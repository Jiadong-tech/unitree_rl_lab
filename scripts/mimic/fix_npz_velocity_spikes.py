#!/usr/bin/env python3
"""Fix velocity spikes in martial arts NPZ files.

Root Cause Analysis
===================
The CMU ASF+AMC → G1 CSV conversion (cmu_amc_to_csv.py) produces joint angles
that sometimes "wrap" — the same physical joint configuration gets mapped to
two different angle representations.  For example:

    Joint 19 (left_wrist_roll): -2.618 → +2.618  (Δ=5.236 rad in 1 frame!)

The csv_to_npz.py pipeline then computes joint_vel = gradient(joint_pos), which
produces extreme velocities (>100 rad/s) at these wrap points.

Fix Strategy
============
1. **joint_pos smoothing**: Detect frame-to-frame jumps exceeding a threshold
   (default 2.0 rad) and bridge them with cubic interpolation over a small
   window, choosing the shorter angular path.
2. **joint_vel recomputation**: Recompute velocities from the smoothed positions
   using the same gradient method as csv_to_npz.py.
3. **body velocity smoothing**: Apply a median filter to body_lin_vel_w and
   body_ang_vel_w to remove the corresponding spikes (these are recorded from
   Isaac Sim during playback and are also affected by the position jumps).
4. **Clamp safety**: Final clamp of joint_vel to ±max_vel (G1 motor limit).

Usage
=====
    # Fix all problematic NPZ files
    python scripts/mimic/fix_npz_velocity_spikes.py

    # Fix a specific file
    python scripts/mimic/fix_npz_velocity_spikes.py --files G1_bassai.npz

    # Dry run (analyze without modifying)
    python scripts/mimic/fix_npz_velocity_spikes.py --dry-run

    # Custom threshold
    python scripts/mimic/fix_npz_velocity_spikes.py --jump-threshold 1.5
"""

import argparse
import os
import shutil
from datetime import datetime

import numpy as np
from scipy.ndimage import median_filter


# ---------------------------------------------------------------------------
# Default data directory
# ---------------------------------------------------------------------------
DEFAULT_DATA_DIR = os.path.join(
    os.path.dirname(__file__),
    "..", "..",
    "source", "unitree_rl_lab", "unitree_rl_lab",
    "tasks", "mimic", "robots", "g1_29dof", "martial_arts",
)
DEFAULT_DATA_DIR = os.path.normpath(DEFAULT_DATA_DIR)

# All NPZ files that need checking (the 4 with known issues)
DEFAULT_FILES = [
    "G1_bassai.npz",
    "G1_empi.npz",
    "G1_lunge_punch.npz",
    "G1_side_kick.npz",
]

# G1 motor velocity limits (rad/s) — conservative bound
G1_MAX_JOINT_VEL = 32.0  # Most restrictive motor (7520-14): 32 rad/s


def analyze_npz(filepath: str, jump_threshold: float = 2.0) -> dict:
    """Analyze an NPZ file for velocity spikes and position jumps.

    Returns a dict with diagnostic info.
    """
    data = np.load(filepath)
    jp = data["joint_pos"]      # [T, 29]
    jv = data["joint_vel"]      # [T, 29]
    blv = data["body_lin_vel_w"]  # [T, N_bodies, 3]
    bav = data["body_ang_vel_w"]  # [T, N_bodies, 3]
    fps = float(data["fps"])
    T, J = jp.shape

    diffs = np.diff(jp, axis=0)  # [T-1, 29]

    issues = []
    for j in range(J):
        jump_frames = np.where(np.abs(diffs[:, j]) > jump_threshold)[0]
        if len(jump_frames) > 0:
            max_vel = np.abs(jv[:, j]).max()
            issues.append({
                "joint": j,
                "num_jumps": len(jump_frames),
                "frames": jump_frames.tolist(),
                "max_vel": max_vel,
                "jumps": [(int(f), float(jp[f, j]), float(jp[f + 1, j]), float(diffs[f, j]))
                          for f in jump_frames[:10]],
            })

    return {
        "filepath": filepath,
        "filename": os.path.basename(filepath),
        "T": T,
        "J": J,
        "fps": fps,
        "duration_s": T / fps,
        "max_joint_vel": float(np.abs(jv).max()),
        "max_body_lin_vel": float(np.abs(blv).max()),
        "max_body_ang_vel": float(np.abs(bav).max()),
        "issues": issues,
        "total_jumps": sum(i["num_jumps"] for i in issues),
    }


def fix_joint_pos_jumps(joint_pos: np.ndarray, jump_threshold: float = 2.0,
                        bridge_half_width: int = 3) -> np.ndarray:
    """Fix discontinuous jumps in joint_pos by smooth interpolation bridging.

    Root cause: The CMU → G1 joint mapping sometimes produces equivalent but
    discontinuous angle representations (IK solution flips). These are NOT
    simple ±2π wraps — they are genuine configuration changes that happen to
    be physically equivalent.

    Strategy: For every detected jump, we smooth-bridge using linear
    interpolation over a small window around the discontinuity. This preserves
    the joint's physical range (stays within robot limits) while eliminating
    the velocity spike.

    For paired jumps (out-and-back within a short window), we bridge across
    the entire anomalous segment to avoid leaving artifacts.

    Args:
        joint_pos: [T, J] joint positions array.
        jump_threshold: Minimum frame-to-frame delta (rad) to consider a jump.
        bridge_half_width: Half-window for the smoothing bridge around isolated jumps.

    Returns:
        Fixed joint_pos array (copy, original not modified).
    """
    jp = joint_pos.copy()
    T, J = jp.shape

    for j in range(J):
        # Iteratively fix jumps — each pass may expose secondary jumps at bridge edges
        for pass_idx in range(30):
            diffs = np.diff(jp[:, j])
            jump_frames = np.where(np.abs(diffs) > jump_threshold)[0]
            if len(jump_frames) == 0:
                break

            # Try to find PAIRED jumps: out-jump at frame A, back-jump at frame B
            # (the anomalous segment is [A+1 .. B], typically short)
            processed = set()
            for idx, f in enumerate(jump_frames):
                if f in processed:
                    continue

                # Look for a return jump within 20 frames
                partner = None
                for idx2 in range(idx + 1, len(jump_frames)):
                    f2 = jump_frames[idx2]
                    if f2 - f <= 20:
                        # Check if the two jumps roughly cancel each other
                        delta1 = diffs[f]
                        delta2 = diffs[f2]
                        if abs(delta1 + delta2) < abs(delta1) * 0.5:
                            partner = f2
                            processed.add(f2)
                            break

                if partner is not None:
                    # Paired jump: bridge the entire anomalous segment [f .. partner+1]
                    # using the values just outside as anchors
                    lo = max(0, f - 1)
                    hi = min(T - 1, partner + 2)
                    x_anchors = [lo, hi]
                    y_anchors = [jp[lo, j], jp[hi, j]]
                    x_bridge = np.arange(lo, hi + 1)
                    jp[lo:hi + 1, j] = np.interp(x_bridge, x_anchors, y_anchors)
                else:
                    # Isolated jump: smooth-bridge around the single discontinuity
                    _smooth_bridge(jp, j, f, bridge_half_width, T)

                processed.add(f)

            # Only process the first batch per pass, then re-check
            break  # Let the outer loop re-detect after fixes

    return jp


def _smooth_bridge(jp: np.ndarray, j: int, frame: int,
                   half_width: int, T: int):
    """Smooth a discontinuity at `frame` by cubic interpolation over a window."""
    lo = max(0, frame - half_width)
    hi = min(T - 1, frame + 1 + half_width)

    # Anchor points: the values just outside the bridge window
    x_anchors = [lo, hi]
    y_anchors = [jp[lo, j], jp[hi, j]]

    if hi - lo <= 1:
        return

    # Linear interpolation through the bridge (simple and robust)
    x_bridge = np.arange(lo, hi + 1)
    jp[lo:hi + 1, j] = np.interp(x_bridge, x_anchors, y_anchors)


def recompute_joint_vel(joint_pos: np.ndarray, fps: float) -> np.ndarray:
    """Recompute joint velocities from positions using central differences.

    Matches the method in csv_to_npz.py: torch.gradient with spacing=dt.
    numpy.gradient with edge_order=1 gives the same result.
    """
    dt = 1.0 / fps
    return np.gradient(joint_pos, dt, axis=0)


def smooth_body_velocities(vel: np.ndarray, max_vel: float,
                           median_kernel: int = 5) -> np.ndarray:
    """Smooth body velocity spikes using median filter + clamping.

    Args:
        vel: [T, N_bodies, 3] velocity array.
        max_vel: Maximum allowed velocity magnitude per component.
        median_kernel: Size of the median filter kernel.

    Returns:
        Smoothed velocity array.
    """
    smoothed = vel.copy()
    T, N, D = vel.shape

    for body in range(N):
        for dim in range(D):
            signal = vel[:, body, dim]
            # Only apply median filter if there are spikes
            if np.abs(signal).max() > max_vel:
                smoothed[:, body, dim] = median_filter(signal, size=median_kernel, mode='nearest')

    return smoothed


def fix_npz_file(filepath: str, jump_threshold: float = 2.0,
                 max_joint_vel: float = G1_MAX_JOINT_VEL,
                 backup: bool = True, dry_run: bool = False) -> dict:
    """Fix velocity spikes in a single NPZ file.

    Args:
        filepath: Path to the NPZ file.
        jump_threshold: Threshold for detecting position jumps (rad).
        max_joint_vel: Maximum allowed joint velocity (rad/s).
        backup: Whether to create a backup of the original file.
        dry_run: If True, analyze only without modifying files.

    Returns:
        Dict with before/after statistics.
    """
    # --- Analyze before ---
    before = analyze_npz(filepath, jump_threshold)
    filename = before["filename"]

    if before["total_jumps"] == 0:
        print(f"  ✅ {filename}: No jumps detected (max_vel={before['max_joint_vel']:.1f} rad/s). Skipping.")
        return {"filename": filename, "status": "clean", "before": before, "after": before}

    print(f"  🔧 {filename}: {before['total_jumps']} jumps across "
          f"{len(before['issues'])} joints, max_vel={before['max_joint_vel']:.1f} rad/s")

    if dry_run:
        for issue in before["issues"]:
            print(f"      Joint {issue['joint']}: {issue['num_jumps']} jumps, "
                  f"max_vel={issue['max_vel']:.1f} rad/s")
            for f, v_before, v_after, delta in issue["jumps"][:3]:
                print(f"        frame {f}: {v_before:.3f} → {v_after:.3f} (Δ={delta:.3f})")
        return {"filename": filename, "status": "dry_run", "before": before, "after": None}

    # --- Load data ---
    data = dict(np.load(filepath))
    fps = float(data["fps"])

    # --- Fix joint positions ---
    original_jp = data["joint_pos"].copy()
    data["joint_pos"] = fix_joint_pos_jumps(data["joint_pos"], jump_threshold)

    # --- Recompute joint velocities ---
    data["joint_vel"] = recompute_joint_vel(data["joint_pos"], fps)

    # --- Clamp joint velocities ---
    vel_before_clamp = data["joint_vel"].copy()
    data["joint_vel"] = np.clip(data["joint_vel"], -max_joint_vel, max_joint_vel)
    n_clamped = np.sum(np.abs(vel_before_clamp) > max_joint_vel)

    # --- Smooth body velocities ---
    # body_lin_vel_w: physical spikes from the position jumps
    data["body_lin_vel_w"] = smooth_body_velocities(
        data["body_lin_vel_w"], max_vel=15.0, median_kernel=5
    )
    # body_ang_vel_w: angular velocity spikes
    data["body_ang_vel_w"] = smooth_body_velocities(
        data["body_ang_vel_w"], max_vel=30.0, median_kernel=5
    )

    # --- Backup original ---
    if backup:
        backup_path = filepath.replace(".npz", f"_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.npz")
        shutil.copy2(filepath, backup_path)
        print(f"      Backup saved: {os.path.basename(backup_path)}")

    # --- Save fixed ---
    # Ensure fps is stored as scalar (matching original format)
    save_data = {}
    for key in data:
        if key == "fps":
            save_data[key] = np.array([fps])
        else:
            save_data[key] = data[key]
    np.savez(filepath, **save_data)

    # --- Analyze after ---
    after = analyze_npz(filepath, jump_threshold)

    # --- Report ---
    jp_changed = np.sum(np.abs(original_jp - data["joint_pos"]) > 1e-6)
    print(f"      ✅ Fixed: max_vel {before['max_joint_vel']:.1f} → {after['max_joint_vel']:.1f} rad/s, "
          f"{jp_changed} position samples modified, {n_clamped} velocity samples clamped")
    print(f"         body_lin_vel max: {before['max_body_lin_vel']:.1f} → {after['max_body_lin_vel']:.1f}")
    print(f"         body_ang_vel max: {before['max_body_ang_vel']:.1f} → {after['max_body_ang_vel']:.1f}")

    if after["total_jumps"] > 0:
        print(f"      ⚠️  {after['total_jumps']} jumps remain (may need manual review):")
        for issue in after["issues"]:
            print(f"         Joint {issue['joint']}: {issue['num_jumps']} jumps, "
                  f"max_vel={issue['max_vel']:.1f} rad/s")

    return {
        "filename": filename,
        "status": "fixed",
        "before": before,
        "after": after,
        "jp_changed": int(jp_changed),
        "vel_clamped": int(n_clamped),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Fix velocity spikes in martial arts NPZ files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--data-dir", default=DEFAULT_DATA_DIR,
        help="Directory containing NPZ files.",
    )
    parser.add_argument(
        "--files", nargs="+", default=DEFAULT_FILES,
        help="NPZ filenames to process (relative to data-dir).",
    )
    parser.add_argument(
        "--jump-threshold", type=float, default=2.0,
        help="Min frame-to-frame position jump (rad) to consider a discontinuity.",
    )
    parser.add_argument(
        "--max-joint-vel", type=float, default=G1_MAX_JOINT_VEL,
        help="Max joint velocity (rad/s) for clamping.",
    )
    parser.add_argument(
        "--no-backup", action="store_true",
        help="Don't create backup files.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Analyze only, don't modify files.",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Process ALL NPZ files in data-dir, not just known problematic ones.",
    )
    args = parser.parse_args()

    if args.all:
        args.files = sorted(f for f in os.listdir(args.data_dir) if f.endswith(".npz"))

    print(f"{'=' * 60}")
    print(f"NPZ Velocity Spike Fixer")
    print(f"{'=' * 60}")
    print(f"Data dir: {args.data_dir}")
    print(f"Files: {args.files}")
    print(f"Jump threshold: {args.jump_threshold} rad")
    print(f"Max joint vel: {args.max_joint_vel} rad/s")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'FIX'}")
    print(f"{'=' * 60}\n")

    results = []
    for filename in args.files:
        filepath = os.path.join(args.data_dir, filename)
        if not os.path.isfile(filepath):
            print(f"  ❌ {filename}: File not found!")
            continue
        result = fix_npz_file(
            filepath,
            jump_threshold=args.jump_threshold,
            max_joint_vel=args.max_joint_vel,
            backup=not args.no_backup,
            dry_run=args.dry_run,
        )
        results.append(result)
        print()

    # --- Summary ---
    print(f"{'=' * 60}")
    print("Summary:")
    print(f"{'=' * 60}")
    for r in results:
        status_emoji = {"clean": "✅", "fixed": "🔧", "dry_run": "👁️"}.get(r["status"], "❓")
        if r["status"] == "fixed":
            print(f"  {status_emoji} {r['filename']}: "
                  f"max_vel {r['before']['max_joint_vel']:.1f} → {r['after']['max_joint_vel']:.1f} rad/s, "
                  f"{r['jp_changed']} pos fixed, {r['vel_clamped']} vel clamped")
        else:
            print(f"  {status_emoji} {r['filename']}: {r['status']}")


if __name__ == "__main__":
    main()
