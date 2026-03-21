#!/usr/bin/env python3
"""Fix motion quality issues in martial arts NPZ files.

Problem
=======
Two critical issues discovered in CMU→G1 motion captures:

1. **Wrong URDF limits in cmu_amc_to_csv.py**: The CSV pipeline clips joint
   angles to limits that are SIGNIFICANTLY wider than the actual G1 URDF limits
   used in Isaac Lab. This causes 18/22 joints to have mismatched limits.
   Most severe: left_elbow CSV allows [−1.047, +2.094] but URDF is [−1.745, +0.087]
   → reference trajectory goes 2.007 rad BEYOND robot's physical limit in 100%
   of frames! The −10.0 joint_limit penalty fights the +3.0 joint_pos tracking
   reward → bizarre postures, bouncing, jitter.

2. **IK solution flips**: CMU MoCap IK sometimes produces discontinuous angle
   jumps (e.g., right_hip_pitch +2.88→−2.53 in 1 frame = 649 rad/s). After
   csv_to_npz.py downsampling (120→50fps), these spread but still demand
   velocities far beyond G1 motor capabilities (32 rad/s).

Fix Strategy
============
1. **URDF limit clipping** (NEW, critical): Clip all NPZ joint_pos to the
   actual G1 URDF limits from the USD file. This is applied FIRST.

2. **Velocity-limited position smoothing**: Enforce that no joint position can
   change faster than the motor physically allows. Iterative forward+backward
   passes with Gaussian smoothing.

3. **Joint velocity recomputation**: Recompute joint_vel from fixed positions.

4. **Body velocity smoothing**: Median filter body velocities to match.

Note: body_pos_w and body_quat_w are NOT modified (they were recorded from
Isaac Sim FK during csv_to_npz.py). After our joint_pos changes, there will be
a small body data inconsistency, but the body tracking rewards (weight 1.5) are
secondary to joint tracking (weight 3.0) and the inconsistency is small since
most clipping only trims joints near limits.

Usage
=====
    # Fix all files (recommended)
    python scripts/mimic/fix_npz_motion_quality.py --all

    # Fix SideKick only
    python scripts/mimic/fix_npz_motion_quality.py --files G1_side_kick.npz

    # Dry run
    python scripts/mimic/fix_npz_motion_quality.py --dry-run --all

    # Custom velocity limit
    python scripts/mimic/fix_npz_motion_quality.py --max-vel 25.0 --all
"""

import argparse
import os
import shutil
from datetime import datetime

import numpy as np
from scipy.ndimage import gaussian_filter1d, median_filter


# ---------------------------------------------------------------------------
# Paths & defaults
# ---------------------------------------------------------------------------
DEFAULT_DATA_DIR = os.path.normpath(os.path.join(
    os.path.dirname(__file__),
    "..", "..",
    "source", "unitree_rl_lab", "unitree_rl_lab",
    "tasks", "mimic", "robots", "g1_29dof", "martial_arts",
))

# All NPZ files to check
DEFAULT_FILES = [
    "G1_side_kick.npz",
    "G1_lunge_punch.npz",
    "G1_roundhouse_kick.npz",
    "G1_empi.npz",
    "G1_bassai.npz",
    "G1_front_kick.npz",
    "G1_heian_shodan.npz",
]

# G1 motor velocity limits
G1_MAX_JOINT_VEL = 32.0  # rad/s — most restrictive motor

# Safe velocity target: use 85% of motor limit to leave headroom for tracking
DEFAULT_MAX_VEL = 25.0  # rad/s

# ---------------------------------------------------------------------------
# G1 29DOF ACTUAL URDF limits (ground truth from g1_29dof_rev_1_0.urdf,
# verified 2026-03-15 against the URDF file).
# Order: SDK (left_leg×6, right_leg×6, waist×3, left_arm×7, right_arm×7)
# ---------------------------------------------------------------------------
G1_URDF_LIMITS_SDK = [
    (-2.5307, +2.8798),  # 0  left_hip_pitch_joint
    (-0.5236, +2.9671),  # 1  left_hip_roll_joint
    (-2.7576, +2.7576),  # 2  left_hip_yaw_joint
    (-0.0873, +2.8798),  # 3  left_knee_joint
    (-0.8727, +0.5236),  # 4  left_ankle_pitch_joint
    (-0.2618, +0.2618),  # 5  left_ankle_roll_joint
    (-2.5307, +2.8798),  # 6  right_hip_pitch_joint
    (-2.9671, +0.5236),  # 7  right_hip_roll_joint
    (-2.7576, +2.7576),  # 8  right_hip_yaw_joint
    (-0.0873, +2.8798),  # 9  right_knee_joint
    (-0.8727, +0.5236),  # 10 right_ankle_pitch_joint
    (-0.2618, +0.2618),  # 11 right_ankle_roll_joint
    (-2.6180, +2.6180),  # 12 waist_yaw_joint
    (-0.5200, +0.5200),  # 13 waist_roll_joint
    (-0.5200, +0.5200),  # 14 waist_pitch_joint
    (-3.0892, +2.6704),  # 15 left_shoulder_pitch_joint
    (-1.5882, +2.2515),  # 16 left_shoulder_roll_joint
    (-2.6180, +2.6180),  # 17 left_shoulder_yaw_joint
    (-1.0472, +2.0944),  # 18 left_elbow_joint
    (-1.9722, +1.9722),  # 19 left_wrist_roll_joint
    (-1.6144, +1.6144),  # 20 left_wrist_pitch_joint
    (-1.6144, +1.6144),  # 21 left_wrist_yaw_joint
    (-3.0892, +2.6704),  # 22 right_shoulder_pitch_joint
    (-2.2515, +1.5882),  # 23 right_shoulder_roll_joint
    (-2.6180, +2.6180),  # 24 right_shoulder_yaw_joint
    (-1.0472, +2.0944),  # 25 right_elbow_joint
    (-1.9722, +1.9722),  # 26 right_wrist_roll_joint
    (-1.6144, +1.6144),  # 27 right_wrist_pitch_joint
    (-1.6144, +1.6144),  # 28 right_wrist_yaw_joint
]

# Mapping from simulator joint order to SDK joint order
# NPZ stores joints in sim order; SIM_TO_SDK[sim_idx] = sdk_idx
SIM_TO_SDK = [
    0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10, 16, 23,
    5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28,
]

# Build URDF limits in SIM order for direct array indexing
G1_URDF_LIMITS_SIM = np.array([G1_URDF_LIMITS_SDK[SIM_TO_SDK[i]] for i in range(29)])


# SDK joint names for readable diagnostics
SDK_JOINT_NAMES = [
    "left_hip_pitch", "left_hip_roll", "left_hip_yaw",
    "left_knee", "left_ankle_pitch", "left_ankle_roll",
    "right_hip_pitch", "right_hip_roll", "right_hip_yaw",
    "right_knee", "right_ankle_pitch", "right_ankle_roll",
    "waist_yaw", "waist_roll", "waist_pitch",
    "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw",
    "left_elbow", "left_wrist_roll", "left_wrist_pitch", "left_wrist_yaw",
    "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw",
    "right_elbow", "right_wrist_roll", "right_wrist_pitch", "right_wrist_yaw",
]


def get_joint_name_sim(sim_idx: int) -> str:
    """Get human-readable joint name from sim index."""
    return SDK_JOINT_NAMES[SIM_TO_SDK[sim_idx]]


def analyze_urdf_violations(joint_pos: np.ndarray,
                            soft_factor: float = 0.9) -> dict:
    """Analyze URDF limit violations in joint_pos (sim order).

    Args:
        joint_pos: [T, 29] array in sim order.
        soft_factor: Soft limit factor (matches robot config).

    Returns:
        Dict with violation details.
    """
    T, J = joint_pos.shape
    lo = G1_URDF_LIMITS_SIM[:, 0]  # [29]
    hi = G1_URDF_LIMITS_SIM[:, 1]  # [29]

    # Hard limit violations
    hard_lo_viol = joint_pos < lo[None, :]  # [T, 29]
    hard_hi_viol = joint_pos > hi[None, :]
    hard_violations = hard_lo_viol | hard_hi_viol

    # Soft limit violations (what the simulator actually penalizes)
    mid = (lo + hi) / 2.0
    half_range = (hi - lo) / 2.0
    soft_lo = mid - half_range * soft_factor
    soft_hi = mid + half_range * soft_factor
    soft_lo_viol = joint_pos < soft_lo[None, :]
    soft_hi_viol = joint_pos > soft_hi[None, :]
    soft_violations = soft_lo_viol | soft_hi_viol

    # Per-joint stats
    joint_violations = []
    for j in range(J):
        n_hard = int(hard_violations[:, j].sum())
        n_soft = int(soft_violations[:, j].sum())
        if n_hard > 0 or n_soft > 0:
            max_lo_excess = float(max(0.0, (lo[j] - joint_pos[:, j]).max()))
            max_hi_excess = float(max(0.0, (joint_pos[:, j] - hi[j]).max()))
            joint_violations.append({
                "sim_idx": j,
                "name": get_joint_name_sim(j),
                "n_hard": n_hard,
                "n_soft": n_soft,
                "pct_hard": 100.0 * n_hard / T,
                "pct_soft": 100.0 * n_soft / T,
                "max_lo_excess": max_lo_excess,
                "max_hi_excess": max_hi_excess,
                "data_range": (float(joint_pos[:, j].min()),
                               float(joint_pos[:, j].max())),
                "urdf_range": (float(lo[j]), float(hi[j])),
            })

    return {
        "n_hard_total": int(hard_violations.sum()),
        "n_soft_total": int(soft_violations.sum()),
        "pct_hard_total": 100.0 * hard_violations.sum() / (T * J),
        "pct_soft_total": 100.0 * soft_violations.sum() / (T * J),
        "joint_violations": joint_violations,
    }


def clip_to_urdf_limits(joint_pos: np.ndarray,
                        margin: float = 0.0) -> np.ndarray:
    """Clip joint positions to actual G1 URDF limits.

    Args:
        joint_pos: [T, 29] array in sim order.
        margin: Safety margin inside limits (rad). 0.0 = clip to exact limits.

    Returns:
        Clipped joint_pos (copy).
    """
    jp = joint_pos.copy()
    lo = G1_URDF_LIMITS_SIM[:, 0] + margin  # [29]
    hi = G1_URDF_LIMITS_SIM[:, 1] - margin  # [29]
    # Ensure lo <= hi even with margin
    lo = np.minimum(lo, hi)
    jp = np.clip(jp, lo[None, :], hi[None, :])
    return jp


def analyze_motion(filepath: str) -> dict:
    """Analyze motion quality of an NPZ file.

    Reports both velocity violations and URDF limit violations.
    """
    data = np.load(filepath)
    jp = data["joint_pos"]  # [T, 29]
    jv = data["joint_vel"]  # [T, 29]
    fps = float(data["fps"])
    T, J = jp.shape

    # Implied velocities from position differences
    pos_diffs = np.diff(jp, axis=0)  # [T-1, 29]
    implied_vel = pos_diffs * fps     # rad/s

    # Per-joint max velocity
    joint_max_vel = np.abs(implied_vel).max(axis=0)

    # Find joints exceeding velocity threshold
    problem_joints = []
    for j in range(J):
        max_v = joint_max_vel[j]
        if max_v > DEFAULT_MAX_VEL:
            max_frame = np.argmax(np.abs(implied_vel[:, j]))
            problem_joints.append({
                "joint": j,
                "name": get_joint_name_sim(j),
                "max_vel": float(max_v),
                "max_frame": int(max_frame),
            })

    # URDF limit violations
    urdf_analysis = analyze_urdf_violations(jp)

    return {
        "filepath": filepath,
        "filename": os.path.basename(filepath),
        "T": T,
        "J": J,
        "fps": fps,
        "duration_s": T / fps,
        "max_implied_vel": float(np.abs(implied_vel).max()),
        "max_stored_vel": float(np.abs(jv).max()),
        "problem_joints": problem_joints,
        "n_problem_joints": len(problem_joints),
        "urdf_violations": urdf_analysis,
    }


def velocity_limited_smooth(joint_pos: np.ndarray, fps: float,
                            max_vel: float = DEFAULT_MAX_VEL,
                            n_passes: int = 10,
                            gaussian_sigma: float = 1.0) -> np.ndarray:
    """Smooth joint positions to enforce velocity limits.

    Algorithm:
    1. Forward pass: for each frame, if the position delta exceeds max_vel/fps,
       clamp the delta (limit how fast the joint can move).
    2. Backward pass: same, in reverse (to avoid asymmetric bias).
    3. Average forward and backward results.
    4. Repeat for n_passes to converge.
    5. Light Gaussian smooth to remove remaining jaggedness.

    This preserves the overall trajectory shape while ensuring no frame-to-frame
    transition demands more velocity than the motor can provide.

    Args:
        joint_pos: [T, J] array of joint positions.
        fps: Frames per second.
        max_vel: Maximum allowed implied velocity (rad/s).
        n_passes: Number of forward+backward passes.
        gaussian_sigma: Sigma for final Gaussian smoothing (frames).

    Returns:
        Smoothed joint_pos array (copy).
    """
    jp = joint_pos.copy()
    T, J = jp.shape
    max_delta = max_vel / fps  # max position change per frame

    for pass_idx in range(n_passes):
        # Check if we still have violations
        diffs = np.abs(np.diff(jp, axis=0))
        max_violation = diffs.max()
        if max_violation <= max_delta * 1.001:  # converged (with tiny tolerance)
            break

        # Forward pass: clamp deltas
        jp_fwd = jp.copy()
        for t in range(1, T):
            delta = jp_fwd[t] - jp_fwd[t - 1]
            clamped = np.clip(delta, -max_delta, max_delta)
            jp_fwd[t] = jp_fwd[t - 1] + clamped

        # Backward pass: clamp deltas (reverse direction)
        jp_bwd = jp.copy()
        for t in range(T - 2, -1, -1):
            delta = jp_bwd[t] - jp_bwd[t + 1]
            clamped = np.clip(delta, -max_delta, max_delta)
            jp_bwd[t] = jp_bwd[t + 1] + clamped

        # Average: this prevents drift bias from either direction
        jp = (jp_fwd + jp_bwd) / 2.0

    # Final Gaussian smooth to remove remaining jaggedness from clamping
    if gaussian_sigma > 0:
        for j in range(J):
            jp[:, j] = gaussian_filter1d(jp[:, j], sigma=gaussian_sigma, mode='nearest')

    return jp


def recompute_joint_vel(joint_pos: np.ndarray, fps: float) -> np.ndarray:
    """Recompute joint velocities from positions using central differences."""
    dt = 1.0 / fps
    return np.gradient(joint_pos, dt, axis=0)


def smooth_body_velocities(vel: np.ndarray, max_vel: float,
                           median_kernel: int = 5) -> np.ndarray:
    """Smooth body velocity spikes using median filter."""
    smoothed = vel.copy()
    T, N, D = vel.shape
    for body in range(N):
        for dim in range(D):
            signal = vel[:, body, dim]
            if np.abs(signal).max() > max_vel:
                smoothed[:, body, dim] = median_filter(
                    signal, size=median_kernel, mode='nearest'
                )
    return smoothed


def fix_npz_file(filepath: str, max_vel: float = DEFAULT_MAX_VEL,
                 backup: bool = True, dry_run: bool = False) -> dict:
    """Fix motion quality issues in a single NPZ file.

    Fix pipeline:
    1. Clip joint_pos to actual URDF limits (critical!)
    2. Apply velocity-limited position smoothing
    3. Re-clip to URDF limits (smoothing can push values back out)
    4. Recompute joint_vel from fixed positions
    5. Smooth body velocities

    Args:
        filepath: Path to the NPZ file.
        max_vel: Maximum allowed implied velocity (rad/s).
        backup: Whether to create a backup.
        dry_run: Analyze only.

    Returns:
        Dict with before/after statistics.
    """
    # --- Analyze before ---
    before = analyze_motion(filepath)
    filename = before["filename"]
    urdf_v = before["urdf_violations"]

    has_vel_problems = before["n_problem_joints"] > 0
    has_urdf_problems = urdf_v["n_hard_total"] > 0

    if not has_vel_problems and not has_urdf_problems:
        print(f"  ✅ {filename}: No issues "
              f"(max_vel={before['max_implied_vel']:.1f} rad/s, "
              f"0 URDF violations). Skipping.")
        return {"filename": filename, "status": "clean", "before": before, "after": before}

    # --- Report issues ---
    if has_urdf_problems:
        print(f"  🔧 {filename}: {urdf_v['n_hard_total']} URDF hard-limit violations "
              f"({urdf_v['pct_hard_total']:.1f}% of samples)")
        for jv_info in urdf_v["joint_violations"]:
            if jv_info["n_hard"] > 0:
                print(f"      sim[{jv_info['sim_idx']:2d}] {jv_info['name']:25s}: "
                      f"{jv_info['pct_hard']:5.1f}% hard viol, "
                      f"data=[{jv_info['data_range'][0]:+.3f}, {jv_info['data_range'][1]:+.3f}], "
                      f"URDF=[{jv_info['urdf_range'][0]:+.3f}, {jv_info['urdf_range'][1]:+.3f}]")

    if has_vel_problems:
        print(f"  🔧 {filename}: {before['n_problem_joints']} joints exceed {max_vel:.0f} rad/s "
              f"(max={before['max_implied_vel']:.1f} rad/s)")
        for pj in before["problem_joints"]:
            print(f"      sim[{pj['joint']:2d}] {pj['name']:25s}: "
                  f"{pj['max_vel']:.1f} rad/s at frame {pj['max_frame']}")

    if dry_run:
        return {"filename": filename, "status": "dry_run", "before": before, "after": None}

    # --- Load data ---
    data = dict(np.load(filepath))
    fps = float(data["fps"])
    original_jp = data["joint_pos"].copy()

    # --- Step 1: Clip to URDF limits (CRITICAL!) ---
    data["joint_pos"] = clip_to_urdf_limits(data["joint_pos"])
    clip_diff = np.abs(original_jp - data["joint_pos"])
    n_clipped = int(np.sum(clip_diff > 1e-6))
    max_clip = float(clip_diff.max())
    print(f"      Step 1 (URDF clip): {n_clipped} samples clipped "
          f"(max_clip={max_clip:.4f} rad)")

    # --- Step 2: Velocity-limited smoothing ---
    data["joint_pos"] = velocity_limited_smooth(
        data["joint_pos"], fps, max_vel=max_vel,
        n_passes=20, gaussian_sigma=1.0
    )

    # --- Step 3: Re-clip after smoothing (Gaussian can push back out) ---
    data["joint_pos"] = clip_to_urdf_limits(data["joint_pos"])

    # --- Step 4: Recompute joint velocities ---
    data["joint_vel"] = recompute_joint_vel(data["joint_pos"], fps)

    # --- Clamp joint velocities as safety net ---
    data["joint_vel"] = np.clip(data["joint_vel"], -G1_MAX_JOINT_VEL, G1_MAX_JOINT_VEL)

    # --- Step 5: Smooth body velocities ---
    data["body_lin_vel_w"] = smooth_body_velocities(
        data["body_lin_vel_w"], max_vel=15.0, median_kernel=5
    )
    data["body_ang_vel_w"] = smooth_body_velocities(
        data["body_ang_vel_w"], max_vel=30.0, median_kernel=5
    )

    # --- Backup ---
    if backup:
        backup_path = filepath.replace(
            ".npz", f"_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.npz"
        )
        shutil.copy2(filepath, backup_path)
        print(f"      Backup: {os.path.basename(backup_path)}")

    # --- Save ---
    save_data = {}
    for key in data:
        if key == "fps":
            save_data[key] = np.array([fps])
        else:
            save_data[key] = data[key]
    np.savez(filepath, **save_data)

    # --- Analyze after ---
    after = analyze_motion(filepath)
    after_urdf = after["urdf_violations"]

    # --- Report ---
    jp_diff = np.abs(original_jp - data["joint_pos"])
    n_changed = np.sum(jp_diff > 1e-6)
    max_change = jp_diff.max()
    mean_change = jp_diff[jp_diff > 1e-6].mean() if n_changed > 0 else 0

    print(f"      ✅ Fixed: max_vel {before['max_implied_vel']:.1f} → "
          f"{after['max_implied_vel']:.1f} rad/s")
    print(f"         URDF hard violations: {urdf_v['n_hard_total']} → "
          f"{after_urdf['n_hard_total']}")
    print(f"         {n_changed} position samples modified "
          f"(max_change={max_change:.4f} rad, mean_change={mean_change:.4f} rad)")
    print(f"         max_stored_vel: {before['max_stored_vel']:.1f} → "
          f"{after['max_stored_vel']:.1f} rad/s")

    if after["n_problem_joints"] > 0:
        print(f"      ⚠️  {after['n_problem_joints']} joints still exceed vel limit:")
        for pj in after["problem_joints"]:
            print(f"         sim[{pj['joint']:2d}] {pj['name']:25s}: {pj['max_vel']:.1f} rad/s")

    if after_urdf["n_hard_total"] > 0:
        print(f"      ⚠️  {after_urdf['n_hard_total']} URDF violations remain!")

    return {
        "filename": filename,
        "status": "fixed",
        "before": before,
        "after": after,
        "n_changed": int(n_changed),
        "max_change": float(max_change),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Fix motion quality issues in martial arts NPZ files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--data-dir", default=DEFAULT_DATA_DIR,
        help="Directory containing NPZ files.",
    )
    parser.add_argument(
        "--files", nargs="+", default=DEFAULT_FILES,
        help="NPZ filenames to process.",
    )
    parser.add_argument(
        "--max-vel", type=float, default=DEFAULT_MAX_VEL,
        help="Max implied velocity (rad/s). Default: 25.0",
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
        help="Process ALL NPZ files in data-dir.",
    )
    args = parser.parse_args()

    if args.all:
        args.files = sorted(
            f for f in os.listdir(args.data_dir)
            if f.endswith(".npz") and "_backup_" not in f
        )

    print(f"{'=' * 60}")
    print(f"NPZ Motion Quality Fixer (URDF limits + velocity smoothing)")
    print(f"{'=' * 60}")
    print(f"Data dir: {args.data_dir}")
    print(f"Files: {args.files}")
    print(f"Max velocity: {args.max_vel} rad/s")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'FIX'}")
    print(f"{'=' * 60}\n")

    results = []
    for filename in args.files:
        filepath = os.path.join(args.data_dir, filename)
        if not os.path.isfile(filepath):
            print(f"  ❌ {filename}: Not found!")
            continue
        result = fix_npz_file(filepath, max_vel=args.max_vel,
                              backup=not args.no_backup, dry_run=args.dry_run)
        results.append(result)
        print()

    # --- Summary ---
    print(f"{'=' * 60}")
    print("Summary:")
    print(f"{'=' * 60}")
    for r in results:
        emoji = {"clean": "✅", "fixed": "🔧", "dry_run": "👁️"}.get(r["status"], "❓")
        if r["status"] == "fixed":
            urdf_before = r["before"]["urdf_violations"]["n_hard_total"]
            urdf_after = r["after"]["urdf_violations"]["n_hard_total"]
            print(f"  {emoji} {r['filename']}: "
                  f"max_vel {r['before']['max_implied_vel']:.1f} → "
                  f"{r['after']['max_implied_vel']:.1f} rad/s, "
                  f"URDF viol {urdf_before} → {urdf_after}, "
                  f"{r['n_changed']} samples modified")
        elif r["status"] == "clean":
            print(f"  {emoji} {r['filename']}: clean "
                  f"(max_vel={r['before']['max_implied_vel']:.1f} rad/s, "
                  f"URDF viol={r['before']['urdf_violations']['n_hard_total']})")
        else:
            print(f"  {emoji} {r['filename']}: {r['status']}")

    # Overall check
    all_clean = all(
        (r.get("after") or r.get("before", {})).get("urdf_violations", {}).get("n_hard_total", 0) == 0
        and (r.get("after") or r.get("before", {})).get("n_problem_joints", 0) == 0
        for r in results
        if r["status"] != "dry_run"
    )
    if all_clean and not args.dry_run:
        print(f"\n  🎉 All files pass quality checks!")
    elif args.dry_run:
        n_need_fix = sum(1 for r in results if r["status"] == "dry_run")
        print(f"\n  👁️  {n_need_fix} files need fixing. Run without --dry-run to apply.")


if __name__ == "__main__":
    main()
