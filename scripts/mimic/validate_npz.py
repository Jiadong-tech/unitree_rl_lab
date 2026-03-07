"""Validate martial arts NPZ data files for correctness before training.

Usage:
    python scripts/mimic/validate_npz.py \
        --dir source/unitree_rl_lab/unitree_rl_lab/tasks/mimic/robots/g1_29dof/martial_arts/

Checks performed:
    1. Required keys exist (fps, joint_pos, joint_vel, body_pos_w, etc.)
    2. Shape consistency (all arrays have same number of frames)
    3. FPS is reasonable (30-120 Hz)
    4. Joint angles within physical limits
    5. No NaN or Inf values
    6. Root height is reasonable (not underground)
    7. Duration is reasonable (>0.5s, <60s)
"""

import argparse
import glob
import os
import sys

import numpy as np


# G1 29DOF approximate joint limits (radians) - conservative estimates
G1_JOINT_LIMITS = {
    "min": np.array([
        -1.6, -0.5, -0.5, -0.1, -0.9, -0.3,  # left leg
        -1.6, -0.5, -0.5, -0.1, -0.9, -0.3,  # right leg
        -1.5, -0.4, -0.6,                       # waist
        -3.1, -1.6, -1.6, -2.6, -1.6, -0.5, -0.5,  # left arm
        -3.1, -1.6, -1.6, -2.6, -1.6, -0.5, -0.5,  # right arm
    ]),
    "max": np.array([
        1.6, 0.5, 0.5, 2.5, 0.6, 0.3,   # left leg
        1.6, 0.5, 0.5, 2.5, 0.6, 0.3,   # right leg
        1.5, 0.4, 0.6,                    # waist
        3.1, 1.6, 1.6, 0.1, 1.6, 0.5, 0.5,  # left arm
        3.1, 1.6, 1.6, 0.1, 1.6, 0.5, 0.5,  # right arm
    ]),
}

REQUIRED_KEYS = ["fps", "joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"]


def validate_npz(filepath: str, verbose: bool = True) -> bool:
    """Validate a single NPZ file. Returns True if valid."""
    filename = os.path.basename(filepath)
    errors = []
    warnings = []

    try:
        data = np.load(filepath, allow_pickle=True)
    except Exception as e:
        print(f"  ❌ {filename}: Cannot load file - {e}")
        return False

    # Check required keys
    for key in REQUIRED_KEYS:
        if key not in data:
            errors.append(f"Missing key: '{key}'")

    if errors:
        for e in errors:
            print(f"  ❌ {filename}: {e}")
        return False

    fps = float(data["fps"].flat[0]) if data["fps"].ndim > 0 else float(data["fps"])
    joint_pos = data["joint_pos"]
    joint_vel = data["joint_vel"]
    body_pos_w = data["body_pos_w"]
    body_quat_w = data["body_quat_w"]
    body_lin_vel_w = data["body_lin_vel_w"]
    body_ang_vel_w = data["body_ang_vel_w"]

    n_frames = joint_pos.shape[0]
    duration = (n_frames - 1) / fps

    # FPS check
    if fps < 30 or fps > 120:
        warnings.append(f"Unusual FPS: {fps} (expected 30-120)")
    if abs(fps - 50) > 0.1 and abs(fps - 60) > 0.1:
        warnings.append(f"FPS={fps}, training expects 50Hz output")

    # Duration check
    if duration < 0.5:
        errors.append(f"Too short: {duration:.2f}s (< 0.5s)")
    if duration > 120:
        warnings.append(f"Very long: {duration:.2f}s (> 120s)")

    # Shape consistency
    arrays = {
        "joint_pos": joint_pos,
        "joint_vel": joint_vel,
        "body_pos_w": body_pos_w,
        "body_quat_w": body_quat_w,
        "body_lin_vel_w": body_lin_vel_w,
        "body_ang_vel_w": body_ang_vel_w,
    }
    for name, arr in arrays.items():
        if arr.shape[0] != n_frames:
            errors.append(f"{name} frame count mismatch: {arr.shape[0]} vs {n_frames}")

    # NaN / Inf check
    for name, arr in arrays.items():
        if np.any(np.isnan(arr)):
            errors.append(f"{name} contains NaN values")
        if np.any(np.isinf(arr)):
            errors.append(f"{name} contains Inf values")

    # Joint dimensions
    if joint_pos.ndim == 2:
        n_joints = joint_pos.shape[1]
        if n_joints < 29:
            warnings.append(f"Only {n_joints} joints (expected >= 29 for G1)")
    else:
        errors.append(f"joint_pos has wrong ndim: {joint_pos.ndim} (expected 2)")

    # Joint limit check (only on first 29 columns if available)
    if joint_pos.ndim == 2 and joint_pos.shape[1] >= 29:
        j29 = joint_pos[:, :29]
        for i in range(29):
            col = j29[:, i]
            if np.any(col < G1_JOINT_LIMITS["min"][i] - 0.5):
                warnings.append(f"Joint {i} below soft limit: min={col.min():.3f} vs limit={G1_JOINT_LIMITS['min'][i]:.3f}")
            if np.any(col > G1_JOINT_LIMITS["max"][i] + 0.5):
                warnings.append(f"Joint {i} above soft limit: max={col.max():.3f} vs limit={G1_JOINT_LIMITS['max'][i]:.3f}")

    # Root height check (body_pos_w[:, 0, 2] should be pelvis z)
    if body_pos_w.ndim == 3 and body_pos_w.shape[1] > 0:
        root_z = body_pos_w[:, 0, 2]
        if np.any(root_z < 0):
            warnings.append(f"Root goes underground: min z = {root_z.min():.3f}")
        if np.mean(root_z) < 0.3:
            warnings.append(f"Root very low: mean z = {np.mean(root_z):.3f}")
        if np.mean(root_z) > 2.0:
            warnings.append(f"Root very high: mean z = {np.mean(root_z):.3f}")

    # Quaternion norm check
    if body_quat_w.ndim == 3 and body_quat_w.shape[2] == 4:
        quat_norms = np.linalg.norm(body_quat_w, axis=2)
        if np.any(np.abs(quat_norms - 1.0) > 0.01):
            warnings.append(f"Quaternion norms deviate from 1.0: max deviation = {np.max(np.abs(quat_norms - 1.0)):.4f}")

    # Print results
    if verbose:
        status = "✅" if not errors else "❌"
        print(f"  {status} {filename}")
        print(f"     FPS: {fps}, Frames: {n_frames}, Duration: {duration:.2f}s")
        print(f"     joint_pos: {joint_pos.shape}, body_pos_w: {body_pos_w.shape}")

        if body_pos_w.ndim == 3:
            print(f"     Bodies tracked: {body_pos_w.shape[1]}, Root height: {body_pos_w[:, 0, 2].mean():.3f}m")

        for e in errors:
            print(f"     ❌ ERROR: {e}")
        for w in warnings:
            print(f"     ⚠️  WARNING: {w}")

    return len(errors) == 0


def main():
    parser = argparse.ArgumentParser(description="Validate martial arts NPZ data files.")
    parser.add_argument("--dir", "-d", type=str, required=True, help="Directory containing NPZ files.")
    parser.add_argument("--file", "-f", type=str, default=None, help="Validate a single NPZ file.")
    args = parser.parse_args()

    if args.file:
        files = [args.file]
    else:
        files = sorted(glob.glob(os.path.join(args.dir, "*.npz")))

    if not files:
        print(f"⚠️  No NPZ files found in: {args.dir}")
        print("\nExpected files:")
        expected = [
            "G1_front_kick.npz",
            "G1_roundhouse_kick.npz",
            "G1_side_kick.npz",
            "G1_lunge_punch.npz",
            "G1_heian_shodan.npz",
            "G1_bassai.npz",
            "G1_empi.npz",
        ]
        for f in expected:
            print(f"  - {f}")
        print("\nGenerate with: bash scripts/mimic/martial_arts_pipeline.sh npz")
        return

    print(f"Validating {len(files)} NPZ file(s) in: {args.dir}\n")

    valid = 0
    total = len(files)
    for f in files:
        if validate_npz(f):
            valid += 1
        print()

    print(f"{'=' * 50}")
    print(f"Results: {valid}/{total} files valid")
    if valid == total:
        print("✅ All files passed validation! Ready for training.")
    else:
        print("❌ Some files have errors. Please fix before training.")


if __name__ == "__main__":
    main()
