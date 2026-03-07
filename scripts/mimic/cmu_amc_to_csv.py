"""Parse CMU MoCap ASF+AMC files and convert to G1 29DOF CSV format.

This script directly converts CMU ASF/AMC data to CSV compatible with csv_to_npz.py,
without needing Blender or BVH as an intermediate step.

Usage:
    python scripts/mimic/cmu_amc_to_csv.py \
        --asf <path_to_asf> \
        --amc <path_to_amc> \
        --output <path_to_csv> \
        [--fps 120] \
        [--visualize]

CMU MoCap Subject #135 (Martial Arts) file mapping:
    135_01.amc → Bassai (拔塞) - 49.3s
    135_02.amc → Empi (燕飛) - 42.0s
    135_03.amc → Front Kick (前踢) - 22.2s
    135_04.amc → Heian Shodan (平安初段) - 10.6s
    135_05.amc → Mawashi Geri (回旋踢) - 19.8s
    135_06.amc → Oi-Tsuki (追突/冲拳) - 26.3s
    135_07.amc → Yoko Geri (侧踢) - 11.9s

CSV Format (per row):
    base_pos_x, base_pos_y, base_pos_z,
    base_quat_x, base_quat_y, base_quat_z, base_quat_w,
    joint_1, ..., joint_29

Dependencies:
    numpy, scipy (for rotation math)
"""

import argparse
import csv
import math
import os
import re
import sys
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation


def wrap_to_pi(angle: float) -> float:
    """Wrap angle to [-π, π]."""
    return (angle + math.pi) % (2 * math.pi) - math.pi


# ============================================================================
# ASF Parser
# ============================================================================

@dataclass
class Bone:
    """A single bone in the ASF skeleton."""
    id: int = 0
    name: str = ""
    direction: np.ndarray = field(default_factory=lambda: np.zeros(3))
    length: float = 0.0
    axis: np.ndarray = field(default_factory=lambda: np.zeros(3))
    axis_order: str = "XYZ"
    dof: list = field(default_factory=list)  # e.g. ["rx", "ry", "rz"]
    limits: list = field(default_factory=list)
    parent: str = ""
    children: list = field(default_factory=list)


@dataclass
class Skeleton:
    """ASF skeleton definition."""
    name: str = "VICON"
    length_unit: float = 0.45  # inches to ... CMU uses 0.45
    angle_unit: str = "deg"
    root_order: str = "TX TY TZ RX RY RZ"
    root_position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    root_orientation: np.ndarray = field(default_factory=lambda: np.zeros(3))
    bones: dict = field(default_factory=dict)
    hierarchy: dict = field(default_factory=dict)  # parent -> [children]


def parse_asf(filepath: str) -> Skeleton:
    """Parse an ASF skeleton file."""
    skeleton = Skeleton()

    with open(filepath, 'r') as f:
        lines = f.readlines()

    i = 0
    current_section = None

    while i < len(lines):
        line = lines[i].strip()
        i += 1

        # Skip comments and empty lines
        if not line or line.startswith('#'):
            continue

        # Section headers
        if line.startswith(':'):
            current_section = line.split()[0][1:]  # remove ':'
            if current_section == 'units':
                continue
            elif current_section == 'root':
                continue
            elif current_section == 'bonedata':
                continue
            elif current_section == 'hierarchy':
                continue
            else:
                continue

        # Parse units
        if current_section == 'units':
            parts = line.split()
            if len(parts) >= 2:
                if parts[0] == 'length':
                    skeleton.length_unit = float(parts[1])
                elif parts[0] == 'angle':
                    skeleton.angle_unit = parts[1]

        # Parse root
        elif current_section == 'root':
            parts = line.split()
            if parts[0] == 'order':
                skeleton.root_order = ' '.join(parts[1:])
            elif parts[0] == 'position':
                skeleton.root_position = np.array([float(x) for x in parts[1:4]])
            elif parts[0] == 'orientation':
                skeleton.root_orientation = np.array([float(x) for x in parts[1:4]])

        # Parse bone data
        elif current_section == 'bonedata':
            if line == 'begin':
                bone = Bone()
                while i < len(lines):
                    bline = lines[i].strip()
                    i += 1
                    if bline == 'end':
                        skeleton.bones[bone.name] = bone
                        break
                    parts = bline.split()
                    if not parts:
                        continue
                    if parts[0] == 'id':
                        bone.id = int(parts[1])
                    elif parts[0] == 'name':
                        bone.name = parts[1]
                    elif parts[0] == 'direction':
                        bone.direction = np.array([float(x) for x in parts[1:4]])
                    elif parts[0] == 'length':
                        bone.length = float(parts[1])
                    elif parts[0] == 'axis':
                        bone.axis = np.array([float(x) for x in parts[1:4]])
                        if len(parts) > 4:
                            bone.axis_order = parts[4]
                    elif parts[0] == 'dof':
                        bone.dof = parts[1:]
                    elif parts[0] == 'limits':
                        # Parse limits - may span multiple lines
                        lim_str = bline.replace('limits', '').strip()
                        lim_pairs = re.findall(r'\(([-\d.e+]+)\s+([-\d.e+]+)\)', lim_str)
                        bone.limits = [(float(a), float(b)) for a, b in lim_pairs]
                        # Check for additional limit lines
                        while len(bone.limits) < len(bone.dof):
                            if i < len(lines):
                                next_line = lines[i].strip()
                                extra_pairs = re.findall(r'\(([-\d.e+]+)\s+([-\d.e+]+)\)', next_line)
                                if extra_pairs:
                                    bone.limits.extend([(float(a), float(b)) for a, b in extra_pairs])
                                    i += 1
                                else:
                                    break
                            else:
                                break

        # Parse hierarchy
        elif current_section == 'hierarchy':
            if line in ('begin', 'end'):
                continue
            parts = line.split()
            if len(parts) >= 2:
                parent = parts[0]
                children = parts[1:]
                skeleton.hierarchy[parent] = children
                for child in children:
                    if child in skeleton.bones:
                        skeleton.bones[child].parent = parent

    return skeleton


# ============================================================================
# AMC Parser
# ============================================================================

@dataclass
class AMCFrame:
    """A single frame of AMC motion data."""
    frame_num: int = 0
    bone_data: dict = field(default_factory=dict)  # bone_name -> [values]


def parse_amc(filepath: str) -> list[AMCFrame]:
    """Parse an AMC motion file. Returns list of frames."""
    frames = []

    with open(filepath, 'r') as f:
        lines = f.readlines()

    i = 0
    # Skip header lines
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        if line.startswith(':') or line.startswith('#'):
            continue
        # First numeric line is frame number
        if line.isdigit():
            i -= 1
            break

    current_frame = None
    while i < len(lines):
        line = lines[i].strip()
        i += 1

        if not line:
            continue

        # Check if this is a frame number
        if line.isdigit():
            if current_frame is not None:
                frames.append(current_frame)
            current_frame = AMCFrame(frame_num=int(line))
        else:
            # Bone data line: bone_name val1 val2 ...
            parts = line.split()
            bone_name = parts[0]
            values = [float(x) for x in parts[1:]]
            if current_frame is not None:
                current_frame.bone_data[bone_name] = values

    if current_frame is not None:
        frames.append(current_frame)

    return frames


# ============================================================================
# Forward Kinematics
# ============================================================================

# G1 robot standing pelvis height (from Isaac Sim default root state)
G1_PELVIS_HEIGHT = 0.78  # meters

# CMU_TO_G1_METERS is auto-calibrated per skeleton in calibrate_scale().
# It maps 1 CMU raw unit → meters at G1 scale.
# Initialized to a sane default; overwritten before actual use.
CMU_TO_G1_METERS: float = 0.054  # placeholder, recalculated per-skeleton


def calibrate_scale(skeleton: Skeleton) -> float:
    """Auto-calibrate CMU → G1 meters scale from skeleton leg chain.

    The CMU ASF unit system is arbitrary per subject.  We calibrate by
    measuring the skeleton's left leg chain (lfemur + ltibia) which
    corresponds to the hip-to-ankle distance, and mapping the root TY
    (pelvis height when standing) to G1_PELVIS_HEIGHT.

    Strategy:
      1. Compute leg_chain = lfemur.length + ltibia.length  (raw units)
      2. Empirically, standing root TY ≈ leg_chain (pelvis ≈ hip-to-ankle)
      3. Scale = G1_PELVIS_HEIGHT / leg_chain

    This makes the conversion independent of CMU's `:units length` value
    and works across different subjects.
    """
    lfemur = skeleton.bones.get("lfemur")
    ltibia = skeleton.bones.get("ltibia")
    rfemur = skeleton.bones.get("rfemur")
    rtibia = skeleton.bones.get("rtibia")

    if lfemur and ltibia:
        leg_chain = lfemur.length + ltibia.length
    elif rfemur and rtibia:
        leg_chain = rfemur.length + rtibia.length
    else:
        # Fallback: assume 1 unit ≈ 5.4cm (typical for CMU data)
        print("[WARN] Cannot find femur+tibia bones, using fallback scale")
        return G1_PELVIS_HEIGHT / 14.4

    scale = G1_PELVIS_HEIGHT / leg_chain
    print(f"  [calibrate] leg_chain = {leg_chain:.3f} units → scale = {scale:.5f} m/unit")
    print(f"  [calibrate] Expected pelvis height ≈ {leg_chain * scale:.3f} m")
    return scale


def euler_to_rotation(angles_deg: np.ndarray, order: str = "XYZ") -> Rotation:
    """Convert Euler angles (degrees) to scipy Rotation."""
    # scipy uses lowercase axis names
    return Rotation.from_euler(order.lower(), angles_deg, degrees=True)


def compute_bone_local_rotation(bone: Bone, frame_data: list) -> Rotation:
    """Compute the local rotation of a bone given AMC frame data.

    CMU convention:
        R_local = R_axis * R_amc * R_axis_inv
    where R_axis is the bone's rest-pose axis rotation.
    """
    # Bone axis rotation (rest pose)
    R_axis = euler_to_rotation(bone.axis, bone.axis_order)

    # Build AMC rotation from DOF values
    amc_angles = np.zeros(3)
    dof_map = {"rx": 0, "ry": 1, "rz": 2}
    for idx, dof_name in enumerate(bone.dof):
        if dof_name in dof_map and idx < len(frame_data):
            amc_angles[dof_map[dof_name]] = frame_data[idx]

    R_amc = euler_to_rotation(amc_angles, "XYZ")

    # Final local rotation: R_axis * R_amc * R_axis_inv
    R_local = R_axis * R_amc * R_axis.inv()
    return R_local


def forward_kinematics(
    skeleton: Skeleton,
    frame: AMCFrame,
    scale: float | None = None,
) -> tuple[np.ndarray, Rotation, dict]:
    """Compute forward kinematics for a single frame.

    Args:
        skeleton: Parsed ASF skeleton.
        frame: One AMC frame.
        scale: Meters-per-CMU-unit.  Uses global CMU_TO_G1_METERS if None.

    Returns:
        root_pos: (3,) root position in meters
        root_rot: Rotation of root
        bone_world_transforms: dict of bone_name -> (position, Rotation)
    """
    s = scale if scale is not None else CMU_TO_G1_METERS
    # Root data
    root_data = frame.bone_data.get("root", [0, 0, 0, 0, 0, 0])
    root_pos = np.array(root_data[:3]) * s
    root_euler = np.array(root_data[3:6])
    root_rot = euler_to_rotation(root_euler, "XYZ")

    # Traverse hierarchy and compute world transforms
    bone_transforms = {}  # bone_name -> (world_pos, world_rot)

    def traverse(parent_name, parent_pos, parent_rot):
        children = skeleton.hierarchy.get(parent_name, [])
        for child_name in children:
            if child_name not in skeleton.bones:
                continue
            bone = skeleton.bones[child_name]

            # Compute child local rotation
            child_data = frame.bone_data.get(child_name, [])
            if bone.dof and child_data:
                R_local = compute_bone_local_rotation(bone, child_data)
            else:
                R_local = Rotation.identity()

            # Child world rotation
            child_rot = parent_rot * R_local

            # Child world position: parent_pos + parent_rot * (bone.direction * bone.length)
            offset = bone.direction * bone.length * s
            child_pos = parent_pos + parent_rot.apply(offset)

            bone_transforms[child_name] = (child_pos, child_rot)
            traverse(child_name, child_pos, child_rot)

    # Start from root
    bone_transforms["root"] = (root_pos, root_rot)
    traverse("root", root_pos, root_rot)

    return root_pos, root_rot, bone_transforms


# ============================================================================
# CMU → G1 Joint Mapping
# ============================================================================
# CMU skeleton has 30 bones. G1 has 29 joints.
# We map CMU bone rotations to G1 joint angles.
#
# The mapping extracts specific Euler angle components from CMU bone rotations
# and maps them to G1 joint angles (in radians).
# ============================================================================

# (cmu_bone_name, cmu_euler_axis, sign, g1_joint_index)
# cmu_euler_axis: which Euler axis to extract (0=X, 1=Y, 2=Z) from the local rotation
CMU_TO_G1_MAPPING = [
    # Left leg (G1 joints 0-5)
    ("lfemur",  0, -1.0, 0),   # left_hip_pitch
    ("lfemur",  2, -1.0, 1),   # left_hip_roll   [FIX: +1.0→-1.0, was producing negative values outside [-0.52,2.97]]
    ("lfemur",  1,  1.0, 2),   # left_hip_yaw
    ("ltibia",  0,  1.0, 3),   # left_knee       [FIX: -1.0→+1.0, knee must be positive when bent]
    ("lfoot",   0,  1.0, 4),   # left_ankle_pitch [FIX: -1.0→+1.0]
    ("lfoot",   2,  1.0, 5),   # left_ankle_roll

    # Right leg (G1 joints 6-11)
    ("rfemur",  0, -1.0, 6),   # right_hip_pitch
    ("rfemur",  2, -1.0, 7),   # right_hip_roll
    ("rfemur",  1, -1.0, 8),   # right_hip_yaw
    ("rtibia",  0,  1.0, 9),   # right_knee      [FIX: -1.0→+1.0, knee must be positive when bent]
    ("rfoot",   0,  1.0, 10),  # right_ankle_pitch [FIX: -1.0→+1.0]
    ("rfoot",   2, -1.0, 11),  # right_ankle_roll

    # Waist (G1 joints 12-14) - map from lowerback/upperback
    ("lowerback", 1,  1.0, 12),  # waist_yaw
    ("lowerback", 2,  1.0, 13),  # waist_roll
    ("lowerback", 0, -1.0, 14),  # waist_pitch

    # Left arm (G1 joints 15-21)
    ("lhumerus", 0, -1.0, 15),  # left_shoulder_pitch
    ("lhumerus", 2,  1.0, 16),  # left_shoulder_roll
    ("lhumerus", 1,  1.0, 17),  # left_shoulder_yaw
    ("lradius",  0,  1.0, 18),  # left_elbow  [FIX: -1.0→+1.0, elbow bend is positive]
    ("lwrist",   1,  1.0, 19),  # left_wrist_roll
    ("lhand",    0, -1.0, 20),  # left_wrist_pitch
    ("lhand",    2,  1.0, 21),  # left_wrist_yaw

    # Right arm (G1 joints 22-28)
    ("rhumerus", 0, -1.0, 22),  # right_shoulder_pitch
    ("rhumerus", 2,  1.0, 23),  # right_shoulder_roll [FIX: -1.0→+1.0, was producing [1.02,1.58] outside [-2.09,0.087]]
    ("rhumerus", 1, -1.0, 24),  # right_shoulder_yaw
    ("rradius",  0,  1.0, 25),  # right_elbow [FIX: -1.0→+1.0, elbow bend is positive]
    ("rwrist",   1, -1.0, 26),  # right_wrist_roll
    ("rhand",    0, -1.0, 27),  # right_wrist_pitch
    ("rhand",    2, -1.0, 28),  # right_wrist_yaw
]


def extract_g1_joint_angles(skeleton: Skeleton, frame: AMCFrame) -> np.ndarray:
    """Extract G1 29DOF joint angles from a CMU AMC frame.

    Returns:
        joint_angles: (29,) array in radians
    """
    joint_angles = np.zeros(29)

    for cmu_bone, euler_axis, sign, g1_idx in CMU_TO_G1_MAPPING:
        if cmu_bone not in skeleton.bones:
            continue

        bone = skeleton.bones[cmu_bone]
        frame_data = frame.bone_data.get(cmu_bone, [])

        if not bone.dof or not frame_data:
            continue

        # Get the AMC values for this bone's DOFs
        dof_map = {"rx": 0, "ry": 1, "rz": 2}
        amc_angles = [0.0, 0.0, 0.0]
        for idx, dof_name in enumerate(bone.dof):
            if dof_name in dof_map and idx < len(frame_data):
                amc_angles[dof_map[dof_name]] = frame_data[idx]

        # Convert degrees to radians and wrap to [-π, π]
        angle_rad = math.radians(amc_angles[euler_axis]) * sign
        joint_angles[g1_idx] = wrap_to_pi(angle_rad)

    # Clip to G1 29DOF URDF joint limits (from g1_29dof_rev_1_0.urdf)
    G1_LIMITS = [
        (-2.531, 2.880),  # 0  left_hip_pitch
        (-0.524, 2.967),  # 1  left_hip_roll
        (-2.758, 2.758),  # 2  left_hip_yaw
        (-0.087, 2.880),  # 3  left_knee
        (-0.873, 0.524),  # 4  left_ankle_pitch
        (-0.262, 0.262),  # 5  left_ankle_roll
        (-2.531, 2.880),  # 6  right_hip_pitch
        (-2.967, 0.524),  # 7  right_hip_roll
        (-2.758, 2.758),  # 8  right_hip_yaw
        (-0.087, 2.880),  # 9  right_knee
        (-0.873, 0.524),  # 10 right_ankle_pitch
        (-0.262, 0.262),  # 11 right_ankle_roll
        (-2.618, 2.618),  # 12 waist_yaw
        (-0.520, 0.520),  # 13 waist_roll
        (-0.520, 0.520),  # 14 waist_pitch
        (-3.089, 2.670),  # 15 left_shoulder_pitch
        (-1.588, 2.252),  # 16 left_shoulder_roll
        (-2.618, 2.618),  # 17 left_shoulder_yaw
        (-1.047, 2.094),  # 18 left_elbow
        (-3.089, 2.670),  # 19 left_wrist_roll
        (-1.614, 1.614),  # 20 left_wrist_pitch
        (-1.614, 1.614),  # 21 left_wrist_yaw
        (-2.670, 3.089),  # 22 right_shoulder_pitch
        (-2.252, 1.588),  # 23 right_shoulder_roll
        (-2.618, 2.618),  # 24 right_shoulder_yaw
        (-1.047, 2.094),  # 25 right_elbow
        (-2.670, 3.089),  # 26 right_wrist_roll
        (-1.614, 1.614),  # 27 right_wrist_pitch
        (-1.614, 1.614),  # 28 right_wrist_yaw
    ]
    for j, (lo, hi) in enumerate(G1_LIMITS):
        joint_angles[j] = np.clip(joint_angles[j], lo, hi)

    return joint_angles


def extract_root_pose(frame: AMCFrame, scale: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Extract root position and quaternion from AMC frame.

    Applies coordinate transform: CMU (X-right, Y-up, Z-back) → Isaac (X-forward, Y-left, Z-up)
    and scales position using the calibrated CMU→G1 scale factor.

    Args:
        frame: AMC frame data.
        scale: Meters-per-CMU-unit.  Uses global CMU_TO_G1_METERS if None.

    Returns:
        pos: (3,) position in meters (Isaac Z-up coordinate)
        quat: (4,) quaternion in xyzw format (for CSV compatibility)
    """
    s = scale if scale is not None else CMU_TO_G1_METERS
    root_data = frame.bone_data.get("root", [0, 0, 0, 0, 0, 0])
    cmu_pos = np.array(root_data[:3]) * s
    euler_deg = np.array(root_data[3:6])

    # Coordinate transform: CMU (X, Y, Z) → Isaac (−Z, −X, Y)
    # CMU: X=right, Y=up, Z=back
    # Isaac: X=forward, Y=left, Z=up
    isaac_pos = np.array([
        -cmu_pos[2],   # Isaac X (forward) = -CMU Z (back)
        -cmu_pos[0],   # Isaac Y (left)    = -CMU X (right)
         cmu_pos[1],   # Isaac Z (up)      =  CMU Y (up)
    ])

    # Rotation: apply the same coordinate frame rotation
    # First build CMU rotation
    R_cmu = euler_to_rotation(euler_deg, "XYZ")
    # Coordinate frame rotation: CMU → Isaac
    # This rotates the axes: X→-Z, Y→-X, Z→Y in CMU → -Z→X, -X→Y, Y→Z in Isaac
    R_coord = Rotation.from_matrix(np.array([
        [0, 0, -1],   # Isaac X from -CMU Z
        [-1, 0, 0],   # Isaac Y from -CMU X
        [0, 1, 0],    # Isaac Z from  CMU Y
    ]))
    R_isaac = R_coord * R_cmu * R_coord.inv()

    quat_xyzw = R_isaac.as_quat()  # scipy returns xyzw by default

    return isaac_pos, quat_xyzw


# ============================================================================
# Main conversion
# ============================================================================

def convert_amc_to_csv(
    asf_path: str,
    amc_path: str,
    output_path: str,
    fps: int = 120,
):
    """Convert CMU ASF+AMC to G1-compatible CSV.

    Handles:
      - Auto-calibrated position scale from skeleton leg chain → G1 pelvis height
      - Coordinate system: CMU Y-up → Isaac Z-up
      - Angle wrapping: all joint angles wrapped to [-π, π]

    Args:
        asf_path: Path to .asf skeleton file
        amc_path: Path to .amc motion file
        output_path: Path to output .csv file
        fps: Frame rate of the AMC data (CMU default: 120)
    """
    global CMU_TO_G1_METERS

    print(f"[cmu_to_csv] Parsing ASF: {asf_path}")
    skeleton = parse_asf(asf_path)
    print(f"  Bones: {len(skeleton.bones)}")
    print(f"  Angle unit: {skeleton.angle_unit}")
    print(f"  Length unit: {skeleton.length_unit}")

    # Auto-calibrate position scale from skeleton anatomy
    CMU_TO_G1_METERS = calibrate_scale(skeleton)

    print(f"[cmu_to_csv] Parsing AMC: {amc_path}")
    frames = parse_amc(amc_path)
    print(f"  Frames: {len(frames)}")
    print(f"  Duration: {len(frames) / fps:.1f}s @ {fps}fps")

    # Convert each frame
    rows = []
    for frame in frames:
        pos, quat_xyzw = extract_root_pose(frame, scale=CMU_TO_G1_METERS)
        joint_angles = extract_g1_joint_angles(skeleton, frame)

        row = list(pos) + list(quat_xyzw) + list(joint_angles)
        rows.append(row)

    # Write CSV
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        for row in rows:
            writer.writerow([f"{v:.6f}" for v in row])

    print(f"[cmu_to_csv] Written: {output_path}")
    print(f"  Columns: 3 (pos) + 4 (quat) + 29 (joints) = 36")
    print(f"  Rows: {len(rows)}")

    # Root position stats (Isaac Z-up coordinate)
    all_rows = np.array(rows)
    root_pos = all_rows[:, :3]
    print(f"\n[cmu_to_csv] Root position (Isaac Z-up):")
    print(f"  X (forward): [{root_pos[:,0].min():.3f}, {root_pos[:,0].max():.3f}]")
    print(f"  Y (left):    [{root_pos[:,1].min():.3f}, {root_pos[:,1].max():.3f}]")
    print(f"  Z (height):  [{root_pos[:,2].min():.3f}, {root_pos[:,2].max():.3f}]  (G1 standing ~0.78m)")

    print(f"\n[cmu_to_csv] Next step:")
    print(f"  python scripts/mimic/csv_to_npz.py -f {output_path} --input_fps {fps} --output_fps 50")

    # Print joint angle statistics for debugging
    all_joints = np.array([r[7:] for r in rows])
    print(f"\n[cmu_to_csv] Joint angle statistics (radians):")
    print(f"  {'Joint':<8} {'Min':>8} {'Max':>8} {'Mean':>8} {'Std':>8}")
    print(f"  {'-' * 40}")
    g1_names = [
        "L_hip_p", "L_hip_r", "L_hip_y", "L_knee", "L_ank_p", "L_ank_r",
        "R_hip_p", "R_hip_r", "R_hip_y", "R_knee", "R_ank_p", "R_ank_r",
        "W_yaw", "W_roll", "W_pitch",
        "L_sh_p", "L_sh_r", "L_sh_y", "L_elbow", "L_wr_r", "L_wr_p", "L_wr_y",
        "R_sh_p", "R_sh_r", "R_sh_y", "R_elbow", "R_wr_r", "R_wr_p", "R_wr_y",
    ]
    for j in range(min(29, all_joints.shape[1])):
        col = all_joints[:, j]
        name = g1_names[j] if j < len(g1_names) else f"j{j}"
        print(f"  {name:<8} {col.min():>8.3f} {col.max():>8.3f} {col.mean():>8.3f} {col.std():>8.3f}")


def main():
    parser = argparse.ArgumentParser(description="Convert CMU ASF+AMC to G1 CSV.")
    parser.add_argument("--asf", type=str, required=True, help="Path to .asf skeleton file.")
    parser.add_argument("--amc", type=str, required=True, help="Path to .amc motion file.")
    parser.add_argument("--output", "-o", type=str, required=True, help="Path to output .csv file.")
    parser.add_argument("--fps", type=int, default=120, help="Frame rate (CMU default: 120).")
    args = parser.parse_args()

    convert_amc_to_csv(
        asf_path=args.asf,
        amc_path=args.amc,
        output_path=args.output,
        fps=args.fps,
    )


if __name__ == "__main__":
    main()
