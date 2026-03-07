"""Martial Arts Motion Tracking Environment Configuration for G1 29DOF.

This module defines environment configurations for training the Unitree G1 robot
to perform individual martial arts movements (kata, kicks, punches).

Architecture — Policy Sequencer:
  Each motion is trained as an independent policy. At deployment time, the C++
  State_MartialArtsSequencer chains these policies back-to-back to produce a
  continuous martial arts performance (similar to 2026 Spring Festival Gala).

  Training:  7 independent tasks → 7 ONNX policies
  Deployment: Sequencer loads policies in order, transitions between segments

Data Source:
  CMU MoCap Database Subject #135 — Karate motions (Bassai, Empi, Heian Shodan,
  front kick, roundhouse kick, side kick, lunge punch).

Data Pipeline:
  CMU ASF+AMC → CSV (cmu_amc_to_csv.py) → NPZ (csv_to_npz.py) → Training
"""

from __future__ import annotations

import os

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import unitree_rl_lab.tasks.mimic.mdp as mdp

# Re-use the same G1 29DOF robot configuration as the gangnam_style task.
# All mimic tasks share the identical physical robot; only the motion data differs.
from unitree_rl_lab.tasks.mimic.robots.g1_29dof.gangnanm_style.g1 import (
    G1_ACTION_SCALE,
    G1_CYLINDER_CFG as ROBOT_CFG,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MOTION_DATA_DIR = os.path.dirname(__file__)

# Martial arts has wider velocity perturbation range for robustness
VELOCITY_RANGE = {
    "x": (-0.5, 0.5),
    "y": (-0.5, 0.5),
    "z": (-0.2, 0.2),
    "roll": (-0.52, 0.52),
    "pitch": (-0.52, 0.52),
    "yaw": (-0.78, 0.78),
}

# All 14 tracked body links on G1 (same as dance tasks)
TRACKED_BODY_NAMES = [
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
]

# End-effector body names (hands + feet) - critical for martial arts strikes
END_EFFECTOR_BODIES = [
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
]


# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------


@configclass
class RobotSceneCfg(InteractiveSceneCfg):
    """Scene configuration for martial arts training."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
            project_uvw=True,
        ),
    )
    robot: ArticulationCfg = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(color=(0.13, 0.13, 0.13), intensity=1000.0),
    )
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
        force_threshold=10.0,
        debug_vis=True,
    )


# ---------------------------------------------------------------------------
# MDP Components
# ---------------------------------------------------------------------------


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    JointPositionAction = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=G1_ACTION_SCALE, use_default_offset=True
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        motion_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "motion"})
        motion_anchor_ori_b = ObsTerm(
            func=mdp.motion_anchor_ori_b, params={"command_name": "motion"}, noise=Unoise(n_min=-0.05, n_max=0.05)
        )
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.5, n_max=0.5))
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class PrivilegedCfg(ObsGroup):
        command = ObsTerm(func=mdp.generated_commands, params={"command_name": "motion"})
        motion_anchor_pos_b = ObsTerm(func=mdp.motion_anchor_pos_b, params={"command_name": "motion"})
        motion_anchor_ori_b = ObsTerm(func=mdp.motion_anchor_ori_b, params={"command_name": "motion"})
        body_pos = ObsTerm(func=mdp.robot_body_pos_b, params={"command_name": "motion"})
        body_ori = ObsTerm(func=mdp.robot_body_ori_b, params={"command_name": "motion"})
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)

    policy: PolicyCfg = PolicyCfg()
    critic: PrivilegedCfg = PrivilegedCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.6),
            "dynamic_friction_range": (0.3, 1.2),
            "restitution_range": (0.0, 0.5),
            "num_buckets": 64,
        },
    )

    add_joint_default_pos = EventTerm(
        func=mdp.randomize_joint_default_pos,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "pos_distribution_params": (-0.01, 0.01),
            "operation": "add",
        },
    )

    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "com_range": {"x": (-0.025, 0.025), "y": (-0.05, 0.05), "z": (-0.05, 0.05)},
        },
    )

    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(2.0, 5.0),  # less frequent pushes for martial arts stability
        params={"velocity_range": VELOCITY_RANGE},
    )


# ---------------------------------------------------------------------------
# Reward Configuration for Martial Arts
# ---------------------------------------------------------------------------
#
# Tuning notes (v4 — added joint_pos tracking to fix bizarre posture):
#
#   Root cause of "weird posture" (姿势怪异):
#     body_pos tracks link CoMs (center of mass positions) — a robot can have
#     the right CoM trajectory while bending knees in completely wrong ways.
#     body_ori tracks link orientations — better, but 14 bodies still leaves
#     DOF for wrong joint-level configurations.
#     → NO reward term was directly penalizing wrong joint angles (e.g., wrong
#       knee bend depth, wrong ankle plantar/dorsiflexion).
#
#   Fix (v4): Add motion_joint_pos_error_exp
#     - Directly computes mean squared error over all 29 joint angles
#     - std=0.8 → at per-joint mean error of 0.8 rad: reward = exp(-0.64) ≈ 0.53
#     - weight=2.0 (same as body tracking) — joint angles are as important as
#       body positions for faithful martial arts reproduction
#
#   v3 fixes retained:
#   1. body_ori std: 0.4 → 0.8, weight: 1.0 → 1.5
#      → At error=0.79 rad: std=0.4 gives reward=0.020 (dead!), std=0.8 gives 0.377 (alive!)
#   2. body_pos std: 0.3 → 0.5  (covers 0.24 m initial error with margin)
#   3. Velocity weights: 1.0 → 0.5, stds widened (lin 1.0→1.5, ang 3.14→4.0)
#      → less dominant, won't fight position/orientation tracking
#
#   ee_body_pos termination relaxed (v4):
#     Front kick raises foot to ~0.8m height. z-only threshold 0.25m was
#     terminating episodes during the kick itself. Raised to 0.6m.
# ---------------------------------------------------------------------------


@configclass
class MartialArtsRewardsCfg:
    """Reward terms optimized for martial arts motion tracking (v3)."""

    # -- regularization
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    joint_torque = RewTerm(func=mdp.joint_torques_l2, weight=-1e-5)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.1)
    joint_limit = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-10.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )

    # -- anchor tracking (root body = torso_link)
    motion_global_anchor_pos = RewTerm(
        func=mdp.motion_global_anchor_position_error_exp,
        weight=0.5,
        params={"command_name": "motion", "std": 0.3},
    )
    motion_global_anchor_ori = RewTerm(
        func=mdp.motion_global_anchor_orientation_error_exp,
        weight=0.5,
        params={"command_name": "motion", "std": 0.8},  # ↑ from 0.4 (covers torso rotation error)
    )

    # -- full body tracking
    motion_body_pos = RewTerm(
        func=mdp.motion_relative_body_position_error_exp,
        weight=1.5,
        params={"command_name": "motion", "std": 0.5},  # ↑ from 0.3 (was causing vanishing grad)
    )
    motion_body_ori = RewTerm(
        func=mdp.motion_relative_body_orientation_error_exp,
        weight=1.5,  # ↑ from 1.0 (arms were completely ignored)
        params={"command_name": "motion", "std": 0.8},  # ↑ from 0.4 (covers 0.79 rad init error!)
    )

    # -- joint position tracking (v4: critical for correct martial arts posture)
    # Without this, policy can match body CoM positions with wrong knee/ankle configs.
    # std=0.8: mean per-joint error of 0.8rad → reward≈0.53 (alive gradient)
    motion_joint_pos = RewTerm(
        func=mdp.motion_joint_pos_error_exp,
        weight=2.0,
        params={"command_name": "motion", "std": 0.8},
    )

    # -- velocity tracking (widened std → less dominant vs position)
    motion_body_lin_vel = RewTerm(
        func=mdp.motion_global_body_linear_velocity_error_exp,
        weight=0.5,  # ↓ from 1.0 — don't fight position tracking
        params={"command_name": "motion", "std": 1.5},  # ↑ from 1.0
    )
    motion_body_ang_vel = RewTerm(
        func=mdp.motion_global_body_angular_velocity_error_exp,
        weight=0.5,  # ↓ from 1.0
        params={"command_name": "motion", "std": 4.0},  # ↑ from 3.14
    )

    # -- contact penalty
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[
                    r"^(?!left_ankle_roll_link$)(?!right_ankle_roll_link$)"
                    r"(?!left_wrist_yaw_link$)(?!right_wrist_yaw_link$).+$"
                ],
            ),
            "threshold": 1.0,
        },
    )


@configclass
class MartialArtsTerminationsCfg:
    """Termination terms for martial arts tasks.

    v2: Tightened thresholds to match gangnam_style — prevents bad poses
    from surviving long enough to pollute the training data.
    """

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # Z-only anchor position check — match gangnam (0.25)
    anchor_pos = DoneTerm(
        func=mdp.bad_anchor_pos_z_only,
        params={"command_name": "motion", "threshold": 0.25},  # ↓ from 0.3
    )
    anchor_ori = DoneTerm(
        func=mdp.bad_anchor_ori,
        params={"asset_cfg": SceneEntityCfg("robot"), "command_name": "motion", "threshold": 0.8},  # ↓ from 0.9
    )
    # End-effector tracking: terminate if hands/feet deviate too far
    # v4: Raised from 0.25 to 0.6 — front kick raises foot to ~0.8m,
    # a 0.25m z-threshold was terminating episodes mid-kick!
    ee_body_pos = DoneTerm(
        func=mdp.bad_motion_body_pos_z_only,
        params={
            "command_name": "motion",
            "threshold": 0.6,  # ↑ from 0.25 — allows feet to be raised during kicks
            "body_names": END_EFFECTOR_BODIES,
        },
    )


# ===========================================================================
# Helper: create a MotionCommandCfg for a given NPZ file
# ===========================================================================

def _make_command_cfg(npz_filename: str) -> mdp.MotionCommandCfg:
    """Create a MotionCommandCfg pointing to a specific motion file."""
    return mdp.MotionCommandCfg(
        asset_name="robot",
        motion_file=os.path.join(MOTION_DATA_DIR, npz_filename),
        anchor_body_name="torso_link",
        resampling_time_range=(1.0e9, 1.0e9),
        debug_vis=True,
        pose_range={
            "x": (-0.05, 0.05),
            "y": (-0.05, 0.05),
            "z": (-0.01, 0.01),
            "roll": (-0.1, 0.1),
            "pitch": (-0.1, 0.1),
            "yaw": (-0.2, 0.2),
        },
        velocity_range=VELOCITY_RANGE,
        joint_position_range=(-0.1, 0.1),
        body_names=TRACKED_BODY_NAMES,
        # Adaptive sampling: focus training on hard segments
        adaptive_kernel_size=3,
        adaptive_lambda=0.8,
        adaptive_uniform_ratio=0.1,
        adaptive_alpha=0.002,
    )


# ===========================================================================
# Base Environment Config
# ===========================================================================


@configclass
class MartialArtsBaseEnvCfg(ManagerBasedRLEnvCfg):
    """Base configuration shared by all martial arts tasks."""

    scene: RobotSceneCfg = RobotSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: MartialArtsRewardsCfg = MartialArtsRewardsCfg()
    terminations: MartialArtsTerminationsCfg = MartialArtsTerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum = None

    def __post_init__(self):
        """Post initialization."""
        self.decimation = 4
        self.episode_length_s = 30.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15


# ===========================================================================
# Per-Motion Task Configs
# ===========================================================================
# Each motion gets its own Env + Play config, pointing to its NPZ file.
# The NPZ file naming convention: G1_<motion_name>.npz
# ===========================================================================

# --- Heian Shodan (平安初段 - karate kata) ---

@configclass
class _HeianShodanCommandsCfg:
    motion = _make_command_cfg("G1_heian_shodan.npz")

@configclass
class HeianShodanEnvCfg(MartialArtsBaseEnvCfg):
    commands: _HeianShodanCommandsCfg = _HeianShodanCommandsCfg()

class HeianShodanPlayEnvCfg(HeianShodanEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.episode_length_s = 1e9


# --- Front Kick (前踢) ---

@configclass
class _FrontKickCommandsCfg:
    motion = _make_command_cfg("G1_front_kick.npz")

@configclass
class FrontKickEnvCfg(MartialArtsBaseEnvCfg):
    commands: _FrontKickCommandsCfg = _FrontKickCommandsCfg()

class FrontKickPlayEnvCfg(FrontKickEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.episode_length_s = 1e9


# --- Roundhouse Kick (回旋踢 / Mawashi Geri) ---

@configclass
class _RoundhouseKickCommandsCfg:
    motion = _make_command_cfg("G1_roundhouse_kick.npz")

@configclass
class RoundhouseKickEnvCfg(MartialArtsBaseEnvCfg):
    commands: _RoundhouseKickCommandsCfg = _RoundhouseKickCommandsCfg()

class RoundhouseKickPlayEnvCfg(RoundhouseKickEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.episode_length_s = 1e9


# --- Lunge Punch (冲拳 / Oi-Tsuki) ---

@configclass
class _LungePunchCommandsCfg:
    motion = _make_command_cfg("G1_lunge_punch.npz")

@configclass
class LungePunchEnvCfg(MartialArtsBaseEnvCfg):
    commands: _LungePunchCommandsCfg = _LungePunchCommandsCfg()

class LungePunchPlayEnvCfg(LungePunchEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.episode_length_s = 1e9


# --- Side Kick (侧踢 / Yoko Geri) ---

@configclass
class _SideKickCommandsCfg:
    motion = _make_command_cfg("G1_side_kick.npz")

@configclass
class SideKickEnvCfg(MartialArtsBaseEnvCfg):
    commands: _SideKickCommandsCfg = _SideKickCommandsCfg()

class SideKickPlayEnvCfg(SideKickEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.episode_length_s = 1e9


# --- Bassai (拔塞 - karate kata) ---

@configclass
class _BassaiCommandsCfg:
    motion = _make_command_cfg("G1_bassai.npz")

@configclass
class BassaiEnvCfg(MartialArtsBaseEnvCfg):
    commands: _BassaiCommandsCfg = _BassaiCommandsCfg()
    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 55.0  # Bassai is ~51s

class BassaiPlayEnvCfg(BassaiEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.episode_length_s = 1e9


# --- Empi (燕飛 - karate kata, CMU 135_02) ---

@configclass
class _EmpiCommandsCfg:
    motion = _make_command_cfg("G1_empi.npz")

@configclass
class EmpiEnvCfg(MartialArtsBaseEnvCfg):
    commands: _EmpiCommandsCfg = _EmpiCommandsCfg()
    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 50.0  # Empi is ~43s

class EmpiPlayEnvCfg(EmpiEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.episode_length_s = 1e9
