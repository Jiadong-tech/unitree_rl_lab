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
# Bodies that must NOT touch the ground (for illegal_contact termination)
# ---------------------------------------------------------------------------
# Only feet (ankle_roll_link) and hands (wrist_yaw_link) are allowed ground
# contact. Every other articulation body should be in this list.
#
# IMPORTANT: body_names must match robot.body_names (the 30 articulation
# bodies exposed by PhysX).  Fixed-joint links like "pelvis_contour_link",
# "logo_link", "head_link" are merged into their parent by PhysX and do
# NOT appear as separate articulation bodies — using them here causes a
# runtime crash ("Available strings: [...]").
#
# The 30 articulation bodies are:
#   pelvis, left/right_hip_{pitch,roll,yaw}_link,
#   left/right_knee_link, left/right_ankle_{pitch,roll}_link,
#   waist_{yaw,roll}_link, torso_link,
#   left/right_shoulder_{pitch,roll,yaw}_link,
#   left/right_elbow_link, left/right_wrist_{roll,pitch,yaw}_link
#
# Allowed contacts (4): ankle_roll × 2, wrist_yaw × 2
# Illegal contacts: 30 − 4 = 26
ILLEGAL_CONTACT_BODIES = [
    # pelvis
    "pelvis",
    # legs — hip
    "left_hip_pitch_link",
    "right_hip_pitch_link",
    "left_hip_roll_link",
    "right_hip_roll_link",
    "left_hip_yaw_link",
    "right_hip_yaw_link",
    # legs — knee
    "left_knee_link",
    "right_knee_link",
    # legs — ankle_pitch (above foot, can touch during bad kicks)
    "left_ankle_pitch_link",
    "right_ankle_pitch_link",
    # waist + torso (logo_link / head_link are merged into torso by PhysX)
    "waist_yaw_link",
    "waist_roll_link",
    "torso_link",
    # arms — shoulder
    "left_shoulder_pitch_link",
    "right_shoulder_pitch_link",
    "left_shoulder_roll_link",
    "right_shoulder_roll_link",
    "left_shoulder_yaw_link",
    "right_shoulder_yaw_link",
    # arms — elbow
    "left_elbow_link",
    "right_elbow_link",
    # arms — wrist (roll and pitch; wrist_yaw is the "hand" → allowed)
    "left_wrist_roll_link",
    "right_wrist_roll_link",
    "left_wrist_pitch_link",
    "right_wrist_pitch_link",
]


# ---------------------------------------------------------------------------
# Reward Configuration for Martial Arts
# ---------------------------------------------------------------------------
#
# Tuning notes (v9 — fix bad postures: knees on ground, twisted arms):
#
#   v9 ROOT CAUSE ANALYSIS (from play.py observations):
#     v8 trained model exhibits:
#       - 膝盖着地 (knees touching ground) — no contact termination existed
#       - 胳膊拧过头 (arms twisted unnaturally) — no joint_pos tracking reward
#       - 手势和腿部姿势怪异 (weird hand/leg poses) — degenerate FK solutions
#
#     Root causes:
#       1. motion_joint_pos_error_exp was REMOVED in v8, but it is CRITICAL
#          for martial arts. Body position tracking (CoM of links) allows
#          degenerate solutions: the policy can achieve the same link CoM
#          positions with completely wrong joint configurations.
#          The function's own docstring warns: "without joint_pos reward the
#          policy can achieve similar body positions with completely wrong
#          knee/ankle/arm configurations (姿势怪异)"
#       2. No body-ground contact TERMINATION existed. Only a weak penalty
#          (undesired_contacts, weight=-0.1) that the policy learns to ignore.
#       3. undesired_contacts penalty weight=-0.1 is too small to prevent
#          the policy from exploiting ground contact for stability.
#
#   v9 FIXES:
#     1. RE-ENABLE motion_joint_pos_error_exp (weight=2.0, std=0.8)
#        — Forces policy to match reference joint angles, not just link CoMs
#        — This is THE key fix for twisted arms and weird poses
#     2. ADD illegal_contact termination for knees, elbows, torso, pelvis, head
#        — Hard termination: episode ends immediately on forbidden contact
#        — Uses Isaac Lab's built-in mdp.illegal_contact
#     3. INCREASE undesired_contacts penalty from -0.1 to -1.0
#        — Stronger gradient signal against any non-foot/hand ground contact
#
#   PRESERVED from v8:
#     - Relaxed termination thresholds (anchor_pos=0.5, anchor_ori=1.2,
#       ee_body_pos=0.8) — these correctly fixed v7's training collapse
#     - Body/velocity tracking weights and stds
#     - Action rate penalty at -0.1
#
#   HISTORY:
#     v9: fix bad postures (re-enable joint_pos, add contact termination)
#     v8: fix training collapse (relax terminations, align with gangnam)
#     v7: balanced tracking, BUT anchor_pos termination too strict → collapse
#     v6: joint-only (body data was corrupted by fix_npz_motion_quality.py)
#     v5: joint+body but body data was inconsistent → jitter
#     v4: added joint_pos_error_exp
#     v3: body position/orientation tuning
# ---------------------------------------------------------------------------


@configclass
class MartialArtsRewardsCfg:
    """Reward terms — v9: fix bad postures (knees on ground, twisted arms)."""

    # -- regularization (match gangnam proven values)
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
        weight=1.5,
        params={"command_name": "motion", "std": 0.5},
    )
    motion_global_anchor_ori = RewTerm(
        func=mdp.motion_global_anchor_orientation_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 0.5},
    )

    # -- full body tracking
    motion_body_pos = RewTerm(
        func=mdp.motion_relative_body_position_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 0.3},
    )
    motion_body_ori = RewTerm(
        func=mdp.motion_relative_body_orientation_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 0.4},
    )

    # -- joint position tracking (v9: RE-ENABLED — critical for martial arts!)
    # Without this, body_pos/ori only tracks link CoMs. The policy finds
    # degenerate FK solutions: correct CoM positions but completely wrong
    # joint angles → twisted arms, weird knee/ankle configurations.
    # This was the #1 root cause of "胳膊拧过头" and "姿势怪异".
    motion_joint_pos = RewTerm(
        func=mdp.motion_joint_pos_error_exp,
        weight=2.0,  # v9: re-enabled (was 3.0 in v7, removed in v8)
        params={"command_name": "motion", "std": 0.8},
    )

    # -- end-effector tracking (relaxed for dynamic motions)
    motion_ee_pos = RewTerm(
        func=mdp.motion_relative_body_position_error_exp,
        weight=1.5,
        params={
            "command_name": "motion",
            "std": 0.3,
            "body_names": END_EFFECTOR_BODIES,
        },
    )

    # -- velocity tracking
    motion_body_lin_vel = RewTerm(
        func=mdp.motion_global_body_linear_velocity_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 1.5},
    )
    motion_body_ang_vel = RewTerm(
        func=mdp.motion_global_body_angular_velocity_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 3.14},
    )

    # -- contact penalty (v9: weight -0.1 → -1.0 — much stronger!)
    # v8's -0.1 was too weak; the policy ignored it and put knees on ground.
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,  # v9: 10x stronger penalty for forbidden body contacts
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

    v9: Added illegal_contact termination for body parts that must not
    touch the ground (knees, elbows, torso, pelvis, waist, hips).
    This is a hard termination — episode ends immediately on contact.

    Note: body_names must be actual articulation bodies (the 30 bodies
    exposed by PhysX). Fixed-joint links like pelvis_contour_link,
    logo_link, head_link are merged into their parents and cannot be
    used here.
    """

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # Z-only anchor position check (v8: relaxed to 0.5m for kicks)
    anchor_pos = DoneTerm(
        func=mdp.bad_anchor_pos_z_only,
        params={"command_name": "motion", "threshold": 0.5},
    )
    # Orientation tolerance (v8: relaxed to 1.2 for spinning kicks)
    anchor_ori = DoneTerm(
        func=mdp.bad_anchor_ori,
        params={"asset_cfg": SceneEntityCfg("robot"), "command_name": "motion", "threshold": 1.2},
    )
    # End-effector tracking (v8: relaxed to 0.8 for kick trajectories)
    ee_body_pos = DoneTerm(
        func=mdp.bad_motion_body_pos_z_only,
        params={
            "command_name": "motion",
            "threshold": 0.8,
            "body_names": END_EFFECTOR_BODIES,
        },
    )

    # v9 NEW: Illegal body contact termination
    # Terminate immediately if forbidden body parts touch the ground.
    # This prevents: 膝盖着地 (knees on ground), 胳膊着地 (elbows on ground),
    # torso/pelvis collapse.
    # Only feet (ankle_roll_link) and hands (wrist_yaw_link) are allowed
    # ground contact during martial arts.
    #
    # NOTE: body_names must be actual articulation bodies (30 bodies from
    # PhysX). Fixed-joint links (pelvis_contour_link, logo_link, head_link)
    # are merged into parents and must NOT be used here.
    illegal_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=ILLEGAL_CONTACT_BODIES,
            ),
            "threshold": 100.0,  # N — generous threshold to avoid false positives
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
