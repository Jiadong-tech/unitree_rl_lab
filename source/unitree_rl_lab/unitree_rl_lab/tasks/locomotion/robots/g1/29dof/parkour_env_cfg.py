import math

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, ViewerCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from unitree_rl_lab.assets.robots.unitree import UNITREE_G1_29DOF_CFG as ROBOT_CFG
from unitree_rl_lab.tasks.locomotion import mdp

# =================================================================================================
#  Terrain Configuration for Parkour (Gaps, Hurdles, Stepping Stones)
# =================================================================================================

PARKOUR_TERRAINS_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=5.0, # Reduced border width as requested
    num_rows=9,
    num_cols=20, # Adjusted to 20 to fit 10x0.1 proportions evenly
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    # Mix terrains by splitting into smaller chunks to interleave them (Random-like distribution)
    sub_terrains={
        "flat_warmup": terrain_gen.MeshPlaneTerrainCfg(proportion=0.1),
        
        "mix_gap_1": terrain_gen.MeshGapTerrainCfg(proportion=0.1, gap_width_range=(0.2, 0.8), platform_width=2.0),
        "mix_box_1": terrain_gen.MeshRandomGridTerrainCfg(proportion=0.1, grid_width=0.45, grid_height_range=(0.1, 0.4), platform_width=2.0),
        "mix_stair_1": terrain_gen.MeshPyramidStairsTerrainCfg(proportion=0.1, step_height_range=(0.05, 0.20), step_width=0.3, platform_width=3.0, border_width=1.0, holes=False),
        
        "mix_gap_2": terrain_gen.MeshGapTerrainCfg(proportion=0.1, gap_width_range=(0.3, 0.8), platform_width=2.0),
        "mix_box_2": terrain_gen.MeshRandomGridTerrainCfg(proportion=0.1, grid_width=0.45, grid_height_range=(0.15, 0.4), platform_width=2.0),
        "mix_stair_2": terrain_gen.MeshPyramidStairsTerrainCfg(proportion=0.1, step_height_range=(0.05, 0.20), step_width=0.3, platform_width=3.0, border_width=1.0, holes=False),
        
        "mix_gap_3": terrain_gen.MeshGapTerrainCfg(proportion=0.1, gap_width_range=(0.2, 0.8), platform_width=2.0),
        "mix_box_3": terrain_gen.MeshRandomGridTerrainCfg(proportion=0.1, grid_width=0.45, grid_height_range=(0.1, 0.4), platform_width=2.0),
        
        "flat_cooldown": terrain_gen.MeshPlaneTerrainCfg(proportion=0.1),
    },
)

@configclass
class RobotSceneCfg(InteractiveSceneCfg):
    """Configuration for the parkour scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=PARKOUR_TERRAINS_CFG,
        max_init_terrain_level=PARKOUR_TERRAINS_CFG.num_rows - 1,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    # robots
    robot: ArticulationCfg = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # sensors: ENABLE HEIGHT SCANNER FOR PARKOUR
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[4.0, 1.0]), # Optimized for 5m/s
        debug_vis=False, # Set to True to see the rays in GUI
        mesh_prim_paths=["/World/ground"],
    )
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


@configclass
class EventCfg:
    """Configuration for events."""
    # (Reuse standard physics material randomization)
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.0),
            "dynamic_friction_range": (0.3, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "mass_distribution_params": (-1.0, 3.0),
            "operation": "add",
        },
    )
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
                "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
            },
        },
    )
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (-1.0, 1.0),
        },
    )
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(5.0, 5.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""
    # Aggressive velocity commands for parkour
    base_velocity = mdp.UniformLevelVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=False,
        debug_vis=True,
        ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(0.5, 5.0), # 0.5 to 5.0 m/s
            lin_vel_y=(-0.1, 0.1), 
            ang_vel_z=(-0.5, 0.5)
        ),
        limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.5, 6.0), lin_vel_y=(-0.3, 0.3), ang_vel_z=(-0.5, 0.5)
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    JointPositionAction = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=0.25, use_default_offset=True
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # Proprioception
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, noise=Unoise(n_min=-1.5, n_max=1.5))
        last_action = ObsTerm(func=mdp.last_action)
        
        # Exteroception (Vision/Scanning) - CRITICAL FOR PARKOUR
        height_scanner = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-1.0, 5.0),
            noise=Unoise(n_min=-0.1, n_max=0.1), # Add noise to simulate real sensor imperfection
        )

        def __post_init__(self):
            self.history_length = 5
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    
    # We can keep Critic same as Policy for simplicity in PPO, or add privileged info
    critic: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # -- task (High weight on tracking to force obstacle traversal)
    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=3.0, # Increased to overpower penalties and force traversal
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=0.5, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )

    alive = RewTerm(func=mdp.is_alive, weight=1.0) # Bonus for staying alive on difficult terrain

    # -- penalties
    base_linear_velocity = RewTerm(func=mdp.lin_vel_z_l2, weight=0.0) # Disabled to allow jumping!
    base_angular_velocity = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    
    # Regularization
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-1.0e-7) # Reduced
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01) # Moderate penalty for 5m/s
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-5.0)
    
    # Cosmetical penalties
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[".*_shoulder_.*_joint", ".*_elbow_joint", ".*_wrist_.*"],
            )
        },
    )
    
    joint_deviation_waists = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=["waist.*"],
            )
        },
    )

    joint_deviation_legs = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_roll_joint", ".*_hip_yaw_joint"])},
    )

    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-0.2) # Relaxed to allow pitching for jumps
    
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.25,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
        },
    )

    # Important: Penalize collision with obstacles (except feet)
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="(?!.*ankle.*).*"), "threshold": 1.0},
    )

    # -- feet gait (Optional: can be disabled for pure parkour to allow free gait)
    gait = RewTerm(
        func=mdp.feet_gait,
        weight=1.0, # Keep it for rhythmic movement
        params={
            "period": 0.4, # ~2.5Hz, suited for 5m/s
            "offset": [0.0, 0.5],
            "threshold": 0.55,
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
        },
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    
    # Fall detection
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="torso_link"), "threshold": 1.0},
    )
    # Robot fallen/blown up check
    bad_orientation = DoneTerm(
        func=mdp.bad_orientation,
        params={"limit_angle": math.pi / 2}, # Kill if tilted more than 90 deg
    )
    base_height = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": 0.2}, # Kill if center of mass is too low (fallen)
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)


@configclass
class RobotEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the Parkour environment."""

    # Scene settings
    scene: RobotSceneCfg = RobotSceneCfg(num_envs=4096, env_spacing=2.5)
    # Viewer settings
    viewer: ViewerCfg = ViewerCfg(
        eye=(8.0, 0.0, 5.0),
        lookat=(0.0, 0.0, 1.0),
        origin_type="env",
    )
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        self.scene.contact_forces.update_period = self.sim.dt
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt

        # check if terrain levels curriculum is enabled - if so, enable curriculum for terrain generator
        # this generates terrains with increasing difficulty and is useful for training
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False


class RobotPlayEnvCfg(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.episode_length_s = 1e9
