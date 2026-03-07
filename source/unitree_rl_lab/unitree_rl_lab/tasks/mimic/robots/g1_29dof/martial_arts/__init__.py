import gymnasium as gym

# === 武术动作任务 (每个动作独立训练一个 policy) ===
# 训练完成后, 通过 C++ Policy Sequencer 在部署端串联成完整表演。
# 详见 deploy/include/FSM/State_MartialArtsSequencer.h

# 空手道套路 - 平安初段 (Heian Shodan)
gym.register(
    id="Unitree-G1-29dof-Mimic-MartialArts-HeianShodan",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tracking_env_cfg:HeianShodanEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.tracking_env_cfg:HeianShodanPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.mimic.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)

# 前踢 (Front Kick)
gym.register(
    id="Unitree-G1-29dof-Mimic-MartialArts-FrontKick",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tracking_env_cfg:FrontKickEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.tracking_env_cfg:FrontKickPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.mimic.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)

# 回旋踢 (Mawashi Geri / Roundhouse Kick)
gym.register(
    id="Unitree-G1-29dof-Mimic-MartialArts-RoundhouseKick",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tracking_env_cfg:RoundhouseKickEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.tracking_env_cfg:RoundhouseKickPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.mimic.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)

# 冲拳 (Oi-Tsuki / Lunge Punch)
gym.register(
    id="Unitree-G1-29dof-Mimic-MartialArts-LungePunch",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tracking_env_cfg:LungePunchEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.tracking_env_cfg:LungePunchPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.mimic.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)

# 侧踢 (Yoko Geri / Side Kick)
gym.register(
    id="Unitree-G1-29dof-Mimic-MartialArts-SideKick",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tracking_env_cfg:SideKickEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.tracking_env_cfg:SideKickPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.mimic.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)

# 拔塞 (Bassai - 空手道套路)
gym.register(
    id="Unitree-G1-29dof-Mimic-MartialArts-Bassai",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tracking_env_cfg:BassaiEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.tracking_env_cfg:BassaiPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.mimic.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)

# 燕飛 (Empi - 空手道套路, CMU #135-02)
gym.register(
    id="Unitree-G1-29dof-Mimic-MartialArts-Empi",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tracking_env_cfg:EmpiEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.tracking_env_cfg:EmpiPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.mimic.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)
