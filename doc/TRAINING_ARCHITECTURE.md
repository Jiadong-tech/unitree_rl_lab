# Training Flow Architecture

This document provides a comprehensive overview of the training flow for the Unitree RL Lab project, specifically for `train.py` execution.

## Overview

The training flow is organized into **five core layers**:

1. **Bootloader Layer** - Entry point, command parsing, and simulator initialization
2. **Configuration Layer** - Environment and robot configuration definitions
3. **Simulation Layer** - Physics simulation and scene construction
4. **Task Logic Layer** - MDP implementation (rewards, observations, terminations)
5. **Learning Algorithm Layer** - PPO training loop and policy optimization

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              LAYER 1: BOOTLOADER                                     │
│  ┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────────────┐  │
│  │     train.py        │───▶│   list_envs.py      │───▶│   gym.register()        │  │
│  │   (Entry Point)     │    │ (Package Scanner)   │    │ (Task Registration)     │  │
│  └─────────────────────┘    └─────────────────────┘    └─────────────────────────┘  │
│            │                                                                         │
│            ▼                                                                         │
│  ┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────────────┐  │
│  │    AppLauncher      │───▶│   Isaac Sim App     │───▶│   Hydra Config Parser   │  │
│  │ (Simulator Start)   │    │   (PhysX Engine)    │    │ (@hydra_task_config)    │  │
│  └─────────────────────┘    └─────────────────────┘    └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                             LAYER 2: CONFIGURATION                                   │
│  ┌─────────────────────────────────────┐    ┌─────────────────────────────────────┐ │
│  │         g1.py (Robot Config)        │    │   tracking_env_cfg.py (Env Config) │ │
│  │  • Joint Stiffness/Damping/Armature │    │  • Scene: USD file, terrain        │ │
│  │  • G1_ACTION_SCALE calculation      │    │  • Observations: joint pos/vel     │ │
│  │  • Initial state configuration      │    │  • Actions: 29 joint positions     │ │
│  │  • Actuator configurations          │    │  • Motion: .npz reference file     │ │
│  └─────────────────────────────────────┘    └─────────────────────────────────────┘ │
│                                                                                      │
│  ┌─────────────────────────────────────┐    ┌─────────────────────────────────────┐ │
│  │   rsl_rl_ppo_cfg.py (PPO Config)    │    │        __init__.py (Registration)  │ │
│  │  • Actor-Critic network dims        │    │  • gym.register() for task ID      │ │
│  │  • Learning rate, gamma, lambda     │    │  • Entry points configuration      │ │
│  │  • PPO hyperparameters              │    │                                    │ │
│  └─────────────────────────────────────┘    └─────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                             LAYER 3: SIMULATION                                      │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐│
│  │                        ManagerBasedRLEnv (Isaac Lab)                            ││
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐                  ││
│  │  │ InteractiveScene│  │  Event Manager  │  │ Command Manager │                  ││
│  │  │ • Load USD      │  │ • Reset states  │  │ • Motion samples│                  ││
│  │  │ • Create terrain│  │ • Randomization │  │ • Ref. tracking │                  ││
│  │  │ • Contact sensor│  │ • Push robot    │  │                 │                  ││
│  │  └─────────────────┘  └─────────────────┘  └─────────────────┘                  ││
│  │                                                                                  ││
│  │  ┌─────────────────┐  ┌─────────────────┐                                       ││
│  │  │Observation Mgr  │  │   PhysX Engine  │                                       ││
│  │  │ • Bind data flow│  │ • Rigid body    │                                       ││
│  │  │ • Feed to NN    │  │ • Articulation  │                                       ││
│  │  └─────────────────┘  └─────────────────┘                                       ││
│  └─────────────────────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                            LAYER 4: TASK LOGIC (MDP)                                 │
│  ┌───────────────────────────────┐     ┌───────────────────────────────┐            │
│  │      Motion Tracking          │     │      Reward Manager           │            │
│  │  • Load .npz reference        │     │  • Joint position tracking    │            │
│  │  • Compare robot vs reference │     │  • End effector position      │            │
│  │  • Compute tracking errors    │     │  • Orientation tracking       │            │
│  └───────────────────────────────┘     │  • Velocity tracking          │            │
│                                        │  • Penalty: action rate, acc  │            │
│  ┌───────────────────────────────┐     └───────────────────────────────┘            │
│  │     Termination Manager       │                                                   │
│  │  • Time out                   │     ┌───────────────────────────────┐            │
│  │  • Bad anchor position        │     │      Observation Manager      │            │
│  │  • Bad anchor orientation     │     │  • Policy obs (corrupted)     │            │
│  │  • End effector position      │     │  • Critic obs (privileged)    │            │
│  └───────────────────────────────┘     └───────────────────────────────┘            │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                         LAYER 5: LEARNING ALGORITHM                                  │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐│
│  │                        OnPolicyRunner (rsl_rl)                                  ││
│  │  ┌───────────────────────────────────────────────────────────────────────────┐  ││
│  │  │                         Training Loop                                      │  ││
│  │  │   ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐    │  ││
│  │  │   │    ROLLOUT       │───▶│     LEARNING     │───▶│     LOGGING      │    │  ││
│  │  │   │ • actions → env  │    │ • Compute grads  │    │ • Reward curves  │    │  ││
│  │  │   │ • env → obs, rew │    │ • Update weights │    │ • Loss values    │    │  ││
│  │  │   │ • Parallel envs  │    │ • PPO algorithm  │    │ • Save .pt files │    │  ││
│  │  │   └──────────────────┘    └──────────────────┘    └──────────────────┘    │  ││
│  │  └───────────────────────────────────────────────────────────────────────────┘  ││
│  │                                                                                  ││
│  │  ┌─────────────────────┐     ┌─────────────────────┐                            ││
│  │  │  Actor Network      │     │   Critic Network    │                            ││
│  │  │  [512, 256, 128]    │     │   [512, 256, 128]   │                            ││
│  │  │  Policy π(a|s)      │     │   Value V(s)        │                            ││
│  │  └─────────────────────┘     └─────────────────────┘                            ││
│  └─────────────────────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Layer Details

### Layer 1: Bootloader (启动层)

| Step | File | Action | Result |
|------|------|--------|--------|
| 1.1 | `train.py` | Import & Register | Scans `tasks/` packages, executes `gym.register()` |
| 1.2 | `train.py` | Start Isaac Sim | `AppLauncher` starts Omniverse Kit (PhysX engine) |
| 1.3 | `train.py` | Parse Arguments | `@hydra_task_config` creates `env_cfg` and `agent_cfg` |

**Key Files:**
- `scripts/rsl_rl/train.py` - Main entry point
- `scripts/list_envs.py` - Package scanner and importer

---

### Layer 2: Configuration (配置层)

| Config Type | File | Contents |
|-------------|------|----------|
| Robot | `g1.py` | Joint stiffness, damping, armature, action scale |
| Environment | `tracking_env_cfg.py` | Scene, observations, actions, rewards, terminations |
| PPO Algorithm | `rsl_rl_ppo_cfg.py` | Network architecture, learning hyperparameters |
| Task Registration | `__init__.py` | `gym.register()` with entry points |

**Robot Physical Properties (g1.py):**

| Motor Type | Stiffness | Damping | Armature |
|------------|-----------|---------|----------|
| 5020 | 14.25 | 0.91 | 0.0036 |
| 7520-14 | 40.18 | 2.56 | 0.0102 |
| 7520-22 | 99.10 | 6.31 | 0.0251 |
| 4010 | 16.78 | 1.07 | 0.0043 |

**Action Scale Calculation:**
```
G1_ACTION_SCALE = 0.25 * effort_limit / stiffness
```

---

### Layer 3: Simulation (物理仿真层)

| Manager | Responsibility |
|---------|----------------|
| InteractiveScene | Load USD assets, create terrain, setup lighting |
| Event Manager | Reset robot states, domain randomization, push perturbations |
| Command Manager | Sample and track reference motions from .npz files |
| Observation Manager | Bind sensor data to neural network inputs |

**Scene Configuration:**

| Component | Description |
|-----------|-------------|
| Robot | G1 29DOF USD model |
| Terrain | Plane with configurable friction |
| Lights | Distant light + dome light |
| Sensors | Contact force sensor on all bodies |

---

### Layer 4: Task Logic (任务逻辑层)

#### Observations (神经网络输入)

| Observation | Description | Noise |
|-------------|-------------|-------|
| motion_command | Reference motion phase/state | - |
| motion_anchor_ori_b | Anchor body orientation | ±0.05 |
| base_ang_vel | Base angular velocity | ±0.2 |
| joint_pos_rel | Relative joint positions | ±0.01 |
| joint_vel_rel | Relative joint velocities | ±0.5 |
| last_action | Previous action | - |

#### Rewards (奖励函数)

| Reward Term | Weight | Description |
|-------------|--------|-------------|
| joint_acc | -2.5e-7 | Joint acceleration penalty |
| joint_torque | -1e-5 | Joint torque penalty |
| action_rate_l2 | -1e-1 | Action smoothness penalty |
| joint_limit | -10.0 | Joint limit violation penalty |
| motion_global_anchor_pos | +0.5 | Anchor position tracking |
| motion_global_anchor_ori | +0.5 | Anchor orientation tracking |
| motion_body_pos | +1.0 | Body position tracking |
| motion_body_ori | +1.0 | Body orientation tracking |
| motion_body_lin_vel | +1.0 | Linear velocity tracking |
| motion_body_ang_vel | +1.0 | Angular velocity tracking |
| undesired_contacts | -0.1 | Penalty for non-foot contacts |

#### Termination Conditions (终止条件)

| Condition | Threshold | Description |
|-----------|-----------|-------------|
| time_out | 30s | Episode time limit |
| anchor_pos | 0.25m | Anchor height deviation |
| anchor_ori | 0.8 rad | Anchor orientation deviation |
| ee_body_pos | 0.25m | End effector position deviation |

---

### Layer 5: Learning Algorithm (学习算法层)

#### PPO Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| num_steps_per_env | 24 | Steps before update |
| max_iterations | 30000 | Total training iterations |
| save_interval | 500 | Checkpoint save frequency |
| learning_rate | 1e-3 | Initial learning rate |
| gamma | 0.99 | Discount factor |
| lambda | 0.95 | GAE parameter |
| clip_param | 0.2 | PPO clipping parameter |
| entropy_coef | 0.005 | Entropy bonus coefficient |
| num_learning_epochs | 5 | PPO epochs per update |
| num_mini_batches | 4 | Mini-batches per epoch |

#### Neural Network Architecture

| Network | Hidden Layers | Activation |
|---------|---------------|------------|
| Actor | [512, 256, 128] | ELU |
| Critic | [512, 256, 128] | ELU |

---

## Data Flow Summary

```
┌─────────────────────────────────────────────────────────────────┐
│                     TRAINING DATA FLOW                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Configuration Files                                            │
│       │                                                         │
│       ▼                                                         │
│  ┌─────────────┐                                               │
│  │  env_cfg    │──────┐                                        │
│  │  agent_cfg  │      │                                        │
│  └─────────────┘      │                                        │
│                       ▼                                         │
│               ┌───────────────┐                                │
│               │   gym.make() │                                 │
│               └───────────────┘                                │
│                       │                                         │
│                       ▼                                         │
│               ┌───────────────┐                                │
│               │ManagerBased  │                                 │
│               │   RLEnv      │◀──────────────────┐             │
│               └───────────────┘                   │             │
│                       │                           │             │
│          ┌────────────┼────────────┐              │             │
│          ▼            ▼            ▼              │             │
│    ┌──────────┐ ┌──────────┐ ┌──────────┐        │             │
│    │   obs    │ │  reward  │ │   done   │        │             │
│    └──────────┘ └──────────┘ └──────────┘        │             │
│          │            │            │              │             │
│          └────────────┼────────────┘              │             │
│                       ▼                           │             │
│               ┌───────────────┐                   │             │
│               │ OnPolicyRunner│                   │             │
│               │   (rsl_rl)   │                   │             │
│               └───────────────┘                   │             │
│                       │                           │             │
│          ┌────────────┴────────────┐              │             │
│          ▼                         ▼              │             │
│    ┌──────────┐             ┌──────────┐          │             │
│    │  Actor   │             │  Critic  │          │             │
│    │  π(a|s)  │             │   V(s)   │          │             │
│    └──────────┘             └──────────┘          │             │
│          │                         │              │             │
│          ▼                         ▼              │             │
│    ┌──────────┐             ┌──────────┐          │             │
│    │ actions  │             │ PPO Loss │          │             │
│    └──────────┘             └──────────┘          │             │
│          │                         │              │             │
│          │                         ▼              │             │
│          │                 ┌──────────┐           │             │
│          │                 │  Update  │           │             │
│          │                 │ Weights  │           │             │
│          │                 └──────────┘           │             │
│          │                                        │             │
│          └────────────────────────────────────────┘             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## File Structure Reference

```
unitree_rl_lab/
├── scripts/
│   ├── list_envs.py              # Package scanner
│   └── rsl_rl/
│       └── train.py              # Main entry point
│
└── source/unitree_rl_lab/unitree_rl_lab/
    └── tasks/
        └── mimic/
            ├── agents/
            │   └── rsl_rl_ppo_cfg.py    # PPO configuration
            ├── mdp/                      # MDP functions
            └── robots/
                └── g1_29dof/
                    └── gangnanm_style/
                        ├── __init__.py           # gym.register()
                        ├── g1.py                 # Robot config
                        ├── tracking_env_cfg.py   # Environment config
                        └── *.npz                 # Motion reference
```

---

## Quick Reference Command

```bash
# Training
python scripts/rsl_rl/train.py --headless --task Unitree-G1-29dof-Mimic-Gangnanm-Style

# Or using the convenience script
./unitree_rl_lab.sh -t --task Unitree-G1-29dof-Mimic-Gangnanm-Style
```
