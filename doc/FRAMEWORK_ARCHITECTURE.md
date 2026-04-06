# Unitree RL Lab — Isaac Lab 整体运作框架解读

> 本文档系统性解读 Unitree RL Lab 如何基于 NVIDIA Isaac Lab 构建从数据准备、环境定义、RL 训练到实机部署的全链路框架，以及各文件夹和模块间的连接关系。

---

## 目录

1. [全局架构鸟瞰](#1-全局架构鸟瞰)
2. [目录结构与职责划分](#2-目录结构与职责划分)
3. [Isaac Lab 核心概念速览](#3-isaac-lab-核心概念速览)
4. [环境注册机制：从代码到 gym.make()](#4-环境注册机制从代码到-gymmake)
5. [Python 包结构深度解读](#5-python-包结构深度解读)
6. [MDP 组件架构——任务的灵魂](#6-mdp-组件架构任务的灵魂)
7. [机器人资产层：assets/](#7-机器人资产层assets)
8. [训练脚本执行流程](#8-训练脚本执行流程)
9. [推理脚本与 ONNX 导出](#9-推理脚本与-onnx-导出)
10. [配置系统：CLI → 注册表 → 覆盖链](#10-配置系统cli--注册表--覆盖链)
11. [数据预处理管线（Mimic 专用）](#11-数据预处理管线mimic-专用)
12. [C++ 部署架构](#12-c-部署架构)
13. [两类任务对比：Locomotion vs Mimic](#13-两类任务对比locomotion-vs-mimic)
14. [模块间依赖关系总图](#14-模块间依赖关系总图)

---

## 1. 全局架构鸟瞰

整个项目是一条 **端到端的强化学习流水线**，涵盖从原始动捕数据到真实机器人部署的完整链路：

```
┌──────────────────────────────────────────────────────────────────────────┐
│                           Unitree RL Lab 全局数据流                       │
│                                                                          │
│  ┌──────────┐   ┌──────────┐   ┌──────────────┐   ┌────────────────┐    │
│  │ 原始数据  │──►│ 数据预处理 │──►│ Isaac Lab 训练│──►│  策略导出/部署  │    │
│  │ CMU MoCap │   │ CSV → NPZ │   │ PPO + 4096环境│   │ ONNX + C++ FSM │    │
│  └──────────┘   └──────────┘   └──────────────┘   └────────────────┘    │
│                                                                          │
│  data/           scripts/mimic/  scripts/rsl_rl/    deploy/              │
│  cmu_mocap/                      + source/           robots/             │
│                                  unitree_rl_lab/                         │
└──────────────────────────────────────────────────────────────────────────┘
```

项目支持 **两大类任务**：

| 任务类型 | 目标 | 机器人 | 命令源 |
|---------|------|--------|-------|
| **Locomotion（运动控制）** | 跟踪速度指令行走/跑步 | Go2, H1, G1 | 速度命令（手柄/随机） |
| **Mimic（动作模仿）** | 复现动捕数据中的动作 | G1-29dof | NPZ 动作片段（逐帧参考） |

---

## 2. 目录结构与职责划分

```
unitree_rl_lab/
│
├── source/unitree_rl_lab/          ★ 核心 Python 包（pip install -e 安装）
│   └── unitree_rl_lab/
│       ├── tasks/                  ★ 环境定义（所有任务入口）
│       │   ├── locomotion/           └─ 速度控制运动任务
│       │   └── mimic/                └─ 动作模仿跟踪任务
│       ├── assets/                 ★ 机器人资产（USD/URDF 配置 + 执行器参数）
│       └── utils/                  ★ 工具函数（配置解析、部署导出）
│
├── scripts/
│   ├── rsl_rl/                     ★ 训练/推理入口脚本
│   │   ├── train.py                  └─ 训练主程序
│   │   ├── play.py                   └─ 推理 + ONNX 导出
│   │   └── cli_args.py               └─ CLI 参数定义 + 配置合并
│   └── mimic/                      ★ 数据预处理管线
│       ├── cmu_amc_to_csv.py         └─ CMU 动捕 → CSV
│       ├── csv_to_npz.py            └─ CSV → NPZ（物理仿真生成速度）
│       └── validate_npz.py          └─ NPZ 数据完整性校验
│
├── deploy/                         ★ C++ 实机部署
│   ├── include/                      └─ 公共头文件（FSM、Isaac Lab C++ 接口）
│   ├── robots/                       └─ 各机器人专属 C++ 项目
│   └── thirdparty/                   └─ ONNX Runtime 推理引擎
│
├── data/cmu_mocap/                 ★ 原始 CMU 动捕数据
├── logs/rsl_rl/                    ★ 训练输出（自动生成）
├── doc/                              └─ 文档集合
├── docker/                           └─ Docker 容器化配置
└── pyproject.toml                    └─ 项目构建与依赖配置
```

### 核心设计原则

- **任务与资产分离**：`tasks/` 定义"做什么"，`assets/` 定义"用什么机器人"
- **脚本与库分离**：`scripts/` 是执行入口，`source/` 是可复用的 Python 包
- **训练与部署分离**：Python 端训练 → 导出 ONNX/YAML → C++ 端独立推理

---

## 3. Isaac Lab 核心概念速览

Unitree RL Lab 构建在 NVIDIA Isaac Lab 之上，理解以下概念是读懂代码的前提：

### 3.1 ManagerBasedRLEnv

Isaac Lab 的核心环境类，通过 **Manager 模式** 组合各个 MDP 组件：

```
ManagerBasedRLEnv
├── SceneManager          ← 管理场景实体（机器人、地形、传感器）
├── ActionManager         ← 管理动作空间与执行
├── ObservationManager    ← 管理观测空间（Policy / Critic 分组）
├── RewardManager         ← 管理奖励计算
├── TerminationManager    ← 管理终止条件
├── CommandManager        ← 管理命令生成（速度指令/动作参考）
├── EventManager          ← 管理随机化事件（reset、push 等）
└── CurriculumManager     ← 管理课程学习进度
```

每个 Manager 由**配置类（`*Cfg`）** 驱动，不需要写代码重载——只需填写配置。

### 3.2 configclass 装饰器

Isaac Lab 使用 `@configclass`（基于 dataclass）定义所有配置。配置类可以嵌套、继承、覆盖：

```python
@configclass
class MyEnvCfg(ManagerBasedRLEnvCfg):
    scene = MySceneCfg(num_envs=4096)       # 场景
    observations = MyObsCfg()                # 观测
    rewards = MyRewardsCfg()                 # 奖励
    terminations = MyTerminationsCfg()       # 终止
    commands = MyCommandsCfg()               # 命令
    events = MyEventsCfg()                   # 随机化
    actions = MyActionsCfg()                 # 动作
```

### 3.3 RSL-RL

RSL-RL 是 ETH Zurich 开发的轻量级 On-Policy RL 库，Unitree RL Lab 用它实现 PPO 训练。核心组件：

- `OnPolicyRunner`：训练主循环（rollout → 计算 GAE → PPO 更新）
- `RslRlVecEnvWrapper`：将 Isaac Lab 环境适配为 RSL-RL 接口
- `ActorCritic`：MLP 策略网络（Actor-Critic 架构）

---

## 4. 环境注册机制：从代码到 gym.make()

这是整个框架最核心的连接逻辑——如何把分散在各文件夹中的配置汇聚成一个可用的 Gym 环境。

### 4.1 注册流程

```
                      ┌─ train.py / play.py ─┐
                      │                       │
                      │  import               │
                      │  unitree_rl_lab.tasks  │
                      └───────────┬───────────┘
                                  │
                                  ▼
               tasks/__init__.py 调用 import_packages()
                                  │
                      ┌───────────┼───────────┐
                      ▼           ▼           ▼
              locomotion/      mimic/       （未来扩展）
              __init__.py      __init__.py
                  │                │
                  ▼                ▼
             robots/          robots/g1_29dof/
             ├── go2/         ├── martial_arts/
             │   __init__.py  │   __init__.py
             │   gym.register │   gym.register  ← 多个动作注册
             ├── h1/          ├── gangnanm_style/
             │   __init__.py  │   __init__.py
             │   gym.register │   gym.register
             └── g1/29dof/    └── dance_102/
                 __init__.py      __init__.py
                 gym.register     gym.register
                                  │
                                  ▼
                        Gym 全局注册表 (gym.registry)
                        现在包含所有可用的 Task ID
                                  │
                                  ▼
                        gym.make("Unitree-Go2-Velocity")
                        gym.make("Unitree-G1-29dof-Mimic-MartialArts-HeianShodan")
```

### 4.2 gym.register() 的三个关键参数

每次 `gym.register()` 调用绑定了三个配置入口点：

```python
gym.register(
    id="Unitree-Go2-Velocity",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",    # ① 环境类（固定）
    kwargs={
        "env_cfg_entry_point": "...velocity_env_cfg:RobotEnvCfg",        # ② 训练配置
        "play_env_cfg_entry_point": "...velocity_env_cfg:RobotPlayEnvCfg", # ③ 推理配置
        "rsl_rl_cfg_entry_point": "...rsl_rl_ppo_cfg:BasePPORunnerCfg",  # ④ PPO 配置
    },
)
```

| 参数 | 用途 | 由谁读取 |
|-----|------|---------|
| `env_cfg_entry_point` | 训练用环境配置（多环境、高随机化） | `train.py` → `parse_env_cfg()` |
| `play_env_cfg_entry_point` | 推理用环境配置（少量环境、低/无随机化） | `play.py` → `parse_env_cfg()` |
| `rsl_rl_cfg_entry_point` | PPO 算法超参数 | `cli_args.parse_rsl_rl_cfg()` |

### 4.3 import_packages() 的发现机制

`tasks/__init__.py` 中的这一行是整个注册链的起点：

```python
from isaaclab_tasks.utils import import_packages
import_packages(__name__, _BLACKLIST_PKGS)
```

`import_packages()` 递归遍历 `tasks/` 下所有子包，执行每个 `__init__.py`，从而触发所有 `gym.register()` 调用。**无需手动维护环境列表**——新增任务只需在对应文件夹中添加 `__init__.py` + `gym.register()`。

---

## 5. Python 包结构深度解读

### 5.1 tasks/ — 任务定义层

```
tasks/
├── __init__.py                     ← import_packages() 自动发现入口
│
├── locomotion/                     ─── 速度控制运动任务 ───
│   ├── __init__.py                 ← from .robots import *
│   ├── mdp/                        ← 共享 MDP 组件
│   │   ├── __init__.py             ← 聚合导出 + 继承 isaaclab 默认 MDP
│   │   ├── commands/
│   │   │   └── velocity_command.py ← UniformLevelVelocityCommandCfg
│   │   ├── observations.py         ← gait_phase() 等自定义观测
│   │   ├── rewards.py              ← 步态奖励、能效惩罚等 20+ 函数
│   │   └── curriculums.py          ← 课程学习（逐步增加难度）
│   ├── agents/
│   │   └── rsl_rl_ppo_cfg.py       ← Locomotion PPO 超参数
│   └── robots/                     ← 每个机器人一个子包
│       ├── go2/
│       │   ├── __init__.py         ← gym.register("Unitree-Go2-Velocity")
│       │   └── velocity_env_cfg.py ← RobotEnvCfg + RobotPlayEnvCfg
│       ├── h1/
│       │   ├── __init__.py         ← gym.register("Unitree-H1-Velocity")
│       │   └── velocity_env_cfg.py
│       └── g1/29dof/
│           ├── __init__.py         ← gym.register("Unitree-G1-29dof-Velocity")
│           │                          gym.register("Unitree-G1-29dof-Parkour")
│           ├── velocity_env_cfg.py
│           └── parkour_env_cfg.py
│
└── mimic/                          ─── 动作模仿跟踪任务 ───
    ├── __init__.py                 （空文件）
    ├── mdp/                        ← Mimic 专属 MDP 组件
    │   ├── commands.py             ← ★ MotionCommand（加载 NPZ、逐帧跟踪、自适应采样）
    │   ├── observations.py         ← 身体位置/朝向观测、相对坐标变换
    │   ├── rewards.py              ← 高斯核跟踪奖励（body_pos/ori/vel）
    │   ├── terminations.py         ← 姿态偏差终止（anchor Z/ori、末端执行器）
    │   └── events.py               ← 重置事件 + 随机扰动
    ├── agents/
    │   └── rsl_rl_ppo_cfg.py       ← Mimic PPO 超参数
    └── robots/g1_29dof/            ← 目前仅 G1-29dof 支持 Mimic
        ├── martial_arts/
        │   ├── __init__.py         ← 7 个武术动作 gym.register()
        │   ├── tracking_env_cfg.py ← 环境配置（场景+MDP组件组装）
        │   └── G1_*.npz            ← 动作数据文件（7 个）
        ├── gangnanm_style/
        │   ├── __init__.py         ← gym.register("...-GangnamStyle")
        │   ├── tracking_env_cfg.py
        │   └── g1.py              ← G1 机器人参数覆盖（action_scale 等）
        ├── dance_102/
        └── petite_verses/
```

### 5.2 核心设计：共享 mdp/ + 机器人专属 robots/

**同一类任务（如 locomotion）共享 MDP 组件**，但每个机器人有自己的 `velocity_env_cfg.py` 来组装这些组件、设定具体参数：

```
locomotion/mdp/rewards.py        ← 定义 energy(), feet_gait(), stand_still() 等函数
locomotion/robots/go2/velocity_env_cfg.py   ← 选择用哪些奖励、设什么权重
locomotion/robots/h1/velocity_env_cfg.py    ← 同样的函数库，不同的权重配置
```

这实现了 **算法逻辑复用** + **机器人参数独立** 的解耦。

### 5.3 EnvCfg vs PlayEnvCfg

每个任务同时定义两种配置：

```python
@configclass
class RobotEnvCfg(ManagerBasedRLEnvCfg):
    """训练配置：4096 环境 + 高随机化"""
    scene = RobotSceneCfg(num_envs=4096, env_spacing=2.5)
    # 完整的 domain randomization ...

@configclass
class RobotPlayEnvCfg(RobotEnvCfg):
    """推理配置：少量环境 + 降低/关闭随机化"""
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        # 关闭随机推力等 ...
```

---

## 6. MDP 组件架构——任务的灵魂

`velocity_env_cfg.py` / `tracking_env_cfg.py` 是每个任务的配置核心，它把所有 MDP 组件**声明式地组装**在一起：

```
                    velocity_env_cfg.py (以 Go2 为例)
                    ┌─────────────────────────────┐
                    │  @configclass                │
                    │  class RobotEnvCfg           │
                    │                              │
  SceneCfg ────────►│  scene = RobotSceneCfg()     │  ← 地形、机器人、传感器
  ActionsCfg ──────►│  actions = ActionsCfg()       │  ← JointPositionAction
  CommandsCfg ─────►│  commands = CommandsCfg()     │  ← 速度命令 / 动作参考
  ObsCfg ──────────►│  observations = ObsCfg()      │  ← Policy + Critic 观测组
  RewardsCfg ──────►│  rewards = RewardsCfg()       │  ← 奖励项及权重
  TerminationsCfg ─►│  terminations = TermCfg()     │  ← 终止条件
  EventCfg ────────►│  events = EventCfg()          │  ← 随机化事件
  CurriculumCfg ───►│  curriculum = CurrCfg()       │  ← 难度递增
                    └─────────────────────────────┘
```

### 6.1 Commands（命令生成器）

| 任务类型 | 命令类 | 作用 |
|---------|--------|------|
| Locomotion | `UniformLevelVelocityCommandCfg` | 随机采样速度指令 (vx, vy, ωz) |
| Mimic | `MotionCommand`（自定义） | 从 NPZ 文件逐帧读取参考姿态 |

**Locomotion 命令**：每隔一段时间重采样目标速度（带上下限范围），机器人学习跟踪速度。

**Mimic 命令**：加载预处理的 NPZ 动捕数据到 GPU，每个控制步推进一帧，提供全身参考状态（关节角、身体位姿、速度）。包含 **自适应采样**：失败率高的时间段被更频繁地采样。

### 6.2 Observations（观测空间）

每个环境配置将观测分为两组：

```python
@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):          # Actor 用 —— 部署时可用的传感器数据
        base_ang_vel = ObsTerm(...)     # IMU 角速度 + 噪声
        joint_pos_rel = ObsTerm(...)    # 关节编码器 + 噪声
        joint_vel_rel = ObsTerm(...)    # 关节速度 + 噪声
        last_action = ObsTerm(...)      # 上一步动作
        # Locomotion 额外：projected_gravity, velocity_commands, gait_phase
        # Mimic 额外：motion_command (参考关节角+速度), motion_anchor_ori_b

    @configclass
    class CriticCfg(ObsGroup):          # Critic 用 —— 仿真特权信息（训练专用）
        # 包含 PolicyCfg 的所有内容，外加：
        base_lin_vel = ObsTerm(...)     # 真实线速度（无噪声）
        # Mimic 额外：body_pos, body_ori（14 个身体部位位置/朝向）
```

**关键设计**：Actor 只能获取带噪声的传感器数据（模拟 sim-to-real 迁移），Critic 获取完整特权信息（加速训练）。

### 6.3 Rewards（奖励函数）

**Locomotion 奖励**（~15 项）：

| 类别 | 示例函数 | 目标 |
|------|---------|------|
| 速度跟踪 | `track_lin_vel_xy_exp`, `track_ang_vel_z_exp` | 跟踪目标速度 |
| 基座稳定 | `orientation_l2`, `base_height_l2`, `upward` | 保持躯干稳定 |
| 步态控制 | `feet_gait`, `foot_clearance_reward`, `air_time_variance_penalty` | 自然步态 |
| 能效惩罚 | `energy`, `joint_torque`, `action_rate_l2` | 能量效率 |
| 安全惩罚 | `joint_limit`, `undesired_contacts`, `feet_stumble` | 避免危险 |
| 静止奖励 | `stand_still`, `feet_contact_without_cmd` | 零速度时站稳 |

**Mimic 奖励**（~10 项，均使用高斯核 $r = e^{-\text{error}^2/\sigma^2}$）：

| 类别 | 函数 | 权重 | 目标 |
|------|------|------|------|
| 身体跟踪 | `motion_body_pos` | 1.5 | 14 个身体部位位置 |
| 身体朝向 | `motion_body_ori` | 1.0 | 14 个身体部位朝向 |
| 速度跟踪 | `motion_body_lin_vel` / `ang_vel` | 各 1.0 | 动态匹配 |
| 锚点跟踪 | `motion_global_anchor_pos` / `ori` | 各 0.5 | 躯干全局位姿 |
| 平滑/能效 | `joint_acc`, `action_rate_l2`, `joint_torque` | 负值 | 惩罚抖动 |

### 6.4 Terminations（终止条件）

| 任务类型 | 条件 | 说明 |
|---------|------|------|
| 通用 | `time_out` | 达到最大 episode 时长 |
| Locomotion | `illegal_contact` | 非脚部接触地面 |
| Mimic | `bad_anchor_pos_z_only` | 躯干 Z 高度偏差 > 阈值 |
| Mimic | `bad_anchor_ori` | 躯干朝向偏差过大 |
| Mimic | `bad_motion_body_pos_z_only` | 手脚 Z 高度偏差 > 阈值 |

### 6.5 Events（随机化事件）

每个环境配置定义三类事件触发时机：

```python
@configclass
class EventCfg:
    # ① startup（仅在仿真启动时执行一次）
    physics_material = EventTerm(...)       # 随机化地面摩擦
    add_base_mass = EventTerm(...)          # 随机化躯干质量

    # ② reset（每次 episode 重置时）
    reset_robot = EventTerm(...)            # 重置关节位置/速度
    reset_base = EventTerm(...)             # 重置基座位姿

    # ③ interval（训练过程中周期性触发）
    push_robot = EventTerm(                 # 随机推力（每 2-5 秒）
        func=mdp.push_by_setting_velocity,
        params={"velocity_range": {...}},
        interval_range_s=(2.0, 5.0),
    )
```

---

## 7. 机器人资产层：assets/

```
assets/robots/
├── __init__.py
└── unitree.py    ← 所有 Unitree 机器人配置的定义
```

### 7.1 自定义配置类

```python
class UnitreeArticulationCfg(ArticulationCfg):
    joint_sdk_names: list[str]              # SDK 关节名映射（部署时用）
    soft_joint_pos_limit_factor: float      # 软关节限位系数（0.9 = 允许行程的 90%）
```

### 7.2 支持的机器人

| 配置常量 | 机器人 | 自由度 | 类型 | USD 资产路径 |
|---------|-------|------|------|------------|
| `UNITREE_GO2_CFG` | Go2 | 12 DOF | 四足 | `.../Go2/usd/go2.usd` |
| `UNITREE_GO2W_CFG` | Go2W | 16 DOF | 轮足混合 | `.../Go2W/usd/go2w.usd` |
| `UNITREE_B2_CFG` | B2 | 12 DOF | 四足（大型） | `.../B2/usd/b2.usd` |
| `UNITREE_H1_CFG` | H1 | 20 DOF | 人形 | `.../H1/usd/h1.usd` |
| `UNITREE_G1_23DOF_CFG` | G1-23 | 23 DOF | 人形 | `.../G1/usd/g1_23dof.usd` |
| `UNITREE_G1_29DOF_CFG` | G1-29 | 29 DOF | 人形（含手腕） | `.../G1/usd/g1_29dof.usd` |

每个配置包含：
- **spawn**：USD 文件路径 + 碰撞/物理属性
- **init_state**：初始位姿和关节角度
- **actuators**：按电机型号分组的 PD 增益（stiffness / damping）
- **joint_sdk_names**：对应真实机器人 SDK 的关节名列表

### 7.3 资产与任务的连接

任务配置通过引用资产常量来使用机器人：

```python
# 在 velocity_env_cfg.py 中
from unitree_rl_lab.assets.robots.unitree import UNITREE_GO2_CFG

@configclass
class RobotSceneCfg(InteractiveSceneCfg):
    robot = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
```

`.replace()` 只修改 `prim_path`（USD 场景路径），不改变机器人物理属性。

---

## 8. 训练脚本执行流程

`scripts/rsl_rl/train.py` 是训练的主入口。以下是完整调用链：

```
train.py
  │
  ├─ ① 解析 CLI 参数
  │   parser = argparse.ArgumentParser()
  │   AppLauncher.add_app_launcher_args(parser)      ← Isaac Sim 参数
  │   cli_args.add_rsl_rl_args(parser)               ← RSL-RL 参数
  │   args = parser.parse_args()
  │
  ├─ ② 启动 Isaac Sim
  │   app_launcher = AppLauncher(args)
  │   simulation_app = app_launcher.app
  │
  ├─ ③ 导入环境包（触发 gym.register）
  │   import unitree_rl_lab.tasks                    ← ★ 这一步注册所有环境
  │
  ├─ ④ 解析环境 & 算法配置
  │   env_cfg = parse_env_cfg(task_name, device, num_envs)
  │   agent_cfg = cli_args.parse_rsl_rl_cfg(task_name, args)
  │
  ├─ ⑤ 创建环境
  │   env = gym.make(task_name, cfg=env_cfg)         ← 实例化 ManagerBasedRLEnv
  │   env = RslRlVecEnvWrapper(env)                  ← 适配 RSL-RL 接口
  │
  ├─ ⑥ 创建训练器
  │   runner = OnPolicyRunner(env, agent_cfg.to_dict(), ...)
  │   runner.load(resume_path) if resuming
  │
  ├─ ⑦ 导出部署配置
  │   export_deploy_cfg(env.unwrapped, log_dir)      ← 保存 deploy.yaml
  │
  ├─ ⑧ 训练
  │   runner.learn(num_learning_iterations=agent_cfg.max_iterations)
  │   │
  │   │  每次迭代：
  │   │  ├─ Rollout: 在 4096 个环境中收集 24 步数据
  │   │  ├─ 计算 GAE (Generalized Advantage Estimation)
  │   │  ├─ PPO 更新: 5 epochs × 4 mini-batches
  │   │  └─ 每 N iter 保存 checkpoint (model_*.pt)
  │   │
  └─ ⑨ 关闭
      env.close()
      simulation_app.close()
```

### 代码层面的关键连接

```
train.py
    │
    │  uses
    ├──────► cli_args.py
    │            │  parse_rsl_rl_cfg() 从注册表加载
    │            └──────► gym.registry → rsl_rl_cfg_entry_point
    │                          │
    │                          └──► agents/rsl_rl_ppo_cfg.py :: BasePPORunnerCfg
    │
    │  uses
    ├──────► utils/parser_cfg.py
    │            │  parse_env_cfg() 从注册表加载
    │            └──────► gym.registry → env_cfg_entry_point
    │                          │
    │                          └──► robots/go2/velocity_env_cfg.py :: RobotEnvCfg
    │                                    │
    │                                    ├──► assets/robots/unitree.py (机器人)
    │                                    └──► locomotion/mdp/ (MDP 组件)
    │
    │  uses
    └──────► utils/export_deploy_cfg.py
                 │  读取 env 中的关节信息、action/obs 配置
                 └──► 输出 logs/.../params/deploy.yaml
```

---

## 9. 推理脚本与 ONNX 导出

`scripts/rsl_rl/play.py` 负责加载训练好的策略进行推理，并导出部署格式：

```
play.py
  │
  ├─ ① 解析配置（使用 play_env_cfg_entry_point）
  │   env_cfg = parse_env_cfg(task_name, ...,
  │       entry_point_key="play_env_cfg_entry_point")   ← 推理专用配置
  │
  ├─ ② 查找 checkpoint
  │   ├─ --use_pretrained_checkpoint → 从预训练仓库下载
  │   ├─ --checkpoint <path>         → 直接指定路径
  │   └─ 自动查找 logs/rsl_rl/<experiment>/ 下最新的
  │
  ├─ ③ 创建环境 + 加载策略
  │   env = gym.make(task_name, cfg=env_cfg)
  │   runner = OnPolicyRunner(env, agent_cfg)
  │   runner.load(resume_path)
  │   policy = runner.get_inference_policy()
  │
  ├─ ④ 导出模型
  │   ├─ exported/policy.pt    ← TorchScript JIT
  │   └─ exported/policy.onnx  ← ONNX 格式（含观测归一化参数）
  │
  └─ ⑤ 推理循环
      with torch.inference_mode():
          while simulation_app.is_running():
              actions = policy(obs)
              obs, _, _, _ = env.step(actions)
```

**导出的 ONNX 文件**包含：
- Actor 网络权重
- 观测归一化器（running mean / std）
- 输入/输出维度信息

---

## 10. 配置系统：CLI → 注册表 → 覆盖链

配置的最终值由三层合并决定：

```
                 优先级
                   ↑
    ┌──────────────┼──────────────┐
    │   CLI 参数                   │  最高：--num_envs 64 --seed 42
    ├─────────────────────────────┤
    │   注册表默认值               │  中间：gym.register 中 entry_point 指向的类
    ├─────────────────────────────┤
    │   Isaac Lab 基类默认值       │  最低：ManagerBasedRLEnvCfg 的默认参数
    └─────────────────────────────┘
```

### cli_args.py 的三个核心函数

```python
def add_rsl_rl_args(parser):
    """添加 RSL-RL 专属 CLI 参数到 argparse"""
    # --experiment_name, --run_name, --resume, --checkpoint
    # --logger (wandb/tensorboard), --max_iterations, --seed

def parse_rsl_rl_cfg(task_name, args_cli):
    """从注册表加载 PPO 配置 → 用 CLI 覆盖"""
    cfg = load_cfg_from_registry(task_name, "rsl_rl_cfg_entry_point")
    cfg = update_rsl_rl_cfg(cfg, args_cli)
    return cfg

def update_rsl_rl_cfg(agent_cfg, args_cli):
    """实际的覆盖逻辑"""
    if args_cli.seed is not None:    agent_cfg.seed = args_cli.seed
    if args_cli.resume:              agent_cfg.resume = True
    if args_cli.max_iterations:      agent_cfg.max_iterations = args_cli.max_iterations
    # ...
```

### parse_env_cfg() 的环境配置加载

```python
# utils/parser_cfg.py
def parse_env_cfg(task_name, device, num_envs, entry_point_key="env_cfg_entry_point"):
    cfg = load_cfg_from_registry(task_name, entry_point_key)
    cfg.sim.device = device
    if num_envs is not None:
        cfg.scene.num_envs = num_envs
    return cfg
```

---

## 11. 数据预处理管线（Mimic 专用）

Mimic 任务需要预处理的参考动作数据。整个管线分三阶段：

```
┌────────────┐     cmu_amc_to_csv.py     ┌──────┐     csv_to_npz.py      ┌──────┐
│ ASF + AMC  │ ──────────────────────────►│ CSV  │ ──────────────────────►│ NPZ  │
│ (CMU 135)  │   骨骼解析 + FK +         │      │   Isaac Lab 物理仿真    │      │
│ 原始动捕    │   关节映射 + 坐标变换      │关节角│   → 产生速度数据         │全状态│
└────────────┘                            └──────┘                        └──────┘
      ↓                                                                      ↓
  data/cmu_mocap/135/                                              tasks/mimic/robots/
  135_*.amc + 135.asf                                              g1_29dof/martial_arts/
                                                                   G1_*.npz
```

### 阶段 1：CMU → CSV

`scripts/mimic/cmu_amc_to_csv.py`

- 解析 ASF 骨骼层级 + AMC 逐帧角度
- 正向运动学计算全局关节位置
- 人体骨骼映射到 G1 29DOF 关节
- CMU 坐标系 → Isaac Lab 坐标系转换
- 输出：每行 = `[base_pos(3), base_quat(4), joint_pos(29)]`

### 阶段 2：CSV → NPZ

`scripts/mimic/csv_to_npz.py`

- 在 Isaac Lab 中创建 G1 机器人仿真场景
- 逐帧设置关节位置并步进物理引擎
- 记录所有 30 个刚体的完整状态：
  - `body_pos_w [T, 30, 3]` — 位置
  - `body_quat_w [T, 30, 4]` — 朝向
  - `body_lin_vel_w [T, 30, 3]` — 线速度
  - `body_ang_vel_w [T, 30, 3]` — 角速度
  - `joint_pos [T, 29]` — 关节角（钳制到软限位内）
  - `joint_vel [T, 29]` — 关节速度

**为什么需要物理仿真？** CSV 只有关节角度，缺少速度信息。直接求导噪声大，物理仿真能产生自洽的速度数据。

### 阶段 3：NPZ → 训练

NPZ 文件直接存放在任务文件夹中（`tasks/mimic/robots/g1_29dof/martial_arts/G1_*.npz`），由 `MotionCommand` 在运行时加载到 GPU。

### 辅助工具

| 脚本 | 用途 |
|------|------|
| `validate_npz.py` | 校验 NPZ 完整性（shape、NaN、速度尖峰） |
| `replay_npz.py` | 在 Isaac Sim 中回放 NPZ 验证视觉效果 |
| `fix_npz_velocity_spikes.py` | 修复速度数据中的异常尖峰 |
| `martial_arts_pipeline.sh` | 一键执行全管线 |

---

## 12. C++ 部署架构

训练好的策略通过 ONNX 格式部署到真实机器人。

### 12.1 目录结构

```
deploy/
├── include/                        ← 公共头文件
│   ├── param.h                     ← 参数加载（读取 deploy.yaml）
│   ├── unitree_articulation.h      ← 关节控制接口
│   ├── LinearInterpolator.h        ← 线性插值器
│   ├── FSM/                        ← 有限状态机框架
│   │   ├── BaseState.h             ← 状态基类
│   │   ├── CtrlFSM.h              ← FSM 控制器（状态切换调度）
│   │   ├── FSMState.h             ← 状态枚举定义
│   │   ├── State_Passive.h        ← 被动状态（上电默认，关节软化）
│   │   ├── State_FixStand.h       ← 固定站立（PD 控制到默认姿态）
│   │   ├── State_RLBase.h         ← ★ RL 推理基类（加载 ONNX 策略）
│   │   └── State_MartialArtsSequencer.h  ← ★ 多策略串联状态
│   └── isaaclab/                   ← Isaac Lab C++ 接口镜像
│       ├── algorithms/             ← 算法工具
│       ├── assets/articulation/    ← 关节配置结构
│       ├── devices/keyboard/       ← 键盘输入
│       ├── envs/                   ← C++ 环境接口
│       │   └── mdp/               ← 动作/观测 manager
│       └── manager/               ← 动作/观测管理器
│
├── robots/                         ← 各机器人专属项目
│   ├── go2/                        ← Go2 四足
│   ├── go2w/                       ← Go2W 轮足
│   ├── b2/                         ← B2 四足
│   ├── h1/                         ← H1 人形
│   ├── h1_2/                       ← H1-2 人形
│   ├── g1_23dof/                   ← G1-23 人形
│   └── g1_29dof/                   ← G1-29 人形（含 Mimic 部署）
│       ├── CMakeLists.txt
│       ├── main.cpp               ← 入口（初始化 FSM → 控制循环）
│       ├── config/config.yaml     ← 部署配置（策略路径、段列表）
│       ├── include/               ← 机器人专属头文件
│       └── src/                   ← 状态实现
│
└── thirdparty/
    └── onnxruntime-linux-x64-1.22.0/  ← ONNX 推理运行时
```

### 12.2 训练 → 部署的衔接

训练过程自动生成两份部署所需文件：

```
logs/rsl_rl/<experiment>/<timestamp>/
├── exported/
│   ├── policy.onnx          ← play.py 导出的 Actor 网络 + 归一化参数
│   └── policy.pt            ← TorchScript 版本
└── params/
    └── deploy.yaml          ← train.py 中 export_deploy_cfg() 导出
```

`deploy.yaml` 包含内容：

```yaml
joint_ids_map:      [SDK 关节名 → 索引映射]
step_dt:            0.02           # 控制周期（= sim.dt × decimation）
stiffness:          [各关节 Kp]
damping:            [各关节 Kd]
default_joint_pos:  [默认姿态]

actions:
  JointPositionAction:
    scale:  [各关节缩放系数]
    clip:   [-clip, +clip]       # 动作裁剪范围
    offset: [默认位置偏移]

observations:
  motion_command:       {scale, bias, ...}
  base_ang_vel:         {scale, noise, ...}
  joint_pos_rel:        {scale, noise, ...}
  # ... 所有观测项的配置
```

### 12.3 FSM 运行流程

```
┌───────────┐  L2 按键   ┌────────────┐  自动    ┌──────────────────────┐
│  Passive  │ ─────────► │  FixStand  │ ──────► │  RLBase / Sequencer  │
│  (关节软化) │           │  (站立归位)  │         │  (策略推理)            │
└───────────┘            └────────────┘         └──────────────────────┘
```

`State_RLBase` 每个控制步执行：
1. 收集传感器数据 → 构建观测向量
2. 送入 ONNX Runtime → 推理出动作
3. 动作解码（× scale + offset）→ 关节位置目标
4. 写入 `lowcmd` → 发送给电机

`State_MartialArtsSequencer` 在 RLBase 基础上增加：
- 多段策略串联（7 个武术动作顺序执行）
- 段间过渡保持（默认 1 秒静止过渡）
- 运动数据帧推进（与训练时的 MotionCommand 对应）

---

## 13. 两类任务对比：Locomotion vs Mimic

| 维度 | Locomotion | Mimic |
|------|-----------|-------|
| **目标** | 跟踪速度指令 (vx, vy, ωz) | 复现动捕动作序列 |
| **命令源** | 随机/手柄速度命令 | NPZ 逐帧参考姿态 |
| **支持机器人** | Go2, Go2W, B2, H1, G1-23, G1-29 | G1-29dof |
| **观测特征** | projected_gravity, gait_phase | motion_command, body_pos/ori |
| **核心奖励** | 速度跟踪 + 步态 + 能效 | 全身姿态跟踪（高斯核） |
| **终止条件** | 非法接触 | 姿态偏差（Z 高度、朝向） |
| **特殊机制** | 课程学习（地形难度） | 自适应采样（失败时间段加权） |
| **部署方式** | RLBase（单策略持续运行） | Sequencer（多策略串联） |
| **训练规模** | 50K iter, ~24h | 30K iter, ~12h |
| **数据依赖** | 无（环境自生成命令） | CMU MoCap → CSV → NPZ |

### MDP 组件复用关系

```
isaaclab.envs.mdp (基础 MDP 函数库)
         │
         ├─► locomotion/mdp/__init__.py
         │       import isaaclab + isaaclab_tasks 默认 MDP
         │       + 自定义 commands, observations, rewards, curriculums
         │
         └─► mimic/mdp/
                 完全自定义: commands, observations, rewards,
                            terminations, events
```

Locomotion 的 MDP 继承了 Isaac Lab 和 isaaclab_tasks 的大量默认实现，只做增量扩展；Mimic 的 MDP 几乎完全自定义（因为动作跟踪逻辑与速度控制迥异）。

---

## 14. 模块间依赖关系总图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           完整模块依赖关系                                    │
│                                                                             │
│  ┌─────────────┐         ┌────────────────┐         ┌──────────────────┐    │
│  │ data/       │         │ scripts/mimic/  │         │ scripts/rsl_rl/  │    │
│  │ cmu_mocap/  │ ──(1)──►│ cmu_amc_to_csv │         │ train.py         │    │
│  │ ASF + AMC   │         │ csv_to_npz     │         │ play.py          │    │
│  └─────────────┘         └───────┬────────┘         │ cli_args.py      │    │
│                                  │                   └────────┬─────────┘    │
│                                 (2) NPZ 文件                  │              │
│                                  │                           (5) import      │
│                                  ▼                            │              │
│  ┌───────────────────────────────────────────────────────────▼────────────┐ │
│  │                    source/unitree_rl_lab/                               │ │
│  │                                                                         │ │
│  │  ┌─────────────────────────────────────────────────────────────────┐   │ │
│  │  │                        tasks/                                    │   │ │
│  │  │                                                                  │   │ │
│  │  │  ┌──────────────┐    ┌───────────────┐                          │   │ │
│  │  │  │ locomotion/  │    │    mimic/      │                          │   │ │
│  │  │  │              │    │               │                          │   │ │
│  │  │  │ ┌──────────┐ │    │ ┌──────────┐  │                          │   │ │
│  │  │  │ │   mdp/   │ │    │ │   mdp/   │  │  ← (3) 共享 MDP 组件     │   │ │
│  │  │  │ │commands  │ │    │ │commands  │  │                          │   │ │
│  │  │  │ │rewards   │ │    │ │rewards   │  │                          │   │ │
│  │  │  │ │obs       │ │    │ │obs       │  │                          │   │ │
│  │  │  │ └────┬─────┘ │    │ └────┬─────┘  │                          │   │ │
│  │  │  │      │        │    │      │        │                          │   │ │
│  │  │  │ ┌────▼─────┐ │    │ ┌────▼─────┐  │                          │   │ │
│  │  │  │ │ robots/  │ │    │ │ robots/  │  │  ← (4) 环境配置组装      │   │ │
│  │  │  │ │go2/h1/g1 │ │    │ │g1_29dof/ │  │     引用 mdp + assets    │   │ │
│  │  │  │ │env_cfg + │ │    │ │env_cfg + │  │                          │   │ │
│  │  │  │ │register  │ │    │ │register  │  │                          │   │ │
│  │  │  │ └──────────┘ │    │ └──────────┘  │                          │   │ │
│  │  │  └──────────────┘    └───────────────┘                          │   │ │
│  │  └─────────────────────────────────────────────────────────────────┘   │ │
│  │                                                                         │ │
│  │  ┌──────────────────┐    ┌──────────────────┐                          │ │
│  │  │ assets/robots/   │←───│ tasks/robots/     │  ← (6) 任务引用资产     │ │
│  │  │ unitree.py       │    │ *_env_cfg.py      │                          │ │
│  │  │ GO2/H1/G1 配置   │    │ ROBOT_CFG.replace │                          │ │
│  │  └──────────────────┘    └──────────────────┘                          │ │
│  │                                                                         │ │
│  │  ┌──────────────────┐                                                   │ │
│  │  │ utils/           │                                                   │ │
│  │  │ parser_cfg.py    │←── train.py / play.py 调用                        │ │
│  │  │ export_deploy.py │──── (7) 输出 deploy.yaml ──►┐                     │ │
│  │  └──────────────────┘                              │                    │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                                       │                      │
│                                                       ▼                      │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │                         logs/rsl_rl/                                    │  │
│  │  <experiment>/<timestamp>/                                              │  │
│  │  ├── model_*.pt                    ← 训练 checkpoint                    │  │
│  │  ├── exported/policy.onnx          ← 导出策略    ──(8)──►┐              │  │
│  │  └── params/deploy.yaml            ← 部署配置    ──(8)──►│              │  │
│  └────────────────────────────────────────────────────────┬──┘──────────┘  │
│                                                           │                  │
│                                                           ▼                  │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │                          deploy/                                        │  │
│  │  ┌──────────────┐    ┌──────────────────┐    ┌─────────────────────┐   │  │
│  │  │ include/FSM/ │    │ robots/g1_29dof/ │    │ thirdparty/         │   │  │
│  │  │ RLBase       │◄───│ main.cpp         │───►│ onnxruntime         │   │  │
│  │  │ Sequencer    │    │ config.yaml      │    │ (推理引擎)           │   │  │
│  │  └──────────────┘    └──────────────────┘    └─────────────────────┘   │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘

连接标号说明：
  (1) cmu_amc_to_csv.py / csv_to_npz.py 读取原始动捕数据
  (2) NPZ 文件输出到 tasks/mimic/robots/ 目录下
  (3) mdp/ 定义可复用的 MDP 函数（奖励、观测、命令等）
  (4) robots/*_env_cfg.py 组合 mdp 组件 + 设定参数
  (5) train.py / play.py import tasks 触发环境注册
  (6) *_env_cfg.py 引用 assets/robots/ 中的机器人配置常量
  (7) export_deploy_cfg() 从环境中提取关节/动作/观测配置
  (8) ONNX 策略 + deploy.yaml 被 C++ 部署程序读取
```

---

> **总结**：Unitree RL Lab 的架构核心是 **声明式配置驱动**——通过 `@configclass` 把场景、MDP 组件、PPO 参数全部以数据形式定义，再由 Isaac Lab 的 Manager 系统自动组装成完整的训练环境。整个工程围绕 `gym.register()` 这一注册机制串联所有模块，实现了"增加新任务只需添加配置文件、无需修改框架代码"的可扩展设计。
