# 训练流程架构详解

本文档详细阐述 Unitree RL Lab 项目的 `train.py` 训练执行全流程，从代码架构、配置系统到算法实现进行全面解析。

---

## 目录

1. [项目概览](#项目概览)
2. [知识架构图谱](#知识架构图谱)
3. [五层架构详解](#五层架构详解)
   - [第一层：启动层](#第一层启动层-bootloader)
   - [第二层：配置层](#第二层配置层-configuration)
   - [第三层：物理仿真层](#第三层物理仿真层-simulation)
   - [第四层：任务逻辑层](#第四层任务逻辑层-task-logic)
   - [第五层：学习算法层](#第五层学习算法层-learning-algorithm)
4. [支持的机器人平台](#支持的机器人平台)
5. [关键数据结构](#关键数据结构)
6. [文件目录结构](#文件目录结构)
7. [部署流程](#部署流程)
8. [常用命令参考](#常用命令参考)

---

## 项目概览

Unitree RL Lab 是基于 [Isaac Lab](https://github.com/isaac-sim/IsaacLab) 构建的强化学习训练平台，专门用于宇树机器人（Go2、H1、G1 等）的运动控制策略训练。

### 核心技术栈

| 技术组件 | 版本要求 | 说明 |
|---------|---------|------|
| Isaac Sim | 5.0.0+ | NVIDIA Omniverse 物理仿真引擎 |
| Isaac Lab | 2.2.0+ | 机器人学习开发框架 |
| rsl_rl | 2.3.1+ | RSL 实验室的 PPO 强化学习库 |
| PyTorch | - | 深度学习框架 |
| Gymnasium (Gym) | - | 强化学习环境标准接口 |
| Hydra | - | 配置管理框架 |
| PhysX 5 | - | 物理引擎 |

### 支持的任务类型

| 任务类型 | 说明 | 示例任务 |
|---------|------|---------|
| Locomotion（运动控制） | 速度追踪、步态控制 | `Unitree-G1-29dof-Velocity` |
| Mimic（动作模仿） | 参考动作追踪 | `Unitree-G1-29dof-Mimic-Gangnanm-Style` |

---

## 知识架构图谱

### 系统分层架构

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                                 第一层：启动层 (BOOTLOADER)                               │
│  ┌────────────────────┐    ┌────────────────────┐    ┌──────────────────────────┐       │
│  │    train.py        │───▶│   list_envs.py     │───▶│    gym.register()        │       │
│  │   (程序入口)        │    │  (包扫描与导入)     │    │   (任务注册到Gym系统)     │       │
│  └────────────────────┘    └────────────────────┘    └──────────────────────────┘       │
│            │                                                                             │
│            ▼                                                                             │
│  ┌────────────────────┐    ┌────────────────────┐    ┌──────────────────────────┐       │
│  │   AppLauncher      │───▶│   Isaac Sim App    │───▶│   Hydra 配置解析器        │       │
│  │  (仿真器启动器)     │    │  (Omniverse Kit)   │    │ (@hydra_task_config)     │       │
│  └────────────────────┘    └────────────────────┘    └──────────────────────────┘       │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                                            │
                                            ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                                 第二层：配置层 (CONFIGURATION)                            │
│  ┌──────────────────────────────────────┐    ┌────────────────────────────────────────┐ │
│  │          g1.py (机器人配置)           │    │   tracking_env_cfg.py (环境配置)        │ │
│  │  • 关节刚度 (Stiffness)              │    │  • 场景定义：USD文件、地形              │ │
│  │  • 关节阻尼 (Damping)                │    │  • 观测空间：关节位置/速度              │ │
│  │  • 关节电枢 (Armature)               │    │  • 动作空间：29个关节位置目标           │ │
│  │  • 动作缩放 G1_ACTION_SCALE          │    │  • 运动参考：.npz 动作文件              │ │
│  └──────────────────────────────────────┘    └────────────────────────────────────────┘ │
│                                                                                          │
│  ┌──────────────────────────────────────┐    ┌────────────────────────────────────────┐ │
│  │    rsl_rl_ppo_cfg.py (PPO配置)       │    │        __init__.py (任务注册)          │ │
│  │  • Actor-Critic 网络维度             │    │  • gym.register() 注册任务ID          │ │
│  │  • 学习率、折扣因子 γ、λ              │    │  • 配置入口点绑定                      │ │
│  │  • PPO 超参数配置                    │    │  • 环境配置入口点绑定                  │ │
│  └──────────────────────────────────────┘    └────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                                            │
                                            ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                                 第三层：物理仿真层 (SIMULATION)                           │
│  ┌─────────────────────────────────────────────────────────────────────────────────────┐│
│  │                     ManagerBasedRLEnv (Isaac Lab 核心环境类)                         ││
│  │  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐                   ││
│  │  │ InteractiveScene │  │   Event Manager  │  │  Command Manager │                   ││
│  │  │   交互式场景      │  │     事件管理器    │  │    命令管理器     │                   ││
│  │  │ • 加载USD资产    │  │ • 状态重置       │  │ • 运动采样        │                   ││
│  │  │ • 创建地形       │  │ • 域随机化       │  │ • 参考轨迹追踪    │                   ││
│  │  │ • 接触传感器     │  │ • 外部扰动       │  │                  │                   ││
│  │  └──────────────────┘  └──────────────────┘  └──────────────────┘                   ││
│  │                                                                                      ││
│  │  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐                   ││
│  │  │Observation Mgr   │  │  Reward Manager  │  │Termination Mgr   │                   ││
│  │  │   观测管理器      │  │    奖励管理器     │  │   终止管理器      │                   ││
│  │  │ • 绑定数据流     │  │ • 计算奖励信号    │  │ • 检查终止条件    │                   ││
│  │  │ • 喂给神经网络   │  │ • 惩罚项计算     │  │ • 触发重置        │                   ││
│  │  └──────────────────┘  └──────────────────┘  └──────────────────┘                   ││
│  │                                                                                      ││
│  │  ┌──────────────────────────────────────────────────────────────────────────────┐   ││
│  │  │                        PhysX 5 物理引擎                                       │   ││
│  │  │  • 刚体动力学仿真  • 关节约束求解  • 碰撞检测  • GPU 并行加速                    │   ││
│  │  └──────────────────────────────────────────────────────────────────────────────┘   ││
│  └─────────────────────────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────────────────────────┘
                                            │
                                            ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                              第四层：任务逻辑层 (TASK LOGIC / MDP)                        │
│  ┌────────────────────────────────┐     ┌─────────────────────────────────────┐         │
│  │       MotionCommand 运动命令    │     │        RewardsCfg 奖励配置          │         │
│  │  • 加载 .npz 参考动作文件       │     │  • 追踪奖励（位置、姿态、速度）      │         │
│  │  • 计算当前帧参考姿态           │     │  • 惩罚项（加速度、力矩、动作抖动）  │         │
│  │  • 自适应采样 (AMP风格)         │     │  • 接触惩罚（非期望接触）           │         │
│  └────────────────────────────────┘     └─────────────────────────────────────┘         │
│                                                                                          │
│  ┌────────────────────────────────┐     ┌─────────────────────────────────────┐         │
│  │     ObservationsCfg 观测配置    │     │      TerminationsCfg 终止条件        │         │
│  │  • Policy观测（带噪声）         │     │  • 超时终止                         │         │
│  │  • Critic观测（特权信息）       │     │  • 锚点位置偏差过大                 │         │
│  │  • 关节位置/速度/动作历史       │     │  • 锚点姿态偏差过大                 │         │
│  └────────────────────────────────┘     │  • 末端执行器位置偏差过大           │         │
│                                          └─────────────────────────────────────┘         │
│  ┌────────────────────────────────┐     ┌─────────────────────────────────────┐         │
│  │       EventsCfg 事件配置        │     │      ActionsCfg 动作配置            │         │
│  │  • 物理材质随机化               │     │  • 关节位置目标控制                 │         │
│  │  • 关节默认位置随机化           │     │  • 动作缩放（ACTION_SCALE）         │         │
│  │  • 质心位置随机化               │     │  • 默认位置偏移                     │         │
│  │  • 外部推力扰动                 │     │                                     │         │
│  └────────────────────────────────┘     └─────────────────────────────────────┘         │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                                            │
                                            ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                              第五层：学习算法层 (LEARNING ALGORITHM)                      │
│  ┌─────────────────────────────────────────────────────────────────────────────────────┐│
│  │                         OnPolicyRunner (rsl_rl 训练运行器)                           ││
│  │  ┌───────────────────────────────────────────────────────────────────────────────┐  ││
│  │  │                              训练循环 (Training Loop)                          │  ││
│  │  │   ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐        │  ││
│  │  │   │     ROLLOUT      │───▶│     LEARNING     │───▶│     LOGGING      │        │  ││
│  │  │   │      采样阶段     │    │      学习阶段     │    │      日志阶段     │        │  ││
│  │  │   │ • 策略输出动作   │    │ • 计算优势函数    │    │ • 记录奖励曲线   │        │  ││
│  │  │   │ • 环境执行步进   │    │ • 计算策略梯度    │    │ • 记录损失函数   │        │  ││
│  │  │   │ • 收集(s,a,r,s') │    │ • PPO裁剪更新    │    │ • 保存模型权重   │        │  ││
│  │  │   │ • 并行环境执行   │    │ • 更新网络权重    │    │ • TensorBoard    │        │  ││
│  │  │   └──────────────────┘    └──────────────────┘    └──────────────────┘        │  ││
│  │  └───────────────────────────────────────────────────────────────────────────────┘  ││
│  │                                                                                      ││
│  │  ┌──────────────────────┐     ┌──────────────────────┐                              ││
│  │  │   Actor Network      │     │    Critic Network    │                              ││
│  │  │     策略网络          │     │      价值网络         │                              ││
│  │  │  [512, 256, 128]     │     │   [512, 256, 128]    │                              ││
│  │  │  输出：动作 π(a|s)   │     │   输出：价值 V(s)    │                              ││
│  │  │  激活函数：ELU       │     │   激活函数：ELU      │                              ││
│  │  └──────────────────────┘     └──────────────────────┘                              ││
│  └─────────────────────────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 五层架构详解

### 第一层：启动层 (Bootloader)

启动层负责程序初始化，包括包导入、仿真器启动和配置解析。

#### 执行流程

```
train.py 启动
    │
    ├─▶ sys.path.insert() 添加任务路径
    │
    ├─▶ list_envs.import_packages() 
    │       │
    │       └─▶ 递归扫描 tasks/ 目录
    │               │
    │               └─▶ 执行各 __init__.py 中的 gym.register()
    │
    ├─▶ AppLauncher(args_cli) 
    │       │
    │       └─▶ 启动 Omniverse Kit 进程 (Isaac Sim 后台)
    │               │
    │               └─▶ 初始化 PhysX 物理引擎
    │
    └─▶ @hydra_task_config 装饰器
            │
            ├─▶ 解析 --task 参数
            │
            ├─▶ 获取 env_cfg_entry_point
            │
            └─▶ 获取 rsl_rl_cfg_entry_point
```

#### 关键文件

| 文件 | 路径 | 功能 |
|-----|------|-----|
| train.py | `scripts/rsl_rl/train.py` | 训练主入口 |
| play.py | `scripts/rsl_rl/play.py` | 推理演示脚本 |
| list_envs.py | `scripts/list_envs.py` | 包扫描与任务列表 |
| cli_args.py | `scripts/rsl_rl/cli_args.py` | 命令行参数定义 |

#### 命令行参数

| 参数 | 类型 | 默认值 | 说明 |
|-----|------|-------|------|
| `--task` | str | None | 任务名称（必填） |
| `--num_envs` | int | 配置值 | 并行环境数量 |
| `--headless` | flag | False | 无界面模式 |
| `--video` | flag | False | 录制视频 |
| `--seed` | int | None | 随机种子 |
| `--max_iterations` | int | 配置值 | 最大训练迭代次数 |
| `--distributed` | flag | False | 多GPU分布式训练 |

---

### 第二层：配置层 (Configuration)

配置层定义了机器人物理属性、环境参数和算法超参数。

#### 2.1 机器人配置 (g1.py)

##### 电机型号与物理参数

| 电机型号 | 刚度 (Stiffness) | 阻尼 (Damping) | 电枢 (Armature) | 应用关节 |
|---------|-----------------|----------------|-----------------|---------|
| N7520-14.3 | 40.18 | 2.56 | 0.0102 | 髋关节Pitch/Yaw |
| N7520-22.5 | 99.10 | 6.31 | 0.0251 | 髋关节Roll、膝关节 |
| N5020-16 | 14.25 | 0.91 | 0.0036 | 肩、肘、踝关节 |
| W4010-25 | 16.78 | 1.07 | 0.0043 | 腕关节 |

##### 参数计算公式

刚度和阻尼的计算基于自然频率和阻尼比：

```
NATURAL_FREQ = 10 * 2π = 62.83 rad/s  (10Hz)
DAMPING_RATIO = 2.0

STIFFNESS = ARMATURE × NATURAL_FREQ²
DAMPING = 2 × DAMPING_RATIO × ARMATURE × NATURAL_FREQ
```

##### 动作缩放计算

```python
G1_ACTION_SCALE = 0.25 * effort_limit / stiffness
```

这意味着神经网络输出 1.0 对应的关节位置偏移量。

#### 2.2 环境配置 (tracking_env_cfg.py)

##### 场景配置 (RobotSceneCfg)

| 组件 | 类型 | 说明 |
|-----|------|-----|
| terrain | TerrainImporterCfg | 平面地形，摩擦系数 1.0 |
| robot | ArticulationCfg | G1 29DOF USD 模型 |
| light | AssetBaseCfg | 远光源 + 穹顶光 |
| contact_forces | ContactSensorCfg | 接触力传感器 |

##### 物理材质配置

| 参数 | 值 | 说明 |
|-----|-----|-----|
| static_friction | 1.0 | 静摩擦系数 |
| dynamic_friction | 1.0 | 动摩擦系数 |
| friction_combine_mode | multiply | 摩擦力组合模式 |
| restitution_combine_mode | multiply | 弹性系数组合模式 |

##### 仿真参数

| 参数 | 值 | 说明 |
|-----|-----|-----|
| dt | 0.005s | 仿真步长 (200Hz) |
| decimation | 4 | 控制频率降采样 (50Hz 控制) |
| episode_length_s | 30.0s | 单次回合时长 |
| num_envs | 4096 | 并行环境数量 |

#### 2.3 PPO 算法配置 (rsl_rl_ppo_cfg.py)

##### 训练参数

| 参数 | 值 | 说明 |
|-----|-----|-----|
| num_steps_per_env | 24 | 每次更新前收集的步数 |
| max_iterations | 30000 | 最大训练迭代次数 |
| save_interval | 500 | 模型保存间隔 |
| empirical_normalization | False | 经验归一化 |

##### PPO 超参数

| 参数 | 值 | 说明 |
|-----|-----|-----|
| learning_rate | 1e-3 | 学习率 |
| gamma | 0.99 | 折扣因子 |
| lam (λ) | 0.95 | GAE λ 参数 |
| clip_param | 0.2 | PPO 裁剪参数 |
| entropy_coef | 0.005 | 熵正则化系数 |
| value_loss_coef | 1.0 | 价值损失系数 |
| num_learning_epochs | 5 | 每次更新的训练轮数 |
| num_mini_batches | 4 | 小批次数量 |
| max_grad_norm | 1.0 | 梯度裁剪阈值 |
| schedule | adaptive | 学习率调度策略 |
| desired_kl | 0.01 | 目标 KL 散度 |

##### 神经网络架构

| 网络 | 隐藏层维度 | 激活函数 | 输出 |
|-----|-----------|---------|------|
| Actor (策略网络) | [512, 256, 128] | ELU | 动作分布 π(a\|s) |
| Critic (价值网络) | [512, 256, 128] | ELU | 状态价值 V(s) |

#### 2.4 任务注册 (__init__.py)

```python
gym.register(
    id="Unitree-G1-29dof-Mimic-Gangnanm-Style",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tracking_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.tracking_env_cfg:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.mimic.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)
```

---

### 第三层：物理仿真层 (Simulation)

物理仿真层在 Isaac Sim 中构建虚拟世界并运行物理仿真。

#### 3.1 核心类 ManagerBasedRLEnv

`ManagerBasedRLEnv` 是 Isaac Lab 的核心强化学习环境类，负责协调各个管理器。

```
┌────────────────────────────────────────────────────────────┐
│                   ManagerBasedRLEnv                        │
│  ┌──────────────────────────────────────────────────────┐  │
│  │                 InteractiveScene                     │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌──────────────┐  │  │
│  │  │   Robot     │  │   Terrain   │  │   Sensors    │  │  │
│  │  └─────────────┘  └─────────────┘  └──────────────┘  │  │
│  └──────────────────────────────────────────────────────┘  │
│                           │                                 │
│  ┌────────────────────────┼─────────────────────────────┐  │
│  │        Manager Registry (管理器注册表)                │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐   │  │
│  │  │ Observation │  │   Reward    │  │Termination  │   │  │
│  │  │   Manager   │  │   Manager   │  │  Manager    │   │  │
│  │  └─────────────┘  └─────────────┘  └─────────────┘   │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐   │  │
│  │  │   Action    │  │   Command   │  │    Event    │   │  │
│  │  │   Manager   │  │   Manager   │  │   Manager   │   │  │
│  │  └─────────────┘  └─────────────┘  └─────────────┘   │  │
│  └──────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────┘
```

#### 3.2 管理器详解

##### Event Manager (事件管理器)

| 事件类型 | 触发模式 | 功能 |
|---------|---------|------|
| physics_material | startup | 随机化物理材质（摩擦系数） |
| add_joint_default_pos | startup | 随机化关节默认位置 |
| base_com | startup | 随机化躯干质心位置 |
| push_robot | interval (1-3s) | 施加随机外力扰动 |

##### 域随机化参数

| 参数 | 范围 | 说明 |
|-----|------|-----|
| static_friction | 0.3 ~ 1.6 | 静摩擦系数 |
| dynamic_friction | 0.3 ~ 1.2 | 动摩擦系数 |
| restitution | 0.0 ~ 0.5 | 弹性恢复系数 |
| joint_default_pos | ±0.01 rad | 关节默认位置偏移 |
| base_com_x | ±0.025 m | 质心 X 偏移 |
| base_com_y | ±0.05 m | 质心 Y 偏移 |
| base_com_z | ±0.05 m | 质心 Z 偏移 |

##### 外部扰动 (Velocity Range)

| 方向 | 范围 | 说明 |
|-----|------|-----|
| x | -0.5 ~ 0.5 m/s | 前后速度扰动 |
| y | -0.5 ~ 0.5 m/s | 左右速度扰动 |
| z | -0.2 ~ 0.2 m/s | 上下速度扰动 |
| roll | -0.52 ~ 0.52 rad/s | 滚转角速度扰动 |
| pitch | -0.52 ~ 0.52 rad/s | 俯仰角速度扰动 |
| yaw | -0.78 ~ 0.78 rad/s | 偏航角速度扰动 |

---

### 第四层：任务逻辑层 (Task Logic)

任务逻辑层定义了 MDP（马尔可夫决策过程）的核心元素。

#### 4.1 观测空间 (Observations)

##### Policy 观测（输入到策略网络）

| 观测项 | 维度 | 噪声范围 | 说明 |
|-------|------|---------|------|
| motion_command | 2×n_joints | - | 参考关节位置和速度 |
| motion_anchor_ori_b | 4 | ±0.05 | 锚点姿态四元数 |
| base_ang_vel | 3 | ±0.2 | 基座角速度 |
| joint_pos_rel | n_joints | ±0.01 | 相对关节位置 |
| joint_vel_rel | n_joints | ±0.5 | 相对关节速度 |
| last_action | n_joints | - | 上一帧动作 |

##### Critic 观测（特权信息，仅训练时使用）

| 观测项 | 说明 |
|-------|------|
| command | 参考运动命令 |
| motion_anchor_pos_b | 锚点位置（无噪声） |
| motion_anchor_ori_b | 锚点姿态（无噪声） |
| body_pos | 身体各部位位置 |
| body_ori | 身体各部位姿态 |
| base_lin_vel | 基座线速度 |
| base_ang_vel | 基座角速度 |
| joint_pos | 关节位置（无噪声） |
| joint_vel | 关节速度（无噪声） |
| actions | 当前动作 |

#### 4.2 奖励函数 (Rewards)

##### 追踪奖励（正向激励）

| 奖励项 | 权重 | 标准差 σ | 公式 |
|-------|------|---------|------|
| motion_global_anchor_pos | +0.5 | 0.3 | exp(-\|\|pos_ref - pos\|\|² / σ²) |
| motion_global_anchor_ori | +0.5 | 0.4 | exp(-quat_error² / σ²) |
| motion_body_pos | +1.0 | 0.3 | exp(-mean(\|\|body_pos_ref - body_pos\|\|²) / σ²) |
| motion_body_ori | +1.0 | 0.4 | exp(-mean(quat_error²) / σ²) |
| motion_body_lin_vel | +1.0 | 1.0 | exp(-mean(\|\|lin_vel_ref - lin_vel\|\|²) / σ²) |
| motion_body_ang_vel | +1.0 | 3.14 | exp(-mean(\|\|ang_vel_ref - ang_vel\|\|²) / σ²) |

##### 惩罚项（负向激励）

| 惩罚项 | 权重 | 说明 |
|-------|------|-----|
| joint_acc | -2.5e-7 | 关节加速度 L2 范数 |
| joint_torque | -1e-5 | 关节力矩 L2 范数 |
| action_rate_l2 | -1e-1 | 动作变化率 L2 范数 |
| joint_limit | -10.0 | 关节限位违反惩罚 |
| undesired_contacts | -0.1 | 非期望接触（非脚部） |

#### 4.3 终止条件 (Terminations)

| 条件 | 阈值 | 说明 |
|-----|------|-----|
| time_out | 30s | 回合超时 |
| anchor_pos | 0.25m | 锚点高度偏差过大 |
| anchor_ori | 0.8 rad | 锚点姿态偏差过大 |
| ee_body_pos | 0.25m | 末端执行器位置偏差过大（手脚） |

#### 4.4 运动命令 (MotionCommand)

##### MotionLoader 数据加载

从 `.npz` 文件加载参考动作数据：

| 数据项 | 形状 | 说明 |
|-------|------|-----|
| fps | scalar | 动作帧率 |
| joint_pos | (T, n_joints) | 关节位置序列 |
| joint_vel | (T, n_joints) | 关节速度序列 |
| body_pos_w | (T, n_bodies, 3) | 身体位置序列 |
| body_quat_w | (T, n_bodies, 4) | 身体姿态序列 |
| body_lin_vel_w | (T, n_bodies, 3) | 身体线速度序列 |
| body_ang_vel_w | (T, n_bodies, 3) | 身体角速度序列 |

##### 自适应采样策略

| 参数 | 值 | 说明 |
|-----|-----|-----|
| adaptive_kernel_size | 1 | 卷积核大小 |
| adaptive_lambda | 0.8 | 衰减系数 |
| adaptive_uniform_ratio | 0.1 | 均匀采样比例 |
| adaptive_alpha | 0.001 | 失败统计更新率 |

##### 追踪的身体部位

| 部位名称 | 说明 |
|---------|------|
| pelvis | 骨盆 |
| left/right_hip_roll_link | 髋关节 Roll |
| left/right_knee_link | 膝关节 |
| left/right_ankle_roll_link | 踝关节 |
| torso_link | 躯干 |
| left/right_shoulder_roll_link | 肩关节 |
| left/right_elbow_link | 肘关节 |
| left/right_wrist_yaw_link | 腕关节 |

---

### 第五层：学习算法层 (Learning Algorithm)

学习算法层实现 PPO (Proximal Policy Optimization) 强化学习算法。

#### 5.1 OnPolicyRunner 训练流程

```
初始化
    │
    ├─▶ 创建 Actor-Critic 网络
    │
    ├─▶ 初始化优化器 (Adam)
    │
    └─▶ 进入训练循环
            │
            ├─▶ ROLLOUT 阶段
            │       │
            │       ├─▶ 策略网络输出动作 actions = π(obs)
            │       │
            │       ├─▶ 环境执行动作 obs', rewards, dones = env.step(actions)
            │       │
            │       ├─▶ 存储轨迹 (s, a, r, s', done)
            │       │
            │       └─▶ 重复 num_steps_per_env 次
            │
            ├─▶ LEARNING 阶段
            │       │
            │       ├─▶ 计算 GAE 优势函数
            │       │
            │       ├─▶ 分成 num_mini_batches 个小批次
            │       │
            │       └─▶ 执行 num_learning_epochs 轮 PPO 更新
            │               │
            │               ├─▶ 计算策略损失 L_policy
            │               │
            │               ├─▶ 计算价值损失 L_value
            │               │
            │               ├─▶ 计算熵奖励 L_entropy
            │               │
            │               └─▶ 反向传播更新参数
            │
            └─▶ LOGGING 阶段
                    │
                    ├─▶ 记录奖励、损失等指标
                    │
                    ├─▶ 定期保存模型 (.pt 文件)
                    │
                    └─▶ 输出到 TensorBoard
```

#### 5.2 PPO 算法细节

##### 策略损失 (Policy Loss)

```
ratio = π_θ(a|s) / π_θ_old(a|s)
L_clip = min(ratio × A, clip(ratio, 1-ε, 1+ε) × A)
L_policy = -E[L_clip]
```

其中 ε = 0.2 (clip_param)

##### 价值损失 (Value Loss)

```
L_value = value_loss_coef × (V(s) - V_target)²
```

##### 熵奖励 (Entropy Bonus)

```
L_entropy = -entropy_coef × H[π(·|s)]
```

##### 总损失

```
L_total = L_policy + L_value + L_entropy
```

#### 5.3 GAE (Generalized Advantage Estimation)

```
δ_t = r_t + γ × V(s_{t+1}) - V(s_t)
A_t = Σ (γλ)^l × δ_{t+l}
```

其中 γ = 0.99, λ = 0.95

---

## 支持的机器人平台

### Unitree 四足机器人

| 型号 | 自由度 | 初始高度 | 说明 |
|-----|--------|---------|------|
| Go2 | 12 | 0.4m | 小型四足机器人 |
| Go2W | 16 | 0.45m | 带轮足四足机器人 |
| B2 | 12 | 0.58m | 工业级四足机器人 |

### Unitree 人形机器人

| 型号 | 自由度 | 初始高度 | 说明 |
|-----|--------|---------|------|
| H1 | 20 | 1.1m | 人形机器人（无手） |
| G1 23DOF | 23 | 0.8m | 人形机器人（基础版） |
| G1 29DOF | 29 | 0.8m | 人形机器人（带腕关节） |

### G1 29DOF 关节列表

| 部位 | 关节名称 | 电机型号 |
|-----|---------|---------|
| 左腿 | left_hip_pitch/roll/yaw, left_knee, left_ankle_pitch/roll | N7520/N5020 |
| 右腿 | right_hip_pitch/roll/yaw, right_knee, right_ankle_pitch/roll | N7520/N5020 |
| 腰部 | waist_yaw/roll/pitch | N7520/N5020 |
| 左臂 | left_shoulder_pitch/roll/yaw, left_elbow, left_wrist_roll/pitch/yaw | N5020/W4010 |
| 右臂 | right_shoulder_pitch/roll/yaw, right_elbow, right_wrist_roll/pitch/yaw | N5020/W4010 |

---

## 关键数据结构

### 配置类继承关系

```
ManagerBasedRLEnvCfg
    │
    ├── scene: InteractiveSceneCfg
    │       └── robot: ArticulationCfg
    │               └── actuators: dict[str, ActuatorCfg]
    │
    ├── observations: ObservationsCfg
    │       ├── policy: ObsGroup
    │       └── critic: ObsGroup
    │
    ├── actions: ActionsCfg
    │       └── JointPositionAction: JointPositionActionCfg
    │
    ├── commands: CommandsCfg
    │       └── motion: MotionCommandCfg
    │
    ├── rewards: RewardsCfg
    │       └── [各奖励项]: RewTerm
    │
    ├── terminations: TerminationsCfg
    │       └── [各终止条件]: DoneTerm
    │
    └── events: EventCfg
            └── [各事件]: EventTerm
```

### 训练数据流

```
┌─────────────────────────────────────────────────────────────────────┐
│                        训练数据流向图                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   配置文件                                                          │
│       │                                                             │
│       ▼                                                             │
│   ┌─────────────┐                                                   │
│   │  env_cfg    │                                                   │
│   │  agent_cfg  │                                                   │
│   └─────────────┘                                                   │
│          │                                                          │
│          ▼                                                          │
│   ┌────────────────┐                                                │
│   │  gym.make()    │                                                │
│   └────────────────┘                                                │
│          │                                                          │
│          ▼                                                          │
│   ┌────────────────────────────────────────────┐                    │
│   │          ManagerBasedRLEnv                 │                    │
│   │  ┌────────────┐  ┌──────────┐  ┌────────┐  │                    │
│   │  │    obs     │  │  reward  │  │  done  │  │                    │
│   │  └────────────┘  └──────────┘  └────────┘  │◀──────┐            │
│   └────────────────────────────────────────────┘       │            │
│          │                                             │            │
│          ▼                                             │            │
│   ┌────────────────┐                                   │            │
│   │ RslRlVecEnvWrapper │                               │            │
│   └────────────────┘                                   │            │
│          │                                             │            │
│          ▼                                             │            │
│   ┌────────────────────────────────────────────┐       │            │
│   │           OnPolicyRunner                   │       │            │
│   │  ┌────────────────┐  ┌────────────────┐    │       │            │
│   │  │     Actor      │  │    Critic      │    │       │            │
│   │  │   π(a|s)       │  │     V(s)       │    │       │            │
│   │  └────────────────┘  └────────────────┘    │       │            │
│   │          │                   │             │       │            │
│   │          ▼                   ▼             │       │            │
│   │  ┌────────────────┐  ┌────────────────┐    │       │            │
│   │  │    actions     │  │   PPO Loss     │    │       │            │
│   │  └────────────────┘  └────────────────┘    │       │            │
│   │          │                   │             │       │            │
│   │          │                   ▼             │       │            │
│   │          │           ┌────────────────┐    │       │            │
│   │          │           │ 梯度更新权重    │    │       │            │
│   │          │           └────────────────┘    │       │            │
│   │          │                                 │       │            │
│   └──────────┼─────────────────────────────────┘       │            │
│              │                                         │            │
│              └─────────────────────────────────────────┘            │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 文件目录结构

```
unitree_rl_lab/
├── scripts/                              # 脚本目录
│   ├── list_envs.py                      # 任务列表和包扫描
│   ├── mimic/                            # Mimic 任务相关脚本
│   └── rsl_rl/                           # RSL-RL 训练脚本
│       ├── train.py                      # 训练入口
│       ├── play.py                       # 推理演示
│       └── cli_args.py                   # 命令行参数
│
├── source/unitree_rl_lab/unitree_rl_lab/ # 源代码主目录
│   ├── __init__.py
│   ├── assets/                           # 资产配置
│   │   └── robots/
│   │       ├── unitree.py                # Unitree 机器人配置
│   │       └── unitree_actuators.py      # 电机配置
│   │
│   ├── tasks/                            # 任务定义
│   │   ├── __init__.py
│   │   ├── locomotion/                   # 运动控制任务
│   │   │   ├── agents/                   # 算法配置
│   │   │   ├── mdp/                      # MDP 函数
│   │   │   └── robots/                   # 各机器人任务
│   │   │       ├── g1/
│   │   │       ├── go2/
│   │   │       └── h1/
│   │   │
│   │   └── mimic/                        # 动作模仿任务
│   │       ├── agents/
│   │       │   └── rsl_rl_ppo_cfg.py     # PPO 配置
│   │       ├── mdp/
│   │       │   ├── commands.py           # 运动命令
│   │       │   ├── events.py             # 事件函数
│   │       │   ├── observations.py       # 观测函数
│   │       │   ├── rewards.py            # 奖励函数
│   │       │   └── terminations.py       # 终止函数
│   │       └── robots/
│   │           └── g1_29dof/
│   │               ├── __init__.py
│   │               ├── gangnanm_style/   # 江南Style舞蹈任务
│   │               │   ├── __init__.py   # gym.register()
│   │               │   ├── g1.py         # 机器人配置
│   │               │   ├── tracking_env_cfg.py  # 环境配置
│   │               │   └── *.npz         # 参考动作文件
│   │               └── dance_102/        # 其他舞蹈任务
│   │
│   └── utils/                            # 工具函数
│       ├── export_deploy_cfg.py          # 导出部署配置
│       └── parser_cfg.py                 # 配置解析
│
├── deploy/                               # 部署代码
│   ├── robots/                           # 各机器人控制器
│   │   ├── g1_29dof/
│   │   ├── go2/
│   │   └── h1/
│   ├── include/                          # C++ 头文件
│   └── thirdparty/                       # 第三方库
│
├── doc/                                  # 文档目录
│   ├── TRAINING_ARCHITECTURE.md          # 英文架构文档
│   ├── TRAINING_ARCHITECTURE_CN.md       # 中文架构文档
│   └── licenses/                         # 许可证
│
└── docker/                               # Docker 配置
```

---

## 部署流程

### Sim2Sim (仿真到仿真迁移)

使用 MuJoCo 进行 Sim2Sim 验证：

```bash
# 1. 配置 unitree_mujoco
#    - robot: g1
#    - domain_id: 0
#    - enable_elastic_hand: 1
#    - use_joystck: 1

# 2. 启动 MuJoCo 仿真
cd unitree_mujoco/simulate/build
./unitree_mujoco

# 3. 启动控制器
cd unitree_rl_lab/deploy/robots/g1_29dof/build
./g1_ctrl

# 操作流程：
# 1. [L2 + Up] 站立
# 2. 点击 MuJoCo 窗口，按 8 让脚接触地面
# 3. [R1 + X] 运行策略
# 4. 点击 MuJoCo 窗口，按 9 禁用弹力带
```

### Sim2Real (仿真到实机部署)

```bash
# 确保机载控制程序已关闭
./g1_ctrl --network eth0  # eth0 为网络接口名
```

---

## 常用命令参考

### 训练命令

```bash
# 基础训练
python scripts/rsl_rl/train.py --headless --task Unitree-G1-29dof-Mimic-Gangnanm-Style

# 指定环境数量
python scripts/rsl_rl/train.py --headless --task Unitree-G1-29dof-Velocity --num_envs 2048

# 使用便捷脚本
./unitree_rl_lab.sh -t --task Unitree-G1-29dof-Mimic-Gangnanm-Style

# 带视频录制的训练
python scripts/rsl_rl/train.py --video --task Unitree-G1-29dof-Velocity

# 分布式训练
python scripts/rsl_rl/train.py --distributed --task Unitree-G1-29dof-Velocity
```

### 推理演示

```bash
# 加载最新模型
python scripts/rsl_rl/play.py --task Unitree-G1-29dof-Mimic-Gangnanm-Style

# 使用便捷脚本
./unitree_rl_lab.sh -p --task Unitree-G1-29dof-Mimic-Gangnanm-Style
```

### 任务列表

```bash
# 列出所有可用任务
./unitree_rl_lab.sh -l
```

### 安装

```bash
# 激活 Isaac Lab 环境
conda activate env_isaaclab

# 安装项目
./unitree_rl_lab.sh -i

# 重启 shell 以激活环境变更
```

---

## 参考资料

- [Isaac Lab 官方文档](https://isaac-sim.github.io/IsaacLab)
- [RSL-RL GitHub](https://github.com/leggedrobotics/rsl_rl)
- [Unitree ROS](https://github.com/unitreerobotics/unitree_ros)
- [Unitree MuJoCo](https://github.com/unitreerobotics/unitree_mujoco)
- [Whole Body Tracking](https://github.com/HybridRobotics/whole_body_tracking)

---

*文档版本：v1.0*  
*最后更新：2024年*
