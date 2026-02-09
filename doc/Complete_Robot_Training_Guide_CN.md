# Unitree RL Lab 完整机器人训练指南

> 本指南基于 Unitree G1 29DOF 人形机器人的 Parkour 训练项目，全面覆盖从理论到实践的所有关键知识点。

---

## 目录

1. [项目架构总览](#1-项目架构总览)
2. [核心概念：强化学习与机器人控制](#2-核心概念强化学习与机器人控制)
3. [执行器物理模型详解](#3-执行器物理模型详解)
4. [机器人配置与组装](#4-机器人配置与组装)
5. [环境配置深度解析](#5-环境配置深度解析)
6. [奖励函数设计艺术](#6-奖励函数设计艺术)
7. [地形系统完全指南](#7-地形系统完全指南)
8. [训练流程与参数调优](#8-训练流程与参数调优)
9. [常见问题与解决方案](#9-常见问题与解决方案)
10. [附录：Python 类型提示与配置系统](#10-附录python-类型提示与配置系统)
11. [Sim-to-Real：跨越虚实鸿沟](#11-sim-to-real跨越虚实鸿沟)
12. [扩展技能开发工作流](#12-扩展技能开发工作流)
13. [附录 A：快速参考卡片](#附录-a快速参考卡片)
14. [附录 B：项目目录结构](#附录-b项目目录结构)

---

## 1. 项目架构总览

### 1.1 系统层级结构

```
┌─────────────────────────────────────────────────────────────────┐
│                        训练入口 (Entry Point)                    │
│                         scripts/rsl_rl/train.py                  │
└─────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                      算法层 (Algorithm Layer)                    │
│                           RSL-RL (PPO)                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │   Actor     │  │   Critic    │  │   Rollout Buffer      │  │
│  │  (策略网络)  │  │  (价值网络)  │  │   (轨迹缓冲区)         │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                      环境层 (Environment Layer)                  │
│                    Isaac Lab ManagerBasedRLEnv                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    parkour_env_cfg.py                     │   │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────┐ │   │
│  │  │ 观测管理器   │ │  动作管理器  │ │     奖励管理器       │ │   │
│  │  │ Observations│ │   Actions   │ │      Rewards        │ │   │
│  │  └─────────────┘ └─────────────┘ └─────────────────────┘ │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                       物理层 (Physics Layer)                     │
│                         NVIDIA PhysX / Isaac Sim                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                      unitree.py                           │   │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────┐ │   │
│  │  │  机器人模型  │ │  执行器配置  │ │     地形生成器       │ │   │
│  │  │  (USD/URDF) │ │ (Actuators) │ │  (TerrainGenerator) │ │   │
│  │  └─────────────┘ └─────────────┘ └─────────────────────┘ │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 数据流向图

```
                    ┌──────────────┐
                    │   环境状态    │
                    │ Observation  │
                    └──────┬───────┘
                           │
                           ▼
                    ┌──────────────┐
                    │   策略网络    │
                    │    Actor     │
                    └──────┬───────┘
                           │
                           ▼
                    ┌──────────────┐
                    │   动作输出    │
                    │   Actions    │
                    │ (关节角度目标) │
                    └──────┬───────┘
                           │
                           ▼
                    ┌──────────────┐
                    │   执行器     │
                    │  Actuator   │
                    │ (PD控制+物理) │
                    └──────┬───────┘
                           │
                           ▼
                    ┌──────────────┐
                    │   物理仿真    │
                    │   PhysX     │
                    └──────┬───────┘
                           │
                           ▼
                    ┌──────────────┐
                    │   新状态     │
                    │ + 奖励信号   │
                    └──────────────┘
```

### 1.3 核心文件职责

| 文件 | 职责 | 类比 |
|------|------|------|
| `unitree_actuators.py` | 定义电机物理特性 (扭矩-转速曲线) | 电机规格书 |
| `unitree.py` | 定义机器人整体结构和关节映射 | 机器人装配图 |
| `parkour_env_cfg.py` | 定义训练环境配置 | 训练场设计图 |
| `rewards.py` | 定义奖励函数 | 评分标准 |
| `train.py` | 训练入口脚本 | 教练 |
| `play.py` | 推理/测试脚本 | 考官 |

---

## 2. 核心概念：强化学习与机器人控制

### 2.1 强化学习基础框架

强化学习的核心是 **Agent（智能体）** 与 **Environment（环境）** 的交互循环：

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│   ┌─────────┐    观测 (Observation)    ┌─────────────────┐ │
│   │         │ ◄─────────────────────── │                 │ │
│   │  Agent  │                          │   Environment   │ │
│   │ (策略π) │ ──────────────────────► │   (仿真世界)    │ │
│   │         │    动作 (Action)         │                 │ │
│   └─────────┘                          └─────────────────┘ │
│        ▲                                       │           │
│        │           奖励 (Reward)               │           │
│        └───────────────────────────────────────┘           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 PPO 算法核心思想

PPO (Proximal Policy Optimization) 是目前机器人控制领域最常用的算法。

> **⚠️ 重要区分**：PPO 使用的是 **Rollout Buffer（轨迹缓冲区）**，而非 DQN 等算法使用的 Experience Replay Buffer。
> - **Rollout Buffer**: 存储当前策略收集的轨迹，用完即丢弃（on-policy）
> - **Replay Buffer**: 存储历史经验，可重复采样（off-policy）

```python
# PPO 的核心思想（伪代码）
for iteration in range(max_iterations):
    # 1. 使用当前策略收集轨迹 (On-Policy)
    trajectories = collect_trajectories(policy, environment, num_steps=24)
    
    # 2. 计算优势函数 (GAE - Generalized Advantage Estimation)
    advantages = compute_gae(trajectories, value_function, gamma=0.99, lambda_=0.95)
    
    # 3. 多轮更新策略（带裁剪，防止更新过大）
    for epoch in range(num_epochs):  # 通常 5 轮
        ratio = new_policy(a|s) / old_policy(a|s)
        # 核心：限制策略更新幅度，防止崩溃
        clipped_ratio = clip(ratio, 1-ε, 1+ε)  
        policy_loss = -min(ratio * advantage, clipped_ratio * advantage)
        
        update_policy(policy_loss)
        update_value_function(value_loss)
    
    # 4. 丢弃旧轨迹，下一轮重新收集
    trajectories.clear()
```

**关键参数解释**：

| 参数 | 含义 | 典型值 | 调参建议 |
|------|------|--------|---------|
| `learning_rate` | 学习率 | 1e-3 ~ 1e-4 | 训练不稳定时降低 |
| `num_learning_epochs` | 每批数据训练轮数 | 5 | 过大导致过拟合 |
| `clip_param` (ε) | 策略裁剪范围 | 0.2 | 控制更新步长 |
| `gamma` | 折扣因子 | 0.99 | 越大越看重长期回报 |
| `lam` (λ) | GAE 参数 | 0.95 | 平衡偏差与方差 |
| `entropy_coef` | 熵正则化系数 | 0.01 | 鼓励探索 |
| `num_steps_per_env` | 每环境收集步数 | 24 | 影响轨迹长度 |
| `num_envs` | 并行环境数 | 4096 | 越大采样越快 |

### 2.3 观测空间设计

观测 (Observation) 是 Agent 感知世界的"眼睛"：

```python
# G1 29DOF Parkour 的观测空间示例
Observation = [
    # === 本体感知 (Proprioception) ===
    base_lin_vel (3),      # 基座线速度 [vx, vy, vz]
    base_ang_vel (3),      # 基座角速度 [wx, wy, wz]  
    projected_gravity (3), # 投影重力方向（姿态信息）
    joint_pos (29),        # 各关节角度
    joint_vel (29),        # 各关节角速度
    
    # === 任务目标 ===
    velocity_commands (3), # 速度指令 [vx_cmd, vy_cmd, wz_cmd]
    
    # === 外部感知 (Exteroception) ===
    height_scan (225),     # 地形高度扫描 (15x15 网格)
    
    # === 历史信息 ===
    last_actions (29),     # 上一时刻的动作
]
# 总维度: 3+3+3+29+29+3+225+29 = 324
```

### 2.4 动作空间设计

动作 (Action) 是 Agent 控制机器人的"手"：

```python
# 动作空间：关节角度增量（相对于默认站姿）
Action = [
    # 左腿 (6 DOF)
    left_hip_yaw, left_hip_roll, left_hip_pitch,
    left_knee, left_ankle_pitch, left_ankle_roll,
    
    # 右腿 (6 DOF)
    right_hip_yaw, right_hip_roll, right_hip_pitch,
    right_knee, right_ankle_pitch, right_ankle_roll,
    
    # 腰部 (3 DOF)
    waist_yaw, waist_roll, waist_pitch,
    
    # 左臂 (7 DOF)
    left_shoulder_pitch, left_shoulder_roll, left_shoulder_yaw,
    left_elbow, left_wrist_roll, left_wrist_pitch, left_wrist_yaw,
    
    # 右臂 (7 DOF)
    right_shoulder_pitch, right_shoulder_roll, right_shoulder_yaw,
    right_elbow, right_wrist_roll, right_wrist_pitch, right_wrist_yaw,
]
# 总维度: 6+6+3+7+7 = 29
```

---

## 3. 执行器物理模型详解

### 3.1 扭矩-转速曲线 (T-N Curve)

这是 `unitree_actuators.py` 的核心——它模拟了真实电机的物理极限。

```
            扭矩 (Torque), N·m
                ^
    Y2──────────|
                |──────────────Y1     ← 峰值扭矩区
                |              │\
                |              │ \
                |              │  \    ← 线性衰减区
                |              |   \
    ------------+--------------|-----\--> 转速 (velocity): rad/s
               0              X1    X2
                               ↑     ↑
                               │     └── 空载转速 (No-load speed)
                               └──────── 额定转速 (Rated speed)
```

**物理含义**：

| 参数 | 物理含义 | 影响 |
|------|---------|------|
| **Y1** | 同向峰值扭矩 (扭矩与转速同方向) | 电机输出的最大"推力" |
| **Y2** | 反向峰值扭矩 (扭矩与转速反方向) | 刹车/反向时的能力 |
| **X1** | 额定转速 (Knee Point) | 能保持满扭矩的最大速度 |
| **X2** | 空载转速 (No-load speed) | 无负载时的最大转速 |

**代码实现解析**：

```python
def _clip_effort(self, effort: torch.Tensor) -> torch.Tensor:
    """根据当前转速限制输出扭矩"""
    
    # 判断扭矩和转速是否同方向
    same_direction = (self._joint_vel * effort) > 0
    
    # 同方向用 Y1，反方向用 Y2（反向刹车能力更强）
    max_effort = torch.where(same_direction, self._effort_y1, self._effort_y2)
    
    # 如果转速超过 X1，进入衰减区
    max_effort = torch.where(
        self._joint_vel.abs() < self._velocity_x1,  # 还在额定范围内
        max_effort,                                  # 保持满扭矩
        self._compute_effort_limit(max_effort)       # 线性衰减
    )
    
    return torch.clip(effort, -max_effort, max_effort)
```

### 3.2 摩擦力模型

真实电机存在摩擦损耗，代码通过以下公式模拟：

```python
# 摩擦力 = 静摩擦 + 动摩擦
friction = Fs * tanh(velocity / Va) + Fd * velocity
#          └──────┬──────────────┘   └──────┬──────┘
#                 │                         │
#         静摩擦（低速时饱和）        动摩擦（与速度成正比）
```

**图示**：

```
摩擦力 (Friction)
    ^
    │           ╭──────────────── Fs + Fd*v
    │          ╱
 Fs ├─────────╱
    │        ╱
    │       ╱
    │      ╱
    │     ╱
----│----╱--------------------------------> 速度 (velocity)
    │   ╱
-Fs ├──╱
    │
```

### 3.3 电机型号对比

```python
# 以下是 Unitree 常用电机的参数对比

# G1/H1 腿部主关节电机
class UnitreeActuatorCfg_N7520_14p3:
    X1 = 22.63   # 额定转速 (rad/s) ≈ 216 RPM
    X2 = 35.52   # 空载转速 (rad/s) ≈ 339 RPM  
    Y1 = 71      # 峰值扭矩 (N·m)
    Y2 = 83.3    # 反向峰值扭矩
    Fs = 1.6     # 静摩擦系数
    Fd = 0.16    # 动摩擦系数

# G1/H1 腿部大关节（髋部）
class UnitreeActuatorCfg_N7520_22p5:
    X1 = 14.5    # 转速较低
    X2 = 22.7
    Y1 = 111.0   # 扭矩更大
    Y2 = 131.0

# Go2 四足机器人电机
class UnitreeActuatorCfg_Go2HV:
    X1 = 13.5
    X2 = 30      # 转速高
    Y1 = 20.2    # 扭矩相对小（四足轻）
    Y2 = 23.4

# G1 腰部大扭矩电机
class UnitreeActuatorCfg_M107_24:
    X1 = 8.8
    X2 = 16
    Y1 = 240     # 超大扭矩（承载上半身）
    Y2 = 292.5
```

**性能对比图**：

```
扭矩 (N·m)
    ^
300 │                              ╭── M107_24 (腰部)
    │                            ╱
250 │                          ╱
    │                        ╱
200 │                      ╱
    │                    ╱
150 │        ╭── N7520_22p5 (髋部大关节)
    │      ╱
100 │    ╱───── N7520_14p3 (腿部常规)
    │  ╱
 50 │╱
    │────────── Go2HV (四足)
  0 └──────────────────────────────────> 转速 (rad/s)
    0    10    20    30    40
```

---

## 4. 机器人配置与组装

### 4.1 unitree.py 与 unitree_actuators.py 的关系

```
┌──────────────────────────────────────────────────────────────┐
│                    unitree_actuators.py                       │
│                      (零件库 / Parts)                         │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐  │
│  │ N7520_14p3     │  │ N7520_22p5     │  │ M107_24        │  │
│  │ 腿部常规电机    │  │ 髋部大电机      │  │ 腰部电机       │  │
│  └────────────────┘  └────────────────┘  └────────────────┘  │
└──────────────────────────────────────────────────────────────┘
                              │
                              │ 引用
                              ▼
┌──────────────────────────────────────────────────────────────┐
│                        unitree.py                             │
│                    (整机装配 / Assembly)                       │
│  ┌────────────────────────────────────────────────────────┐  │
│  │                    G1_29dof_Cfg                         │  │
│  │  ├── spawn: UsdFileCfg(usd_path="g1_29dof.usd")        │  │
│  │  ├── init_state: 默认站立姿态                           │  │
│  │  └── actuators:                                         │  │
│  │      ├── 腿部关节 → 使用 N7520_14p3                     │  │
│  │      ├── 髋部关节 → 使用 N7520_22p5                     │  │
│  │      ├── 腰部关节 → 使用 M107_24                        │  │
│  │      └── 手臂关节 → 使用 N5020_16                       │  │
│  └────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
```

### 4.2 机器人配置详解

```python
# unitree.py 中的机器人配置示例

@configclass
class UNITREE_G1_29DOF_CFG(UnitreeArticulationCfg):
    """G1 29自由度机器人配置"""
    
    # 1. 3D 模型文件
    spawn = UnitreeUsdFileCfg(
        usd_path=f"{UNITREE_MODEL_DIR}/g1_29dof.usd",
    )
    
    # 2. 初始状态（站立姿态）
    init_state = ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.78),  # 初始位置 (x, y, z)，z=0.78 是站立高度
        joint_pos={
            # 腿部初始弯曲角度
            "left_hip_pitch_joint": -0.1,
            "left_knee_joint": 0.3,
            "left_ankle_pitch_joint": -0.2,
            # ... 其他关节
        },
    )
    
    # 3. 执行器配置（关节 -> 电机映射）
    actuators = {
        # 腿部常规关节
        "legs": unitree_actuators.UnitreeActuatorCfg_N7520_14p3(
            joint_names_expr=[
                "left_hip_yaw_joint", "left_hip_roll_joint",
                "right_hip_yaw_joint", "right_hip_roll_joint",
                # ... 膝盖、脚踝等
            ],
            stiffness=200.0,  # PD 控制器刚度
            damping=10.0,     # PD 控制器阻尼
        ),
        
        # 髋部大关节（需要更大扭矩）
        "hip_pitch": unitree_actuators.UnitreeActuatorCfg_N7520_22p5(
            joint_names_expr=[
                "left_hip_pitch_joint",
                "right_hip_pitch_joint",
            ],
            stiffness=200.0,
            damping=10.0,
        ),
        
        # 腰部关节（承载上半身，最大扭矩）
        "torso": unitree_actuators.UnitreeActuatorCfg_M107_24(
            joint_names_expr=["torso_joint"],
            stiffness=300.0,
            damping=15.0,
        ),
    }
    
    # 4. 关节 SDK 名称映射（用于部署）
    joint_sdk_names = [
        "left_hip_yaw", "left_hip_roll", "left_hip_pitch",
        # ... 按 SDK 顺序排列
    ]
```

### 4.3 PD 控制器原理

执行器内部使用 PD 控制器将动作（目标角度）转换为扭矩：

```python
# PD 控制公式
torque = Kp * (target_pos - current_pos) + Kd * (target_vel - current_vel)
#        └─────────┬─────────────────┘   └─────────────┬───────────────┘
#              位置误差 × 刚度                    速度误差 × 阻尼

# 其中：
# Kp = stiffness (刚度)
# Kd = damping (阻尼)
# target_vel 通常为 0（静态目标）
```

**参数调节指南**：

| 参数 | 效果 | 过大 | 过小 |
|------|------|------|------|
| **Stiffness (Kp)** | 响应速度 | 振荡、不稳定 | 软趴趴、跟不上目标 |
| **Damping (Kd)** | 阻尼振荡 | 迟钝、过阻尼 | 振荡、抖动 |

---

## 5. 环境配置深度解析

### 5.1 parkour_env_cfg.py 整体结构

```python
@configclass
class G1_29dof_ParkourEnvCfg(ManagerBasedRLEnvCfg):
    """Parkour 环境配置主类"""
    
    # === 1. 场景配置 ===
    scene: RobotSceneCfg = RobotSceneCfg(...)
    
    # === 2. 观测配置 ===
    observations: ObservationsCfg = ObservationsCfg(...)
    
    # === 3. 动作配置 ===
    actions: ActionsCfg = ActionsCfg(...)
    
    # === 4. 命令配置 ===
    commands: CommandsCfg = CommandsCfg(...)
    
    # === 5. 奖励配置 ===
    rewards: RewardsCfg = RewardsCfg(...)
    
    # === 6. 终止条件 ===
    terminations: TerminationsCfg = TerminationsCfg(...)
    
    # === 7. 课程学习 ===
    curriculum: CurriculumCfg = CurriculumCfg(...)
    
    # === 8. 域随机化 ===
    events: EventCfg = EventCfg(...)
```

### 5.2 场景配置 (SceneCfg)

```python
@configclass  
class RobotSceneCfg(InteractiveSceneCfg):
    """定义仿真场景中的所有实体"""
    
    # 1. 地形
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=PARKOUR_TERRAINS_CFG,  # 地形生成器配置
    )
    
    # 2. 机器人
    robot: ArticulationCfg = UNITREE_G1_29DOF_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot"
    )
    
    # 3. 接触力传感器（脚底）
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*_ankle_roll_link",
        track_air_time=True,  # 追踪腾空时间
    )
    
    # 4. 高度扫描传感器（感知前方地形）
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
        attach_yaw_only=True,  # 只跟随偏航角，不跟随俯仰
        pattern_cfg=GridPatternCfg(
            resolution=0.1,        # 每个点间隔 0.1m
            size=[4.0, 1.0],       # 前方 4m × 左右 1m
        ),
        mesh_prim_paths=["/World/ground"],
    )
```

### 5.3 观测配置详解

```python
@configclass
class ObservationsCfg:
    """观测空间定义"""
    
    @configclass
    class PolicyCfg(ObsGroup):
        """策略网络的输入"""
        
        # 本体感知
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)           # 基座线速度
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)           # 基座角速度
        projected_gravity = ObsTerm(func=mdp.projected_gravity) # 姿态
        
        # 关节状态
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,     # 相对于默认姿态的偏移
            noise=Unoise(n_min=-0.01, n_max=0.01),  # 添加噪声，增强鲁棒性
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            noise=Unoise(n_min=-1.5, n_max=1.5),
        )
        
        # 任务目标
        velocity_commands = ObsTerm(func=mdp.generated_commands)
        
        # 历史动作
        last_actions = ObsTerm(func=mdp.last_action)
        
        # 外部感知（地形）
        height_scan = ObsTerm(
            func=mdp.height_scan,
            clip=(-1.0, 1.0),  # 限制范围，防止异常值
        )
        
    policy: PolicyCfg = PolicyCfg()
```

### 5.4 动作配置

```python
@configclass
class ActionsCfg:
    """动作空间定义"""
    
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],          # 匹配所有关节
        scale=0.5,                   # 动作缩放因子
        use_default_offset=True,     # 基于默认站姿
    )
```

**动作缩放的意义**：

```python
# 实际目标角度 = 默认站姿 + 动作输出 × scale
target_joint_pos = default_joint_pos + action * scale

# scale = 0.5 意味着：
# - 网络输出 1.0 → 实际偏移 0.5 rad ≈ 28.6°
# - 网络输出范围 [-1, 1] → 实际偏移 [-0.5, 0.5] rad
```

### 5.5 命令配置 (速度指令)

```python
@configclass
class CommandsCfg:
    """速度指令生成器"""
    
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        heading_command=True,    # 使用航向控制而非角速度
        resampling_time_range=(10.0, 10.0),  # 每 10 秒采样新指令
        
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.5, 5.0),    # 前进速度 0.5~5.0 m/s
            lin_vel_y=(-0.5, 0.5),   # 侧移速度
            heading=(-3.14, 3.14),   # 航向角（全范围）
        ),
    )
```

---

## 6. 奖励函数设计艺术

### 6.1 奖励设计原则

```
┌────────────────────────────────────────────────────────────────┐
│                        奖励设计黄金法则                          │
├────────────────────────────────────────────────────────────────┤
│ 1. 明确目标：奖励应直接引导达成最终目标                           │
│ 2. 分层设计：主要目标 > 次要目标 > 风格偏好                       │
│ 3. 避免冲突：确保奖励之间不相互矛盾                               │
│ 4. 适度惩罚：惩罚过重会导致策略过于保守                           │
│ 5. 可调试性：使用权重 (weight) 而非改变公式                      │
└────────────────────────────────────────────────────────────────┘
```

### 6.2 奖励函数数学形式

```python
# 常用的奖励函数形式

# 1. 线性奖励（追踪误差）
reward = -|error|

# 2. 指数核（平滑追踪）
reward = exp(-error² / σ²)
# σ 控制容忍度，σ 越大越宽容

# 3. 平方惩罚
reward = -error²

# 4. 截断惩罚（超过阈值才惩罚）
reward = -max(0, |value| - threshold)
```

### 6.3 Parkour 奖励配置详解

```python
@configclass
class RewardsCfg:
    """Parkour 训练的奖励函数配置"""
    
    # ==================== 主要目标：速度追踪 ====================
    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=3.0,  # 最高权重！核心目标
        params={"command_name": "base_velocity", "std": 0.5},
    )
    # 数学形式: reward = exp(-||v_actual - v_command||² / 0.5²)
    
    # ==================== 安全约束：存活奖励 ====================
    is_alive = RewTerm(
        func=mdp.is_alive,
        weight=1.0,  # 只要没摔就给奖励
    )
    
    # ==================== 运动风格：步态周期 ====================
    gait = RewTerm(
        func=mdp.feet_gait,
        weight=0.5,
        params={
            "std": 0.1,
            "synced": False,    # 非同步 = 交替步态（走路/跑步）
            "period": 0.4,      # 步态周期 0.4 秒（快速步态）
            "command_name": "base_velocity",
        },
    )
    
    # ==================== 稳定性：姿态保持 ====================
    flat_orientation = RewTerm(
        func=mdp.flat_orientation_exp,
        weight=0.3,
        params={"std": 0.3},
    )
    # 惩罚身体倾斜过大
    
    # ==================== 平滑性：动作惩罚 ====================
    action_rate = RewTerm(
        func=mdp.action_rate_l2,
        weight=-0.01,  # 负权重 = 惩罚
    )
    # 惩罚动作剧烈变化，使运动平滑
    
    # ==================== 关节保护：扭矩惩罚 ====================
    joint_torques = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-0.0002,
    )
    # 惩罚大扭矩输出，保护电机
    
    # ==================== 危险行为：终止惩罚 ====================
    termination = RewTerm(
        func=mdp.is_terminated,
        weight=-10.0,  # 强力惩罚摔倒等终止行为
    )
```

### 6.4 步态奖励 (feet_gait) 详解

这是实现自然行走/跑步的关键奖励：

```python
def feet_gait(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    command_name: str,
    std: float,
    period: float,
    synced: bool = False,
) -> torch.Tensor:
    """
    鼓励双脚按照特定周期交替接触/离开地面
    
    参数:
        asset_cfg: 机器人资产配置
        sensor_cfg: 接触传感器配置
        command_name: 速度指令名称
        std: 高斯核标准差（容忍度）
        period: 步态周期（秒）
        synced: 
            True  = 同步步态（双脚同时着地，如跳跃）
            False = 交替步态（左右脚交替，如走路/跑步）
    
    返回:
        奖励张量，shape: (num_envs,)
    """
    
    # 获取接触传感器
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    
    # 获取速度指令
    command = env.command_manager.get_command(command_name)
    
    # 计算目标相位 (0~1 的周期信号)
    # 相位随时间推进：phase = (t / period) mod 1
    phase = env.episode_length_buf[:, None] * env.step_dt / period
    
    # 计算左右脚的目标相位
    if synced:
        # 同步：双脚相位相同 (跳跃/双脚同时落地)
        foot_phase = torch.cat([phase, phase], dim=1)
    else:
        # 交替：右脚相位偏移 0.5 (走路/跑步)
        foot_phase = torch.cat([phase, phase + 0.5], dim=1)
    
    # 计算目标接触状态
    # 使用余弦函数：cos(2π*phase) > 0 时应该接触地面
    # 即 phase ∈ [0, 0.25] ∪ [0.75, 1] 时接触，[0.25, 0.75] 时腾空
    target_contact = torch.cos(2 * torch.pi * foot_phase) > 0
    
    # 获取实际接触状态（接触力 > 阈值）
    actual_contact = contact_sensor.data.net_forces_w_history[:, :, :, 2].max(dim=-1).values > 1.0
    
    # 计算奖励：目标与实际接触状态的匹配度
    error = (target_contact.float() - actual_contact.float()).abs()
    reward = torch.exp(-error.sum(dim=1) / std)
    
    # 静止时（低速）不给步态奖励，避免原地踏步
    stance_mask = command[:, 0].abs() < 0.1  # 前进速度 < 0.1 m/s
    reward[stance_mask] = 0.0
    
    return reward
```

**⚠️ 注意事项**：
1. **`synced=False`** 是最常用的设置，对应正常行走/跑步的交替步态
2. **`period` 应该与速度匹配**：高速需要更短的周期（更快的步频）
3. **静止时不给步态奖励**：防止机器人原地踏步来刷分

**周期参数对运动风格的影响**：

| Period (秒) | 步频 (步/秒) | 运动风格 |
|-------------|-------------|---------|
| 1.0 | 1.0 | 慢走 |
| 0.6 | 1.67 | 快走 |
| 0.4 | 2.5 | 慢跑 |
| 0.3 | 3.33 | 快跑 |
| 0.2 | 5.0 | 冲刺 |

---

## 7. 地形系统完全指南

### 7.1 地形生成器架构

```python
@configclass
class TerrainGeneratorCfg:
    """地形生成器配置"""
    
    size: tuple[float, float]      # 地形总尺寸 (length, width)
    border_width: float            # 边界宽度（平地缓冲区）
    num_rows: int                  # 行数（难度梯度）
    num_cols: int                  # 列数（类型变化）
    
    sub_terrains: dict[str, SubTerrainBaseCfg]  # 子地形配置
```

### 7.2 地形布局可视化

```
┌─────────────────────────────────────────────────────────────────┐
│                         border_width                             │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  (0,0)    (0,1)    (0,2)    (0,3)    (0,4)    (0,5)     │   │
│   │  flat     gap      box      stair    flat     gap       │   │
│   ├─────────────────────────────────────────────────────────┤   │
│   │  (1,0)    (1,1)    (1,2)    (1,3)    (1,4)    (1,5)     │   │
│   │  box      stair    flat     gap      box      stair     │   │ ← num_rows
│   ├─────────────────────────────────────────────────────────┤   │
│   │  (2,0)    (2,1)    (2,2)    (2,3)    (2,4)    (2,5)     │   │
│   │  stair    flat     gap      box      stair    flat      │   │
│   └─────────────────────────────────────────────────────────┘   │
│                         └────────── num_cols ──────────┘        │
└─────────────────────────────────────────────────────────────────┘
```

### 7.3 Parkour 混合地形配置

```python
PARKOUR_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(20.0, 20.0),
    border_width=5.0,  # 5m 平地边界
    num_rows=4,
    num_cols=8,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=True,  # 启用课程学习
    
    sub_terrains={
        # === 热身区（平地）===
        "flat_warmup": HfDiscreteObstaclesTerrainCfg(
            proportion=0.1,
            size=(8.0, 8.0),
            obstacle_height_mode="fixed",
            obstacle_height_range=(0.0, 0.0),
        ),
        
        # === 间隙地形 ===
        "mix_gap_1": MeshGapTerrainCfg(
            proportion=0.15,
            size=(8.0, 8.0),
            platform_width=0.8,
            gap_width_range=(0.2, 0.6),  # 宽度 20~60 cm
        ),
        
        # === 障碍物（箱子）===
        "mix_box_1": MeshRandomGridTerrainCfg(
            proportion=0.15,
            size=(8.0, 8.0),
            grid_width=0.5,
            grid_height_range=(0.1, 0.25),  # 高度 10~25 cm
        ),
        
        # === 楼梯 ===
        "mix_stair_1": MeshPyramidStairsTerrainCfg(
            proportion=0.15,
            size=(8.0, 8.0),
            step_height_range=(0.1, 0.2),  # 台阶高度 10~20 cm
            step_width=0.3,
        ),
        
        # 更多交替地形...
        "mix_gap_2": MeshGapTerrainCfg(...),
        "mix_box_2": MeshRandomGridTerrainCfg(...),
        "mix_stair_2": MeshPyramidStairsTerrainCfg(...),
    },
)
```

### 7.4 地形类型详解

```
1. MeshGapTerrainCfg (间隙地形)
   ┌────┐      ┌────┐      ┌────┐
   │    │      │    │      │    │
   │    │      │    │      │    │
   └────┘      └────┘      └────┘
        ↑ gap ↑      ↑ gap ↑

2. MeshRandomGridTerrainCfg (随机方块)
   ┌──┐  ┌────┐     ┌─┐
   │  │  │    │     │ │  ┌──┐
   │  │  │    │  ┌──┤ │  │  │
   └──┘  └────┘  │  └─┘  └──┘
                 └──┘

3. MeshPyramidStairsTerrainCfg (金字塔楼梯)
                 ┌───┐
              ┌──┤   │
           ┌──┤  │   │
        ┌──┤  │  │   │
   ─────┴──┴──┴──┴───┴─────

4. HfDiscreteObstaclesTerrainCfg (离散障碍)
        ▲         ▲
       ╱ ╲       ╱ ╲      ▲
   ───╱───╲─────╱───╲────╱─╲───
```

### 7.5 课程学习 (Curriculum)

```python
@configclass
class CurriculumCfg:
    """课程学习配置"""
    
    terrain_levels = CurrTerm(
        func=mdp.terrain_levels_vel,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "min_distance": 5.0,   # 走 5m 升级
            "max_distance": 15.0,  # 走 15m 后稳定
        },
    )
```

**课程学习流程**：

```
Level 0 (Easy)     Level 1 (Medium)    Level 2 (Hard)
┌────────────┐     ┌────────────┐      ┌────────────┐
│ 小间隙      │  →  │ 中等间隙    │  →   │ 大间隙      │
│ 低台阶      │     │ 中等台阶    │      │ 高台阶      │
│ 小障碍      │     │ 中等障碍    │      │ 大障碍      │
└────────────┘     └────────────┘      └────────────┘
      ↑                  ↑                   ↑
  机器人走得远       自动升级难度         最终挑战
```

---

## 8. 训练流程与参数调优

### 8.1 训练命令

```bash
# 基础训练
./unitree_rl_lab.sh -t --task Unitree-G1-29dof-Parkour --headless

# 带参数训练
python scripts/rsl_rl/train.py \
    --task Unitree-G1-29dof-Parkour \
    --headless \
    --num_envs 4096 \
    --max_iterations 10000

# 恢复训练
./unitree_rl_lab.sh -t --task Unitree-G1-29dof-Parkour --resume

# 从特定 checkpoint 恢复
python scripts/rsl_rl/train.py \
    --task Unitree-G1-29dof-Parkour \
    --load_run 2026-01-18_12-00-00 \
    --checkpoint model_5000.pt
```

### 8.2 训练监控

```bash
# 使用 TensorBoard 监控
tensorboard --logdir logs/rsl_rl/

# 关键指标
# - mean_reward: 平均奖励（应持续上升）
# - mean_episode_length: 平均 episode 长度（应增加）
# - policy_loss: 策略损失（应下降后稳定）
# - value_loss: 价值损失（应下降后稳定）
```

### 8.3 推理/测试

```bash
# 可视化测试
python scripts/rsl_rl/play.py --task Unitree-G1-29dof-Parkour

# 录制视频
python scripts/rsl_rl/play.py \
    --task Unitree-G1-29dof-Parkour \
    --video \
    --video_length 1000
```

### 8.4 超参数调优指南

| 问题现象 | 可能原因 | 解决方案 |
|---------|---------|---------|
| 奖励不增长 | 学习率过高/过低 | 调整 `learning_rate` |
| 策略振荡 | clip_param 过大 | 降低 `clip_param` |
| 探索不足 | entropy 过小 | 增加 `entropy_coef` |
| 过拟合 | epoch 过多 | 减少 `num_learning_epochs` |
| 机器人摔倒多 | 终止惩罚太小 | 增加 `termination` 权重 |
| 步态不自然 | 步态奖励太低 | 增加 `gait` 权重 |
| 动作抖动 | 动作惩罚太低 | 增加 `action_rate` 权重 |

### 8.5 典型训练曲线

```
Mean Reward
    ^
400 │                              ╭────── 收敛
    │                            ╱
300 │                          ╱
    │                        ╱
200 │                      ╱
    │                    ╱
100 │       ╭──────────╱
    │      ╱   ← 快速上升期
  0 │─────╱
    └──────────────────────────────────────> Iterations
         1k    2k    3k    4k    5k
```

---

## 9. 常见问题与解决方案

### 9.1 训练相关

**Q: 训练中断了，如何恢复？**

```bash
# 方法 1: 自动查找最新 checkpoint
./unitree_rl_lab.sh -t --task <TaskName> --resume

# 方法 2: 指定 run 目录
python scripts/rsl_rl/train.py --task <TaskName> --load_run <run_folder>

# 方法 3: 指定具体 checkpoint
python scripts/rsl_rl/train.py --task <TaskName> \
    --load_run 2026-01-18_12-00-00 \
    --checkpoint model_5000.pt
```

**Q: 机器人老是摔倒？**

1. 增加存活奖励：`is_alive.weight = 2.0`
2. 增加终止惩罚：`termination.weight = -20.0`
3. 增加姿态奖励：`flat_orientation.weight = 0.5`
4. 降低目标速度：`lin_vel_x = (0.0, 2.0)`

**Q: 机器人走路姿势奇怪？**

1. 添加关节偏离惩罚：
```python
joint_deviation = RewTerm(
    func=mdp.joint_deviation_l1,
    weight=-0.1,
    params={"asset_cfg": SceneEntityCfg("robot")},
)
```
2. 增加步态奖励权重
3. 添加动作平滑惩罚

### 9.2 环境相关

**Q: 地形类报错 `AttributeError`？**

Isaac Lab 版本不同，可用的地形类不同。检查可用类：

```python
from isaaclab.terrains.trimesh import mesh_terrains
print(dir(mesh_terrains))
```

常用替代：
- `MeshHurdleTerrainCfg` → `MeshRandomGridTerrainCfg`
- `MeshStepTerrainCfg` → `MeshPyramidStairsTerrainCfg`

**Q: 传感器数据异常？**

```python
# 检查 ray caster 配置
height_scanner = RayCasterCfg(
    ...
    debug_vis=True,  # 开启可视化调试
)
```

### 9.3 推理相关

**Q: `play.py` 找不到 checkpoint？**

1. 检查日志目录结构：
```bash
ls -la logs/rsl_rl/<ExperimentName>/
```

2. 手动指定：
```bash
python scripts/rsl_rl/play.py --task <TaskName> \
    --load_run <folder_name> \
    --checkpoint model_5000.pt
```

**Q: 推理时机器人行为与训练不同？**

1. 检查是否使用了相同的环境配置
2. 检查是否有域随机化差异
3. 确认观测噪声设置

---

## 10. 附录：Python 类型提示与配置系统

### 10.1 类型提示语法

```python
# 基础类型提示
def greet(name: str) -> str:
    return f"Hello, {name}"

# 类属性类型提示
class Person:
    name: str           # 声明 name 是字符串类型
    age: int            # 声明 age 是整数类型
    
    def __init__(self, name: str, age: int):
        self.name = name
        self.age = age

# 泛型类型
from typing import List, Dict, Optional

def process(items: List[int]) -> Dict[str, float]:
    ...

# Optional 表示可能为 None
def find(id: int) -> Optional[Person]:
    ...
```

### 10.2 Isaac Lab @configclass 装饰器

```python
from isaaclab.utils import configclass

@configclass
class MyConfig:
    """配置类示例"""
    
    # 带默认值的属性
    learning_rate: float = 1e-3
    batch_size: int = 256
    
    # 嵌套配置
    network: NetworkCfg = NetworkCfg()

# 使用配置
cfg = MyConfig()
cfg.learning_rate = 5e-4  # 修改参数

# 或直接初始化
cfg = MyConfig(learning_rate=5e-4, batch_size=512)
```

### 10.3 场景配置中的类型注解

```python
@configclass
class RobotSceneCfg(InteractiveSceneCfg):
    """场景配置"""
    
    # 类型注解：robot 必须是 ArticulationCfg 或其子类
    robot: ArticulationCfg = UNITREE_G1_29DOF_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot"
    )
    
    # 类型注解：contact_forces 必须是 ContactSensorCfg
    contact_forces: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*_ankle_roll_link",
    )
```

**类型注解的作用**：

1. **代码提示**：IDE 可以提供自动补全
2. **错误检查**：静态分析器可以发现类型错误
3. **文档作用**：明确参数期望类型
4. **运行时验证**：某些框架会在运行时验证类型

---

## 11. Sim-to-Real：跨越虚实鸿沟

将仿真训练的模型部署到真实机器人（Sim-to-Real）是强化学习最大的挑战。本节介绍核心策略。

### 11.1 差距来源 (The Gap)

| 来源 | 说明 | 对策 |
|------|------|------|
| **动力学误差** | 仿真物理引擎与真实物理的差异（如摩擦、弹性） | 域随机化 (Domain Randomization) |
| **执行器延时** | 通信延迟、电机响应滞后 | 建模延时 + 随机化延迟 |
| **传感器噪声** | 真实IMU和电机编码器有噪声 | 在观测中加入高斯噪声 |
| **地形感知** | 真实深度相机有盲区和噪点 | 训练时模拟丢包和随机遮挡 |

### 11.2 域随机化 (Domain Randomization)

我们在 `parkour_env_cfg.py` 的 `events` 部分配置随机化，迫使 Agent 适应各种物理参数的变化。

```python
@configclass
class EventCfg:
    """域随机化配置"""

    # 1. 物理属性随机化
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        params={
            "static_friction_range": (0.5, 1.25),  # 地面摩擦力变化
            "dynamic_friction_range": (0.5, 1.25),
            "restitution_range": (0.0, 0.2),       # 地面弹性
        },
    )

    # 2. 机器人质量随机化 (模拟负载变化)
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        params={"mass_distribution_params": (-2.0, 3.0)}, # 质量 ±kg
    )

    # 3. 初始状态随机化 (防止过拟合特定出生点)
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        params={"pose_range": {"x": (-0.5, 0.5), "yaw": (-3.14, 3.14)}},
    )
```

**关键策略**：
如果机器人在真机上站不稳，尝试**增大随机化范围**（特别是推力/质量/摩擦力），让 Agent 学会更鲁棒的策略。

### 11.3 执行器网络 (Actuator Network)

简单的 PD 控制有时不足以模拟复杂的电机特性。进阶做法是训练一个 **Actuator Net**（MLP），输入 `(q, dq, action)`，输出 `torque`，用来替代理想的 PD 公式。

> **注意**：Unitree RL Lab 目前使用精心标定的 `UnitreeActuator` 类（基于 T-N 曲线）来近似真实电机，这是比纯 PD 更准确的物理建模。

---

## 12. 扩展技能开发工作流

如果你想为一个新任务（比如“后空翻”或“搬运箱子”）开发策略，请遵循此工作流：

### 第一步：定义任务目标的 MDP
1.  **新建环境配置**：复制 `parkour_env_cfg.py` 为 `backflip_env_cfg.py`。
2.  **修改观测**：是否需要感知高度？是否需要感知目标物体？
3.  **设计动作**：后空翻可能需要更大的动作幅度，修改 `ActionCfg.scale`。

### 第二步：奖励工程 (Reward Engineering)
这是最难的一步。
*   **初期**：给予密集的引导奖励（如 `tracking_lin_vel`）。
*    **中期**：加入风格奖励（如 `feet_air_time`，确保腾空）。
*   **后期**：精调权重，去除辅助奖励。

**(示例) 后空翻奖励设计**：
```python
# 核心奖励：基座俯仰角速度达到目标值
backflip_rotation = RewTerm(
    func=mdp.track_ang_vel_y,
    weight=5.0,
    params={"target_vel": -6.0},  # 快速向后旋转
)

# 辅助奖励：跳跃高度
jump_height = RewTerm(
    func=mdp.base_height_target,
    weight=1.0,
    params={"target_height": 1.2},
)
```

### 第三步：课程学习 (Curriculum)
不要让机器人一开始就做完整动作。
1.  **Level 1**: 学习原地跳跃。
2.  **Level 2**: 学习向后跳。
3.  **Level 3**: 尝试空翻。

### 第四步：仿真验证
使用 `play.py` 观察：
*   动作是否自然？
*   是否有高频抖动（对电机有害）？
*   是否利用了仿真器的 Bug（例如利用穿模获得推力）？

---

## 结语

本指南涵盖了 Unitree RL Lab 项目的核心知识：

1. **架构层面**：从入口脚本到物理仿真的完整流程
2. **物理层面**：执行器模型、扭矩限制、摩擦建模
3. **环境层面**：观测、动作、奖励的设计与配置
4. **训练层面**：PPO 算法、超参数调优、课程学习
5. **调试层面**：常见问题诊断与解决方案
6. **部署层面**：Sim-to-Real 的关键策略

通过理解这些内容，你应该能够：
- 配置和训练自定义的运动技能
- 诊断和解决训练中的问题
- 设计适合特定任务的奖励函数
- 理解代码结构并进行修改扩展
- 为真机部署做好准备

---

## 附录 A：快速参考卡片

### 常用命令速查

```bash
# 列出所有可用任务
./unitree_rl_lab.sh -l

# 训练（无渲染）
./unitree_rl_lab.sh -t --task Unitree-G1-29dof-Parkour --headless

# 恢复训练
./unitree_rl_lab.sh -t --task Unitree-G1-29dof-Parkour --resume

# 推理测试
./unitree_rl_lab.sh -p --task Unitree-G1-29dof-Parkour

# 录制视频
python scripts/rsl_rl/play.py --task Unitree-G1-29dof-Parkour --video --video_length 500

# 监控训练
tensorboard --logdir logs/rsl_rl/
```

### 关键文件路径

| 用途 | 路径 |
|------|------|
| 环境配置 | `source/unitree_rl_lab/unitree_rl_lab/tasks/<robot>/<task>/` |
| 机器人资产 | `source/unitree_rl_lab/unitree_rl_lab/assets/robots/` |
| 奖励函数 | `source/unitree_rl_lab/unitree_rl_lab/tasks/mdp/rewards.py` |
| 训练日志 | `logs/rsl_rl/<experiment_name>/` |
| 部署代码 | `deploy/robots/<robot_name>/` |

### 奖励权重经验值

| 奖励类型 | 推荐权重范围 | 说明 |
|---------|-------------|------|
| 速度追踪 | 1.0 ~ 5.0 | 核心目标，应最高 |
| 存活奖励 | 0.5 ~ 2.0 | 防止过早终止 |
| 步态奖励 | 0.2 ~ 1.0 | 风格塑造 |
| 姿态保持 | 0.1 ~ 0.5 | 稳定性 |
| 动作平滑 | -0.001 ~ -0.05 | 惩罚项（负数） |
| 扭矩惩罚 | -0.0001 ~ -0.001 | 保护电机 |
| 终止惩罚 | -5.0 ~ -20.0 | 强力惩罚摔倒 |

---

## 附录 B：项目目录结构

```
unitree_rl_lab/
├── scripts/                    # 入口脚本
│   ├── rsl_rl/
│   │   ├── train.py           # 训练入口
│   │   ├── play.py            # 推理入口
│   │   └── cli_args.py        # 命令行参数
│   └── list_envs.py           # 列出可用环境
│
├── source/unitree_rl_lab/      # 主代码包
│   └── unitree_rl_lab/
│       ├── assets/             # 机器人资产
│       │   └── robots/
│       │       ├── unitree.py           # 机器人配置
│       │       └── unitree_actuators.py # 执行器配置
│       │
│       └── tasks/              # 任务定义
│           ├── mdp/            # MDP 组件
│           │   ├── rewards.py  # 奖励函数
│           │   ├── observations.py
│           │   └── terminations.py
│           │
│           └── <robot>/        # 机器人专属任务
│               └── <task>/
│                   └── *_env_cfg.py
│
├── deploy/                     # C++ 部署代码
│   ├── include/               # 头文件
│   └── robots/                # 机器人专属部署
│       ├── go2/
│       ├── g1_29dof/
│       └── h1/
│
├── logs/rsl_rl/               # 训练日志和模型
└── doc/                       # 文档
```

---

祝训练顺利！🤖

---

*本文档基于 Unitree RL Lab 项目，结合 Isaac Lab 框架编写。*
*最后更新：2026 年 1 月*
