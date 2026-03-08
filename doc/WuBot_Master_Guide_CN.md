# 🥋 武bot (Wu-Bot) 完整解析与开发指导手册

> **版本**: v5.0 — 基于源代码逐行审计撰写
> **适用对象**: 希望深入理解本项目架构、调参逻辑、并推进完整武术表演部署的开发者
> **最后更新**: 2026-03-07

---

## 目录

1. [项目愿景与核心创新](#1-项目愿景与核心创新)
2. [系统架构全景图](#2-系统架构全景图)
3. [数据管线深度解析](#3-数据管线深度解析)
4. [环境配置逐层拆解](#4-环境配置逐层拆解)
5. [奖励函数设计哲学](#5-奖励函数设计哲学)
6. [Front Kick 动作诊断与调参](#6-front-kick-动作诊断与调参)
7. [从单个动作到完整武术套路](#7-从单个动作到完整武术套路)
8. [C++ 部署端架构解析](#8-c-部署端架构解析)
9. [改进路线图与高级优化](#9-改进路线图与高级优化)
10. [快速参考卡片](#10-快速参考卡片)

---

## 1. 项目愿景与核心创新

### 1.1 目标

让 Unitree G1（29自由度）人形机器人完整表演一套武术招式，实现类似 **2026年春节联欢晚会机器人武术表演** 的效果。

### 1.2 核心难点

武术动作与普通行走/舞蹈的本质区别：

| 特性 | 行走 (Locomotion) | 舞蹈 (Gangnam Style) | 武术 (Martial Arts) |
|------|-----------|------|------|
| 支撑相 | 双脚交替 | 多数双脚 | 大量单脚（踢腿时） |
| 力学特性 | 周期性、对称 | 节奏性、低冲击 | 爆发性、高冲击、非对称 |
| 质心偏移 | 小幅前后 | 中等左右 | 极端偏移（高踢腿） |
| 容错性 | 高（自恢复步态） | 中 | 低（单脚站立时一推即倒） |
| 关节速度 | 低~中 | 中 | 极高（出拳/踢腿瞬间） |

### 1.3 架构创新：Policy Sequencer（策略串联器）

**核心洞察**：用一个网络学所有动作 → 灾难性遗忘 + 动作软绵。

**解决方案**：

```
训练层 (Isaac Lab)                    部署层 (C++)
┌─────────────────┐                 ┌───────────────────────────────┐
│ Front Kick  → ONNX_1 ─────┐      │  State_MartialArtsSequencer   │
│ Lunge Punch → ONNX_2 ─────┤      │                               │
│ Side Kick   → ONNX_3 ─────┼─────→│  ONNX_1 → hold → ONNX_2 →   │
│ Roundhouse  → ONNX_4 ─────┤      │  hold → ONNX_3 → hold → ...  │
│ Heian Shodan→ ONNX_5 ─────┤      │                               │
│ Bassai      → ONNX_6 ─────┤      │  YAML 配置：顺序可自由编排    │
│ Empi        → ONNX_7 ─────┘      └───────────────────────────────┘
└─────────────────┘
    7 个独立训练任务                    1 个 C++ 状态机串联播放
```

**为什么不拼接 NPZ 训一个模型**（已废弃的 Combo NPZ 方案）：
- 长序列 Credit Assignment 困难（踢腿不好 → 40秒前的冲拳被惩罚？）
- 一个动作训崩 → 整条链路崩溃
- 固定编排顺序，无法灵活组合

---

## 2. 系统架构全景图

### 2.1 文件结构与职责

```
unitree_rl_lab/
│
├── source/unitree_rl_lab/unitree_rl_lab/tasks/mimic/
│   ├── mdp/                           ← 🧠 共享 MDP 核心逻辑
│   │   ├── commands.py                # MotionCommand + MotionLoader + 自适应采样
│   │   ├── rewards.py                 # 追踪奖励函数 (位置/朝向/速度/关节)
│   │   ├── observations.py            # 机器人/动捕观测量
│   │   ├── terminations.py            # 终止条件 (偏差过大即重置)
│   │   └── events.py                  # 域随机化 (CoM/默认关节位/摩擦)
│   │
│   ├── agents/rsl_rl_ppo_cfg.py       ← PPO 超参数 (所有mimic任务共享)
│   │
│   └── robots/g1_29dof/
│       ├── gangnanm_style/            ← 🕺 舞蹈任务 (成熟参考)
│       │   ├── g1.py                  # G1_CYLINDER_CFG 机器人物理配置
│       │   └── tracking_env_cfg.py    # 舞蹈环境配置 (奖励基线)
│       │
│       └── martial_arts/             ← 🥋 武术任务 (本项目核心)
│           ├── __init__.py            # 7 个 gym.register 任务注册
│           ├── tracking_env_cfg.py    # 武术环境配置 (定制奖励+终止)
│           ├── G1_*.csv               # 7 套动捕数据 (CMU→G1 映射)
│           └── G1_*.npz               # 7 套训练用参考轨迹
│
├── scripts/mimic/
│   ├── martial_arts_pipeline.sh       ← 一键管线 (csv/npz/validate/train/deploy)
│   ├── cmu_amc_to_csv.py             # CMU ASF+AMC → G1 CSV 转换
│   ├── csv_to_npz.py                 # CSV → NPZ (Isaac Sim 回放录制)
│   └── validate_npz.py               # NPZ 质量检查
│
├── deploy/
│   ├── include/FSM/
│   │   └── State_MartialArtsSequencer.h  ← 🔑 C++ 策略串联核心
│   └── robots/g1_29dof/
│       ├── config/config.yaml         # 状态机配置 + segments 编排
│       └── include/State_Mimic.h      # 单动作 Mimic 部署基类
│
└── data/cmu_mocap/135/               ← CMU 原始数据 (Subject #135 空手道)
```

### 2.2 共享 vs 专有

武术项目遵循**最大复用**原则——只有 3 样东西是武术专有的：

| 组件 | 共享自 | 武术专有修改 |
|------|--------|-------------|
| 机器人物理模型 | `gangnanm_style/g1.py` | 无（完全复用 G1_CYLINDER_CFG） |
| MDP 核心函数 | `mimic/mdp/` | `motion_joint_pos_error_exp` (v4 新增) |
| PPO 超参数 | `agents/rsl_rl_ppo_cfg.py` | 无（完全复用 BasePPORunnerCfg） |
| 奖励权重 | — | `MartialArtsRewardsCfg`（武术专用权重） |
| 终止条件 | — | `MartialArtsTerminationsCfg`（ee_body_pos 放宽） |
| 动捕数据 | — | 7 套 CSV/NPZ |
| C++ 串联器 | — | `State_MartialArtsSequencer.h` |

---

## 3. 数据管线深度解析

### 3.1 完整数据流

```
CMU #135 (ASF+AMC, 120fps, Y-up, 角度制)
    │
    │  cmu_amc_to_csv.py
    │  ├── 骨骼定义解析: ASF → 骨骼链长度 + 层级关系
    │  ├── 坐标变换: CMU (X右,Y上,Z后) → Isaac (X前,Y左,Z上)
    │  ├── 身高标定: 人类leg_chain → G1骨盆高度 0.78m
    │  └── 关节映射: 30个CMU骨骼 → 29个G1关节角度
    ▼
G1_xxx.csv (每行36列: 3 root_pos + 4 root_quat + 29 joint_angles)
    │
    │  csv_to_npz.py (需要 Isaac Sim 运行)
    │  ├── 帧率降采样: 120fps → 50fps (线性插值)
    │  ├── Isaac Sim 回放: 将CSV逐帧写入仿真关节 → 自动计算正运动学
    │  └── 录制输出: body_pos_w, body_quat_w, body_lin_vel_w,
    │               body_ang_vel_w, joint_pos, joint_vel  全部存入NPZ
    ▼
G1_xxx.npz (形状示例: body_pos_w=[1150, 14, 3] 表示1150帧×14个body×3D坐标)
    │
    │  MotionLoader (commands.py) 在训练时加载
    ▼
PPO 训练 (4096并行环境, 30000 iterations, ~10小时)
    │
    │  play.py 自动导出
    ▼
policy.onnx → C++ State_MartialArtsSequencer 串联部署
```

### 3.2 NPZ 文件内部结构

`MotionLoader.__init__()` 加载以下字段：

| 字段名 | 形状 | 含义 |
|--------|------|------|
| `fps` | 标量 | 帧率 (50) |
| `joint_pos` | `[T, 29]` | 每帧的关节角度 (rad) |
| `joint_vel` | `[T, 29]` | 每帧的关节速度 (rad/s) |
| `body_pos_w` | `[T, N_bodies, 3]` | 每帧的 body 世界坐标 |
| `body_quat_w` | `[T, N_bodies, 4]` | 每帧的 body 世界朝向四元数 |
| `body_lin_vel_w` | `[T, N_bodies, 3]` | 每帧的 body 线速度 |
| `body_ang_vel_w` | `[T, N_bodies, 3]` | 每帧的 body 角速度 |

其中 `T` = 总帧数 (front_kick ≈ 1150帧 @ 50fps ≈ 23秒)。

### 3.3 14 个追踪 Body

```python
TRACKED_BODY_NAMES = [
    "pelvis",                      # 骨盆 (根节点)
    "left_hip_roll_link",          # 左髋
    "left_knee_link",              # 左膝
    "left_ankle_roll_link",        # 左踝 ← 末端执行器
    "right_hip_roll_link",         # 右髋
    "right_knee_link",             # 右膝
    "right_ankle_roll_link",       # 右踝 ← 末端执行器
    "torso_link",                  # 躯干 (锚点 anchor)
    "left_shoulder_roll_link",     # 左肩
    "left_elbow_link",             # 左肘
    "left_wrist_yaw_link",         # 左腕 ← 末端执行器
    "right_shoulder_roll_link",    # 右肩
    "right_elbow_link",            # 右肘
    "right_wrist_yaw_link",        # 右腕 ← 末端执行器
]
```

其中 4 个末端执行器 (hands + feet) 是武术的核心：拳头打到位、脚踢到高度。

---

## 4. 环境配置逐层拆解

### 4.1 物理仿真参数 (`MartialArtsBaseEnvCfg.__post_init__`)

```python
self.decimation = 4          # 控制频率 = 1/(0.005*4) = 50 Hz
self.episode_length_s = 30.0 # 每个episode最长30秒
self.sim.dt = 0.005          # 物理步长 = 200 Hz (PhysX)
```

**含义**：物理引擎以 200Hz 运行，但 RL 策略以 50Hz 做决策。每做一次决策，物理引擎走 4 步。

### 4.2 机器人物理模型 (`g1.py`)

G1 的驱动器参数源自**真实电机的物理测量值**：

```
NATURAL_FREQ = 10 × 2π ≈ 62.8 rad/s    ← 电机固有频率 10Hz
DAMPING_RATIO = 2.0                      ← 临界阻尼的2倍 (过阻尼)
```

刚度/阻尼按**电机型号**区分：

| 电机型号 | 关节 | 刚度 (Nm/rad) | 阻尼 (Nms/rad) |
|---------|------|---------|--------|
| 7520-22 | hip_roll, knee | 99.1 | 6.31 |
| 7520-14 | hip_pitch/yaw | 40.2 | 2.56 |
| 5020 | ankle, shoulder | 14.3 (×2=28.5 for ankle) | 0.91 (×2=1.81) |
| 4010 | wrist | 16.8 | 1.07 |

**`G1_ACTION_SCALE` 的计算逻辑**：

```python
# 动态计算: action_scale = 0.25 × effort_limit / stiffness
# 物理含义: 当网络输出±1时, 关节目标偏移 = ±(effort_limit/stiffness) × 0.25
# 这确保了不同电机型号有合理的运动范围
```

### 4.3 域随机化 (`EventCfg`)

武术相比舞蹈的域随机化差异：

| 参数 | 武术 | 舞蹈 | 原因 |
|------|------|------|------|
| `push_robot` 间隔 | 2~5秒 | 1~3秒 | 武术动作复杂，过于频繁推扰会阻碍学习 |
| `velocity_range.roll/pitch` | ±0.52 | ±0.52 | 相同 |
| `velocity_range.yaw` | ±0.78 | ±0.78 | 相同 |
| `physics_material` | 相同 | 相同 | 摩擦力随机化范围一致 |
| `base_com` | 相同 | 相同 | 重心随机化范围一致 |
| `add_joint_default_pos` | ±0.01 rad | ±0.01 rad | 模拟关节校准误差 |

### 4.4 观测空间 (`ObservationsCfg`)

#### Policy 观测 (部署时使用):

| 观测量 | 维度 | 噪声 | 说明 |
|--------|------|------|------|
| `motion_command` | 29+29=58 | 无 | 目标关节角度+速度 (来自NPZ当前帧) |
| `motion_anchor_ori_b` | 6 | ±0.05 | 当前锚点朝向误差 (旋转矩阵的前两列) |
| `base_ang_vel` | 3 | ±0.2 | 机体角速度 |
| `joint_pos_rel` | 29 | ±0.01 | 当前关节角度相对默认值的偏差 |
| `joint_vel_rel` | 29 | ±0.5 | 当前关节速度 |
| `last_action` | 29 | 无 | 上一帧的动作输出 |

**总观测维度**: 58 + 6 + 3 + 29 + 29 + 29 = **154**

#### Critic 特权观测 (仅训练时使用):

额外包含 `base_lin_vel`、`motion_anchor_pos_b`、`body_pos`、`body_ori` 等真值信息，让 Critic 能更好评估状态价值。

### 4.5 终止条件 (`MartialArtsTerminationsCfg`)

```python
# 锚点高度偏差 > 0.25m → 终止 (防止蹲下或跳起太多)
anchor_pos = bad_anchor_pos_z_only(threshold=0.25)

# 躯干朝向偏差 > 0.8 → 终止 (防止翻倒)
anchor_ori = bad_anchor_ori(threshold=0.8)

# 末端执行器Z轴偏差 > 0.6m → 终止
# ⚠️ 关键修改 (v4): 从 0.25m 放宽到 0.6m
# 原因: 正踢腿时脚部升高 ~0.8m, 0.25m阈值会在踢腿中途直接终止episode!
ee_body_pos = bad_motion_body_pos_z_only(threshold=0.6)
```

---

## 5. 奖励函数设计哲学

### 5.1 奖励架构总览

```
Total Reward = Σ(weight_i × reward_i)

正则化惩罚 (负权重 → 防止过激动作):
  ├── joint_acc_l2         × -2.5e-7    关节加速度惩罚
  ├── joint_torques_l2     × -1e-5      关节力矩惩罚
  ├── action_rate_l2       × -0.1       动作变化率惩罚
  ├── joint_pos_limits     × -10.0      关节极限惩罚
  └── undesired_contacts   × -0.1       非预期接触惩罚

锚点追踪 (躯干跟踪):
  ├── anchor_pos           × 0.5        躯干位置追踪
  └── anchor_ori           × 0.5        躯干朝向追踪

全身追踪 (14个body):
  ├── body_pos             × 1.5        身体各部位位置追踪
  ├── body_ori             × 1.5        身体各部位朝向追踪
  └── joint_pos (v4新增)   × 2.0        29个关节角度追踪

速度追踪 (辅助):
  ├── body_lin_vel         × 0.5        线速度追踪
  └── body_ang_vel         × 0.5        角速度追踪
```

### 5.2 exp 核函数机制

所有追踪奖励使用相同的 exp 核函数：

$$R = \exp\left(-\frac{\text{mean}(\|q_{\text{ref}} - q_{\text{robot}}\|^2)}{\sigma^2}\right)$$

**关键理解**：σ越小 → 奖励衰减越快 → 追踪越严格，但梯度消失风险越大。

| σ值 | 当误差=0.3时 | 当误差=0.5时 | 当误差=0.8时 |
|-----|-------------|-------------|-------------|
| 0.3 | 0.37 | 0.06 | 0.001 |
| 0.5 | 0.70 | 0.37 | 0.08 |
| 0.8 | 0.87 | 0.67 | 0.37 |

**教训**：v1-v3 中很多追踪项用 σ=0.3~0.4，在武术的大幅动作误差下奖励直接趋近0（梯度消失），策略学不到任何东西。

### 5.3 武术 vs 舞蹈的奖励差异（源代码逐项对比）

| 奖励项 | 武术 (weight/std) | 舞蹈 (weight/std) | 武术修改原因 |
|--------|----------|----------|------------|
| `body_pos` | **1.5** / std=**0.5** | 1.0 / std=0.3 | 初始误差~0.24m, std=0.3时梯度近乎消失 |
| `body_ori` | **1.5** / std=**0.8** | 1.0 / std=0.4 | 初始误差~0.79rad, std=0.4时reward=0.02(死!) |
| `joint_pos` | **2.0** / std=0.8 | ❌ 不存在 | **v4核心修复**: 防止"对的位置、错的姿势" |
| `body_lin_vel` | **0.5** / std=**1.5** | 1.0 / std=1.0 | 降权+放宽, 防止速度追踪抢占位置追踪 |
| `body_ang_vel` | **0.5** / std=**4.0** | 1.0 / std=3.14 | 同上 |
| `anchor_ori` | 0.5 / std=**0.8** | 0.5 / std=0.4 | 躯干旋转误差较大需放宽 |
| `ee_body_pos` 终止 | threshold=**0.6** | threshold=0.25 | 正踢腿脚高~0.8m, 0.25m会杀死踢腿 |

### 5.4 v4 新增 `motion_joint_pos_error_exp` 的核心作用

**问题根源**：`body_pos` 追踪 link 的质心位置，`body_ori` 追踪 link 的朝向。但一个机器人可以用**完全错误的关节配置**达到近似正确的质心轨迹（例如：膝盖过伸 + 踝关节过弯 → 质心不变但姿势扭曲）。

**解决方案**：直接追踪 29 个关节角度：

```python
def motion_joint_pos_error_exp(...):
    ref_joint_pos = command.joint_pos          # NPZ中的参考关节角
    robot_joint_pos = command.robot_joint_pos  # 机器人实际关节角
    error = torch.mean(torch.square(ref_joint_pos - robot_joint_pos), dim=-1)
    return torch.exp(-error / std**2)
```

**效果**：强制每个关节都接近参考值 → 消除"对的位置、错的姿势"现象。

### 5.5 自适应采样 (Adaptive Sampling)

`MotionCommand._adaptive_sampling()` 实现了**困难片段聚焦训练**：

```
时间轴: |---起手式---|---准备---|-!-踢腿-!-|---收腿---|---结束---|
失败率:      低          低        高!         中          低
采样概率:    5%         10%       50%!        25%         10%
```

- `adaptive_alpha=0.002`: EMA平滑系数，缓慢更新失败率统计
- `adaptive_kernel_size=3`: 卷积核大小，让相邻 bin 也被关注
- `adaptive_lambda=0.8`: 核权重衰减
- `adaptive_uniform_ratio=0.1`: 10% 概率均匀采样 (防止遗忘)

---

## 6. Front Kick 动作诊断与调参

### 6.1 当前训练状态

最新训练运行：`2026-03-04_23-00-52`，已完成 30000 iterations（保存了 model_29999.pt）。ONNX 已导出至 `exported/` 目录。

### 6.2 "怪异"的具体诊断

通过逐行对比武术和舞蹈的配置，以及代码中的注释历史（v1→v2→v3→v4），以下是 front_kick "动作怪异" 的可能原因和对应修复：

#### 问题 A: 上半身不自然扭动 / 手臂乱挥

**根因**：`motion_joint_pos` 的 std=0.8 对于 29 个关节来说仍然偏宽松。当整体 MSE = 0.64 rad² 时 reward ≈ 0.37，这意味着平均每个关节允许偏差 ~0.15 rad（约8.6度）。手臂因为力矩限制小（effort_limit=25Nm），在被推扰后恢复慢，容易晃荡。

**现已实施调整 (v5)**：

```python
# 简单方案: 收紧 std, 提升 weight
motion_joint_pos = RewTerm(
    func=mdp.motion_joint_pos_error_exp,
    weight=3.0,                    # ↑ 从 2.0
    params={"command_name": "motion", "std": 0.6},  # ↓ 从 0.8
)
```

**进阶方案**（需要扩展 `rewards.py`）：把 `joint_pos` 拆为上/下半身, 给不同权重:
- 上半身 (waist/shoulder/elbow/wrist): weight=1.5, std=0.5 → 严格保持姿势
- 下半身 (hip/knee/ankle): weight=2.5, std=0.8 → 允许腿部适应平衡

#### 问题 B: 踢腿时躯干过度前倾 / 后仰

**根因**：`anchor_ori` 的 std=0.8、weight=0.5 过于宽松。正踢腿时，策略为了保持平衡会大幅弯腰，导致观感"佝偻"。

**现已实施调整 (v5)**：

```python
motion_global_anchor_ori = RewTerm(
    func=mdp.motion_global_anchor_orientation_error_exp,
    weight=1.0,                    # ↑ 从 0.5 翻倍，强迫躯干稳定
    params={"command_name": "motion", "std": 0.8},  
)
```

#### 问题 C: 动作不够爆发 / 出现抖动

**根因**：`action_rate_l2` 的 weight=-0.1 会惩罚动作的快速变化。对于武术的瞬态爆发动作来说，这个惩罚过强。

**现已实施调整 (v5)**：

```python
action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.05)  # ↓ 从 -0.1 减半，解锁瞬间爆发力
```

#### 问题 D: 踢腿幅度不够

**根因**：全身 `body_pos` 的 std=0.5 对于脚尖在空间中 0.8m 的大幅移动仍可能不够。可以给末端执行器额外加权。

**现已实施调整 (v5)**：

```python
# 新增: 末端执行器专项追踪 (只追踪手脚以模拟春晚级爆发力与准度)
motion_ee_pos = RewTerm(
    func=mdp.motion_relative_body_position_error_exp,
    weight=3.0,  # 极高权重
    params={
        "command_name": "motion",
        "std": 0.2,   # 极苛刻要求
        "body_names": END_EFFECTOR_BODIES,
    },
)

# 新增: 末端执行器的打出瞬时速度，确保招式利落
motion_ee_lin_vel = RewTerm(
    func=mdp.motion_global_body_linear_velocity_error_exp,
    weight=1.0,  
    params={"command_name": "motion", "std": 1.0, "body_names": END_EFFECTOR_BODIES},
)
```

### 6.3 已经上线的 v5 调参汇总 (2026 最新代码已应用)

| 奖励项 | v4 (当前) | v5 (建议) | 修改原因 |
|--------|----------|----------|---------|
| `action_rate_l2` | weight=-0.1 | **-0.05** | 允许更爆发的出拳/踢腿 |
| `anchor_ori` | w=0.5, σ=0.8 | **w=1.0, σ=0.5** | 防止佝偻 |
| `joint_pos` | w=2.0, σ=0.8 | **w=3.0, σ=0.6** | 消除怪异姿势 |
| 其他项 | 保持 | 保持 | v4已调好 |

### 6.4 训练调试流程

```bash
# 1. 应用修改后重新训练 (先跑 5000 iterations 看趋势)
python scripts/rsl_rl/train.py \
    --task Unitree-G1-29dof-Mimic-MartialArts-FrontKick \
    --headless --max_iterations 5000

# 2. TensorBoard 对比新旧运行
tensorboard --logdir logs/rsl_rl/unitree_g1_29dof_mimic_martialarts_frontkick

# 3. 可视化播放
python scripts/rsl_rl/play.py \
    --task Unitree-G1-29dof-Mimic-MartialArts-FrontKick

# 4. 录制视频对比
python scripts/rsl_rl/play.py \
    --task Unitree-G1-29dof-Mimic-MartialArts-FrontKick --video
```

---

## 7. 从单个动作到完整武术套路

### 7.1 阶段路线图

```
阶段1 (当前)          阶段2              阶段3              阶段4
━━━━━━━━━━          ━━━━━━━━━━        ━━━━━━━━━━        ━━━━━━━━━━
Front Kick         扩展武器库          Sim2Sim验证        真机部署
精调至自然          训练其他6个动作     Mujoco交叉验证     吊架→落地
                   调整Sequencer过渡   排查接触力问题     连续表演
```

### 7.2 阶段 1: Front Kick 精调 (当前)

**检查清单**:
- [ ] 踢腿高度是否达到参考动作的 80% 以上？
- [ ] 上半身是否保持相对直立？（不佝偻）
- [ ] 收腿后是否能稳定站立？（不倾倒）
- [ ] 在 `push_robot` 扰动下是否仍能完成动作？
- [ ] 手臂是否保持合理姿势？（不乱挥）

### 7.3 阶段 2: 扩展武器库

Front Kick 调好后，用相同的 `MartialArtsRewardsCfg` 训练其他动作：

```bash
# 推荐训练顺序 (按难度递增)

# ⭐⭐ 简单动作 (先练)
python scripts/rsl_rl/train.py --task Unitree-G1-29dof-Mimic-MartialArts-LungePunch --headless

# ⭐⭐⭐ 中等动作
python scripts/rsl_rl/train.py --task Unitree-G1-29dof-Mimic-MartialArts-SideKick --headless
python scripts/rsl_rl/train.py --task Unitree-G1-29dof-Mimic-MartialArts-RoundhouseKick --headless

# ⭐⭐⭐⭐ 复杂套路
python scripts/rsl_rl/train.py --task Unitree-G1-29dof-Mimic-MartialArts-HeianShodan --headless

# ⭐⭐⭐⭐⭐ 长套路 (可能需要 50000+ iterations)
python scripts/rsl_rl/train.py --task Unitree-G1-29dof-Mimic-MartialArts-Bassai --headless
python scripts/rsl_rl/train.py --task Unitree-G1-29dof-Mimic-MartialArts-Empi --headless

# 或一键:
bash scripts/mimic/martial_arts_pipeline.sh train all
```

**动作特殊调整**:
- **Side Kick / Roundhouse Kick**: 腿侧面/后方运动更极端，可能需要放宽 `ee_body_pos` 终止阈值到 0.8m
- **Bassai / Empi**: 长达 43~51秒的套路，`episode_length_s` 已在配置中单独设为 50~55秒

### 7.4 阶段 3: 动作串联

#### 7.4.1 过渡问题

**核心难题**: 冲拳结束时的姿态 ≠ 正踢的标准起手式。

```
冲拳结束:                    正踢起手:
   ▯ (前倾, 右拳伸出)          ▯ (直立, 双手收回)
  /|\                         /|\
  / \                         / \
  完全不同的关节配置!
```

#### 7.4.2 解决方案（现已全面实现风险规避）

基于上述串联时可能发生的**连环翻车（Cascading Failure）**风险，项目中已采用**双重保险**来彻底规避前后动作的依赖：

**1. 部署层 (C++): 动作间平滑插值过渡（已在 `State_MartialArtsSequencer.h` 中实现）**
此前仅简单使用 Transition Hold（直接锁死电机保持僵硬姿态，待下一策略加载后容易产生剧烈的动作跳变），现已全面升级为**关节级平滑线性插值**。
* **机制**: 当段落 A 结束时，保存最后时刻的所有关节目标角（`start_q`）；立即在后台加载段落 B，并步进一次以获取段落 B 的起手第一帧关节坐标（`target_q`）。
* **平滑**: 在 `transition_hold_s_` 的时间窗口内，以高频进行线性插值（`alpha` 平滑），将机器人安全柔和地“拉回”到下一套动作的标准起手式，彻底截断了上一个动作导致的姿态代偿传染到下一个动作。

```cpp
// 核心逻辑已实现在 policy_loop() 与 run() 的结合中：
// 3. Linearly interpolate between start_q and target_q
int hold_steps = static_cast<int>(transition_hold_s_ / current_env_->step_dt);
for (int step = 0; step <= hold_steps && policy_thread_running_; ++step) {
    float alpha = static_cast<float>(step) / hold_steps;
    std::vector<float> interp_q(start_q.size());
    for(size_t i=0; i<start_q.size(); ++i) {
        interp_q[i] = start_q[i] + alpha * (target_q[i] - start_q[i]);
    }
    transition_q_ = interp_q; // 发送给 1000Hz run() 循环
    std::this_thread::sleep_for(std::chrono::milliseconds(int(current_env_->step_dt * 1000)));
}
```

**2. 训练层 (Python): 鲁棒性注入（防止小偏差被放大）**
哪怕有插值，起动时机器人的动量和现实偏差仍客观存在。因此在训练单动作阶段，我们加大初始状态的随机噪声。
这迫使网络学会在初始姿态出现轻微偏差（如插值精度有限导致关节偏差）时，具备极强的强行纠偏能力，而不是随着初始误差产生“雪崩”。

```python
# 修改 tracking_env_cfg.py 中的 _make_command_cfg 加大初始扰动
def _make_command_cfg(npz_filename):
    return mdp.MotionCommandCfg(
        ...,
        pose_range={
            "roll": (-0.3, 0.3),    # 提供偏斜起步抗性
            "pitch": (-0.3, 0.3),   
        },
        joint_position_range=(-0.3, 0.3),  
    )
```

### 7.5 阶段 4: 部署

#### 导出 ONNX

```bash
# play.py 自动导出到 logs/rsl_rl/<experiment>/exported/
python scripts/rsl_rl/play.py --task Unitree-G1-29dof-Mimic-MartialArts-FrontKick

# 收集所有 ONNX 到部署目录
bash scripts/mimic/martial_arts_pipeline.sh deploy
```

#### Mujoco Sim2Sim 验证

在真机前必须做：
1. 将 ONNX 放入 `deploy/` 对应目录
2. 用 Mujoco 跑一遍（接触力学不同于 PhysX）
3. 确认关节速度不超限、动作不抖动

#### 真机安全检查清单

- [ ] 电机电流峰值 < 安全阈值？
- [ ] 关节速度峰值 < velocity_limit_sim？
- [ ] 踢腿时对地反力是否合理？
- [ ] 吊架上连续执行 3 次是否稳定？

#### 编排表演

修改 `deploy/robots/g1_29dof/config/config.yaml`：

```yaml
MartialArtsSequencer:
  transition_hold_s: 1.0
  segments:
    # 自由编排顺序!
    - policy_dir: config/policy/mimic/martial_arts/front_kick/
      motion_file: .../G1_front_kick.csv
      fps: 50
    - policy_dir: config/policy/mimic/martial_arts/lunge_punch/
      motion_file: .../G1_lunge_punch.csv
      fps: 50
    # 可以重复、跳过、调换任何动作
```

操控方式：
| 按键组合 | 动作 |
|---------|------|
| `LT + up` | Passive → FixStand |
| `RB + X` | FixStand → Velocity (行走) |
| `LT(2s) + right` | Velocity → **武术表演** |
| `LT + B` | 任何状态 → Passive (紧急停止) |

---

## 8. C++ 部署端架构解析

### 8.1 状态机 (FSM) 总览

```
                     LT + up
    ┌─────────┐  ──────────→  ┌──────────┐  RB + X   ┌──────────┐
    │ Passive │               │ FixStand │  ───────→  │ Velocity │
    │  (安全) │  ←──────────  │  (站立)  │           │  (行走)  │
    └─────────┘   LT + B     └──────────┘           └────┬─────┘
         ▲                         ▲                      │
         │                         │                      │ LT(2s)+right
         │ LT + B                  │ RB + X               ▼
         │                         │               ┌──────────────┐
         └─────────────────────────┴───────────────│ MartialArts  │
                                                   │ Sequencer    │
                                                   └──────────────┘
                                                   自动播放 → 完成后
                                                   回到 FixStand
```

### 8.2 `State_MartialArtsSequencer` 执行流程

```cpp
enter()
  └→ load_segment(0) → start_policy_thread()

policy_loop() [独立线程]:
  for each segment (0..N-1):
    │
    ├─ load_segment(i)               // 加载 ONNX + motion CSV
    ├─ env->reset()                  // 初始化观测
    ├─ while elapsed < duration:
    │    env->step()                 // 推理 → action
    │    [run() 在主线程写入电机]
    │
    ├─ transition_hold (1.0s)        // 保持当前姿势
    └─ 下一个 segment

  finished_ = true → 自动回 FixStand
```

### 8.3 安全机制

```cpp
// 1. 演完自动退出
registered_checks: finished_ == true → FixStand

// 2. 摔倒保护
registered_checks: bad_orientation(1.0) → Passive
```

---

## 9. 改进路线图与高级优化

### 9.1 短期 (本月)

| 优先级 | 改进项 | 具体做法 |
|--------|--------|----------|
| ~~P0~~ | ~~Front Kick 姿势修复~~ | **(已完成)** 收紧 joint_pos std, 提升 anchor_ori weight |
| ~~P0~~ | ~~放松 action_rate~~ | **(已完成)** weight -0.1 → -0.05 |
| ~~P0~~ | ~~末端极速与精准追踪~~ | **(已完成)** 引入 `motion_ee_pos` 和 `motion_ee_lin_vel` 保证“春晚级”爆发力与精准打击 |
| P1 | 训练 Lunge Punch | 同配置直接训练 |
| P1 | 训练 Side Kick | 可能需放宽 ee_body_pos 到 0.8m |
| ~~P2~~ | ~~实现插值过渡~~ | **(已完成)** 在 C++ Sequencer 中植入了安全平滑切换机制，解决连环翻车风险 |

### 9.2 中期 (1-2月)

| 改进项 | 说明 |
|--------|------|
| 上下半身分离追踪 | 扩展 `motion_joint_pos_error_exp` 支持 `joint_names` 过滤 |
| 末端执行器专项奖励 | 手脚 body_pos 单独更高权重 |
| 重心约束 (CoM/ZMP) | 单腿支撑时质心投影必须在支撑脚上方 |
| 鲁棒过渡训练 | 加大 joint_position_range 和 pose_range |
| 对称性训练 | 随机翻转左右, 右踢学会左踢 |

### 9.3 长期 (高级研究方向)

| 改进项 | 说明 |
|--------|------|
| 视觉打靶 | 加入目标沙袋, 奖励接触力 |
| 对抗训练 | 训练"对手"推扰, 互相博弈 |
| 风格迁移 | 保持关键帧, RL 自由发挥过渡 |
| 电机建模 | 反电动势、发热限流等约束 |

---

## 10. 快速参考卡片

### 10.1 常用命令

```bash
# 安装
./unitree_rl_lab.sh -i

# 列出武术任务
./unitree_rl_lab.sh -l | grep Martial

# 训练
python scripts/rsl_rl/train.py --task Unitree-G1-29dof-Mimic-MartialArts-FrontKick --headless

# 恢复训练
python scripts/rsl_rl/train.py --task Unitree-G1-29dof-Mimic-MartialArts-FrontKick --headless \
    --resume --checkpoint logs/rsl_rl/.../model_29999.pt

# 播放
python scripts/rsl_rl/play.py --task Unitree-G1-29dof-Mimic-MartialArts-FrontKick

# 录制视频
python scripts/rsl_rl/play.py --task Unitree-G1-29dof-Mimic-MartialArts-FrontKick --video

# TensorBoard
tensorboard --logdir logs/rsl_rl/unitree_g1_29dof_mimic_martialarts_frontkick

# 一键全流程
bash scripts/mimic/martial_arts_pipeline.sh all
```

### 10.2 关键文件速查

| 要修改什么 | 去哪个文件 |
|-----------|-----------|
| 奖励权重/std | `martial_arts/tracking_env_cfg.py` → `MartialArtsRewardsCfg` |
| 终止条件阈值 | `martial_arts/tracking_env_cfg.py` → `MartialArtsTerminationsCfg` |
| 域随机化范围 | `martial_arts/tracking_env_cfg.py` → `EventCfg` |
| 初始状态扰动 | `tracking_env_cfg.py` → `_make_command_cfg()` → `pose_range` |
| 电机 PD 增益 | `gangnanm_style/g1.py` → `G1_CYLINDER_CFG` |
| PPO 超参数 | `mimic/agents/rsl_rl_ppo_cfg.py` → `BasePPORunnerCfg` |
| 奖励函数实现 | `mimic/mdp/rewards.py` |
| 动捕加载逻辑 | `mimic/mdp/commands.py` → `MotionLoader` + `MotionCommand` |
| C++ 串联器 | `deploy/include/FSM/State_MartialArtsSequencer.h` |
| 部署编排配置 | `deploy/robots/g1_29dof/config/config.yaml` → `MartialArtsSequencer` |

### 10.3 TensorBoard 关键指标

| 指标名 | 理想趋势 | 异常信号 |
|--------|---------|---------|
| `reward/total` | 持续上升 → 收敛 | 下降 = 奖励冲突 |
| `error_joint_pos` | 持续下降 | 不降 = joint_pos 权重太低 |
| `error_body_pos` | 持续下降 | 不降 = body_pos std 太小(梯度消失) |
| `error_anchor_rot` | 下降到 <0.3 | >0.5 = anchor_ori 权重太低 |
| `sampling_entropy` | 先降后稳 | 趋近0 = 过拟合某段 |
| `sampling_top1_prob` | <0.3 | >0.5 = 反复在同一段失败 |
| `episode_length` | 接近 `episode_length_s` | 很短 = 终止条件太严 |

---

> **写在最后**: 武bot 项目的精髓不在于某一个奖励函数的权重，而在于**分而治之 + 工程化串联**的架构思想。7 个独立策略各自精雕细琢，由 C++ 状态机在毫秒级精度下无缝切换——这不是单一的 RL 问题，而是一个系统工程问题。祝你的机器人打出一套漂亮的拳！🥋
