# 武术机器人项目架构与算法原理

> 目标：让 Unitree G1 29DOF 机器人复现 2026 年春节联欢晚会武术机器人表演。
>
> 核心思路：CMU 动作捕捉数据 → 独立 RL 策略训练（7 个动作）→ C++ Policy Sequencer 串联部署。

---

## 目录

1. [整体架构概览](#1-整体架构概览)
2. [数据流水线](#2-数据流水线)
3. [训练环境核心：MotionCommand](#3-训练环境核心motioncommand)
4. [强化学习算法：PPO + 运动跟踪](#4-强化学习算法ppo--运动跟踪)
5. [奖励函数设计](#5-奖励函数设计)
6. [终止条件设计](#6-终止条件设计)
7. [自适应采样机制](#7-自适应采样机制)
8. [观测空间设计](#8-观测空间设计)
9. [动作空间与执行器](#9-动作空间与执行器)
10. [Policy Sequencer 部署架构](#10-policy-sequencer-部署架构)
11. [代码文件索引](#11-代码文件索引)
12. [完整训练工作流](#12-完整训练工作流)
13. [关键超参数速查](#13-关键超参数速查)

---

## 1. 整体架构概览

```
┌─────────────────────────────────────────────────────────────────┐
│                        项目总体数据流                             │
│                                                                 │
│  CMU MoCap                Isaac Lab 训练              C++ 部署   │
│  ASF + AMC  ──►  CSV  ──►  NPZ  ──►  PPO训练  ──►  ONNX Policy │
│  (Subject                                                       │
│   #135)                                              ▼          │
│                                               Policy Sequencer  │
│                                               (7 策略串联)        │
│                                                    ▼            │
│                                               G1 实体机器人       │
└─────────────────────────────────────────────────────────────────┘
```

### 7 个独立武术动作（每个独立训练一个策略）

| 动作名 | Task ID | 数据来源 | 时长 |
|--------|---------|---------|------|
| 平安初段 (Heian Shodan) | `...-HeianShodan` | CMU 135_04 | ~11s |
| 前踢 (Front Kick)       | `...-FrontKick`   | CMU 135_03 | ~23s |
| 回旋踢 (Roundhouse Kick)| `...-RoundhouseKick`| CMU 135_05 | ~20s |
| 冲拳 (Lunge Punch)      | `...-LungePunch`  | CMU 135_06 | ~26s |
| 侧踢 (Side Kick)        | `...-SideKick`    | CMU 135_07 | ~12s |
| 拔塞 (Bassai)           | `...-Bassai`      | CMU 135_01 | ~51s |
| 燕飞 (Empi)             | `...-Empi`        | CMU 135_02 | ~43s |

> 完整 Task ID 前缀：`Unitree-G1-29dof-Mimic-MartialArts-`

---

## 2. 数据流水线

### 阶段 1：CMU ASF+AMC → CSV

**脚本**：`scripts/mimic/cmu_amc_to_csv.py`

CMU MoCap 格式：
- `.asf`：骨骼定义（骨骼名称、长度、连接关系、关节自由度）
- `.amc`：逐帧运动数据（各关节角度）

转换工作：
1. 解析 ASF 骨骼树，建立 bone hierarchy
2. 逐帧读取 AMC 关节角，通过正向运动学（FK）计算每根骨骼的全局位置/朝向
3. 将人体骨骼关节映射到 G1 29DOF 关节（手动定义的关节对应关系）
4. 输出 CSV，每行格式：
   ```
   base_pos_x, base_pos_y, base_pos_z,
   base_quat_x, base_quat_y, base_quat_z, base_quat_w,
   joint_1, ..., joint_29
   ```

CMU 坐标系 → Isaac Lab 坐标系的转换也在这里完成。

### 阶段 2：CSV → NPZ（Isaac Lab 仿真重放）

**脚本**：`scripts/mimic/csv_to_npz.py`

这一步**不是简单的格式转换**，而是在 Isaac Lab 中实际运行物理仿真：

```
CSV 关节角序列
      │
      ▼
  Isaac Lab 物理仿真
  (SimulationContext + G1 机器人)
      │  每帧写入关节位置，前向模拟
      │  收集所有刚体的真实物理状态
      ▼
NPZ 文件（keys：fps, joint_pos, joint_vel,
                body_pos_w, body_quat_w,
                body_lin_vel_w, body_ang_vel_w）
```

为什么需要物理仿真？
- CSV 只有关节角，缺少**关节速度**、**刚体速度**
- 仿真能产生物理上自洽的速度数据（通过有限差分 + 物理引擎）
- G1 有 30 个刚体（`body_pos_w` shape: `[N, 30, 3]`），仿真同时记录所有刚体状态

关键参数：
- `output_fps=50`（训练控制频率）
- 输入支持任意 fps，内部用线性插值（位置）+ SLERP（四元数）重采样

### 阶段 3：NPZ → 训练

NPZ 文件存放位置（直接在任务包内，`MOTION_DATA_DIR = os.path.dirname(__file__)`）：
```
source/unitree_rl_lab/.../martial_arts/
    G1_front_kick.npz
    G1_heian_shodan.npz
    ...（共 7 个）
```

---

## 3. 训练环境核心：MotionCommand

**文件**：`source/unitree_rl_lab/.../mdp/commands.py`

`MotionCommand` 是整个运动跟踪系统的核心，继承自 Isaac Lab 的 `CommandTerm`。

### 职责

```
NPZ 文件
   │  MotionLoader（加载到 GPU Tensor）
   ▼
MotionCommand
   ├── 时间步推进 (_update_command，每个控制步 +1)
   ├── 提供参考状态（当前帧的 body_pos/quat/vel）
   ├── 相对坐标变换（body_pos_relative_w）
   ├── Episode 重置时的自适应时间点采样
   └── Debug 可视化（Isaac Sim 中显示绿色/红色目标标记）
```

### 相对坐标变换

这是最关键的设计——训练时参考动作会跟随机器人的位置移动（仅跟 Yaw 方向，不跟 Roll/Pitch）：

```python
# delta_pos_w：参考动作平移到当前机器人位置（Z 轴用参考值）
delta_pos_w[..., 2] = anchor_pos_w_repeat[..., 2]  # 保留参考Z高度

# delta_ori_w：只取 Yaw 分量的旋转差
delta_ori_w = yaw_quat(quat_mul(robot_anchor_quat_w, quat_inv(anchor_quat_w)))

# 变换后的参考 body 位置/朝向
body_pos_relative_w = delta_pos_w + quat_apply(delta_ori_w, body_pos_w - anchor_pos_w)
body_quat_relative_w = quat_mul(delta_ori_w, body_quat_w)
```

效果：机器人可以在任意 XY 位置、任意 Yaw 朝向完成动作，参考轨迹会自动跟随对齐。这是让 4096 个并行环境都能独立训练的关键。

### anchor body

Anchor body = `torso_link`（躯干）。所有其他身体部位的位置都是**相对于 torso_link** 计算的，这样策略学到的是"身体各部位相对躯干的姿态"，与绝对世界坐标无关。

---

## 4. 强化学习算法：PPO + 运动跟踪

### 框架：RSL-RL（OnPolicyRunner）

**文件**：`source/unitree_rl_lab/.../agents/rsl_rl_ppo_cfg.py`

```python
BasePPORunnerCfg:
    num_steps_per_env = 24        # 每次收集 24 步（0.48s @ 50Hz）
    max_iterations = 30000        # 总共 30000 次更新
    
    policy（Actor-Critic）：
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = "elu"
    
    algorithm（PPO）：
        clip_param = 0.2          # PPO clip
        num_learning_epochs = 5   # 每批数据学习 5 遍
        num_mini_batches = 4      # 分 4 个 mini-batch
        learning_rate = 1e-3      # 自适应学习率
        gamma = 0.99              # 折扣因子
        lam = 0.95                # GAE λ
        desired_kl = 0.01         # 目标 KL 散度（用于自适应 lr）
        entropy_coef = 0.005      # 熵正则（鼓励探索）
```

### 并行规模

```
4096 个并行环境 × 24 步/轮 = 98,304 样本/次更新
共 30,000 次更新 = ~29 亿 environment steps
```

### Actor-Critic 架构

```
观测（Actor Policy obs）
  ├── motion_command:         58 维（当前帧 joint_pos + joint_vel）
  ├── motion_anchor_ori_b:     6 维（目标 torso 朝向，相对机器人坐标系）
  ├── base_ang_vel:            3 维（机器人角速度）+ noise
  ├── joint_pos_rel:          29 维（当前关节角）+ noise
  ├── joint_vel_rel:          29 维（当前关节速度）+ noise
  └── last_action:            29 维（上一步动作）
  = 154 维 → Actor MLP → 29 维关节位置目标

观测（Critic Privileged obs）
  ├── command:                58 维
  ├── motion_anchor_pos_b:     3 维（无噪声）
  ├── motion_anchor_ori_b:     6 维（无噪声）
  ├── body_pos:             14×3=42 维（14个tracked body的位置）
  ├── body_ori:             14×6=84 维（14个tracked body的朝向矩阵前两列）
  ├── base_lin_vel:            3 维
  ├── base_ang_vel:            3 维
  ├── joint_pos:              29 维
  ├── joint_vel:              29 维
  └── actions:                29 维
  = ~289 维 → Critic MLP → 值函数 V(s)
```

> Actor 只用带噪声的有限观测（模拟真实传感器），Critic 用完整特权信息加速训练（部署时不需要 Critic）。

---

## 5. 奖励函数设计

**文件**：`source/unitree_rl_lab/.../mdp/rewards.py`
**配置**：`tracking_env_cfg.py` → `MartialArtsRewardsCfg`

所有跟踪奖励使用**指数形式**（高斯核）：

$$r = e^{-\frac{\text{error}^2}{\sigma^2}}$$

误差为 0 时奖励 = 1.0，误差越大奖励越接近 0。`std` 控制奖励对误差的敏感程度（越小越尖锐）。

### 当前奖励配置（v2，调优后）

| 奖励项 | weight | std | 作用 |
|--------|--------|-----|------|
| `motion_global_anchor_pos` | +0.5 | 0.3m | torso 全局位置跟踪 |
| `motion_global_anchor_ori` | +0.5 | 0.4rad | torso 全局朝向跟踪 |
| `motion_body_pos` | **+1.5** | 0.3m | 14 个身体部位相对位置（核心驱动力） |
| `motion_body_ori` | +1.0 | 0.4rad | 14 个身体部位朝向 |
| `motion_body_lin_vel` | +1.0 | 1.0m/s | 身体部位线速度 |
| `motion_body_ang_vel` | +1.0 | 3.14rad/s | 身体部位角速度 |
| `joint_acc` | −2.5e-7 | — | 关节加速度惩罚（平滑性） |
| `joint_torque` | −1e-5 | — | 关节扭矩惩罚（能效） |
| `action_rate_l2` | −0.1 | — | 动作变化率惩罚（防抖动） |
| `joint_limit` | −10.0 | — | 关节极限硬惩罚 |
| `undesired_contacts` | −0.1 | — | 非预期碰撞（手脚除外） |

### 奖励设计关键原则

**① anchor 权重不能超过 body_pos 权重**
- 如果 anchor（root 跟踪）权重过高，策略会优先学"站稳不倒"而忽视四肢姿态
- 正确关系：`body_pos weight (1.5) > anchor weight (0.5)`

**② std 不能太小**
- std=0.2m 意味着误差 0.2m 时奖励仅剩 ~37%，梯度极其尖锐
- 训练初期误差很大，太小的 std 导致奖励几乎为 0，策略无法学习
- std=0.3m 是合理的起点

**③ 正向奖励总和 >> 负向惩罚**
- 正向奖励总和上限：0.5+0.5+1.5+1.0+1.0+1.0 = **5.5**
- 负向惩罚量级远小于正向奖励，策略的主要优化方向是"做对动作"而非"避免惩罚"

---

## 6. 终止条件设计

**文件**：`source/unitree_rl_lab/.../mdp/terminations.py`
**配置**：`tracking_env_cfg.py` → `MartialArtsTerminationsCfg`

| 终止条件 | 实现 | 阈值 | 说明 |
|----------|------|------|------|
| `time_out` | `time_out` | 30s | 时间到正常结束 |
| `anchor_pos` | `bad_anchor_pos_z_only` | **0.25m** | torso Z 轴高度偏差过大（不检查 XY，允许横向移动） |
| `anchor_ori` | `bad_anchor_ori` | **0.8** | torso 朝向与参考差异（比较重力向量的 Z 分量差） |
| `ee_body_pos` | `bad_motion_body_pos_z_only` | **0.25m** | 手腕/脚踝 Z 高度偏差过大 |

### 为什么只检查 Z 轴

踢腿、冲拳时机器人的 XY 位置可能与参考有合理偏差（地面摩擦、惯性），但高度（Z）偏差意味着要倒下，应立即终止。

### 终止阈值的设计哲学

- **越严格**（阈值越小）：坏 episode 更早结束 → 训练数据更"纯净" → 策略学不到用错误姿态勉强存活的技巧
- **越宽松**（阈值越大）：episode 存活更长 → 但可能让策略学会"歪着姿势也能活下去"
- v1 用了 0.3m（太宽松），v2 改回 0.25m（与 gangnam_style 一致）

---

## 7. 自适应采样机制

**文件**：`commands.py` → `_adaptive_sampling()`

这是运动跟踪 RL 的关键技术，解决"难动作段学不好"的问题。

### 问题

训练时 episode 从随机时间点开始（Reference State Initialization, RSI）。如果从头均匀采样，容易反复练简单段（如站立），难做的段（如高踢腿）采样不足。

### 解决方案：基于失败率的自适应采样

```
把整段动作按时间等分为 N 个 bin（约 1s/bin）

失败统计：
  每次 episode 因 terminated（非 timeout）结束，记录它是在哪个 bin 失败的
  → bin_failed_count：指数移动平均（alpha=0.002）

采样概率：
  P(bin_i) ∝ bin_failed_count[i] + uniform_ratio/N
               ↑失败越多的段被采样越频繁    ↑保留基础均匀探索

核平滑：
  对 bin_failed_count 做一维卷积（kernel_size=3, lambda=0.8）
  → 相邻 bin 互相影响，避免孤立 spike
```

指标监控：
- `sampling_entropy`：采样分布的熵（越低 = 越集中在难段）
- `sampling_top1_bin`：当前最难的时间段位置（归一化到 [0,1]）
- `sampling_top1_prob`：最难时间段的采样概率

---

## 8. 观测空间设计

### Policy 观测（Actor 用，部署时输入）

```python
PolicyCfg:
    motion_command        # 当前帧参考关节位置 + 速度（29+29=58维）
                          # 告诉策略"现在应该做什么动作"
    motion_anchor_ori_b   # 目标 torso 朝向（相对机器人自身坐标系，6维）
                          # + 均匀噪声 ±0.05
    base_ang_vel          # 机器人自身角速度（3维）+ 噪声 ±0.2
    joint_pos_rel         # 当前关节角（29维）+ 噪声 ±0.01 rad
    joint_vel_rel         # 当前关节速度（29维）+ 噪声 ±0.5 rad/s
    last_action           # 上一步输出动作（29维，无噪声）
```

**为什么加噪声**：仿真到实体（Sim-to-Real）迁移，真实传感器有噪声，训练时加噪声使策略对传感器误差有鲁棒性。

### Critic 观测（训练时专用，部署时丢弃）

Critic 还额外得到 `body_pos`、`body_ori`、`base_lin_vel`、`motion_anchor_pos_b` 等特权信息，用于准确估计 V(s)，但**不参与 Actor 决策**。

---

## 9. 动作空间与执行器

### 动作：关节位置目标（29 DOF）

策略输出的是**相对于 default pose 的关节位置偏移量**，经过 `scale` 系数缩放后加上 default offset：

```python
# 各关节的 action_scale（越大越"有力"）
legs：         scale ≈ 0.35-0.55 rad
ankles/waist： scale ≈ 0.44 rad
arms：         scale ≈ 0.44 rad
wrist:         scale ≈ 0.07 rad（手腕减小，避免过激）
```

缩放系数由 `G1_ACTION_SCALE` 根据各关节的 `行程/2π` 比例计算。

### 执行器：隐式 PD 控制（ImplicitActuator）

Isaac Lab 的隐式执行器直接在物理引擎层面实现 PD 控制：

$$\tau = k_p (q_{target} - q) + k_d (\dot{q}_{target} - \dot{q})$$

G1 各关节刚度（`k_p`）由电机型号决定：
- 腿部（7520-22电机）：k_p ≈ 99.1，k_d ≈ 6.3
- 腿部（7520-14电机）：k_p ≈ 40.2，k_d ≈ 2.6
- 手臂（5020电机）：k_p ≈ 14.3，k_d ≈ 0.9
- 手腕（4010电机）：k_p ≈ 16.8，k_d ≈ 1.1

设计公式：`k_p = J × ωn²`，`k_d = 2 × ζ × J × ωn`，其中 `ωn = 10Hz × 2π`，`ζ = 2.0`（过阻尼，稳定性优先）。

### 领域随机化（Domain Randomization）

训练时每个 episode 开始时随机化：
- 地面摩擦系数：static [0.3, 1.6]，dynamic [0.3, 1.2]
- 躯干质心位置：XYZ ±0.025-0.05m
- 关节初始角度：±0.01 rad（在 NPZ 参考值基础上抖动）
- 每 2-5s 施加随机速度扰动（`push_robot`）

这些随机化使策略对真实机器人的个体差异和外部干扰有鲁棒性。

---

## 10. Policy Sequencer 部署架构

**文件**：`deploy/include/FSM/State_MartialArtsSequencer.h`
**配置**：`deploy/robots/g1_29dof/config/config.yaml`

### 整体 FSM 结构

```
G1 机器人 FSM：
    State_Passive          ← 上电默认，关节软化
    State_FixStand         ← 站立归位
    State_MartialArtsSequencer  ← 武术表演（L2 长按 + 右摇杆触发）
```

### Policy Sequencer 工作流程

```cpp
enter():
    current_segment = 0
    load_segment(0)      // 加载 ONNX policy + 运动文件
    start_policy_thread()

policy_loop():
    while not finished:
        current_env.reset()       // 初始化到运动数据第一帧
        
        while elapsed < segment_duration:
            current_env.step()    // 推理一步 ONNX 策略
            write joints to lowcmd
            sleep(step_dt = 0.02s)
        
        current_segment++
        
        // 过渡保持（默认 1s）
        hold current pose for transition_hold_s
        
        load_segment(current_segment)  // 加载下一个策略

run():
    // 主线程每帧读取 policy_thread 写好的 joint targets
    send lowcmd to robot
```

### 每个 Segment 的内容

```yaml
MartialArtsSequencer:
  transition_hold_s: 1.0
  segments:
    - policy_dir: "logs/rsl_rl/unitree_g1_29dof_mimic_martialarts_heianshodan/.../exported"
      motion_file: "G1_heian_shodan.npz"
      fps: 50
    - policy_dir: "logs/rsl_rl/unitree_g1_29dof_mimic_martialarts_frontkick/.../exported"
      motion_file: "G1_front_kick.npz"
      fps: 50
    # ... 共 7 个
```

### ONNX 导出

`play.py` 在推理时自动导出 ONNX：

```python
export_policy_as_onnx(
    actor_critic,
    normalizer,   # 观测归一化器（均值/方差）
    path=os.path.join(log_dir, "exported/policy.onnx")
)
```

导出内容包含：Actor 网络权重 + 观测归一化参数，一个 `.onnx` 文件完整封装推理所需的一切。

---

## 11. 代码文件索引

```
unitree_rl_lab/
│
├── scripts/
│   ├── mimic/
│   │   ├── cmu_amc_to_csv.py      # 阶段1: CMU ASF+AMC → G1 CSV
│   │   ├── csv_to_npz.py          # 阶段2: CSV → NPZ（Isaac Lab 物理仿真）
│   │   ├── replay_npz.py          # 验证工具: 在 Isaac Sim 中回放 NPZ
│   │   ├── validate_npz.py        # 验证工具: 检查 NPZ 完整性
│   │   └── martial_arts_pipeline.sh  # 一键流水线脚本
│   │
│   └── rsl_rl/
│       ├── train.py               # 训练入口
│       └── play.py                # 推理/导出 ONNX 入口
│
├── source/unitree_rl_lab/unitree_rl_lab/
│   ├── tasks/mimic/
│   │   ├── mdp/
│   │   │   ├── commands.py        # ★ MotionCommand（参考动作管理、自适应采样）
│   │   │   ├── rewards.py         # ★ 奖励函数实现（指数误差奖励）
│   │   │   ├── terminations.py    # 终止条件实现
│   │   │   └── observations.py    # 观测函数（body_pos_b, anchor_ori_b 等）
│   │   │
│   │   ├── agents/
│   │   │   └── rsl_rl_ppo_cfg.py  # ★ PPO 超参数配置
│   │   │
│   │   └── robots/g1_29dof/
│   │       ├── gangnanm_style/
│   │       │   ├── g1.py          # ★ G1 机器人物理参数（刚度、阻尼、执行器）
│   │       │   └── tracking_env_cfg.py  # 江南 Style 任务配置（参考基准）
│   │       │
│   │       └── martial_arts/
│   │           ├── __init__.py    # ★ 7 个任务的 gym.register
│   │           ├── tracking_env_cfg.py  # ★ 武术任务配置（奖励/终止/环境）
│   │           └── G1_*.npz       # 7 个运动数据文件（训练时读取）
│   │
│   └── assets/robots/unitree/     # G1 USD 模型文件路径
│
└── deploy/
    ├── include/
    │   └── FSM/
    │       ├── State_MartialArtsSequencer.h  # ★ C++ Policy Sequencer
    │       └── State_RLBase.h                # 基础 RL 部署状态
    │
    └── robots/g1_29dof/
        ├── config/config.yaml     # 部署配置（Sequencer 段列表）
        └── main.cpp               # 机器人程序入口
```

---

## 12. 完整训练工作流

### 前置准备

```bash
# 激活环境（每次新终端都需要）
conda activate env_isaaclab
cd /home/jiadong/unitree_rl_lab
```

### Step 1：验证 NPZ 数据

```bash
python scripts/mimic/validate_npz.py
```

### Step 2：训练单个动作

```bash
# 以 FrontKick 为例，~12小时
python scripts/rsl_rl/train.py \
    --task Unitree-G1-29dof-Mimic-MartialArts-FrontKick \
    --headless
```

训练日志保存在 `logs/rsl_rl/unitree_g1_29dof_mimic_martialarts_frontkick/<日期时间>/`

训练指标监控（关注以下数值收敛）：
| 指标 | 初期(~500iter) | 收敛(~30000iter) |
|------|--------------|-----------------|
| `mean_reward` | ~0.5-1.0 | ~3.0-4.5 |
| `error_body_pos` | ~0.3m | ~0.05-0.1m |
| `error_anchor_pos` | ~0.15m | ~0.02-0.05m |
| `episode_length` | ~3-8s | ~15-25s |

### Step 3：验证（推理 + 导出 ONNX）

```bash
# 可视化推理（需要显示器）
python scripts/rsl_rl/play.py \
    --task Unitree-G1-29dof-Mimic-MartialArts-FrontKick

# 录制视频
python scripts/rsl_rl/play.py \
    --task Unitree-G1-29dof-Mimic-MartialArts-FrontKick \
    --video --video_length 1145
```

执行 `play.py` 后，ONNX 自动导出到：
```
logs/rsl_rl/unitree_g1_29dof_mimic_martialarts_frontkick/<日期>/exported/policy.onnx
```

### Step 4：训练其余 6 个动作（按顺序）

```bash
# 按时长从短到长训练（短的先验证）
python scripts/rsl_rl/train.py --task Unitree-G1-29dof-Mimic-MartialArts-HeianShodan --headless
python scripts/rsl_rl/train.py --task Unitree-G1-29dof-Mimic-MartialArts-SideKick --headless
python scripts/rsl_rl/train.py --task Unitree-G1-29dof-Mimic-MartialArts-RoundhouseKick --headless
python scripts/rsl_rl/train.py --task Unitree-G1-29dof-Mimic-MartialArts-LungePunch --headless
python scripts/rsl_rl/train.py --task Unitree-G1-29dof-Mimic-MartialArts-Empi --headless
python scripts/rsl_rl/train.py --task Unitree-G1-29dof-Mimic-MartialArts-Bassai --headless
```

> **注意**：RTX 4070 Ti (12GB) 不能同时训练和推理，需逐个完成。

### Step 5：部署配置

更新 `deploy/robots/g1_29dof/config/config.yaml` 中的 `policy_dir` 路径指向实际训练结果目录，然后 CMake 编译 `deploy/robots/g1_29dof/`。

---

## 13. 关键超参数速查

### 仿真参数
| 参数 | 值 | 说明 |
|------|-----|------|
| `sim.dt` | 0.005s | 物理仿真步长（200Hz） |
| `decimation` | 4 | 控制频率 = 200/4 = **50Hz** |
| `episode_length_s` | 30s（一般）/ 55s（Bassai） | Episode 最大时长 |
| `num_envs` | 4096 | 并行环境数 |

### 训练参数
| 参数 | 值 | 说明 |
|------|-----|------|
| `num_steps_per_env` | 24 | 每次 rollout 长度（0.48s） |
| `max_iterations` | 30,000 | 总训练轮数（~12h） |
| `learning_rate` | 1e-3（自适应） | 当 KL > 0.01 时降低 lr |
| `save_interval` | 500 | 每 500 iter 保存 checkpoint |

### 奖励参数（v2）
| 参数 | 值 |
|------|----|
| body_pos weight | **1.5** |
| body_pos std | **0.3m** |
| anchor weight | **0.5** |
| anchor std (pos) | **0.3m** |
| action_rate penalty | **-0.1** |

### 终止阈值（v2）
| 参数 | 值 |
|------|----|
| anchor Z 高度偏差 | **0.25m** |
| anchor 朝向偏差 | **0.8**（重力Z差） |
| EE body Z 偏差 | **0.25m** |
