# 武术机器人（WuBot）完全精通指南

> **从零基础到完全精通：Unitree G1 29DOF 武术动作训练与部署**
>
> 本文档全面介绍 WuBot 项目的数据流水线、算法架构、奖励设计、参数调优及部署方案，
> 覆盖从 CMU 动作捕捉数据到 2026 春节联欢晚会武术机器人表演的完整技术链路。

---

## 目录

- [第一部分：项目总览](#第一部分项目总览)
  - [1.1 项目目标](#11-项目目标)
  - [1.2 核心思想：分而治之](#12-核心思想分而治之)
  - [1.3 总体架构图](#13-总体架构图)
  - [1.4 七个武术动作一览](#14-七个武术动作一览)
- [第二部分：数据流水线](#第二部分数据流水线)
  - [2.1 数据来源：CMU Motion Capture](#21-数据来源cmu-motion-capture)
  - [2.2 阶段一：CMU ASF+AMC → CSV](#22-阶段一cmu-asfamc--csv)
  - [2.3 阶段二：CSV → NPZ（物理仿真重放）](#23-阶段二csv--npz物理仿真重放)
  - [2.4 数据质量关键：关节限位钳位](#24-数据质量关键关节限位钳位)
  - [2.5 NPZ 文件内容详解](#25-npz-文件内容详解)
- [第三部分：机器人物理模型](#第三部分机器人物理模型)
  - [3.1 G1 29DOF 关节定义](#31-g1-29dof-关节定义)
  - [3.2 执行器（Actuator）配置](#32-执行器actuator配置)
  - [3.3 动作缩放（Action Scale）](#33-动作缩放action-scale)
  - [3.4 USD vs URDF 关节限位差异](#34-usd-vs-urdf-关节限位差异)
  - [3.5 碰撞体几何分析](#35-碰撞体几何分析)
- [第四部分：训练环境架构](#第四部分训练环境架构)
  - [4.1 ManagerBasedRLEnv 总体框架](#41-managerbasedrlenv-总体框架)
  - [4.2 MotionCommand：运动跟踪核心](#42-motioncommand运动跟踪核心)
  - [4.3 相对坐标变换：yaw-only 对齐](#43-相对坐标变换yaw-only-对齐)
  - [4.4 观测空间设计](#44-观测空间设计)
  - [4.5 动作空间设计](#45-动作空间设计)
  - [4.6 Domain Randomization（域随机化）](#46-domain-randomization域随机化)
- [第五部分：强化学习算法](#第五部分强化学习算法)
  - [5.1 PPO（Proximal Policy Optimization）](#51-ppoproximal-policy-optimization)
  - [5.2 Actor-Critic 网络架构](#52-actor-critic-网络架构)
  - [5.3 Asymmetric Actor-Critic](#53-asymmetric-actor-critic)
  - [5.4 PPO 超参数详解](#54-ppo-超参数详解)
- [第六部分：奖励函数设计（核心）](#第六部分奖励函数设计核心)
  - [6.1 奖励设计哲学](#61-奖励设计哲学)
  - [6.2 指数核函数（Exp Kernel）](#62-指数核函数exp-kernel)
  - [6.3 锚点（Anchor）跟踪奖励](#63-锚点anchor跟踪奖励)
  - [6.4 全身体追踪奖励](#64-全身体追踪奖励)
  - [6.5 关节位置跟踪奖励（关键！）](#65-关节位置跟踪奖励关键)
  - [6.6 末端执行器跟踪奖励](#66-末端执行器跟踪奖励)
  - [6.7 速度跟踪奖励](#67-速度跟踪奖励)
  - [6.8 正则化惩罚](#68-正则化惩罚)
  - [6.9 接触惩罚](#69-接触惩罚)
  - [6.10 奖励权重总表（v9）](#610-奖励权重总表v9)
- [第七部分：终止条件设计](#第七部分终止条件设计)
  - [7.1 终止条件的作用](#71-终止条件的作用)
  - [7.2 锚点位置/朝向终止](#72-锚点位置朝向终止)
  - [7.3 末端执行器位置终止](#73-末端执行器位置终止)
  - [7.4 非法接触终止（v9 新增）](#74-非法接触终止v9-新增)
  - [7.5 终止条件总表（v9）](#75-终止条件总表v9)
- [第八部分：自适应采样机制](#第八部分自适应采样机制)
  - [8.1 为什么需要自适应采样](#81-为什么需要自适应采样)
  - [8.2 算法原理](#82-算法原理)
  - [8.3 参数调优指南](#83-参数调优指南)
- [第九部分：部署架构](#第九部分部署架构)
  - [9.1 Policy Sequencer 原理](#91-policy-sequencer-原理)
  - [9.2 C++ 实现架构](#92-c-实现架构)
  - [9.3 过渡插值（Transition Hold）](#93-过渡插值transition-hold)
  - [9.4 部署配置文件](#94-部署配置文件)
  - [9.5 Sim2Sim → Sim2Real 工作流](#95-sim2sim--sim2real-工作流)
- [第十部分：完整训练工作流](#第十部分完整训练工作流)
  - [10.1 环境准备](#101-环境准备)
  - [10.2 数据制备](#102-数据制备)
  - [10.3 训练命令](#103-训练命令)
  - [10.4 监控训练（TensorBoard）](#104-监控训练tensorboard)
  - [10.5 推理验证（Play）](#105-推理验证play)
  - [10.6 导出与部署](#106-导出与部署)
- [第十一部分：参数调优方法论](#第十一部分参数调优方法论)
  - [11.1 奖励权重调优原则](#111-奖励权重调优原则)
  - [11.2 终止阈值调优原则](#112-终止阈值调优原则)
  - [11.3 常见问题诊断与解决](#113-常见问题诊断与解决)
  - [11.4 版本演进经验总结（v6→v9）](#114-版本演进经验总结v6v9)
- [第十二部分：代码文件索引](#第十二部分代码文件索引)
- [附录](#附录)
  - [A. SIM 关节顺序](#a-sim-关节顺序)
  - [B. 碰撞体列表](#b-碰撞体列表)
  - [C. 关键公式汇总](#c-关键公式汇总)

---

# 第一部分：项目总览

## 1.1 项目目标

让 **Unitree G1 29DOF 人形机器人** 学会执行7个空手道/武术动作，然后在真实硬件上将这些动作串联成一场完整的武术表演——类似于2026年春节联欢晚会上的武术机器人表演。

核心技术路径：
```
人类动作捕捉 → 物理仿真 → 强化学习策略训练 → 实体机器人部署
```

## 1.2 核心思想：分而治之

**为什么不训练一个万能策略？**

7个武术动作差异巨大（套路长达51秒 vs 单踢12秒），如果用一个策略同时学所有动作：
- 观测/动作空间复杂度暴增
- 不同动作的奖励信号相互干扰
- 训练不稳定，难以收敛

**我们的方案：**
1. **训练阶段**：每个动作独立训练一个 PPO 策略 → 7个 ONNX 模型
2. **部署阶段**：用 C++ Policy Sequencer 按顺序串联执行，段间平滑过渡

这是经典的 **"分解-征服-组合"** 策略，每个子问题简单且可独立调优。

## 1.3 总体架构图

```
┌───────────────────────────────────────────────────────────────────┐
│                    WuBot 项目总体数据流                             │
│                                                                   │
│  CMU MoCap             Isaac Lab 训练              C++ 部署       │
│  ASF + AMC  ──► CSV ──► NPZ ──► PPO训练 ──► ONNX Policy         │
│  (Subject #135)                    │                    │         │
│                                    │              ┌─────┴─────┐   │
│  cmu_amc_to_csv.py  csv_to_npz.py │              │  Policy   │   │
│                                    │              │ Sequencer │   │
│                              ┌─────┴─────┐        │ (7策略    │   │
│                              │ 7个独立   │        │  串联)    │   │
│                              │ 训练任务  │        └─────┬─────┘   │
│                              └───────────┘              │         │
│                                                   G1 实体机器人    │
└───────────────────────────────────────────────────────────────────┘
```

## 1.4 七个武术动作一览

| # | 动作名 | 日文名 | Task ID 后缀 | CMU 编号 | 时长 | 特点 |
|---|--------|--------|-------------|----------|------|------|
| 1 | 平安初段 | Heian Shodan | `HeianShodan` | 135_04 | ~11s | 空手道基础型，步法+手技 |
| 2 | 前踢 | Mae Geri | `FrontKick` | 135_03 | ~23s | 前方直踢，单腿平衡 |
| 3 | 回旋踢 | Mawashi Geri | `RoundhouseKick` | 135_05 | ~20s | 旋转踢击，全身协调 |
| 4 | 冲拳 | Oi-Tsuki | `LungePunch` | 135_06 | ~26s | 弓步冲拳，重心前移 |
| 5 | 侧踢 | Yoko Geri | `SideKick` | 135_07 | ~12s | 侧方踢击，核心稳定 |
| 6 | 拔塞 | Bassai | `Bassai` | 135_01 | ~51s | 高级套路，多段组合 |
| 7 | 燕飞 | Empi | `Empi` | 135_02 | ~43s | 高级套路，含跳跃 |

> **完整 Task ID 格式**：`Unitree-G1-29dof-Mimic-MartialArts-{后缀}`
>
> 例如：`Unitree-G1-29dof-Mimic-MartialArts-FrontKick`

---

# 第二部分：数据流水线

## 2.1 数据来源：CMU Motion Capture

数据来自 [CMU Motion Capture Database](http://mocap.cs.cmu.edu/) 的 **Subject #135**，
该被试者录制了一系列空手道（Karate）动作。

CMU MoCap 格式：
- **`.asf`** (Acclaim Skeleton File)：骨骼定义文件
  - 骨骼名称、长度、连接关系
  - 每个关节的自由度（channels）和旋转顺序
- **`.amc`** (Acclaim Motion Capture)：逐帧运动数据
  - 每帧每个关节的旋转角度

> ⚠️ CMU 数据是**人体骨骼**格式，需要转换为 G1 机器人的 29 个关节角。

## 2.2 阶段一：CMU ASF+AMC → CSV

**脚本**：`scripts/mimic/cmu_amc_to_csv.py`

转换流程：
```
ASF 骨骼树 + AMC 关节角
        │
        ▼
  正向运动学（FK）计算
  人体每根骨骼的全局位姿
        │
        ▼
  关节映射：人体骨骼 → G1 29DOF
  (手动定义的对应关系)
        │
        ▼
  坐标系转换：CMU → Isaac Lab
        │
        ▼
  CSV 输出 (每行 = 1帧)
```

CSV 每行格式：
```
base_pos_x, base_pos_y, base_pos_z,           # 基座位置 (3)
base_quat_x, base_quat_y, base_quat_z, base_quat_w,  # 基座朝向 (4, xyzw)
joint_1, joint_2, ..., joint_29                # SDK 关节角 (29)
```

> **注意**：CSV 中的四元数是 `xyzw` 格式，后续转 NPZ 时会转为 `wxyz`。

## 2.3 阶段二：CSV → NPZ（物理仿真重放）

**脚本**：`scripts/mimic/csv_to_npz.py`

**这一步不是简单的格式转换**，而是在 Isaac Lab 中实际运行物理仿真引擎：

```python
# 核心逻辑（简化版）
for each_frame in csv_data:
    # 1. 写入关节状态
    robot.write_joint_state_to_sim(joint_pos_clamped, joint_vel_clamped)
    robot.write_root_state_to_sim(root_state)

    # 2. 渲染（不做物理步进！）
    sim.render()  # 只做正向运动学，不做动力学
    scene.update(dt)

    # 3. 收集 FK 计算的真实刚体数据
    log["body_pos_w"].append(robot.data.body_pos_w)
    log["body_quat_w"].append(robot.data.body_quat_w)
    log["body_lin_vel_w"].append(robot.data.body_lin_vel_w)
    log["body_ang_vel_w"].append(robot.data.body_ang_vel_w)
```

**为什么要用仿真引擎做 FK？**

1. **一致性**：训练时 Isaac Lab 用 PhysX 引擎计算刚体位姿。如果 NPZ 中的 body 数据是用自定义 FK 计算的，由于浮点精度、关节建模差异等原因，会与训练时的实际 body 位姿有偏差。偏差导致奖励信号不准确，策略学到错误的行为。

2. **关节限位钳位**：在写入 sim 之前，关节角被钳位到 USD 软限位。这保证了 NPZ body 数据对应的是物理可行的关节配置。

### 帧率插值

CSV 输入帧率（通常 60 FPS）可能与训练帧率（50 FPS）不同。`csv_to_npz.py` 内置了插值逻辑：

- **位置**：线性插值（lerp）
- **朝向**：球面线性插值（slerp）
- **速度**：通过 `torch.gradient()` 从插值后的轨迹计算
- **角速度**：通过 SO(3) 导数计算：$\omega = \frac{\text{axis\_angle}(q_{t+1} \cdot q_{t-1}^{-1})}{2\Delta t}$

## 2.4 数据质量关键：关节限位钳位

这是数据流水线中**最关键的一步**，直接决定训练成败：

```python
# csv_to_npz.py 中的钳位逻辑
joint_lo = robot.data.soft_joint_pos_limits[0, :, 0]
joint_hi = robot.data.soft_joint_pos_limits[0, :, 1]
joint_pos_clamped = torch.clamp(joint_pos, joint_lo, joint_hi)
```

如果不钳位会怎样？

- NPZ 中的 `joint_pos` 超出关节限位 → 训练时 `joint_pos_error` 奖励驱动策略去追踪超限的目标
- 但同时 `joint_limit` 惩罚在阻止关节超限
- **两个奖励信号冲突** → 策略困惑 → 抖动、飞出、训练崩溃

> **MotionCommand 也做了二次钳位**：即使 NPZ 数据已经钳位，在训练加载时还会再次
> clamp 到 `soft_joint_pos_limits`，作为安全保障。

## 2.5 NPZ 文件内容详解

| Key | Shape | 说明 |
|-----|-------|------|
| `fps` | scalar | 帧率（通常 50） |
| `joint_pos` | `[T, 29]` | 关节角度（Isaac Lab 关节顺序） |
| `joint_vel` | `[T, 29]` | 关节角速度 |
| `body_pos_w` | `[T, B, 3]` | 所有刚体世界坐标位置（B=39 for G1） |
| `body_quat_w` | `[T, B, 4]` | 所有刚体世界坐标朝向（wxyz） |
| `body_lin_vel_w` | `[T, B, 3]` | 所有刚体线速度 |
| `body_ang_vel_w` | `[T, B, 3]` | 所有刚体角速度 |

> T = 帧数，B = 刚体数量（G1 29DOF 有 39 个刚体）

---

# 第三部分：机器人物理模型

## 3.1 G1 29DOF 关节定义

G1 机器人有 29 个可驱动关节。在 Isaac Lab 仿真中，关节按照**交错的左-右-腰**顺序排列（SIM 关节顺序）：

| SIM索引 | 关节名 | 位置 |
|---------|--------|------|
| 0 | `left_hip_pitch_joint` | 左髋俯仰 |
| 1 | `right_hip_pitch_joint` | 右髋俯仰 |
| 2 | `waist_yaw_joint` | 腰部偏航 |
| 3 | `left_hip_roll_joint` | 左髋侧摆 |
| 4 | `right_hip_roll_joint` | 右髋侧摆 |
| 5 | `waist_roll_joint` | 腰部侧摆 |
| 6 | `left_hip_yaw_joint` | 左髋旋转 |
| 7 | `right_hip_yaw_joint` | 右髋旋转 |
| 8 | `waist_pitch_joint` | 腰部俯仰 |
| 9 | `left_knee_joint` | 左膝 |
| 10 | `right_knee_joint` | 右膝 |
| 11 | `left_shoulder_pitch_joint` | 左肩俯仰 |
| 12 | `right_shoulder_pitch_joint` | 右肩俯仰 |
| 13 | `left_ankle_pitch_joint` | 左踝俯仰 |
| 14 | `right_ankle_pitch_joint` | 右踝俯仰 |
| 15 | `left_shoulder_roll_joint` | 左肩侧摆 |
| 16 | `right_shoulder_roll_joint` | 右肩侧摆 |
| 17 | `left_ankle_roll_joint` | 左踝侧摆 |
| 18 | `right_ankle_roll_joint` | 右踝侧摆 |
| 19 | `left_shoulder_yaw_joint` | 左肩旋转 |
| 20 | `right_shoulder_yaw_joint` | 右肩旋转 |
| 21 | `left_elbow_joint` | 左肘 |
| 22 | `right_elbow_joint` | 右肘 |
| 23 | `left_wrist_roll_joint` | 左腕滚转 |
| 24 | `right_wrist_roll_joint` | 右腕滚转 |
| 25 | `left_wrist_pitch_joint` | 左腕俯仰 |
| 26 | `right_wrist_pitch_joint` | 右腕俯仰 |
| 27 | `left_wrist_yaw_joint` | 左腕偏航 |
| 28 | `right_wrist_yaw_joint` | 右腕偏航 |

> ⚠️ SDK 关节顺序（用于真实硬件通信）与 SIM 关节顺序不同！
> `export_deploy_cfg.py` 会在训练结束时导出 `joint_ids_map` 映射表。

## 3.2 执行器（Actuator）配置

G1 使用 **ImplicitActuatorCfg**（隐式执行器），参数基于电机物理特性：

```
自然频率 ω = 10 × 2π = 62.83 rad/s
阻尼比 ζ = 2.0（过阻尼，确保稳定）

刚度 K = armature × ω²
阻尼 D = 2 × ζ × armature × ω
```

| 电机型号 | 转子惯量 (armature) | 刚度 K | 阻尼 D | 使用位置 |
|---------|---------------------|--------|--------|---------|
| 5020 | 0.003610 | 14.25 | 0.907 | 肩部、肘部 |
| 7520-14 | 0.010178 | 40.18 | 2.558 | 髋俯仰/偏航、膝、腰偏航 |
| 7520-22 | 0.025102 | 99.10 | 6.309 | 髋侧摆、膝 |
| 4010 | 0.004250 | 16.78 | 1.068 | 腕俯仰、腕偏航 |

**分组配置**（`g1.py`）：

| 组名 | 关节 | 力矩限制 | 速度限制 |
|------|------|---------|---------|
| `legs` | hip_yaw/roll/pitch, knee | 88~139 Nm | 20~32 rad/s |
| `feet` | ankle_pitch/roll | 50 Nm | 37 rad/s |
| `waist` | waist_roll/pitch | 50 Nm | 37 rad/s |
| `waist_yaw` | waist_yaw | 88 Nm | 32 rad/s |
| `arms` | shoulder_*, elbow, wrist_* | 5~25 Nm | 22~37 rad/s |

## 3.3 动作缩放（Action Scale）

策略输出 $a \in [-1, 1]^{29}$，经过 Action Scale 转换为关节位置偏移量：

$$\Delta q_i = a_i \times \text{scale}_i$$

其中：

$$\text{scale}_i = 0.25 \times \frac{\text{effort\_limit}_i}{\text{stiffness}_i}$$

这个设计确保策略输出 $a=1$ 时，关节偏移量约为最大力矩可达范围的 25%。
不同关节因电机规格不同，scale 值不同：

- 腿部大关节：scale 较大（力矩大、刚度高）
- 手腕小关节：scale 较小（力矩小、需要精细控制）

## 3.4 USD vs URDF 关节限位差异

G1 的 USD 模型和 URDF 模型在 **9 个关节** 上存在限位差异：

| 关节 | URDF 限位 | USD 限位 | 差异来源 |
|------|----------|---------|---------|
| `wrist_pitch` | ±0.3491 | ±1.6144 | USD 为仿真放宽 |
| `wrist_yaw` | ±0.5236 | ±1.6144 | 同上 |
| `ankle_roll` | ±0.2618 | ±0.2618 | 一致 |
| ... | ... | ... | ... |

**关键设计决策**：数据制备和训练都使用 **USD 限位**，保持内部一致性。
部署到真实硬件时，在 C++ 层额外钳位到 URDF 限位。

> `soft_joint_pos_limit_factor = 0.9` 进一步将有效范围缩小到 USD 限位的 90%。

## 3.5 碰撞体几何分析

G1 29DOF 的 USD 模型在 PhysX 加载后暴露 **30 个 articulation body**。

URDF 中还有一些固定关节链接（如 `pelvis_contour_link`、`logo_link`、`head_link`），
但 PhysX 会将固定关节链接**合并到父刚体**中，它们不会出现在 `robot.body_names` 列表中。

> ⚠️ **在 `SceneEntityCfg(body_names=...)` 中只能使用 articulation body 名字**。
> 使用已合并的固定关节链接名字（如 `pelvis_contour_link`）会导致运行时崩溃：
> `Available strings: [...]` 错误。

---

# 第四部分：训练环境架构

## 4.1 ManagerBasedRLEnv 总体框架

Isaac Lab 的 `ManagerBasedRLEnv` 使用**管理器模式**组织环境逻辑：

```
ManagerBasedRLEnv
├── SceneManager        → 物理场景（地形、机器人、灯光、传感器）
├── ActionManager       → 动作处理（scale、offset、clip）
├── ObservationManager  → 观测计算（policy obs + critic obs）
├── CommandManager      → 指令生成（MotionCommand 运动跟踪）
├── RewardManager       → 奖励计算（多项奖励加权求和）
├── TerminationManager  → 终止判断（多条件 OR 逻辑）
├── EventManager        → 随机化事件（startup + interval）
└── CurriculumManager   → 课程学习（本项目未使用）
```

每个 Manager 内包含多个 **Term**（项），在 `tracking_env_cfg.py` 中通过 `@configclass` 配置。

## 4.2 MotionCommand：运动跟踪核心

`MotionCommand`（定义在 `commands.py`）是整个运动跟踪训练的**核心组件**。它负责：

### 职责

1. **加载参考动作**：从 NPZ 加载运动数据到 GPU
2. **时间步进**：管理每个环境的当前播放时间
3. **坐标变换**：将参考动作从世界坐标转换为机器人相对坐标
4. **自适应采样**：根据失败统计动态调整采样分布
5. **重采样初始化**：在 episode 开始时设置机器人到参考姿态
6. **关节限位钳位**：在加载时将 NPZ joint_pos 钳位到 USD 软限位

### 数据流

```
NPZ (joint_pos, body_pos_w, body_quat_w, ...)
      │
      ▼  MotionLoader (加载到 GPU)
      │
      ▼  time_steps 索引
      │
      ├──► joint_pos / joint_vel (关节目标)
      ├──► body_pos_w / body_quat_w (刚体世界位姿)
      ├──► anchor_pos_w / anchor_quat_w (锚点位姿)
      │
      ▼  _update_command() 坐标变换
      │
      └──► body_pos_relative_w / body_quat_relative_w (相对位姿)
           └──► 用于奖励计算
```

### 关键属性

| 属性 | Shape | 说明 |
|------|-------|------|
| `joint_pos` | `[E, 29]` | 当前帧参考关节角 |
| `joint_vel` | `[E, 29]` | 当前帧参考关节速度 |
| `anchor_pos_w` | `[E, 3]` | 锚点世界位置 |
| `anchor_quat_w` | `[E, 4]` | 锚点世界朝向 |
| `body_pos_relative_w` | `[E, 14, 3]` | 相对坐标下的刚体位置 |
| `body_quat_relative_w` | `[E, 14, 4]` | 相对坐标下的刚体朝向 |

> E = 环境数量

## 4.3 相对坐标变换：yaw-only 对齐

**为什么不直接用世界坐标？**

- 参考动作的起始位置/朝向是固定的
- 但训练时机器人会被随机初始化到不同位置/朝向
- 如果用世界坐标，位置差异会导致巨大的跟踪误差

**yaw-only 对齐**：

```python
# 只提取偏航（yaw）差异，忽略 roll 和 pitch
delta_ori_w = yaw_quat(quat_mul(robot_anchor_quat, quat_inv(ref_anchor_quat)))

# 用 yaw 旋转变换参考动作的位置
body_pos_relative = delta_pos + quat_apply(delta_ori_w, body_pos_w - anchor_pos_w)

# 用 yaw 旋转变换参考动作的朝向
body_quat_relative = quat_mul(delta_ori_w, body_quat_w)
```

直觉理解：
1. 计算机器人当前朝向与参考动作朝向的 **yaw 差**
2. 用这个 yaw 差旋转所有参考刚体的位姿
3. 结果：参考动作 "看起来" 总是朝着机器人当前方向
4. Z 轴（高度）直接用参考值，不做变换（重力方向不变）

## 4.4 观测空间设计

### Policy 观测（Actor 网络输入）

策略网络只能看到**机器人自身能感知的信息**：

| 观测项 | 维度 | 说明 | 噪声 |
|--------|------|------|------|
| `motion_command` | 58 | joint_pos(29) + joint_vel(29) | 无 |
| `motion_anchor_ori_b` | 6 | 锚点朝向（旋转矩阵前2列）| ±0.05 |
| `base_ang_vel` | 3 | 基座角速度（IMU） | ±0.2 |
| `joint_pos_rel` | 29 | 关节位置（相对默认） | ±0.01 |
| `joint_vel_rel` | 29 | 关节速度 | ±0.5 |
| `last_action` | 29 | 上一步动作 | 无 |
| **总计** | **154** | | |

### Critic 观测（Critic 网络输入，含特权信息）

Critic 网络可以看到**仿真中的完美信息**（不加噪声）：

| 观测项 | 维度 | 说明 |
|--------|------|------|
| `command` | 58 | joint_pos + joint_vel |
| `motion_anchor_pos_b` | 3 | 锚点相对位置 |
| `motion_anchor_ori_b` | 6 | 锚点相对朝向 |
| `body_pos` | 42 | 14个刚体相对位置 |
| `body_ori` | 84 | 14个刚体相对朝向（旋转矩阵前2列）|
| `base_lin_vel` | 3 | 基座线速度 |
| `base_ang_vel` | 3 | 基座角速度 |
| `joint_pos` | 29 | 关节位置 |
| `joint_vel` | 29 | 关节速度 |
| `actions` | 29 | 上一步动作 |

> **Asymmetric Actor-Critic**：Actor 看不到 anchor_pos_b、body_pos/ori、base_lin_vel，
> 但 Critic 可以看到。这让 Critic 能更准确地估计 Value 函数，提供更好的梯度信号。

## 4.5 动作空间设计

- **维度**：29（每个关节一个动作值）
- **范围**：$[-1, 1]^{29}$（由 `clip_actions` 裁剪）
- **类型**：关节位置偏移（JointPositionAction）
- **计算**：目标关节位置 = default_joint_pos + action × scale

```python
# 内部实现
target_q = self._offset + action * self._scale  # offset = default_joint_pos
# 然后由 PhysX 的 PD 控制器驱动关节到 target_q
```

## 4.6 Domain Randomization（域随机化）

为了让策略在真实硬件上也能工作，训练时引入多种随机化：

### Startup 随机化（初始化时一次性设置）

| 项目 | 参数 | 说明 |
|------|------|------|
| 物理材料 | 静摩擦 0.3~1.6，动摩擦 0.3~1.2 | 不同地面材质 |
| 关节默认位置 | ±0.01 rad 扰动 | 模拟校准误差 |
| 质心偏移 | x ±0.025, y ±0.05, z ±0.05 m | 负载不确定性 |

### Interval 随机化（训练中周期性触发）

| 项目 | 频率 | 参数 | 说明 |
|------|------|------|------|
| 外力推动 | 2~5秒 | lin_vel ±0.5, ang_vel ±0.78 | 模拟外部扰动 |

### Resample 随机化（episode 开始时）

| 项目 | 参数 | 说明 |
|------|------|------|
| 初始位姿偏移 | pos ±0.05m, ori ±0.2rad | 不从精确参考点开始 |
| 初始速度 | lin_vel ±0.5, ang_vel ±0.78 | 带初速度开始 |
| 初始关节角 | ±0.1 rad 扰动 | 关节不精确初始化 |

---

# 第五部分：强化学习算法

## 5.1 PPO（Proximal Policy Optimization）

本项目使用 **RSL-RL** 库的 PPO 实现（`OnPolicyRunner`）。

PPO 是一种 **On-Policy** 策略梯度方法，核心思想：
1. 用当前策略采集数据
2. 计算优势函数（GAE）
3. 对策略参数做多步更新，但用 clipping 限制每步更新幅度

策略梯度（Clipped Surrogate Objective）：

$$L^{\text{CLIP}} = \mathbb{E}\left[\min\left(r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t\right)\right]$$

其中：
- $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_\text{old}}(a_t|s_t)}$ 是概率比
- $\hat{A}_t$ 是广义优势估计（GAE）
- $\epsilon = 0.2$ 是 clipping 参数

GAE（Generalized Advantage Estimation）：

$$\hat{A}_t = \sum_{l=0}^{\infty} (\gamma\lambda)^l \delta_{t+l}$$

$$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

## 5.2 Actor-Critic 网络架构

```
Actor 网络（策略）                    Critic 网络（价值函数）
┌──────────────────┐              ┌──────────────────┐
│  Policy Obs (154) │              │ Critic Obs (286+) │
│        ↓         │              │        ↓         │
│  Linear(512) + ELU│              │  Linear(512) + ELU│
│        ↓         │              │        ↓         │
│  Linear(256) + ELU│              │  Linear(256) + ELU│
│        ↓         │              │        ↓         │
│  Linear(128) + ELU│              │  Linear(128) + ELU│
│        ↓         │              │        ↓         │
│  Linear(29)       │              │  Linear(1)        │
│  μ(s) → 高斯均值  │              │  V(s) → 状态价值  │
└──────────────────┘              └──────────────────┘
      ↓
  action = μ(s) + σ·ε
  σ: 初始 std = 1.0, 可学习
```

**网络规格**：
- 隐藏层：`[512, 256, 128]`
- 激活函数：ELU（比 ReLU 在负区域有梯度，训练更稳定）
- 初始噪声标准差：`init_noise_std = 1.0`（较大，鼓励早期探索）
- 总参数量：约 `154×512 + 512×256 + 256×128 + 128×29 ≈ 246K`（Actor）

## 5.3 Asymmetric Actor-Critic

本项目使用 **非对称 Actor-Critic** 设计：

- **Actor** 只看 Policy Obs（模拟真实传感器的信息）
- **Critic** 看 Privileged Obs（包含仿真特权信息）

优势：
- Critic 有更多信息 → 更准确的 Value 估计 → 更好的 GAE → 更稳定的策略更新
- Actor 只依赖真实可获取的信息 → 可以直接部署到硬件（无需 teacher-student 蒸馏）

## 5.4 PPO 超参数详解

| 参数 | 值 | 说明 |
|------|-----|------|
| `num_steps_per_env` | 24 | 每个环境采集24步后更新一次 |
| `max_iterations` | 30000 | 最大训练迭代次数 |
| `num_learning_epochs` | 5 | 每次更新的 epoch 数 |
| `num_mini_batches` | 4 | mini-batch 数量 |
| `clip_param` | 0.2 | PPO clipping 参数 ε |
| `learning_rate` | 1e-3 | 初始学习率 |
| `schedule` | `"adaptive"` | 自适应学习率（根据 KL 散度调整）|
| `desired_kl` | 0.01 | 目标 KL 散度 |
| `gamma` | 0.99 | 折扣因子 |
| `lam` | 0.95 | GAE λ 参数 |
| `entropy_coef` | 0.005 | 熵正则化系数 |
| `value_loss_coef` | 1.0 | Value loss 权重 |
| `max_grad_norm` | 1.0 | 梯度裁剪 |

**自适应学习率**：
- 如果 KL 散度 > `desired_kl`×2 → 学习率 /= 1.5（更新太大，减速）
- 如果 KL 散度 < `desired_kl`/2 → 学习率 *= 1.5（更新太小，加速）
- 这防止了早期训练时学习率过大导致策略崩溃

**为什么 entropy_coef = 0.005？**
- 太大（如 0.01）：策略过于随机，学不到精确动作
- 太小（如 0.001）：过早收敛到次优解，丧失探索能力
- 0.005 是武术跟踪的经验最优值

---

# 第六部分：奖励函数设计（核心）

> **奖励函数是整个项目最重要的设计决策。** 错误的奖励设计会导致：
> 策略耍小聪明（reward hacking）、姿态畸形、训练崩溃。

## 6.1 奖励设计哲学

### 三层目标

1. **跟踪层**（最重要）：让机器人尽可能精确地复现参考动作
2. **正则化层**：防止高频抖动、过大力矩、关节超限
3. **安全层**：惩罚/终止危险的身体接触

### 权重分配原则

```
正奖励总和 >> |负奖励总和|
```

正奖励驱动策略学习目标行为，负奖励约束行为边界。
如果负奖励过大，策略会 "学会不动"（standing still is safe）。

## 6.2 指数核函数（Exp Kernel）

所有跟踪奖励使用统一的**指数核**形式：

$$r = \exp\left(-\frac{\text{error}^2}{\sigma^2}\right)$$

- $r \in (0, 1]$：误差为0时奖励最大 = 1，误差增大时平滑衰减
- $\sigma$ 控制 "容忍度"：
  - $\sigma$ 大 → 曲线平坦 → 对误差不敏感 → 容易获得奖励
  - $\sigma$ 小 → 曲线陡峭 → 对误差敏感 → 难以获得奖励

**调参直觉**：

| $\sigma$ | 误差 = $\sigma$ 时的奖励 | 适用场景 |
|----------|----------------------|---------|
| 0.3 | 0.37 | 精确跟踪（末端执行器） |
| 0.5 | 0.37 | 中等精度（锚点位置） |
| 0.8 | 0.37 | 宽松跟踪（全身关节） |
| 1.5 | 0.37 | 非常宽松（线速度） |
| 3.14 | 0.37 | 几乎不惩罚（角速度） |

> **规律**：误差等于 $\sigma$ 时，奖励恒等于 $e^{-1} \approx 0.37$。
> 选择 $\sigma$ 就是在定义 "可接受的误差量级"。

## 6.3 锚点（Anchor）跟踪奖励

锚点 = `torso_link`（躯干），代表机器人的全局位姿。

### 锚点位置跟踪

$$r_{\text{anchor\_pos}} = \exp\left(-\frac{\|p_{\text{ref}} - p_{\text{robot}}\|^2}{\sigma^2}\right)$$

- **weight = 1.5**，$\sigma = 0.5$
- 追踪躯干在世界坐标系中的 3D 位置
- $\sigma = 0.5$ m 表示 "50cm 以内的误差可以接受"

### 锚点朝向跟踪

$$r_{\text{anchor\_ori}} = \exp\left(-\frac{\text{quat\_error}(q_{\text{ref}}, q_{\text{robot}})^2}{\sigma^2}\right)$$

- **weight = 1.0**，$\sigma = 0.5$
- 使用四元数误差（角度量级）

## 6.4 全身体追踪奖励

对 14 个跟踪刚体计算位置/朝向误差的**平均值**：

### 14 个跟踪刚体

```
pelvis, left_hip_roll_link, left_knee_link, left_ankle_roll_link,
right_hip_roll_link, right_knee_link, right_ankle_roll_link,
torso_link, left_shoulder_roll_link, left_elbow_link,
left_wrist_yaw_link, right_shoulder_roll_link, right_elbow_link,
right_wrist_yaw_link
```

### Body Position

$$r_{\text{body\_pos}} = \exp\left(-\frac{\frac{1}{14}\sum_{i=1}^{14}\|p_i^{\text{ref}} - p_i^{\text{robot}}\|^2}{\sigma^2}\right)$$

- **weight = 1.0**，$\sigma = 0.3$
- 使用**相对坐标**（经过 yaw-only 对齐）

### Body Orientation

$$r_{\text{body\_ori}} = \exp\left(-\frac{\frac{1}{14}\sum_{i=1}^{14}\text{quat\_error}(q_i^{\text{ref}}, q_i^{\text{robot}})^2}{\sigma^2}\right)$$

- **weight = 1.0**，$\sigma = 0.4$

## 6.5 关节位置跟踪奖励（关键！）

**这是 v9 最重要的修复之一**，在 v8 中被错误移除后导致了严重的姿态畸形。

$$r_{\text{joint\_pos}} = \exp\left(-\frac{\frac{1}{29}\sum_{j=1}^{29}(q_j^{\text{ref}} - q_j^{\text{robot}})^2}{\sigma^2}\right)$$

- **weight = 2.0**（v9），$\sigma = 0.8$

### 为什么 body_pos/ori 不够？为什么还需要 joint_pos？

**关键洞察：正向运动学的多解性（FK Degeneracy）**

一个刚体（link）的质心位置由多个关节共同决定。不同的关节角组合可以产生相同的质心位置！

例如：膝关节弯曲 30° + 髋关节俯仰 20° 的大腿质心位置，可能与膝关节弯曲 -10° + 髋关节俯仰 45° 得到的位置很接近。

**没有 joint_pos 奖励时**：策略只优化 body 质心位置 → 找到"简单但错误"的关节配置 → **膝盖反弯、胳膊拧过头、姿势怪异**。

**有 joint_pos 奖励时**：策略被迫同时匹配每个关节的具体角度 → 消除多解性 → 姿态正确。

> 📌 **教训**：永远不要只跟踪 body 位姿，一定要同时跟踪 joint 角度！

## 6.6 末端执行器跟踪奖励

$$r_{\text{ee\_pos}} = \exp\left(-\frac{\frac{1}{4}\sum_{i=1}^{4}\|p_i^{\text{ref}} - p_i^{\text{robot}}\|^2}{\sigma^2}\right)$$

- **weight = 1.5**，$\sigma = 0.3$
- 追踪 4 个末端执行器：`left/right_ankle_roll_link`（脚）+ `left/right_wrist_yaw_link`（手）
- 武术动作中手脚的精确位置至关重要（踢击/出拳的落点）

> 注意：这复用了 `motion_relative_body_position_error_exp` 函数，
> 但通过 `body_names` 参数只选择末端执行器。

## 6.7 速度跟踪奖励

### 线速度

$$r_{\text{lin\_vel}} = \exp\left(-\frac{\frac{1}{14}\sum_{i}\|\dot{p}_i^{\text{ref}} - \dot{p}_i^{\text{robot}}\|^2}{\sigma^2}\right)$$

- **weight = 1.0**，$\sigma = 1.5$ m/s
- 较宽松的 $\sigma$，因为速度的绝对精度不如位置重要

### 角速度

$$r_{\text{ang\_vel}} = \exp\left(-\frac{\frac{1}{14}\sum_{i}\|\omega_i^{\text{ref}} - \omega_i^{\text{robot}}\|^2}{\sigma^2}\right)$$

- **weight = 1.0**，$\sigma = 3.14$ rad/s
- 非常宽松，主要提供方向性引导

## 6.8 正则化惩罚

| 惩罚项 | 权重 | 公式 | 作用 |
|--------|------|------|------|
| `joint_acc` | -2.5e-7 | $\sum \ddot{q}^2$ | 抑制关节加速度（平滑运动） |
| `joint_torque` | -1e-5 | $\sum \tau^2$ | 抑制过大力矩（节能） |
| `action_rate` | -0.1 | $\sum (a_t - a_{t-1})^2$ | 抑制动作突变（平滑控制） |
| `joint_limit` | -10.0 | 超限距离 | 强力阻止关节超出限位 |

**action_rate 的权重选择**：

- `-0.01`：几乎无约束，动作会抖动
- `-0.1`：轻微平滑，推荐（v9 使用）
- `-1.0`：过度平滑，动作迟钝，跟不上快速武术动作

## 6.9 接触惩罚

```python
undesired_contacts = RewTerm(
    func=mdp.undesired_contacts,
    weight=-1.0,  # v9: 从 -0.1 提升到 -1.0
    params={
        "sensor_cfg": SceneEntityCfg(
            "contact_forces",
            body_names=[
                r"^(?!left_ankle_roll_link$)(?!right_ankle_roll_link$)"
                r"(?!left_wrist_yaw_link$)(?!right_wrist_yaw_link$).+$"
            ],
        ),
        "threshold": 1.0,  # N
    },
)
```

**正则表达式** `^(?!...)...+$` 的含义：
- 匹配所有 body name，**排除**脚（ankle_roll）和手（wrist_yaw）
- 即：除了脚和手，其他任何 body 接触地面都会产生惩罚

**v8 → v9 权重变化**：-0.1 → -1.0

v8 的 -0.1 太弱，策略发现 "跪在地上" 获得的跟踪奖励 > 接触惩罚，于是选择跪着做动作。
v9 提升到 -1.0 后，接触惩罚的梯度信号足够强，策略被迫保持站立。

## 6.10 奖励权重总表（v9）

### 正奖励（跟踪目标，总和 = 10.0）

| 奖励项 | 权重 | σ | 追踪目标 |
|--------|------|---|---------|
| `anchor_pos` | 1.5 | 0.5 | 躯干世界位置 |
| `anchor_ori` | 1.0 | 0.5 | 躯干世界朝向 |
| `body_pos` | 1.0 | 0.3 | 14 刚体相对位置 |
| `body_ori` | 1.0 | 0.4 | 14 刚体相对朝向 |
| **`joint_pos`** | **2.0** | **0.8** | **29 关节角度** |
| `ee_pos` | 1.5 | 0.3 | 4 末端执行器位置 |
| `lin_vel` | 1.0 | 1.5 | 14 刚体线速度 |
| `ang_vel` | 1.0 | 3.14 | 14 刚体角速度 |

### 负奖励（正则化 + 安全）

| 惩罚项 | 权重 | 说明 |
|--------|------|------|
| `joint_acc` | -2.5e-7 | 加速度 L2 |
| `joint_torque` | -1e-5 | 力矩 L2 |
| `action_rate` | -0.1 | 动作变化率 L2 |
| `joint_limit` | -10.0 | 关节超限 |
| `undesired_contacts` | -1.0 | 非法接触 |

---

# 第七部分：终止条件设计

## 7.1 终止条件的作用

终止条件（Termination）决定 episode 何时结束。两种类型：

1. **time_out**（超时）：正常结束，不算失败，GAE 计算时 bootstrap value
2. **非 time_out**：失败终止，GAE 计算时 value = 0

**设计原则**：
- 太严格 → episode 太短 → 策略学不到完整动作 → 训练崩溃
- 太宽松 → 策略可以在错误状态下长期运行 → 学到坏习惯
- 需要**刚好在策略无法恢复时终止**

## 7.2 锚点位置/朝向终止

### anchor_pos_z_only

```python
# 只检查 Z 轴（高度）偏差
|z_ref - z_robot| > 0.5m  →  终止
```

**为什么只检查 Z 轴？**
- 武术动作有大幅度的水平移动（弓步、侧踢）
- 检查全 3D 距离会因水平漂移误触发终止
- Z 轴偏差意味着 "摔倒"，是真正需要终止的情况

**v8 放宽到 0.5m 的原因**：
- v7 用 0.25m → 高踢时躯干高度变化大 → 提前终止 → 训练崩溃
- 0.5m 允许更大的垂直运动范围

### anchor_ori

```python
# 检查重力投影差异
|g_ref · z - g_robot · z| > 1.2  →  终止
```

- 比较参考和实际朝向的重力方向投影
- 1.2 约对应 ~60° 的朝向偏差
- v7 的 0.8（~40°）对回旋踢来说太严格

## 7.3 末端执行器位置终止

```python
# 4 个末端执行器的 Z 轴偏差
any(|z_ref_i - z_robot_i| > 0.8m)  →  终止
```

- 检查手脚的高度是否严重偏离参考
- v8 放宽到 0.8m：高踢时脚的 Z 变化可达 0.7m

## 7.4 非法接触终止（v9 新增）

**这是 v9 解决 "膝盖着地" 问题的关键修复。**

```python
illegal_contact = DoneTerm(
    func=mdp.illegal_contact,
    params={
        "sensor_cfg": SceneEntityCfg(
            "contact_forces",
            body_names=ILLEGAL_CONTACT_BODIES,  # 26 个身体
        ),
        "threshold": 100.0,  # N
    },
)
```

### 26 个非法接触身体

**允许接触的**（4个）：
- `left/right_ankle_roll_link`（脚底）— 正常站立
- `left/right_wrist_yaw_link`（手）— 武术中可能触地

**禁止接触的**（26个，包括但不限于）：
- 骨盆（`pelvis`）
- 膝盖（`left/right_knee_link`）
- 手肘（`left/right_elbow_link`）
- 躯干（`torso_link`）
- 腰部（`waist_yaw_link`、`waist_roll_link`）
- 所有髋关节链接
- 所有肩关节链接
- 腕部非末端链接

> ⚠️ **关键发现**：`pelvis_contour_link`、`logo_link`、`head_link` 在 URDF 中存在，
> 但它们是**固定关节链接**，被 PhysX 合并到父链接中，**不出现在 articulation body 列表中**！
> 如果在 `body_names` 中使用这些名字，Isaac Lab 会报 "Available strings" 错误并崩溃。
> 必须使用实际的 articulation body 名字（共 30 个）。

### threshold = 100N 的选择

- 太低（10N）：正常动作中的轻微碰撞触发误终止
- 太高（1000N）：只有重摔才终止，膝盖软着地 不会触发
- 100N：约 10kg 的力 → 有意义的接触 → 合理阈值

## 7.5 终止条件总表（v9）

| 终止项 | 类型 | 条件 | 阈值 | 说明 |
|--------|------|------|------|------|
| `time_out` | timeout | 超过 episode_length_s | 30s | 正常结束 |
| `anchor_pos` | failure | Z 偏差 | 0.5m | 摔倒检测 |
| `anchor_ori` | failure | 朝向偏差 | 1.2 | 翻倒检测 |
| `ee_body_pos` | failure | 末端 Z 偏差 | 0.8m | 手脚失控 |
| `illegal_contact` | failure | 26体接触力 | 100N | 膝/肘/头触地 |

---

# 第八部分：自适应采样机制

## 8.1 为什么需要自适应采样

一段武术动作中有 **难度不均** 的片段：

- 站立部分（简单）→ 策略很快学会
- 高踢部分（困难）→ 策略反复失败

如果均匀随机采样起始帧：
- 大部分 episode 从简单片段开始 → 策略过拟合简单部分
- 困难片段的采样比例太低 → 学不到困难部分

**自适应采样**：根据失败统计，增加困难片段的采样概率。

## 8.2 算法原理

### 时间分箱（Binning）

将整段动作按时间分成 $N$ 个 bin：

$$N = \left\lfloor\frac{T_{\text{total}}}{dt}\right\rfloor + 1$$

其中 $dt = \text{decimation} \times \text{sim\_dt} = 4 \times 0.005 = 0.02s$。

### 失败计数与 EMA 更新

每次 episode 终止（非 timeout）时：
1. 记录终止时的 bin 索引
2. 用 **指数移动平均（EMA）** 更新失败统计：

$$f_{\text{new}} = \alpha \cdot f_{\text{current}} + (1-\alpha) \cdot f_{\text{old}}$$

$\alpha = 0.002$（非常慢的更新，确保统计稳定）

### 核平滑

对失败分布进行 **1D 卷积平滑**，防止尖刺采样：

$$p_{\text{smooth}} = f * k$$

其中核 $k = [\lambda^0, \lambda^1, \lambda^2]$（归一化后），$\lambda = 0.8$，核大小 = 3。

### 采样概率

$$p_i = \frac{f_i^{\text{smooth}} + u/N}{\sum_j (f_j^{\text{smooth}} + u/N)}$$

$u = 0.1$（uniform_ratio），确保每个 bin 至少有 10% 基础概率，避免完全忽略简单片段。

### 采样流程

```
bin_failed_count  →  EMA 更新  →  核平滑  →  + uniform  →  归一化  →  multinomial 采样
                                                                          │
                                                                          ▼
                                                                 sampled_bin + random_offset
                                                                          │
                                                                          ▼
                                                                    time_step
```

## 8.3 参数调优指南

| 参数 | 默认值 | 作用 | 调优建议 |
|------|--------|------|---------|
| `adaptive_alpha` | 0.002 | EMA 更新速率 | 增大→更快适应；减小→更稳定 |
| `adaptive_kernel_size` | 3 | 平滑核大小 | 增大→更平滑的采样分布 |
| `adaptive_lambda` | 0.8 | 核衰减系数 | 增大→更强的邻域扩散 |
| `adaptive_uniform_ratio` | 0.1 | 均匀混合比例 | 增大→更多探索；减小→更多利用 |

### 训练监控指标

| 指标 | TensorBoard key | 含义 | 健康范围 |
|------|-----------------|------|---------|
| 采样熵 | `sampling_entropy` | 采样分布的均匀程度 | 0.5~0.9 |
| Top1概率 | `sampling_top1_prob` | 最热 bin 的采样概率 | < 0.3 |
| Top1位置 | `sampling_top1_bin` | 最难片段的位置 | 应与已知难点一致 |

- 熵太低（< 0.3）：采样过度集中在某几个 bin → 可能过拟合
- 熵太高（> 0.95）：几乎均匀采样 → 自适应没有生效

---

# 第九部分：部署架构

## 9.1 Policy Sequencer 原理

训练得到 7 个独立的 ONNX 策略后，需要将它们串联成一场完整的武术表演。

**核心挑战**：
- 每个策略只学会了一个动作
- 策略之间没有 "交接" 训练
- 直接切换会导致突然的关节跳变 → 不安全

**解决方案：Policy Sequencer**
```
Segment 0 (heian_shodan)
    → 执行策略 0 直到动作结束
    → transition_hold（1秒平滑过渡）
Segment 1 (front_kick)
    → 执行策略 1 直到动作结束
    → transition_hold
...
Segment 6 (empi)
    → 执行策略 6 直到动作结束
    → 回到 FixStand
```

## 9.2 C++ 实现架构

```
State_MartialArtsSequencer : FSMState
├── segments_: vector<SequencerSegment>  // 7 个动作段
├── current_env_: ManagerBasedRLEnv      // 当前段的推理环境
├── policy_thread_: thread               // 推理线程
│
├── enter()     → 加载第一个段，启动推理线程
├── run()       → 在主线程中写关节命令
├── exit()      → 停止推理线程，清理资源
│
└── policy_loop()  // 推理线程主循环
    ├── 执行当前段（按 step_dt 节奏推理）
    ├── 段结束 → 如果还有下一段:
    │   ├── 捕获当前关节位置 start_q
    │   ├── 加载下一段并 reset
    │   ├── 获取新段初始关节位置 target_q
    │   └── 在 transition_hold_s 内线性插值 start_q → target_q
    └── 所有段执行完毕 → finished = true
```

### 双线程架构

```
主线程（2ms 控制周期）               推理线程（20ms 推理周期）
┌──────────────────┐              ┌──────────────────┐
│  读取传感器       │              │  读取传感器       │
│  ↓               │              │  计算观测         │
│  from policy:    │              │  ONNX 推理        │
│  action[] ────────│──── 共享 ────│──► action[]      │
│  ↓               │              │  sleep(step_dt)   │
│  写入电机命令     │              │  loop...          │
│  loop...         │              └──────────────────┘
└──────────────────┘
```

## 9.3 过渡插值（Transition Hold）

段间切换时的线性插值：

```cpp
for (int step = 0; step <= hold_steps; ++step) {
    float alpha = float(step) / hold_steps;
    for (size_t i = 0; i < num_joints; ++i) {
        interp_q[i] = start_q[i] + alpha * (target_q[i] - start_q[i]);
    }
    // 写入关节命令
    sleep(step_dt);
}
```

- `transition_hold_s = 1.0`（默认 1 秒）
- 插值步数 = 1.0 / 0.02 = 50 步
- 关节从上一段末端位置平滑过渡到下一段初始位置
- 过渡期间 PD 增益保持不变

## 9.4 部署配置文件

`deploy/robots/g1_29dof/config/config.yaml`：

```yaml
MartialArtsSequencer:
  transitions:
    Passive: LT + B.on_pressed
    FixStand: RB + X.on_pressed
  transition_hold_s: 1.0
  segments:
    - policy_dir: config/policy/mimic/martial_arts/front_kick/
      motion_file: config/policy/mimic/martial_arts/front_kick/params/G1_front_kick.csv
      fps: 50
    - policy_dir: config/policy/mimic/martial_arts/lunge_punch/
      ...
    # ... 共 7 个段
```

每个 `policy_dir` 包含：
```
policy_dir/
├── exported/
│   └── policy.onnx       ← 训练导出的 ONNX 模型
└── params/
    ├── deploy.yaml        ← 部署参数（关节映射、增益、观测配置）
    └── G1_xxx.csv         ← 动作参考数据（用于运动命令时间步进）
```

`deploy.yaml` 由 `export_deploy_cfg.py` 在训练结束时自动生成，包含：
- `joint_ids_map`：SIM→SDK 关节映射
- `step_dt`：推理步长
- `stiffness` / `damping`：PD 增益
- `actions.scale` / `actions.offset`：动作缩放和偏移
- `observations`：各观测项的 scale 和 clip

## 9.5 Sim2Sim → Sim2Real 工作流

```
1. Isaac Lab 训练  →  ONNX Policy
          │
2. Mujoco Sim2Sim  →  在 Mujoco 中验证策略（不同物理引擎）
          │
3. 真实硬件部署   →  Unitree SDK + C++ FSM
```

Sim2Sim 验证的意义：
- Isaac Lab 使用 PhysX 引擎
- Mujoco 使用不同的物理模型
- 如果策略在两个引擎中都表现良好 → 对物理参数不敏感 → 更可能在真实硬件上工作
- 如果在 Mujoco 中失败 → 说明策略过度拟合了 PhysX 的特性

---

# 第十部分：完整训练工作流

## 10.1 环境准备

```bash
# 1. 确保 Isaac Lab 已安装
# 2. 安装本项目
cd unitree_rl_lab
./unitree_rl_lab.sh -i  # 以 editable 模式安装

# 3. 验证安装
python scripts/list_envs.py
# 应该能看到 7 个 MartialArts 任务
```

## 10.2 数据制备

```bash
# 1. CMU ASF+AMC → CSV
python scripts/mimic/cmu_amc_to_csv.py -f data/cmu_mocap/135/135_03.amc

# 2. CSV → NPZ
python scripts/mimic/csv_to_npz.py \
    -f source/.../martial_arts/G1_front_kick.csv \
    --input_fps 60 \
    --output_fps 50 \
    --headless
```

> NPZ 文件应放在 `source/.../martial_arts/` 目录下，与 `tracking_env_cfg.py` 同级。

## 10.3 训练命令

```bash
# 训练单个动作（以前踢为例）
python scripts/rsl_rl/train.py \
    --task Unitree-G1-29dof-Mimic-MartialArts-FrontKick \
    --headless \
    --num_envs 512

# RTX 4070 Ti (12GB) 推荐 num_envs=512
# RTX 4090 (24GB) 可以用 num_envs=4096
# 训练约 30000 iterations，约需 6~12 小时

# 恢复训练
python scripts/rsl_rl/train.py \
    --task Unitree-G1-29dof-Mimic-MartialArts-FrontKick \
    --headless --num_envs 512 --resume
```

**训练所有 7 个动作**：

```bash
# 可以串行训练
for task in HeianShodan FrontKick RoundhouseKick LungePunch SideKick Bassai Empi; do
    python scripts/rsl_rl/train.py \
        --task "Unitree-G1-29dof-Mimic-MartialArts-${task}" \
        --headless --num_envs 512
done
```

## 10.4 监控训练（TensorBoard）

```bash
tensorboard --logdir logs/rsl_rl/ --port 6006
```

### 关键指标

| 指标 | 健康范围 | 异常信号 |
|------|---------|---------|
| `Episode/mean_reward` | 持续上升 | 持续下降 → 奖励设计有问题 |
| `Episode/mean_length` | 接近 episode_length_s | 持续很短 → 终止太严格 |
| `Train/mean_std` | 逐渐减小 | 不减小 → 策略没收敛 |
| `error_anchor_pos` | 逐渐减小 | 增大 → 锚点跟踪失败 |
| `error_body_pos` | 逐渐减小 | 增大 → 身体跟踪失败 |
| `error_joint_pos` | 逐渐减小 | 增大 → 关节跟踪失败 |
| `sampling_entropy` | 0.5~0.9 | < 0.3 → 采样过度集中 |

### 训练曲线典型形态

```
reward
  ↑
  │           ╱─────── 收敛
  │         ╱
  │       ╱
  │     ╱
  │   ╱
  │ ╱
  │╱
  └────────────────── iterations
  0    5K   10K  15K  20K  25K  30K
```

- 前 5K：快速上升期（学会基本站立和粗略跟踪）
- 5K~15K：稳步提升期（精细化动作跟踪）
- 15K~30K：精修期（微调动作细节，收敛）

## 10.5 推理验证（Play）

```bash
# 可视化推理
python scripts/rsl_rl/play.py \
    --task Unitree-G1-29dof-Mimic-MartialArts-FrontKick

# 录制视频
python scripts/rsl_rl/play.py \
    --task Unitree-G1-29dof-Mimic-MartialArts-FrontKick \
    --video --video_length 500

# 指定 checkpoint
python scripts/rsl_rl/play.py \
    --task Unitree-G1-29dof-Mimic-MartialArts-FrontKick \
    --checkpoint /path/to/model_30000.pt
```

### 检查要点

✅ 动作整体流畅，与参考动作匹配
✅ 无关节反弯、胳膊拧弯
✅ 站立稳定，无膝盖/手肘触地
✅ 踢击/出拳到位，末端轨迹正确
✅ 过渡平滑，无突变跳动

❌ 如果看到问题 → 参见 [第十一部分：参数调优方法论](#第十一部分参数调优方法论)

## 10.6 导出与部署

训练完成后，`play.py` 会自动导出：
- `exported/policy.onnx`：ONNX 格式模型
- `exported/policy.pt`：JIT 格式模型

训练时 `train.py` 自动导出：
- `params/deploy.yaml`：部署参数

将这些文件复制到部署目录：
```
deploy/robots/g1_29dof/config/policy/mimic/martial_arts/<motion_name>/
├── exported/policy.onnx
└── params/
    ├── deploy.yaml
    └── G1_<motion_name>.csv
```

---

# 第十一部分：参数调优方法论

> 本章基于 v6→v9 的真实迭代经验，记录了调参过程中的具体教训。

## 11.1 奖励权重调优原则

### 原则 1：正奖励总和要远大于负奖励量级

```
如果 |负奖励| 接近 正奖励总和 → 策略学到 "不动最安全"
```

v9 的平衡：正奖励 ≈ 10.0，负正则化 ≈ 0.15（可忽略），接触惩罚 ≈ -1.0（显著但不压制）。

### 原则 2：σ 参数反映物理量级

| 跟踪类型 | 典型误差量级 | 推荐 σ 范围 |
|---------|-------------|------------|
| 位置 (m) | 0.1~0.5 | 0.3~0.5 |
| 朝向 (rad) | 0.2~0.8 | 0.4~0.5 |
| 关节角 (rad) | 0.3~1.0 | 0.5~0.8 |
| 线速度 (m/s) | 0.5~2.0 | 1.0~1.5 |
| 角速度 (rad/s) | 1.0~5.0 | 2.0~3.14 |

### 原则 3：权重代表优先级

- 最重要的奖励给最大权重
- v9 中 `joint_pos`(2.0) > `anchor_pos`(1.5) = `ee_pos`(1.5) > 其他(1.0)
- 这反映了 "关节角正确 > 位置到位 > 朝向/速度匹配" 的优先级

### 原则 4：先粗后细

1. 初始训练用较大的 σ（容易获得奖励，建立基本行为）
2. 观察基本行为正确后，逐步减小 σ 提升精度
3. 切勿一开始就用很小的 σ → 策略无法获得有效奖励信号 → 不收敛

## 11.2 终止阈值调优原则

### 原则 1：终止阈值 = "不可恢复" 的边界

不要用终止条件来 "塑造" 行为（那是奖励的工作）。终止条件应该标记 "机器人已经进入无法恢复的状态"。

### 原则 2：先松后紧

1. 初始用宽松的阈值训练
2. 观察策略行为后，根据需要收紧
3. 如果收紧后训练崩溃 → 说明收紧过度

### 原则 3：观察 episode 长度

- episode 平均长度 << episode_length_s → 终止太频繁 → 放宽阈值
- episode 平均长度 ≈ episode_length_s → 终止适中
- 从不终止 → 可能需要更严格的终止条件或奖励修改

## 11.3 常见问题诊断与解决

### 问题 1：膝盖着地 / 身体接触地面

**症状**：机器人跪在地上做动作

**原因分析**：
1. 无接触终止条件 → 策略发现跪着更稳定
2. 接触惩罚太弱 → 跟踪奖励 > 接触惩罚

**解决方案**：
- 添加 `illegal_contact` 终止（threshold=100N）
- 增大 `undesired_contacts` 权重到 -1.0
- 检查 ILLEGAL_CONTACT_BODIES 是否正确（注意 pelvis vs pelvis_contour_link）

### 问题 2：胳膊拧弯 / 关节配置错误

**症状**：身体位置大致正确，但关节角完全错误

**原因分析**：FK 多解性 — body_pos 奖励有多个局部最优，策略找到了错误的解

**解决方案**：
- 启用 `motion_joint_pos_error_exp` 奖励（weight ≥ 2.0）
- 这是最根本的修复，直接约束关节角

### 问题 3：训练崩溃 / episode 很短

**症状**：reward 持续下降，episode_length 很短

**原因分析**：终止条件过于严格 → 策略没有足够的探索空间

**解决方案**：
- 检查哪个终止条件触发最频繁（查看 TensorBoard 的 termination 日志）
- 放宽对应阈值
- v7→v8 的教训：anchor_pos 从 0.25→0.5m，anchor_ori 从 0.8→1.2

### 问题 4：动作抖动 / 不平滑

**症状**：关节快速振荡

**原因分析**：
1. action_rate 惩罚太小
2. NPZ 数据中 joint_vel 有噪声
3. 关节限位钳位不正确导致目标跳变

**解决方案**：
- 增大 action_rate 权重（但不要超过 0.5）
- 检查 NPZ 数据质量
- 确认 csv_to_npz.py 中的钳位逻辑正确

### 问题 5：策略忽略某些身体部位

**症状**：上半身或下半身跟踪很差

**原因分析**：body_pos/ori 是对 14 个刚体求平均 → 某些刚体的误差被平均稀释

**解决方案**：
- 增加 ee_pos 权重（突出末端执行器）
- 考虑为特定刚体组添加额外奖励项

## 11.4 版本演进经验总结（v6→v9）

| 版本 | 问题 | 修复 | 教训 |
|------|------|------|------|
| v6 | NPZ body 数据损坏 | 只用 joint_pos 训练 | 数据质量是一切的基础 |
| v7 | 训练崩溃（episode 太短）| 放宽终止阈值 | 终止条件不能太严格 |
| v8 | 膝盖着地、胳膊拧弯 | 移除 joint_pos（错误！）| 千万不要移除关键奖励 |
| v9 | 同 v8 + 正确修复 | 恢复 joint_pos + 添加 illegal_contact | 问题的根因是奖励缺失，不是终止缺失 |

**核心教训**：
1. **数据 > 奖励 > 终止**：数据质量最重要，其次是奖励设计，最后才是终止条件
2. **不要删除有效的奖励项**：即使暂时看不到效果，也不要贸然删除
3. **分层验证**：先验证数据（NPZ 回放）→ 再验证奖励（reward 曲线）→ 最后验证行为（play.py）
4. **记录每次改动**：在代码注释中写清楚版本历史和修改原因

---

# 第十二部分：代码文件索引

## 核心训练代码

| 文件 | 路径 | 说明 |
|------|------|------|
| `tracking_env_cfg.py` | `tasks/mimic/robots/g1_29dof/martial_arts/` | 环境配置（v9）|
| `g1.py` | `tasks/mimic/robots/g1_29dof/gangnanm_style/` | G1 机器人物理配置 |
| `commands.py` | `tasks/mimic/mdp/` | MotionCommand + MotionLoader |
| `rewards.py` | `tasks/mimic/mdp/` | 所有奖励函数 |
| `terminations.py` | `tasks/mimic/mdp/` | 自定义终止函数 |
| `observations.py` | `tasks/mimic/mdp/` | 观测函数 |
| `events.py` | `tasks/mimic/mdp/` | 域随机化事件 |
| `__init__.py` | `tasks/mimic/robots/g1_29dof/martial_arts/` | 7 个 gym.register |
| `rsl_rl_ppo_cfg.py` | `tasks/mimic/agents/` | PPO 超参数 |

## 脚本

| 文件 | 路径 | 说明 |
|------|------|------|
| `train.py` | `scripts/rsl_rl/` | 训练入口 |
| `play.py` | `scripts/rsl_rl/` | 推理/视频入口 |
| `csv_to_npz.py` | `scripts/mimic/` | CSV→NPZ 数据制备 |
| `cmu_amc_to_csv.py` | `scripts/mimic/` | CMU→CSV 数据转换 |
| `list_envs.py` | `scripts/` | 列出所有可用任务 |

## 部署代码

| 文件 | 路径 | 说明 |
|------|------|------|
| `State_MartialArtsSequencer.h` | `deploy/include/FSM/` | Policy Sequencer |
| `config.yaml` | `deploy/robots/g1_29dof/config/` | 部署 FSM 配置 |
| `export_deploy_cfg.py` | `utils/` | 导出部署参数 |

## 数据文件

| 文件格式 | 位置 | 说明 |
|---------|------|------|
| `G1_*.npz` | `tasks/.../martial_arts/` | 训练用参考动作 |
| `G1_*.csv` | 同上 | 原始 CSV 关节数据 |
| CMU ASF/AMC | `data/cmu_mocap/135/` | CMU 原始数据 |

---

# 附录

## A. SIM 关节顺序

Isaac Lab 中的关节索引顺序是**交错的左-右-腰**模式：

```
[0]  left_hip_pitch      [1]  right_hip_pitch     [2]  waist_yaw
[3]  left_hip_roll       [4]  right_hip_roll       [5]  waist_roll
[6]  left_hip_yaw        [7]  right_hip_yaw        [8]  waist_pitch
[9]  left_knee           [10] right_knee
[11] left_shoulder_pitch [12] right_shoulder_pitch
[13] left_ankle_pitch    [14] right_ankle_pitch
[15] left_shoulder_roll  [16] right_shoulder_roll
[17] left_ankle_roll     [18] right_ankle_roll
[19] left_shoulder_yaw   [20] right_shoulder_yaw
[21] left_elbow          [22] right_elbow
[23] left_wrist_roll     [24] right_wrist_roll
[25] left_wrist_pitch    [26] right_wrist_pitch
[27] left_wrist_yaw      [28] right_wrist_yaw
```

> ⚠️ SDK 关节顺序不同！CSV 使用 SDK 顺序，NPZ 使用 SIM 顺序。
> `csv_to_npz.py` 通过 `robot_joint_indexes` 做映射。

## B. 碰撞体列表

### 30 个 Articulation Body（PhysX 暴露的刚体）

这些是 `robot.body_names` 返回的全部刚体，也是 `body_names` 参数中唯一可用的名字：

```
pelvis,
left_hip_pitch_link, left_hip_roll_link, left_hip_yaw_link,
left_knee_link, left_ankle_pitch_link, left_ankle_roll_link,  ← 允许接触（脚）
right_hip_pitch_link, right_hip_roll_link, right_hip_yaw_link,
right_knee_link, right_ankle_pitch_link, right_ankle_roll_link,  ← 允许接触（脚）
waist_yaw_link, waist_roll_link,
torso_link,
left_shoulder_pitch_link, left_shoulder_roll_link, left_shoulder_yaw_link,
left_elbow_link, left_wrist_roll_link, left_wrist_pitch_link,
left_wrist_yaw_link,  ← 允许接触（手）
right_shoulder_pitch_link, right_shoulder_roll_link, right_shoulder_yaw_link,
right_elbow_link, right_wrist_roll_link, right_wrist_pitch_link,
right_wrist_yaw_link  ← 允许接触（手）
```

### 被 PhysX 合并的固定关节链接（不在 articulation body 列表中！）

这些链接在 URDF 中存在，但因为是固定关节，PhysX 将它们合并到父刚体中：

```
pelvis_contour_link → 合并到 pelvis
logo_link → 合并到 torso_link
head_link → 合并到 torso_link
waist_pitch_link → 合并到 waist_roll_link
imu_link, d435i_link, mid360_link → 合并到各自父链接
left_rubber_hand, right_rubber_hand → 合并到 wrist_yaw_link
```

> ⚠️ **切勿在 `body_names` 中使用这些名字！** 否则会导致运行时崩溃。

## C. 关键公式汇总

### 指数核奖励

$$r = \exp\left(-\frac{e^2}{\sigma^2}\right)$$

### PPO Clipped Objective

$$L^{\text{CLIP}} = \mathbb{E}\left[\min\left(r_t \hat{A}_t,\ \text{clip}(r_t, 1-\epsilon, 1+\epsilon) \hat{A}_t\right)\right]$$

### GAE

$$\hat{A}_t = \sum_{l=0}^{\infty} (\gamma\lambda)^l \delta_{t+l}, \quad \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

### Action Scale

$$\text{scale}_i = 0.25 \times \frac{E_i}{K_i}$$

### 自适应采样 EMA

$$f_{\text{new}} = \alpha \cdot f_{\text{current}} + (1-\alpha) \cdot f_{\text{old}}$$

### 角速度（SO3 导数）

$$\omega_t = \frac{\text{axis\_angle}(q_{t+1} \cdot q_{t-1}^{-1})}{2\Delta t}$$

### yaw-only 坐标变换

$$\Delta_{\text{ori}} = \text{yaw}(q_{\text{robot}} \cdot q_{\text{ref}}^{-1})$$

$$p_{\text{relative}} = \Delta_p + \Delta_{\text{ori}} \cdot (p_{\text{body}} - p_{\text{anchor}})$$

---

> **文档版本**：v9 | **最后更新**：2026-03
>
> **核心团队**：WuBot Project
