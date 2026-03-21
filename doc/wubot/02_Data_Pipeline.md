# 02 — 数据管线深度解析

> **阅读时间**: 15 分钟 · **难度**: ⭐⭐ · **前置要求**: 01_Project_Overview
>
> **学完你能**: 理解 NPZ 文件里每个字段的含义、知道 14 个追踪 Body 是谁、能验证数据质量

---

## 1. 完整数据流

```
CMU #135 (ASF+AMC, 120fps, Y-up, 角度制)
    │
    │  scripts/mimic/cmu_amc_to_csv.py
    │  ├── 骨骼定义解析: ASF → 骨骼链长度 + 层级关系
    │  ├── 坐标变换: CMU (X右,Y上,Z后) → Isaac (X前,Y左,Z上)
    │  ├── 身高标定: 人类 leg_chain → G1 骨盆高度 0.78m
    │  └── 关节映射: 30个CMU骨骼 → 29个G1关节角度
    ▼
G1_xxx.csv (每行36列: 3 root_pos + 4 root_quat + 29 joint_angles)
    │
    │  scripts/mimic/csv_to_npz.py (需要 Isaac Sim 运行)
    │  ├── 帧率降采样: 120fps → 50fps (线性插值)
    │  ├── Isaac Sim 回放: CSV 逐帧写入仿真关节 → 正运动学
    │  └── 录制输出: body 位置/朝向/速度 + joint 角度/速度
    ▼
G1_xxx.npz → MotionLoader 在训练时加载
    ▼
PPO 训练 (4096并行环境, 30000 iterations)
    ▼
policy.onnx → C++ 部署
```

### 1.1 为什么需要两步转换？

| 步骤 | 输入 | 输出 | 为什么不能跳过 |
|------|------|------|---------------|
| CMU→CSV | ASF+AMC (人类骨骼) | G1关节角 | 人类和机器人的骨骼完全不同，需要映射 |
| CSV→NPZ | 关节角度序列 | body 位置/速度 | 需要正运动学计算每个 link 的世界坐标，Isaac Sim 自动完成 |

> 💡 **关键理解**：CSV 只有关节角度，NPZ 多了 body 的位置/速度/朝向——这些是 reward 计算需要的参考值。

---

## 2. NPZ 文件内部结构

`MotionLoader.__init__()` 加载以下字段：

| 字段名 | 形状 | 单位 | 含义 |
|--------|------|------|------|
| `fps` | `[1]` | Hz | 帧率（固定 50） |
| `joint_pos` | `[T, 29]` | rad | 每帧的 29 个关节角度 |
| `joint_vel` | `[T, 29]` | rad/s | 每帧的 29 个关节速度 |
| `body_pos_w` | `[T, 30, 3]` | m | 每帧 30 个 body 的世界坐标 |
| `body_quat_w` | `[T, 30, 4]` | - | 每帧 30 个 body 的世界朝向（四元数 wxyz） |
| `body_lin_vel_w` | `[T, 30, 3]` | m/s | 每帧 30 个 body 的线速度 |
| `body_ang_vel_w` | `[T, 30, 3]` | rad/s | 每帧 30 个 body 的角速度 |

其中 `T` = 总帧数，`30` = G1 机器人的全部 body 数量（但训练只追踪其中 14 个）。

---

## 3. 14 个追踪 Body

训练时并不追踪全部 30 个 body，只追踪最重要的 14 个：

```
                    torso_link ← 锚点(anchor)
                        │
            ┌───────────┴───────────┐
    left_shoulder           right_shoulder
    _roll_link              _roll_link
        │                       │
    left_elbow              right_elbow
    _link                   _link
        │                       │
    left_wrist  ★           right_wrist  ★
    _yaw_link               _yaw_link
                        
                    pelvis ← 根节点
            ┌───────────┴───────────┐
    left_hip_roll           right_hip_roll
    _link                   _link
        │                       │
    left_knee               right_knee
    _link                   _link
        │                       │
    left_ankle  ★           right_ankle  ★
    _roll_link              _roll_link

    ★ = 末端执行器 (END_EFFECTOR_BODIES)
```

### 3.1 为什么是这 14 个？

- **骨盆 (pelvis)**: 根节点，全身运动的参考点
- **躯干 (torso_link)**: 锚点，用于计算锚点位置/朝向误差
- **髋/膝/踝 × 2**: 腿部运动链，控制站立和踢腿
- **肩/肘/腕 × 2**: 臂部运动链，控制出拳和格挡
- **4个末端执行器**: 手脚的精确位置，武术打击的核心

### 3.2 末端执行器为什么特别重要？

武术的核心是**打击到位**：拳头打到指定位置、脚踢到指定高度。所以 v5 新增了专门的末端执行器 reward：

```python
END_EFFECTOR_BODIES = [
    "left_ankle_roll_link",   # 左脚 — 踢腿目标
    "right_ankle_roll_link",  # 右脚 — 踢腿目标
    "left_wrist_yaw_link",    # 左拳 — 出拳目标
    "right_wrist_yaw_link",   # 右拳 — 出拳目标
]
```

---

## 4. 七套数据质量报告

所有 NPZ 文件经过速度尖峰修复（`scripts/mimic/fix_npz_velocity_spikes.py`），当前数据质量：

| 动作 | 帧数 | 时长 | joint_pos 范围 | joint_vel 最大值 | 状态 |
|------|------|------|---------------|-----------------|------|
| front_kick | 1145 | 22.9s | [-1.58, 2.00] | 28.9 rad/s | ✅ 干净 |
| heian_shodan | 548 | 11.0s | [-1.58, 2.60] | 27.1 rad/s | ✅ 干净 |
| roundhouse_kick | 1025 | 20.5s | [-2.34, 2.15] | 29.7 rad/s | ✅ 干净 |
| side_kick | 613 | 12.3s | [-1.76, 2.15] | 31.6 rad/s | ✅ 修复(3处) |
| lunge_punch | 1359 | 27.2s | [-1.69, 2.87] | 32.0 rad/s | ✅ 修复(35处) |
| bassai | 2548 | 51.0s | [-2.71, 2.67] | 32.0 rad/s | ✅ 修复(301处) |
| empi | 2168 | 43.4s | [-3.09, 2.71] | 32.0 rad/s | ✅ 修复(283处) |

### 4.1 速度尖峰是怎么来的？

CMU→G1 的关节映射中，IK（逆运动学）求解器偶尔会在相邻帧之间跳到不同的解（解翻转），导致关节角度突变 → 速度尖峰高达 150+ rad/s。

### 4.2 怎么验证数据质量？

```python
import numpy as np

f = np.load("G1_front_kick.npz")
jv = f['joint_vel']
print(f"max |vel|: {np.abs(jv).max():.1f} rad/s")  # 应 ≤ 32
print(f"spikes > 32: {(np.abs(jv) > 32).sum()}")    # 应 = 0
```

---

## 5. 数据在训练中的使用方式

`MotionCommand`（定义在 `mimic/mdp/commands.py`）在训练时做以下事情：

1. **加载** NPZ 到 GPU tensor
2. **采样起点**：每个 episode 从某一帧开始（自适应采样偏向困难片段）
3. **逐帧推进**：每个 step 前进到下一帧，提供参考值
4. **计算误差**：将参考值传给 reward 函数计算追踪误差

```
NPZ 数据:   [frame_0] [frame_1] [frame_2] ... [frame_T]
                                    ↑
                              当前参考帧
                                    │
                        ┌───────────┴───────────┐
                    参考关节角              参考body位置
                        │                       │
                    ↓ reward ↓              ↓ reward ↓
                joint_pos_error          body_pos_error
```

---

## 检查清单

- [ ] NPZ 中 `joint_pos` 和 `body_pos_w` 的区别是什么？
- [ ] 为什么训练只追踪 14 个 body 而不是全部 30 个？
- [ ] 4 个末端执行器分别对应机器人的哪个部位？
- [ ] 速度尖峰的根本原因是什么？

---

**下一步** → [03_Environment_Config.md](03_Environment_Config.md) 了解训练环境的物理仿真配置
