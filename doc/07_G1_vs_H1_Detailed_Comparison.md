# G1 vs H1：详细对比与选择指南

## 概览对比

| 特性 | Go2（四足） | H1（人形1.0） | G1-29DOF（人形2.0） |
|-----|-----------|-------------|------------------|
| **出厂年份** | 2023 | 2023 | 2024 |
| **身体形态** | 犬形 (4足) | 人形 (躯干+4肢+头) | 人形 (躯干+4肢+双臂) |
| **高度** | ~45 cm | ~110 cm | ~80 cm |
| **质量** | ~3 kg | ~47 kg | ~55 kg |
| **总自由度 (DOF)** | 12 | 20 | 29 |
| **腿部自由度** | 3×4 = 12 | 7×2 + 1 躯干 = 15 | 6×2 + 1 躯干 = 13 |
| **臂部自由度** | 0 | 3×2 = 6 | 5×2 = 10 |
| **其他DOF** | 0 | 头部(0) | 头部(0) |
| **典型应用** | 越野、速度 | 走路、基础操作 | 舞蹈、复杂动作 |
| **现存 RL 任务** | 1 (Velocity) | 1 (Velocity) | 4 (1 Velocity + 3 Mimic) |
| **学习难度** | ⭐ 简单 | ⭐⭐ 中等 | ⭐⭐⭐ 困难 |

---

## 详细参数对比

### 1. 机械结构

#### Go2
```
Front Legs (FL):        Rear Legs (RL):
  + Hip (Abduction)       + Hip (Abduction)
  + Thigh                 + Thigh
  + Calf                  + Calf
  (3 DOF per leg)         (3 DOF per leg)

Back Legs 类似 (FR, RR)

总计：4 腿 × 3 DOF = 12 DOF
```

#### H1
```
Left Leg:               Right Leg:
  + Hip (Roll/Pitch/Yaw)  + Hip (Roll/Pitch/Yaw)
  + Knee (Pitch)          + Knee (Pitch)
  + Ankle (Pitch/Roll)    + Ankle (Pitch/Roll)
  (7 DOF per leg)         (7 DOF per leg)

躯干：Yaw (1 DOF)

Arms: 无

总计：7+7+1 = 15 DOF (腿) + 0 (臂) + 待查头部 = 20 DOF
```

#### G1-29DOF
```
Left Leg:               Right Leg:
  + Hip (Roll/Pitch/Yaw)  + Hip (Roll/Pitch/Yaw)
  + Knee (Pitch)          + Knee (Pitch)
  + Ankle (Pitch/Roll)    + Ankle (Pitch/Roll)
  (6 DOF per leg)         (6 DOF per leg)

躯干：Yaw (1 DOF)

Left Arm:               Right Arm:
  + Shoulder (P/R/Y)      + Shoulder (P/R/Y)
  + Elbow (Pitch)         + Elbow (Pitch)
  + Wrist (Roll)          + Wrist (Roll)
  (5 DOF per arm)         (5 DOF per arm)

总计：6+6+1+5+5 = 23 DOF 基础，还有手腕/手指等 = 29 DOF
```

### 2. 执行器（电动机）参数

从 `source/unitree_rl_lab/assets/robots/unitree.py` 提取：

#### Go2 (GO2HV)
```python
GO2HV:
  effort_limit: 23.5 Nm      # 最大扭矩
  velocity_limit: 30.0 rad/s  # 最大角速度
  stiffness: 25.0 Nm/rad      # 关节刚度（越大越硬）
  damping: 0.5 Nm/(rad/s)     # 阻尼（越大摆动越少）
```

#### H1 (混合型)
```python
GO2HV-1 (脚踝、肩膀):
  effort_limit: 40 Nm
  velocity_limit: 9 rad/s
  stiffness: 40-100 Nm/rad
  damping: 2.0 Nm/(rad/s)

GO2HV-2 (肘部、肩膀Yaw):
  effort_limit: 18 Nm
  velocity_limit: 20 rad/s
  stiffness: 50 Nm/rad
  damping: 2.0 Nm/(rad/s)

M107-24-1 (膝盖 - 强力):
  effort_limit: 300 Nm
  velocity_limit: 14 rad/s
  stiffness: 200 Nm/rad
  damping: 4.0 Nm/(rad/s)

M107-24-2 (臀部、躯干 - 超强力):
  effort_limit: 200 Nm
  velocity_limit: 23 rad/s
  stiffness: 150-300 Nm/rad
  damping: 2-6 Nm/(rad/s)
```

#### G1-29DOF (混合型)
```python
N7520-14.3 (髋部Pitch、Yaw、腰部):
  effort_limit_sim: 88 Nm
  velocity_limit_sim: 32 rad/s
  stiffness: 100-200 Nm/rad
  damping: 2-5 Nm/(rad/s)

N7520-22.5 (髋部Roll、膝盖):
  effort_limit_sim: 139 Nm
  velocity_limit_sim: 20 rad/s
  stiffness: 100-150 Nm/rad
  damping: 2-4 Nm/(rad/s)

N5020-16 (肩膀、肘部、腕部):
  effort_limit_sim: 25 Nm
  velocity_limit_sim: 37 rad/s
  stiffness: 40 Nm/rad
  damping: 1 Nm/(rad/s)

N5020-16-parallel (脚踝):
  effort_limit_sim: 35 Nm
  velocity_limit_sim: 30 rad/s
  stiffness: 40 Nm/rad
  damping: 2 Nm/(rad/s)
```

### 3. 参数解释

**Effort Limit (最大扭矩)**
- 单位：Nm (牛顿米)
- 含义：电动机能提供的最大力量
- Go2: ~24 Nm → 轻便
- H1: 40-300 Nm → 强力多样
- G1: 25-139 Nm → 中等强力

**Velocity Limit (最大角速度)**
- 单位：rad/s
- 含义：关节最快能转的速度
- Go2: 30 rad/s → 中等快
- H1: 9-23 rad/s → 较慢（为了力量）
- G1: 20-37 rad/s → 快（特别是手臂）

**Stiffness (刚度)**
- 单位：Nm/rad
- 含义：给定偏差，关节能产生多大的对抗力量
- 高刚度：硬、反应快、耗能大、容易过度补偿
- 低刚度：软、反应慢、耗能小、灵活
- Go2: 25 Nm/rad → 软，易弯曲
- H1: 40-200 Nm/rad → 硬，稳定
- G1: 40-200 Nm/rad → 与 H1 类似

**Damping (阻尼)**
- 单位：Nm/(rad/s)
- 含义：速度相关的摩擦力（防止摆动）
- 高阻尼：摆动少、反应迟钝
- 低阻尼：灵活、容易振荡
- Go2: 0.5-0.01 → 很软
- H1: 2-6 → 硬
- G1: 1-5 → 与 H1 类似

---

## 代码结构对比

### 文件组织

所有三个机器人遵循同样的模式：

```
tasks/locomotion/
└── robots/
    ├── go2/
    │   └── velocity_env_cfg.py
    ├── h1/
    │   └── velocity_env_cfg.py
    └── g1/
        └── 29dof/
            └── velocity_env_cfg.py
```

### 配置文件结构完全相同

每个 `velocity_env_cfg.py` 都定义：

```python
@configclass
class RobotSceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(...)  # 地形定义
    robot = ROBOT_CFG.replace(...)     # 机器人模型

@configclass
class EventCfg:
    physics_material = EventTerm(...)  # 环境随机化

@configclass
class ObservationsCfg:
    obs_policy = ObsGroup(...)         # 观测空间

@configclass
class RewardsCfg:
    track_lin_vel_xy_exp = RewTerm()   # 各种奖励

@configclass
class TerminationsCfg:
    base_height = DoneTerm(...)        # 终止条件

@configclass
class RobotEnvCfg(ManagerBasedRLEnvCfg):
    # 把上面的组装起来
```

**区别**：
- 只有 `ROBOT_CFG` 不同（指向不同的机器人）
- 所有其他部分（奖励函数、观测等）逻辑相同
- 但实际数值可能需要微调（如终止条件的高度）

---

## 任务对比

### Locomotion (走路)

| 机器人 | 任务数 | 任务名 | 难度 | 备注 |
|------|------|------|------|-----|
| Go2 | 1 | Velocity | ⭐ | 最快学习 |
| H1 | 1 | Velocity | ⭐⭐ | 人形，但无臂 |
| G1 | 1 | Velocity | ⭐⭐⭐ | 最复杂 |

所有三个任务本质相同：学习在命令速度下走路

### Mimic (模仿)

| 机器人 | 任务数 | 任务名 | 数据来源 | 难度 |
|------|------|------|--------|------|
| Go2 | 0 | N/A | N/A | N/A |
| H1 | 0 | N/A | N/A | N/A |
| G1 | 3 | Dance-102, Gangnam Style, Petite Verses | 人类动作捕捉 | ⭐⭐⭐ |

为什么只有 G1？
- Go2 是四足，人类舞蹈数据不适用
- H1 没有双臂，难以复制全身舞蹈
- G1 有双臂，能更准确地模仿人类动作

---

## 选择机器人的决策树

### 场景 1：我是完全新手，想快速上手

**选择：Go2 Velocity**

原因：
- 只有 12 DOF，最简单
- 环境配置最少，最容易理解
- 训练速度最快（几分钟出结果）
- 足以理解整个 RL 框架

建议：
```bash
# 快速验证环境是否安装正确
python scripts/rsl_rl/train.py --task Unitree-Go2-Velocity --headless --max_iterations=100

# 如果 reward 在上升，说明一切正常
# 然后再考虑学习其他机器人
```

---

### 场景 2：我想学习人形机器人的走路

**选择：H1 Velocity**

原因：
- 人形结构（20 DOF），比 Go2 更接近真实机器人
- 比 G1 简单（无臂）
- 有躯干旋转，能学习更复杂的平衡

建议：
```bash
# H1 训练会比 Go2 慢，但概念相同
python scripts/rsl_rl/train.py --task Unitree-H1-Velocity --headless

# 修改奖励函数时，可能需要微调权重
# （因为 H1 比 Go2 更容易摔倒）
```

重点关注：
- 躯干的旋转如何影响走路
- 修改 `stiffness` 参数看稳定性变化

---

### 场景 3：我想学习动作模仿（舞蹈）

**选择：G1-29DOF Mimic**

原因：
- 唯一有舞蹈数据的机器人
- 具有完整的身体结构（腿 + 双臂）
- 最有趣（能看到机器人跳舞）

建议：
```bash
# 选择一个舞蹈（建议先试 Gangnam Style）
python scripts/rsl_rl/train.py --task Unitree-G1-29dof-Mimic-Gangnanm-Style --headless

# 这会花较长时间（Go2 Velocity 的 3-5 倍）
# 但效果最酷！
```

重点关注：
- 观测中的"参考关节位置"
- 奖励函数中的"位置追踪误差"和"速度追踪误差"
- `.npz` 文件中的参考动作如何加载

---

### 场景 4：我想对比三个机器人的训练效率

**建议的实验设计**：

```bash
# 1. Go2 Velocity - 基线
time python scripts/rsl_rl/train.py --task Unitree-Go2-Velocity --headless --max_iterations=500

# 2. H1 Velocity - 看是否更复杂
time python scripts/rsl_rl/train.py --task Unitree-H1-Velocity --headless --max_iterations=500

# 3. G1 Velocity - 最复杂的走路
time python scripts/rsl_rl/train.py --task Unitree-G1-29dof-Velocity --headless --max_iterations=500
```

预期结果：
- Go2: 耗时 ~5 分钟，Reward 快速上升
- H1: 耗时 ~15 分钟，Reward 上升较慢
- G1: 耗时 ~30 分钟，Reward 上升最慢

**为什么？**
- DOF 越多 → 状态空间越大 → 需要更多样本学习
- 状态空间大 → 同样的训练迭代需要更多计算

---

## 改装指南：从一个机器人改为另一个

### Go2 → H1 (添加上身和躯干)

**你需要改什么**：

1. **配置文件**
   ```python
   # 原来
   from unitree_rl_lab.assets.robots.unitree import UNITREE_GO2_CFG as ROBOT_CFG
   
   # 改为
   from unitree_rl_lab.assets.robots.unitree import UNITREE_H1_CFG as ROBOT_CFG
   ```

2. **观测空间** - 可能需要调整
   ```python
   # H1 的关节更多，可能需要选择性地观测
   # 比如只关注腿部，忽略臂部
   ```

3. **初始姿态** - H1 需要不同的初始位置
   ```python
   init_state = ArticulationCfg.InitialStateCfg(
       pos=(0.0, 0.0, 1.1),    # H1 更高
       joint_pos={
           ".*_hip_pitch_joint": -0.1,
           ".*_knee_joint": 0.3,  # H1 膝盖配置不同
           ".*_ankle_joint": -0.2,
       },
   )
   ```

4. **终止条件** - H1 更高，摔倒的高度阈值不同
   ```python
   base_height = DoneTerm(
       func=mdp.base_height_below_threshold,
       params={"threshold": 0.5},  # 从 0.2 改为 0.5（因为 H1 更高）
   )
   ```

5. **重新训练并调整**
   ```bash
   python scripts/rsl_rl/train.py --task Unitree-H1-Velocity --headless --max_iterations=500
   ```

### H1 → G1-29DOF (仅腿部)

基本类似，但：
- G1 的腿部结构与 H1 略有不同（6 DOF vs 7 DOF）
- 需要调整关节名称和初始姿态

### 直接创建新任务（高级）

复制 `tasks/locomotion/robots/go2/` 整个目录为新名字，修改：
1. 导入的 `ROBOT_CFG`
2. 场景配置（地形、传感器）
3. 奖励和观测（可选）
4. 在 `__init__.py` 中注册

---

## 性能基准 (Benchmarks)

### 训练速度

假设在单 GPU (RTX 4090) 上训练 1000 iterations：

| 机器人 | 环境并行数 | 预计时间 | Reward 收敛 |
|------|---------|--------|----------|
| Go2 | 4096 | 10 min | 500 iter |
| H1 | 2048 | 30 min | 800 iter |
| G1 (Velocity) | 1024 | 60 min | 1000 iter |
| G1 (Mimic) | 512 | 120 min | 1500 iter |

### 推理性能

在 C++ 部署中的实时性：

| 机器人 | 神经网络大小 | 推理延迟 | 是否实时 (50 Hz) |
|------|-----------|--------|--------------|
| Go2 | ~100 KB | <2 ms | ✓ |
| H1 | ~200 KB | <5 ms | ✓ |
| G1 | ~300 KB | <10 ms | ✓ |

所有机器人都可以实时运行（满足 20 Hz 的机器人控制频率）

---

## 常见问题

### Q: G1 和 H1 哪个更"人形"？

**A:** 从人形程度上：
- **H1**：腿部更接近人类（膝盖、脚踝）
- **G1**：身体比例更接近人类，有双臂

从运动能力上：
- **H1**：可能更稳定（更重）
- **G1**：可能更灵活（更轻、有臂）

### Q: 为什么 G1 有两个版本 (23DOF 和 29DOF)？

**A:** 
- **G1-23DOF**：较轻，用于快速迭代
- **G1-29DOF**：完整版，多了手指/手腕关节，能做更复杂的舞蹈

项目中主要用 29DOF（有更多任务）

### Q: 我能混合 Go2 的训练设置和 H1 的机器人吗？

**A:** 理论上可以，但需要小心：
- 观测维度必须匹配
- 关节名称必须兼容
- 初始姿态需要调整

建议：先完全复制一个现有配置，再做最小改动

### Q: 两个机器人训练的模型能互换吗？

**A:** 不能！

原因：
- 神经网络的输入层大小由观测维度决定
- Go2 的观测维度 ≠ H1 的观测维度
- 输出层大小由关节数决定
- Go2: 12 关节 vs H1: 20 关节

### Q: G1 Velocity 和 G1 Mimic 有什么区别？

**A:**

| 维度 | Velocity | Mimic |
|-----|----------|-------|
| 目标 | 跟踪速度命令 | 模仿人类舞蹈 |
| 参考数据 | 无（随机生成命令） | 有（.npz 动作文件） |
| 关键奖励 | 速度追踪 | 关节位置/速度追踪 |
| 训练时间 | 较短 | 较长 |
| 实用性 | 部署到真机 | 展示酷炫效果 |

---

## 建议的学习路径

```
Week 1:
  Day 1-2: 运行 Go2 Velocity，理解框架
  Day 3: 修改 Go2 的奖励函数，观察变化
  Day 4-5: 阅读 velocity_env_cfg.py 代码

Week 2:
  Day 1-2: 切换到 H1 Velocity，对比
  Day 3: 分析 H1 vs Go2 的配置差异
  Day 4: 调整 H1 的参数以改进训练

Week 3:
  Day 1-2: 尝试 G1-29DOF-Velocity
  Day 3: 读懂 G1 的完整配置
  Day 4-5: 尝试 G1 Mimic 任务（选一个舞蹈）

Week 4:
  选一个感兴趣的方向深入：
  - 修改奖励函数
  - 创建新任务
  - 部署到 C++ / 真机
```

---

## 总结

| 我想做 | 选择 | 耗时 | 难度 |
|------|-----|------|------|
| 快速验证环境 | Go2 Velocity | 10 min | ⭐ |
| 学习基础 RL | Go2 Velocity | 1-2 h | ⭐ |
| 学习人形机器人 | H1 Velocity | 2-3 h | ⭐⭐ |
| 学习动作模仿 | G1 Mimic | 4-6 h | ⭐⭐⭐ |
| 研究舞蹈生成 | G1 Mimic (all) | 10+ h | ⭐⭐⭐ |
| 部署到真机 | 先用 Go2, 再试 H1/G1 | 2-3 周 | ⭐⭐⭐⭐ |
