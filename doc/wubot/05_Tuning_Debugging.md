# 05 — 调参与问题诊断

> **阅读时间**: 15 分钟 · **难度**: ⭐⭐⭐ · **前置要求**: 04_Reward_Design
>
> **学完你能**: 看到训练效果不好时快速定位原因、用 TensorBoard 读懂关键指标、按决策树找到该调哪个参数

---

## 1. 问题诊断决策树

看到机器人动作不对？按这个流程走：

```
机器人动作不对
    │
    ├── 完全不动 / 只站着 ──→ 问题 E: 奖励设计问题
    │
    ├── 动了但幅度不够 ──→ 问题 D: 踢腿/出拳不到位
    │
    ├── 动了但姿势怪异 ──→ 问题 A: 手臂乱挥 / 关节扭曲
    │                  └─→ 问题 B: 躯干佝偻
    │
    ├── 动作正确但抖动 ──→ 问题 C: 不够爆发 / 抖动
    │
    └── 训练中途频繁重置 ──→ 问题 F: 终止条件太严
```

---

## 2. 六大问题诊断卡

### 问题 A: 手臂乱挥 / 关节扭曲

**现象**: 腿在做正确动作，但手臂在乱晃或呈不自然角度

**根因**: `joint_pos` 的 σ 太宽松，手臂因力矩小（effort_limit=25Nm）恢复慢

**TensorBoard 信号**: `error_joint_pos` 不再下降，或上半身 error 明显高于下半身

**修复** (v5 已应用):
```python
# tracking_env_cfg.py → MartialArtsRewardsCfg
motion_joint_pos = RewTerm(
    weight=3.0,    # ↑ 从 2.0（加强约束）
    params={"std": 0.6},  # ↓ 从 0.8（更严格）
)
```

**进阶方案**: 拆分上下半身，给不同权重:
- 上半身 (waist/shoulder/elbow/wrist): weight=1.5, std=0.5
- 下半身 (hip/knee/ankle): weight=2.5, std=0.8

---

### 问题 B: 躯干佝偻 / 过度前倾

**现象**: 踢腿时弯腰驼背，像老人而不是武术选手

**根因**: 策略为保持平衡倾向弯腰，`anchor_ori` 约束太弱

**TensorBoard 信号**: `error_anchor_rot` > 0.5 且不再下降

**修复** (v5 已应用):
```python
motion_global_anchor_ori = RewTerm(
    weight=1.0,    # ↑ 从 0.5（翻倍）
    params={"std": 0.5},  # ↓ 从 0.8（更严格）
)
```

---

### 问题 C: 动作不够爆发 / 出现抖动

**现象**: 出拳/踢腿软绵绵，或动作过程中出现细微抖动

**根因**: `action_rate_l2` 惩罚太强，压制了快速动作变化

**TensorBoard 信号**: action 的方差很小，或 action_rate reward 过度主导

**修复** (v5 已应用):
```python
action_rate_l2 = RewTerm(weight=-0.05)  # ↓ 从 -0.1（减半，解锁爆发力）
```

---

### 问题 D: 踢腿/出拳幅度不够

**现象**: 正踢高度不到位，拳头没有伸展到位

**根因**: 全身 `body_pos` 对手脚的约束力不够（手脚只占 14 个 body 中的 4 个）

**TensorBoard 信号**: `error_body_pos` 下降但末端执行器 error 仍然高

**修复** (v5 已应用):
```python
# 新增末端执行器专项追踪
motion_ee_pos = RewTerm(
    weight=3.0,     # 极高权重
    params={"std": 0.2, "body_names": END_EFFECTOR_BODIES},  # 极严格
)
motion_ee_lin_vel = RewTerm(
    weight=1.0,
    params={"std": 1.0, "body_names": END_EFFECTOR_BODIES},
)
```

---

### 问题 E: 机器人完全不动 / 只站着

**现象**: 训练了很多轮，机器人还是站在原地不动或微微晃

**可能原因** (按概率排序):

1. **σ 太小** → 初始 reward ≈ 0 → 梯度消失
   - 检查: TensorBoard 中所有追踪 reward 是否接近 0
   - 修复: 增大 σ 值

2. **终止条件太严** → episode 太短，学不到东西
   - 检查: `episode_length` 是否远小于预期
   - 修复: 放宽终止条件阈值

3. **action_rate 惩罚太大** → "不动"是最安全的选择
   - 检查: action_rate reward 是否占总 reward 的大比例
   - 修复: 降低 action_rate 权重

4. **数据质量问题** → NPZ 数据有速度尖峰
   - 检查: `max(|joint_vel|)` 是否 > 50 rad/s
   - 修复: 运行 `fix_npz_velocity_spikes.py`

---

### 问题 F: 训练中频繁重置

**现象**: episode 平均长度很短（< 5秒），策略没有足够时间学习

**可能原因**:

1. **`ee_body_pos` 阈值太严** → 踢腿时被终止
   - 检查: 放宽到 0.6~0.8m
   
2. **`anchor_pos` 阈值太严** → 质心偏移时被终止
   - 检查: 当前 0.25m，对于武术可能需要放宽

3. **`anchor_ori` 阈值太严** → 转身时被终止
   - 检查: 对于有大幅旋转的动作（HeianShodan），0.8 可能太严

**TensorBoard 信号**: `episode_length` 指标远小于 `episode_length_s`

---

## 3. TensorBoard 关键指标速查

```bash
tensorboard --logdir logs/rsl_rl/unitree_g1_29dof_mimic_martialarts_frontkick
```

| 指标名 | 理想趋势 | 异常信号 | 对应问题 |
|--------|---------|---------|---------|
| `reward/total` | 持续上升 → 收敛 | 下降或不动 | 奖励冲突 / σ太小 |
| `error_joint_pos` | 持续下降 | 不降 | joint_pos 权重太低 (问题A) |
| `error_body_pos` | 持续下降 | 不降 | body_pos σ太小 (问题E) |
| `error_anchor_rot` | 下降到 <0.3 | >0.5 | anchor_ori 权重太低 (问题B) |
| `episode_length` | 接近 30s | <5s | 终止条件太严 (问题F) |
| `sampling_entropy` | 先降后稳 | 趋近0 | 过拟合某段 |
| `sampling_top1_prob` | <0.3 | >0.5 | 反复在同一段失败 |

---

## 4. 调参工作流

### 4.1 标准 4 步流程

```bash
# Step 1: 短训练验证趋势 (5000 iterations, ~30分钟)
python scripts/rsl_rl/train.py \
    --task Unitree-G1-29dof-Mimic-MartialArts-FrontKick \
    --headless --max_iterations 5000 --num_envs 512

# Step 2: TensorBoard 看曲线
tensorboard --logdir logs/rsl_rl/unitree_g1_29dof_mimic_martialarts_frontkick

# Step 3: 可视化播放
python scripts/rsl_rl/play.py \
    --task Unitree-G1-29dof-Mimic-MartialArts-FrontKick

# Step 4: 录制视频对比
python scripts/rsl_rl/play.py \
    --task Unitree-G1-29dof-Mimic-MartialArts-FrontKick --video
```

### 4.2 调参原则

1. **一次只改一个参数** — 否则无法判断是哪个参数的效果
2. **先跑 5000 轮看趋势** — 不需要跑完 30000 轮才能判断方向
3. **看 TensorBoard 而不是看视频** — 数据比视觉更可靠
4. **改权重优先，改 σ 其次** — 权重控制"优先级"，σ 控制"灵敏度"

---

## 5. v5 调参汇总表

| 奖励项 | v3→v4 | v4→v5 | 当前值 | 修改原因 |
|--------|-------|-------|--------|---------|
| `action_rate` | -0.1 | **-0.05** | -0.05 | 解锁爆发力 |
| `anchor_ori` w | 0.5 | **1.0** | 1.0 | 防佝偻 |
| `anchor_ori` σ | 0.8 | **0.5** | 0.5 | 更严格的直立约束 |
| `joint_pos` w | 新增2.0 | **3.0** | 3.0 | 消除怪异姿势 |
| `joint_pos` σ | 0.8 | **0.6** | 0.6 | 更严格的关节约束 |
| `ee_pos` | — | **新增** | w=3.0, σ=0.2 | 手脚精准打击 |
| `ee_lin_vel` | — | **新增** | w=1.0, σ=1.0 | 打击利落度 |
| `ee_body_pos` 终止 | 0.25→**0.6** | 0.6 | 0.6m | 允许踢腿 |

---

## 检查清单

- [ ] 看到"手臂乱挥"应该调哪个参数？
- [ ] TensorBoard 中 `episode_length` 很短意味着什么？
- [ ] 为什么要先跑 5000 轮而不是直接跑 30000 轮？
- [ ] v5 相比 v4 主要改了哪三个东西？

---

**下一步** → [06_Training_Roadmap.md](06_Training_Roadmap.md) 了解 7 个动作的训练顺序和串联方案
