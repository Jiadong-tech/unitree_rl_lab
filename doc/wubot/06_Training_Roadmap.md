# 06 — 训练路线图

> **阅读时间**: 15 分钟 · **难度**: ⭐⭐ · **前置要求**: 01_Project_Overview
>
> **学完你能**: 知道 7 个动作该按什么顺序训练、理解动作间过渡问题和解决方案、规划完整的训练计划

---

## 1. 四阶段路线图

```
阶段1 (当前)          阶段2              阶段3              阶段4
━━━━━━━━━━          ━━━━━━━━━━        ━━━━━━━━━━        ━━━━━━━━━━
单个动作精调         扩展武器库          Sim2Sim验证        真机部署
                   训练其他6个动作     Mujoco交叉验证     吊架→落地
                   调整过渡参数        排查接触力问题     连续表演
```

---

## 2. 阶段 1：单个动作精调

以 FrontKick 为标杆，反复调参直到满意。

**检查清单**:
- [ ] 踢腿高度达到参考动作的 80% 以上？
- [ ] 上半身保持相对直立（不佝偻）？
- [ ] 收腿后能稳定站立（不倾倒）？
- [ ] `push_robot` 推扰下仍能完成动作？
- [ ] 手臂保持合理姿势（不乱挥）？

**训练命令**:
```bash
# RTX 4070 Ti (12GB) 建议 num_envs=512
python scripts/rsl_rl/train.py \
    --task Unitree-G1-29dof-Mimic-MartialArts-FrontKick \
    --headless --num_envs 512
```

---

## 3. 阶段 2：扩展武器库

### 3.1 推荐训练顺序（按难度递增）

| 顺序 | 动作 | 难度 | 时长 | 特殊注意 |
|------|------|------|------|---------|
| 1 | **FrontKick** 正踢 | ⭐⭐ | 22.9s | 标杆动作，先调好 |
| 2 | **LungePunch** 冲拳 | ⭐⭐ | 27.2s | 主要是手臂动作，较容易 |
| 3 | **SideKick** 侧踢 | ⭐⭐⭐ | 12.3s | 腿侧向运动，可能需放宽 ee 阈值 |
| 4 | **RoundhouseKick** 回旋踢 | ⭐⭐⭐ | 20.5s | 旋转+踢腿组合 |
| 5 | **HeianShodan** 平安初段 | ⭐⭐⭐⭐ | 11.0s | 完整套路，大量方向变换 |
| 6 | **Bassai** 拔塞 | ⭐⭐⭐⭐⭐ | 51.0s | 长套路，可能需 50000+ iterations |
| 7 | **Empi** 燕飞 | ⭐⭐⭐⭐⭐ | 43.4s | 长套路，包含跳跃动作 |

### 3.2 训练命令

```bash
# ⭐⭐ 简单动作 (先练)
python scripts/rsl_rl/train.py --task Unitree-G1-29dof-Mimic-MartialArts-LungePunch --headless --num_envs 512

# ⭐⭐⭐ 中等动作
python scripts/rsl_rl/train.py --task Unitree-G1-29dof-Mimic-MartialArts-SideKick --headless --num_envs 512
python scripts/rsl_rl/train.py --task Unitree-G1-29dof-Mimic-MartialArts-RoundhouseKick --headless --num_envs 512

# ⭐⭐⭐⭐ 复杂套路
python scripts/rsl_rl/train.py --task Unitree-G1-29dof-Mimic-MartialArts-HeianShodan --headless --num_envs 512

# ⭐⭐⭐⭐⭐ 长套路 (可能需要 50000+ iterations)
python scripts/rsl_rl/train.py --task Unitree-G1-29dof-Mimic-MartialArts-Bassai --headless --num_envs 512
python scripts/rsl_rl/train.py --task Unitree-G1-29dof-Mimic-MartialArts-Empi --headless --num_envs 512
```

### 3.3 动作特殊调整建议

| 动作 | 可能需要的调整 |
|------|--------------|
| SideKick / RoundhouseKick | `ee_body_pos` 终止阈值放宽到 0.8m（腿侧面运动更极端） |
| HeianShodan | `anchor_ori` 阈值可能需放宽（有大幅转身） |
| Bassai / Empi | `episode_length_s` 已在配置中设为 50~55s；可能需 50000+ iterations |

---

## 4. 阶段 3：动作串联

### 4.1 过渡问题

**核心难题**：冲拳结束时的姿态 ≠ 正踢的标准起手式。

```
冲拳结束:                    正踢起手:
   ▯ (前倾, 右拳伸出)          ▯ (直立, 双手收回)
  /|\                         /|\
  / \                         / \
  完全不同的关节配置!
```

如果直接硬切换，机器人会因为关节突变而摔倒。

### 4.2 双重保险解决方案

**保险 1：C++ 部署层 — 平滑插值过渡**

`State_MartialArtsSequencer.h` 中已实现关节级线性插值：

```
段落A结束 → 保存末尾关节角 (start_q)
               ↓
加载段落B → 获取起手第一帧关节角 (target_q)
               ↓
过渡时间窗口内 → alpha 从 0→1 线性插值
               ↓
          start_q + alpha × (target_q - start_q)
               ↓
          平滑到达 → 段落B正式开始
```

**保险 2：训练层 — 鲁棒性注入**

训练时加大初始状态随机噪声，迫使策略学会从偏离的初始姿态中纠偏：

```python
# _make_command_cfg 中的初始扰动
pose_range = {
    "roll": (-0.1, 0.1),      # 起始姿态扰动
    "pitch": (-0.1, 0.1),
    "yaw": (-0.2, 0.2),
}
joint_position_range = (-0.1, 0.1)  # 关节角度扰动
```

> 💡 **进阶**：如果过渡效果不好，可以增大 `joint_position_range` 到 ±0.3，让策略更能适应初始偏差。

---

## 5. GPU 内存与训练时间

### 5.1 不同 num_envs 的内存需求

| num_envs | 估计显存 | 适用 GPU | 训练速度 |
|----------|---------|---------|---------|
| 4096 (默认) | ~16 GB | RTX 4090 / A100 | 最快 (~3h/30000iter) |
| 2048 | ~10 GB | RTX 3090 / 4080 | 快 (~5h) |
| 1024 | ~7 GB | RTX 3080 | 中 (~8h) |
| 512 | ~5 GB | **RTX 4070 Ti 12GB** | 慢 (~12h) |
| 256 | ~3 GB | RTX 3060 | 很慢 (~20h) |

### 5.2 训练时间估算

| 动作 | 帧数 | 建议iterations | 512 envs 估计时间 |
|------|------|---------------|-----------------|
| FrontKick | 1145 | 30000 | ~12h |
| LungePunch | 1359 | 30000 | ~12h |
| SideKick | 613 | 30000 | ~10h |
| RoundhouseKick | 1025 | 30000 | ~12h |
| HeianShodan | 548 | 30000 | ~10h |
| Bassai | 2548 | **50000** | ~20h |
| Empi | 2168 | **50000** | ~18h |

---

## 6. 恢复训练

如果训练中断，可以从最新 checkpoint 恢复：

```bash
python scripts/rsl_rl/train.py \
    --task Unitree-G1-29dof-Mimic-MartialArts-FrontKick \
    --headless --num_envs 512 \
    --resume --checkpoint logs/rsl_rl/unitree_g1_29dof_mimic_martialarts_frontkick/<run>/model_XXXX.pt
```

---

## 检查清单

- [ ] 7 个动作中应该先训练哪个？
- [ ] SideKick 可能需要什么特殊调整？
- [ ] 动作间过渡的两层保险分别是什么？
- [ ] RTX 4070 Ti 12GB 应该用多少 num_envs？

---

**下一步** → [07_Deployment_Guide.md](07_Deployment_Guide.md) 了解如何将训练好的策略部署到真机
