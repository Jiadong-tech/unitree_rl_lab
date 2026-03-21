# 08 — 快速参考卡

> **用法**: 打印贴在显示器旁边，需要时查阅 · **难度**: ⭐
>
> 不需要阅读，直接**查**。

---

## 1. 常用命令

```bash
# ═══════════ 安装 ═══════════
./unitree_rl_lab.sh -i                # 安装（editable mode）

# ═══════════ 查看任务 ═══════════
./unitree_rl_lab.sh -l | grep Martial  # 列出武术任务

# ═══════════ 训练 ═══════════
python scripts/rsl_rl/train.py \
    --task Unitree-G1-29dof-Mimic-MartialArts-FrontKick \
    --headless --num_envs 512          # RTX 4070 Ti 12GB 建议

# ═══════════ 恢复训练 ═══════════
python scripts/rsl_rl/train.py \
    --task Unitree-G1-29dof-Mimic-MartialArts-FrontKick \
    --headless --num_envs 512 \
    --resume --checkpoint logs/rsl_rl/.../model_XXXX.pt

# ═══════════ 播放 ═══════════
python scripts/rsl_rl/play.py \
    --task Unitree-G1-29dof-Mimic-MartialArts-FrontKick

# ═══════════ 录制视频 ═══════════
python scripts/rsl_rl/play.py \
    --task Unitree-G1-29dof-Mimic-MartialArts-FrontKick --video

# ═══════════ TensorBoard ═══════════
tensorboard --logdir logs/rsl_rl/unitree_g1_29dof_mimic_martialarts_frontkick

# ═══════════ 验证 NPZ 数据 ═══════════
python -c "
import numpy as np
f = np.load('path/to/G1_xxx.npz')
jv = f['joint_vel']
print(f'max |vel|: {abs(jv).max():.1f}')
print(f'spikes>32: {(abs(jv)>32).sum()}')
"
```

---

## 2. 七个任务名

| 短名 | 完整 Task Name |
|------|---------------|
| FrontKick | `Unitree-G1-29dof-Mimic-MartialArts-FrontKick` |
| LungePunch | `Unitree-G1-29dof-Mimic-MartialArts-LungePunch` |
| SideKick | `Unitree-G1-29dof-Mimic-MartialArts-SideKick` |
| RoundhouseKick | `Unitree-G1-29dof-Mimic-MartialArts-RoundhouseKick` |
| HeianShodan | `Unitree-G1-29dof-Mimic-MartialArts-HeianShodan` |
| Bassai | `Unitree-G1-29dof-Mimic-MartialArts-Bassai` |
| Empi | `Unitree-G1-29dof-Mimic-MartialArts-Empi` |

---

## 3. 关键文件速查

| 要修改什么 | 去哪个文件 |
|-----------|-----------|
| 奖励权重/σ | `martial_arts/tracking_env_cfg.py` → `MartialArtsRewardsCfg` |
| 终止条件阈值 | `martial_arts/tracking_env_cfg.py` → `MartialArtsTerminationsCfg` |
| 域随机化范围 | `martial_arts/tracking_env_cfg.py` → `EventCfg` |
| 初始状态扰动 | `tracking_env_cfg.py` → `_make_command_cfg()` → `pose_range` |
| 并行环境数 | `tracking_env_cfg.py` → `MartialArtsBaseEnvCfg` → `num_envs` |
| 电机 PD 增益 | `gangnanm_style/g1.py` → `G1_CYLINDER_CFG` |
| PPO 超参数 | `mimic/agents/rsl_rl_ppo_cfg.py` → `BasePPORunnerCfg` |
| 奖励函数实现 | `mimic/mdp/rewards.py` |
| 动捕加载逻辑 | `mimic/mdp/commands.py` → `MotionLoader` + `MotionCommand` |
| C++ 串联器 | `deploy/include/FSM/State_MartialArtsSequencer.h` |
| 部署编排配置 | `deploy/robots/g1_29dof/config/config.yaml` → segments |

---

## 4. 当前 v5 奖励参数

| 奖励项 | 权重 | σ | 类型 |
|--------|------|---|------|
| joint_acc | -2.5e-7 | — | 正则化 |
| joint_torque | -1e-5 | — | 正则化 |
| action_rate | **-0.05** | — | 正则化 |
| joint_limit | -10.0 | — | 正则化 |
| undesired_contacts | -0.1 | — | 正则化 |
| anchor_pos | 0.5 | 0.3 | 锚点 |
| anchor_ori | **1.0** | **0.5** | 锚点 |
| body_pos | 1.5 | 0.5 | 全身 |
| body_ori | 1.5 | 0.8 | 全身 |
| joint_pos | **3.0** | **0.6** | 全身 |
| ee_pos | **3.0** | **0.2** | 末端 |
| ee_lin_vel | **1.0** | **1.0** | 末端 |
| body_lin_vel | 0.5 | 1.5 | 速度 |
| body_ang_vel | 0.5 | 4.0 | 速度 |

**加粗** = v5 修改或新增

---

## 5. 终止条件阈值

| 条件 | 阈值 | 含义 |
|------|------|------|
| `anchor_pos` (Z) | 0.25m | 躯干高度偏离 |
| `anchor_ori` | 0.8 | 躯干朝向偏差 |
| `ee_body_pos` (Z) | **0.6m** | 末端执行器高度偏离 |
| `time_out` | 30s | 最大 episode 时长 |

---

## 6. TensorBoard 指标速查

| 指标 | 理想趋势 | 异常 → 可能原因 |
|------|---------|----------------|
| `reward/total` | 持续↑ → 收敛 | 不动 → σ太小 / 奖励冲突 |
| `error_joint_pos` | 持续↓ | 不降 → joint_pos 权重太低 |
| `error_body_pos` | 持续↓ | 不降 → body_pos σ太小 |
| `error_anchor_rot` | ↓ 到 <0.3 | >0.5 → anchor_ori 权重太低 |
| `episode_length` | 接近 30s | <5s → 终止条件太严 |
| `sampling_entropy` | 先↓后稳 | →0 → 过拟合某段 |
| `sampling_top1_prob` | <0.3 | >0.5 → 反复同一段失败 |

---

## 7. GPU 内存速查

| num_envs | 估计显存 | 适用 GPU |
|----------|---------|---------|
| 4096 | ~16 GB | RTX 4090 / A100 |
| 2048 | ~10 GB | RTX 3090 / 4080 |
| 1024 | ~7 GB | RTX 3080 |
| **512** | **~5 GB** | **RTX 4070 Ti 12GB** |
| 256 | ~3 GB | RTX 3060 |

---

## 8. 紧急命令

```bash
# 紧急停止 (C++ 部署时): LT + B → Passive 状态

# 杀死训练进程
pkill -f "train.py"

# 查看 GPU 使用
nvidia-smi

# 清理 OOM 后的残留进程
kill $(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
```

---

**返回** → [00_WuBot_Learning_Index.md](00_WuBot_Learning_Index.md)
