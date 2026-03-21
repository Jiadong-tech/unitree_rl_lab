# 01 — 项目概览与核心架构

> **阅读时间**: 10 分钟 · **难度**: ⭐ · **前置要求**: 无
>
> **学完你能**: 向别人一句话解释武bot是什么、画出系统架构图、知道改代码去哪找文件

---

## 1. 项目愿景

让 **Unitree G1**（29自由度人形机器人）完整表演一套武术招式，实现类似**春节联欢晚会机器人武术表演**的效果。

7 个独立的武术动作，各自训练出一个神经网络策略，最后由 C++ 状态机串联成完整表演。

---

## 2. 武术 vs 普通运动的区别

这是理解整个项目技术选择的基础：

| 特性 | 行走 | 舞蹈 (Gangnam Style) | 🥋 武术 |
|------|------|------|------|
| 支撑相 | 双脚交替 | 多数双脚 | **大量单脚**（踢腿时） |
| 力学 | 周期性、对称 | 节奏性、低冲击 | **爆发性、高冲击、非对称** |
| 质心偏移 | 小幅前后 | 中等左右 | **极端偏移**（高踢腿时） |
| 容错性 | 高（自恢复） | 中 | **低**（单脚站立一推即倒） |
| 关节速度 | 低~中 | 中 | **极高**（出拳/踢腿瞬间） |

> 💡 **这就是为什么不能直接套用行走/舞蹈的参数——武术需要专门的奖励设计和终止条件。**

---

## 3. 核心架构：Policy Sequencer（策略串联器）

这是整个项目最重要的设计决策。

### 3.1 方案对比

| 方案 | 做法 | 问题 |
|------|------|------|
| ❌ 单一网络学所有 | 一个大模型学7个动作 | 灾难性遗忘、动作软绵 |
| ❌ 拼接NPZ训一个模型 | 把7段动作拼成长序列 | Credit Assignment 困难、一处崩全崩 |
| ✅ **Policy Sequencer** | 7个独立模型 + C++串联 | 各自精雕细琢、灵活编排 |

### 3.2 架构图

```
训练层 (Python / Isaac Lab)              部署层 (C++)
┌─────────────────────────┐            ┌──────────────────────────────┐
│  FrontKick   → ONNX_1  ─┐           │  State_MartialArtsSequencer  │
│  LungePunch  → ONNX_2  ─┤           │                              │
│  SideKick    → ONNX_3  ─┤           │  ONNX_1 → 过渡 → ONNX_2 →  │
│  Roundhouse  → ONNX_4  ─┼──────────→│  过渡 → ONNX_3 → 过渡 → …  │
│  HeianShodan → ONNX_5  ─┤           │                              │
│  Bassai      → ONNX_6  ─┤           │  YAML 配置: 顺序自由编排     │
│  Empi        → ONNX_7  ─┘           └──────────────────────────────┘
└─────────────────────────┘
     7 个独立训练任务                       1 个 C++ 状态机串联播放
```

### 3.3 关键优势

- **独立训练**：踢腿训崩了不影响冲拳
- **独立调参**：每个动作可以用不同的 reward 权重
- **自由编排**：只需修改 YAML 就能改变表演顺序
- **渐进开发**：先做好一个动作，再扩展到全部

---

## 4. 代码结构

### 4.1 整体布局

```
unitree_rl_lab/
│
├── source/unitree_rl_lab/unitree_rl_lab/tasks/mimic/
│   │
│   ├── mdp/                           ← 🧠 共享核心逻辑
│   │   ├── commands.py                # 动捕数据加载 + 自适应采样
│   │   ├── rewards.py                 # 追踪奖励函数
│   │   ├── observations.py            # 观测量计算
│   │   ├── terminations.py            # 终止条件
│   │   └── events.py                  # 域随机化
│   │
│   ├── agents/rsl_rl_ppo_cfg.py       ← PPO 超参数（所有mimic共享）
│   │
│   └── robots/g1_29dof/
│       ├── gangnanm_style/            ← 🕺 舞蹈（成熟参考）
│       │   ├── g1.py                  # G1 机器人物理配置
│       │   └── tracking_env_cfg.py    # 舞蹈环境配置
│       │
│       └── martial_arts/              ← 🥋 武术（本项目核心）
│           ├── __init__.py            # 7 个 gym.register 注册
│           ├── tracking_env_cfg.py    # 武术环境配置
│           └── G1_*.npz              # 7 套训练数据
│
├── scripts/rsl_rl/
│   ├── train.py                       ← 训练入口
│   └── play.py                        ← 推理 + ONNX 导出
│
└── deploy/
    └── include/FSM/
        └── State_MartialArtsSequencer.h  ← C++ 策略串联
```

### 4.2 共享 vs 专有

武术项目遵循**最大复用**原则——只有 3 样东西是武术专有的：

| 组件 | 来源 | 武术专有？ |
|------|------|-----------|
| 机器人物理模型 | `gangnanm_style/g1.py` | ❌ 完全复用 |
| MDP 核心函数 | `mimic/mdp/` | ❌ 复用 + 1个新函数 |
| PPO 超参数 | `agents/rsl_rl_ppo_cfg.py` | ❌ 完全复用 |
| **奖励权重** | — | ✅ `MartialArtsRewardsCfg` |
| **终止条件** | — | ✅ `MartialArtsTerminationsCfg` |
| **动捕数据** | — | ✅ 7 套 NPZ |
| **C++ 串联器** | — | ✅ `State_MartialArtsSequencer.h` |

> 💡 **设计哲学**：改奖励权重和终止条件阈值就能适配不同动作，不需要大改代码。

---

## 5. 七个武术动作一览

| 动作 | 日文名 | NPZ 文件 | 帧数 | 时长 | 难度 |
|------|--------|----------|------|------|------|
| 正踢 | Mae Geri | `G1_front_kick.npz` | 1145 | 22.9s | ⭐⭐ |
| 冲拳 | Oi-Tsuki | `G1_lunge_punch.npz` | 1359 | 27.2s | ⭐⭐ |
| 侧踢 | Yoko Geri | `G1_side_kick.npz` | 613 | 12.3s | ⭐⭐⭐ |
| 回旋踢 | Mawashi Geri | `G1_roundhouse_kick.npz` | 1025 | 20.5s | ⭐⭐⭐ |
| 平安初段 | Heian Shodan | `G1_heian_shodan.npz` | 548 | 11.0s | ⭐⭐⭐⭐ |
| 拔塞 | Bassai | `G1_bassai.npz` | 2548 | 51.0s | ⭐⭐⭐⭐⭐ |
| 燕飞 | Empi | `G1_empi.npz` | 2168 | 43.4s | ⭐⭐⭐⭐⭐ |

---

## 检查清单

读完本文后，你应该能回答：

- [ ] 武bot 为什么不用一个网络学所有动作？
- [ ] `tracking_env_cfg.py` 在哪个目录下？
- [ ] 武术和舞蹈共享了哪些代码，专有了哪些？
- [ ] 7 个动作中最简单和最难的分别是什么？

---

**下一步** → [02_Data_Pipeline.md](02_Data_Pipeline.md) 了解动捕数据怎么变成训练用的 NPZ
