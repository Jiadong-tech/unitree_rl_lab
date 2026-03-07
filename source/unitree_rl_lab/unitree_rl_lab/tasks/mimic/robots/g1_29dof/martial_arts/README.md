# 🥋 G1 武术动作模仿 — Policy Sequencer 架构

使用 Unitree G1 (29DOF) 人形机器人还原 **2026年春节联欢晚会机器人武术表演**。

> ⚠️ 春晚原始 MoCap 数据为宇树科技专有。本项目使用 **CMU MoCap Database Subject #135** (空手道) 作为开源替代。

---

## 🏗️ 架构设计 — 为什么用 Policy Sequencer

### 核心问题
春晚的武术表演是一段**连续的成套动作** (前踢 → 冲拳 → 回旋踢 → ...)，如何实现？

### 两种方案对比

| | ~~Combo NPZ (已废弃)~~ | **Policy Sequencer (当前)** |
|---|---|---|
| 训练 | 拼接 NPZ，训练 1 个 policy | 7 个独立 policy，分别训练 |
| 收敛性 | 差 — 长序列 credit assignment 困难 | 好 — 每段独立收敛 |
| 过渡 | 自然 (参考轨迹连续) | 1s 保持姿态 + 自动切换 |
| 鲁棒性 | 一个动作失败 → 全链崩溃 | 独立 — 可重试、可跳过 |
| 部署灵活性 | 固定顺序 | YAML 配置，随意编排 |

**结论**: 独立训练 + C++ 调度器串联 = 更可靠、更灵活、与春晚实际方案一致。

---

## 📋 任务列表 (7 个独立 policy)

| 动作 | 任务 ID | CMU 来源 | 时长 | 难度 |
|------|---------|----------|------|------|
| 前踢 | `...-MartialArts-FrontKick` | 135_03 | ~23s | ⭐⭐ |
| 冲拳 | `...-MartialArts-LungePunch` | 135_06 | ~27s | ⭐⭐ |
| 侧踢 | `...-MartialArts-SideKick` | 135_07 | ~12s | ⭐⭐⭐ |
| 回旋踢 | `...-MartialArts-RoundhouseKick` | 135_05 | ~21s | ⭐⭐⭐ |
| 平安初段 | `...-MartialArts-HeianShodan` | 135_04 | ~11s | ⭐⭐⭐⭐ |
| 拔塞 | `...-MartialArts-Bassai` | 135_01 | ~51s | ⭐⭐⭐⭐⭐ |
| 燕飛 | `...-MartialArts-Empi` | 135_02 | ~43s | ⭐⭐⭐⭐⭐ |

> 任务 ID 前缀: `Unitree-G1-29dof-Mimic`

---

## 🚀 完整工作流

### 前置条件
- NVIDIA Isaac Sim 4.x + Isaac Lab
- `./unitree_rl_lab.sh -i` 安装本包
- Python 3.10+, numpy, scipy

### 一键全流程 (CSV → NPZ → 训练 → 部署打包)
```bash
bash scripts/mimic/martial_arts_pipeline.sh all
```

### 分步运行

```bash
# Step 1: CMU ASF+AMC → G1 CSV (已完成, CSV 已包含在本目录)
bash scripts/mimic/martial_arts_pipeline.sh csv

# Step 2: CSV → NPZ (需要 Isaac Sim)
bash scripts/mimic/martial_arts_pipeline.sh npz

# Step 3: 验证 NPZ 数据
bash scripts/mimic/martial_arts_pipeline.sh validate

# Step 4: 训练 (可选分组: all/kicks/punch/kata/<task_key>)
bash scripts/mimic/martial_arts_pipeline.sh train kicks     # 只练踢腿
bash scripts/mimic/martial_arts_pipeline.sh train all       # 全部训练

# Step 5: 收集 ONNX 到部署目录
bash scripts/mimic/martial_arts_pipeline.sh deploy
```

### 推理/演示 (单个动作)
```bash
python scripts/rsl_rl/play.py --task Unitree-G1-29dof-Mimic-MartialArts-FrontKick
python scripts/rsl_rl/play.py --task Unitree-G1-29dof-Mimic-MartialArts-FrontKick --video
```

---

## 🔗 Policy Sequencer — 成套动作串联

### 工作原理

```
   C++ FSM 状态机
   ┌──────────────────────────────────────────────────┐
   │  State_MartialArtsSequencer                      │
   │                                                  │
   │  Segment 0        Segment 1        Segment 2     │
   │  ┌──────────┐    ┌──────────┐    ┌──────────┐   │
   │  │front_kick│─1s─│lunge_    │─1s─│roundhouse│   │
   │  │  .onnx   │hold│punch.onnx│hold│_kick.onnx│...│
   │  └──────────┘    └──────────┘    └──────────┘   │
   │       23s             27s             21s        │
   └──────────────────────────────────────────────────┘
```

1. 从 YAML 配置加载 segment 列表 (policy_dir + motion_file)
2. 依次执行每个 segment 的 ONNX policy
3. 每个 segment 播放 `duration_s` 秒 (由 motion CSV 帧数决定)
4. segment 之间保持当前姿态 `transition_hold_s` 秒
5. 全部播完后自动回到 FixStand 状态

### 部署配置 (`deploy/robots/g1_29dof/config/config.yaml`)

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
      motion_file: ...
    # ... 自由编排顺序和组合
```

### 操控方式

| 按键组合 | 动作 |
|---------|------|
| `L2 + Up` | Passive → FixStand |
| `R1 + X` | FixStand → Velocity (正常行走) |
| `L2(2s) + Right` | Velocity → **武术表演模式** |
| `L2 + B` | 任何状态 → Passive (紧急停止) |

---

## 📂 精简文件结构

```
martial_arts/                       ← Isaac Lab task package
├── __init__.py                     # 7 个 gym.register 任务注册
├── tracking_env_cfg.py             # 环境配置 (场景/观测/奖励/终止)
├── README.md                       # 本文件
├── G1_front_kick.csv               # CMU→G1 转换后的动作数据
├── G1_roundhouse_kick.csv
├── G1_side_kick.csv
├── G1_lunge_punch.csv
├── G1_heian_shodan.csv
├── G1_bassai.csv
├── G1_empi.csv
└── G1_*.npz                        # ⬜ Isaac Sim 生成 (Step 2)

deploy/
├── include/FSM/
│   └── State_MartialArtsSequencer.h  # 🔑 Policy Sequencer (串联核心)
└── robots/g1_29dof/
    ├── main.cpp                      # 包含 Sequencer 头文件
    └── config/
        ├── config.yaml               # MartialArtsSequencer 段落配置
        └── policy/mimic/martial_arts/ # ⬜ 训练后放入 ONNX + CSV

scripts/mimic/
├── martial_arts_pipeline.sh        # 统一管线 (csv/npz/validate/train/deploy)
├── cmu_amc_to_csv.py               # CMU → G1 CSV 转换
├── csv_to_npz.py                   # CSV → NPZ (Isaac Sim)
└── validate_npz.py                 # NPZ 质量检查
```

---

## 🔧 数据管线

```
CMU #135 (ASF+AMC, 120fps, Y-up, 角度制)
    │  cmu_amc_to_csv.py
    │  ├── 自动标定: leg_chain → G1骨盆高度 0.78m
    │  ├── 坐标变换: CMU(X右Y上Z后) → Isaac(X前Y左Z上)
    │  └── 关节映射: 30 CMU bones → 29 G1 joints
    ▼
G1_xxx.csv (36列: 3pos + 4quat + 29joints)
    │  csv_to_npz.py (Isaac Sim 回放)
    │  ├── 插值: 120fps → 50fps
    │  └── 录制: body_pos/quat/vel + joint_pos/vel
    ▼
G1_xxx.npz → PPO 训练 (4096并行, 30000 iterations)
    │  play.py 自动导出
    ▼
policy.onnx → C++ State_MartialArtsSequencer 串联部署
```

---

## 🎯 奖励设计 (vs 舞蹈任务)

| 参数 | 武术 | 舞蹈 | 原因 |
|------|------|------|------|
| anchor_pos std | 0.2 | 0.3 | 更严格重心控制 (踢腿时单脚站立) |
| anchor_ori std | 0.3 | 0.4 | 更严格躯干朝向 (冲拳方向性) |
| body_pos weight | 1.5 | 1.0 | 重视肢体位置精度 (出拳/踢腿到位) |
| joint_acc weight | -1.5e-7 | -2.5e-7 | 允许爆发力 (快速出拳/踢腿) |
| push interval | 2-5s | 1-3s | 减少训练扰动 (复杂动作需稳定学习) |

---

## 🔄 与主框架的关系

```
unitree_rl_lab (主框架)
├── tasks/mimic/mdp/          ← 共享: MotionCommand, rewards, observations
├── tasks/mimic/agents/       ← 共享: BasePPORunnerCfg (PPO 超参数)
├── gangnanm_style/g1.py      ← 共享: G1_CYLINDER_CFG 机器人物理配置
├── scripts/rsl_rl/train.py   ← 共享: 训练入口
├── scripts/rsl_rl/play.py    ← 共享: 推理/导出入口
│
└── martial_arts/             ← 本项目专有
    ├── tracking_env_cfg.py   # 武术专用奖励权重 + 终止条件
    ├── __init__.py           # 7 个任务注册
    ├── G1_*.csv              # 7 套动作数据
    └── State_MartialArtsSequencer.h  # C++ 串联部署
```

**设计原则**: 不重复发明轮子。武术项目仅包含 **差异化** 的部分 (数据 + 奖励参数 + 部署串联器)，其余全部复用主框架。
