# 宇树机器人 × Isaac Lab 强化学习项目 —— 通俗全面手册

> **目标读者**：有基本编程经验，想系统理解"机器人怎么通过 RL 学会走路/打拳"的开发者。
>
> **阅读时间**：完整阅读约 45 分钟。可按目录跳读感兴趣的章节。
>
> **核心理念**：用"开餐厅"的比喻贯穿全文，让你不读一行代码也能理解整个系统。

---

## 目录

- [第一篇 全景概览：这个项目到底在做什么？](#第一篇-全景概览这个项目到底在做什么)
  - [1.1 一句话总结](#11-一句话总结)
  - [1.2 端到端的流水线全貌](#12-端到端的流水线全貌)
  - [1.3 两大类任务：走路 vs 打拳](#13-两大类任务走路-vs-打拳)
- [第二篇 餐厅比喻：用生活常识理解整套架构](#第二篇-餐厅比喻用生活常识理解整套架构)
  - [2.1 七个角色，一家餐厅](#21-七个角色一家餐厅)
  - [2.2 一次完整的"点菜到上桌"流程](#22-一次完整的点菜到上桌流程)
- [第三篇 代码地图：每个文件夹是干什么的？](#第三篇-代码地图每个文件夹是干什么的)
  - [3.1 目录结构总览](#31-目录结构总览)
  - [3.2 核心 Python 包详解](#32-核心-python-包详解)
  - [3.3 三大分离原则](#33-三大分离原则)
- [第四篇 Isaac Lab 核心概念：像搭积木一样拼环境](#第四篇-isaac-lab-核心概念像搭积木一样拼环境)
  - [4.1 ManagerBasedRLEnv：总指挥](#41-managerbasedrlenv总指挥)
  - [4.2 八大 Manager 与 MDP](#42-八大-manager-与-mdp)
  - [4.3 @configclass：类型安全的配置系统](#43-configclass类型安全的配置系统)
- [第五篇 七大设计模式：为什么代码要这样写？](#第五篇-七大设计模式为什么代码要这样写)
  - [5.1 策略模式 — 可替换的调料包](#51-策略模式--可替换的调料包)
  - [5.2 组合模式 — 班长管帮厨的树状架构](#52-组合模式--班长管帮厨的树状架构)
  - [5.3 工厂方法 — 报菜名下单](#53-工厂方法--报菜名下单)
  - [5.4 生成器模式 — 分步骤写菜谱](#54-生成器模式--分步骤写菜谱)
  - [5.5 模板方法 — 固定的烹饪流水线](#55-模板方法--固定的烹饪流水线)
  - [5.6 适配器模式 — 碗碟转外卖餐盒](#56-适配器模式--碗碟转外卖餐盒)
  - [5.7 状态模式 — 机器人的行为档位](#57-状态模式--机器人的行为档位)
  - [5.8 模式协作全景图](#58-模式协作全景图)
- [第六篇 走路任务详解（Locomotion）](#第六篇-走路任务详解locomotion)
  - [6.1 Go2 四足机器狗的速度跟踪](#61-go2-四足机器狗的速度跟踪)
  - [6.2 环境配置逐项解读](#62-环境配置逐项解读)
  - [6.3 奖励函数设计哲学](#63-奖励函数设计哲学)
  - [6.4 观测空间：机器人能"看到"什么](#64-观测空间机器人能看到什么)
- [第七篇 武术模仿任务详解（Mimic）](#第七篇-武术模仿任务详解mimic)
  - [7.1 从动捕数据到机器人打拳](#71-从动捕数据到机器人打拳)
  - [7.2 数据流水线三阶段](#72-数据流水线三阶段)
  - [7.3 MotionCommand：运动跟踪的大脑](#73-motioncommand运动跟踪的大脑)
  - [7.4 奖励函数：怎么让机器人打得像](#74-奖励函数怎么让机器人打得像)
  - [7.5 七个独立策略 → 一场完整表演](#75-七个独立策略--一场完整表演)
- [第八篇 训练全流程：从敲命令到出模型](#第八篇-训练全流程从敲命令到出模型)
  - [8.1 训练脚本 train.py 逐行解读](#81-训练脚本-trainpy-逐行解读)
  - [8.2 PPO 训练循环：1 万次迭代里发生了什么](#82-ppo-训练循环1-万次迭代里发生了什么)
  - [8.3 推理脚本 play.py 与模型导出](#83-推理脚本-playpy-与模型导出)
- [第九篇 部署：从仿真到真实机器人](#第九篇-部署从仿真到真实机器人)
  - [9.1 C++ 部署架构总览](#91-c-部署架构总览)
  - [9.2 有限状态机 FSM 详解](#92-有限状态机-fsm-详解)
  - [9.3 Policy Sequencer：串联七个武术动作](#93-policy-sequencer串联七个武术动作)
- [第十篇 实操速查：我想做 XX 该改哪里？](#第十篇-实操速查我想做-xx-该改哪里)
- [附录 A：设计模式速查表](#附录-a设计模式速查表)
- [附录 B：关键超参数速查](#附录-b关键超参数速查)
- [附录 C：常见问题 FAQ](#附录-c常见问题-faq)

---

# 第一篇 全景概览：这个项目到底在做什么？

## 1.1 一句话总结

> 用 NVIDIA Isaac Lab 物理仿真引擎，通过强化学习（PPO 算法），训练宇树机器人（Go2 机器狗、H1/G1 人形机器人）学会走路和打武术，然后导出模型部署到真实机器人上。

## 1.2 端到端的流水线全貌

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        完整生命周期                                       │
│                                                                          │
│  【数据准备】          【仿真训练】           【实机部署】                  │
│                                                                          │
│  CMU 动捕数据     →   Isaac Lab 物理仿真  →   C++ ONNX 推理              │
│  (真人武术动作)       (4096 个机器人并行)      (真实机器人实时运行)         │
│                                                                          │
│  ASF+AMC → CSV     Python PPO 训练          FSM 状态机控制               │
│  CSV → NPZ         30000 次迭代              Passive→站立→执行            │
│  (物理重放生成速度)  保存 .pt 模型             ONNX Runtime 推理           │
│                    导出 .onnx + deploy.yaml                              │
│                                                                          │
│  data/cmu_mocap/   scripts/rsl_rl/          deploy/robots/               │
│  scripts/mimic/    source/unitree_rl_lab/                                │
└──────────────────────────────────────────────────────────────────────────┘
```

**通俗理解**：就像教小孩学武术——
1. **找教材**：从 CMU 大学的动作捕捉数据库下载真人武术动作（相当于教学视频）
2. **练习**：在电脑里搭建 4096 个虚拟道场，让 4096 个机器人同时练习（仿真训练）
3. **考核**：练好了导出"肌肉记忆"（神经网络权重）
4. **上场**：把"肌肉记忆"装到真实机器人里执行

## 1.3 两大类任务：走路 vs 打拳

| 特征 | Locomotion（运动控制） | Mimic（动作模仿） |
|:---|:---|:---|
| **目标** | 跟随速度指令行走/跑步 | 精确复现动捕数据中的动作 |
| **机器人** | Go2（四足狗）、H1、G1（人形） | G1-29dof（人形，29 个关节） |
| **命令源** | 手柄/随机生成的速度指令 | NPZ 文件中的逐帧参考姿态 |
| **关键奖励** | 速度跟踪、能量效率、步态 | 全身关节位置/朝向/速度跟踪 |
| **应用场景** | 日常移动、送货、巡逻 | 春晚武术表演、舞蹈、竞技 |
| **训练环境数** | 4096 | 4096 |
| **训练迭代** | ~30000 | ~50000 |

---

# 第二篇 餐厅比喻：用生活常识理解整套架构

## 2.1 七个角色，一家餐厅

想象你要开一家**连锁智能餐厅**——不同分店做不同菜系（Go2 做快餐、G1 做法式大餐），但所有分店共用一套管理系统。这个项目的 7 个设计模式，就像餐厅里 7 个不同的管理制度：

| 设计模式 | 餐厅里的角色 | 解决什么问题 |
|:---:|:---|:---|
| **工厂方法** | 📋 菜单系统 | 顾客报菜名下单，不需要知道后厨怎么做 |
| **生成器** | 📖 菜谱（配方） | 复杂的菜品分步骤写配方，不用一个巨大的说明书 |
| **组合模式** | 👨‍🍳 班长管帮厨 | 厨师长喊"出菜！"，班长自动协调所有帮厨 |
| **策略模式** | 🧂 可替换的调料包 | 同一道菜，可以换辣椒、换花椒，接口统一 |
| **模板方法** | 🏭 固定流水线 | 备料→下锅→翻炒→出锅，顺序固定，内容灵活 |
| **适配器** | 📦 碗碟转外卖餐盒 | 堂食用碗碟，外卖要换包装盒 |
| **状态模式** | 🔄 营业状态切换 | 准备中→营业→清场→关店，不同状态不同行为 |

## 2.2 一次完整的"点菜到上桌"流程

```
顾客走进餐厅（运行 python train.py --task Unitree-Go2-Velocity）

  ① 【菜单系统/工厂方法】
     "我要一份 Go2-Velocity"
     服务员翻菜单 → 找到这道菜对应哪个厨师、哪套配方
     （gym.make() 从注册表中查找环境配置）

  ② 【菜谱/生成器模式】
     后厨拿到配方，分步骤准备：
     - 第1步：选食材 → 场景（Go2 机器人 + 地形 + 传感器）
     - 第2步：定刀工 → 动作空间（12 个关节位置控制）
     - 第3步：调酱料 → 奖励函数（16 个奖励项 + 权重）
     - 第4步：定摆盘 → 观测空间（角速度、关节位置…）
     - 第5步：定火候 → 终止条件（摔倒就结束）
     - 第6步：加花椒 → 随机扰动（随机推力、摩擦力变化）
     （RobotEnvCfg 分步组装 scene/actions/rewards/observations/...）

  ③ 【班长管帮厨/组合模式】
     所有帮厨就位：
     ├── 酱料班（RewardManager）: 15 个帮厨分别计算不同奖励
     ├── 配菜班（ObservationManager）: 6 个帮厨采集不同观测
     ├── 质检班（TerminationManager）: 3 个帮厨检查终止条件
     └── 传菜班（ActionManager）: 1 个帮厨执行关节动作
     厨师长只需说"开工！"，不需要知道每个人具体做什么

  ④ 【碗碟转餐盒/适配器模式】
     后厨做好的菜用碗碟装（Gymnasium 接口）
     转交给外卖平台要换标准餐盒（RSL-RL 接口）
     （RslRlVecEnvWrapper 转换接口格式）

  ⑤ 【流水线/模板方法 + 调料包/策略模式】
     开始炒菜！流水线循环 30000 次：
     │  工位1: 下动作指令（把神经网络输出的关节角度发给机器人）
     │  工位2: 物理仿真（PhysX 引擎模拟一步物理）
     │  工位3: 采集观测（读取关节角度、速度、重力方向…）
     │  工位4: 计算奖励（用调料包——energy(), orientation()...）
     │  工位5: 检查终止（机器人摔倒了？超时了？）
     │  工位6: 更新命令（给新的速度目标）
     │  工位7: 处理重置（摔倒的环境重新开始）
     流水线固定不变 ← 模板方法
     每个工位用什么调料可以换 ← 策略模式

  ⑥ 训练完成！出锅！
     保存 .pt 模型 → 导出 .onnx + deploy.yaml

  ⑦ 【开关门/状态模式】部署到真实机器人
     Passive（关节软化/安全模式）
       → 按 L2 → FixStand（站立归位）
         → 按 R2 → RLBase（执行 RL 策略）
           → 按 L1 → 回到 Passive
```

---

# 第三篇 代码地图：每个文件夹是干什么的？

## 3.1 目录结构总览

```
unitree_rl_lab/
│
├── source/unitree_rl_lab/          ★ 核心 Python 包（pip install -e 安装）
│   └── unitree_rl_lab/
│       ├── tasks/                  ★ 所有任务的定义（环境 = 机器人 + MDP 组件）
│       │   ├── locomotion/           ├── 走路任务（Go2/H1/G1）
│       │   │   ├── mdp/               │   ├── 共享的 MDP 函数库（奖励、观测…）
│       │   │   ├── agents/             │   ├── PPO 超参数
│       │   │   └── robots/             │   └── 各机器人专属配置
│       │   │       ├── go2/            │       ├── Go2 环境配置
│       │   │       ├── h1/             │       ├── H1 环境配置
│       │   │       └── g1/29dof/       │       └── G1 环境配置
│       │   └── mimic/                └── 模仿任务（动作跟踪）
│       │       ├── mdp/                    ├── Mimic 专属 MDP 函数库
│       │       ├── agents/                 ├── Mimic PPO 超参数
│       │       └── robots/g1_29dof/        └── G1 各动作配置
│       │           ├── martial_arts/           ├── 7 个武术动作 + NPZ 数据
│       │           ├── gangnanm_style/         ├── 江南 Style 舞蹈
│       │           ├── dance_102/              ├── 舞蹈 102
│       │           └── petite_verses/          └── 小诗舞蹈
│       ├── assets/                 ★ 机器人模型参数（关节刚度、阻尼、初始姿势）
│       │   └── robots/
│       │       ├── unitree.py          ├── Go2/H1/G1 的物理参数定义
│       │       └── unitree_actuators.py└── 电机参数
│       └── utils/                  ★ 工具函数
│           ├── parser_cfg.py           ├── 配置解析
│           └── export_deploy_cfg.py    └── 导出部署配置
│
├── scripts/
│   ├── rsl_rl/                     ★ 训练/推理入口
│   │   ├── train.py                   ├── 训练主程序
│   │   ├── play.py                    ├── 推理/播放/ONNX 导出
│   │   └── cli_args.py                └── 命令行参数
│   ├── mimic/                      ★ 数据预处理工具
│   │   ├── cmu_amc_to_csv.py          ├── CMU 动捕 → CSV
│   │   ├── csv_to_npz.py             ├── CSV → NPZ（物理仿真）
│   │   └── validate_npz.py           └── NPZ 数据校验
│   └── list_envs.py               ★ 列出所有已注册的任务
│
├── deploy/                         ★ C++ 实机部署
│   ├── include/
│   │   ├── FSM/                       ├── 有限状态机
│   │   │   ├── BaseState.h               │   ├── 状态基类
│   │   │   ├── CtrlFSM.h                 │   ├── 状态机控制器
│   │   │   ├── State_Passive.h            │   ├── 安全/被动状态
│   │   │   ├── State_FixStand.h           │   ├── 站立状态
│   │   │   ├── State_RLBase.h             │   ├── RL 策略执行状态
│   │   │   └── State_MartialArtsSequencer.h│  └── 武术动作串联器
│   │   └── isaaclab/                  └── Isaac Lab C++ 镜像接口
│   └── robots/                        └── 各机器人部署项目（CMake）
│       ├── go2/
│       ├── h1/
│       ├── g1_23dof/
│       └── g1_29dof/
│
├── data/cmu_mocap/                 ★ CMU 原始动捕数据
├── logs/rsl_rl/                    ★ 训练输出（自动生成）
├── doc/                            ★ 文档集合
└── pyproject.toml                  ★ 项目构建配置
```

## 3.2 核心 Python 包详解

整个 `source/unitree_rl_lab/` 下的代码可以分为**三层**：

```
                    ┌─────────────────────────────┐
                    │  Layer 1: 任务定义 (tasks/) │  ← 你最常改的地方
                    │  "做什么"                     │
                    └─────────────┬───────────────┘
                                  │ 引用
                    ┌─────────────▼───────────────┐
                    │  Layer 2: 机器人资产 (assets/)│  ← 定义机器人硬件参数
                    │  "用什么"                     │
                    └─────────────┬───────────────┘
                                  │ 引用
                    ┌─────────────▼───────────────┐
                    │  Layer 3: Isaac Lab 框架     │  ← NVIDIA 提供，你不用改
                    │  (ManagerBasedRLEnv)         │
                    └─────────────────────────────┘
```

**tasks/** 内部的组织逻辑：

- `mdp/` 目录放**共享函数库**（奖励函数、观测函数等），所有机器人都可以调用
- `robots/` 目录放**机器人专属配置**（选择用 mdp/ 中的哪些函数，参数设多少）
- `agents/` 目录放 **PPO 超参数**

这样设计的好处：写一次 `energy()` 奖励函数，Go2、H1、G1 都能用，只是权重不同。

## 3.3 三大分离原则

| 原则 | 拆分方式 | 好处 |
|:---|:---|:---|
| **任务与资产分离** | `tasks/` 定义"做什么"，`assets/` 定义"用什么机器人" | 同一个机器人可以做不同任务 |
| **脚本与库分离** | `scripts/` 是执行入口，`source/` 是可复用包 | 库代码可以被任何脚本调用 |
| **训练与部署分离** | Python 训练 → 导出 ONNX/YAML → C++ 部署 | 训练和部署互不影响 |

---

# 第四篇 Isaac Lab 核心概念：像搭积木一样拼环境

## 4.1 ManagerBasedRLEnv：总指挥

Isaac Lab 的核心设计理念是：**你不需要写代码来定义环境，只需要填配置表**。

`ManagerBasedRLEnv` 就是这个"总指挥"——你告诉它用什么 Manager、每个 Manager 里放什么 Term，它自动帮你组装出一个完整的 RL 训练环境。

**传统做法 vs Isaac Lab 做法**：

```python
# ❌ 传统做法：继承 + 手写所有逻辑
class MyEnv(BaseEnv):
    def compute_reward(self):
        r1 = self.calc_velocity_reward()
        r2 = self.calc_energy_penalty()
        r3 = self.calc_orientation_reward()
        return 1.5 * r1 - 0.02 * r2 - 2.5 * r3  # 硬编码

# ✅ Isaac Lab 做法：填配置表
@configclass
class RewardsCfg:
    velocity  = RewTerm(func=mdp.track_lin_vel_xy_exp, weight=1.5)
    energy    = RewTerm(func=mdp.energy,                weight=-0.02)
    orient    = RewTerm(func=mdp.flat_orientation_l2,   weight=-2.5)
    # 想加一个？加一行就行。想删一个？删一行就行。
```

## 4.2 八大 Manager 与 MDP

一个强化学习环境（MDP = 马尔可夫决策过程）需要定义：状态、动作、奖励、转移、终止。Isaac Lab 把这些拆成**八大 Manager**，每个负责一个方面：

```
ManagerBasedRLEnv（总指挥）
│
├── 1️⃣ SceneManager（场景管理器）
│      负责：加载机器人 3D 模型、创建地形、放置传感器
│      对应配置：scene = RobotSceneCfg(...)
│
├── 2️⃣ ActionManager（动作管理器）
│      负责：将神经网络输出转换为关节指令
│      对应配置：actions = ActionsCfg(...)
│      例如：网络输出 [-0.3, 0.5, ...] → 乘以 scale → 加上默认角度 → 发给关节
│
├── 3️⃣ ObservationManager（观测管理器）
│      负责：收集机器人的感知信息，喂给神经网络
│      对应配置：observations = ObservationsCfg(...)
│      包含两组：
│        ├── policy（给 Actor 用的，可以加噪声模拟传感器不精确）
│        └── critic（给 Critic 用的，可以有特权信息）
│
├── 4️⃣ RewardManager（奖励管理器）
│      负责：计算所有奖励项，加权求和
│      对应配置：rewards = RewardsCfg(...)
│      例如：速度跟踪 ×1.5 + 能量惩罚 ×-2e-5 + 姿态惩罚 ×-2.5 + ...
│
├── 5️⃣ TerminationManager（终止管理器）
│      负责：判断 episode 是否结束
│      对应配置：terminations = TerminationsCfg(...)
│      例如：超时终止、摔倒终止、偏离参考轨迹终止
│
├── 6️⃣ CommandManager（命令管理器）
│      负责：生成训练命令（目标速度 或 参考动作）
│      对应配置：commands = CommandsCfg(...)
│      Locomotion：随机生成 (vx, vy, ωz) 速度指令
│      Mimic：从 NPZ 文件逐帧读取参考姿态
│
├── 7️⃣ EventManager（事件管理器）
│      负责：随机化扰动（Domain Randomization）
│      对应配置：events = EventCfg(...)
│      例如：随机推力、随机改变摩擦系数、随机改变质量分布
│
└── 8️⃣ CurriculumManager（课程管理器）
       负责：根据训练进度调节难度
       对应配置：curriculum = CurriculumCfg(...)
       例如：一开始在平地上走，later 在碎石路上走
```

**核心理解**：每个 Manager 就像一个"部门经理"，手下有一堆"员工"（Term）。你只需要在配置表里告诉每个部门有几个员工、各做什么事就行。

## 4.3 @configclass：类型安全的配置系统

Isaac Lab 不用 YAML/JSON 来写配置，而是用 Python 的 `@configclass`（基于 dataclass）：

```python
@configclass
class RobotEnvCfg(ManagerBasedRLEnvCfg):
    """一个完整的 Go2 走路环境配置"""

    # 场景：Go2 机器狗 + 碎石路地形 + 接触力传感器 + 高度扫描器
    scene = RobotSceneCfg(num_envs=4096, env_spacing=2.5)

    # 动作：12 个关节的位置控制，scale=0.25
    actions = ActionsCfg()

    # 观测：角速度、重力方向、关节位置、步态相位...（6 项给 policy，8 项给 critic）
    observations = ObservationsCfg()

    # 命令：随机速度指令，前后 [-1.0, 2.0] m/s，左右 [-0.5, 0.5] m/s
    commands = CommandsCfg()

    # 奖励：16 个奖励项，权重精心调配
    rewards = RewardsCfg()

    # 终止：超时 20 秒、非法接触（头碰地）、姿态翻转
    terminations = TerminationsCfg()

    # 随机化：摩擦力、质量分布、外部推力
    events = EventCfg()

    # 课程：逐渐增加地形难度和速度范围
    curriculum = CurriculumCfg()
```

**为什么不用 YAML？** 三大好处：
1. **类型安全**：写错字段名 IDE 直接报红
2. **可继承**：PlayEnvCfg 直接继承 EnvCfg，只覆盖少数参数
3. **IDE 友好**：Ctrl+Click 跳转到函数定义，自动补全

---

# 第五篇 七大设计模式：为什么代码要这样写？

> 以下每个模式都用"代码 + 比喻 + 核心洞察"三位一体的方式讲解。

## 5.1 策略模式 — 可替换的调料包

**痛点**：奖励函数有十几种，观测函数也有十几种，怎么灵活切换而不改底层代码？

**类比**：做菜时的调料包——只要是"辣味"就行，用朝天椒还是花椒随便换。

**代码实例**：

```python
# 调料包 A：计算能量消耗
def energy(env, asset_cfg) -> torch.Tensor:
    qvel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    qfrc = asset.data.applied_torque[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(qvel) * torch.abs(qfrc), dim=-1)

# 调料包 B：计算姿态偏差
def orientation_l2(env, desired_gravity, asset_cfg) -> torch.Tensor:
    cos_dist = torch.sum(asset.data.projected_gravity_b * desired_gravity, dim=-1)
    return torch.square(0.5 * cos_dist + 0.5)

# 配置中选择用哪些调料包、用多少
@configclass
class RewardsCfg:
    energy      = RewTerm(func=mdp.energy,           weight=-2e-5)   # 用 A
    orientation = RewTerm(func=mdp.orientation_l2,    weight=-2.5)    # 用 B
    # 想换？改 func= 就行，整个框架不用动
```

**核心洞察**：所有奖励/观测函数都遵循统一签名 `(env, **params) → Tensor`，就像所有调料包都是"打开→撒上→完成"。Manager 只负责调用，不关心具体是哪个函数。这是日常调参最频繁接触的模式——**90% 的实验就是在这里换函数和调权重**。

## 5.2 组合模式 — 班长管帮厨的树状架构

**痛点**：RewardManager 下面有 15 个奖励项，ObservationManager 下面有 8 个观测项，怎么统一管理？

**类比**：厨师长说"出菜！"不需要逐一指挥每个帮厨——班长会自动协调。

```
ManagerBasedRLEnv（厨师长）
  ├── RewardManager（酱料班长）
  │     ├── track_lin_vel_xy  ×1.5     ← 帮厨 1
  │     ├── energy            ×-2e-5   ← 帮厨 2
  │     ├── orientation       ×-2.5    ← 帮厨 3
  │     └── ... 共 16 个帮厨
  ├── ObservationManager（配菜班长）
  │     ├── PolicyGroup（给顾客看的菜）
  │     │     ├── base_ang_vel        ← 帮厨
  │     │     ├── joint_pos_rel       ← 帮厨
  │     │     └── gait_phase          ← 帮厨
  │     └── CriticGroup（给后厨参考的备注）
  │           ├── base_lin_vel        ← 帮厨
  │           └── joint_effort        ← 帮厨
  └── TerminationManager（质检班长）
        ├── time_out                  ← 帮厨
        └── illegal_contact           ← 帮厨
```

**代码关键**：厨师长只需一行调用，不管下面有几个帮厨：

```python
# env.step() 内部
rewards = self.reward_manager.compute()        # 自动遍历 16 个 RewTerm，加权求和
obs = self.observation_manager.compute()       # 自动遍历所有 ObsTerm，拼接成向量
terminated = self.termination_manager.compute() # 自动遍历所有 TermTerm，逻辑或
```

**核心洞察**：增减奖励项，只改配置里的一行，不改 RewardManager 的代码。这就是"开闭原则"——新增功能不用改已有代码。

## 5.3 工厂方法 — 报菜名下单

**痛点**：项目有 20+ 个不同任务（Go2走路、G1武术-平安初段、G1武术-前踢…），`train.py` 怎么知道创建哪个环境？

**类比**：菜单上每道菜有编号，顾客报编号，后厨自动按配方做。

**注册"菜品"**（每个任务文件夹的 `__init__.py`）：

```python
# Go2 的菜品注册
gym.register(
    id="Unitree-Go2-Velocity",                              # 菜名
    entry_point="isaaclab.envs:ManagerBasedRLEnv",          # 通用工厂
    kwargs={
        "env_cfg_entry_point": "...velocity_env_cfg:RobotEnvCfg",  # 配方
    },
)

# G1 武术-平安初段的菜品注册
gym.register(
    id="Unitree-G1-29dof-Mimic-MartialArts-HeianShodan",   # 另一道菜
    entry_point="isaaclab.envs:ManagerBasedRLEnv",          # 同一个工厂！
    kwargs={
        "env_cfg_entry_point": "...tracking_env_cfg:HeianShodanEnvCfg",  # 不同配方
    },
)
```

**下单**（`train.py`）：

```python
# 顾客只需报菜名，不需要知道任何内部细节
env = gym.make("Unitree-Go2-Velocity", cfg=env_cfg)
```

**核心洞察**：新增一个任务？只需新建文件夹 + 写 `__init__.py` 注册 + 写环境配置。`train.py` **一行不用改**——它只认菜名，不认厨师。

**自动发现机制**：`tasks/__init__.py` 会递归扫描所有子包，执行每个 `__init__.py` 里的 `gym.register()`。所以你新加的任务会被自动发现，无需手动维护列表。

## 5.4 生成器模式 — 分步骤写菜谱

**痛点**：一个 RL 环境配置有几十个组件（场景+动作+观测+奖励+终止+事件+课程），不能全塞进一个构造函数。

**类比**：菜谱分步骤——"第1步选食材、第2步调酱料、第3步下锅…"

```python
@configclass
class RobotEnvCfg(ManagerBasedRLEnvCfg):
    scene   = RobotSceneCfg(num_envs=4096)     # 步骤1: 选食材
    actions = ActionsCfg()                       # 步骤2: 定刀工
    observations = ObservationsCfg()             # 步骤3: 定摆盘
    commands = CommandsCfg()                      # 步骤4: 定菜式
    rewards = RewardsCfg()                        # 步骤5: 调酱料
    terminations = TerminationsCfg()              # 步骤6: 定火候
    events  = EventCfg()                          # 步骤7: 加花椒
    curriculum = CurriculumCfg()                  # 步骤8: 进阶路线

    def __post_init__(self):                      # 步骤9: 最终微调
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
```

**不同菜品的变体**——推理配置直接继承训练配置，只改少数参数：

```python
@configclass
class RobotPlayEnvCfg(RobotEnvCfg):
    """推理配置：继承训练配置，覆盖少数参数"""
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32           # 32 个环境够了
        # 关闭随机推力...
```

**核心洞察**：Go2、H1、G1 的配方**步骤相同、内容不同**。这就是"不同生成器产出不同产品"——代码结构一致，参数各异。

## 5.5 模板方法 — 固定的烹饪流水线

**痛点**：无论训练什么任务，`env.step()` 的流程必须是固定的——先执行动作、再物理仿真、再算观测、再算奖励…顺序不能乱。

**类比**：流水线——备料→下锅→翻炒→出锅，顺序固定，每个工位做什么可以变。

```python
class ManagerBasedRLEnv:
    def step(self, action):
        # ═══ 以下步骤顺序不可更改 ═══

        # Step 1: 执行动作（把关节角度发给机器人）
        self.action_manager.process_action(action)

        # Step 2: 物理仿真（decimation=4，即 4 次子步骤）
        for _ in range(self.cfg.decimation):
            self.scene.write_data_to_sim()
            self.sim.step()       # PhysX 引擎算一步物理
            self.scene.update()

        # Step 3: 计算观测（读取机器人的感知数据）
        obs = self.observation_manager.compute()

        # Step 4: 计算奖励（用 16 个奖励函数加权求和）
        reward = self.reward_manager.compute()

        # Step 5: 检查终止（摔倒？超时？）
        terminated, truncated = self.termination_manager.compute()

        # Step 6: 更新命令（新的速度目标/下一帧参考动作）
        self.command_manager.compute()

        # Step 7: 处理重置（终止的环境重新开始）
        reset_ids = (terminated | truncated).nonzero()
        self.event_manager.apply(mode="reset", env_ids=reset_ids)

        return obs, reward, terminated, info
```

**核心洞察**：Go2 走路和 G1 打拳用的是**完全相同的 `step()` 骨架**，只是每个步骤里调用的具体函数不同（通过配置注入）。这是"组合优于继承"的经典实践。

## 5.6 适配器模式 — 碗碟转外卖餐盒

**痛点**：Isaac Lab 遵循 Gymnasium 接口，但 RSL-RL 训练库需要自己的接口格式。

**类比**：餐厅用碗碟（Gymnasium），外卖平台要标准餐盒（RSL-RL），需要一个"转换插头"。

```python
# 碗碟（Gymnasium 接口）
env = gym.make("Unitree-Go2-Velocity", cfg=env_cfg)

# 转换插头！
env = RslRlVecEnvWrapper(env, clip_actions=True)

# 外卖平台（RSL-RL）可以用了
runner = OnPolicyRunner(env, agent_cfg.to_dict(), ...)
```

| Gymnasium 格式 | RSL-RL 格式 |
|:---|:---|
| `env.step() → (obs, reward, terminated, truncated, info)` | `env.step() → (obs, privileged_obs, rewards, dones, infos)` |
| `env.reset() → (obs, info)` | `env.get_observations() → obs` |
| `env.observation_space` | `env.num_obs` / `env.num_actions` |

**核心洞察**：以后想换 Stable-Baselines3？只需写个新的 `SB3VecEnvWrapper`，环境代码一行不改。同样，`export_deploy_cfg()` 也是一种"适配"——把 Python 配置转成 C++ 能读的 YAML。

## 5.7 状态模式 — 机器人的行为档位

**痛点**：真实机器人不能上电就飙策略，需要安全的启动/停止流程。

**类比**：汽车的档位——P 档（驻车）→ D 档（行驶）→ P 档（停车），不同档位不同行为。

```
   按 L2              站稳后按 R2           按 L1
Passive ─────────► FixStand ─────────► RLBase ─────────► Passive
(关节软化/安全)     (PD 控制站立)       (RL 策略推理)     (安全回归)
```

C++ 代码结构：

```cpp
// 统一接口
class BaseState {
    virtual void enter() = 0;     // 进入状态时执行
    virtual void run() = 0;       // 每个控制周期执行
    virtual void exit() = 0;      // 退出状态时执行
};

// CtrlFSM（状态机控制器）不关心当前是哪个状态
void CtrlFSM::run_() {
    currentState->run();          // 多态调用
    // 检查是否需要切换...
    if (needTransition) {
        currentState->exit();
        currentState = nextState;
        currentState->enter();
    }
}
```

**核心洞察**：每个状态是独立的类，职责清晰。比用巨大的 `if/else` 链好维护得多，而且**对于真实硬件来说安全性至关重要**——状态隔离防止意外行为。

## 5.8 模式协作全景图

一次完整的训练中，7 个模式像齿轮一样啮合运转：

```
时间线 ──────────────────────────────────────────────────────────────────►

① import tasks     ② parse_env_cfg    ③ RobotEnvCfg 组装    ④ gym.make()
（插件发现+工厂）  （工厂方法）        （生成器模式）         （工厂方法）
   扫描注册所有任务    从注册表加载配置    分步组装8大Manager    创建环境实例

⑤ Manager 初始化    ⑥ RslRlVecEnvWrapper  ⑦ env.step() × 30000
（组合模式）         （适配器模式）         （模板方法 + 策略模式）
   构建 Term 树        转换接口格式          固定流程 + 可替换算法

⑧ 保存模型 + 导出 ONNX                     ⑨ C++ FSM 部署
                                             （状态模式）
                                              Passive → FixStand → RLBase
```

| 阶段 | 模式 | 通俗理解 |
|:---:|:---:|:---|
| 启动 | 插件发现+工厂 | 印刷菜单，让所有菜品可以被点 |
| 点菜 | 工厂方法 | 顾客报菜名，后厨自动找配方 |
| 备菜 | 生成器 | 按菜谱分步准备 |
| 分工 | 组合 | 班长协调所有帮厨就位 |
| 接单 | 适配器 | 碗碟→外卖餐盒 |
| 炒菜 | 模板方法+策略 | 流水线开工，每个工位用不同调料 |
| 上桌 | 状态 | 机器人安全启动+执行+停止 |

## 5.9 模式嵌套关系 — 七大模式的因果链与主从层级

> **核心问题**：这 7 个模式不是并列关系——有的模式**住在另一个模式体内**，有的模式**是另一个模式的因**，有的模式**在某一步骤里悄悄化身为另一个模式**。理解这种嵌套关系，才算真正理解框架。

### 5.9.1 三层嵌套关系图

从"最外层"到"最内层"，模式可以画成一个俄罗斯套娃：

```
┌──────────────────────────────────────────────────────────────────────┐
│ 模板方法（env.step 主循环）                                           │
│                                                                      │
│   step 1: action_manager.process_action(action)                      │
│   step 2: sim.step()  × decimation                                   │
│   step 3: observation_manager.compute()  ◄─┐                        │
│   step 4: reward_manager.compute()       ◄─┤ 组合模式               │
│   step 5: termination_manager.compute()  ◄─┤ （Manager遍历Term树）   │
│   step 6: command_manager.compute()      ◄─┤                        │
│   step 7: event_manager.apply("reset")   ◄─┘                        │
│                                                                      │
│   ┌──────────────────────────────────────────────────────────────┐   │
│   │ 组合模式（Manager → Term 树）                                 │   │
│   │                                                              │   │
│   │   RewardManager.compute():                                   │   │
│   │     for term in self.terms:                                  │   │
│   │       value += term.weight * term.func(env, **term.params)   │   │
│   │                                          ▲                   │   │
│   │                                          │                   │   │
│   │   ┌──────────────────────────────────────┼───────────────┐   │   │
│   │   │ 策略模式（可替换的 func）              │               │   │   │
│   │   │                                      │               │   │   │
│   │   │  track_lin_vel_xy_exp()    ◄─────────┤               │   │   │
│   │   │  energy()                  ◄─────────┤               │   │   │
│   │   │  feet_gait()              ◄─────────┤               │   │   │
│   │   │  joint_mirror()            ◄─────────┘               │   │   │
│   │   └──────────────────────────────────────────────────────┘   │   │
│   └──────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────┘
```

**通俗理解**：
- **模板方法**是最外层的"流水线"，决定了 step 1 → step 2 → … → step 7 的**不可更改的顺序**。
- 流水线的每个工位（step 3/4/5/6/7）内部，是**组合模式**——Manager 遍历它名下的所有 Term。
- 每个 Term 的具体计算，是一个**策略函数**——可以随时替换。

**一句话**：**模板方法驱动组合，组合调用策略**。这是运行时最核心的三层嵌套。

### 5.9.2 因果触发链

模式之间还有"谁触发谁"的因果关系。从训练脚本的第一行到最后一行：

```
工厂方法 ──触发──► 生成器 ──产出──► 组合 ──被套入──► 模板方法
  (gym.make)       (EnvCfg)       (Manager-Term)    (env.step)
      │                                                  │
      └──► 适配器 ──包裹──► 模板方法的输出               │
           (Wrapper)        (obs,rew→RSL-RL格式)         │
                                                         │
                                               训练完成，导出模型
                                                         │
                                                         ▼
                                                    状态模式
                                                  (C++ FSM 部署)
```

**逐步因果分析**：

| 顺序 | 因（触发者） | 果（被触发者） | 具体发生了什么 |
|:---:|:---:|:---:|:---|
| ① | **工厂方法** | **生成器** | `gym.make("Unitree-Go2-Velocity")` → 从注册表取出 `env_cfg_entry_point` → 实例化 `RobotEnvCfg` |
| ② | **生成器** | **策略（引用）** | `RobotEnvCfg` 的每个 `RewTerm(func=energy, weight=-0.001)` 引用了一个策略函数 |
| ③ | **生成器** | **组合** | `ManagerBasedRLEnv.__init__()` 读取 `RobotEnvCfg`，把每个 `TermCfg` 实例化为 `Term`，挂在 `Manager` 树上 |
| ④ | **工厂方法** | **适配器** | `env = gym.make(...)` 之后立刻 `env = RslRlVecEnvWrapper(env)` 包裹 |
| ⑤ | **模板方法** | **组合+策略** | `env.step()` 的每个步骤调用 `Manager.compute()`，触发树遍历 + 策略函数执行 |
| ⑥ | **训练完成** | **状态模式** | 导出 ONNX，C++ FSM 在 `State_RLBase` 中加载模型，开始部署推理 |

### 5.9.3 模式之间的"蕴含"关系

"蕴含"指的是：**模式 A 的某个步骤里，本质上就是模式 B 在运作**。

```
┌──────────────────────────────────────────────────────────────┐
│                      蕴含关系矩阵                             │
│                                                              │
│  工厂方法的"创建产品"步骤  ═══蕴含═══►  生成器               │
│  │  gym.make() 内部调用 EnvCfg.__post_init__() 组装配置      │
│  │                                                           │
│  生成器的"每个构建步骤"    ═══蕴含═══►  策略（引用层）        │
│  │  RewardsCfg 里每个 RewTerm 都 func= 一个策略函数          │
│  │                                                           │
│  组合模式的"每个叶子节点"  ═══蕴含═══►  策略（执行层）        │
│  │  Term.compute() 实际就是调用 term.func(env, **params)     │
│  │                                                           │
│  模板方法的"每个固定步骤"  ═══蕴含═══►  组合                  │
│  │  step 3~7 每个都是一次 Manager.compute() 树遍历           │
│  │                                                           │
│  适配器的"被包裹对象"      ═══蕴含═══►  模板方法              │
│  │  Wrapper.step() 内部调用 self.env.step() = 模板方法       │
│  │                                                           │
│  状态模式的"RLBase状态"    ═══蕴含═══►  策略（简化版）        │
│  │  State_RLBase 内部用 ONNX 推理，本质是策略执行            │
│  │                                                           │
│  状态模式的"武术序列状态"  ═══蕴含═══►  组合+策略（简化版）   │
│  │  MartialArtsSequencer 管理 7 个策略模型的序列切换          │
└──────────────────────────────────────────────────────────────┘
```

### 5.9.4 主从关系总结

从"谁是容器、谁是被容纳者"的角度，可以画出清晰的主从层级：

```
                    ┌─────────────┐
                    │  工厂方法    │  ← 入口（触发一切）
                    │  gym.make   │
                    └──────┬──────┘
                           │ 触发
                    ┌──────▼──────┐
                    │   生成器     │  ← 组装者（配置阶段的主角）
                    │  EnvCfg     │
                    └──────┬──────┘
                           │ 其每个步骤引用
                    ┌──────▼──────────────────┐
                    │  策略                     │  ← 原子单元（最小可替换粒子）
                    │  func=energy/gait/...    │
                    └──────┬──────────────────-┘
                           │ 被挂载到
                    ┌──────▼──────┐
                    │   组合       │  ← 结构骨架（运行时的主角）
                    │  Manager树  │
                    └──────┬──────┘
                           │ 被嵌入
                    ┌──────▼──────┐
                    │  模板方法    │  ← 运行时驱动器（最外层循环）
                    │  env.step   │
                    └──────┬──────┘
                           │ 被包裹
                    ┌──────▼──────┐
                    │   适配器     │  ← 接口转换层（训练库的对接点）
                    │  Wrapper    │
                    └──────┬──────┘
                           │ 训练完成后
                    ┌──────▼──────┐
                    │  状态模式    │  ← 部署终端（C++ 世界的入口）
                    │  C++ FSM   │
                    └─────────────┘
```

**七种模式的角色定位**：

| 模式 | 角色 | 生命周期 | 被谁包含 | 包含谁 |
|:---|:---|:---|:---|:---|
| **策略** | 原子单元 | 贯穿始终 | 被生成器引用、被组合执行 | 不包含其他模式 |
| **生成器** | 配置组装者 | 初始化阶段 | 被工厂方法触发 | 引用策略函数 |
| **工厂方法** | 入口开关 | 初始化阶段 | 无（最顶层触发者） | 触发生成器 |
| **组合** | 运行时骨架 | 运行阶段 | 被模板方法的每个步骤调用 | 执行策略函数 |
| **模板方法** | 运行时引擎 | 运行阶段 | 被适配器包裹 | 调用组合模式 |
| **适配器** | 接口胶水 | 运行阶段 | 被 RSL-RL Runner 使用 | 包裹模板方法 |
| **状态** | 部署终端 | 部署阶段 | 无（C++世界顶层） | 内嵌简化版策略 |

### 5.9.5 一个 step 中的模式"洋葱图"

在训练阶段执行一步 `runner.learn()` → `env.step(action)` 时，模式的嵌套像剥洋葱：

```
RSL-RL OnPolicyRunner.learn()
 └─► 适配器: RslRlVecEnvWrapper.step(action)      ← 第1层：接口转换
      └─► 模板方法: ManagerBasedRLEnv.step(action)  ← 第2层：固定流程
           ├─► action_manager.process_action()
           ├─► sim.step() × decimation
           ├─► 组合: observation_manager.compute()   ← 第3层：树遍历
           │    ├─► 策略: base_lin_vel(env)           ← 第4层：具体算法
           │    ├─► 策略: base_ang_vel(env)
           │    └─► 策略: joint_pos(env)
           ├─► 组合: reward_manager.compute()         ← 第3层
           │    ├─► 策略: track_lin_vel_xy_exp(env)   ← 第4层
           │    ├─► 策略: energy(env)
           │    └─► 策略: feet_gait(env)
           ├─► 组合: termination_manager.compute()    ← 第3层
           │    ├─► 策略: time_out(env)               ← 第4层
           │    └─► 策略: illegal_contact(env)
           └─► 组合: command_manager.compute()        ← 第3层
                └─► 策略: UniformVelocityCommand()    ← 第4层
```

**结论**：一次 `step()` 调用，穿越了 **适配器 → 模板方法 → 组合 → 策略** 四层模式嵌套。这不是设计者刻意为之，而是**关注点分离**自然产生的结果——每个模式只解决一个问题，组合起来就形成了层级。

### 5.9.6 双生命线：配置期 vs 运行期

七个模式分布在两条生命线上，各自的主从关系不同：

```
━━━━ 配置期生命线 ━━━━                  ━━━━ 运行期生命线 ━━━━

工厂方法（主）                           模板方法（主）
  │                                       │
  ├── 生成器（从）                         ├── 组合（从）
  │     │                                 │     │
  │     └── 策略引用（从的从）              │     └── 策略执行（从的从）
  │                                       │
  └── 适配器（并列）                       └── 适配器（包裹层）

配置期的"策略"只是引用                   运行期的"策略"才真正执行
(func=energy 存在配置里)                 (energy(env) 被调用计算)
```

**关键洞察**：**策略模式跨越了两条生命线**——它在配置期是"被引用的函数名"，在运行期是"被调用的计算逻辑"。这也是为什么策略被称为"原子单元"——它是连接配置与运行的桥梁。

---

# 第六篇 走路任务详解（Locomotion）

## 6.1 Go2 四足机器狗的速度跟踪

**目标**：给 Go2 一个速度指令 (vx, vy, ωz)，它就能以该速度稳定行走。

```
控制器发指令               Go2 执行
(vx=1.0, vy=0, ωz=0)  →  以 1 m/s 向前走
(vx=0, vy=0.5, ωz=0)  →  以 0.5 m/s 向左横移
(vx=0, vy=0, ωz=1.0)  →  原地旋转
```

**机器人参数**（来自 `assets/robots/unitree.py`）：
- 12 个关节（每条腿 3 个：hip、thigh、calf）
- 初始姿态：站立，身体离地 0.4m
- 执行器类型：UnitreeActuatorCfg_Go2HV（刚度 25.0，阻尼 0.5）

## 6.2 环境配置逐项解读

以 Go2 的 `velocity_env_cfg.py` 为例，解读每个组件：

### 场景（Scene）

```python
class RobotSceneCfg:
    terrain = TerrainImporterCfg(...)       # 碎石路地形（可选平地）
    robot = UNITREE_GO2_CFG                  # Go2 机器人模型
    height_scanner = RayCasterCfg(...)       # 地形高度扫描器（176 个射线）
    contact_forces = ContactSensorCfg(...)   # 足底接触力传感器
```

**通俗理解**：搭建 4096 个平行训练场，每个场上放一只机器狗、一片地形，机器狗脚上装触觉传感器、身上装地形扫描器。

### 动作（Actions）

```python
class ActionsCfg:
    joint_pos = JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],     # 所有 12 个关节
        scale=0.25,             # 输出缩放（网络输出 ×0.25 再加上默认角度）
        use_default_offset=True # 以默认站姿为零位
    )
```

**通俗理解**：神经网络输出 12 个数字（范围约 [-1, 1]），乘以 0.25 后加到默认站姿角度上，就是目标关节角度。PD 控制器把关节驱动到目标位置。

### 奖励（Rewards）—— 最核心的部分

Go2 使用 **16 个奖励项**，分为四大类：

```
┌─────────────────────────────────────────────────────────────┐
│                    奖励函数组合（Go2 速度跟踪）               │
│                                                             │
│  ★ 任务目标奖励（正向激励】                                   │
│  ├── track_lin_vel_xy_exp  ×1.5   跟踪 XY 速度指令          │
│  ├── track_ang_vel_z_exp   ×0.75  跟踪旋转速度指令           │
│  └── feet_gait             ×0.5   保持正确步态节奏           │
│                                                             │
│  ★ 基座惩罚（负向抑制）                                      │
│  ├── orientation_l2        ×-2.5  保持身体水平              │
│  ├── base_lin_vel_z_l2     ×-1.5  抑制上下颠簸              │
│  └── base_ang_vel_xy_l2    ×-0.05 抑制翻滚/俯仰            │
│                                                             │
│  ★ 关节惩罚（负向抑制）                                      │
│  ├── energy                ×-2e-5 减少能量浪费              │
│  ├── action_rate_l2        ×-0.01 动作平滑（不要抖）         │
│  ├── joint_acc_l2          ×-2.5e-7 关节加速度平滑           │
│  ├── joint_pos_limits      ×-10.0 避免撞到关节极限           │
│  └── joint_mirror          ×-0.2  左右腿对称（走直线）       │
│                                                             │
│  ★ 接触奖励（步态）                                          │
│  ├── foot_clearance_reward ×0.3   抬脚离地                  │
│  ├── feet_stumble          ×-1.0  惩罚脚碰障碍              │
│  ├── feet_too_near         ×-0.5  两脚不要太近              │
│  └── air_time_variance     ×-1.0  四条腿节奏一致            │
└─────────────────────────────────────────────────────────────┘
```

**通俗理解**：就像老师给学生打分——跟上节奏加分、浪费体力扣分、动作不标准扣分。这套分数体系精心设计，让机器人自然学会高效、稳定、协调的步态。

### 终止条件（Terminations）

```python
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)  # 20 秒超时
    base_contact = DoneTerm(                                 # 身体碰地
        func=mdp.illegal_contact,
        params={"sensor_cfg": ..., "threshold": 1.0}
    )
    bad_orientation = DoneTerm(func=mdp.bad_orientation, ...) # 翻转
```

## 6.3 奖励函数设计哲学

**核心思想**：**不告诉机器人"怎么走"，而是告诉它"走得好是什么样"**。

机器人自己通过试错摸索出最佳策略。这就是强化学习的精髓——你定义"好坏的标准"（奖励），算法自己找"最好的做法"（策略）。

奖励设计的几个关键原则：
1. **正向目标 + 负向约束**：速度跟踪是正奖励（鼓励），能量惩罚是负奖励（抑制）
2. **权重调配**：最重要的用大权重（如速度跟踪 ×1.5），次要的用小权重（如能量 ×-2e-5）
3. **指数核 vs L2 核**：指数核 `exp(-error/σ²)` 对大误差更宽容，L2 `error²` 是严格惩罚

## 6.4 观测空间：机器人能"看到"什么

| 观测项 | 维度 | 给 Policy | 给 Critic | 说明 |
|:---|:---:|:---:|:---:|:---|
| base_ang_vel | 3 | ✅ | ✅ | 身体角速度（带噪声 ±0.2） |
| projected_gravity | 3 | ✅ | ✅ | 重力方向（判断身体是否水平） |
| velocity_commands | 3 | ✅ | ✅ | 当前速度指令 |
| joint_pos_rel | 12 | ✅ | ✅ | 关节角偏离默认位置（带噪声） |
| joint_vel_rel | 12 | ✅ | ✅ | 关节角速度 |
| last_action | 12 | ✅ | ✅ | 上一步的动作 |
| gait_phase | 2 | ✅ | ✅ | sin/cos 步态相位时钟 |
| base_lin_vel | 3 | ❌ | ✅ | 身体线速度（特权信息） |
| joint_effort | 12 | ❌ | ✅ | 关节力矩（特权信息） |

**为什么 Policy 和 Critic 的观测不同？**

这叫**非对称观测**（Asymmetric Actor-Critic）：
- **Policy（Actor）** 只能看到真实传感器能测到的（加了噪声），因为部署时真实机器人只有这些传感器
- **Critic** 可以看到"上帝视角"的特权信息（精确速度、力矩），它只在训练时使用，帮助更准确地估计价值函数

---

# 第七篇 武术模仿任务详解（Mimic）

## 7.1 从动捕数据到机器人打拳

**目标**：让 G1-29dof 人形机器人精确复现 CMU 动捕数据库 Subject #135 的空手道动作。

7 个独立武术动作：

| 动作名 | Task ID 后缀 | CMU 数据 | 时长 |
|:---|:---|:---|:---|
| 平安初段 (Heian Shodan) | HeianShodan | 135_04 | ~11s |
| 前踢 (Front Kick) | FrontKick | 135_03 | ~23s |
| 回旋踢 (Roundhouse Kick) | RoundhouseKick | 135_05 | ~20s |
| 冲拳 (Lunge Punch) | LungePunch | 135_06 | ~26s |
| 侧踢 (Side Kick) | SideKick | 135_07 | ~12s |
| 拔塞 (Bassai) | Bassai | 135_01 | ~51s |
| 燕飞 (Empi) | Empi | 135_02 | ~43s |

## 7.2 数据流水线三阶段

```
阶段 1                   阶段 2                    阶段 3
CMU ASF+AMC              CSV                       NPZ
(人体骨骼动画)     →     (机器人关节角度)     →    (完整物理状态)
                                                     ↓
cmu_amc_to_csv.py        csv_to_npz.py           用于训练
```

### 阶段 1：ASF+AMC → CSV（骨骼映射）

- **输入**：CMU 的 `.asf`（骨骼定义）+ `.amc`（逐帧关节角度）
- **核心工作**：把**人体骨骼**的运动映射到**G1 机器人**的 29 个关节上
- **难点**：人有 50+ 个关节，G1 只有 29 个，需要手动定义对应关系
- **输出**：CSV 文件，每行 = `base_pos(3) + base_quat(4) + joint_angles(29)`

### 阶段 2：CSV → NPZ（物理仿真重放）

这一步**不是简单的格式转换**，而是在 Isaac Lab 中实际运行物理仿真！

- **为什么？** CSV 只有关节角度，没有**速度**信息。通过物理仿真"重放"这些角度，引擎可以自动计算所有刚体的速度、加速度
- **输出**：NPZ 文件包含 `joint_pos`, `joint_vel`, `body_pos_w`, `body_quat_w`, `body_lin_vel_w`, `body_ang_vel_w`
- **参数**：fps=50（训练控制频率）

### 阶段 3：NPZ → 训练

NPZ 文件直接放在任务文件夹内（如 `martial_arts/G1_front_kick.npz`），被 `MotionCommand` 加载使用。

## 7.3 MotionCommand：运动跟踪的大脑

`MotionCommand` 是 Mimic 任务的核心组件，它继承自 Isaac Lab 的 `CommandTerm`，负责：

```
NPZ 文件                              MotionCommand
(参考动作)                              │
   │── MotionLoader（加载到 GPU）        │── 时间步推进（每步 +1）
   │                                      │── 提供当前帧参考状态
   │                                      │── 相对坐标变换（跟随机器人位移）
   │                                      │── 自适应采样（难的地方多练）
   └──────────────────────────────────────└── Debug 可视化（绿色/红色标记）
```

### 关键设计：相对坐标变换

训练时，参考动作会**跟随机器人的 Yaw（偏航）方向和 XY 位置移动**，但保持参考的 Z 高度：

```
参考动作原始位置          机器人实际位置          变换后的参考位置
(0, 0, 0.8)     +     (2.0, 1.0, ?)    →    (2.0, 1.0, 0.8)
                                               跟随 XY，保留参考 Z
```

这样设计的原因：如果参考动作固定在世界坐标，机器人一旦偏移就永远追不上了。相对变换让参考动作"黏在"机器人身旁。

### 关键设计：自适应采样

Episode 重置时，不是从动作开头重新开始，而是根据**历史失败率**对时间点采样——失败多的片段有更大概率被选中练习。这就像学跆拳道的学生，侧踢总摔就多练侧踢。

## 7.4 奖励函数：怎么让机器人打得像

Mimic 任务的奖励函数围绕**全身跟踪精度**设计，使用**高斯核**（指数核）：

```
reward = exp(-error² / σ²)
```

| 奖励项 | σ 值 | 说明 |
|:---|:---:|:---|
| anchor_position_error | 0.3 | 骨盆位置跟踪（根节点） |
| anchor_orientation_error | 0.3 | 骨盆朝向跟踪 |
| body_position_error | 0.5 | 14 个刚体位置跟踪 |
| body_orientation_error | 0.3 | 14 个刚体朝向跟踪 |
| body_linear_velocity_error | 1.0 | 刚体线速度跟踪 |
| body_angular_velocity_error | 0.5 | 刚体角速度跟踪 |
| joint_pos_error | 0.8 | 29 个关节角度跟踪（防止姿态怪异） |

**为什么用高斯核而不是 L2？** 高斯核有上界（最大 1.0），对大误差不会产生巨大的梯度，训练更稳定。

### 惩罚项

```
action_rate_l2    ×-0.01   动作平滑
joint_acc_l2      ×-2.5e-7 关节加速度平滑
```

### 终止条件（比 Locomotion 更严格）

```
time_out              超时（动作时长+缓冲）
bad_anchor_pos_z_only 骨盆高度偏离参考 >0.3m（飞起来或摔倒）
bad_anchor_ori        骨盆朝向偏离参考 >阈值
bad_motion_body_pos   末端执行器位置偏离 >0.5m（手脚位置太离谱）
```

## 7.5 七个独立策略 → 一场完整表演

```
训练阶段（Python）                    部署阶段（C++）
┌──────────────┐                    ┌──────────────────────────┐
│ 独立训练 7 个  │                    │  Policy Sequencer        │
│ ONNX 策略     │     ──导出──►     │                          │
│               │                    │  策略1 → 过渡 → 策略2    │
│ HeianShodan   │                    │  → 过渡 → 策略3 → ...   │
│ FrontKick     │                    │                          │
│ RoundhouseKick│                    │  串联成连续表演            │
│ LungePunch    │                    │  （2026 春晚武术机器人）   │
│ SideKick      │                    └──────────────────────────┘
│ Bassai        │
│ Empi          │
└──────────────┘
```

C++ 的 `State_MartialArtsSequencer` 把多个策略按顺序播放，在策略之间有简短的"保持当前姿态"过渡（`transition_hold_s: 1.0`），确保平滑衔接。

---

# 第八篇 训练全流程：从敲命令到出模型

## 8.1 训练脚本 train.py 逐行解读

```bash
# 训练 Go2 走路（headless 无渲染，更快）
python scripts/rsl_rl/train.py --task Unitree-Go2-Velocity --headless

# 训练 G1 武术-前踢（50000 次迭代）
python scripts/rsl_rl/train.py --task Unitree-G1-29dof-Mimic-MartialArts-FrontKick --headless --max_iterations 50000
```

`train.py` 内部做了什么（对应五大层）：

```
Layer 1 | 启动层
│  1. import unitree_rl_lab.tasks → 自动注册所有任务
│  2. 解析 CLI 参数（task、num_envs、seed…）
│  3. AppLauncher 启动 Isaac Sim（PhysX 引擎）
│
Layer 2 | 配置层
│  4. @hydra_task_config 加载环境配置+PPO配置
│  5. CLI 参数覆盖（num_envs、max_iterations…）
│
Layer 3 | 仿真层
│  6. gym.make("Unitree-Go2-Velocity") → 创建 ManagerBasedRLEnv
│     → 内部自动构建所有 Manager + 加载机器人模型 + 创建地形
│
Layer 4 | 适配层
│  7. RslRlVecEnvWrapper 包装（Gymnasium → RSL-RL 接口）
│  8. OnPolicyRunner 初始化（PPO + Actor-Critic 网络）
│
Layer 5 | 训练层
│  9. runner.learn(30000) → PPO 主循环
│  10. 每次迭代：rollout → compute GAE → PPO update → logging
│  11. 定期保存 checkpoint（model_*.pt）
│  12. 保存配置快照 + export_deploy_cfg(deploy.yaml)
```

## 8.2 PPO 训练循环：1 万次迭代里发生了什么

```
┌─────────────── 一次 PPO 迭代 ═══════════════════════════════┐
│                                                              │
│  Phase 1: Rollout（数据采集）                                 │
│  ┌────────────────────────────────────────────────────────┐  │
│  │ for step in range(24):  # 24 步数据                     │  │
│  │     actions = policy(obs)           # 神经网络推理       │  │
│  │     obs, rewards, dones, infos = env.step(actions)     │  │
│  │     storage.add(obs, actions, rewards, dones, values)  │  │
│  │                                                         │  │
│  │  并行运行 4096 个环境 × 24 步 = 98304 条经验              │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                              │
│  Phase 2: GAE（计算优势函数）                                  │
│  ┌────────────────────────────────────────────────────────┐  │
│  │ advantages = compute_gae(rewards, values, gamma=0.99)  │  │
│  │ returns = advantages + values                          │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                              │
│  Phase 3: PPO Update（策略优化）                              │
│  ┌────────────────────────────────────────────────────────┐  │
│  │ for epoch in range(5):  # 5 轮优化                      │  │
│  │     for batch in mini_batches:                         │  │
│  │         ratio = π_new(a|s) / π_old(a|s)               │  │
│  │         clip_ratio = clip(ratio, 1-ε, 1+ε)            │  │
│  │         policy_loss = -min(ratio*A, clip_ratio*A)      │  │
│  │         value_loss  = (V(s) - returns)²               │  │
│  │         optimizer.step(policy_loss + 0.5*value_loss)   │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                              │
│  Phase 4: Logging（记录日志）                                 │
│  ┌────────────────────────────────────────────────────────┐  │
│  │ 输出到 TensorBoard / WandB：                            │  │
│  │   mean_reward, mean_episode_length, policy_loss, ...   │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                              │
└═════════════════════════════════════════════════════════════┘
```

**PPO 超参数**（Locomotion）：

| 参数 | 值 | 含义 |
|:---|:---|:---|
| num_steps_per_env | 24 | 每次迭代每个环境采集 24 步数据 |
| max_iterations | 30000 | 最大训练迭代次数 |
| policy_network | [512, 256, 128] | Actor 网络结构 |
| value_network | [512, 256, 128] | Critic 网络结构 |
| learning_rate | 1e-3 | 学习率 |
| gamma | 0.99 | 折扣因子 |
| lambda | 0.95 | GAE lambda |
| clip_param | 0.2 | PPO clip 系数 |
| entropy_coef | 0.01 | 熵奖励系数 |

## 8.3 推理脚本 play.py 与模型导出

```bash
# 推理（可视化）
python scripts/rsl_rl/play.py --task Unitree-Go2-Velocity

# 录制视频
python scripts/rsl_rl/play.py --task Unitree-Go2-Velocity --video --video_length 200
```

`play.py` 的关键流程：
1. 使用 `play_env_cfg_entry_point`（32 个环境，关闭随机扰动）
2. 自动寻找最新 checkpoint（或指定 `--checkpoint`）
3. **导出模型**为 JIT (`policy.pt`) 和 ONNX (`policy.onnx`)
4. 推理循环：`obs → policy(obs) → actions → env.step(actions)`

**模型导出路径**：`logs/rsl_rl/<experiment>/<datetime>/exported/policy.onnx`

---

# 第九篇 部署：从仿真到真实机器人

## 9.1 C++ 部署架构总览

```
训练输出                          C++ 部署端
──────────                        ──────────
policy.onnx  ─────────►  ONNX Runtime 推理
deploy.yaml  ─────────►  读取关节映射、PD增益、观测定义
                                │
                         ┌──────▼──────┐
                         │   CtrlFSM   │
                         │   状态机     │
                         └──────┬──────┘
                                │
                    ┌───────────┼───────────┐
                    ▼           ▼           ▼
                 Passive    FixStand    RLBase
                 安全待机    PD站立      RL推理
```

`deploy.yaml` 是训练和部署之间的"桥梁文件"，由 `export_deploy_cfg()` 自动生成，包含：
- `joint_ids_map`：Isaac Lab 关节序号 → SDK 关节序号的映射
- `stiffness` / `damping`：PD 控制增益
- `default_joint_pos`：默认站姿角度
- `actions`：动作空间的 scale、offset、clip
- `observations`：观测空间的定义（scale、clip、history）
- `commands`：速度指令范围

## 9.2 有限状态机 FSM 详解

```cpp
// BaseState 定义了状态接口
class BaseState {
    virtual void enter() {}      // 进入时
    virtual void pre_run() {}    // 每帧前处理
    virtual void run() {}        // 每帧核心逻辑
    virtual void post_run() {}   // 每帧后处理
    virtual void exit() {}       // 退出时
    // 状态转换条件列表
    std::vector<std::pair<std::function<bool()>, int>> registered_checks;
};

// CtrlFSM 以 1kHz 频率运行控制循环
class CtrlFSM {
    void run_() {
        currentState->pre_run();
        currentState->run();
        currentState->post_run();

        // 检查是否需要切换状态
        for (auto& check : currentState->registered_checks) {
            if (check.first()) {  // 条件满足
                currentState->exit();
                currentState = findState(check.second);
                currentState->enter();
                break;
            }
        }
    }
};
```

### 三个核心状态

**State_Passive（被动/安全）**：
- 所有关节力矩清零，机器人自然放松
- 上电后的默认状态，确保安全

**State_FixStand（站立归位）**：
- 使用线性插值器从当前姿态平滑过渡到站立姿态
- 有 PD 增益，关节被驱动到目标位置
- 从配置文件读取 `ts`（时间点）和 `qs`（目标关节角度序列）

**State_RLBase（RL 策略执行）**：
- 启动策略推理线程（与控制线程并行）
- 每 `step_dt` 秒执行一次完整的观测→推理→动作循环
- 设置关节 PD 增益为训练时的值

## 9.3 Policy Sequencer：串联七个武术动作

`State_MartialArtsSequencer` 的工作原理：

```yaml
# deploy 端配置（config.yaml）
MartialArtsSequencer:
  transition_hold_s: 1.0        # 动作之间保持 1 秒
  segments:
    - { policy_dir: "heian_shodan/", motion_file: "...", fps: 50 }
    - { policy_dir: "front_kick/",   motion_file: "...", fps: 50 }
    - { policy_dir: "roundhouse/",   motion_file: "...", fps: 50 }
    # ... 7 个动作按顺序排列
```

执行流程：
1. 加载第 1 个策略（heian_shodan）的 ONNX 文件和对应动作数据
2. 按照动作时长执行策略推理
3. 动作完成后，保持当前姿态 1 秒（过渡期）
4. 加载第 2 个策略（front_kick），继续执行
5. 重复直到所有 7 个动作完成
6. 结束后自动切回 FixStand 状态

安全保护：如果执行过程中检测到姿态异常（`bad_orientation`），立即切回 Passive 状态。

---

# 第十篇 实操速查：我想做 XX 该改哪里？

| 我想做的事情 | 要改的文件 | 改什么 |
|:---|:---|:---|
| **加一个新的奖励函数** | `tasks/locomotion/mdp/rewards.py` | 写一个 `def my_reward(env, ...) -> Tensor` 函数 |
| **给某个任务加上这个奖励** | `robots/go2/velocity_env_cfg.py` | 在 `RewardsCfg` 里加一行 `my_reward = RewTerm(func=..., weight=...)` |
| **调奖励权重** | `robots/go2/velocity_env_cfg.py` | 修改 `RewTerm(..., weight=新权重)` |
| **加一个新的观测项** | `tasks/locomotion/mdp/observations.py` + `velocity_env_cfg.py` | 写函数 + 在 `PolicyCfg` 加 `ObsTerm` |
| **换一个机器人** | `assets/robots/unitree.py` | 定义新的 `UNITREE_XX_CFG` |
| **新增一个任务（如 Go2 跑酷）** | 新建 `robots/go2/parkour_env_cfg.py` + 修改 `__init__.py` | 写 EnvCfg + `gym.register()` |
| **改训练超参数** | `agents/rsl_rl_ppo_cfg.py` | 修改 learning_rate、网络结构等 |
| **改物理仿真步长** | `velocity_env_cfg.py` 的 `__post_init__` | 修改 `self.sim.dt` 和 `self.decimation` |
| **减少训练环境数** | 命令行 `--num_envs 512` 或配置中的 `scene.num_envs` | - |
| **添加 Domain Randomization** | `velocity_env_cfg.py` 的 `EventCfg` | 加 `EventTerm(func=..., mode="interval")` |
| **部署到新机器人** | `deploy/robots/新机器人/` | CMake 项目 + 适配 FSM 状态 |

---

# 附录 A：设计模式速查表

| 模式 | 在哪里 | 一句话说明 | 你什么时候会碰到它 |
|:---|:---|:---|:---|
| **策略模式** | `mdp/rewards.py` / `mdp/observations.py` | 函数可替换 | 每次改奖励/观测都在用 |
| **组合模式** | `ManagerBasedRLEnv` 内的 Manager-Term 树 | 统一接口管理 | 理解 env.step() 时 |
| **工厂方法** | `__init__.py` 里的 `gym.register()` | 字符串建环境 | 新增任务时 |
| **生成器模式** | `@configclass` 的 `*EnvCfg` | 分步组装配置 | 修改环境配置时 |
| **模板方法** | `ManagerBasedRLEnv.step()` | 固定流程骨架 | 理解训练循环时 |
| **适配器模式** | `RslRlVecEnvWrapper` | 转接口格式 | 换训练框架时 |
| **状态模式** | `deploy/include/FSM/` | 行为状态切换 | 部署到真实机器人时 |

---

# 附录 B：关键超参数速查

## Locomotion 训练参数

| 参数 | 值 | 说明 |
|:---|:---|:---|
| num_envs | 4096 | 并行环境数 |
| sim.dt | 0.005 | 物理仿真时间步 (200Hz) |
| decimation | 4 | 控制频率 = 1/(0.005×4) = 50Hz |
| episode_length_s | 20.0 | Episode 最大时长 |
| max_iterations | 30000 | 训练迭代次数 |
| num_steps_per_env | 24 | 每次 rollout 步数 |
| learning_rate | 1e-3 | 学习率 |
| actor_hidden_dims | [512, 256, 128] | Actor 网络 |
| critic_hidden_dims | [512, 256, 128] | Critic 网络 |

## Mimic (武术) 训练参数

| 参数 | 值 | 说明 |
|:---|:---|:---|
| num_envs | 4096 | 并行环境数 |
| sim.dt | 0.005 | 物理仿真时间步 |
| decimation | 4 | 控制频率 50Hz |
| max_iterations | 50000 | 训练迭代次数（比 Locomotion 多） |
| num_steps_per_env | 48 | 每次 rollout 步数（动作更长） |
| adaptive_lambda | 0.95 | 自适应采样衰减系数 |

---

# 附录 C：常见问题 FAQ

### Q1: 为什么 play.py 启动就退出了？
**A**: 最常见原因是找不到 checkpoint。检查 `logs/rsl_rl/<experiment>/` 目录下是否有 `model_*.pt` 文件。可以用 `--checkpoint <path>` 明确指定。

### Q2: 新增一个任务需要改 train.py 吗？
**A**: 不需要。只需：① 新建任务文件夹 ② 写 `velocity_env_cfg.py` ③ 在 `__init__.py` 里 `gym.register()` ④ 直接 `--task 新任务名` 训练。train.py 通过工厂方法自动发现。

### Q3: 怎么理解 decimation=4？
**A**: 物理仿真以 200Hz 运行（dt=0.005），但神经网络以 50Hz 决策（200/4=50）。也就是说，网络输出一个动作，物理引擎执行 4 步后，网络才做下一次决策。这是为了在真实性和计算效率之间取平衡。

### Q4: num_envs=4096 是什么意思？
**A**: 同时在 GPU 上跑 4096 个独立的仿真环境，每个环境里有一个机器人独立训练。这是 Isaac Lab 的核心优势——大规模并行仿真，一次采集 4096 条经验。

### Q5: Policy 和 Critic 观测不一样合理吗？
**A**: 完全合理，这叫**非对称 Actor-Critic**。Critic 在训练时使用特权信息（如精确的基座速度），帮助更准确地估计价值函数。部署时只用 Policy（Actor），它只依赖真实传感器数据。

### Q6: 怎么从零新增一个机器人？
需要三步：
1. **资产定义**：在 `assets/robots/unitree.py` 添加新的 `UNITREE_XX_CFG`（关节名、刚度、阻尼、初始姿态）
2. **任务配置**：在 `tasks/locomotion/robots/` 下新建文件夹，写 `velocity_env_cfg.py`（选择合适的奖励、观测、传感器配置）
3. **注册任务**：在 `__init__.py` 中 `gym.register("Unitree-XX-Velocity", ...)`

### Q7: 训练好的模型怎么部署到真实机器人？
1. `play.py` 自动导出 `policy.onnx` 和 `deploy.yaml`
2. 在 `deploy/robots/<robot>/` 创建 C++ 项目
3. 配置 `config.yaml`（引用 policy 路径和 deploy.yaml 参数）
4. 编译运行，使用 FSM 控制：Passive → FixStand → RLBase

### Q8: 设计模式看不懂怎么办？
推荐学习路径：
1. 先阅读 [Refactoring Guru](https://refactoringguru.cn/design-patterns/catalog) 上的通俗解释
2. 再对照本手册第五篇的代码+比喻
3. 重点理解**策略模式**和**组合模式**——这两个在日常开发中最常碰到
4. 项目的 `doc/DESIGN_PATTERNS_TUTORIAL.md` 有更详细的代码级教程

---

> **最后总结**：整个项目的设计哲学可以用一句话概括——**"让你只关注奖励怎么调、观测怎么加，不碰底层引擎代码"**。7 个设计模式协同工作，把一个极其复杂的"从仿真到实机"系统，变成了"填配置表+写小函数"的简单操作。掌握了这个框架，你就拥有了训练任意宇树机器人完成任意任务的能力。
