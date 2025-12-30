# Unitree RL Lab 文档导航指南

这是为 Unitree RL Lab 项目编写的完整文档库。选择适合你的文档开始学习。

---

## 🚀 快速开始（5分钟）

### 完全新手？从这里开始

1. **读这个** → `06_Quick_Reference_and_Decision_Tree.md` 的前两部分
   - 了解核心命令
   - 决策树帮你选择合适的机器人

2. **跑这个** 
   ```bash
   python scripts/list_envs.py  # 看有什么任务
   ./unitree_rl_lab.sh -p --task Unitree-Go2-Velocity  # 看演示
   ```

3. **接下来** → 继续下一部分

---

## 📚 完整文档库

### 第1层：项目概览
📄 **`05_Mastering_Unitree_RL_Lab.md`**
- **适合**：想系统学习项目的人
- **内容**：10阶段学习路线（从地形图到C++部署）
- **特点**：详细但需要2-3周时间
- **建议**：打印或分阶段阅读

### 第2层：快速参考
📄 **`06_Quick_Reference_and_Decision_Tree.md`**
- **适合**：想快速查询的人
- **内容**：决策树、快速命令、一页纸总结
- **特点**：即插即用，无需完整阅读
- **建议**：修改代码前快速查看

### 第3层：机器人对比
📄 **`07_G1_vs_H1_Detailed_Comparison.md`**
- **适合**：需要在 Go2/H1/G1 间选择的人
- **内容**：详细的参数对比、决策树、改装指南
- **特点**：表格多、对比清晰
- **建议**：选定机器人后深入阅读

---

## 📖 按任务选择文档

### 任务 A：我想快速验证环境是否安装正确

**推荐阅读顺序**：
1. `06_Quick_Reference_and_Decision_Tree.md` → "快速诊断" 部分
2. `05_Mastering_Unitree_RL_Lab.md` → 第二阶段（动手实验）

**命令**：
```bash
# 1. 查看任务列表
python scripts/list_envs.py

# 2. 快速训练 50 步（验证环境）
python scripts/rsl_rl/train.py --task Unitree-Go2-Velocity \
    --headless --max_iterations=50
```

**预期**：看到终端打印 Reward 数值，说明环境正常

---

### 任务 B：我想从零开始学习这个项目

**推荐阅读顺序**：
1. `06_Quick_Reference_and_Decision_Tree.md` → "快速参考卡" + "决策树"
2. `05_Mastering_Unitree_RL_Lab.md` → 第1-4阶段（打好基础）
3. `07_G1_vs_H1_Detailed_Comparison.md` → 选定主要学习的机器人
4. `05_Mastering_Unitree_RL_Lab.md` → 第5-9阶段（深入代码）

**预计耗时**：1-2 周

**关键实验**：
- Week 1：跑通 Go2 Velocity，修改一个权重参数
- Week 2：切换到 H1 或 G1，对比行为变化

---

### 任务 C：我想立刻修改参数并看到效果

**推荐阅读**：
1. `06_Quick_Reference_and_Decision_Tree.md` → "文件导航速查表"
2. `06_Quick_Reference_and_Decision_Tree.md` → "快速实验模板 A"

**步骤**：
```bash
# 1. 找到配置文件（Go2 为例）
# source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/go2/velocity_env_cfg.py

# 2. 修改 weight=1.0 为 weight=2.0

# 3. 训练
python scripts/rsl_rl/train.py --task Unitree-Go2-Velocity --headless

# 4. 观看效果
./unitree_rl_lab.sh -p --task Unitree-Go2-Velocity
```

**预计耗时**：10 分钟

---

### 任务 D：我想学习舞蹈模仿（Mimic）

**推荐阅读顺序**：
1. `06_Quick_Reference_and_Decision_Tree.md` → "决策树"
2. `05_Mastering_Unitree_RL_Lab.md` → 第4阶段（Mimic 任务详解）
3. `05_Mastering_Unitree_RL_Lab.md` → 第6阶段（参考动作数据）
4. `07_G1_vs_H1_Detailed_Comparison.md` → "为什么只有 G1？"

**推荐任务**：`Unitree-G1-29dof-Mimic-Gangnanm-Style`（江南 Style 舞蹈）

**关键代码文件**：
- `tasks/mimic/robots/g1_29dof/gangnanm_style/G1_gangnam_style_V01.bvh_60hz.npz` - 动作数据
- `tasks/mimic/robots/g1_29dof/gangnanm_style/tracking_env_cfg.py` - 环境配置

**预计耗时**：2-3 小时学习 + 2-4 小时训练

---

### 任务 E：我想部署到真机或 C++ 环境

**推荐阅读顺序**：
1. `05_Mastering_Unitree_RL_Lab.md` → 第8阶段（C++ 部署入门）
2. `deploy/` 目录下的各机器人的 README（如果有）
3. `deploy/robots/<robot_name>/CMakeLists.txt` - 编译配置

**关键文件**：
- `deploy/include/unitree_articulation.h` - 机器人接口
- `deploy/include/FSM/State_RLBase.h` - RL 策略执行
- `deploy/robots/<robot_name>/main.cpp` - 程序入口

**预计耗时**：3-5 小时（假设已有训练好的模型）

---

### 任务 F：我想创建全新的任务

**推荐阅读顺序**：
1. `05_Mastering_Unitree_RL_Lab.md` → 第3-4阶段（理解配置）
2. `06_Quick_Reference_and_Decision_Tree.md` → "快速实验模板"
3. 现有任务源码（作为模板）

**步骤**：
1. 复制现有任务目录
2. 修改 `velocity_env_cfg.py` 或 `tracking_env_cfg.py`
3. 在 `__init__.py` 中注册新任务
4. 重新训练并测试

**预计耗时**：2-3 小时

---

## 🎯 按知识点快速查询

### "我想理解..."

| 知识点 | 文档 | 位置 |
|-------|-----|------|
| ...MDP 是什么 | 05 | 第三阶段 |
| ...奖励函数如何工作 | 05 | 第三阶段 → 3.1 → RewardsCfg |
| ...观测空间是什么 | 05 | 第三阶段 → 3.1 → ObservationsCfg |
| ...参考动作如何加载 | 05 | 第六阶段 |
| ...两个机器人的区别 | 07 | 详细参数对比 |
| ...怎么修改地形 | 06 | 文件导航速查表 |
| ...怎么创建新任务 | 05 | 第九阶段项目C |
| ...怎么部署到真机 | 05 | 第八阶段 |
| ...训练为什么这么慢 | 06 | 快速诊断 → 问题4 |
| ...Reward 为什么不上升 | 06 | 快速诊断 → 问题3 |

---

## 💡 推荐的学习时间表

### 周期 1：基础阶段（第1-2周）

**目标**：能跑通完整流程，理解基本概念

**Day 1**：
- 读 `06_Quick_Reference_and_Decision_Tree.md` (30 min)
- 运行 `list_envs.py` (5 min)
- 运行 `train.py --max_iterations=50` (15 min)

**Day 2-3**：
- 读 `05_Mastering_Unitree_RL_Lab.md` 第1-2阶段 (1 h)
- 完成"实验 A"和"实验 B" (1 h 训练)

**Day 4-5**：
- 读 `05_Mastering_Unitree_RL_Lab.md` 第3阶段 (1 h)
- 在代码中找到 `RewardsCfg` 和 `ObservationsCfg` (30 min)

**Day 6-7**：
- 完成"实验项目 A"：修改奖励权重 (1-2 h)

### 周期 2：进阶阶段（第3-4周）

**目标**：能修改配置、创建新任务、理解机器人差异

**Day 1-2**：
- 读 `07_G1_vs_H1_Detailed_Comparison.md` (1 h)
- 选定要深入学习的机器人 (H1 or G1)

**Day 3-4**：
- 读 `05_Mastering_Unitree_RL_Lab.md` 第4-6阶段 (1.5 h)
- 在所选机器人上运行训练 (2-4 h)

**Day 5-7**：
- 完成"实验项目 B"或"项目 C" (3-5 h)

### 周期 3+：专项深化（可选）

选择以下任一方向：
- **算法研究**：修改 PPO 或创建新奖励函数
- **部署**：学习第8阶段，编译 C++ 代码
- **舞蹈**：学习 Mimic 任务，创建新舞蹈

---

## 📋 核心文件清单

### 配置文件（你会经常改这些）

```
source/unitree_rl_lab/unitree_rl_lab/
├── tasks/
│   ├── locomotion/
│   │   ├── robots/
│   │   │   ├── go2/velocity_env_cfg.py      ← Go2 环境定义
│   │   │   ├── h1/velocity_env_cfg.py        ← H1 环境定义
│   │   │   └── g1/29dof/velocity_env_cfg.py  ← G1 环境定义
│   │   └── mdp/
│   │       ├── rewards.py                    ← 奖励函数库
│   │       ├── observations.py               ← 观测函数库
│   │       └── commands.py                   ← 命令定义
│   └── mimic/
│       ├── robots/g1_29dof/
│       │   ├── gangnanm_style/tracking_env_cfg.py  ← 江南Style配置
│       │   ├── dance_102/tracking_env_cfg.py       ← 舞蹈102配置
│       │   └── petite_verses/tracking_env_cfg.py   ← 小诗歌配置
│       └── mdp/
│           ├── rewards.py
│           └── observations.py
└── assets/robots/
    └── unitree.py                            ← 机器人物理参数
```

### 脚本文件（主要使用这些）

```
scripts/
├── train.py     ← 训练脚本
├── play.py      ← 推理脚本
└── cli_args.py  ← 命令行参数处理
```

### 部署文件（C++ 相关）

```
deploy/
├── include/     ← C++ 头文件
└── robots/      ← 各机器人实现
    ├── go2/main.cpp
    ├── h1/main.cpp
    └── g1_29dof/main.cpp
```

---

## 📞 困境排查流程

**我的训练不工作！**

→ 按顺序检查：

1. 环境是否安装正确？
   - 查看：`06_Quick_Reference_and_Decision_Tree.md` → 快速诊断 → 问题1

2. Reward 是否在上升？
   - 查看：`06_Quick_Reference_and_Decision_Tree.md` → 快速诊断 → 问题3

3. 训练速度是否太慢？
   - 查看：`06_Quick_Reference_and_Decision_Tree.md` → 快速诊断 → 问题4

4. 机器人在仿真中行为怪异？
   - 查看：`06_Quick_Reference_and_Decision_Tree.md` → 快速诊断 → 问题5

5. 仍未解决？
   - 查看：`05_Mastering_Unitree_RL_Lab.md` → 附录 FAQ

---

## 📊 文档关系图

```
快速开始 (5 min)
    ↓
06_Quick_Reference.md (10 min)
    ├─ 决策树 → 选机器人
    ├─ 快速命令 → 跑起来
    └─ 快速诊断 → 解决问题
    ↓
05_Mastering.md (2-3 weeks)
    ├─ 第1-2阶段：体验
    ├─ 第3-4阶段：理解代码
    ├─ 第5-6阶段：执行流
    ├─ 第7-8阶段：机器人/部署
    └─ 第9-10阶段：深化
    ↓
07_G1_vs_H1.md (如需对比)
    └─ 机器人详细参数
    └─ 改装指南
    ├─ 选择机器人
    └─ 建议学习路径
```

---

## 🎓 预计学习耗时

| 目标 | 推荐文档 | 耗时 | 难度 |
|-----|--------|------|------|
| 验证环境安装 | 06 | 15 min | ⭐ |
| 跑通第一个任务 | 06 + 05(1-2阶) | 1 h | ⭐ |
| 理解项目框架 | 05 | 4-6 h | ⭐⭐ |
| 创建新任务 | 05 + 06 | 3-5 h | ⭐⭐ |
| 完全掌握项目 | 所有文档 | 2-3 weeks | ⭐⭐⭐ |
| 部署到真机 | 05(8阶) + deploy/ | 3-5 h (+ 训练) | ⭐⭐⭐⭐ |

---

## 💾 文件大小参考

```
05_Mastering_Unitree_RL_Lab.md      ~50 KB  (10,000+ 字)
06_Quick_Reference_and_Decision_Tree.md  ~40 KB  (8,000+ 字)
07_G1_vs_H1_Detailed_Comparison.md  ~45 KB  (9,000+ 字)
```

**建议**：先快速看 06（可打印）做快速参考，再选择性阅读 05 和 07

---

## 🔗 推荐外部资源

- **IsaacLab 官方**：https://isaac-sim.github.io/IsaacLab/main
- **RSL-RL**：https://github.com/leggedrobotics/rsl_rl
- **强化学习基础**：Sutton & Barto 书籍 / Berkeley CS285 课程
- **PPO 论文**：https://arxiv.org/abs/1707.06347

---

## ✅ 阅读完成检查清单

阅读完 `05_Mastering_Unitree_RL_Lab.md` 后，你应该能：

- [ ] 解释什么是 MDP（Observation, Action, Reward）
- [ ] 打开配置文件，找到 `RewardsCfg` 和 `ObservationsCfg`
- [ ] 修改一个奖励权重参数并重新训练
- [ ] 对比两个不同机器人的配置差异
- [ ] 理解为什么 Mimic 任务只有 G1
- [ ] 能够用 TensorBoard 查看训练日志
- [ ] 知道如何创建一个新的任务（至少在概念上）

---

## 🤝 贡献与反馈

如果你发现文档有错误或不清楚的地方，欢迎提出建议！

常见的反馈：
- 某部分代码示例过时
- 某个概念解释不清楚
- 想要更多实际例子
- 希望添加新的主题

---

**祝你学习愉快！选择一个文档开始阅读吧！** 🚀
