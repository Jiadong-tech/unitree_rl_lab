# Unitree RL Lab：快速参考与决策树

## 快速参考卡（打印版）

### 核心命令

```bash
# 查看所有可用任务
python scripts/list_envs.py

# 训练
./unitree_rl_lab.sh -t --task <TaskName> --headless

# 推理（查看效果）
./unitree_rl_lab.sh -p --task <TaskName>

# 查看日志
tensorboard --logdir logs/rsl_rl/
```

### 文件导航速查表

| 我想做... | 打开这个文件 | 修改什么 |
|---------|-----------|--------|
| 修改走路速度范围 | `tasks/locomotion/mdp/commands.py` | `vel_command_range` |
| 调整奖励权重 | `tasks/locomotion/robots/<robot>/velocity_env_cfg.py` | `RewardsCfg` 中的 `weight` |
| 添加新的奖励函数 | `tasks/locomotion/mdp/rewards.py` | 新函数 + `RewardsCfg` 中注册 |
| 改变观测空间 | `tasks/locomotion/robots/<robot>/velocity_env_cfg.py` | `ObservationsCfg` |
| 修改机器人物理参数 | `source/unitree_rl_lab/assets/robots/unitree.py` | `UNITREE_*_CFG` 中的 `stiffness`, `damping` |
| 改地形 | `tasks/locomotion/robots/<robot>/velocity_env_cfg.py` | `RobotSceneCfg.terrain` |
| 创建新 Mimic 任务 | 复制 `tasks/mimic/robots/g1_29dof/gangnanm_style/` | 修改 `tracking_env_cfg.py` 和 `__init__.py` |

---

## 决策树：我应该做什么？

```
┌─ 我是完全新手吗？
│  ├─ YES → 跟着 05_Mastering_Unitree_RL_Lab.md 的前两阶段
│  │        （先跑 Play 和 Train 看看效果）
│  └─ NO  → 继续下一步
│
├─ 我想快速验证一个想法吗？
│  ├─ YES → 使用 Go2 Velocity 任务
│  │        (最快，12 DOF)
│  └─ NO  → 继续下一步
│
├─ 我想做舞蹈/动作模仿吗？
│  ├─ YES → 使用 G1-29DOF Mimic 任务
│  │        (3 种舞蹈可选，需要较长训练)
│  └─ NO  → 继续下一步
│
├─ 我想做通用走路策略吗？
│  ├─ YES → H1 Velocity 任务
│  │        (人形，中等复杂度)
│  └─ NO  → 继续下一步
│
└─ 我想修改代码吗？
   ├─ 想改奖励函数
   │  └─ 打开 tasks/locomotion/mdp/rewards.py
   │     修改 RewardsCfg 中的 weight 参数
   │
   ├─ 想改观测空间
   │  └─ 打开 tasks/locomotion/robots/<robot>/velocity_env_cfg.py
   │     修改 ObservationsCfg
   │
   └─ 想改机器人物理参数
      └─ 打开 source/unitree_rl_lab/assets/robots/unitree.py
         修改 UNITREE_*_CFG 中的 stiffness/damping
```

---

## 快速诊断：遇到问题怎么办？

### 问题 1：程序启动报错

```
ModuleNotFoundError: No module named 'isaaclab'
```
**解决**：确认 Isaac Lab 环境已激活
```bash
conda activate env_isaaclab
```

---

### 问题 2：没有预训练模型（Play 失败）

```
FileNotFoundError: No checkpoint found in ...
```
**解决**：首先训练一个模型
```bash
python scripts/rsl_rl/train.py --task Unitree-Go2-Velocity \
    --headless --max_iterations=50  # 快速测试，只训练 50 步
```

---

### 问题 3：Reward 一直是 0 或负数

**可能原因**：
1. 观测空间设置错误 → 检查 `ObservationsCfg`
2. 奖励权重全为 0 → 检查 `RewardsCfg` 中的 `weight`
3. 终止条件太严格 → 机器人一开始就摔倒 → 检查 `TerminationsCfg`

**快速诊断**：
```python
# 打印观测向量的维度和值
from isaaclab.envs import gym
env = gym.make("Unitree-Go2-Velocity")
obs, _ = env.reset()
print(f"Observation shape: {obs.shape}")
print(f"Observation sample: {obs[0]}")
```

---

### 问题 4：训练速度太慢

**可能原因**：
1. 没有用 `--headless` → **加上它**
   ```bash
   python scripts/rsl_rl/train.py --task ... --headless
   ```
2. 用了 Go2 之外的机器人 → **切换到 Go2**（最快）
3. GPU 没有用上 → 检查 CUDA/GPU 状态
   ```bash
   nvidia-smi  # 查看 GPU 使用
   ```

---

### 问题 5：机器人在仿真中走不稳定

**可能原因**：
1. 奖励函数不合理 → 调整权重
2. 机器人参数不对 → 检查 `stiffness` 和 `damping`
   - 太软：`stiffness` 太低 → 机器人无力
   - 太硬：`damping` 太低 → 抖动
3. 学习率太高 → 策略不稳定 → 降低学习率

---

## 快速实验模板

### 实验 A：修改单个奖励权重

```python
# 1. 打开配置文件
# source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/go2/velocity_env_cfg.py

# 2. 找到这段代码：
class RewardsCfg:
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=1.0,  # <-- 改这里
        params={...}
    )

# 3. 尝试不同的值：
# weight=0.5   # 降低速度的重要性
# weight=2.0   # 提高速度的重要性

# 4. 重新训练并对比
python scripts/rsl_rl/train.py --task Unitree-Go2-Velocity --headless --max_iterations=100
```

### 实验 B：对比两个机器人的训练速度

```bash
# Go2（12 DOF，最快）
time python scripts/rsl_rl/train.py --task Unitree-Go2-Velocity --headless --max_iterations=100

# H1（20 DOF，中等）
time python scripts/rsl_rl/train.py --task Unitree-H1-Velocity --headless --max_iterations=100
```

### 实验 C：可视化参考动作（Mimic 任务）

```python
import numpy as np
import matplotlib.pyplot as plt

# 加载舞蹈数据
data = np.load('source/unitree_rl_lab/unitree_rl_lab/tasks/mimic/robots/g1_29dof/gangnanm_style/G1_gangnam_style_V01.bvh_60hz.npz')
motion = data['motion']  # shape: (N_frames, N_joints)

# 绘制第一个关节的角度变化
plt.plot(motion[:, 0])
plt.xlabel('Frame')
plt.ylabel('Joint Angle (rad)')
plt.title('G1 Gangnam Style - Joint 0 Trajectory')
plt.show()
```

---

## 调参指南

### 当机器人走得太慢时

```python
# 1. 增加速度奖励权重
track_lin_vel_xy_exp: weight = 2.0  (从 1.0 改为 2.0)

# 2. 或者减少惩罚项
# 如 action_rate_l2 的权重
action_rate_l2: weight = -0.001  (从 -0.01 改为 -0.001)

# 3. 或者扩大目标速度范围
# 在 tasks/locomotion/mdp/commands.py 中
lin_vel_x: (0.0, 3.0)  # 改为更大的范围
```

### 当机器人走得不稳定时

```python
# 1. 增加平衡奖励
base_height_l2: weight = 0.5  # 维持躯干高度

# 2. 增加机器人刚度（更难弯曲）
# 在 source/unitree_rl_lab/assets/robots/unitree.py 中
stiffness=30.0  # 从 25.0 改为 30.0

# 3. 增加阻尼（阻止摆动）
damping=1.0  # 从 0.5 改为 1.0
```

### 当机器人站不起来时

```python
# 1. 调整初始姿态（更接近站立）
init_state = ArticulationCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.5),  # z 坐标增加（更高）
    joint_pos={
        ".*_hip_joint": 0.0,   # 腰更直
        ".*_knee_joint": 0.5,  # 膝盖弯曲
        ".*_calf_joint": -1.0, # 小腿向下
    }
)

# 2. 给一个更温和的终止条件
base_height_lower_than_threshold: threshold = 0.15  # 从 0.2 改为 0.15
# （允许机器人更低）
```

---

## TensorBoard 日志解读

```bash
# 启动 TensorBoard
tensorboard --logdir logs/rsl_rl/

# 打开浏览器访问 http://localhost:6006
```

**关键指标**：

| 指标 | 含义 | 好的迹象 |
|-----|------|--------|
| Cumulative Reward | 累积奖励 | 单调上升 |
| Episode Length | 单个 episode 长度 | 逐渐增加 |
| Success Rate | 成功率 | 从 0% 升到接近 100% |
| Policy Loss | 策略损失 | 逐渐减小 |
| Value Loss | 价值函数损失 | 逐渐减小 |

---

## 一页纸总结：核心概念

### 1. **MDP (Markov Decision Process)**
```
State (观测) → Agent → Action (动作) → Environment → Reward (奖励)
                  ↑                          ↓
                  └──────────────────────────┘
```

### 2. **RL 的三个核心问题**
```
Observation (我看到什么?)       → 定义在 observations.py
Action      (我能做什么?)       → 由神经网络输出（无需手动定义）
Reward      (什么是成功?)       → 定义在 rewards.py
```

### 3. **训练流程**
```
1. 初始化环境和神经网络
2. 循环：
   a. 获取观测 obs
   b. 神经网络推理 → 输出动作
   c. 环境执行动作 → 返回 obs, reward, done
   d. 累积数据用于更新神经网络参数
3. 保存最优模型
```

### 4. **如何修改智能体的行为**
```
改 Observations → 智能体看到的信息不同
改 Rewards     → 智能体的目标不同（最直接有效）
改 Actions     → 改神经网络输出的维度（通常不改）
改 Environment → 改地形、物理参数等
```

---

## 下一步行动清单

- [ ] 阅读 `05_Mastering_Unitree_RL_Lab.md` 前两个阶段
- [ ] 运行 `./unitree_rl_lab.sh -p --task Unitree-Go2-Velocity` 看现有模型
- [ ] 运行 `python scripts/rsl_rl/train.py --task Unitree-Go2-Velocity --headless --max_iterations=50` 体验训练
- [ ] 打开一个配置文件，找到 `RewardsCfg` 类，理解各个奖励项
- [ ] 修改一个权重参数，重新训练，观察行为变化
- [ ] 在 `05_Mastering_Unitree_RL_Lab.md` 中选择一个实践项目完成
- [ ] 深入阅读选定机器人的物理参数配置

---

## 进阶资源

- **IsaacLab 官方文档**：https://isaac-sim.github.io/IsaacLab/main
- **RSL-RL（训练算法）**：https://github.com/leggedrobotics/rsl_rl
- **PPO 算法论文**：https://arxiv.org/abs/1707.06347
- **强化学习入门**：Sutton & Barto《Reinforcement Learning》

---

## 常见修改清单

### 快速调参模板

```python
# 在 velocity_env_cfg.py 中

class RewardsCfg:
    # 模板：(函数, 权重, 参数)
    
    # 原始奖励（跟踪目标）
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=1.0,        # ← 调这个数字 (1.0 = 重要, 0.1 = 不重要)
        params={
            "std_xy": 0.5,  # ← 或调这个
            "command_name": "base_vel_cmd",
        },
    )
    
    # 惩罚项（什么不应该做）
    action_rate_l2 = RewTerm(
        func=mdp.action_rate_l2,
        weight=-0.01,      # ← 负数 = 惩罚
        params={},
    )
```

### 检查清单

运行训练前：
- [ ] 是否用了 `--headless`？
- [ ] 是否用了最简单的任务（Go2）来测试？
- [ ] 奖励函数的权重加起来是否合理？
- [ ] 观测维度是否过大（可能导致训练很慢）？

训练过程中：
- [ ] Reward 是否在上升？
- [ ] Episode 长度是否在增加？
- [ ] GPU 是否被充分利用（>50%）？

训练完后：
- [ ] 推理时机器人的行为是否符合预期？
- [ ] 是否尝试不同权重的对比？
