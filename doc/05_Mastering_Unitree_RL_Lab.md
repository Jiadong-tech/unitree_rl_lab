# 掌握 Unitree RL Lab 的完整指南

## 概述

Unitree RL Lab 是一个基于 **IsaacLab** 的强化学习框架，支持多个 Unitree 机器人（Go2、H1、G1）在仿真环境中训练 RL 策略，并将其部署到真实机器人。

该项目包含两大类任务：
- **Locomotion（足式运动）**：让机器人学会在复杂地形上走路
- **Mimic（动作模仿）**：让机器人学会模仿人类的特定动作（如江南Style舞蹈）

---

## 第一阶段：项目地形图 (15分钟)

### 1.1 核心目录结构

```
unitree_rl_lab/
├── source/unitree_rl_lab/          # 主Python包
│   └── unitree_rl_lab/
│       ├── tasks/                  # RL环境定义（核心）
│       │   ├── locomotion/         # 走路任务
│       │   └── mimic/              # 舞蹈模仿任务
│       ├── assets/                 # 机器人模型配置
│       └── ...
├── scripts/                         # 训练和推理脚本
│   └── rsl_rl/
│       ├── train.py               # 训练脚本
│       ├── play.py                # 推理脚本
│       └── cli_args.py            # 参数处理
├── deploy/                         # C++ 部署代码
│   └── robots/
│       ├── go2/, h1/, g1_*/       # 各机器人的实现
│       └── include/               # C++ 头文件
└── doc/                           # 文档
```

### 1.2 快速了解：三个机器人的对比

| 维度 | Go2 | H1 | G1-29DOF |
|-----|-----|-----|----------|
| **形态** | 四足犬型 | 人形(躯干+四肢+头) | 人形(躯干+四肢+双臂) |
| **DOF (自由度)** | 12 | 20 | 29 |
| **关键特点** | 轻便、快速、敏捷 | 平衡更好、有躯干转动 | 最灵活、双臂可操作 |
| **适用任务** | 速度控制、越野 | 通用走路、平衡 | 复杂模仿、舞蹈 |
| **计算复杂度** | 低 | 中 | 高 |
| **适合新手** | ✓ | ✓ | 后来阶段 |

**初期建议**：从 Go2 的 Velocity 任务开始学习 RL 环境结构。

---

## 第二阶段：动手实验 - 跑通完整流程 (1小时)

### 2.1 环境验证
首先确保环境安装正确：
```bash
cd /home/jiadong/unitree_rl_lab
python scripts/list_envs.py
```

你应该看到 6 个可用的任务：
1. Unitree-G1-29dof-Velocity
2. Unitree-Go2-Velocity
3. Unitree-H1-Velocity
4. Unitree-G1-29dof-Mimic-Dance-102
5. Unitree-G1-29dof-Mimic-Gangnanm-Style
6. Unitree-G1-29dof-Mimic-Petite-Verses

### 2.2 实验 A：Play（推理）- 看看成品长什么样

```bash
# 运行 Go2 走路推理（如果有预训练模型）
./unitree_rl_lab.sh -p --task Unitree-Go2-Velocity

# 或者使用完整命令
python scripts/rsl_rl/play.py --task Unitree-Go2-Velocity
```

**预期结果**：看到 Isaac Sim 窗口，机器人在平地上走动。

**如果报错**：说明缺少预训练模型，这是正常的。继续下一步训练一个。

### 2.3 实验 B：Train（训练）- 让机器人自己学习

```bash
# 训练 Go2，headless 模式（不显示画面，更快）
./unitree_rl_lab.sh -t --task Unitree-Go2-Velocity --headless

# 或使用完整命令，训练 50 iterations 快速验证
python scripts/rsl_rl/train.py --task Unitree-Go2-Velocity --headless --max_iterations=50
```

**观察内容**：
- 终端会打印每个 iteration 的 FPS、Reward、Success Rate 等
- 日志存储在 `logs/rsl_rl/<TaskName>/<Timestamp>/`
- 模型参数存储在 `logs/rsl_rl/<TaskName>/<Timestamp>/models/`

**这个阶段的收获**：
- 理解"环境定义 → 训练 → 模型保存"的完整流程
- 看到 reward 随着训练逐渐增加（如果一切正常）

### 2.4 验证刚训练的模型

```bash
# 用刚训练的最新模型推理
python scripts/rsl_rl/play.py --task Unitree-Go2-Velocity
```

机器人应该会走得比随机初始化的版本好。

---

## 第三阶段：深入代码核心 - 环境配置 (2小时)

### 3.1 任务配置文件详解

任务定义的入口是 **`tracking_env_cfg.py`** 或 **`velocity_env_cfg.py`**。

以 Go2 Velocity 为例，打开：
```
source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/go2/velocity_env_cfg.py
```

**关键类（按执行顺序）**：

1. **`RobotSceneCfg`** - 场景定义
   ```python
   class RobotSceneCfg(InteractiveSceneCfg):
       # 定义地形
       terrain = TerrainImporterCfg(...)
       
       # 定义机器人
       robot = ArticulationCfg = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
       
       # 定义传感器（高度扫描、接触传感器）
       height_scanner = RayCasterCfg(...)
       contact_forces = ContactSensorCfg(...)
   ```

2. **`EventCfg`** - 环境事件（重置、随机化）
   ```python
   class EventCfg:
       # 重置时随机化摩擦系数
       physics_material = EventTerm(...)
       
       # 训练中随机化重力
       gravity = EventTerm(...)
   ```

3. **`ObservationsCfg`** - 机器人"看到"什么
   ```python
   class ObservationsCfg:
       obs_policy = ObsGroup(
           observations=[
               ObsTerm(func=mdp.base_lin_vel, ...),     # 线速度
               ObsTerm(func=mdp.base_ang_vel, ...),     # 角速度
               ObsTerm(func=mdp.projected_gravity, ...), # 重力方向
               ObsTerm(func=mdp.height_scan, ...),      # 地形高度
               ...
           ]
       )
   ```

4. **`RewardsCfg`** - 机器人获得什么奖励
   ```python
   class RewardsCfg:
       # 向目标速度方向移动获得奖励
       track_lin_vel_xy_exp = RewTerm(
           func=mdp.track_lin_vel_xy_exp,
           weight=1.0,
           params={...}
       )
       
       # 角速度过大受罚
       ang_vel_xy_l2 = RewTerm(
           func=mdp.ang_vel_xy_l2,
           weight=-0.05,
           params={...}
       )
   ```

5. **`TerminationsCfg`** - 什么时候结束一个 episode
   ```python
   class TerminationsCfg:
       # 机器人摔倒（腹部高度太低）
       base_height = DoneTerm(
           func=mdp.base_height_below_threshold,
           params={"threshold": 0.2, ...}
       )
   ```

6. **`RobotEnvCfg`** - 把所有上面的配置组合起来
   ```python
   @configclass
   class RobotEnvCfg(ManagerBasedRLEnvCfg):
       scene: RobotSceneCfg = RobotSceneCfg()
       observations: ObservationsCfg = ObservationsCfg()
       rewards: RewardsCfg = RewardsCfg()
       terminations: TerminationsCfg = TerminationsCfg()
       ...
   ```

### 3.2 实战：修改一个简单参数并重新训练

**目标**：让机器人更喜欢快速移动

**步骤**：

1. 打开配置文件：
   ```
   source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/go2/velocity_env_cfg.py
   ```

2. 找到 `RewardsCfg` 类，定位到速度奖励项：
   ```python
   track_lin_vel_xy_exp = RewTerm(
       func=mdp.track_lin_vel_xy_exp,
       weight=1.0,  # <-- 这里控制速度追踪的重要性
       params={...}
   )
   ```

3. 将 `weight` 从 `1.0` 改成 `2.0`（加强奖励）

4. 重新训练：
   ```bash
   python scripts/rsl_rl/train.py --task Unitree-Go2-Velocity --headless --max_iterations=100
   ```

5. 观察结果：训练曲线应该更陡峭，机器人学习的更快

**为什么会这样**？因为奖励权重高了，机器人会为了最大化总奖励而更积极地学习高速移动。

### 3.3 MDP (Markov Decision Process) 组件详解

强化学习的核心是 MDP，包含：

- **Observations（观测）**：机器人的"眼睛和耳朵"，定义在 `tasks/locomotion/mdp/observations.py`
- **Rewards（奖励）**：机器人的"目标"，定义在 `tasks/locomotion/mdp/rewards.py`
- **Commands（命令）**：外部给机器人下达的目标，定义在 `tasks/locomotion/mdp/commands/`
- **Terminations（终止条件）**：什么时候算一个 episode 结束

**查看这些文件**：
```bash
ls -la source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/mdp/
```

你会看到 `observations.py`, `rewards.py`, `commands.py` 等。这些定义了所有可用的观测和奖励函数。

---

## 第四阶段：掌握两大任务类型 (2小时)

### 4.1 Locomotion（走路任务）

**定义**：训练机器人在命令速度下走路，同时保持平衡。

**结构**：
```
tasks/locomotion/
├── robots/
│   ├── go2/velocity_env_cfg.py      # Go2 的配置
│   ├── h1/velocity_env_cfg.py        # H1 的配置
│   └── g1/29dof/velocity_env_cfg.py  # G1 的配置
├── mdp/
│   ├── commands.py                   # 速度命令定义
│   ├── observations.py               # 观测空间
│   ├── rewards.py                    # 奖励函数
│   └── curriculums.py                # 课程学习
└── agents/
    └── rsl_rl_ppo_cfg.py            # PPO 算法配置
```

**输入**（Observations）：
- 基座速度（x, y, z）
- 基座角速度
- 投影重力
- 高度扫描（周围地形）
- 关节位置/速度

**输出**（Actions）：
- 12 个（Go2）或 20-29 个（H1/G1）关节目标位置

**目标**（Commands）：
- 目标线速度（vx, vy）：随机化变化，训练出通用的速度控制策略

### 4.2 Mimic（模仿任务）- 以 Gangnam Style 为例

**定义**：训练机器人学会模仿人类的特定动作（动作捕捉数据）。

**目录结构**：
```
tasks/mimic/
├── robots/g1_29dof/
│   ├── gangnanm_style/              # <-- 江南Style舞蹈
│   │   ├── G1_gangnam_style_V01.bvh_60hz.npz  # 动作数据
│   │   ├── tracking_env_cfg.py      # 环境配置
│   │   └── g1.py                    # G1 参数
│   ├── dance_102/                   # 另一个舞蹈
│   └── petite_verses/               # 又一个舞蹈
├── mdp/
│   ├── observations.py              # 观测（参考动作 + 当前状态）
│   ├── rewards.py                   # 奖励（动作模仿误差）
│   ├── terminations.py              # 终止条件（摔倒）
│   └── ...
└── agents/
    └── rsl_rl_ppo_cfg.py
```

**关键文件解析**：

1. **`G1_gangnam_style_V01.bvh_60hz.npz`** - 动作数据
   - BVH（Biovision Hierarchy）：业界标准的动作捕捉格式
   - `.npz`：Numpy 压缩文件，包含关节角度序列
   - 60Hz：每秒 60 帧

2. **`tracking_env_cfg.py`** - 环境配置
   ```python
   class RobotSceneCfg(InteractiveSceneCfg):
       # ...与 Locomotion 类似，但没有地形命令
   
   class ObservationsCfg:
       obs_policy = ObsGroup(
           observations=[
               ObsTerm(func=mdp.base_lin_vel, ...),
               # 关键！参考动作
               ObsTerm(func=mdp.target_joint_pos, ...),  
               ObsTerm(func=mdp.target_joint_vel, ...),
               # 当前状态
               ObsTerm(func=mdp.joint_pos, ...),
               ObsTerm(func=mdp.joint_vel, ...),
           ]
       )
   
   class RewardsCfg:
       # 关键！模仿损失
       joint_pos_tracking = RewTerm(...)   # 位置误差
       joint_vel_tracking = RewTerm(...)   # 速度误差
   ```

3. **`g1.py`** - 机器人参数
   ```python
   # G1 的关节刚度、阻尼等物理参数
   # 这些参数影响机器人对指令的响应速度和平滑度
   ```

**Mimic 的核心机制**：
```
参考动作（从 .npz 读取）
    ↓
环境在每个时间步提供参考关节角度给智能体
    ↓
智能体学习输出接近参考动作的关节指令
    ↓
获得基于动作误差的奖励
```

---

## 第五阶段：从配置到代码 - 理解执行流 (2小时)

### 5.1 训练脚本的执行流

打开 `scripts/rsl_rl/train.py`，你会看到：

```python
def main():
    # 1. 解析命令行参数
    env_cfg = parse_cfg(args, task_name)
    
    # 2. 创建 RL 环境
    env = gym.make(task_name, cfg=env_cfg)
    
    # 3. 创建 RSL-RL PPO 算法
    runner = PPOTrainer(
        cfg=agent_cfg,
        env=env,
        device='cuda'
    )
    
    # 4. 训练循环
    while runner.alg.frame_count < runner.max_frames:
        actions = runner.get_actions(...)
        env_step_data = env.step(actions)
        runner.process_env_data(env_step_data)
        runner.alg.update()
```

**关键理解点**：
- `env_cfg` 来自你修改的 `velocity_env_cfg.py` 或 `tracking_env_cfg.py`
- `gym.make()` 会动态导入并实例化环境
- 每个 `env.step()` 会执行 Observations → Actions → Rewards 的完整 MDP 流程

### 5.2 推理脚本的执行流

打开 `scripts/rsl_rl/play.py`，看起来类似，但：

```python
def main():
    # 1. 加载训练好的模型权重
    actor_critic = load_checkpoint(checkpoint_path)
    
    # 2. 创建环境
    env = gym.make(task_name, cfg=env_cfg)
    
    # 3. 推理循环
    while True:
        obs = env.reset()
        for step in range(max_episode_length):
            actions = actor_critic(obs)  # <-- 神经网络推理
            obs, rewards, done, info = env.step(actions)
            if rendering:
                env.render()
```

---

## 第六阶段：参考动作数据 - Mimic 任务的"老师" (1小时)

### 6.1 查看 .npz 文件内容

```bash
python3 << 'EOF'
import numpy as np
data = np.load('source/unitree_rl_lab/unitree_rl_lab/tasks/mimic/robots/g1_29dof/gangnanm_style/G1_gangnam_style_V01.bvh_60hz.npz')
print("Keys in npz:", data.files)
print("Shape of motion data:", data['motion'].shape)
print("First 5 frames:")
print(data['motion'][:5, :10])  # 前 5 帧，前 10 个关节
EOF
```

**预期输出**：
```
Keys in npz: ['motion', 'frame_time', ...]
Shape of motion data: (N_frames, N_joints)  # 比如 (1200, 29) 表示 1200 帧，29 个关节
```

### 6.2 理解动作数据在环境中的使用

查看 `tasks/mimic/mdp/observations.py`，找到函数 `target_joint_pos()`：

```python
def target_joint_pos(env, asset_cfg: SceneEntityCfg, ...):
    """返回当前时间步的参考关节位置"""
    current_frame = env.current_frame_idx
    reference_motion = env.reference_motion  # 从 .npz 加载
    return reference_motion[current_frame]   # 返回该帧的所有关节角度
```

**时间同步**：
- 环境每个 step 自动推进 `current_frame_idx`
- 智能体看到的"目标"就是动作捕捉数据中当前帧的关节位置
- 智能体学习输出尽可能接近这个目标的关节命令

---

## 第七阶段：G1 vs H1 对比 - 选择合适的机器人 (1小时)

### 7.1 详细对比表

| 维度 | Go2 | H1 | G1-29DOF |
|-----|-----|-----|----------|
| **形态** | 四足犬型 | 人形（1个躯干+2条腿+2条臂） | 人形（1个躯干+2条腿+2条臂） |
| **高度** | ~45cm | ~110cm | ~80cm |
| **质量** | ~3kg | ~47kg | ~55kg |
| **腿部DOF** | 12（4腿×3） | 14（双腿7×2 + 躯干转） | 12（双腿6×2） |
| **臂部DOF** | 0 | 6（双臂3×2） | 10（双臂5×2） |
| **额外DOF** | 0 | 1（躯干Yaw） | 1（躯干Yaw） |
| **执行器类型** | GO2HV | 混合（多个型号） | 混合（高性能） |
| **最大关节扭矩** | ~23.5 Nm | 200 Nm (hip) | 139 Nm (hip) |
| **典型应用** | 越野、速度、敏捷 | 通用走路、臂操作 | 舞蹈模仿、复杂动作 |
| **学习难度** | ⭐ 简单 | ⭐⭐ 中等 | ⭐⭐⭐ 困难 |
| **仿真耗时** | 最快 | 中等 | 最慢 |
| **现存任务** | Velocity | Velocity | Velocity + 3种舞蹈 |

### 7.2 适用场景指南

**选择 Go2** 当你想：
- 快速验证 RL 环境的基本结构
- 测试新的奖励函数或课程学习策略
- 训练速度控制策略
- ✓ **新手首选**

**选择 H1** 当你想：
- 在更复杂的身体结构上训练（相比 Go2）
- 测试躯干转动对走路的影响
- 验证通用的人形走路策略
- ✓ **进阶学习**

**选择 G1-29DOF** 当你想：
- 学习动作模仿（有现成的参考动作）
- 理解全身协调（腿部 + 双臂）
- 测试支持最多自由度的策略
- ✗ **训练最慢，适合有经验的开发者**

### 7.3 代码对比：哪些文件不同？

**相同的**：
- `mdp/` 目录结构（观测、奖励、命令）
- 训练脚本逻辑
- PPO 算法配置

**不同的**：
- `robots/<robot_name>/velocity_env_cfg.py` - 环境配置
- `assets/robots/unitree.py` 中的 `UNITREE_*_CFG` - 机器人物理参数

**为什么会不同**？
- 每个机器人的自由度、身体比例、执行器特性都不同
- 环境配置必须适应这些差异（比如初始姿态、传感器位置、关节范围等）

---

## 第八阶段：C++ 部署入门 (1.5小时)

### 8.1 部署的三个阶段

1. **导出模型**（Python）
   ```bash
   python scripts/rsl_rl/play.py \
       --task Unitree-Go2-Velocity \
       --checkpoint logs/rsl_rl/... \
       --export_model  # 导出为 ONNX 格式
   ```

2. **C++ 编译**（编译为二进制）
   ```bash
   cd deploy/robots/go2
   mkdir build && cd build
   cmake .. && make -j4
   ```

3. **真机部署**（拷贝到机器人，运行）
   ```bash
   scp build/go2_control unitree@<robot_ip>:~/
   ssh unitree@<robot_ip> ~/go2_control
   ```

### 8.2 C++ 代码结构

```
deploy/
├── include/
│   ├── unitree_articulation.h      # 机器人关节接口
│   ├── param.h                      # 参数定义
│   └── FSM/
│       ├── BaseState.h              # 有限状态机基类
│       └── State_RLBase.h           # RL 策略执行状态
└── robots/<robot_name>/
    ├── main.cpp                     # 程序入口
    ├── CMakeLists.txt
    └── src/                         # 机器人特定实现
```

**关键理解**：
- Python 训练的神经网络权重导出为 ONNX
- C++ 程序加载 ONNX 模型
- 每个控制周期：读取传感器 → 调用 NN 推理 → 输出关节命令

---

## 第九阶段：实践项目 (3小时)

### 9.1 项目 A：自定义速度命令

**目标**：修改 Locomotion 任务，让机器人学会只向前走（不向左右移动）

**步骤**：
1. 打开 `tasks/locomotion/mdp/commands.py`
2. 找到 `VelocityCommandCfg` 类，修改速度范围：
   ```python
   vel_command_range = {
       "lin_vel_x": (0.0, 2.0),    # 只向前 (0-2 m/s)
       "lin_vel_y": (0.0, 0.0),    # 禁止左右移动
       "ang_vel_z": (-1.0, 1.0),   # 允许原地转
   }
   ```
3. 重新训练并观察机器人的行为变化

### 9.2 项目 B：调整奖励权重

**目标**：让机器人优先保持平衡而不是高速运动

**步骤**：
1. 打开 `tasks/locomotion/robots/go2/velocity_env_cfg.py`
2. 在 `RewardsCfg` 中：
   ```python
   # 降低速度追踪权重
   track_lin_vel_xy_exp = RewTerm(..., weight=0.5, ...)  # 从 1.0 改为 0.5
   
   # 提高平衡奖励权重
   base_height_l2 = RewTerm(..., weight=1.0, ...)        # 增加高度维持奖励
   ```
3. 训练并对比行为

### 9.3 项目 C：创建新的舞蹈 Mimic 任务

**目标**：基于现有的 Gangnam Style 模板，创建新的模仿任务

**步骤**：
1. 获取新的 BVH 动作捕捉文件（或重用现有的）
2. 复制 `gangnanm_style/` 目录为 `my_custom_dance/`
3. 修改文件：
   - `__init__.py` - 注册新任务
   - `tracking_env_cfg.py` - 如需要调整奖励权重
   - `g1.py` - 如需要调整 G1 参数

---

## 第十阶段：高级主题 (选学)

### 10.1 课程学习（Curriculum Learning）

训练刚开始时，地形简单→逐渐增加地形难度，帮助机器人渐进学习。

查看文件：`tasks/locomotion/mdp/curriculums.py`

### 10.2 领域随机化（Domain Randomization）

在仿真中随机化摩擦系数、重力大小等参数，使训练出的策略更robust，能更好地泛化到真机。

查看文件：`tasks/locomotion/robots/go2/velocity_env_cfg.py` 中的 `EventCfg` 类

### 10.3 多环境并行训练

IsaacLab 支持同时运行多个并行环境（GPU加速），大幅加快训练。

细节：见 `scripts/rsl_rl/train.py` 中的 `num_envs` 参数

---

## 总结：学习路线图

```
┌─────────────────────────────────────────────────────┐
│ 第 1 阶段：地形图（15 分钟）                           │
│ - 项目结构、三个机器人概览                               │
└─────────────┬───────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────────────────┐
│ 第 2 阶段：动手实验（1 小时）                          │
│ - Play → Train → Play，跑通完整流程                   │
└─────────────┬───────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────────────────┐
│ 第 3 阶段：代码核心（2 小时）                          │
│ - 环境配置文件（.py）详解、MDP 组件                     │
└─────────────┬───────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────────────────┐
│ 第 4 阶段：两大任务类型（2 小时）                     │
│ - Locomotion vs Mimic，了解各自的特点                 │
└─────────────┬───────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────────────────┐
│ 第 5 阶段：执行流（2 小时）                           │
│ - 从配置到代码：train.py 和 play.py 如何工作           │
└─────────────┬───────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────────────────┐
│ 第 6 阶段：参考数据（1 小时）                         │
│ - .npz 文件、动作捕捉数据如何用于 Mimic 任务           │
└─────────────┬───────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────────────────┐
│ 第 7 阶段：G1 vs H1（1 小时）                        │
│ - 机器人对比、选择合适的平台                          │
└─────────────┬───────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────────────────┐
│ 第 8 阶段：C++ 部署（1.5 小时）                       │
│ - 导出模型、编译、部署流程                            │
└─────────────┬───────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────────────────┐
│ 第 9 阶段：实践项目（3 小时）                         │
│ - A. 自定义速度命令 B. 调整奖励 C. 新舞蹈任务          │
└─────────────┬───────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────────────────┐
│ 第 10 阶段：高级主题（选学）                         │
│ - 课程学习、域随机化、并行训练                        │
└─────────────────────────────────────────────────────┘

总耗时：约 15 小时（可根据深度调整）
```

---

## 附录：常见问题 FAQ

### Q1：为什么我的训练 reward 没有上升？
**A**：可能原因：
- 学习率太高（导致不稳定）→ 调整 `agents/rsl_rl_ppo_cfg.py` 中的 `lr`
- 奖励权重设置不合理 → 检查 `RewardsCfg`
- 环境配置有问题 → 检查观测空间是否正确

### Q2：Play 时报错 "No checkpoint found"
**A**：解决方案：
```bash
# 1. 确认训练日志存在
ls logs/rsl_rl/

# 2. 手动指定检查点
python scripts/rsl_rl/play.py --task Unitree-Go2-Velocity \
    --checkpoint logs/rsl_rl/<TaskName>/<Timestamp>/models/model_<iteration>.pt
```

### Q3：G1 和 H1 哪个更适合开始学习？
**A**：从 **Go2** 开始（最简单）→ 然后 **H1**（中等）→ 最后 **G1**（复杂）

如果你直接要做舞蹈，可跳过 Go2/H1，直接用 G1 Mimic 任务。

### Q4：如何创建完全新的任务？
**A**：参考现有任务目录结构，复制一个任务，修改：
- `velocity_env_cfg.py` 或 `tracking_env_cfg.py`
- 观测函数、奖励函数
- 在 `__init__.py` 中注册新任务

### Q5：真机部署失败了怎么办？
**A**：按顺序检查：
1. ONNX 模型导出是否成功？
2. C++ 编译是否通过？
3. 机器人 SDK 版本是否匹配？
4. 传感器数据是否正确上报？

---

## 推荐学习顺序

1. **周一**：阶段 1-2（地形图 + 跑通流程）
2. **周二**：阶段 3-4（配置文件 + 两大任务类型）
3. **周三**：阶段 5-6（执行流 + 参考数据）
4. **周四**：阶段 7（机器人对比）
5. **周五**：阶段 9（实践项目 A/B）
6. **周末**：阶段 9C + 10（深化学习）

预计总耗时：**1-2 周**（取决于你的深度学习基础）

---

## 最后的建议

- **永远先跑通一个完整的例子再深入代码**
- **通过修改参数来理解代码**（不要只读代码）
- **在修改前备份原文件**
- **使用 `--headless` 加快训练速度**
- **定期查看 TensorBoard 日志**：
  ```bash
  tensorboard --logdir logs/rsl_rl/<TaskName>
  ```
