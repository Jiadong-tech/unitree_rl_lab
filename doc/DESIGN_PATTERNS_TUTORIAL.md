# 通过 Refactoring Guru 学习 Isaac Lab 框架设计模式 —— 完整教程

> **前言**：Isaac Lab 是 NVIDIA 为机器人强化学习开发的仿真框架，它的代码架构中大量运用了经典设计模式。本教程以 [refactoringguru.cn](https://refactoringguru.cn/design-patterns/catalog) 上的设计模式为知识源，结合 Unitree RL Lab 中的真实代码，带你逐个理解这些模式在工业级 RL 框架中的实际应用。
>
> **学习方式**：每章先阅读 Refactoring Guru 上的对应页面，理解模式的通用概念，然后对照 Isaac Lab 的真实代码加深理解。
>
> **适合读者**：有基本 Python 经验，想理解大型 RL 框架如何组织代码的开发者。

---

## 目录

- [第一章 策略模式（Strategy）— MDP 组件的灵魂](#第一章-策略模式strategy-mdp-组件的灵魂)
- [第二章 组合模式（Composite）— Manager 的树状架构](#第二章-组合模式composite-manager-的树状架构)
- [第三章 工厂方法（Factory Method）— 环境的创建机制](#第三章-工厂方法factory-method-环境的创建机制)
- [第四章 生成器模式（Builder）— 复杂环境的逐步组装](#第四章-生成器模式builder-复杂环境的逐步组装)
- [第五章 模板方法（Template Method）— 仿真循环的固定骨架](#第五章-模板方法template-method-仿真循环的固定骨架)
- [第六章 适配器模式（Adapter）— 框架间的桥梁](#第六章-适配器模式adapter-框架间的桥梁)
- [第七章 状态模式（State）— 机器人部署的 FSM](#第七章-状态模式state-机器人部署的-fsm)
- [第八章 综合实战：追踪一个完整的训练流程](#第八章-综合实战追踪一个完整的训练流程)
- [第九章 进阶：未被 Refactoring Guru 覆盖的模式](#第九章-进阶未被-refactoring-guru-覆盖的模式)
- [附录 A：模式速查表](#附录-a模式速查表)
- [附录 B：推荐学习路径](#附录-b推荐学习路径)

---

## 第一章 策略模式（Strategy）— MDP 组件的灵魂

### 📖 先读这个

**Refactoring Guru 页面**：[refactoringguru.cn/design-patterns/strategy](https://refactoringguru.cn/design-patterns/strategy)

重点关注：
- "策略模式建议将所有能以不同方式完成的算法提取为独立的类，称为策略"
- "上下文（Context）不会自行执行算法，而是将工作委派给策略对象"
- 类图中 Context → Strategy 接口 → ConcreteStrategyA/B/C 的关系

### 🤖 Isaac Lab 中怎么用的

在 Isaac Lab 中，**每一个 `ObsTerm`、`RewTerm`、`EventTerm`、`CommandTerm` 都是一个"策略对象"**。它们的"算法"（即具体的计算函数）可以独立替换，而 Manager（上下文）不需要知道具体用了哪个。

#### 具体例子：奖励函数是可替换的策略

**策略接口**：所有奖励函数都遵循统一签名 `(env, **params) → Tensor`

```python
# 策略 A：计算能量消耗
def energy(env, asset_cfg=SceneEntityCfg("robot")) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    qvel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    qfrc = asset.data.applied_torque[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(qvel) * torch.abs(qfrc), dim=-1)

# 策略 B：计算方向对齐
def orientation_l2(env, desired_gravity, asset_cfg=SceneEntityCfg("robot")) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    desired_gravity = torch.tensor(desired_gravity, device=env.device)
    cos_dist = torch.sum(asset.data.projected_gravity_b * desired_gravity, dim=-1)
    normalized = 0.5 * cos_dist + 0.5
    return torch.square(normalized)

# 策略 C：计算关节对称性
def joint_mirror(env, asset_cfg, mirror_joints) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    reward = torch.zeros(env.num_envs, device=env.device)
    for joint_pair in mirror_joints:
        reward += torch.sum(torch.square(
            asset.data.joint_pos[:, joint_pair[0]] - asset.data.joint_pos[:, joint_pair[1]]
        ), dim=-1)
    return reward
```

**上下文（Context）**：`RewardsCfg` 不关心具体函数细节，只通过 `RewTerm` 持有策略引用：

```python
@configclass
class RewardsCfg:
    # 只需要指定 func=（哪个策略）和 weight=（权重），不需要了解函数内部逻辑
    track_lin_vel_xy = RewTerm(func=mdp.track_lin_vel_xy_exp, weight=1.5, params={...})
    energy           = RewTerm(func=mdp.energy,                weight=-2e-5)
    orientation      = RewTerm(func=mdp.flat_orientation_l2,   weight=-2.5)
    joint_mirror     = RewTerm(func=mdp.joint_mirror,          weight=-0.5, params={...})
```

**替换策略**：想换一个奖励函数？只需改 `func=` 指向的函数：

```python
# 原来用 L2 误差
orientation = RewTerm(func=mdp.flat_orientation_l2, weight=-2.5)

# 换成指数核，一行改动，整个框架不需要任何其他修改
orientation = RewTerm(func=mdp.orientation_exp, weight=-2.5, params={"std": 0.5})
```

#### 同样的模式出现在观测中

```python
def gait_phase(env, period: float) -> torch.Tensor:
    """策略 A：步态相位观测"""
    global_phase = (env.episode_length_buf * env.step_dt) % period / period
    phase = torch.zeros(env.num_envs, 2, device=env.device)
    phase[:, 0] = torch.sin(global_phase * torch.pi * 2.0)
    phase[:, 1] = torch.cos(global_phase * torch.pi * 2.0)
    return phase
```

```python
@configclass
class PolicyCfg(ObsGroup):
    # 观测项也是可替换的策略
    gait_phase = ObsTerm(func=mdp.gait_phase, params={"period": 0.6})    # 策略 A
    base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2)) # 策略 B
    joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01)) # 策略 C
```

### 💡 核心洞察

| Refactoring Guru 概念 | Isaac Lab 对应 |
|----------------------|---------------|
| Context（上下文） | `RewardManager` / `ObservationManager` |
| Strategy 接口 | 函数签名 `(env, **params) → Tensor` |
| ConcreteStrategy | `energy()`, `orientation_l2()`, `gait_phase()` 等具体函数 |
| setStrategy() | 在 `*Cfg` 中修改 `func=` 参数 |

### ✅ 练习

1. 阅读 `source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/mdp/rewards.py`，找出至少 5 个不同的"策略"函数
2. 对比 Go2 和 H1 的 `velocity_env_cfg.py`，看它们如何选择不同的奖励策略组合
3. 尝试自己写一个新的奖励函数，遵循 `(env, **params) → Tensor` 签名

---

## 第二章 组合模式（Composite）— Manager 的树状架构

### 📖 先读这个

**Refactoring Guru 页面**：[refactoringguru.cn/design-patterns/composite](https://refactoringguru.cn/design-patterns/composite)

重点关注：
- "组合模式建议使用一个通用接口来与产品和盒子进行交互"
- "对于树状结构，容器和叶子节点都实现相同的接口"
- 树状结构图：Container 包含 Leaf 和其他 Container

### 🤖 Isaac Lab 中怎么用的

`ManagerBasedRLEnv` 是一个**容器**，包含多个 **Manager**（子容器），每个 Manager 再包含多个 **Term**（叶子节点）。整体形成一棵树：

```
ManagerBasedRLEnv                                    ← 根容器
├── RewardManager                                    ← 容器节点
│   ├── RewTerm: track_lin_vel_xy     (weight=1.5)  ← 叶子
│   ├── RewTerm: energy               (weight=-2e-5)← 叶子
│   ├── RewTerm: flat_orientation_l2  (weight=-2.5) ← 叶子
│   ├── RewTerm: joint_pos_limits     (weight=-10)  ← 叶子
│   └── ... (15+ 个奖励项)
├── ObservationManager                               ← 容器节点
│   ├── ObsGroup: policy                             ← 子容器
│   │   ├── ObsTerm: base_ang_vel                   ← 叶子
│   │   ├── ObsTerm: joint_pos_rel                  ← 叶子
│   │   └── ObsTerm: gait_phase                     ← 叶子
│   └── ObsGroup: critic                             ← 子容器
│       ├── ObsTerm: base_lin_vel                   ← 叶子
│       └── ObsTerm: joint_effort                   ← 叶子
├── ActionManager                                    ← 容器节点
│   └── ActionTerm: JointPositionAction             ← 叶子
├── CommandManager                                   ← 容器节点
│   └── CommandTerm: UniformVelocityCommand         ← 叶子
├── TerminationManager                               ← 容器节点
│   ├── TermTerm: time_out                          ← 叶子
│   └── TermTerm: illegal_contact                   ← 叶子
└── EventManager                                     ← 容器节点
    ├── EventTerm: reset_robot (on_reset)           ← 叶子
    ├── EventTerm: push_robot  (interval)           ← 叶子
    └── EventTerm: physics_material (on_startup)    ← 叶子
```

#### 统一接口：compute()

组合模式的关键是**容器和叶子有统一的操作方式**。在 Isaac Lab 中，无论 Manager 下面有几个 Term，调用方式完全一样：

```python
# env.step() 内部（简化）：
rewards = self.reward_manager.compute()        # 自动遍历所有 RewTerm，加权求和
obs = self.observation_manager.compute()       # 自动遍历所有 ObsTerm，拼接成向量
terminated = self.termination_manager.compute() # 自动遍历所有 TermTerm，逻辑或
```

**RewardManager 不需要知道它下面挂了多少个 RewTerm**——可能是 5 个，也可能是 20 个，`compute()` 的调用方式完全不变。

#### 在配置中体现的树状结构

```python
@configclass
class RobotEnvCfg(ManagerBasedRLEnvCfg):
    """这就是在配置层面组装树结构"""

    # 一级节点：各 Manager 的配置
    scene = RobotSceneCfg(num_envs=4096)      # SceneManager
    actions = ActionsCfg()                      # ActionManager
    observations = ObservationsCfg()            # ObservationManager ← 包含 PolicyCfg + CriticCfg
    rewards = RewardsCfg()                      # RewardManager      ← 包含 15+ 个 RewTerm
    terminations = TerminationsCfg()            # TerminationManager ← 包含多个 TermTerm
    events = EventCfg()                         # EventManager       ← 包含多个 EventTerm
    commands = CommandsCfg()                     # CommandManager     ← 包含 CommandTerm
    curriculum = CurriculumCfg()                # CurriculumManager
```

### 💡 核心洞察

| Refactoring Guru 概念 | Isaac Lab 对应 |
|----------------------|---------------|
| Component（通用接口） | 所有 Term 都有 `compute()` / `__call__()` |
| Leaf（叶子） | `RewTerm`, `ObsTerm`, `EventTerm`, `TermTerm` |
| Composite（容器） | `RewardManager`, `ObservationManager`, `ObsGroup` |
| 遍历子元素 | Manager 的 `compute()` 内部遍历所有 Term |

### ✅ 练习

1. 数一数 `velocity_env_cfg.py` 中 `RewardsCfg` 的叶子节点数量
2. 观测空间是**两层组合**（Manager → ObsGroup → ObsTerm），画出它的完整树状图
3. 思考：如果你要新增一个奖励项，只需要在 `RewardsCfg` 里加一行——为什么不需要修改 `RewardManager` 的代码？

---

## 第三章 工厂方法（Factory Method）— 环境的创建机制

### 📖 先读这个

**Refactoring Guru 页面**：[refactoringguru.cn/design-patterns/factory-method](https://refactoringguru.cn/design-patterns/factory-method)

重点关注：
- "工厂方法建议使用特殊的工厂方法代替直接使用 new 运算符"
- "子类可以修改工厂方法返回的对象类型"
- Creator → Product 的解耦关系

### 🤖 Isaac Lab 中怎么用的

Isaac Lab 使用 **OpenAI Gymnasium 的注册表工厂**。这是工厂方法模式的一个变体——不用继承来定义工厂，而是用注册表 + 字符串 ID。

#### 第一步：注册"配方"（定义产品如何创建）

每个任务在 `__init__.py` 中注册自己的创建配方：

```python
# tasks/locomotion/robots/go2/__init__.py
import gymnasium as gym

gym.register(
    id="Unitree-Go2-Velocity",                                  # 产品名
    entry_point="isaaclab.envs:ManagerBasedRLEnv",              # 工厂类
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotEnvCfg",  # 产品配置 A
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotPlayEnvCfg",  # 产品配置 B
        "rsl_rl_cfg_entry_point": "...rsl_rl_ppo_cfg:BasePPORunnerCfg",     # 算法配置
    },
)
```

```python
# tasks/mimic/robots/g1_29dof/martial_arts/__init__.py
gym.register(
    id="Unitree-G1-29dof-Mimic-MartialArts-HeianShodan",       # 另一个产品名
    entry_point="isaaclab.envs:ManagerBasedRLEnv",              # 同一个工厂类！
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tracking_env_cfg:HeianShodanEnvCfg",  # 不同的产品配置
        "play_env_cfg_entry_point": f"{__name__}.tracking_env_cfg:HeianShodanPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "...rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)
```

注意：**7 个武术动作 × 2 个运动任务 × 6 个机器人 = 大量产品**，但它们都用同一个工厂类 `ManagerBasedRLEnv`，只是配置不同。

#### 第二步：通过工厂创建产品（不需要知道具体类）

```python
# train.py 中——调用者只需要一个字符串 ID
env = gym.make("Unitree-Go2-Velocity", cfg=env_cfg)

# 效果等同于：
# from isaaclab.envs import ManagerBasedRLEnv
# from unitree_rl_lab.tasks.locomotion.robots.go2.velocity_env_cfg import RobotEnvCfg
# cfg = RobotEnvCfg()
# env = ManagerBasedRLEnv(cfg=cfg)
```

**调用者（train.py）完全不需要知道：**
- 具体的环境配置类在哪个文件
- 用了哪个机器人
- MDP 组件是怎么组装的

它只需要一个字符串 `"Unitree-Go2-Velocity"`。

#### 第三步：配置也通过工厂加载

```python
# utils/parser_cfg.py
def parse_env_cfg(task_name, device, num_envs, entry_point_key="env_cfg_entry_point"):
    # 从注册表查找配置类并实例化
    cfg = load_cfg_from_registry(task_name, entry_point_key)
    # 覆盖运行时参数
    cfg.sim.device = device
    if num_envs is not None:
        cfg.scene.num_envs = num_envs
    return cfg
```

`load_cfg_from_registry()` 就是一个工厂方法——给它一个任务名和 key，它自动找到正确的配置类并创建实例。

### 💡 核心洞察

| Refactoring Guru 概念 | Isaac Lab 对应 |
|----------------------|---------------|
| Creator（创建者） | `gym.make()` + `gym.registry` |
| Product（产品） | `ManagerBasedRLEnv` 实例 |
| ConcreteCreator | 每个 `__init__.py` 中的 `gym.register()` 调用 |
| factoryMethod() | `load_cfg_from_registry(task_name, key)` |

### 与传统工厂方法的区别

传统工厂方法用**继承**（子类覆盖 `createProduct()`），Isaac Lab 用**注册表**（字符串映射到配置入口）。效果相同——调用者与具体产品解耦。

### ✅ 练习

1. 在 `tasks/mimic/robots/g1_29dof/martial_arts/__init__.py` 中，找出所有注册的产品 ID
2. 从 `train.py` 开始，追踪 `gym.make()` 最终如何创建了一个 `ManagerBasedRLEnv`
3. 思考：如果你要新增一个 "Unitree-Go2-Terrain" 任务，需要修改 `train.py` 吗？（答案是不需要——只需新增 `__init__.py` + `terrain_env_cfg.py`）

---

## 第四章 生成器模式（Builder）— 复杂环境的逐步组装

### 📖 先读这个

**Refactoring Guru 页面**：[refactoringguru.cn/design-patterns/builder](https://refactoringguru.cn/design-patterns/builder)

重点关注：
- "生成器模式建议将对象构建代码从产品类中抽取出来"
- "分步骤创建对象，不同生成器以不同方式实现这些步骤"
- Director 和 Builder 的关系

### 🤖 Isaac Lab 中怎么用的

一个完整的 RL 训练环境是非常复杂的对象——它需要：场景（地形 + 机器人 + 传感器）、动作空间、观测空间、奖励函数、终止条件、随机化事件、命令生成器。

`@configclass` 配置体系就是一个 **生成器模式**——逐步组装这些组件，最终传给 `ManagerBasedRLEnv` 的构造函数。

#### 生成器：逐步组装

```python
@configclass
class RobotEnvCfg(ManagerBasedRLEnvCfg):
    """每一个属性赋值 = Builder 的一个 build step"""

    # Step 1: 建造场景（地形 + 机器人 + 传感器）
    scene = RobotSceneCfg(num_envs=4096, env_spacing=2.5)

    # Step 2: 建造动作空间
    actions = ActionsCfg()

    # Step 3: 建造观测空间
    observations = ObservationsCfg()

    # Step 4: 建造命令生成器
    commands = CommandsCfg()

    # Step 5: 建造奖励系统
    rewards = RewardsCfg()

    # Step 6: 建造终止条件
    terminations = TerminationsCfg()

    # Step 7: 建造随机化事件
    events = EventCfg()

    # Step 8: 建造课程学习
    curriculum = CurriculumCfg()

    # Step 9: 后处理调整（Director 的微调）
    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
```

#### 不同的"生成器"产出不同的产品

Go2、H1、G1 各有自己的 `velocity_env_cfg.py`，使用**相同的步骤**（都要定义 scene/actions/rewards/...）但**产出不同的环境**：

```python
# Go2 的生成器
class Go2EnvCfg(ManagerBasedRLEnvCfg):
    scene = Go2SceneCfg(num_envs=4096)            # 四足场景
    rewards = Go2RewardsCfg()                       # 四足奖励（步态、足底等）
    commands = VelocityCommandCfg(lin_vel_x=(-0.1, 0.1))  # 保守速度范围

# H1 的生成器
class H1EnvCfg(ManagerBasedRLEnvCfg):
    scene = H1SceneCfg(num_envs=4096)             # 人形场景
    rewards = H1RewardsCfg()                        # 人形奖励（手臂、躯干等）
    commands = VelocityCommandCfg(lin_vel_x=(-0.3, 1.0))  # 更大速度范围
```

#### 训练配置 vs 推理配置——同一份蓝图的变体

```python
# 训练用生成器（完整规模）
@configclass
class RobotEnvCfg(ManagerBasedRLEnvCfg):
    scene = RobotSceneCfg(num_envs=4096)
    events = EventCfg()  # 含随机推力等

# 推理用生成器（基于训练版本修改）
@configclass
class RobotPlayEnvCfg(RobotEnvCfg):
    """继承训练配置，覆盖部分步骤"""
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50          # 减少环境数
        # 关闭随机推力等...
```

### 💡 核心洞察

| Refactoring Guru 概念 | Isaac Lab 对应 |
|----------------------|---------------|
| Builder（生成器） | 每个 `*EnvCfg` 类 |
| buildStepA/B/C | scene= / actions= / rewards= 各属性 |
| Director（主管） | `__post_init__()` + `ManagerBasedRLEnv.__init__()` |
| Product（产品） | 最终的 `ManagerBasedRLEnv` 实例 |
| 不同生成器 | Go2EnvCfg / H1EnvCfg / G1EnvCfg |

### ✅ 练习

1. 对比 Go2 和 H1 的 `velocity_env_cfg.py`，找出 "Step" 相同但 "内容" 不同的地方
2. 阅读 `RobotPlayEnvCfg`，理解它如何通过继承 + 覆盖实现"变体生成器"
3. 思考：如果没有 Builder 模式，你需要一个多大的构造函数来创建环境？

---

## 第五章 模板方法（Template Method）— 仿真循环的固定骨架

### 📖 先读这个

**Refactoring Guru 页面**：[refactoringguru.cn/design-patterns/template-method](https://refactoringguru.cn/design-patterns/template-method)

重点关注：
- "模板方法建议将算法分解为一系列步骤，将这些步骤转换为方法，并在模板方法中依次调用"
- "子类可以重写某些步骤，但不能改变整体结构"
- 骨架方法 vs 可覆盖步骤

### 🤖 Isaac Lab 中怎么用的

`ManagerBasedRLEnv.step()` 是一个**模板方法**——它定义了仿真循环的固定步骤顺序，但每一步的具体内容由配置（而非子类）决定。

#### 固定的执行骨架

```python
# Isaac Lab 内部的 env.step()（简化版，展示核心逻辑）
class ManagerBasedRLEnv:
    def step(self, action):
        # ——— 以下步骤顺序不可更改 ———

        # Step 1: 处理动作
        self.action_manager.process_action(action)

        # Step 2: 物理仿真（decimation 次子步骤）
        for _ in range(self.cfg.decimation):
            self.scene.write_data_to_sim()
            self.sim.step()
            self.scene.update()

        # Step 3: 计算观测
        obs = self.observation_manager.compute()

        # Step 4: 计算奖励
        reward = self.reward_manager.compute()

        # Step 5: 检查终止
        terminated, truncated = self.termination_manager.compute()

        # Step 6: 更新命令
        self.command_manager.compute()

        # Step 7: 处理重置（用事件管理器）
        reset_ids = (terminated | truncated).nonzero()
        self.event_manager.apply(mode="reset", env_ids=reset_ids)

        # Step 8: 课程更新
        self.curriculum_manager.compute()

        return obs, reward, terminated, info
```

#### "可覆盖的步骤"通过配置注入

传统模板方法用**子类继承**来覆盖步骤，Isaac Lab 用**配置注入**实现同样效果——更灵活：

```
模板方法 step()                  不同任务的"覆盖"
━━━━━━━━━━━━━━━                  ━━━━━━━━━━━━━━━━
Step 3: compute obs              Locomotion: [ang_vel, gravity, joint_pos, gait_phase]
                                 Mimic:      [motion_command, anchor_ori, joint_pos]

Step 4: compute rewards          Locomotion: [track_vel, energy, orientation]
                                 Mimic:      [body_pos, body_ori, body_vel]

Step 5: check termination        Locomotion: [illegal_contact, timeout]
                                 Mimic:      [anchor_drift, timeout]

Step 6: update commands          Locomotion: UniformVelocityCommand (随机速度)
                                 Mimic:      MotionCommand (NPZ 逐帧)
```

**步骤骨架不变，内容可替换**——这正是模板方法的核心思想。

### 💡 核心洞察

| Refactoring Guru 概念 | Isaac Lab 对应 |
|----------------------|---------------|
| AbstractClass | `ManagerBasedRLEnv` |
| templateMethod() | `step()` |
| step1(), step2() ... | `action_manager.process()`, `reward_manager.compute()` 等 |
| ConcreteClass 重写步骤 | 不同的 `*Cfg` 注入不同的 Manager 内容 |

### 与传统模板方法的区别

传统做法：通过**继承**创建 `Go2Env(BaseEnv)` 并重写 `compute_reward()`。
Isaac Lab 做法：通过**配置**注入不同的 `RewardsCfg`，`step()` 框架完全不变。

这是"组合优于继承"原则的实践——用模板方法的思想，但用组合模式来实现"可覆盖的步骤"。

### ✅ 练习

1. 想象如果没有模板方法，Go2 和 H1 的训练循环各自需要写多少重复代码
2. `step()` 中 7 个步骤的执行顺序为什么重要？（提示：奖励计算需要先有最新观测）
3. 对比 Refactoring Guru 书中的"一步步泡茶/泡咖啡"例子和 Isaac Lab 的"一步步走仿真循环"

---

## 第六章 适配器模式（Adapter）— 框架间的桥梁

### 📖 先读这个

**Refactoring Guru 页面**：[refactoringguru.cn/design-patterns/adapter](https://refactoringguru.cn/design-patterns/adapter)

重点关注：
- "适配器是一个特殊的对象，能够转换对象接口，使其能与其他对象进行交互"
- "适配器不仅可以转换数据格式，还可以帮助不同接口的对象合作"
- 方钉和圆孔的经典比喻

### 🤖 Isaac Lab 中怎么用的

Isaac Lab 环境遵循 Gymnasium 接口，但 RSL-RL 训练库需要自己的向量化接口。`RslRlVecEnvWrapper` 就是一个**适配器**。

#### 接口不兼容

```
Isaac Lab (Gymnasium 接口)          RSL-RL (自有接口)
━━━━━━━━━━━━━━━━━━━━━━━━           ━━━━━━━━━━━━━━━━
env.step(action)                   env.step(action)
→ (obs, reward, terminated,        → (obs, privileged_obs,
   truncated, info)                    rewards, dones, infos)

env.reset()                        env.get_observations()
→ (obs, info)                      → obs

env.observation_space              env.num_obs
                                   env.num_privileged_obs
                                   env.num_actions
```

同样是 `step()`，但**返回值格式完全不同**。

#### 适配器：RslRlVecEnvWrapper

```python
# train.py 中
env = gym.make("Unitree-Go2-Velocity", cfg=env_cfg)    # Gymnasium 接口
env = RslRlVecEnvWrapper(env, clip_actions=True)         # ← 适配器！
# 现在 env 变成了 RSL-RL 可用的接口

runner = OnPolicyRunner(env, ...)  # RSL-RL 可以正常使用
```

适配器在幕后做的事情：
- 将 Gymnasium 的 `(obs, reward, terminated, truncated, info)` 转换为 RSL-RL 的 `(obs, privileged_obs, rewards, dones, infos)`
- 暴露 `env.num_obs`、`env.num_actions` 等 RSL-RL 需要的属性
- 处理动作裁剪（`clip_actions`）

#### 同样出现在部署导出中

```python
# export_deploy_cfg() 也是一种"适配"
# 将 Python 训练环境的配置转换为 C++ 部署端能读取的 YAML 格式
export_deploy_cfg(env.unwrapped, log_dir)
# 输出：deploy.yaml（关节映射、PD 增益、观测定义 → C++ 可读）
```

### 💡 核心洞察

| Refactoring Guru 概念 | Isaac Lab 对应 |
|----------------------|---------------|
| Client（客户端） | RSL-RL 的 `OnPolicyRunner` |
| Service（不兼容的服务） | Isaac Lab 的 `ManagerBasedRLEnv`（Gymnasium 接口）|
| Adapter（适配器） | `RslRlVecEnvWrapper` |
| 接口转换 | `(obs, reward, terminated, truncated, info)` → `(obs, priv_obs, rewards, dones, infos)` |

### ✅ 练习

1. 思考：如果将来要换用 Stable-Baselines3 训练框架，需要改 `ManagerBasedRLEnv` 的代码吗？（答案是不需要——只需写一个新的适配器 `SB3VecEnvWrapper`）
2. `export_deploy_cfg()` 本质上是把 Python 对象"适配"成 YAML 文本，这也是适配器思想的体现

---

## 第七章 状态模式（State）— 机器人部署的 FSM

### 📖 先读这个

**Refactoring Guru 页面**：[refactoringguru.cn/design-patterns/state](https://refactoringguru.cn/design-patterns/state)

重点关注：
- "状态模式建议将所有特定于状态的行为抽取到一组独立的类中"
- "原始对象将工作委派给这些状态对象，而不是自行实现"
- Context → State 接口 → ConcreteStateA/B/C

### 🤖 Isaac Lab 中怎么用的

C++ 部署端 (`deploy/`) 使用有限状态机（FSM）控制机器人，每个状态是一个独立的类。

#### 状态接口

```cpp
// deploy/include/FSM/BaseState.h（简化）
class BaseState {
public:
    virtual void enter() = 0;          // 进入状态时
    virtual void run() = 0;            // 每个控制周期
    virtual void exit() = 0;           // 退出状态时
    virtual FSMStateName checkTransition() = 0;  // 检查是否要切换状态
};
```

#### 具体状态

```
┌───────────────────────────────────────────────────────────┐
│                    CtrlFSM (Context)                       │
│                                                           │
│   ┌──────────┐    ┌─────────────┐    ┌─────────────────┐ │
│   │ Passive  │───►│  FixStand   │───►│ RLBase /        │ │
│   │          │    │             │    │ MartialArts     │ │
│   │ 关节软化  │    │ PD 控制站立  │    │ Sequencer       │ │
│   │ enter()  │    │ enter()     │    │ enter()          │ │
│   │ run()    │    │ run()       │    │ run()            │ │
│   │ exit()   │    │ exit()      │    │ exit()           │ │
│   └──────────┘    └─────────────┘    └─────────────────┘ │
│                                                           │
│   currentState->run()  // Context 不关心当前是哪个状态     │
└───────────────────────────────────────────────────────────┘
```

#### 状态转换由状态自身决定

```cpp
// 每个状态自行决定是否切换
FSMStateName State_Passive::checkTransition() {
    if (joystick.L2_pressed())
        return FSMStateName::FIXSTAND;     // 按 L2 → 切换到站立
    return FSMStateName::PASSIVE;          // 否则保持当前状态
}

FSMStateName State_FixStand::checkTransition() {
    if (joystick.R2_pressed() && stand_complete)
        return FSMStateName::RLBASE;       // 站稳后按 R2 → 切换到 RL 推理
    return FSMStateName::FIXSTAND;
}
```

#### CtrlFSM（Context）的控制循环

```cpp
// deploy/include/FSM/CtrlFSM.h（简化）
class CtrlFSM {
    BaseState* currentState;
    
    void update() {
        // 检查是否需要切换状态
        FSMStateName nextState = currentState->checkTransition();
        if (nextState != currentStateName) {
            currentState->exit();      // 退出当前状态
            currentState = states[nextState];
            currentState->enter();     // 进入新状态
        }
        currentState->run();           // 执行当前状态的逻辑
    }
};
```

### 💡 核心洞察

| Refactoring Guru 概念 | Isaac Lab 对应 |
|----------------------|---------------|
| Context | `CtrlFSM` |
| State 接口 | `BaseState`（enter/run/exit/checkTransition） |
| ConcreteStateA | `State_Passive`（关节软化） |
| ConcreteStateB | `State_FixStand`（站立归位） |
| ConcreteStateC | `State_RLBase`（RL 策略推理） |
| ConcreteStateD | `State_MartialArtsSequencer`（多策略串联） |
| setState() | `checkTransition()` 返回新状态名 |

### ✅ 练习

1. 画出完整的状态转换图：Passive → FixStand → RLBase，标注触发条件
2. Refactoring Guru 中用"音乐播放器"（播放/暂停/停止）解释状态模式——与机器人 FSM 对比
3. 思考：为什么不用 `if/else` 链代替状态模式？（想象有 10 个状态时代码会多混乱）

---

## 第八章 综合实战：追踪一个完整的训练流程

现在把前 7 章的模式串联起来，追踪一次完整的训练过程中各模式的协作。

### 场景：运行 `python train.py --task Unitree-Go2-Velocity`

```
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  ① train.py 启动                                                    │
│     import unitree_rl_lab.tasks                                     │
│     ↓                                                               │
│  ② 【工厂方法 + 插件发现】                                            │
│     tasks/__init__.py → import_packages()                          │
│     递归发现所有 __init__.py → 执行 gym.register()                    │
│     注册表中现在有 "Unitree-Go2-Velocity" 等所有任务                    │
│     ↓                                                               │
│  ③ 【工厂方法】                                                      │
│     env_cfg = parse_env_cfg("Unitree-Go2-Velocity")                │
│     → load_cfg_from_registry() 查找 entry_point → 实例化 RobotEnvCfg │
│     ↓                                                               │
│  ④ 【生成器模式】                                                     │
│     RobotEnvCfg 内部组装：                                            │
│     scene=RobotSceneCfg → actions=ActionsCfg → rewards=RewardsCfg   │
│     → observations=ObsCfg → terminations=TermCfg → events=EventCfg  │
│     ↓                                                               │
│  ⑤ 【工厂方法】                                                      │
│     env = gym.make("Unitree-Go2-Velocity", cfg=env_cfg)            │
│     → ManagerBasedRLEnv(cfg) 读取配置，创建所有 Manager               │
│     ↓                                                               │
│  ⑥ 【组合模式】                                                      │
│     ManagerBasedRLEnv 内部构建树：                                     │
│     ├── RewardManager (15 个 RewTerm)                               │
│     ├── ObservationManager (2 个 ObsGroup, 各含多个 ObsTerm)         │
│     ├── ActionManager (1 个 JointPositionAction)                    │
│     └── ...                                                         │
│     ↓                                                               │
│  ⑦ 【适配器模式】                                                     │
│     env = RslRlVecEnvWrapper(env)                                   │
│     Gymnasium 接口 → RSL-RL 接口                                     │
│     ↓                                                               │
│  ⑧ 训练循环开始：runner.learn()                                       │
│     ↓                                                               │
│  ⑨ 【模板方法 + 策略模式】每个 env.step():                              │
│     │  Step 1: action_manager.apply(action)                         │
│     │  Step 2: sim.step() × decimation                              │
│     │  Step 3: obs_manager.compute()                                │
│     │     ├── 【策略】gait_phase(period=0.6)                         │
│     │     ├── 【策略】base_ang_vel(noise=±0.2)                       │
│     │     └── 【策略】joint_pos_rel(noise=±0.01)                     │
│     │  Step 4: reward_manager.compute()                             │
│     │     ├── 【策略】track_lin_vel_xy_exp × 1.5                     │
│     │     ├── 【策略】energy × -2e-5                                 │
│     │     └── 【策略】flat_orientation_l2 × -2.5                     │
│     │  Step 5: termination_manager.compute()                        │
│     │  Step 6: command_manager.compute()                            │
│     │  Step 7: event_manager.apply(reset)                           │
│     │                                                               │
│     ↓ 重复 30,000 次迭代                                              │
│                                                                     │
│  ⑩ 训练完成，保存 checkpoint + 导出 deploy.yaml                        │
│     → 部署端使用【状态模式】FSM 加载 ONNX 策略                          │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 模式协作时间线

| 阶段 | 时刻 | 使用的模式 | 作用 |
|------|------|----------|------|
| 启动 | `import tasks` | 插件发现 | 自动注册所有环境 |
| 配置 | `parse_env_cfg()` | 工厂方法 | 从注册表加载配置 |
| 组装 | `RobotEnvCfg` 实例化 | 生成器 | 逐步组装环境组件 |
| 创建 | `gym.make()` | 工厂方法 | 创建环境实例 |
| 构建 | Manager 初始化 | 组合 | 构建 Term 树 |
| 适配 | `RslRlVecEnvWrapper` | 适配器 | 转换接口格式 |
| 运行 | `env.step()` | 模板方法 | 固定步骤骨架 |
| 每步 | Manager.compute() | 策略 + 组合 | 遍历 Term、调用策略函数 |
| 部署 | C++ FSM | 状态 | 机器人行为切换 |

### 深度解析：设计模式在项目启动与训练流程中的核心意义

在上面追踪的完整流程中，每个设计模式都不是为了“炫技”而存在的，而是精准地解决了复杂的真实业务痛点。将这套架构对应到“控制流程”，能够让我们从更高维度去理解整个框架：

**1. 准备阶段：解决“高内聚低耦合”与“扩展性”问题**
- **插件发现（Plugin Discovery）与工厂方法（Factory Method）**：在系统启动时（`train.py`），最头疼的问题通常是要维护一个巨大的 `if-else` 或字典来映射所有的任务名字与任务类。通过 `tasks/__init__.py` 的插件发现机制和 `gym.make()`，新增任务时开发者只需**新建文件**并使用装饰器注册即可。主程序对具体环境**完全无感知**，贯彻了“开闭原则”（对扩展开放，对修改封闭）。

**2. 配置阶段：解决“参数地狱”与“复杂对象构建”问题**
- **生成器模式（Builder Pattern）**：强化学习环境包含数十成百个配置项（地形、动作空间、几十个奖励函数项、噪声注入序列等）。如果全放在构造函数里，会变成不可维护的“参数地狱”。`RobotEnvCfg` 使用生成器模式将配置分割为 `scene`, `actions`, `rewards`, `observations` 等独立模块分步组装。开发者可以随心所欲地替换某个子模块进行实验，而不会干扰其他配置。

**3. 构建阶段：解决“多层级管理”和“兼容性”问题**
- **组合模式（Composite Pattern）**：`ManagerBasedRLEnv` 内部管理着不同类的管理器。组合模式的优势在于“整体和部分的统一性”。比如你配置了 15 个奖励项（Term），`RewardManager` 用组合模式将它们挂在同一棵树下。当环境在 `step()` 里调用 `compute()` 时，Manager 会自动遍历树底层的各个小策略进行计算并汇总，顶层调用极其清爽。
- **适配器模式（Adapter Pattern）**：众所周知，RL算法库（如 RSL-RL, SB3）和环境库（如 Gymnasium, Isaac Gym）往往接口不匹配。`RslRlVecEnvWrapper` 就像一个转换插头，使得底层辛苦搭建的环境能无缝对接各大顶级算法框架。后期如果要换 PPO 算法后端，只需新增一个 Adapter，核心环境逻辑一行不用改。

**4. 运行阶段：解决“流程固化”与“细节灵活”问题**
- **模板方法（Template Method）与策略模式（Strategy Pattern）**：`env.step()` 是训练最频繁调用的函数，具有严格的先后顺序（动作下发 -> 物理仿真 -> 观测计算 -> 奖励计算 -> 终止判定）。**模板方法**锁死了这套主干流程，保证了系统的时序安全。而骨架留出的各个“插槽”，则由各个具体的**策略模式**函数（如具体的 `track_lin_vel_xy_exp` 或 `energy`, `gait_phase`）填充。这让算法工程师可以在不碰引擎底层代码的情况下，仅凭调参和自定义策略函数就实现无尽的魔改。

**5. 落地部署阶段：解决“真实流控安全”问题**
- **状态模式（State Pattern）**：当策略被训练成 ONNX 权重落地到真实 C++ 机器狗时，机器人面临更加多变的环境。使用 FSM（有限状态机）结合状态模式，把 `BaseState`, `State_FixStand`, `State_RLBase` 切分得干干净净。避免了巨型的 `switch-case` 逻辑，在切换机器犬状态时极其安全稳妥，这对于实物硬件保护至关重要。

**总结**：设计模式在这套 RL 框架中，就像工厂里的模具和流水线：它让你（开发者）不用每一项任务都像手工作坊一样从头编写屎山代码；它利用接口和模式设定好规则，让你把全部精力聚焦在“奖励怎么调、观测怎么加”这些**真正决定算法效果**的核心痛点上，极大抹平了工业级强化学习落地的复杂性门槛。

---

## 第九章 进阶：未被 Refactoring Guru 覆盖的模式

Isaac Lab 中还有一些设计模式在 Refactoring Guru 上没有专门页面，但值得了解：

### 9.1 注册表模式（Registry Pattern）

`gym.register()` / `gym.make()` 使用的核心机制。注册表是一个全局字典，映射字符串 ID 到创建逻辑。

```python
# 注册（定义时）
gym.register(id="Unitree-Go2-Velocity", entry_point=..., kwargs=...)

# 查找（使用时）
gym.registry["Unitree-Go2-Velocity"]  # → 返回注册的 entry_point + kwargs
```

这是工厂方法的加强版——工厂不硬编码产品列表，而是运行时动态注册。

**延伸学习**：虽然 Refactoring Guru 没有专门页面，但在工厂方法页面的 "适用场景" 中有提及类似思路。

### 9.2 插件架构（Plugin Architecture）

```python
# tasks/__init__.py
from isaaclab_tasks.utils import import_packages
import_packages(__name__, _BLACKLIST_PKGS)  # 递归扫描并导入所有包
```

`import_packages()` 自动发现所有子模块并执行其 `__init__.py`。新增任务不需要修改任何现有代码——只需创建新文件夹 + `__init__.py`。

这不是 GoF 经典模式，而是框架级设计，常见于 Python 的 `entry_points`、Django 的 `INSTALLED_APPS` 等。

### 9.3 配置即代码（Configuration as Code）

Isaac Lab 的 `@configclass` 让配置具有：
- **类型安全**（不同于 YAML/JSON 字典）
- **可继承**（`PlayEnvCfg(EnvCfg)` 通过继承覆盖）
- **IDE 友好**（自动补全、跳转定义）

这结合了 Builder 模式和领域特定语言（DSL）的思想。

---

## 附录 A：模式速查表

| 模式 | Refactoring Guru 链接 | Isaac Lab 中的位置 | 一句话说明 |
|------|----------------------|-------------------|----------|
| **Strategy** | [/strategy](https://refactoringguru.cn/design-patterns/strategy) | `mdp/rewards.py`, `mdp/observations.py` | 奖励/观测函数可替换 |
| **Composite** | [/composite](https://refactoringguru.cn/design-patterns/composite) | `ManagerBasedRLEnv` 内的 Manager-Term 树 | 统一接口遍历所有组件 |
| **Factory Method** | [/factory-method](https://refactoringguru.cn/design-patterns/factory-method) | `gym.register()` + `gym.make()` | 字符串 ID 创建环境 |
| **Builder** | [/builder](https://refactoringguru.cn/design-patterns/builder) | `@configclass` 配置组装 | 逐步构建复杂环境 |
| **Template Method** | [/template-method](https://refactoringguru.cn/design-patterns/template-method) | `ManagerBasedRLEnv.step()` | 固定步骤骨架 |
| **Adapter** | [/adapter](https://refactoringguru.cn/design-patterns/adapter) | `RslRlVecEnvWrapper` | 转换接口格式 |
| **State** | [/state](https://refactoringguru.cn/design-patterns/state) | `deploy/include/FSM/` | 机器人行为状态切换 |

---

## 附录 B：推荐学习路径

### 路径 1：从零开始（约 2-3 天）

```
Day 1 上午：Refactoring Guru 基础
├── 阅读 "什么是设计模式？" 总览页面
├── 阅读 Strategy（策略）完整页面 + Python 代码示例
└── 对照 Isaac Lab 的 rewards.py 理解策略模式

Day 1 下午：组合与工厂
├── 阅读 Composite（组合）完整页面
├── 画出 Isaac Lab 的 Manager-Term 树状图
├── 阅读 Factory Method（工厂方法）完整页面
└── 追踪 gym.register() → gym.make() 的完整流程

Day 2 上午：生成器与模板方法
├── 阅读 Builder（生成器）完整页面
├── 对照 velocity_env_cfg.py 理解逐步组装
├── 阅读 Template Method（模板方法）完整页面
└── 理解 env.step() 的固定骨架

Day 2 下午：适配器与状态
├── 阅读 Adapter（适配器）完整页面
├── 理解 RslRlVecEnvWrapper 的接口转换
├── 阅读 State（状态）完整页面
└── 理解 deploy/ 的 FSM 架构

Day 3：综合实战
├── 从 train.py 第一行开始，逐行追踪每个模式的出现
├── 尝试新增一个自定义奖励函数（体验 Strategy）
├── 尝试新增一个任务 gym.register()（体验 Factory）
└── 回顾整个框架，画出完整的模式协作图
```

### 路径 2：有设计模式基础，快速映射（约半天）

```
1. 直接阅读本教程第八章（综合实战），了解全貌
2. 对照附录 A 速查表，快速定位每个模式的代码位置
3. 重点阅读 Strategy + Composite 两章（这两个是 Isaac Lab 最核心的模式）
4. 浏览 velocity_env_cfg.py 完整文件，体会 Builder + Strategy + Composite 如何协作
```

### 路径 3：只想理解框架，不深入模式理论（约 2 小时）

```
1. 阅读第八章的"模式协作时间线"表格
2. 阅读第一章（Strategy）的 "Isaac Lab 中怎么用的" 部分
3. 阅读第二章（Composite）的 "Isaac Lab 中怎么用的" 部分
4. 跳到附录 A 速查表，从代码入手逆向理解
```

---

> **总结**：Isaac Lab 框架之所以能用统一的 `ManagerBasedRLEnv` 支持从四足狗到人形机器人、从速度跟踪到武术模仿的多样任务，靠的就是这 7 个设计模式的有机组合。Strategy 让算法可替换，Composite 让组件可组合，Factory 让环境可注册，Builder 让配置可组装，Template Method 让流程可固定，Adapter 让框架可对接，State 让部署可切换。掌握了这些，你就掌握了 Isaac Lab 的设计哲学。
