# 07 — 部署指南

> **阅读时间**: 15 分钟 · **难度**: ⭐⭐⭐ · **前置要求**: 06_Training_Roadmap
>
> **学完你能**: 导出 ONNX 模型、理解 C++ FSM 状态机的工作方式、知道真机部署前需要做哪些安全检查

---

## 1. 从训练到部署的完整流程

```
训练完成 (model_29999.pt)
    │
    │  play.py 自动导出
    ▼
ONNX + JIT 模型 (exported/ 目录)
    │
    │  手动复制
    ▼
deploy/robots/g1_29dof/config/policy/mimic/martial_arts/
    │
    │  Mujoco 验证
    ▼
Sim2Sim 通过？
    │
    ├── YES → 真机部署 (先吊架, 后落地)
    └── NO  → 回去调参
```

---

## 2. 导出 ONNX

`play.py` 在推理时会**自动导出**模型到 `exported/` 目录：

```bash
# 播放并自动导出
python scripts/rsl_rl/play.py \
    --task Unitree-G1-29dof-Mimic-MartialArts-FrontKick

# 导出文件位置:
# logs/rsl_rl/unitree_g1_29dof_mimic_martialarts_frontkick/<latest_run>/exported/
#   ├── policy.pt    (JIT 格式)
#   └── policy.onnx  (ONNX 格式, C++ 部署用)
```

### 收集所有 ONNX

```bash
# 将所有动作的 ONNX 收集到部署目录
DEPLOY_DIR=deploy/robots/g1_29dof/config/policy/mimic/martial_arts

for motion in front_kick lunge_punch side_kick roundhouse_kick heian_shodan bassai empi; do
    mkdir -p $DEPLOY_DIR/$motion/
    cp logs/rsl_rl/unitree_g1_29dof_mimic_martialarts_${motion}/<latest>/exported/policy.onnx \
       $DEPLOY_DIR/$motion/
done
```

---

## 3. C++ FSM 状态机

### 3.1 状态转换图

```
                     LT + up
    ┌─────────┐  ──────────→  ┌──────────┐  RB + X   ┌──────────┐
    │ Passive │               │ FixStand │  ───────→  │ Velocity │
    │  (安全) │  ←──────────  │  (站立)  │           │  (行走)  │
    └─────────┘   LT + B     └──────────┘           └────┬─────┘
         ▲                         ▲                      │
         │                         │                      │ LT(2s)+right
         │ LT + B                  │ 自动返回              ▼
         │                         │               ┌──────────────┐
         └─────────────────────────┴───────────────│ MartialArts  │
                                                   │ Sequencer    │
                                                   └──────────────┘
                                                   自动播放全部段落
                                                   → 完成后回 FixStand
```

### 3.2 操控方式

| 按键组合 | 动作 |
|---------|------|
| `LT + up` | Passive → FixStand |
| `RB + X` | FixStand → Velocity (行走) |
| `LT(长按2s) + right` | Velocity → **武术表演** |
| `LT + B` | 任何状态 → Passive (**紧急停止**) |

### 3.3 Sequencer 执行流程

```cpp
enter()
  └→ load_segment(0) → start_policy_thread()

policy_loop() [独立线程]:
  for each segment (0..N-1):
    │
    ├─ load_segment(i)           // 加载 ONNX + motion CSV
    ├─ env->reset()              // 初始化观测
    │
    ├─ while elapsed < duration:
    │    env->step()             // ONNX 推理 → action
    │    [run() 在主线程以 1000Hz 写入电机]
    │
    ├─ 保存末尾关节角 (start_q)
    ├─ 加载下一段, 获取起手关节角 (target_q)
    ├─ 线性插值过渡 (transition_hold_s)
    │    alpha = step / total_steps
    │    interp_q = start_q + alpha × (target_q - start_q)
    │
    └─ 下一个 segment

  finished_ = true → 自动回 FixStand
```

### 3.4 安全机制

```cpp
// 1. 演完自动退出
registered_checks: finished_ == true → FixStand

// 2. 摔倒保护
registered_checks: bad_orientation(1.0) → Passive
```

---

## 4. 编排表演

修改 YAML 配置文件即可改变表演顺序：

```yaml
# deploy/robots/g1_29dof/config/config.yaml

MartialArtsSequencer:
  transition_hold_s: 1.0    # 段落间过渡时间

  segments:
    # 自由编排! 可以重复、跳过、调换
    - policy_dir: config/policy/mimic/martial_arts/front_kick/
      motion_file: .../G1_front_kick.csv
      fps: 50

    - policy_dir: config/policy/mimic/martial_arts/lunge_punch/
      motion_file: .../G1_lunge_punch.csv
      fps: 50

    - policy_dir: config/policy/mimic/martial_arts/roundhouse_kick/
      motion_file: .../G1_roundhouse_kick.csv
      fps: 50

    # 可以加更多段落...
```

---

## 5. Sim2Sim 验证 (Mujoco)

在真机前**必须**先在 Mujoco 中验证（PhysX 和 Mujoco 的接触力学有差异）：

### 5.1 验证清单

- [ ] 关节速度峰值 < velocity_limit？
- [ ] 动作轨迹视觉上合理？
- [ ] 连续执行 3 次无摔倒？
- [ ] 过渡时无剧烈跳变？

---

## 6. 真机部署

### 6.1 部署流程

```
1. Sim2Sim 验证通过
    ↓
2. 吊架测试 (机器人悬挂, 脚不着地)
    ├── 检查电机电流峰值
    ├── 检查关节速度
    └── 确认无异常振动
    ↓
3. 降地测试 (脚着地, 有安全绳)
    ├── 单个动作测试
    └── 确认站立稳定
    ↓
4. 自由测试 (无安全绳)
    ├── 连续表演 3 次
    └── 确认鲁棒性
    ↓
5. 正式表演
```

### 6.2 真机安全检查清单

- [ ] 电机电流峰值 < 安全阈值？
- [ ] 关节速度峰值 < velocity_limit_sim？
- [ ] 踢腿时对地反力是否合理？
- [ ] 吊架上连续执行 3 次是否稳定？
- [ ] 紧急停止按钮 (LT + B) 测试通过？

### 6.3 C++ 编译

```bash
cd deploy/robots/g1_29dof
mkdir build && cd build
cmake ..
make -j$(nproc)
```

---

## 7. 改进路线图

### 7.1 短期 (本月)

| 优先级 | 项目 | 状态 |
|--------|------|------|
| ~~P0~~ | Front Kick 姿势修复 | ✅ v5 已完成 |
| ~~P0~~ | 放松 action_rate | ✅ v5 已完成 |
| ~~P0~~ | 末端执行器追踪 | ✅ v5 已完成 |
| ~~P0~~ | 插值过渡 | ✅ C++ 已实现 |
| P1 | 训练 LungePunch | 🔄 待训练 |
| P1 | 训练 SideKick | 🔄 待训练 |

### 7.2 中期 (1-2月)

| 项目 | 说明 |
|------|------|
| 上下半身分离追踪 | 扩展 `rewards.py` 支持 `joint_names` 过滤 |
| 重心约束 (CoM/ZMP) | 单腿支撑时质心投影须在支撑脚上方 |
| 鲁棒过渡训练 | 加大 `joint_position_range` 和 `pose_range` |
| 对称性训练 | 随机翻转左右，右踢学会左踢 |

### 7.3 长期 (高级研究方向)

| 项目 | 说明 |
|------|------|
| 视觉打靶 | 加入目标沙袋，奖励接触力 |
| 对抗训练 | 训练"对手"推扰，互相博弈 |
| 风格迁移 | 保持关键帧，RL 自由发挥过渡 |
| 电机建模 | 反电动势、发热限流等约束 |

---

## 检查清单

- [ ] play.py 导出的 ONNX 在哪个目录下？
- [ ] 紧急停止的按键组合是什么？
- [ ] 部署前为什么要先做 Sim2Sim 验证？
- [ ] 编排表演顺序需要修改哪个文件？

---

**返回** → [00_WuBot_Learning_Index.md](00_WuBot_Learning_Index.md) 查看学习导航
