import numpy as np
import os

# 路径指向 Gangnam Style 的动作数据文件
file_path = 'source/unitree_rl_lab/unitree_rl_lab/tasks/mimic/robots/g1_29dof/gangnanm_style/G1_gangnam_style_V01.bvh_60hz.npz'

if not os.path.exists(file_path):
    print(f"错误: 找不到文件 {file_path}")
else:
    # 加载数据
    data = np.load(file_path)

    print("="*50)
    print(f"正在查看文件: {os.path.basename(file_path)}")
    print("="*50)

    # 打印所有的键（Keys）
    print(f"数据包含的键 (Keys): {data.files}")

    # 遍历每个键，打印其维度（Shape）
    for key in data.files:
        print(f"- 键 '{key}' 的维度 (Shape): {data[key].shape}")

    # 详细打印 motion 数据的前几行
    if 'motion' in data:
        print("\n" + "-"*30)
        print("Motion 数据采样 (前 5 帧, 前 10 列):")
        # 前 5 帧，前 10 个维度（通常是 root pos, root ori, joint pos 等）
        print(data['motion'][:5, :10])
        print("-"*30)

    print("\n提示: motion 数据的每一行代表一帧，每一列通常对应关节角度或躯干状态。")
