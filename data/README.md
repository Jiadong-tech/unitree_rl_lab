# Martial Arts Motion Data
# =======================
# This directory holds intermediate files for the motion capture pipeline.
#
# Workflow: FBX → BVH → CSV → NPZ
#
# data/
# ├── fbx/    ← Download from Mixamo (FBX Binary, Without Skin, 60fps)
# ├── bvh/    ← Converted from FBX using fbx_to_bvh.py (Blender)
# └── csv/    ← Retargeted to G1 skeleton using retarget_bvh_to_g1.py (Blender)
#
# Final NPZ files go to:
#   source/unitree_rl_lab/unitree_rl_lab/tasks/mimic/robots/g1_29dof/martial_arts/
#
# See martial_arts/README.md for full instructions.
