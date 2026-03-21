#!/bin/bash
# Regenerate all 7 martial arts NPZ files from CSV source.
#
# WHY: The current NPZ files were corrupted by fix_npz_motion_quality.py
#      which used wrong URDF joint limits (20/29 joints had scrambled values).
#      The CSV source data is correct (max URDF violation = 0.0004 rad).
#
# csv_to_npz.py now clamps joints to URDF soft limits before FK,
# ensuring consistent joint_pos + body_pos_w data.
#
# USAGE:
#   conda activate env_isaaclab
#   cd /home/jiadong/unitree_rl_lab
#   bash scripts/mimic/regenerate_all_npz.sh
#
# NOTE: Requires Isaac Sim (uses sim.render() for FK).
#       Each file takes ~30-60 seconds. Total ~5-7 minutes.

set -e

MA_DIR="source/unitree_rl_lab/unitree_rl_lab/tasks/mimic/robots/g1_29dof/martial_arts"

# Backup current NPZ files
BACKUP_DIR="${MA_DIR}/backup_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"
echo "[INFO] Backing up current NPZ files to $BACKUP_DIR"
cp ${MA_DIR}/G1_*.npz "$BACKUP_DIR/" 2>/dev/null || true

# All motions with their input FPS
# CMU MoCap Subject 135 data was captured at 120fps
declare -A MOTIONS
MOTIONS["G1_bassai"]=120
MOTIONS["G1_empi"]=120
MOTIONS["G1_front_kick"]=120
MOTIONS["G1_heian_shodan"]=120
MOTIONS["G1_lunge_punch"]=120
MOTIONS["G1_roundhouse_kick"]=120
MOTIONS["G1_side_kick"]=120

echo ""
echo "============================================"
echo " Regenerating all martial arts NPZ files"
echo "============================================"

for MOTION in "${!MOTIONS[@]}"; do
    FPS=${MOTIONS[$MOTION]}
    CSV="${MA_DIR}/${MOTION}.csv"
    
    if [ ! -f "$CSV" ]; then
        echo "[WARN] CSV not found: $CSV, skipping"
        continue
    fi
    
    echo ""
    echo "--- Processing: $MOTION (input_fps=$FPS) ---"
    python scripts/mimic/csv_to_npz.py \
        -f "$CSV" \
        --input_fps "$FPS" \
        --headless
    
    echo "[OK] $MOTION.npz regenerated"
done

echo ""
echo "============================================"
echo " All NPZ files regenerated successfully!"
echo " Backup saved to: $BACKUP_DIR"
echo "============================================"
echo ""
echo "Next steps:"
echo "  1. Verify with: python -c 'import numpy as np; d=np.load(\"${MA_DIR}/G1_side_kick.npz\"); print(d[\"joint_pos\"].shape)'"
echo "  2. Train: python scripts/rsl_rl/train.py --headless --task Unitree-G1-29dof-Mimic-MartialArts-SideKick --num_envs 512"
