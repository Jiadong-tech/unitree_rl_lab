#!/usr/bin/env bash
# =============================================================================
# 🥋 Martial Arts Data Pipeline — CMU MoCap → CSV → NPZ → Training
# =============================================================================
#
# This single script orchestrates the entire martial arts data + training
# pipeline.  Run it with a STAGE argument to execute a specific step, or
# run without arguments to see help.
#
# Usage:
#   bash scripts/mimic/martial_arts_pipeline.sh csv        # Step 1: AMC → CSV
#   bash scripts/mimic/martial_arts_pipeline.sh npz        # Step 2: CSV → NPZ (Isaac Sim)
#   bash scripts/mimic/martial_arts_pipeline.sh validate   # Step 3: Validate NPZ
#   bash scripts/mimic/martial_arts_pipeline.sh train [GROUP]  # Step 4: Train
#   bash scripts/mimic/martial_arts_pipeline.sh all        # Run steps 1-4
#
# Prerequisites:
#   - Step csv:      numpy, scipy
#   - Step npz:      Isaac Sim / Isaac Lab activated
#   - Step train:    Isaac Sim / Isaac Lab + unitree_rl_lab installed
# =============================================================================

set -e

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

CMU_DATA_DIR="$REPO_ROOT/data/cmu_mocap/135"
ASF_FILE="$CMU_DATA_DIR/135.asf"
MA_DIR="$REPO_ROOT/source/unitree_rl_lab/unitree_rl_lab/tasks/mimic/robots/g1_29dof/martial_arts"
CMU_TO_CSV="$SCRIPT_DIR/cmu_amc_to_csv.py"
CSV_TO_NPZ="$SCRIPT_DIR/csv_to_npz.py"
VALIDATE="$SCRIPT_DIR/validate_npz.py"

INPUT_FPS=120   # CMU MoCap frame rate
OUTPUT_FPS=50   # Isaac Sim control rate (decimation=4, dt=0.005)

# CMU #135 file → motion name mapping
declare -A MOTION_MAP=(
    ["135_01"]="G1_bassai"
    ["135_02"]="G1_empi"
    ["135_03"]="G1_front_kick"
    ["135_04"]="G1_heian_shodan"
    ["135_05"]="G1_roundhouse_kick"
    ["135_06"]="G1_lunge_punch"
    ["135_07"]="G1_side_kick"
)

# Task IDs for training (easy → hard)
declare -A TASKS=(
    ["front_kick"]="Unitree-G1-29dof-Mimic-MartialArts-FrontKick"
    ["lunge_punch"]="Unitree-G1-29dof-Mimic-MartialArts-LungePunch"
    ["side_kick"]="Unitree-G1-29dof-Mimic-MartialArts-SideKick"
    ["roundhouse_kick"]="Unitree-G1-29dof-Mimic-MartialArts-RoundhouseKick"
    ["heian_shodan"]="Unitree-G1-29dof-Mimic-MartialArts-HeianShodan"
    ["bassai"]="Unitree-G1-29dof-Mimic-MartialArts-Bassai"
    ["empi"]="Unitree-G1-29dof-Mimic-MartialArts-Empi"
)
TRAIN_ORDER=(front_kick lunge_punch side_kick roundhouse_kick heian_shodan bassai empi)

# Colors
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'

# ---------------------------------------------------------------------------
# Stage 1: CMU ASF+AMC → G1 CSV
# ---------------------------------------------------------------------------
stage_csv() {
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}  Stage 1: CMU ASF+AMC → G1 CSV${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

    if [ ! -s "$ASF_FILE" ]; then
        echo -e "${RED}ERROR: ASF file missing: $ASF_FILE${NC}"
        echo "Place CMU #135 data in: $CMU_DATA_DIR/"
        exit 1
    fi

    local ok=0 fail=0
    for key in "${!MOTION_MAP[@]}"; do
        local amc="$CMU_DATA_DIR/${key}.amc"
        local name="${MOTION_MAP[$key]}"
        local csv="$MA_DIR/${name}.csv"
        [ ! -f "$amc" ] && echo -e "${YELLOW}⏭  $key.amc not found, skip${NC}" && continue

        echo -e "${BLUE}  $key.amc → $name.csv${NC}"
        if python "$CMU_TO_CSV" --asf "$ASF_FILE" --amc "$amc" -o "$csv" --fps $INPUT_FPS; then
            echo -e "${GREEN}  ✓ $name.csv${NC}"; ok=$((ok+1))
        else
            echo -e "${RED}  ✗ Failed${NC}"; fail=$((fail+1))
        fi
    done
    echo -e "\n  Converted: ${GREEN}$ok${NC}  Failed: ${RED}$fail${NC}"
}

# ---------------------------------------------------------------------------
# Stage 2: CSV → NPZ (requires Isaac Sim)
# ---------------------------------------------------------------------------
stage_npz() {
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}  Stage 2: CSV → NPZ (Isaac Sim)${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

    local csvs=("$MA_DIR"/G1_*.csv)
    if [ ${#csvs[@]} -eq 0 ]; then
        echo -e "${RED}No CSV files found. Run 'csv' stage first.${NC}"; exit 1
    fi

    local ok=0 fail=0 skip=0
    for csv in "${csvs[@]}"; do
        local base=$(basename "$csv")
        local npz="${csv%.csv}.npz"
        if [ -f "$npz" ] && [ "$1" != "--force" ]; then
            echo -e "${YELLOW}⏭  $base (NPZ exists, use --force to overwrite)${NC}"
            skip=$((skip+1)); continue
        fi
        echo -e "${BLUE}  $base → $(basename "$npz")${NC}"
        if python "$CSV_TO_NPZ" -f "$csv" --input_fps $INPUT_FPS --output_fps $OUTPUT_FPS --headless; then
            echo -e "${GREEN}  ✓ $(basename "$npz")${NC}"; ok=$((ok+1))
        else
            echo -e "${RED}  ✗ Failed${NC}"; fail=$((fail+1))
        fi
    done
    echo -e "\n  Converted: ${GREEN}$ok${NC}  Skipped: ${YELLOW}$skip${NC}  Failed: ${RED}$fail${NC}"
}

# ---------------------------------------------------------------------------
# Stage 3: Validate NPZ
# ---------------------------------------------------------------------------
stage_validate() {
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}  Stage 3: Validate NPZ files${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    python "$VALIDATE" -d "$MA_DIR"
}

# ---------------------------------------------------------------------------
# Stage 4: Train
# ---------------------------------------------------------------------------
stage_train() {
    local group="${1:-all}"
    local order=()

    case "$group" in
        kick|kicks)    order=(front_kick side_kick roundhouse_kick) ;;
        punch)         order=(lunge_punch) ;;
        kata)          order=(heian_shodan bassai empi) ;;
        all)           order=("${TRAIN_ORDER[@]}") ;;
        *)
            if [[ -v "TASKS[$group]" ]]; then
                order=("$group")
            else
                echo -e "${RED}Unknown group: $group${NC}"
                echo "Options: all | kicks | punch | kata | <task_key>"
                echo "Keys: ${!TASKS[*]}"; exit 1
            fi ;;
    esac

    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}  Stage 4: Training (${#order[@]} tasks)${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

    local ok=0 fail=0
    for key in "${order[@]}"; do
        local tid="${TASKS[$key]}"
        echo -e "\n${YELLOW}▶ $key ($tid)${NC}"
        if python scripts/rsl_rl/train.py --task "$tid" --headless; then
            echo -e "${GREEN}  ✓ $key${NC}"; ok=$((ok+1))
        else
            echo -e "${RED}  ✗ $key${NC}"; fail=$((fail+1))
        fi
    done
    echo -e "\n  Succeeded: ${GREEN}$ok${NC}  Failed: ${RED}$fail${NC}"
}

# ---------------------------------------------------------------------------
# Stage 5: Deploy — collect trained policies for C++ sequencer
# ---------------------------------------------------------------------------
DEPLOY_DIR="$REPO_ROOT/deploy/robots/g1_29dof/config/policy/mimic/martial_arts"

stage_deploy() {
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}  Stage 5: Collect policies for deployment${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

    local ok=0 skip=0
    for key in "${TRAIN_ORDER[@]}"; do
        local task_id="${TASKS[$key]}"
        # Find latest log dir for this task
        local exp_name=$(echo "$task_id" | tr '[:upper:]' '[:lower:]' | tr '-' '_')
        local log_base="$REPO_ROOT/logs/rsl_rl"
        local latest_log=$(find "$log_base" -maxdepth 2 -name "$exp_name" -type d 2>/dev/null | head -1)

        if [ -z "$latest_log" ]; then
            echo -e "${YELLOW}⏭  $key — no training logs found${NC}"
            skip=$((skip+1)); continue
        fi

        # Find the latest timestamped run
        local latest_run=$(ls -dt "$latest_log"/*/ 2>/dev/null | head -1)
        if [ -z "$latest_run" ]; then
            echo -e "${YELLOW}⏭  $key — no runs found in $latest_log${NC}"
            skip=$((skip+1)); continue
        fi

        local dest="$DEPLOY_DIR/$key"
        mkdir -p "$dest/exported" "$dest/params"

        # Copy ONNX policy
        if [ -f "$latest_run/exported/policy.onnx" ]; then
            cp "$latest_run/exported/policy.onnx" "$dest/exported/"
        else
            echo -e "${YELLOW}⏭  $key — no policy.onnx found${NC}"
            skip=$((skip+1)); continue
        fi

        # Copy deploy config
        if [ -f "$latest_run/params/deploy.yaml" ]; then
            cp "$latest_run/params/deploy.yaml" "$dest/params/"
        fi

        # Copy motion CSV
        local csv="$MA_DIR/G1_${key}.csv"
        if [ -f "$csv" ]; then
            cp "$csv" "$dest/params/"
        fi

        echo -e "${GREEN}  ✓ $key → $dest${NC}"; ok=$((ok+1))
    done
    echo -e "\n  Deployed: ${GREEN}$ok${NC}  Skipped: ${YELLOW}$skip${NC}"
    echo -e "\n${BLUE}Deploy directory: $DEPLOY_DIR${NC}"
    echo -e "${BLUE}Config.yaml MartialArtsSequencer segments point to these paths.${NC}"
}


# ---------------------------------------------------------------------------
# Help / Entry
# ---------------------------------------------------------------------------
show_help() {
    echo -e "${BLUE}🥋 Martial Arts Pipeline${NC}"
    echo ""
    echo "Usage: $0 <stage> [options]"
    echo ""
    echo "Stages:"
    echo "  csv              CMU ASF+AMC → G1 CSV  (needs numpy, scipy)"
    echo "  npz [--force]    CSV → NPZ             (needs Isaac Sim)"
    echo "  validate         Validate NPZ files"
    echo "  train [GROUP]    Train policies         (needs Isaac Sim + RSL-RL)"
    echo "                   GROUP: all|kicks|punch|kata|<task_key>"
    echo "  deploy           Collect trained policies for C++ sequencer"
    echo "  all              Run csv → npz → validate → train all"
    echo ""
    echo "Data locations:"
    echo "  CMU raw data:    $CMU_DATA_DIR/"
    echo "  CSV/NPZ data:    $MA_DIR/"
    echo "  Training logs:   logs/rsl_rl/"
    echo "  Deploy policies: $DEPLOY_DIR/"
}

# Main dispatch
case "${1:-help}" in
    csv)        stage_csv ;;
    npz)        stage_npz "$2" ;;
    validate)   stage_validate ;;
    train)      stage_train "$2" ;;
    deploy)     stage_deploy ;;
    all)
        stage_csv
        echo ""
        stage_npz
        echo ""
        stage_validate
        echo ""
        stage_train all
        echo ""
        stage_deploy
        ;;
    *)          show_help ;;
esac
