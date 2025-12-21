# Unitree RL Lab Copilot Instructions

## Project Overview
This project provides Reinforcement Learning environments for Unitree robots (Go2, H1, G1) using NVIDIA Isaac Lab. It supports training in Isaac Sim and deployment via C++.

## Key Workflows & Commands

### 1. Environment Management
- **List Tasks:** `./unitree_rl_lab.sh -l` (or `python scripts/list_envs.py`)
- **Install:** `./unitree_rl_lab.sh -i` (installs in editable mode)

### 2. Training (RSL-RL)
- **Command:** `python scripts/rsl_rl/train.py --task <TaskName> --headless`
- **Wrapper:** `./unitree_rl_lab.sh -t --task <TaskName>`
- **Logs:** Stored in `logs/rsl_rl/<ExperimentName>/<Date_Time>`

### 3. Inference / Play
- **Command:** `python scripts/rsl_rl/play.py --task <TaskName>`
- **Wrapper:** `./unitree_rl_lab.sh -p --task <TaskName>`
- **Important:**
  - Requires a trained checkpoint in `logs/rsl_rl/`.
  - Use `--video` to record (auto-terminates after `--video_length` steps).
  - Use `--checkpoint <path>` to specify a model file explicitly.

### 4. Deployment (C++)
- **Location:** `deploy/` directory.
- **Build:** Standard CMake workflow in `deploy/robots/<robot_name>`.
- **Sim2Sim:** Verify trained policies in Mujoco before real robot deployment.

## Codebase Structure
- `source/unitree_rl_lab/`: Main Python package.
  - `tasks/`: Environment definitions (ManagerBasedRLEnv).
  - `assets/`: Robot assets (USD/URDF configuration).
- `scripts/rsl_rl/`: Entry points for training and playing.
- `deploy/`: C++ SDK for robot deployment.

## Common Patterns
- **Config Parsing:** Uses `cli_args.py` to merge RSL-RL and AppLauncher arguments.
- **Environment:** Uses `gym.make` with Isaac Lab's `ManagerBasedRLEnv`.
- **Checkpoints:** `play.py` attempts to auto-locate the latest checkpoint if not provided.

## Troubleshooting
- **Play Termination:** `play.py` terminates if:
  - The simulation window is closed.
  - `--video` is set (stops after fixed steps).
  - No checkpoint is found (check console output).
