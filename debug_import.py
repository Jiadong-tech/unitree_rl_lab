
import sys
import os

# Add the source directory to sys.path explicitly to mimic editable install if needed, 
# but let's first see what standard import does.
sys.path.append(os.path.abspath("source"))

try:
    from unitree_rl_lab.tasks.mimic.robots.g1_29dof.petite_verses.tracking_env_cfg import CommandsCfg
    print(f"Imported from: {sys.modules['unitree_rl_lab.tasks.mimic.robots.g1_29dof.petite_verses.tracking_env_cfg'].__file__}")
    print(f"Motion file in config: {CommandsCfg.motion.motion_file}")
except ImportError as e:
    print(f"ImportError: {e}")
except Exception as e:
    print(f"Error: {e}")
