import inspect
import sys
import os

# Add source to path so we can import unitree_rl_lab
sys.path.append(os.path.join(os.getcwd(), "source"))

try:
    from unitree_rl_lab.tasks.locomotion.mdp import feet_slide
    print("Source code for feet_slide:")
    print(inspect.getsource(feet_slide))
except ImportError as e:
    print(f"ImportError: {e}")
except Exception as e:
    print(f"Error: {e}")
