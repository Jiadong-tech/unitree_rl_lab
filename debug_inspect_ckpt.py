
import torch
import os

checkpoint_path = "/home/jiadong/unitree_rl_lab/logs/rsl_rl/unitree_g1_29dof_parkour/2026-01-19_18-24-30/model_60100.pt"

if not os.path.exists(checkpoint_path):
    print(f"Checkpoint not found at {checkpoint_path}")
    exit(1)

try:
    print(f"Loading checkpoint: {checkpoint_path}")
    loaded_dict = torch.load(checkpoint_path, map_location="cpu")
    
    model_state = loaded_dict['model_state_dict']
    print("Keys in model_state_dict:", model_state.keys())
    
    if 'std' in model_state:
        std = model_state['std']
        print(f"std values: {std}")
        if torch.isnan(std).any():
            print("CRITICAL: 'std' contains NaNs!")
        if torch.isinf(std).any():
            print("CRITICAL: 'std' contains Infs!")
        if (std < 0).any():
            print("CRITICAL: 'std' contains negative values!")
            
    # Check other weights for NaNs
    for key, val in model_state.items():
        if torch.is_tensor(val):
            if torch.isnan(val).any():
                print(f"CRITICAL: Layer '{key}' contains NaNs!")
            elif torch.isinf(val).any():
                print(f"CRITICAL: Layer '{key}' contains Infs!")
                
    optimizer_state = loaded_dict.get('optimizer_state_dict', {})
    print("Optimizer state checked (summary omitted).")

except Exception as e:
    print(f"Error inspecting checkpoint: {e}")
