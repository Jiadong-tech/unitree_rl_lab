
import torch
import os
import argparse

def fix_checkpoint(path, obs_dim=2735):
    print(f"Processing checkpoint: {path}")
    if not os.path.exists(path):
        print("Error: File not found.")
        return

    # Load on CPU
    data = torch.load(path, map_location="cpu")
    
    # Check if obs_norm_state_dict exists
    if "obs_norm_state_dict" in data:
        print("Checkpoint already has 'obs_norm_state_dict'. No changes needed.")
        return

    print("Injecting default 'obs_norm_state_dict'...")
    
    # Create default running mean/var (Standard Normal)
    # RSL-RL RunningMeanStd expects: 'count', 'running_mean', 'running_var'
    obs_norm_state_dict = {
        "count": torch.tensor([1.0], dtype=torch.float),
        "running_mean": torch.zeros(obs_dim, dtype=torch.float),
        "running_var": torch.ones(obs_dim, dtype=torch.float)
    }
    
    data["obs_norm_state_dict"] = obs_norm_state_dict
    
    # Save back
    backup_path = path + ".bak"
    os.rename(path, backup_path)
    print(f"Backed up original to: {backup_path}")
    
    torch.save(data, path)
    print(f"Successfully saved patched checkpoint to: {path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Path to the .pt checkpoint file")
    args = parser.parse_args()
    
    fix_checkpoint(args.path)
