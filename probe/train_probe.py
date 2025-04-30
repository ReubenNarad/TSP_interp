import os
import argparse
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import json
import sys
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from policy.policy_hooked import HookedAttentionModelPolicy
from policy.reinforce_clipped import REINFORCEClipped

# Simplified train_probe, only for regression
def train_regression_probe(probe_model, loader, loss_fn, optimizer, num_epochs, device, task_name="Regression"):
    """Helper function to train a regression probe."""
    losses = []
    for epoch in range(num_epochs):
        probe_model.train()
        epoch_losses = []
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = probe_model(xb).squeeze(-1) # Direct prediction
            loss = loss_fn(pred, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        mean_loss = np.mean(epoch_losses)
        losses.append(mean_loss)
        log_str = f"Epoch {epoch+1}/{num_epochs} [{task_name}] - Loss (MSE): {mean_loss:.6f}"

        if (epoch+1) % 10 == 0 or epoch == 0:
            print(log_str)

    results = {"losses": losses}
    return results

def main(args):
    run_path = f"runs/{args.run_name}"
    probe_dir = os.path.join(run_path, "probe")
    os.makedirs(probe_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Data Loading ---
    print("--- Loading Data ---")
    # Load config
    with open(os.path.join(run_path, "config.json"), "r") as f:
        config = json.load(f)
    # Load environment
    env_path = os.path.join(run_path, "env.pkl")
    with open(env_path, "rb") as f:
        env = pickle.load(f)

    # Load generated instances and metrics from baseline_probe.pkl
    baseline_probe_path = os.path.join(probe_dir, "baseline_probe.pkl")
    print(f"Attempting to load data from: {baseline_probe_path}")
    with open(baseline_probe_path, "rb") as f:
        probe_data = pickle.load(f)

    if "locs" not in probe_data:
        raise KeyError("'locs' key not found in baseline_probe.pkl.")
    if "run_times" not in probe_data or not isinstance(probe_data["run_times"], list) or len(probe_data["run_times"]) == 0:
         raise KeyError("'run_times' key not found or has incorrect format.")

    locs = probe_data["locs"].to(device)
    run_times_list = probe_data["run_times"][0]
    num_instances = locs.shape[0]
    num_run_times = len(run_times_list)
    print(f"DEBUG: Loaded locs shape: {locs.shape}, Found {num_run_times} run times.")

    # Convert run_times to tensor and ensure consistency
    run_times_tensor = torch.tensor(run_times_list, dtype=torch.float).to(device)

    if num_instances != num_run_times:
        print(f"WARNING: Mismatch between locs ({num_instances}) and run times ({num_run_times}). Using {min(num_instances, num_run_times)}.")
        num_instances = min(num_instances, num_run_times)
        locs = locs[:num_instances]
        y_all = run_times_tensor[:num_instances]
    else:
         y_all = run_times_tensor

    print(f"Using {num_instances} instances for training and plotting.")

    # Extract LP statistics
    lp_rows = probe_data["lp_rows"][0]
    lp_cols = probe_data["lp_cols"][0] 
    lp_nonzeros = probe_data["lp_nonzeros"][0]
    
    # Convert to numpy for plotting
    rows_np = lp_rows.numpy()
    cols_np = lp_cols.numpy()
    nonzeros_np = lp_nonzeros.numpy()
    
    # Filter out any negative values (failed instances)
    valid_mask = (rows_np > 0) & (cols_np > 0) & (nonzeros_np > 0)
    rows_np = rows_np[valid_mask]
    cols_np = cols_np[valid_mask]
    nonzeros_np = nonzeros_np[valid_mask]
    
    # Plot histograms for LP statistics
    
    # 1. LP Rows Histogram
    plt.figure(figsize=(10, 6))
    plt.hist(rows_np, bins=30, color='lightgreen', edgecolor='black')
    plt.title(f'Histogram of LP Rows ({num_instances} instances)')
    plt.xlabel('Number of Rows')
    plt.ylabel('Frequency')
    plt.grid(True, axis='y', alpha=0.5)
    rows_hist_path = os.path.join(probe_dir, 'lp_rows_histogram.png')
    plt.savefig(rows_hist_path)
    plt.close()
    print(f"Saved LP rows histogram to {rows_hist_path}")
    
    # 2. LP Columns Histogram
    plt.figure(figsize=(10, 6))
    plt.hist(cols_np, bins=30, color='lightblue', edgecolor='black')
    plt.title(f'Histogram of LP Columns ({num_instances} instances)')
    plt.xlabel('Number of Columns')
    plt.ylabel('Frequency')
    plt.grid(True, axis='y', alpha=0.5)
    cols_hist_path = os.path.join(probe_dir, 'lp_cols_histogram.png')
    plt.savefig(cols_hist_path)
    plt.close()
    print(f"Saved LP columns histogram to {cols_hist_path}")
    
    # 3. LP Nonzeros Histogram
    plt.figure(figsize=(10, 6))
    plt.hist(nonzeros_np, bins=30, color='salmon', edgecolor='black')
    plt.title(f'Histogram of LP Nonzeros ({num_instances} instances)')
    plt.xlabel('Number of Nonzeros')
    plt.ylabel('Frequency')
    plt.grid(True, axis='y', alpha=0.5)
    nonzeros_hist_path = os.path.join(probe_dir, 'lp_nonzeros_histogram.png')
    plt.savefig(nonzeros_hist_path)
    plt.close()
    print(f"Saved LP nonzeros histogram to {nonzeros_hist_path}")
    
    # 4. Run times histogram (existing code)
    # Plot histogram
    hist_path = os.path.join(probe_dir, 'run_times_histogram.png')
    regenerate_plot = True
    if regenerate_plot:
        plt.figure(figsize=(10, 6))
        plt.hist(y_all.cpu().numpy(), bins=30, color='skyblue', edgecolor='black')
        plt.title(f'Histogram of Concorde Run Times ({num_instances} instances)')
        plt.xlabel('Run Time (seconds)')
        plt.ylabel('Frequency')
        plt.grid(True, axis='y', alpha=0.5)
        plt.savefig(hist_path)
        plt.close()
        print(f"Saved run time histogram for {num_instances} instances to {hist_path}")

    # --- Activation Loading ---
    print("--- Loading Model and Getting Activations ---")
    # Load model checkpoint
    checkpoint_dir = os.path.join(run_path, "checkpoints")
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{args.epoch}.ckpt")
    if not os.path.exists(checkpoint_path):
        import glob
        checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "checkpoint_epoch_*.ckpt"))
        if not checkpoint_files: raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
        checkpoint_path = max(checkpoint_files, key=os.path.getctime)
        print(f"Using latest checkpoint: {checkpoint_path}")

    # Build policy and model
    policy = HookedAttentionModelPolicy(
        env_name=env.name,
        embed_dim=config["embed_dim"],
        num_encoder_layers=config["n_encoder_layers"],
        num_heads=8,
        temperature=config["temperature"],
    )
    model = REINFORCEClipped.load_from_checkpoint(
        checkpoint_path, env=env, policy=policy, strict=False
    )
    model.eval().to(device)
    policy = model.policy.to(device)

    # Get activations for the loaded instances
    instance_data_td = {"locs": locs}
    with torch.no_grad():
        encoder_out, _ = policy.encoder(instance_data_td)
        X_all = encoder_out.mean(dim=1) # Use mean activation across nodes
        feat_dim = X_all.shape[1]

    # --- Data Setup for Single Regression Probe ---
    print(f"--- Setting up data for regression probe on all {num_instances} instances ---")
    dataset = TensorDataset(X_all, y_all) # Use all data
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # --- Train Single Regression Probe ---
    print(f"\n--- Training Regression Probe (Predicting time for all instances) ---")
    probe_reg = nn.Linear(feat_dim, 1).to(device)
    optimizer_reg = optim.Adam(probe_reg.parameters(), lr=args.lr)
    loss_fn_reg = nn.MSELoss() # Use MSE for regression

    # Use the simplified training function
    reg_log = train_regression_probe(probe_reg, loader, loss_fn_reg, optimizer_reg, args.num_epochs, device, "Regression (All Data)")

    # --- Save Results ---
    # Use simple filenames now
    model_name = "linear_probe.pt"
    log_name = "probe_train_log.json"

    torch.save(probe_reg.state_dict(), os.path.join(probe_dir, model_name))
    with open(os.path.join(probe_dir, log_name), "w") as f:
        json.dump(reg_log, f, indent=2)
    print(f"Saved Regression probe to {os.path.join(probe_dir, model_name)}")
    print(f"Saved training log to {os.path.join(probe_dir, log_name)}")

    print("\n--- Probe Training Complete ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a single regression probe for Concorde run time.")
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--epoch", type=int, required=True, help="Epoch of model checkpoint to use")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()
    main(args) 