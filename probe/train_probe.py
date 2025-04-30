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
def train_regression_probe(probe_model, loader, loss_fn, optimizer, num_epochs, device, task_name="Regression", patience=50):
    """Helper function to train a regression probe."""
    losses = []
    best_loss = float('inf')
    patience_counter = 0
    
    print(f"Training for up to {num_epochs} epochs with patience={patience}...")
    
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

        # Print more frequently for longer training runs
        if (epoch+1) % 25 == 0 or epoch == 0:
            print(log_str)
            
        # Early stopping check
        if mean_loss < best_loss:
            best_loss = mean_loss
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1} as loss hasn't improved for {patience} epochs.")
            print(f"Best loss: {best_loss:.6f}")
            break

    results = {"losses": losses, "best_loss": best_loss}
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
    print(f"Attempting to load probe data from: {baseline_probe_path}")
    with open(baseline_probe_path, "rb") as f:
        probe_data = pickle.load(f)

    if "locs" not in probe_data:
        raise KeyError("'locs' key not found in baseline_probe.pkl.")
    # Load LP Nonzeros instead of run_times
    if "lp_nonzeros" not in probe_data or not isinstance(probe_data["lp_nonzeros"], list) or len(probe_data["lp_nonzeros"]) == 0:
         raise KeyError("'lp_nonzeros' key not found or has incorrect format.")

    locs = probe_data["locs"].to(device)
    # Target variable is now LP Nonzeros
    lp_nonzeros_list = probe_data["lp_nonzeros"][0]
    num_instances = locs.shape[0]
    num_nonzeros = len(lp_nonzeros_list)
    print(f"DEBUG: Loaded locs shape: {locs.shape}, Found {num_nonzeros} LP nonzeros values.")

    # Convert lp_nonzeros to tensor and ensure consistency
    # Convert to float for regression target
    lp_nonzeros_tensor = torch.tensor(lp_nonzeros_list, dtype=torch.float).to(device)

    if num_instances != num_nonzeros:
        print(f"WARNING: Mismatch between locs ({num_instances}) and nonzeros ({num_nonzeros}). Using {min(num_instances, num_nonzeros)}.")
        num_instances = min(num_instances, num_nonzeros)
        locs = locs[:num_instances]
        y_all = lp_nonzeros_tensor[:num_instances] # Target is nonzeros
    else:
         y_all = lp_nonzeros_tensor # Target is nonzeros

    print(f"Using {num_instances} instances for training and plotting.")

    # Extract other LP statistics if needed for context or checks, but nonzeros is the target
    lp_rows = probe_data["lp_rows"][0]
    lp_cols = probe_data["lp_cols"][0]
    # lp_nonzeros = probe_data["lp_nonzeros"][0] # Already loaded as target

    # Print raw nonzeros stats before normalization
    print(f"Raw nonzeros stats: min={y_all.min().item():.1f}, max={y_all.max().item():.1f}, mean={y_all.mean().item():.1f}, std={y_all.std().item():.1f}")

    # Convert to numpy for plotting
    rows_np = lp_rows.numpy()
    cols_np = lp_cols.numpy()
    # Use the target tensor for plotting nonzeros
    nonzeros_np = y_all.cpu().numpy()

    # Filter out any negative values (failed instances) from all stats if plotting them
    valid_mask = (rows_np > 0) & (cols_np > 0) & (nonzeros_np >= 0) # Nonzeros can be 0
    # Only plot nonzeros histogram now
    nonzeros_np_valid = nonzeros_np[valid_mask]

    # --- Plotting Section (Only Nonzeros Histogram) ---
    hist_path = os.path.join(probe_dir, 'lp_nonzeros_histogram.png')
    regenerate_plot = True # Or make this an arg
    if regenerate_plot:
        plt.figure(figsize=(10, 6))
        plt.hist(nonzeros_np_valid, bins=30, color='salmon', edgecolor='black')
        plt.title(f'Histogram of LP Nonzeros ({len(nonzeros_np_valid)} valid instances)')
        plt.xlabel('Number of Nonzeros')
        plt.ylabel('Frequency')
        plt.grid(True, axis='y', alpha=0.5)
        plt.savefig(hist_path)
        plt.close()
        print(f"Saved LP nonzeros histogram to {hist_path}")


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

    # --- Normalize target values before training ---
    print("--- Normalizing target values ---")
    # Calculate mean and std of nonzeros
    y_mean = y_all.mean()
    y_std = y_all.std()
    # Store these for denormalizing predictions later
    normalization_stats = {"mean": y_mean.item(), "std": y_std.item()}
    
    # Normalize targets
    y_normalized = (y_all - y_mean) / y_std
    print(f"Normalized nonzeros stats: min={y_normalized.min().item():.4f}, max={y_normalized.max().item():.4f}, mean={y_normalized.mean().item():.4f}, std={y_normalized.std().item():.4f}")
    
    # Save normalization stats for later use (e.g., when applying the probe)
    with open(os.path.join(probe_dir, "normalization_stats.json"), "w") as f:
        json.dump(normalization_stats, f)
    
    # --- Data Setup for Single Regression Probe ---
    print(f"--- Setting up data for regression probe on all {num_instances} instances ---")
    # Use normalized values for training
    dataset = TensorDataset(X_all, y_normalized) # Use normalized y values
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # --- Train Single Regression Probe ---
    print(f"\n--- Training Regression Probe (Predicting LP Nonzeros) ---")
    probe_reg = nn.Linear(feat_dim, 1).to(device)
    optimizer_reg = optim.Adam(probe_reg.parameters(), lr=args.lr)
    loss_fn_reg = nn.MSELoss() # Use MSE for regression

    # Use the simplified training function
    reg_log = train_regression_probe(
        probe_reg, 
        loader, 
        loss_fn_reg, 
        optimizer_reg, 
        args.num_epochs, 
        device, 
        task_name="Regression (LP Nonzeros)",
        patience=args.patience
    )

    # Evaluate probe on training data
    print("\n--- Evaluating probe on training data ---")
    probe_reg.eval()
    with torch.no_grad():
        # Get predictions for all data
        pred_normalized = probe_reg(X_all).squeeze(-1)
        # Denormalize predictions
        pred_original = pred_normalized * y_std + y_mean
        
        # Calculate MSE on original scale
        mse_original = ((pred_original - y_all) ** 2).mean().item()
        # Calculate MAE (Mean Absolute Error) 
        mae_original = (pred_original - y_all).abs().mean().item()
        # Calculate MAPE (Mean Absolute Percentage Error) - skip zero values to avoid division by zero
        non_zero_mask = y_all != 0
        mape = (((pred_original[non_zero_mask] - y_all[non_zero_mask]).abs() / y_all[non_zero_mask]) * 100).mean().item()
        
        # Calculate R-squared (coefficient of determination)
        y_mean_val = y_all.mean()
        ss_total = ((y_all - y_mean_val) ** 2).sum().item()
        ss_residual = ((y_all - pred_original) ** 2).sum().item()
        r_squared = 1 - (ss_residual / ss_total)
        
        print(f"Evaluation Metrics:")
        print(f"MSE on original scale: {mse_original:.2f}")
        print(f"MAE on original scale: {mae_original:.2f}")
        print(f"MAPE: {mape:.2f}%")
        print(f"R-squared: {r_squared:.4f}")
        
        # Calculate percentage of predictions within X% of actual value
        error_thresholds = [10, 20, 30, 50]
        for thresh in error_thresholds:
            pct_errors = ((pred_original[non_zero_mask] - y_all[non_zero_mask]).abs() / y_all[non_zero_mask]) * 100
            within_thresh = (pct_errors <= thresh).float().mean().item() * 100
            print(f"Predictions within {thresh}% error: {within_thresh:.2f}%")
        
        # Print some sample predictions
        print("\nSample predictions (original scale):")
        num_samples = min(10, len(X_all))
        indices = torch.randperm(len(X_all))[:num_samples]
        for i, idx in enumerate(indices):
            actual = y_all[idx].item()
            predicted = pred_original[idx].item()
            error = abs(predicted - actual)
            error_percent = 100 * error / actual if actual != 0 else float('inf')
            print(f"Sample {i+1}: Actual={actual:.1f}, Predicted={predicted:.1f}, Error={error:.1f} ({error_percent:.1f}%)")
            
        # Create and save a scatter plot of predictions vs actuals
        plt.figure(figsize=(10, 8))
        plt.scatter(y_all.cpu().numpy(), pred_original.cpu().numpy(), alpha=0.5)
        
        # Add a perfect prediction line
        min_val = min(y_all.min().item(), pred_original.min().item())
        max_val = max(y_all.max().item(), pred_original.max().item())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        
        plt.xlabel('Actual LP Nonzeros')
        plt.ylabel('Predicted LP Nonzeros')
        plt.title('Predicted vs Actual LP Nonzeros')
        plt.grid(True, alpha=0.3)
        
        # Add stats to the plot
        plt.text(0.05, 0.95, f'R² = {r_squared:.4f}', transform=plt.gca().transAxes)
        plt.text(0.05, 0.90, f'MAE = {mae_original:.2f}', transform=plt.gca().transAxes)
        plt.text(0.05, 0.85, f'MAPE = {mape:.2f}%', transform=plt.gca().transAxes)
        
        scatter_path = os.path.join(probe_dir, 'prediction_scatter.png')
        plt.savefig(scatter_path)
        plt.close()
        print(f"Saved prediction scatter plot to {scatter_path}")

    # --- Save Results ---
    # Update filenames for nonzeros probe
    model_name = "linear_probe_nonzeros.pt"
    log_name = "probe_train_log_nonzeros.json"

    torch.save(probe_reg.state_dict(), os.path.join(probe_dir, model_name))
    with open(os.path.join(probe_dir, log_name), "w") as f:
        json.dump(reg_log, f, indent=2)
    print(f"Saved LP Nonzeros Regression probe to {os.path.join(probe_dir, model_name)}")
    print(f"Saved training log to {os.path.join(probe_dir, log_name)}")

    print("\n--- Probe Training Complete ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a single regression probe for Concorde LP nonzeros.")
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--epoch", type=int, required=True, help="Epoch of model checkpoint to use")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=5000, help="Number of epochs for probe training (default: 5000)")
    parser.add_argument("--lr", type=float, default=1e-6, help="Learning rate (default: 1e-6)")
    parser.add_argument("--patience", type=int, default=50, help="Patience parameter for early stopping")
    args = parser.parse_args()
    main(args) 