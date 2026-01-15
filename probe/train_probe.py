import os
import argparse
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import json
import sys
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from policy.policy_hooked import HookedAttentionModelPolicy
from policy.reinforce_clipped import REINFORCEClipped

# Simplified train_probe, only for regression
def train_regression_probe(probe_model, train_loader, val_loader, loss_fn, optimizer, num_epochs, device, 
                           y_mean, y_std, val_freq=50, task_name="Regression", patience=50, l1_lambda=0.0):
    """Helper function to train a regression probe with periodic validation."""
    train_losses = []
    val_losses = []
    best_loss = float('inf')
    patience_counter = 0
    val_metrics = []
    
    print(f"Training for up to {num_epochs} epochs with patience={patience}, validating every {val_freq} epochs...")
    
    for epoch in range(num_epochs):
        # Training phase
        probe_model.train()
        epoch_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = probe_model(xb).squeeze(-1) # Direct prediction
            
            # Calculate base loss
            loss = loss_fn(pred, yb)
            
            # Add L1 regularization if specified
            if l1_lambda > 0:
                l1_penalty = sum(p.abs().sum() for p in probe_model.parameters())
                loss = loss + l1_lambda * l1_penalty

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        mean_loss = np.mean(epoch_losses)
        train_losses.append(mean_loss)
        
        # Validation phase (less frequently)
        if epoch % val_freq == 0 or epoch == num_epochs - 1:
            probe_model.eval()
            val_epoch_losses = []
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    pred = probe_model(xb).squeeze(-1)
                    
                    # Base loss only for validation (no regularization)
                    loss = loss_fn(pred, yb)
                    val_epoch_losses.append(loss.item())
            
            val_mean_loss = np.mean(val_epoch_losses)
            val_losses.append((epoch, val_mean_loss))  # Store epoch with loss value
            
            # Quick MSE calculation on unnormalized values for better interpretability
            pred_unnorm = pred * y_std.to(device) + y_mean.to(device)
            yb_unnorm = yb * y_std.to(device) + y_mean.to(device)
            orig_mse = ((pred_unnorm - yb_unnorm) ** 2).mean().item()
            
            # Store validation metrics for this epoch
            val_metrics.append({
                'epoch': epoch,
                'normalized_loss': val_mean_loss,
                'original_mse': orig_mse
            })
            
            # Include L1 info in log if used
            l1_info = f", L1: {l1_lambda:.2e}" if l1_lambda > 0 else ""
            log_str = f"Epoch {epoch+1}/{num_epochs} [{task_name}] - Train Loss: {mean_loss:.6f}, Val Loss: {val_mean_loss:.6f}, Val MSE (original): {orig_mse:.2f}{l1_info}"
            print(log_str)
        elif (epoch+1) % 25 == 0 or epoch == 0:
            # Just print training loss on other epochs
            l1_info = f", L1: {l1_lambda:.2e}" if l1_lambda > 0 else ""
            print(f"Epoch {epoch+1}/{num_epochs} [{task_name}] - Train Loss: {mean_loss:.6f}{l1_info}")
            
        # Early stopping check (using training loss for consistency with original code)
        if mean_loss < best_loss:
            best_loss = mean_loss
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1} as loss hasn't improved for {patience} epochs.")
            print(f"Best training loss: {best_loss:.6f}")
            break

    results = {
        "train_losses": train_losses, 
        "val_losses": val_losses, 
        "best_loss": best_loss,
        "val_metrics": val_metrics,
        "l1_lambda": l1_lambda  # Store L1 parameter in results
    }
    return results

def evaluate_probe(probe_model, X, y, y_mean, y_std, device, set_name="Test"):
    """Evaluate probe on given data."""
    probe_model.eval()
    # Make sure y_mean and y_std are on the same device as y
    y_mean_device = y_mean.to(device)
    y_std_device = y_std.to(device)
    with torch.no_grad():
        # Get predictions
        X_device = X.to(device)
        y_device = y.to(device)
        pred_normalized = probe_model(X_device).squeeze(-1)
        # Denormalize predictions
        pred_original = pred_normalized * y_std_device + y_mean_device
        y_original = y_device * y_std_device + y_mean_device
        
        # Calculate metrics on original scale
        mse_original = ((pred_original - y_original) ** 2).mean().item()
        mae_original = (pred_original - y_original).abs().mean().item()
        
        # Calculate MAPE (Mean Absolute Percentage Error)
        non_zero_mask = y_original != 0
        mape = (((pred_original[non_zero_mask] - y_original[non_zero_mask]).abs() / y_original[non_zero_mask]) * 100).mean().item()
        
        # Calculate R-squared
        y_mean_val = y_original.mean()
        ss_total = ((y_original - y_mean_val) ** 2).sum().item()
        ss_residual = ((y_original - pred_original) ** 2).sum().item()
        r_squared = 1 - (ss_residual / ss_total)
        
        print(f"{set_name} Set Metrics:")
        print(f"MSE: {mse_original:.2f}")
        print(f"MAE: {mae_original:.2f}")
        print(f"MAPE: {mape:.2f}%")
        print(f"R-squared: {r_squared:.4f}")
        
        # Calculate percentage of predictions within X% of actual value
        error_thresholds = [10, 20, 30, 50]
        within_thresh_results = {}
        for thresh in error_thresholds:
            pct_errors = ((pred_original[non_zero_mask] - y_original[non_zero_mask]).abs() / y_original[non_zero_mask]) * 100
            within_thresh = (pct_errors <= thresh).float().mean().item() * 100
            within_thresh_results[str(thresh)] = within_thresh
            print(f"Predictions within {thresh}% error: {within_thresh:.2f}%")
        
        # Print sample predictions if it's the test set
        if set_name == "Test":
            print("\nSample predictions (original scale):")
            num_samples = min(10, len(X))
            indices = torch.randperm(len(X_device))[:num_samples]
            for i, idx in enumerate(indices):
                actual = y_original[idx].item()
                predicted = pred_original[idx].item()
                error = abs(predicted - actual)
                error_percent = 100 * error / actual if actual != 0 else float('inf')
                print(f"Sample {i+1}: Actual={actual:.1f}, Predicted={predicted:.1f}, Error={error:.1f} ({error_percent:.1f}%)")
    
    results = {
        "mse": mse_original,
        "mae": mae_original,
        "mape": mape,
        "r_squared": r_squared,
        "within_threshold": within_thresh_results
    }
    return results, pred_original.cpu(), y_original.cpu()

def main(args):
    run_path = f"runs/{args.run_name}"
    probe_dir = os.path.join(run_path, "probe")
    os.makedirs(probe_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Data Loading ---
    print("--- Loading Data ---")
    # Load config
    config_path = os.path.join(run_path, "config.json")
    print(f"Loading config from: {config_path}")
    with open(config_path, "r") as f:
        config = json.load(f)
    # Load environment
    env_path = os.path.join(run_path, "env.pkl")
    print(f"Loading environment from: {env_path}")
    with open(env_path, "rb") as f:
        env = pickle.load(f)
    print("Environment loaded.")

    # Load generated instances and metrics from baseline_probe.pkl
    baseline_probe_path = os.path.join(probe_dir, "baseline_probe.pkl")
    print(f"Attempting to load probe data from: {baseline_probe_path}")
    with open(baseline_probe_path, "rb") as f:
        probe_data = pickle.load(f)
    print("Probe data loaded.")

    if "locs" not in probe_data:
        raise KeyError("'locs' key not found in baseline_probe.pkl.")
    # Load LP Nonzeros instead of run_times
    if "lp_nonzeros" not in probe_data or not isinstance(probe_data["lp_nonzeros"], list) or len(probe_data["lp_nonzeros"]) == 0:
         raise KeyError("'lp_nonzeros' key not found or has incorrect format.")

    print("Processing loaded data...")
    locs = probe_data["locs"].to(device)
    # Target variable is now LP Nonzeros
    lp_nonzeros_list = probe_data["lp_nonzeros"][0]
    num_instances = locs.shape[0]
    num_nodes = locs.shape[1]  # Number of nodes in each TSP instance
    print(f"Number of nodes per instance: {num_nodes}")
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
    print("Data processing complete.")

    # --- Plotting Section (Only Nonzeros Histogram) ---
    print("--- Plotting Histogram ---")
    hist_path = os.path.join(probe_dir, 'lp_nonzeros_histogram.png')
    regenerate_plot = True # Or make this an arg
    if regenerate_plot:
        print(f"Generating histogram plot at {hist_path}...")
        plt.figure(figsize=(10, 6))
        plt.hist(nonzeros_np_valid, bins=30, color='salmon', edgecolor='black')
        plt.title(f'Histogram of LP Nonzeros ({len(nonzeros_np_valid)} valid instances)')
        plt.xlabel('Number of Nonzeros')
        plt.ylabel('Frequency')
        plt.grid(True, axis='y', alpha=0.5)
        plt.savefig(hist_path)
        plt.close()
        print(f"Saved LP nonzeros histogram to {hist_path}")
    print("--- Plotting Complete ---")


    # --- Activation Loading ---
    print("--- Loading Model and Getting Activations ---")
    # Load model checkpoint
    checkpoint_dir = os.path.join(run_path, "checkpoints")
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{args.epoch}.ckpt")
    print(f"Checking for checkpoint at: {checkpoint_path}")
    if not os.path.exists(checkpoint_path):
        print("Exact checkpoint not found, searching for latest...")
        import glob
        checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "checkpoint_epoch_*.ckpt"))
        if not checkpoint_files: raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
        checkpoint_path = max(checkpoint_files, key=os.path.getctime)
        print(f"Using latest checkpoint: {checkpoint_path}")

    # Build policy and model
    print("Building policy...")
    policy = HookedAttentionModelPolicy(
        env_name=env.name,
        embed_dim=config["embed_dim"],
        num_encoder_layers=config["n_encoder_layers"],
        num_heads=int(config.get("num_heads", 8)),
        temperature=config["temperature"],
    )
    print("Loading model from checkpoint...")
    model = REINFORCEClipped.load_from_checkpoint(
        checkpoint_path, env=env, policy=policy, strict=False
    )
    model.eval().to(device)
    policy = model.policy.to(device)
    print("Model loaded and put in eval mode.")

    # Get activations for the loaded instances
    instance_data_td = {"locs": locs}
    print(f"Running encoder forward pass for {num_instances} instances...")
    with torch.no_grad():
        encoder_out, _ = policy.encoder(instance_data_td)
        # No pooling - use the full encoder output
        embed_dim = encoder_out.shape[2]
        print(f"Encoder output shape: {encoder_out.shape}")
        
        # Reshape encoder output to [batch_size, num_nodes * embed_dim]
        # This preserves all node-level information
        X_all = encoder_out.reshape(num_instances, -1)
        orig_feat_dim = X_all.shape[1]
        print(f"Original feature dimension: {orig_feat_dim}")
        
        # Apply PCA dimensionality reduction if requested
        if args.pca_components > 0:
            print(f"Applying PCA to reduce dimensions from {orig_feat_dim} to {args.pca_components}...")
            
            # Move data to CPU for scikit-learn
            X_cpu = X_all.cpu().numpy()
            
            # Fit PCA and transform the data
            pca = PCA(n_components=args.pca_components)
            X_reduced = pca.fit_transform(X_cpu)
            
            # Calculate explained variance
            explained_var = np.sum(pca.explained_variance_ratio_) * 100
            print(f"PCA retains {explained_var:.2f}% of the variance with {args.pca_components} components")
            
            # Convert back to tensor and move to the right device
            X_all = torch.tensor(X_reduced, dtype=torch.float32).to(device)
            
            # Create a plot of explained variance
            plt.figure(figsize=(10, 5))
            plt.plot(np.cumsum(pca.explained_variance_ratio_) * 100)
            plt.xlabel('Number of Components')
            plt.ylabel('Cumulative Explained Variance (%)')
            plt.title('PCA Explained Variance')
            plt.grid(True, alpha=0.3)
            plt.axhline(y=95, color='r', linestyle='--', label='95% Threshold')
            plt.axvline(x=args.pca_components, color='g', linestyle='--', 
                        label=f'Selected Components ({args.pca_components})')
            plt.legend()
            pca_plot_path = os.path.join(probe_dir, 'pca_explained_variance.png')
            plt.savefig(pca_plot_path)
            plt.close()
            print(f"Saved PCA explained variance plot to {pca_plot_path}")
            
        feat_dim = X_all.shape[1]
        print(f"Final feature dimension: {feat_dim}")
    print(f"Encoder forward pass complete. Features shape: {X_all.shape}")

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
    
    # --- Create train/test split ---
    print("\n--- Creating train/test split (80/20) ---")
    # Create full dataset
    full_dataset = TensorDataset(X_all.cpu(), y_normalized.cpu())
    
    # Calculate sizes for train/test split (80/20)
    num_train = int(0.8 * len(full_dataset))
    num_test = len(full_dataset) - num_train
    
    # Create the splits
    train_dataset, test_dataset = random_split(
        full_dataset, 
        [num_train, num_test],
        generator=torch.Generator().manual_seed(42)  # Set seed for reproducibility
    )
    
    print(f"Split dataset into {num_train} training samples and {num_test} test samples")
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    # Smaller batch size for test to get more detailed metrics
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # --- Train Single Regression Probe ---
    print(f"\n--- Training Regression Probe (Predicting LP Nonzeros) ---")
    print(f"Creating linear probe with input dim {feat_dim}")
    probe_reg = nn.Linear(feat_dim, 1).to(device)
    print(f"Number of parameters in probe: {sum(p.numel() for p in probe_reg.parameters())}")
    
    # Add L2 regularization to avoid overfitting with high-dimensional input
    optimizer_reg = optim.Adam(probe_reg.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn_reg = nn.MSELoss() # Use MSE for regression

    # Report regularization settings
    reg_info = f"Using regularization: L2={args.weight_decay:.2e}"
    if args.l1_lambda > 0:
        reg_info += f", L1={args.l1_lambda:.2e}"
    print(reg_info)

    # Use the updated training function with train and test loaders for validation
    reg_log = train_regression_probe(
        probe_reg, 
        train_loader, 
        test_loader,  # Use test set as validation set
        loss_fn_reg, 
        optimizer_reg, 
        args.num_epochs, 
        device,
        y_mean,
        y_std,
        val_freq=args.val_freq,  # Add validation frequency parameter
        task_name="Regression (LP Nonzeros)",
        patience=args.patience,
        l1_lambda=args.l1_lambda  # Pass L1 regularization parameter
    )
    
    # Plot training and validation losses
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(reg_log["train_losses"])), reg_log["train_losses"], label='Training Loss')
    val_epochs, val_losses = zip(*reg_log["val_losses"])
    plt.plot(val_epochs, val_losses, 'o-', label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    loss_plot_path = os.path.join(probe_dir, 'training_validation_loss.png')
    plt.savefig(loss_plot_path)
    plt.close()
    print(f"Saved training/validation loss plot to {loss_plot_path}")

    # --- Evaluate probe on both train and test data ---
    print("\n--- Evaluating probe on training data ---")
    # Extract X and y from train dataset
    X_train = torch.stack([x for x, _ in [train_dataset[i] for i in range(len(train_dataset))]])
    y_train = torch.stack([y for _, y in [train_dataset[i] for i in range(len(train_dataset))]])
    
    # Evaluate on training data
    train_results, train_pred, train_actual = evaluate_probe(
        probe_reg, X_train, y_train, y_mean, y_std, device, set_name="Training"
    )
    
    print("\n--- Evaluating probe on test data ---")
    # Extract X and y from test dataset
    X_test = torch.stack([x for x, _ in [test_dataset[i] for i in range(len(test_dataset))]])
    y_test = torch.stack([y for _, y in [test_dataset[i] for i in range(len(test_dataset))]])
    
    # Evaluate on test data
    test_results, test_pred, test_actual = evaluate_probe(
        probe_reg, X_test, y_test, y_mean, y_std, device, set_name="Test"
    )
    
    # Create and save scatter plots for both train and test
    plt.figure(figsize=(18, 8))
    
    # Training set plot
    plt.subplot(1, 2, 1)
    plt.scatter(train_actual.cpu().numpy(), train_pred.cpu().numpy(), alpha=0.5, color='blue')
    min_val = min(train_actual.min().item(), train_pred.min().item())
    max_val = max(train_actual.max().item(), train_pred.max().item())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    plt.xlabel('Actual LP Nonzeros')
    plt.ylabel('Predicted LP Nonzeros')
    plt.title('Training Set: Predicted vs Actual')
    plt.grid(True, alpha=0.3)
    plt.text(0.05, 0.95, f'R² = {train_results["r_squared"]:.4f}', transform=plt.gca().transAxes)
    plt.text(0.05, 0.90, f'MAE = {train_results["mae"]:.2f}', transform=plt.gca().transAxes)
    plt.text(0.05, 0.85, f'MAPE = {train_results["mape"]:.2f}%', transform=plt.gca().transAxes)
    
    # Test set plot
    plt.subplot(1, 2, 2)
    plt.scatter(test_actual.cpu().numpy(), test_pred.cpu().numpy(), alpha=0.5, color='green')
    min_val = min(test_actual.min().item(), test_pred.min().item())
    max_val = max(test_actual.max().item(), test_pred.max().item())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    plt.xlabel('Actual LP Nonzeros')
    plt.ylabel('Predicted LP Nonzeros')
    plt.title('Test Set: Predicted vs Actual')
    plt.grid(True, alpha=0.3)
    plt.text(0.05, 0.95, f'R² = {test_results["r_squared"]:.4f}', transform=plt.gca().transAxes)
    plt.text(0.05, 0.90, f'MAE = {test_results["mae"]:.2f}', transform=plt.gca().transAxes)
    plt.text(0.05, 0.85, f'MAPE = {test_results["mape"]:.2f}%', transform=plt.gca().transAxes)
    
    plt.tight_layout()
    scatter_path = os.path.join(probe_dir, 'prediction_scatter_train_test.png')
    plt.savefig(scatter_path)
    plt.close()
    print(f"Saved train/test prediction scatter plots to {scatter_path}")

    # --- Save Results ---
    print("--- Saving Results ---")
    # Update filenames for nonzeros probe
    model_name = "linear_probe_nonzeros.pt"
    log_name = "probe_train_log_nonzeros.json"

    model_save_path = os.path.join(probe_dir, model_name)
    print(f"Saving probe model state dict to {model_save_path}")
    torch.save(probe_reg.state_dict(), model_save_path)
    
    # Update training log to include both train and test results
    reg_log["train_evaluation"] = train_results
    reg_log["test_evaluation"] = test_results
    
    log_save_path = os.path.join(probe_dir, log_name)
    print(f"Saving training log to {log_save_path}")
    with open(log_save_path, "w") as f:
        json.dump(reg_log, f, indent=2)
    print(f"Saved LP Nonzeros Regression probe to {model_save_path}")
    print(f"Saved training log to {log_save_path}")

    print("\n--- Probe Training Complete ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a single regression probe for Concorde LP nonzeros.")
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--epoch", type=int, required=True, help="Epoch of model checkpoint to use")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=5000, help="Number of epochs for probe training (default: 5000)")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate (default: 1e-5)")
    parser.add_argument("--patience", type=int, default=50, help="Patience parameter for early stopping")
    parser.add_argument("--weight_decay", type=float, default=1e-3, help="L2 regularization weight (default: 1e-3)")
    parser.add_argument("--l1_lambda", type=float, default=0.0, help="L1 regularization weight (default: 0.0)")
    parser.add_argument("--val_freq", type=int, default=50, help="Frequency for validation (default: every 50 epochs)")
    parser.add_argument("--pca_components", type=int, default=50, help="Number of PCA components (0 to disable)")
    args = parser.parse_args()
    main(args) 
