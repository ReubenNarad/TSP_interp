import os
import sys
import argparse
import pickle
import json
import glob
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.kernel_ridge import KernelRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.colors as mcolors

# Fix imports from parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from policy.policy_hooked import HookedAttentionModelPolicy
from policy.reinforce_clipped import REINFORCEClipped
from sae.model.sae_model import TopKSparseAutoencoder
from rl4co.envs import TSPEnv


# --- Model Loading Functions (Keep as standalone or move into Collector) ---
def load_tsp_model(run_path, config, env, device):
    """Loads the TSP policy model."""
    checkpoint_dir = run_path / "checkpoints"
    checkpoint_files = glob.glob(str(checkpoint_dir / "checkpoint_epoch_*.ckpt"))
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint files found in {checkpoint_dir}")
    latest_checkpoint = max(
        checkpoint_files,
        key=lambda f: int(os.path.basename(f).split("checkpoint_epoch_")[1].split(".ckpt")[0])
    )
    policy = HookedAttentionModelPolicy(
        env_name=env.name,
        embed_dim=config['embed_dim'],
        num_encoder_layers=config['n_encoder_layers'],
        num_heads=int(config.get("num_heads", 8)),
        temperature=config['temperature'],
    )
    model = REINFORCEClipped.load_from_checkpoint(
        latest_checkpoint,
        env=env,
        policy=policy,
        strict=False # Allow loading even if some keys mismatch slightly
    )
    model = model.to(device)
    model.eval()
    print(f"Loaded TSP model from checkpoint: {latest_checkpoint}")
    return model

def load_sae_model(sae_path, run_path, device):
    """Loads the SAE model."""
    sae_config_path = sae_path / "sae_config.json"
    if not sae_config_path.exists():
        raise FileNotFoundError(f"SAE config not found at {sae_config_path}")
    with open(sae_config_path, "r") as f:
        sae_config = json.load(f)

    sae_model_path = sae_path / "sae_final.pt"
    # Add logic to find latest checkpoint if final doesn't exist (similar to TSP model)
    if not sae_model_path.exists():
         # Try to find the latest checkpoint
        checkpoint_dir = sae_path / "checkpoints"
        if checkpoint_dir.exists():
            checkpoint_files = glob.glob(str(checkpoint_dir / "sae_epoch_*.pt"))
            if checkpoint_files:
                latest_checkpoint = max(
                    checkpoint_files,
                    key=lambda f: int(os.path.basename(f).split("sae_epoch_")[1].split(".pt")[0])
                )
                sae_model_path = Path(latest_checkpoint)
            else:
                 raise FileNotFoundError(f"SAE model file not found in {sae_path} or checkpoints.")
        else:
            raise FileNotFoundError(f"SAE model file not found at {sae_model_path} and no checkpoints directory.")

    # Determine input dimension from an activation file
    activation_dir = run_path / "sae" / "activations"
    activation_files = glob.glob(str(activation_dir / "activations_epoch_*.pt"))
    if not activation_files:
        raise FileNotFoundError(f"No activation files found in {activation_dir}")
    activations_sample = torch.load(activation_files[0], map_location='cpu')
    activation_key = sae_config.get("activation_key", "encoder_output")
    if activation_key not in activations_sample:
         possible_keys = [k for k in activations_sample if "encoder" in k and "output" in k]
         if not possible_keys:
              raise ValueError(f"Activation key '{activation_key}' not found in sample file {activation_files[0]} and no fallback found.")
         activation_key = possible_keys[0]
         print(f"Warning: Using fallback activation key '{activation_key}'")
    input_dim = activations_sample[activation_key].shape[-1]

    expansion_factor = sae_config.get("expansion_factor", 2.0)
    latent_dim = int(expansion_factor * input_dim)
    k_ratio = sae_config.get("k_ratio", 0.05)
    sae_model = TopKSparseAutoencoder(
        input_dim=input_dim,
        latent_dim=latent_dim,
        k_ratio=k_ratio,
        tied_weights=sae_config.get("tied_weights", False),
        bias_decay=sae_config.get("bias_decay", 0.0),
        dict_init=sae_config.get("init_method", "uniform")
    )
    checkpoint = torch.load(sae_model_path, map_location=device)
    if "model_state_dict" in checkpoint:
        sae_model.load_state_dict(checkpoint["model_state_dict"])
    else:
        sae_model.load_state_dict(checkpoint)
    sae_model = sae_model.to(device)
    sae_model.eval()
    print(f"Loaded SAE model from {sae_model_path}")
    print(f"SAE model: {input_dim} -> {latent_dim}, k_ratio={k_ratio}")
    return sae_model

# --- Data Collector Class ---
class LocationDataCollector:
    def __init__(self, run_name: str, sae_run_name: str, device: torch.device):
        self.run_name = run_name
        self.sae_run_name = sae_run_name
        self.device = device
        self.run_path = Path("./runs") / run_name
        self.sae_path = self.run_path / "sae" / "sae_runs" / sae_run_name

        self._load_dependencies()

    def _load_dependencies(self):
        # Load Env
        env_path = self.run_path / "env.pkl"
        if not env_path.exists(): raise FileNotFoundError(f"Env not found: {env_path}")
        with open(env_path, "rb") as f: self.env = pickle.load(f)
        print(f"Collector: Loaded environment from {env_path}")

        # Load Config
        config_path = self.run_path / "config.json"
        if not config_path.exists(): raise FileNotFoundError(f"Config not found: {config_path}")
        with open(config_path, "r") as f: self.config = json.load(f)
        print(f"Collector: Loaded config from {config_path}")

        # Load Models
        self.tsp_model = load_tsp_model(self.run_path, self.config, self.env, self.device)
        self.sae_model = load_sae_model(self.sae_path, self.run_path, self.device)

    def collect(self, feature_idx: int, num_instances: int):
        """Collects node locations (X) and feature activations (y)."""
        print(f"Collector: Collecting data for feature {feature_idx} from {num_instances} instances...")
        instances = self.env.reset(batch_size=[num_instances]).to(self.device)

        with torch.no_grad():
            batch = instances
            h, _ = self.tsp_model.policy.encoder(batch)
            if len(h.shape) == 3:
                b_size, n_nodes, f_dim = h.shape
                model_acts_flat = h.reshape(-1, f_dim)
            else: model_acts_flat = h

            _, sae_acts_flat = self.sae_model(model_acts_flat)

            if len(h.shape) == 3:
                sae_acts = sae_acts_flat.reshape(b_size, n_nodes, -1)
            else: sae_acts = sae_acts_flat

            target_acts = sae_acts[:, :, feature_idx]
            locs = batch['locs']
            X = locs.reshape(-1, 2).cpu().numpy()
            y = target_acts.reshape(-1).cpu().numpy()

        print(f"Collector: Collected {X.shape[0]} data points.")
        return X, y


# --- Base Regressor Class (Optional - for potential future sharing) ---
# class BaseLocationRegressor:
#     def __init__(self):
#         self.X, self.y = None, None
#         self.X_train, self.X_test = None, None
#         self.y_train, self.y_test = None, None
#         self.grid_points, self.xx, self.yy = None, None, None
#         self.activation_cmap = plt.cm.get_cmap('viridis')
#         self.error_cmap = plt.cm.get_cmap('Reds')

#     def _prepare_grid(self):
#         # ... (grid generation logic) ...
#         pass

#     def _split_data(self):
#          self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
#              self.X, self.y, test_size=0.2, random_state=42
#          )


# --- RBF Ridge Regressor Class ---
class RBFRidgeLocationRegressor:
    def __init__(self, alpha=0.1, gamma=1.0):
        self.alpha = alpha
        self.gamma = gamma
        self.krr_activation = KernelRidge(kernel='rbf', alpha=self.alpha, gamma=self.gamma)
        self.krr_error = KernelRidge(kernel='rbf', alpha=self.alpha, gamma=self.gamma)
        self.score_activation = None
        self.score_error = None
        self.X, self.y = None, None
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.y_error_krr_train, self.y_error_krr_test = None, None
        self.activation_cmap = plt.cm.get_cmap('viridis')
        self.error_cmap = plt.cm.get_cmap('Reds')

    def fit(self, X, y):
        print("\n--- Fitting RBF Kernel Ridge Models ---")
        self.X, self.y = X, y
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        # Fit activation model
        print("Fitting Activation Model...")
        self.krr_activation.fit(self.X_train, self.y_train)
        self.score_activation = self.krr_activation.score(self.X_test, self.y_test)
        print(f"KRR Activation R^2: {self.score_activation:.4f}")
        # Calculate errors
        y_pred_krr_full = self.krr_activation.predict(self.X)
        y_error_krr = np.abs(self.y - y_pred_krr_full)
        self.y_error_krr_train, self.y_error_krr_test = train_test_split(
            y_error_krr, test_size=0.2, random_state=42
        )
        # Fit error model
        print("Fitting Error Model...")
        self.krr_error.fit(self.X_train, self.y_error_krr_train)
        self.score_error = self.krr_error.score(self.X_test, self.y_error_krr_test)
        print(f"KRR Error R^2: {self.score_error:.4f}")

    def _generate_grid(self):
        grid_res = 100
        x_min, x_max = self.X[:, 0].min() - 0.05, self.X[:, 0].max() + 0.05
        y_min, y_max = self.X[:, 1].min() - 0.05, self.X[:, 1].max() + 0.05
        x_lin = np.linspace(x_min, x_max, grid_res)
        y_lin = np.linspace(y_min, y_max, grid_res)
        xx, yy = np.meshgrid(x_lin, y_lin)
        grid_points = np.vstack([xx.ravel(), yy.ravel()]).T
        return xx, yy, grid_points, x_lin, y_lin

    def visualize_activation(self, feature_idx, run_name, sae_run_name):
        if self.X_test is None: raise RuntimeError("Model not fitted yet.")
        print("Generating KRR Activation visualization...")
        xx, yy, grid_points, x_lin, y_lin = self._generate_grid()
        Z_activation = self.krr_activation.predict(grid_points).reshape(xx.shape)

        fig, ax = plt.subplots(figsize=(10, 10))
        contour = ax.contourf(xx, yy, Z_activation, levels=20, cmap=self.activation_cmap, alpha=0.3)
        fig.colorbar(contour, ax=ax, label='Predicted Activation (KRR)')
        ax.scatter(self.X_test[:, 0], self.X_test[:, 1], c=self.y_test, cmap=self.activation_cmap, s=30, alpha=0.6, label='Test Data')
        ax.set_title(f"KRR Activation Regression: Feat {feature_idx} (R2={self.score_activation:.3f})\nRun: {run_name}, SAE: {sae_run_name}")
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_aspect('equal'); ax.legend(); plt.tight_layout()
        filename = f"krr_regression_activation_feature_{feature_idx}.png"
        plt.savefig(filename); print(f"Saved: {filename}")
        return fig, ax

    def visualize_error(self, feature_idx, run_name, sae_run_name):
        if self.X_test is None: raise RuntimeError("Model not fitted yet.")
        print("Generating KRR Error visualization...")
        xx, yy, grid_points, x_lin, y_lin = self._generate_grid()
        Z_error = self.krr_error.predict(grid_points).reshape(xx.shape)

        fig, ax = plt.subplots(figsize=(10, 10))
        contour = ax.contourf(xx, yy, Z_error, levels=20, cmap=self.error_cmap, alpha=0.3)
        fig.colorbar(contour, ax=ax, label='Predicted Activation Error (KRR)')
        norm_err = plt.Normalize(vmin=self.y_error_krr_test.min(), vmax=self.y_error_krr_test.max())
        ax.scatter(self.X_test[:, 0], self.X_test[:, 1], c=self.y_error_krr_test, cmap=self.error_cmap, norm=norm_err,
                   s=30, alpha=0.7, edgecolors='grey', linewidths=0.5, label='Test Data (True Error)')
        ax.set_title(f"KRR Error Regression: Feat {feature_idx} (R2={self.score_error:.3f})\nWhere KRR Prediction Fails")
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_aspect('equal'); ax.legend(); plt.tight_layout()
        filename = f"krr_regression_error_feature_{feature_idx}.png"
        plt.savefig(filename); print(f"Saved: {filename}")
        return fig, ax


# --- KNN Regressor Class ---
class KNNLocationRegressor:
    def __init__(self, k=10):
        self.k = k
        self.knn_activation = KNeighborsRegressor(n_neighbors=self.k, weights='distance')
        self.knn_error = KNeighborsRegressor(n_neighbors=self.k, weights='distance')
        self.score_activation = None
        self.score_error = None
        self.X, self.y = None, None
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.y_error_knn_train, self.y_error_knn_test = None, None
        self.activation_cmap = plt.cm.get_cmap('viridis')
        self.error_cmap = plt.cm.get_cmap('Reds')


    def fit(self, X, y):
        print("\n--- Fitting KNN Models ---")
        self.X, self.y = X, y
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        # Fit activation model
        print(f"Fitting Activation Model (k={self.k})...")
        self.knn_activation.fit(self.X_train, self.y_train)
        self.score_activation = self.knn_activation.score(self.X_test, self.y_test)
        print(f"KNN Activation R^2: {self.score_activation:.4f}")
        # Calculate errors
        y_pred_knn_full = self.knn_activation.predict(self.X)
        y_error_knn = np.abs(self.y - y_pred_knn_full)
        self.y_error_knn_train, self.y_error_knn_test = train_test_split(
            y_error_knn, test_size=0.2, random_state=42
        )
        # Fit error model
        print(f"Fitting Error Model (k={self.k})...")
        self.knn_error.fit(self.X_train, self.y_error_knn_train)
        self.score_error = self.knn_error.score(self.X_test, self.y_error_knn_test)
        print(f"KNN Error R^2: {self.score_error:.4f}")

    def _generate_grid(self):
        # Same grid generation as KRR
        grid_res = 100
        x_min, x_max = self.X[:, 0].min() - 0.05, self.X[:, 0].max() + 0.05
        y_min, y_max = self.X[:, 1].min() - 0.05, self.X[:, 1].max() + 0.05
        x_lin = np.linspace(x_min, x_max, grid_res)
        y_lin = np.linspace(y_min, y_max, grid_res)
        xx, yy = np.meshgrid(x_lin, y_lin)
        grid_points = np.vstack([xx.ravel(), yy.ravel()]).T
        return xx, yy, grid_points, x_lin, y_lin

    def visualize_activation(self, feature_idx, run_name, sae_run_name):
        if self.X_test is None: raise RuntimeError("Model not fitted yet.")
        print("Generating KNN Activation visualization...")
        xx, yy, grid_points, x_lin, y_lin = self._generate_grid()
        Z_activation = self.knn_activation.predict(grid_points).reshape(xx.shape)

        fig, ax = plt.subplots(figsize=(10, 10))
        contour = ax.contourf(xx, yy, Z_activation, levels=20, cmap=self.activation_cmap, alpha=0.3)
        fig.colorbar(contour, ax=ax, label=f'Predicted Activation (KNN k={self.k})')
        ax.scatter(self.X_test[:, 0], self.X_test[:, 1], c=self.y_test, cmap=self.activation_cmap, s=30, alpha=0.6, label='Test Data')
        ax.set_title(f"KNN Activation Regression: Feat {feature_idx} (R2={self.score_activation:.3f}, k={self.k})\nRun: {run_name}, SAE: {sae_run_name}")
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_aspect('equal'); ax.legend(); plt.tight_layout()
        filename = f"knn_regression_activation_feature_{feature_idx}_k{self.k}.png"
        plt.savefig(filename); print(f"Saved: {filename}")
        return fig, ax

    def visualize_error(self, feature_idx, run_name, sae_run_name):
        if self.X_test is None: raise RuntimeError("Model not fitted yet.")
        print("Generating KNN Error visualization...")
        xx, yy, grid_points, x_lin, y_lin = self._generate_grid()
        Z_error = self.knn_error.predict(grid_points).reshape(xx.shape)

        fig, ax = plt.subplots(figsize=(10, 10))
        contour = ax.contourf(xx, yy, Z_error, levels=20, cmap=self.error_cmap, alpha=0.3)
        fig.colorbar(contour, ax=ax, label=f'Predicted Activation Error (KNN k={self.k})')
        norm_err = plt.Normalize(vmin=self.y_error_knn_test.min(), vmax=self.y_error_knn_test.max())
        ax.scatter(self.X_test[:, 0], self.X_test[:, 1], c=self.y_error_knn_test, cmap=self.error_cmap, norm=norm_err,
                   s=30, alpha=0.7, edgecolors='grey', linewidths=0.5, label='Test Data (True Error)')
        ax.set_title(f"KNN Error Regression: Feat {feature_idx} (R2={self.score_error:.3f}, k={self.k})\nWhere KNN Prediction Fails")
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_aspect('equal'); ax.legend(); plt.tight_layout()
        filename = f"knn_regression_error_feature_{feature_idx}_k{self.k}.png"
        plt.savefig(filename); print(f"Saved: {filename}")
        return fig, ax


# --- Main Execution ---
def main():
    parser = argparse.ArgumentParser(description="Test Location Dependence Regressions for SAE features.")
    parser.add_argument("--run_name", type=str, required=True, help="Name of the TSP policy run")
    parser.add_argument("--sae_run_name", type=str, required=True, help="Name of the SAE run")
    parser.add_argument("--feature", type=int, required=True, help="Index of the SAE feature.")
    parser.add_argument("--num_instances", type=int, default=100, help="Number of instances for data.")
    parser.add_argument("--device", type=str, default=None, help="Device ('cpu', 'cuda'). Auto-detects if None.")
    parser.add_argument("--krr_alpha", type=float, default=0.1, help="Regularization strength (alpha) for KRR.")
    parser.add_argument("--krr_gamma", type=float, default=100.0, help="Gamma parameter for KRR RBF kernel.")
    parser.add_argument("--knn_k", type=int, default=10, help="Number of neighbors (k) for KNN.")
    args = parser.parse_args()

    # --- Setup Device ---
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # --- Collect Data ---
    collector = LocationDataCollector(args.run_name, args.sae_run_name, device)
    X, y = collector.collect(args.feature, args.num_instances)

    # --- Run KRR ---
    rbf_regressor = RBFRidgeLocationRegressor(alpha=args.krr_alpha, gamma=args.krr_gamma)
    rbf_regressor.fit(X, y)
    rbf_regressor.visualize_activation(args.feature, args.run_name, args.sae_run_name)
    rbf_regressor.visualize_error(args.feature, args.run_name, args.sae_run_name)

    # --- Run KNN ---
    knn_regressor = KNNLocationRegressor(k=args.knn_k)
    knn_regressor.fit(X, y)
    knn_regressor.visualize_activation(args.feature, args.run_name, args.sae_run_name)
    knn_regressor.visualize_error(args.feature, args.run_name, args.sae_run_name)

    print("\nRegression test complete.")
    plt.show() # Show all plots at the end

if __name__ == "__main__":
    main()
