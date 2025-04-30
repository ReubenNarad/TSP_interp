# experiments/cluster_feature/analyze_cluster_features.py
import os
import sys
import pickle
import json
import argparse
from pathlib import Path
import math

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support
from scipy.stats import pearsonr
from tqdm import tqdm
import glob
import base64
from io import BytesIO

# Add project root directory (parent of the parent of the script's directory) to sys.path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

# Manually added imports - might need adjustment based on actual model structure
from policy.policy_hooked import HookedAttentionModelPolicy
from policy.reinforce_clipped import REINFORCEClipped
from sae.model.sae_model import TopKSparseAutoencoder
from rl4co.envs import TSPEnv
from rl4co.envs.routing import TSPGenerator

# --- Configuration ---
RUN_NAME = "clip_tiny_lr_huge_embed_512_tight_clusters"
SAE_RUN_NAME = "sae_l10.001_ef4.0_k0.05_04-17_11:05:11"
BASE_RUN_PATH = Path(f"runs/{RUN_NAME}")
SAE_PATH = BASE_RUN_PATH / "sae" / "sae_runs" / SAE_RUN_NAME
OUTPUT_DIR = Path("experiments/cluster_feature/visualizations")
NUM_LOC = 100 # Default number of locations, adjust if needed from config
NUM_INSTANCES = 10  # Number of instances to analyze

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Device Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Modified Cluster Generator ---
class ClustersWithIDs:
    """
    Samples from a random number of gaussians, each with random mean and cov.
    Also returns cluster IDs for each node.
    Creates new clusters for each sample() call with tight, compact clusters.
    Allows points to extend beyond the [0,1] range.
    (Adapted from distributions.py)
    """
    def __init__(self, min_clusters=5, max_clusters=10, num_loc=100):
        super().__init__()
        self.min_clusters = min_clusters
        self.max_clusters = max_clusters
        self.num_loc = num_loc # Store num_loc

    def sample(self, size):
        # Ensure size is correctly interpreted (might be list or TensorDict)
        if isinstance(size, (list, tuple)):
             batch_size = size[0]
             # num_loc should be taken from self.num_loc if size only provides batch
             if len(size) > 1:
                 num_loc = size[1]
             else:
                 num_loc = self.num_loc
        elif isinstance(size, int):
             batch_size = size
             num_loc = self.num_loc
        else:
             # Assuming TensorDict or similar structure, try accessing batch_size
             try:
                 batch_size = size.batch_size[0] if len(size.batch_size) > 0 else 1
                 num_loc = self.num_loc # Rely on init parameter
             except AttributeError:
                 raise TypeError(f"Unsupported size type: {type(size)}")


        # Use self.num_loc for consistency
        num_loc = self.num_loc

        # Create empty tensor for coordinates and cluster IDs
        coords = torch.zeros(batch_size, num_loc, 2, device=device)
        cluster_ids = torch.zeros(batch_size, num_loc, dtype=torch.long, device=device)

        # For each batch, create new clusters
        for i in range(batch_size):
            # Generate new clusters for this batch
            n_clusters = torch.randint(self.min_clusters, self.max_clusters + 1, (1,)).item()
            clusters = []

            for j in range(n_clusters):
                # Define the cluster's mean (center)
                mean = 0.1 + 0.8 * torch.rand(2, device=device)

                # Create a proper random covariance matrix with random orientation
                std_x = (0.004 + 0.002 * torch.rand(1, device=device)).sqrt()
                std_y = (0.004 + 0.002 * torch.rand(1, device=device)).sqrt()
                theta = 2 * math.pi * torch.rand(1, device=device)
                c, s = torch.cos(theta), torch.sin(theta)
                rotation = torch.tensor([[c, -s], [s, c]], device=device).squeeze()
                scales = torch.diag(torch.tensor([std_x**2, std_y**2], device=device).squeeze())
                cov = rotation @ scales @ rotation.T
                clusters.append({'mean': mean, 'cov': cov})

            # Calculate how many points to sample from each cluster
            points_per_cluster = torch.full((n_clusters,), num_loc // n_clusters, device=device)
            remainder = num_loc % n_clusters
            if remainder > 0:
                points_per_cluster[:remainder] += 1

            point_idx = 0
            # Sample from each cluster
            for c_idx, cluster_info in enumerate(clusters):
                mean, cov = cluster_info['mean'], cluster_info['cov']
                n_points = points_per_cluster[c_idx].item()

                try:
                    L = torch.linalg.cholesky(cov)
                    z = torch.randn(n_points, 2, device=device)
                    samples = torch.matmul(z, L.T) + mean
                except Exception as e: # Catch LinAlgError specifically?
                    print(f"Warning: Cholesky decomposition failed for cluster {c_idx} in batch {i}. Using diagonal cov. Error: {e}")
                    std = torch.sqrt(torch.diag(cov))
                    samples = torch.randn(n_points, 2, device=device) * std + mean

                coords[i, point_idx:point_idx + n_points] = samples
                cluster_ids[i, point_idx:point_idx + n_points] = c_idx # Assign cluster ID
                point_idx += n_points

            # Shuffle the points and their corresponding cluster IDs together
            idx = torch.randperm(num_loc, device=device)
            coords[i] = coords[i][idx]
            cluster_ids[i] = cluster_ids[i][idx]

        # Return as a dictionary similar to Tensordict structure
        # Note: rl4co environments expect a TensorDict, but for direct use here,
        # a dictionary is fine. If integrating with rl4co trainer later, wrap this.
        return {'locs': coords, 'cluster_ids': cluster_ids}


# --- Loading Functions ---
def load_config(config_path):
    print(f"Loading config from {config_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}")
    with open(config_path, "r") as f:
        config = json.load(f)
    return config

def load_tsp_model(run_path, config, env_name):
    print("Loading TSP model...")
    checkpoint_dir = run_path / "checkpoints"
    checkpoint_files = glob.glob(str(checkpoint_dir / "checkpoint_epoch_*.ckpt"))
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint files found in {checkpoint_dir}")

    latest_checkpoint = max(
        checkpoint_files,
        key=lambda f: int(os.path.basename(f).split("checkpoint_epoch_")[1].split(".ckpt")[0])
    )
    print(f"Using TSP checkpoint: {latest_checkpoint}")

    # Recreate policy based on config
    policy = HookedAttentionModelPolicy(
        env_name=env_name, # Assuming env_name is simple like 'tsp'
        embed_dim=config['embed_dim'],
        num_encoder_layers=config['n_encoder_layers'],
        num_heads=8, # Assuming 8 heads, adjust if in config
        temperature=config.get('temperature', 1.0), # Use get for flexibility
        # Add other args like dropout if they were used and are in config
        dropout=config.get('dropout', 0.0),
        attention_dropout=config.get('attention_dropout', 0.0),
    )

    # Load the Lightning Module (REINFORCEClipped in this case)
    # We need a dummy env instance to load, but won't use its generator
    dummy_env = TSPEnv(generator=TSPGenerator(num_loc=config['num_loc'])) # Use config num_loc
    model = REINFORCEClipped.load_from_checkpoint(
        latest_checkpoint,
        env=dummy_env, # Pass the dummy env
        policy=policy,
        strict=False # Allow missing keys if architecture changed slightly
    )
    model = model.to(device)
    model.eval()
    print("TSP model loaded.")
    return model

def load_sae_model(sae_path, tsp_config):
    print("Loading SAE model...")
    sae_config_path = sae_path / "sae_config.json"
    sae_config = load_config(sae_config_path)

    # Find the model file
    sae_model_path = sae_path / "sae_final.pt"
    if not sae_model_path.exists():
        checkpoint_dir = sae_path / "checkpoints"
        if checkpoint_dir.exists():
            checkpoint_files = glob.glob(str(checkpoint_dir / "sae_epoch_*.pt"))
            if checkpoint_files:
                latest_checkpoint = max(
                    checkpoint_files,
                    key=lambda f: int(os.path.basename(f).split("sae_epoch_")[1].split(".pt")[0])
                )
                sae_model_path = Path(latest_checkpoint)
    if not sae_model_path.exists():
        raise FileNotFoundError(f"SAE model (.pt) not found in {sae_path} or its checkpoints.")
    print(f"Using SAE model checkpoint: {sae_model_path}")

    # Determine input dimension from TSP config
    input_dim = tsp_config['embed_dim'] # SAE input is TSP model's embed_dim
    activation_key = sae_config.get("activation_key", "encoder_output") # Check which activation it used

    expansion_factor = sae_config.get("expansion_factor", 2.0)
    latent_dim = int(expansion_factor * input_dim)
    k_ratio = sae_config.get("k_ratio", 0.05)

    sae_model = TopKSparseAutoencoder(
        input_dim=input_dim,
        latent_dim=latent_dim,
        k_ratio=k_ratio,
        # Pass other potential args from sae_config if needed
    )

    checkpoint = torch.load(sae_model_path, map_location=device)
    state_dict_key = "model_state_dict" if "model_state_dict" in checkpoint else None

    if state_dict_key:
        sae_model.load_state_dict(checkpoint[state_dict_key])
    else:
         # Try loading directly if it's just the state dict
        try:
            sae_model.load_state_dict(checkpoint)
        except RuntimeError as e:
            print(f"Error loading SAE state dict: {e}")
            print("Attempting to load with strict=False")
            sae_model.load_state_dict(checkpoint, strict=False)


    sae_model = sae_model.to(device)
    sae_model.eval()
    print(f"SAE model loaded. Input dim: {input_dim}, Latent dim: {latent_dim}, Activation key: {activation_key}")
    return sae_model, activation_key

# --- Main Analysis Logic ---
def analyze_single_instance(tsp_model, sae_model, activation_key, instance_data, instance_idx=0, output_dir=None):
    """
    Analyze a single TSP instance for feature-cluster correlations.
    
    Args:
        tsp_model: The TSP policy model
        sae_model: The SAE model
        activation_key: Key for accessing the right activations
        instance_data: Dictionary containing 'locs' and 'cluster_ids'
        instance_idx: Index of the instance to analyze
        output_dir: Directory to save visualizations (default: None)
    
    Returns:
        List of feature-cluster correlation results
    """
    print(f"\n--- Analyzing Instance {instance_idx} ---")
    
    # Use provided output_dir or fallback to default
    viz_dir = output_dir if output_dir is not None else OUTPUT_DIR
    
    tsp_policy = tsp_model.policy # Get the underlying policy model

    # Prepare instance data for the model (needs to be in a dict-like structure)
    # Convert numpy arrays from generator to tensors if needed
    locs_tensor = torch.tensor(instance_data['locs'][instance_idx], dtype=torch.float32, device=device).unsqueeze(0) # Add batch dim
    cluster_ids_np = instance_data['cluster_ids'][instance_idx].cpu().numpy() # Keep IDs as numpy for analysis

    # Wrap in a simple dictionary for the policy forward pass
    model_input = {'locs': locs_tensor}

    # --- Get Activations ---
    tsp_policy.activation_cache = {} # Clear previous cache if any
    tsp_policy.hooks = {} # Clear hooks if reusing policy instance
    tsp_policy._setup_hooks() # Re-attach hooks

    # Manually add hooks if _setup_hooks doesn't capture encoder_output correctly
    # Example:
    # def get_encoder_output(module, input, output):
    #     # Output might be a tuple (hidden_states, init_embeds)
    #     if isinstance(output, tuple):
    #         tsp_policy.activation_cache['encoder_output'] = output[0] # Assuming first element is embeddings
    #     else:
    #         tsp_policy.activation_cache['encoder_output'] = output
    # handle = tsp_policy.encoder.register_forward_hook(get_encoder_output)
    # tsp_policy.hooks['encoder_output_manual'] = handle


    with torch.no_grad():
        # Run policy encoder forward pass
        # The encoder usually takes the input dict directly
        encoder_output = tsp_policy.encoder(model_input)
        # The encoder output might be a tuple, e.g., (node_embeddings, graph_embedding)
        # We need the node embeddings. Let's assume it's the first element if tuple.
        if isinstance(encoder_output, tuple):
            model_activations = encoder_output[0]
        else:
            model_activations = encoder_output

        # If manual hook was used, retrieve from cache:
        # model_activations = tsp_policy.activation_cache.get(activation_key)
        # if model_activations is None:
        #     raise ValueError(f"Could not retrieve '{activation_key}' from policy cache after forward pass.")

        # Ensure model_activations is [batch_size, num_nodes, features] = [1, num_loc, embed_dim]
        if model_activations.shape[0] != 1 or model_activations.shape[1] != NUM_LOC:
             raise ValueError(f"Unexpected model activation shape: {model_activations.shape}")

        # Reshape for SAE: [1 * num_nodes, features]
        model_acts_flat = model_activations.reshape(-1, model_activations.shape[-1])

        # Get SAE activations
        _, sae_acts_flat = sae_model(model_acts_flat)

        # Reshape back: [1, num_nodes, latent_dim]
        sae_acts = sae_acts_flat.reshape(1, NUM_LOC, -1)

    # Remove hooks after use
    for handle in tsp_policy.hooks.values():
        handle.remove()
    tsp_policy.hooks = {}

    # --- Correlation Analysis ---
    sae_acts_instance = sae_acts[0].cpu().numpy() # Shape: [num_nodes, latent_dim]
    num_features = sae_acts_instance.shape[1]
    unique_clusters = np.unique(cluster_ids_np)
    print(f"Found {len(unique_clusters)} clusters in this instance.")
    print(f"SAE has {num_features} features.")

    results = []
    for feature_idx in tqdm(range(num_features), desc="Analyzing Features"):
        feature_activations = sae_acts_instance[:, feature_idx]
        
        for cluster_id in unique_clusters:
            # Create binary target vector for this cluster
            is_in_cluster = (cluster_ids_np == cluster_id).astype(int)
            
            # Check if either array is constant before calculating correlation
            if np.all(feature_activations == feature_activations[0]) or np.all(is_in_cluster == is_in_cluster[0]):
                # If either array is constant, set correlation and p-value to 0
                corr, p_value = 0.0, 1.0
            else:
                # Calculate Pearson correlation
                corr, p_value = pearsonr(feature_activations, is_in_cluster)
            
            # Calculate Precision/Recall/F1 using a threshold (e.g., > 0)
            # This helps interpret if the feature "selects" the cluster nodes
            feature_binary_select = (feature_activations > 0).astype(int)
            precision, recall, f1, _ = precision_recall_fscore_support(
                is_in_cluster, feature_binary_select, average='binary', zero_division=0
            )

            results.append({
                'feature_idx': feature_idx,
                'cluster_id': cluster_id,
                'correlation': corr if not np.isnan(corr) else 0.0, # Handle NaN correlation (e.g., constant vector)
                'p_value': p_value if not np.isnan(p_value) else 1.0,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            })

    # --- Find Best Matches ---
    # Sort by absolute correlation first, then maybe F1 score
    results.sort(key=lambda x: (abs(x['correlation']), x['f1_score']), reverse=True)

    print("\nTop 10 Feature-Cluster Matches (Sorted by Abs Correlation, then F1):")
    for i in range(min(10, len(results))):
        res = results[i]
        print(f"  Rank {i+1}: Feature {res['feature_idx']} <-> Cluster {res['cluster_id']} | "
              f"Corr: {res['correlation']:.4f} (p={res['p_value']:.3f}) | "
              f"F1: {res['f1_score']:.4f} (P: {res['precision']:.3f}, R: {res['recall']:.3f})")

    # --- Visualize Top Match for Each Cluster ---
    if results:
        # Group results by cluster_id
        cluster_to_features = {}
        for res in results:
            cluster_id = res['cluster_id']
            if cluster_id not in cluster_to_features:
                cluster_to_features[cluster_id] = []
            cluster_to_features[cluster_id].append(res)
        
        # For each cluster, find the best matching feature
        for cluster_id, cluster_results in cluster_to_features.items():
            # Sort features for this cluster by absolute correlation
            cluster_results.sort(key=lambda x: (abs(x['correlation']), x['f1_score']), reverse=True)
            best_feature = cluster_results[0]
            feature_idx = best_feature['feature_idx']
            
            print(f"\nVisualizing best feature for Cluster {cluster_id}: Feature {feature_idx}")
            print(f"  Corr: {best_feature['correlation']:.4f}, F1: {best_feature['f1_score']:.4f}")
            print(f"  Precision: {best_feature['precision']:.4f}, Recall: {best_feature['recall']:.4f}")
            
            node_coords = instance_data['locs'][instance_idx].cpu().numpy()
            feature_acts = sae_acts_instance[:, feature_idx]
            
            # Create figure with two subplots with uniform sizes
            fig = plt.figure(figsize=(16, 8))
            
            # Use gridspec to have more control over subplot sizes
            from matplotlib import gridspec
            gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 0.05])
            
            # Create the subplots
            ax1 = plt.subplot(gs[0])  # Left plot (clusters)
            ax2 = plt.subplot(gs[1])  # Right plot (feature activation)
            cax = plt.subplot(gs[2])  # Space for colorbar
            
            # Create a colormap for the clusters - use a more vibrant one
            num_clusters = len(unique_clusters)
            # Replace 'tab20' with a higher contrast colormap
            cmap = plt.get_cmap('Accent', num_clusters)  # Much more contrast than tab20
            # If more than 8 clusters (Accent limit), fall back to a combined approach
            if num_clusters > 8:
                # Create a list of distinct colors by combining colormaps
                base_colors = plt.get_cmap('Accent', 8)(range(8))
                extra_colors = plt.get_cmap('Accent', num_clusters-8)(range(num_clusters-8))
                all_colors = np.vstack([base_colors, extra_colors])
                cmap = plt.matplotlib.colors.ListedColormap(all_colors)
            
            # First plot all clusters with modest alpha
            for idx, clust_id in enumerate(unique_clusters):
                mask = (cluster_ids_np == clust_id)
                # Lower alpha for non-focal clusters
                alpha_val = 0.4 if clust_id != cluster_id else 0.9
                # Larger size for focal cluster
                size_val = 50 if clust_id != cluster_id else 80
                # Different edge color and width for focal cluster
                edge_color = 'none' if clust_id != cluster_id else 'black'
                edge_width = 0 if clust_id != cluster_id else 1
                
                ax1.scatter(
                    node_coords[mask, 0],
                    node_coords[mask, 1],
                    c=[cmap(idx)],  # Use consistent color from colormap
                    s=size_val,
                    alpha=alpha_val,
                    edgecolors=edge_color,
                    linewidths=edge_width,
                    label=f'Cluster {clust_id}' + (' (Target)' if clust_id == cluster_id else '')
                )
            
            ax1.set_title(f'Ground Truth Clusters (Highlighting Cluster {cluster_id})')
            ax1.set_xlabel('X Coordinate')
            ax1.set_ylabel('Y Coordinate')
            ax1.set_aspect('equal')
            ax1.grid(True)
            ax1.legend(loc='upper right')
            
            # Plot 2: Feature Activation
            # Normalize the feature activations to 0-1 range for visualization
            feature_min = feature_acts.min()
            feature_max = feature_acts.max()
            if feature_max > feature_min:  # Avoid division by zero
                feature_acts_normalized = (feature_acts - feature_min) / (feature_max - feature_min)
            else:
                feature_acts_normalized = feature_acts * 0  # All zeros if all values are identical
            
            norm = plt.Normalize(vmin=0, vmax=1)  # Fixed 0-1 normalization
            cmap = plt.get_cmap('viridis')
            scatter = ax2.scatter(
                node_coords[:, 0], 
                node_coords[:, 1], 
                c=feature_acts_normalized, 
                cmap=cmap, 
                norm=norm, 
                s=50, 
                alpha=0.8
            )
            
            # Add colorbar in the dedicated axis - vertical but same height as plots
            cbar = plt.colorbar(scatter, cax=cax)
            cbar.set_label(f'Feature {feature_idx} Activation (Normalized)')
            
            # Update the title to include the actual min/max values
            ax2.set_title(f'Feature {feature_idx} Activation\nCorr: {best_feature["correlation"]:.3f}, Range: [{feature_min:.2f}, {feature_max:.2f}]')
            
            ax2.set_xlabel('X Coordinate')
            ax2.set_ylabel('Y Coordinate')
            ax2.set_aspect('equal')
            ax2.grid(True)
            
            # Set the limits to be identical for both plots
            x_min, x_max = 0, 1
            y_min, y_max = 0, 1
            ax1.set_xlim([x_min, x_max])
            ax1.set_ylim([y_min, y_max])
            ax2.set_xlim([x_min, x_max])
            ax2.set_ylim([y_min, y_max])
            
            plt.tight_layout()
            
            # Save the figure
            save_path = viz_dir / f"cluster_{cluster_id}_best_feature_{feature_idx}.png"
            plt.savefig(save_path, dpi=150)
            print(f"Saved visualization to {save_path}")
            plt.close(fig)
            
        # Also visualize the overall best match with consistent sizes
        top_match = results[0]
        feature_to_viz = top_match['feature_idx']
        cluster_to_viz = top_match['cluster_id']
        print(f"\nVisualizing overall best match: Feature {feature_to_viz} and Cluster {cluster_to_viz}")
        
        node_coords = instance_data['locs'][instance_idx].cpu().numpy()
        feature_acts_viz = sae_acts_instance[:, feature_to_viz]
        
        # Create figure with fixed layout
        fig = plt.figure(figsize=(16, 8))
        from matplotlib import gridspec
        gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 0.05])
        
        # Use a more vibrant colormap for clusters
        cluster_cmap = plt.get_cmap('Accent', min(8, len(unique_clusters)))
        # If more than 8 clusters, combine colormaps
        if len(unique_clusters) > 8:
            base_colors = plt.get_cmap('Accent', 8)(range(8))
            extra_colors = plt.get_cmap('Accent', len(unique_clusters)-8)(range(len(unique_clusters)-8))
            all_colors = np.vstack([base_colors, extra_colors])
            cluster_cmap = plt.matplotlib.colors.ListedColormap(all_colors)
        
        # Create the subplots
        ax1 = plt.subplot(gs[0])
        ax2 = plt.subplot(gs[1])
        cax = plt.subplot(gs[2])
        
        # Plot 1: Ground Truth Clusters (all clusters with different colors)
        scatter1 = ax1.scatter(node_coords[:, 0], node_coords[:, 1], c=cluster_ids_np, cmap=cluster_cmap, s=50, alpha=0.8)
        ax1.set_title(f'Instance {instance_idx}: Ground Truth Clusters ({len(unique_clusters)} clusters)')
        ax1.set_xlabel('X Coordinate')
        ax1.set_ylabel('Y Coordinate')
        ax1.set_aspect('equal')
        ax1.grid(True)
        
        # Create legend for clusters
        handles, labels = scatter1.legend_elements(prop="colors", alpha=0.6)
        legend_labels = [f'Cluster {i}' for i in range(len(unique_clusters))]
        ax1.legend(handles, legend_labels, title="Clusters", loc='upper right')
        
        # Plot 2: Feature Activation
        # Normalize the feature activations to 0-1 range for visualization
        feature_min = feature_acts_viz.min()
        feature_max = feature_acts_viz.max()
        if feature_max > feature_min:  # Avoid division by zero
            feature_acts_viz_normalized = (feature_acts_viz - feature_min) / (feature_max - feature_min)
        else:
            feature_acts_viz_normalized = feature_acts_viz * 0  # All zeros if all values are identical
        
        norm = plt.Normalize(vmin=0, vmax=1)  # Fixed 0-1 normalization
        cmap = plt.get_cmap('viridis')
        scatter2 = ax2.scatter(node_coords[:, 0], node_coords[:, 1], c=feature_acts_viz_normalized, cmap=cmap, norm=norm, s=50, alpha=0.8)
        
        # Update the title to include the actual min/max values
        ax2.set_title(f'Feature {feature_to_viz} Activation (Normalized)\n' +
                     f'Corr: {top_match["correlation"]:.3f}, F1: {top_match["f1_score"]:.3f}, Range: [{feature_min:.2f}, {feature_max:.2f}]')
        
        ax2.set_xlabel('X Coordinate')
        ax2.set_ylabel('Y Coordinate')
        ax2.set_aspect('equal')
        ax2.grid(True)
        
        # Add colorbar to dedicated axis
        cbar = plt.colorbar(scatter2, cax=cax)
        cbar.set_label(f'Feature {feature_to_viz} Activation (Normalized)')
        
        # Set the limits to be identical for both plots
        x_min, x_max = 0, 1
        y_min, y_max = 0, 1
        ax1.set_xlim([x_min, x_max])
        ax1.set_ylim([y_min, y_max])
        ax2.set_xlim([x_min, x_max])
        ax2.set_ylim([y_min, y_max])
        
        plt.tight_layout()
        
        # Save the figure
        save_path = viz_dir / f"overall_best_match_feature_{feature_to_viz}_cluster_{cluster_to_viz}.png"
        plt.savefig(save_path, dpi=150)
        print(f"Saved overall best match visualization to {save_path}")
        plt.close(fig)
        
    else:
        print("No results to visualize.")

    return results

# Add this function before the main script execution
def create_interactive_html(all_results, output_dir):
    """Create an interactive HTML file to browse visualizations."""
    
    print("Generating interactive HTML viewer...")
    
    html_path = output_dir / "interactive_viewer.html"
    
    # Collect instance and cluster information
    instance_data = {}
    
    # Collect the structure of available visualizations
    for instance_idx, instance_results in enumerate(all_results):
        instance_dir = output_dir / f"instance_{instance_idx}"
        
        # Group results by cluster_id
        cluster_to_features = {}
        for res in instance_results:
            cluster_id = res['cluster_id']
            if cluster_id not in cluster_to_features:
                cluster_to_features[cluster_id] = []
            cluster_to_features[cluster_id].append(res)
                
        # Store best feature for each cluster
        best_features = {}
        for cluster_id, cluster_results in cluster_to_features.items():
            # Sort features for this cluster by absolute correlation
            cluster_results.sort(key=lambda x: (abs(x['correlation']), x['f1_score']), reverse=True)
            best_feature = cluster_results[0]
            
            # Convert NumPy types to Python native types
            cluster_id_py = int(cluster_id) if isinstance(cluster_id, (np.integer, np.int64)) else cluster_id
            feature_idx_py = int(best_feature['feature_idx']) if isinstance(best_feature['feature_idx'], (np.integer, np.int64)) else best_feature['feature_idx']
            
            best_features[str(cluster_id_py)] = {
                'feature_idx': feature_idx_py,
                'correlation': float(best_feature['correlation']),
                'f1_score': float(best_feature['f1_score']),
                'image_path': f"instance_{instance_idx}/cluster_{cluster_id_py}_best_feature_{feature_idx_py}.png"
            }
            
        # Count unique clusters in this instance
        unique_clusters = set(res['cluster_id'] for res in instance_results)
        # Convert to Python native integers
        unique_clusters_py = [int(c) if isinstance(c, (np.integer, np.int64)) else c for c in unique_clusters]
        
        # Store instance data
        instance_data[str(instance_idx)] = {
            'clusters': sorted(unique_clusters_py),
            'best_features': best_features,
        }
    
    # Generate HTML file
    with open(html_path, 'w') as f:
        f.write("""<!DOCTYPE html>
<html>
<head>
    <title>TSP SAE Feature Analysis</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 0;
            background-color: #f5f5f5;
            color: #333;
            height: 100vh;
            display: flex;
            flex-direction: column;
        }
        .container {
            max-width: 90%;
            margin: 0 auto;
            flex: 1;
            display: flex;
            flex-direction: column;
            padding: 40px;
        }
        h1 {
            color: #333;
            margin-top: 0;
            padding-bottom: 10px;
            border-bottom: 1px solid #ddd;
            font-size: 1.6em;
            margin-bottom: 20px;
        }
        .control-panel {
            display: flex;
            align-items: center;
            gap: 25px;
            margin-bottom: 20px;
        }
        .control-group {
            display: flex;
            align-items: center;
        }
        label {
            font-weight: bold;
            margin-right: 12px;
            white-space: nowrap;
            font-size: 16px;
        }
        select {
            padding: 10px 12px;
            border: 1px solid #ccc;
            border-radius: 4px;
            min-width: 250px;
            font-size: 16px;
        }
        .image-container {
            flex: 1;
            display: flex;
            justify-content: center;
            align-items: center;
            overflow: auto;
            margin: 0 30px;
        }
        img {
            max-width: 100%;
            max-height: 100%;
            object-fit: contain;
        }
        #current-view {
            font-size: 16px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>TSP SAE Feature Analysis - """ + RUN_NAME + """ - """ + SAE_RUN_NAME + """</h1>
        
        <div class="control-panel">
            <div class="control-group">
                <label for="instance-select">Instance:</label>
                <select id="instance-select" onchange="updateView()">
""")

        # Add instance options
        for instance_idx in sorted([int(k) for k in instance_data.keys()]):
            clusters_count = len(instance_data[str(instance_idx)]["clusters"])
            f.write(f'                <option value="{instance_idx}">Instance {instance_idx} ({clusters_count} clusters)</option>\n')

        f.write("""                </select>
            </div>
            
            <div class="control-group">
                <label for="cluster-select">Cluster:</label>
                <select id="cluster-select" onchange="updateClusterView()">
                    <!-- Cluster options will be populated by JavaScript -->
                </select>
            </div>
            
            <div id="current-view" style="margin-left: auto; font-style: italic;"></div>
        </div>
        
        <div class="image-container">
            <img id="display-image" src="" alt="Visualization">
        </div>
    </div>

    <script>
    // Store all data about instances and visualizations
    """)

        # Add JavaScript data object with explicit type conversion
        def convert_to_serializable(obj):
            if isinstance(obj, dict):
                return {str(convert_to_serializable(k)): convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(i) for i in obj]
            elif isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
                return float(obj)
            else:
                return obj
        
        serializable_data = convert_to_serializable(instance_data)
        f.write(f"const instanceData = {json.dumps(serializable_data)};\n")
        
        f.write("""
    // Function to update the cluster dropdown options based on selected instance
    function updateClusterOptions() {
        const instanceSelect = document.getElementById('instance-select');
        const clusterSelect = document.getElementById('cluster-select');
        const selectedInstance = instanceSelect.value;
        
        // Clear existing options
        clusterSelect.innerHTML = '';
        
        // Add new options based on selected instance - keep clusters sorted numerically
        const clusters = instanceData[selectedInstance].clusters.sort((a, b) => a - b);
        
        clusters.forEach(clusterId => {
            const clusterIdStr = clusterId.toString();
            const feature = instanceData[selectedInstance].best_features[clusterIdStr].feature_idx;
            const corr = Math.abs(instanceData[selectedInstance].best_features[clusterIdStr].correlation).toFixed(3);
            const option = document.createElement('option');
            option.value = clusterIdStr;
            option.textContent = `Cluster ${clusterId} (Feature ${feature}, Corr: ${corr})`;
            clusterSelect.appendChild(option);
        });
    }
    
    // Function to update view label
    function updateViewLabel(instanceIdx, clusterId) {
        const currentView = document.getElementById('current-view');
        const featureData = instanceData[instanceIdx].best_features[clusterId];
        
        currentView.textContent = `Instance ${instanceIdx}: Best Feature for Cluster ${clusterId} (Feature ${featureData.feature_idx})`;
    }
    
    // Function to update the view based on instance selection
    function updateView() {
        // Update cluster options 
        updateClusterOptions();
        
        // Select first cluster and update view
        const clusterSelect = document.getElementById('cluster-select');
        if (clusterSelect.options.length > 0) {
            clusterSelect.selectedIndex = 0;
            updateClusterView();
        }
    }
    
    // Function to update view when cluster changes
    function updateClusterView() {
        const instanceSelect = document.getElementById('instance-select');
        const clusterSelect = document.getElementById('cluster-select');
        const displayImage = document.getElementById('display-image');
        
        const selectedInstance = instanceSelect.value;
        const selectedCluster = clusterSelect.value;
        
        // Show selected cluster's image if it exists
        const imagePath = instanceData[selectedInstance].best_features[selectedCluster].image_path;
        displayImage.src = imagePath;
        
        // Update view label
        updateViewLabel(selectedInstance, selectedCluster);
    }
    
    // Initialize the view
    window.onload = function() {
        updateView();
    };
    </script>
</body>
</html>""")

    print(f"Interactive HTML viewer created at {html_path}")
    return html_path

# --- Script Execution ---
if __name__ == "__main__":
    # Load configs
    tsp_config = load_config(BASE_RUN_PATH / "config.json")
    # Update NUM_LOC from config if needed
    NUM_LOC = tsp_config.get('num_loc', NUM_LOC)

    # Load models
    env_name = tsp_config.get('env_name', 'tsp')
    tsp_model = load_tsp_model(BASE_RUN_PATH, tsp_config, env_name)
    sae_model, activation_key = load_sae_model(SAE_PATH, tsp_config)

    # Generate instances with cluster IDs
    print(f"\nGenerating {NUM_INSTANCES} instances with {NUM_LOC} nodes each...")
    cluster_gen = ClustersWithIDs(min_clusters=3, max_clusters=7, num_loc=NUM_LOC)
    
    # Generate all instances at once
    instance_data = cluster_gen.sample(size=[NUM_INSTANCES])

    # Process each instance
    all_results = []
    for instance_idx in range(NUM_INSTANCES):
        print(f"\n=== Processing Instance {instance_idx} ===")
        
        # Create instance-specific output directory
        instance_output_dir = OUTPUT_DIR / f"instance_{instance_idx}"
        os.makedirs(instance_output_dir, exist_ok=True)
        
        # Run analysis for this instance with the specific output directory
        instance_results = analyze_single_instance(
            tsp_model,
            sae_model,
            activation_key,
            instance_data,
            instance_idx=instance_idx,
            output_dir=instance_output_dir  # Pass the directory
        )
        all_results.append(instance_results)
        
        # Save instance-specific results
        results_save_path = instance_output_dir / f"analysis_results.json"
        # Convert numpy types for JSON serialization
        serializable_results = []
        for r in instance_results:
            serializable_item = {}
            for k, v in r.items():
                # Convert all numpy types to Python native types
                if isinstance(v, (np.integer, np.int64, np.int32, np.int16, np.int8)):
                    serializable_item[k] = int(v)
                elif isinstance(v, (np.floating, np.float64, np.float32, np.float16)):
                    serializable_item[k] = float(v)
                else:
                    serializable_item[k] = v
            serializable_results.append(serializable_item)

        with open(results_save_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        print(f"Saved instance results to {results_save_path}")
    
    # Create a summary of all instances
    summary_path = OUTPUT_DIR / "analysis_summary.txt"
    with open(summary_path, 'w') as f:
        f.write(f"Analysis of {NUM_INSTANCES} TSP instances with SAE features\n")
        f.write(f"Run: {RUN_NAME}\n")
        f.write(f"SAE: {SAE_RUN_NAME}\n\n")
        
        for instance_idx, instance_results in enumerate(all_results):
            # Get the top 5 matches for this instance
            top_matches = sorted(
                instance_results, 
                key=lambda x: (abs(x['correlation']), x['f1_score']), 
                reverse=True
            )[:5]
            
            f.write(f"Instance {instance_idx} - Top 5 Feature-Cluster Matches:\n")
            for i, match in enumerate(top_matches):
                f.write(f"  {i+1}. Feature {match['feature_idx']} <-> Cluster {match['cluster_id']} | ")
                f.write(f"Corr: {match['correlation']:.4f}, F1: {match['f1_score']:.4f}\n")
            f.write("\n")
    
    print(f"\nAnalysis complete for {NUM_INSTANCES} instances.")
    print(f"Results saved to {OUTPUT_DIR}")
    print(f"Summary saved to {summary_path}")
    
    # After processing all instances and creating the summary file
    interactive_html_path = create_interactive_html(all_results, OUTPUT_DIR)
    
    print(f"Interactive HTML viewer created at {interactive_html_path}")
    print("\nAnalysis complete.")