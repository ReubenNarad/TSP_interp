# This is where we will load up the SAE model and visualize the leaned features

import os
import sys
import argparse
import pickle
import json
import glob
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from tqdm import tqdm

# Fix imports from parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from policy.policy_hooked import HookedAttentionModelPolicy
from policy.reinforce_clipped import REINFORCEClipped
from sae.model.sae_model import TopKSparseAutoencoder

# Import TSPModelVisualizer from plotter
from policy.plotter import TSPModelVisualizer

class FeatureAnalyzer:
    """Analyzer for SAE features on TSP instances"""
    
    def __init__(
        self, 
        run_path: str, 
        sae_path: str, 
        feature_indices: Optional[List[int]] = None,
        device: Optional[str] = None,
        checkpoint_epoch: Optional[int] = None
    ):
        """
        Initialize the feature analyzer.
        
        Args:
            run_path: Path to the main TSP run directory
            sae_path: Path to the specific SAE run to analyze
            feature_indices: List of feature indices to analyze (default: auto-select top-k)
            device: Device to run analysis on
            checkpoint_epoch: Specific model checkpoint epoch to use
        """
        self.run_path = Path(run_path)
        self.sae_path = Path(sae_path)
        self.feature_indices = feature_indices
        self.checkpoint_epoch = checkpoint_epoch
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Load environment, TSP model, and SAE model
        self.env = self._load_env()
        self.config = self._load_config()
        self.tsp_model = self._load_tsp_model()
        self.sae_model = self._load_sae_model()
        
        # Create TSP visualizer
        self.tsp_visualizer = TSPModelVisualizer(
            run_path=self.run_path,
            checkpoint_epoch=self.checkpoint_epoch,
            device=self.device
        )
        
        # Create output directory for visualizations
        self.viz_dir = self.sae_path / "feature_analysis"
        os.makedirs(self.viz_dir, exist_ok=True)
        
        # Store instance-feature activations for indexing
        self.activation_index = {}
        
    def _load_env(self):
        """Load the environment from the run path"""
        env_path = self.run_path / "env.pkl"
        if not env_path.exists():
            raise FileNotFoundError(f"Environment file not found at {env_path}")
        
        with open(env_path, "rb") as f:
            env = pickle.load(f)
        
        print(f"Loaded environment from {env_path}")
        return env
    
    def _load_config(self):
        """Load the configuration from the run path"""
        config_path = self.run_path / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found at {config_path}")
        
        with open(config_path, "r") as f:
            config = json.load(f)
        
        print(f"Loaded config from {config_path}")
        return config
    
    def _load_tsp_model(self):
        """Load the TSP model from the latest checkpoint"""
        checkpoint_dir = self.run_path / "checkpoints"
        checkpoint_files = glob.glob(str(checkpoint_dir / "checkpoint_epoch_*.ckpt"))
        
        if not checkpoint_files:
            raise FileNotFoundError(f"No checkpoint files found in {checkpoint_dir}")
        
        # Get the latest checkpoint by epoch number
        latest_checkpoint = max(
            checkpoint_files,
            key=lambda f: int(os.path.basename(f).split("checkpoint_epoch_")[1].split(".ckpt")[0])
        )
        
        # Create policy model
        policy = HookedAttentionModelPolicy(
            env_name=self.env.name,
            embed_dim=self.config['embed_dim'],
            num_encoder_layers=self.config['n_encoder_layers'],
            num_heads=8,
            temperature=self.config['temperature'],
        )
        
        # Load checkpoint
        model = REINFORCEClipped.load_from_checkpoint(
            latest_checkpoint,
            env=self.env,
            policy=policy,
            strict=False
        )
        model = model.to(self.device)
        model.eval()
        
        print(f"Loaded TSP model from checkpoint: {latest_checkpoint}")
        return model
    
    def _load_sae_model(self):
        """Load the SAE model from the SAE path"""
        # Find the SAE configuration
        sae_config_path = self.sae_path / "sae_config.json"
        if not sae_config_path.exists():
            raise FileNotFoundError(f"SAE config not found at {sae_config_path}")
        
        with open(sae_config_path, "r") as f:
            sae_config = json.load(f)
        
        # Find the model file (sae_final.pt)
        sae_model_path = self.sae_path / "sae_final.pt"
        if not sae_model_path.exists():
            # Try to find the latest checkpoint
            checkpoint_dir = self.sae_path / "checkpoints"
            if checkpoint_dir.exists():
                checkpoint_files = glob.glob(str(checkpoint_dir / "sae_epoch_*.pt"))
                if checkpoint_files:
                    latest_checkpoint = max(
                        checkpoint_files,
                        key=lambda f: int(os.path.basename(f).split("sae_epoch_")[1].split(".pt")[0])
                    )
                    sae_model_path = Path(latest_checkpoint)
        
        if not sae_model_path.exists():
            raise FileNotFoundError(f"SAE model not found at {sae_model_path}")
        
        # Find the activation path to determine input dimension
        activation_dir = self.run_path / "sae" / "activations"
        activation_files = glob.glob(str(activation_dir / "activations_epoch_*.pt"))
        
        if not activation_files:
            raise FileNotFoundError(f"No activation files found in {activation_dir}")
        
        # Load one activation file to determine input dimension
        activations = torch.load(activation_files[0])
        activation_key = sae_config.get("activation_key", "encoder_output")
        
        if activation_key not in activations:
            # Try to find a key that might match
            for key in activations.keys():
                if "encoder" in key and "output" in key:
                    activation_key = key
                    break
            else:
                raise ValueError(f"Activation key {activation_key} not found in {activation_files[0]}")
        
        input_dim = activations[activation_key].shape[1]
        
        # Create SAE model
        expansion_factor = sae_config.get("expansion_factor", 2.0)
        latent_dim = int(expansion_factor * input_dim)
        k_ratio = sae_config.get("k_ratio", 0.05)
        
        # Create and load the model
        sae_model = TopKSparseAutoencoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            k_ratio=k_ratio,
            tied_weights=sae_config.get("tied_weights", False),
            bias_decay=sae_config.get("bias_decay", 0.0),
            dict_init=sae_config.get("init_method", "uniform")
        )
        
        # Load state dict
        checkpoint = torch.load(sae_model_path, map_location=self.device)
        # Check if this is a training checkpoint (with model_state_dict) or just model weights
        if "model_state_dict" in checkpoint:
            sae_model.load_state_dict(checkpoint["model_state_dict"])
        else:
            sae_model.load_state_dict(checkpoint)
        sae_model = sae_model.to(self.device)
        sae_model.eval()
        
        print(f"Loaded SAE model from {sae_model_path}")
        print(f"SAE model: {input_dim} -> {latent_dim}, k_ratio={k_ratio}")
        
        return sae_model
    
    def collect_activations(self, instances, batch_size=64):
        """
        Collect SAE feature activations for the given instances.
        
        Args:
            instances: TSP instances to analyze
            batch_size: Batch size for processing
        
        Returns:
            Feature activations of shape [num_instances, num_nodes, latent_dim]
        """
        self.tsp_model.eval()
        self.sae_model.eval()
        
        # Process in batches to avoid OOM
        all_model_acts = []
        all_sae_acts = []
        
        with torch.no_grad():
            for i in range(0, len(instances['locs']), batch_size):
                # Create a batch of the correct size
                batch = {k: v[i:i+batch_size] for k, v in instances.items()}
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Get model activations - directly access the encoder
                # The encoder returns a tuple of (h, init_h) where h is the processed embedding
                h, init_h = self.tsp_model.policy.encoder(batch)
                
                # Store the encoder output in activation cache manually
                self.tsp_model.policy.activation_cache["encoder_output"] = h
                
                # Use the processed embeddings (h) as our model activations
                model_acts = h
                
                # Reshape if needed [batch_size, num_nodes, feat_dim]
                if len(model_acts.shape) == 3:
                    batch_size, num_nodes, feat_dim = model_acts.shape
                    # Flatten to [batch_size*num_nodes, feat_dim]
                    model_acts_flat = model_acts.reshape(-1, feat_dim)
                else:
                    model_acts_flat = model_acts
                
                # Forward pass through SAE
                _, sae_acts = self.sae_model(model_acts_flat)
                
                # Reshape back to [batch_size, num_nodes, latent_dim]
                if len(model_acts.shape) == 3:
                    sae_acts = sae_acts.reshape(batch_size, num_nodes, -1)
                
                all_model_acts.append(model_acts)
                all_sae_acts.append(sae_acts)
        
        # Combine results
        model_activations = torch.cat(all_model_acts, dim=0)
        sae_activations = torch.cat(all_sae_acts, dim=0)
        
        return model_activations, sae_activations
    
    def find_top_features(self, num_instances=100, num_features=6):
        """
        Find the most active features across random instances.
        
        Args:
            num_instances: Number of instances to generate
            num_features: Number of most active features to find
            
        Returns:
            List of top feature indices
        """
        if self.feature_indices is not None:
            return self.feature_indices
        
        print(f"Finding {num_features} most active features from {num_instances} instances...")
        
        # Generate instances
        instances = self.env.reset(batch_size=[num_instances]).to(self.device)
        
        # Get activations
        _, sae_acts = self.collect_activations(instances)
        
        # Average across instances and nodes
        # Shape: [num_instances, num_nodes, latent_dim] -> [latent_dim]
        avg_activations = sae_acts.mean(dim=(0, 1))
        
        # Get top features
        top_values, top_indices = torch.topk(avg_activations, num_features)
        
        top_indices = top_indices.cpu().numpy()
        top_values = top_values.cpu().numpy()
        
        # Print top features and their average activations
        print("Top features by average activation:")
        for i, (idx, val) in enumerate(zip(top_indices, top_values)):
            print(f"  {i+1}. Feature {idx}: {val:.6f}")
        
        return top_indices.tolist()
    
    def visualize_feature_activation(self, instance, sae_activations, feature_idx, 
                                     instance_idx=0, ax=None, save_path=None,
                                     show_colorbar=True, max_val=None, show_solution=False):
        """
        Visualize feature activation on a TSP instance, with optional solution overlay.
        
        Args:
            instance: TSP instance
            sae_activations: SAE activations for instance nodes
            feature_idx: Feature index to visualize
            instance_idx: Instance index in the batch (if multiple)
            ax: Matplotlib axis (optional)
            save_path: Path to save the figure (optional)
            show_colorbar: Whether to show the colorbar
            max_val: Maximum value for color normalization
            show_solution: Whether to show solution path overlaid
            
        Returns:
            Matplotlib axis
        """
        # Create a new figure if ax is not provided
        create_new_fig = ax is None
        if create_new_fig:
            fig, ax = plt.subplots(figsize=(10, 10))
        
        # Extract node coordinates for this instance
        node_coords = instance['locs'][instance_idx].cpu().numpy()
        
        # Get activations for the specified feature
        node_activations = sae_activations[instance_idx, :, feature_idx].cpu().numpy()
        
        # Store average activation for this feature on this instance
        instance_key = f"instance_{instance_idx}"
        if instance_key not in self.activation_index:
            self.activation_index[instance_key] = {}
        self.activation_index[instance_key][feature_idx] = float(node_activations.mean())  # Convert to Python float
        
        # Create color normalization
        if max_val is None:
            norm = Normalize(vmin=0, vmax=node_activations.max())
        else:
            norm = Normalize(vmin=0, vmax=max_val)
        
        # Plot nodes colored by activation
        scatter = ax.scatter(
            node_coords[:, 0], 
            node_coords[:, 1],
            c=node_activations,
            cmap='viridis',
            s=100,
            alpha=1,
            edgecolors='black',
            norm=norm
        )
        
        # Add colorbar
        if show_colorbar:
            plt.colorbar(scatter, ax=ax, label=f'Feature {feature_idx} Activation')
        
        # Add title and labels for feature plot
        avg_activation = node_activations.mean()
        ax.set_title(f'Feature {feature_idx} Activation (Avg: {avg_activation:.4f})')
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        
        # Set equal aspect ratio and limits
        ax.set_aspect('equal')
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        
        # Show solution overlay if requested
        if show_solution:
            try:
                # Create a fresh instance
                single_instance = self.env.reset(batch_size=[1])
                single_instance = single_instance.to(self.device)
                # Set locations to match the instance we're visualizing
                single_instance['locs'][0] = instance['locs'][instance_idx].clone()
                
                # Use plotter to get the solution
                solution = self.tsp_visualizer.decode_solution(single_instance)
                
                # Get node coordinates and solution path
                coords = node_coords
                path = solution['actions'][0].cpu().numpy()
                
                # Complete the tour by connecting back to start
                tour_path = np.append(path, path[0])
                
                # Plot tour with arrows - use consistent head size and lower zorder
                for idx in range(len(tour_path) - 1):
                    i, j = tour_path[idx], tour_path[idx + 1]
                    # Calculate vector components
                    dx = coords[j, 0] - coords[i, 0]
                    dy = coords[j, 1] - coords[i, 1]
                    
                    # Use fixed arrow head sizes instead of scaling them
                    ax.arrow(
                        coords[i, 0], coords[i, 1],  # Start point
                        dx, dy,  # Direction vector
                        head_width=0.01,  # Fixed head width
                        head_length=0.03,  # Fixed head length
                        fc='black', ec='black',
                        alpha=0.5,
                        length_includes_head=True,
                        zorder=0  # Lower zorder so arrows appear under nodes
                    )
                
                # Add solution info to title
                reward = solution['reward'][0].item()
                distance = -reward  # Negative because reward is negative of distance
                ax.set_title(f'Feature {feature_idx} Activation (Avg: {avg_activation:.4f})\nTour Length: {distance:.4f}')
            except Exception as e:
                print(f"Error generating solution: {e}")
        
        plt.tight_layout()
        
        # Save the figure if a path is provided
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved visualization to {save_path}")
            
            # Close the figure if we created it
            if create_new_fig:
                plt.close(fig)
        
        return ax  # Return the axis
    
    def visualize_top_features_for_instance(self, instance, sae_activations, 
                                           feature_indices, instance_idx=0, save_path=None,
                                           show_solution=False):
        """
        Visualize multiple features for a single TSP instance.
        
        Args:
            instance: TSP instance
            sae_activations: SAE activations
            feature_indices: List of feature indices to visualize
            instance_idx: Instance index in the batch
            save_path: Path to save the figure
            show_solution: Whether to show TSP solution beside each feature
        """
        num_features = len(feature_indices)
        
        # If showing solutions, we'll have a 2-column layout per feature
        if show_solution:
            fig, axes = plt.subplots(num_features, 2, figsize=(20, 10*num_features))
            
            # Convert axes to proper shape for indexing
            if num_features == 1:
                axes = axes.reshape(1, 2)
            
            # Find global max for consistent coloring
            max_val = 0
            for i, feature_idx in enumerate(feature_indices):
                act = sae_activations[instance_idx, :, feature_idx].max().item()
                max_val = max(max_val, act)
            
            # Create a fresh instance for solution visualization
            try:
                single_instance = self.env.reset(batch_size=[1])
                single_instance = single_instance.to(self.device)
                single_instance['locs'][0] = instance['locs'][instance_idx].clone()
                solution = self.tsp_visualizer.decode_solution(single_instance)
                coords = instance['locs'][instance_idx].cpu().numpy()
                path = solution['actions'][0].cpu().numpy()
                tour_path = np.append(path, path[0])
                
                # Compute tour length for reference
                reward = solution['reward'][0].item()
                distance = -reward  # Negative because reward is negative of distance
                tour_length = distance
                
                # Visualize each feature with solution
                for i, feature_idx in enumerate(feature_indices):
                    # Plot feature activation
                    feature_ax = self.visualize_feature_activation(
                        instance, sae_activations, feature_idx, 
                        instance_idx, axes[i, 0], max_val=max_val,
                        show_solution=False  # Don't show solution here
                    )
                    
                    # Plot solution on second axis
                    sol_ax = axes[i, 1]
                    
                    # Plot nodes
                    sol_ax.scatter(coords[:, 0], coords[:, 1], c='blue', s=100, zorder=2)
                        
                    # Plot tour with arrows - use consistent head size and lower zorder
                    for idx in range(len(tour_path) - 1):
                        i, j = tour_path[idx], tour_path[idx + 1]
                        # Calculate vector components
                        dx = coords[j, 0] - coords[i, 0]
                        dy = coords[j, 1] - coords[i, 1]
                        
                        # Use fixed arrow head sizes instead of scaling them
                        sol_ax.arrow(
                            coords[i, 0], coords[i, 1],  # Start point
                            dx, dy,  # Direction vector
                            head_width=0.01,  # Fixed head width
                            head_length=0.03,  # Fixed head length
                            fc='black', ec='black',
                            alpha=0.5,
                            length_includes_head=True,
                            zorder=0  # Lower zorder so arrows appear under nodes
                        )
                    
                    # Set title and formatting
                    sol_ax.set_title(f'TSP Solution (Tour Length: {tour_length:.4f})')
                    sol_ax.set_xlabel('X Coordinate')
                    sol_ax.set_ylabel('Y Coordinate')
                    sol_ax.set_aspect('equal')
                    sol_ax.set_xlim(-0.05, 1.05)
                    sol_ax.set_ylim(-0.05, 1.05)
                
            except Exception as e:
                print(f"Error generating solutions: {e}")
                # We'll still show feature activations and just display error for solutions
                for i, feature_idx in enumerate(feature_indices):
                    # Plot feature activation
                    self.visualize_feature_activation(
                        instance, sae_activations, feature_idx, 
                        instance_idx, axes[i, 0], max_val=max_val,
                        show_solution=False
                    )
                    
                    # Show error on solution axis
                    axes[i, 1].text(0.5, 0.5, "Solution visualization failed", 
                                 ha='center', va='center', fontsize=12)
                    axes[i, 1].set_title("Error")
                    axes[i, 1].set_aspect('equal')
                    axes[i, 1].set_xlim(-0.05, 1.05)
                    axes[i, 1].set_ylim(-0.05, 1.05)
        else:
            # Standard layout without solutions
            rows = int(np.ceil(num_features / 2))
            cols = min(2, num_features)
            
            fig, axes = plt.subplots(rows, cols, figsize=(10*cols, 10*rows))
            
            if num_features == 1:
                axes = np.array([axes])
            
            axes = axes.flatten()
            
            # Find global max for consistent coloring
            max_val = 0
            for i, feature_idx in enumerate(feature_indices):
                act = sae_activations[instance_idx, :, feature_idx].max().item()
                max_val = max(max_val, act)
            
            # Visualize each feature
            for i, feature_idx in enumerate(feature_indices):
                if i < len(axes):
                    self.visualize_feature_activation(
                        instance, sae_activations, feature_idx, 
                        instance_idx, axes[i], max_val=max_val,
                        show_solution=False
                    )
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved visualization to {save_path}")
            plt.close(fig)
        
        return fig, axes
    
    def visualize_feature_overlay(self, instances, sae_activations, feature_idx, 
                               num_instances=10, save_path=None, show_solution=False):
        """
        Create an overlay visualization showing how a feature activates across multiple instances.
        
        Args:
            instances: TSP instances to analyze
            sae_activations: SAE activations for instances
            feature_idx: Feature index to visualize
            num_instances: Number of instances to overlay (default: 10)
            save_path: Path to save the figure (optional)
            show_solution: Whether to overlay a solution path (only for first instance)
            
        Returns:
            The matplotlib figure
        """
        # Use a maximum of available instances
        num_instances = min(num_instances, len(instances['locs']))
        
        # Create the figure with extra space for the legend and colorbar
        fig, ax = plt.subplots(figsize=(14, 12), constrained_layout=True)
        
        # Define a color normalization
        norm = Normalize(vmin=0, vmax=1.0)
        
        # Plot each instance with different markers
        markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
        
        # Create legend handles
        legend_handles = []
        
        # Plot each instance
        for i in range(num_instances):
            # Get node coordinates
            node_coords = instances['locs'][i].cpu().numpy()
            
            # Get activations for this feature
            node_activations = sae_activations[i, :, feature_idx].cpu().numpy()
            
            # Create a scatter plot with a unique marker for this instance
            marker = markers[i % len(markers)]
            scatter = ax.scatter(
                node_coords[:, 0], 
                node_coords[:, 1],
                c=node_activations,
                marker=marker,
                s=80,
                alpha=0.8,
                edgecolors='black',
                linewidths=0.5,
                cmap='viridis',
                norm=norm,
                zorder=2
            )
            
            # Add to legend
            legend_handles.append(plt.Line2D([0], [0], marker=marker, color='gray', 
                                           markerfacecolor='gray', markersize=8, 
                                           label=f'Instance {i}'))
            
            # Solution overlay is disabled by default to ensure core functionality works
            # if show_solution and i == 0:
            #     try:
            #         # Solution overlay code would go here
            #         pass
            #     except Exception as e:
            #         print(f"Error generating solution for overlay: {e}")
        
        # Add a colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label(f'Feature {feature_idx} Activation')
        
        # Add title and labels
        ax.set_title(f'Feature {feature_idx} Overlay Across {num_instances} Instances')
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        
        # Add legend
        ax.legend(handles=legend_handles, bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Set equal aspect ratio
        ax.set_aspect('equal')
        
        # Set limits
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        
        # Adjust layout
        # plt.tight_layout()
        
        # Save the figure if a path is provided
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved overlay visualization to {save_path}")
            plt.close(fig)
        
        return fig
    
    def generate_overlay_visualizations(self, num_instances=10, num_overlays=10, 
                                       show_solution=True):
        """
        Generate overlay visualizations for each feature.
        
        Args:
            num_instances: Number of instances to include in each overlay
            num_overlays: Number of overlay visualizations to create
            show_solution: Whether to overlay the TSP solution path
        """
        print(f"Generating overlay visualizations...")
        
        # Create output directory for overlays
        overlay_dir = self.viz_dir / "feature_overlays"
        os.makedirs(overlay_dir, exist_ok=True)
        
        # Generate instances
        instances = self.env.reset(batch_size=[num_instances]).to(self.device)
        
        # Get activations
        _, sae_activations = self.collect_activations(instances)
        
        # Generate overlay for each feature
        for feature_idx in self.feature_indices:
            save_path = overlay_dir / f"feature_{feature_idx}_overlay.png"
            self.visualize_feature_overlay(
                instances, sae_activations, feature_idx, 
                num_instances, save_path=save_path,
                show_solution=show_solution
            )
        
        print(f"Overlay visualizations saved to {overlay_dir}")
        
        return overlay_dir
    
    def analyze_instances(self, num_instances=10, batch_size=16, generate_overlays=True,
                         show_solution=True, num_features=10, skip_multi_feature=False):
        """
        Analyze multiple instances and visualize feature activations.
        
        Args:
            num_instances: Number of instances to analyze
            batch_size: Batch size for processing
            generate_overlays: Whether to generate overlay visualizations
            show_solution: Whether to overlay the TSP solution paths
            num_features: Number of top features to analyze
            skip_multi_feature: If True, skip generating multi-feature visualizations
        """
        print(f"Analyzing {num_instances} instances...")
        
        # Get feature indices to analyze
        if self.feature_indices is None:
            self.feature_indices = self.find_top_features(num_features=num_features)
        
        # Generate instances
        instances = self.env.reset(batch_size=[num_instances]).to(self.device)
        
        # Get activations
        _, sae_activations = self.collect_activations(instances, batch_size=batch_size)
        
        # Create output directories
        feature_dirs = {}
        for feature_idx in self.feature_indices:
            feature_dir = self.viz_dir / f"feature_{feature_idx}"
            os.makedirs(feature_dir, exist_ok=True)
            feature_dirs[feature_idx] = feature_dir
        
        # Create a directory for multi-feature visualizations (if needed)
        if not skip_multi_feature:
            multi_feature_dir = self.viz_dir / "multi_feature"
            os.makedirs(multi_feature_dir, exist_ok=True)
        
        # Visualize each instance
        for instance_idx in tqdm(range(num_instances)):
            # Visualize each feature separately
            for feature_idx in self.feature_indices:
                save_path = feature_dirs[feature_idx] / f"instance_{instance_idx}.png"
                self.visualize_feature_activation(
                    instances, sae_activations, feature_idx, 
                    instance_idx, save_path=save_path,
                    show_solution=show_solution
                )
            
            # Visualize all features for this instance (if not skipped)
            if not skip_multi_feature:
                save_path = multi_feature_dir / f"instance_{instance_idx}.png"
                self.visualize_top_features_for_instance(
                    instances, sae_activations, self.feature_indices, 
                    instance_idx, save_path=save_path,
                    show_solution=show_solution
                )
        
        # Convert any NumPy types to Python native types for JSON serialization
        serializable_index = {}
        for instance_key, features in self.activation_index.items():
            serializable_index[instance_key] = {
                int(feat_idx): float(activation)
                for feat_idx, activation in features.items()
            }
        
        # Save the activation index
        index_path = self.viz_dir / "activation_index.json"
        with open(index_path, "w") as f:
            json.dump(serializable_index, f, indent=2)
        
        # Generate overlay visualizations if requested
        if generate_overlays:
            overlay_dir = self.generate_overlay_visualizations(show_solution=show_solution, num_instances=num_instances)
        
        print(f"Analysis complete. Results saved to {self.viz_dir}")
        
        # Create an HTML index file for easy browsing
        self.create_html_index()
    
    def create_html_index(self):
        """Create an HTML index for easy browsing of visualizations"""
        index_path = self.viz_dir / "index.html"
        
        with open(index_path, "w") as f:
            f.write("<!DOCTYPE html>\n")
            f.write("<html><head><title>SAE Feature Analysis</title>\n")
            f.write("<style>\n")
            f.write("body { font-family: Arial, sans-serif; margin: 20px; }\n")
            f.write(".container { display: flex; flex-wrap: wrap; }\n")
            f.write(".feature { margin: 10px; padding: 10px; border: 1px solid #ccc; }\n")
            f.write(".feature-title { font-weight: bold; margin-bottom: 10px; }\n")
            f.write(".instance-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; }\n")
            f.write(".instance { margin: 5px; }\n")
            f.write("img { max-width: 300px; border: 1px solid #eee; }\n")
            f.write(".collapsed { display: none; }\n")
            f.write(".collapsible { cursor: pointer; padding: 10px; background-color: #f1f1f1; border: none; text-align: left; outline: none; font-size: 16px; font-weight: bold; width: 100%; }\n")
            f.write(".active, .collapsible:hover { background-color: #ddd; }\n")
            f.write(".collapsible:after { content: '\\002B'; float: right; }\n") 
            f.write(".active:after { content: '\\2212'; }\n")
            f.write("</style>\n")
            
            # Add JavaScript for collapsible sections
            f.write("<script>\n")
            f.write("function toggleSection(sectionId) {\n")
            f.write("  var content = document.getElementById(sectionId);\n")
            f.write("  var button = document.querySelector('[data-target=\"' + sectionId + '\"]');\n")
            f.write("  if (content.style.display === 'block') {\n")
            f.write("    content.style.display = 'none';\n")
            f.write("    button.classList.remove('active');\n")
            f.write("  } else {\n")
            f.write("    content.style.display = 'block';\n")
            f.write("    button.classList.add('active');\n")
            f.write("  }\n")
            f.write("}\n")
            f.write("</script>\n")
            
            f.write("</head><body>\n")
            
            f.write(f"<h1>SAE Feature Analysis</h1>\n")
            f.write(f"<p>Run: {self.run_path.name}</p>\n")
            f.write(f"<p>SAE: {self.sae_path.name}</p>\n")
            
            # Add collapsible section for feature overlays
            overlay_dir = self.viz_dir / "feature_overlays"
            if overlay_dir.exists():
                f.write("<button class='collapsible active' data-target='overlay-section' onclick='toggleSection(\"overlay-section\")'>Feature Overlays (Multiple Instances)</button>\n")
                f.write("<div id='overlay-section' style='display: block;'>\n")
                f.write("<div class='instance-grid'>\n")
                
                for feature_idx in self.feature_indices:
                    overlay_file = f"feature_{feature_idx}_overlay.png"
                    overlay_path = overlay_dir / overlay_file
                    
                    if overlay_path.exists():
                        f.write(f"<div class='instance'>\n")
                        f.write(f"<p>Feature {feature_idx} Overlay</p>\n")
                        f.write(f"<a href='feature_overlays/{overlay_file}' target='_blank'>\n")
                        f.write(f"<img src='feature_overlays/{overlay_file}'>\n")
                        f.write(f"</a>\n")
                        f.write("</div>\n")
                
                f.write("</div>\n")
                f.write("</div>\n")
            
            # Add collapsible section for multi-feature visualizations
            if not hasattr(self, 'skip_multi_feature') or not self.skip_multi_feature:
                f.write("<button class='collapsible' data-target='multi-feature-section' onclick='toggleSection(\"multi-feature-section\")'>Multi-Feature Visualizations</button>\n")
                f.write("<div id='multi-feature-section' style='display: none;'>\n")
                
                # Only show the grid if we have instances
                if self.activation_index:
                    f.write("<div class='instance-grid'>\n")
                    multi_feature_dir = "multi_feature"
                    for i in range(len(self.activation_index)):
                        instance_file = f"instance_{i}.png"
                        f.write(f"<div class='instance'>\n")
                        f.write(f"<p>Instance {i}</p>\n")
                        f.write(f"<a href='{multi_feature_dir}/{instance_file}' target='_blank'>\n")
                        f.write(f"<img src='{multi_feature_dir}/{instance_file}'>\n")
                        f.write(f"</a>\n")
                        f.write("</div>\n")
                    f.write("</div>\n")
                else:
                    f.write("<p>No instances analyzed yet.</p>\n")
                
                f.write("</div>\n")
            
            # Add collapsible section for individual features
            f.write("<button class='collapsible' data-target='individual-features-section' onclick='toggleSection(\"individual-features-section\")'>Individual Features</button>\n")
            f.write("<div id='individual-features-section' style='display: none;'>\n")
            
            if self.feature_indices:
                for feature_idx in self.feature_indices:
                    feature_section_id = f"feature-{feature_idx}-section"
                    f.write(f"<button class='collapsible' data-target='{feature_section_id}' onclick='toggleSection(\"{feature_section_id}\")'> Feature {feature_idx}</button>\n")
                    f.write(f"<div id='{feature_section_id}' style='display: none;'>\n")
                    
                    # Find top instances for this feature
                    feature_activations = {
                        instance: data[feature_idx] 
                        for instance, data in self.activation_index.items() 
                        if feature_idx in data
                    }
                    
                    if feature_activations:
                        # Sort instances by activation
                        sorted_instances = sorted(
                            feature_activations.items(), 
                            key=lambda x: x[1], 
                            reverse=True
                        )
                        
                        # Show top 6 instances or fewer if not enough
                        top_instances = sorted_instances[:min(6, len(sorted_instances))]
                        f.write("<div class='instance-grid'>\n")
                        for instance, activation in top_instances:
                            instance_idx = int(instance.split("_")[1])
                            instance_file = f"instance_{instance_idx}.png"
                            feature_dir = f"feature_{feature_idx}"
                            
                            f.write(f"<div class='instance'>\n")
                            f.write(f"<p>Activation: {activation:.4f}</p>\n")
                            f.write(f"<a href='{feature_dir}/{instance_file}' target='_blank'>\n")
                            f.write(f"<img src='{feature_dir}/{instance_file}'>\n")
                            f.write(f"</a>\n")
                            f.write("</div>\n")
                        
                        f.write("</div>\n")
                    else:
                        f.write("<p>No instances yet for this feature.</p>\n")
                    
                    f.write("</div>\n")  # Close feature section
            else:
                f.write("<p>No features selected for analysis.</p>\n")
            
            f.write("</div>\n")  # Close individual features section
            f.write("</body></html>\n")
        
        print(f"Created HTML index at {index_path}")

def main():
    parser = argparse.ArgumentParser(description="Analyze SAE features on TSP instances")
    
    parser.add_argument("--run_path", type=str, required=True,
                       help="Path to the main TSP run directory")
    parser.add_argument("--sae_path", type=str, required=True,
                       help="Path to the specific SAE run to analyze")
    parser.add_argument("--features", type=int, nargs="+", default=None,
                       help="Specific feature indices to analyze (default: auto-select top-k)")
    parser.add_argument("--num_features", type=int, default=10,
                       help="Number of most active features to analyze if --features is not specified")
    parser.add_argument("--num_instances", type=int, default=20,
                       help="Number of instances to analyze")
    parser.add_argument("--batch_size", type=int, default=16,
                       help="Batch size for processing")
    parser.add_argument("--device", type=str, default=None,
                       help="Device to run analysis on (default: auto-select)")
    parser.add_argument("--no_overlays", action="store_true",
                       help="Skip generating overlay visualizations")
    parser.add_argument("--checkpoint_epoch", type=int, default=None,
                       help="Specific model checkpoint epoch to use")
    parser.add_argument("--show_solution", action="store_true",
                       help="Show solution paths side-by-side with feature activations")
    parser.add_argument("--no_multi_feature", action="store_true",
                       help="Skip generating multi-feature visualizations")
    
    args = parser.parse_args()
    
    analyzer = FeatureAnalyzer(
        run_path=args.run_path,
        sae_path=args.sae_path,
        feature_indices=args.features,
        device=args.device,
        checkpoint_epoch=args.checkpoint_epoch
    )
    
    analyzer.analyze_instances(
        num_instances=args.num_instances,
        batch_size=args.batch_size,
        generate_overlays=not args.no_overlays,
        show_solution=args.show_solution,
        num_features=args.num_features,
        skip_multi_feature=args.no_multi_feature
    )

if __name__ == "__main__":
    main()


