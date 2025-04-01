import os
import glob
import json
import pickle
import argparse
import sys

import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import imageio

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from policy_hooked import HookedAttentionModelPolicy
from reinforce_clipped import REINFORCEClipped

def main(run_path: str, fps: float, batch_size: int = 4, num_instances: int = 5):
    # Convert fps to duration (seconds per frame)
    duration = 1.0 / fps
    
    # 1. Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 2. Load environment and configuration.
    config_path = os.path.join(run_path, "config.json")
    with open(config_path, "r") as f:
        config = json.load(f)
    
    env_path = os.path.join(run_path, "env.pkl")
    with open(env_path, "rb") as f:
        env = pickle.load(f)
    
    # Create activations directory if it doesn't exist.
    activations_dir = os.path.join(run_path, "sae/activations")
    os.makedirs(activations_dir, exist_ok=True)
    
    # Create directories for each instance
    for i in range(num_instances):
        instance_dir = os.path.join(activations_dir, f"instance_{i+1}")
        os.makedirs(instance_dir, exist_ok=True)
    
    # 3. Load baseline (optimal tour) and validation data (true node locations).
    baseline_path = os.path.join(run_path, "baseline.pkl")
    with open(baseline_path, "rb") as f:
        baseline = pickle.load(f)
    
    val_td_path = os.path.join(run_path, "val_td.pkl")
    with open(val_td_path, "rb") as f:
        val_td = pickle.load(f)

    # Use a cyclical (rainbow) colormap for the PCA plots.
    cyclical_cmap = "hsv"

    # Prepare dictionaries to store generated intermediate image filenames for each instance.
    activation_img_files = {i: [] for i in range(num_instances)}
    pca_model_img_files = {i: [] for i in range(num_instances)}
    pca_optimal_img_files = {i: [] for i in range(num_instances)}

    # 4. Get all checkpoint paths sorted by epoch number.
    checkpoint_dir = os.path.join(run_path, "checkpoints")
    checkpoint_paths = glob.glob(os.path.join(checkpoint_dir, "checkpoint_epoch_*.ckpt"))
    
    def extract_epoch(cp_path):
        fname = os.path.basename(cp_path)
        epoch_str = fname.split("checkpoint_epoch_")[1].split(".ckpt")[0]
        return int(epoch_str)
    
    checkpoint_paths = sorted(checkpoint_paths, key=extract_epoch)

    print(f"Found {len(checkpoint_paths)} checkpoints to process.")

    # 5. Iterate over each checkpoint and generate plots.
    for cp in checkpoint_paths:
        epoch = extract_epoch(cp)
        print(f"\nProcessing checkpoint for epoch {epoch} ...")
        
        # Each iteration creates a fresh policy so that hooks get registered cleanly.
        policy = HookedAttentionModelPolicy(
            env_name=env.name,
            embed_dim=config['embed_dim'],
            num_encoder_layers=config['n_encoder_layers'],
            num_heads=8,
            temperature=config['temperature'],
        ).to(device)
        
        # Load the model checkpoint.
        model = REINFORCEClipped.load_from_checkpoint(
            cp,
            env=env,
            policy=policy,
            strict=False
        ).to(device)
        model.eval()
        
        # 6. Generate input batch and perform forward pass.
        td = env.reset(batch_size=batch_size).to(device)
        output = model.policy(td, env, phase="test", return_actions=True)
        
        # Process each instance
        for instance_idx in range(num_instances):
            # 7. Get activations from each encoder layer.
            num_layers = len(model.policy.encoder.net.layers)
            activations = []
            for layer_idx in range(num_layers):
                act_name = f"encoder_layer_{layer_idx}"
                activation = model.policy.get_activation(act_name)
                if activation is not None:
                    print(f"Epoch {epoch}: Activation for {act_name} has shape {activation.shape}")
                    activations.append(activation[instance_idx].detach().cpu().numpy())
                else:
                    print(f"Epoch {epoch}: No activation found for {act_name}")
                    activations.append(None)
            
            instance_dir = os.path.join(activations_dir, f"instance_{instance_idx+1}")
            
            # Plot 1: All activations per checkpoint.
            fig, axes = plt.subplots(num_layers, 1, figsize=(12, 2 * num_layers))
            if num_layers == 1:
                axes = [axes]
            for idx, (ax, act) in enumerate(zip(axes, activations)):
                if act is not None:
                    im = ax.imshow(act, cmap="RdBu", interpolation="nearest", aspect="auto")
                    plt.colorbar(im, ax=ax, label="Activation Value", shrink=0.95, pad=0.02)
                else:
                    ax.text(0.5, 0.5, "No Activation", horizontalalignment="center", verticalalignment="center")
                ax.set_title(f"Layer {idx}")
                ax.set_ylabel("Node Index")
                if idx == num_layers - 1:
                    ax.set_xlabel("Embedding Dimension")
                else:
                    ax.set_xticklabels([])
            fig.suptitle(f"Activations - Epoch {epoch} (Instance {instance_idx+1})", fontsize=16)
            fig.tight_layout(rect=[0, 0, 1, 0.93])
            act_img_path = os.path.join(instance_dir, f"all_activations_epoch_{epoch:04d}.png")
            fig.savefig(act_img_path)
            plt.close(fig)
            activation_img_files[instance_idx].append(act_img_path)
            
            # Prepare tours for PCA plots.
            # For the optimal tour, load tour from baseline.
            tour_optimal = baseline["actions"][0][instance_idx].to(device)
            # For the model tour, use the output from the forward pass.
            tour_model = output['actions'][instance_idx]
            node_locations = val_td["locs"][instance_idx].cpu()
            
            def get_normalized_positions(tour):
                # Create a tensor of the same size as tour
                tour_pos = torch.zeros_like(tour, device=device)
                # For each position in the tour, store its order
                # Need to handle the tour as a sequence, not assign values directly
                for i, node_idx in enumerate(tour):
                    tour_pos[node_idx] = i
                return tour_pos.float() / len(tour)
            
            norm_pos_opt = get_normalized_positions(tour_optimal)
            norm_pos_model = get_normalized_positions(tour_model)
            
            # Plot 2: PCA plots colored by optimal tour.
            fig_opt, axes_opt = plt.subplots(1, num_layers + 1, figsize=(4 * (num_layers + 1), 4))
            # Left-most subplot: true node locations using cyclical colormap.
            ax = axes_opt[0]
            scatter = ax.scatter(node_locations[:, 0], node_locations[:, 1],
                                 c=norm_pos_opt.cpu(),
                                 cmap=cyclical_cmap, alpha=0.6, s=50)
            ax.set_title("True Node Locations\n(optimal tour)")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.grid(True, alpha=0.3)
            plt.colorbar(scatter, ax=ax, label="Position in optimal tour")
            
            # For each encoder layer, compute PCA over the node embeddings.
            for idx, act in enumerate(activations):
                ax = axes_opt[idx + 1]
                if act is not None:
                    pca = PCA(n_components=2)
                    embeddings_2d = pca.fit_transform(act)
                    scatter = ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                                         c=norm_pos_opt.cpu(),
                                         cmap=cyclical_cmap, alpha=0.6, s=50)
                    ax.set_title(f"Layer {idx}\nExplained var: {pca.explained_variance_ratio_[0]:.2f}, {pca.explained_variance_ratio_[1]:.2f}")
                    plt.colorbar(scatter, ax=ax, label="Position in optimal tour")
                else:
                    ax.text(0.5, 0.5, "No Activation", horizontalalignment="center", verticalalignment="center")
                ax.set_xlabel("First PC")
                ax.set_ylabel("Second PC")
                ax.grid(True, alpha=0.3)
            fig_opt.suptitle(f"PCA (Optimal Tour) - Epoch {epoch} (Instance {instance_idx+1})", fontsize=16)
            fig_opt.tight_layout(rect=[0, 0, 1, 0.93])
            opt_img_path = os.path.join(instance_dir, f"pca_by_layer_optimal_tour_epoch_{epoch:04d}.png")
            fig_opt.savefig(opt_img_path)
            plt.close(fig_opt)
            pca_optimal_img_files[instance_idx].append(opt_img_path)
            
            # Plot 3: PCA plots colored by model tour.
            fig_model, axes_model = plt.subplots(1, num_layers + 1, figsize=(4 * (num_layers + 1), 4))
            # Left-most subplot: true node locations using cyclical colormap.
            ax = axes_model[0]
            scatter = ax.scatter(node_locations[:, 0], node_locations[:, 1],
                                 c=norm_pos_model.cpu(),
                                 cmap=cyclical_cmap, alpha=0.6, s=50)
            ax.set_title("True Node Locations\n(model tour)")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.grid(True, alpha=0.3)
            plt.colorbar(scatter, ax=ax, label="Position in model tour")
            
            # For each encoder layer, PCA over the activations.
            for idx, act in enumerate(activations):
                ax = axes_model[idx + 1]
                if act is not None:
                    pca = PCA(n_components=2)
                    embeddings_2d = pca.fit_transform(act)
                    scatter = ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                                         c=norm_pos_model.cpu(),
                                         cmap=cyclical_cmap, alpha=0.6, s=50)
                    ax.set_title(f"Layer {idx}\nExplained var: {pca.explained_variance_ratio_[0]:.2f}, {pca.explained_variance_ratio_[1]:.2f}")
                    plt.colorbar(scatter, ax=ax, label="Position in model tour")
                else:
                    ax.text(0.5, 0.5, "No Activation", horizontalalignment="center", verticalalignment="center")
                ax.set_xlabel("First PC")
                ax.set_ylabel("Second PC")
                ax.grid(True, alpha=0.3)
            fig_model.suptitle(f"PCA (Model Tour) - Epoch {epoch} (Instance {instance_idx+1})", fontsize=16)
            fig_model.tight_layout(rect=[0, 0, 1, 0.93])
            model_img_path = os.path.join(instance_dir, f"pca_by_layer_model_tour_epoch_{epoch:04d}.png")
            fig_model.savefig(model_img_path)
            plt.close(fig_model)
            pca_model_img_files[instance_idx].append(model_img_path)
    
    # 8. Create gifs from the accumulated images for each instance.
    def create_gif(image_files, gif_path, duration):
        images = []
        for filename in image_files:
            images.append(imageio.imread(filename))
        imageio.mimsave(gif_path, images, duration=duration)
    
    for instance_idx in range(num_instances):
        instance_dir = os.path.join(activations_dir, f"instance_{instance_idx+1}")
        gif_activation = os.path.join(instance_dir, "all_activations.gif")
        gif_optimal = os.path.join(instance_dir, "pca_by_layer_optimal_tour.gif")
        gif_model = os.path.join(instance_dir, "pca_by_layer_model_tour.gif")
        
        create_gif(activation_img_files[instance_idx], gif_activation, duration)
        create_gif(pca_optimal_img_files[instance_idx], gif_optimal, duration)
        create_gif(pca_model_img_files[instance_idx], gif_model, duration)
        
        print(f"\nCreated gifs for instance {instance_idx+1}:")
        print(f"Activations gif:\n  {gif_activation}")
        print(f"PCA Optimal Tour gif:\n  {gif_optimal}")
        print(f"PCA Model Tour gif:\n  {gif_model}")
    
    # 9. Delete intermediate image files.
    for file_lists in [activation_img_files, pca_optimal_img_files, pca_model_img_files]:
        for img_list in file_lists.values():
            for file in img_list:
                os.remove(file)
    print("\nDeleted intermediate image files.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate activation and PCA GIFs for all checkpoints in a run."
    )
    parser.add_argument(
        "--run_path",
        type=str,
        default="runs/TSP100_uniform_02-06_12:17:31",
        help="Path to the run directory"
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=4.0,
        help="Frames per second in the GIF"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for the environment reset"
    )
    parser.add_argument(
        "--num_instances",
        type=int,
        default=4,
        help="Number of instances to process"
    )
    args = parser.parse_args()
    
    main(args.run_path, args.fps, args.batch_size, args.num_instances)
