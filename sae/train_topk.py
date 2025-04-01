import os
import sys
import json
import pickle
import argparse
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import glob

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from sae directory
from sae.model.sae_model import TopKSparseAutoencoder
from sae.utils import visualize_activation_statistics, get_l0_sparsity, get_dead_features

def train_sae(args):
    """Train a sparse autoencoder on collected activations"""
    # Load configuration
    with open(os.path.join(args.run_dir, "config.json"), "r") as f:
        config = json.load(f)
    
    # Create a unique identifier for this SAE run
    timestamp = datetime.now().strftime("%m-%d_%H:%M:%S")
    sae_run_name = f"sae_l1{args.l1_coef}_ef{args.expansion_factor}_k{args.k_ratio}_{timestamp}"
    
    # Create base SAE directory
    sae_base_dir = os.path.join(args.run_dir, "sae")
    os.makedirs(sae_base_dir, exist_ok=True)
    
    # Create individual run directory
    sae_runs_dir = os.path.join(sae_base_dir, "sae_runs")
    os.makedirs(sae_runs_dir, exist_ok=True)
    
    # Create this specific run directory
    sae_dir = os.path.join(sae_runs_dir, sae_run_name)
    os.makedirs(sae_dir, exist_ok=True)
    
    # Save SAE configuration
    sae_config = vars(args)
    with open(os.path.join(sae_dir, "sae_config.json"), "w") as f:
        json.dump(sae_config, f, indent=2)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Construct the activation path from the run directory and epoch
    activation_dir = os.path.join(args.run_dir, "sae", "activations")
    run_name = os.path.basename(os.path.normpath(args.run_dir))
    
    if args.epoch is None:
        # Find the latest activation file if no specific epoch is provided
        activation_files = glob.glob(os.path.join(activation_dir, "activations_epoch_*.pt"))
        if not activation_files:
            raise ValueError(f"No activation files found in {activation_dir}")
        
        # Get latest epoch
        epochs = [int(f.split('_epoch_')[1].split('.pt')[0]) for f in activation_files]
        latest_epoch = max(epochs)
        args.epoch = latest_epoch
    
    activation_path = os.path.join(activation_dir, f"activations_epoch_{args.epoch}.pt")
    
    # Check if the activation file exists
    if not os.path.exists(activation_path):
        raise ValueError(f"Activation file not found at {activation_path}")
    
    # Load activations
    print(f"Loading activations from {activation_path}")
    data = torch.load(activation_path)
    print(f"Successfully loaded tensor with keys: {data.keys() if isinstance(data, dict) else 'tensor'}")
    
    # Extract relevant activations
    if args.activation_key in data:
        activations = data[args.activation_key]
        print(f"Loaded activations with shape: {activations.shape}")
    else:
        print(f"Available keys in activation file: {list(data.keys())}")
        raise ValueError(f"Activation key '{args.activation_key}' not found in data")
    
    # Convert to torch tensor if needed
    if not isinstance(activations, torch.Tensor):
        activations = torch.tensor(activations, dtype=torch.float32)
    
    # Normalize if requested
    if args.normalize:
        print("Normalizing activations")
        mean = activations.mean(dim=0, keepdim=True)
        std = activations.std(dim=0, keepdim=True) + 1e-5
        activations = (activations - mean) / std
    
    # Visualize original activations
    fig = visualize_activation_statistics(activations, title="Original Activation Statistics")
    fig.savefig(os.path.join(sae_dir, "original_activations.png"))
    plt.close(fig)
    
    # Dataset and dataloader
    dataset = TensorDataset(activations)
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=args.num_workers
    )
    
    # Initialize SAE model
    input_dim = activations.shape[1]
    latent_dim = int(args.expansion_factor * input_dim)
    
    model = TopKSparseAutoencoder(
        input_dim=input_dim,
        latent_dim=latent_dim,
        k_ratio=args.k_ratio,
        tied_weights=args.tied_weights,
        bias_decay=args.bias_decay,
        dict_init=args.init_method
    )
    model.to(device)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=5, 
        verbose=True
    )
    
    # Training loop
    print(f"Training SAE for {args.num_epochs} epochs")
    
    # For tracking metrics
    train_losses = []
    recon_losses = []
    l1_sparsities = []
    dead_neuron_counts = []
    
    for epoch in range(args.num_epochs):
        model.train()
        epoch_losses = []
        
        # Process batches
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.num_epochs}")
        for batch in pbar:
            x = batch[0].to(device)
            
            # Forward pass
            x_recon, z = model(x)
            
            # Compute losses
            recon_loss = torch.mean((x_recon - x) ** 2)
            recon_losses.append(recon_loss.item())
            
            # L1 regularization on activations
            l1_loss = args.l1_coef * torch.mean(torch.sum(torch.abs(z), dim=1))
            
            # L2 regularization on decoder bias (helps prevent exploding biases)
            if model.tied_weights:
                bias_loss = args.bias_decay * torch.sum(model.decoder_bias ** 2)
            else:
                bias_loss = args.bias_decay * torch.sum(model.decoder.bias ** 2)
            
            # Total loss
            loss = recon_loss + l1_loss + bias_loss
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            
            # Clip gradients if needed
            if args.clip_grad_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                
            optimizer.step()
            
            # Track metrics
            epoch_losses.append(loss.item())
            
            # Update progress bar
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "recon": f"{recon_loss.item():.4f}",
                "l1": f"{l1_loss.item():.4f}"
            })
        
        # Calculate epoch metrics
        avg_loss = np.mean(epoch_losses)
        train_losses.append(avg_loss)
        
        # Update learning rate
        scheduler.step(avg_loss)
        
        # Evaluate model on full dataset
        model.eval()
        with torch.no_grad():
            # Process in batches to avoid OOM
            all_z = []
            for batch in DataLoader(dataset, batch_size=args.batch_size):
                x = batch[0].to(device)
                _, z = model(x)
                all_z.append(z)
            
            all_z = torch.cat(all_z, dim=0)
            
            # Calculate sparsity metrics
            l1_sparsity = model.get_l1_sparsity(all_z)
            
            l1_sparsities.append(l1_sparsity)
            
            # Calculate dead neurons
            dead_neurons = model.get_dead_neurons().sum().item()
            dead_neuron_counts.append(dead_neurons)
            
            print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, L1={l1_sparsity:.1f}, Dead={dead_neurons}/{latent_dim}")
        
        # Reinitialize dead neurons if needed
        if args.reinit_dead and (epoch + 1) % args.reinit_freq == 0:
            model.reset_dead_neurons(reinit_type=args.init_method)
        
        # Save checkpoint periodically
        if (epoch + 1) % args.save_freq == 0 or epoch == args.num_epochs - 1:
            checkpoint_dir = os.path.join(sae_dir, "checkpoints")
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, f"sae_epoch_{epoch+1}.pt")
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_loss,
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(sae_dir, "sae_final.pt"))
    
    # Visualize results
    plt.figure(figsize=(12, 8))
    
    # Plot training loss
    plt.subplot(2, 2, 1)
    plt.plot(train_losses)
    plt.title("Total Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    
    # Replace L0 sparsity with reconstruction loss
    plt.subplot(2, 2, 2)
    plt.plot(recon_losses)
    plt.title("Reconstruction Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    
    # Plot L1 sparsity
    plt.subplot(2, 2, 3)
    plt.plot(l1_sparsities)
    plt.title("L1 Sparsity")
    plt.xlabel("Epoch")
    plt.ylabel("Average L1 Norm")
    
    # Plot dead neurons
    plt.subplot(2, 2, 4)
    plt.plot(dead_neuron_counts)
    plt.title("Dead Neurons")
    plt.xlabel("Epoch")
    plt.ylabel("Count")
    
    plt.tight_layout()
    plt.savefig(os.path.join(sae_dir, "training_metrics.png"))
    
    # Analyze final model
    model.eval()
    with torch.no_grad():
        # Get feature statistics
        all_z = []
        for batch in DataLoader(dataset, batch_size=args.batch_size):
            x = batch[0].to(device)
            _, z = model(x)
            all_z.append(z)
        
        all_z = torch.cat(all_z, dim=0)
        
        # Visualize latent space statistics
        fig = visualize_activation_statistics(all_z, title="SAE Latent Activation Statistics")
        fig.savefig(os.path.join(sae_dir, "latent_activations.png"))
        plt.close(fig)
        
        # Calculate and display feature usage statistics
        feature_usage = (all_z > 0).float().sum(dim=0).cpu()
        feature_usage = feature_usage / len(dataset)
        
        plt.figure(figsize=(10, 6))
        plt.hist(feature_usage.numpy(), bins=50)
        plt.title("Feature Usage Distribution")
        plt.xlabel("Usage Fraction")
        plt.ylabel("Count")
        plt.savefig(os.path.join(sae_dir, "feature_usage.png"))
        plt.close()
        
        # Save feature usage data
        torch.save(feature_usage, os.path.join(sae_dir, "feature_usage.pt"))
        
    # Create a symlink to the latest run for convenience
    latest_link_path = os.path.join(sae_base_dir, "latest")
    if os.path.exists(latest_link_path) and os.path.islink(latest_link_path):
        os.unlink(latest_link_path)
    
    try:
        # Use relative path for symlink to avoid issues with different mount points
        rel_path = os.path.relpath(sae_dir, sae_base_dir)
        os.symlink(rel_path, latest_link_path)
        print(f"Created symlink to latest run at {latest_link_path}")
    except (OSError, NotImplementedError) as e:
        # Symlinks might not be supported on some systems (like Windows without admin)
        print(f"Could not create symlink: {e}")
        
    print(f"SAE training completed. Results saved to: {sae_dir}")
    
    # Return the directory where results are stored for reference
    return sae_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Sparse Autoencoder on TSP model activations")
    
    # Data arguments
    parser.add_argument("--run_dir", type=str, required=True, 
                      help="Directory with model checkpoints and activations")
    parser.add_argument("--epoch", type=int, default=None,
                      help="Epoch number of the activation file to use (will use latest if not specified)")
    parser.add_argument("--activation_key", type=str, default="encoder_out",
                      help="Key in the activations dictionary to use")
    parser.add_argument("--normalize", action="store_true",
                      help="Whether to normalize the activations")
    
    # Model arguments
    parser.add_argument("--expansion_factor", type=float, default=2.0,
                      help="Latent dimension as a multiple of input dimension")
    parser.add_argument("--k_ratio", type=float, default=0.05,
                      help="Fraction of latent units to keep active")
    parser.add_argument("--tied_weights", action="store_true",
                      help="Whether to tie encoder and decoder weights")
    parser.add_argument("--init_method", type=str, default="uniform",
                      choices=["uniform", "normal", "xavier"],
                      help="Weight initialization method")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=512,
                      help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=100,
                      help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3,
                      help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.0,
                      help="Weight decay (L2 penalty)")
    parser.add_argument("--l1_coef", type=float, default=1e-3,
                      help="L1 regularization coefficient")
    parser.add_argument("--bias_decay", type=float, default=1e-5,
                      help="L2 regularization for decoder bias")
    parser.add_argument("--clip_grad_norm", type=float, default=1.0,
                      help="Gradient norm clipping (0 to disable)")
    
    # Misc arguments
    parser.add_argument("--reinit_dead", action="store_true",
                      help="Whether to reinitialize dead neurons")
    parser.add_argument("--reinit_freq", type=int, default=2,
                      help="How often to check for and reinitialize dead neurons")
    parser.add_argument("--save_freq", type=int, default=10,
                      help="How often to save checkpoints")
    parser.add_argument("--num_workers", type=int, default=4,
                      help="Number of dataloader workers")
    
    args = parser.parse_args()
    train_sae(args)
