import torch
import matplotlib.pyplot as plt
import json
import numpy as np
from typing import Dict, Any
import os
import matplotlib.patches as patches

from net import TSPModel
from TSP_env import TSP_env

def load_model(checkpoint_path: str) -> tuple[TSPModel, Dict[str, Any]]:
    """Loads model and config from checkpoint."""
    checkpoint = torch.load(checkpoint_path)
    config = checkpoint['config']
    
    # Infer grid size from the data path (e.g., "5x5" from the run directory name)
    run_dir = os.path.dirname(checkpoint_path)
    grid_size = int(os.path.basename(run_dir).split('x')[0])
    
    # Initialize model with config
    gpt2_config = {
        'n_positions': config['max_context_length'] + 10,
        'n_embd': config['hidden_dim'],
        'n_layer': config.get('n_layers', 6),  # Default to 6 layers
        'n_head': config.get('n_heads', 4),    # Default to 4 heads
        'resid_pdrop': 0.1,
        'embd_pdrop': 0.1,
        'attn_pdrop': 0.1,
        'use_cache': False
    }
    
    model = TSPModel(
        hidden_dim=config['hidden_dim'],
        grid_size=grid_size,
        max_context_length=config['max_context_length'],
        gpt2_config=gpt2_config  # Pass the GPT2 config
    )
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, config

def visualize_step(env: TSP_env, logits: torch.Tensor, step: int, save_dir: str):
    """Visualizes model logits for a single step."""
    # Create figure with three subplots side by side
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5))
    
    # Plot environment state in the first subplot
    current_ax = plt.gca()
    current_fig = plt.gcf()
    
    plt.sca(ax1)
    env.visualize()
    ax1.set_aspect('equal')
    
    plt.sca(current_ax)
    ax1.set_title(f'Environment State - Step {step}')
    
    # Reshape logits to grid
    grid_size = int(np.sqrt(logits.size(0)))
    logits_grid = logits.reshape(grid_size, grid_size).cpu().detach().numpy()
    
    # Find global min and max for consistent colorbar
    vmin = logits_grid.min()
    vmax = logits_grid.max()
    
    # Plot unmasked logits
    im2 = ax2.imshow(logits_grid, cmap='viridis', origin='lower', vmin=vmin, vmax=vmax)
    ax2.set_title(f'Model Logits (Unmasked) - Step {step}')
    plt.colorbar(im2, ax=ax2)
    
    # Get valid moves mask and create masked array
    valid_moves = env.valid_moves_mask()
    valid_moves_grid = np.zeros((grid_size, grid_size))
    for move in valid_moves:
        valid_moves_grid[move['x'], move['y']] = 1
    masked_logits = np.ma.masked_where(valid_moves_grid == 0, logits_grid)
    
    # Plot masked logits
    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color='black')
    im3 = ax3.imshow(masked_logits, cmap=cmap, origin='lower', vmin=vmin, vmax=vmax)
    ax3.set_title(f'Model Logits (Masked) - Step {step}')
    plt.colorbar(im3, ax=ax3)
    
    # Set aspect ratio for other plots
    ax2.set_aspect('equal')
    ax3.set_aspect('equal')
    
    # Save and close
    plt.savefig(os.path.join(save_dir, f'step_{step}.png'), bbox_inches='tight', dpi=300)
    plt.close()

def evaluate_trajectory(model: TSPModel, env: TSP_env, save_dir: str):
    """Evaluates model on a single trajectory and visualizes each step."""
    device = next(model.parameters()).device
    os.makedirs(save_dir, exist_ok=True)
    
    # Initial state
    state_tensor, attention_mask = env.format_state()
    coords = state_tensor[:, :2].unsqueeze(0).to(device)  # Add batch dimension
    states = state_tensor[:, 2:].unsqueeze(0).to(device)
    attention_mask = attention_mask.unsqueeze(0).to(device)
    
    # For each step in trajectory
    for step in range(len(env.tour) - 1):
        # Get model predictions
        with torch.no_grad():
            logits = model(coords, states, attention_mask)
            logits = logits.squeeze(0)  # Remove batch dimension
        
        # Visualize current step
        visualize_step(env, logits, step, save_dir)
        
        # Take optimal step
        optimal_next = env.get_optimal_next_step()
        env.step(optimal_next)
        
        # Update state tensors
        state_tensor, attention_mask = env.format_state()
        coords = state_tensor[:, :2].unsqueeze(0).to(device)
        states = state_tensor[:, 2:].unsqueeze(0).to(device)
        attention_mask = attention_mask.unsqueeze(0).to(device)

if __name__ == "__main__":
    # Load model from checkpoint
    checkpoint_path = "runs/10x10_TSP10_n_80000_11-14_23:09/checkpoint_epoch_20.pt"
    model, config = load_model(checkpoint_path)
    
    # Load a single instance
    with open('data/grid_10/dataset_10_10k.jsonl', 'r') as f:
        instance = json.loads(f.readline())
    
    # Create environment
    env = TSP_env(instance, max_context_length=config['max_context_length'])
    
    # Create directory for visualization
    save_dir = 'visualization'
    
    # Run evaluation
    evaluate_trajectory(model, env, save_dir)
    print(f"Visualization saved in {save_dir}")