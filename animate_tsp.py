import torch
import matplotlib.pyplot as plt
import json
import numpy as np
import os
from matplotlib.animation import PillowWriter
from typing import List, Dict, Any
from matplotlib.patches import Patch

from net import TSPModel
from TSP_env import TSP_env
from eval import load_model

# Explicitly list checkpoints to visualize
CHECKPOINTS = [f"runs/20x20_TSP20_n_80000_11-16_16:47/checkpoint_epoch_{epoch}.pt" for epoch in range(1, 30)]
DATASET = 'data/grid_20/dataset_20_tiny_rotated.jsonl'
RUN_NAME = CHECKPOINTS[0].split('/')[-2]
STEP = 14
FPS = 6

def create_logits_evolution_gif(checkpoint_paths: List[str], 
                                env: TSP_env,
                                trajectory_step: int,
                                save_path: str = f'visualization/{RUN_NAME}_{STEP}.gif',
                                fps: int = FPS):
    """
    Creates an animation showing how model logits evolve across different checkpoints
    for a specific environment state and trajectory step.
    
    Args:
        checkpoint_paths: List of paths to model checkpoints
        env: The environment to evaluate
        trajectory_step: Which step in the trajectory to visualize
        save_path: Where to save the output GIF
        fps: Frames per second in the output GIF
    """
    
    # Validate trajectory step
    if trajectory_step >= len(env.tour) - 1:
        raise ValueError(f"Trajectory step {trajectory_step} is too large. Maximum allowed step is {len(env.tour) - 2}")
    
    # Advance environment to desired step
    for _ in range(trajectory_step):
        next_node_id = env.tour[_ + 1]
        next_pos = env.get_node_position(next_node_id)
        env.step(next_pos)
    
    # **Step 1: Determine Global vmin and vmax across all checkpoints**
    global_vmin = float('inf')
    global_vmax = float('-inf')
    
    print("Determining global vmin and vmax across all checkpoints...")
    for checkpoint_path in checkpoint_paths:
        model, config = load_model(checkpoint_path)
        device = next(model.parameters()).device
        
        state_tensor, attention_mask = env.format_state()
        coords = state_tensor[:, :2].unsqueeze(0).to(device)
        states = state_tensor[:, 2:].unsqueeze(0).to(device)
        attention_mask = attention_mask.unsqueeze(0).to(device)
        
        with torch.no_grad():
            logits = model(coords, states, attention_mask)
            logits = logits.squeeze(0)
        
        # Process logits
        grid_size = int(np.sqrt(logits.size(0)))
        logits_grid = logits.reshape(grid_size, grid_size).cpu().detach().numpy()
        
        # Update global min and max
        current_min = logits_grid.min()
        current_max = logits_grid.max()
        if current_min < global_vmin:
            global_vmin = current_min
        if current_max > global_vmax:
            global_vmax = current_max
        
        # Cleanup
        del model
        torch.cuda.empty_cache()
    
    print(f"Global vmin: {global_vmin}, Global vmax: {global_vmax}")
    
    # **Step 2: Setup the figure with fixed colorbar limits**
    fig = plt.figure(figsize=(20, 5), constrained_layout=True)
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.1])  # Add space for colorbar
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])
    cbar_ax = fig.add_subplot(gs[3])  # Create colorbar axis once
    
    # Define color mappings
    color_map = {
        'Current & Depot': [1, 0, 1],     # Purple
        'Current': [1, 0, 0],             # Red
        'Depot': [1, 1, 0],               # Yellow
        'Unvisited': [0.5, 0.5, 0.5],     # Gray
        'Optimal Next': [0, 0, 1],        # Blue
    }
    
    # Create legend elements once
    legend_elements = [
        Patch(facecolor=color_map[label], edgecolor='k', label=label)
        for label in color_map
    ]
    
    # Create the legend once
    fig.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.02, 0.5))
    
    # **Step 3: Initialize the colorbar with global_vmin and global_vmax**
    # Temporarily plot an empty image to initialize the colorbar
    temp_im = ax2.imshow(np.zeros((1,1)), cmap='viridis', vmin=global_vmin, vmax=global_vmax)
    colorbar = fig.colorbar(temp_im, cax=cbar_ax)
    colorbar.set_label('Logits')
    
    # Setup the figure for animation
    writer = PillowWriter(fps=fps)
    
    # Begin saving frames
    with writer.saving(fig, save_path, dpi=100):
        for checkpoint_path in checkpoint_paths:
            print(f"Processing checkpoint: {checkpoint_path}")
            # Clear previous plots
            ax1.clear()
            ax2.clear()
            ax3.clear()
            
            # Re-create the legend (optional if necessary)
            fig.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.02, 0.5))
            
            # Create environment visualization grid with black background
            grid = np.zeros((env.grid_size, env.grid_size, 3))  # Black background instead of white
            
            # **Exclude Visited Nodes by Skipping Them**
            for node in env.coords:
                x, y = node['x'], node['y']
                if env.state[x, y] == env.VISITED:
                    continue  # Skip Visited nodes; they remain as background
                grid[x, y] = color_map['Unvisited']
            
            # Get optimal next node
            optimal_next = env.get_optimal_next_step()
            
            # Update grid based on state for non-Visited nodes
            for node in env.coords:
                x, y = node['x'], node['y']
                if env.state[x, y] == env.VISITED:
                    continue  # Skip Visited nodes
                if x == env.current_pos['x'] and y == env.current_pos['y'] and env.state[x, y] == env.DEPOT:
                    grid[x, y] = color_map['Current & Depot']
                elif x == env.current_pos['x'] and y == env.current_pos['y']:
                    grid[x, y] = color_map['Current']
                elif env.state[x, y] == env.DEPOT:
                    grid[x, y] = color_map['Depot']
                elif x == optimal_next['x'] and y == optimal_next['y']:
                    grid[x, y] = color_map['Optimal Next']
                else:
                    grid[x, y] = color_map['Unvisited']
            
            # Plot environment state without Visited nodes
            ax1.imshow(grid, origin='lower')
            
            # Draw lines for optimal route
            for i in range(len(env.tour)):
                # Get current and next node positions
                current_node_id = env.tour[i]
                next_node_id = env.tour[(i + 1) % len(env.tour)]
                
                current_pos = env.get_node_position(current_node_id)
                next_pos = env.get_node_position(next_node_id)
                
                # Draw line between nodes
                ax1.plot([current_pos['y'], next_pos['y']], 
                        [current_pos['x'], next_pos['x']], 
                        'w-', alpha=0.3, linewidth=1)  # white lines with some transparency
        
            ax1.set_title(f'Input State (Step {trajectory_step})')
            ax1.axis('off')
            
            # Get model predictions
            model, config = load_model(checkpoint_path)
            epoch = int(os.path.basename(checkpoint_path).split('_')[-1].split('.')[0])
            device = next(model.parameters()).device
            
            state_tensor, attention_mask = env.format_state()
            coords = state_tensor[:, :2].unsqueeze(0).to(device)
            states = state_tensor[:, 2:].unsqueeze(0).to(device)
            attention_mask = attention_mask.unsqueeze(0).to(device)
            
            with torch.no_grad():
                logits = model(coords, states, attention_mask)
                logits = logits.squeeze(0)
            
            # Process logits
            grid_size = int(np.sqrt(logits.size(0)))
            logits_grid = logits.reshape(grid_size, grid_size).cpu().detach().numpy()
            
            # **Use global_vmin and global_vmax instead of frame-specific vmin and vmax**
            
            # Plot unmasked logits
            im2 = ax2.imshow(logits_grid, cmap='viridis', origin='lower', vmin=global_vmin, vmax=global_vmax)
            ax2.set_title(f'Model Logits (Unmasked) - Epoch {epoch}')
            ax2.axis('off')
            
            # Create and plot masked logits
            valid_moves = env.valid_moves_mask()
            valid_moves_grid = np.zeros((grid_size, grid_size))
            for move in valid_moves:
                valid_moves_grid[move['x'], move['y']] = 1
            masked_logits = np.ma.masked_where(valid_moves_grid == 0, logits_grid)
            
            cmap = plt.cm.viridis.copy()
            cmap.set_bad(color='black')
            im3 = ax3.imshow(masked_logits, cmap=cmap, origin='lower', vmin=global_vmin, vmax=global_vmax)
            ax3.set_title(f'Model Logits (Masked) - Epoch {epoch}')
            ax3.axis('off')
            
            # **Update the images without altering the colorbar**
            # Since colorbar is already initialized, no need to update it
            
            # Set aspect ratio for all plots
            ax1.set_aspect('equal')
            ax2.set_aspect('equal')
            ax3.set_aspect('equal')
            
            # Capture the frame
            writer.grab_frame()
            
            # Cleanup
            del model
            torch.cuda.empty_cache()
    
    plt.close()
    env.reset()

if __name__ == "__main__":
    # Load a single instance
    with open(DATASET, 'r') as f:
        instance = json.loads(f.readline())
    
    # Create environment (using config from first checkpoint for max_context_length)
    first_checkpoint = torch.load(CHECKPOINTS[0], map_location='cpu')
    max_context_length = first_checkpoint['config']['max_context_length']
    env = TSP_env(instance, max_context_length=max_context_length)
    
    # Print tour length and create animation with valid step
    print(f"Tour length: {len(env.tour)}")
    step = min(STEP, len(env.tour) - 2)  # Use step 5 or last valid step if tour is shorter
    print(f"Creating animation for step {step}")
    create_logits_evolution_gif(CHECKPOINTS, env, step)
