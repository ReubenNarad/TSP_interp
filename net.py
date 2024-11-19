# TSP Solver Network Implementation

import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2Model
from typing import List
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.animation import FuncAnimation, PillowWriter

class TSPEmbedding(nn.Module):
    """
    Embeds node features into the model's hidden dimension.
    
    Inputs:
        coords: Tensor of shape [batch_size, max_nodes, 2]
        states: Tensor of shape [batch_size, max_nodes, 2]
    
    Output:
        embeddings: Tensor of shape [batch_size, max_nodes, hidden_dim]
    """
    def __init__(self, hidden_dim: int = 768):
        super(TSPEmbedding, self).__init__()
        self.hidden_dim = hidden_dim
        
        # Split hidden dim into components
        self.coord_dim = hidden_dim // 4    # For (x,y) coordinates
        self.state_dim = hidden_dim // 4    # For binary state flags (current, depot)
        self.dist_dim = hidden_dim // 2     # For distance feature
        
        # Embedding layers
        self.coord_embedding = nn.Linear(2, self.coord_dim)
        self.state_embedding = nn.Linear(2, self.state_dim)  # For is_current and is_depot
        self.dist_embedding = nn.Linear(1, self.dist_dim)   # For distance to current
        self.dropout = nn.Dropout(p=0.1)
    
    def forward(self, coords: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for TSPEmbedding.
        
        Args:
            coords (torch.Tensor): [batch_size, max_nodes, 2]
            states (torch.Tensor): [batch_size, max_nodes, 2]
        
        Returns:
            torch.Tensor: [batch_size, max_nodes, hidden_dim]
        """
        # Split input tensor
        coord_input = coords  # Already [batch, nodes, 2]
        state_input = states[:, :, :2]  # [batch, nodes, 2] - current & depot flags
        dist_input = states[:, :, 2:3]  # [batch, nodes, 1] - distance feature
        
        # Individual embeddings
        coord_embeds = self.coord_embedding(coord_input)
        state_embeds = self.state_embedding(state_input)
        dist_embeds = self.dist_embedding(dist_input)
        
        # Combine embeddings
        embeddings = torch.cat([coord_embeds, state_embeds, dist_embeds], dim=-1)
        return self.dropout(embeddings)

class TSPUnembedding(nn.Module):
    """
    Projects hidden states to 100x100 grid logits.
    
    Input:
        hidden_states: Tensor of shape [batch_size, hidden_dim]
    
    Output:
        logits: Tensor of shape [batch_size, grid_size * grid_size]
    """
    def __init__(self, hidden_dim: int = 32, grid_size: int = 10):
        super(TSPUnembedding, self).__init__()
        self.hidden_dim = hidden_dim
        self.grid_size = grid_size
        self.output_dim = grid_size * grid_size
        self.unembed = nn.Linear(hidden_dim, self.output_dim)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for TSPUnembedding.
        
        Args:
            hidden_states (torch.Tensor): [batch_size, hidden_dim]
        
        Returns:
            torch.Tensor: [batch_size, grid_size * grid_size]
        """
        logits = self.unembed(hidden_states)  # [batch_size, grid_size * grid_size]
        return logits

class TSPModel(nn.Module):
    """
    Complete TSP Solver Model integrating Embedding, GPT2 Transformer, and Unembedding layers.
    
    Inputs:
        coords: Tensor of shape [batch_size, max_nodes, 2]
        states: Tensor of shape [batch_size, max_nodes, 2]
        attention_mask: Tensor of shape [batch_size, max_nodes]
    
    Output:
        logits: Tensor of shape [batch_size, grid_size * grid_size] - logits for next position prediction
    """
    def __init__(self, hidden_dim: int = 768, grid_size: int = 100, max_context_length: int = 50, gpt2_config: dict = None):
        super(TSPModel, self).__init__()
        self.embedding = TSPEmbedding(hidden_dim=hidden_dim)
        self.unembedding = TSPUnembedding(hidden_dim=hidden_dim, grid_size=grid_size)

        if gpt2_config is None:
            # Define default GPT2 configuration
            gpt2_config = {
                'n_positions': max_context_length + 10,
                'n_embd': hidden_dim,
                'n_layer': 12,  # Number of transformer layers
                'n_head': 12,   # Number of attention heads
                'resid_pdrop': 0.1,
                'embd_pdrop': 0.1,
                'attn_pdrop': 0.1,
                'use_cache': False,
            }

        config = GPT2Config(
            n_positions=gpt2_config['n_positions'],
            n_embd=gpt2_config['n_embd'],
            n_layer=gpt2_config['n_layer'],
            n_head=gpt2_config['n_head'],
            resid_pdrop=gpt2_config['resid_pdrop'],
            embd_pdrop=gpt2_config['embd_pdrop'],
            attn_pdrop=gpt2_config['attn_pdrop'],
            use_cache=gpt2_config['use_cache'],
        )
        self.transformer = GPT2Model(config)

    def prepare_batch_from_envs(self, envs: List['TSP_env']) -> dict:
        """
        Prepares a batch of data from a list of TSP_env instances.
        
        Args:
            envs (List[TSP_env]): List of TSP_env instances to be batched.
        
        Returns:
            dict: A dictionary containing:
                - 'coords': [batch_size, max_nodes, 2]
                - 'states': [batch_size, max_nodes, 2]
                - 'attention_mask': [batch_size, max_nodes]
        """
        batch_size = len(envs)
        max_nodes = self.max_context_length
        
        coords_list = []
        states_list = []
        attention_masks = []
        
        for env in envs:
            state_tensor = env.format_state()  # [max_context_length, 4]
            
            # Split into coords and states
            coords = state_tensor[:, :2]  # [max_context_length, 2]
            states = state_tensor[:, 2:4]  # [max_context_length, 2]
            
            coords_list.append(coords)
            states_list.append(states)
            attention_masks.append(torch.ones(max_nodes, dtype=torch.long))  # All are valid in fixed context
        
        # Stack into batched tensors
        coords = torch.stack(coords_list)  # [batch_size, max_nodes, 2]
        states = torch.stack(states_list)  # [batch_size, max_nodes, 2]
        attention_mask = torch.stack(attention_masks)  # [batch_size, max_nodes]
        
        return {
            'coords': coords,
            'states': states,
            'attention_mask': attention_mask
        }

    def forward(self, coords: torch.Tensor, states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the TSPModel.
        
        Args:
            coords (torch.Tensor): [batch_size, max_nodes, 2]
            states (torch.Tensor): [batch_size, max_nodes, 2]
            attention_mask (torch.Tensor): [batch_size, max_nodes]
        
        Returns:
            torch.Tensor: [batch_size, grid_size * grid_size] - logits for next position prediction
        """
            # Embedding
        embeddings = self.embedding(coords, states)  # [batch_size, max_nodes, hidden_dim]
        
        # GPT2 transformer
        transformer_outputs = self.transformer(
            inputs_embeds=embeddings,
            attention_mask=attention_mask
        )
        hidden_states = transformer_outputs.last_hidden_state  # [batch_size, max_nodes, hidden_dim]
        
        # Get only the current position's hidden state (last non-masked position)
        current_mask = attention_mask.sum(dim=1) - 1  # [batch_size]
        batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
        current_hidden = hidden_states[batch_indices, current_mask]  # [batch_size, hidden_dim]
        
        # Unembedding only for current position
        logits = self.unembedding(current_hidden)  # [batch_size, grid_size * grid_size]
        
        return logits

if __name__ == "__main__":
    # Test the TSPModel with fixed max_context_length
    import json
    import torch
    from TSP_env import TSP_env

    # Initialize the model
    print("Initializing model...")
    fixed_max_context_length = 50  # Example value
    model = TSPModel(hidden_dim=768, grid_size=100, max_context_length=fixed_max_context_length)
    print("Model initialized.")

    # Load a sample instance
    with open('dataset_20_sample.jsonl', 'r') as f:
        instances = [json.loads(line) for line in f.readlines()]
    
    # Create batch of environments with fixed max_context_length
    envs = [TSP_env(instance, max_context_length=fixed_max_context_length) for instance in instances]
    batch = model.prepare_batch_from_envs(envs)
    
    # Get the legal moves grid for visualization (example for first env)
    legal_moves = envs[0].valid_moves_mask()
    legal_moves_grid = torch.zeros((100, 100))
    for move in legal_moves:
        legal_moves_grid[move['x'], move['y']] = 1

    # Forward pass
    logits = model(batch['coords'], batch['states'], batch['attention_mask'])

    # Visualize the logits after masking with invalid moves as black
    # Reshape logits to match legal_moves_grid dimensions
    logits_reshaped = logits[0].detach().reshape(100, 100).numpy()
    legal_moves_grid_np = legal_moves_grid.numpy()

    # Create a masked array where legal_moves_grid is 0
    masked_logits = np.ma.masked_where(legal_moves_grid_np == 0, logits_reshaped)

    # Create a colormap and set the 'bad' (masked) values to black
    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color='black')

    # Plot the masked logits
    plt.figure()
    plt.imshow(masked_logits, cmap=cmap, origin='lower')
    plt.colorbar()
    plt.title("Logits Visualization with Legal Moves Mask")
    plt.show()

    # Create animation of steps
    fig, ax = plt.subplots(figsize=(10, 8))
