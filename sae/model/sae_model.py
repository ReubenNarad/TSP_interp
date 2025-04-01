import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from policy.policy_hooked import HookedAttentionModelPolicy

class TopKSparseAutoencoder(nn.Module):
    """
    Sparse Autoencoder with top-k activation for mechanistic interpretability, 
    following the approach from Gao et al. (OpenAI)
    """
    
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        k: int = None,
        k_ratio: float = 0.05,
        tied_weights: bool = False,
        bias_decay: float = 0.0,
        dict_init: str = "uniform",
    ):
        """
        Initialize the sparse autoencoder.
        
        Args:
            input_dim: Dimension of input activations
            latent_dim: Dimension of the latent space (typically larger than input_dim)
            k: Number of active units to keep in top-k. If None, use k_ratio
            k_ratio: Fraction of units to keep active if k is None
            tied_weights: Whether to tie encoder and decoder weights
            bias_decay: L2 regularization strength on decoder bias
            dict_init: Initialization strategy for dictionary ("normal", "uniform" or "xavier")
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.tied_weights = tied_weights
        self.bias_decay = bias_decay
        
        # Determine k (number of active features) based on k or k_ratio
        self.k = k if k is not None else max(1, int(k_ratio * latent_dim))
        print(f"Using top-{self.k} activation (out of {latent_dim} features)")
        
        # Encoder (dictionary)
        self.encoder = nn.Linear(input_dim, latent_dim, bias=True)
        
        # Decoder (dictionary transpose + bias)
        if tied_weights:
            self.decoder_weight = None  # Will use encoder.weight.t() during forward pass
            self.decoder_bias = nn.Parameter(torch.zeros(input_dim))
        else:
            self.decoder = nn.Linear(latent_dim, input_dim)
        
        # Initialize dictionary
        self._init_dictionary(dict_init)
        
        # For tracking dead neurons
        self.register_buffer("neuron_activity", torch.zeros(latent_dim))
        
    def _init_dictionary(self, init_type: str):
        """Initialize encoder/decoder weights"""
        if init_type == "uniform":
            # Initialize with uniform distribution as in the paper
            nn.init.kaiming_uniform_(self.encoder.weight, nonlinearity='relu')
            if not self.tied_weights:
                nn.init.kaiming_uniform_(self.decoder.weight, nonlinearity='linear')
        elif init_type == "normal":
            # Initialize with normal distribution
            nn.init.kaiming_normal_(self.encoder.weight, nonlinearity='relu')
            if not self.tied_weights:
                nn.init.kaiming_normal_(self.decoder.weight, nonlinearity='linear')
        elif init_type == "xavier":
            # Xavier/Glorot initialization
            nn.init.xavier_uniform_(self.encoder.weight)
            if not self.tied_weights:
                nn.init.xavier_uniform_(self.decoder.weight)
        else:
            raise ValueError(f"Unknown initialization type: {init_type}")
        
        # Initialize biases to zero
        nn.init.zeros_(self.encoder.bias)
        if not self.tied_weights:
            nn.init.zeros_(self.decoder.bias)
            
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input activations to sparse latent representation.
        
        Args:
            x: Input tensor of shape [batch_size, input_dim]
            
        Returns:
            Sparse latent representation of shape [batch_size, latent_dim]
        """
        # Pre-activation
        z = self.encoder(x)
        
        # Apply top-k sparsity
        # Find values and indices of top-k activations for each sample
        values, _ = torch.topk(z, k=self.k, dim=1)
        # Get threshold value (the smallest value in the top-k)
        thresholds = values[:, -1].unsqueeze(1)
        # Apply thresholding: keep only values >= threshold
        z_sparse = F.relu(z - thresholds)
        
        # Update neuron activity statistics
        if self.training:
            self.neuron_activity += (z_sparse > 0).float().sum(dim=0)
            
        return z_sparse
        
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent representation back to input space.
        
        Args:
            z: Latent tensor of shape [batch_size, latent_dim]
            
        Returns:
            Reconstructed tensor of shape [batch_size, input_dim]
        """
        if self.tied_weights:
            # Use transposed encoder weights
            return F.linear(z, self.encoder.weight.t(), self.decoder_bias)
        else:
            return self.decoder(z)
            
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the autoencoder.
        
        Args:
            x: Input tensor of shape [batch_size, input_dim]
            
        Returns:
            Tuple of (reconstructed_input, latent_representation)
        """
        # Encode
        z = self.encode(x)
        
        # Decode
        x_recon = self.decode(z)
        
        return x_recon, z
    
    def get_dead_neurons(self, threshold: float = 1e-5) -> torch.Tensor:
        """Return binary mask of dead neurons (never activate above threshold)"""
        if self.neuron_activity.sum() == 0:
            print("Warning: Neuron activity is not being tracked or no forward passes yet")
            return torch.zeros(self.latent_dim, dtype=torch.bool, device=self.neuron_activity.device)
        return self.neuron_activity < threshold
    
    def reset_dead_neurons(self, reinit_type: str = "uniform"):
        """
        Reinitialize dead neurons.
        
        Args:
            reinit_type: Reinitialization method ("uniform", "normal", "data_based")
        """
        dead_mask = self.get_dead_neurons()
        dead_count = dead_mask.sum().item()
        
        if dead_count == 0:
            return 0
        
        print(f"Reinitializing {dead_count} dead neurons")
        
        if reinit_type == "uniform":
            # Reinitialize with uniform distribution
            nn.init.kaiming_uniform_(self.encoder.weight[dead_mask], nonlinearity='relu')
            if not self.tied_weights:
                nn.init.kaiming_uniform_(self.decoder.weight[:, dead_mask], nonlinearity='linear')
        elif reinit_type == "normal":
            # Reinitialize with normal distribution
            nn.init.kaiming_normal_(self.encoder.weight[dead_mask], nonlinearity='relu')
            if not self.tied_weights:
                nn.init.kaiming_normal_(self.decoder.weight[:, dead_mask], nonlinearity='linear')
        else:
            raise ValueError(f"Unknown reinitialization type: {reinit_type}")
        
        # Reset bias values
        nn.init.zeros_(self.encoder.bias[dead_mask])
        
        # Reset activity counter for reinitialized neurons
        self.neuron_activity[dead_mask] = 0
        
        return dead_count
    
    def get_l0_sparsity(self, z: torch.Tensor) -> float:
        """Calculate average L0 sparsity (number of non-zero features)"""
        return (z > 0).float().sum(dim=1).mean().item()
    
    def get_l1_sparsity(self, z: torch.Tensor) -> float:
        """Calculate average L1 sparsity (sum of absolute values)"""
        return z.abs().sum(dim=1).mean().item()
    
    def get_feature_activations(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get activations of all latent features for the input.
        
        Args:
            x: Input tensor of shape [batch_size, input_dim]
            
        Returns:
            z: Latent activations of shape [batch_size, latent_dim]
        """
        # Set to evaluation mode to ensure consistent behavior
        was_training = self.training
        self.eval()
        
        with torch.no_grad():
            # Get latent activations
            z = self.encode(x)
        
        # Restore previous training state
        if was_training:
            self.train()
            
        return z
    
    def get_feature_importance(self, activation_dataset: torch.Tensor, top_k: int = 10) -> tuple:
        """
        Compute overall importance of each feature based on average activation.
        
        Args:
            activation_dataset: Dataset of activations with shape [batch_size, input_dim]
            top_k: Number of top features to return
            
        Returns:
            Tuple of (indices, values) for top-k features
        """
        self.eval()
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(activation_dataset),
            batch_size=512, shuffle=False
        )
        
        all_activations = []
        with torch.no_grad():
            for batch in loader:
                _, z = self(batch[0])
                all_activations.append(z)
        
        # Concatenate all batches
        all_z = torch.cat(all_activations, dim=0)
        
        # Calculate mean activation per feature
        mean_activations = all_z.mean(dim=0)
        
        # Get top k features
        values, indices = torch.topk(mean_activations, top_k)
        
        return indices.cpu(), values.cpu()
    
    