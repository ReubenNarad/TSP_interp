import torch
from typing import Dict, List, Callable, Optional
from rl4co.models.zoo.am.policy import AttentionModelPolicy

class HookedAttentionModelPolicy(AttentionModelPolicy):
    """
    Extension of AttentionModelPolicy that allows registering hooks for mechanistic interpretability.
    Adds functionality to capture intermediate activations from the encoder layers.
    """
    def __init__(self, *args, **kwargs):
        # Pass all kwargs to the parent class (including any dropout parameters)
        super().__init__(*args, **kwargs)
        self.hooks = {}
        self.activation_cache = {}
        self._setup_hooks()
    
    def _setup_hooks(self):
        """Setup the hooks for each encoder layer"""
        for layer_idx, layer in enumerate(self.encoder.net.layers):
            # Register hook for the output of each encoder layer
            def get_hook(layer_idx):
                def hook(module, input, output):
                    self.activation_cache[f'encoder_layer_{layer_idx}'] = output
                return hook
            
            # Register the hook on the MultiHeadAttentionLayer
            handle = layer.register_forward_hook(get_hook(layer_idx))
            self.hooks[f'encoder_layer_{layer_idx}'] = handle
    
    def clear_hooks(self):
        """Remove all registered hooks"""
        for handle in self.hooks.values():
            handle.remove()
        self.hooks.clear()
    
    def clear_cache(self):
        """Clear the activation cache"""
        self.activation_cache.clear()
    
    def get_activation(self, name: str) -> Optional[torch.Tensor]:
        """
        Retrieve activation from cache by name
        
        Args:
            name: Name of the activation to retrieve (e.g., 'encoder_layer_0')
        
        Returns:
            torch.Tensor or None if activation not found
        """
        return self.activation_cache.get(name)
    
    def forward(self, *args, **kwargs):
        """
        Forward pass that automatically clears the activation cache before each run
        """
        self.clear_cache()
        return super().forward(*args, **kwargs)
