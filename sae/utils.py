import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Optional, Tuple, Union

def visualize_activation_statistics(activations: torch.Tensor, title: str = "Activation Statistics"):
    """Visualize basic statistics about activations"""
    act_mean = activations.mean(dim=0)
    act_std = activations.std(dim=0)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot histogram of activation means
    ax1.hist(act_mean.cpu().numpy(), bins=50)
    ax1.set_title(f"Distribution of Activation Means")
    ax1.set_xlabel("Mean Activation Value")
    ax1.set_ylabel("Count")
    
    # Plot histogram of activation standard deviations
    ax2.hist(act_std.cpu().numpy(), bins=50)
    ax2.set_title(f"Distribution of Activation Std Devs")
    ax2.set_xlabel("Std Dev of Activation")
    ax2.set_ylabel("Count")
    
    plt.suptitle(title)
    plt.tight_layout()
    
    return fig

def get_l0_sparsity(activations: torch.Tensor) -> float:
    """Calculate L0 sparsity (average number of non-zero features)"""
    return (activations > 0).float().sum(dim=1).mean().item()

def get_dead_features(activations: torch.Tensor) -> float:
    """Calculate fraction of dead features (never activate)"""
    return (activations.sum(dim=0) == 0).float().mean().item() 