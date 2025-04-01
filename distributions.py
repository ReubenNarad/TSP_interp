from torch import tensor, zeros, randperm, rand, cos, sin
import torch
from torch.distributions import Uniform
import math

class Uniform:
    """
    Samples points from a uniform distribution.
    """
    def __init__(self, min_x=0.0, max_x=1.0, min_y=0.0, max_y=1.0):
        super().__init__()
        self.min_x = min_x
        self.max_x = max_x
        self.min_y = min_y
        self.max_y = max_y

    def sample(self, size):
        batch_size, num_loc, _ = size
        
        # Sample x and y coordinates separately
        x_coords = torch.rand(batch_size, num_loc) * (self.max_x - self.min_x) + self.min_x
        y_coords = torch.rand(batch_size, num_loc) * (self.max_y - self.min_y) + self.min_y
        
        # Stack them together
        coords = torch.stack([x_coords, y_coords], dim=-1)
        return coords


class DualUniform:
    """
    Samples 50/50 from two uniform distributions defined by their corner coordinates.
    
    Args:
        corner1_min: (x_min, y_min) for first distribution
        corner1_max: (x_max, y_max) for first distribution
        corner2_min: (x_min, y_min) for second distribution
        corner2_max: (x_max, y_max) for second distribution
    """
    def __init__(self, corner1_min=(0.0, 0.0), corner1_max=(0.4, 0.4),
                 corner2_min=(0.6, 0.6), corner2_max=(1.0, 1.0)):
        super().__init__()
        self.corner1_min = tensor(corner1_min)
        self.corner1_max = tensor(corner1_max)
        self.corner2_min = tensor(corner2_min)
        self.corner2_max = tensor(corner2_max)

    def sample(self, size):
        batch_size, num_loc, _ = size
        
        # Create empty tensor for coordinates
        coords = zeros(batch_size, num_loc, 2)
        
        # For each batch
        for i in range(batch_size):
            # Sample half the points from first distribution
            n_first = num_loc // 2
            coords[i, :n_first] = rand(n_first, 2) * (self.corner1_max - self.corner1_min) + self.corner1_min
            
            # Sample remaining points from second distribution
            coords[i, n_first:] = rand(num_loc - n_first, 2) * (self.corner2_max - self.corner2_min) + self.corner2_min
            
            # Shuffle the points
            idx = randperm(num_loc)
            coords[i] = coords[i][idx]

        return coords

class RandomUniform:
    """
    Draws a random number (1-5) -> n uniform distributions, then draws their boundaries from a uniform distribution.
    Each distribution is elongated/skinny with a random orientation.
    """
    def __init__(self, min_loc=0.0, max_loc=1.0, length_factor=5.0, min_clusters=1, max_clusters=5):
        super().__init__()
        self.min_loc = min_loc
        self.max_loc = max_loc
        self.length_factor = length_factor
        self.min_clusters = min_clusters
        self.max_clusters = max_clusters

    def sample(self, size):
        batch_size, num_loc, _ = size
        
        # Get the device from the input size tensor if it's a tensor, otherwise default to CPU
        device = size.device if torch.is_tensor(size) else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create tensors directly on the correct device
        coords = torch.zeros(batch_size, num_loc, 2, device=device)
        
        # Generate all random numbers at once on the device
        num_clusters = torch.randint(self.min_clusters, self.max_clusters + 1, (batch_size,), device=device)
        max_points_per_cluster = (num_loc + num_clusters.max().item() - 1) // num_clusters.max().item()
        
        # Pre-generate all random numbers we'll need
        centers = torch.rand(batch_size, num_clusters, 2, device=device)
        lengths = torch.rand(batch_size, num_clusters, device=device) * 0.7 + 0.4  # Length between 0.4 and 1.1
        widths = lengths / self.length_factor  # Much smaller width
        
        point_idx = 0
        for i in range(batch_size):
            points_per_cluster = num_loc // num_clusters
            
            for c in range(num_clusters):
                cluster_points = points_per_cluster if c < num_clusters - 1 else (num_loc - point_idx)
                
                # Generate points in a rectangle - no rotation
                x_offsets = (torch.rand(cluster_points, device=device) - 0.5) * lengths[i, c]
                y_offsets = (torch.rand(cluster_points, device=device) - 0.5) * widths[i, c]
                
                coords[i, point_idx:point_idx + cluster_points, 0] = centers[i, c, 0] + x_offsets
                coords[i, point_idx:point_idx + cluster_points, 1] = centers[i, c, 1] + y_offsets
                
                point_idx += cluster_points
            point_idx = 0  # Reset for next batch
            
        return coords * (self.max_loc - self.min_loc) + self.min_loc


class FuzzyCircle:
    """
    Samples points in a circular pattern with normally distributed radial noise.
    For each batch, the radius mean and standard deviation are themselves
    sampled from uniform distributions.
    
    Args:
        radius_mean_lower: Lower bound for sampling the mean radius
        radius_mean_upper: Upper bound for sampling the mean radius
        radius_std_lower: Lower bound for sampling the radius standard deviation
        radius_std_upper: Upper bound for sampling the radius standard deviation
        center: (x, y) coordinates of circle center, defaults to (0.5, 0.5)
        random_center: If True, ignores the center parameter and generates a random
                     center for each batch within the unit square [0,1]×[0,1]
        center_x_range: (min_x, max_x) range for random center x-coordinate
        center_y_range: (min_y, max_y) range for random center y-coordinate
    """
    def __init__(self, radius_mean_lower=0.3, radius_mean_upper=0.5, 
                radius_std_lower=0.02, radius_std_upper=0.05, center=(0.5, 0.5),
                random_center=False, center_x_range=(0.2, 0.8), center_y_range=(0.2, 0.8)):
        super().__init__()
        self.radius_mean_lower = radius_mean_lower
        self.radius_mean_upper = radius_mean_upper
        self.radius_std_lower = radius_std_lower
        self.radius_std_upper = radius_std_upper
        self.center = torch.tensor(center)
        self.random_center = random_center
        self.center_x_range = center_x_range
        self.center_y_range = center_y_range

    def sample(self, size):
        batch_size, num_loc, _ = size
        
        # Get the device from the input size tensor if it's a tensor, otherwise default to CPU
        device = size.device if torch.is_tensor(size) else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create empty tensor for coordinates
        coords = torch.zeros(batch_size, num_loc, 2, device=device)
        
        for i in range(batch_size):
            # Sample the radius mean and std for this batch
            radius_mean = torch.empty(1, device=device).uniform_(self.radius_mean_lower, self.radius_mean_upper)
            radius_std = torch.empty(1, device=device).uniform_(self.radius_std_lower, self.radius_std_upper)
            
            # Sample a random center if specified
            if self.random_center:
                center_x = torch.empty(1, device=device).uniform_(self.center_x_range[0], self.center_x_range[1])
                center_y = torch.empty(1, device=device).uniform_(self.center_y_range[0], self.center_y_range[1])
                center = torch.tensor([center_x.item(), center_y.item()], device=device)
            else:
                center = self.center.to(device)
            
            # Sample uniform angles
            angles = 2 * math.pi * torch.rand(num_loc, device=device)
            
            # Using scalar values from the tensors
            radius_mean_val = radius_mean.item()
            radius_std_val = radius_std.item()
            distances = torch.normal(
                mean=radius_mean_val, 
                std=radius_std_val, 
                size=(num_loc,), 
                device=device
            )
            
            # Convert to cartesian coordinates
            coords[i, :, 0] = distances * torch.cos(angles) + center[0]
            coords[i, :, 1] = distances * torch.sin(angles) + center[1]
            
        return coords


class HybridSampler:
    """
    A hybrid sampler that draws from multiple distributions with equal or specified probabilities.
    
    Args:
        distributions: List of distribution objects that implement a sample method
        probabilities: Optional list of probabilities corresponding to each distribution.
                      If None, samples evenly from all distributions.
    """
    def __init__(
        self, 
        distributions=None,
        probabilities=None
    ):
        super().__init__()
        
        # Set default distributions if none provided
        if distributions is None:
            distributions = [
                Uniform()
            ]
        
        self.distributions = distributions
        
        # Validate and normalize probabilities if provided
        if probabilities is not None:
            if len(probabilities) != len(distributions):
                raise ValueError("Length of probabilities must match length of distributions")
            
            # Convert to tensor and normalize
            self.probabilities = torch.tensor(probabilities, dtype=torch.float)
            self.probabilities = self.probabilities / self.probabilities.sum()
        else:
            # Equal probabilities for all distributions
            n_dists = len(distributions)
            self.probabilities = torch.ones(n_dists) / n_dists

    def sample(self, size):
        batch_size, num_loc, _ = size
        
        # Get the device from the input size tensor
        device = size.device if torch.is_tensor(size) else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Move probabilities to the same device
        probabilities = self.probabilities.to(device)
        
        # Create tensor to store results
        coords = torch.zeros(batch_size, num_loc, 2, device=device)
        
        # For each instance in the batch, decide which distribution to use
        for i in range(batch_size):
            # Sample distribution index based on probabilities
            dist_idx = torch.multinomial(probabilities, 1).item()
            
            # Sample from the selected distribution
            coords[i] = self.distributions[dist_idx].sample((1, num_loc, 2))[0]
        
        return coords


# Demonstration with quick plot
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style="whitegrid")


    hybrid_sampler = HybridSampler([
        FuzzyCircle(
            radius_mean_lower=0.3,
            radius_mean_upper=0.4,
            radius_std_lower=0.02,
            radius_std_upper=0.1,
            random_center=True,
            center_x_range=(0.3, 0.7),
            center_y_range=(0.3, 0.7)
        ),
        RandomUniform(
            min_loc=0.0,
            max_loc=1.0,
            length_factor=5.0
        ),
        Uniform(
            min_x=0.0,
            max_x=1.0,
            min_y=0.0,
            max_y=1.0
        )
    ])
    
    # Sample points from the hybrid sampler
    coords = hybrid_sampler.sample((10, 100, 2))
    
    # Create a figure with subplots for each batch
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()
    
    # Plot each batch
    for i in range(10):
        coords_np = coords[i].cpu().numpy()
        axes[i].scatter(coords_np[:, 0], coords_np[:, 1], alpha=0.5, s=10)
        axes[i].set_xlim(-0.1, 1.1)
        axes[i].set_ylim(-0.1, 1.1)
        axes[i].set_title(f"Batch {i+1}")
    
    plt.tight_layout()
    plt.savefig("hybrid_sampler.png")
    plt.close()