#!/usr/bin/env python3

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass 
class TrajectoryState:
    """Represents the state at a specific point in a TSP trajectory"""
    current_node: torch.Tensor  # [batch_size] - current node for each instance
    visited_mask: torch.Tensor  # [batch_size, num_nodes] - True for visited nodes
    step: int  # Which step in the trajectory we're at


class NearestNeighborPolicy:
    """
    Implements a greedy nearest neighbor TSP heuristic that can be executed
    from any point in a trajectory, respecting the same constraints as the neural model.
    """
    
    def __init__(self, device=None):
        self.device = device or torch.device("cpu")
    
    def compute_distances(self, instances: Dict) -> torch.Tensor:
        """
        Compute pairwise distances between all nodes.
        
        Args:
            instances: TSP instances containing node locations
            
        Returns:
            Distance matrix [batch_size, num_nodes, num_nodes]
        """
        locs = instances['locs']  # [batch_size, num_nodes, 2]
        batch_size, num_nodes, _ = locs.shape
        
        # Compute pairwise Euclidean distances
        # Expand to [batch_size, num_nodes, 1, 2] and [batch_size, 1, num_nodes, 2]
        locs_i = locs.unsqueeze(2)  # [batch_size, num_nodes, 1, 2]
        locs_j = locs.unsqueeze(1)  # [batch_size, 1, num_nodes, 2]
        
        # Compute squared differences and sum over coordinate dimension
        squared_diffs = (locs_i - locs_j).pow(2).sum(dim=-1)  # [batch_size, num_nodes, num_nodes]
        distances = squared_diffs.sqrt()
        
        return distances
    
    def get_next_node(
        self, 
        instances: Dict, 
        current_node: torch.Tensor, 
        visited_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Select the next node using nearest neighbor heuristic.
        
        Args:
            instances: TSP instances
            current_node: [batch_size] - current node for each instance
            visited_mask: [batch_size, num_nodes] - True for visited nodes
            
        Returns:
            [batch_size] - selected next node for each instance
        """
        distances = self.compute_distances(instances)  # [batch_size, num_nodes, num_nodes]
        batch_size, num_nodes, _ = distances.shape
        
        # Extract distances from current node to all other nodes
        # Use advanced indexing to get distances from current_node to all nodes
        batch_indices = torch.arange(batch_size, device=self.device)
        current_distances = distances[batch_indices, current_node]  # [batch_size, num_nodes]
        
        # Mask out visited nodes by setting their distances to infinity
        masked_distances = current_distances.clone()
        masked_distances[visited_mask] = float('inf')
        
        # Also mask out the current node itself
        masked_distances[batch_indices, current_node] = float('inf')
        
        # Select the nearest unvisited node
        next_node = torch.argmin(masked_distances, dim=1)
        
        return next_node
    
    def get_trajectory_actions(
        self, 
        instances: Dict, 
        start_state: Optional[TrajectoryState] = None
    ) -> Dict:
        """
        Generate a complete trajectory using nearest neighbor heuristic.
        
        Args:
            instances: TSP instances
            start_state: Optional starting state. If None, starts from node 0.
            
        Returns:
            Dictionary with actions, rewards, and trajectory info
        """
        locs = instances['locs']
        batch_size, num_nodes, _ = locs.shape
        
        if start_state is None:
            # Start from depot (node 0)
            current_node = torch.zeros(batch_size, dtype=torch.long, device=self.device)
            visited_mask = torch.zeros(batch_size, num_nodes, dtype=torch.bool, device=self.device)
            visited_mask[:, 0] = True  # Mark depot as visited
            start_step = 0
        else:
            current_node = start_state.current_node
            visited_mask = start_state.visited_mask.clone()
            start_step = start_state.step
        
        # Store the trajectory
        actions = []
        trajectory_nodes = [current_node.clone()]
        
        # Generate trajectory
        for step in range(start_step, num_nodes - 1):
            next_node = self.get_next_node(instances, current_node, visited_mask)
            actions.append(next_node)
            
            # Update state
            current_node = next_node
            visited_mask.scatter_(1, next_node.unsqueeze(1), True)
            trajectory_nodes.append(current_node.clone())
        
        # Complete the tour by returning to depot
        depot = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        actions.append(depot)
        trajectory_nodes.append(depot)
        
        # Convert to tensors
        actions_tensor = torch.stack(actions, dim=1)  # [batch_size, num_steps]
        trajectory_tensor = torch.stack(trajectory_nodes, dim=1)  # [batch_size, num_steps+1]
        
        # Compute tour length (reward is negative tour length)
        tour_length = self._compute_tour_length(instances, trajectory_tensor)
        reward = -tour_length
        
        return {
            'actions': actions_tensor,
            'trajectory': trajectory_tensor,
            'reward': reward,
            'tour_length': tour_length
        }
    
    def get_single_step_action(
        self,
        instances: Dict,
        current_node: torch.Tensor,
        visited_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Get the action that nearest neighbor policy would take at a single step.
        
        Args:
            instances: TSP instances
            current_node: [batch_size] - current node
            visited_mask: [batch_size, num_nodes] - visited nodes
            
        Returns:
            [batch_size] - next node that NN policy would select
        """
        return self.get_next_node(instances, current_node, visited_mask)
    
    def _compute_tour_length(self, instances: Dict, trajectory: torch.Tensor) -> torch.Tensor:
        """
        Compute the total length of tours.
        
        Args:
            instances: TSP instances
            trajectory: [batch_size, num_steps] - sequence of visited nodes
            
        Returns:
            [batch_size] - tour length for each instance
        """
        locs = instances['locs']  # [batch_size, num_nodes, 2]
        batch_size, num_steps = trajectory.shape
        
        # Get coordinates of trajectory nodes
        batch_indices = torch.arange(batch_size, device=self.device).unsqueeze(1)
        trajectory_coords = locs[batch_indices, trajectory]  # [batch_size, num_steps, 2]
        
        # Compute distances between consecutive nodes
        coord_diffs = trajectory_coords[:, 1:] - trajectory_coords[:, :-1]  # [batch_size, num_steps-1, 2]
        step_distances = coord_diffs.pow(2).sum(dim=-1).sqrt()  # [batch_size, num_steps-1]
        
        # Sum to get total tour length
        tour_length = step_distances.sum(dim=1)  # [batch_size]
        
        return tour_length


class NearestNeighborComparator:
    """
    Extends behavior comparison to include nearest neighbor policy comparisons.
    """
    
    def __init__(self, nn_policy: NearestNeighborPolicy):
        self.nn_policy = nn_policy
    
    def compare_with_nearest_neighbor(
        self,
        instances: Dict,
        neural_actions: torch.Tensor,
        trajectory_state: TrajectoryState
    ) -> Dict:
        """
        Compare neural model actions with nearest neighbor policy at specific trajectory points.
        
        Args:
            instances: TSP instances
            neural_actions: [batch_size] - actions chosen by neural model
            trajectory_state: Current state in trajectory
            
        Returns:
            Dictionary with comparison metrics
        """
        # Get what nearest neighbor policy would do at this state
        nn_actions = self.nn_policy.get_single_step_action(
            instances, 
            trajectory_state.current_node, 
            trajectory_state.visited_mask
        )
        
        # Compute agreement metrics
        agreement = (neural_actions == nn_actions).float()
        agreement_rate = agreement.mean().item()
        
        # Compute distance-based metrics
        distances = self.nn_policy.compute_distances(instances)
        batch_size = neural_actions.shape[0]
        batch_indices = torch.arange(batch_size, device=neural_actions.device)
        
        # Distance to neural model's choice
        neural_distances = distances[batch_indices, trajectory_state.current_node, neural_actions]
        
        # Distance to NN policy's choice  
        nn_distances = distances[batch_indices, trajectory_state.current_node, nn_actions]
        
        return {
            'agreement_rate': agreement_rate,
            'agreements': agreement,
            'neural_actions': neural_actions,
            'nn_actions': nn_actions,
            'neural_distances': neural_distances,
            'nn_distances': nn_distances,
            'distance_ratio': neural_distances / (nn_distances + 1e-8)  # How much farther neural choice is
        }
    
    def extract_trajectory_state(
        self, 
        instances: Dict, 
        actions: torch.Tensor, 
        step: int
    ) -> TrajectoryState:
        """
        Extract trajectory state at a specific step from a sequence of actions.
        
        Args:
            instances: TSP instances
            actions: [batch_size, num_steps] - full action sequence
            step: Which step to extract state for
            
        Returns:
            TrajectoryState at the specified step
        """
        batch_size, num_nodes = instances['locs'].shape[:2]
        
        if step == 0:
            # Initial state - starting from depot
            current_node = torch.zeros(batch_size, dtype=torch.long, device=actions.device)
            visited_mask = torch.zeros(batch_size, num_nodes, dtype=torch.bool, device=actions.device)
            visited_mask[:, 0] = True
        else:
            # Reconstruct state from actions
            current_node = actions[:, step - 1]  # Previous action is current node
            visited_mask = torch.zeros(batch_size, num_nodes, dtype=torch.bool, device=actions.device)
            
            # Mark depot as visited
            visited_mask[:, 0] = True
            
            # Mark all previously visited nodes
            for s in range(step):
                visited_mask.scatter_(1, actions[:, s].unsqueeze(1), True)
        
        return TrajectoryState(
            current_node=current_node,
            visited_mask=visited_mask,
            step=step
        ) 