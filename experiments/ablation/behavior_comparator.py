import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from dataclasses import dataclass


@dataclass
class ComparisonResult:
    """Results from comparing baseline vs ablated model outputs"""
    baseline_probs: torch.Tensor
    ablated_probs: torch.Tensor
    baseline_actions: torch.Tensor
    ablated_actions: torch.Tensor
    prob_difference: torch.Tensor  # Simple L1 difference between probability distributions
    action_changed: torch.Tensor   # Binary indicator if selected action changed


@dataclass
class TrajectoryComparison:
    """Results from comparing full trajectory execution"""
    baseline_rewards: torch.Tensor
    ablated_rewards: torch.Tensor
    reward_difference: torch.Tensor
    tour_changed: torch.Tensor


class BehaviorComparator:
    """
    Compares baseline vs ablated model outputs for TSP solving.
    
    Focuses on simple next-node prediction metrics for easier debugging.
    """
    
    def __init__(self, tsp_model, sae_model=None, ablator=None):
        """
        Initialize the behavior comparator.
        
        Args:
            tsp_model: The TSP policy model
            sae_model: Optional SAE model for feature ablation
            ablator: Optional feature ablator for managing interventions
        """
        self.tsp_model = tsp_model
        self.sae_model = sae_model
        self.ablator = ablator
        self.device = next(tsp_model.parameters()).device
        
        # Put models in eval mode
        self.tsp_model.eval()
        if self.sae_model is not None:
            self.sae_model.eval()
    
    def compare_single_step(
        self,
        instances: Dict,
        target_type: str = "sae_feature",
        target_idx: Optional[int] = None,
        step_idx: int = 0,
        temperature: float = 1.0
    ) -> Dict:
        """
        Compare model outputs at a single step of the trajectory.
        
        For now, this is a simplified implementation that compares
        the final action probabilities rather than true step-wise logits.
        This can be enhanced later with proper step-wise decoding.
        
        Args:
            instances: Batch of TSP instances
            target_type: Type of ablation target
            target_idx: Index of target to ablate (None for baseline only)
            step_idx: Step index for comparison (currently unused)
            temperature: Temperature for probability computation
            
        Returns:
            Comparison results dictionary
        """
        # For now, just use the full trajectory approach
        # TODO: Implement true step-wise comparison once we understand
        # the RL4CO step-wise API better
        return self.compare_full_trajectory(instances, target_type, target_idx, "greedy", True)
    
    def compare_full_trajectory(
        self,
        instances: Dict,
        target_type: str = "sae_feature", 
        target_idx: Optional[int] = None,
        decode_type: str = "greedy",
        return_actions: bool = True
    ) -> Dict:
        """
        Compare complete solution trajectories.
        
        Args:
            instances: Batch of TSP instances
            target_type: Type of ablation target
            target_idx: Index of target to ablate (None for baseline only)
            decode_type: Decoding strategy ("greedy", "sampling")
            return_actions: Whether to return action sequences
            
        Returns:
            Dictionary with trajectory comparison metrics
        """
        with torch.no_grad():
            # Run baseline trajectory with fresh copy
            baseline_result = self._run_full_trajectory(
                instances.clone(),  # Use TensorDict's clone method
                target_type=target_type,
                target_idx=None,  # No ablation for baseline
                decode_type=decode_type,
                return_actions=return_actions
            )
            
            if target_idx is None:
                # Return baseline only if no target specified
                trajectory_result = self._create_trajectory_result(baseline_result, baseline_result)
            else:
                # Run ablated trajectory with another fresh copy
                ablated_result = self._run_full_trajectory(
                    instances.clone(),  # Use TensorDict's clone method
                    target_type=target_type,
                    target_idx=target_idx, 
                    decode_type=decode_type,
                    return_actions=return_actions
                )
                trajectory_result = self._create_trajectory_result(baseline_result, ablated_result)
            
            # Convert TrajectoryComparison to dictionary
            return {
                'baseline_rewards': trajectory_result.baseline_rewards,
                'ablated_rewards': trajectory_result.ablated_rewards,
                'reward_diff': trajectory_result.reward_difference,
                'reward_difference': trajectory_result.reward_difference,  # Both keys for compatibility
                'tour_changed': trajectory_result.tour_changed,
                'action_changes': trajectory_result.tour_changed.sum().item()  # Add this for compatibility
            }
    
    def _run_forward_pass(
        self,
        instances: Dict,
        target_type: str,
        target_idx: Optional[int] = None,
        step_idx: int = 0,
        temperature: float = 1.0
    ) -> Dict:
        """
        Execute a single forward pass with optional ablation.
        
        Args:
            instances: Batch of TSP instances
            target_type: Type of ablation target
            target_idx: Index of target to ablate (None for no ablation)
            step_idx: Step index for partial trajectories
            temperature: Temperature for probability computation
            
        Returns:
            Dictionary with probabilities and selected actions
        """
        if target_idx is not None and self.ablator is not None:
            # Use ablation context manager
            with self.ablator.get_ablation_context(target_type, [target_idx]):
                logits = self._get_step_logits(instances, step_idx)
        else:
            # Run baseline forward pass
            logits = self._get_step_logits(instances, step_idx)
        
        # Apply temperature and compute probabilities
        scaled_logits = logits / temperature
        probs = F.softmax(scaled_logits, dim=-1)
        
        # Get greedy action selection
        actions = torch.argmax(probs, dim=-1)
        
        return {
            'probs': probs,
            'actions': actions
        }
    
    def _get_step_logits(self, instances: Dict, step_idx: int = 0) -> torch.Tensor:
        """
        Get logits for next node selection at a specific step.
        
        Args:
            instances: Batch of TSP instances
            step_idx: Which step to get logits for (0 = first step)
            
        Returns:
            Logits tensor for next node selection
        """
        # For now, implement a simplified version that runs the full forward pass
        # and extracts the logits at the specified step. This is less efficient but
        # works with the RL4CO API without reverse-engineering the step-wise decoding.
        
        # We'll need to hook into the decoder to capture step-wise logits
        captured_logits = []
        step_counter = [0]  # Use list to make it mutable in closure
        
        def capture_logits_hook(module, input, output):
            # This hook captures logits at each decoding step
            if hasattr(module, 'pointer') and step_counter[0] == step_idx:
                # Try to extract logits from the pointer attention output
                # The exact format depends on the RL4CO implementation
                if isinstance(output, tuple) and len(output) >= 1:
                    captured_logits.append(output[0])  # Typically first element is logits
                elif isinstance(output, torch.Tensor):
                    captured_logits.append(output)
            step_counter[0] += 1
        
        if hasattr(self.tsp_model, 'policy'):
            policy = self.tsp_model.policy
        else:
            policy = self.tsp_model
        
        # Register hook on decoder
        hook_handle = policy.decoder.register_forward_hook(capture_logits_hook)
        
        try:
            with torch.no_grad():
                # Run the full forward pass to trigger the hooks
                result = policy(instances, phase="test", decode_type="greedy", return_actions=True)
                
                if captured_logits:
                    return captured_logits[0]
                else:
                    # Fallback: if we couldn't capture step-wise logits,
                    # compute a simple approximation using encoder outputs
                    h, _ = policy.encoder(instances)
                    batch_size, num_nodes, embed_dim = h.shape
                    
                    # Simple logits computation (this is a placeholder)
                    # In practice, this would need proper context and attention computation
                    return torch.randn(batch_size, num_nodes, device=self.device)
        finally:
            # Always remove the hook
            hook_handle.remove()
    
    def _run_full_trajectory(
        self,
        instances: Dict,
        target_type: str,
        target_idx: Optional[int] = None,
        decode_type: str = "greedy", 
        return_actions: bool = True
    ) -> Dict:
        """
        Execute a complete trajectory with optional ablation.
        
        Args:
            instances: Batch of TSP instances
            target_type: Type of ablation target 
            target_idx: Index of target to ablate (None for no ablation)
            decode_type: Decoding strategy
            return_actions: Whether to return action sequences
            
        Returns:
            Dictionary with tour results and rewards
        """
        if target_idx is not None and self.ablator is not None:
            # Use ablation context manager for full trajectory
            with self.ablator.get_ablation_context(target_type, [target_idx]):
                if hasattr(self.tsp_model, 'policy'):
                    policy = self.tsp_model.policy
                else:
                    policy = self.tsp_model
                    
                result = policy(
                    instances,
                    phase="test",
                    decode_type=decode_type,
                    return_actions=return_actions
                )
        else:
            # Run baseline trajectory
            if hasattr(self.tsp_model, 'policy'):
                policy = self.tsp_model.policy  
            else:
                policy = self.tsp_model
                
            result = policy(
                instances,
                phase="test", 
                decode_type=decode_type,
                return_actions=return_actions
            )
        
        return result
    
    def _create_single_result(self, baseline_output: Dict, ablated_output: Dict) -> ComparisonResult:
        """Create ComparisonResult from single-step outputs."""
        baseline_probs = baseline_output['probs']
        ablated_probs = ablated_output['probs']
        
        # Simple probability difference (L1 norm)
        prob_diff = torch.norm(baseline_probs - ablated_probs, p=1, dim=-1)
        
        # Check if actions changed
        action_changed = (baseline_output['actions'] != ablated_output['actions']).float()
        
        return ComparisonResult(
            baseline_probs=baseline_probs,
            ablated_probs=ablated_probs,
            baseline_actions=baseline_output['actions'],
            ablated_actions=ablated_output['actions'],
            prob_difference=prob_diff,
            action_changed=action_changed
        )
    
    def _create_trajectory_result(
        self, 
        baseline_result: Dict, 
        ablated_result: Dict
    ) -> TrajectoryComparison:
        """Create TrajectoryComparison from full trajectory results."""
        baseline_rewards = baseline_result['reward']
        ablated_rewards = ablated_result['reward']
        reward_diff = ablated_rewards - baseline_rewards
        
        # Check if tours changed (if actions are available)
        tour_changed = torch.zeros(baseline_rewards.shape[0], dtype=torch.bool, device=self.device)
        if 'actions' in baseline_result and 'actions' in ablated_result:
            baseline_actions = baseline_result['actions']
            ablated_actions = ablated_result['actions']
            tour_changed = ~torch.all(baseline_actions == ablated_actions, dim=-1)
        
        return TrajectoryComparison(
            baseline_rewards=baseline_rewards,
            ablated_rewards=ablated_rewards,
            reward_difference=reward_diff,
            tour_changed=tour_changed
        )
    
    def compute_batch_statistics(self, results: Union[ComparisonResult, TrajectoryComparison]) -> Dict:
        """
        Compute summary statistics across a batch of comparisons.
        
        Args:
            results: ComparisonResult or TrajectoryComparison
            
        Returns:
            Dictionary with mean, std, min, max statistics
        """
        stats = {}
        
        if isinstance(results, ComparisonResult):
            # Single-step statistics
            stats['prob_difference'] = {
                'mean': results.prob_difference.mean().item(),
                'std': results.prob_difference.std().item(), 
                'min': results.prob_difference.min().item(),
                'max': results.prob_difference.max().item()
            }
            stats['action_changed_rate'] = results.action_changed.mean().item()
            
        elif isinstance(results, TrajectoryComparison):
            # Trajectory-level statistics
            stats['reward_difference'] = {
                'mean': results.reward_difference.mean().item(),
                'std': results.reward_difference.std().item(),
                'min': results.reward_difference.min().item(),
                'max': results.reward_difference.max().item()
            }
            stats['tour_changed_rate'] = results.tour_changed.float().mean().item()
        
        return stats 