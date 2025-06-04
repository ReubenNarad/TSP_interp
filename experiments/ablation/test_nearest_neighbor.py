#!/usr/bin/env python3

import torch
import pickle
import json
from pathlib import Path
import sys
import os

# Add parent directories to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from feature_ablator import FeatureAblator, load_tsp_model, load_sae_model
from behavior_comparator import BehaviorComparator
from nearest_neighbor_policy import NearestNeighborPolicy, NearestNeighborComparator, TrajectoryState

# Import RL4CO classes for safe loading
from rl4co.envs.routing.tsp.env import TSPEnv

# Add safe globals for PyTorch 2.6 compatibility
torch.serialization.add_safe_globals([TSPEnv])

def test_nearest_neighbor_policy():
    """Test the nearest neighbor policy implementation"""
    run_name = "Long_RandomUniform"
    sae_run_name = "sae_l10.001_ef4.0_k0.1_04-03_10:39:46"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Set up paths
    run_path = Path("../../runs") / run_name
    sae_path = run_path / "sae" / "sae_runs" / sae_run_name
    
    print("Loading models...")
    # Load environment
    env_path = run_path / "env.pkl"
    with open(env_path, "rb") as f:
        env = pickle.load(f)
    
    # Load config
    config_path = run_path / "config.json"
    with open(config_path, "r") as f:
        config = json.load(f)
    
    # Load models
    tsp_model = load_tsp_model(run_path, config, env, device)
    sae_model = load_sae_model(sae_path, run_path, device)
    
    print("Testing nearest neighbor policy...")
    
    # Create test instances
    test_instances = env.reset(batch_size=[2]).to(device)
    
    # Initialize nearest neighbor policy
    nn_policy = NearestNeighborPolicy(device=device)
    nn_comparator = NearestNeighborComparator(nn_policy)
    
    print("\n=== Testing Complete NN Trajectory ===")
    # Test complete nearest neighbor trajectory
    nn_result = nn_policy.get_trajectory_actions(test_instances)
    print(f"NN trajectory shape: {nn_result['actions'].shape}")
    print(f"NN tour lengths: {nn_result['tour_length']}")
    print(f"NN rewards: {nn_result['reward']}")
    
    print("\n=== Testing Neural Model Trajectory ===")
    # Get neural model trajectory for comparison
    with torch.no_grad():
        if hasattr(tsp_model, 'policy'):
            policy = tsp_model.policy
        else:
            policy = tsp_model
        neural_result = policy(test_instances, phase="test", decode_type="greedy", return_actions=True)
    
    print(f"Neural trajectory shape: {neural_result['actions'].shape}")
    print(f"Neural tour lengths (negative rewards): {-neural_result['reward']}")
    print(f"Neural rewards: {neural_result['reward']}")
    
    print("\n=== Comparing Performance ===")
    nn_tour_length = nn_result['tour_length'].mean().item()
    neural_tour_length = (-neural_result['reward']).mean().item()
    print(f"Average NN tour length: {nn_tour_length:.4f}")
    print(f"Average Neural tour length: {neural_tour_length:.4f}")
    print(f"Neural improvement over NN: {((nn_tour_length - neural_tour_length) / nn_tour_length * 100):.2f}%")
    
    print("\n=== Testing Step-wise Comparison ===")
    # Test step-wise comparison at multiple points
    test_steps = [0, 5, 10, 20, 30] if test_instances['locs'].shape[1] > 30 else [0, 1, 2, 3, 4]
    
    for step in test_steps:
        if step >= neural_result['actions'].shape[1]:
            continue
            
        print(f"\n--- Step {step} ---")
        
        # Extract trajectory state at this step
        trajectory_state = nn_comparator.extract_trajectory_state(
            test_instances, 
            neural_result['actions'], 
            step
        )
        
        # Get neural model's action at this step
        neural_action = neural_result['actions'][:, step]
        
        # Compare with nearest neighbor
        comparison = nn_comparator.compare_with_nearest_neighbor(
            test_instances,
            neural_action,
            trajectory_state
        )
        
        print(f"  Agreement rate: {comparison['agreement_rate']:.3f}")
        print(f"  Neural actions: {neural_action.cpu().numpy()}")
        print(f"  NN actions: {comparison['nn_actions'].cpu().numpy()}")
        print(f"  Neural distances: {comparison['neural_distances'].cpu().numpy()}")
        print(f"  NN distances: {comparison['nn_distances'].cpu().numpy()}")
        print(f"  Distance ratios: {comparison['distance_ratio'].cpu().numpy()}")
    
    print("\n=== Testing with Feature Ablation ===")
    # Now test how feature ablation affects NN agreement
    ablator = FeatureAblator(tsp_model, sae_model)
    
    # Test step 5 with and without ablation
    test_step = 5
    if test_step < neural_result['actions'].shape[1]:
        print(f"\nTesting step {test_step} with feature ablation...")
        
        # Baseline neural trajectory
        baseline_trajectory_state = nn_comparator.extract_trajectory_state(
            test_instances,
            neural_result['actions'],
            test_step
        )
        baseline_action = neural_result['actions'][:, test_step]
        baseline_comparison = nn_comparator.compare_with_nearest_neighbor(
            test_instances,
            baseline_action,
            baseline_trajectory_state
        )
        
        # Ablated neural trajectory
        fresh_instances = env.reset(batch_size=[2]).to(device)  # Use fresh instances
        with ablator.get_ablation_context("sae_feature", [0]):  # Ablate feature 0
            ablated_result = policy(fresh_instances, phase="test", decode_type="greedy", return_actions=True)
        
        ablated_trajectory_state = nn_comparator.extract_trajectory_state(
            fresh_instances,
            ablated_result['actions'],
            test_step
        )
        ablated_action = ablated_result['actions'][:, test_step]
        ablated_comparison = nn_comparator.compare_with_nearest_neighbor(
            fresh_instances,
            ablated_action,
            ablated_trajectory_state
        )
        
        print(f"Baseline NN agreement: {baseline_comparison['agreement_rate']:.3f}")
        print(f"Ablated NN agreement: {ablated_comparison['agreement_rate']:.3f}")
        print(f"Agreement change: {ablated_comparison['agreement_rate'] - baseline_comparison['agreement_rate']:.3f}")
    
    print("\nNearest neighbor policy test complete!")

if __name__ == "__main__":
    test_nearest_neighbor_policy() 