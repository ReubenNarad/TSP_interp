#!/usr/bin/env python3

import torch
import pickle
import json
from pathlib import Path
import sys
import os
import time

# Add parent directories to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from feature_ablator import FeatureAblator, load_tsp_model, load_sae_model
from behavior_comparator import BehaviorComparator
from behaviors.nearest_neighbor_policy import NearestNeighborPolicy, NearestNeighborComparator, TrajectoryState

# Import RL4CO classes for safe loading
from rl4co.envs.routing.tsp.env import TSPEnv

# Add safe globals for PyTorch 2.6 compatibility
torch.serialization.add_safe_globals([TSPEnv])

def find_active_features(tsp_model, sae_model, test_instances, top_k=10):
    """Find which SAE features are actually active on the test instances"""
    print("Finding active features...")
    
    # Get policy for forward passes
    if hasattr(tsp_model, 'policy'):
        policy = tsp_model.policy
    else:
        policy = tsp_model
    
    with torch.no_grad():
        # Get encoder output
        h, _ = policy.encoder(test_instances)
        if len(h.shape) == 3:
            b_size, n_nodes, f_dim = h.shape
            model_acts_flat = h.reshape(-1, f_dim)
        else: 
            model_acts_flat = h

        # Pass through SAE
        _, sae_acts_flat = sae_model(model_acts_flat)
        
        # Compute feature norms across all nodes/instances
        feature_norms = torch.norm(sae_acts_flat, dim=0)  # Shape: [num_features]
    
    # Find top-k most active features
    active_indices = torch.nonzero(feature_norms > 0).squeeze(-1)
    if len(active_indices) == 0:
        print("Warning: No active features found!")
        return []
    
    # Sort by activation strength
    active_norms = feature_norms[active_indices]
    sorted_indices = torch.argsort(active_norms, descending=True)
    top_active_features = active_indices[sorted_indices[:top_k]].cpu().tolist()
    
    print(f"Found {len(active_indices)} active features total")
    print(f"Top {len(top_active_features)} most active features:")
    for i, feat_idx in enumerate(top_active_features):
        norm_val = feature_norms[feat_idx].item()
        print(f"  {i+1}. Feature {feat_idx}: norm = {norm_val:.4f}")
    
    return top_active_features

def test_active_features_nn_comparison():
    """Test batch comparison of ACTIVE features using NN agreement rate CHANGE"""
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
    
    # Create ablator and NN comparator
    ablator = FeatureAblator(tsp_model, sae_model)
    nn_policy = NearestNeighborPolicy(device=device)
    nn_comparator = NearestNeighborComparator(nn_policy)
    
    # Test parameters
    batch_size = 8
    test_steps = [0, 5, 10, 15]
    
    # Create FIXED test instances
    fixed_instances = env.reset(batch_size=[batch_size]).to(device)
    
    # STEP 1: Find active features
    active_features = find_active_features(tsp_model, sae_model, fixed_instances, top_k=10)
    
    if not active_features:
        print("No active features found - exiting")
        return []
    
    print(f"\nTesting {len(active_features)} active features with batch size {batch_size}")
    print(f"Testing at trajectory steps: {test_steps}")
    print(f"Key insight: Measuring CHANGE in NN agreement (baseline - ablated)")
    
    # Get policy for forward passes
    if hasattr(tsp_model, 'policy'):
        policy = tsp_model.policy
    else:
        policy = tsp_model
    
    print(f"\n=== Computing Baseline NN Agreement (once, reused for all features) ===")
    baseline_start = time.time()
    
    # Run baseline once on fixed instances
    with torch.no_grad():
        baseline_result = policy(fixed_instances.clone(), phase="test", decode_type="greedy", return_actions=True)
    
    # Compute baseline NN agreement on the same instances
    baseline_step_agreements = []
    for step in test_steps:
        if step >= baseline_result['actions'].shape[1]:
            continue
            
        # Extract baseline trajectory state
        baseline_trajectory_state = nn_comparator.extract_trajectory_state(
            fixed_instances,
            baseline_result['actions'],
            step
        )
        
        # Get baseline model's action at this step
        baseline_action = baseline_result['actions'][:, step]
        
        # Compare baseline with NN
        baseline_comparison = nn_comparator.compare_with_nearest_neighbor(
            fixed_instances,
            baseline_action,
            baseline_trajectory_state
        )
        
        baseline_step_agreements.append(baseline_comparison['agreement_rate'])
    
    baseline_avg_agreement = sum(baseline_step_agreements) / len(baseline_step_agreements) if baseline_step_agreements else 0.0
    baseline_time = time.time() - baseline_start
    
    print(f"Baseline forward pass: {baseline_time:.3f}s")
    print(f"Baseline tour lengths: {(-baseline_result['reward']).mean().item():.4f}")
    print(f"Baseline NN agreement: {baseline_avg_agreement:.3f}")
    
    # Store results for each feature
    feature_results = []
    
    print(f"\n=== Testing Each Active Feature (N={len(active_features)} forward passes) ===")
    total_ablation_start = time.time()
    
    for feature_idx in active_features:
        print(f"\n--- Feature {feature_idx} ---")
        feature_start_time = time.time()
        
        # Run ablated forward pass on the SAME fixed instances
        with torch.no_grad():
            with ablator.get_ablation_context("sae_feature", [feature_idx]):
                ablated_result = policy(fixed_instances.clone(), phase="test", decode_type="greedy", return_actions=True)
        
        ablation_time = time.time() - feature_start_time
        
        # Compare ablated model with NN at each test step
        ablated_step_agreements = []
        
        for step in test_steps:
            if step >= ablated_result['actions'].shape[1]:
                continue
                
            # Extract ablated trajectory state
            ablated_trajectory_state = nn_comparator.extract_trajectory_state(
                fixed_instances,
                ablated_result['actions'],
                step
            )
            
            # Get ablated model's action at this step
            ablated_action = ablated_result['actions'][:, step]
            
            # Compare ablated with NN
            ablated_comparison = nn_comparator.compare_with_nearest_neighbor(
                fixed_instances,
                ablated_action,
                ablated_trajectory_state
            )
            
            ablated_step_agreements.append(ablated_comparison['agreement_rate'])
        
        ablated_avg_agreement = sum(ablated_step_agreements) / len(ablated_step_agreements) if ablated_step_agreements else 0.0
        
        # Compute the CHANGE in NN agreement (this is the key insight!)
        agreement_change = baseline_avg_agreement - ablated_avg_agreement
        
        # Store results for this feature
        feature_result = {
            'feature_idx': feature_idx,
            'ablation_time': ablation_time,
            'baseline_agreement': baseline_avg_agreement,
            'ablated_agreement': ablated_avg_agreement,
            'agreement_change': agreement_change,  # POSITIVE = feature helps NN behavior
            'tour_length': (-ablated_result['reward']).mean().item(),
            'baseline_step_agreements': baseline_step_agreements,
            'ablated_step_agreements': ablated_step_agreements,
        }
        feature_results.append(feature_result)
        
        print(f"  Time: {ablation_time:.3f}s")
        print(f"  Baseline NN agreement: {baseline_avg_agreement:.3f}")
        print(f"  Ablated NN agreement: {ablated_avg_agreement:.3f}")
        print(f"  Change in NN agreement: {agreement_change:+.3f} ({'helps' if agreement_change > 0 else 'hinders'} NN behavior)")
    
    total_ablation_time = time.time() - total_ablation_start
    
    print(f"\n=== Summary Results ===")
    print(f"Baseline time: {baseline_time:.3f}s")
    print(f"Total ablation time: {total_ablation_time:.3f}s") 
    print(f"Average per-feature time: {total_ablation_time/len(active_features):.3f}s")
    print(f"Total time: {baseline_time + total_ablation_time:.3f}s")
    print(f"Efficiency: {len(active_features)+1} forward passes vs {2*len(active_features)} (naive)")
    
    print(f"\n=== Feature Impact Rankings ===")
    # Sort by absolute change (most impactful regardless of direction)
    sorted_by_impact = sorted(feature_results, key=lambda x: abs(x['agreement_change']), reverse=True)
    
    print("Most impactful features (largest change in NN agreement):")
    for i, result in enumerate(sorted_by_impact[:5]):
        direction = "promotes" if result['agreement_change'] > 0 else "inhibits"
        print(f"  {i+1}. Feature {result['feature_idx']}: {result['agreement_change']:+.3f} change ({direction} NN behavior)")
    
    print(f"\n=== Features That Promote NN Behavior ===")
    promoters = [r for r in feature_results if r['agreement_change'] > 0]
    promoters.sort(key=lambda x: x['agreement_change'], reverse=True)
    
    if promoters:
        print("Features that help model mimic NN (positive change):")
        for i, result in enumerate(promoters[:3]):
            print(f"  {i+1}. Feature {result['feature_idx']}: {result['agreement_change']:+.3f}")
    else:
        print("No features found that promote NN behavior")
    
    print(f"\n=== Features That Inhibit NN Behavior ===")
    inhibitors = [r for r in feature_results if r['agreement_change'] < 0]
    inhibitors.sort(key=lambda x: x['agreement_change'])  # Most negative first
    
    if inhibitors:
        print("Features that prevent model from mimicking NN (negative change):")
        for i, result in enumerate(inhibitors[:3]):
            print(f"  {i+1}. Feature {result['feature_idx']}: {result['agreement_change']:+.3f}")
    else:
        print("No features found that inhibit NN behavior")
    
    print(f"\n=== Step-wise Analysis ===")
    print("NN Agreement change by trajectory step:")
    for step_idx, step in enumerate(test_steps):
        baseline_step_vals = [r['baseline_step_agreements'][step_idx] for r in feature_results if step_idx < len(r['baseline_step_agreements'])]
        ablated_step_vals = [r['ablated_step_agreements'][step_idx] for r in feature_results if step_idx < len(r['ablated_step_agreements'])]
        
        if baseline_step_vals and ablated_step_vals:
            avg_baseline = sum(baseline_step_vals) / len(baseline_step_vals)
            avg_ablated = sum(ablated_step_vals) / len(ablated_step_vals)
            avg_change = avg_baseline - avg_ablated
            print(f"  Step {step}: {avg_change:+.3f} average change across all active features")
    
    print("\nActive features NN comparison test complete!")
    print("Key insight: Positive change = feature promotes NN behavior, Negative = feature inhibits NN behavior")
    return feature_results

if __name__ == "__main__":
    results = test_active_features_nn_comparison() 