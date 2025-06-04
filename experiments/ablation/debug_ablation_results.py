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
from behaviors.nearest_neighbor_policy import NearestNeighborPolicy, NearestNeighborComparator

# Import RL4CO classes for safe loading
from rl4co.envs.routing.tsp.env import TSPEnv
torch.serialization.add_safe_globals([TSPEnv])

def debug_ablation_results():
    """Debug whether different feature ablations actually produce different results"""
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
    
    # Create ablator
    ablator = FeatureAblator(tsp_model, sae_model)
    
    # Get policy
    if hasattr(tsp_model, 'policy'):
        policy = tsp_model.policy
    else:
        policy = tsp_model
    
    # Create FIXED test instances
    fixed_instances = env.reset(batch_size=[4]).to(device)  # Smaller batch for debugging
    
    print(f"Fixed instances shape: {fixed_instances['locs'].shape}")
    print(f"First instance first few locations:\n{fixed_instances['locs'][0, :3]}")
    
    # Test features [0, 1, 2]
    features_to_test = [0, 1, 2]
    results = {}
    
    print(f"\n=== Testing Features {features_to_test} ===")
    
    for feature_idx in features_to_test:
        print(f"\n--- Feature {feature_idx} ---")
        
        with torch.no_grad():
            with ablator.get_ablation_context("sae_feature", [feature_idx]):
                result = policy(fixed_instances.clone(), phase="test", decode_type="greedy", return_actions=True)
        
        # Store key info
        results[feature_idx] = {
            'actions': result['actions'].cpu(),
            'rewards': result['reward'].cpu(),
            'tour_length': (-result['reward']).mean().item()
        }
        
        print(f"  Tour length: {results[feature_idx]['tour_length']:.6f}")
        print(f"  First action sequence: {result['actions'][0, :5].cpu().tolist()}")
        print(f"  First reward: {result['reward'][0].item():.6f}")
    
    print(f"\n=== Comparison Results ===")
    
    # Compare actions between features
    for i, feat_a in enumerate(features_to_test):
        for feat_b in features_to_test[i+1:]:
            actions_a = results[feat_a]['actions']
            actions_b = results[feat_b]['actions']
            
            # Check if actions are identical
            actions_identical = torch.equal(actions_a, actions_b)
            
            # Check tour length difference
            length_diff = abs(results[feat_a]['tour_length'] - results[feat_b]['tour_length'])
            
            print(f"Feature {feat_a} vs Feature {feat_b}:")
            print(f"  Actions identical: {actions_identical}")
            print(f"  Tour length diff: {length_diff:.6f}")
            
            if not actions_identical:
                # Find first difference
                diff_mask = actions_a != actions_b
                if diff_mask.any():
                    first_diff = torch.nonzero(diff_mask)[0]
                    batch, step = first_diff[0].item(), first_diff[1].item()
                    print(f"  First difference at batch {batch}, step {step}: {actions_a[batch, step]} vs {actions_b[batch, step]}")
    
    # Test baseline (no ablation)
    print(f"\n=== Baseline (No Ablation) ===")
    with torch.no_grad():
        baseline_result = policy(fixed_instances.clone(), phase="test", decode_type="greedy", return_actions=True)
    
    baseline_tour_length = (-baseline_result['reward']).mean().item()
    print(f"Baseline tour length: {baseline_tour_length:.6f}")
    print(f"Baseline first action sequence: {baseline_result['actions'][0, :5].cpu().tolist()}")
    
    # Compare each feature result to baseline
    print(f"\n=== Feature vs Baseline Comparison ===")
    for feature_idx in features_to_test:
        actions_same_as_baseline = torch.equal(results[feature_idx]['actions'], baseline_result['actions'].cpu())
        length_diff_from_baseline = abs(results[feature_idx]['tour_length'] - baseline_tour_length)
        
        print(f"Feature {feature_idx} vs Baseline:")
        print(f"  Actions identical to baseline: {actions_same_as_baseline}")
        print(f"  Tour length diff from baseline: {length_diff_from_baseline:.6f}")
    
    return results

if __name__ == "__main__":
    debug_ablation_results() 