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

# Import RL4CO classes for safe loading
from rl4co.envs.routing.tsp.env import TSPEnv

# Add safe globals for PyTorch 2.6 compatibility
torch.serialization.add_safe_globals([TSPEnv])

def test_behavior_comparator():
    """Test the behavior comparator implementation"""
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
    
    # Create ablator and comparator
    ablator = FeatureAblator(tsp_model, sae_model)
    comparator = BehaviorComparator(tsp_model, sae_model, ablator)
    
    print("Testing behavior comparator...")
    
    # Create test instances
    test_instances = env.reset(batch_size=[2]).to(device)
    
    # Test single step comparison (which now falls back to full trajectory)
    print("\n=== Testing Single Step Comparison ===")
    try:
        single_result = comparator.compare_single_step(
            test_instances, 
            target_type="sae_feature", 
            target_idx=0,
            step_idx=0
        )
        print(f"Single step comparison successful!")
        print(f"Result keys: {single_result.keys()}")
    except Exception as e:
        print(f"Error in single step comparison: {e}")
    
    # Test full trajectory comparison
    print("\n=== Testing Full Trajectory Comparison ===")
    try:
        full_result = comparator.compare_full_trajectory(
            test_instances,
            target_type="sae_feature",
            target_idx=1
        )
        print(f"Full trajectory comparison successful!")
        print(f"Result keys: {full_result.keys()}")
        
        # Print some metrics
        print(f"Reward difference: {full_result.get('reward_diff', 'N/A')}")
        print(f"Action changes: {full_result.get('action_changes', 'N/A')}")
        
    except Exception as e:
        print(f"Error in full trajectory comparison: {e}")
    
    print("\nBehavior comparator test complete!")

if __name__ == "__main__":
    test_behavior_comparator() 