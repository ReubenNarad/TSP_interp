#!/usr/bin/env python3

import torch
import pickle
import json
from pathlib import Path
import sys
import os

# Add parent directories to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from feature_ablator import load_tsp_model, load_sae_model

# Add safe globals for PyTorch 2.6 compatibility
from rl4co.envs import TSPEnv
torch.serialization.add_safe_globals([TSPEnv])

def test_tensordict_slicing():
    """Test if TensorDict slicing is causing the decoding issue"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    run_path = Path('../../runs/Long_RandomUniform')
    
    # Load environment
    env_path = run_path / 'env.pkl'
    with open(env_path, 'rb') as f:
        env = pickle.load(f)
    
    # Load config and model
    config_path = run_path / 'config.json'
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    tsp_model = load_tsp_model(run_path, config, env, device)
    policy = tsp_model.policy
    
    print("Testing TensorDict slicing...")
    
    # Test with small instances
    instances = env.reset(batch_size=[10]).to(device)
    print(f'Full instances shape: {instances["locs"].shape}')
    
    # Test full instances
    print('Testing full instances...')
    try:
        result1 = policy(instances, phase='test', decode_type='greedy', return_actions=True)
        print(f'Full instances result: {result1["actions"].shape}')
        print("✅ Full instances work!")
    except Exception as e:
        print(f"❌ Full instances failed: {e}")
        return
    
    # Test sliced instances
    print('Testing sliced instances...')
    try:
        batch_slice = instances[0:5]
        print(f'Sliced instances shape: {batch_slice["locs"].shape}')
        result2 = policy(batch_slice, phase='test', decode_type='greedy', return_actions=True)
        print(f'Sliced instances result: {result2["actions"].shape}')
        print("✅ Sliced instances work!")
    except Exception as e:
        print(f"❌ Sliced instances failed: {e}")
        return
    
    # Test with even smaller slices
    print('Testing very small slice...')
    try:
        tiny_slice = instances[0:2]
        result3 = policy(tiny_slice, phase='test', decode_type='greedy', return_actions=True)
        print(f'Tiny slice result: {result3["actions"].shape}')
        print("✅ Tiny slice works!")
    except Exception as e:
        print(f"❌ Tiny slice failed: {e}")
        return
        
    print("All slicing tests passed!")

if __name__ == "__main__":
    test_tensordict_slicing() 