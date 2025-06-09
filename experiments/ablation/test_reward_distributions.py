#!/usr/bin/env python3

import os
import sys
import argparse
import pickle
import json
import glob
from pathlib import Path
import time
from collections import defaultdict

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add parent directories to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from policy.policy_hooked import HookedAttentionModelPolicy
from policy.reinforce_clipped import REINFORCEClipped
from sae.model.sae_model import TopKSparseAutoencoder
from rl4co.envs import TSPEnv
from feature_ablator import load_tsp_model, load_sae_model, FeatureAblator

# Add safe globals for PyTorch 2.6 compatibility
torch.serialization.add_safe_globals([TSPEnv])


class NearestNeighborRewardTester:
    """
    Efficiently test nearest neighbor reward distributions across all SAE features.
    This helps understand noise patterns before implementing bandit algorithms.
    """
    
    def __init__(self, run_name: str, sae_run_name: str, device: torch.device):
        self.run_name = run_name
        self.sae_run_name = sae_run_name
        self.device = device
        # Fix path to point to runs from experiments/ablation
        self.run_path = Path("../../runs") / run_name
        self.sae_path = self.run_path / "sae" / "sae_runs" / sae_run_name
        
        # Results storage
        self.feature_rewards = defaultdict(list)  # feature_idx -> [rewards]
        self.feature_stats = {}  # feature_idx -> {mean, std, min, max, etc.}
        
        self._load_models()
    
    def _load_models(self):
        """Load TSP model, SAE model, and environment"""
        print("Loading models...")
        
        # Load environment
        env_path = self.run_path / "env.pkl"
        if not env_path.exists():
            raise FileNotFoundError(f"Environment not found: {env_path}")
        with open(env_path, "rb") as f:
            self.env = pickle.load(f)
        
        # Load config
        config_path = self.run_path / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")
        with open(config_path, "r") as f:
            self.config = json.load(f)
        
        # Use the working load functions from feature_ablator.py
        self.tsp_model = load_tsp_model(self.run_path, self.config, self.env, self.device)
        self.sae_model = load_sae_model(self.sae_path, self.run_path, self.device)
        
        # Create ablator for feature ablation
        self.ablator = FeatureAblator(self.tsp_model, self.sae_model)
        
        # Get policy for forward passes
        if hasattr(self.tsp_model, 'policy'):
            self.policy = self.tsp_model.policy
        else:
            self.policy = self.tsp_model
        
        print(f"Loaded models: TSP solver + SAE with {self.sae_model.latent_dim} features")
    
    def _compute_nearest_neighbor_reward_from_actions(self, baseline_actions, ablated_actions, instances):
        """
        Compute nearest neighbor reward from full action trajectories.
        
        Now checks multiple steps along the trajectory to see if ablation affects
        nearest neighbor decisions at different points.
        
        Args:
            baseline_actions: [batch_size, seq_len] - actions from unmodified model
            ablated_actions: [batch_size, seq_len] - actions from ablated model  
            instances: batch of TSP instances
            
        Returns:
            rewards: [batch_size] - reward for each instance
        """
        batch_size = baseline_actions.shape[0]
        seq_len = baseline_actions.shape[1]
        locs = instances['locs']  # [batch_size, num_nodes, 2]
        
        # Test multiple steps for more signal
        test_steps = [0, 10, 20, 30, 40] if seq_len > 40 else [0, seq_len//4, seq_len//2, 3*seq_len//4]
        test_steps = [s for s in test_steps if s < seq_len]  # Only valid steps
        
        batch_rewards = []
        
        # For each instance, compare nearest neighbor agreement across multiple steps
        for b in range(batch_size):
            instance_locs = locs[b]  # [num_nodes, 2]
            step_rewards = []
            
            for step in test_steps:
                # Get visited nodes up to this step
                if step == 0:
                    visited = torch.tensor([0], device=instance_locs.device)  # Start with depot
                    current_pos = instance_locs[0]  # At depot
                else:
                    visited = torch.cat([torch.tensor([0], device=instance_locs.device), 
                                       baseline_actions[b, :step]])  # Depot + previous actions
                    current_pos = instance_locs[baseline_actions[b, step-1]]  # At last visited node
                
                # Find unvisited nodes
                all_nodes = torch.arange(instance_locs.shape[0], device=instance_locs.device)
                unvisited_mask = ~torch.isin(all_nodes, visited)
                unvisited_nodes = all_nodes[unvisited_mask]
                
                if len(unvisited_nodes) == 0:
                    continue  # No more nodes to visit
                
                # Find nearest unvisited neighbor
                distances_to_unvisited = torch.norm(
                    instance_locs[unvisited_nodes] - current_pos.unsqueeze(0), dim=1
                )
                nearest_neighbor_idx = unvisited_nodes[torch.argmin(distances_to_unvisited)]
                
                # Check if models chose nearest neighbor
                baseline_action = baseline_actions[b, step].item()
                ablated_action = ablated_actions[b, step].item()
                
                baseline_chooses_nn = (baseline_action == nearest_neighbor_idx.item())
                ablated_chooses_nn = (ablated_action == nearest_neighbor_idx.item())
                
                # Reward is change in NN agreement (positive = baseline was more NN-like)
                step_reward = float(baseline_chooses_nn) - float(ablated_chooses_nn)
                step_rewards.append(step_reward)
            
            # Average reward across all tested steps
            instance_reward = np.mean(step_rewards) if step_rewards else 0.0
            batch_rewards.append(instance_reward)
        
        return torch.tensor(batch_rewards)
    
    def test_all_features(self, num_instances: int = 1000, batch_size: int = 32):
        """
        Test all SAE features efficiently by caching baseline computation.
        
        Args:
            num_instances: Total number of instances to test
            batch_size: Batch size for processing
        """
        print(f"Testing all {self.sae_model.latent_dim} SAE features on {num_instances} instances...")
        
        # For efficiency, we'll generate instances in batches and run all tests on the same instances
        # to ensure fair comparison
        
        # Generate all instances in manageable batches
        print("Generating test instances...")
        all_instances = []
        all_baseline_actions = []
        
        num_batches = (num_instances + batch_size - 1) // batch_size
        
        for batch_idx in range(num_batches):
            actual_batch_size = min(batch_size, num_instances - batch_idx * batch_size)
            
            # Generate fresh instances for this batch
            batch_instances = self.env.reset(batch_size=[actual_batch_size]).to(self.device)
            all_instances.append(batch_instances)
            
            # Run baseline computation for this batch - CLONE THE INSTANCES!
            with torch.no_grad():
                baseline_result = self.policy(batch_instances.clone(), phase="test", decode_type="greedy", return_actions=True)
                all_baseline_actions.append(baseline_result['actions'])
        
        print(f"Generated {num_instances} instances in {num_batches} batches")
        
        # Test each feature
        print("Testing individual features...")
        
        for feature_idx in tqdm(range(self.sae_model.latent_dim), desc="Testing features"):
            feature_rewards = []
            
            # Test this feature on all batches
            for batch_idx in range(num_batches):
                batch_instances = all_instances[batch_idx]
                baseline_actions = all_baseline_actions[batch_idx]
                
                with torch.no_grad():
                    # Run ablated forward pass - CLONE THE INSTANCES!
                    with self.ablator.get_ablation_context("sae_feature", [feature_idx]):
                        ablated_result = self.policy(batch_instances.clone(), phase="test", decode_type="greedy", return_actions=True)
                
                # Compute rewards for this batch
                batch_rewards = self._compute_nearest_neighbor_reward_from_actions(
                    baseline_actions, ablated_result['actions'], batch_instances
                )
                feature_rewards.extend(batch_rewards.tolist())
            
            # Store results for this feature
            self.feature_rewards[feature_idx] = feature_rewards
            
            # Compute summary statistics
            rewards_array = np.array(feature_rewards)
            self.feature_stats[feature_idx] = {
                'mean': float(np.mean(rewards_array)),
                'std': float(np.std(rewards_array)),
                'min': float(np.min(rewards_array)),
                'max': float(np.max(rewards_array)),
                'median': float(np.median(rewards_array)),
                'q25': float(np.percentile(rewards_array, 25)),
                'q75': float(np.percentile(rewards_array, 75)),
                'abs_mean': float(np.mean(np.abs(rewards_array))),  # Mean absolute effect
                'signal_to_noise': float(np.abs(np.mean(rewards_array)) / (np.std(rewards_array) + 1e-8))
            }
    
    def save_results(self, output_dir: str = "experiments/ablation/reward_distributions"):
        """Save all results to disk"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save raw rewards
        with open(output_path / "feature_rewards.pkl", "wb") as f:
            pickle.dump(dict(self.feature_rewards), f)
        
        # Save summary statistics
        with open(output_path / "feature_stats.json", "w") as f:
            json.dump(self.feature_stats, f, indent=2)
        
        print(f"Saved results to {output_path}")
    
    def generate_analysis(self, output_dir: str = "experiments/ablation/reward_distributions"):
        """Generate analysis plots and summaries"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Summary statistics across all features
        all_means = [stats['mean'] for stats in self.feature_stats.values()]
        all_stds = [stats['std'] for stats in self.feature_stats.values()]
        all_abs_means = [stats['abs_mean'] for stats in self.feature_stats.values()]
        all_snr = [stats['signal_to_noise'] for stats in self.feature_stats.values()]
        
        # Plot 1: Distribution of mean effects
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.hist(all_means, bins=50, alpha=0.7, edgecolor='black')
        plt.xlabel('Mean Reward Effect')
        plt.ylabel('Number of Features')
        plt.title('Distribution of Mean Effects Across Features')
        plt.axvline(0, color='red', linestyle='--', alpha=0.5)
        
        plt.subplot(2, 2, 2)
        plt.hist(all_stds, bins=50, alpha=0.7, edgecolor='black', color='orange')
        plt.xlabel('Standard Deviation of Rewards')
        plt.ylabel('Number of Features')
        plt.title('Distribution of Reward Noise Across Features')
        
        plt.subplot(2, 2, 3)
        plt.hist(all_abs_means, bins=50, alpha=0.7, edgecolor='black', color='green')
        plt.xlabel('Mean Absolute Effect')
        plt.ylabel('Number of Features')
        plt.title('Distribution of Absolute Effects')
        
        plt.subplot(2, 2, 4)
        plt.hist(all_snr, bins=50, alpha=0.7, edgecolor='black', color='purple')
        plt.xlabel('Signal-to-Noise Ratio')
        plt.ylabel('Number of Features')
        plt.title('Distribution of Signal-to-Noise Ratios')
        
        plt.tight_layout()
        plt.savefig(output_path / "reward_distribution_summary.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot 2: Top features by different metrics
        # Sort features by different criteria
        by_abs_mean = sorted(self.feature_stats.items(), key=lambda x: x[1]['abs_mean'], reverse=True)
        by_snr = sorted(self.feature_stats.items(), key=lambda x: x[1]['signal_to_noise'], reverse=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Top 20 by absolute mean effect
        top_20_abs = by_abs_mean[:20]
        feature_ids = [str(x[0]) for x in top_20_abs]
        abs_means = [x[1]['abs_mean'] for x in top_20_abs]
        
        axes[0, 0].bar(range(len(feature_ids)), abs_means, color='green', alpha=0.7)
        axes[0, 0].set_xticks(range(len(feature_ids)))
        axes[0, 0].set_xticklabels(feature_ids, rotation=45)
        axes[0, 0].set_title('Top 20 Features by Absolute Mean Effect')
        axes[0, 0].set_ylabel('Mean Absolute Effect')
        
        # Top 20 by signal-to-noise ratio
        top_20_snr = by_snr[:20]
        feature_ids_snr = [str(x[0]) for x in top_20_snr]
        snr_values = [x[1]['signal_to_noise'] for x in top_20_snr]
        
        axes[0, 1].bar(range(len(feature_ids_snr)), snr_values, color='purple', alpha=0.7)
        axes[0, 1].set_xticks(range(len(feature_ids_snr)))
        axes[0, 1].set_xticklabels(feature_ids_snr, rotation=45)
        axes[0, 1].set_title('Top 20 Features by Signal-to-Noise Ratio')
        axes[0, 1].set_ylabel('Signal-to-Noise Ratio')
        
        # Histogram of rewards for top feature by absolute effect
        top_feature_idx = by_abs_mean[0][0]
        top_feature_rewards = self.feature_rewards[top_feature_idx]
        
        axes[1, 0].hist(top_feature_rewards, bins=50, alpha=0.7, edgecolor='black')
        axes[1, 0].set_xlabel('Reward Value')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title(f'Reward Distribution for Feature {top_feature_idx}\n(Top by Absolute Effect)')
        axes[1, 0].axvline(np.mean(top_feature_rewards), color='red', linestyle='--', label='Mean')
        axes[1, 0].legend()
        
        # Histogram of rewards for top feature by SNR
        top_snr_feature_idx = by_snr[0][0]
        top_snr_feature_rewards = self.feature_rewards[top_snr_feature_idx]
        
        axes[1, 1].hist(top_snr_feature_rewards, bins=50, alpha=0.7, edgecolor='black', color='purple')
        axes[1, 1].set_xlabel('Reward Value')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title(f'Reward Distribution for Feature {top_snr_feature_idx}\n(Top by SNR)')
        axes[1, 1].axvline(np.mean(top_snr_feature_rewards), color='red', linestyle='--', label='Mean')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(output_path / "top_features_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Print summary statistics
        print("\n" + "="*60)
        print("REWARD DISTRIBUTION ANALYSIS SUMMARY")
        print("="*60)
        print(f"Total features tested: {len(self.feature_stats)}")
        print(f"Instances per feature: {len(next(iter(self.feature_rewards.values())))}")
        print()
        print("OVERALL STATISTICS:")
        print(f"Mean effect size: {np.mean(all_means):.6f} ± {np.std(all_means):.6f}")
        print(f"Mean absolute effect: {np.mean(all_abs_means):.6f} ± {np.std(all_abs_means):.6f}")
        print(f"Mean noise level (std): {np.mean(all_stds):.6f} ± {np.std(all_stds):.6f}")
        print(f"Mean signal-to-noise ratio: {np.mean(all_snr):.3f} ± {np.std(all_snr):.3f}")
        print()
        print("TOP 5 FEATURES BY ABSOLUTE EFFECT:")
        for i, (feat_idx, stats) in enumerate(by_abs_mean[:5]):
            print(f"{i+1}. Feature {feat_idx}: {stats['abs_mean']:.6f} (SNR: {stats['signal_to_noise']:.3f})")
        print()
        print("TOP 5 FEATURES BY SIGNAL-TO-NOISE RATIO:")
        for i, (feat_idx, stats) in enumerate(by_snr[:5]):
            print(f"{i+1}. Feature {feat_idx}: SNR {stats['signal_to_noise']:.3f} (Effect: {stats['abs_mean']:.6f})")
        
        # Save text summary
        with open(output_path / "summary.txt", "w") as f:
            f.write("REWARD DISTRIBUTION ANALYSIS SUMMARY\n")
            f.write("="*60 + "\n")
            f.write(f"Total features tested: {len(self.feature_stats)}\n")
            f.write(f"Instances per feature: {len(next(iter(self.feature_rewards.values())))}\n\n")
            f.write("OVERALL STATISTICS:\n")
            f.write(f"Mean effect size: {np.mean(all_means):.6f} ± {np.std(all_means):.6f}\n")
            f.write(f"Mean absolute effect: {np.mean(all_abs_means):.6f} ± {np.std(all_abs_means):.6f}\n")
            f.write(f"Mean noise level (std): {np.mean(all_stds):.6f} ± {np.std(all_stds):.6f}\n")
            f.write(f"Mean signal-to-noise ratio: {np.mean(all_snr):.3f} ± {np.std(all_snr):.3f}\n\n")
            
            f.write("TOP 10 FEATURES BY ABSOLUTE EFFECT:\n")
            for i, (feat_idx, stats) in enumerate(by_abs_mean[:10]):
                f.write(f"{i+1}. Feature {feat_idx}: {stats['abs_mean']:.6f} (SNR: {stats['signal_to_noise']:.3f})\n")
            
            f.write("\nTOP 10 FEATURES BY SIGNAL-TO-NOISE RATIO:\n")
            for i, (feat_idx, stats) in enumerate(by_snr[:10]):
                f.write(f"{i+1}. Feature {feat_idx}: SNR {stats['signal_to_noise']:.3f} (Effect: {stats['abs_mean']:.6f})\n")


def main():
    # Hardcoded parameters - FULL RUN
    run_name = "Long_RandomUniform"
    sae_run_name = "sae_l10.001_ef4.0_k0.1_04-03_10:39:46"
    num_instances = 100  # Good balance of signal vs speed
    batch_size = 16     
    output_dir = "experiments/ablation/reward_distributions"
    
    # FULL RUN: Test all features now that we have signal!
    # max_features_to_test = 50  # Comment out for full run
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Testing run: {run_name}")
    print(f"SAE run: {sae_run_name}")
    print(f"Instances per feature: {num_instances}")
    print(f"FULL RUN: Testing all 1024 SAE features!")
    print(f"Output directory: {output_dir}")
    print()
    
    # Initialize tester
    tester = NearestNeighborRewardTester(run_name, sae_run_name, device)
    
    print(f"Testing all {tester.sae_model.latent_dim} SAE features...")
    
    # Run the test
    start_time = time.time()
    tester.test_all_features(num_instances, batch_size)
    end_time = time.time()
    
    print(f"\nCompleted testing in {end_time - start_time:.1f} seconds")
    
    # Save results and generate analysis
    tester.save_results(output_dir)
    tester.generate_analysis(output_dir)
    
    print(f"\nResults saved to: {output_dir}")
    print(f"\nFULL ANALYSIS COMPLETE! 🎉")
    print(f"We found {len([f for f in tester.feature_stats.values() if f['abs_mean'] > 0.5])} high-impact features!")
    
    # Print top 10 by absolute effect
    by_abs_effect = sorted(tester.feature_stats.items(), key=lambda x: x[1]['abs_mean'], reverse=True)
    print(f"\nTOP 10 FEATURES BY NEAREST NEIGHBOR IMPACT:")
    for i, (feat_idx, stats) in enumerate(by_abs_effect[:10]):
        print(f"{i+1:2d}. Feature {feat_idx:3d}: {stats['abs_mean']:.3f} effect (SNR: {stats['signal_to_noise']:.2f})")
    
    return tester.feature_stats


if __name__ == "__main__":
    main() 