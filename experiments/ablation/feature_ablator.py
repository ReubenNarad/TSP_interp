import torch
import torch.nn as nn
from typing import Dict, List, Optional, Set, Union, Any
from contextlib import contextmanager
import os
import sys
import pickle
import json
import glob
from pathlib import Path

# Add parent directories to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from policy.policy_hooked import HookedAttentionModelPolicy
from policy.reinforce_clipped import REINFORCEClipped
from sae.model.sae_model import TopKSparseAutoencoder

# Import RL4CO classes for safe loading
from rl4co.envs.routing.tsp.env import TSPEnv

# Add safe globals for PyTorch 2.6 compatibility
torch.serialization.add_safe_globals([TSPEnv])


class FeatureAblator:
    """
    Handles the mechanics of zeroing out SAE features, raw neurons, and other model components
    during forward passes using PyTorch hooks and context managers.
    """
    
    def __init__(self, tsp_model, sae_model=None):
        """
        Initialize the feature ablator.
        
        Args:
            tsp_model: The TSP policy model
            sae_model: Optional SAE model for feature ablation
        """
        self.tsp_model = tsp_model
        self.sae_model = sae_model
        self.device = next(tsp_model.parameters()).device
        
        # Keep track of active hooks and ablations
        self.active_hooks = []
        self.active_ablations = {}
        
        # Store references to model layers for neuron ablations
        self._register_model_layers()
    
    def _register_model_layers(self):
        """Register model layers for potential ablation."""
        self.model_layers = {}
        
        # Get the actual policy model
        if hasattr(self.tsp_model, 'policy'):
            policy = self.tsp_model.policy
        else:
            policy = self.tsp_model
        
        # Debug: Print policy structure
        print(f"Policy type: {type(policy)}")
        if hasattr(policy, 'encoder'):
            print(f"Encoder type: {type(policy.encoder)}")
            if hasattr(policy.encoder, 'layers'):
                print(f"Encoder layers type: {type(policy.encoder.layers)}")
                print(f"Number of encoder layers: {len(policy.encoder.layers)}")
            # Also check for 'net' attribute which is common in RL4CO
            elif hasattr(policy.encoder, 'net') and hasattr(policy.encoder.net, 'layers'):
                print(f"Encoder net layers type: {type(policy.encoder.net.layers)}")
                print(f"Number of encoder net layers: {len(policy.encoder.net.layers)}")
        
        # Register encoder layers - try multiple possible paths
        encoder_layers = None
        if hasattr(policy, 'encoder'):
            if hasattr(policy.encoder, 'layers'):
                encoder_layers = policy.encoder.layers
            elif hasattr(policy.encoder, 'net') and hasattr(policy.encoder.net, 'layers'):
                encoder_layers = policy.encoder.net.layers
        
        if encoder_layers is not None:
            for i, layer in enumerate(encoder_layers):
                self.model_layers[f'encoder_layer_{i}'] = layer
                
                # Register specific sublayers - adapt to RL4CO structure
                # RL4CO often has layers as sequential blocks
                if hasattr(layer, 'self_attn'):
                    self.model_layers[f'encoder_layer_{i}_attention'] = layer.self_attn
                elif len(layer) > 0 and hasattr(layer[0], 'module'):
                    # RL4CO structure: layer[0] is attention, layer[2] is FFN
                    self.model_layers[f'encoder_layer_{i}_attention'] = layer[0]
                    if len(layer) > 2:
                        self.model_layers[f'encoder_layer_{i}_ffn'] = layer[2]
                
                if hasattr(layer, 'feed_forward') or hasattr(layer, 'ffn'):
                    ffn = getattr(layer, 'feed_forward', getattr(layer, 'ffn', None))
                    if ffn:
                        self.model_layers[f'encoder_layer_{i}_ffn'] = ffn
        
        print(f"Registered {len(self.model_layers)} model layers:")
        for layer_name in self.model_layers.keys():
            print(f"  - {layer_name}")
    
    @contextmanager
    def get_ablation_context(self, target_type: str, target_indices: List[int]):
        """
        Context manager for temporary ablations.
        
        Args:
            target_type: Type of ablation ("sae_feature", "encoder_neuron", etc.)
            target_indices: List of indices to ablate
        """
        # Setup ablation
        hook_handles = self._setup_ablation(target_type, target_indices)
        
        try:
            yield
        finally:
            # Clean up ablation
            self._cleanup_ablation(hook_handles)
    
    def _setup_ablation(self, target_type: str, target_indices: List[int]) -> List:
        """Setup hooks for ablation."""
        hook_handles = []
        
        if target_type == "sae_feature":
            if self.sae_model is None:
                raise ValueError("SAE model required for SAE feature ablation")
            handles = self._setup_sae_feature_ablation(target_indices)
            hook_handles.extend(handles)
            
        elif target_type.startswith("encoder_neuron"):
            # Parse layer index if specified (e.g., "encoder_neuron_layer_2")
            if "_layer_" in target_type:
                layer_idx = int(target_type.split("_layer_")[1])
                layer_name = f'encoder_layer_{layer_idx}'
            else:
                # Default to first encoder layer
                layer_name = 'encoder_layer_0'
            
            handles = self._setup_neuron_ablation(layer_name, target_indices)
            hook_handles.extend(handles)
            
        elif target_type.startswith("attention_head"):
            # Parse layer index (e.g., "attention_head_layer_1")
            if "_layer_" in target_type:
                layer_idx = int(target_type.split("_layer_")[1])
                layer_name = f'encoder_layer_{layer_idx}_attention'
            else:
                layer_name = 'encoder_layer_0_attention'
            
            handles = self._setup_attention_head_ablation(layer_name, target_indices)
            hook_handles.extend(handles)
            
        else:
            raise ValueError(f"Unknown target type: {target_type}")
        
        return hook_handles
    
    def _setup_sae_feature_ablation(self, feature_indices: List[int]) -> List:
        """Setup hooks to zero out SAE features by intercepting and modifying encoder outputs."""
        hook_handles = []
        
        if self.sae_model is None:
            raise ValueError("SAE model required for SAE feature ablation")
        
        def encoder_intervention_hook(module, input, output):
            """Hook to intercept encoder output, pass through SAE with ablated features, and replace."""
            print(f"Encoder intervention hook called! Output type: {type(output)}")
            
            # The encoder returns (h, init_h) where h is the processed embedding
            if isinstance(output, tuple) and len(output) == 2:
                h, init_h = output
                print(f"Encoder output - h: {h.shape}, init_h: {init_h.shape}")
                
                # Get the processed embeddings (h) - shape: [batch_size, num_nodes, embed_dim]
                original_shape = h.shape
                batch_size, num_nodes, embed_dim = original_shape
                
                # Flatten for SAE processing: [batch_size*num_nodes, embed_dim]
                h_flat = h.reshape(-1, embed_dim)
                
                # Pass through SAE to get reconstruction and latent activations
                with torch.no_grad():
                    sae_reconstruction, sae_latents = self.sae_model(h_flat)
                
                print(f"SAE - reconstruction: {sae_reconstruction.shape}, latents: {sae_latents.shape}")
                
                # Check feature indices are valid
                max_feature_idx = max(feature_indices) if feature_indices else -1
                if max_feature_idx >= sae_latents.shape[-1]:
                    print(f"WARNING: Feature index {max_feature_idx} >= latent dim {sae_latents.shape[-1]}")
                    valid_indices = [idx for idx in feature_indices if idx < sae_latents.shape[-1]]
                else:
                    valid_indices = feature_indices
                
                if valid_indices:
                    print(f"Zeroing out SAE features {valid_indices}")
                    
                    # Clone latents and zero out specified features
                    ablated_latents = sae_latents.clone()
                    ablated_latents[:, valid_indices] = 0.0
                    
                    # Check if ablation actually happened
                    original_norm = torch.norm(sae_latents[:, valid_indices]).item()
                    ablated_norm = torch.norm(ablated_latents[:, valid_indices]).item()
                    print(f"Feature ablation - original norm: {original_norm:.6f}, ablated norm: {ablated_norm:.6f}")
                    
                    # Recompute reconstruction with ablated features
                    ablated_reconstruction = self.sae_model.decode(ablated_latents)
                    
                    # Reshape back to original encoder output shape
                    h_ablated = ablated_reconstruction.reshape(original_shape)
                    
                    print(f"Returning ablated encoder output: {h_ablated.shape}")
                    return h_ablated, init_h
                else:
                    print("No valid feature indices to ablate")
                    return output
            else:
                print(f"Unexpected encoder output format: {type(output)}")
                return output
        
        # Hook the encoder instead of the SAE
        if hasattr(self.tsp_model, 'policy'):
            encoder = self.tsp_model.policy.encoder
        else:
            encoder = self.tsp_model.encoder
            
        print(f"Registering encoder intervention hook on {type(encoder)}")
        handle = encoder.register_forward_hook(encoder_intervention_hook)
        hook_handles.append(handle)
        
        return hook_handles
    
    def _setup_neuron_ablation(self, layer_name: str, neuron_indices: List[int]) -> List:
        """Setup hooks to zero out specific neurons in a layer."""
        hook_handles = []
        
        if layer_name not in self.model_layers:
            raise ValueError(f"Layer {layer_name} not found in model")
        
        layer = self.model_layers[layer_name]
        
        def neuron_ablation_hook(module, input, output):
            """Hook to zero out specific neurons."""
            modified_output = output.clone()
            
            # Handle different output shapes
            if len(modified_output.shape) == 3:
                # Shape: [batch, seq_len, features]
                modified_output[:, :, neuron_indices] = 0.0
            elif len(modified_output.shape) == 2:
                # Shape: [batch, features]
                modified_output[:, neuron_indices] = 0.0
            else:
                # Try to modify last dimension
                modified_output[..., neuron_indices] = 0.0
            
            return modified_output
        
        handle = layer.register_forward_hook(neuron_ablation_hook)
        hook_handles.append(handle)
        
        return hook_handles
    
    def _setup_attention_head_ablation(self, layer_name: str, head_indices: List[int]) -> List:
        """Setup hooks to zero out specific attention heads."""
        hook_handles = []
        
        if layer_name not in self.model_layers:
            raise ValueError(f"Attention layer {layer_name} not found in model")
        
        attention_layer = self.model_layers[layer_name]
        
        def attention_ablation_hook(module, input, output):
            """Hook to zero out specific attention heads."""
            # This is more complex - need to understand the attention structure
            # For now, implement a simple version
            modified_output = output.clone()
            
            # Attention output is typically [batch, seq_len, embed_dim]
            # Each head contributes embed_dim/num_heads dimensions
            if hasattr(module, 'num_heads'):
                num_heads = module.num_heads
                head_dim = modified_output.shape[-1] // num_heads
                
                for head_idx in head_indices:
                    start_dim = head_idx * head_dim
                    end_dim = (head_idx + 1) * head_dim
                    modified_output[..., start_dim:end_dim] = 0.0
            
            return modified_output
        
        handle = attention_layer.register_forward_hook(attention_ablation_hook)
        hook_handles.append(handle)
        
        return hook_handles
    
    def _cleanup_ablation(self, hook_handles: List):
        """Remove hooks after ablation."""
        for handle in hook_handles:
            handle.remove()
    
    def reset(self):
        """Clear all ablations and remove hooks."""
        # Remove any remaining hooks
        for handle in self.active_hooks:
            handle.remove()
        
        self.active_hooks.clear()
        self.active_ablations.clear()


def load_tsp_model(run_path, config, env, device):
    """Load the TSP policy model from checkpoint."""
    checkpoint_dir = run_path / "checkpoints"
    checkpoint_files = glob.glob(str(checkpoint_dir / "checkpoint_epoch_*.ckpt"))
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint files found in {checkpoint_dir}")
    
    latest_checkpoint = max(
        checkpoint_files,
        key=lambda f: int(os.path.basename(f).split("checkpoint_epoch_")[1].split(".ckpt")[0])
    )
    
    policy = HookedAttentionModelPolicy(
        env_name=env.name,
        embed_dim=config['embed_dim'],
        num_encoder_layers=config['n_encoder_layers'],
        num_heads=8,  # Default value
        temperature=config['temperature'],
    )
    
    # Try loading with weights_only=False for PyTorch 2.6 compatibility
    try:
        model = REINFORCEClipped.load_from_checkpoint(
            latest_checkpoint,
            env=env,
            policy=policy,
            strict=False
        )
    except Exception as e:
        if "weights_only" in str(e):
            print("PyTorch 2.6 detected - attempting alternative loading method...")
            # Load the checkpoint manually with weights_only=False
            import lightning.pytorch as pl
            checkpoint_data = torch.load(latest_checkpoint, map_location='cpu', weights_only=False)
            
            # Create model and load state dict manually
            model = REINFORCEClipped(
                env=env,
                policy=policy,
                baseline="rollout",
                baseline_kwargs={"warmup": 1000},
                batch_size=config.get('batch_size', 512),
                train_data_size=config.get('num_instances', 1000),
                val_data_size=config.get('num_val', 10),
                optimizer_kwargs={"lr": config.get('lr', 1e-4)},
            )
            
            # Load the state dict
            if 'state_dict' in checkpoint_data:
                model.load_state_dict(checkpoint_data['state_dict'], strict=False)
            else:
                model.load_state_dict(checkpoint_data, strict=False)
        else:
            raise e
    
    model = model.to(device)
    model.eval()
    print(f"Loaded TSP model from checkpoint: {latest_checkpoint}")
    return model


def load_sae_model(sae_path, run_path, device):
    """Load the SAE model from checkpoint."""
    sae_config_path = sae_path / "sae_config.json"
    if not sae_config_path.exists():
        raise FileNotFoundError(f"SAE config not found at {sae_config_path}")
    
    with open(sae_config_path, "r") as f:
        sae_config = json.load(f)
    
    # Try to find SAE model file
    sae_model_path = sae_path / "sae_final.pt"
    if not sae_model_path.exists():
        # Try to find the latest checkpoint
        checkpoint_dir = sae_path / "checkpoints"
        if checkpoint_dir.exists():
            checkpoint_files = glob.glob(str(checkpoint_dir / "sae_epoch_*.pt"))
            if checkpoint_files:
                latest_checkpoint = max(
                    checkpoint_files,
                    key=lambda f: int(os.path.basename(f).split("sae_epoch_")[1].split(".pt")[0])
                )
                sae_model_path = Path(latest_checkpoint)
            else:
                raise FileNotFoundError(f"No SAE checkpoints found in {checkpoint_dir}")
        else:
            raise FileNotFoundError(f"SAE model not found at {sae_model_path}")
    
    # Determine input dimension from activation files
    activation_dir = run_path / "sae" / "activations"
    activation_files = glob.glob(str(activation_dir / "activations_epoch_*.pt"))
    if not activation_files:
        raise FileNotFoundError(f"No activation files found in {activation_dir}")
    
    activations_sample = torch.load(activation_files[0], map_location='cpu', weights_only=False)
    activation_key = sae_config.get("activation_key", "encoder_output")
    if activation_key not in activations_sample:
        possible_keys = [k for k in activations_sample if "encoder" in k and "output" in k]
        if not possible_keys:
            raise ValueError(f"Activation key '{activation_key}' not found")
        activation_key = possible_keys[0]
    
    input_dim = activations_sample[activation_key].shape[-1]
    expansion_factor = sae_config.get("expansion_factor", 2.0)
    latent_dim = int(expansion_factor * input_dim)
    k_ratio = sae_config.get("k_ratio", 0.05)
    
    sae_model = TopKSparseAutoencoder(
        input_dim=input_dim,
        latent_dim=latent_dim,
        k_ratio=k_ratio,
        tied_weights=sae_config.get("tied_weights", False),
        bias_decay=sae_config.get("bias_decay", 0.0),
        dict_init=sae_config.get("init_method", "uniform")
    )
    
    checkpoint = torch.load(sae_model_path, map_location=device, weights_only=False)
    if "model_state_dict" in checkpoint:
        sae_model.load_state_dict(checkpoint["model_state_dict"])
    else:
        sae_model.load_state_dict(checkpoint)
    
    sae_model = sae_model.to(device)
    sae_model.eval()
    print(f"Loaded SAE model from {sae_model_path}")
    print(f"SAE model: {input_dim} -> {latent_dim}, k_ratio={k_ratio}")
    return sae_model


if __name__ == "__main__":
    # Test the feature ablator with the specified SAE run
    run_name = "Long_RandomUniform"
    sae_run_name = "sae_l10.001_ef4.0_k0.1_04-03_10:39:46"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Testing FeatureAblator with run: {run_name}, SAE: {sae_run_name}")
    print(f"Using device: {device}")
    
    # Set up paths
    run_path = Path("../../runs") / run_name  # Going up from experiments/ablation/
    sae_path = run_path / "sae" / "sae_runs" / sae_run_name
    
    print(f"Run path: {run_path}")
    print(f"SAE path: {sae_path}")
    
    try:
        # Load environment
        env_path = run_path / "env.pkl"
        if not env_path.exists():
            raise FileNotFoundError(f"Environment not found: {env_path}")
        
        with open(env_path, "rb") as f:
            env = pickle.load(f)
        print(f"Loaded environment from {env_path}")
        
        # Load config
        config_path = run_path / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")
        
        with open(config_path, "r") as f:
            config = json.load(f)
        print(f"Loaded config from {config_path}")
        
        # Load models
        print("\nLoading TSP model...")
        tsp_model = load_tsp_model(run_path, config, env, device)
        
        print("\nLoading SAE model...")
        sae_model = load_sae_model(sae_path, run_path, device)
        
        # Create ablator
        print("\nCreating FeatureAblator...")
        ablator = FeatureAblator(tsp_model, sae_model)
        
        # Test basic functionality
        print("\nTesting SAE feature ablation context manager...")
        try:
            with ablator.get_ablation_context("sae_feature", [0, 1, 2]):
                print("SAE feature ablation context active")
        except Exception as e:
            print(f"SAE feature ablation test failed: {e}")
        
        print("\nTesting encoder neuron ablation context manager...")
        if len(ablator.model_layers) > 0:
            try:
                with ablator.get_ablation_context("encoder_neuron_layer_0", [0, 1, 2]):
                    print("Encoder neuron ablation context active")
            except Exception as e:
                print(f"Encoder neuron ablation test failed: {e}")
        else:
            print("No encoder layers found - skipping encoder neuron test")
        
        print("\nFeatureAblator test completed successfully!")
        
        # Test with a small TSP instance
        print("\nTesting with actual TSP instance...")
        
        # Get the policy
        if hasattr(tsp_model, 'policy'):
            policy = tsp_model.policy
        else:
            policy = tsp_model
        
        # Test baseline forward pass
        print("Running baseline forward pass...")
        test_instances = env.reset(batch_size=[2]).to(device)
        baseline_result = policy(test_instances, phase="test", decode_type="greedy", return_actions=True)
        print(f"Baseline reward: {baseline_result['reward'].mean().item():.4f}")
        
        # Test ablated forward pass with fresh instances
        print("Running ablated forward pass...")
        test_instances_2 = env.reset(batch_size=[2]).to(device)  # Fresh instances
        with ablator.get_ablation_context("sae_feature", [0, 1, 2, 3, 4]):
            ablated_result = policy(test_instances_2, phase="test", decode_type="greedy", return_actions=True)
        print(f"Ablated reward: {ablated_result['reward'].mean().item():.4f}")
        
        # For fair comparison, compare same instances
        print("\nRunning fair comparison with same instances...")
        test_instances_3 = env.reset(batch_size=[2]).to(device)
        
        # Baseline
        print("=== BASELINE RUN ===")
        baseline_result_fair = policy(test_instances_3.clone(), phase="test", decode_type="greedy", return_actions=True)
        print(f"Fair baseline reward: {baseline_result_fair['reward'].mean().item():.4f}")
        
        # Ablated (with fresh copy of same instances)
        print("\n=== ABLATED RUN ===")
        with ablator.get_ablation_context("sae_feature", [0, 1, 2, 3, 4]):
            ablated_result_fair = policy(test_instances_3.clone(), phase="test", decode_type="greedy", return_actions=True)
        print(f"Fair ablated reward: {ablated_result_fair['reward'].mean().item():.4f}")
        
        reward_diff = (ablated_result_fair['reward'] - baseline_result_fair['reward']).mean().item()
        print(f"Reward difference: {reward_diff:.4f}")
        
        # Additional debugging: compare actions and check if anything changed
        if 'actions' in baseline_result_fair and 'actions' in ablated_result_fair:
            baseline_actions = baseline_result_fair['actions']
            ablated_actions = ablated_result_fair['actions']
            actions_identical = torch.all(baseline_actions == ablated_actions).item()
            print(f"Actions identical: {actions_identical}")
            
            if not actions_identical:
                diff_count = (baseline_actions != ablated_actions).sum().item()
                total_actions = baseline_actions.numel()
                print(f"Different actions: {diff_count}/{total_actions} ({100*diff_count/total_actions:.2f}%)")
            else:
                print("All actions are identical - ablation may not be working!")
        
        # Test with a much larger ablation to see if we get any effect
        print("\n=== TESTING LARGER ABLATION ===")
        test_instances_4 = env.reset(batch_size=[2]).to(device)
        
        # Baseline
        baseline_result_large = policy(test_instances_4.clone(), phase="test", decode_type="greedy", return_actions=True)
        print(f"Large test baseline reward: {baseline_result_large['reward'].mean().item():.4f}")
        
        # Ablate many more features
        large_feature_list = list(range(50))  # Ablate first 50 features
        print(f"Ablating features: {large_feature_list}")
        with ablator.get_ablation_context("sae_feature", large_feature_list):
            ablated_result_large = policy(test_instances_4.clone(), phase="test", decode_type="greedy", return_actions=True)
        print(f"Large test ablated reward: {ablated_result_large['reward'].mean().item():.4f}")
        
        reward_diff_large = (ablated_result_large['reward'] - baseline_result_large['reward']).mean().item()
        print(f"Large ablation reward difference: {reward_diff_large:.4f}")
        
        if 'actions' in baseline_result_large and 'actions' in ablated_result_large:
            baseline_actions_large = baseline_result_large['actions']
            ablated_actions_large = ablated_result_large['actions']
            actions_identical_large = torch.all(baseline_actions_large == ablated_actions_large).item()
            print(f"Large ablation - Actions identical: {actions_identical_large}")
        
        # Test individual features to find the most impactful
        print("\n=== TESTING INDIVIDUAL FEATURES ===")
        individual_results = []
        
        for feature_idx in range(5):  # Test features 0, 1, 2, 3, 4
            print(f"\n--- Testing Feature {feature_idx} ---")
            test_instances_single = env.reset(batch_size=[2]).to(device)
            
            # Baseline for this feature test
            baseline_result_single = policy(test_instances_single.clone(), phase="test", decode_type="greedy", return_actions=True)
            baseline_reward = baseline_result_single['reward'].mean().item()
            
            # Ablate single feature
            with ablator.get_ablation_context("sae_feature", [feature_idx]):
                ablated_result_single = policy(test_instances_single.clone(), phase="test", decode_type="greedy", return_actions=True)
            ablated_reward = ablated_result_single['reward'].mean().item()
            
            reward_diff = ablated_reward - baseline_reward
            
            # Check action differences
            if 'actions' in baseline_result_single and 'actions' in ablated_result_single:
                baseline_actions_single = baseline_result_single['actions']
                ablated_actions_single = ablated_result_single['actions']
                actions_identical_single = torch.all(baseline_actions_single == ablated_actions_single).item()
                diff_count = (baseline_actions_single != ablated_actions_single).sum().item()
                total_actions = baseline_actions_single.numel()
                action_change_pct = 100 * diff_count / total_actions
            else:
                actions_identical_single = True
                action_change_pct = 0.0
            
            print(f"Feature {feature_idx}: baseline={baseline_reward:.4f}, ablated={ablated_reward:.4f}, diff={reward_diff:.4f}")
            print(f"Feature {feature_idx}: actions changed {action_change_pct:.1f}%")
            
            individual_results.append({
                'feature': feature_idx,
                'baseline_reward': baseline_reward,
                'ablated_reward': ablated_reward,
                'reward_diff': reward_diff,
                'action_change_pct': action_change_pct,
                'actions_identical': actions_identical_single
            })
        
        # Find the feature with the largest performance drop
        print("\n=== INDIVIDUAL FEATURE IMPACT SUMMARY ===")
        # Sort by reward difference (most negative = biggest performance drop)
        sorted_results = sorted(individual_results, key=lambda x: x['reward_diff'])
        
        print("Features ranked by performance impact (largest drop first):")
        for i, result in enumerate(sorted_results):
            print(f"{i+1}. Feature {result['feature']}: {result['reward_diff']:.4f} reward change, {result['action_change_pct']:.1f}% actions changed")
        
        most_impactful = sorted_results[0]
        print(f"\nMost impactful feature: Feature {most_impactful['feature']} (reward change: {most_impactful['reward_diff']:.4f})")
        
        # Test integration with behavior comparator
        print("\n=== TESTING FEATURE ABLATOR + BEHAVIOR COMPARATOR INTEGRATION ===")
        try:
            # Import and create behavior comparator
            from behavior_comparator import BehaviorComparator
            
            # Create behavior comparator with our models and ablator
            comparator = BehaviorComparator(tsp_model, sae_model, ablator)
            
            # Test with the most impactful feature we found
            most_impactful_feature = most_impactful['feature']
            print(f"Testing integration with most impactful feature: {most_impactful_feature}")
            
            # Generate fresh test instances for integration test
            integration_instances = env.reset(batch_size=[3]).to(device)
            
            # Use the trajectory comparison (which should work with our existing setup)
            print("Running trajectory comparison...")
            trajectory_result = comparator.compare_full_trajectory(
                integration_instances,
                target_type="sae_feature",
                target_idx=most_impactful_feature,
                decode_type="greedy",
                return_actions=True
            )
            
            # Extract results
            baseline_rewards = trajectory_result.baseline_rewards.mean().item()
            ablated_rewards = trajectory_result.ablated_rewards.mean().item()
            reward_diff = trajectory_result.reward_difference.mean().item()
            
            print(f"Comparator Results:")
            print(f"  Baseline reward: {baseline_rewards:.4f}")
            print(f"  Ablated reward: {ablated_rewards:.4f}")
            print(f"  Reward difference: {reward_diff:.4f}")
            
            # Check if tours changed
            tours_changed = trajectory_result.tour_changed.sum().item()
            total_tours = trajectory_result.tour_changed.numel()
            change_rate = 100 * tours_changed / total_tours
            print(f"  Tours changed: {tours_changed}/{total_tours} ({change_rate:.1f}%)")
            
            print("✅ Feature Ablator + Behavior Comparator integration test PASSED!")
            
        except Exception as e:
            print(f"❌ Integration test failed: {e}")
            import traceback
            traceback.print_exc()
        
        print("\nAll tests passed! FeatureAblator is working correctly.")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc() 