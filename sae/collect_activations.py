import os
import glob
import json
import pickle
import argparse
from typing import Dict, List, Optional, Tuple

import torch
import numpy as np
from tqdm import tqdm
from torchrl.data import Composite

# Ensure torch.load defaults to weights_only=False for trusted legacy checkpoints
os.environ.setdefault("TORCH_LOAD_WEIGHTS_ONLY", "0")
_original_torch_load = torch.load
def _torch_load_with_weights(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_torch_load(*args, **kwargs)
torch.load = _torch_load_with_weights

# Fix imports from parent directory
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from policy.policy_hooked import HookedAttentionModelPolicy
from policy.matnet_hooked import EnhancedHookedMatNetPolicy
from policy.reinforce_clipped import REINFORCEClipped

class EnhancedHookedPolicy(HookedAttentionModelPolicy):
    """Enhanced hooked policy that captures encoder-decoder interface activations"""
    
    def _setup_hooks(self):
        """Setup hooks for encoder layers and encoder-decoder interface"""
        # First, set up the regular encoder layer hooks
        super()._setup_hooks()
        
        # Now add a hook for the encoder-decoder interface
        def encoder_output_hook(module, input, output):
            # Make sure we're storing just the tensor, not a tuple
            self.activation_cache['encoder_output'] = output
            
        def decoder_input_hook(module, input, output):
            # Store the first input tensor to the decoder
            if isinstance(input, tuple) and len(input) > 0:
                self.activation_cache['decoder_input'] = input[0]  
            else:
                print(f"Unexpected input type to decoder: {type(input)}")
                self.activation_cache['decoder_input'] = None
            
        # Register hooks on the encoder module's output and decoder's input
        encoder_handle = self.encoder.register_forward_hook(encoder_output_hook)
        self.hooks['encoder_output'] = encoder_handle
        
        decoder_handle = self.decoder.register_forward_hook(decoder_input_hook)
        self.hooks['decoder_input'] = decoder_handle

def collect_activations(args):
    """Collect activations from a model checkpoint"""
    # Construct full run path from run name
    run_path = os.path.join("./runs", args.run_name)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Silence warnings if requested
    if args.silence_warnings:
        import warnings
        warnings.filterwarnings("ignore", message="Found keys that are not in the model state dict")
        warnings.filterwarnings("ignore", message="Attribute .* is an instance of `nn.Module`")
        warnings.filterwarnings("ignore", message="You are using `torch.load` with `weights_only=False`")
        
    # Helper function to extract epoch number from checkpoint path
    def extract_epoch(cp_path):
        fname = os.path.basename(cp_path)
        epoch_str = fname.split("checkpoint_epoch_")[1].split(".ckpt")[0]
        return int(epoch_str)
    
    # Load environment and configuration
    config_path = os.path.join(run_path, "config.json")
    with open(config_path, "r") as f:
        config = json.load(f)
    
    # Get values from config instead of args
    num_instances = config.get('num_instances', 1000)
    checkpoint_freq = config.get('checkpoint_freq', 10)
    num_epochs = config.get('num_epochs', 100)  # For finding final checkpoint
    
    env_path = os.path.join(run_path, "env.pkl")
    with open(env_path, "rb") as f:
        env = pickle.load(f)

    def _patch_spec(spec):
        """Ensure Composite specs have newer attributes for compatibility with old pickles."""
        if isinstance(spec, Composite):
            if not hasattr(spec, "data_cls"):
                spec.data_cls = None
            if not hasattr(spec, "step_mdp_static"):
                spec.step_mdp_static = False
            for child in spec.values():
                if child is not None:
                    _patch_spec(child)

    for spec_name in ["input_spec", "output_spec", "observation_spec", "reward_spec"]:
        spec = getattr(env, spec_name, None)
        if spec is not None:
            _patch_spec(spec)
    
    # Create the output directory inside the run directory
    activations_dir = os.path.join(run_path, "sae", "activations")
    os.makedirs(activations_dir, exist_ok=True)
    
    # Load validation data to use as inputs
    val_td_path = os.path.join(run_path, "val_td.pkl")
    with open(val_td_path, "rb") as f:
        val_td = pickle.load(f)
    
    # Default number of instances for activation collection - much higher for SAE training
    default_activation_instances = 10000  # 10K instances by default

    # Determine number of instances to use
    if args.num_instances:  # If user-specified, use that value
        instances_to_use = args.num_instances
    else:  # Otherwise use a higher default for SAE training
        instances_to_use = default_activation_instances
    
    print(f"Generating {instances_to_use} instances for activation collection...")
    
    # Get all checkpoints
    checkpoint_dir = os.path.join(run_path, "checkpoints")
    
    if args.checkpoint is not None:
        # Use a specific checkpoint based on epoch number
        specific_checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{args.checkpoint}.ckpt")
        if os.path.exists(specific_checkpoint_path):
            checkpoint_paths = [specific_checkpoint_path]
        else:
            raise ValueError(f"Checkpoint for epoch {args.checkpoint} not found at {specific_checkpoint_path}")
    else:
        # Determine if we should use a specific checkpoint (final) or all of them
        final_only = args.final_only
        
        if final_only:
            # Use only the final checkpoint based on num_epochs in config
            final_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{num_epochs}.ckpt")
            if os.path.exists(final_path):
                checkpoint_paths = [final_path]
            else:
                # If the exact epoch doesn't exist, find the latest checkpoint
                all_checkpoints = glob.glob(os.path.join(checkpoint_dir, "checkpoint_epoch_*.ckpt"))
                if not all_checkpoints:
                    raise ValueError(f"No checkpoints found in {checkpoint_dir}")
                checkpoint_paths = [max(all_checkpoints, key=extract_epoch)]
        else:
            # Use all checkpoints in the directory
            checkpoint_paths = glob.glob(os.path.join(checkpoint_dir, "checkpoint_epoch_*.ckpt"))
            
            checkpoint_paths = sorted(checkpoint_paths, key=extract_epoch)
            
            # Apply checkpoint frequency from config
            if checkpoint_freq > 1:
                checkpoint_paths = [cp for i, cp in enumerate(checkpoint_paths) 
                                  if extract_epoch(cp) % checkpoint_freq == 0]
    
    print(f"Found {len(checkpoint_paths)} checkpoints to process.")
    
    # Decide which hooked policy family to use based on the env/config.
    # For our repo, MatNet runs use ATSPEnv and provide a cost_matrix (pool/tsplib).
    is_matnet = (getattr(env, "name", None) == "atsp") and (
        config.get("pool_dir") is not None or config.get("tsplib_path") is not None or config.get("env_name") == "atsp"
    )

    # Process each checkpoint
    for cp_path in tqdm(checkpoint_paths):
        # Create an enhanced policy model (AttentionModel or MatNet)
        if is_matnet:
            policy = EnhancedHookedMatNetPolicy(
                env_name=env.name,
                embed_dim=int(config["embed_dim"]),
                num_encoder_layers=int(config["n_encoder_layers"]),
                num_heads=int(config.get("num_heads", 8)),
                normalization=str(config.get("normalization", "instance")),
                use_graph_context=bool(config.get("use_graph_context", False)),
                bias=bool(config.get("bias", False)),
                init_embedding_mode=str(config.get("init_embedding_mode", "random_onehot")),
                tanh_clipping=float(config.get("tanh_clipping", 10.0)),
                temperature=float(config.get("temperature", 1.0)),
            )
        else:
            policy = EnhancedHookedPolicy(
                env_name=env.name,
                embed_dim=config["embed_dim"],
                num_encoder_layers=config["n_encoder_layers"],
                num_heads=int(config.get("num_heads", 8)),
                temperature=config["temperature"],
            )
        
        # Load checkpoint
        checkpoint_name = os.path.basename(cp_path)
        epoch = extract_epoch(cp_path)
        print(f"\nProcessing checkpoint for epoch {epoch}...")
        
        model = REINFORCEClipped.load_from_checkpoint(
            cp_path,
            env=env,
            policy=policy,
            strict=False
        )
        model.eval()
        
        # Debug: print hook points for visualization
        print("\nHook attachment points:")
        print(f"Policy:  {type(model.policy)}")
        print(f"Encoder: {type(model.policy.encoder)}")
        print(f"Decoder: {type(model.policy.decoder)}")
        
        # Get policy from loaded model
        policy = model.policy.to(device)
        
        accumulated = {}
        total_instances = 0
        chunk_size = args.chunk_size if hasattr(args, "chunk_size") and args.chunk_size else instances_to_use
        remaining = instances_to_use
        num_nodes = None
        
        while remaining > 0:
            current = min(chunk_size, remaining)
            data = env.reset(batch_size=[current]).to(device)
            if num_nodes is None:
                if "locs" in data.keys(True, True):
                    num_nodes = data["locs"].shape[1]
                elif "cost_matrix" in data.keys(True, True):
                    num_nodes = data["cost_matrix"].shape[1]
                else:
                    raise KeyError(f"Could not infer num_nodes from td keys: {list(data.keys(True, True))}")
            
            with torch.no_grad():
                _ = policy(data, decode_type="greedy")
            
            for name, activation in policy.activation_cache.items():
                if activation is None:
                    continue
                
                def _flatten_and_store(tensor, key):
                    if len(tensor.shape) != 3:
                        print(f"Warning: Activation {key} has shape {tensor.shape}, expected 3D tensor. Skipping.")
                        return
                    flat = tensor.reshape(-1, tensor.shape[-1]).cpu()
                    accumulated.setdefault(key, []).append(flat)
                
                if name.startswith('encoder_layer_') or name.startswith('encoder_output') or name == 'decoder_input':
                    if isinstance(activation, tuple):
                        if len(activation) == 2:
                            first_tensor = activation[0]
                            _flatten_and_store(first_tensor, name)
                            if args.store_attention_weights:
                                _flatten_and_store(activation[1], f"{name}_attn_weights")
                        else:
                            print(f"Warning: Unexpected tuple length {len(activation)} for {name}")
                    else:
                        _flatten_and_store(activation, name)
            
            total_instances += current
            remaining -= current
            torch.cuda.empty_cache()
        
        activations = {name: torch.cat(chunks, dim=0) for name, chunks in accumulated.items()}
        
        activation_path = os.path.join(activations_dir, f"activations_epoch_{epoch}.pt")
        torch.save(activations, activation_path)
        print(f"Saved activations to {activation_path}")
        
        metadata = {
            'epoch': epoch,
            'shapes': {name: tensor.shape for name, tensor in activations.items()},
            'model_config': config,
            'num_instances': total_instances,
            'num_nodes': num_nodes if num_nodes is not None else 0,
        }
        
        metadata_path = os.path.join(activations_dir, f"metadata_epoch_{epoch}.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        del activations, accumulated, model, policy
        torch.cuda.empty_cache()
    
    print(f"Activations saved to {activations_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Collect activations from model checkpoints for SAE training"
    )
    parser.add_argument(
        "--run_name",
        type=str,
        required=True,
        help="Name of the run (folder name in ./runs/)"
    )
    parser.add_argument(
        "--checkpoint",
        type=int,
        default=None,
        help="Specific epoch number to process (if provided, only this checkpoint will be used)"
    )
    parser.add_argument(
        "--final_only",
        action="store_true",
        help="Only process the final checkpoint (based on num_epochs in config)"
    )
    parser.add_argument(
        "--num_instances",
        type=int,
        default=None,
        help="Number of instances to process (overrides config.json if specified)"
    )
    parser.add_argument(
        "--store_attention_weights",
        action="store_true",
        help="Store attention weights in addition to node embeddings"
    )
    parser.add_argument(
        "--silence_warnings",
        action="store_true",
        help="Silence warnings about keys not in model state dict"
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=None,
        help="Number of instances to process per forward pass (reduces GPU memory usage). Defaults to all instances."
    )
    
    args = parser.parse_args()
    collect_activations(args)
