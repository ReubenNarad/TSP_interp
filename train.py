# NOTE IN THE MORNING:
# This run is experimenting with both weight decay (adamw), lower lr, and more epochs

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from typing import Dict, Any, List
import os
from tqdm import tqdm
import argparse
import json
import datetime
import shutil
import matplotlib.pyplot as plt

from net import TSPModel
from dataset import TSPDataset, collate_fn
from TSP_env import TSP_env

class TSPTrainer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config

        # Create datasets first to infer grid size
        dataset = TSPDataset(config['data_path'], config['max_context_length'])
        self.grid_size = dataset.grid_size
        
        # Split dataset
        val_size = int(len(dataset) * config['val_split'])
        train_size = len(dataset) - val_size
        self.train_dataset, self.val_dataset = random_split(
            dataset, [train_size, val_size]
        )

        # Create unique run directory with lr and instance size
        timestamp = datetime.datetime.now().strftime("%m-%d_%H:%M")
        # Extract instance size from data path (assuming format like 'dataset_20_sample.jsonl')
        instance_size = os.path.basename(config['data_path']).split('_')[1]
        run_name = f"{self.grid_size}x{self.grid_size}_TSP{instance_size}_n_{len(self.train_dataset)}_{timestamp}"
        self.run_dir = os.path.join('runs', run_name)
        os.makedirs(self.run_dir, exist_ok=True)
        
        # Save config file
        config_path = os.path.join(self.run_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
            
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model with inferred grid size
        gpt2_config = {
            'n_positions': config['max_context_length'] + 10,
            'n_embd': config['hidden_dim'],
            'n_layer': config.get('n_layers', 12),
            'n_head': config.get('n_heads', 12),
            'resid_pdrop': 0.05,
            'embd_pdrop': 0.05,
            'attn_pdrop': 0.05,
            'use_cache': False,
        }
        self.model = TSPModel(
            hidden_dim=config['hidden_dim'],
            grid_size=self.grid_size,  # Use inferred grid size
            max_context_length=config['max_context_length'],
            gpt2_config=gpt2_config
        ).to(self.device)
        
        # Initialize optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config['lr'],
            weight_decay=config.get('weight_decay', 1e-5),  # Default to 1e-5 if not specified
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Initialize loss function
        self.criterion = nn.CrossEntropyLoss(reduction='sum')
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            collate_fn=collate_fn
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            collate_fn=collate_fn
        )

    def train_epoch(self) -> Dict[str, float]:
        """Trains the model for one epoch and returns average loss and accuracy."""
        self.model.train()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        num_batches = 0
        
        for batch in tqdm(self.train_loader, desc="Batch"):
            # Move batch to device
            coords = batch['coords'].to(self.device)
            states = batch['states'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            indices = batch['indices'].to(self.device)
            
            # Get environments for this batch
            envs = [self.train_dataset.dataset.envs[idx.item()] for idx in indices]
            batch_size = len(envs)
            
            # Initialize trajectory loss
            trajectory_loss = 0
            
            # Create copies of the environments for trajectory rollout
            env_copies = [env.clone() for env in envs]
            
            # Store initial tensors
            current_coords = coords.clone()
            current_states = states.clone()
            current_attention_mask = attention_mask.clone()
            
            # Zero gradients at the start of trajectory
            self.optimizer.zero_grad()
            
            # For each step in the trajectory
            for step in tqdm(range(len(envs[0].tour) - 1), desc="Horizon", leave=False):
                # Get model predictions for current state
                logits = self.model(current_coords, current_states, current_attention_mask)
                
                # Get targets for the current positions
                targets = torch.zeros(batch_size, dtype=torch.long, device=self.device)
                for i, env in enumerate(env_copies):
                    if step < len(env.tour) - 1:
                        next_node_id = env.tour[step + 1]
                        next_pos = env.get_node_position(next_node_id)
                        targets[i] = next_pos['x'] * self.grid_size + next_pos['y']
                
                # Get output mask for valid moves
                output_mask = self._get_output_mask(envs, step)  # [batch_size, grid_size * grid_size]

                # Apply mask to logits by setting invalid moves to large negative value
                invalid_moves = ~output_mask.bool()
                logits = logits.masked_fill(invalid_moves, float('-inf'))

                # Get active batch elements (environments that still have steps)
                active_envs = torch.tensor([step < len(env.tour) - 1 for env in envs], 
                                          device=self.device)

                if active_envs.any():
                    # Calculate loss for active environments only
                    masked_logits = logits[active_envs]
                    masked_targets = targets[active_envs]
                    
                    # Calculate loss and normalize by total number of valid moves
                    step_loss = self.criterion(masked_logits, masked_targets)
                    num_valid_moves = output_mask[active_envs].sum()
                    step_loss = step_loss / num_valid_moves
                    trajectory_loss += step_loss
                    
                    # Calculate accuracy
                    predictions = masked_logits.argmax(dim=1)
                    correct_predictions += (predictions == masked_targets).sum().item()
                    total_predictions += masked_targets.size(0)
                
                # Update environments and create new state tensors
                next_coords = []
                next_states = []
                next_attention_masks = []
                
                for i, env in enumerate(env_copies):
                    if step < len(env.tour) - 1:
                        next_node_id = env.tour[step + 1]
                        next_pos = env.get_node_position(next_node_id)
                        env.step({'x': next_pos['x'], 'y': next_pos['y']})
                        
                        # Get new state tensors with distance features
                        state_tensor, mask = env.format_state()
                        next_coords.append(state_tensor[:, :2])  # x,y coords
                        next_states.append(state_tensor[:, 2:])  # current, depot, distance features
                        next_attention_masks.append(mask)
                
                # Stack new tensors
                current_coords = torch.stack(next_coords).to(self.device)
                current_states = torch.stack(next_states).to(self.device)
                current_attention_mask = torch.stack(next_attention_masks).to(self.device)
            
            # Average loss over trajectory steps and backpropagate
            trajectory_loss = trajectory_loss / (len(envs[0].tour) - 1) if (len(envs[0].tour) - 1) > 0 else 0.0
            trajectory_loss.backward()
            self.optimizer.step()
            
            total_loss += trajectory_loss.item()
            num_batches += 1
            
            # Reset original environments for next batch
            for env in envs:
                env.reset()
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
        
        return {
            'train_loss': avg_loss,
            'train_accuracy': accuracy
        }

    def _get_output_mask(self, envs: List[TSP_env], step: int) -> torch.Tensor:
        """
        Generates an output mask for the current step using valid moves from each environment.
        
        Args:
            envs (List[TSP_env]): List of environment instances.
            step (int): Current step in the trajectory.
        
        Returns:
            torch.Tensor: A tensor of shape [batch_size, grid_size * grid_size] containing 1s for valid moves
                         and 0s for invalid moves.
        """
        batch_size = len(envs)
        mask = torch.zeros((batch_size, self.grid_size * self.grid_size), device=self.device)
        
        for i, env in enumerate(envs):
            if step < len(env.tour) - 1:
                # Get valid moves from environment
                valid_moves = env.valid_moves_mask()
                for move in valid_moves:
                    idx = move['x'] * self.grid_size + move['y']
                    mask[i, idx] = 1
        
        return mask

    def validate(self) -> Dict[str, float]:
        """Validates the model and returns metrics."""
        self.model.eval()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.val_loader, desc="Validating")):
                # Move batch to device
                coords = batch['coords'].to(self.device)
                states = batch['states'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                indices = batch['indices'].to(self.device)
                
                # Get environments for this batch
                envs = [self.val_dataset.dataset.envs[idx.item()] for idx in indices]
                batch_size = len(envs)
                
                # Initialize trajectory loss and prediction counts
                trajectory_loss = 0
                
                # Create copies of the environments for trajectory rollout
                env_copies = [env.clone() for env in envs]
                
                # Store initial tensors
                current_coords = coords.clone()
                current_states = states.clone()
                current_attention_mask = attention_mask.clone()
                
                # For each step in the trajectory
                for step in tqdm(range(len(envs[0].tour) - 1), desc="Horizon", leave=False):
                    # Get model predictions for current state
                    logits = self.model(current_coords, current_states, current_attention_mask)
                    
                    # Get targets for the current positions
                    targets = torch.zeros(batch_size, dtype=torch.long, device=self.device)
                    for i, env in enumerate(env_copies):
                        if step < len(env.tour) - 1:
                            next_node_id = env.tour[step + 1]
                            next_pos = env.get_node_position(next_node_id)
                            targets[i] = next_pos['x'] * self.grid_size + next_pos['y']
                    
                    # Get output mask for valid moves
                    output_mask = self._get_output_mask(envs, step)  # [batch_size, grid_size * grid_size]

                    # Apply mask to logits by setting invalid moves to large negative value
                    invalid_moves = ~output_mask.bool()
                    logits = logits.masked_fill(invalid_moves, float('-inf'))

                    # Get active batch elements (environments that still have steps)
                    active_envs = torch.tensor([step < len(env.tour) - 1 for env in envs], 
                                              device=self.device)

                    if active_envs.any():
                        # Calculate loss for active environments only
                        masked_logits = logits[active_envs]
                        masked_targets = targets[active_envs]
                        
                        # Calculate loss and normalize by total number of valid moves
                        step_loss = self.criterion(masked_logits, masked_targets)
                        num_valid_moves = output_mask[active_envs].sum()
                        step_loss = step_loss / num_valid_moves
                        trajectory_loss += step_loss
                        
                        # Calculate accuracy
                        predictions = masked_logits.argmax(dim=1)
                        correct_predictions += (predictions == masked_targets).sum().item()
                        total_predictions += masked_targets.size(0)
                    
                    # Update environments and create new state tensors
                    next_coords = []
                    next_states = []
                    next_attention_masks = []
                    
                    for i, env in enumerate(env_copies):
                        if step < len(env.tour) - 1:
                            next_node_id = env.tour[step + 1]
                            next_pos = env.get_node_position(next_node_id)
                            env.step({'x': next_pos['x'], 'y': next_pos['y']})
                            
                            # Get new state tensors with distance features
                            state_tensor, mask = env.format_state()
                            next_coords.append(state_tensor[:, :2])  # x,y coords
                            next_states.append(state_tensor[:, 2:])  # current, depot, distance features
                            next_attention_masks.append(mask)
                    
                    # Stack new tensors
                    if next_coords:
                        current_coords = torch.stack(next_coords).to(self.device)
                        current_states = torch.stack(next_states).to(self.device)
                        current_attention_mask = torch.stack(next_attention_masks).to(self.device)
                    else:
                        break  # No more steps to process
                
                # Average loss over trajectory steps
                avg_trajectory_loss = trajectory_loss / (len(envs[0].tour) - 1) if (len(envs[0].tour) - 1) > 0 else 0.0
                total_loss += avg_trajectory_loss
                
                # Reset original environments for next batch
                for env in envs:
                    env.reset()
        
        avg_loss = total_loss / len(self.val_loader) if len(self.val_loader) > 0 else 0.0
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
        
        return {
            'val_loss': avg_loss,
            'val_accuracy': accuracy
        }

    def _get_optimal_targets(self, envs: List[TSP_env], step: int) -> torch.Tensor:
        """
        Gets optimal target indices for a batch of environments.
        
        Args:
            envs (List[TSP_env]): List of TSP_env instances in the batch.
            step (int): Current step in the tour.
            
        Returns:
            torch.Tensor: Tensor of shape [batch_size] with target indices
        """
        batch_size = len(envs)
        targets = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        
        for i, env in enumerate(envs):
            targets[i] = env.get_optimal_target_idx(step, self.grid_size)
        
        return targets

    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], path: str):
        """Saves model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'config': self.config  # Also save config with checkpoint
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str):
        """Loads model checkpoint."""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['epoch'], checkpoint['metrics']

    def train(self):
        """Runs the full training loop for the specified number of epochs."""
        training_losses = []
        training_accuracies = []
        val_losses = []
        val_accuracies = []

        for epoch in range(1, self.config['num_epochs'] + 1):
            print(f"\nEpoch {epoch}/{self.config['num_epochs']}")
            
            # Train
            train_metrics = self.train_epoch()
            training_losses.append(train_metrics['train_loss'])
            training_accuracies.append(train_metrics['train_accuracy'])
            print(f"Train Loss: {train_metrics['train_loss']:.4f}")
            
            # Validate
            val_metrics = self.validate()
            val_losses.append(val_metrics['val_loss'])
            val_accuracies.append(val_metrics['val_accuracy'])
            print(f"Val Loss: {val_metrics['val_loss']:.4f}")
            print(f"Val Accuracy: {val_metrics['val_accuracy']:.4f}")
            
            # Save checkpoint every 5 epochs
            if epoch % 1 == 0 or epoch == self.config['num_epochs']:
                checkpoint_path = os.path.join(
                    self.run_dir,
                    f"checkpoint_epoch_{epoch}.pt"
                )
                self.save_checkpoint(epoch, val_metrics, checkpoint_path)
                print(f"Checkpoint saved at {checkpoint_path}")
            
            # Save plot every 10 epochs or on final epoch
            if epoch % 10 == 0 or epoch == self.config['num_epochs']:
                # Create figure with two y-axes
                fig, ax1 = plt.subplots(figsize=(10, 5))
                ax2 = ax1.twinx()
                
                # Plot losses on the first y-axis
                ln1 = ax1.plot(range(1, epoch + 1), training_losses, 
                              label='Training Loss', color='blue')
                ln2 = ax1.plot(range(1, epoch + 1), val_losses, 
                              label='Validation Loss', color='green')
                ax1.set_xlabel('Epoch')
                ax1.set_ylabel('Loss')
                
                # Plot accuracies on the second y-axis
                ln3 = ax2.plot(range(1, epoch + 1), training_accuracies, 
                               label='Training Accuracy', color='orange')
                ln4 = ax2.plot(range(1, epoch + 1), val_accuracies, 
                               label='Validation Accuracy', color='red')
                ax2.set_ylabel('Accuracy')
                
                # Combine legends
                lns = ln1 + ln2 + ln3 + ln4
                labs = [l.get_label() for l in lns]
                ax1.legend(lns, labs, loc='upper right')
                plt.title(f"Training and Validation Loss (Batch size: {self.config['batch_size']}, LR: {self.config['lr']})")
                plt.grid(True)

                # Save plot and close figure
                plot_path = os.path.join(self.run_dir, f"training_plot_epoch_{epoch}.png")
                plt.savefig(plot_path)
                plt.close()

                # Save metrics as JSON
                metrics = {
                    'training_losses': [float(loss) for loss in training_losses],
                    'training_accuracies': [float(acc) for acc in training_accuracies],
                    'val_losses': [float(loss) for loss in val_losses],
                    'val_accuracies': [float(acc) for acc in val_accuracies]
                }
                metrics_path = os.path.join(self.run_dir, f"metrics_epoch_{epoch}.json")
                with open(metrics_path, 'w') as f:
                    json.dump(metrics, f, indent=4)

def main():
    parser = argparse.ArgumentParser(description='Train TSP Solver')
    parser.add_argument('--data_path', type=str, required=True, help='Path to data file')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--val_split', type=float, default=0.2, help='Validation split ratio')
    parser.add_argument('--plot', type=bool, default=False, help='Plot training and validation losses')
    parser.add_argument('--max_context_length', type=int, required=True, help='Fixed maximum context length')
    parser.add_argument('--n_layers', type=int, default=6, help='Number of transformer layers')
    parser.add_argument('--n_heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay')
    args = parser.parse_args()

    # Training configuration
    config = {
        'hidden_dim': 64,
        'lr': args.lr,
        'batch_size': args.batch_size,
        'num_epochs': args.num_epochs,
        'data_path': args.data_path,
        'val_split': args.val_split,
        'plot': args.plot,
        'max_context_length': args.max_context_length,
        'n_layers': args.n_layers,
        'n_heads': args.n_heads,
        'weight_decay': args.weight_decay,
        'use_distance_features': True,
        'distance_embedding_dim': 128 // 4,
    }
     
    # Initialize trainer (no need to create checkpoint dir, it's handled in __init__)
    trainer = TSPTrainer(config)
    
    # Start training
    trainer.train()

if __name__ == "__main__":
    # Define a small test configuration
    main()
    # test_config = {
    #     'data_path': 'dataset_5_sample_small.jsonl',  # Use a small dataset for quick testing
    #     'batch_size': 2,  # Small batch size for quick iteration
    #     'num_epochs': 1,  # Only one epoch to test the setup
    #     'lr': 1e-3,  # A reasonable learning rate for testing
    #     'val_split': 0.2,  # Validation split
    #     'plot': False,  # Disable plotting for quick tests
    #     'max_context_length': 5,  # Match the dataset's instance size
    #     'n_layers': 2,  # Fewer layers for faster testing
    #     'n_heads': 2,  # Fewer heads for faster testing
    #     'weight_decay': 1e-5,  # Small weight decay
    #     'use_distance_features': True,
    #     'distance_embedding_dim': 32,  # Adjusted for smaller hidden_dim
    #     'hidden_dim': 64,
    # }

    # # Initialize the trainer with the test configuration
    # trainer = TSPTrainer(test_config)

    # # Run a single training epoch to test the setup
    # print("\nRunning a single training epoch for testing...")
    # trainer.train_epoch()

    # # Run a single validation step to test the setup
    # print("\nRunning a single validation step for testing...")
    # val_metrics = trainer.validate()
    # print(f"Validation Loss: {val_metrics['val_loss']}")
    # print(f"Validation Accuracy: {val_metrics['val_accuracy']}")

    # # Optionally, you can also test saving and loading checkpoints
    # checkpoint_path = 'test_checkpoint.pt'
    # trainer.save_checkpoint(1, val_metrics, checkpoint_path)
    # print(f"Checkpoint saved at {checkpoint_path}")