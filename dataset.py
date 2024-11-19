# This will have a class for the dataset, which will read the jsonl file and create a dataset of TSP_envs.
# It should have methods for accessing the data in batches

# dataset.py

import json
from typing import List, Dict, Any
import torch
from torch.utils.data import Dataset, DataLoader
from TSP_env import TSP_env

class TSPDataset(Dataset):
    """
    PyTorch Dataset for Traveling Salesman Problem (TSP) instances.

    This dataset reads TSP instances from a JSONL file, initializes TSP_env environments,
    and provides methods to access data samples suitable for training the TSPModel.

    Attributes:
        file_path (str): Path to the JSONL file containing TSP instances.
        envs (List[TSP_env]): List of TSP_env instances.
        max_context_length (int): Fixed maximum context length for all samples.
    """

    def __init__(self, file_path: str, max_context_length: int):
        """
        Initializes the TSPDataset by loading TSP instances from a JSONL file.

        Args:
            file_path (str): Path to the JSONL file containing TSP instances.
            max_context_length (int): Fixed maximum number of nodes per sample.
        """
        self.file_path = file_path
        self.max_context_length = max_context_length
        self.envs = self._load_envs()
        
        # Infer grid size from the data
        self.grid_size = self._infer_grid_size()
        
    def _load_envs(self) -> List[TSP_env]:
        """
        Loads TSP environments from the JSONL file.

        Returns:
            List[TSP_env]: A list of TSP_env instances.
        """
        envs = []
        with open(self.file_path, 'r') as f:
            for line in f:
                data = json.loads(line.strip())
                env = TSP_env(data, max_context_length=self.max_context_length)  # Pass the parameter
                envs.append(env)
        return envs

    def _infer_grid_size(self) -> int:
        """Infers grid size by finding the maximum x and y coordinates."""
        max_x = max_y = 0
        for env in self.envs:
            for node in env.coords:
                max_x = max(max_x, node['x'])
                max_y = max(max_y, node['y'])
        return max(max_x, max_y) + 1  # Add 1 since coordinates are 0-based

    def __len__(self) -> int:
        """
        Returns the number of TSP instances in the dataset.

        Returns:
            int: Number of TSP instances.
        """
        return len(self.envs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Gets a single item from the dataset.
        
        Args:
            idx (int): Index of the item to get.
            
        Returns:
            Dict[str, torch.Tensor]: Dictionary containing:
                - coords: tensor of shape [max_context_length, 2]
                - states: tensor of shape [max_context_length, 3]
                - attention_mask: tensor of shape [max_context_length]
                - index: tensor containing original index in dataset
        """
        env = self.envs[idx]
        state_tensor, attention_mask = env.format_state()
        
        return {
            'coords': state_tensor[:, :2],         # x,y coordinates
            'states': state_tensor[:, 2:],         # current, depot, distance
            'attention_mask': attention_mask,
            'index': idx
        }

def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function to handle batching of fixed-sized TSP instances.

    Ensures that all samples in the batch conform to the fixed max_context_length.

    Args:
        batch (List[Dict[str, Any]]): List of samples from TSPDataset.

    Returns:
        Dict[str, torch.Tensor]: Batched tensors.
            - 'coords': [batch_size, max_context_length, 2]
            - 'states': [batch_size, max_context_length, 3]
            - 'attention_mask': [batch_size, max_context_length]
            - 'indices': [batch_size]
    """
    batch_size = len(batch)
    max_context_length = batch[0]['coords'].size(0)
    coords_batch = torch.zeros((batch_size, max_context_length, 2), dtype=torch.float)
    states_batch = torch.zeros((batch_size, max_context_length, 3), dtype=torch.float)
    attention_mask_batch = torch.zeros((batch_size, max_context_length), dtype=torch.bool)
    indices_batch = torch.zeros(batch_size, dtype=torch.long)

    for i, sample in enumerate(batch):
        coords_batch[i] = sample['coords']
        states_batch[i] = sample['states']
        attention_mask_batch[i] = sample['attention_mask']
        indices_batch[i] = sample['index']

    return {
        'coords': coords_batch,
        'states': states_batch,
        'attention_mask': attention_mask_batch,
        'indices': indices_batch
    }

if __name__ == "__main__":
    # Create dataset instance with fixed max_context_length
    fixed_max_context_length = 50
    dataset = TSPDataset('dataset_20_sample.jsonl', max_context_length=fixed_max_context_length)
    print(f"Dataset size: {len(dataset)} instances")
    
    # Get first sample
    first_sample = dataset[0]
    print("\nFirst sample structure:")
    for key, value in first_sample.items():
        print(f"{key} shape: {value.shape}" if isinstance(value, torch.Tensor) else f"{key}: {value}")
    print(first_sample['coords'])
    print(first_sample['states'])

    # Print the valid moves in the first sample
    env = dataset.envs[0]
    valid_moves = env.valid_moves_mask()
    print("\nValid moves for first sample:")
    print(f"Number of valid moves: {len(valid_moves)}")
    print("Valid move coordinates:")
    for move in valid_moves:
        print(f"x: {move['x']}, y: {move['y']}")

    # Print the optimal next step in the first sample
    optimal_next = env.get_optimal_next_step()
    print(f"\nOptimal next step for first sample: {optimal_next}")
    # take a step
    env.step(optimal_next)
    print(env.valid_moves_mask())

    # Create a small dataloader
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
    
    # Get first batch
    batch = next(iter(dataloader))
    print("\nBatch structure:")
    for key, value in batch.items():
        print(f"{key} shape: {value.shape}")
