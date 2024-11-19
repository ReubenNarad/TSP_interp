import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional
import copy

class TSP_env:
    """
    Represents an instance of the Traveling Salesman Problem (TSP).
    
    Attributes:
        instance_size (int): Number of nodes in the TSP instance.
        coords (List[Dict[str, int]]): List of node coordinates with 'id', 'x', and 'y'.
        tour (List[int]): The optimal tour sequence.
        state (np.ndarray): 100x100 array representing the state of each node.
        tour_so_far (List[int]): List of node IDs representing the tour taken so far.
        max_context_length (int): Fixed maximum number of nodes to consider.
    """
    
    # State encoding
    UNVISITED = 0
    VISITED = 1
    DEPOT = 2
    CURRENT = 3

    def __init__(self, data: Dict[str, Any], max_context_length: int):
        """
        Initializes the TSP_env with data from a single JSON line.
        
        Args:
            data (Dict[str, Any]): A dictionary containing 'instance_size', 'coords', and 'tour'.
            max_context_length (int): Fixed maximum number of nodes to consider.
        """
        self.instance_size = data['instance_size']
        self.coords = data['coords']
        self.tour = data['tour']
        self.max_context_length = max_context_length
        
        # Infer grid size from coordinates
        self.grid_size = max(
            max(node['x'] for node in self.coords),
            max(node['y'] for node in self.coords)
        ) + 1  # Add 1 since coordinates are 0-based
        
        # Initialize a grid of inferred size
        self.state = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)
        
        # Initialize all nodes as unvisited
        for node in self.coords:
            x, y = node['x'], node['y']
            self.state[x, y] = self.UNVISITED
        
        # Set the depot (node with id 0)
        initial_node = self.coords[0]
        self.state[initial_node['x'], initial_node['y']] = self.DEPOT

        # Set the current position to also be at the depot
        self.current_pos = {'x': initial_node['x'], 'y': initial_node['y']}
        
        # Initialize tour history
        self.tour_so_far = [initial_node['id']]

    def step(self, action: Dict[str, int]) -> None:
        """
        Takes an action by moving to the next node specified by the action.
        
        Args:
            action (Dict[str, int]): A dictionary with 'x' and 'y' coordinates of the next node.
        """
        action_x, action_y = action['x'], action['y']
        
        # Find the node corresponding to the action coordinates
        action_node = next((node for node in self.coords if node['x'] == action_x and node['y'] == action_y), None)
        if action_node is None:
            raise ValueError(f"No node found at coordinates ({action_x}, {action_y}).")
        
        action_id = action_node['id']
        
        # Update previous current position
        if self.state[self.current_pos['x'], self.current_pos['y']] == self.CURRENT:
            self.state[self.current_pos['x'], self.current_pos['y']] = self.VISITED
        
        # Update new position
        self.current_pos = {'x': action_x, 'y': action_y}
        self.state[action_x, action_y] = self.CURRENT
        
        # Append to tour history
        self.tour_so_far.append(action_id)


    def valid_moves_mask(self) -> List[Dict[str, int]]:
        """
        Creates a list of valid moves in the format {'x': x, 'y': y}.
        
        Returns:
            List[Dict[str, int]]: List of valid moves where each move is a dictionary 
            containing 'x' and 'y' coordinates.
            - If there are unvisited nodes, returns all unvisited nodes
            - If all nodes are visited except depot, returns only the depot
        """
        # Get all unvisited nodes (excluding depot)
        unvisited_moves = [
            {'x': node['x'], 'y': node['y']} 
            for node in self.coords 
            if self.state[node['x'], node['y']] == self.UNVISITED
        ]
        
        # If there are unvisited nodes, they are the only valid moves
        if unvisited_moves:
            return unvisited_moves
        
        # If all nodes are visited, only the depot is a valid move
        depot_node = self.coords[0]  # depot is always the first node
        return [{'x': depot_node['x'], 'y': depot_node['y']}]

    def format_state(self) -> tuple[torch.Tensor, torch.Tensor]:
        max_nodes = self.max_context_length
        sequence = []
        
        # Get current position
        current_x = self.current_pos['x']
        current_y = self.current_pos['y']
        
        # Calculate grid size once
        grid_size = max(max(node['x'] for node in self.coords), 
                       max(node['y'] for node in self.coords)) + 1
        
        # Add nodes in order: current, depot, then unvisited
        for node in self.coords:
            x, y = node['x'], node['y']
            is_current = (x == current_x and y == current_y)
            is_depot = (self.state[x, y] == self.DEPOT)
            
            # Calculate relative distances using actual grid size
            dist_to_current = np.sqrt((x - current_x)**2 + (y - current_y)**2) / grid_size
            
            # Prioritize current node and depot
            if is_current or is_depot:
                sequence.insert(0 if is_current else 1, (x, y, is_current, is_depot, dist_to_current))
        
        # Add unvisited nodes
        for node in self.coords:
            x, y = node['x'], node['y']
            if self.state[x, y] == self.UNVISITED:
                dist_to_current = np.sqrt((x - current_x)**2 + (y - current_y)**2) / grid_size
                sequence.append((x, y, False, False, dist_to_current))
        
        # Convert to tensor
        state_tensor = torch.zeros((max_nodes, 5), dtype=torch.float)
        attention_mask = torch.ones(max_nodes, dtype=torch.bool)
        
        # Fill tensors with sequence data
        for idx, (x, y, is_current, is_depot, dist_to_current) in enumerate(sequence[:max_nodes]):
            state_tensor[idx] = torch.tensor([
                x / grid_size,  # Normalize by grid_size (5) instead of 100
                y / grid_size,
                float(is_current),
                float(is_depot),
                dist_to_current
            ])
        
        # Set attention mask to False for padding positions
        if len(sequence) < max_nodes:
            attention_mask[len(sequence):] = False
        
        return state_tensor, attention_mask

    def reset(self) -> None:
        """
        Resets the environment to the initial state.
        """
        self.__init__({
            'instance_size': self.instance_size,
            'coords': self.coords,
            'tour': self.tour
        }, max_context_length=self.max_context_length)

    def is_done(self) -> bool:
        """
        Checks if all nodes have been visited.
        
        Returns:
            bool: True if all nodes are visited, False otherwise.
        """
        return len(self.tour_so_far) >= self.instance_size

    def get_current_node(self) -> Dict[str, int]:
        """
        Retrieves the current node's coordinates.
        
        Returns:
            Dict[str, int]: A dictionary with 'x' and 'y' of the current node.
        """
        current_node_id = self.tour_so_far[-1]
        current_node = next(node for node in self.coords if node['id'] == current_node_id)
        return {'x': current_node['x'], 'y': current_node['y']}
    
    def visualize(self, title=None) -> None:
        """
        Visualizes the current state of the TSP environment with dynamic grid sizing.
        """
        # Get min and max coordinates to show full grid
        min_x = 0  # Start at 0
        max_x = self.grid_size - 1  # grid_size is 5, so this will be 4
        min_y = 0
        max_y = self.grid_size - 1
        
        # Create grid of appropriate size
        grid_height = max_x - min_x + 1
        grid_width = max_y - min_y + 1
        grid = np.zeros((grid_height, grid_width, 3))
        
        # Define color mappings
        color_map = {
            'Current & Depot': [1, 0, 1],     # Purple
            'Current': [1, 0, 0],             # Red
            'Depot': [1, 1, 0],               # Yellow
            'Unvisited': [0.5, 0.5, 0.5],     # Gray
            'Visited': [0, 0, 0],             # Black
            'Optimal Next': [0, 0, 1],        # Blue
        }
        
        # Get optimal next node
        optimal_next = self.get_optimal_next_step()
        
        # First set all nodes to unvisited (gray)
        for node in self.coords:
            adj_x = node['x'] - min_x
            adj_y = node['y'] - min_y
            grid[adj_x, adj_y] = color_map['Unvisited']
        
        # Then update based on state
        for node in self.coords:
            adj_x = node['x'] - min_x
            adj_y = node['y'] - min_y
            
            if node['x'] == self.current_pos['x'] and node['y'] == self.current_pos['y'] and self.state[node['x'], node['y']] == self.DEPOT:
                color = color_map['Current & Depot']
            elif node['x'] == self.current_pos['x'] and node['y'] == self.current_pos['y']:
                color = color_map['Current']
            elif self.state[node['x'], node['y']] == self.DEPOT:
                color = color_map['Depot']
            elif self.state[node['x'], node['y']] == self.VISITED:
                color = color_map['Visited']
            elif node['x'] == optimal_next['x'] and node['y'] == optimal_next['y']:
                color = color_map['Optimal Next']
            else:
                color = color_map['Unvisited']
            grid[adj_x, adj_y] = color
        
        # Remove the plt.figure() call since we're using subplots
        plt.imshow(grid, origin='lower')
        plt.title(title if title else "TSP Environment Visualization")
        
        # Create custom legend (without Visited)
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=color_map['Current & Depot'], edgecolor='k', label='Current & Depot'),
            Patch(facecolor=color_map['Current'], edgecolor='k', label='Current'),
            Patch(facecolor=color_map['Depot'], edgecolor='k', label='Depot'),
            Patch(facecolor=color_map['Unvisited'], edgecolor='k', label='Unvisited'),
            Patch(facecolor=color_map['Optimal Next'], edgecolor='k', label='Optimal Next'),
        ]
        # plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(-0.15, 0.5))
        
        # Remove axis labels
        plt.axis('off')
        
        # Remove plt.show() since we'll show all subplots at once

    def get_optimal_next_step(self) -> Dict[str, int]:
        """
        Returns the coordinates of the next node in the optimal tour.
        
        Returns:
            Dict[str, int]: A dictionary with 'x' and 'y' coordinates of the next optimal node.
        """
        current_node_id = self.tour_so_far[-1]
        current_tour_index = self.tour.index(current_node_id)
        next_node_id = self.tour[(current_tour_index + 1) % self.instance_size]
        
        next_node = next(node for node in self.coords if node['id'] == next_node_id)
        return {'x': next_node['x'], 'y': next_node['y']}

    def get_node_position(self, node_id: int) -> Dict[str, int]:
        """
        Retrieves the position (x, y) of a node given its ID.
        
        Args:
            node_id (int): The ID of the node.
        
        Returns:
            Dict[str, int]: A dictionary with 'x' and 'y' coordinates.
        """
        node = next((node for node in self.coords if node['id'] == node_id), None)
        if node is None:
            raise ValueError(f"Node with ID {node_id} not found.")
        return {'x': node['x'], 'y': node['y']}

    def get_optimal_target_idx(self, step: int, grid_size: int = 100) -> int:
        """
        Gets the index of the optimal next position for a given step.
        
        Args:
            step (int): Current step in the tour.
            grid_size (int): Size of the grid (default 100).
        
        Returns:
            int: Index in range [0, grid_size^2) or -1 if step is invalid
        """
        if step >= len(self.tour):
            return -1
        
        next_node_id = self.tour[step]
        next_node = self.get_node_position(next_node_id)
        return next_node['x'] * grid_size + next_node['y']

    def clone(self):
        """Creates a deep copy of the environment."""
        return copy.deepcopy(self)



# Test code
if __name__ == "__main__":
    # Load the first instance from the dataset
    with open('data/dataset_5_rotated.jsonl', 'r') as f:
        # Read tenth line
        for _ in range(3):
            f.readline()
        tenth_instance = json.loads(f.readline())
    
    # Create environment with fixed max_context_length
    fixed_max_context_length = 50
    env = TSP_env(tenth_instance, max_context_length=fixed_max_context_length)
    
    # Step 0 (initial state)
    env.visualize(title='Step 0')
    plt.savefig('step0.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    # Step 1
    optimal_next = env.get_optimal_next_step()
    env.step(optimal_next)
    env.visualize(title='Step 1')
    plt.savefig('step1.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    # Step 5
    for _ in range(4):  # 4 more steps to reach step 5
        optimal_next = env.get_optimal_next_step()
        env.step(optimal_next)
    env.visualize(title='Step 5')
    plt.savefig('step5.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    # Step 15
    for _ in range(10):  # 10 more steps to reach step 15
        optimal_next = env.get_optimal_next_step()
        env.step(optimal_next)
    env.visualize(title='Step 15')
    plt.savefig('step15.png', bbox_inches='tight', dpi=300)
    plt.close()