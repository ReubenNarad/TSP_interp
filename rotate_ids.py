import json
import random

def rotate_ids(instance, rotation=None):
    """
    Rotate which physical locations correspond to which IDs, while keeping the same
    physical path through space.
    """
    n = instance['instance_size']
    
    if rotation is None:
        rotation = random.randint(1, n-1)
    
    # Create new coords by rotating the physical locations
    new_coords = []
    for i in range(n):
        old_pos = (i + rotation) % n
        new_coords.append({
            'id': i,
            'x': instance['coords'][old_pos]['x'],
            'y': instance['coords'][old_pos]['y']
        })
    
    # Rotate the tour sequence to match the new ID assignments
    new_tour = [(i - rotation) % n for i in instance['tour']]
    
    # Rotate tour until it starts with 0
    start_idx = new_tour.index(0)
    new_tour = new_tour[start_idx:] + new_tour[:start_idx]
    
    return {
        'instance_size': n,
        'coords': new_coords,
        'tour': new_tour
    }

# Process the input file
with open('data/grid_20/dataset_20_100k.jsonl', 'r') as f:
    instances = [json.loads(line) for line in f]

# Rotate each instance
rotated_instances = [rotate_ids(instance) for instance in instances]

# Write output
with open('data/grid_20/dataset_20_100k_rotated.jsonl', 'w') as f:
    for instance in rotated_instances:
        f.write(json.dumps(instance) + '\n')
