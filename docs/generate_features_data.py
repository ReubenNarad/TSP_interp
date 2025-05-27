#!/usr/bin/env python3
"""
Script to generate Jekyll-compatible features data from activation_index.json
"""

import json
import yaml
from collections import defaultdict

def main():
    # Load the activation index
    with open('activation_index.json', 'r') as f:
        activation_data = json.load(f)
    
    # Process the data to get feature information
    feature_data = defaultdict(list)
    
    # Collect all activations for each feature across instances
    for instance_id, features in activation_data.items():
        instance_num = int(instance_id.split('_')[1])
        
        for feature_id, activation in features.items():
            feature_data[int(feature_id)].append({
                'instance_id': instance_num,
                'activation': activation
            })
    
    # Calculate average activations and sort instances by activation
    features_list = []
    for feature_id, instances in feature_data.items():
        # Sort instances by activation (highest first)
        instances.sort(key=lambda x: x['activation'], reverse=True)
        
        # Calculate average activation
        avg_activation = sum(inst['activation'] for inst in instances) / len(instances)
        
        # Format for Jekyll
        feature_info = {
            'id': feature_id,
            'activation': round(avg_activation, 4),
            'instances': [
                {
                    'id': inst['instance_id'],
                    'activation': round(inst['activation'], 4)
                }
                for inst in instances
            ]
        }
        features_list.append(feature_info)
    
    # Sort features by average activation (highest first)
    features_list.sort(key=lambda x: x['activation'], reverse=True)
    
    # Create the final data structure
    output_data = {'features': features_list}
    
    # Write to YAML file
    with open('_data/features.yml', 'w') as f:
        yaml.dump(output_data, f, default_flow_style=False, sort_keys=False)
    
    print(f"Generated features data for {len(features_list)} features")
    print("Top 5 features by activation:")
    for i, feature in enumerate(features_list[:5]):
        print(f"  {i+1}. Feature {feature['id']}: {feature['activation']}")

if __name__ == '__main__':
    main() 