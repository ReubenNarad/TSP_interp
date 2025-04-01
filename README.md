# TSP-RL: Interpretable Reinforcement Learning for the Traveling Salesman Problem

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Note**: This is an ongoing research project. Code and documentation are continuously being improved.

## Overview

This repository contains an implementation of a reinforcement learning approach to solve the Traveling Salesman Problem (TSP) with a focus on interpretability using Sparse Autoencoders (SAE).

The project aims to:
1. Train transformer-based policies to solve TSP instances using reinforcement learning
2. Extract interpretable features from the learned neural representations using sparse autoencoders
3. Visualize and analyze these features to understand what patterns the neural network has learned

## Project Structure and Architecture

The project is structured around three main components:

1. **Environment (`env/`)**: TSP environment generator and Concorde solver integration
2. **Policy (`policy/`)**: Transformer-based policy models for solving TSP instances using reinforcement learning
3. **Sparse Autoencoder (`sae/`)**: Extracts interpretable features from the learned policy representation
4. **Shell Scripts (`*.sh`)**: Automation scripts for the complete workflow

## Installation

```bash
# Clone the repository
git 
cd tsp_interp

# Install dependencies
pip install -r requirements.txt
```

## Workflow and Usage

The project follows a 4-step workflow:

### 1. Training a Policy
Currently, we use a Graph Attention Network (cite Kool et al), implemented by the RL4CO library (cite) REINFORCE method (modified to use clipped rewards), and TSP environment. The default TSP environment generates distances by drawing node locations from a uniform distribution in the unit square, with the option to modify it.

Specify the policy run name and hyperparameters before running:
```bash
bash train_policy.sh
```

Key hyperparameters in `train_policy.sh`:
- `run_name`: Name of the policy run
- `num_epochs`: Number of training epochs
- `num_instances`: Number of TSP instances per epoch
- `num_loc`: Number of locations in each TSP instance
- `embed_dim`: Dimensionality of transformer embeddings
- `n_encoder_layers`: Number of transformer encoder layers

### 2. Collecting Activations

After training the policy, collect activations from the encoder output by running the policy on a set of TSP instances drawn from the same distribution as the policy's training. This will be used as the training data for the sparse autoencoder:

```bash
# Specify the policy run name before running
bash collect_activations.sh
```

### 3. Training a Sparse Autoencoder

Train the SAE on the collected activations. We use a top-k sparse autoencoder (cite).

```bash
# Specify the policy run name before running
bash train_sae.sh
```

Key hyperparameters in `train_sae.sh`:
- `l1_coef`: L1 sparsity penalty coefficient
- `expansion_factor`: Ratio of latent dimensions to input dimensions
- `k_ratio`: Fraction of latent units allowed to be active

### 4. Analyzing Learned Features

Finally, visualize and analyze the most important features of the SAE. In `analyze_features.sh`, specify:
- `num_instances`: Number of instances to analyze
- `batch_size`: Batch size for feature analysis
- `num_features`: Number of features to analyze

```bash
# Specify the policy run name and the SAE run name before running
bash analyze_features.sh
```

This will generate a gallery of feature activations across num_instances instances.

## Feature Visualization

The sparse autoencoder learns interpretable features from the policy's neural representations. Here are some examples of the discovered features:

### Feature Gallery

![Feature Example 1](images/feature_example1.png)
![Feature Example 2](images/feature_example2.png)

<details>
<summary>More visualizations</summary>

Feature visualizations can be found in the run directories after running `analyze_features.sh`. The script generates visualizations like:

- Feature activations on individual instances
- Feature overlays across multiple instances
- Solution paths with feature activation overlays

Example paths to visualization directories:
```
runs/[run_name]/sae/sae_runs/[sae_run_name]/feature_analysis/
```

To add more visualizations to this README, copy them to the `images/` directory and link them here.

</details>

## Results and Findings

This research is ongoing, but preliminary results suggest that:

1. The transformer-based policy can learn effective strategies for solving TSP instances
2. The sparse autoencoder successfully extracts interpretable features from the policy's internal representations
3. Many of these features correspond to meaningful spatial patterns relevant to solving TSP

## Future Work

Potential directions for future development:

1. Add support for more TSP variants (e.g., Capacitated Vehicle Routing Problem)
2. Create a web interface for interactive exploration of learned features
3. Implement automated testing
4. Add parallel processing support for faster training and analysis

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Concorde TSP Solver](http://www.math.uwaterloo.ca/tsp/concorde.html) for optimal solutions

## Citation

If you use this code in your research, please cite:

```
@misc{TSP-RL,
  author = {Your Name},
  title = {TSP-RL: Interpretable Reinforcement Learning for the Traveling Salesman Problem},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/yourusername/tsp-rl}
}
``` 