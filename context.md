# TSP-interp: Mechanistic Interpretability for the Traveling Salesman Problem

## Project Overview

This repository applies mechanistic interpretability techniques to understand neural networks trained to solve the Traveling Salesman Problem (TSP). The core idea is to analyze internal representations of a transformer-based policy model using Sparse Autoencoders (SAE) to extract interpretable features.

The project aims to turn "black-box" neural network models for combinatorial optimization into more transparent and interpretable systems. By understanding what these models learn, the project hopes to gain insights into both neural networks and the structure of TSP problems.

## Repository Structure

The repository is organized into several key directories:

### 1. Environment (`env/`)
- **generator.py**: Implements TSP instance generation with various distribution options
- **solve_td_with_concorde.py**: Interface to the Concorde TSP solver for optimal solution benchmarking

### 2. Policy Models (`policy/`)
- **train_vanilla.py**: Main policy training script using RL methods
- **reinforce_clipped.py**: Implementation of modified REINFORCE algorithm with clipped rewards
- **policy_hooked.py**: Modified policy model with hooks for extracting internal activations
- **eval.py**: Evaluation scripts to test policy performance
- **plotter.py**: Visualization tools for policy behavior

### 3. Sparse Autoencoder (`sae/`)
- **collect_activations.py**: Extracts policy network activations for SAE training
- **train_topk.py**: Implements top-k sparse autoencoder training
- **feature_analysis.py**: Analyzes and visualizes learned SAE features
- **regression_test.py**: Tests relationships between features and problem attributes
- **activations_gif.py**: Creates visualizations of activations over time

### 4. Probes (`probe/`)
- **train_probe.py**: Trains linear probes to understand what information is encoded in network layers
- **concorde_for_probe.py**: Concorde solver integration for probe training

### 5. Scripts
- **train_policy.sh**: Script to train the neural policy model
- **collect_activations.sh**: Script to extract activations from the policy 
- **train_sae.sh**: Script to train the sparse autoencoder
- **analyze_features.sh**: Script to analyze and visualize SAE features

### 6. Visualization
- **readme_images/**: Contains visualizations for the README
- Various PNG files: Feature visualization outputs

## Workflow

The project follows a four-step workflow:

1. **Train a Policy**: A transformer-based neural network is trained with reinforcement learning to solve TSP instances
2. **Collect Activations**: The trained policy is run on TSP instances, and activations from its encoder layers are collected
3. **Train a Sparse Autoencoder**: An SAE is trained on these activations to extract interpretable features
4. **Analyze Features**: The learned features are visualized and analyzed to understand what patterns the policy has learned

### Detailed Workflow (Bash Scripts)

The repository provides bash scripts to execute each step of the workflow:

#### 1. Training the TSP Policy (`train_policy.sh`)
```bash
bash train_policy.sh
```
Key parameters:
- `run_name`: Name identifier for the training run
- `num_epochs`: Number of training epochs
- `num_instances`: Number of TSP instances per epoch
- `num_loc`: Number of locations in each TSP instance (problem size)
- `embed_dim`: Dimensionality of transformer embeddings
- `n_encoder_layers`: Number of transformer encoder layers
- `batch_size`: Batch size for training

This script calls `policy/train_vanilla.py` to train a transformer-based policy using REINFORCE.

#### 2. Collecting Activations (`collect_activations.sh`)
```bash
bash collect_activations.sh
```
Key parameters:
- `run_name`: Name of the policy run to collect activations from
- `num_instances`: Number of TSP instances to process
- `batch_size`: Batch size for processing

This script calls `sae/collect_activations.py` to run the trained policy on TSP instances and extract activations from its encoder layers.

#### 3. Training the Sparse Autoencoder (`train_sae.sh`)
```bash
bash train_sae.sh
```
Key parameters:
- `run_name`: Name of the policy run
- `sae_run_name`: Name identifier for the SAE run
- `l1_coef`: L1 sparsity penalty coefficient
- `expansion_factor`: Ratio of latent dimensions to input dimensions
- `k_ratio`: Fraction of latent units allowed to be active
- `batch_size`: Batch size for training

This script calls `sae/train_topk.py` to train a top-k sparse autoencoder on the collected activations.

#### 4. Analyzing Features (`analyze_features.sh`)
```bash
bash analyze_features.sh
```
Key parameters:
- `run_name`: Name of the policy run
- `sae_run_name`: Name of the SAE run
- `num_instances`: Number of instances to analyze
- `batch_size`: Batch size for feature analysis
- `num_features`: Number of features to analyze
- `nodes_to_highlight`: Number of nodes to highlight in visualizations

This script calls `sae/feature_analysis.py` to generate visualizations and analyses of the learned SAE features.

#### 5. Probe Training (Manual Process)
For training probes, use the Python scripts directly:
```bash
python probe/train_probe.py --run_name <policy_run> --sae_run_name <sae_run> --target <target_metric>
```
Key parameters:
- `run_name`: Name of the policy run
- `sae_run_name`: Name of the SAE run (if using SAE features)
- `target`: The target metric to predict (e.g., "optimal_length", "difficulty")
- `layer`: Which layer's activations to use for the probe

The probe training process helps to understand what information is encoded in the policy's neural representations by training linear models to predict properties of TSP instances.

## Dependencies

The project relies on:
- **PyTorch**: Deep learning framework for neural network implementation
- **RL4CO** (Reinforcement Learning for Combinatorial Optimization): A library implementing reinforcement learning approaches for solving combinatorial optimization problems. The project relies heavily on RL4CO's TSP environment and attention-based architecture. (https://github.com/ai4co/rl4co)
- **Concorde TSP Solver**: Industry-standard exact solver for the TSP, used to generate optimal solutions as a benchmark (http://www.math.uwaterloo.ca/tsp/concorde.html)
  - **Note**: The `concorde` command must be available on the system PATH. Concorde is only available for Windows and Linux systems, not for macOS. The code directly calls the `concorde` command-line utility.
- **NumPy**: For numerical computations and array operations
- **Matplotlib**: For creating visualizations and plots
- **TensorDict**: Data structure for efficient handling of tensors
- **SciPy**: For scientific computing and statistical tools
- **scikit-learn**: For machine learning utilities including regression models
- **Lightning**: For structured PyTorch training

The complete list of Python dependencies can be found in `requirements.txt`.

## Key Technical Concepts

1. **TSP Environment**: TSP instances are created by generating city locations (typically uniformly distributed in a unit square) and computing pairwise distances.

2. **Policy Architecture**: Uses a Graph Attention Network (based on Kool et al.) implemented through RL4CO, trained with REINFORCE.

3. **Sparse Autoencoder**: A top-k SAE extracts sparse features from the dense neural representations of the policy.

4. **Feature Analysis**: Visualization tools show how features activate for different TSP instance properties.

5. **Probes**: Linear models trained to predict properties of TSP instances from neural representations.

## Research Goals

The repository supports ongoing research exploring:
- How neural networks encode spatial information in TSP instances
- What features are most important for solving TSP efficiently
- How different layers of the policy network build upon each other
- The relationship between learned features and problem difficulty

The ultimate goal is to bridge the gap between neural and symbolic approaches to combinatorial optimization by making neural solutions more interpretable and extracting useful heuristics.
