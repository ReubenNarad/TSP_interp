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
- **feature_analysis.py**: Analyzes and visualizes learned SAE features with extensive visualization capabilities
- **regression_test.py**: Tests relationships between features and problem attributes
- **activations_gif.py**: Creates visualizations of activations over time
- **utils.py**: Utility functions for activation analysis and visualization
- **model/**: Contains SAE model implementations

### 4. Probes (`probe/`)
- **train_probe.py**: Trains linear probes to understand what information is encoded in network layers
- **concorde_for_probe.py**: Concorde solver integration for probe training

### 5. Documentation and Web Interface (`docs/`)
- **index.html**: Interactive web documentation with feature exploration interface
- **assets/**: CSS and JavaScript files for the web interface
- **feature_overlays/**: Generated overlay visualizations for features
- **demo_overlays/**: Curated example feature visualizations
- **activation_index.json**: Index of feature activations for the interactive interface
- **generate_features_data.py**: Script to generate data for the web interface
- Various PNG and GIF files for documentation and demonstrations

### 6. Distribution Classes (`distributions.py`)
A comprehensive module containing multiple TSP instance distribution classes:
- **Uniform**: Standard uniform distribution in unit square
- **RandomUniform**: Uniform within randomly sampled rectangles  
- **DualUniform**: 50/50 split between two uniform distributions
- **NRandomUniform**: Random number of elongated uniform clusters
- **FuzzyCircle**: Circular patterns with normally distributed radial noise
- **Clusters**: Gaussian cluster-based distributions
- **SimpleClusters**: Simplified clustering with configurable parameters
- **HybridSampler**: Combines multiple distributions with specified probabilities

### 7. Experiments (`experiments/`)
- **cluster_feature/**: Contains experimental analysis of cluster-based features with interactive HTML viewers

### 8. Scripts
- **train_policy.sh**: Script to train the neural policy model
- **collect_activations.sh**: Script to extract activations from the policy 
- **train_sae.sh**: Script to train the sparse autoencoder
- **analyze_features.sh**: Script to analyze and visualize SAE features

### 9. Visualization and Results
- **readme_images/**: Contains visualizations for the README
- Various feature visualization PNG files and analysis results
- **runs/**: Training run outputs and checkpoints

### 10. Cross-Layer Transcoders (`clt/`)
- **model.py**: Sparse CLT architecture mirroring the top-k SAE but mapping between layers
- **train_transcoder.py**: W&B-ready trainer that consumes existing activation dumps and fits cross-layer transcoders
- Training artifacts are written under `runs/<policy_run>/clt/clt_runs/<source__to__target>/`

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
- `chunk_size`: If set when calling `sae.collect_activations` directly, limits the number of instances processed per forward pass to avoid GPU OOM.

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

## Interactive Web Documentation

The project includes a comprehensive web interface (`docs/index.html`) that provides:

- **Interactive Feature Explorer**: Browse and visualize SAE features with a dropdown interface
- **Feature Activation Overlays**: Multi-instance visualizations showing how features respond to different TSP instances
- **Mathematical Documentation**: Detailed explanations of the SAE architecture and forward pass
- **Feature Analysis Results**: Categorized examples of discovered feature types (edge detectors, cluster focusers, linear separators, etc.)
- **Responsive Design**: Modern web interface with proper styling and navigation

The web interface can be served locally or deployed for easy sharing of results and demonstrations.

## Advanced Visualization Capabilities

The `sae/feature_analysis.py` module provides extensive visualization capabilities:

- **Feature Overlay Visualizations**: Show how features activate across multiple TSP instances simultaneously
- **Per-Instance Analysis**: Detailed feature activation maps for individual instances
- **Solution Path Overlays**: Optional TSP solution path visualization alongside feature activations
- **Interactive HTML Generation**: Automatic generation of browsable HTML interfaces for results
- **Activation Statistics**: Tools for analyzing feature activation patterns and statistics
- **Batch Processing**: Efficient analysis of large numbers of instances and features

## Distribution Flexibility

The `distributions.py` module provides extensive flexibility for training and testing on different types of TSP instances:

- **Uniform Distributions**: Standard and parameterized uniform sampling
- **Cluster-Based**: Various clustering approaches with configurable parameters
- **Geometric Patterns**: Circular and other geometric distributions
- **Hybrid Approaches**: Combine multiple distributions with specified probabilities
- **Random Parameterization**: Distributions with randomly sampled parameters per instance

This allows for comprehensive testing of how the neural solver generalizes across different problem structures.

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

4. **Feature Analysis**: Comprehensive visualization tools show how features activate for different TSP instance properties, including overlay visualizations across multiple instances.

5. **Interactive Documentation**: Web-based interface for exploring and understanding discovered features.

6. **Distribution Variety**: Extensive collection of TSP instance distributions for training and evaluation.

## Research Goals

The repository supports ongoing research exploring:
- How neural networks encode spatial information in TSP instances
- What features are most important for solving TSP efficiently
- How different layers of the policy network build upon each other
- The relationship between learned features and problem difficulty
- Interpretability techniques for combinatorial optimization

The ultimate goal is to bridge the gap between neural and symbolic approaches to combinatorial optimization by making neural solutions more interpretable and extracting useful heuristics.

## Web Interface and Demo

The project includes a sophisticated web interface that allows for:
- Interactive exploration of discovered SAE features
- Real-time visualization of feature activations
- Comprehensive documentation with mathematical explanations
- Demonstration of different feature types and their behaviors
- Easy sharing and presentation of research results

This makes the research accessible to both technical and non-technical audiences, facilitating better understanding and communication of the interpretability findings.
