# Ablation Study Implementation Plan

## Overview
Implement a bandit-based ablation study system to efficiently identify the most impactful SAE features in our TSP solver. The goal is to find features that causally affect the model's behavior, specifically testing the "nearest neighbor mimicking" hypothesis.

## Core Architecture

### 1. Feature Ablation Engine (`ablation_engine.py`)
**Purpose**: Core orchestrator that manages the ablation pipeline

**Key Components**:
- `AblationEngine` class that coordinates experiments
- Manages model loading, instance generation, and result aggregation
- Handles baseline vs ablated model comparisons
- Interfaces with bandit selector and reward functions

**Methods**:
- `__init__(run_name, sae_run_name, device)` - Load models and setup
- `run_experiment(num_instances, budget, reward_function, bandit_config)` - Main experiment loop
- `ablate_feature(feature_idx, instances)` - Run single feature ablation
- `save_results(output_dir)` - Save experiment results

### 2. Feature Ablator (`feature_ablator.py`)
**Purpose**: Handles the mechanics of zeroing out SAE features OR raw neurons during forward passes

**Key Components**:
- `FeatureAblator` class that modifies activations at different model layers
- Context manager for temporary ablations
- Hook system to intercept and modify both SAE outputs and raw model activations
- Support for multiple ablation targets (SAE features, encoder neurons, attention heads, etc.)

**Methods**:
- `__init__(tsp_model, sae_model=None)` - Setup hooks for both models
- `ablate_sae_features(feature_indices)` - Context manager for SAE feature ablation
- `ablate_raw_neurons(layer_name, neuron_indices)` - Context manager for raw neuron ablation
- `ablate_attention_heads(layer_idx, head_indices)` - Context manager for attention head ablation
- `_sae_intervention_hook(module, input, output)` - Hook function to zero SAE features
- `_neuron_intervention_hook(module, input, output)` - Hook function to zero raw neurons
- `reset()` - Clear all ablations and remove hooks

**Ablation Types Supported**:
- **SAE Features**: Zero out specific dimensions in SAE latent space
- **Encoder Neurons**: Zero out specific neurons in transformer encoder layers
- **Attention Heads**: Zero out specific attention heads
- **Feedforward Neurons**: Zero out specific neurons in FFN layers
- **Embedding Dimensions**: Zero out specific embedding dimensions

### 3. Behavior Comparator (`behavior_comparator.py`)
**Purpose**: Compares baseline vs ablated model outputs

**Key Components**:
- `BehaviorComparator` class for running paired comparisons
- Handles both single-step and trajectory-level comparisons
- Manages batched processing for efficiency

**Methods**:
- `__init__(tsp_model, sae_model, ablator)` - Setup models
- `compare_single_step(instances, feature_idx, step_idx)` - Compare next-node predictions
- `compare_full_trajectory(instances, feature_idx)` - Compare complete solutions
- `_run_forward_pass(instances, ablated_features=None)` - Execute model forward pass

### 4. Reward Functions (`reward_functions.py`)
**Purpose**: Define different metrics for measuring ablation impact

**Base Class**:
- `RewardFunction` - Abstract base class defining interface
- `compute_reward(baseline_output, ablated_output, instances)` - Main method

**Implementations**:
- `NearestNeighborReward` - Tests NN mimicking hypothesis
- `TourLengthReward` - Measures impact on solution quality  
- `ProbabilityDivergenceReward` - Measures distribution shift
- `AttentionChangeReward` - Measures attention pattern changes

### 5. Bandit Selector (`bandit_selector.py`)
**Purpose**: Implements bandit algorithms for efficient feature/neuron selection

**Key Components**:
- `BanditSelector` base class
- Multiple algorithm implementations (UCB, Thompson Sampling, etc.)
- Top-p selection logic (nucleus-style selection)
- Support for different ablation target types (SAE features vs raw neurons)

**Implementations**:
- `UCBSelector` - Upper Confidence Bound algorithm
- `ThompsonSelector` - Thompson Sampling 
- `EpsilonGreedySelector` - ε-greedy exploration

**Methods**:
- `select_next_target()` - Choose next ablation target (feature or neuron) to test
- `update_reward(target_id, reward)` - Update beliefs after observation
- `get_top_p_targets(p)` - Return targets contributing p% of total impact
- `set_target_space(target_type, target_range)` - Configure what to ablate (SAE features, layer neurons, etc.)

### 6. Experiment Orchestrator (`ablation_experiment.py`)
**Purpose**: High-level interface for running complete experiments

**Key Components**:
- `AblationExperiment` class that manages full pipeline
- Configuration management and result visualization
- Multiple experiment types and batch processing

**Methods**:
- `__init__(config)` - Load configuration and initialize components
- `run_nearest_neighbor_experiment()` - Specific NN hypothesis test
- `run_comprehensive_scan()` - Test multiple reward functions
- `visualize_results()` - Generate plots and summaries

## Data Flow Pipeline

### Phase 1: Initialization
1. Load TSP model checkpoint and SAE model
2. Initialize feature ablator with SAE hooks
3. Setup behavior comparator with models
4. Configure bandit selector with hyperparameters
5. Generate or load test TSP instances

### Phase 2: Bandit-Guided Search  
1. **Target Selection**: Bandit algorithm selects next ablation target (SAE feature, neuron, attention head, etc.)
2. **Baseline Run**: Execute forward pass with unmodified model
3. **Ablated Run**: Execute forward pass with selected target zeroed out
4. **Comparison**: Compute reward using specified reward function
5. **Update**: Feed reward back to bandit algorithm
6. **Repeat**: Continue until budget exhausted or convergence

### Phase 3: Analysis & Results
1. Extract top-p features from bandit selector
2. Run detailed analysis on identified important features
3. Generate visualizations and statistical summaries
4. Save results and experiment metadata

## Implementation Details

### Hook Integration
- Use PyTorch forward hooks to intercept SAE outputs
- Modify activations in-place during ablation context
- Ensure hooks are properly cleaned up after experiments

### Efficiency Considerations  
- Batch multiple instances together for GPU efficiency
- Cache baseline computations when testing multiple features
- Use memory-efficient attention computation for long sequences
- Implement checkpointing for long-running experiments

### Experiment Configuration
```python
# SAE Feature Ablation Experiment
sae_experiment_config = {
    "run_name": "policy_run_name",
    "sae_run_name": "sae_run_name", 
    "ablation_type": "sae_features",
    "target_range": range(1024),  # All SAE features
    "num_instances": 1000,
    "budget": 5000,  # Max ablation evaluations
    "bandit_algorithm": "ucb",
    "reward_function": "nearest_neighbor",
    "top_p_threshold": 0.8,
    "batch_size": 32,
    "device": "cuda"
}

# Raw Neuron Ablation Experiment  
neuron_experiment_config = {
    "run_name": "policy_run_name",
    "sae_run_name": None,  # No SAE needed
    "ablation_type": "encoder_neurons",
    "target_layer": "encoder.layers.2",  # Target specific layer
    "target_range": range(256),  # All neurons in that layer
    "num_instances": 1000,
    "budget": 2000,
    "bandit_algorithm": "thompson_sampling",
    "reward_function": "tour_length",
    "top_p_threshold": 0.9,
    "batch_size": 32,
    "device": "cuda"
}

# Attention Head Ablation Experiment
attention_experiment_config = {
    "run_name": "policy_run_name", 
    "sae_run_name": None,
    "ablation_type": "attention_heads",
    "target_layer": 1,  # Layer index
    "target_range": range(8),  # All 8 attention heads
    "num_instances": 500,
    "budget": 64,  # Small budget since only 8 heads
    "bandit_algorithm": "epsilon_greedy",
    "reward_function": "probability_divergence",
    "top_p_threshold": 0.8,
    "batch_size": 16,
    "device": "cuda"
}
```

### Key Metrics to Track
- Feature impact scores (reward values)
- Bandit algorithm convergence 
- Computational efficiency (features tested vs budget)
- Statistical significance of identified features
- Ablation effect sizes and confidence intervals

## File Structure
```
experiments/ablation/
├── plan.md                    # This file
├── proposal.md               # Project proposal
├── ablation_engine.py        # Core orchestrator
├── feature_ablator.py        # Feature zeroing mechanics  
├── behavior_comparator.py    # Baseline vs ablated comparison
├── reward_functions.py       # Reward function implementations
├── bandit_selector.py        # Bandit algorithm implementations
├── ablation_experiment.py    # High-level experiment interface
├── configs/                  # Experiment configurations
├── results/                  # Experiment outputs
└── utils.py                  # Helper functions

```

## Testing Strategy
1. **Unit Tests**: Test each component in isolation
2. **Integration Tests**: Test full pipeline with small feature sets
3. **Validation**: Compare results with manual ablation studies
4. **Benchmarking**: Measure efficiency gains vs exhaustive search

## Expected Outputs
1. **Feature Rankings**: Ordered list of most impactful features
2. **Impact Quantification**: Numerical scores for each tested feature  
3. **Hypothesis Validation**: Evidence for/against NN mimicking hypothesis
4. **Efficiency Analysis**: Comparison of bandit vs exhaustive search
5. **Visualizations**: Feature impact plots, bandit convergence curves
6. **Statistical Reports**: Confidence intervals, significance tests

This modular design allows for easy extension to new reward functions, bandit algorithms, and analysis methods while maintaining clean separation of concerns.