# Project Plan for Adding Model Interpretability Features

## Overview

This plan outlines the steps to integrate model interpretability features into the existing Traveling Salesman Problem (TSP) project. The goal is to track and visualize the model's activations, enabling users to interactively explore attention mechanisms and activation patterns. Additionally, the plan includes collecting activation data for training sparse autoencoders (SAEs).

## Table of Contents

1. [Architecture Changes](#architecture-changes)
   - [Model Instrumentation](#model-instrumentation)
   - [Data Collection System](#data-collection-system)
   - [Visualization Components](#visualization-components)
2. [Tool Integration](#tool-integration)
   - [nnsight Integration](#nnsight-integration)
   - [SAElens Integration](#saelens-integration)
3. [New Components](#new-components)
   - [Activation Manager](#activation-manager)
   - [Web UI Component](#web-ui-component)
   - [Analysis Pipeline](#analysis-pipeline)
4. [Additional Tools](#additional-tools)
5. [Implementation Timeline](#implementation-timeline)
6. [Responsibilities](#responsibilities)

---

## Architecture Changes

### Model Instrumentation

**Objective:** Capture and manage activations during model forward passes.

**Tasks:**

1. **Identify Key Layers:**
   - Determine which layers in `net.py` (e.g., Transformer layers, MLP layers) require activation hooks.
   
2. **Add Hooks:**
   - In `net.py`, within the `TSPEmbedding` class, implement forward hooks to capture:
     - Attention weights
     - Residual stream activations
     - MLP outputs

3. **Create Wrapper Class:**
   - Develop a wrapper around the existing model to handle the registration and removal of hooks.
   - File Path: `net.py`
   - Class: `TSPEmbedding`

4. **Store Activation Data:**
   - Design a mechanism to temporarily store activation data during inference.
   - Ensure thread-safety if using multiprocessing.

### Data Collection System

**Objective:** Efficiently collect and store activation data for analysis and visualization.

**Tasks:**

1. **Design Storage Format:**
   - Choose a format (e.g., HDF5, JSON, or binary files) to store large activation tensors.
   
2. **Implement Storage Pipeline:**
   - In `dataset.py`, extend the `TSPDataset` class to include:
     - Activation data capturing during data loading.
     - Saving activation data to the chosen format.

3. **Metadata Management:**
   - Store metadata such as instance ID, layer names, and timestamps.
   
4. **Optimization:**
   - Ensure minimal overhead during data collection to prevent performance degradation.

### Visualization Components

**Objective:** Develop interactive visualizations for model activations and attention patterns.

**Tasks:**

1. **Node Visualization:**
   - Enhance existing visualization in `TSP_env.py` to overlay activation data.
   
2. **Attention Heatmaps:**
   - Implement heatmaps showing attention scores between nodes.
   
3. **Activation Patterns:**
   - Visualize residual stream and MLP activations with color coding or intensity indicators.
   
4. **Connection Strength Indicators:**
   - Display lines or arrows between nodes with strength proportional to attention weights.

---

## Tool Integration

### nnsight Integration

**Objective:** Utilize nnsight for advanced model inspection and real-time activation tracking.

**Tasks:**

1. **Setup nnsight:**
   - Install and configure nnsight within the project environment.
   
2. **Integrate with Model:**
   - In `net.py`, ensure that nnsight can access the model’s layers and activations.
   
3. **Real-Time Tracking:**
   - Configure nnsight to monitor activations during inference.
   
4. **Interactive Visualization:**
   - Leverage nnsight’s UI capabilities to create interactive dashboards.

### SAElens Integration

**Objective:** Use SAElens for training sparse autoencoders on collected activation data.

**Tasks:**

1. **Install SAElens:**
   - Add SAElens to the project dependencies.
   
2. **Develop Training Pipeline:**
   - In `dataset.py`, create functions to preprocess and feed activation data to SAElens.
   
3. **Model Configuration:**
   - Define SAE architectures suitable for the activation data dimensions.
   
4. **Analysis Tools:**
   - Utilize SAElens’ analysis features to interpret learned features from SAEs.

---

## New Components

### Activation Manager

**Objective:** Manage the tracking and storage of activations seamlessly.

**Tasks:**

1. **Create Activation Manager Class:**
   - File Path: `net.py`
   - Class Name: `ActivationManager`
   
2. **Functionalities:**
   - Register and deregister hooks.
   - Handle storage and retrieval of activation data.
   - Interface with the data collection system.

3. **API Design:**
   - Provide methods to access activation data for specific instances and layers.

### Web UI Component

**Objective:** Develop an interactive web interface for visualizing activations and attention.

**Tasks:**

1. **Choose Framework:**
   - Recommended: Streamlit or Dash for rapid development.
   
2. **Frontend Design:**
   - Design UI layouts featuring:
     - Node maps with hover capabilities.
     - Attention heatmaps.
     - Activation pattern overlays.
   
3. **Backend Integration:**
   - Connect the frontend to the activation manager to fetch real-time data.
   
4. **Interactive Elements:**
   - Implement hover events to highlight related nodes based on attention scores.

### Analysis Pipeline

**Objective:** Process and analyze collected activation data for insights.

**Tasks:**

1. **Data Processing Scripts:**
   - Develop scripts to clean and preprocess activation data.
   
2. **Feature Extraction:**
   - Implement methods to extract meaningful features from activations.
   
3. **Correlation Analysis:**
   - Analyze relationships between different activation patterns and model performance.
   
4. **Reporting Tools:**
   - Generate reports or visual summaries of analysis findings.

---

## Additional Tools

1. **Weights & Biases (W&B):**
   - **Purpose:** Experiment tracking and visualization.
   - **Integration Steps:**
     - Install W&B.
     - Initialize W&B in training scripts.
     - Log activation summaries and metrics.

2. **Captum:**
   - **Purpose:** Advanced interpretability methods.
   - **Integration Steps:**
     - Install Captum.
     - Implement attribution analyses (e.g., Integrated Gradients) in `net.py`.
     - Visualize attribution results in the Web UI.

3. **Streamlit:**
   - **Purpose:** Rapid prototyping of the Web UI.
   - **Integration Steps:**
     - Develop Streamlit apps for visualization.
     - Deploy alongside the existing application or as a separate service.

---

## Implementation Timeline

| Phase                | Duration | Tasks                                                                                             |
|----------------------|----------|---------------------------------------------------------------------------------------------------|
| **Phase 1: Preparation**      | 1 Week   | - Finalize tool selections (nnsight, SAElens, Streamlit) <br> - Set up project environment     |
| **Phase 2: Model Instrumentation** | 2 Weeks  | - Implement activation hooks in `net.py` <br> - Develop `ActivationManager`                 |
| **Phase 3: Data Collection System** | 2 Weeks  | - Design storage format <br> - Extend `TSPDataset` for activation data <br> - Optimize pipeline |
| **Phase 4: Visualization Components** | 3 Weeks  | - Develop node visualization enhancements <br> - Implement attention heatmaps <br> - Integrate with Web UI |
| **Phase 5: Tool Integration** | 2 Weeks  | - Integrate nnsight <br> - Integrate SAElens <br> - Setup W&B and Captum                      |
| **Phase 6: New Components Development** | 3 Weeks  | - Develop Web UI <br> - Create Analysis Pipeline                                              |
| **Phase 7: Testing & Optimization** | 2 Weeks  | - Test end-to-end data flow <br> - Optimize performance <br> - Fix bugs                      |
| **Phase 8: Documentation & Deployment** | 1 Week | - Document new features <br> - Deploy Web UI and analysis tools                             |

---

## Responsibilities

- **[Your Name]:**
  - Lead the integration of model instrumentation.
  - Develop the `ActivationManager` class.
  - Oversee data collection and storage systems.

- **[Team Member 1]:**
  - Handle tool integrations (nnsight, SAElens, Captum).
  - Develop visualization components.

- **[Team Member 2]:**
  - Create the Web UI using Streamlit or Dash.
  - Implement the analysis pipeline.

- **[Team Member 3]:**
  - Manage experiment tracking with Weights & Biases.
  - Assist in testing and optimization phases.

---

## Notes

- **Performance Considerations:**
  - Ensure that activation hooks do not introduce significant overhead.
  - Optimize data storage to handle large activation datasets efficiently.

- **Scalability:**
  - Design the system to accommodate larger TSP instances in the future.
  
- **Security:**
  - If deploying the Web UI, ensure secure access controls to protect model and activation data.

- **Collaboration:**
  - Utilize version control (e.g., Git) to manage changes.
  - Regularly update the team on progress and challenges.

---

## Conclusion

By following this detailed plan, the project will successfully incorporate robust model interpretability features, enhancing both the usability and analytical capabilities of the TSP solver. The integration of visualization tools and analysis pipelines will facilitate deeper insights into the model's decision-making processes, ultimately contributing to more efficient and transparent solutions.
