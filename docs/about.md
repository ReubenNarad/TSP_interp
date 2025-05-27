## About SAE Feature Analysis

<div style="width: 100vw; margin-left: calc(-50vw + 50%); padding: 4rem 0; margin-bottom: 3rem;">
  <div style="max-width: 1200px; margin: 0 auto; padding: 0 2rem;">
    <h1 style="text-align: center; font-size: 3rem; font-weight: 700; color: #333; margin-bottom: 1rem;">
      TSP Solver Mechanistic Interpretability
    </h1>
    <p style="text-align: center; font-size: 1.2rem; color: #666; margin-bottom: 3rem; max-width: 800px; margin-left: auto; margin-right: auto;">
      We investigate the internal mechanisms used by deep learning models for the Traveling Salesman Problem using sparse autoencoder feature analysis and circuit tracing methodology.
    </p>
    
    <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 2rem; margin-top: 3rem;">
      <!-- TSP Solver Panel -->
      <div style="background: white; border-radius: 12px; padding: 2rem; box-shadow: 0 4px 12px rgba(0,0,0,0.1); text-align: center; cursor: pointer; transition: transform 0.2s ease;" onclick="document.getElementById('tsp-solver').scrollIntoView()">
        <div style="width: 100%; height: 200px; background: #f8f9fa; border-radius: 8px; margin-bottom: 1.5rem; display: flex; align-items: center; justify-content: center; color: #999; font-style: italic;">
          [TSP Architecture Image]
        </div>
        <h3 style="color: #6f42c1; margin-bottom: 1rem; font-size: 1.3rem;">TSP Solver</h3>
      </div>
      
      <!-- Sparse Autoencoder Panel -->
      <div style="background: white; border-radius: 12px; padding: 2rem; box-shadow: 0 4px 12px rgba(0,0,0,0.1); text-align: center; cursor: pointer; transition: transform 0.2s ease;" onclick="document.getElementById('sparse-autoencoder').scrollIntoView()">
        <div style="width: 100%; height: 200px; background: #f8f9fa; border-radius: 8px; margin-bottom: 1.5rem; display: flex; align-items: center; justify-content: center; color: #999; font-style: italic;">
          [SAE Diagram Image]
        </div>
        <h3 style="color: #6f42c1; margin-bottom: 1rem; font-size: 1.3rem;">Sparse Autoencoder</h3>
      </div>
      
      <!-- Feature Analysis Panel -->
      <div style="background: white; border-radius: 12px; padding: 2rem; box-shadow: 0 4px 12px rgba(0,0,0,0.1); text-align: center; cursor: pointer; transition: transform 0.2s ease;" onclick="document.getElementById('feature-analysis').scrollIntoView()">
        <div style="width: 100%; height: 200px; background: #f8f9fa; border-radius: 8px; margin-bottom: 1.5rem; display: flex; align-items: center; justify-content: center; color: #999; font-style: italic;">
          [Feature Visualization Image]
        </div>
        <h3 style="color: #6f42c1; margin-bottom: 1rem; font-size: 1.3rem;">Feature Analysis</h3>
      </div>
    </div>
  </div>
</div>

<div style='max-width: 800px; margin: 0 auto; line-height: 1.6;'>
  <h3>Overview</h3>
  <p>Sparse coding has garnered interest in the Ai interpretability community for discovering features and circuits within the latent space of neural networks. While most of this interpretability work has focused on language models, the same techniques can be applied to other modalities.</p>
  <p>We are interested in interpreting Deep Learning models for OR problems, starting with the TSP. Many recent efforts (cite Kool et all, cite Pointer networks, cite Cycleformer) have used deep learning to either learn heuristics  to solve or to directly solve the TSP.</p>
  <p>By investigating the latent space of these models, we hope to: </p>
  <ul>
    <li>Learn details about the model itself, such as debugging and training dynamics</li>
    <li>Identify motifs that could be externally useful to understanding the TSP domain</li>
    <li>Extract other learned attributes of TSP instances for transfer learning</li>
  </ul>
  
  <!-- TSP Solver Section -->
  <div class="collapsible-section" id="tsp-solver">
    <h3 class="section-header">TSP Solver</h3>
    <div class="section-content">
      <div class="section-nav">
        <a href="#architecture">Architecture</a> | 
        <a href="#training-params">Training Parameters</a> | 
        <a href="#data" style="color: #999;">Data (TODO)</a> | 
        <a href="#results" style="color: #999;">Results (TODO)</a>
      </div>
      
      <h4 id="architecture">Architecture</h4>
      <div style='text-align: center; margin: 2rem 0;'>
        <img src='architecture.png' alt='SAE Feature Analysis Architecture' style='width: 160%; max-width: none; height: auto; border: 1px solid #ddd; border-radius: 8px; margin-left: -35%;'>
        <p style='font-style: italic; color: #666; margin-top: 0.5rem;'>System architecture showing the flow from TSP instances through the policy network to SAE feature extraction and analysis.</p>
        <p> We use (RL4CO)'s implementation of (Kool et al.)'s Transformer Pointer Network, which performs next-node prediction to autoregressively construct TSP solutions. We focus on TSP instances with a fixed size of 100 nodes. </p>
        <p>  </p>    
      </div>
      
      <h4 id="training-params">Training Parameters</h4>
      <table style="margin: 1rem 0; border-collapse: collapse; width: 100%;">
        <caption style="font-weight: bold; margin-bottom: 0.5rem;">Training Parameters</caption>
        <thead>
          <tr style="background-color: #f5f5f5;">
            <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Parameter</th>
            <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Value</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td style="border: 1px solid #ddd; padding: 8px;">lr</td>
            <td style="border: 1px solid #ddd; padding: 8px;">1e-05</td>
          </tr>
          <tr style="background-color: #f9f9f9;">
            <td style="border: 1px solid #ddd; padding: 8px;">num_epochs</td>
            <td style="border: 1px solid #ddd; padding: 8px;">1000</td>
          </tr>
          <tr>
            <td style="border: 1px solid #ddd; padding: 8px;">num_instances</td>
            <td style="border: 1px solid #ddd; padding: 8px;">100000</td>
          </tr>
          <tr style="background-color: #f9f9f9;">
            <td style="border: 1px solid #ddd; padding: 8px;">num_val</td>
            <td style="border: 1px solid #ddd; padding: 8px;">100</td>
          </tr>
          <tr>
            <td style="border: 1px solid #ddd; padding: 8px;">num_loc</td>
            <td style="border: 1px solid #ddd; padding: 8px;">100</td>
          </tr>
          <tr style="background-color: #f9f9f9;">
            <td style="border: 1px solid #ddd; padding: 8px;">temperature</td>
            <td style="border: 1px solid #ddd; padding: 8px;">1.0</td>
          </tr>
          <tr>
            <td style="border: 1px solid #ddd; padding: 8px;">embed_dim</td>
            <td style="border: 1px solid #ddd; padding: 8px;">256</td>
          </tr>
          <tr style="background-color: #f9f9f9;">
            <td style="border: 1px solid #ddd; padding: 8px;">n_encoder_layers</td>
            <td style="border: 1px solid #ddd; padding: 8px;">5</td>
          </tr>
          <tr>
            <td style="border: 1px solid #ddd; padding: 8px;">checkpoint_freq</td>
            <td style="border: 1px solid #ddd; padding: 8px;">20</td>
          </tr>
          <tr style="background-color: #f9f9f9;">
            <td style="border: 1px solid #ddd; padding: 8px;">dropout</td>
            <td style="border: 1px solid #ddd; padding: 8px;">0.1</td>
          </tr>
          <tr>
            <td style="border: 1px solid #ddd; padding: 8px;">attention_dropout</td>
            <td style="border: 1px solid #ddd; padding: 8px;">0.0</td>
          </tr>
        </tbody>
      </table>
    </div>
  </div>
  
  <!-- Sparse Autoencoder Section -->
  <div class="collapsible-section" id="sparse-autoencoder">
    <h3 class="section-header">Sparse Autoencoder</h3>
    <div class="section-content">
      <p>Sparse Autoencoders are neural network architectures designed to learn efficient representations of data by enforcing sparsity constraints. They are particularly useful for understanding the internal representations learned by larger neural networks.</p>
      
      <h4>Current Analysis</h4>
      <p><strong>Run:</strong> Hybrid_3_1e-5</p>
      <p><strong>SAE Model:</strong> sae_l10.001_ef4.0_k0.1_03-21_17:41:07</p>
      <p>This analysis includes data from multiple features with varying activation strengths, providing insights into the learned representations and their interpretability.</p>
    </div>
  </div>
  
  <!-- Feature Analysis Section -->
  <div class="collapsible-section" id="feature-analysis">
    <h3 class="section-header">Feature Analysis</h3>
    <div class="section-content">
      <h4>Features</h4>
      <ul>
        <li><strong>Per Feature View:</strong> Explore individual features and their top activating instances</li>
        <li><strong>Per Instance View:</strong> Examine specific instances and see which features they activate</li>
        <!-- I think we removed this? ^^ -->
        <li><strong>Interactive Visualization:</strong> Click on images to view them in full resolution</li>
        <li><strong>Activation Analysis:</strong> View activation strengths and patterns across different features</li>
      </ul>
      
      <h4>How to Use</h4>
      <p>Use the dropdown menu above to switch between different views:</p>
      <ul>
        <li>Select "Per Feature" to explore individual features and their most activating instances</li>
        <li>Select "Per Instance" to examine specific instances and see which features they activate most strongly</li>
        <li>Click on any image to view it in full resolution in a new tab</li>
      </ul>
      
      <p><em>This is placeholder content that will be expanded with a full article about the methodology, results, and implications of this SAE analysis.</em></p>
    </div>
  </div>
</div> 