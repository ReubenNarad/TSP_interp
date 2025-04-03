import torch
import matplotlib.pyplot as plt
import os, argparse, pickle
import imageio.v2 as imageio
import numpy as np
from scipy.optimize import curve_fit
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from distributions import Uniform, DualUniform, RandomUniform, FuzzyCircle, HybridSampler


def main(args):
    run_path = f'runs/{args.run_name}'
    n_epochs = args.num_epochs
    plot_instance = 1

    env = pickle.load(open(f'{run_path}/env.pkl', 'rb'))
    instance = pickle.load(open(f'{run_path}/val_td.pkl', 'rb'))

    # 1) Load the solved baseline
    with open(f'{run_path}/baseline.pkl', 'rb') as f:
        baseline = pickle.load(f)

    # Render the baseline (optimal) solution
    baseline_dist_for_instance = -baseline["rewards"][0][plot_instance].item()
    fig = env.render(instance[plot_instance], baseline["actions"][0][plot_instance])
    plt.title(f"Optimal Distance: {baseline_dist_for_instance:.2f}")
    plt.savefig(f'{run_path}/optimal_solution_{plot_instance}.png')
    plt.close()

    # 2) Load the training results for each epoch and render solutions
    os.makedirs(f'{run_path}/renders', exist_ok=True)
    results = []
    step = 2
    
    for i in range(0, n_epochs, step):
        # Load results for this epoch
        with open(f'{run_path}/results/results_epoch_{i}.pkl', 'rb') as f:
            td_epoch = pickle.load(f)
        results.append(td_epoch)
        
        # Compute distance for this epoch (recall: distance = -reward)
        distance = -td_epoch["rewards"][plot_instance].item()

        # Render solution for this epoch
        fig = env.render(instance[plot_instance], td_epoch["actions"][plot_instance])
        plt.title(f"Epoch {i} — Distance: {distance:.2f}")
        plt.savefig(f'{run_path}/renders/epoch_{i}_{plot_instance}.png')
        plt.close()

    # 2.5) Create a single GIF from the saved PNG files using imageio
    gif_path = f"{run_path}/animation_{plot_instance}.gif"
    # Read all images first
    images = []
    for i in range(0, n_epochs, step):
        filename = f'{run_path}/renders/epoch_{i}_{plot_instance}.png'
        images.append(imageio.imread(filename))

    # Write GIF with fps parameter
    imageio.mimsave(gif_path, images, fps=args.fps)
    print(f"GIF saved at {gif_path}")

    # 3) Get the baseline's average distance
    baseline_distance = -baseline["rewards"][0].mean().item()

    # 4) Compute the average distance for each epoch
    epoch_distances = []
    for i in range(0, n_epochs, step):
        with open(f'{run_path}/results/results_epoch_{i}.pkl', 'rb') as f:
            td_epoch = pickle.load(f)
        avg_dist = -td_epoch["rewards"][0].mean().item() if isinstance(td_epoch["rewards"], list) else -td_epoch["rewards"].mean().item()
        epoch_distances.append(avg_dist)

    # 5) Plot and fit double exponential decay
    plt.figure(figsize=(8, 5))
    
    # Adjust start_epoch based on available data
    min_epochs_for_fit = 3  # Minimum number of points needed for a reasonable fit
    start_epoch = min(10, n_epochs // 2)  # Use either 10 or half of total epochs
    start_idx = start_epoch // step
    
    epochs = np.array(range(0, n_epochs, step))  # Changed to start from 0
    distances = np.array(epoch_distances)
    
    # Only perform curve fitting if we have enough data points
    if len(distances) >= min_epochs_for_fit:
        # Modified double exponential decay function with fixed asymptote
        def double_exp_decay(x, a1, b1, a2, b2):
            return a1 * np.exp(-b1 * x) + a2 * np.exp(-b2 * x) + baseline_distance
        
        # Initial guess: one fast decay and one slow decay component
        p0 = [0.5, 0.01, 0.5, 0.001]
        try:
            popt, pcov = curve_fit(double_exp_decay, epochs, distances, p0=p0, maxfev=10000)
            a1, b1, a2, b2 = popt
            
            # Generate smooth curve for plotting
            x_smooth = np.linspace(0, n_epochs, 1000)  # Changed to start from 0
            y_smooth = double_exp_decay(x_smooth, a1, b1, a2, b2)
            
            # Print fitting results
            print(f"\nDouble Exponential Decay Fit Results:")
            print(f"Fast component: {a1:.3f} * exp(-{b1:.6f} * x)")
            print(f"Slow component: {a2:.3f} * exp(-{b2:.6f} * x)")
            print(f"Fixed asymptote (baseline): {baseline_distance:.3f}")
            
            plt.plot(x_smooth, y_smooth, 'g:', label='Double Exp Fit (Fixed Asymptote)')
        except (RuntimeError, TypeError) as e:
            print(f"Curve fitting failed: {e}")
    else:
        print(f"Not enough epochs for curve fitting (need at least {min_epochs_for_fit}, got {len(distances)})")
    
    print(f"Current best distance: {min(epoch_distances):.3f}")
    
    plt.plot(range(0, n_epochs, step), epoch_distances, marker="o", markersize=2, label="Avg Distance per Epoch")
    plt.axhline(y=baseline_distance, color="red", linestyle="--", label="Baseline (Optimal)")
    plt.xlabel("Epoch")
    plt.ylabel("Distance")
    plt.ylim(baseline_distance / 1.1, baseline_distance * 1.2)
    plt.yscale('log')
    plt.title(f"Average Distance per Training Epoch vs. Baseline (every {step} epochs)")
    plt.legend()
    plt.savefig(f'{run_path}/train_plot.png')
    plt.close()

    # 6) Create edge overlap plot
    plt.figure(figsize=(8, 5))
    
    # Convert baseline route to edges
    baseline_route = baseline["actions"][0][plot_instance]
    optimal_edges = set(
        tuple(sorted((int(baseline_route[i].item()), int(baseline_route[(i+1) % len(baseline_route)].item()))))
        for i in range(len(baseline_route))
    )
    
    edge_overlaps = []
    
    # Calculate edge overlap for each epoch
    for i in range(0, n_epochs, step):
        with open(f'{run_path}/results/results_epoch_{i}.pkl', 'rb') as f:
            td_epoch = pickle.load(f)
            
        # Convert route to edges
        current_route = td_epoch["actions"][plot_instance]
        current_edges = set(
            tuple(sorted((int(current_route[i].item()), int(current_route[(i+1) % len(current_route)].item()))))
            for i in range(len(current_route))
        )
        
        overlap = len(optimal_edges.intersection(current_edges)) / len(optimal_edges)
        edge_overlaps.append(overlap)
    
    plt.plot(range(0, n_epochs, step), edge_overlaps, marker="o", markersize=2, 
             color='purple', label="Edge Overlap with Optimal")
    plt.xlabel("Epoch")
    plt.ylabel("Edge Overlap Ratio")
    plt.ylim(0, 1)
    plt.title(f"Edge Overlap with Optimal Solution (every {step} epochs)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(f'{run_path}/edge_overlap_plot.png')
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--fps", type=int, default=20)
    args = parser.parse_args()

    main(args)


# Optionally, remove the individual PNGs to keep things clean
# import glob
# for png_file in glob.glob(f"{run_path}/renders/*.png"):
#     if os.path.basename(png_file).startswith("epoch_"):
#         os.remove(png_file)


