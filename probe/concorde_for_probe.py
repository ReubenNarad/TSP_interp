# solve with concorde, collecting run times as well
import torch
import subprocess, os, pickle, argparse, re, time
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from distributions import Uniform, DualUniform, RandomUniform, FuzzyCircle, HybridSampler
# Assuming TSPEnv is needed if we load env.pkl which might contain it
from rl4co.envs import TSPEnv

def write_tsplib_file(filename, coords):
    with open(filename, 'w') as f:
        f.write("NAME: td_tsp\n")
        f.write("TYPE: TSP\n")
        f.write(f"DIMENSION: {coords.shape[0]}\n")
        f.write("EDGE_WEIGHT_TYPE: EUC_2D\n")
        f.write("NODE_COORD_SECTION\n")
        for idx, (x, y) in enumerate(coords):
            f.write(f"{idx + 1} {float(x*100):.4f} {float(y*100):.4f}\n")
        f.write("EOF\n")

def run_concorde(instance_filename, solution_filename):
    base_path = os.path.splitext(instance_filename)[0]
    bb_nodes = -1
    run_time = -1.0
    lp_rows = -1
    lp_cols = -1
    lp_nonzeros = -1
    
    try:
        start = time.time()
        result = subprocess.run(
            ['concorde', '-o', solution_filename, instance_filename],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        end = time.time()
        run_time = end - start

        # Extract B&B nodes
        match = re.search(r"Number of bbnodes:\s*(\d+)", result.stdout)
        if match:
            bb_nodes = int(match.group(1))
        else:
            match_err = re.search(r"Number of bbnodes:\s*(\d+)", result.stderr)
            if match_err:
                bb_nodes = int(match_err.group(1))

        # Extract run time
        match_time = re.search(r"Total Running Time:\s*([0-9.]+)", result.stdout)
        if match_time:
            run_time = float(match_time.group(1))
        else:
            match_time_err = re.search(r"Total Running Time:\s*([0-9.]+)", result.stderr)
            if match_time_err:
                run_time = float(match_time_err.group(1))
        
        # Extract LP statistics
        lp_stats_match = re.search(r"Final LP has (\d+) rows, (\d+) columns, (\d+) nonzeros", result.stdout)
        if lp_stats_match:
            lp_rows = int(lp_stats_match.group(1))
            lp_cols = int(lp_stats_match.group(2))
            lp_nonzeros = int(lp_stats_match.group(3))
        else:
            lp_stats_match_err = re.search(r"Final LP has (\d+) rows, (\d+) columns, (\d+) nonzeros", result.stderr)
            if lp_stats_match_err:
                lp_rows = int(lp_stats_match_err.group(1))
                lp_cols = int(lp_stats_match_err.group(2))
                lp_nonzeros = int(lp_stats_match_err.group(3))

    except subprocess.CalledProcessError as e:
        print(f"Error executing Concorde for {instance_filename}:")
        print(e.stderr)
        
        # Try to extract info from error output as before
        match = re.search(r"Number of bbnodes:\s*(\d+)", e.stderr)
        if match:
            bb_nodes = int(match.group(1))
        match_time = re.search(r"Total Running Time:\s*([0-9.]+)", e.stderr)
        if match_time:
            run_time = float(match_time.group(1))
        lp_stats_match = re.search(r"Final LP has (\d+) rows, (\d+) columns, (\d+) nonzeros", e.stderr)
        if lp_stats_match:
            lp_rows = int(lp_stats_match.group(1))
            lp_cols = int(lp_stats_match.group(2))
            lp_nonzeros = int(lp_stats_match.group(3))
    
    # Clean up files as before
    extensions = ['.pul', '.sav', '.res', '.mas', '.pix', '.sol']
    base_stub = os.path.basename(base_path)
    for ext in extensions:
        temp_file = f"{base_stub}{ext}"
        o_temp_file = "O" + temp_file
        if os.path.exists(temp_file):
            os.remove(temp_file)
        if os.path.exists(o_temp_file):
            os.remove(o_temp_file)
            
    return bb_nodes, run_time, lp_rows, lp_cols, lp_nonzeros

def read_solution_file(filename):
    tour = []
    with open(filename, 'r') as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if i == 0:
            continue
        numbers = [int(x) for x in line.strip().split()]
        for num in numbers:
            if num == -1:
                break
            tour.append(num)
    return tour

def compute_tour_length(coords, route):
    route_coords = coords[route]
    dist_vec = (route_coords - route_coords.roll(-1, dims=0)).pow(2).sum(-1).sqrt()
    return dist_vec.sum()

def main(args):
    run_path = f'runs/{args.run_name}'
    probe_dir = os.path.join(run_path, 'probe')
    output_td = os.path.join(probe_dir, 'baseline_probe.pkl')
    os.makedirs(probe_dir, exist_ok=True)

    env_path = os.path.join(run_path, "env.pkl")
    with open(env_path, "rb") as f:
        env = pickle.load(f)
    print(f"Loaded environment from {env_path}")

    print(f"Generating {args.num_instances} new instances...")
    generated_td = env.reset(batch_size=[args.num_instances]).to('cpu')
    locs = generated_td["locs"] # Shape: [num_instances, num_loc, 2]
    batch_size = locs.shape[0]
    print(f"Generated {batch_size} instances with {locs.shape[1]} nodes each.")


    all_solutions = []
    all_rewards = []
    all_bb_nodes = []
    all_run_times = []
    all_lp_rows = []     # New lists for LP statistics
    all_lp_cols = []
    all_lp_nonzeros = []

    scratch_folder = "concorde_temp"
    os.makedirs(scratch_folder, exist_ok=True)

    print(f"Solving {batch_size} generated instances with Concorde (tracking LP stats)...")
    from tqdm import tqdm
    for i in tqdm(range(batch_size), desc="Solving Instances"):
        coords_i = locs[i] # Get coordinates for the i-th generated instance
        tsp_filename = os.path.join(scratch_folder, f"instance_{i}.tsp")
        sol_filename = os.path.join(scratch_folder, f"solution_{i}.sol")
        
        write_tsplib_file(tsp_filename, coords_i)
        
        bb_nodes, run_time, lp_rows, lp_cols, lp_nonzeros = run_concorde(tsp_filename, sol_filename)

        # Handle cases where Concorde might fail to produce a solution file
        if os.path.exists(sol_filename):
            route_1based = read_solution_file(sol_filename)
            # Convert 1-based route from Concorde to 0-based
            # Ensure route has the correct number of nodes, otherwise pad or handle error
            if len(route_1based) == coords_i.shape[0]:
                 route_0based = [x - 1 for x in route_1based]
                 route_t = torch.tensor(route_0based, dtype=torch.long)
                 tour_length = compute_tour_length(coords_i, route_t)
                 reward = -tour_length.clone().detach()
            else:
                 print(f"Warning: Solution for instance {i} has incorrect length ({len(route_1based)} vs {coords_i.shape[0]}). Skipping.")
                 # Append placeholders or handle as needed
                 route_t = torch.zeros(coords_i.shape[0], dtype=torch.long) # Placeholder
                 reward = torch.tensor(float('-inf')) # Placeholder
                 bb_nodes = -1 # Mark as failed
                 run_time = -1.0 # Mark as failed
                 lp_rows, lp_cols, lp_nonzeros = -1, -1, -1 # Mark LP stats as failed too
        else:
            print(f"Warning: Solution file not found for instance {i}. Skipping.")
            route_t = torch.zeros(coords_i.shape[0], dtype=torch.long) # Placeholder
            reward = torch.tensor(float('-inf')) # Placeholder
            bb_nodes = -1 # Mark as failed
            run_time = -1.0 # Mark as failed
            lp_rows, lp_cols, lp_nonzeros = -1, -1, -1 # Mark LP stats as failed too


        all_solutions.append(route_t)
        all_rewards.append(reward)
        all_bb_nodes.append(bb_nodes)
        all_run_times.append(run_time)
        all_lp_rows.append(lp_rows)        # Save LP statistics
        all_lp_cols.append(lp_cols)
        all_lp_nonzeros.append(lp_nonzeros)

    print("\n[main] Aggregating results...")
    optimal_routes = torch.stack(all_solutions, dim=0)
    optimal_rewards = torch.stack(all_rewards, dim=0)
    optimal_bb_nodes = torch.tensor(all_bb_nodes, dtype=torch.long)
    optimal_run_times = torch.tensor(all_run_times, dtype=torch.float)
    optimal_lp_rows = torch.tensor(all_lp_rows, dtype=torch.long)       # New tensors
    optimal_lp_cols = torch.tensor(all_lp_cols, dtype=torch.long)
    optimal_lp_nonzeros = torch.tensor(all_lp_nonzeros, dtype=torch.long)

    # Filter out failed instances for calculating averages
    valid_mask = optimal_rewards != float('-inf')
    num_solved = valid_mask.sum().item()
    print(f"\nSuccessfully solved {num_solved}/{batch_size} instances.")

    if num_solved > 0:
        avg_dist = -optimal_rewards[valid_mask].mean().item()
        valid_nodes_mask = (optimal_bb_nodes[valid_mask] != -1)
        avg_nodes = optimal_bb_nodes[valid_mask][valid_nodes_mask].float().mean().item() if valid_nodes_mask.any() else 'N/A'
        valid_times_mask = (optimal_run_times[valid_mask] > 0)
        avg_time = optimal_run_times[valid_mask][valid_times_mask].mean().item() if valid_times_mask.any() else 'N/A'
        avg_rows = optimal_lp_rows[valid_mask].float().mean().item() if valid_mask.any() else 'N/A'
        avg_cols = optimal_lp_cols[valid_mask].float().mean().item() if valid_mask.any() else 'N/A'
        avg_nonzeros = optimal_lp_nonzeros[valid_mask].float().mean().item() if valid_mask.any() else 'N/A'
        print(f"Avg distance (solved): {avg_dist:.4f}, Avg B&B nodes: {avg_nodes}, Avg time: {avg_time:.4f}s")
        print(f"Avg LP stats - Rows: {avg_rows:.1f}, Cols: {avg_cols:.1f}, Nonzeros: {avg_nonzeros:.1f}")

    print("[main] Preparing results dictionary...")
    results = {
        'locs': locs, # Save the generated locations!
        'actions': [optimal_routes],
        'rewards': [optimal_rewards],
        'bb_nodes': [optimal_bb_nodes],
        'run_times': [optimal_run_times],
        'lp_rows': [optimal_lp_rows],
        'lp_cols': [optimal_lp_cols],
        'lp_nonzeros': [optimal_lp_nonzeros]
    }

    print(f"[main] Saving results to {output_td}...")
    with open(output_td, 'wb') as f:
        pickle.dump(results, f)
    print(f"Saved results for {batch_size} generated instances (including locs) to {output_td}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate TSP instances, solve with Concorde, and record metrics.")
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--num_instances", type=int, default=5000, help="Number of instances to generate and solve")
    args = parser.parse_args()
    main(args)