import torch
import subprocess, os, pickle, argparse
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from distributions import Uniform, DualUniform, RandomUniform, FuzzyCircle, HybridSampler

def write_tsplib_file(filename, coords):
    """
    Write a single TSP instance to file in TSPLIB format.

    Args:
        filename (str): Path to file to be written.
        coords (Tensor): Float tensor of shape [num_locs, 2].
                         Each row is (x, y) in [0,1] or some range.
    """
    with open(filename, 'w') as f:
        f.write("NAME: td_tsp\n")
        f.write("TYPE: TSP\n")
        f.write(f"DIMENSION: {coords.shape[0]}\n")
        f.write("EDGE_WEIGHT_TYPE: EUC_2D\n")
        f.write("NODE_COORD_SECTION\n")
        for idx, (x, y) in enumerate(coords):
            # TSPLIB node indices start at 1
            # Scale coordinates by 100 for concorde stability (should not affect solution)
            f.write(f"{idx + 1} {float(x*100):.4f} {float(y*100):.4f}\n")
        f.write("EOF\n")


def run_concorde(instance_filename, solution_filename):
    """
    Run Concorde on the given TSP instance file and save solution to solution_filename.

    Args:
        instance_filename (str): Path to .tsp file in TSPLIB format.
        solution_filename (str): Desired output .sol or .txt file from Concorde.
    """
    base_path = os.path.splitext(instance_filename)[0]
    try:
        subprocess.run(
            ['concorde', '-o', solution_filename, instance_filename],
            check=True,
            text=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE
        )
    except subprocess.CalledProcessError as e:
        print("Error executing Concorde:")
        print(e.stderr)
        raise

    # Clean up Concorde's intermediate files
    extensions = ['.pul', '.sav', '.res', '.mas', '.pix', '.sol']
    base_stub = os.path.basename(base_path)
    for ext in extensions:
        # Concorde uses <basename> + ext or 'O' + <basename> + ext
        temp_file = f"{base_stub}{ext}"
        o_temp_file = "O" + temp_file
        if os.path.exists(temp_file):
            os.remove(temp_file)
        if os.path.exists(o_temp_file):
            os.remove(o_temp_file)


def read_solution_file(filename):
    """
    Concorde's solution file expects lines with city indices (1-based)
    separated by whitespace, ending with -1. This function reads them
    and returns a route in 0-based indexing.

    Args:
        filename (str): Path to the concorde solution file.

    Returns:
        tour (list): The route as a list of node indices (0-based).
    """
    tour = []
    with open(filename, 'r') as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        # The first line is often a second copy of dimension,
        # so we skip it or parse carefully
        if i == 0:
            # Could parse dimension from here if needed, but skip
            continue
        # Each line can contain multiple nodes, separated by space, ending with -1
        # e.g. "1 3 4 -1"
        numbers = [int(x) for x in line.strip().split()]
        # Exclude the trailing -1 (if present)
        for num in numbers:
            if num == -1:
                break
            tour.append(num)
    return tour


def compute_tour_length(coords, route):
    """
    Compute the total Euclidean tour length for a route that visits
    all nodes in 'route' order and returns to the start.

    coords: [num_locs, 2]  FloatTensor
    route: [num_locs]  1D long tensor with the visiting order (0-based).

    Returns:
        Scalar float for total distance of the route (closed tour).
    """
    # Gather the coordinates in the visiting order
    route_coords = coords[route]

    # Compute pairwise distances, including the wrap from last to first
    # roll(-1) shifts everything by one to line up neighbors
    dist_vec = (route_coords - route_coords.roll(-1, dims=0)).pow(2).sum(-1).sqrt()
    return dist_vec.sum()


def main(args):
    run_path = f'runs/{args.run_name}'
    input_td = os.path.join(run_path, 'val_td.pkl')
    output_td = os.path.join(run_path, 'baseline.pkl')

    with open(input_td, 'rb') as f:
        td = pickle.load(f)
    
    locs = td["locs"]  # shape: [batch_size, num_locs, 2]
    batch_size = locs.shape[0]
    
    all_solutions = []
    all_rewards = []  # We'll store -tour_length here, consistent with how TSPEnv does it

    # Create a local scratch folder for TSP files
    scratch_folder = "concorde_temp"
    os.makedirs(scratch_folder, exist_ok=True)

    print(f"Solving {batch_size} instances...")
    for i in range(batch_size):
        coords_i = locs[i]  # [num_locs, 2]
        tsp_filename = os.path.join(scratch_folder, f"instance_{i}.tsp")
        sol_filename = os.path.join(scratch_folder, f"solution_{i}.sol")

        write_tsplib_file(tsp_filename, coords_i)

        # 3) Run Concorde
        run_concorde(tsp_filename, sol_filename)

        # 4) Read the solution route
        route_0based = read_solution_file(sol_filename)

        # Convert route to a torch tensor
        route_t = torch.tensor(route_0based, dtype=torch.long)

        # Compute the TSP reward like in TSPEnv (reward = -tour_length)
        tour_length = compute_tour_length(coords_i, route_t)
        reward = -tour_length.clone().detach()

        all_solutions.append(route_t)
        all_rewards.append(reward)

    # Convert to tensors
    optimal_routes = torch.stack(all_solutions, dim=0)      # [batch_size, num_locs]
    optimal_rewards = torch.stack(all_rewards, dim=0)       # [batch_size]

    # Store similarly to how training code stores actions/rewards
    results = {
        'actions': [optimal_routes],  # same shape as your model's output
        'rewards': [optimal_rewards]  # match the style: a list with one tensor
    }

    print(f"Optimally solved {batch_size} instances with avg distance {- optimal_rewards.mean().item()}")

    # 5) Save the results
    with open(output_td, 'wb') as f:
        pickle.dump(results, f)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, default=None)
    args = parser.parse_args()
    main(args)