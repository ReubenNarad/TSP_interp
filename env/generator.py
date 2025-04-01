import torch
from typing import Union, Sequence
from tensordict import TensorDict
from rl4co.envs.common.utils import Generator  # Ensure correct import path

class CustomTSPGenerator(Generator):
    """
    A custom TSP generator that places half of the cities in one uniform region
    and the other half in a non-overlapping region.
    """
    def __init__(
        self,
        num_loc: int = 20,
        min_loc: float = 0.0,
        cluster_bound: float = 0.5,
        max_loc: float = 1.0,
        init_sol_type: str = "random",
    ):
        """
        Args:
            num_loc: Total number of locations.
            min_loc: Minimum coordinate for cluster 1 (e.g., 0.0).
            cluster_bound: Boundary between cluster 1 and cluster 2 (e.g., 0.5).
            max_loc: Maximum coordinate for cluster 2 (e.g., 1.0).
            init_sol_type: "random" or "greedy" or any custom logic for an initial solution.
        """
        super().__init__()
        self.num_loc = num_loc
        self.min_loc = min_loc
        self.cluster_bound = cluster_bound
        self.max_loc = max_loc
        self.init_sol_type = init_sol_type

    def _generate(self, batch_size: Union[int, Sequence[int], torch.Size]) -> TensorDict:
        """
        Generate a batch of TSP instances.

        Returns:
            A TensorDict with shape [batch_size] whose entries include:
              - "locs": (batch_size, num_loc, 2) coordinates of the TSP cities
              - "init_sol": (batch_size, num_loc) an (optional) initial solution
        """
        # Log the incoming batch_size for debugging
        print(f"BATCH SIZE: {batch_size}")

        # Initialize batch_size_int to None
        batch_size_int = None

        # Handle different types of batch_size
        if isinstance(batch_size, torch.Size):
            # Always treat empty torch.Size as batch_size=1
            batch_size_int = 1
        elif isinstance(batch_size, Sequence):
            if len(batch_size) == 0:
                batch_size_int = 1
            else:
                batch_size_int = batch_size[0]
        elif isinstance(batch_size, int):
            batch_size_int = batch_size
        else:
            raise ValueError(f"batch_size must be int, torch.Size, or a sequence, got {batch_size}")

        # Further validation
        if not isinstance(batch_size_int, int) or batch_size_int <= 0:
            raise ValueError(f"Processed batch_size must be a positive integer, got {batch_size_int}")

        # Now, batch_size_int is a valid integer
        # Split the total number of cities into two groups
        cluster1_size = self.num_loc // 2
        cluster2_size = self.num_loc - cluster1_size

        # Sample cluster1 uniformly from [min_loc, cluster_bound] x [min_loc, cluster_bound]
        locs_c1 = torch.rand(batch_size_int, cluster1_size, 2)
        locs_c1 = locs_c1 * (self.cluster_bound - self.min_loc) + self.min_loc

        # Sample cluster2 uniformly from [cluster_bound, max_loc] x [cluster_bound, max_loc]
        locs_c2 = torch.rand(batch_size_int, cluster2_size, 2)
        locs_c2 = locs_c2 * (self.max_loc - self.cluster_bound) + self.cluster_bound

        # Concatenate the two sets of cities
        locs = torch.cat([locs_c1, locs_c2], dim=1)
        
        # Optionally shuffle them so they're not grouped by cluster in the final array
        perm = torch.stack([torch.randperm(self.num_loc) for _ in range(batch_size_int)], dim=0)
        locs = torch.gather(
            locs, 
            1, 
            perm.unsqueeze(-1).expand(batch_size_int, self.num_loc, 2)
        )

        # Build initial solutions if desired
        print(f"INIT SOL TYPE: {self.init_sol_type}")
        if self.init_sol_type == "random":
            # A random permutation of city indices for each instance
            init_sol = torch.stack([torch.randperm(self.num_loc) for _ in range(batch_size_int)])
        else:
            # Or do some heuristic, or skip entirely.
            init_sol = torch.zeros(batch_size_int, self.num_loc, dtype=torch.long)

        # Wrap this up in a TensorDict
        data = TensorDict(
            {
                "locs": locs,              # (batch_size, num_loc, 2)
                "init_sol": init_sol,      # (batch_size, num_loc)
            },
            batch_size=[batch_size_int],
        )
        return data