# Plan: Seattle Road-Network TSP RL Environment

Goal: Train a policy on road-network (non-Euclidean) TSP instances sampled from a real map region (e.g. Seattle) where edge costs come from shortest-path travel times (or distances) on the road graph.

## 0) Decisions to Lock In
- **Map region**: pick `pbf` + `bbox` (lat_min, lat_max, lon_min, lon_max), plus an optional tighter `sample_bbox` for “addresses”.
- **Cost semantics**: `time` (seconds) vs `distance` (meters), symmetric vs asymmetric.
- **Instance size**: training `N` (e.g. 50/100) and target evaluation `N`.
- **Pool size**: `K` (e.g. 5k–20k) total “addresses” (OSM nodes) used to form a reusable pool.

## 1) Build the Road Graph (One-Time per bbox)
- Load the OSM network from `pbf` + `bbox`.
- Convert to a weighted graph (Networkit) using `time` or `distance` weights.
- Keep only the largest connected component.
- Persist metadata: bbox, network type, graph stats.

## 2) Choose the Address Pool (One-Time per pool)
- Sample `K` graph nodes inside `sample_bbox` (or the full bbox) using a sampling strategy:
  - uniform random (default), or
  - BFS-ball sampling (for “connected neighborhoods”), or
  - stratified grid sampling (for coverage).
- Persist:
  - `pool_node_ids.json`: list of OSM node ids (and bbox/network_type/seed).
  - `pool_coords_lonlat.npy`: `K×2` lon/lat (for plotting).

## 3) Compute the `K×K` Cost Matrix (Offline, Chunked)
Baseline implementation (fast enough for Seattle-sized graphs):
- Run `K` single-source shortest-path computations on the road graph and gather distances to the other `K` pool nodes.
- Use chunking + memmap storage:
  - Write to `pool_cost_matrix.memmap` (shape `K×K`, dtype `float32` or quantized `uint16/uint32` with a scale factor).
  - Optionally store symmetry handling (`min`, `avg`, `none`) as a config flag rather than rewriting.

Optional “hierarchy engine” implementation (OSRM/Valhalla/GraphHopper):
- Preprocess the road graph once (CH/MLD).
- Use the engine’s many-to-many matrix API in blocks (e.g. 512×512 or 1024×1024) to fill the memmap.
- This is most useful when the raw graph is much larger than the Seattle bbox and Dijkstra becomes expensive.

Validation checks after build:
- diagonal is 0
- no NaNs/Infs (or a defined fallback)
- random triangle inequality spot checks (won’t strictly hold for directed times unless symmetrized)

## 4) RL4CO Environment: Sample `N` Nodes → Return `N×N` Submatrix
Implement a generator/env that:
- Loads the persisted pool:
  - `pool_coords_lonlat.npy` (optional, for plotting)
  - `pool_cost_matrix.memmap` (required)
- For each instance:
  - sample indices `idx ∈ [0..K)` of size `N` without replacement
  - return `td["cost_matrix"] = cost[idx, idx]`
  - optionally return `td["locs"] = coords[idx]` (lon/lat) so plotting does not need extra files
- Train with `ATSPEnv` (supports `cost_matrix`) + `MatNetPolicy`.

Generalization strategy:
- create multiple pools (different seeds / neighborhoods) and mix them, or
- hold out a disjoint set of pool nodes for validation/test, or
- use multiple bboxes.

## 5) Training + Evaluation
- Training:
  - `policy/train_matnet.py` as the entrypoint (already exists), extended to read the new pool+memmap format.
- Monitoring:
  - track `val/reward` on a fixed validation batch sampled once from the pool.
  - periodically compute a Concorde baseline for small `N` (only feasible for small instances and symmetric costs).
- Rendering:
  - For GIFs/plots, snap edges to road paths using Dijkstra on the *road graph* and plot lon/lat with road background.
  - Ensure we use the pool’s original bbox and node_ids (not nearest-neighbor snapping) when available.

## 6) Deliverables
- `scripts/build_seattle_pool.py` (or similar): creates pool nodes + memmap cost matrix + metadata.
- `env/pool_submatrix_generator.py`: samples `N` nodes from pool and returns `N×N` cost matrices.
- Updated training script to use the pool generator.
- Updated plotting/GIF pipeline to match the GEPA-style road-snapped visualization.

## 7) Risks / Constraints
- Storage scales as `O(K^2)`: choose `K` based on disk/RAM budget; prefer memmap + quantization.
- Directed travel times imply ATSP; Concorde baselines require symmetrization (or an ATSP solver).
- Throughput depends on whether you do offline precompute (fast training) vs on-the-fly shortest paths (slow training).

