from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Literal, Tuple

import networkit as nk  # type: ignore
import numpy as np

BBox = Tuple[float, float, float, float]  # lat_min, lat_max, lon_min, lon_max
Weight = Literal["time", "distance"]


@dataclass(frozen=True)
class OSMGraph:
    g: nk.graph.Graph
    coords_lonlat: np.ndarray  # shape (n, 2) -> [lon, lat]
    node_ids: list[int]  # graph-index -> OSM node id
    id_to_idx: dict[int, int]  # OSM node id -> graph-index
    stats: Dict[str, Any]


def _bbox_to_pyrosm(bbox: BBox) -> list[float]:
    lat_min, lat_max, lon_min, lon_max = bbox
    return [lon_min, lat_min, lon_max, lat_max]


def _compute_length_m(edge_row: Any) -> float:
    try:
        from shapely.geometry import LineString  # type: ignore
    except Exception:
        LineString = None  # type: ignore
    length = getattr(edge_row, "length", None)
    if length is not None:
        try:
            return float(length)
        except Exception:
            pass
    geom = getattr(edge_row, "geometry", None)
    if LineString is not None and isinstance(geom, LineString):
        return float(geom.length)
    return 0.0


def _parse_maxspeed_mps(ms: Any, default_kmh: float = 35.0, convert_mph: bool = False) -> float:
    if ms is None:
        return default_kmh / 3.6
    if isinstance(ms, list) and ms:
        ms = ms[0]
    try:
        if isinstance(ms, str):
            s = ms.strip().lower()
            token = s.split()[0]
            v = float(token)
            if convert_mph and "mph" in s:
                v = v * 1.60934
            return v / 3.6
        return float(ms) / 3.6
    except Exception:
        return default_kmh / 3.6


def load_osm_network(pbf: Path, bbox: BBox, network_type: str = "driving"):
    import pyrosm  # type: ignore
    bb = _bbox_to_pyrosm(bbox)
    osm = pyrosm.OSM(str(pbf), bounding_box=bb)
    nodes_df, edges_df = osm.get_network(network_type=network_type, nodes=True, extra_attributes=["maxspeed"])
    return nodes_df, edges_df


def build_graph(
    nodes_df,
    edges_df,
    *,
    weight: Weight,
    default_kmh: float = 35.0,
    convert_mph: bool = False,
) -> OSMGraph:
    id_to_idx: Dict[int, int] = {}
    node_ids: list[int] = []
    lats: list[float] = []
    lons: list[float] = []

    for row in nodes_df.itertuples(index=False):
        nid = int(getattr(row, "id"))
        id_to_idx[nid] = len(node_ids)
        node_ids.append(nid)
        lats.append(float(getattr(row, "lat")))
        lons.append(float(getattr(row, "lon")))

    g = nk.graph.Graph(n=len(node_ids), weighted=True, directed=False)
    lengths = []
    speeds = []
    default_speed = default_kmh / 3.6

    for row in edges_df.itertuples(index=False):
        u = getattr(row, "u", None)
        v = getattr(row, "v", None)
        if u is None or v is None:
            continue
        if u not in id_to_idx or v not in id_to_idx:
            continue
        ui = id_to_idx[int(u)]
        vi = id_to_idx[int(v)]
        length_m = _compute_length_m(row)
        if weight == "time":
            speed_mps = _parse_maxspeed_mps(
                getattr(row, "maxspeed", None), default_kmh=default_kmh, convert_mph=convert_mph
            )
            speed_mps = speed_mps if speed_mps > 0 else default_speed
            w = length_m / speed_mps
            speeds.append(speed_mps)
        else:
            w = length_m
        g.addEdge(ui, vi, w=w)
        lengths.append(length_m)

    stats: Dict[str, Any] = {
        "nodes": g.numberOfNodes(),
        "edges": g.numberOfEdges(),
        "avg_length_m": float(np.mean(lengths)) if lengths else None,
        "avg_speed_mps": float(np.mean(speeds)) if speeds and weight == "time" else None,
        "weight": weight,
        "default_kmh": default_kmh,
    }
    coords = np.column_stack([np.array(lons), np.array(lats)])
    return OSMGraph(g=g, coords_lonlat=coords, node_ids=node_ids, id_to_idx=id_to_idx, stats=stats)


def largest_component(graph: OSMGraph) -> OSMGraph:
    ug = nk.graphtools.toUndirected(graph.g)
    comps = nk.components.ConnectedComponents(ug)
    comps.run()
    comps_list = comps.getComponents()
    if not comps_list:
        return graph
    keep_nodes = max(comps_list, key=len)
    mapping = {old: new for new, old in enumerate(keep_nodes)}

    sub = nk.graph.Graph(n=len(keep_nodes), weighted=True, directed=False)
    for u, v, w in graph.g.iterEdgesWeights():
        if u in mapping and v in mapping:
            sub.addEdge(mapping[u], mapping[v], w=w)

    sub_coords = graph.coords_lonlat[keep_nodes]
    sub_node_ids = [graph.node_ids[i] for i in keep_nodes]
    sub_id_to_idx = {nid: i for i, nid in enumerate(sub_node_ids)}
    return OSMGraph(g=sub, coords_lonlat=sub_coords, node_ids=sub_node_ids, id_to_idx=sub_id_to_idx, stats=graph.stats)
