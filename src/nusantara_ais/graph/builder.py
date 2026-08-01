"""
Graph builder: assembles temporal snapshots of the spatially-aware
heterogeneous maritime graph as `torch_geometric.data.HeteroData` objects.

Snapshot generation
--------------------
Rather than a single static graph over the whole dataset (which would be
memory-prohibitive at fleet scale and would not let the model reason about
"what is happening right now"), the timeline is discretized into
non-overlapping snapshots of `snapshot_interval_min` minutes. Each snapshot
S_t contains every vessel_state node whose timestamp falls in
[t, t + snapshot_window_min) -- i.e. windows OVERLAP by
(snapshot_window_min - snapshot_interval_min), which lets a slow-moving
context (e.g. a loitering event) be visible and consistently connected
across several consecutive snapshots rather than being sliced awkwardly at
a hard boundary. Context nodes (port, shipping_lane_segment, grid_cell) are
shared/reconstructed identically across all snapshots since they are
static; only their EDGES to vessel_state nodes vary per snapshot.

This design directly implements the "Snapshot Generation" and "Dynamic
graph construction" requirements: the graph is a sequence of HeteroData
snapshots, each independently batchable by PyG's HeteroData DataLoader,
which also solves "Graph batching" and "Memory optimization" (each snapshot
holds only the vessel_state nodes active in its window, not the entire
fleet-history, bounding memory by fleet density * window length rather than
by total dataset size).
"""
from __future__ import annotations

import logging
from typing import List

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

from ..config import Config
from ..data.ingestion import MaritimeDataset
from . import edges as E
from . import nodes as N

logger = logging.getLogger("nusantara_ais.graph")


def assign_snapshots(meta: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    t0 = meta["timestamp"].min()
    minutes_since_start = (meta["timestamp"] - t0).dt.total_seconds() / 60.0
    meta = meta.copy()
    meta["snapshot_id"] = (minutes_since_start // cfg.graph.snapshot_interval_min).astype(int)
    return meta


class MaritimeGraphBuilder:
    def __init__(self, cfg: Config, dataset: MaritimeDataset):
        self.cfg = cfg
        self.dataset = dataset

        self.port_x, self.port_meta = N.build_port_features(dataset.ports, cfg)
        self.lane_x, self.lane_meta = N.build_lane_segment_features(dataset.lanes, cfg)
        self.port_lane_edge_index, self.port_lane_edge_attr = E.build_port_lane_edges(
            dataset.ports.reset_index(drop=True), self.lane_meta
        )

    def build_snapshots(self, vessel_df: pd.DataFrame) -> List[HeteroData]:
        cfg = self.cfg
        vessel_x, vessel_meta = N.build_vessel_state_features(vessel_df, cfg)
        vessel_meta = assign_snapshots(vessel_meta, cfg)

        grid_x, grid_meta, cell_ids_full = N.build_grid_cell_features(
            vessel_meta, self.dataset.bathymetry, self.dataset.protected_areas, cfg
        )
        located_in_index = E.build_located_in_edges(cell_ids_full, grid_meta["cell_id"].values)
        grid_adj_index = E.build_grid_adjacency_edges(grid_meta, cfg.graph.grid_cell_size_deg)

        snapshots: List[HeteroData] = []
        window_steps = max(1, cfg.graph.snapshot_window_min // cfg.graph.snapshot_interval_min)
        snapshot_ids = sorted(vessel_meta["snapshot_id"].unique())

        for s_id in snapshot_ids:
            window_mask = (vessel_meta["snapshot_id"] >= s_id) & (vessel_meta["snapshot_id"] < s_id + window_steps)
            local_idx = np.where(window_mask.values)[0]
            if len(local_idx) < 2:
                continue

            data = HeteroData()
            local_meta = vessel_meta.iloc[local_idx].reset_index(drop=True)
            local_x = vessel_x[local_idx]

            data["vessel_state"].x = local_x
            data["vessel_state"].mmsi = local_meta["mmsi"].values
            data["vessel_state"].timestamp = local_meta["timestamp"].values
            data["vessel_state"].ari = torch.tensor(local_meta["ari"].values, dtype=torch.float32)
            if "behavior_label" in local_meta.columns:
                data["vessel_state"].behavior_label = local_meta["behavior_label"].values

            data["port"].x = self.port_x
            data["shipping_lane_segment"].x = self.lane_x
            data["grid_cell"].x = grid_x

            t_idx, t_attr = E.build_temporal_edges(local_meta, tau_temporal_min=60.0)
            data["vessel_state", "temporal_next", "vessel_state"].edge_index = t_idx
            data["vessel_state", "temporal_next", "vessel_state"].edge_attr = t_attr

            sp_idx, sp_attr = E.build_spatial_proximity_edges(
                local_meta, cfg.graph.spatial_proximity_radius_km, cfg.graph.max_spatial_neighbors,
                snapshot_col="snapshot_id",
            )
            data["vessel_state", "spatial_proximity", "vessel_state"].edge_index = sp_idx
            data["vessel_state", "spatial_proximity", "vessel_state"].edge_attr = sp_attr

            port_idx, port_attr = E.build_nearest_context_edges(
                local_meta, self.port_meta, ("lat", "lon"), ("lat", "lon"), max_distance_km=50.0
            )
            data["vessel_state", "near_port", "port"].edge_index = port_idx
            data["vessel_state", "near_port", "port"].edge_attr = port_attr

            lane_idx, lane_attr = E.build_nearest_context_edges(
                local_meta, self.lane_meta, ("lat", "lon"), ("mid_lat", "mid_lon"),
                max_distance_km=cfg.graph.spatial_proximity_radius_km,
            )
            data["vessel_state", "on_lane", "shipping_lane_segment"].edge_index = lane_idx
            data["vessel_state", "on_lane", "shipping_lane_segment"].edge_attr = lane_attr

            local_cell_ids = cell_ids_full[local_idx]
            data["vessel_state", "located_in", "grid_cell"].edge_index = E.build_located_in_edges(
                local_cell_ids, grid_meta["cell_id"].values
            )
            data["grid_cell", "adjacent", "grid_cell"].edge_index = grid_adj_index
            data["port", "connected_by_lane", "shipping_lane_segment"].edge_index = self.port_lane_edge_index
            data["port", "connected_by_lane", "shipping_lane_segment"].edge_attr = self.port_lane_edge_attr

            data = _add_reverse_edges(data)
            snapshots.append(data)

        logger.info(f"Built {len(snapshots)} graph snapshots "
                    f"(interval={cfg.graph.snapshot_interval_min}min, window={cfg.graph.snapshot_window_min}min)")
        return snapshots


def _add_reverse_edges(data: HeteroData) -> HeteroData:
    """PyG heterogeneous message passing requires an edge type in both
    directions to be defined explicitly for bipartite relations (context
    nodes must also receive/send messages back for a symmetric HetGAT
    convolution stack). Homogeneous relations already used as undirected
    proxies (spatial_proximity, adjacent) get a mirrored reverse edge too so
    that the GATConv used per-relation sees a symmetric neighborhood."""
    new_edges = {}
    for (src_type, rel, dst_type), store in list(data.edge_items()):
        if src_type == dst_type and rel in ("spatial_proximity", "adjacent"):
            rev_index = store.edge_index.flip(0)
            key = (dst_type, f"rev_{rel}", src_type)
            new_edges[key] = {"edge_index": rev_index}
            if "edge_attr" in store:
                new_edges[key]["edge_attr"] = store.edge_attr
        elif src_type != dst_type:
            rev_index = store.edge_index.flip(0)
            key = (dst_type, f"rev_{rel}", src_type)
            new_edges[key] = {"edge_index": rev_index}
            if "edge_attr" in store:
                new_edges[key]["edge_attr"] = store.edge_attr
    for key, attrs in new_edges.items():
        for attr_name, val in attrs.items():
            data[key][attr_name] = val
    return data
