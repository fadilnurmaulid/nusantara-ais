from pathlib import Path

import torch
from torch_geometric.data import HeteroData

ROOT = Path(__file__).resolve().parents[3]

def to_tensor(df):

    return torch.tensor(
        df.values,
        dtype=torch.float32
    )


def edge_tensor(edge):

    return torch.tensor(
        edge[["source", "target"]].values.T,
        dtype=torch.long
    )


def build_dataset(

    ais_nodes,
    ais_x,

    port_nodes,
    trip_nodes,
    protected_nodes,

    next_edges,
    port_edges,
    trip_edges,
    protected_edges

):

    data = HeteroData()

    # ============================================
    # AIS NODE
    # ============================================

    data["ais"].x = to_tensor(ais_x)

    # ============================================
    # PORT
    # ============================================

    data["port"].x = torch.ones(
        (
            len(port_nodes),
            1
        )
    )

    # ============================================

    data["trip"].x = torch.tensor(

        trip_nodes[
            [
                "duration",
                "points",
                "mean_speed",
            ]
        ].values,

        dtype=torch.float32

    )

    # ============================================

    data["protected"].x = torch.ones(
        (
            len(protected_nodes),
            1
        )
    )

    # ============================================
    # EDGE
    # ============================================

    data[
        "ais",
        "next",
        "ais"
    ].edge_index = edge_tensor(next_edges)

    data[
        "ais",
        "near",
        "port"
    ].edge_index = edge_tensor(port_edges)

    data[
        "ais",
        "trip",
        "trip"
    ].edge_index = edge_tensor(trip_edges)

    data[
        "ais",
        "protected",
        "protected"
    ].edge_index = edge_tensor(
        protected_edges
    )

    return data

def load_dataset():

    data = torch.load(
        ROOT / "data" / "processed" / "hetero_graph.pt",
        weights_only=False,
    )

    return data