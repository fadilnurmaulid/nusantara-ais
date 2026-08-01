from pathlib import Path
import sys

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.nusantara_ais.graph.dataset import build_dataset

# ==========================================
# LOAD
# ==========================================

print("Loading nodes...")

ais_nodes = pd.read_parquet(
    ROOT / "data/processed/ais_nodes.parquet"
)

ais_x = pd.read_parquet(
    ROOT / "data/processed/ais_features.parquet"
)

port_nodes = pd.read_parquet(
    ROOT / "data/processed/port_nodes.parquet"
)

trip_nodes = pd.read_parquet(
    ROOT / "data/processed/trip_nodes.parquet"
)

protected_nodes = pd.read_parquet(
    ROOT / "data/processed/protected_nodes.parquet"
)

print("Loading edges...")

next_edges = pd.read_parquet(
    ROOT / "data/processed/next_edges.parquet"
)

port_edges = pd.read_parquet(
    ROOT / "data/processed/port_edges.parquet"
)

trip_edges = pd.read_parquet(
    ROOT / "data/processed/trip_edges.parquet"
)

protected_edges = pd.read_parquet(
    ROOT / "data/processed/protected_edges.parquet"
)

# ==========================================
# BUILD
# ==========================================

print("Building graph...")

data = build_dataset(
    ais_nodes,
    ais_x,
    port_nodes,
    trip_nodes,
    protected_nodes,
    next_edges,
    port_edges,
    trip_edges,
    protected_edges,
)

# ==========================================
# SAVE
# ==========================================

out = ROOT / "data/processed/hetero_graph.pt"

torch.save(data, out)

print("\nDone")
print(data)
print()
print("Saved:", out)