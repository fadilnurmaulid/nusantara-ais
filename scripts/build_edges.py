from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

sys.path.append(str(ROOT))

from src.nusantara_ais.graph.edges import (
    build_next_edges,
    build_port_edges,
    build_trip_edges,
    build_protected_edges,
)

OUT = ROOT / "data" / "processed"

print("Loading nodes...")

ais_nodes = pd.read_parquet(
    OUT / "ais_nodes.parquet"
)

port_nodes = pd.read_parquet(
    OUT / "port_nodes.parquet"
)

trip_nodes = pd.read_parquet(
    OUT / "trip_nodes.parquet"
)

protected_nodes = pd.read_parquet(
    OUT / "protected_nodes.parquet"
)

print("Building NEXT edges...")

next_edges = build_next_edges(ais_nodes)

print("Building PORT edges...")

port_edges = build_port_edges(
    ais_nodes,
    port_nodes,
)

print("Building TRIP edges...")

trip_edges = build_trip_edges(
    ais_nodes,
    trip_nodes,
)

print("Building PROTECTED edges...")

protected_edges = build_protected_edges(
    ais_nodes,
    protected_nodes,
)

next_edges.to_parquet(
    OUT / "next_edges.parquet",
    index=False,
)

port_edges.to_parquet(
    OUT / "port_edges.parquet",
    index=False,
)

trip_edges.to_parquet(
    OUT / "trip_edges.parquet",
    index=False,
)

protected_edges.to_parquet(
    OUT / "protected_edges.parquet",
    index=False,
)

print()

print("========== DONE ==========")

print("NEXT :", len(next_edges))
print("PORT :", len(port_edges))
print("TRIP :", len(trip_edges))
print("PROTECTED :", len(protected_edges))