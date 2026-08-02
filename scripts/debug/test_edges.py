from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "src"))

import pandas as pd

from nusantara_ais.graph.nodes import *
from nusantara_ais.graph.edges import *

df = pd.read_parquet(
    ROOT / "data/processed/ais_indonesia_clean.parquet"
)

ais_nodes, _ = build_ais_nodes(df)
ports = build_port_nodes(df)
trips = build_trip_nodes(df)
protected = build_protected_nodes(df)

next_edges = build_next_edges(ais_nodes)
port_edges = build_port_edges(ais_nodes, ports)
trip_edges = build_trip_edges(ais_nodes, trips)
protected_edges = build_protected_edges(
    ais_nodes,
    protected
)

print("=" * 40)
print("NEXT EDGE")
print(next_edges.shape)

print("=" * 40)
print("PORT EDGE")
print(port_edges.shape)

print("=" * 40)
print("TRIP EDGE")
print(trip_edges.shape)

print("=" * 40)
print("PROTECTED EDGE")
print(protected_edges.shape)

print("\nExample NEXT")
print(next_edges.head())

print("\nExample PORT")
print(port_edges.head())