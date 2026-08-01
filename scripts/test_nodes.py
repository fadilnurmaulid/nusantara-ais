from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

import pandas as pd

from nusantara_ais.graph.nodes import (
    build_ais_nodes,
    build_port_nodes,
    build_trip_nodes,
    build_protected_nodes,
)

df = pd.read_parquet(
    ROOT / "data/processed/ais_indonesia_clean.parquet"
)

ais_nodes, x = build_ais_nodes(df)
port_nodes = build_port_nodes(df)
trip_nodes = build_trip_nodes(df)
protected_nodes = build_protected_nodes(df)

print("=" * 40)
print("AIS")
print(len(ais_nodes))
print(x.shape)

print("=" * 40)
print("PORT")
print(len(port_nodes))

print("=" * 40)
print("TRIP")
print(len(trip_nodes))

print("=" * 40)
print("PROTECTED")
print(len(protected_nodes))