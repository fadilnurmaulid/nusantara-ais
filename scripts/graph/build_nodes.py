from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

sys.path.append(str(ROOT))

from src.nusantara_ais.graph.nodes import (
    build_ais_nodes,
    build_port_nodes,
    build_trip_nodes,
    build_protected_nodes,
)

DATA = ROOT / "data" / "processed" / "ais_dataset_v1.parquet"

print("Loading dataset...")

df = pd.read_parquet(DATA)

print(df.shape)

print("Building AIS nodes...")

ais_nodes, ais_features = build_ais_nodes(df)

print("Building Port nodes...")

port_nodes = build_port_nodes(df)

print("Building Trip nodes...")

trip_nodes = build_trip_nodes(df)

print("Building Protected nodes...")

protected_nodes = build_protected_nodes(df)

OUT = ROOT / "data" / "processed"

ais_nodes.to_parquet(
    OUT / "ais_nodes.parquet",
    index=False,
)

ais_features.to_parquet(
    OUT / "ais_features.parquet",
    index=False,
)

port_nodes.to_parquet(
    OUT / "port_nodes.parquet",
    index=False,
)

trip_nodes.to_parquet(
    OUT / "trip_nodes.parquet",
    index=False,
)

protected_nodes.to_parquet(
    OUT / "protected_nodes.parquet",
    index=False,
)

print()

print("========== DONE ==========")

print("AIS :", len(ais_nodes))
print("PORT :", len(port_nodes))
print("TRIP :", len(trip_nodes))
print("PROTECTED :", len(protected_nodes))