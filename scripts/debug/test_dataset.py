from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]

sys.path.append(str(ROOT / "src"))

import pandas as pd

from nusantara_ais.graph.nodes import *
from nusantara_ais.graph.edges import *
from nusantara_ais.graph.dataset import *

df = pd.read_parquet(
    ROOT / "data/processed/ais_indonesia_clean.parquet"
)

ais_nodes, x = build_ais_nodes(df)

ports = build_port_nodes(df)

trips = build_trip_nodes(df)

protected = build_protected_nodes(df)

next_edges = build_next_edges(ais_nodes)

port_edges = build_port_edges(
    ais_nodes,
    ports
)

trip_edges = build_trip_edges(
    ais_nodes,
    trips
)

protected_edges = build_protected_edges(
    ais_nodes,
    protected
)

graph = build_dataset(

    ais_nodes,
    x,

    ports,
    trips,
    protected,

    next_edges,
    port_edges,
    trip_edges,
    protected_edges

)

print(graph)

print()

print(graph.metadata())

print()

print(graph["ais"].x.shape)

print(graph["trip"].x.shape)

print(graph["ais", "next", "ais"].edge_index.shape)

print(graph["ais", "near", "port"].edge_index.shape)