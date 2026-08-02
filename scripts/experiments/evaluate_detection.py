from pathlib import Path
import sys

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.nusantara_ais.graph.dataset import load_dataset
from src.nusantara_ais.models.baseline import (
    GCNAutoEncoder,
    GATAutoEncoder,
    GraphSAGEAutoEncoder,
)
from src.nusantara_ais.models.hetgat import HetGATAutoEncoder

# ==========================================================
# DEVICE
# ==========================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device :", device)

# ==========================================================
# GRAPH
# ==========================================================

print("Loading graph...")

data = load_dataset().to(device)

dataset = pd.read_parquet(
    ROOT / "data" / "processed" / "ais_dataset_v1.parquet"
)

x = data["ais"].x

edge_index = data[
    "ais",
    "next",
    "ais"
].edge_index

# ==========================================================
# FUNCTION
# ==========================================================

def evaluate(name, score):

    tmp = dataset.copy()

    tmp["score"] = score.cpu().numpy()

    top100 = (
        tmp
        .sort_values(
            "score",
            ascending=False,
        )
        .head(100)
    )

    return {

        "model": name,

        "mean_score": float(score.mean()),

        "std_score": float(score.std()),

        "max_score": float(score.max()),

        "top100_unique_mmsi":
            top100["mmsi"].nunique(),

        "top100_unique_trip":
            top100["trip_id"].nunique(),

        "top100_inside_eez":

            int(
                top100[
                    "inside_indonesia_eez"
                ].sum()
            ),

        "top100_shipping_lane":

            int(
                top100[
                    "inside_shipping_lane"
                ].sum()
            ),

        "top100_protected":

            int(
                top100[
                    "inside_protected_area"
                ].sum()
            ),

    }

# ==========================================================
# RESULT
# ==========================================================

results = []

MODEL_DIR = ROOT / "data" / "model"

# ==========================================================
# GCN
# ==========================================================

model = GCNAutoEncoder(
    in_channels=x.shape[1]
).to(device)

model.load_state_dict(
    torch.load(
        MODEL_DIR / "gcn.pt",
        map_location=device,
        weights_only=True,
    )
)

model.eval()

with torch.no_grad():

    x_hat, z, loss = model(
        x,
        edge_index,
    )

    score = (
        (x - x_hat)
        .pow(2)
        .mean(dim=1)
    )

results.append(
    evaluate(
        "GCN",
        score,
    )
)

# ==========================================================
# GAT
# ==========================================================

model = GATAutoEncoder(
    in_channels=x.shape[1]
).to(device)

model.load_state_dict(
    torch.load(
        MODEL_DIR / "gat.pt",
        map_location=device,
        weights_only=True,
    )
)

model.eval()

with torch.no_grad():

    x_hat, z, loss = model(
        x,
        edge_index,
    )

    score = (
        (x - x_hat)
        .pow(2)
        .mean(dim=1)
    )

results.append(
    evaluate(
        "GAT",
        score,
    )
)

# ==========================================================
# GraphSAGE
# ==========================================================

model = GraphSAGEAutoEncoder(
    in_channels=x.shape[1]
).to(device)

model.load_state_dict(
    torch.load(
        MODEL_DIR / "sage.pt",
        map_location=device,
        weights_only=True,
    )
)

model.eval()

with torch.no_grad():

    x_hat, z, loss = model(
        x,
        edge_index,
    )

    score = (
        (x - x_hat)
        .pow(2)
        .mean(dim=1)
    )

results.append(
    evaluate(
        "GraphSAGE",
        score,
    )
)

# ==========================================================
# HetGAT
# ==========================================================

model = HetGATAutoEncoder(
    metadata=data.metadata(),
).to(device)

model.load_state_dict(
    torch.load(
        ROOT /
        "data" /
        "processed" /
        "hetgat_model.pt",
        map_location=device,
        weights_only=True,
    )
)

model.eval()

with torch.no_grad():

    x_hat, z, loss = model(data)

    score = model.reconstruction_error_raw(
        data,
        x_hat,
    )

results.append(
    evaluate(
        "HetGAT",
        score,
    )
)

# ==========================================================
# SAVE
# ==========================================================

result = pd.DataFrame(results)

print("\n")
print(result)

OUT = ROOT / "results"

OUT.mkdir(
    exist_ok=True,
)

result.to_csv(
    OUT / "detection_comparison.csv",
    index=False,
)

print("\nSaved :")
print(
    OUT / "detection_comparison.csv"
)