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

# =====================================================
# DEVICE
# =====================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device :", device)

# =====================================================
# GRAPH
# =====================================================

print("Loading graph...")

data = load_dataset().to(device)

x = data["ais"].x

edge_index = data[
    "ais",
    "next",
    "ais"
].edge_index

# =====================================================
# RESULT
# =====================================================

results = []

MODEL_DIR = ROOT / "data" / "model"

# =====================================================
# BASELINE
# =====================================================

baseline_models = [

    (
        "GCN",
        GCNAutoEncoder(
            in_channels=x.shape[1]
        ),
        MODEL_DIR / "gcn.pt",
    ),

    (
        "GAT",
        GATAutoEncoder(
            in_channels=x.shape[1]
        ),
        MODEL_DIR / "gat.pt",
    ),

    (
        "GraphSAGE",
        GraphSAGEAutoEncoder(
            in_channels=x.shape[1]
        ),
        MODEL_DIR / "sage.pt",
    ),

]

for name, model, weight in baseline_models:

    print(f"\n{name}")

    model.load_state_dict(

        torch.load(
            weight,
            map_location=device,
            weights_only=True,
        )

    )

    model.to(device)
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

    results.append({

        "model": name,

        "loss": float(loss),

        "mean_score": float(score.mean()),
        "std_score": float(score.std()),
        "min_score": float(score.min()),
        "max_score": float(score.max()),

    })

# =====================================================
# HETGAT
# =====================================================

print("\nHetGAT")

hetgat = HetGATAutoEncoder(
    metadata=data.metadata(),
).to(device)

hetgat.load_state_dict(

    torch.load(

        ROOT /
        "data" /
        "processed" /
        "hetgat_model.pt",

        map_location=device,
        weights_only=True,

    )

)

hetgat.eval()

with torch.no_grad():

    x_hat, z, loss = hetgat(data)

    score = hetgat.reconstruction_error_raw(
        data,
        x_hat,
    )

results.append({

    "model": "HetGAT",

    "loss": float(loss),

    "mean_score": float(score.mean()),
    "std_score": float(score.std()),
    "min_score": float(score.min()),
    "max_score": float(score.max()),

})

# =====================================================
# SAVE
# =====================================================

result = pd.DataFrame(results)

print("\n==============================")
print(result)

OUT = ROOT / "results"

OUT.mkdir(
    exist_ok=True,
)

result.to_csv(

    OUT /
    "baseline_comparison.csv",

    index=False,

)

print("\nSaved :")
print(
    OUT /
    "baseline_comparison.csv"
)