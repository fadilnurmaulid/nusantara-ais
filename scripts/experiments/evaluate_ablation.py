from pathlib import Path
import sys

import pandas as pd
import torch
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.nusantara_ais.models.hetgat import HetGATAutoEncoder

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

GRAPH_DIR = ROOT / "data" / "processed" / "ablation"
MODEL_DIR = ROOT / "data" / "model" / "ablation"
RESULT_DIR = ROOT / "results"

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

variants = [
    "full",
    "no_ari",
    "no_trip",
    "no_port",
    "no_protected",
]

rows = []

print("Device :", DEVICE)

for name in variants:

    print("\n" + "=" * 50)
    print(name)
    print("=" * 50)

    graph = torch.load(
        GRAPH_DIR / f"{name}.pt",
        weights_only=False,
    ).to(DEVICE)

    model = HetGATAutoEncoder(
        metadata=graph.metadata(),
        ais_in_channels=graph["ais"].x.size(1),
        trip_in_channels=graph["trip"].x.size(1),
        hidden_channels=128,
        latent_channels=64,
        heads=4,
        dropout=0.2,
    ).to(DEVICE)

    model.load_state_dict(
        torch.load(
            MODEL_DIR / f"{name}.pt",
            map_location=DEVICE,
            weights_only=False,
        )
    )

    model.eval()

    with torch.no_grad():

        x_hat, z, loss = model(graph)

        score = model.reconstruction_error_raw(
            graph,
            x_hat,
        ).cpu().numpy()

    rows.append(

        {

            "variant": name,
            "loss": float(loss.item()),
            "mean_score": float(score.mean()),
            "std_score": float(score.std()),
            "max_score": float(score.max()),

        }

    )

df = pd.DataFrame(rows)

print("\n")
print(df)

csv_path = RESULT_DIR / "ablation.csv"

df.to_csv(
    csv_path,
    index=False,
)

# ==========================================
# Plot
# ==========================================

plt.figure(figsize=(8,5))

plt.bar(
    df["variant"],
    df["loss"],
)

for i, v in enumerate(df["loss"]):

    plt.text(
        i,
        v + 0.002,
        f"{v:.3f}",
        ha="center",
        fontsize=9,
    )

plt.ylabel("Reconstruction Loss")
plt.xlabel("Ablation Variant")
plt.title("HetGAT Ablation Study")

plt.tight_layout()

plot_path = RESULT_DIR / "ablation.png"

plt.savefig(
    plot_path,
    dpi=300,
)

plt.close()

print("\nSaved :")
print(csv_path)
print(plot_path)