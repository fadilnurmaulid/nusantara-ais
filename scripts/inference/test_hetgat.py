from pathlib import Path
import sys

import torch
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.nusantara_ais.graph.dataset import load_dataset
from src.nusantara_ais.models.hetgat import HetGATAutoEncoder
from src.nusantara_ais.models.anomaly_score import anomaly_score

# =====================================================
# DEVICE
# =====================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device :", device)

# =====================================================
# LOAD GRAPH
# =====================================================

print("Loading graph...")

data = load_dataset().to(device)

print(data)

# =====================================================
# LOAD MODEL
# =====================================================

print("\nLoading trained HetGAT model...")

model = HetGATAutoEncoder(
    metadata=data.metadata(),
    hidden_channels=128,
    latent_channels=64,
    heads=4,
    dropout=0.2,
).to(device)

state_dict = torch.load(
    ROOT / "data/processed/hetgat_model.pt",
    map_location=device,
)
model.load_state_dict(state_dict)

model.eval()

# =====================================================
# INFERENCE
# =====================================================

print("\nRunning inference...")

with torch.no_grad():
    x_hat, z, loss = model(data)

    # Reconstruction error in the ORIGINAL (unnormalized) feature scale
    # -- more interpretable than error in the normalized training space.
    rec_error = model.reconstruction_error_raw(data, x_hat)

    score = anomaly_score(
        (data["ais"].x - model.ais_mean) / model.ais_std,
        x_hat,
        z,
    )

print("Reconstruction loss (normalized space) :", loss.item())

# =====================================================
# SAVE OUTPUTS
# =====================================================

OUT = ROOT / "data/processed"

torch.save(z.cpu(), OUT / "hetgat_embedding.pt")
torch.save(score.cpu(), OUT / "hetgat_anomaly_score.pt")

print("\nSaved:")
print(" -", OUT / "hetgat_embedding.pt", "shape:", tuple(z.shape))
print(" -", OUT / "hetgat_anomaly_score.pt", "shape:", tuple(score.shape))

# =====================================================
# TOP 20 ANOMALY
# =====================================================

print("\nTop 20 anomalies (by anomaly_score):\n")

score_np = score.cpu().numpy()
rec_error_np = rec_error.cpu().numpy()

try:
    ais_nodes = pd.read_parquet(OUT / "ais_nodes.parquet")
    id_cols = [c for c in ["node_id", "mmsi", "timestamp"] if c in ais_nodes.columns]
    ranking = ais_nodes[id_cols].copy()
except FileNotFoundError:
    ranking = pd.DataFrame({"node_id": range(len(score_np))})

ranking["anomaly_score"] = score_np
ranking["reconstruction_error"] = rec_error_np

top20 = ranking.sort_values("anomaly_score", ascending=False).head(20)

print(top20.to_string(index=False))

print("\nDone.")
