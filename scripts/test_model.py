from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.nusantara_ais.graph.dataset import load_dataset
from src.nusantara_ais.models.gae import GraphAutoEncoder
from src.nusantara_ais.models.anomaly_score import anomaly_score

# ==========================================
# LOAD GRAPH
# ==========================================

print("Loading graph...")

data = load_dataset()

print(data)

# ==========================================
# MODEL
# ==========================================

model = GraphAutoEncoder(
    in_channels=data["ais"].x.shape[1],
)

model.load_state_dict(
    torch.load(
        ROOT / "data/model/gae.pt",
        weights_only=True,
    )
)

model.eval()

# ==========================================
# FORWARD
# ==========================================

with torch.no_grad():
    x_hat, z, loss = model(data)

print("\nNode feature")
print(data["ais"].x.shape)

print("\nEmbedding")
print(z.shape)

print("\nReconstruction")
print(x_hat.shape)

print("\nLoss")
print(loss.item())

# ==========================================
# SAVE EMBEDDING
# ==========================================

torch.save(
    z,
    ROOT / "data/processed/gae_embedding.pt",
)

print("\nEmbedding saved")

# ==========================================
# ANOMALY SCORE
# ==========================================

score = anomaly_score(
    data["ais"].x,
    x_hat,
    z,
)

print("\nAnomaly score")

print(score.shape)

print("Min :", score.min().item())
print("Max :", score.max().item())
print("Mean:", score.mean().item())

torch.save(
    score,
    ROOT / "data/processed/anomaly_score.pt",
)

print("\nAnomaly score saved")

# ==========================================
# TOP ANOMALY
# ==========================================

top = torch.topk(
    score,
    20,
)

print("\nTop 20 anomaly")

for rank, (idx, s) in enumerate(
    zip(top.indices.tolist(), top.values.tolist()),
    1,
):
    print(
        f"{rank:2d}. AIS #{idx:5d} -> {s:.4f}"
    )