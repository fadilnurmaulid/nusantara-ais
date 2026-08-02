from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.nusantara_ais.graph.dataset import load_dataset
from src.nusantara_ais.models.hetgat import HetGATAutoEncoder

# =====================================================
# DEVICE
# =====================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device :", device)

# =====================================================
# LOAD GRAPH
# =====================================================

print("Loading graph...")

data = load_dataset().to(device)

# =====================================================
# MODEL
# =====================================================

model = HetGATAutoEncoder(
    metadata=data.metadata(),
    hidden_channels=128,
    latent_channels=64,
    heads=4,
    dropout=0.2,
).to(device)

# Fit input normalization statistics (mean/std per feature) once, from
# the training graph. Raw AIS/trip features in hetero_graph.pt are not
# normalized (e.g. distances in meters, trip duration in seconds), so
# skipping this step causes the reconstruction loss to explode.
model.fit_normalizer(data)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3,
    weight_decay=1e-5,
)

# =====================================================
# TRAIN
# =====================================================

epochs = 200

print("\nStart Training\n")

for epoch in range(1, epochs + 1):

    model.train()

    optimizer.zero_grad()

    x_hat, z, loss = model(data)

    loss.backward()

    optimizer.step()

    if epoch % 10 == 0:

        print(
            f"Epoch {epoch:03d} | Loss {loss.item():.6f}"
        )

# =====================================================
# SAVE
# =====================================================

torch.save(
    model.state_dict(),
    ROOT / "data/processed/hetgat_model.pt",
)

print("\nTraining Finished")
print("Final Loss :", loss.item())
print("Model saved.")