from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]

sys.path.append(str(ROOT))

from src.nusantara_ais.graph.dataset import load_dataset
from src.nusantara_ais.models.gae import GraphAutoEncoder

print("Loading graph...")

data = load_dataset()

print(data)

x = data["ais"].x

edge_index = data[
    "ais",
    "next",
    "ais",
].edge_index

print()

print("Node feature :", x.shape)
print("Edge :", edge_index.shape)

model = GraphAutoEncoder(

    in_channels=x.shape[1],

    hidden_channels=64,

    latent_channels=64,

)

z = model(

    x,

    edge_index,

)

print()

print("Embedding")

print(z.shape)

loss = model.recon_loss(

    z,

    edge_index,

)

print()

print("Reconstruction loss")

print(loss.item())