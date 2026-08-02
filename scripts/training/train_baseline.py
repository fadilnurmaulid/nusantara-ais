from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.nusantara_ais.graph.dataset import load_dataset
from src.nusantara_ais.models.baseline import (
    GCNAutoEncoder,
    GATAutoEncoder,
    GraphSAGEAutoEncoder,
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device :", DEVICE)

print("Loading graph...")

data = load_dataset()

x = data["ais"].x.to(DEVICE)

edge_index = data[
    "ais",
    "next",
    "ais"
].edge_index.to(DEVICE)

MODELS = {

    "gcn": GCNAutoEncoder,

    "gat": GATAutoEncoder,

    "sage": GraphSAGEAutoEncoder,

}

MODEL_DIR = ROOT / "data" / "model"

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

for name, Model in MODELS.items():

    print()
    print("=" * 50)
    print(name.upper())
    print("=" * 50)

    model = Model(
        in_channels=x.shape[1],
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-5,
    )

    loss_history = []

    for epoch in range(1, 201):

        model.train()

        optimizer.zero_grad()

        _, _, loss = model(
            x,
            edge_index,
        )

        loss.backward()

        optimizer.step()

        loss_history.append(
            loss.item()
        )

        if epoch % 10 == 0:

            print(
                f"Epoch {epoch:03d} | Loss {loss.item():.6f}"
            )

    torch.save(
        model.state_dict(),
        MODEL_DIR / f"{name}.pt",
    )

    torch.save(
        torch.tensor(loss_history),
        MODEL_DIR / f"{name}_loss.pt",
    )

    print()

    print("Saved :", MODEL_DIR / f"{name}.pt")