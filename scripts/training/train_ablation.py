from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.nusantara_ais.models.hetgat import HetGATAutoEncoder

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

GRAPH_DIR = ROOT / "data" / "processed" / "ablation"
MODEL_DIR = ROOT / "data" / "model" / "ablation"

MODEL_DIR.mkdir(
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

epochs = 200

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

    model.fit_normalizer(graph)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-5,
    )

    for epoch in range(1, epochs + 1):

        model.train()

        optimizer.zero_grad()

        _, _, loss = model(graph)

        loss.backward()

        optimizer.step()

        if epoch % 10 == 0:

            print(
                f"{name} | Epoch {epoch:03d} | Loss {loss.item():.6f}"
            )

    torch.save(
        model.state_dict(),
        MODEL_DIR / f"{name}.pt",
    )

    print(
        f"\nSaved : {MODEL_DIR / f'{name}.pt'}"
    )

print("\nDone.")