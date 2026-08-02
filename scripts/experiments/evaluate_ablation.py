from pathlib import Path
import sys

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.nusantara_ais.models.hetgat import HetGATAutoEncoder

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

variants = [
    "full",
    "no_ari",
    "no_trip",
    "no_port",
    "no_protected",
]

rows = []

for name in variants:

    print(f"\n{name}")

    graph = torch.load(
        ROOT / "data" / "processed" / "ablation" / f"{name}.pt",
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

    for epoch in range(100):

        model.train()

        optimizer.zero_grad()

        x_hat, z, loss = model(graph)

        loss.backward()

        optimizer.step()

        if epoch % 10 == 0:
            print(
            f"{name} | Epoch {epoch:03d} | Loss {loss.item():6f}"
            )

    model.eval()

    with torch.no_grad():

        x_hat, z, loss = model(graph)

        score = model.reconstruction_error_raw(
            graph,
            x_hat,
        )

    rows.append({

        "variant": name,

        "loss": float(loss),

        "mean_score": float(score.mean()),

        "std_score": float(score.std()),

        "max_score": float(score.max()),

    })

df = pd.DataFrame(rows)

print(df)

save_path = ROOT / "results" / "ablation.csv"

df.to_csv(
    save_path,
    index=False,
)

print("\nSaved :")
print(save_path)