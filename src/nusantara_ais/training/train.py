from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]

from src.nusantara_ais.graph.dataset import load_dataset
from src.nusantara_ais.models.gae import GraphAutoEncoder


def train(
    epochs=200,
    lr=1e-3,
    device="cpu",
):

    data = load_dataset().to(device)

    model = GraphAutoEncoder(
        in_channels=data["ais"].x.shape[1]
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
    )

    history = []

    print("=" * 50)

    for epoch in range(1, epochs + 1):

        model.train()

        optimizer.zero_grad()

        x_hat, z, loss = model(data)

        loss.backward()

        optimizer.step()

        history.append(loss.item())

        if epoch % 10 == 0:

            print(
                f"Epoch {epoch:03d} | Loss {loss.item():.6f}"
            )

    save_dir = ROOT / "data" / "model"
    save_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        model.state_dict(),
        save_dir / "gae.pt",
    )

    torch.save(
        history,
        save_dir / "gae_loss.pt",
    )

    print("\nTraining Finished")

    print("Final Loss :", history[-1])

    return model