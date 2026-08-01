import torch
import torch.nn.functional as F

from torch_geometric.nn import GCNConv


class GraphAutoEncoder(torch.nn.Module):

    def __init__(
        self,
        in_channels,
        hidden_channels=128,
        latent_channels=64,
    ):

        super().__init__()

        # Encoder
        self.conv1 = GCNConv(
            in_channels,
            hidden_channels,
        )

        self.conv2 = GCNConv(
            hidden_channels,
            latent_channels,
        )

        # Decoder
        self.decoder = torch.nn.Sequential(

            torch.nn.Linear(
                latent_channels,
                hidden_channels,
            ),

            torch.nn.ReLU(),

            torch.nn.Linear(
                hidden_channels,
                in_channels,
            ),

        )

    # ======================================

    def encode(self, x, edge_index):

        x = self.conv1(x, edge_index)
        x = F.relu(x)

        z = self.conv2(
            x,
            edge_index,
        )

        return z

    # ======================================

    def decode(self, z):

        return self.decoder(z)

    # ======================================

    def forward(self, data):

        x = data["ais"].x

        edge_index = data[
            "ais",
            "next",
            "ais"
        ].edge_index

        z = self.encode(
            x,
            edge_index,
        )

        x_hat = self.decode(z)

        loss = F.mse_loss(
            x_hat,
            x,
        )

        return x_hat, z, loss

    # ======================================

    def get_embedding(self, data):

        x = data["ais"].x

        edge_index = data[
            "ais",
            "next",
            "ais"
        ].edge_index

        return self.encode(
            x,
            edge_index,
        )