import torch
import torch.nn.functional as F

from torch_geometric.nn import (
    GCNConv,
    GATConv,
    SAGEConv,
)


# ==========================================================
# BASE
# ==========================================================

class BaseAutoEncoder(torch.nn.Module):

    def __init__(
        self,
        encoder,
        in_channels,
        hidden_channels=128,
        latent_channels=64,
    ):

        super().__init__()

        self.encoder = encoder(
            in_channels,
            hidden_channels,
            latent_channels,
        )

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

    def forward(self, x, edge_index):

        z = self.encoder(
            x,
            edge_index,
        )

        x_hat = self.decoder(z)

        loss = F.mse_loss(
            x_hat,
            x,
        )

        return x_hat, z, loss


# ==========================================================
# GCN
# ==========================================================

class GCNEncoder(torch.nn.Module):

    def __init__(
        self,
        in_channels,
        hidden_channels,
        latent_channels,
    ):

        super().__init__()

        self.conv1 = GCNConv(
            in_channels,
            hidden_channels,
        )

        self.conv2 = GCNConv(
            hidden_channels,
            latent_channels,
        )

    def forward(
        self,
        x,
        edge_index,
    ):

        x = self.conv1(
            x,
            edge_index,
        )

        x = F.relu(x)

        x = self.conv2(
            x,
            edge_index,
        )

        return x


class GCNAutoEncoder(BaseAutoEncoder):

    def __init__(
        self,
        in_channels,
        hidden_channels=128,
        latent_channels=64,
    ):

        super().__init__(
            GCNEncoder,
            in_channels,
            hidden_channels,
            latent_channels,
        )


# ==========================================================
# GAT
# ==========================================================

class GATEncoder(torch.nn.Module):

    def __init__(
        self,
        in_channels,
        hidden_channels,
        latent_channels,
    ):

        super().__init__()

        self.conv1 = GATConv(
            in_channels,
            hidden_channels,
            heads=4,
            concat=False,
        )

        self.conv2 = GATConv(
            hidden_channels,
            latent_channels,
            heads=1,
            concat=False,
        )

    def forward(
        self,
        x,
        edge_index,
    ):

        x = self.conv1(
            x,
            edge_index,
        )

        x = F.elu(x)

        x = self.conv2(
            x,
            edge_index,
        )

        return x


class GATAutoEncoder(BaseAutoEncoder):

    def __init__(
        self,
        in_channels,
        hidden_channels=128,
        latent_channels=64,
    ):

        super().__init__(
            GATEncoder,
            in_channels,
            hidden_channels,
            latent_channels,
        )


# ==========================================================
# GraphSAGE
# ==========================================================

class GraphSAGEEncoder(torch.nn.Module):

    def __init__(
        self,
        in_channels,
        hidden_channels,
        latent_channels,
    ):

        super().__init__()

        self.conv1 = SAGEConv(
            in_channels,
            hidden_channels,
        )

        self.conv2 = SAGEConv(
            hidden_channels,
            latent_channels,
        )

    def forward(
        self,
        x,
        edge_index,
    ):

        x = self.conv1(
            x,
            edge_index,
        )

        x = F.relu(x)

        x = self.conv2(
            x,
            edge_index,
        )

        return x


class GraphSAGEAutoEncoder(BaseAutoEncoder):

    def __init__(
        self,
        in_channels,
        hidden_channels=128,
        latent_channels=64,
    ):

        super().__init__(
            GraphSAGEEncoder,
            in_channels,
            hidden_channels,
            latent_channels,
        )