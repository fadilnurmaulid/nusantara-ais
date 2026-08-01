import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GATConv
from torch_geometric.nn import HeteroConv


class HetGATAutoEncoder(nn.Module):

    def __init__(
        self,
        metadata,
        hidden_channels=128,
        latent_channels=64,
        heads=4,
        dropout=0.2,
    ):

        super().__init__()

        self.dropout = dropout

        # ======================================================
        # INPUT PROJECTION
        # ======================================================

        self.ais_proj = nn.Linear(27, hidden_channels)

        self.port_proj = nn.Linear(1, hidden_channels)

        self.trip_proj = nn.Linear(3, hidden_channels)

        self.protected_proj = nn.Linear(
            1,
            hidden_channels,
        )

        # ======================================================
        # HETERO GAT LAYER 1
        # ======================================================

        self.conv1 = HeteroConv(

            {

                (
                    "ais",
                    "next",
                    "ais",
                ): GATConv(
                    (-1, -1),
                    hidden_channels,
                    heads=heads,
                    concat=False,
                    add_self_loops=False,
                ),

                (
                    "ais",
                    "near",
                    "port",
                ): GATConv(
                    (-1, -1),
                    hidden_channels,
                    heads=heads,
                    concat=False,
                    add_self_loops=False,
                ),

                (
                    "port",
                    "rev_near",
                    "ais",
                ): GATConv(
                    (-1, -1),
                    hidden_channels,
                    heads=heads,
                    concat=False,
                    add_self_loops=False,
                ),

                (
                    "ais",
                    "trip",
                    "trip",
                ): GATConv(
                    (-1, -1),
                    hidden_channels,
                    heads=heads,
                    concat=False,
                    add_self_loops=False,
                ),

                (
                    "trip",
                    "rev_trip",
                    "ais",
                ): GATConv(
                    (-1, -1),
                    hidden_channels,
                    heads=heads,
                    concat=False,
                    add_self_loops=False,
                ),

                (
                    "ais",
                    "protected",
                    "protected",
                ): GATConv(
                    (-1, -1),
                    hidden_channels,
                    heads=heads,
                    concat=False,
                    add_self_loops=False,
                ),

                (
                    "protected",
                    "rev_protected",
                    "ais",
                ): GATConv(
                    (-1, -1),
                    hidden_channels,
                    heads=heads,
                    concat=False,
                    add_self_loops=False,
                ),

            },

            aggr="sum",

        )

        # ======================================================
        # HETERO GAT LAYER 2
        # ======================================================

        self.conv2 = HeteroConv(

            {

                edge: GATConv(
                    (-1, -1),
                    latent_channels,
                    heads=1,
                    concat=False,
                    add_self_loops=False,
                )

                for edge in metadata[1]

            },

            aggr="sum",

        )

        # ======================================================
        # DECODER
        # ======================================================

        self.decoder = nn.Sequential(

            nn.Linear(
                latent_channels,
                hidden_channels,
            ),

            nn.ReLU(),

            nn.Dropout(dropout),

            nn.Linear(
                hidden_channels,
                27,
            ),

        )

    # ======================================================

    def project(self, data):

        return {

            "ais": self.ais_proj(
                data["ais"].x
            ),

            "port": self.port_proj(
                data["port"].x
            ),

            "trip": self.trip_proj(
                data["trip"].x
            ),

            "protected": self.protected_proj(
                data["protected"].x
            ),

        }

    # ======================================================

    def encode(self, data):

        x_dict = self.project(data)

        x_dict = self.conv1(
            x_dict,
            data.edge_index_dict,
        )

        for key in x_dict:

            x_dict[key] = F.relu(
                x_dict[key]
            )

            x_dict[key] = F.dropout(
                x_dict[key],
                p=self.dropout,
                training=self.training,
            )

        x_dict = self.conv2(
            x_dict,
            data.edge_index_dict,
        )

        return x_dict["ais"]

    # ======================================================

    def decode(self, z):

        return self.decoder(z)

    # ======================================================

    def forward(self, data):

        z = self.encode(data)

        x_hat = self.decode(z)

        loss = F.mse_loss(
            x_hat,
            data["ais"].x,
        )

        return x_hat, z, loss

    # ======================================================

    def get_embedding(self, data):

        return self.encode(data)