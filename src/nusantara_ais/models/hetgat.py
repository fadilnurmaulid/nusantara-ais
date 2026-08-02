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
        # INPUT NORMALIZATION (per node type, z-score)
        # ======================================================
        # `hetero_graph.pt` stores raw engineered features for the "ais"
        # and "trip" node types (e.g. trip duration in seconds reaches
        # ~1e8, distance_from_port in meters reaches ~1e6). Feeding these
        # directly into Linear + GATConv makes activations and the MSE
        # reconstruction loss explode / diverge.
        #
        # These buffers hold mean/std computed once (from the graph the
        # model first sees, i.e. the training graph) and are saved and
        # restored with the model's state_dict, so test/inference uses
        # the exact same statistics as training. This does NOT change
        # the dataset or graph files on disk, and does not change which
        # features are used -- only how they are scaled internally
        # before the first Linear projection.

        self.register_buffer("ais_mean", torch.zeros(27))
        self.register_buffer("ais_std", torch.ones(27))

        self.register_buffer("trip_mean", torch.zeros(3))
        self.register_buffer("trip_std", torch.ones(3))

        self.register_buffer("_norm_fitted", torch.tensor(False))

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

    @torch.no_grad()
    def fit_normalizer(self, data, eps=1e-6):
        """
        Compute per-feature mean/std for the "ais" and "trip" node types
        from `data` and store them in buffers. Must be called once on
        the training graph before the first forward pass. "port" and
        "protected" features are constant (all ones) so they don't need
        normalization.
        """

        ais_x = data["ais"].x
        self.ais_mean.copy_(ais_x.mean(dim=0))
        self.ais_std.copy_(ais_x.std(dim=0) + eps)

        trip_x = data["trip"].x
        self.trip_mean.copy_(trip_x.mean(dim=0))
        self.trip_std.copy_(trip_x.std(dim=0) + eps)

        self._norm_fitted.fill_(True)

    # ======================================================

    def project(self, data):

        if not bool(self._norm_fitted):
            raise RuntimeError(
                "HetGATAutoEncoder: normalizer statistics not fitted. "
                "Call model.fit_normalizer(data) once on the training "
                "graph before running the model."
            )

        ais_x = (data["ais"].x - self.ais_mean) / self.ais_std
        trip_x = (data["trip"].x - self.trip_mean) / self.trip_std

        return {

            "ais": self.ais_proj(ais_x),

            "port": self.port_proj(
                data["port"].x
            ),

            "trip": self.trip_proj(trip_x),

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

        # The decoder reconstructs the *normalized* AIS features (same
        # space the encoder consumes). Comparing against the raw,
        # unnormalized data["ais"].x here is what causes the MSE loss to
        # explode, since raw feature magnitudes range up to ~1e6.
        ais_x_norm = (data["ais"].x - self.ais_mean) / self.ais_std

        loss = F.mse_loss(
            x_hat,
            ais_x_norm,
        )

        return x_hat, z, loss

    # ======================================================

    def get_embedding(self, data):

        return self.encode(data)

    # ======================================================

    def reconstruction_error_raw(self, data, x_hat):
        """
        Per-node reconstruction error (mean squared error across
        features), computed in the ORIGINAL feature scale rather than
        the normalized training scale. This is more interpretable as an
        anomaly score than the normalized-space error, since it isn't
        dominated by whichever feature happens to have the largest
        z-scored variance.
        """

        x_hat_raw = x_hat * self.ais_std + self.ais_mean

        return ((data["ais"].x - x_hat_raw) ** 2).mean(dim=1)