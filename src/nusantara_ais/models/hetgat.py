import torch
import torch.nn.functional as F

from torch_geometric.nn import HeteroConv
from torch_geometric.nn import GATConv


class HetGAT(torch.nn.Module):

    def __init__(
        self,
        metadata,
        hidden_dim=64,
        out_dim=64,
        heads=4
    ):
        super().__init__()

        self.conv1 = HeteroConv({

            edge: GATConv(
                (-1, -1),
                hidden_dim,
                heads=heads,
                add_self_loops=False
            )

            for edge in metadata[1]

        })

        self.conv2 = HeteroConv({

            edge: GATConv(
                (-1, -1),
                out_dim,
                heads=1,
                concat=False,
                add_self_loops=False
            )

            for edge in metadata[1]

        })

    def forward(self, x_dict, edge_index_dict):

        x = self.conv1(
            x_dict,
            edge_index_dict
        )

        x = {
            k: F.elu(v)
            for k, v in x.items()
        }

        x = self.conv2(
            x,
            edge_index_dict
        )

        return x