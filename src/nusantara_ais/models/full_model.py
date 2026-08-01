import torch

from .hetgat import HetGAT
from .gae import GraphAutoEncoder


class NusantaraAIS(torch.nn.Module):

    def __init__(
        self,
        metadata
    ):
        super().__init__()

        encoder = HetGAT(metadata)

        self.gae = GraphAutoEncoder(
            encoder
        )

    def forward(
        self,
        data
    ):

        return self.gae.encode(
            data.x_dict,
            data.edge_index_dict
        )