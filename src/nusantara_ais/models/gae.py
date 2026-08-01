import torch
import torch.nn.functional as F


class DotDecoder(torch.nn.Module):

    def forward(
        self,
        z,
        edge_index
    ):

        src = z[edge_index[0]]
        dst = z[edge_index[1]]

        score = (src * dst).sum(dim=1)

        return torch.sigmoid(score)


class GraphAutoEncoder(torch.nn.Module):

    def __init__(
        self,
        encoder
    ):
        super().__init__()

        self.encoder = encoder
        self.decoder = DotDecoder()

    def encode(
        self,
        x_dict,
        edge_index_dict
    ):

        return self.encoder(
            x_dict,
            edge_index_dict
        )

    def reconstruction_loss(
        self,
        z,
        edge_index
    ):

        pred = self.decoder(
            z,
            edge_index
        )

        label = torch.ones_like(pred)

        return F.binary_cross_entropy(
            pred,
            label
        )