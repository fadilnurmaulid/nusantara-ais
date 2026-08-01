import torch


def anomaly_score(
    embedding,
    centroid
):

    return torch.norm(
        embedding - centroid,
        dim=1
    )