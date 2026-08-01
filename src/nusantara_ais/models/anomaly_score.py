import torch


def reconstruction_error(
    x,
    x_hat,
):

    return ((x - x_hat) ** 2).mean(dim=1)


def embedding_norm(z):

    return torch.norm(
        z,
        dim=1,
    )


def anomaly_score(
    x,
    x_hat,
    z,
):

    rec = reconstruction_error(
        x,
        x_hat,
    )

    emb = embedding_norm(z)

    score = rec + 0.1 * emb

    return score