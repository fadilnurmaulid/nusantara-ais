r"""
Graph AutoEncoder (GAE) over HetGAT vessel_state embeddings.

Only `vessel_state` nodes are decoded (ports/lanes/grid-cells are context
that HAS ALREADY been fused into vessel_state embeddings by HetGAT's
message passing; decoding them separately would not serve the anomaly-score
objective).

Decoders
--------
1. Feature decoder D_feat: MLP  z_i -> \hat{x}_i \in R^{F_v}
   Reconstruction loss (per node): L_feat_i = || x_i - \hat{x}_i ||_2^2 / F_v
   A vessel behaving in a way inconsistent with the spatial/temporal context
   HetGAT has fused into z_i will be reconstructed poorly -- this is the
   PRIMARY anomaly signal (an autoencoder trained predominantly on normal
   behavior will fail to reconstruct atypical feature combinations).

2. Structure decoder D_struct: inner-product link predictor
       \hat{A}_{ij} = \sigma( z_i^T z_j )
   trained with binary cross-entropy against the TRUE temporal_next +
   spatial_proximity adjacency of the vessel_state-vessel_state subgraph,
   using `negative_sampling_ratio` negative (non-edge) pairs per positive
   edge (standard GAE/VGAE structure loss, Kipf & Welling 2016). A vessel
   whose embedding cannot predict its OWN real spatio-temporal edges (e.g.
   because HetGAT has fused in context wildly inconsistent with its
   trajectory) contributes a high structure-reconstruction error --
   this specifically targets encounter/rendezvous-implausible and
   context-implausible (e.g., "on a shipping lane" edge type mismatch)
   anomalies that a pure feature-reconstruction loss would miss.

3. ARI-consistency head D_ari: MLP  z_i -> \hat{ari}_i \in [0, 1] (sigmoid)
   Auxiliary supervised regression loss: L_ari_i = (ari_i - \hat{ari}_i)^2.
   This does NOT let the ARI leak into the anomaly score by definition
   (ARI is combined with reconstruction error only at the very end, in
   `anomaly_score.py`) -- its purpose here is purely regularizing: it forces
   the embedding z_i to retain information that is PREDICTIVE of physical
   reliability, which empirically stabilizes training and prevents the
   autoencoder from learning a reconstruction shortcut that ignores
   reliability-relevant dynamics (see ablation in evaluation/ablation.py).

Total loss
-----------
    L = recon_loss_weight * mean_i(L_feat_i)
      + structure_loss_weight * BCE(\hat{A}, A)
      + ari_loss_weight * mean_i(L_ari_i)
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import ModelConfig


class FeatureDecoder(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class ARIHead(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, 1), nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).squeeze(-1)


class GraphAutoEncoder(nn.Module):
    def __init__(self, hidden_dim: int, feature_out_dim: int, model_cfg: ModelConfig):
        super().__init__()
        self.to_latent = nn.Linear(hidden_dim, model_cfg.gae_latent_dim)
        self.feature_decoder = FeatureDecoder(model_cfg.gae_latent_dim, model_cfg.gae_decoder_hidden, feature_out_dim)
        self.ari_head = ARIHead(model_cfg.gae_latent_dim, model_cfg.gae_decoder_hidden)
        self.cfg = model_cfg

    def encode(self, hetgat_vessel_embedding: torch.Tensor) -> torch.Tensor:
        return self.to_latent(hetgat_vessel_embedding)

    def decode_structure(self, z: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index
        return torch.sigmoid((z[src] * z[dst]).sum(dim=-1))

    def negative_sample(self, edge_index: torch.Tensor, n_nodes: int, ratio: float) -> torch.Tensor:
        n_neg = max(1, int(edge_index.size(1) * ratio))
        device = edge_index.device
        neg_src = torch.randint(0, n_nodes, (n_neg,), device=device)
        neg_dst = torch.randint(0, n_nodes, (n_neg,), device=device)
        return torch.stack([neg_src, neg_dst])

    def forward(self, hetgat_vessel_embedding: torch.Tensor, x_original: torch.Tensor,
                pos_edge_index: torch.Tensor, ari_target: torch.Tensor
                ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        z = self.encode(hetgat_vessel_embedding)
        x_hat = self.feature_decoder(z)
        ari_hat = self.ari_head(z)

        feat_loss = F.mse_loss(x_hat, x_original)

        n_nodes = z.size(0)
        if pos_edge_index.numel() > 0 and n_nodes > 1:
            neg_edge_index = self.negative_sample(pos_edge_index, n_nodes, self.cfg.negative_sampling_ratio)
            pos_pred = self.decode_structure(z, pos_edge_index)
            neg_pred = self.decode_structure(z, neg_edge_index)
            pos_labels = torch.ones_like(pos_pred)
            neg_labels = torch.zeros_like(neg_pred)
            struct_loss = F.binary_cross_entropy(
                torch.cat([pos_pred, neg_pred]), torch.cat([pos_labels, neg_labels])
            )
        else:
            struct_loss = torch.tensor(0.0, device=z.device)

        ari_loss = F.mse_loss(ari_hat, ari_target)

        total = (
            self.cfg.recon_loss_weight * feat_loss
            + self.cfg.structure_loss_weight * struct_loss
            + self.cfg.ari_loss_weight * ari_loss
        )
        per_node_feat_error = ((x_hat - x_original) ** 2).mean(dim=-1)

        return total, {
            "feat_loss": feat_loss.detach(), "struct_loss": struct_loss.detach(),
            "ari_loss": ari_loss.detach(), "z": z.detach(),
            "x_hat": x_hat.detach(), "ari_hat": ari_hat.detach(),
            "per_node_feat_error": per_node_feat_error.detach(),
        }
