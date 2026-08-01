"""
NusantaraAISModel: end-to-end composition of

    Physics-informed ARI (precomputed feature, not learned here)
       -> HetGATEncoder (heterogeneous message passing)
       -> GraphAutoEncoder (feature + structure reconstruction, ARI head)
       -> Context-aware Maritime Anomaly Score (post-hoc combination)

This module owns only the two learned sub-networks (HetGAT, GAE); the
anomaly score itself is a deterministic, non-parametric function of their
outputs (see `anomaly_score.py`) and is computed by the training/inference
loops, not inside `forward`, so it can be recomputed at evaluation time
with different score weights without retraining (needed for the ARI
sensitivity study).
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch_geometric.data import HeteroData

from ..config import ModelConfig
from .hetgat import HetGATEncoder
from .gae import GraphAutoEncoder


class NusantaraAISModel(nn.Module):
    def __init__(self, node_feature_dims: Dict[str, int], edge_types, model_cfg: ModelConfig):
        super().__init__()
        self.encoder = HetGATEncoder(node_feature_dims, edge_types, model_cfg)
        self.gae = GraphAutoEncoder(model_cfg.hidden_dim, node_feature_dims["vessel_state"], model_cfg)

    def forward(self, data: HeteroData) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        x_dict = self.encoder(data)
        vessel_embedding = x_dict["vessel_state"]

        pos_edge_index = data["vessel_state", "temporal_next", "vessel_state"].edge_index
        sp_edge_index = data["vessel_state", "spatial_proximity", "vessel_state"].edge_index
        combined_pos_edges = torch.cat([pos_edge_index, sp_edge_index], dim=1) if sp_edge_index.numel() else pos_edge_index

        loss, aux = self.gae(
            vessel_embedding, data["vessel_state"].x, combined_pos_edges, data["vessel_state"].ari
        )
        aux["vessel_embedding"] = vessel_embedding.detach()
        aux["combined_edge_index"] = combined_pos_edges.detach()
        return loss, aux
