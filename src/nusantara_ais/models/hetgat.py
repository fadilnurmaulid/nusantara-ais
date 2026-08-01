r"""
Heterogeneous Graph Attention Network (HetGAT) encoder.

Architecture
------------
Each layer l is a `torch_geometric.nn.HeteroConv` wrapping one `GATConv` per
(src_type, relation, dst_type) triple present in the graph (including the
reverse relations added in `graph/builder._add_reverse_edges`, so every
context node type also receives updated messages). `HeteroConv` with
aggregation "sum" combines, per node type, the messages arriving from every
incoming relation -- this is the standard, most reproducible heterogeneous
message-passing scheme (Schlichtkrull et al. R-GCN generalized to attention;
Wang et al. HAN's per-relation-then-fuse pattern) and is what "HetGAT" refers
to throughout the paper.

Per-relation attention (single GATConv layer), for relation r connecting
source node i (type A) to target node j (type B):

    e_{ij}^{(r)} = LeakyReLU( a_r^T [ W_r h_i \, \| \, W_r h_j \, \| \, w_{ij} ] )
    \alpha_{ij}^{(r)} = softmax_j( e_{ij}^{(r)} )                (over neighbors j of i, per head)
    h_i'^{(r)} = \sigma( \sum_{j \in N_r(i)} \alpha_{ij}^{(r)} W_r h_j )

where w_{ij} is the (scalar) edge_attr computed in `graph/edges.py` --
PyG's GATConv natively supports `edge_dim` to fold a scalar/vector edge
feature into the attention logit, which is how the temporal decay and
inverse-distance edge weights actually influence attention rather than only
gating the adjacency structure.

Multi-head attention (H = `hetgat_heads`) is averaged (not concatenated) on
the FINAL layer to keep the output dimensionality equal to `hidden_dim`
regardless of head count (required so the GAE decoder has a fixed input
size), and concatenated on intermediate layers (standard GAT practice,
increases capacity without over-compressing).

Since node types have different raw feature dimensionalities (vessel_state
has ~40 dims, port has 3, lane has 5, grid_cell has 4), a per-node-type
input linear projection maps every type into a common `hidden_dim` BEFORE
the first HetGAT layer -- this is required by HeteroConv/GATConv (which
expects consistent hidden sizes across types being aggregated together) and
is mathematically just W_in^{(type)} h_i + b^{(type)}, one projection matrix
per node type.

Tensor dimensions (per snapshot, batch of B graphs handled by PyG's
automatic disjoint-union batching so shapes below are for a single graph):
    vessel_state.x        : [N_v, F_v]   (F_v = number of engineered features, ~40)
    port.x                 : [N_p, 3]
    shipping_lane_segment.x: [N_l, 5]
    grid_cell.x             : [N_g, 4]
    -> after input projection, every type: [N_type, hidden_dim]
    -> after L HetGAT layers, every type: [N_type, hidden_dim]  (final layer averages heads)
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import GATConv, HeteroConv

from ..config import ModelConfig, GraphConfig


def _activation(name: str):
    return {"elu": F.elu, "relu": F.relu, "gelu": F.gelu}.get(name, F.elu)


class HetGATEncoder(nn.Module):
    def __init__(self, node_feature_dims: Dict[str, int], edge_types: List[Tuple[str, str, str]],
                 model_cfg: ModelConfig):
        super().__init__()
        self.hidden_dim = model_cfg.hidden_dim
        self.n_layers = model_cfg.hetgat_layers
        self.heads = model_cfg.hetgat_heads
        self.act = _activation(model_cfg.activation)

        self.input_proj = nn.ModuleDict({
            ntype: nn.Linear(dim, self.hidden_dim) for ntype, dim in node_feature_dims.items()
        })

        self.layers = nn.ModuleList()
        for layer_idx in range(self.n_layers):
            is_last = layer_idx == self.n_layers - 1
            concat = not is_last
            out_channels_per_head = self.hidden_dim // self.heads if concat else self.hidden_dim
            conv_dict = {}
            for (src, rel, dst) in edge_types:
                in_dim = self.hidden_dim if layer_idx == 0 else self.hidden_dim
                conv_dict[(src, rel, dst)] = GATConv(
                    in_dim, out_channels_per_head, heads=self.heads, concat=concat,
                    edge_dim=1, dropout=model_cfg.hetgat_dropout, add_self_loops=False,
                )
            self.layers.append(HeteroConv(conv_dict, aggr="sum"))

        self.dropout = nn.Dropout(model_cfg.hetgat_dropout)

    def forward(self, data: HeteroData) -> Dict[str, torch.Tensor]:
        x_dict = {ntype: self.input_proj[ntype](data[ntype].x) for ntype in self.input_proj}
        edge_index_dict = {}
        edge_attr_dict = {}
        for edge_type in data.edge_types:
            store = data[edge_type]
            edge_index_dict[edge_type] = store.edge_index
            if "edge_attr" in store and store.edge_attr is not None and store.edge_attr.numel() > 0:
                edge_attr_dict[edge_type] = store.edge_attr
            else:
                n_edges = store.edge_index.size(1)
                edge_attr_dict[edge_type] = torch.ones((n_edges, 1), device=store.edge_index.device)

        for layer_idx, conv in enumerate(self.layers):
            x_dict = conv(x_dict, edge_index_dict, edge_attr_dict=edge_attr_dict)
            x_dict = {k: self.dropout(self.act(v)) for k, v in x_dict.items()}
        return x_dict
