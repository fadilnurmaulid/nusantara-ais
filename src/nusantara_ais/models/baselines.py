"""
Baseline models for comparative evaluation.

All baselines consume the SAME flattened per-vessel-state feature matrix
used as `vessel_state.x` (i.e. no graph structure) except GraphSAGE / GAT /
HetGAT-only, which consume the same heterogeneous snapshots as the full
model but without the GraphAutoEncoder's structure/ARI-consistency losses
(feature-reconstruction-only, isolating the contribution of the full GAE
objective in the ablation study).

    Isolation Forest    -- unsupervised, axis-aligned partitioning; the
                           classic tabular anomaly-detection baseline.
    One-Class SVM        -- unsupervised, kernelized boundary around the
                           dense region of "normal" feature space.
    XGBoost               -- supervised gradient-boosted trees trained on
                           behavior_label (used ONLY when ground-truth
                           labels are available, i.e. synthetic validation;
                           demonstrates the ceiling a fully-supervised
                           tabular model reaches on this feature set).
    GraphSAGE             -- homogeneous-graph inductive GNN baseline
                           (mean-aggregator) applied to the vessel_state
                           spatial_proximity + temporal_next subgraph only
                           (context node types collapsed out), showing the
                           benefit of heterogeneity.
    GAT                    -- homogeneous single-relation attention GNN,
                           isolating the benefit of heterogeneous relations
                           vs. attention alone.
    HetGAT (ours, no GAE) -- the paper's encoder with a simple feature-
                           reconstruction-only decoder (no structure/ARI
                           losses), isolating the GAE's contribution.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from torch_geometric.data import HeteroData
from torch_geometric.nn import GATConv, SAGEConv
from xgboost import XGBClassifier


def run_isolation_forest(X: np.ndarray, contamination: float = 0.15, seed: int = 42) -> np.ndarray:
    model = IsolationForest(contamination=contamination, random_state=seed, n_estimators=200)
    model.fit(X)
    raw = -model.score_samples(X)  # higher = more anomalous
    return (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)


def run_one_class_svm(X: np.ndarray, nu: float = 0.15) -> np.ndarray:
    model = OneClassSVM(nu=nu, kernel="rbf", gamma="scale")
    model.fit(X)
    raw = -model.decision_function(X)
    return (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)


def run_xgboost_supervised(X_train, y_train, X_test, seed: int = 42) -> Tuple[XGBClassifier, np.ndarray]:
    model = XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
        random_state=seed, n_jobs=4,
    )
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    return model, proba


class HomogeneousGraphSAGE(nn.Module):
    """Vessel_state-only GraphSAGE encoder + linear feature-reconstruction
    decoder (autoencoder-style anomaly baseline)."""

    def __init__(self, in_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, in_dim)

    def forward(self, x, edge_index):
        h = F.elu(self.conv1(x, edge_index))
        h = F.elu(self.conv2(h, edge_index))
        x_hat = self.decoder(h)
        per_node_error = ((x_hat - x) ** 2).mean(dim=-1)
        return per_node_error, h


class HomogeneousGAT(nn.Module):
    """Vessel_state-only single-relation GAT encoder + linear decoder."""

    def __init__(self, in_dim: int, hidden_dim: int = 128, heads: int = 4):
        super().__init__()
        self.conv1 = GATConv(in_dim, hidden_dim // heads, heads=heads, concat=True)
        self.conv2 = GATConv(hidden_dim, hidden_dim, heads=1, concat=False)
        self.decoder = nn.Linear(hidden_dim, in_dim)

    def forward(self, x, edge_index):
        h = F.elu(self.conv1(x, edge_index))
        h = F.elu(self.conv2(h, edge_index))
        x_hat = self.decoder(h)
        per_node_error = ((x_hat - x) ** 2).mean(dim=-1)
        return per_node_error, h


def homogeneous_edge_index(data: HeteroData) -> torch.Tensor:
    """Merges temporal_next + spatial_proximity vessel_state<->vessel_state
    edges into a single homogeneous edge_index for the GraphSAGE/GAT
    baselines (which do not model heterogeneity)."""
    t = data["vessel_state", "temporal_next", "vessel_state"].edge_index
    s = data["vessel_state", "spatial_proximity", "vessel_state"].edge_index
    return torch.cat([t, s], dim=1) if s.numel() else t
