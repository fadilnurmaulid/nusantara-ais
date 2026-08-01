"""
Validation pipeline.

Runs the trained model over a snapshot dataset and returns, per node, the
raw signals needed by the anomaly-score module (feature-reconstruction
error, structure error, ARI) plus ground-truth behavior labels IF present
(only available for synthetic validation; the model never trains on them).
Also aggregates dataset-level loss components for reporting.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from ..config import Config
from ..graph.dataset import MaritimeSnapshotDataset
from ..models.anomaly_score import AnomalyScoreConfig, compute_anomaly_score, compute_structure_error, risk_tier
from ..models.full_model import NusantaraAISModel


@torch.no_grad()
def run_inference(model: NusantaraAISModel, dataset: MaritimeSnapshotDataset, device: torch.device,
                   score_cfg: AnomalyScoreConfig = None) -> Dict[str, np.ndarray]:
    model.eval()
    score_cfg = score_cfg or AnomalyScoreConfig()
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    all_mmsi, all_ts, all_labels = [], [], []
    all_feat_err, all_struct_err, all_ari = [], [], []
    total_loss_components = {"feat_loss": [], "struct_loss": [], "ari_loss": []}

    for batch in loader:
        batch = batch.to(device)
        loss, aux = model(batch)
        for k in total_loss_components:
            total_loss_components[k].append(aux[k].item())

        z = aux["z"]
        edge_index = aux["combined_edge_index"]
        n_nodes = z.size(0)
        e_struct = compute_structure_error(z.cpu(), edge_index.cpu(), n_nodes)

        # NOTE: PyG's DataLoader/Batch collation wraps non-tensor attributes
        # (plain numpy arrays such as mmsi/timestamp/behavior_label) in a
        # length-`batch_size` Python list even at batch_size=1 -- it does
        # NOT concatenate them like tensor attributes. Since this loader
        # always uses batch_size=1, each list has exactly one element,
        # which we unwrap here before accumulating across snapshots.
        all_feat_err.append(aux["per_node_feat_error"].cpu().numpy())
        all_struct_err.append(e_struct)
        all_ari.append(batch["vessel_state"].ari.cpu().numpy())
        all_mmsi.append(np.asarray(batch["vessel_state"].mmsi[0]))
        all_ts.append(np.asarray(batch["vessel_state"].timestamp[0]))
        if hasattr(batch["vessel_state"], "behavior_label"):
            all_labels.append(np.asarray(batch["vessel_state"].behavior_label[0]))

    feat_err = np.concatenate(all_feat_err)
    struct_err = np.concatenate(all_struct_err)
    ari = np.concatenate(all_ari)
    mmsi = np.concatenate(all_mmsi)
    ts = np.concatenate(all_ts)

    result = compute_anomaly_score(feat_err, struct_err, ari, score_cfg)
    result["mmsi"] = mmsi
    result["timestamp"] = ts
    result["ari"] = ari
    result["risk_tier"] = risk_tier(result["anomaly_score"])
    result["loss_components"] = {k: float(np.mean(v)) for k, v in total_loss_components.items()}
    if all_labels:
        result["behavior_label"] = np.concatenate(all_labels)
    return result
