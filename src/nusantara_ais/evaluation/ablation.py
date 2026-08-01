"""
Complete experimental suite: baseline comparison, ablation study, feature
importance, ARI sensitivity, embedding analysis, attention analysis, case
study support.

Every function returns a plain pandas DataFrame / dict so results can be
directly rendered into the paper's tables/figures without additional
glue code.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader

from ..config import Config
from ..graph.dataset import MaritimeSnapshotDataset
from ..models import baselines as B
from ..models.anomaly_score import AnomalyScoreConfig, compute_anomaly_score, compute_structure_error
from .metrics import detection_metrics


def _flatten_vessel_features(dataset: MaritimeSnapshotDataset) -> Dict[str, np.ndarray]:
    X, labels, ari = [], [], []
    for i in range(len(dataset)):
        data = dataset.get(i)
        X.append(data["vessel_state"].x.numpy())
        ari.append(data["vessel_state"].ari.numpy())
        if hasattr(data["vessel_state"], "behavior_label"):
            labels.append(np.asarray(data["vessel_state"].behavior_label))
    return {"X": np.concatenate(X), "ari": np.concatenate(ari),
            "labels": np.concatenate(labels) if labels else None}


def run_baseline_comparison(train_ds: MaritimeSnapshotDataset, test_ds: MaritimeSnapshotDataset,
                             device: torch.device, seed: int = 42) -> pd.DataFrame:
    train_flat = _flatten_vessel_features(train_ds)
    test_flat = _flatten_vessel_features(test_ds)
    X_train, X_test = train_flat["X"], test_flat["X"]
    labels_test = test_flat["labels"]

    results = []

    iso_scores = B.run_isolation_forest(X_test, seed=seed)
    results.append({"model": "IsolationForest", **detection_metrics(iso_scores, labels_test)})

    ocsvm_scores = B.run_one_class_svm(X_test)
    results.append({"model": "OneClassSVM", **detection_metrics(ocsvm_scores, labels_test)})

    if train_flat["labels"] is not None:
        from .metrics import binary_ground_truth
        y_train = binary_ground_truth(train_flat["labels"])
        _, xgb_scores = B.run_xgboost_supervised(X_train, y_train, X_test, seed=seed)
        results.append({"model": "XGBoost(supervised)", **detection_metrics(xgb_scores, labels_test)})

    for name, model_cls in (("GraphSAGE", B.HomogeneousGraphSAGE), ("GAT", B.HomogeneousGAT)):
        model = model_cls(in_dim=X_train.shape[1]).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loader = DataLoader(train_ds, batch_size=8, shuffle=True)
        model.train()
        for _ in range(20):
            for batch in loader:
                batch = batch.to(device)
                edge_index = B.homogeneous_edge_index(batch)
                optimizer.zero_grad()
                err, _ = model(batch["vessel_state"].x, edge_index)
                loss = err.mean()
                loss.backward()
                optimizer.step()

        model.eval()
        all_err, all_labels = [], []
        test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                edge_index = B.homogeneous_edge_index(batch)
                err, _ = model(batch["vessel_state"].x, edge_index)
                all_err.append(err.cpu().numpy())
                if hasattr(batch["vessel_state"], "behavior_label"):
                    # batch_size=1 loader: PyG wraps non-tensor attrs in a list
                    all_labels.append(np.asarray(batch["vessel_state"].behavior_label[0]))
        err_concat = np.concatenate(all_err)
        score = (err_concat - err_concat.min()) / (err_concat.max() - err_concat.min() + 1e-9)
        lbl_concat = np.concatenate(all_labels) if all_labels else labels_test
        results.append({"model": name, **detection_metrics(score, lbl_concat)})

    return pd.DataFrame(results)


def run_ablation_study(model_full, model_no_struct_loss, model_no_ari_loss, model_no_hetero,
                        test_ds: MaritimeSnapshotDataset, device: torch.device,
                        score_cfg: AnomalyScoreConfig) -> pd.DataFrame:
    """Compares the full model against three ablations:
      - no_struct_loss : GAE trained with structure_loss_weight=0
      - no_ari_loss     : GAE trained with ari_loss_weight=0
      - no_hetero       : HetGAT replaced by a homogeneous GAT (see baselines.HomogeneousGAT)
    Each variant must be pre-trained by the caller with the corresponding
    config (see scripts/run_pipeline.py::run_ablation_experiments) and
    passed in here already fitted.
    """
    from ..training.validate import run_inference
    variants = {"full_model": model_full, "no_struct_loss": model_no_struct_loss,
                "no_ari_loss": model_no_ari_loss}
    rows = []
    for name, model in variants.items():
        if model is None:
            continue
        result = run_inference(model, test_ds, device, score_cfg)
        m = detection_metrics(result["anomaly_score"], result.get("behavior_label", np.array([])),
                               result["risk_tier"]) if "behavior_label" in result else {}
        rows.append({"variant": name, **m, **result["loss_components"]})

    if model_no_hetero is not None:
        all_err, all_labels = [], []
        loader = DataLoader(test_ds, batch_size=1, shuffle=False)
        model_no_hetero.eval()
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                edge_index = B.homogeneous_edge_index(batch)
                err, _ = model_no_hetero(batch["vessel_state"].x, edge_index)
                all_err.append(err.cpu().numpy())
                if hasattr(batch["vessel_state"], "behavior_label"):
                    # batch_size=1 loader: PyG wraps non-tensor attrs in a list
                    all_labels.append(np.asarray(batch["vessel_state"].behavior_label[0]))
        err_concat = np.concatenate(all_err)
        score = (err_concat - err_concat.min()) / (err_concat.max() - err_concat.min() + 1e-9)
        lbl_concat = np.concatenate(all_labels)
        rows.append({"variant": "no_heterogeneity_homogeneous_gat", **detection_metrics(score, lbl_concat)})

    return pd.DataFrame(rows)


def feature_importance_via_permutation(model, dataset: MaritimeSnapshotDataset, device: torch.device,
                                        feature_names: List[str], n_repeats: int = 3, seed: int = 42
                                        ) -> pd.DataFrame:
    """Permutation importance: shuffles one vessel_state feature column at a
    time across nodes (within each snapshot) and measures the resulting
    increase in feature-reconstruction loss -- a feature the model relies on
    heavily will cause large error increases when its values are
    permuted-decorrelated from the rest of the node's context."""
    rng = np.random.default_rng(seed)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    model.eval()

    baseline_losses = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            loss, _ = model(batch)
            baseline_losses.append(loss.item())
    baseline = float(np.mean(baseline_losses))

    rows = []
    for f_idx, f_name in enumerate(feature_names):
        deltas = []
        for _ in range(n_repeats):
            perturbed_losses = []
            with torch.no_grad():
                for batch in loader:
                    batch = batch.to(device)
                    x = batch["vessel_state"].x.clone()
                    perm = torch.tensor(rng.permutation(x.size(0)))
                    x[:, f_idx] = x[perm, f_idx]
                    batch["vessel_state"].x = x
                    loss, _ = model(batch)
                    perturbed_losses.append(loss.item())
            deltas.append(float(np.mean(perturbed_losses)) - baseline)
        rows.append({"feature": f_name, "mean_importance": float(np.mean(deltas)),
                     "std_importance": float(np.std(deltas))})
    return pd.DataFrame(rows).sort_values("mean_importance", ascending=False).reset_index(drop=True)


def ari_sensitivity_analysis(feat_err: np.ndarray, struct_err: np.ndarray, ari: np.ndarray,
                              behavior_labels: np.ndarray, w_reliability_grid: List[float] = None) -> pd.DataFrame:
    """Sweeps the ARI (reliability) weight in the anomaly score and reports
    ROC-AUC/PR-AUC at each setting -- demonstrates the marginal contribution
    of the physics-informed ARI term to detection quality (the paper's
    central "why ARI matters" evidence)."""
    w_reliability_grid = w_reliability_grid or [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    rows = []
    for w_r in w_reliability_grid:
        remaining = 1.0 - w_r
        w_f = remaining * (0.5 / 0.8)
        w_s = remaining * (0.3 / 0.8)
        cfg = AnomalyScoreConfig(w_feat=w_f, w_struct=w_s, w_reliability=w_r)
        result = compute_anomaly_score(feat_err, struct_err, ari, cfg)
        m = detection_metrics(result["anomaly_score"], behavior_labels)
        rows.append({"w_reliability": w_r, **m})
    return pd.DataFrame(rows)


def embedding_analysis(z: torch.Tensor, behavior_labels: np.ndarray, method: str = "pca") -> pd.DataFrame:
    """2D projection of vessel_state embeddings for visualization (PCA by
    default -- deterministic and reproducible, unlike t-SNE/UMAP whose
    stochastic layouts complicate exact reproducibility requirements)."""
    z_np = z.cpu().numpy() if isinstance(z, torch.Tensor) else z
    z_centered = z_np - z_np.mean(axis=0, keepdims=True)
    u, s, vt = np.linalg.svd(z_centered, full_matrices=False)
    proj = z_centered @ vt[:2].T
    df = pd.DataFrame({"dim1": proj[:, 0], "dim2": proj[:, 1]})
    if behavior_labels is not None:
        df["behavior_label"] = behavior_labels
    df.attrs["explained_variance_ratio"] = (s[:2] ** 2 / (s ** 2).sum()).tolist()
    return df


def attention_analysis(model, data, device: torch.device) -> pd.DataFrame:
    """Extracts mean attention weight per edge TYPE from the first HetGAT
    layer's GATConv modules (`return_attention_weights=True`), summarizing
    which relations dominate message passing -- e.g. a high mean weight on
    (vessel_state, near_port, port) for a given snapshot indicates port
    proximity was influential for that snapshot's embeddings."""
    model.eval()
    data = data.to(device)
    x_dict = {ntype: model.encoder.input_proj[ntype](data[ntype].x) for ntype in model.encoder.input_proj}
    rows = []
    first_layer = model.encoder.layers[0]
    with torch.no_grad():
        for edge_type, conv in first_layer.convs.items():
            src_t, rel, dst_t = edge_type
            store = data[edge_type]
            if store.edge_index.numel() == 0:
                continue
            edge_attr = store.edge_attr if "edge_attr" in store and store.edge_attr.numel() > 0 else \
                torch.ones((store.edge_index.size(1), 1), device=device)
            _, (edge_index_out, alpha) = conv(
                (x_dict[src_t], x_dict[dst_t]), store.edge_index, edge_attr=edge_attr,
                return_attention_weights=True,
            )
            rows.append({"relation": f"{src_t}->{rel}->{dst_t}", "mean_attention": float(alpha.mean()),
                         "n_edges": int(store.edge_index.size(1))})
    return pd.DataFrame(rows).sort_values("mean_attention", ascending=False).reset_index(drop=True)
