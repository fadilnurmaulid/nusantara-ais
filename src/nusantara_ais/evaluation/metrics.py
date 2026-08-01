"""
Evaluation metrics.

Since the model predicts a CONTINUOUS anomaly score (never a hard illegal-
fishing classification, per the project's explicit non-goal), quantitative
detection quality against the synthetic ground-truth behavior labels is
measured with THRESHOLD-FREE ranking metrics:

  - ROC-AUC       : probability a random anomalous vessel-state is ranked
                    above a random normal one
  - PR-AUC (average precision): more informative than ROC-AUC under the
                    realistic severe class imbalance of maritime anomalies
  - Anomaly-tier recall: fraction of true anomalies captured within the
                    HIGH/CRITICAL risk tiers (an operationally meaningful,
                    threshold-based secondary metric)

"Anomalous" ground truth = behavior_label NOT IN {normal_transit,
normal_fishing} (i.e. every simulated anomaly class is pooled into a single
binary target purely for evaluation; the model itself never sees this
binary label).
"""
from __future__ import annotations

import time
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

NORMAL_LABELS = {"normal_transit", "normal_fishing"}


def binary_ground_truth(behavior_labels: np.ndarray) -> np.ndarray:
    return np.array([0 if lbl in NORMAL_LABELS else 1 for lbl in behavior_labels])


def detection_metrics(anomaly_score: np.ndarray, behavior_labels: np.ndarray, risk_tier: np.ndarray = None
                       ) -> Dict[str, float]:
    y_true = binary_ground_truth(behavior_labels)
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return {"roc_auc": float("nan"), "pr_auc": float("nan")}

    roc_auc = roc_auc_score(y_true, anomaly_score)
    pr_auc = average_precision_score(y_true, anomaly_score)

    metrics = {"roc_auc": float(roc_auc), "pr_auc": float(pr_auc),
               "n_samples": int(len(y_true)), "n_anomalous": int(y_true.sum())}

    if risk_tier is not None:
        high_or_critical = np.isin(risk_tier, ["HIGH", "CRITICAL"])
        recall_at_tier = (high_or_critical & (y_true == 1)).sum() / max(y_true.sum(), 1)
        precision_at_tier = (high_or_critical & (y_true == 1)).sum() / max(high_or_critical.sum(), 1)
        metrics["tier_recall"] = float(recall_at_tier)
        metrics["tier_precision"] = float(precision_at_tier)
    return metrics


def per_class_mean_score(anomaly_score: np.ndarray, behavior_labels: np.ndarray) -> pd.DataFrame:
    df = pd.DataFrame({"behavior_label": behavior_labels, "anomaly_score": anomaly_score})
    return df.groupby("behavior_label")["anomaly_score"].agg(["mean", "std", "count"]).reset_index()


def false_positive_analysis(anomaly_score: np.ndarray, behavior_labels: np.ndarray, threshold_quantile: float = 0.85
                             ) -> pd.DataFrame:
    """Vessel-states flagged HIGH/CRITICAL (top (1-threshold_quantile)) that
    are actually normal traffic -- the false-positive population operators
    would need to triage; broken down by behavior_label to see whether a
    specific normal-traffic pattern (e.g. legitimate anchoring) is
    systematically over-scored."""
    threshold = np.quantile(anomaly_score, threshold_quantile)
    flagged = anomaly_score >= threshold
    y_true = binary_ground_truth(behavior_labels)
    fp_mask = flagged & (y_true == 0)
    fp_labels = pd.Series(behavior_labels[fp_mask])
    return fp_labels.value_counts().rename_axis("normal_behavior_type").reset_index(name="false_positive_count")


def failure_analysis(anomaly_score: np.ndarray, behavior_labels: np.ndarray, threshold_quantile: float = 0.85
                      ) -> pd.DataFrame:
    """Anomalous ground-truth vessel-states that were NOT flagged (false
    negatives / missed detections), broken down by anomaly type -- reveals
    which anomaly classes the model under-detects."""
    threshold = np.quantile(anomaly_score, threshold_quantile)
    flagged = anomaly_score >= threshold
    y_true = binary_ground_truth(behavior_labels)
    fn_mask = (~flagged) & (y_true == 1)
    fn_labels = pd.Series(behavior_labels[fn_mask])
    return fn_labels.value_counts().rename_axis("missed_anomaly_type").reset_index(name="false_negative_count")


def runtime_and_complexity(model, dataset, device, n_repeats: int = 3) -> Dict[str, float]:
    """Empirical runtime profiling + theoretical complexity note.

    Theoretical complexity per snapshot with N_v vessel_state nodes,
    N_c context nodes, E edges, hidden dim d, heads h, L layers:
      HetGAT forward:  O(L * (E * d/h * h + N * d^2)) = O(L * E * d + L * N * d^2)
      GAE decode:       feature decoder O(N * d * F_v); structure decoder
                        O(E_pos * d + E_neg * d) (inner products, linear in
                        edges sampled, NOT quadratic in N since only
                        `negative_sampling_ratio * E_pos` negatives are drawn)
      Overall: O(L * E * d + N * d * max(d, F_v))  -- linear in both nodes
      and edges per snapshot, which is what makes the sliding-window
      snapshot design (bounded N, E per snapshot) tractable at fleet scale.
    """
    from torch_geometric.loader import DataLoader
    import torch

    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    times = []
    model.eval()
    with torch.no_grad():
        for _ in range(n_repeats):
            for batch in loader:
                batch = batch.to(device)
                t0 = time.perf_counter()
                _ = model(batch)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                times.append(time.perf_counter() - t0)

    n_params = sum(p.numel() for p in model.parameters())
    return {
        "mean_forward_ms": float(np.mean(times) * 1000),
        "p95_forward_ms": float(np.percentile(times, 95) * 1000),
        "n_trainable_params": int(n_params),
        "n_snapshots_profiled": len(dataset) * n_repeats,
    }
