r"""
Context-aware Maritime Anomaly Score -- CONTRIBUTION 3.

Definition
----------
For vessel_state node i, three raw error/reliability signals are already
available after a forward pass of the full model:

    e_feat(i)   = per-node feature reconstruction MSE  (from GraphAutoEncoder)
    e_struct(i) = 1 - mean_{j in N(i)} \hat{A}_{ij}      (mean predicted link
                  probability to i's TRUE spatio-temporal neighbors; low
                  value = the model could not predict i's own real edges)
    r(i)        = 1 - ARI(i)                              (unreliability)

Each signal lives on a different, dataset-dependent scale, so each is first
mapped to a percentile rank within the current evaluation population
(robust to outliers and requires no fixed normalization constants that
would need re-tuning as fleet composition changes):

    \tilde{e}_feat(i)   = rank_percentile( e_feat(i) )     in [0, 1]
    \tilde{e}_struct(i) = rank_percentile( e_struct(i) )   in [0, 1]
    \tilde{r}(i)         = rank_percentile( r(i) )           in [0, 1]

Context-aware Maritime Anomaly Score:

    AnomalyScore(i) = w_f * \tilde{e}_feat(i) + w_s * \tilde{e}_struct(i) + w_r * \tilde{r}(i)

with (w_f, w_s, w_r) = (0.5, 0.3, 0.2) by default -- feature-reconstruction
error is the dominant signal (it directly encodes "this vessel's full
context-fused behavior is atypical"), structure error is secondary (catches
context/rendezvous implausibility), and the ARI/unreliability term is a
smaller but non-trivial contribution so that an otherwise behaviorally
normal-looking vessel with strong evidence of AIS unreliability (blackout,
spoofing) is still elevated -- this is precisely what makes the score
"context-aware": it fuses learned behavioral abnormality (from the graph)
with physically-grounded reliability evidence (from the ARI) rather than
relying on either alone.

The weights are exposed via `AnomalyScoreConfig` and are the target of the
paper's "ARI sensitivity" experiment (see evaluation/ablation.py).

AnomalyScore(i) in [0, 1] by construction (convex combination of three
values already in [0, 1]). Downstream `risk_assessment` maps the continuous
score onto qualitative risk tiers for operational use, WITHOUT ever
collapsing it back to a hard illegal-fishing classification (per the
project's explicit non-goal).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch


@dataclass
class AnomalyScoreConfig:
    w_feat: float = 0.5
    w_struct: float = 0.3
    w_reliability: float = 0.2


def _percentile_rank(values: np.ndarray) -> np.ndarray:
    order = values.argsort().argsort()
    n = len(values)
    return order / max(n - 1, 1)


def compute_structure_error(z: torch.Tensor, true_edge_index: torch.Tensor, n_nodes: int) -> np.ndarray:
    """e_struct(i) = 1 - mean predicted link probability to i's true neighbors."""
    if true_edge_index.numel() == 0:
        return np.zeros(n_nodes)
    with torch.no_grad():
        src, dst = true_edge_index
        pred = torch.sigmoid((z[src] * z[dst]).sum(dim=-1))
    sums = torch.zeros(n_nodes)
    counts = torch.zeros(n_nodes)
    sums.index_add_(0, src.cpu(), pred.cpu())
    counts.index_add_(0, src.cpu(), torch.ones_like(pred.cpu()))
    mean_pred = torch.where(counts > 0, sums / counts.clamp(min=1), torch.zeros_like(sums))
    e_struct = 1.0 - mean_pred.numpy()
    e_struct[counts.numpy() == 0] = e_struct[counts.numpy() > 0].mean() if (counts.numpy() > 0).any() else 0.0
    return e_struct


def compute_anomaly_score(per_node_feat_error: np.ndarray, e_struct: np.ndarray, ari: np.ndarray,
                           score_cfg: AnomalyScoreConfig) -> Dict[str, np.ndarray]:
    e_feat_pct = _percentile_rank(per_node_feat_error)
    e_struct_pct = _percentile_rank(e_struct)
    unreliability_pct = _percentile_rank(1.0 - ari)

    score = (
        score_cfg.w_feat * e_feat_pct
        + score_cfg.w_struct * e_struct_pct
        + score_cfg.w_reliability * unreliability_pct
    )
    return {
        "anomaly_score": score,
        "feat_error_percentile": e_feat_pct,
        "struct_error_percentile": e_struct_pct,
        "unreliability_percentile": unreliability_pct,
    }


def risk_tier(score: np.ndarray) -> np.ndarray:
    """Maps the continuous Context-aware Maritime Anomaly Score onto four
    qualitative operational risk tiers for maritime-surveillance triage.
    Thresholds are quartile-based by default (data-driven, not fixed
    magic numbers) so tiering is stable across deployments with different
    score distributions; a fixed-threshold variant is provided in
    evaluation/metrics.py for cross-run comparability."""
    tiers = np.empty(len(score), dtype=object)
    q1, q2, q3 = np.quantile(score, [0.5, 0.85, 0.97])
    tiers[score <= q1] = "LOW"
    tiers[(score > q1) & (score <= q2)] = "MODERATE"
    tiers[(score > q2) & (score <= q3)] = "HIGH"
    tiers[score > q3] = "CRITICAL"
    return tiers
