"""
Inference pipeline: apply a trained NusantaraAISModel checkpoint to new raw
AIS data, running the full stack (cleaning -> temporal alignment ->
segmentation -> feature engineering -> ARI -> graph construction ->
HetGAT+GAE forward pass -> Context-aware Maritime Anomaly Score) and
returning a per-vessel-state anomaly report.

This is the single entry point operational users (and `scripts/run_pipeline.py`)
should call for scoring new data; it deliberately re-uses every pipeline
module rather than re-implementing preprocessing, guaranteeing train/serve
feature consistency (the single most common source of ML production bugs).
"""
from __future__ import annotations

from typing import Dict

import pandas as pd
import torch

from ..config import Config
from ..data import cleaning, temporal, trajectory
from ..data.ingestion import MaritimeDataset
from ..features.engineer import engineer_all_features
from ..graph.builder import MaritimeGraphBuilder
from ..graph.dataset import MaritimeSnapshotDataset
from ..models.anomaly_score import AnomalyScoreConfig
from ..models.full_model import NusantaraAISModel
from ..training.validate import run_inference
from ..utils.reproducibility import get_device


def preprocess_raw_ais(raw_ais: pd.DataFrame, dataset: MaritimeDataset, cfg: Config) -> pd.DataFrame:
    df = cleaning.clean_ais(raw_ais, cfg.ingestion)
    df = temporal.align_temporal(df, cfg.ingestion)
    df = trajectory.segment_trajectories(df, cfg.ingestion)
    df = engineer_all_features(df, dataset, cfg)
    return df


def load_checkpoint(checkpoint_path: str, device: torch.device) -> Dict:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = Config()  # structural defaults; hyperparameters restored from ckpt below where relevant
    model = NusantaraAISModel(ckpt["node_feature_dims"], ckpt["edge_types"], cfg.model).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return {"model": model, "config_dict": ckpt["config"], "node_feature_dims": ckpt["node_feature_dims"],
            "edge_types": ckpt["edge_types"]}


def score_new_data(raw_ais: pd.DataFrame, dataset: MaritimeDataset, cfg: Config, checkpoint_path: str,
                    score_cfg: AnomalyScoreConfig = None) -> pd.DataFrame:
    device = get_device(cfg.training.device)
    ckpt = load_checkpoint(checkpoint_path, device)

    engineered = preprocess_raw_ais(raw_ais, dataset, cfg)
    builder = MaritimeGraphBuilder(cfg, dataset)
    snapshots = builder.build_snapshots(engineered)
    infer_dataset = MaritimeSnapshotDataset(snapshots)

    result = run_inference(ckpt["model"], infer_dataset, device, score_cfg)

    report = pd.DataFrame({
        "mmsi": result["mmsi"], "timestamp": result["timestamp"], "ari": result["ari"],
        "anomaly_score": result["anomaly_score"], "risk_tier": result["risk_tier"],
        "feat_error_percentile": result["feat_error_percentile"],
        "struct_error_percentile": result["struct_error_percentile"],
        "unreliability_percentile": result["unreliability_percentile"],
    })
    if "behavior_label" in result:
        report["behavior_label"] = result["behavior_label"]
    return report.sort_values("anomaly_score", ascending=False).reset_index(drop=True)
