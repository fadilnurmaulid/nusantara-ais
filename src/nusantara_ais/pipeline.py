"""
End-to-end pipeline orchestrator.

    Raw AIS -> Cleaning -> Trajectory Segmentation -> Spatial Feature
    Engineering -> AIS Reliability Index -> Heterogeneous Graph Construction
    -> HetGAT -> Graph AutoEncoder -> Context-aware Maritime Anomaly Score
    -> Risk Assessment

This module is the single reference implementation of that pipeline
diagram; `scripts/run_pipeline.py` is a thin CLI wrapper around it.
"""
from __future__ import annotations

import logging
import os
from typing import Dict, Optional

import pandas as pd
import torch

from .config import Config
from .data import cleaning, spatial_join, temporal, trajectory
from .data.ingestion import DataIngestor, MaritimeDataset
from .evaluation import ablation, metrics, visualization
from .features.engineer import engineer_all_features
from .graph.builder import MaritimeGraphBuilder
from .graph.dataset import chronological_split
from .models.anomaly_score import AnomalyScoreConfig
from .training.train import train_model
from .training.validate import run_inference
from .utils.reproducibility import get_device, set_global_seed, setup_logger

logger = logging.getLogger("nusantara_ais.pipeline")


def run_full_pipeline(cfg: Optional[Config] = None, n_vessels: int = 60, points_per_vessel: int = 144,
                       generate_figures: bool = True, run_baselines: bool = True) -> Dict:
    cfg = cfg or Config()
    set_global_seed(cfg.training.seed)
    log = setup_logger("nusantara_ais.pipeline", cfg.paths.abs(cfg.paths.logs_dir))

    log.info("=" * 70)
    log.info("STAGE 0/9 -- Data ingestion (real datasets if present, else synthetic)")
    raw_dir = cfg.paths.abs(cfg.paths.raw_dir)
    real_data_present = any(
        os.path.exists(os.path.join(raw_dir, f"{name}.parquet")) or os.path.exists(os.path.join(raw_dir, f"{name}.csv"))
        for name in ("ais_raw", "ports", "shipping_lanes", "coastline", "bathymetry", "bmkg_weather", "protected_areas")
    )
    if not real_data_present:
        # (Re)generate synthetic data at the requested fleet size -- this is
        # what makes --n-vessels / --points-per-vessel on the CLI actually
        # take effect instead of silently reusing whatever synthetic data
        # happens to already be cached on disk from a previous run.
        from .data import synthetic
        synthetic.generate_all(cfg, n_vessels=n_vessels, points_per_vessel=points_per_vessel, seed=cfg.training.seed)
    ingestor = DataIngestor(cfg)
    dataset: MaritimeDataset = ingestor.load()
    raw_ais = dataset.ais
    log.info(f"Loaded {len(raw_ais)} raw AIS messages across {raw_ais['mmsi'].nunique()} vessels")

    log.info("STAGE 1/9 -- Cleaning")
    df = cleaning.clean_ais(raw_ais, cfg.ingestion)

    log.info("STAGE 2/9 -- Temporal alignment")
    df = temporal.align_temporal(df, cfg.ingestion)

    log.info("STAGE 3/9 -- Trajectory segmentation")
    df = trajectory.segment_trajectories(df, cfg.ingestion)

    log.info("STAGE 4/9 -- Spatial feature engineering + AIS Reliability Index")
    df = engineer_all_features(df, dataset, cfg)

    # behavior_label survives resampling as a forward-filled categorical (synthetic ground truth only)
    if "behavior_label" in raw_ais.columns and "behavior_label" not in df.columns:
        label_lookup = raw_ais.groupby("mmsi")["behavior_label"].first()
        df["behavior_label"] = df["mmsi"].map(label_lookup)

    log.info("STAGE 5/9 -- Heterogeneous graph construction (temporal snapshots)")
    builder = MaritimeGraphBuilder(cfg, dataset)
    snapshots = builder.build_snapshots(df)
    train_ds, val_ds, test_ds = chronological_split(snapshots, cfg.training)
    log.info(f"Snapshots: train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

    log.info("STAGE 6/9 -- Training (HetGAT + Graph AutoEncoder)")
    train_result = train_model(cfg, train_ds, val_ds)
    model = train_result["model"]
    device = train_result["device"]

    log.info("STAGE 7/9 -- Inference + Context-aware Maritime Anomaly Score")
    score_cfg = AnomalyScoreConfig()
    test_result = run_inference(model, test_ds, device, score_cfg)

    log.info("STAGE 8/9 -- Evaluation")
    eval_report = {}
    if "behavior_label" in test_result:
        eval_report["detection_metrics"] = metrics.detection_metrics(
            test_result["anomaly_score"], test_result["behavior_label"], test_result["risk_tier"]
        )
        eval_report["per_class_scores"] = metrics.per_class_mean_score(
            test_result["anomaly_score"], test_result["behavior_label"]
        )
        eval_report["false_positive_analysis"] = metrics.false_positive_analysis(
            test_result["anomaly_score"], test_result["behavior_label"]
        )
        eval_report["failure_analysis"] = metrics.failure_analysis(
            test_result["anomaly_score"], test_result["behavior_label"]
        )
        eval_report["ari_sensitivity"] = ablation.ari_sensitivity_analysis(
            test_result["feat_error_percentile"], test_result["struct_error_percentile"],
            test_result["ari"], test_result["behavior_label"]
        )

    eval_report["runtime_complexity"] = metrics.runtime_and_complexity(model, test_ds, device)

    if run_baselines and "behavior_label" in test_result:
        log.info("Running baseline comparison (IsolationForest, OneClassSVM, XGBoost, GraphSAGE, GAT)")
        eval_report["baseline_comparison"] = ablation.run_baseline_comparison(train_ds, test_ds, device,
                                                                               seed=cfg.training.seed)

    log.info("STAGE 9/9 -- Visualization")
    figures = {}
    if generate_figures:
        figures["dataset_statistics"] = visualization.plot_dataset_statistics(df, cfg)
        figures["anomaly_score_distribution"] = visualization.plot_anomaly_score_distribution(test_result, cfg)
        if "behavior_label" in test_result:
            figures["roc_pr_curves"] = visualization.plot_roc_pr_curves(
                test_result["anomaly_score"], test_result["behavior_label"], cfg
            )
        sample_batch = test_ds.get(0)
        with torch.no_grad():
            z_sample = model.encoder(sample_batch.to(device))["vessel_state"]
        embed_labels = getattr(sample_batch["vessel_state"], "behavior_label", None)
        embed_df = ablation.embedding_analysis(z_sample, embed_labels)
        figures["embedding_projection"] = visualization.plot_embedding_projection(embed_df, cfg)

        attn_df = ablation.attention_analysis(model, test_ds.get(0), device)
        figures["attention_summary"] = visualization.plot_attention_summary(attn_df, cfg)

        if "baseline_comparison" in eval_report:
            figures["baseline_comparison"] = visualization.plot_ablation_comparison(
                eval_report["baseline_comparison"], cfg
            )

    log.info("Pipeline complete.")
    return {
        "engineered_df": df, "train_ds": train_ds, "val_ds": val_ds, "test_ds": test_ds,
        "model": model, "train_history": train_result["history"], "test_result": test_result,
        "eval_report": eval_report, "figures": figures, "device": device,
    }
