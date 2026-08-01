"""
Visualization pipeline. Every function saves a PNG to `PathConfig.figures_dir`
and returns the saved path, so `scripts/run_pipeline.py` can call the whole
suite and present a manifest of figures.
"""
from __future__ import annotations

import os
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..config import Config


def _savefig(fig, cfg: Config, name: str) -> str:
    out_dir = cfg.paths.abs(cfg.paths.figures_dir)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_dataset_statistics(df: pd.DataFrame, cfg: Config) -> str:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].hist(df["sog"].dropna(), bins=40, color="#2b6cb0")
    axes[0, 0].set_title("SOG distribution (knots)")

    if "behavior_label" in df.columns:
        df["behavior_label"].value_counts().plot(kind="barh", ax=axes[0, 1], color="#2f855a")
        axes[0, 1].set_title("Behavior class counts")
    else:
        axes[0, 1].axis("off")

    axes[1, 0].hist(df["ari"].dropna(), bins=40, color="#c05621")
    axes[1, 0].set_title("ARI distribution")

    axes[1, 1].scatter(df["lon"], df["lat"], s=1, alpha=0.3, c="#4a5568")
    axes[1, 1].set_title("Spatial footprint (lat/lon)")
    axes[1, 1].set_xlabel("lon"); axes[1, 1].set_ylabel("lat")

    fig.tight_layout()
    return _savefig(fig, cfg, "dataset_statistics")


def plot_anomaly_score_distribution(result: dict, cfg: Config) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].hist(result["anomaly_score"], bins=40, color="#805ad5")
    axes[0].set_title("Context-aware Maritime Anomaly Score distribution")
    axes[0].set_xlabel("anomaly score")

    if "behavior_label" in result:
        df = pd.DataFrame({"score": result["anomaly_score"], "label": result["behavior_label"]})
        df.boxplot(column="score", by="label", ax=axes[1], rot=45)
        axes[1].set_title("Anomaly score by behavior class")
        plt.suptitle("")
    else:
        axes[1].axis("off")

    fig.tight_layout()
    return _savefig(fig, cfg, "anomaly_score_distribution")


def plot_roc_pr_curves(anomaly_score: np.ndarray, behavior_labels: np.ndarray, cfg: Config) -> str:
    from sklearn.metrics import roc_curve, precision_recall_curve
    from .metrics import binary_ground_truth

    y_true = binary_ground_truth(behavior_labels)
    fpr, tpr, _ = roc_curve(y_true, anomaly_score)
    prec, rec, _ = precision_recall_curve(y_true, anomaly_score)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(fpr, tpr, color="#2b6cb0"); axes[0].plot([0, 1], [0, 1], "--", color="gray")
    axes[0].set_title("ROC curve"); axes[0].set_xlabel("FPR"); axes[0].set_ylabel("TPR")

    axes[1].plot(rec, prec, color="#c05621")
    axes[1].set_title("Precision-Recall curve"); axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")

    fig.tight_layout()
    return _savefig(fig, cfg, "roc_pr_curves")


def plot_embedding_projection(embedding_df: pd.DataFrame, cfg: Config) -> str:
    fig, ax = plt.subplots(figsize=(7, 6))
    if "behavior_label" in embedding_df.columns:
        for label, g in embedding_df.groupby("behavior_label"):
            ax.scatter(g["dim1"], g["dim2"], s=6, alpha=0.6, label=label)
        ax.legend(fontsize=7, markerscale=2)
    else:
        ax.scatter(embedding_df["dim1"], embedding_df["dim2"], s=6, alpha=0.6)
    ax.set_title("Vessel-state embedding (PCA projection)")
    fig.tight_layout()
    return _savefig(fig, cfg, "embedding_projection")


def plot_attention_summary(attention_df: pd.DataFrame, cfg: Config) -> str:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(attention_df["relation"], attention_df["mean_attention"], color="#3182ce")
    ax.set_title("Mean HetGAT attention by relation (layer 1)")
    ax.set_xlabel("mean attention weight")
    fig.tight_layout()
    return _savefig(fig, cfg, "attention_summary")


def plot_ablation_comparison(ablation_df: pd.DataFrame, cfg: Config) -> str:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    metric_col = "roc_auc" if "roc_auc" in ablation_df.columns else ablation_df.columns[1]
    key_col = "variant" if "variant" in ablation_df.columns else "model"
    ax.bar(ablation_df[key_col], ablation_df[metric_col], color="#38a169")
    ax.set_title(f"{metric_col} across variants")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    return _savefig(fig, cfg, "ablation_comparison")


def plot_case_study(case_df: pd.DataFrame, cfg: Config, mmsi: int) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(case_df["timestamp"], case_df["anomaly_score"], color="#c53030")
    axes[0].set_title(f"MMSI {mmsi}: anomaly score over time")
    axes[0].tick_params(axis="x", rotation=30)

    sc = axes[1].scatter(case_df["lon"], case_df["lat"], c=case_df["anomaly_score"], cmap="Reds", s=15)
    axes[1].set_title("Trajectory colored by anomaly score")
    plt.colorbar(sc, ax=axes[1])

    fig.tight_layout()
    return _savefig(fig, cfg, f"case_study_mmsi_{mmsi}")
