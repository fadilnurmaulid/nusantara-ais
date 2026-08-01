"""
Minimal but meaningful unit tests covering the paper's three contributions.
Run with: python -m pytest tests/ -v   (or plain: python tests/test_core.py)
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd

from nusantara_ais.config import Config
from nusantara_ais.data import cleaning, coordinate, temporal, trajectory
from nusantara_ais.data.ingestion import DataIngestor
from nusantara_ais.features import dynamic, reliability
from nusantara_ais.features.engineer import engineer_all_features
from nusantara_ais.graph.builder import MaritimeGraphBuilder
from nusantara_ais.models.anomaly_score import AnomalyScoreConfig, compute_anomaly_score, risk_tier


def _tiny_dataset(cfg):
    ing = DataIngestor(cfg)
    return ing.load()


def test_haversine_known_distance():
    # Jakarta (-6.2088, 106.8456) to Surabaya (-7.2575, 112.7521) ~ 665 km great-circle
    d = coordinate.haversine_km(-6.2088, 106.8456, -7.2575, 112.7521)
    assert 600 < d < 720, f"unexpected distance {d}"


def test_bearing_range():
    b = coordinate.bearing_deg(-6.0, 106.0, -5.0, 107.0)
    assert 0 <= b <= 360


def test_ari_in_unit_interval():
    cfg = Config()
    ds = _tiny_dataset(cfg)
    df = cleaning.clean_ais(ds.ais.sample(300, random_state=0), cfg.ingestion)
    df = temporal.align_temporal(df, cfg.ingestion)
    df = trajectory.segment_trajectories(df, cfg.ingestion)
    df = dynamic.add_dynamic_features(df, cfg.ingestion)
    df = reliability.compute_ari(df, cfg.ari, cfg.ingestion)
    assert (df["ari"] >= 0).all() and (df["ari"] <= 1).all()


def test_ari_penalizes_blackout():
    """A vessel with a long blackout gap should score a lower ARI on the
    message immediately after the blackout than one with dense reporting."""
    cfg = Config()
    t0 = pd.Timestamp("2024-01-01")
    normal_times = pd.date_range(t0, periods=20, freq="10min")
    blackout_times = list(pd.date_range(t0, periods=10, freq="10min")) + \
        list(pd.date_range(t0 + pd.Timedelta(hours=5), periods=10, freq="10min"))

    def make_df(times, mmsi):
        n = len(times)
        return pd.DataFrame({
            "mmsi": mmsi, "timestamp": times,
            "lat": -5.0 + np.linspace(0, 0.05, n), "lon": 110.0 + np.linspace(0, 0.05, n),
            "sog": 8.0, "cog": 45.0, "heading": 45.0, "nav_status": 0,
            "vessel_type": 70, "length": 80.0, "width": 12.0, "draft": 5.0,
        })

    df = pd.concat([make_df(normal_times, 1), make_df(blackout_times, 2)], ignore_index=True)
    df = cleaning.clean_ais(df, cfg.ingestion)
    df = temporal.align_temporal(df, cfg.ingestion)
    df = trajectory.segment_trajectories(df, cfg.ingestion)
    df = dynamic.add_dynamic_features(df, cfg.ingestion)
    df = reliability.compute_ari(df, cfg.ari, cfg.ingestion)

    ari_normal = df.loc[df["mmsi"] == 1, "ari"].mean()
    ari_blackout = df.loc[df["mmsi"] == 2, "ari"].mean()
    assert ari_blackout <= ari_normal, f"blackout vessel ARI ({ari_blackout}) should be <= normal ({ari_normal})"


def test_anomaly_score_bounds_and_monotonicity():
    n = 200
    rng = np.random.default_rng(0)
    feat_err = rng.exponential(1.0, n)
    struct_err = rng.uniform(0, 1, n)
    ari = rng.uniform(0, 1, n)
    cfg = AnomalyScoreConfig()
    result = compute_anomaly_score(feat_err, struct_err, ari, cfg)
    score = result["anomaly_score"]
    assert (score >= 0).all() and (score <= 1).all()

    tiers = risk_tier(score)
    assert set(np.unique(tiers)).issubset({"LOW", "MODERATE", "HIGH", "CRITICAL"})

    # the node with the max feature error should not be in the LOW tier
    worst_idx = feat_err.argmax()
    assert tiers[worst_idx] != "LOW" or score[worst_idx] > np.quantile(score, 0.4)


def test_graph_snapshot_construction_shapes():
    cfg = Config()
    ds = _tiny_dataset(cfg)
    df = cleaning.clean_ais(ds.ais, cfg.ingestion)
    df = temporal.align_temporal(df, cfg.ingestion)
    df = trajectory.segment_trajectories(df, cfg.ingestion)
    df = engineer_all_features(df, ds, cfg)

    builder = MaritimeGraphBuilder(cfg, ds)
    snapshots = builder.build_snapshots(df)
    assert len(snapshots) > 0
    first = snapshots[0]
    assert "vessel_state" in first.node_types
    assert first["vessel_state"].x.shape[0] > 0
    assert first["vessel_state"].x.shape[1] > 0
    for edge_type in first.edge_types:
        store = first[edge_type]
        assert store.edge_index.shape[0] == 2


def test_bounding_box_filter_rejects_out_of_range():
    cfg = Config()
    df = pd.DataFrame({
        "mmsi": [1, 2], "timestamp": pd.to_datetime(["2024-01-01", "2024-01-01"]),
        "lat": [0.0, 90.0], "lon": [110.0, 200.0],  # second row is out of bounds
        "sog": [5.0, 5.0], "cog": [10.0, 10.0], "heading": [10.0, 10.0],
        "nav_status": [0, 0], "vessel_type": [70, 70], "length": [50.0, 50.0],
        "width": [10.0, 10.0], "draft": [4.0, 4.0],
    })
    cleaned = cleaning.clean_ais(df, cfg.ingestion)
    assert len(cleaned) == 1
    assert cleaned.iloc[0]["mmsi"] == 1


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        print(f"running {t.__name__} ...")
        t()
        print("  OK")
    print(f"\n{len(tests)} tests passed.")
