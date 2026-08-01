"""
Trajectory segmentation.

A vessel's raw AIS stream spans arbitrary calendar time and may include long
periods in port, at anchor, or genuinely dark. For graph-snapshot and
sequence-feature purposes we split each MMSI's stream into `segment_id`s
whenever the *raw* (pre-interpolation) inter-message gap exceeds
`segment_gap_threshold_min` -- this is a materially longer threshold than the
blackout threshold, since not every AIS gap should fragment a trajectory
(short gaps are common and handled by interpolation), but a multi-hour
silence usually indicates the vessel entered port, powered down, or the
"trip" genuinely ended.

Segments with fewer than `min_segment_points` are dropped as too short to
support meaningful trajectory-level features (turning rate, angular
velocity, etc. need at least a few consecutive points).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import IngestionConfig


def segment_trajectories(df: pd.DataFrame, cfg: IngestionConfig) -> pd.DataFrame:
    df = df.sort_values([cfg.id_col, cfg.time_col]).reset_index(drop=True)
    gap_min = df.groupby(cfg.id_col)[cfg.time_col].diff().dt.total_seconds().div(60.0).fillna(0.0)
    new_segment = (gap_min > cfg.segment_gap_threshold_min).astype(int)
    # cumulative sum per-vessel gives a monotonically increasing segment index
    seg_within_vessel = new_segment.groupby(df[cfg.id_col]).cumsum()
    df = df.copy()
    df["segment_id"] = df[cfg.id_col].astype(str) + "_SEG_" + seg_within_vessel.astype(str)

    seg_sizes = df.groupby("segment_id").size()
    valid_segments = seg_sizes[seg_sizes >= cfg.min_segment_points].index
    n_dropped_segments = (seg_sizes < cfg.min_segment_points).sum()
    df = df[df["segment_id"].isin(valid_segments)].reset_index(drop=True)
    df.attrs["n_dropped_short_segments"] = int(n_dropped_segments)
    return df
