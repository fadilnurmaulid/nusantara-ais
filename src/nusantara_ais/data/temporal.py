"""
Temporal alignment.

Raw AIS messages arrive at irregular intervals (Class A transponders may
report every few seconds while transitioning, but every few minutes while
cruising; Class B / satellite AIS is sparser still). To make trajectories
comparable across vessels and usable for fixed-window graph snapshots, every
per-vessel stream is resampled onto a common `resample_interval_min` grid:

  * Gaps shorter than `max_interp_gap_min` are filled by linear interpolation
    of lat/lon (great-circle-aware spherical linear interpolation is
    approximated by linear interpolation in projected meters, which is
    accurate at this spatial/temporal scale) and by forward-fill for
    categorical fields (nav_status, vessel_type).
  * Gaps longer than `max_interp_gap_min` are NOT interpolated across --
    doing so would fabricate a plausible-looking trajectory through a period
    where the vessel could have gone anywhere, silently destroying the
    blackout signal the ARI and anomaly score depend on. Instead the gap is
    left as a discontinuity and annotated on the following real message with
    `ais_gap_min` (time since previous real message) and, if it exceeds
    `blackout_threshold_min`, `is_blackout_event = True`.
"""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from ..config import IngestionConfig


def _resample_single_vessel(g: pd.DataFrame, cfg: IngestionConfig) -> pd.DataFrame:
    g = g.sort_values(cfg.time_col).reset_index(drop=True)

    # record raw inter-message gap BEFORE resampling (used for ais_gap / blackout features)
    raw_gap_min = g[cfg.time_col].diff().dt.total_seconds().div(60.0)
    g["_raw_gap_min"] = raw_gap_min.fillna(0.0)
    g["_is_blackout_event"] = g["_raw_gap_min"] > cfg.blackout_threshold_min

    start, end = g[cfg.time_col].iloc[0], g[cfg.time_col].iloc[-1]
    if start == end:
        grid = pd.DatetimeIndex([start])
    else:
        grid = pd.date_range(start, end, freq=f"{cfg.resample_interval_min}min")

    g_idx = g.set_index(cfg.time_col)
    resampled = g_idx.reindex(g_idx.index.union(grid)).sort_index()

    # interpolate numeric navigational fields time-aware, but only within
    # max_interp_gap_min of a real observation
    numeric_cols = ["lat", "lon", "sog", "cog", "heading", "length", "width", "draft"]
    numeric_cols = [c for c in numeric_cols if c in resampled.columns]

    time_since_obs = pd.Series(resampled.index, index=resampled.index)
    has_obs = resampled["mmsi"].notna()
    last_obs_time = time_since_obs.where(has_obs).ffill()
    next_obs_time = time_since_obs.where(has_obs).bfill()
    gap_minutes = ((next_obs_time - last_obs_time).dt.total_seconds() / 60.0)

    interpolated = resampled[numeric_cols].interpolate(method="time", limit_area="inside")
    within_limit = gap_minutes <= cfg.max_interp_gap_min
    for col in numeric_cols:
        resampled[col] = np.where(within_limit.values, interpolated[col].values, resampled[col].values)

    for col in ["nav_status", "vessel_type", "mmsi"]:
        if col in resampled.columns:
            resampled[col] = resampled[col].ffill()

    resampled = resampled.reindex(grid)
    resampled["mmsi"] = g["mmsi"].iloc[0]
    resampled.index.name = cfg.time_col
    resampled = resampled.reset_index()

    # propagate gap annotations onto the resampled grid at the nearest real timestamp
    real_times = g[cfg.time_col].values
    gap_lookup = dict(zip(g[cfg.time_col], g["_raw_gap_min"]))
    blackout_lookup = dict(zip(g[cfg.time_col], g["_is_blackout_event"]))
    resampled["ais_gap_min"] = resampled[cfg.time_col].map(gap_lookup).fillna(0.0)
    resampled["is_blackout_event"] = resampled[cfg.time_col].map(blackout_lookup).fillna(False)
    resampled["is_interpolated"] = ~resampled[cfg.time_col].isin(real_times)

    return resampled.dropna(subset=["lat", "lon"]).reset_index(drop=True)


def align_temporal(df: pd.DataFrame, cfg: IngestionConfig) -> pd.DataFrame:
    """Resample every vessel's stream onto a fixed cadence grid.

    Returns a concatenated, per-vessel-resampled DataFrame with additional
    columns: ais_gap_min, is_blackout_event, is_interpolated.
    """
    out: List[pd.DataFrame] = []
    for mmsi, g in df.groupby(cfg.id_col, sort=False):
        out.append(_resample_single_vessel(g, cfg))
    result = pd.concat(out, ignore_index=True)
    return result.sort_values([cfg.id_col, cfg.time_col]).reset_index(drop=True)
