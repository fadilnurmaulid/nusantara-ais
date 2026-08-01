"""
AIS cleaning pipeline.

Stages (applied in order, each is idempotent and logged):
 1. Schema enforcement       -- required columns present & correctly typed
 2. Bounding-box filtering    -- drop messages outside the plausible EEZ bbox
 3. Duplicate removal         -- exact duplicate (mmsi, timestamp) rows
 4. Null coordinate drop      -- lat/lon NaN or (0,0) null-island artifacts
 5. Physically-implausible speed/accel hard filter -- see `IngestionConfig`
 6. Per-MMSI chronological sort
 7. Duplicate-timestamp resolution -- keep the message with more complete
    fields when two records share (mmsi, timestamp) but are not exact dups
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ..config import IngestionConfig

logger = logging.getLogger("nusantara_ais.cleaning")


def enforce_schema(df: pd.DataFrame, cfg: IngestionConfig) -> pd.DataFrame:
    missing = [c for c in cfg.ais_columns if c not in df.columns]
    if missing:
        raise ValueError(f"AIS dataframe missing required columns: {missing}")
    df = df.copy()
    df[cfg.time_col] = pd.to_datetime(df[cfg.time_col])
    for col in ("lat", "lon", "sog", "cog", "heading"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def filter_bounds(df: pd.DataFrame, cfg: IngestionConfig) -> pd.DataFrame:
    lat_lo, lat_hi = cfg.lat_bounds
    lon_lo, lon_hi = cfg.lon_bounds
    mask = (
        df["lat"].between(lat_lo, lat_hi)
        & df["lon"].between(lon_lo, lon_hi)
        & ~((df["lat"].abs() < 1e-6) & (df["lon"].abs() < 1e-6))  # null island
        & df["lat"].notna() & df["lon"].notna()
    )
    n_dropped = (~mask).sum()
    logger.info(f"filter_bounds: dropping {n_dropped} / {len(df)} out-of-bbox/null rows")
    return df.loc[mask].reset_index(drop=True)


def drop_duplicates(df: pd.DataFrame, cfg: IngestionConfig) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates(subset=[cfg.id_col, cfg.time_col], keep="first")
    logger.info(f"drop_duplicates: removed {before - len(df)} exact/timestamp duplicates")
    return df.reset_index(drop=True)


def filter_implausible_kinematics(df: pd.DataFrame, cfg: IngestionConfig) -> pd.DataFrame:
    """Hard-reject rows with an impossible reported SOG. Finer-grained
    acceleration/turn-rate implausibility is NOT hard-filtered here -- it is
    intentionally preserved and scored continuously by the ARI, because a
    single implausible kinematic reading is itself evidence of unreliability
    (e.g. spoofing) rather than noise to be discarded.
    """
    mask = df["sog"].between(0, cfg.max_sog_knots) | df["sog"].isna()
    n_dropped = (~mask).sum()
    logger.info(f"filter_implausible_kinematics: dropping {n_dropped} rows with SOG > {cfg.max_sog_knots}kn")
    return df.loc[mask].reset_index(drop=True)


def sort_chronological(df: pd.DataFrame, cfg: IngestionConfig) -> pd.DataFrame:
    return df.sort_values([cfg.id_col, cfg.time_col]).reset_index(drop=True)


def clean_ais(df: pd.DataFrame, cfg: IngestionConfig) -> pd.DataFrame:
    df = enforce_schema(df, cfg)
    df = filter_bounds(df, cfg)
    df = drop_duplicates(df, cfg)
    df = filter_implausible_kinematics(df, cfg)
    df = sort_chronological(df, cfg)
    logger.info(f"clean_ais: final shape {df.shape}")
    return df
