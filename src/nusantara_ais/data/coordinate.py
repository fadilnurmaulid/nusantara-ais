"""
Coordinate transformation.

All distance-based features (distance_to_port, distance_to_shipping_lane,
distance_to_coastline, spatial-proximity graph edges) require a metric CRS,
since raw lat/lon degrees are not distance-preserving (1 degree of longitude
shrinks by cos(latitude) toward the poles, which is significant across
Indonesia's ~35 degrees of longitude span at low latitude). We use
`CRSConfig.projected_crs_proj4` (an azimuthal equidistant projection centered
on the archipelago) via `pyproj` for exact projected (x, y) meters, and the
haversine formula for fast great-circle distance where pairwise projection is
unnecessary (e.g. kNN spatial queries).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pyproj import Transformer

from ..config import CRSConfig

_EARTH_R_KM = 6371.0088  # IUGG mean Earth radius


def get_transformer(crs_cfg: CRSConfig) -> Transformer:
    return Transformer.from_crs(crs_cfg.geographic_crs, crs_cfg.projected_crs_proj4, always_xy=True)


def to_projected(lon: np.ndarray, lat: np.ndarray, crs_cfg: CRSConfig) -> tuple:
    transformer = get_transformer(crs_cfg)
    x, y = transformer.transform(lon, lat)
    return np.asarray(x), np.asarray(y)


def haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Vectorized great-circle distance in kilometers.

    d = 2R * asin( sqrt( sin^2(dphi/2) + cos(phi1)cos(phi2)sin^2(dlambda/2) ) )
    """
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.clip(np.sqrt(a), -1, 1))
    return _EARTH_R_KM * c


def bearing_deg(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Initial great-circle bearing (degrees, 0-360) from point 1 to point 2."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlon = lon2 - lon1
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    theta = np.arctan2(x, y)
    return (np.degrees(theta) + 360) % 360


def add_projected_columns(df: pd.DataFrame, crs_cfg: CRSConfig,
                           lat_col: str = "lat", lon_col: str = "lon") -> pd.DataFrame:
    x, y = to_projected(df[lon_col].values, df[lat_col].values, crs_cfg)
    df = df.copy()
    df["proj_x"], df["proj_y"] = x, y
    return df
