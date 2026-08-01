"""
Spatial joins.

Implements the geospatial "distance-to" and "containment" primitives shared
by feature engineering and graph construction:

  - nearest_port / distance_to_port_km            (BallTree, haversine metric)
  - nearest_shipping_lane_segment / distance_to_shipping_lane_km
        (point-to-segment great-circle distance, vectorized)
  - distance_to_coastline_km                       (BallTree over coastline vertices)
  - bathymetry_m                                   (nearest bathymetry grid cell)
  - inside_eez / inside_protected_area             (haversine radius test; the
        protected areas are modeled as circles which is a reasonable and fast
        approximation given MPA boundaries are not sharply enforced physical
        edges but buffer zones)

BallTree with the haversine metric gives O(log n) nearest-neighbor queries
directly on (lat, lon) in radians without needing a projected CRS for this
step, which keeps this module dependency-light and fast even for hundreds of
thousands of AIS points against a modest number of ports/lane vertices.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

from .coordinate import haversine_km

_EARTH_R_KM = 6371.0088


def _build_balltree(lat: np.ndarray, lon: np.ndarray) -> BallTree:
    coords_rad = np.radians(np.column_stack([lat, lon]))
    return BallTree(coords_rad, metric="haversine")


def nearest_port(df: pd.DataFrame, ports: pd.DataFrame) -> pd.DataFrame:
    tree = _build_balltree(ports["lat"].values, ports["lon"].values)
    query = np.radians(df[["lat", "lon"]].values)
    dist_rad, idx = tree.query(query, k=1)
    df = df.copy()
    df["distance_to_port_km"] = dist_rad[:, 0] * _EARTH_R_KM
    df["nearest_port_id"] = ports["port_id"].values[idx[:, 0]]
    return df


def nearest_coastline(df: pd.DataFrame, coastline: pd.DataFrame) -> pd.DataFrame:
    tree = _build_balltree(coastline["lat"].values, coastline["lon"].values)
    query = np.radians(df[["lat", "lon"]].values)
    dist_rad, _ = tree.query(query, k=1)
    df = df.copy()
    df["distance_to_coastline_km"] = dist_rad[:, 0] * _EARTH_R_KM
    return df


def _point_segment_distance_km(p_lat, p_lon, a_lat, a_lon, b_lat, b_lon) -> np.ndarray:
    """Approximate point-to-great-circle-segment distance by projecting to a
    local equirectangular plane (accurate for segment lengths of tens of km,
    which National Shipping Lane segments are by construction)."""
    lat0 = (a_lat + b_lat) / 2.0
    scale = np.cos(np.radians(lat0))

    def to_xy(lat, lon):
        return (lon * scale, lat)

    px, py = to_xy(p_lat, p_lon)
    ax, ay = to_xy(a_lat, a_lon)
    bx, by = to_xy(b_lat, b_lon)

    abx, aby = bx - ax, by - ay
    seg_len_sq = abx ** 2 + aby ** 2
    seg_len_sq = np.where(seg_len_sq == 0, 1e-12, seg_len_sq)
    t = ((px - ax) * abx + (py - ay) * aby) / seg_len_sq
    t = np.clip(t, 0.0, 1.0)
    proj_x = ax + t * abx
    proj_y = ay + t * aby
    dx_deg = px - proj_x
    dy_deg = py - proj_y
    dist_deg = np.sqrt(dx_deg ** 2 + dy_deg ** 2)
    km_per_deg = 111.32
    return dist_deg * km_per_deg


def nearest_shipping_lane(df: pd.DataFrame, lanes: pd.DataFrame, candidate_k: int = 5) -> pd.DataFrame:
    """Find nearest National Shipping Lane segment.

    Strategy: first use a BallTree over segment MIDPOINTS to shortlist the
    `candidate_k` nearest segments per point (fast), then compute exact
    point-to-segment distance only against that shortlist (accurate),
    avoiding an O(n_points * n_segments) brute force pass.
    """
    mid_lat = (lanes["lat_start"] + lanes["lat_end"]) / 2.0
    mid_lon = (lanes["lon_start"] + lanes["lon_end"]) / 2.0
    tree = _build_balltree(mid_lat.values, mid_lon.values)
    query = np.radians(df[["lat", "lon"]].values)
    k = min(candidate_k, len(lanes))
    _, idx = tree.query(query, k=k)

    best_dist = np.full(len(df), np.inf)
    best_seg = np.empty(len(df), dtype=object)

    p_lat = df["lat"].values
    p_lon = df["lon"].values
    seg_ids = lanes["segment_id"].values
    a_lat_all, a_lon_all = lanes["lat_start"].values, lanes["lon_start"].values
    b_lat_all, b_lon_all = lanes["lat_end"].values, lanes["lon_end"].values

    for c in range(k):
        seg_idx = idx[:, c]
        d = _point_segment_distance_km(
            p_lat, p_lon, a_lat_all[seg_idx], a_lon_all[seg_idx], b_lat_all[seg_idx], b_lon_all[seg_idx]
        )
        better = d < best_dist
        best_dist = np.where(better, d, best_dist)
        best_seg = np.where(better, seg_ids[seg_idx], best_seg)

    df = df.copy()
    df["distance_to_shipping_lane_km"] = best_dist
    df["nearest_shipping_lane_id"] = best_seg
    return df


def bathymetry_lookup(df: pd.DataFrame, bathy: pd.DataFrame) -> pd.DataFrame:
    tree = _build_balltree(bathy["lat"].values, bathy["lon"].values)
    query = np.radians(df[["lat", "lon"]].values)
    _, idx = tree.query(query, k=1)
    df = df.copy()
    df["bathymetry_m"] = bathy["depth_m"].values[idx[:, 0]]
    return df


def inside_eez(df: pd.DataFrame, eez_bbox: tuple) -> pd.DataFrame:
    """Fallback EEZ containment: bounding-box test used when only a coarse
    EEZ extent is available (this dataset is marked optional in the spec)."""
    lat_lo, lat_hi, lon_lo, lon_hi = eez_bbox
    df = df.copy()
    df["inside_eez"] = (
        df["lat"].between(lat_lo, lat_hi) & df["lon"].between(lon_lo, lon_hi)
    )
    return df


def inside_protected_area(df: pd.DataFrame, protected_areas: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    inside = np.zeros(len(df), dtype=bool)
    nearest_area_id = np.empty(len(df), dtype=object)
    min_dist = np.full(len(df), np.inf)
    for _, area in protected_areas.iterrows():
        d = haversine_km(df["lat"].values, df["lon"].values, area["lat_center"], area["lon_center"])
        inside |= d <= area["radius_km"]
        better = d < min_dist
        min_dist = np.where(better, d, min_dist)
        nearest_area_id = np.where(better, area["area_id"], nearest_area_id)
    df["inside_protected_area"] = inside
    df["nearest_protected_area_id"] = nearest_area_id
    df["distance_to_protected_area_km"] = min_dist
    return df


def join_environmental(df: pd.DataFrame, weather: pd.DataFrame, time_col: str = "timestamp",
                        max_time_diff_h: int = 3) -> pd.DataFrame:
    """Join nearest BMKG station (spatially) at the nearest available
    timestamp (temporally, within `max_time_diff_h`)."""
    stations = weather[["station_lat", "station_lon"]].drop_duplicates().reset_index(drop=True)
    tree = _build_balltree(stations["station_lat"].values, stations["station_lon"].values)
    query = np.radians(df[["lat", "lon"]].values)
    _, idx = tree.query(query, k=1)
    df = df.copy()
    df["_station_lat"] = stations["station_lat"].values[idx[:, 0]]
    df["_station_lon"] = stations["station_lon"].values[idx[:, 0]]

    weather_sorted = weather.sort_values(time_col)
    merged_rows = []
    for (s_lat, s_lon), group in df.groupby(["_station_lat", "_station_lon"]):
        station_weather = weather_sorted[
            (weather_sorted["station_lat"] == s_lat) & (weather_sorted["station_lon"] == s_lon)
        ]
        merged = pd.merge_asof(
            group.sort_values(time_col), station_weather.sort_values(time_col),
            on=time_col, direction="nearest",
            tolerance=pd.Timedelta(hours=max_time_diff_h),
            suffixes=("", "_env"),
        )
        merged_rows.append(merged)
    result = pd.concat(merged_rows, ignore_index=True)
    for col in ("wind_speed_ms", "wave_height_m", "wave_direction_deg", "weather_code"):
        if col not in result.columns:
            result[col] = np.nan
    return result.drop(columns=["_station_lat", "_station_lon"], errors="ignore")
