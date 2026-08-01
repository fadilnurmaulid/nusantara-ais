"""
Synthetic dataset generator.

Rationale
---------
Global Fishing Watch AIS, BMKG (wave/wind/weather), Indonesian bathymetry,
coastline, port, and BIG National Shipping Lane data are licensed / gated
datasets that cannot be redistributed or fetched inside this sandbox. To keep
the ENTIRE pipeline runnable and testable end-to-end, this module generates
statistically- and physically-realistic synthetic stand-ins with the exact
same schema the real ingestion module expects (see `data/ingestion.py`).

Swapping synthetic sources for the real ones requires touching only
`ingestion.py` (point the loader at the real GFW/BMKG/BIG files) -- every
downstream module (cleaning, feature engineering, ARI, graph construction,
model) is schema-driven and does not know or care whether the data is real.

Vessel behavior classes simulated (ground-truth labels are retained ONLY for
evaluation; the model itself never sees them):
    - normal_transit       : follows shipping lanes, plausible kinematics
    - normal_fishing       : characteristic loitering/zig-zag inside legal zones
    - illegal_fishing       : loitering inside protected/foreign-adjacent waters
    - ais_blackout          : long transmission gap then reappearance
    - ais_spoofing          : implausible position jumps / identity inconsistency
    - loitering_anomaly     : stationary in a non-fishing, non-anchorage area
    - protected_area_intrusion : transits through a marine protected polygon
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd

from ..config import Config


# --------------------------------------------------------------------------- #
# Static context layers
# --------------------------------------------------------------------------- #
def generate_ports(n_ports: int = 12, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    # Rough coastal placement inside the Indonesian bounding box
    lats = rng.uniform(-9.5, 4.5, n_ports)
    lons = rng.uniform(96.0, 139.0, n_ports)
    names = [f"PORT_{i:03d}" for i in range(n_ports)]
    return pd.DataFrame({"port_id": names, "lat": lats, "lon": lons,
                          "capacity_teu": rng.integers(500, 50000, n_ports)})


def generate_shipping_lanes(n_lanes: int = 8, points_per_lane: int = 10, seed: int = 1) -> pd.DataFrame:
    """National Shipping Lane Network: piecewise-linear corridors between port hubs."""
    rng = np.random.default_rng(seed)
    rows = []
    for lane_idx in range(n_lanes):
        lat0, lon0 = rng.uniform(-9.0, 4.0), rng.uniform(97.0, 138.0)
        lat1, lon1 = rng.uniform(-9.0, 4.0), rng.uniform(97.0, 138.0)
        lats = np.linspace(lat0, lat1, points_per_lane) + rng.normal(0, 0.05, points_per_lane)
        lons = np.linspace(lon0, lon1, points_per_lane) + rng.normal(0, 0.05, points_per_lane)
        for seg_idx in range(points_per_lane - 1):
            rows.append({
                "lane_id": f"LANE_{lane_idx:02d}",
                "segment_id": f"LANE_{lane_idx:02d}_SEG_{seg_idx:03d}",
                "lat_start": lats[seg_idx], "lon_start": lons[seg_idx],
                "lat_end": lats[seg_idx + 1], "lon_end": lons[seg_idx + 1],
            })
    return pd.DataFrame(rows)


def generate_coastline(n_points: int = 400, seed: int = 2) -> pd.DataFrame:
    """A stylized closed polyline approximating an archipelagic coastline."""
    rng = np.random.default_rng(seed)
    theta = np.linspace(0, 2 * np.pi, n_points)
    base_lat, base_lon = -2.5, 118.0
    r = 6.0 + 2.0 * np.sin(4 * theta) + rng.normal(0, 0.3, n_points)
    lats = base_lat + r * np.sin(theta) * 0.6
    lons = base_lon + r * np.cos(theta)
    return pd.DataFrame({"lat": lats, "lon": lons, "seq": np.arange(n_points)})


def generate_bathymetry_grid(resolution_deg: float = 0.5, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    lats = np.arange(-11.5, 6.5, resolution_deg)
    lons = np.arange(94.5, 141.5, resolution_deg)
    grid_lat, grid_lon = np.meshgrid(lats, lons)
    # Smooth pseudo-bathymetry: deeper offshore, shallow near a synthetic coastline sinusoid
    coast_lat_at_lon = -2.5 + 3.0 * np.sin((grid_lon - 118.0) / 8.0)
    dist_from_coast = np.abs(grid_lat - coast_lat_at_lon)
    depth = -50 - dist_from_coast * 800 + rng.normal(0, 100, grid_lat.shape)
    depth = np.clip(depth, -6000, -5)
    return pd.DataFrame({
        "lat": grid_lat.ravel(), "lon": grid_lon.ravel(), "depth_m": depth.ravel(),
    })


def generate_protected_areas(n_areas: int = 3, seed: int = 4) -> List[dict]:
    rng = np.random.default_rng(seed)
    areas = []
    for i in range(n_areas):
        lat_c = rng.uniform(-8.0, 3.0)
        lon_c = rng.uniform(98.0, 136.0)
        areas.append({"area_id": f"MPA_{i:02d}", "lat_center": lat_c, "lon_center": lon_c,
                       "radius_km": rng.uniform(20, 60)})
    return areas


def generate_bmkg_weather(n_stations: int = 20, n_days: int = 14, seed: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    lats = rng.uniform(-9.5, 4.5, n_stations)
    lons = rng.uniform(96.0, 139.0, n_stations)
    times = pd.date_range("2024-01-01", periods=n_days * 24, freq="h")
    for s in range(n_stations):
        wind = np.clip(rng.normal(6, 2.5, len(times)), 0, None)
        wave = np.clip(rng.normal(1.2, 0.6, len(times)), 0, None)
        wave_dir = rng.uniform(0, 360, len(times))
        weather_code = rng.integers(0, 4, len(times))  # 0 clear,1 cloudy,2 rain,3 storm
        for t_idx, t in enumerate(times):
            rows.append({
                "station_lat": lats[s], "station_lon": lons[s], "timestamp": t,
                "wind_speed_ms": wind[t_idx], "wave_height_m": wave[t_idx],
                "wave_direction_deg": wave_dir[t_idx], "weather_code": int(weather_code[t_idx]),
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# AIS vessel trajectory simulation
# --------------------------------------------------------------------------- #
_KNOTS_TO_MS = 0.514444
_EARTH_R_KM = 6371.0


def _move(lat, lon, heading_deg, speed_knots, dt_min):
    dist_km = speed_knots * _KNOTS_TO_MS * (dt_min * 60) / 1000.0
    ang_dist = dist_km / _EARTH_R_KM
    hd = np.radians(heading_deg)
    lat1, lon1 = np.radians(lat), np.radians(lon)
    lat2 = np.arcsin(np.sin(lat1) * np.cos(ang_dist) + np.cos(lat1) * np.sin(ang_dist) * np.cos(hd))
    lon2 = lon1 + np.arctan2(np.sin(hd) * np.sin(ang_dist) * np.cos(lat1),
                              np.cos(ang_dist) - np.sin(lat1) * np.sin(lat2))
    return np.degrees(lat2), np.degrees(lon2)


@dataclass
class VesselSimSpec:
    mmsi: int
    behavior: str
    start_lat: float
    start_lon: float
    n_points: int
    dt_min: int = 10


def _simulate_vessel(spec: VesselSimSpec, rng: np.random.Generator, protected_areas: List[dict],
                      ports: pd.DataFrame) -> pd.DataFrame:
    lat, lon = spec.start_lat, spec.start_lon
    heading = rng.uniform(0, 360)
    speed = rng.uniform(6, 14)
    t0 = pd.Timestamp("2024-01-01") + pd.Timedelta(minutes=int(rng.integers(0, 2000)))
    rows = []
    blackout_at = None
    if spec.behavior == "ais_blackout":
        blackout_at = rng.integers(int(spec.n_points * 0.3), int(spec.n_points * 0.6))
    spoof_jump_at = None
    if spec.behavior == "ais_spoofing":
        spoof_jump_at = rng.integers(int(spec.n_points * 0.3), int(spec.n_points * 0.7))

    port_target = ports.sample(1, random_state=int(rng.integers(1e6))).iloc[0]

    for i in range(spec.n_points):
        t = t0 + pd.Timedelta(minutes=i * spec.dt_min)

        if spec.behavior == "ais_blackout" and blackout_at is not None and i == blackout_at:
            gap_steps = rng.integers(6, 18)  # 60-180 min silent gap
            t0 = t0 + pd.Timedelta(minutes=int(gap_steps * spec.dt_min))
            lat, lon = _move(lat, lon, heading, speed, gap_steps * spec.dt_min)

        if spec.behavior == "normal_transit":
            target_heading = (np.degrees(np.arctan2(port_target["lon"] - lon, port_target["lat"] - lat))) % 360
            heading = 0.9 * heading + 0.1 * target_heading + rng.normal(0, 2)
            speed = np.clip(speed + rng.normal(0, 0.3), 4, 18)
        elif spec.behavior in ("normal_fishing", "illegal_fishing"):
            heading = (heading + rng.normal(0, 25)) % 360
            speed = np.clip(rng.normal(2.5, 1.2), 0.2, 6)
        elif spec.behavior in ("loitering_anomaly",):
            heading = (heading + rng.normal(0, 40)) % 360
            speed = np.clip(rng.normal(0.8, 0.5), 0, 2.5)
        elif spec.behavior == "protected_area_intrusion":
            area = protected_areas[int(rng.integers(0, len(protected_areas)))]
            target_heading = (np.degrees(np.arctan2(area["lon_center"] - lon, area["lat_center"] - lat))) % 360
            heading = 0.85 * heading + 0.15 * target_heading + rng.normal(0, 4)
            speed = np.clip(speed + rng.normal(0, 0.4), 4, 14)
        elif spec.behavior == "ais_spoofing":
            heading = (heading + rng.normal(0, 5)) % 360
            speed = np.clip(speed + rng.normal(0, 0.4), 4, 15)
        else:
            heading = (heading + rng.normal(0, 8)) % 360
            speed = np.clip(speed + rng.normal(0, 0.5), 2, 14)

        lat, lon = _move(lat, lon, heading, speed, spec.dt_min)
        lat = float(np.clip(lat, -11.4, 6.4))
        lon = float(np.clip(lon, 94.6, 141.4))

        if spec.behavior == "ais_spoofing" and spoof_jump_at is not None and i == spoof_jump_at:
            # Teleport: physically impossible jump inconsistent with reported SOG
            lat = float(np.clip(lat + rng.normal(0, 2.5), -11.4, 6.4))
            lon = float(np.clip(lon + rng.normal(0, 2.5), 94.6, 141.4))

        reported_sog = speed if spec.behavior != "ais_spoofing" else speed * rng.uniform(0.3, 0.6)
        rows.append({
            "mmsi": spec.mmsi, "timestamp": t, "lat": lat, "lon": lon,
            "sog": max(reported_sog, 0), "cog": heading % 360,
            "heading": (heading + rng.normal(0, 3)) % 360,
            "nav_status": 0 if speed > 1 else 1,
            "vessel_type": rng.choice([30, 37, 70, 80]),  # fishing, pleasure, cargo, tanker (GFW-style codes)
            "length": rng.uniform(15, 120), "width": rng.uniform(4, 20), "draft": rng.uniform(2, 10),
            "behavior_label": spec.behavior,
        })
    return pd.DataFrame(rows)


def generate_ais_fleet(n_vessels: int = 60, points_per_vessel: int = 144, seed: int = 42,
                        protected_areas: List[dict] = None, ports: pd.DataFrame = None) -> pd.DataFrame:
    """Simulate a fleet with a realistic mixture of behaviors.

    Class balance approximates real-world sparsity of anomalies: most traffic
    is legitimate transit/fishing, anomalies are a minority class (as in the
    real maritime-surveillance setting the anomaly score targets).
    """
    rng = np.random.default_rng(seed)
    behaviors = (
        ["normal_transit"] * int(n_vessels * 0.35)
        + ["normal_fishing"] * int(n_vessels * 0.30)
        + ["illegal_fishing"] * int(n_vessels * 0.10)
        + ["ais_blackout"] * int(n_vessels * 0.08)
        + ["ais_spoofing"] * int(n_vessels * 0.07)
        + ["loitering_anomaly"] * int(n_vessels * 0.05)
        + ["protected_area_intrusion"] * int(n_vessels * 0.05)
    )
    while len(behaviors) < n_vessels:
        behaviors.append("normal_transit")
    rng.shuffle(behaviors)

    frames = []
    for i, behavior in enumerate(behaviors[:n_vessels]):
        mmsi = 500_000_000 + i
        start_lat = rng.uniform(-9.0, 4.0)
        start_lon = rng.uniform(97.0, 138.0)
        spec = VesselSimSpec(mmsi=mmsi, behavior=behavior, start_lat=start_lat, start_lon=start_lon,
                              n_points=points_per_vessel)
        frames.append(_simulate_vessel(spec, rng, protected_areas, ports))
    df = pd.concat(frames, ignore_index=True)
    return df.sort_values(["mmsi", "timestamp"]).reset_index(drop=True)


def generate_all(cfg: Config, n_vessels: int = 60, points_per_vessel: int = 144,
                  seed: int = 42) -> dict:
    ports = generate_ports(seed=seed)
    lanes = generate_shipping_lanes(seed=seed + 1)
    coastline = generate_coastline(seed=seed + 2)
    bathy = generate_bathymetry_grid(seed=seed + 3)
    protected_areas = generate_protected_areas(seed=seed + 4)
    weather = generate_bmkg_weather(seed=seed + 5)
    ais = generate_ais_fleet(n_vessels=n_vessels, points_per_vessel=points_per_vessel, seed=seed,
                              protected_areas=protected_areas, ports=ports)

    out_dir = cfg.paths.abs(cfg.paths.synthetic_dir)
    os.makedirs(out_dir, exist_ok=True)
    ais.to_parquet(os.path.join(out_dir, "ais_raw.parquet"))
    ports.to_parquet(os.path.join(out_dir, "ports.parquet"))
    lanes.to_parquet(os.path.join(out_dir, "shipping_lanes.parquet"))
    coastline.to_parquet(os.path.join(out_dir, "coastline.parquet"))
    bathy.to_parquet(os.path.join(out_dir, "bathymetry.parquet"))
    weather.to_parquet(os.path.join(out_dir, "bmkg_weather.parquet"))
    pd.DataFrame(protected_areas).to_parquet(os.path.join(out_dir, "protected_areas.parquet"))

    return {
        "ais": ais, "ports": ports, "lanes": lanes, "coastline": coastline,
        "bathymetry": bathy, "weather": weather, "protected_areas": protected_areas,
    }
