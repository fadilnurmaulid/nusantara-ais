"""
NUSANTARA-AIS :: Configuration system.

Design rationale
----------------
A single dataclass tree is used instead of a raw dict so that (a) every field
is typed and IDE/static-analysis friendly, (b) defaults are defined exactly
once and are guaranteed consistent across the ingestion, feature-engineering,
graph-construction, model and training modules, and (c) experiments are fully
reproducible: a run's exact configuration can be serialized to YAML and
re-loaded byte-for-byte later (required for the reproducibility contribution
of the paper).

All CRS, projection, physical and hyper-parameter constants that appear
throughout the pipeline are declared here ONCE and imported everywhere else,
so there is a single source of truth.
"""
from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import yaml


# --------------------------------------------------------------------------- #
# Path configuration
# --------------------------------------------------------------------------- #
@dataclass
class PathConfig:
    project_root: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    synthetic_dir: str = "data/synthetic"
    checkpoint_dir: str = "outputs/checkpoints"
    figures_dir: str = "outputs/figures"
    logs_dir: str = "outputs/logs"
    reports_dir: str = "outputs/reports"

    def abs(self, rel: str) -> str:
        return os.path.join(self.project_root, rel)


# --------------------------------------------------------------------------- #
# Coordinate reference system configuration
# --------------------------------------------------------------------------- #
@dataclass
class CRSConfig:
    # WGS84 geographic (input AIS lat/lon)
    geographic_crs: str = "EPSG:4326"
    # Indonesia falls across UTM zones 46N-54S. We reproject to the national
    # equal-area/equidistant projection used for area & distance-preserving
    # computation across the whole archipelago instead of a single UTM zone
    # (which would distort distances far from its central meridian).
    # World Azimuthal Equidistant centered on the Indonesian archipelago
    # centroid gives centimeter-consistent distance behaviour across the
    # entire EEZ, which is essential for distance_to_* features.
    projected_crs_proj4: str = (
        "+proj=aeqd +lat_0=-2.5 +lon_0=118.0 +x_0=0 +y_0=0 "
        "+ellps=WGS84 +units=m +no_defs"
    )
    earth_radius_m: float = 6_371_000.0


# --------------------------------------------------------------------------- #
# Data ingestion / cleaning configuration
# --------------------------------------------------------------------------- #
@dataclass
class IngestionConfig:
    # Global Fishing Watch AIS field schema (post-normalization column names)
    ais_columns: List[str] = field(default_factory=lambda: [
        "mmsi", "timestamp", "lat", "lon", "sog", "cog", "heading",
        "nav_status", "vessel_type", "length", "width", "draft",
    ])
    time_col: str = "timestamp"
    id_col: str = "mmsi"

    # Physical plausibility bounds used for hard-rejection cleaning
    lat_bounds: tuple = (-11.5, 6.5)     # Indonesian EEZ approx bounding box
    lon_bounds: tuple = (94.5, 141.5)
    max_sog_knots: float = 60.0           # implausible speed above this (physics-informed ARI reuses this)
    max_accel_mps2: float = 1.5           # implausible acceleration for a vessel
    max_turn_rate_deg_s: float = 15.0     # implausible turning rate

    # Temporal alignment
    resample_interval_min: int = 10       # AIS messages are resampled/interpolated to a fixed cadence
    max_interp_gap_min: int = 60          # do not interpolate across gaps longer than this (treat as blackout)
    blackout_threshold_min: int = 30      # gap beyond this is flagged as an AIS blackout event

    # Trajectory segmentation
    segment_gap_threshold_min: int = 180  # split a trajectory into a new segment after this silence
    min_segment_points: int = 5


# --------------------------------------------------------------------------- #
# AIS Reliability Index (ARI) configuration
# --------------------------------------------------------------------------- #
@dataclass
class ARIConfig:
    # Weighted linear-in-log-odds combination of reliability sub-scores.
    # Weights are chosen so that spoofing / kinematic implausibility dominate
    # (they are the strongest physical evidence of unreliable AIS) while gap
    # ratio and coordinate-jump contribute secondary evidence. Weights sum to 1.
    w_kinematic: float = 0.30      # acceleration / turn-rate plausibility
    w_jump: float = 0.20           # coordinate jump (implied speed vs reported speed)
    w_gap: float = 0.20            # AIS gap ratio
    w_blackout: float = 0.15       # blackout ratio
    w_spoof: float = 0.15          # spoofing indicator (identity/position consistency)

    # Sigmoid steepness controlling how sharply a sub-score saturates to 0/1
    sigmoid_k_kinematic: float = 4.0
    sigmoid_k_jump: float = 3.0

    # Reference max implied speed multiplier: an implied speed more than this
    # multiple of the max plausible vessel speed is treated as a hard jump
    jump_speed_multiplier: float = 1.5


# --------------------------------------------------------------------------- #
# Feature engineering configuration
# --------------------------------------------------------------------------- #
@dataclass
class FeatureConfig:
    dynamic_features: List[str] = field(default_factory=lambda: [
        "sog", "cog_sin", "cog_cos", "heading_sin", "heading_cos",
        "acceleration", "turning_rate", "angular_velocity",
        "ais_gap_min", "blackout_duration_min",
    ])
    spatial_features: List[str] = field(default_factory=lambda: [
        "distance_to_port_km", "distance_to_shipping_lane_km",
        "distance_to_coastline_km", "bathymetry_m", "inside_eez",
        "nearest_port_id", "nearest_shipping_lane_id",
    ])
    environmental_features: List[str] = field(default_factory=lambda: [
        "wind_speed_ms", "wave_height_m", "wave_direction_deg", "weather_code",
    ])
    reliability_features: List[str] = field(default_factory=lambda: [
        "coordinate_jump_score", "heading_instability_score",
        "acceleration_anomaly_score", "ais_gap_ratio", "blackout_ratio",
        "spoofing_indicator", "ari",
    ])
    historical_features: List[str] = field(default_factory=lambda: [
        "revisit_frequency", "anchoring_duration_min", "stop_frequency",
        "avg_speed_hist", "avg_heading_hist", "trip_duration_min",
    ])
    # Auto-proposed additional features (documented in README, sec. "Extra Features")
    extra_features: List[str] = field(default_factory=lambda: [
        "loitering_index", "eez_boundary_proximity_km",
        "dark_activity_ratio", "speed_variance_hist",
        "course_over_ground_entropy", "port_dwell_ratio",
        "time_of_day_sin", "time_of_day_cos", "day_of_week_sin", "day_of_week_cos",
    ])
    stop_speed_threshold_knots: float = 0.5
    anchoring_min_duration_min: int = 60
    loitering_radius_km: float = 3.0
    loitering_min_duration_min: int = 120
    history_window_days: int = 30


# --------------------------------------------------------------------------- #
# Graph construction configuration
# --------------------------------------------------------------------------- #
@dataclass
class GraphConfig:
    node_types: List[str] = field(default_factory=lambda: [
        "vessel_state", "port", "shipping_lane_segment", "grid_cell",
    ])
    edge_types: List[tuple] = field(default_factory=lambda: [
        ("vessel_state", "temporal_next", "vessel_state"),
        ("vessel_state", "spatial_proximity", "vessel_state"),
        ("vessel_state", "near_port", "port"),
        ("vessel_state", "on_lane", "shipping_lane_segment"),
        ("vessel_state", "located_in", "grid_cell"),
        ("grid_cell", "adjacent", "grid_cell"),
        ("port", "connected_by_lane", "shipping_lane_segment"),
    ])
    # Spatial proximity edge radius (km) for vessel_state-vessel_state graph
    spatial_proximity_radius_km: float = 5.0
    max_spatial_neighbors: int = 10
    # Snapshot / temporal windowing
    snapshot_interval_min: int = 60
    snapshot_window_min: int = 360        # look-back window aggregated into one snapshot
    grid_cell_size_deg: float = 0.25      # ~27km equatorial grid for the grid_cell node type
    # kNN search backend
    neighbor_search_backend: str = "balltree"  # haversine-metric BallTree, O(log n) query


# --------------------------------------------------------------------------- #
# Model hyper-parameters
# --------------------------------------------------------------------------- #
@dataclass
class ModelConfig:
    hidden_dim: int = 128
    hetgat_layers: int = 2
    hetgat_heads: int = 4
    hetgat_dropout: float = 0.2
    gae_latent_dim: int = 32
    gae_decoder_hidden: int = 64
    negative_sampling_ratio: float = 1.0
    recon_loss_weight: float = 1.0
    structure_loss_weight: float = 0.5
    ari_loss_weight: float = 0.3
    activation: str = "elu"


# --------------------------------------------------------------------------- #
# Training configuration
# --------------------------------------------------------------------------- #
@dataclass
class TrainingConfig:
    epochs: int = 100
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-5
    early_stopping_patience: int = 15
    lr_scheduler: str = "cosine"
    grad_clip_norm: float = 5.0
    seed: int = 42
    num_workers: int = 2
    device: str = "cuda_if_available"
    val_split: float = 0.15
    test_split: float = 0.15
    amp: bool = True                       # mixed precision on GPU


@dataclass
class Config:
    paths: PathConfig = field(default_factory=PathConfig)
    crs: CRSConfig = field(default_factory=CRSConfig)
    ingestion: IngestionConfig = field(default_factory=IngestionConfig)
    ari: ARIConfig = field(default_factory=ARIConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    experiment_name: str = "nusantara_ais_baseline"

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)

    @classmethod
    def load(cls, path: str) -> "Config":
        with open(path, "r") as f:
            raw = yaml.safe_load(f)
        return cls(
            paths=PathConfig(**raw.get("paths", {})),
            crs=CRSConfig(**raw.get("crs", {})),
            ingestion=IngestionConfig(**raw.get("ingestion", {})),
            ari=ARIConfig(**raw.get("ari", {})),
            features=FeatureConfig(**raw.get("features", {})),
            graph=GraphConfig(**raw.get("graph", {})),
            model=ModelConfig(**raw.get("model", {})),
            training=TrainingConfig(**raw.get("training", {})),
            experiment_name=raw.get("experiment_name", "nusantara_ais_baseline"),
        )


def get_default_config() -> Config:
    return Config()
