from pathlib import Path

import geopandas as gpd
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

# =====================================================
# PATH
# =====================================================

AIS_PATH = (
    ROOT
    / "data"
    / "processed"
    / "ais_indonesia_clean.parquet"
)

WDPA_PATH = (
    ROOT
    / "data"
    / "processed"
    / "wdpa_indonesia.gpkg"
)

# =====================================================
# LOAD AIS
# =====================================================

print("Loading AIS...")

ais = pd.read_parquet(AIS_PATH)

print("Original rows :", len(ais))

ais = gpd.GeoDataFrame(
    ais,
    geometry=gpd.points_from_xy(
        ais.lon,
        ais.lat
    ),
    crs="EPSG:4326"
)

# =====================================================
# LOAD WDPA
# =====================================================

print("Loading Protected Areas...")

wdpa = gpd.read_file(WDPA_PATH)

# Keep only required columns
wdpa = wdpa[
    [
        "NAME",
        "IUCN_CAT",
        "NO_TAKE",
        "geometry"
    ]
]

# =====================================================
# PROJECT
# =====================================================

print("Projecting to EPSG:3857...")

ais = ais.to_crs(3857)
wdpa = wdpa.to_crs(3857)

# =====================================================
# INTERSECTION
# =====================================================

print("Checking intersection...")

inside = gpd.sjoin(
    ais,
    wdpa,
    how="left",
    predicate="intersects"
)

inside = (
    inside
    .sort_index()
    .drop_duplicates(
        subset=["mmsi", "timestamp", "lat", "lon"],
        keep="first"
    )
)

inside["inside_protected_area"] = (
    inside["index_right"].notna()
)

# =====================================================
# NEAREST PROTECTED AREA
# =====================================================

print("Finding nearest protected area...")

nearest = gpd.sjoin_nearest(
    ais,
    wdpa,
    how="left",
    distance_col="distance_meter"
)

nearest = (
    nearest
    .sort_values("distance_meter")
    .drop_duplicates(
        subset=["mmsi", "timestamp", "lat", "lon"],
        keep="first"
    )
)

nearest = nearest.rename(
    columns={
        "NAME": "protected_name",
        "IUCN_CAT": "protected_category",
        "NO_TAKE": "protected_no_take"
    }
)

nearest["distance_to_protected_km"] = (
    nearest["distance_meter"] / 1000
)

# =====================================================
# MERGE FEATURES
# =====================================================

nearest = nearest.drop(
    columns=[
        "index_right",
        "distance_meter"
    ],
    errors="ignore"
)

inside = inside[
    [
        "mmsi",
        "timestamp",
        "lat",
        "lon",
        "inside_protected_area"
    ]
]

joined = nearest.merge(
    inside,
    on=[
        "mmsi",
        "timestamp",
        "lat",
        "lon"
    ],
    how="left"
)

joined["inside_protected_area"] = (
    joined["inside_protected_area"]
    .fillna(False)
)

# =====================================================
# CLEAN
# =====================================================

joined = joined.to_crs(4326)

joined.to_parquet(
    AIS_PATH,
    index=False
)

# =====================================================
# REPORT
# =====================================================

print("\n========== RESULT ==========")

print("Rows :", len(joined))

print("\nInside Protected Area")

print(
    joined["inside_protected_area"]
    .value_counts()
)

print("\nDistance to Protected Area (km)")

print(
    joined["distance_to_protected_km"]
    .describe()
)

print("\nTop Protected Areas")

print(
    joined["protected_name"]
    .value_counts()
    .head(20)
)

print("\nSaved to")

print(AIS_PATH)