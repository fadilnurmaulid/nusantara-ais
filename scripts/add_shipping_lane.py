from pathlib import Path

import geopandas as gpd
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

# =====================================================
# PATH
# =====================================================

AIS_PATH = ROOT / "data" / "processed" / "ais_indonesia_clean.parquet"

LANE_PATH = (
    ROOT
    / "data"
    / "raw"
    / "ShippingLane"
    / "APNPR.shp"
)

# =====================================================
# LOAD
# =====================================================

print("Loading AIS...")

df = pd.read_parquet(AIS_PATH)

print(f"Original rows : {len(df)}")

# ID unik setiap AIS point
df = df.reset_index(drop=True)
df["ais_id"] = df.index

gdf = gpd.GeoDataFrame(
    df,
    geometry=gpd.points_from_xy(df.lon, df.lat),
    crs="EPSG:4326"
)

print("Loading Shipping Lane...")

lanes = gpd.read_file(LANE_PATH)

# =====================================================
# PROJECT TO METERS
# =====================================================

print("Projecting to EPSG:3857...")

gdf = gdf.to_crs(3857)
lanes = lanes.to_crs(3857)

# =====================================================
# NEAREST LANE
# =====================================================

print("Finding nearest shipping lane...")

joined = gpd.sjoin_nearest(
    gdf,
    lanes[["geometry"]],
    how="left",
    distance_col="distance_to_lane_m"
)

print(f"Rows after join : {len(joined)}")

# =====================================================
# KEEP CLOSEST MATCH
# =====================================================

joined = (
    joined
    .sort_values("distance_to_lane_m")
    .drop_duplicates(subset="ais_id", keep="first")
    .sort_values("ais_id")
)

print(f"Rows after dedup : {len(joined)}")

# =====================================================
# FEATURES
# =====================================================

joined["distance_to_lane_km"] = (
    joined["distance_to_lane_m"] / 1000.0
)

joined["inside_shipping_lane"] = (
    joined["distance_to_lane_km"] <= 5.0
)

# =====================================================
# CLEAN
# =====================================================

drop_cols = [
    "ais_id",
    "index_right",
    "distance_to_lane_m",
]

joined.drop(
    columns=[c for c in drop_cols if c in joined.columns],
    inplace=True
)

joined = joined.to_crs(4326)

joined.to_parquet(
    AIS_PATH,
    index=False
)

# =====================================================
# REPORT
# =====================================================

print("\n========== RESULT ==========")

print(f"Final rows : {len(joined)}")

print("\nDistance to shipping lane (km)")
print(joined["distance_to_lane_km"].describe())

print("\nInside shipping lane")
print(joined["inside_shipping_lane"].value_counts())

print("\nFive nearest distances")
print(
    joined["distance_to_lane_km"]
    .nsmallest(5)
)

print("\nSaved to")
print(AIS_PATH)