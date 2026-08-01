from pathlib import Path

import pandas as pd
import geopandas as gpd
import rasterio

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

BATH_PATH = (
    ROOT
    / "data"
    / "raw"
    / "AIS"
    / "bathymetry.tif"
)

# =====================================================
# LOAD AIS
# =====================================================

print("Loading AIS...")

df = pd.read_parquet(AIS_PATH)

print("Rows :", len(df))

# =====================================================
# OPEN RASTER
# =====================================================

print("Loading Bathymetry...")

src = rasterio.open(BATH_PATH)

# =====================================================
# SAMPLE DEPTH
# =====================================================

print("Sampling bathymetry...")

coords = list(zip(df.lon, df.lat))

depth = []

for value in src.sample(coords):
    depth.append(float(value[0]))

df["bathymetry_m"] = depth

# =====================================================
# INVALID VALUE
# =====================================================

nodata = src.nodata

if nodata is not None:
    df.loc[df["bathymetry_m"] == nodata, "bathymetry_m"] = pd.NA

# =====================================================
# SAVE
# =====================================================

df.to_parquet(
    AIS_PATH,
    index=False
)

# =====================================================
# REPORT
# =====================================================

print("\n========== RESULT ==========")

print(df["bathymetry_m"].describe())

print()

print("Missing :",
      df["bathymetry_m"].isna().sum())

print()

print("Five deepest")

print(
    df["bathymetry_m"]
    .sort_values()
    .head()
)

print()

print("Five shallowest")

print(
    df["bathymetry_m"]
    .sort_values(ascending=False)
    .head()
)

print()

print("Saved to")

print(AIS_PATH)