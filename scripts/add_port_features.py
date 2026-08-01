from pathlib import Path

import geopandas as gpd
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

# =====================================================
# LOAD AIS
# =====================================================

ais = pd.read_parquet(
    ROOT / "data" / "processed" / "ais_indonesia_clean.parquet"
)

ais = gpd.GeoDataFrame(
    ais,
    geometry=gpd.points_from_xy(ais.lon, ais.lat),
    crs="EPSG:4326"
)

# =====================================================
# LOAD PORT
# =====================================================

ports = gpd.read_file(
    ROOT / "data" / "processed" / "ports_indonesia.gpkg"
)

ports = ports.to_crs(3857)
ais = ais.to_crs(3857)

# =====================================================
# NEAREST PORT
# =====================================================

joined = gpd.sjoin_nearest(
    ais,
    ports[["PORT_NAME", "geometry"]],
    how="left",
    distance_col="distance_meter"
)

# =====================================================
# FEATURE
# =====================================================

joined["distance_to_port_km"] = joined["distance_meter"] / 1000

joined = joined.rename(
    columns={
        "PORT_NAME": "nearest_port"
    }
)

joined = joined.drop(
    columns=["index_right", "distance_meter"]
)

joined = joined.to_crs(4326)

# =====================================================
# SAVE
# =====================================================

joined.to_parquet(
    ROOT / "data" / "processed" / "ais_indonesia_clean.parquet",
    index=False
)

print(joined[[
    "nearest_port",
    "distance_to_port_km"
]].head())

print()
print("DONE")
print("Rows :", len(joined))