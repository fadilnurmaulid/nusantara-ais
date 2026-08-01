from pathlib import Path
import geopandas as gpd
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

# ==========================
# Load AIS
# ==========================

ais = pd.read_parquet(
    ROOT / "data" / "processed" / "ais_indonesia_clean.parquet"
)

gdf = gpd.GeoDataFrame(
    ais,
    geometry=gpd.points_from_xy(ais.lon, ais.lat),
    crs="EPSG:4326"
)

# ==========================
# Load EEZ
# ==========================

eez = gpd.read_file(
    ROOT / "data" / "raw" / "EEZ" / "eez_v12.shp"
)

# Indonesia saja
eez = eez[eez["ISO_SOV1"] == "IDN"]

print(eez[["GEONAME", "ISO_SOV1"]])

print("Jumlah polygon:", len(eez))