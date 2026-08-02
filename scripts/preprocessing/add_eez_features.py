from pathlib import Path

import geopandas as gpd
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

ais_path = ROOT / "data/processed/ais_indonesia_clean.parquet"

eez_path = (
    ROOT
    / "data/raw/EEZ"
    / "eez_v12.shp"
)

print("Load AIS...")
df = pd.read_parquet(ais_path)

gdf = gpd.GeoDataFrame(
    df,
    geometry=gpd.points_from_xy(df.lon, df.lat),
    crs="EPSG:4326"
)

print("Load EEZ...")
eez = gpd.read_file(eez_path)

# hanya EEZ Indonesia
eez = eez[
    eez["ISO_SOV1"] == "IDN"
][["GEONAME", "geometry"]]

print("Spatial Join...")

joined = gpd.sjoin(
    gdf,
    eez,
    predicate="within",
    how="left"
)

joined["inside_indonesia_eez"] = (
    joined["GEONAME"].notna()
)

joined.rename(
    columns={
        "GEONAME": "eez_name"
    },
    inplace=True
)

joined.drop(
    columns=["index_right"],
    inplace=True
)

joined.to_parquet(
    ais_path,
    index=False
)

print(joined[
    [
        "eez_name",
        "inside_indonesia_eez"
    ]
].head())

print(joined["inside_indonesia_eez"].value_counts())