from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

df = pd.read_parquet(
    ROOT/"results"/"ais_scored.parquet"
)

gdf = gpd.GeoDataFrame(
    df,
    geometry=gpd.points_from_xy(
        df.lon,
        df.lat
    ),
    crs="EPSG:4326"
)

fig, ax = plt.subplots(figsize=(12,10))

gdf.plot(
    ax=ax,
    column="anomaly_score",
    markersize=3,
    legend=True,
)

plt.savefig(
    ROOT/"results"/"anomaly_map.png",
    dpi=300
)

print("Saved")