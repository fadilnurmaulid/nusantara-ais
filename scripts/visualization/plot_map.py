from pathlib import Path
import sys

import matplotlib.pyplot as plt
import geopandas as gpd
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

# ==========================================
# LOAD DATA
# ==========================================

print("Loading dataset...")

df = pd.read_parquet(
    ROOT / "results" / "ais_scored.parquet"
)

coast = gpd.read_file(
    ROOT / "data" / "raw" / "Coastline" / "ne_10m_coastline.shp"
)

# ==========================================
# HOTSPOT
# ==========================================

hotspot = df[
    df["risk_level"].isin(
        [
            "High",
            "Extreme",
        ]
    )
]

print()

print("Hotspot :", len(hotspot))

# ==========================================
# FIGURE
# ==========================================

fig, ax = plt.subplots(
    figsize=(12, 8)
)

# ==========================================
# INDONESIA COASTLINE
# ==========================================

coast.plot(
    ax=ax,
    color="black",
    linewidth=0.5,
)

# ==========================================
# ALL AIS POINT
# ==========================================

ax.scatter(
    df["lon"],
    df["lat"],
    s=2,
    color="lightgray",
    alpha=0.25,
    label="AIS Trajectory",
)

# ==========================================
# HOTSPOT
# ==========================================

scatter = ax.scatter(
    hotspot["lon"],
    hotspot["lat"],
    c=hotspot["anomaly_score"],
    cmap="Reds",
    s=70,
    edgecolors="black",
    linewidths=0.4,
    label="High-risk Anomaly",
    zorder=5,
)

# ==========================================
# COLORBAR
# ==========================================

cbar = plt.colorbar(
    scatter,
    ax=ax,
)

cbar.set_label(
    "Anomaly Score"
)

# ==========================================
# MAP LIMIT
# ==========================================

ax.set_xlim(94, 142)
ax.set_ylim(-12, 8)

# ==========================================
# GRID
# ==========================================

ax.grid(
    alpha=0.3,
    linestyle="--",
)

# ==========================================
# LABEL
# ==========================================

ax.set_xlabel(
    "Longitude (°)"
)

ax.set_ylabel(
    "Latitude (°)"
)

ax.set_title(
    "Spatial Distribution of High-Risk Maritime Anomalies"
)

ax.legend(
    loc="lower left"
)

plt.tight_layout()

# ==========================================
# SAVE
# ==========================================

OUT = ROOT / "results" / "anomaly_map.png"

plt.savefig(
    OUT,
    dpi=300,
    bbox_inches="tight",
)

plt.close()

print()
print("Saved:", OUT)