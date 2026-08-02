from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

df = pd.read_parquet(
    ROOT / "results" / "ais_scored.parquet"
)

# hanya anomaly tinggi
hotspot = df[
    df["risk_level"].isin(
        ["High", "Extreme"]
    )
]

print()

print("Hotspot points")
print(len(hotspot))

print()

print("EEZ")

print(
    hotspot["eez_name"]
    .value_counts()
    .head(20)
)

print()

print("Nearest Port")

print(
    hotspot["nearest_port"]
    .value_counts()
    .head(20)
)

print()

print("Protected")

print(
    hotspot["protected_name"]
    .value_counts()
    .head(20)
)

hotspot.to_csv(
    ROOT / "results" / "hotspot.csv",
    index=False
)

print()

print("Saved")