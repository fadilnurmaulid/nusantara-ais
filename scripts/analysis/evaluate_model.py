from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

df = pd.read_parquet(
    ROOT/"results"/"ais_scored.parquet"
)

print()

print("========== MODEL ==========")

print()

print("Nodes :", len(df))

print(
    "Extreme :",
    (df.risk_level=="Extreme").sum()
)

print(
    "High :",
    (df.risk_level=="High").sum()
)

print(
    "Moderate :",
    (df.risk_level=="Moderate").sum()
)

print(
    "Normal :",
    (df.risk_level=="Normal").sum()
)

print()

print(df.anomaly_score.describe())