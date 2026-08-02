from pathlib import Path
import pandas as pd
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

DATA = ROOT/"results"/"ais_scored.parquet"

print("Loading dataset...")

df = pd.read_parquet(DATA)

from src.nusantara_ais.analysis.evaluate import *

print()

print(anomaly_summary(df))

print()

risk = categorize(df)

print()

print(
    risk["risk_level"].value_counts()
)

risk.to_parquet(
    ROOT/"results"/"ais_scored.parquet",
    index=False
)

print()

print("Saved")