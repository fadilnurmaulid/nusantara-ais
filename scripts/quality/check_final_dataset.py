from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

DATA = (
    ROOT
    / "data"
    / "processed"
    / "ais_indonesia_clean.parquet"
)

print("Loading dataset...")

df = pd.read_parquet(DATA)

print("\n==============================")
print("SHAPE")
print("==============================")

print(df.shape)

print("\n==============================")
print("COLUMN")
print("==============================")

for c in df.columns:
    print(c)

print("\n==============================")
print("MISSING")
print("==============================")

print(df.isna().sum())

print("\n==============================")
print("DUPLICATE")
print("==============================")

print(df.duplicated().sum())

print("\n==============================")
print("INFINITE")
print("==============================")

numeric = df.select_dtypes(include=np.number)

print(np.isinf(numeric).sum())

print("\n==============================")
print("NUMERIC SUMMARY")
print("==============================")

print(numeric.describe().T)

print("\n==============================")
print("MMSI")
print("==============================")

print(df.mmsi.nunique())

print("\n==============================")
print("TRIP")
print("==============================")

print(df.trip_id.nunique())

print("\n==============================")
print("EEZ")
print("==============================")

print(df.inside_indonesia_eez.value_counts())

print("\n==============================")
print("LANE")
print("==============================")

print(df.inside_shipping_lane.value_counts())

print("\n==============================")
print("PROTECTED")
print("==============================")

print(df.inside_protected_area.value_counts())

print("\n==============================")
print("STOP")
print("==============================")

print(df.is_stop.value_counts())

print("\n==============================")
print("LABEL")
print("==============================")

print(df.is_fishing.value_counts())

print("\n==============================")
print("PORT")
print("==============================")

print(df.nearest_port.value_counts().head(20))