from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

AIS_PATH = (
    ROOT
    / "data"
    / "processed"
    / "ais_indonesia_clean.parquet"
)

print("Loading AIS...")

df = pd.read_parquet(AIS_PATH)

print("Rows :", len(df))

# =====================================================
# SORT
# =====================================================

print("Sorting...")

df = df.sort_values(
    ["mmsi", "timestamp"]
).reset_index(drop=True)

# =====================================================
# TIMESTAMP
# =====================================================

print("Converting timestamp...")

df["timestamp"] = pd.to_datetime(df["timestamp"])

# =====================================================
# BASIC TEMPORAL FEATURES
# =====================================================

print("Extract temporal features...")

df["year"] = df.timestamp.dt.year.astype("int16")

df["month"] = df.timestamp.dt.month.astype("int8")

df["day"] = df.timestamp.dt.day.astype("int8")

df["hour"] = df.timestamp.dt.hour.astype("int8")

df["minute"] = df.timestamp.dt.minute.astype("int8")

df["day_of_week"] = df.timestamp.dt.dayofweek.astype("int8")

df["is_weekend"] = (
    df.day_of_week >= 5
).astype("int8")

# =====================================================
# TIME GAP
# =====================================================

print("Computing time gap...")

df["time_gap_second"] = (
    df
    .groupby("mmsi")["timestamp"]
    .diff()
    .dt.total_seconds()
)

df["time_gap_second"] = (
    df["time_gap_second"]
    .fillna(0)
    .astype("float32")
)

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

print()

print(df[[
    "year",
    "month",
    "day",
    "hour",
    "minute",
    "day_of_week",
    "is_weekend",
    "time_gap_second"
]].head())

print()

print(df["time_gap_second"].describe())

print()

print("Weekend")

print(
    df["is_weekend"].value_counts()
)

print()

print("Hour distribution")

print(
    df["hour"].value_counts().sort_index()
)

print()

print("Saved")

print(AIS_PATH)