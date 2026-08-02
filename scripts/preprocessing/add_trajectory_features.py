from pathlib import Path

import numpy as np
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

df = df.sort_values(
    ["mmsi", "timestamp"]
).reset_index(drop=True)

# =====================================================
# PREVIOUS VALUES
# =====================================================

print("Previous position...")

g = df.groupby("mmsi")

df["prev_lat"] = g["lat"].shift(1)

df["prev_lon"] = g["lon"].shift(1)

df["previous_speed"] = g["speed"].shift(1)

df["previous_course"] = g["course"].shift(1)

# =====================================================
# ACCELERATION
# =====================================================

print("Acceleration...")

dt = df["time_gap_second"].replace(0, np.nan)

df["acceleration"] = (
    (df["speed"] - df["previous_speed"])
    / dt
)

df["acceleration"] = (
    df["acceleration"]
    .fillna(0)
    .astype("float32")
)

# =====================================================
# COURSE CHANGE
# =====================================================

print("Course change...")

df["course_change"] = (
    (df["course"] - df["previous_course"])
    .abs()
)

df["course_change"] = (
    np.minimum(
        df["course_change"],
        360 - df["course_change"]
    )
)

df["course_change"] = (
    df["course_change"]
    .fillna(0)
    .astype("float32")
)

# =====================================================
# HAVERSINE
# =====================================================

print("Distance travelled...")

R = 6371.0

lat1 = np.radians(df["prev_lat"])
lon1 = np.radians(df["prev_lon"])

lat2 = np.radians(df["lat"])
lon2 = np.radians(df["lon"])

dlat = lat2 - lat1
dlon = lon2 - lon1

a = (
    np.sin(dlat / 2) ** 2
    + np.cos(lat1)
    * np.cos(lat2)
    * np.sin(dlon / 2) ** 2
)

c = 2 * np.arctan2(
    np.sqrt(a),
    np.sqrt(1 - a)
)

distance = R * c

df["distance_previous_km"] = (
    pd.Series(distance)
    .fillna(0)
    .astype("float32")
)

# =====================================================
# STOP FLAG
# =====================================================

print("Stop flag...")

df["is_stop"] = (
    df["speed"] < 0.5
).astype("int8")

# =====================================================
# TRIP ID
# =====================================================

print("Trip segmentation...")

new_trip = (
    (df["time_gap_second"] > 3600 * 12)
    | (g["mmsi"].cumcount() == 0)
)

df["trip_id"] = (
    new_trip
    .groupby(df["mmsi"])
    .cumsum()
    .astype("int32")
)

# =====================================================
# DROP TEMP
# =====================================================

df = df.drop(
    columns=[
        "prev_lat",
        "prev_lon"
    ]
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
    "previous_speed",
    "acceleration",
    "course_change",
    "distance_previous_km",
    "trip_id",
    "is_stop"
]].head())

print()

print(df["acceleration"].describe())

print()

print(df["distance_previous_km"].describe())

print()

print("Trips :", df["trip_id"].nunique())

print()

print("Stop")

print(df["is_stop"].value_counts())

print()

print("Saved")

print(AIS_PATH)