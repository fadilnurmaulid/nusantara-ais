from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

DATA = ROOT / "data" / "processed" / "ais_indonesia_clean.parquet"

print("Loading dataset...")

df = pd.read_parquet(DATA)

# =====================================================
# COURSE
# =====================================================

df["course"] = (
    df.groupby("mmsi")["course"]
      .ffill()
      .bfill()
      .fillna(0)
)

# =====================================================
# PREVIOUS SPEED
# =====================================================

df["previous_speed"] = (
    df["previous_speed"]
      .fillna(0)
)

# =====================================================
# PREVIOUS COURSE
# =====================================================

df["previous_course"] = (
    df["previous_course"]
      .fillna(df["course"])
      .fillna(0)
)

# =====================================================
# SAVE
# =====================================================

df.to_parquet(DATA, index=False)

print("\nDone.\n")

print(df[[
    "course",
    "previous_speed",
    "previous_course"
]].isna().sum())

print("\nRemaining missing values:\n")
print(df.isna().sum())