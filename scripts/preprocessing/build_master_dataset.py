from pathlib import Path
import pandas as pd

# =====================================================
# Configuration
# =====================================================

RAW_DIR = Path("data/raw/AIS/anonymized_ais_training_data")
OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "ais_indonesia_clean.parquet"

# Indonesia Bounding Box
LAT_MIN = -11.0
LAT_MAX = 6.0
LON_MIN = 95.0
LON_MAX = 141.0

# =====================================================
# Load Dataset
# =====================================================

csv_files = sorted(RAW_DIR.glob("*.csv"))

if len(csv_files) == 0:
    raise FileNotFoundError(f"Tidak ada CSV pada:\n{RAW_DIR}")

print("=" * 60)
print("Loading AIS datasets...")
print("=" * 60)

dfs = []

for file in csv_files:

    print(f"Reading {file.name}")

    df = pd.read_csv(file)

    # -----------------------------------------------
    # Filter Indonesia
    # -----------------------------------------------

    df = df[
        (df["lat"] >= LAT_MIN)
        & (df["lat"] <= LAT_MAX)
        & (df["lon"] >= LON_MIN)
        & (df["lon"] <= LON_MAX)
    ]

    dfs.append(df)

print()
print("Merging datasets...")

df = pd.concat(dfs, ignore_index=True)

# =====================================================
# Cleaning
# =====================================================

print()
print("=" * 60)
print("Cleaning")
print("=" * 60)

print("Initial rows :", len(df))

# Remove duplicate
before = len(df)
df = df.drop_duplicates()
print("Duplicate removed :", before - len(df))

# Remove missing speed/course
before = len(df)
df = df.dropna(subset=["speed", "course"])
print("Missing removed :", before - len(df))

# Invalid course
invalid_course = (df["course"] > 360).sum()

df.loc[df["course"] > 360, "course"] = pd.NA

print("Invalid course (>360):", invalid_course)

# Convert timestamp
df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")

# Numeric type optimization
float_cols = [
    "distance_from_shore",
    "distance_from_port",
    "speed",
    "course",
    "lat",
    "lon",
]

for col in float_cols:
    df[col] = pd.to_numeric(df[col], downcast="float")

df["mmsi"] = df["mmsi"].astype("int64")
df["is_fishing"] = df["is_fishing"].astype("int8")

# Sort
df = df.sort_values(
    ["mmsi", "timestamp"],
    ascending=True
).reset_index(drop=True)

# =====================================================
# Statistics
# =====================================================

print()
print("=" * 60)
print("Dataset Summary")
print("=" * 60)

print("Rows            :", len(df))
print("Unique MMSI     :", df["mmsi"].nunique())
print("Start           :", df["timestamp"].min())
print("End             :", df["timestamp"].max())
print("Missing values")
print(df.isna().sum())

print()
print(df.head())

# =====================================================
# Save
# =====================================================

print()
print("=" * 60)
print("Saving...")
print("=" * 60)

df.to_parquet(
    OUTPUT_FILE,
    index=False
)

print(f"Saved -> {OUTPUT_FILE}")

print()
print("Done.")