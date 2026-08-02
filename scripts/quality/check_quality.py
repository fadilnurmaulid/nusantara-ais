import pandas as pd
from pathlib import Path

files = list(Path("anonymized_ais_training_data").glob("*.csv"))

dfs = []

for f in files:
    df = pd.read_csv(f)

    df = df[
        (df["lat"] >= -11) &
        (df["lat"] <= 6) &
        (df["lon"] >= 95) &
        (df["lon"] <= 141)
    ]

    dfs.append(df)

df = pd.concat(dfs, ignore_index=True)

print("="*50)

print("Shape")
print(df.shape)

print("="*50)

print("Missing value")
print(df.isna().sum())

print("="*50)

print("Duplicate")
print(df.duplicated().sum())

print("="*50)

print("Speed")
print(df["speed"].describe())

print("="*50)

print("Course")
print(df["course"].describe())

print("="*50)

print("Timestamp")
print(pd.to_datetime(df["timestamp"], unit="s").describe())

print("="*50)

print("Fishing label")
print(df["is_fishing"].value_counts())