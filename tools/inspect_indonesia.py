import pandas as pd
from pathlib import Path

files = list(Path("anonymized_ais_training_data").glob("*.csv"))

indo = []

for f in files:
    df = pd.read_csv(f)

    df = df[
        (df["lat"] >= -11) &
        (df["lat"] <= 6) &
        (df["lon"] >= 95) &
        (df["lon"] <= 141)
    ]

    indo.append(df)

indo = pd.concat(indo, ignore_index=True)

print("Jumlah data :", len(indo))
print()

print("Jumlah MMSI :", indo["mmsi"].nunique())
print()

print("Tanggal awal :", pd.to_datetime(indo["timestamp"], unit="s").min())
print("Tanggal akhir:", pd.to_datetime(indo["timestamp"], unit="s").max())
print()

print(indo.head())