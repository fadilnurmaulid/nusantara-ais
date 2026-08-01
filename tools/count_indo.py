from pathlib import Path
import pandas as pd

folder = Path(r"D:\Fadil\Activities\GEMASTIK26\naisdataset\AIS\anonymized_ais_training_data")

total = 0

for file in folder.glob("*.csv"):
    df = pd.read_csv(file)

    indo = df[
        (df["lat"] >= -11) &
        (df["lat"] <= 6) &
        (df["lon"] >= 95) &
        (df["lon"] <= 141)
    ]

    total += len(indo)

print("TOTAL DATA INDONESIA =", total)
print(indo.columns.tolist())