from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

df = pd.read_parquet(
    ROOT / "results" / "ais_scored.parquet"
)

plt.figure(figsize=(8,5))

plt.hist(
    df["anomaly_score"],
    bins=100,
)

plt.yscale("log")

plt.xlabel("Anomaly Score")
plt.ylabel("Frequency (log)")
plt.title("Distribution of Anomaly Score")

plt.tight_layout()

OUT = ROOT / "results"

plt.savefig(
    OUT / "histogram_anomaly.png",
    dpi=300,
)

print("Saved:", OUT / "histogram_anomaly.png")