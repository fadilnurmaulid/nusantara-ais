from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]

# ==========================================
# LOAD
# ==========================================

print("Loading dataset...")

df = pd.read_parquet(
    ROOT / "data/processed/ais_dataset_v1.parquet"
)

score = torch.load(
    ROOT / "data/processed/hetgat_anomaly_score.pt",
    weights_only=False,
)

embedding = torch.load(
    ROOT / "data/processed/hetgat_embedding.pt",
    weights_only=False,
)

# ==========================================
# ADD SCORE
# ==========================================

df["anomaly_score"] = score.numpy()

# embedding norm
df["embedding_norm"] = embedding.norm(dim=1).numpy()

# ==========================================
# SORT
# ==========================================

df = df.sort_values(
    "anomaly_score",
    ascending=False,
)

# ==========================================
# TOP 100
# ==========================================

top100 = df.head(100)

# ==========================================
# SAVE
# ==========================================

OUT = ROOT / "results"

OUT.mkdir(exist_ok=True)

top100.to_csv(
    OUT / "top100_anomaly.csv",
    index=False,
)

df.to_parquet(
    OUT / "ais_scored.parquet",
    index=False,
)

# ==========================================
# REPORT
# ==========================================

print()

print("========== SUMMARY ==========")

print(df["anomaly_score"].describe())

print()

print("Top Vessel")

print(
    top100["mmsi"].value_counts().head(20)
)

print()

print("Saved")

print(OUT / "top100_anomaly.csv")
print(OUT / "ais_scored.parquet")