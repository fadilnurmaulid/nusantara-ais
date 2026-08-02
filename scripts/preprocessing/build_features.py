from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

sys.path.append(str(ROOT / "src"))

from nusantara_ais.features.engineer import build_features
from nusantara_ais.features.normalize import normalize
from nusantara_ais.features.reliability import compute_ari

DATA_PATH = ROOT / "data" / "processed" / "ais_dataset_v1.parquet"

print("Loading dataset...")

df = pd.read_parquet(DATA_PATH)

print(df.shape)

print("Engineering features...")

df = build_features(df)
df = compute_ari(df)
df = normalize(df)

print(df.shape)

print("Saving...")

df.to_parquet(
    DATA_PATH,
    index=False
)

print()

print("Done")
print(df.columns)