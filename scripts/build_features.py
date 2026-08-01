from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

import pandas as pd

from nusantara_ais.features.engineer import build_features

ROOT = Path(__file__).resolve().parents[1]

path = ROOT / "data" / "processed" / "ais_indonesia_clean.parquet"

df = pd.read_parquet(path)

df = build_features(df)

df.to_parquet(path, index=False)

print(df.columns)
print(df.shape)