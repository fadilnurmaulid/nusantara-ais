from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import torch

from src.nusantara_ais.graph.dataset import load_dataset
from src.nusantara_ais.models.full_model import NusantaraAIS

data = load_dataset()

model = NusantaraAIS(data.metadata())

embedding = model(data)

print()

for k, v in embedding.items():
    print(k, v.shape)