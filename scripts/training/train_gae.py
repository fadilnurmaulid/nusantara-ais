from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]

sys.path.append(str(ROOT))

from src.nusantara_ais.training.train import train

train(
    epochs=200,
    lr=1e-3,
)