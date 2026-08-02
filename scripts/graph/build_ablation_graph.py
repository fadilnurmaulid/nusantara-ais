from pathlib import Path
import sys
import copy
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.nusantara_ais.graph.dataset import load_dataset

OUT = ROOT / "data" / "processed" / "ablation"
OUT.mkdir(parents=True, exist_ok=True)

print("Loading graph...")
graph = load_dataset()

# =====================================================
# FULL
# =====================================================

torch.save(
    graph,
    OUT / "full.pt",
)

print("Saved full.pt")

# =====================================================
# NO ARI
# kolom terakhir = ari
# =====================================================

g = copy.deepcopy(graph)

g["ais"].x[:, -1] = 0

torch.save(
    g,
    OUT / "no_ari.pt",
)

print("Saved no_ari.pt")

# =====================================================
# NO TRIP
# =====================================================

g = copy.deepcopy(graph)

del g["ais", "trip", "trip"]
del g["trip", "rev_trip", "ais"]

torch.save(
    g,
    OUT / "no_trip.pt",
)

print("Saved no_trip.pt")

# =====================================================
# NO PORT
# =====================================================

g = copy.deepcopy(graph)

del g["ais", "near", "port"]
del g["port", "rev_near", "ais"]

torch.save(
    g,
    OUT / "no_port.pt",
)

print("Saved no_port.pt")

# =====================================================
# NO PROTECTED
# =====================================================

g = copy.deepcopy(graph)

del g["ais", "protected", "protected"]
del g["protected", "rev_protected", "ais"]

torch.save(
    g,
    OUT / "no_protected.pt",
)

print("Saved no_protected.pt")

print("\nDone.")