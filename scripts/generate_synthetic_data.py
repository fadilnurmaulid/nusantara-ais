#!/usr/bin/env python3
"""Generate/refresh the synthetic AIS + BMKG + bathymetry + coastline +
ports + shipping-lane dataset used when real (licensed) datasets are not
present under data/raw/. Writes parquet files to data/synthetic/.

Usage:
    python scripts/generate_synthetic_data.py --n-vessels 60 --points-per-vessel 144
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from nusantara_ais.config import Config
from nusantara_ais.data import synthetic


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-vessels", type=int, default=60)
    parser.add_argument("--points-per-vessel", type=int, default=144)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = Config()
    data = synthetic.generate_all(cfg, n_vessels=args.n_vessels, points_per_vessel=args.points_per_vessel,
                                   seed=args.seed)
    out_dir = cfg.paths.abs(cfg.paths.synthetic_dir)
    print(f"Synthetic dataset written to {out_dir}")
    for name, df in data.items():
        if hasattr(df, "shape"):
            print(f"  {name}: {df.shape}")


if __name__ == "__main__":
    main()
