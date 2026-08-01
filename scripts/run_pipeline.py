#!/usr/bin/env python3
"""
CLI entrypoint: runs the full NUSANTARA-AIS pipeline end-to-end and writes
a JSON/markdown experiment report + figures to `outputs/reports`.

Usage:
    python scripts/run_pipeline.py --n-vessels 60 --points-per-vessel 144
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from nusantara_ais.config import Config
from nusantara_ais.pipeline import run_full_pipeline


def _json_default(o):
    import numpy as np
    import pandas as pd
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, pd.DataFrame):
        return o.to_dict(orient="records")
    return str(o)


def main():
    parser = argparse.ArgumentParser(description="Run the NUSANTARA-AIS end-to-end pipeline")
    parser.add_argument("--n-vessels", type=int, default=60)
    parser.add_argument("--points-per-vessel", type=int, default=144)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--no-baselines", action="store_true")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()

    cfg = Config()
    if args.epochs is not None:
        cfg.training.epochs = args.epochs

    result = run_full_pipeline(
        cfg, n_vessels=args.n_vessels, points_per_vessel=args.points_per_vessel,
        generate_figures=not args.no_figures, run_baselines=not args.no_baselines,
    )

    report_dir = cfg.paths.abs(cfg.paths.reports_dir)
    os.makedirs(report_dir, exist_ok=True)

    serializable_eval = {}
    for k, v in result["eval_report"].items():
        serializable_eval[k] = v.to_dict(orient="records") if hasattr(v, "to_dict") else v

    report = {
        "experiment_name": cfg.experiment_name,
        "n_vessels": args.n_vessels,
        "points_per_vessel": args.points_per_vessel,
        "n_train_snapshots": len(result["train_ds"]),
        "n_val_snapshots": len(result["val_ds"]),
        "n_test_snapshots": len(result["test_ds"]),
        "best_val_loss": result["train_history"][-1]["val_loss"] if result["train_history"] else None,
        "n_epochs_run": len(result["train_history"]),
        "evaluation": serializable_eval,
        "figures": result["figures"],
    }

    report_path = os.path.join(report_dir, "experiment_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=_json_default)

    print(f"\nExperiment report written to {report_path}")
    if "detection_metrics" in result["eval_report"]:
        print("Detection metrics:", result["eval_report"]["detection_metrics"])
    print("Figures:", list(result["figures"].values()))


if __name__ == "__main__":
    main()
