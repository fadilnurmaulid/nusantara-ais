"""Seed control and device selection for full reproducibility."""
from __future__ import annotations

import logging
import os
import random
import sys
from typing import Optional

import numpy as np
import torch


def set_global_seed(seed: int) -> None:
    """Fix every source of randomness reachable from this process.

    Covers: Python's `random`, NumPy, PyTorch CPU/CUDA RNGs, PYTHONHASHSEED
    (affects dict/set iteration order in some CPython builds), and forces
    deterministic cuDNN kernels (at a performance cost, which is acceptable
    for the reproducibility guarantees required by the paper).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def get_device(preference: str = "cuda_if_available") -> torch.device:
    if preference == "cpu":
        return torch.device("cpu")
    if preference == "cuda_if_available" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def setup_logger(name: str, log_dir: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)
        fh = logging.FileHandler(os.path.join(log_dir, f"{name}.log"))
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger
