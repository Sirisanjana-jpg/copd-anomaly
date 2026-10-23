"""
Utility functions for deterministic seeding, logging, caching, and integrity validation.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
from typing import Any

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Sets random seeds across random, numpy, and torch for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_logger(name: str = "copd_anomaly", level: int = logging.INFO) -> logging.Logger:
    """Configures and returns a structured logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def compute_config_hash(cfg: dict[str, Any] | Any) -> str:
    """Computes a deterministic SHA256 hex digest from a dictionary configuration."""
    if hasattr(cfg, "__dict__"):
        cfg_dict = cfg.__dict__
    elif isinstance(cfg, dict):
        cfg_dict = cfg
    else:
        cfg_dict = {"value": str(cfg)}

    serialized = json.dumps(cfg_dict, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]

