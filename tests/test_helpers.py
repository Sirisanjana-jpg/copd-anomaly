"""
Tests for utility functions: seed determinism, logger, and config hashing.
"""

import numpy as np
import pytest
import torch

from copd_anomaly.utils.helpers import compute_config_hash, get_logger, set_seed


@pytest.mark.synthetic
def test_set_seed_determinism():
    """Verify that set_seed ensures reproducible random generation."""
    set_seed(123)
    np_rand_1 = np.random.rand(5)
    torch_rand_1 = torch.rand(5)

    set_seed(123)
    np_rand_2 = np.random.rand(5)
    torch_rand_2 = torch.rand(5)

    assert np.allclose(np_rand_1, np_rand_2)
    assert torch.allclose(torch_rand_1, torch_rand_2)


@pytest.mark.synthetic
def test_compute_config_hash():
    """Verify that config hashing produces deterministic digests."""
    cfg1 = {"model": "resnet18", "lr": 0.001, "layers": [1, 2, 3]}
    cfg2 = {"layers": [1, 2, 3], "lr": 0.001, "model": "resnet18"}  # Permuted keys
    cfg3 = {"model": "resnet18", "lr": 0.002, "layers": [1, 2, 3]}  # Changed value

    hash1 = compute_config_hash(cfg1)
    hash2 = compute_config_hash(cfg2)
    hash3 = compute_config_hash(cfg3)

    assert hash1 == hash2
    assert hash1 != hash3
    assert len(hash1) == 16


@pytest.mark.synthetic
def test_get_logger():
    """Verify logger creation and message formatting."""
    logger = get_logger("test_logger")
    assert logger.name == "test_logger"
    assert len(logger.handlers) >= 1
