"""Checkpoint and density-model persistence failure tests."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest
import torch

from copd_anomaly.anomaly.density import GaussianMixtureDensity
from scripts.train_encoder import _atomic_torch_save


def test_atomic_checkpoint_round_trip_and_temp_cleanup(tmp_path: Path) -> None:
    destination = tmp_path / "encoder_latest.pt"
    _atomic_torch_save(
        {"training_mode": "real_patch_training", "epoch": 2, "value": torch.tensor([1.0])},
        destination,
    )

    loaded = torch.load(destination, map_location="cpu", weights_only=False)
    assert loaded["epoch"] == 2
    assert torch.equal(loaded["value"], torch.tensor([1.0]))
    assert not list(tmp_path.glob(".*.tmp"))


def test_failed_checkpoint_write_preserves_previous_checkpoint(tmp_path: Path) -> None:
    destination = tmp_path / "encoder_latest.pt"
    _atomic_torch_save({"training_mode": "real_patch_training", "epoch": 1}, destination)

    # Torch/Python versions may surface this serialization failure as either
    # pickle.PicklingError or AttributeError for a local lambda.
    with pytest.raises((pickle.PicklingError, AttributeError)):
        _atomic_torch_save(
            {"training_mode": "real_patch_training", "bad_value": lambda: None},
            destination,
        )

    loaded = torch.load(destination, map_location="cpu", weights_only=False)
    assert loaded["epoch"] == 1
    assert not list(tmp_path.glob(".*.tmp"))


def test_density_fit_rejects_non_finite_embeddings() -> None:
    model = GaussianMixtureDensity(n_components=2)
    embeddings = np.array([[0.0, 1.0], [np.nan, 2.0], [3.0, 4.0]], dtype=np.float32)
    with pytest.raises(ValueError, match="NaN or infinite"):
        model.fit(embeddings)


def test_unfitted_density_model_cannot_be_saved(tmp_path: Path) -> None:
    model = GaussianMixtureDensity(n_components=2)
    destination = tmp_path / "density.joblib"
    with pytest.raises(RuntimeError, match="before fitting"):
        model.save(str(destination))
    assert not destination.exists()
