"""Unit tests for CT-RATE split integrity and streaming preflight safety."""
from __future__ import annotations

import pandas as pd
import pytest

from copd_anomaly.data.ctrate_streamer import (
    create_tiered_splits,
    extract_patient_id,
    verify_split_leakage,
)
from scripts.stream_ctrate import stream_ctrate


def test_extract_patient_id_uses_split_prefix() -> None:
    assert extract_patient_id("train_10001_a_1.nii.gz") == "train_10001"
    assert extract_patient_id("valid_20001_a_1.nii.gz") == "valid_20001"


def test_create_tiered_splits_is_patient_disjoint() -> None:
    train = pd.DataFrame(
        {"VolumeName": [f"train_{idx:05d}_a_1.nii.gz" for idx in range(130)]}
    )
    valid = pd.DataFrame(
        {"VolumeName": [f"valid_{idx:05d}_a_1.nii.gz" for idx in range(130, 200)]}
    )
    splits = create_tiered_splits(train, valid, tier="lean", seed=42)
    patients = {
        name: set(frame["patient_id"].astype(str))
        for name, frame in splits.items()
    }
    assert len(splits["train"]) == 100
    assert len(splits["val"]) == 30
    assert len(splits["test"]) == 70
    assert patients["train"].isdisjoint(patients["val"])
    assert patients["train"].isdisjoint(patients["test"])
    assert patients["val"].isdisjoint(patients["test"])


def test_split_leakage_is_rejected() -> None:
    with pytest.raises(AssertionError, match="Patient leakage detected"):
        verify_split_leakage({"patient_a"}, {"patient_a"}, {"patient_b"})


def test_streaming_requires_token_before_file_or_network_access(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HF_TOKEN", "")
    monkeypatch.setenv("HF_READ", "")
    with pytest.raises(EnvironmentError, match="CT-RATE access token not found"):
        stream_ctrate(
            train_metadata_path=tmp_path / "missing_train.csv",
            valid_metadata_path=tmp_path / "missing_valid.csv",
            output_dir=tmp_path / "out",
        )
    assert not (tmp_path / "out").exists()
