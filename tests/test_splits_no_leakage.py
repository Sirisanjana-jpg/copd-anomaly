"""
Unit tests for CT-RATE patient-level disjoint splitting and zero-leakage verification.
"""

from __future__ import annotations

import pandas as pd
import pytest

from copd_anomaly.data.ctrate_streamer import (
    create_tiered_splits,
    extract_patient_id,
    verify_split_leakage,
)


@pytest.fixture
def synthetic_metadata_cohort():
    """Generates synthetic CT-RATE train and valid metadata with multiple reconstructions per subject."""
    train_records = []
    # 200 train subjects, 2 scans each
    for i in range(200):
        pat_id = f"train_{1000 + i}"
        train_records.append(
            {
                "VolumeName": f"{pat_id}_a_1.nii.gz",
                "patient_id": pat_id,
                "Emphysema": 0 if i < 150 else 1,
                "Mosaic attenuation pattern": 0,
                "Lung opacity": 0,
                "Consolidation": 0,
                "Atelectasis": 0,
                "Pulmonary fibrotic sequela": 0,
                "Bronchiectasis": 0,
            }
        )
        train_records.append(
            {
                "VolumeName": f"{pat_id}_b.nii.gz",
                "patient_id": pat_id,
                "Emphysema": 0 if i < 150 else 1,
                "Mosaic attenuation pattern": 0,
                "Lung opacity": 0,
                "Consolidation": 0,
                "Atelectasis": 0,
                "Pulmonary fibrotic sequela": 0,
                "Bronchiectasis": 0,
            }
        )

    valid_records = []
    # 100 valid subjects, 2 scans each
    for i in range(100):
        pat_id = f"valid_{2000 + i}"
        valid_records.append(
            {
                "VolumeName": f"{pat_id}_a_1.nii.gz",
                "patient_id": pat_id,
                "Emphysema": 0 if i % 2 == 0 else 1,
                "Mosaic attenuation pattern": 0,
                "Lung opacity": 0,
                "Consolidation": 0,
                "Atelectasis": 0,
                "Pulmonary fibrotic sequela": 0,
                "Bronchiectasis": 0,
            }
        )

    return pd.DataFrame(train_records), pd.DataFrame(valid_records)


def test_patient_id_extraction():
    """Verify patient ID parsing from volume file names."""
    assert extract_patient_id("train_10042_a_1.nii.gz") == "train_10042"
    assert extract_patient_id("dataset/valid/valid_0055_b.nii.gz") == "valid_0055"
    assert extract_patient_id("scan_99999.nii.gz") == "patient_99999"


def test_split_leakage_assertion_pass_and_fail():
    """Verify verify_split_leakage passes on disjoint sets and raises AssertionError on leakage."""
    set_a = {"p1", "p2", "p3"}
    set_b = {"p4", "p5"}
    set_c = {"p6", "p7"}

    # Disjoint sets should pass without error
    verify_split_leakage(set_a, set_b, set_c)

    # Introduce leakage between train and val
    leaky_b = {"p3", "p4", "p5"}
    with pytest.raises(AssertionError, match="Patient leakage detected"):
        verify_split_leakage(set_a, leaky_b, set_c)

    # Introduce leakage between train and test
    leaky_c = {"p1", "p7"}
    with pytest.raises(AssertionError, match="Patient leakage detected"):
        verify_split_leakage(set_a, set_b, leaky_c)


def test_tiered_splits_creation(synthetic_metadata_cohort):
    """Verify creation of patient-disjoint lean tier splits."""
    train_meta, valid_meta = synthetic_metadata_cohort

    splits = create_tiered_splits(train_meta, valid_meta, tier="lean", seed=42)

    df_train = splits["train"]
    df_val = splits["val"]
    df_test = splits["test"]

    # In 'lean' tier: train=100, val=30, test=70
    assert len(df_train) == 100
    assert len(df_val) == 30
    assert len(df_test) == 70

    # Ensure single volume per patient
    assert df_train["patient_id"].nunique() == 100
    assert df_val["patient_id"].nunique() == 30
    assert df_test["patient_id"].nunique() == 70

    # Check preferred suffix selection
    assert df_train["VolumeName"].str.endswith("_a_1.nii.gz").all()

    # CRITICAL: Verify zero patient overlap across all three splits
    patients_train = set(df_train["patient_id"])
    patients_val = set(df_val["patient_id"])
    patients_test = set(df_test["patient_id"])

    assert patients_train.isdisjoint(patients_val)
    assert patients_train.isdisjoint(patients_test)
    assert patients_val.isdisjoint(patients_test)
