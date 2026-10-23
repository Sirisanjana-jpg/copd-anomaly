"""
Unit tests for CT-RATE label extraction, normal-lung criteria, and intensity windowing.
"""

from __future__ import annotations

import pandas as pd
import pytest

from copd_anomaly.labels.abnormalities import (
    TARGET_NORMAL_NEGATIVE_COLUMNS,
    filter_normal_controls,
    is_normal_control,
)
from copd_anomaly.patches.sampler import window_and_to_uint8


def test_is_normal_control_criteria():
    """Verify strict normal control definition across the 7 required pulmonary targets."""
    # Completely clean scan
    clean_row = {col: 0 for col in TARGET_NORMAL_NEGATIVE_COLUMNS}
    clean_row["Cardiomegaly"] = 1  # Extrapulmonary abnormality allowed
    assert is_normal_control(clean_row) is True

    # Scan with single pulmonary abnormality
    for target in TARGET_NORMAL_NEGATIVE_COLUMNS:
        bad_row = clean_row.copy()
        bad_row[target] = 1
        assert is_normal_control(bad_row) is False, f"Failed for {target}=1"

        bad_row_str = clean_row.copy()
        bad_row_str[target] = "1"
        assert is_normal_control(bad_row_str) is False, f"Failed for {target}='1'"


def test_filter_normal_controls_dataframe():
    """Verify filter_normal_controls retains only true normal lung subjects."""
    records = [
        {"scan_id": "s1", "Emphysema": 0, "Consolidation": 0, "Lung opacity": 0, "Atelectasis": 0, "Bronchiectasis": 0, "Mosaic attenuation pattern": 0, "Pulmonary fibrotic sequela": 0},
        {"scan_id": "s2", "Emphysema": 1, "Consolidation": 0, "Lung opacity": 0, "Atelectasis": 0, "Bronchiectasis": 0, "Mosaic attenuation pattern": 0, "Pulmonary fibrotic sequela": 0},
        {"scan_id": "s3", "Emphysema": 0, "Consolidation": 1, "Lung opacity": 0, "Atelectasis": 0, "Bronchiectasis": 0, "Mosaic attenuation pattern": 0, "Pulmonary fibrotic sequela": 0},
        {"scan_id": "s4", "Emphysema": 0, "Consolidation": 0, "Lung opacity": 0, "Atelectasis": 0, "Bronchiectasis": 0, "Mosaic attenuation pattern": 0, "Pulmonary fibrotic sequela": 0},
    ]
    df = pd.DataFrame(records)
    filtered = filter_normal_controls(df)
    assert len(filtered) == 2
    assert list(filtered["scan_id"]) == ["s1", "s4"]
