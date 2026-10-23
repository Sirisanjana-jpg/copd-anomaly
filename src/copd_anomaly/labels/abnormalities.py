"""
CT-RATE Abnormality label definitions and normal control filtering.

Enforces zero-abnormality criteria for normal lung controls:
All of the following must be 0 (negative):
1. Emphysema
2. Mosaic attenuation pattern
3. Lung opacity
4. Consolidation
5. Atelectasis
6. Pulmonary fibrotic sequela
7. Bronchiectasis
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TARGET_NORMAL_NEGATIVE_COLUMNS = [
    "Emphysema",
    "Mosaic attenuation pattern",
    "Lung opacity",
    "Consolidation",
    "Atelectasis",
    "Pulmonary fibrotic sequela",
    "Bronchiectasis",
]

TARGET_EVAL_COLUMNS = [
    "Emphysema",
    "Mosaic attenuation pattern",
]


def is_normal_control(row: pd.Series | dict) -> bool:
    """
    Checks if a CT-RATE scan meets the strict normal control definition.
    All target abnormality columns must be 0 (or False / absent).
    """
    for col in TARGET_NORMAL_NEGATIVE_COLUMNS:
        # Check standard and snake_case column names
        val = None
        if col in row:
            val = row[col]
        elif col.lower().replace(" ", "_") in row:
            val = row[col.lower().replace(" ", "_")]

        if val is not None and pd.notna(val):
            # Treat numeric > 0 or bool True as abnormal
            if isinstance(val, (int, float, np.integer, np.floating)) and val > 0:
                return False
            if isinstance(val, bool) and val is True:
                return False
            if str(val).strip() in ("1", "1.0", "True", "true", "positive"):
                return False

    return True


def filter_normal_controls(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filters a metadata DataFrame down to scans with zero reported abnormalities.
    """
    mask = df.apply(is_normal_control, axis=1)
    return df[mask].reset_index(drop=True)
