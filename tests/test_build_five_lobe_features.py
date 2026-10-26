from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from scripts.build_five_lobe_features import LOBE_LABELS, build_lobe_feature_table


def test_builds_five_lobe_feature_rows_from_nifti(tmp_path: Path) -> None:
    ct_dir, mask_dir = tmp_path / "ct", tmp_path / "masks"
    ct_dir.mkdir()
    mask_dir.mkdir()
    scan_id = "train_7_a_1"
    ct = np.full((10, 10, 10), -900, dtype=np.float32)
    mask = np.zeros((10, 10, 10), dtype=np.uint8)
    for label in LOBE_LABELS:
        mask[label - 1 : label + 1, :, :] = label
    affine = np.diag([1.0, 1.0, 1.0, 1.0])
    ct_path = ct_dir / f"{scan_id}.nii.gz"
    mask_path = mask_dir / f"{scan_id}_lobes.nii.gz"
    nib.save(nib.Nifti1Image(ct, affine), ct_path)
    nib.save(nib.Nifti1Image(mask, affine), mask_path)

    output = tmp_path / "features.csv"
    frame = build_lobe_feature_table(ct_dir, mask_dir, output)
    assert len(frame) == 5
    assert set(frame["lobe_id"]) == set(LOBE_LABELS.values())
    assert "low_attenuation_pct_le_minus950" in frame.columns
    assert pd.read_csv(output).shape[0] == 5
