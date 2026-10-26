from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from scripts.ct_only_assessment import extract_features, train_labelled_baseline


def _write_case(root: Path, case_id: str) -> None:
    ct_dir = root / "ct"
    mask_dir = root / "lobes"
    ct_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    ct = np.full((20, 20, 20), -850, dtype=np.int16)
    mask = np.zeros((20, 20, 20), dtype=np.uint8)
    for label in range(1, 6):
        mask[(label - 1) * 4 : label * 4, :, :] = label
        ct[(label - 1) * 4 : label * 4, :, :] = -960 if label % 2 else -800
    affine = np.diag([1.0, 1.0, 1.0, 1.0])
    nib.save(nib.Nifti1Image(ct, affine), ct_dir / f"{case_id}_insp.nii.gz")
    nib.save(nib.Nifti1Image(mask, affine), mask_dir / f"{case_id}_insp_lobes.nii.gz")


def test_extract_features_reports_whole_lung_and_five_lobes(tmp_path: Path) -> None:
    _write_case(tmp_path, "case001")
    output = tmp_path / "reports" / "features.csv"
    frame = extract_features(tmp_path / "ct", tmp_path / "lobes", output)
    assert output.is_file()
    assert len(frame) == 1
    assert "whole_lung_laa950_pct" in frame.columns
    assert "right_upper_laa950_pct" in frame.columns
    assert "left_lower_volume_ml" in frame.columns
    assert frame.loc[0, "right_upper_laa950_pct"] == pytest.approx(100.0)
    assert frame.loc[0, "right_middle_laa950_pct"] == pytest.approx(0.0)


def test_extract_features_rejects_missing_lobe_mask(tmp_path: Path) -> None:
    (tmp_path / "ct").mkdir()
    nib.save(
        nib.Nifti1Image(np.full((10, 10, 10), -800, dtype=np.int16), np.eye(4)),
        tmp_path / "ct" / "case001_insp.nii.gz",
    )
    (tmp_path / "lobes").mkdir()
    with pytest.raises(FileNotFoundError, match="Missing five-lobe mask"):
        extract_features(tmp_path / "ct", tmp_path / "lobes", tmp_path / "features.csv")


def test_report_proxy_model_is_explicitly_not_clinically_validated(tmp_path: Path) -> None:
    rng = np.random.default_rng(7)
    rows = []
    labels = []
    for idx in range(40):
        case_id = f"case{idx:03d}"
        y = idx % 2
        rows.append(
            {
                "case_id": case_id,
                "ct_path": f"{case_id}.nii.gz",
                "whole_lung_laa950_pct": float(rng.normal(5 + y * 20, 2)),
                "whole_lung_mean_hu": float(rng.normal(-800 - y * 30, 10)),
            }
        )
        labels.append({"case_id": case_id, "copd_label": y, "patient_id": case_id})
    features = pd.DataFrame(rows)
    labels_path = tmp_path / "labels.csv"
    pd.DataFrame(labels).to_csv(labels_path, index=False)
    result = train_labelled_baseline(features, labels_path, tmp_path / "model", "report")
    assert result["label_source"] == "report"
    assert result["clinical_validation_status"] == "NOT_CLINICALLY_VALIDATED"
    assert (tmp_path / "model" / "evaluation_metrics.json").is_file()


def test_ct_emphysema_bands_are_not_gold_grades() -> None:
    from scripts.ct_only_assessment import ct_emphysema_band

    assert ct_emphysema_band(2.0) == "low_ct_emphysema_burden_lt6pct"
    assert ct_emphysema_band(6.0) == "intermediate_ct_emphysema_burden_6_to_lt14pct"
    assert ct_emphysema_band(14.0) == "high_ct_emphysema_burden_ge14pct"
