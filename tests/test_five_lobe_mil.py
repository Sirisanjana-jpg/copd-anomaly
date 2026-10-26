from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.train_five_lobe_mil import LOBE_ORDER, train_lobe_mil
from scripts.train_multilabel_head import CTRATE_LABELS


def test_five_lobe_mil_writes_scan_metrics_and_lobe_hypotheses(tmp_path: Path) -> None:
    rng = np.random.default_rng(4)
    lobe_rows, label_rows = [], []
    for patient in range(80):
        scan = f"train_{patient}_a_1"
        labels = [int((patient + j) % (j + 3) == 0) for j in range(len(CTRATE_LABELS))]
        label_rows.append({"VolumeName": scan, **dict(zip(CTRATE_LABELS, labels, strict=True))})
        for i, lobe in enumerate(LOBE_ORDER):
            lobe_rows.append({
                "VolumeName": scan, "patient_id": f"p{patient}", "lobe_id": lobe,
                "mean_hu": float(labels[7] * (i + 1) + rng.normal()),
                "low_attenuation_pct": float(labels[13] * (5 - i) + rng.normal()),
                "volume_ml": float(100 + i + rng.normal()),
            })
    feature_path, label_path = tmp_path / "lobes.csv", tmp_path / "labels.csv"
    pd.DataFrame(lobe_rows).to_csv(feature_path, index=False)
    pd.DataFrame(label_rows).to_csv(label_path, index=False)
    report = train_lobe_mil(feature_path, label_path, tmp_path / "out", epochs=3, seed=7)
    assert report["localization_status"].startswith("WEAK_LOCALIZATION")
    assert len(report["per_label_metrics"]) == 18
    assert (tmp_path / "out" / "five_lobe_mil.pt").is_file()
    assert (tmp_path / "out" / "holdout_scan_burden_profile.csv").is_file()
    profile = pd.read_csv(tmp_path / "out" / "holdout_scan_burden_profile.csv")
    assert profile["ct_burden_status"].eq("STRUCTURAL_PROFILE_NOT_CLINICAL_SEVERITY").all()
    out = pd.read_csv(tmp_path / "out" / "holdout_lobe_hypotheses.csv")
    assert set(out.lobe_id) == set(LOBE_ORDER)
    assert len(out) == report["n_test_scans"] * 5


def test_five_lobe_mil_rejects_incomplete_lobes(tmp_path: Path) -> None:
    feature_path, label_path = tmp_path / "lobes.csv", tmp_path / "labels.csv"
    pd.DataFrame([{"scan_id": "x", "lobe_id": "LUL", "f0": 1.0}]).to_csv(feature_path, index=False)
    pd.DataFrame([{"scan_id": "x", **dict.fromkeys(CTRATE_LABELS, 0)}]).to_csv(label_path, index=False)
    with pytest.raises(ValueError, match="complete five-lobe scans"):
        train_lobe_mil(feature_path, label_path, tmp_path / "out")
