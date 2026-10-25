"""Safety tests for held-out report-label proxy evaluation."""

from __future__ import annotations

import pandas as pd
import pytest

from scripts.evaluate_proxy import evaluate_report_label_proxy


def test_proxy_evaluation_requires_test_split_and_exact_patient_join(tmp_path) -> None:
    scores_path = tmp_path / "patient_scores.csv"
    labels_path = tmp_path / "labels.csv"
    output_path = tmp_path / "metrics.json"
    pd.DataFrame(
        {
            "patient_id": [f"p{i}" for i in range(10)],
            "split": ["test"] * 10,
            "data_status": ["REAL_CT_PATCH_INFERENCE_UNVALIDATED"] * 10,
            "anomaly_mean": [float(i) for i in range(10)],
            "anomaly_p90": [float(i + 1) for i in range(10)],
            "anomaly_p95": [float(i + 2) for i in range(10)],
            "anomaly_max": [float(i + 3) for i in range(10)],
        }
    ).to_csv(scores_path, index=False)
    pd.DataFrame({"patient_id": [f"p{i}" for i in range(10)], "label": [0] * 5 + [1] * 5}).to_csv(
        labels_path, index=False
    )

    result = evaluate_report_label_proxy(
        scores_path,
        labels_path,
        output_path,
        label_definition="test-only synthetic fixture; not a research label",
        n_bootstraps=10,
    )

    assert output_path.is_file()
    assert result["data_status"] == "CT_RATE_REPORT_LABEL_PROXY_NOT_CLINICAL_DIAGNOSIS"
    assert result["clinical_claims_allowed"] is False
    assert result["n_patients"] == 10


def test_proxy_evaluation_rejects_training_split(tmp_path) -> None:
    scores_path = tmp_path / "patient_scores.csv"
    labels_path = tmp_path / "labels.csv"
    pd.DataFrame(
        {
            "patient_id": ["p0", "p1"],
            "split": ["train", "train"],
            "data_status": ["REAL_CT_PATCH_INFERENCE_UNVALIDATED"] * 2,
            "anomaly_mean": [0.1, 0.9],
        }
    ).to_csv(scores_path, index=False)
    pd.DataFrame({"patient_id": ["p0", "p1"], "label": [0, 1]}).to_csv(labels_path, index=False)

    with pytest.raises(ValueError, match="Evaluation must use only split"):
        evaluate_report_label_proxy(
            scores_path,
            labels_path,
            tmp_path / "metrics.json",
            label_definition="unit test",
            n_bootstraps=5,
        )
    assert not (tmp_path / "metrics.json").exists()
