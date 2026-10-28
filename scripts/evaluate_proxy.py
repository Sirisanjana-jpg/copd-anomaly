"""Evaluate a held-out CT-RATE report-label proxy from real patient scores.

This is not clinical COPD diagnosis, spirometry-confirmed disease classification, or
exacerbation prediction. The label CSV must be created and reviewed separately.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from copd_anomaly.evaluation.metrics import ModelEvaluator
from copd_anomaly.utils.helpers import get_logger

logger = get_logger("evaluate_proxy")
ALLOWED_SCORES = {"anomaly_mean", "anomaly_p90", "anomaly_p95", "anomaly_max"}


def evaluate_report_label_proxy(
    patient_scores_path: Path,
    labels_path: Path,
    output_path: Path,
    label_definition: str,
    score_column: str = "anomaly_mean",
    expected_split: str = "test",
    n_bootstraps: int = 500,
) -> dict[str, object]:
    if not label_definition.strip():
        raise ValueError("A concise, documented --label-definition is required.")
    if score_column not in ALLOWED_SCORES:
        raise ValueError(f"score_column must be one of {sorted(ALLOWED_SCORES)}.")
    if n_bootstraps < 1:
        raise ValueError("n_bootstraps must be >= 1.")
    for path in (patient_scores_path, labels_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required evaluation input not found: {path}")
    scores = pd.read_csv(patient_scores_path)
    labels = pd.read_csv(labels_path)
    score_required = {"patient_id", "split", "data_status", score_column}
    label_required = {"patient_id", "label"}
    if not score_required.issubset(scores.columns):
        raise ValueError(
            f"Patient score table is missing columns: {sorted(score_required - set(scores.columns))}."
        )
    if not label_required.issubset(labels.columns):
        raise ValueError(
            f"Label table is missing columns: {sorted(label_required - set(labels.columns))}."
        )
    if scores.empty or labels.empty:
        raise ValueError("Patient score and label tables must both be non-empty.")
    if scores["data_status"].astype(str).ne("REAL_CT_PATCH_INFERENCE_UNVALIDATED").any():
        raise ValueError("Score table does not have expected real-inference provenance.")
    splits = set(scores["split"].astype(str))
    if splits != {expected_split}:
        raise ValueError(
            f"Evaluation must use only split={expected_split!r}; score table contains {sorted(splits)}."
        )
    if "split" in labels.columns and set(labels["split"].astype(str)) != {expected_split}:
        raise ValueError(
            f"Label table must contain only split={expected_split!r}; "
            f"found {sorted(set(labels['split'].astype(str)))}."
        )
    if scores["patient_id"].astype(str).duplicated().any():
        raise ValueError("Patient score table must have one row per patient.")
    if labels["patient_id"].astype(str).duplicated().any():
        raise ValueError("Label table must have one row per patient.")
    y = pd.to_numeric(labels["label"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(y).all() or not set(np.unique(y)).issubset({0.0, 1.0}):
        raise ValueError("Labels must be finite binary values 0/1.")
    y_score_series = pd.to_numeric(scores[score_column], errors="coerce")
    if not np.isfinite(y_score_series.to_numpy(dtype=float)).all():
        raise ValueError(f"Score column {score_column!r} contains non-finite values.")

    score_ids = set(scores["patient_id"].astype(str))
    label_ids = set(labels["patient_id"].astype(str))
    if score_ids != label_ids:
        raise ValueError(
            "Held-out patient IDs do not match exactly between scores and labels; "
            f"missing labels={len(score_ids - label_ids)}, unmatched labels={len(label_ids - score_ids)}."
        )
    labels = labels.assign(patient_id=labels["patient_id"].astype(str))
    scores = scores.assign(patient_id=scores["patient_id"].astype(str))
    merged = scores[["patient_id", score_column]].merge(
        labels[["patient_id", "label"]], on="patient_id", validate="one_to_one"
    )
    metrics = ModelEvaluator().evaluate_proxy_auc(
        y_true=merged["label"].to_numpy(dtype=int),
        y_score=merged[score_column].to_numpy(dtype=float),
        n_bootstraps=n_bootstraps,
    )
    result: dict[str, object] = {
        **metrics,
        "label_definition": label_definition,
        "score_column": score_column,
        "evaluation_split": expected_split,
        "n_patients": int(len(merged)),
        "data_status": "CT_RATE_REPORT_LABEL_PROXY_NOT_CLINICAL_DIAGNOSIS",
        "clinical_claims_allowed": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    json.loads(temporary.read_text(encoding="utf-8"))
    temporary.replace(output_path)
    logger.warning(
        "Saved report-label proxy metrics for %d held-out patients. This is not clinical COPD diagnosis.",
        len(merged),
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate held-out CT-RATE report-label proxy metrics."
    )
    parser.add_argument("--patient-scores", type=Path, required=True)
    parser.add_argument(
        "--labels", type=Path, required=True, help="Reviewed CSV with patient_id,label[,split]."
    )
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/real/reports/report_label_proxy_metrics.json")
    )
    parser.add_argument(
        "--label-definition", required=True, help="Documented report-label rule/version."
    )
    parser.add_argument("--score-column", choices=sorted(ALLOWED_SCORES), default="anomaly_mean")
    parser.add_argument("--expected-split", choices=["test"], default="test")
    parser.add_argument("--bootstraps", type=int, default=500)
    args = parser.parse_args()
    evaluate_report_label_proxy(
        patient_scores_path=args.patient_scores,
        labels_path=args.labels,
        output_path=args.output,
        label_definition=args.label_definition,
        score_column=args.score_column,
        expected_split=args.expected_split,
        n_bootstraps=args.bootstraps,
    )


if __name__ == "__main__":
    main()
