"""Train an exploratory multi-label chest CT finding head from scan embeddings.

Embeddings must be extracted from CT image data (for example, pooled 3D encoder
patch embeddings). CT-RATE's 18 labels are report-derived pseudo-labels, not
radiologist-adjudicated voxel masks or independently confirmed diagnoses.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

CTRATE_LABELS = [
    "Medical material",
    "Arterial wall calcification",
    "Cardiomegaly",
    "Pericardial effusion",
    "Coronary artery wall calcification",
    "Hiatal hernia",
    "Lymphadenopathy",
    "Emphysema",
    "Atelectasis",
    "Lung nodule",
    "Lung opacity",
    "Pulmonary fibrotic sequela",
    "Pleural effusion",
    "Mosaic attenuation pattern",
    "Peribronchial thickening",
    "Consolidation",
    "Bronchiectasis",
    "Interlobular septal thickening",
]


def _find_key(frame: pd.DataFrame, candidates: tuple[str, ...], description: str) -> str:
    for name in candidates:
        if name in frame.columns:
            return name
    raise ValueError(f"Could not find {description} column. Tried: {candidates}.")


def _patient_group(volume_name: str) -> str:
    value = str(volume_name)
    # CT-RATE IDs typically look like train_123_a_1 / valid_53_a_1.
    match = re.match(r"^(train|valid|validation)_([0-9]+)(?:_|$)", value, re.IGNORECASE)
    if match:
        return f"{match.group(1).lower()}_{match.group(2)}"
    # If an upstream embedding table has patient IDs encoded in the scan key,
    # use its provided patient_id column instead of relying on this fallback.
    return value


def train_multilabel_head(
    embeddings_csv: Path,
    labels_csv: Path,
    output_dir: Path,
    test_size: float = 0.2,
    seed: int = 42,
) -> dict[str, object]:
    if not 0.1 <= test_size <= 0.4:
        raise ValueError("test_size must be between 0.1 and 0.4.")
    embeddings = pd.read_csv(embeddings_csv)
    labels = pd.read_csv(labels_csv)
    emb_key = _find_key(
        embeddings, ("VolumeName", "volume_name", "scan_id", "case_id"), "embedding scan ID"
    )
    label_key = _find_key(
        labels, ("VolumeName", "volume_name", "scan_id", "case_id"), "label scan ID"
    )
    missing_labels = [name for name in CTRATE_LABELS if name not in labels.columns]
    if missing_labels:
        raise ValueError(
            "Labels CSV is missing CT-RATE finding columns: " + ", ".join(missing_labels)
        )
    if embeddings[emb_key].duplicated().any():
        raise ValueError(
            f"Embeddings CSV has duplicate scan IDs in {emb_key}; aggregate patches to one row per scan."
        )
    if labels[label_key].duplicated().any():
        raise ValueError(
            f"Labels CSV has duplicate scan IDs in {label_key}; use the official per-volume label table."
        )

    patient_key = next(
        (name for name in ("patient_id", "PatientID", "patient") if name in embeddings.columns),
        None,
    )
    feature_cols = [
        name for name in embeddings.columns
        if name not in {emb_key, "patient_id", "PatientID", "patient", "split"}
        and pd.api.types.is_numeric_dtype(embeddings[name])
    ]
    if not feature_cols:
        raise ValueError("No numeric image embedding columns were found.")
    joined = embeddings.merge(
        labels[[label_key, *CTRATE_LABELS]],
        left_on=emb_key,
        right_on=label_key,
        how="inner",
        validate="one_to_one",
    )
    if len(joined) < 50:
        raise ValueError(
            f"Only {len(joined)} embedding/label rows matched; need at least 50 for this exploratory baseline. "
            "Check volume-name conventions and embedding coverage."
        )
    if patient_key:
        # Keep the patient key outside the model features and merge it by scan ID.
        group_lookup = embeddings.set_index(emb_key)[patient_key].astype(str)
        groups = joined[emb_key].map(group_lookup).astype(str)
    else:
        groups = joined[emb_key].map(_patient_group).astype(str)
    if groups.nunique() < 10:
        raise ValueError("At least 10 independent patient groups are required for a holdout.")
    y_frame = joined[CTRATE_LABELS].apply(pd.to_numeric, errors="coerce")
    if y_frame.isna().any().any():
        raise ValueError("Finding labels contain missing/non-numeric values; resolve label encoding first.")
    if not set(np.unique(y_frame.to_numpy())).issubset({0, 1}):
        raise ValueError("This baseline expects binary 0/1 CT-RATE pseudo-labels only.")
    if (y_frame.sum(axis=0) == 0).any():
        absent = list(y_frame.columns[y_frame.sum(axis=0) == 0])
        raise ValueError(f"These labels have no positive examples in the matched rows: {absent}")

    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(splitter.split(joined, groups=groups))
    train_frame, test_frame = joined.iloc[train_idx], joined.iloc[test_idx]
    y_train = train_frame[CTRATE_LABELS].astype(int)
    y_test = test_frame[CTRATE_LABELS].astype(int)
    if any(y_train[col].nunique() < 2 for col in CTRATE_LABELS):
        raise ValueError(
            "At least one finding has a single class in the training split. Use more cases or "
            "a prespecified patient-level split; do not split reconstructions from the same patient."
        )

    model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        OneVsRestClassifier(
            LogisticRegression(class_weight="balanced", max_iter=2000, random_state=seed)
        ),
    )
    model.fit(train_frame[feature_cols], y_train)
    probabilities = model.predict_proba(test_frame[feature_cols])
    per_label: dict[str, dict[str, float | None]] = {}
    aucs, aps = [], []
    for idx, name in enumerate(CTRATE_LABELS):
        truth = y_test[name].to_numpy()
        score = probabilities[:, idx]
        auc = float(roc_auc_score(truth, score)) if np.unique(truth).size == 2 else None
        ap = float(average_precision_score(truth, score)) if truth.sum() > 0 else None
        per_label[name] = {"roc_auc": auc, "average_precision": ap, "test_positive_count": int(truth.sum())}
        if auc is not None:
            aucs.append(auc)
        if ap is not None:
            aps.append(ap)

    report = {
        "task": "18-label chest CT finding classification from image embeddings",
        "label_source": "CT-RATE report-derived pseudo-labels",
        "n_matched_scans": int(len(joined)),
        "n_train_scans": int(len(train_frame)),
        "n_test_scans": int(len(test_frame)),
        "n_patient_groups": int(groups.nunique()),
        "embedding_features": feature_cols,
        "macro_roc_auc_over_evaluable_labels": float(np.mean(aucs)) if aucs else None,
        "macro_average_precision_over_evaluable_labels": float(np.mean(aps)) if aps else None,
        "per_label_metrics": per_label,
        "clinical_validation_status": "NOT_CLINICALLY_VALIDATED",
        "limitations": [
            "CT-RATE labels are extracted from reports and are noisy proxies.",
            "The random patient-grouped holdout is exploratory; use the official patient-disjoint validation cohort for final internal evaluation.",
            "Scan-level classification does not provide lesion localization or pixel/voxel segmentation.",
            "Predicted findings are research outputs, not autonomous clinical diagnoses.",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": model, "feature_columns": feature_cols, "label_columns": CTRATE_LABELS},
        output_dir / "ct_rate_multilabel_head.joblib",
    )
    prediction_frame = test_frame[[emb_key]].copy()
    prediction_frame = prediction_frame.rename(columns={emb_key: "volume_name"})
    for idx, name in enumerate(CTRATE_LABELS):
        prediction_frame[f"probability__{name}"] = probabilities[:, idx]
        prediction_frame[f"label__{name}"] = y_test[name].to_numpy()
    prediction_frame.to_csv(output_dir / "holdout_predictions.csv", index=False)
    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fit an exploratory 18-label CT-RATE classifier on image-derived scan embeddings."
    )
    parser.add_argument("--embeddings-csv", type=Path, required=True,
                        help="One row per CT volume, with scan ID and numeric image embedding columns.")
    parser.add_argument("--labels-csv", type=Path, required=True,
                        help="CT-RATE official train/validation multi-abnormality labels CSV.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/real/ct_rate_multilabel"))
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(train_multilabel_head(
        args.embeddings_csv, args.labels_csv, args.output_dir, args.test_size, args.seed
    ), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
