"""CT-only COPD research assessment from inspiratory CT and five-lobe masks.

Computes quantitative CT features without requiring expiratory scans. Optional
classification requires an explicitly declared label source; report-derived labels
are weak proxies and must never be described as clinical COPD diagnosis.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import nibabel as nib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Research-only inspiratory CT emphysema bands reported in published cohorts.
# These are NOT GOLD grades and are not diagnostic cutoffs.
def ct_emphysema_band(laa950_pct: float) -> str:
    if laa950_pct < 6.0:
        return "low_ct_emphysema_burden_lt6pct"
    if laa950_pct < 14.0:
        return "intermediate_ct_emphysema_burden_6_to_lt14pct"
    return "high_ct_emphysema_burden_ge14pct"


LOBE_NAMES = {
    1: "right_upper",
    2: "right_middle",
    3: "right_lower",
    4: "left_upper",
    5: "left_lower",
}


def _case_id(path: Path) -> str:
    name = path.name
    for suffix in ("_insp.nii.gz", ".nii.gz", ".nii"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def _summarize_region(hu: np.ndarray, spacing_mm: tuple[float, float, float]) -> dict[str, float]:
    values = np.asarray(hu, dtype=np.float32)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("A lung/lobe region is empty or contains no finite CT values.")
    voxel_ml = float(np.prod(spacing_mm) / 1000.0)
    return {
        "voxel_count": int(values.size),
        "volume_ml": float(values.size * voxel_ml),
        "mean_hu": float(np.mean(values)),
        "median_hu": float(np.median(values)),
        "p15_hu": float(np.percentile(values, 15)),
        "std_hu": float(np.std(values)),
        "laa950_pct": float(np.mean(values <= -950.0) * 100.0),
        "laa910_pct": float(np.mean(values <= -910.0) * 100.0),
    }


def extract_features(ct_dir: Path, lobe_dir: Path, output_csv: Path) -> pd.DataFrame:
    if not ct_dir.is_dir():
        raise FileNotFoundError(f"CT directory not found: {ct_dir}")
    if not lobe_dir.is_dir():
        raise FileNotFoundError(f"Lobe-mask directory not found: {lobe_dir}")
    scans = sorted(ct_dir.glob("*.nii.gz"))
    scans = [p for p in scans if not p.name.endswith("_lobes.nii.gz")]
    if not scans:
        raise FileNotFoundError(f"No .nii.gz CT scans found under {ct_dir}")

    records: list[dict[str, object]] = []
    for ct_path in scans:
        case_id = _case_id(ct_path)
        mask_path = lobe_dir / f"{case_id}_lobes.nii.gz"
        if not mask_path.is_file():
            raise FileNotFoundError(
                f"Missing five-lobe mask for {ct_path.name}: expected {mask_path}. "
                "Generate and quality-check masks before quantitative assessment."
            )
        ct_img = nib.load(str(ct_path))
        mask_img = nib.load(str(mask_path))
        if ct_img.shape[:3] != mask_img.shape[:3] or not np.allclose(
            ct_img.affine, mask_img.affine, rtol=0.0, atol=1e-3
        ):
            raise ValueError(f"CT/mask geometry mismatch for case {case_id}.")
        ct = ct_img.get_fdata(dtype=np.float32)
        labels_float = mask_img.get_fdata(dtype=np.float32)
        if ct.ndim != 3 or labels_float.ndim != 3:
            raise ValueError(f"Expected 3D CT and mask volumes for case {case_id}.")
        if not np.isfinite(ct).all() or not np.isfinite(labels_float).all():
            raise ValueError(f"Non-finite CT or lobe labels for case {case_id}.")
        if not np.equal(labels_float, np.round(labels_float)).all():
            raise ValueError(f"Non-integer lobe labels for case {case_id}.")
        labels = labels_float.astype(np.uint8)
        present = set(np.unique(labels).astype(int)) - {0}
        if present != set(LOBE_NAMES):
            raise ValueError(
                f"Expected lobe IDs 1-5 for {case_id}; found {sorted(present)}."
            )
        # HU values outside a plausible CT range usually indicate an invalid conversion.
        lung_values = ct[labels > 0]
        if lung_values.size == 0 or np.percentile(lung_values, 1) < -1200 or np.percentile(
            lung_values, 99
        ) > 1500:
            raise ValueError(
                f"CT intensities for {case_id} do not look like calibrated HU. "
                "Check the DICOM/NIfTI conversion and rescale slope/intercept."
            )

        spacing = tuple(float(v) for v in ct_img.header.get_zooms()[:3])
        row: dict[str, object] = {"case_id": case_id, "ct_path": str(ct_path)}
        whole_features = _summarize_region(ct[labels > 0], spacing)
        row.update({f"whole_lung_{k}": v for k, v in whole_features.items()})
        row["whole_lung_ct_emphysema_band"] = ct_emphysema_band(whole_features["laa950_pct"])
        for lobe_id, lobe_name in LOBE_NAMES.items():
            lobe_features = _summarize_region(ct[labels == lobe_id], spacing)
            row.update({f"{lobe_name}_{k}": v for k, v in lobe_features.items()})
            row[f"{lobe_name}_ct_emphysema_band"] = ct_emphysema_band(
                lobe_features["laa950_pct"]
            )
        row["severity_label_scope"] = "CT_EMPHYSEMA_BURDEN_ONLY_NOT_GOLD"
        records.append(row)

    frame = pd.DataFrame(records)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_csv, index=False)
    return frame


def train_labelled_baseline(
    features: pd.DataFrame,
    labels_csv: Path,
    output_dir: Path,
    label_source: str,
) -> dict[str, object]:
    """Fit a simple patient-disjoint baseline; not a clinically validated model."""
    if label_source not in {"spirometry", "report"}:
        raise ValueError("label_source must be 'spirometry' or 'report'.")
    labels = pd.read_csv(labels_csv)
    required = {"case_id", "copd_label"}
    if not required.issubset(labels.columns):
        raise ValueError(f"Labels CSV must contain columns: {sorted(required)}.")
    if labels["case_id"].duplicated().any():
        raise ValueError("Labels CSV must have one row per case_id; resolve duplicates first.")
    if not set(pd.to_numeric(labels["copd_label"], errors="coerce").dropna().unique()).issubset(
        {0, 1}
    ):
        raise ValueError("copd_label must contain only binary values 0 or 1.")
    labels["copd_label"] = pd.to_numeric(labels["copd_label"], errors="raise").astype(int)
    merged = features.merge(labels, on="case_id", how="inner", validate="one_to_one")
    if len(merged) < 20 or merged["copd_label"].nunique() != 2:
        raise ValueError(
            "At least 20 labelled cases and both classes are required for this exploratory "
            "holdout. More data will usually be needed for reliable estimates."
        )
    group_col = "patient_id" if "patient_id" in merged.columns else "case_id"
    groups = merged[group_col].astype(str)
    if groups.nunique() < 8:
        raise ValueError("Too few independent patients for a patient-disjoint holdout.")
    feature_cols = [
        col for col in features.columns
        if col not in {"case_id", "ct_path"} and pd.api.types.is_numeric_dtype(features[col])
    ]
    if not feature_cols:
        raise ValueError("No numeric CT features found for modelling.")

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    train_idx, test_idx = next(splitter.split(merged, merged["copd_label"], groups))
    train, test = merged.iloc[train_idx], merged.iloc[test_idx]
    if train["copd_label"].nunique() != 2 or test["copd_label"].nunique() != 2:
        raise ValueError(
            "The patient-disjoint split did not contain both classes in train and test. "
            "Collect more patients or create a prespecified split; do not fall back to random "
            "scan-level splitting."
        )

    model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42),
    )
    model.fit(train[feature_cols], train["copd_label"])
    probabilities = model.predict_proba(test[feature_cols])[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(test["copd_label"], predictions, labels=[0, 1]).ravel()
    metrics = {
        "task": "report_label_proxy_classification" if label_source == "report"
        else "spirometry_labelled_internal_holdout",
        "label_source": label_source,
        "n_total_cases": int(len(merged)),
        "n_train_cases": int(len(train)),
        "n_test_cases": int(len(test)),
        "n_independent_patients": int(groups.nunique()),
        "patient_group_column": group_col,
        "features": feature_cols,
        "roc_auc": float(roc_auc_score(test["copd_label"], probabilities)),
        "average_precision": float(average_precision_score(test["copd_label"], probabilities)),
        "balanced_accuracy": float(balanced_accuracy_score(test["copd_label"], predictions)),
        "sensitivity_at_0_5": float(tp / (tp + fn)) if tp + fn else None,
        "specificity_at_0_5": float(tn / (tn + fp)) if tn + fp else None,
        "brier_score": float(brier_score_loss(test["copd_label"], probabilities)),
        "clinical_validation_status": "NOT_CLINICALLY_VALIDATED",
        "note": (
            "Report-derived labels are weak proxies, not clinical diagnoses."
            if label_source == "report"
            else "Internal holdout only; independent external validation is still required."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": model, "feature_columns": feature_cols, "label_source": label_source},
        output_dir / "ct_only_baseline.joblib",
    )
    pd.DataFrame(
        {
            "case_id": test["case_id"].values,
            "y_true": test["copd_label"].values,
            "probability": probabilities,
            "prediction_at_0_5": predictions,
            "label_source": label_source,
        }
    ).to_csv(output_dir / "holdout_predictions.csv", index=False)
    (output_dir / "evaluation_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Quantify inspiratory CT emphysema features per lung lobe; optional research baseline."
    )
    parser.add_argument("--ct-dir", type=Path, required=True, help="Directory of calibrated CT .nii.gz files.")
    parser.add_argument("--lobe-dir", type=Path, required=True, help="Directory of matching *_lobes.nii.gz masks.")
    parser.add_argument("--output-csv", type=Path, default=Path("outputs/real/reports/ct_lobe_features.csv"))
    parser.add_argument("--labels-csv", type=Path, help="Optional CSV: case_id,copd_label[,patient_id].")
    parser.add_argument("--label-source", choices=["spirometry", "report"], default=None)
    parser.add_argument("--model-output-dir", type=Path, default=Path("outputs/real/reports/ct_only_model"))
    args = parser.parse_args()
    if bool(args.labels_csv) != bool(args.label_source):
        parser.error("--labels-csv and --label-source must be supplied together.")
    features = extract_features(args.ct_dir, args.lobe_dir, args.output_csv)
    print(f"Wrote CT-only quantitative features for {len(features)} scans to {args.output_csv}")
    if args.labels_csv:
        metrics = train_labelled_baseline(features, args.labels_csv, args.model_output_dir, args.label_source)
        print(json.dumps(metrics, indent=2))
    else:
        print(
            "No labels supplied: no diagnostic classifier was trained. "
            "These outputs quantify CT appearance and are not a COPD diagnosis."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
