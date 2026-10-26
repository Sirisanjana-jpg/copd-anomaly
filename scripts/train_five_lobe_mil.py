"""Weakly supervised five-lobe multi-label CT finding model.

Input: one row per (scan, lobe), numeric image-derived features, and a scan-level
CT-RATE label CSV. A noisy-OR multiple-instance objective trains lobe scores from
scan-level labels. Per-lobe scores are *weak localization hypotheses*, not validated
radiologist localization. This script does not read raw CT or create lobe masks.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from torch import nn

from scripts.train_multilabel_head import CTRATE_LABELS

LOBE_ORDER = ("LUL", "LLL", "RUL", "RML", "RLL")
SCAN_KEYS = ("VolumeName", "volume_name", "scan_id", "case_id")


class NoisyOrLobeMIL(nn.Module):
    """Shared per-lobe linear head; noisy-OR combines five lobe probabilities."""

    def __init__(self, n_features: int, n_labels: int) -> None:
        super().__init__()
        self.lobe_head = nn.Linear(n_features, n_labels)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x shape: batch x 5 lobes x features
        lobe_logits = self.lobe_head(x)
        lobe_probs = torch.sigmoid(lobe_logits).clamp(1e-6, 1 - 1e-6)
        scan_probs = 1.0 - torch.prod(1.0 - lobe_probs, dim=1)
        return scan_probs.clamp(1e-6, 1 - 1e-6), lobe_probs


def _key(df: pd.DataFrame, what: str) -> str:
    for candidate in SCAN_KEYS:
        if candidate in df.columns:
            return candidate
    raise ValueError(f"Missing {what} scan-ID column; expected one of {SCAN_KEYS}.")


def train_lobe_mil(
    lobe_features_csv: Path,
    labels_csv: Path,
    output_dir: Path,
    epochs: int = 100,
    test_size: float = 0.2,
    seed: int = 42,
) -> dict[str, object]:
    if epochs < 1:
        raise ValueError("epochs must be >= 1")
    features = pd.read_csv(lobe_features_csv)
    labels = pd.read_csv(labels_csv)
    feature_scan_key = _key(features, "lobe feature")
    label_scan_key = _key(labels, "label")
    required = {"lobe_id"}
    if not required.issubset(features.columns):
        raise ValueError("Lobe feature CSV must contain lobe_id with LUL, LLL, RUL, RML, RLL.")
    missing = [name for name in CTRATE_LABELS if name not in labels.columns]
    if missing:
        raise ValueError("Labels CSV is missing CT-RATE finding columns: " + ", ".join(missing))
    features["lobe_id"] = features["lobe_id"].astype(str).str.upper()
    if not set(features["lobe_id"]).issubset(set(LOBE_ORDER)):
        raise ValueError(f"Unsupported lobe_id. Use exactly these IDs: {LOBE_ORDER}.")
    if features.duplicated([feature_scan_key, "lobe_id"]).any():
        raise ValueError("Duplicate scan/lobe rows found; expected exactly one row per lobe.")
    feature_cols = [
        c for c in features.columns
        if c not in {feature_scan_key, "lobe_id", "patient_id", "PatientID", "patient", "split"}
        and pd.api.types.is_numeric_dtype(features[c])
    ]
    if not feature_cols:
        raise ValueError("No numeric lobe-level feature columns found.")
    label_ids = labels[label_scan_key].astype(str)
    if label_ids.duplicated().any():
        raise ValueError("Duplicate scan IDs in labels CSV.")
    label_table = labels.copy()
    label_table[label_scan_key] = label_ids
    y_frame = label_table[CTRATE_LABELS].apply(pd.to_numeric, errors="coerce")
    if y_frame.isna().any().any() or not set(np.unique(y_frame.to_numpy())).issubset({0, 1}):
        raise ValueError("CT-RATE labels must be complete binary 0/1 values.")
    label_table[CTRATE_LABELS] = y_frame.astype(np.float32)
    scans = []
    target_ids = []
    group_ids = []
    patient_key = next((c for c in ("patient_id", "PatientID", "patient") if c in features), None)
    label_lookup = label_table.set_index(label_scan_key)
    matched = set(features[feature_scan_key].astype(str)) & set(label_table[label_scan_key].astype(str))
    for scan_id in sorted(matched):
        rows = features[features[feature_scan_key].astype(str) == scan_id].copy()
        if set(rows["lobe_id"]) != set(LOBE_ORDER) or len(rows) != 5:
            continue
        rows = rows.set_index("lobe_id").loc[list(LOBE_ORDER)]
        matrix = rows[feature_cols].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=np.float32)
        if not np.isfinite(matrix).all():
            continue
        scans.append(matrix)
        target_ids.append(label_lookup.loc[scan_id, CTRATE_LABELS].to_numpy(dtype=np.float32))
        group_ids.append(str(rows[patient_key].iloc[0]) if patient_key else scan_id)
    if len(scans) < 30:
        raise ValueError(f"Only {len(scans)} complete five-lobe scans matched; need at least 30.")
    x = np.stack(scans)
    y = np.stack(target_ids)
    groups = np.asarray(group_ids)
    if len(np.unique(groups)) < 10:
        raise ValueError("Need at least 10 independent patient groups.")
    train_idx, test_idx = next(GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed).split(x, y, groups))
    mean = x[train_idx].reshape(-1, x.shape[-1]).mean(axis=0)
    std = x[train_idx].reshape(-1, x.shape[-1]).std(axis=0)
    std[std < 1e-6] = 1.0
    x = np.nan_to_num((x - mean) / std, nan=0.0, posinf=0.0, neginf=0.0)
    torch.manual_seed(seed)
    model = NoisyOrLobeMIL(x.shape[-1], len(CTRATE_LABELS))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=1e-3)
    xt = torch.tensor(x, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32)
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        probs, _ = model(xt[train_idx])
        loss = nn.functional.binary_cross_entropy(probs, yt[train_idx])
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        scan_probs, lobe_probs = model(xt[test_idx])
    p = scan_probs.numpy()
    lp = lobe_probs.numpy()
    per_label = {}
    aucs, aps = [], []
    for j, label in enumerate(CTRATE_LABELS):
        truth = y[test_idx, j]
        auc = float(roc_auc_score(truth, p[:, j])) if np.unique(truth).size == 2 else None
        ap = float(average_precision_score(truth, p[:, j])) if truth.sum() else None
        per_label[label] = {"roc_auc": auc, "average_precision": ap, "test_positive_count": int(truth.sum())}
        if auc is not None:
            aucs.append(auc)
        if ap is not None:
            aps.append(ap)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(), "feature_columns": feature_cols,
        "label_columns": CTRATE_LABELS, "lobe_order": LOBE_ORDER,
        "feature_mean": mean.tolist(), "feature_std": std.tolist(),
    }, output_dir / "five_lobe_mil.pt")
    loc_rows = []
    # Keep output ordering aligned to the scan tensors, rather than the raw CSV.
    valid_ids = []
    for scan_id in sorted(matched):
        rows = features[features[feature_scan_key].astype(str) == scan_id]
        if set(rows["lobe_id"]) == set(LOBE_ORDER) and len(rows) == 5:
            rows = rows.set_index("lobe_id").loc[list(LOBE_ORDER)]
            mat = rows[feature_cols].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=np.float32)
            if np.isfinite(mat).all():
                valid_ids.append(scan_id)
    for k, scan_index in enumerate(test_idx):
        for lobe_idx, lobe in enumerate(LOBE_ORDER):
            row = {"volume_name": valid_ids[scan_index], "lobe_id": lobe}
            row.update({f"weak_probability__{name}": float(lp[k, lobe_idx, j]) for j, name in enumerate(CTRATE_LABELS)})
            loc_rows.append(row)
    pd.DataFrame(loc_rows).to_csv(output_dir / "holdout_lobe_hypotheses.csv", index=False)
    report = {
        "task": "five-lobe weakly supervised multi-label CT finding prediction",
        "label_source": "CT-RATE scan/report-level pseudo-labels",
        "pooling": "noisy-OR over five lobe-level probabilities",
        "lobe_order": list(LOBE_ORDER), "feature_columns": feature_cols,
        "n_matched_complete_scans": len(scans), "n_train_scans": len(train_idx),
        "n_test_scans": len(test_idx), "n_patient_groups": int(len(np.unique(groups))),
        "macro_roc_auc_over_evaluable_labels": float(np.mean(aucs)) if aucs else None,
        "macro_average_precision_over_evaluable_labels": float(np.mean(aps)) if aps else None,
        "per_label_metrics": per_label,
        "localization_status": "WEAK_LOCALIZATION_HYPOTHESES_NOT_RADIOLOGIST_VALIDATED",
        "clinical_validation_status": "NOT_CLINICALLY_VALIDATED",
        "limitations": [
            "Training labels are scan-level report-derived pseudo-labels, not lobe annotations.",
            "Lobe probabilities are weak localization hypotheses; do not interpret them as validated lesion locations.",
            "This model consumes precomputed lobe features; it does not read raw CT or segment lobes.",
            "Clinical COPD severity requires independent clinical reference outcomes such as spirometry.",
        ],
    }
    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lobe-features-csv", type=Path, required=True,
                        help="One row per scan/lobe with scan_id, lobe_id, optional patient_id and numeric features.")
    parser.add_argument("--labels-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/real/five_lobe_mil"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(train_lobe_mil(args.lobe_features_csv, args.labels_csv, args.output_dir,
                                    args.epochs, args.test_size, args.seed), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
