"""Build lobe-wise features from real patch anomaly scores and validated lobe IDs.

Real mode never invents scores or lobe IDs. It requires the output of the real patch
inference command and a complete five-lobe assignment for each analyzed scan.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from copd_anomaly.features.lobe_builder import LobeFeatureBuilder
from copd_anomaly.interfaces import PatchRecord
from copd_anomaly.utils.helpers import get_logger

logger = get_logger("lobe_analysis")
LOBE_NAMES = {1: "rul", 2: "rml", 3: "rll", 4: "lul", 5: "lll"}


def run_lobe_analysis(
    manifest_path: Path = Path("data/processed/dirlab_manifest.parquet"),
    prm_report_path: Path = Path("reports/dirlab_prm_stats.csv"),
    output_csv: Path = Path("outputs/real/reports/lobe_features.csv"),
    dry_run: bool = False,
    patch_scores_path: Path = Path("outputs/real/reports/patch_anomaly_scores.csv"),
) -> pd.DataFrame:
    output_parts = {part.lower() for part in output_csv.parts}
    if dry_run and "real" in output_parts:
        raise ValueError("Synthetic lobe reports cannot be written under a real output directory.")
    if not dry_run and "synthetic" in output_parts:
        raise ValueError("Real lobe reports cannot be written under a synthetic output directory.")

    if not dry_run:
        if not patch_scores_path.is_file():
            raise FileNotFoundError(
                f"Real patch-score table not found: {patch_scores_path}. Run "
                "scripts/score_anomaly.py --real-inference first."
            )
        scores_df = pd.read_csv(patch_scores_path)
        required = {
            "patch_id", "patient_id", "scan_id", "lobe_id", "lung_fraction",
            "volume_weight", "laa950_fraction", "anomaly_score",
        }
        missing = required - set(scores_df.columns)
        if missing:
            raise ValueError(f"Patch-score table is missing required columns: {sorted(missing)}.")
        if scores_df.empty:
            raise ValueError("Patch-score table is empty; no lobe report was written.")
        if scores_df["patch_id"].astype(str).duplicated().any():
            raise ValueError("Patch-score table contains duplicate patch IDs.")
        score_values = pd.to_numeric(scores_df["anomaly_score"], errors="coerce").to_numpy(float)
        if not np.isfinite(score_values).all():
            raise ValueError("Patch-score table contains non-finite anomaly scores.")
        lobe_values = pd.to_numeric(scores_df["lobe_id"], errors="coerce").to_numpy(float)
        if not np.isfinite(lobe_values).all() or not np.equal(lobe_values, np.floor(lobe_values)).all():
            raise ValueError("Patch-score table contains invalid lobe IDs.")
        if not set(lobe_values.astype(int)).issubset({1, 2, 3, 4, 5}):
            raise ValueError(
                "Real lobe analysis requires a validated five-lobe mask for every patch; "
                "found unknown/unassigned lobe IDs. Rebuild patches with --lobe-masks-dir."
            )

        prm_by_case: dict[str, dict[str, float]] = {}
        if prm_report_path.is_file():
            prm_df = pd.read_csv(prm_report_path)
            if "case_id" in prm_df.columns:
                for _, prm_row in prm_df.iterrows():
                    prm_by_case[str(prm_row["case_id"])] = {
                        key: float(value)
                        for key, value in prm_row.items()
                        if key != "case_id" and isinstance(value, (int, float, np.number))
                        and pd.notna(value)
                    }

        builder = LobeFeatureBuilder()
        records_by_scan = []
        for (patient_id, scan_id), group in scores_df.groupby(["patient_id", "scan_id"], sort=True):
            observed = set(pd.to_numeric(group["lobe_id"]).astype(int))
            if observed != {1, 2, 3, 4, 5}:
                raise ValueError(
                    f"Scan {scan_id} has patches in lobes {sorted(observed)}; all five labels "
                    "must be represented before emitting a complete lobe-wise feature row."
                )
            records = []
            for _, row in group.iterrows():
                center = (
                    int(row["center_z"]) if "center_z" in row and pd.notna(row["center_z"]) else 0,
                    int(row["center_y"]) if "center_y" in row and pd.notna(row["center_y"]) else 0,
                    int(row["center_x"]) if "center_x" in row and pd.notna(row["center_x"]) else 0,
                )
                records.append(
                    PatchRecord(
                        patch_id=str(row["patch_id"]),
                        patient_id=str(patient_id),
                        scan_id=str(scan_id),
                        center_zyx=center,
                        lobe_id=int(row["lobe_id"]),
                        lung_fraction=float(row["lung_fraction"]),
                        volume_weight=float(row["volume_weight"]),
                        laa950_fraction=float(row["laa950_fraction"]),
                    )
                )
            case_id = str(scan_id).split("_")[0]
            features = builder.build_features(
                group["anomaly_score"].to_numpy(dtype=float),
                records,
                prm_by_case.get(case_id, {}),
            )
            features["patient_id"] = str(patient_id)
            features["scan_id"] = str(scan_id)
            features["data_status"] = "REAL_CT_PATCH_INFERENCE_UNVALIDATED"
            records_by_scan.append(features)

        df = pd.DataFrame(records_by_scan)
        if df.empty:
            raise ValueError("No scan-level feature rows were created.")
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        temp_path = output_csv.with_suffix(".csv.tmp")
        df.to_csv(temp_path, index=False)
        pd.read_csv(temp_path, nrows=1)
        temp_path.replace(output_csv)
        logger.info("Saved %d real scan-level lobe feature rows to %s", len(df), output_csv)
        return df

    logger.warning(
        "SYNTHETIC DRY RUN ONLY: generated scores and lobe IDs are artificial, not patient data."
    )
    output_csv = (
        Path("outputs/synthetic/reports/dirlab_lobe_features.csv")
        if output_csv == Path("reports/dirlab_lobe_features.csv")
        else output_csv
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    builder = LobeFeatureBuilder()
    rng = np.random.RandomState(42)
    records_by_case = []
    for case_idx in range(1, 11):
        case_id = f"synthetic_copd{case_idx}"
        n_patches = 50
        scores = rng.normal(loc=12.0 + (case_idx % 3) * 3, scale=3.0, size=n_patches)
        records = [
            PatchRecord(
                patch_id=f"{case_id}_p{idx:03d}",
                patient_id=case_id,
                scan_id=f"{case_id}_synthetic",
                center_zyx=(20 + (idx % 20), 100, 100),
                lobe_id=(idx % 5) + 1,
                lung_fraction=0.85,
                volume_weight=1.0,
                laa950_fraction=float(rng.uniform(0.01, 0.15)),
            )
            for idx in range(n_patches)
        ]
        features = builder.build_features(scores, records, {})
        features["case_id"] = case_id
        features["data_status"] = "SYNTHETIC_DRY_RUN_NOT_REAL_DIRLAB"
        records_by_case.append(features)
    df = pd.DataFrame(records_by_case)
    df = df[["case_id"] + [col for col in df.columns if col != "case_id"]]
    df.to_csv(output_csv, index=False)
    logger.info("Saved synthetic smoke-test lobe report to %s", output_csv)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Build lobe-wise features from patch anomaly scores.")
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/dirlab_manifest.parquet"))
    parser.add_argument("--prm-report", type=Path, default=Path("reports/dirlab_prm_stats.csv"))
    parser.add_argument("--patch-scores", type=Path, default=Path("outputs/real/reports/patch_anomaly_scores.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs/real/reports/lobe_features.csv"))
    parser.add_argument("--dry-run", action="store_true", help="Generate synthetic data for code tests only")
    args = parser.parse_args()
    run_lobe_analysis(
        manifest_path=args.manifest,
        prm_report_path=args.prm_report,
        output_csv=args.output,
        dry_run=args.dry_run,
        patch_scores_path=args.patch_scores,
    )


if __name__ == "__main__":
    main()
