"""
Lobe-wise quantitative analysis entry point.

Real lobe-wise analysis must consume actual patch scores and validated lobe labels.
The current repository does not yet connect those inputs to this script, so real mode
fails explicitly instead of generating fabricated patient features.
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


def run_lobe_analysis(
    manifest_path: Path = Path("data/processed/dirlab_manifest.parquet"),
    prm_report_path: Path = Path("reports/dirlab_prm_stats.csv"),
    output_csv: Path = Path("reports/dirlab_lobe_features.csv"),
    dry_run: bool = False,
) -> pd.DataFrame:
    output_parts = {part.lower() for part in output_csv.parts}
    if dry_run and "real" in output_parts:
        raise ValueError("Synthetic lobe reports cannot be written under a real output directory.")
    if not dry_run and "synthetic" in output_parts:
        raise ValueError("Real lobe reports cannot be written under a synthetic output directory.")
    if not dry_run:
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"DIR-Lab manifest not found: {manifest_path}. No lobe report was written."
            )
        raise NotImplementedError(
            "Real lobe analysis is not wired yet: this script does not load real patch scores or "
            "validated lobe masks. Refusing to fabricate lobe IDs/scores. Complete segmentation, "
            "patch-to-lobe mapping, and real anomaly inference first."
        )

    logger.warning(
        "SYNTHETIC DRY RUN ONLY: generated patch scores/lobe IDs are artificial and are not "
        "DIR-Lab measurements."
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
    parser = argparse.ArgumentParser(description="Run lobe-wise quantitative feature analysis.")
    parser.add_argument(
        "--manifest", type=Path, default=Path("data/processed/dirlab_manifest.parquet")
    )
    parser.add_argument("--prm-report", type=Path, default=Path("reports/dirlab_prm_stats.csv"))
    parser.add_argument("--output", type=Path, default=Path("reports/dirlab_lobe_features.csv"))
    parser.add_argument(
        "--dry-run", action="store_true", help="Generate synthetic data for code smoke tests only"
    )
    args = parser.parse_args()
    run_lobe_analysis(
        manifest_path=args.manifest,
        prm_report_path=args.prm_report,
        output_csv=args.output,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
