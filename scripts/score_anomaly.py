"""
Evaluate report-label proxy metrics.

This script currently supports an explicit synthetic smoke test only. Real inference
must be wired to a patient/patch manifest before a real evaluation report is produced.
Synthetic results are never labeled as clinical COPD classification.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from copd_anomaly.evaluation.metrics import ModelEvaluator
from copd_anomaly.utils.helpers import get_logger

logger = get_logger("score_anomaly")
PROXY_WARNING = (
    "SYNTHETIC DRY RUN — report-label proxy, not COPD classification; not real CT evaluation."
)


def score_and_evaluate(
    density_model_path: Path,
    output_report_path: Path = Path("reports/proxy_auc_report.csv"),
    dry_run: bool = False,
    n_bootstraps: int = 500,
) -> pd.DataFrame:
    if n_bootstraps < 1:
        raise ValueError("n_bootstraps must be >= 1.")
    output_parts = {part.lower() for part in output_report_path.parts}
    if dry_run and "real" in output_parts:
        raise ValueError("Synthetic evaluation reports cannot be written under a real output directory.")
    if not dry_run and "synthetic" in output_parts:
        raise ValueError("Real evaluation reports cannot be written under a synthetic output directory.")
    if dry_run:
        if output_report_path == Path("reports/proxy_auc_report.csv"):
            output_report_path = Path("outputs/synthetic/reports/proxy_auc_report.csv")
        logger.warning(PROXY_WARNING)
        rng = np.random.RandomState(42)
        y_true = np.array([0] * 50 + [1] * 50)
        scores_mean = np.concatenate([rng.normal(10.0, 3.0, 50), rng.normal(18.0, 4.0, 50)])
        scores_p90 = np.concatenate([rng.normal(12.0, 3.0, 50), rng.normal(22.0, 4.0, 50)])
        df_patients = pd.DataFrame(
            {
                "patient_id": [f"synthetic_{i:03d}" for i in range(100)],
                "y_true": y_true,
                "score_mean": scores_mean,
                "score_p90": scores_p90,
            }
        )
        metric_type = PROXY_WARNING
    else:
        if not density_model_path.is_file():
            raise FileNotFoundError(
                f"Density model not found: {density_model_path}. No report was written. "
                "Fit a real density model first, or use --dry-run for synthetic smoke testing."
            )
        raise NotImplementedError(
            "Real patient-level scoring is not implemented yet: the current script has no real "
            "patient/patch inference input. Refusing to write an empty or fabricated real evaluation. "
            "Use --dry-run only to test the metrics code."
        )

    evaluator = ModelEvaluator()
    results = []
    for score_col in ("score_mean", "score_p90"):
        metrics = evaluator.evaluate_proxy_auc(
            y_true=df_patients["y_true"].values,
            y_score=df_patients[score_col].values,
            n_bootstraps=n_bootstraps,
        )
        metrics["aggregator"] = score_col
        metrics["data_status"] = metric_type
        results.append(metrics)
    report = pd.DataFrame(results)
    output_report_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(output_report_path, index=False)
    logger.info("Saved explicitly synthetic proxy smoke-test report to %s", output_report_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate report-label proxy metrics.")
    parser.add_argument(
        "--density-model", type=Path, default=Path("outputs/real/checkpoints/density_model.joblib")
    )
    parser.add_argument("--output-report", type=Path, default=Path("reports/proxy_auc_report.csv"))
    parser.add_argument("--dry-run", action="store_true", help="Evaluate synthetic scores only")
    parser.add_argument("--bootstraps", type=int, default=500)
    args = parser.parse_args()
    score_and_evaluate(
        density_model_path=args.density_model,
        output_report_path=args.output_report,
        dry_run=args.dry_run,
        n_bootstraps=args.bootstraps,
    )


if __name__ == "__main__":
    main()
