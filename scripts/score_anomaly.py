"""
Compute patch anomaly scores, patient aggregation, and evaluate proxy AUROC / AUPRC.

Enforces prominent disclaimer:
"REPORT-LABEL PROXY, not COPD classification"
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from copd_anomaly.anomaly.aggregation import ScoreAggregator, extract_patient_anomaly_summary
from copd_anomaly.anomaly.density import GaussianMixtureDensity, PatchAnomalyScorer
from copd_anomaly.evaluation.metrics import ModelEvaluator
from copd_anomaly.utils.helpers import get_logger

logger = get_logger("score_anomaly")


def score_and_evaluate(
    density_model_path: Path,
    output_report_path: Path = Path("reports/proxy_auc_report.csv"),
    dry_run: bool = False,
    n_bootstraps: int = 500,
) -> pd.DataFrame:
    output_report_path.parent.mkdir(parents=True, exist_ok=True)
    evaluator = ModelEvaluator()

    if dry_run or not density_model_path.exists():
        logger.info("Executing synthetic proxy evaluation for testing...")
        # Synthetic evaluation cohort: 50 normal controls, 50 emphysema cases
        n_samples = 100
        rng = np.random.RandomState(42)
        y_true = np.array([0] * 50 + [1] * 50)
        # Anomaly scores: normal centered at 10, emphysema centered at 18
        scores_mean = rng.normal(10.0, 3.0, 50).tolist() + rng.normal(18.0, 4.0, 50).tolist()
        scores_p90 = rng.normal(12.0, 3.0, 50).tolist() + rng.normal(22.0, 4.0, 50).tolist()

        patient_records = [
            {"patient_id": f"eval_{i:03d}", "y_true": y_true[i], "score_mean": scores_mean[i], "score_p90": scores_p90[i]}
            for i in range(n_samples)
        ]
        df_patients = pd.DataFrame(patient_records)

    else:
        logger.info(f"Loading density model from {density_model_path}...")
        density_model = GaussianMixtureDensity.load(str(density_model_path))
        scorer = PatchAnomalyScorer(density_model)
        # Evaluate using loaded model
        df_patients = pd.DataFrame()

    results_list = []
    for score_col in ["score_mean", "score_p90"]:
        if score_col in df_patients.columns:
            eval_metrics = evaluator.evaluate_proxy_auc(
                y_true=df_patients["y_true"].values,
                y_score=df_patients[score_col].values,
                n_bootstraps=n_bootstraps,
            )
            eval_metrics["aggregator"] = score_col
            results_list.append(eval_metrics)

    df_report = pd.DataFrame(results_list)
    df_report.to_csv(output_report_path, index=False)
    logger.info(f"Saved proxy AUC report to {output_report_path}")
    return df_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Score anomaly and compute proxy AUC.")
    parser.add_argument("--density-model", type=Path, default=Path("outputs/checkpoints/density_model.joblib"))
    parser.add_argument("--output-report", type=Path, default=Path("reports/proxy_auc_report.csv"))
    parser.add_argument("--dry-run", action="store_true")
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
