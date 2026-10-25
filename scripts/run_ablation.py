"""
Multimodal Feature Ablation Study on Synthetic COPD Cohort.

Evaluates 5 model configurations:
1. Clinical Covariates Only
2. Spirometry Only
3. Imaging Biomarkers Only (Anomaly + PRM + Spatial Gradients)
4. Clinical + Spirometry
5. Full Multimodal Fusion (Clinical + Spirometry + Imaging)

Computes cross-validated AUROC, AUPRC, Brier score, and delta p-values.
MANDATORY: Prominently banners all results as synthetic.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from copd_anomaly.synthetic.cohort_generator import (
    SYNTHETIC_BANNER,
    generate_synthetic_copd_cohort,
)
from copd_anomaly.utils.helpers import get_logger

logger = get_logger("run_ablation")

ABLATION_FEATURE_SETS = {
    "1_clinical_only": ["age", "sex_M", "bmi", "pack_years", "prior_exacerbations_count"],
    "2_spirometry_only": ["fev1_pct_predicted", "fvc_pct_predicted", "fev1_fvc_ratio"],
    "3_imaging_only": [
        "global_anomaly_score",
        "prm_emph_pct",
        "prm_fsad_pct",
        "upper_lower_ratio",
        "heterogeneity_index",
    ],
    "4_clinical_plus_spiro": [
        "age",
        "sex_M",
        "bmi",
        "pack_years",
        "prior_exacerbations_count",
        "fev1_pct_predicted",
        "fvc_pct_predicted",
        "fev1_fvc_ratio",
    ],
    "5_full_multimodal": [
        "age",
        "sex_M",
        "bmi",
        "pack_years",
        "prior_exacerbations_count",
        "fev1_pct_predicted",
        "fvc_pct_predicted",
        "fev1_fvc_ratio",
        "global_anomaly_score",
        "prm_emph_pct",
        "prm_fsad_pct",
        "upper_lower_ratio",
        "heterogeneity_index",
    ],
}


def run_ablation_experiments(
    cohort_path: Path = Path("data/processed/synthetic_cohort.parquet"),
    output_report: Path = Path("outputs/synthetic/reports/multimodal_ablation_results.csv"),
    n_splits: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    logger.info(SYNTHETIC_BANNER)
    if "real" in {part.lower() for part in output_report.parts}:
        raise ValueError(
            "Synthetic ablation reports cannot be written under a real output directory."
        )
    output_report.parent.mkdir(parents=True, exist_ok=True)

    if not cohort_path.exists():
        logger.warning(f"Cohort file not found at {cohort_path}. Generating on-the-fly...")
        df = generate_synthetic_copd_cohort(n_patients=500, seed=seed)
    else:
        df = pd.read_parquet(cohort_path)

    # Encode sex as binary indicator
    df["sex_M"] = (df["sex"] == "M").astype(float)
    y = df["exacerbation_12m_event"].values

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    ablation_predictions = {name: np.zeros(len(df)) for name in ABLATION_FEATURE_SETS}

    logger.info(
        f"Running {n_splits}-fold cross-validation across {len(ABLATION_FEATURE_SETS)} configurations..."
    )

    for train_idx, val_idx in cv.split(df, y):
        df_train, df_val = df.iloc[train_idx], df.iloc[val_idx]
        y_train = y[train_idx]

        for name, feats in ABLATION_FEATURE_SETS.items():
            scaler = StandardScaler()
            x_train = scaler.fit_transform(df_train[feats].values)
            x_val = scaler.transform(df_val[feats].values)

            clf = LogisticRegression(C=1.0, max_iter=200, random_state=seed)
            clf.fit(x_train, y_train)

            probs = clf.predict_proba(x_val)[:, 1]
            ablation_predictions[name][val_idx] = probs

    # Baseline predictions for statistical significance comparison
    baseline_preds = ablation_predictions["4_clinical_plus_spiro"]

    results = []
    for name in ABLATION_FEATURE_SETS:
        preds = ablation_predictions[name]
        auc = roc_auc_score(y, preds)
        pr_auc = average_precision_score(y, preds)
        brier = brier_score_loss(y, preds)

        # Paired t-test on log-loss residuals vs baseline
        res_baseline = (y - baseline_preds) ** 2
        res_current = (y - preds) ** 2
        diff = res_baseline - res_current
        t_stat, p_val = stats.ttest_1samp(diff, 0.0)

        results.append(
            {
                "banner": SYNTHETIC_BANNER,
                "configuration": name,
                "auroc": round(float(auc), 4),
                "auprc": round(float(pr_auc), 4),
                "brier_score": round(float(brier), 4),
                "delta_auroc_vs_baseline": round(float(auc - roc_auc_score(y, baseline_preds)), 4),
                "p_value_vs_baseline": round(float(p_val), 4)
                if name != "4_clinical_plus_spiro"
                else 1.0,
                "statistically_significant": bool(p_val < 0.05 and name != "4_clinical_plus_spiro"),
            }
        )
        logger.info(
            f"Config '{name}': AUROC = {auc:.4f}, AUPRC = {pr_auc:.4f}, Brier = {brier:.4f}"
        )

    df_results = pd.DataFrame(results)
    df_results.to_csv(output_report, index=False)
    logger.info(f"Saved multimodal ablation report to {output_report}")
    return df_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multimodal feature ablation experiments.")
    parser.add_argument(
        "--cohort", type=Path, default=Path("data/processed/synthetic_cohort.parquet")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("reports/multimodal_ablation_results.csv")
    )
    parser.add_argument("--splits", type=int, default=5)
    args = parser.parse_args()

    run_ablation_experiments(
        cohort_path=args.cohort, output_report=args.output, n_splits=args.splits
    )


if __name__ == "__main__":
    main()
