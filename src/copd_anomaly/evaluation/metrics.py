"""
Evaluation Metrics: Proxy AUROC, AUPRC, and Bootstrap Confidence Intervals.

ENFORCEMENT NOTICE:
All classification metrics against radiology report abnormality classifications
are strictly PROXY benchmarks and MUST be labelled:
"REPORT-LABEL PROXY, not COPD classification".
Real COPD classification requires spirometry (FEV1/FVC < 0.70).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from copd_anomaly.interfaces import Evaluator
from copd_anomaly.registration.registrar import evaluate_tre
from copd_anomaly.utils.helpers import get_logger

logger = get_logger("evaluator")

PROXY_DISCLAIMER = (
    "DISCLAIMER: REPORT-LABEL PROXY, not COPD classification. "
    "Radiology report extraction does not substitute for spirometric confirmation (FEV1/FVC < 0.70)."
)


def compute_bootstrap_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric_fn: Any,
    n_bootstraps: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float, float]:
    """
    Computes metric and empirical bootstrap (1 - alpha) confidence interval.

    Returns:
        (point_estimate, ci_lower, ci_upper)
    """
    n_samples = len(y_true)
    point_est = float(metric_fn(y_true, y_score))

    rng = np.random.RandomState(seed)
    boot_scores = []

    for _ in range(n_bootstraps):
        boot_idx = rng.randint(0, n_samples, n_samples)
        y_t_b = y_true[boot_idx]
        y_s_b = y_score[boot_idx]

        # Check that both classes are present in the bootstrap resample
        if len(np.unique(y_t_b)) < 2:
            continue

        try:
            val = metric_fn(y_t_b, y_s_b)
            boot_scores.append(val)
        except Exception:
            continue

    if not boot_scores:
        return point_est, point_est, point_est

    lower_p = (alpha / 2.0) * 100.0
    upper_p = (1.0 - alpha / 2.0) * 100.0
    ci_lower = float(np.percentile(boot_scores, lower_p))
    ci_upper = float(np.percentile(boot_scores, upper_p))

    return point_est, ci_lower, ci_upper


class ModelEvaluator(Evaluator):
    """
    Evaluates registration accuracy and proxy anomaly detection performance.
    """

    def evaluate_tre(
        self,
        fixed_landmarks: np.ndarray,
        warped_landmarks: np.ndarray,
        spacing_zyx: tuple[float, float, float] = (1.0, 1.0, 1.0),
    ) -> dict[str, float]:
        """Computes Target Registration Error statistics in mm."""
        return evaluate_tre(fixed_landmarks, warped_landmarks, spacing_zyx=spacing_zyx)

    def evaluate_proxy_auc(
        self,
        y_true: np.ndarray,
        y_score: np.ndarray,
        n_bootstraps: int = 1000,
    ) -> dict[str, Any]:
        """
        Evaluates AUROC and AUPRC with 95% bootstrap confidence intervals.
        """
        y_true = np.asarray(y_true).astype(int)
        y_score = np.asarray(y_score).astype(float)

        if len(np.unique(y_true)) < 2:
            raise ValueError("y_true must contain at least two classes to evaluate AUROC/AUPRC.")

        # Compute point estimates and 95% CIs
        auc, auc_lo, auc_hi = compute_bootstrap_ci(
            y_true, y_score, roc_auc_score, n_bootstraps=n_bootstraps
        )
        pr_auc, pr_lo, pr_hi = compute_bootstrap_ci(
            y_true, y_score, average_precision_score, n_bootstraps=n_bootstraps
        )

        # Optimal Youden J threshold
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        j_scores = tpr - fpr
        opt_idx = int(np.argmax(j_scores))
        opt_thresh = float(thresholds[opt_idx])
        opt_sens = float(tpr[opt_idx])
        opt_spec = float(1.0 - fpr[opt_idx])

        results = {
            "disclaimer": PROXY_DISCLAIMER,
            "metric_type": "REPORT-LABEL PROXY, not COPD classification",
            "n_samples": len(y_true),
            "n_positives": int(np.sum(y_true == 1)),
            "n_negatives": int(np.sum(y_true == 0)),
            "auroc": auc,
            "auroc_ci95_low": auc_lo,
            "auroc_ci95_high": auc_hi,
            "auprc": pr_auc,
            "auprc_ci95_low": pr_lo,
            "auprc_ci95_high": pr_hi,
            "optimal_threshold": opt_thresh,
            "sensitivity_at_opt": opt_sens,
            "specificity_at_opt": opt_spec,
            "balanced_accuracy_at_opt": (opt_sens + opt_spec) / 2.0,
        }

        logger.info(
            f"Proxy Evaluation [{results['metric_type']}]:\n"
            f"  AUROC: {auc:.3f} (95% CI: [{auc_lo:.3f}, {auc_hi:.3f}])\n"
            f"  AUPRC: {pr_auc:.3f} (95% CI: [{pr_lo:.3f}, {pr_hi:.3f}])\n"
            f"  Optimal Threshold: {opt_thresh:.3f} (Sens: {opt_sens:.1%}, Spec: {opt_spec:.1%})"
        )
        return results
