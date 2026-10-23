"""Model evaluation, registration benchmarks, and proxy diagnostic metrics."""

from copd_anomaly.evaluation.metrics import (
    PROXY_DISCLAIMER,
    ModelEvaluator,
    compute_bootstrap_ci,
)

__all__ = [
    "ModelEvaluator",
    "compute_bootstrap_ci",
    "PROXY_DISCLAIMER",
]
