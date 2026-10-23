"""
Patch-to-Patient Anomaly Score Aggregators.

Maps fine-grained 3D volumetric patch anomaly scores s(x) to a holistic
scan-level anomaly severity index S(X).
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from copd_anomaly.interfaces import Aggregator


class ScoreAggregator(Aggregator):
    """Unified patch score aggregator supporting multiple pooling functions."""

    def __init__(
        self,
        method: Literal["mean", "p90", "p95", "top_k", "weighted", "lobe_max"] = "p90",
        top_k: int = 10,
    ) -> None:
        self.method = method
        self.top_k = top_k

    def aggregate(
        self,
        patch_scores: np.ndarray,
        volume_weights: np.ndarray | None = None,
        lobe_ids: np.ndarray | None = None,
    ) -> float:
        """
        Aggregates patch anomaly scores into a patient-level scalar score.

        Args:
            patch_scores: (N_patches,) float32 anomaly scores.
            volume_weights: Optional (N_patches,) sampling weights.
            lobe_ids: Optional (N_patches,) lobe assignments (1..5).

        Returns:
            Scalar anomaly score S(X).
        """
        if len(patch_scores) == 0:
            return 0.0

        if self.method == "mean":
            return float(np.mean(patch_scores))
        if self.method == "p90":
            return float(np.percentile(patch_scores, 90))
        if self.method == "p95":
            return float(np.percentile(patch_scores, 95))
        if self.method == "top_k":
            k = min(self.top_k, len(patch_scores))
            top_k_vals = np.partition(patch_scores, -k)[-k:]
            return float(np.mean(top_k_vals))
        if self.method == "weighted":
            if volume_weights is not None and np.sum(volume_weights) > 0:
                return float(np.average(patch_scores, weights=volume_weights))
            return float(np.mean(patch_scores))
        if self.method == "lobe_max":
            if lobe_ids is not None and np.any(lobe_ids > 0):
                lobe_means = []
                for lobe_id in range(1, 6):
                    in_lobe = lobe_ids == lobe_id
                    if np.any(in_lobe):
                        lobe_means.append(float(np.mean(patch_scores[in_lobe])))
                if lobe_means:
                    return float(np.max(lobe_means))
            return float(np.mean(patch_scores))

        return float(np.mean(patch_scores))


def extract_patient_anomaly_summary(
    patch_scores: np.ndarray,
    lobe_ids: np.ndarray | None = None,
) -> dict[str, float]:
    """Extracts multi-faceted patient-level anomaly score metrics."""
    if len(patch_scores) == 0:
        return {}

    summary = {
        "anomaly_mean": float(np.mean(patch_scores)),
        "anomaly_std": float(np.std(patch_scores)),
        "anomaly_median": float(np.median(patch_scores)),
        "anomaly_p90": float(np.percentile(patch_scores, 90)),
        "anomaly_p95": float(np.percentile(patch_scores, 95)),
        "anomaly_max": float(np.max(patch_scores)),
    }

    if lobe_ids is not None and np.any(lobe_ids > 0):
        lobe_means = {}
        for lobe_id, name in enumerate(["rul", "rml", "rll", "lul", "lll"], start=1):
            mask = lobe_ids == lobe_id
            if np.any(mask):
                val = float(np.mean(patch_scores[mask]))
                summary[f"{name}_anomaly_mean"] = val
                lobe_means[name] = val

        if (
            "rul" in lobe_means
            and "lul" in lobe_means
            and "rll" in lobe_means
            and "lll" in lobe_means
        ):
            upper_mean = (lobe_means["rul"] + lobe_means["lul"]) / 2.0
            lower_mean = (lobe_means["rll"] + lobe_means["lll"]) / 2.0
            summary["anomaly_upper_lower_ratio"] = float(upper_mean / (lower_mean + 1e-6))

        if len(lobe_means) == 5:
            summary["anomaly_heterogeneity_index"] = float(np.std(list(lobe_means.values())))

    return summary

