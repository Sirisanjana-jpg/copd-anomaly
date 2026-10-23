"""
Lobe-Wise Quantitative Feature Builder.

Synthesizes volumetric patch anomaly scores, spatial lobe segmentations,
and Parametric Response Mapping into a comprehensive, tabular patient biomarker representation.
"""

from __future__ import annotations

import numpy as np

from copd_anomaly.interfaces import FeatureBuilder, PatchRecord
from copd_anomaly.utils.helpers import get_logger

logger = get_logger("lobe_builder")


class LobeFeatureBuilder(FeatureBuilder):
    """
    Constructs global, lobe-resolved, and spatial asymmetry features.
    """

    def build_features(
        self,
        patch_scores: np.ndarray,
        records: list[PatchRecord],
        prm_stats: dict[str, float] | None = None,
    ) -> dict[str, float]:
        """
        Builds complete quantitative feature vector.

        Returns:
            Dictionary containing global, lobe-level, and spatial heterogeneity metrics.
        """
        features: dict[str, float] = {}

        if len(patch_scores) == 0:
            logger.warning("Empty patch scores provided to build_features.")
            return features

        # Global Anomaly Scores
        features["global_anomaly_mean"] = float(np.mean(patch_scores))
        features["global_anomaly_std"] = float(np.std(patch_scores))
        features["global_anomaly_p90"] = float(np.percentile(patch_scores, 90))
        features["global_anomaly_p95"] = float(np.percentile(patch_scores, 95))
        features["global_anomaly_max"] = float(np.max(patch_scores))

        # Mean patch LAA-950 fraction
        laa_vals = [r.laa950_fraction for r in records]
        features["mean_patch_laa950"] = float(np.mean(laa_vals)) if laa_vals else 0.0

        # Lobe-Specific Anomaly Scores
        lobe_names = {1: "rul", 2: "rml", 3: "rll", 4: "lul", 5: "lll"}
        lobe_scores: dict[str, list[float]] = {name: [] for name in lobe_names.values()}

        for score, rec in zip(patch_scores, records):
            if rec.lobe_id in lobe_names:
                lobe_scores[lobe_names[rec.lobe_id]].append(score)

        lobe_means = {}
        for name, scores in lobe_scores.items():
            if scores:
                val = float(np.mean(scores))
                features[f"{name}_anomaly_mean"] = val
                features[f"{name}_anomaly_p90"] = float(np.percentile(scores, 90))
                lobe_means[name] = val
            else:
                features[f"{name}_anomaly_mean"] = features["global_anomaly_mean"]
                lobe_means[name] = features["global_anomaly_mean"]

        # Spatial Gradient: Upper / Lower Ratio
        # Upper: RUL + LUL; Lower: RLL + LLL
        upper_mean = (lobe_means["rul"] + lobe_means["lul"]) / 2.0
        lower_mean = (lobe_means["rll"] + lobe_means["lll"]) / 2.0
        features["upper_lower_ratio"] = float(upper_mean / (lower_mean + 1e-6))

        # Spatial Gradient: Left / Right Asymmetry
        right_mean = (lobe_means["rul"] + lobe_means["rml"] + lobe_means["rll"]) / 3.0
        left_mean = (lobe_means["lul"] + lobe_means["lll"]) / 2.0
        features["left_right_asymmetry"] = float(abs(right_mean - left_mean))

        # Spatial Heterogeneity Index (Standard deviation of the 5 lobe means)
        features["heterogeneity_index"] = float(np.std(list(lobe_means.values())))

        # Integrate PRM Stats if provided
        if prm_stats:
            for k, v in prm_stats.items():
                if isinstance(v, (int, float, np.floating, np.integer)):
                    features[f"prm_{k}"] = float(v)

        return features
