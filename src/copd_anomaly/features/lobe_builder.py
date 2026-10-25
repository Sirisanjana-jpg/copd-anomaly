"""
Lobe-wise quantitative feature builder.

Missing lobe coverage is represented explicitly; it is never silently imputed with
the global score, which could disguise failed or absent anatomical segmentation.
"""
from __future__ import annotations

import numpy as np

from copd_anomaly.interfaces import FeatureBuilder, PatchRecord
from copd_anomaly.utils.helpers import get_logger

logger = get_logger("lobe_builder")


class LobeFeatureBuilder(FeatureBuilder):
    """Construct global, lobe-resolved, and spatial asymmetry features."""

    def build_features(
        self,
        patch_scores: np.ndarray,
        records: list[PatchRecord],
        prm_stats: dict[str, float] | None = None,
    ) -> dict[str, float]:
        features: dict[str, float] = {}
        scores = np.asarray(patch_scores, dtype=float)
        if scores.size == 0:
            logger.warning("Empty patch scores provided to build_features.")
            return features
        if scores.ndim != 1 or len(scores) != len(records):
            raise ValueError(
                f"patch_scores must be 1-D and aligned with records; got {scores.shape} scores "
                f"and {len(records)} records."
            )
        if not np.isfinite(scores).all():
            raise ValueError("patch_scores contains NaN or infinite values.")

        features["global_anomaly_mean"] = float(np.mean(scores))
        features["global_anomaly_std"] = float(np.std(scores))
        features["global_anomaly_p90"] = float(np.percentile(scores, 90))
        features["global_anomaly_p95"] = float(np.percentile(scores, 95))
        features["global_anomaly_max"] = float(np.max(scores))

        laa_vals = [r.laa950_fraction for r in records if r.laa950_fraction is not None]
        features["mean_patch_laa950"] = float(np.mean(laa_vals)) if laa_vals else float("nan")

        lobe_names = {1: "rul", 2: "rml", 3: "rll", 4: "lul", 5: "lll"}
        lobe_scores: dict[str, list[float]] = {name: [] for name in lobe_names.values()}
        for score, record in zip(scores, records, strict=True):
            if record.lobe_id in lobe_names:
                lobe_scores[lobe_names[record.lobe_id]].append(float(score))

        lobe_means: dict[str, float] = {}
        present_lobes = []
        for lobe_id, name in lobe_names.items():
            vals = lobe_scores[name]
            features[f"{name}_patch_count"] = len(vals)
            features[f"{name}_present"] = int(bool(vals))
            if vals:
                lobe_means[name] = float(np.mean(vals))
                features[f"{name}_anomaly_mean"] = lobe_means[name]
                features[f"{name}_anomaly_p90"] = float(np.percentile(vals, 90))
                present_lobes.append(name)
            else:
                lobe_means[name] = float("nan")
                features[f"{name}_anomaly_mean"] = float("nan")
                features[f"{name}_anomaly_p90"] = float("nan")
                logger.warning("No valid patches assigned to lobe %s (label %d).", name, lobe_id)

        features["lobes_observed_count"] = len(present_lobes)
        # Spatial summaries are meaningful only when all constituent lobes are observed.
        if all(name in present_lobes for name in ("rul", "lul", "rll", "lll")):
            upper = (lobe_means["rul"] + lobe_means["lul"]) / 2.0
            lower = (lobe_means["rll"] + lobe_means["lll"]) / 2.0
            features["upper_lower_ratio"] = float(upper / lower) if lower != 0 else float("nan")
        else:
            features["upper_lower_ratio"] = float("nan")
        if all(name in present_lobes for name in ("rul", "rml", "rll", "lul", "lll")):
            right = (lobe_means["rul"] + lobe_means["rml"] + lobe_means["rll"]) / 3.0
            left = (lobe_means["lul"] + lobe_means["lll"]) / 2.0
            features["left_right_asymmetry"] = float(abs(right - left))
            features["heterogeneity_index"] = float(np.std(list(lobe_means.values())))
        else:
            features["left_right_asymmetry"] = float("nan")
            features["heterogeneity_index"] = float("nan")

        if prm_stats:
            for key, value in prm_stats.items():
                if isinstance(value, (int, float, np.floating, np.integer)):
                    features[f"prm_{key}"] = float(value)
        return features
