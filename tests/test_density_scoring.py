"""
Unit tests for density estimation (GMM, RealNVP), patch anomaly scoring, aggregators,
and proxy AUROC evaluation with bootstrap confidence intervals.
"""

from __future__ import annotations

import numpy as np
import pytest

from copd_anomaly.anomaly.aggregation import (
    ScoreAggregator,
    extract_patient_anomaly_summary,
)
from copd_anomaly.anomaly.density import (
    GaussianMixtureDensity,
    PatchAnomalyScorer,
    RealNVPFlowDensity,
)
from copd_anomaly.evaluation.metrics import PROXY_DISCLAIMER, ModelEvaluator


@pytest.fixture
def synthetic_density_data():
    """Generates synthetic normal embeddings and abnormal outliers."""
    rng = np.random.RandomState(42)
    dim = 16
    # 200 normal points sampled from tight origin Gaussian
    z_normal = rng.normal(0.0, 1.0, size=(200, dim)).astype(np.float32)

    # Inliers for testing
    z_inliers = rng.normal(0.0, 1.0, size=(20, dim)).astype(np.float32)

    # Extreme outliers shifted by 8 standard deviations
    z_outliers = rng.normal(8.0, 1.0, size=(20, dim)).astype(np.float32)

    return {
        "z_normal": z_normal,
        "z_inliers": z_inliers,
        "z_outliers": z_outliers,
    }


def test_gmm_density_and_scoring(synthetic_density_data):
    """Verify GMM assigns higher log_prob to inliers and higher anomaly scores to outliers."""
    gmm = GaussianMixtureDensity(n_components=4, covariance_type="diag", random_state=42)
    gmm.fit(synthetic_density_data["z_normal"])

    log_p_in = gmm.log_prob(synthetic_density_data["z_inliers"])
    log_p_out = gmm.log_prob(synthetic_density_data["z_outliers"])

    # Normal points must have significantly higher log likelihood than outliers
    assert np.mean(log_p_in) > np.mean(log_p_out)

    scorer = PatchAnomalyScorer(gmm)
    scores_in = scorer.score_patches(synthetic_density_data["z_inliers"])
    scores_out = scorer.score_patches(synthetic_density_data["z_outliers"])

    # Outliers must have higher anomaly score s(x) = -log p(z)
    assert np.mean(scores_out) > np.mean(scores_in)


def test_realnvp_flow_density(synthetic_density_data):
    """Verify RealNVP normalizing flow fits and evaluates log probabilities."""
    flow_model = RealNVPFlowDensity(dim=16, n_layers=2, epochs=5, lr=1e-3)
    flow_model.fit(synthetic_density_data["z_normal"])

    lp_in = flow_model.log_prob(synthetic_density_data["z_inliers"])
    lp_out = flow_model.log_prob(synthetic_density_data["z_outliers"])

    assert len(lp_in) == 20
    assert not np.isnan(lp_in).any()
    # Outliers should have lower density
    assert np.mean(lp_in) > np.mean(lp_out)


def test_score_aggregators():
    """Verify patch-to-patient score aggregation strategies."""
    patch_scores = np.array([1.0, 2.0, 3.0, 4.0, 10.0], dtype=np.float32)
    weights = np.array([1.0, 1.0, 1.0, 1.0, 5.0], dtype=np.float32)
    lobe_ids = np.array([1, 1, 2, 2, 3], dtype=np.uint8)

    agg_mean = ScoreAggregator(method="mean")
    assert np.isclose(agg_mean.aggregate(patch_scores), 4.0)

    agg_p90 = ScoreAggregator(method="p90")
    assert agg_p90.aggregate(patch_scores) > agg_mean.aggregate(patch_scores)

    agg_weighted = ScoreAggregator(method="weighted")
    # Weighted by 10.0 having weight 5.0 -> pulls score higher than mean
    assert agg_weighted.aggregate(patch_scores, volume_weights=weights) > 4.0

    agg_lobe = ScoreAggregator(method="lobe_max")
    # Lobe 3 contains only the outlier (10.0), so max lobe mean is 10.0
    assert np.isclose(agg_lobe.aggregate(patch_scores, lobe_ids=lobe_ids), 10.0)

    summary = extract_patient_anomaly_summary(patch_scores, lobe_ids)
    assert "anomaly_mean" in summary
    assert "anomaly_p90" in summary
    assert "rul_anomaly_mean" in summary


def test_evaluator_proxy_auc_and_disclaimer():
    """Verify ModelEvaluator computes AUROC, AUPRC, bootstrap CIs, and enforces proxy disclaimer."""
    evaluator = ModelEvaluator()

    # Clear separation: 20 controls with low scores, 20 cases with high scores
    y_true = np.array([0] * 20 + [1] * 20)
    rng = np.random.RandomState(42)
    y_score = np.concatenate([rng.normal(5.0, 1.0, 20), rng.normal(15.0, 1.0, 20)])

    res = evaluator.evaluate_proxy_auc(y_true, y_score, n_bootstraps=100)

    assert "auroc" in res
    assert res["auroc"] > 0.95
    assert "auroc_ci95_low" in res
    assert "auroc_ci95_high" in res
    assert res["auroc_ci95_low"] <= res["auroc"] <= res["auroc_ci95_high"]

    # CRITICAL: Verify mandatory proxy disclaimer
    assert "REPORT-LABEL PROXY, not COPD classification" in res["metric_type"]
    assert PROXY_DISCLAIMER in res["disclaimer"]
