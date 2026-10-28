"""
Unit tests for synthetic cohort generation, multimodal fusion, MC uncertainty, and ablations.
"""

from __future__ import annotations

from pathlib import Path

import torch

from copd_anomaly.synthetic.cohort_generator import (
    SYNTHETIC_BANNER,
    generate_synthetic_copd_cohort,
)
from copd_anomaly.synthetic.multimodal import (
    ExacerbationRiskHead,
    MultimodalFusionModel,
    MultimodalRiskPipeline,
)


def test_synthetic_cohort_generation():
    """Verify synthetic cohort distributions, clinical correlations, and mandatory banner."""
    df = generate_synthetic_copd_cohort(n_patients=100, seed=42)

    assert len(df) == 100
    assert "banner" in df.columns
    assert df["banner"].iloc[0] == SYNTHETIC_BANNER

    # Check plausible demographic ranges
    assert df["age"].min() >= 45.0
    assert df["age"].max() <= 90.0
    assert set(df["sex"].unique()) == {"M", "F"}
    assert df["fev1_pct_predicted"].min() >= 15.0
    assert df["fev1_pct_predicted"].max() <= 120.0

    # GOLD stages 0 to 4
    assert set(df["gold_stage"].unique()).issubset({0, 1, 2, 3, 4})

    # Event rate should be non-trivial
    event_rate = df["exacerbation_12m_event"].mean()
    assert 0.10 <= event_rate <= 0.60


def test_multimodal_fusion_and_risk_head():
    """Verify forward and backward passes for fusion network and risk head."""
    batch_size = 4
    d_ct = 64
    d_clin = 8
    d_fused = 32

    x_ct = torch.randn(batch_size, d_ct)
    x_clin = torch.randn(batch_size, d_clin)

    fusion = MultimodalFusionModel(d_ct=d_ct, d_clin=d_clin, d_hidden=32, d_fused=d_fused)
    head = ExacerbationRiskHead(d_fused=d_fused)

    h_fused = fusion(x_ct, x_clin)
    assert h_fused.shape == (batch_size, d_fused)

    logits = head(h_fused)
    assert logits.shape == (batch_size, 1)

    # Loss computation and backward gradient flow
    targets = torch.tensor([[1.0], [0.0], [1.0], [0.0]])
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)
    loss.backward()

    assert fusion.ct_proj[0].weight.grad is not None
    assert head.classifier[0].weight.grad is not None


def test_monte_carlo_uncertainty_estimation():
    """Verify epistemic uncertainty estimation via Monte Carlo Dropout."""
    pipeline = MultimodalRiskPipeline(d_ct=32, d_clin=6, d_fused=16, dropout=0.3)

    x_ct = torch.randn(5, 32)
    x_clin = torch.randn(5, 6)

    res = pipeline.predict_with_uncertainty(x_ct, x_clin, n_mc_samples=15)

    assert "mean_risk" in res
    assert "epistemic_uncertainty" in res
    assert "ci90_low" in res
    assert "ci90_high" in res

    mean_risk = res["mean_risk"]
    assert mean_risk.shape == (5,)
    assert torch.all(mean_risk >= 0.0) and torch.all(mean_risk <= 1.0)

    # Uncertainty should be non-negative
    assert torch.all(res["epistemic_uncertainty"] >= 0.0)


def test_ablation_experiment_runner(tmp_path: Path):
    """Verify cross-validated ablation experiment runner on synthetic cohort."""
    from scripts.run_ablation import run_ablation_experiments

    cohort_file = tmp_path / "test_syn_cohort.parquet"
    df = generate_synthetic_copd_cohort(n_patients=60, seed=42)
    df.to_parquet(cohort_file, index=False)

    report_file = tmp_path / "ablation_results.csv"
    res_df = run_ablation_experiments(
        cohort_path=cohort_file,
        output_report=report_file,
        n_splits=3,
        seed=42,
    )

    assert len(res_df) == 5
    assert "auroc" in res_df.columns
    assert "brier_score" in res_df.columns
    assert "statistically_significant" in res_df.columns
    assert report_file.exists()
