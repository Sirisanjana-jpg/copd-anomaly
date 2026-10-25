"""Safety tests for fail-closed real-data paths and explicit synthetic provenance."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from copd_anomaly.features.lobe_builder import LobeFeatureBuilder
from copd_anomaly.interfaces import PatchRecord
from scripts.fit_density import fit_density_estimator
from scripts.lobe_analysis import run_lobe_analysis
from scripts.score_anomaly import score_and_evaluate
from scripts.train_encoder import train_contrastive_encoder


def test_encoder_refuses_missing_manifest_without_checkpoint(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Patch manifest not found"):
        train_contrastive_encoder(
            shards_dir=tmp_path / "missing",
            checkpoint_dir=tmp_path / "checkpoints",
            epochs=1,
            batch_size=2,
        )
    assert not list(tmp_path.rglob("*.pt"))


def test_density_refuses_missing_manifest(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Patch manifest not found"):
        fit_density_estimator(
            shards_dir=tmp_path / "missing",
            encoder_ckpt=tmp_path / "encoder.pt",
            output_model_path=tmp_path / "density.joblib",
        )
    assert not list(tmp_path.rglob("*.joblib"))


def test_real_scoring_does_not_emit_empty_report(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Density model not found"):
        score_and_evaluate(
            density_model_path=tmp_path / "missing.joblib",
            output_report_path=tmp_path / "report.csv",
        )
    assert not (tmp_path / "report.csv").exists()


def test_lobe_analysis_refuses_to_fabricate_real_scores(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.parquet"
    manifest.touch()
    with pytest.raises(NotImplementedError, match="does not load real patch scores"):
        run_lobe_analysis(
            manifest_path=manifest,
            output_csv=tmp_path / "lobe_features.csv",
        )
    assert not (tmp_path / "lobe_features.csv").exists()


def test_missing_lobe_is_nan_not_global_mean() -> None:
    records = [
        PatchRecord(
            patch_id="p1",
            patient_id="patient1",
            scan_id="scan1",
            center_zyx=(1, 1, 1),
            lobe_id=1,
            lung_fraction=0.9,
            volume_weight=1.0,
            laa950_fraction=0.02,
        ),
        PatchRecord(
            patch_id="p2",
            patient_id="patient1",
            scan_id="scan1",
            center_zyx=(2, 2, 2),
            lobe_id=2,
            lung_fraction=0.9,
            volume_weight=1.0,
            laa950_fraction=0.03,
        ),
    ]
    features = LobeFeatureBuilder().build_features(np.array([2.0, 4.0]), records)
    assert features["rul_anomaly_mean"] == 2.0
    assert features["rml_anomaly_mean"] == 4.0
    assert np.isnan(features["rll_anomaly_mean"])
    assert features["rll_present"] == 0
    assert features["lobes_observed_count"] == 2
    assert np.isnan(features["upper_lower_ratio"])
    assert np.isnan(features["heterogeneity_index"])


def test_synthetic_score_report_is_marked_and_separated(tmp_path: Path) -> None:
    report_path = tmp_path / "synthetic" / "proxy.csv"
    report = score_and_evaluate(
        density_model_path=tmp_path / "unused.joblib",
        output_report_path=report_path,
        dry_run=True,
        n_bootstraps=20,
    )
    assert report_path.exists()
    assert report["data_status"].str.contains("SYNTHETIC DRY RUN").all()
    assert report["metric_type"].str.contains("proxy", case=False).all()
