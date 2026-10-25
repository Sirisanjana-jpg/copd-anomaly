"""Tests for training preflight diagnostics."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from copd_anomaly.interfaces import PatchRecord
from copd_anomaly.patches.dataset import ShardedPatchWriter
from scripts.diagnose_training import diagnose_training_inputs


def _record() -> PatchRecord:
    return PatchRecord(
        patch_id="patch-1",
        patient_id="patient-1",
        scan_id="scan-1",
        center_zyx=(2, 2, 2),
        lobe_id=None,
        lung_fraction=0.9,
        volume_weight=1.0,
        laa950_fraction=0.02,
    )


def test_preflight_passes_for_valid_patch_shards(tmp_path: Path) -> None:
    shards = tmp_path / "shards"
    writer = ShardedPatchWriter(shards, shard_size=2)
    writer.add_patches(np.zeros((1, 1, 8, 8, 8), dtype=np.uint8), [_record()])
    writer.close()

    report = diagnose_training_inputs(
        shards_dir=shards,
        checkpoint_dir=tmp_path / "checkpoints",
        min_free_gb=0,
    )

    assert report["status"] == "passed"
    assert report["patch_count"] == 1
    assert report["shard_count"] == 1
    assert (tmp_path / "checkpoints").is_dir()


def test_preflight_reports_corrupt_patch_dataset(tmp_path: Path) -> None:
    shards = tmp_path / "shards"
    shards.mkdir()
    (shards / "patch_manifest.csv").write_text("not,a,valid,manifest\n", encoding="utf-8")

    report = diagnose_training_inputs(
        shards_dir=shards,
        checkpoint_dir=tmp_path / "checkpoints",
        min_free_gb=0,
    )

    assert report["status"] == "failed"
    assert any("Patch dataset validation failed" in error for error in report["errors"])
