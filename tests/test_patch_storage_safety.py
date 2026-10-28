"""Crash-safety and corruption checks for patch shard persistence and loading."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from copd_anomaly.interfaces import PatchRecord
from copd_anomaly.patches.dataset import ShardedPatchDataset, ShardedPatchWriter


def _record(index: int = 0) -> PatchRecord:
    return PatchRecord(
        patch_id=f"patch-{index}",
        patient_id="patient-1",
        scan_id="scan-1",
        center_zyx=(2, 2, 2),
        lobe_id=None,
        lung_fraction=0.9,
        volume_weight=1.0,
        laa950_fraction=0.02,
    )


def test_writer_rejects_patch_record_count_mismatch(tmp_path: Path) -> None:
    writer = ShardedPatchWriter(tmp_path / "shards", shard_size=2)
    patches = np.zeros((2, 1, 8, 8, 8), dtype=np.uint8)
    with pytest.raises(ValueError, match="Patch/record count mismatch"):
        writer.add_patches(patches, [_record()])
    assert not list((tmp_path / "shards").glob("shard_*.npy"))


def test_writer_rejects_wrong_dtype_before_writing(tmp_path: Path) -> None:
    writer = ShardedPatchWriter(tmp_path / "shards", shard_size=2)
    patches = np.zeros((1, 1, 8, 8, 8), dtype=np.float32)
    with pytest.raises(ValueError, match="must be uint8"):
        writer.add_patches(patches, [_record()])
    assert not list((tmp_path / "shards").glob("shard_*.npy"))


def test_writer_publishes_readable_shard_and_manifest(tmp_path: Path) -> None:
    output = tmp_path / "shards"
    writer = ShardedPatchWriter(output, shard_size=2)
    patches = np.zeros((2, 1, 8, 8, 8), dtype=np.uint8)
    writer.add_patches(patches, [_record(0), _record(1)])
    manifest = writer.close()

    assert manifest.is_file()
    assert not list(output.glob(".*.tmp"))
    dataset = ShardedPatchDataset(output)
    assert len(dataset) == 2
    patch, metadata = dataset[1]
    assert tuple(patch.shape) == (1, 8, 8, 8)
    assert metadata["patch_id"] == "patch-1"


def test_dataset_fails_fast_when_manifest_points_outside_shard(tmp_path: Path) -> None:
    output = tmp_path / "shards"
    writer = ShardedPatchWriter(output, shard_size=2)
    writer.add_patches(
        np.zeros((1, 1, 8, 8, 8), dtype=np.uint8),
        [_record()],
    )
    writer.close()

    import pandas as pd

    manifest_path = output / "patch_manifest.parquet"
    manifest = pd.read_parquet(manifest_path)
    manifest.loc[0, "index_in_shard"] = 9
    manifest.to_parquet(manifest_path, index=False)

    with pytest.raises(IndexError, match="shard contains only"):
        ShardedPatchDataset(output)


def test_dataset_fails_fast_on_missing_shard(tmp_path: Path) -> None:
    output = tmp_path / "shards"
    writer = ShardedPatchWriter(output, shard_size=2)
    writer.add_patches(
        np.zeros((1, 1, 8, 8, 8), dtype=np.uint8),
        [_record()],
    )
    writer.close()
    (output / "shard_0000.npy").unlink()

    with pytest.raises(FileNotFoundError, match="missing shard files"):
        ShardedPatchDataset(output)
