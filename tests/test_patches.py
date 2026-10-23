"""
Unit tests for 3D patch extraction, uint8 windowing, and sharded dataset streaming.
Uses small synthetic volumetric phantoms.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest
import torch

from copd_anomaly.patches.dataset import ShardedPatchDataset, ShardedPatchWriter
from copd_anomaly.patches.sampler import (
    UniformLungPatchSampler,
    uint8_to_float32,
    window_and_to_uint8,
)


def test_window_and_uint8_scaling():
    """Verify linear HU mapping from [-1024, 0] to [0, 255] and back to float32."""
    hu_vals = np.array([-1024.0, -512.0, 0.0, -2000.0, 500.0], dtype=np.float32)
    uint8_vals = window_and_to_uint8(hu_vals, window_hu=(-1024.0, 0.0))

    assert uint8_vals.dtype == np.uint8
    assert uint8_vals[0] == 0  # -1024 HU -> 0
    assert uint8_vals[2] == 255  # 0 HU -> 255
    assert np.isclose(uint8_vals[1], 128, atol=2)  # -512 HU -> midpoint ~128
    assert uint8_vals[3] == 0  # Clipped below -> 0
    assert uint8_vals[4] == 255  # Clipped above -> 255

    float_vals = uint8_to_float32(uint8_vals)
    assert float_vals.dtype == np.float32
    assert np.isclose(float_vals[0], 0.0)
    assert np.isclose(float_vals[2], 1.0)
    assert np.isclose(float_vals[1], 0.5, atol=0.01)


def test_patch_sampler_constraints():
    """Verify patch sampler enforces min_lung_fraction and returns expected tensor shapes."""
    # Synthetic volume: shape (60, 60, 60)
    shape = (60, 60, 60)
    vol_insp = np.full(shape, -850.0, dtype=np.float32)
    vol_exp = np.full(shape, -750.0, dtype=np.float32)

    # Synthetic lung mask: center cube is lung, borders are background
    mask = np.zeros(shape, dtype=np.uint8)
    mask[5:55, 5:55, 5:55] = 1

    lobe_mask = np.full(shape, 1, dtype=np.uint8)  # Lobe 1 (RUL)

    sampler = UniformLungPatchSampler(patch_size=30, min_lung_fraction=0.70)
    patches, records = sampler.extract_patches(
        volume_insp=vol_insp,
        volume_exp=vol_exp,
        lung_mask=mask,
        lobe_mask=lobe_mask,
        patch_size=30,
        max_patches_per_scan=5,
        patient_id="case_syn",
        scan_id="case_syn_insp",
    )

    assert len(records) > 0
    assert patches.shape == (len(records), 2, 30, 30, 30)
    assert patches.dtype == np.uint8

    for rec in records:
        assert rec.lung_fraction >= 0.70
        assert rec.patient_id == "case_syn"
        assert rec.lobe_id == 1
        assert rec.center_zyx[0] >= 15


def test_sharded_patch_writer_and_dataset(tmp_path: Path):
    """Verify out-of-core sharded patch writer, manifest generation, and PyTorch dataset reading."""
    writer = ShardedPatchWriter(output_dir=tmp_path / "shards", shard_size=2)

    # Create dummy patches
    p_shape = (5, 1, 16, 16, 16)
    patches = np.full(p_shape, 128, dtype=np.uint8)
    # Set one patch with high emphysema (LAA-950)
    from copd_anomaly.interfaces import PatchRecord

    records = [
        PatchRecord(
            patch_id=f"p_{i}",
            patient_id="pat_0",
            scan_id="scan_0",
            center_zyx=(8, 8, 8),
            lobe_id=1,
            lung_fraction=0.85,
            volume_weight=1.0,
            laa950_fraction=0.005 if i < 3 else 0.05,  # 3 normal, 2 emphysema
        )
        for i in range(5)
    ]

    writer.add_patches(patches, records)
    manifest_path = writer.close()

    assert manifest_path.exists()
    # Shard size was 2, so 5 patches should produce 3 shard files: shard_0000, shard_0001, shard_0002
    shard_files = list((tmp_path / "shards").glob("shard_*.npy"))
    assert len(shard_files) == 3

    # Test full dataset
    ds = ShardedPatchDataset(tmp_path / "shards")
    assert len(ds) == 5

    tensor, meta = ds[0]
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (1, 16, 16, 16)
    assert tensor.dtype == torch.float32
    assert np.isclose(tensor.mean().item(), 128.0 / 255.0, atol=0.02)
    assert meta["patch_id"] == "p_0"

    # Test normal-only filtering
    ds_normal = ShardedPatchDataset(
        tmp_path / "shards", normal_only=True, normal_emphysema_cap=0.01
    )
    assert len(ds_normal) == 3
