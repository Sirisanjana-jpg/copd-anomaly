"""
Unit tests for lung segmentation, deformable registration, TRE evaluation, and PRM feature extraction.
Uses synthetic 3D fixtures.
"""

from __future__ import annotations

import numpy as np
import pytest

from copd_anomaly.features.prm import compute_prm_map, compute_prm_stats
from copd_anomaly.registration.registrar import SimpleITKRegistrar, evaluate_tre
from copd_anomaly.segmentation.segmenter import LungSegmenter


@pytest.fixture
def synthetic_pair_phantom():
    """
    Creates a small synthetic paired CT phantom (Z=16, Y=32, X=32).
    Fixed: central lung region (-850 HU), background body (+40 HU), air (-1024 HU).
    Moving: shifted slightly.
    """
    shape = (16, 32, 32)
    vol_fixed = np.full(shape, 40.0, dtype=np.float32)  # soft tissue body
    vol_fixed[0, :, :] = -1024.0
    vol_fixed[-1, :, :] = -1024.0
    vol_fixed[:, 0, :] = -1024.0
    vol_fixed[:, -1, :] = -1024.0
    vol_fixed[:, :, 0] = -1024.0
    vol_fixed[:, :, -1] = -1024.0

    # Two lung cylinders
    # Right lung: y in [8, 24], x in [6, 14]
    vol_fixed[3:13, 8:24, 6:14] = -850.0
    # Left lung: y in [8, 24], x in [18, 26]
    vol_fixed[3:13, 8:24, 18:26] = -850.0

    # Moving: shifted by 1 voxel in X
    vol_moving = np.full(shape, 40.0, dtype=np.float32)
    vol_moving[3:13, 8:24, 7:15] = -800.0
    vol_moving[3:13, 8:24, 19:27] = -800.0

    mask = ((vol_fixed >= -950.0) & (vol_fixed <= -500.0)).astype(np.uint8)

    # 5-lobe synthetic mask
    lobe_mask = np.zeros(shape, dtype=np.uint8)
    # Upper lobes (z in [3, 8])
    lobe_mask[3:8, 8:24, 6:14] = 1  # RUL
    lobe_mask[3:8, 8:24, 18:26] = 4  # LUL
    # Middle lobe (z in [8, 10], right)
    lobe_mask[8:10, 8:24, 6:14] = 2  # RML
    # Lower lobes (z in [10, 13])
    lobe_mask[10:13, 8:24, 6:14] = 3  # RLL
    lobe_mask[8:13, 8:24, 18:26] = 5  # LLL

    return {
        "vol_fixed": vol_fixed,
        "vol_moving": vol_moving,
        "mask": mask,
        "lobe_mask": lobe_mask,
        "spacing_zyx": (2.0, 1.0, 1.0),
    }


def test_lung_segmenter_threshold(synthetic_pair_phantom):
    """Verify threshold segmentation identifies both lung regions and fills holes."""
    segmenter = LungSegmenter(backend="threshold")
    mask = segmenter.segment(
        synthetic_pair_phantom["vol_fixed"],
        synthetic_pair_phantom["spacing_zyx"],
    )
    assert mask.shape == synthetic_pair_phantom["vol_fixed"].shape
    assert mask.dtype == np.uint8
    assert np.sum(mask == 1) > 100
    # Air border should be background 0
    assert mask[0, 0, 0] == 0


def test_evaluate_tre_calculation():
    """Verify Target Registration Error math with synthetic known landmark displacement."""
    fixed_pts = np.array([[10.0, 10.0, 10.0], [20.0, 20.0, 20.0]], dtype=np.float32)
    # Moving shifted by dx=3, dy=4, dz=0 mm with 1mm spacing -> Euclidean distance is 5.0 mm
    moving_pts = fixed_pts + np.array([3.0, 4.0, 0.0], dtype=np.float32)
    # Warped brings points closer: dx=1, dy=0, dz=0 -> 1.0 mm
    warped_pts = fixed_pts + np.array([1.0, 0.0, 0.0], dtype=np.float32)

    stats = evaluate_tre(fixed_pts, moving_pts, warped_pts, spacing_zyx=(1.0, 1.0, 1.0))
    assert np.isclose(stats["initial_mean_tre_mm"], 5.0)
    assert np.isclose(stats["mean_tre_mm"], 1.0)
    assert np.isclose(stats["tre_reduction_pct"], 80.0)


def test_deformable_registration_fast_mode(synthetic_pair_phantom):
    """Verify SimpleITK deformable registration completes and transforms points."""
    registrar = SimpleITKRegistrar(fast_mode=True)
    res = registrar.register(
        fixed_volume=synthetic_pair_phantom["vol_fixed"],
        moving_volume=synthetic_pair_phantom["vol_moving"],
        fixed_mask=synthetic_pair_phantom["mask"],
        spacing_zyx=synthetic_pair_phantom["spacing_zyx"],
    )

    assert res.warped_volume.shape == synthetic_pair_phantom["vol_fixed"].shape
    assert res.displacement_field is not None
    assert res.displacement_field.shape == (*synthetic_pair_phantom["vol_fixed"].shape, 3)

    test_pts = np.array([[10.0, 10.0, 5.0], [20.0, 15.0, 6.0]], dtype=np.float32)
    transformed_pts = registrar.transform_points(
        test_pts, res.transform, spacing_zyx=synthetic_pair_phantom["spacing_zyx"]
    )
    assert transformed_pts.shape == test_pts.shape
    assert not np.isnan(transformed_pts).any()


def test_prm_mapping_and_statistics(synthetic_pair_phantom):
    """Verify PRM categorical classifications and lobe statistics."""
    shape = (10, 20, 20)
    mask = np.ones(shape, dtype=np.uint8)

    # 4 quadrants with specific HU profiles
    vol_insp = np.full(shape, -850.0, dtype=np.float32)
    vol_exp = np.full(shape, -750.0, dtype=np.float32)

    # Section 1: Normal (Insp -850, Exp -750) -> already set
    # Section 2: fSAD (Insp -850, Exp -900)
    vol_exp[:, :, 10:] = -900.0

    # Section 3: Emphysema (Insp -980, Exp -960)
    vol_insp[:, 10:, :] = -980.0
    vol_exp[:, 10:, :] = -960.0

    prm_map = compute_prm_map(vol_insp, vol_exp, mask)
    # Check categorical labels exist
    assert 1 in prm_map  # Normal
    assert 2 in prm_map  # fSAD
    assert 3 in prm_map  # Emphysema

    stats = compute_prm_stats(
        vol_insp,
        vol_exp,
        mask,
        lobe_mask=None,
        spacing_zyx=(1.0, 1.0, 1.0),
    )
    assert "prm_norm_pct" in stats
    assert "prm_fsad_pct" in stats
    assert "prm_emph_pct" in stats
    assert "laa950_insp_pct" in stats
    assert stats["prm_emph_pct"] > 0
    assert stats["laa950_insp_pct"] > 0


def test_lobe_feature_builder():
    """Verify LobeFeatureBuilder calculates spatial gradients and PRM integration."""
    from copd_anomaly.features.lobe_builder import LobeFeatureBuilder
    from copd_anomaly.interfaces import PatchRecord

    builder = LobeFeatureBuilder()
    scores = np.array([10.0, 12.0, 20.0, 22.0, 15.0], dtype=np.float32)
    # 5 patches in 5 lobes: 1: RUL, 2: RML, 3: RLL, 4: LUL, 5: LLL
    records = [
        PatchRecord(
            patch_id=f"p_{i}",
            patient_id="case_1",
            scan_id="case_1_insp",
            center_zyx=(10, 20, 20),
            lobe_id=i + 1,
            lung_fraction=0.85,
            volume_weight=1.0,
            laa950_fraction=0.05,
        )
        for i in range(5)
    ]

    prm_dummy = {"norm_pct": 70.0, "fsad_pct": 20.0, "emph_pct": 10.0}
    feats = builder.build_features(scores, records, prm_dummy)

    assert "global_anomaly_mean" in feats
    assert "rul_anomaly_mean" in feats
    assert "upper_lower_ratio" in feats
    assert "left_right_asymmetry" in feats
    assert "heterogeneity_index" in feats
    assert feats["prm_norm_pct"] == 70.0


def test_reconstruct_anomaly_volume():
    """Verify 3D anomaly heatmap volume reconstruction from patch scores."""
    from copd_anomaly.interfaces import PatchRecord
    from copd_anomaly.viz.heatmaps import reconstruct_anomaly_volume

    shape = (20, 20, 20)
    records = [
        PatchRecord(
            patch_id="p_0",
            patient_id="c1",
            scan_id="s1",
            center_zyx=(10, 10, 10),
            lobe_id=1,
            lung_fraction=1.0,
            volume_weight=1.0,
            laa950_fraction=0.0,
        )
    ]
    scores = np.array([25.0], dtype=np.float32)

    vol = reconstruct_anomaly_volume(shape, records, scores, patch_size=6)
    assert vol.shape == shape
    assert vol.dtype == np.float32
    # Center voxel should have received score 25.0
    assert np.isclose(vol[10, 10, 10], 25.0)
    # Far corner voxel should be 0.0
    assert vol[0, 0, 0] == 0.0

