"""
Unit tests for DIRLabLoader using small synthetic raw fixtures.
"""

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from copd_anomaly.data.dirlab_loader import DIRLabLoader


@pytest.fixture
def synthetic_dirlab_env(tmp_path: Path):
    """Creates a temporary synthetic DIR-Lab raw image and configuration."""
    cfg_path = tmp_path / "test_dirlab_cases.yaml"
    # Small synthetic dimensions: X=16, Y=16, Z=8
    dims = [16, 16, 8]
    spacing = [1.0, 1.0, 2.5]

    yaml_content = f"""
defaults:
  dtype: "<i2"
  hu_offset: 1000
  padding_value: -2000
cases:
  test_case:
    dims: [{dims[0]}, {dims[1]}, {dims[2]}]
    spacing_mm: [{spacing[0]}, {spacing[1]}, {spacing[2]}]
"""
    cfg_path.write_text(yaml_content, encoding="utf-8")

    # Generate synthetic raw data
    # Create lung voxels around -850 HU (stored as -850 + 1000 = 150)
    # Outside padding as -2000
    n_voxels = dims[0] * dims[1] * dims[2]
    synthetic_stored = np.full(n_voxels, 150, dtype=np.int16)
    # Set corner voxels as padding (-2000)
    synthetic_stored[0:10] = -2000

    raw_img_path = tmp_path / "test_case_iBHCT.img"
    synthetic_stored.tofile(raw_img_path)

    # Generate synthetic 1-based landmarks
    # Points near center: x=8, y=8, z=4 (in 1-based: x=9, y=9, z=5)
    landmarks_1based = np.array(
        [
            [9.0, 9.0, 5.0],
            [8.0, 8.0, 4.0],
            [10.0, 10.0, 6.0],
        ],
        dtype=np.float32,
    )
    lm_path = tmp_path / "test_case_300_iBH_xyz_r1.txt"
    np.savetxt(lm_path, landmarks_1based)

    return {
        "cfg_path": cfg_path,
        "raw_img_path": raw_img_path,
        "lm_path": lm_path,
        "dims": dims,
        "spacing": spacing,
        "tmp_path": tmp_path,
    }


@pytest.mark.synthetic
def test_read_raw_volume_correct(synthetic_dirlab_env):
    """Verify raw volume reading, shape, HU conversion, and padding handling."""
    loader = DIRLabLoader(synthetic_dirlab_env["cfg_path"])
    dims = synthetic_dirlab_env["dims"]
    vol_hu = loader.read_raw_volume(synthetic_dirlab_env["raw_img_path"], dims)

    # Shape should be (Z, Y, X) = (8, 16, 16)
    assert vol_hu.shape == (dims[2], dims[1], dims[0])

    # Check HU values: stored 150 -> 150 - 1000 = -850 HU
    # Padding -2000 -> air -1024 HU
    assert np.isclose(vol_hu[4, 8, 8], -850.0)
    assert np.isclose(vol_hu.reshape(-1)[0], -1024.0)


@pytest.mark.synthetic
def test_read_raw_volume_wrong_size_failure(synthetic_dirlab_env):
    """Verify that reader fails loudly with ValueError when file size does not match dimensions."""
    loader = DIRLabLoader(synthetic_dirlab_env["cfg_path"])
    wrong_dims = [32, 32, 16]  # Expects much larger byte size

    with pytest.raises(ValueError, match="File size mismatch"):
        loader.read_raw_volume(synthetic_dirlab_env["raw_img_path"], wrong_dims)


@pytest.mark.synthetic
def test_verify_hu_offset_success_and_failure(synthetic_dirlab_env):
    """Verify automated HU offset validation."""
    loader = DIRLabLoader(synthetic_dirlab_env["cfg_path"])
    dims = synthetic_dirlab_env["dims"]
    vol_hu = loader.read_raw_volume(synthetic_dirlab_env["raw_img_path"], dims)

    # Plausible range: median is -850 HU, should pass
    is_valid, median_hu = loader.verify_hu_offset(vol_hu, expected_range=(-950.0, -750.0))
    assert is_valid is True
    assert np.isclose(median_hu, -850.0)

    # Intentionally impossible range should raise AssertionError
    with pytest.raises(AssertionError, match="Plausibility check failed"):
        loader.verify_hu_offset(vol_hu, expected_range=(-500.0, -100.0))


@pytest.mark.synthetic
def test_load_landmarks_index_base_detection(synthetic_dirlab_env):
    """Verify landmark loader detects 1-based indexing and provides 0-based coordinates."""
    loader = DIRLabLoader(synthetic_dirlab_env["cfg_path"])
    dims = synthetic_dirlab_env["dims"]
    shape_zyx = (dims[2], dims[1], dims[0])

    vol_hu = loader.read_raw_volume(synthetic_dirlab_env["raw_img_path"], dims)
    mask = loader.crude_lung_mask(vol_hu)

    coords_0based, chosen_base, containment = loader.load_landmarks(
        synthetic_dirlab_env["lm_path"], shape_zyx, mask
    )

    assert chosen_base == 1
    assert containment > 0.9
    # Ground truth was 1-based [9, 9, 5] -> 0-based should be [8, 8, 4]
    assert np.allclose(coords_0based[0], [8.0, 8.0, 4.0])


@pytest.mark.synthetic
def test_save_and_load_nifti(synthetic_dirlab_env):
    """Verify NIfTI volume saving with correct affine and units."""
    loader = DIRLabLoader(synthetic_dirlab_env["cfg_path"])
    dims = synthetic_dirlab_env["dims"]
    vol_hu = loader.read_raw_volume(synthetic_dirlab_env["raw_img_path"], dims)
    spacing_zyx = (2.5, 1.0, 1.0)

    out_path = synthetic_dirlab_env["tmp_path"] / "saved_scan.nii.gz"
    loader.save_nifti(vol_hu, spacing_zyx, out_path)

    assert out_path.exists()
    loaded_nii = nib.load(str(out_path))
    # In NIfTI, shape is (X, Y, Z) = (16, 16, 8)
    assert loaded_nii.shape == (dims[0], dims[1], dims[2])
    assert np.isclose(loaded_nii.header.get_zooms(), (1.0, 1.0, 2.5)).all()


def test_dirlab_manifest_parquet_if_present():
    """Verify that if dirlab_manifest.parquet is generated, it contains all 20 scans with verified HU."""
    manifest_path = Path("data/processed/dirlab_manifest.parquet")
    if not manifest_path.exists():
        pytest.skip("Manifest not generated on disk yet.")

    import pandas as pd
    df = pd.read_parquet(manifest_path)
    assert len(df) == 20
    assert "hu_offset_verified" in df.columns
    assert df["hu_offset_verified"].all()
    assert set(df["phase"].unique()) == {"insp", "exp"}
    assert df["patient_id"].nunique() == 10

