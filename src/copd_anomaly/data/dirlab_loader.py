"""
DIR-Lab raw scan ingestion, validation, NIfTI conversion, and landmark loading.

Handles:
1. Little-endian int16 headerless raw .img parsing with reshape order (Z, Y, X).
2. HU conversion: HU = stored - 1000; padding (-2000) -> -1024 HU.
3. Automated HU offset validation checking lung median in [-950, -750] HU.
4. Landmark loading with automated 0-based vs 1-based index detection via lung mask containment.
"""

from __future__ import annotations

from pathlib import Path
import shutil
import zipfile
import numpy as np
import nibabel as nib
import yaml

from copd_anomaly.interfaces import ScanMetadata
from copd_anomaly.utils.helpers import get_logger

logger = get_logger("dirlab_loader")


class DIRLabLoader:
    """Loader and preprocessor for DIR-Lab COPD cases."""

    def __init__(self, config_path: str | Path = "configs/data/dirlab_cases.yaml") -> None:
        self.config_path = Path(config_path)
        with open(self.config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.defaults = self.config.get("defaults", {})
        self.cases_cfg = self.config.get("cases", {})

    def get_case_config(self, case_id: str) -> dict:
        case_id = case_id.lower()
        if case_id not in self.cases_cfg:
            raise KeyError(f"Case '{case_id}' not found in {self.config_path}")
        return self.cases_cfg[case_id]

    def extract_case(self, zip_path: Path, extract_dir: Path) -> dict[str, Path]:
        """
        Extracts raw image and landmark files from zip, ignoring macOS artifacts.
        Resumable: skips extraction if target files already exist.
        """
        extract_dir.mkdir(parents=True, exist_ok=True)
        extracted_files = {}

        with zipfile.ZipFile(zip_path, "r") as zf:
            for member in zf.infolist():
                if "__MACOSX" in member.filename or member.filename.endswith(".DS_Store"):
                    continue

                filename = Path(member.filename).name
                if not filename:
                    continue

                target_path = extract_dir / filename
                if not target_path.exists() or target_path.stat().st_size != member.file_size:
                    with zf.open(member) as source, open(target_path, "wb") as target:
                        shutil.copyfileobj(source, target, length=1024 * 1024)

                if filename.endswith(".img"):
                    if "_iBHCT" in filename:
                        extracted_files["insp_img"] = target_path
                    elif "_eBHCT" in filename:
                        extracted_files["exp_img"] = target_path
                elif filename.endswith(".txt") and "300" in filename:
                    if "_iBH_" in filename:
                        extracted_files["insp_landmarks"] = target_path
                    elif "_eBH_" in filename:
                        extracted_files["exp_landmarks"] = target_path

        return extracted_files

    def read_raw_volume(
        self,
        img_path: Path,
        dims_xyz: list[int],
        dtype_str: str = "<i2",
        hu_offset: int = 1000,
        padding_val: int = -2000,
    ) -> np.ndarray:
        """
        Reads little-endian int16 raw volume, reshapes to (Z, Y, X), and converts to HU.
        """
        expected_bytes = 2 * dims_xyz[0] * dims_xyz[1] * dims_xyz[2]
        actual_bytes = img_path.stat().st_size
        if actual_bytes != expected_bytes:
            raise ValueError(
                f"File size mismatch for {img_path.name}: expected {expected_bytes} bytes for dims {dims_xyz}, "
                f"got {actual_bytes} bytes."
            )

        raw_data = np.fromfile(img_path, dtype=np.dtype(dtype_str))
        x_dim, y_dim, z_dim = dims_xyz
        volume = raw_data.reshape((z_dim, y_dim, x_dim))

        # Identify padding on int16 volume before converting to float
        # Padding values are <= -1500 (standard padding is -2000)
        padding_mask = volume <= -1500

        # Convert to float32 in place to avoid keeping duplicate arrays in RAM
        volume_hu = volume.astype(np.float32)
        del raw_data, volume
        volume_hu -= float(hu_offset)
        # Set padding to air (-1024 HU)
        volume_hu[padding_mask] = -1024.0
        del padding_mask

        return volume_hu

    def crude_lung_mask(self, volume_hu: np.ndarray) -> np.ndarray:
        """
        Generates a fast crude lung mask by thresholding HU in [-950, -500].
        """
        return ((volume_hu >= -950.0) & (volume_hu <= -500.0)).astype(np.uint8)

    def verify_hu_offset(
        self,
        volume_hu: np.ndarray,
        expected_range: tuple[float, float] = (-950.0, -750.0),
    ) -> tuple[bool, float]:
        """
        Computes median lung HU inside crude lung mask and asserts it is in plausible range.
        """
        mask = self.crude_lung_mask(volume_hu)
        lung_voxels = volume_hu[mask == 1]
        if len(lung_voxels) == 0:
            raise ValueError("Crude lung mask is empty; cannot verify HU offset.")

        median_hu = float(np.median(lung_voxels))
        min_exp, max_exp = expected_range
        is_valid = min_exp <= median_hu <= max_exp
        if not is_valid:
            raise AssertionError(
                f"Plausibility check failed: median lung HU is {median_hu:.1f}, "
                f"expected between {min_exp} and {max_exp} HU."
            )
        return is_valid, median_hu

    def load_landmarks(
        self,
        landmarks_path: Path,
        shape_zyx: tuple[int, int, int],
        lung_mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, int, float]:
        """
        Loads 300 landmarks (columns x, y, z), determines 0-based vs 1-based indexing,
        and returns integer voxel indices (N, 3) as (x_col, y_row, z_slice).

        Returns:
            coords: (N, 3) float32 coordinates (x, y, z)
            index_base: 0 or 1
            in_lung_fraction: float fraction of landmarks inside crude lung mask
        """
        data = np.loadtxt(landmarks_path, dtype=np.float32)
        if data.shape[1] != 3:
            raise ValueError(f"Expected (N, 3) landmarks in {landmarks_path}, got shape {data.shape}")

        z_max, y_max, x_max = shape_zyx

        def test_containment(base: int) -> float:
            shifted = data - float(base)
            x_int = np.round(shifted[:, 0]).astype(int)
            y_int = np.round(shifted[:, 1]).astype(int)
            z_int = np.round(shifted[:, 2]).astype(int)

            within_bounds = (
                (x_int >= 0) & (x_int < x_max) &
                (y_int >= 0) & (y_int < y_max) &
                (z_int >= 0) & (z_int < z_max)
            )
            if not np.all(within_bounds):
                return 0.0

            if lung_mask is not None:
                # lung_mask shape is (Z, Y, X)
                mask_hits = lung_mask[z_int, y_int, x_int]
                return float(np.mean(mask_hits))
            return 1.0

        score_0 = test_containment(0)
        score_1 = test_containment(1)

        # Choose the base with higher in-mask fraction or valid bounds
        if score_1 >= score_0:
            chosen_base = 1
            best_score = score_1
        else:
            chosen_base = 0
            best_score = score_0

        coords_0based = data - float(chosen_base)
        logger.info(
            f"Landmark base detection for {landmarks_path.name}: "
            f"Base 1: {score_1 * 100:.1f}%, Base 0: {score_0 * 100:.1f}%. Selected: Base {chosen_base}"
        )
        return coords_0based, chosen_base, best_score

    def save_nifti(
        self,
        volume_hu: np.ndarray,
        spacing_zyx: tuple[float, float, float],
        output_path: Path,
    ) -> None:
        """
        Saves volume as NIfTI-1 (.nii.gz) with diagonal affine matrix from spacing.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # NIfTI affine: diagonal (sx, sy, sz)
        sz, sy, sx = spacing_zyx
        affine = np.diag([sx, sy, sz, 1.0])
        # NIfTI array shape conventionally (X, Y, Z); transpose from (Z, Y, X)
        volume_xyz = np.transpose(volume_hu, (2, 1, 0))
        img = nib.Nifti1Image(volume_xyz, affine)
        img.header.set_xyzt_units("mm")
        nib.save(img, str(output_path))
        logger.info(f"Saved NIfTI volume to {output_path}")
