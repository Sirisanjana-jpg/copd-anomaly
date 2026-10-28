"""
CT-RATE Streaming Ingestion, Patient-Level Disjoint Splitting, and Immediate Cleanup.

Enforces zero disk exhaustion:
1. Candidate subjects are selected offline using metadata CSVs.
2. Disjoint patient splits (ctrate_train, ctrate_val, ctrate_test) are verified with zero patient leakage.
3. Volumes are streamed one-by-one from Hugging Face Hub using HF_TOKEN.
4. Volumes are immediately deleted from disk as soon as patches are extracted.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from nibabel.processing import resample_to_output

from copd_anomaly.patches.dataset import ShardedPatchWriter
from copd_anomaly.patches.sampler import UniformLungPatchSampler
from copd_anomaly.segmentation.segmenter import LungSegmenter
from copd_anomaly.utils.helpers import get_logger

logger = get_logger("ctrate_streamer")

TIERS = {
    "lean": {"train": 100, "val": 30, "test": 70},
    "medium": {"train": 300, "val": 90, "test": 210},
    "full": {"train": 1000, "val": 300, "test": 700},
}
def ctrate_remote_path(volume_name: str) -> str:
    """Build the nested CT-RATE Hub path from a volume filename."""
    name = Path(volume_name).name

    match = re.fullmatch(
        r"(train|valid)_(\d+)_([a-z]+)_(\d+)\.nii\.gz",
        name,
    )
    if not match:
        raise ValueError(f"Unexpected CT-RATE filename: {name!r}")

    split, subject_num, series, _reconstruction = match.groups()
    subject = f"{split}_{subject_num}"
    series_dir = f"{subject}_{series}"

    return f"dataset/{split}/{subject}/{series_dir}/{name}"

def load_ctrate_hu(
    image_path: str | Path,
    slope: float,
    intercept: float,
) -> tuple[nib.Nifti1Image, np.ndarray, np.ndarray]:
    """Load CT-RATE stored values and apply CSV rescale metadata once.

    Returns:
        image: Canonical NIfTI image, retaining its spatial geometry.
        hu_xyz: HU volume in canonical XYZ orientation.
        valid_xyz: Boolean mask identifying non-padding source voxels.
    """
    if not np.isfinite(slope) or slope == 0:
        raise ValueError(f"Invalid CT-RATE rescale slope: {slope!r}")
    if not np.isfinite(intercept):
        raise ValueError(f"Invalid CT-RATE rescale intercept: {intercept!r}")

    source_img = nib.load(str(image_path))
    raw = np.asarray(source_img.dataobj.get_unscaled(), dtype=np.float32)

    if not np.isfinite(raw).all():
        raise ValueError(f"Non-finite stored voxel values in {image_path}")

    # CT-RATE raw-zero voxels are treated as padding for this dataset.
    # Keep the mask separately; do not infer validity from HU values.
    valid_source = raw != 0

    hu = raw * np.float32(slope) + np.float32(intercept)
    hu[~valid_source] = -1024.0

    # Apply orientation to the HU data and its validity mask together.
    hu_img = nib.Nifti1Image(hu, source_img.affine, source_img.header)
    hu_img.set_sform(source_img.get_sform(), code=int(source_img.header["sform_code"]))
    hu_img.set_qform(source_img.get_qform(), code=int(source_img.header["qform_code"]))

    canonical_img = nib.as_closest_canonical(hu_img)

    valid_img = nib.Nifti1Image(
        valid_source.astype(np.uint8),
        source_img.affine,
        source_img.header,
    )
    valid_img.set_sform(
        source_img.get_sform(),
        code=int(source_img.header["sform_code"]),
    )
    valid_img.set_qform(
        source_img.get_qform(),
        code=int(source_img.header["qform_code"]),
    )
    canonical_valid_img = nib.as_closest_canonical(valid_img)
    valid_xyz = canonical_valid_img.get_fdata(dtype=np.float32) > 0.5

    # Resample both onto the same 1 mm output grid.
    canonical_img = resample_to_output(
        canonical_img, voxel_sizes=(1.0, 1.0, 1.0), order=1
    )
    valid_resampled = resample_to_output(
        canonical_valid_img, voxel_sizes=(1.0, 1.0, 1.0), order=0
    )

    hu_xyz = canonical_img.get_fdata(dtype=np.float32)
    valid_xyz = valid_resampled.get_fdata(dtype=np.float32) > 0.5

    # Prevent interpolated values outside the source image from being treated
    # as real CT voxels.
    hu_xyz[~valid_xyz] = -1024.0

    return canonical_img, hu_xyz, valid_xyz


def extract_patient_id(volume_or_path: str) -> str:
    """
    Extracts canonical subject/patient ID from CT-RATE filename or path.
    Example: 'train_10001_a_1.nii.gz' -> 'patient_train_10001' or '10001'
    """
    name = Path(volume_or_path).name
    match = re.search(r"((?:train|valid)_\d+)", name)
    if match:
        return match.group(1)
    match = re.search(r"(\d{4,})", name)
    if match:
        return f"patient_{match.group(1)}"
    return name.split(".")[0]


def verify_split_leakage(
    train_patients: set[str],
    val_patients: set[str],
    test_patients: set[str],
) -> None:
    """Asserts zero patient overlap between train, val, and test splits."""
    train_val_leak = train_patients.intersection(val_patients)
    train_test_leak = train_patients.intersection(test_patients)
    val_test_leak = val_patients.intersection(test_patients)

    if train_val_leak or train_test_leak or val_test_leak:
        raise AssertionError(
            "Patient leakage detected between splits!\n"
            f"Train/Val overlap ({len(train_val_leak)}): {train_val_leak}\n"
            f"Train/Test overlap ({len(train_test_leak)}): {train_test_leak}\n"
            f"Val/Test overlap ({len(val_test_leak)}): {val_test_leak}"
        )
    logger.info("Split verification successful: Zero patient leakage across train, val, and test.")


def create_tiered_splits(
    train_metadata: pd.DataFrame,
    valid_metadata: pd.DataFrame,
    tier: str = "lean",
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """Constructs patient-disjoint train, val, and test subsets by configuration tier."""
    if tier not in TIERS:
        raise ValueError(f"Unknown CT-RATE tier {tier!r}; choose one of {sorted(TIERS)}.")
    for name, frame in (("train", train_metadata), ("valid", valid_metadata)):
        if "VolumeName" not in frame.columns:
            raise ValueError(f"{name} metadata must contain a 'VolumeName' column.")
        if frame.empty:
            raise ValueError(f"{name} metadata is empty.")
        if (
            frame["VolumeName"].isna().any()
            or (frame["VolumeName"].astype(str).str.strip() == "").any()
        ):
            raise ValueError(f"{name} metadata contains empty VolumeName values.")
        if frame["VolumeName"].astype(str).duplicated().any():
            raise ValueError(f"{name} metadata contains duplicate VolumeName entries.")
    tier_cfg = TIERS[tier]
    n_train = tier_cfg["train"]
    n_val = tier_cfg["val"]
    n_test = tier_cfg["test"]

    train_df = train_metadata.copy()
    valid_df = valid_metadata.copy()

    if "patient_id" not in train_df.columns:
        train_df["patient_id"] = train_df["VolumeName"].apply(extract_patient_id)
    if "patient_id" not in valid_df.columns:
        valid_df["patient_id"] = valid_df["VolumeName"].apply(extract_patient_id)

    unique_train_patients = np.array(train_df["patient_id"].unique())
    rng = np.random.RandomState(seed)
    rng.shuffle(unique_train_patients)

    if len(unique_train_patients) < (n_train + n_val):
        raise ValueError(
            "Not enough unique train subjects "
            f"({len(unique_train_patients)}) for train={n_train} and val={n_val}."
        )

    train_patient_set = set(unique_train_patients[:n_train])
    val_patient_set = set(unique_train_patients[n_train : n_train + n_val])

    unique_valid_patients = np.array(valid_df["patient_id"].unique())
    rng_test = np.random.RandomState(seed + 1)
    rng_test.shuffle(unique_valid_patients)

    filtered_test_patients = [
        patient
        for patient in unique_valid_patients
        if patient not in train_patient_set and patient not in val_patient_set
    ]
    if len(filtered_test_patients) < n_test:
        raise ValueError(
            f"Not enough unique valid subjects ({len(filtered_test_patients)}) for test={n_test}."
        )

    test_patient_set = set(filtered_test_patients[:n_test])
    verify_split_leakage(train_patient_set, val_patient_set, test_patient_set)

    def select_canonical_volumes(
        df: pd.DataFrame,
        target_patients: set[str],
        split_name: str,
    ) -> pd.DataFrame:
        subset = df[df["patient_id"].isin(target_patients)].copy()
        subset["is_pref_suffix"] = subset["VolumeName"].str.contains(r"_a_1\.nii\.gz|_a\.nii\.gz")
        subset = subset.sort_values(by=["patient_id", "is_pref_suffix"], ascending=[True, False])
        canonical = subset.drop_duplicates(subset=["patient_id"], keep="first").copy()
        canonical["split"] = split_name
        return canonical.reset_index(drop=True)

    split_train_df = select_canonical_volumes(train_df, train_patient_set, "ctrate_train")
    split_val_df = select_canonical_volumes(train_df, val_patient_set, "ctrate_val")
    split_test_df = select_canonical_volumes(valid_df, test_patient_set, "ctrate_test")

    return {
        "train": split_train_df,
        "val": split_val_df,
        "test": split_test_df,
    }


class CTRATEStreamer:
    """Streams single CT-RATE volumes from Hugging Face, extracts patches, and deletes raws."""

    def __init__(
        self,
        repo_id: str = "ibrahimhamamci/CT-RATE",
        temp_dir: str | Path = "data/interim/ctrate_temp",
        patch_size: int = 50,
        max_patches_per_scan: int = 100,
        use_lobe_segmentation: bool = False,
    ) -> None:
        self.repo_id = repo_id
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.patch_size = patch_size
        self.max_patches_per_scan = max_patches_per_scan
        self.sampler = UniformLungPatchSampler(patch_size=patch_size)
        self.use_lobe_segmentation = use_lobe_segmentation
        self.segmenter = LungSegmenter(
            backend="totalsegmentator" if use_lobe_segmentation else "threshold"
        )
        
    def stream_and_process_scan(
        self,
        volume_name: str,
        patient_id: str,
        writer: ShardedPatchWriter,
        slope: float,
        intercept: float,
    ) -> int:
        """Streams a single scan and extracts patches while enforcing cleanup."""
        token = os.environ.get("HF_TOKEN") or os.environ.get("HF_READ")
        downloaded_path: Path | None = None
        stage = "validate_remote_path"

        try:
            from huggingface_hub import hf_hub_download

            basename = Path(volume_name).name
            if basename != volume_name or not basename.endswith(".nii.gz"):
                raise ValueError(
                    f"VolumeName must be a plain .nii.gz filename, got {volume_name!r}."
                )
            if basename.startswith("train_"):
                subfolder = "train"
            elif basename.startswith("valid_"):
                subfolder = "valid"
            else:
                raise ValueError(
                    f"Cannot determine CT-RATE split from filename {basename!r}; "
                    "expected train_* or valid_*."
                )
            stage = "download"
            logger.info("Downloading %s from %s...", volume_name, self.repo_id)
            remote_path = ctrate_remote_path(basename)

            local_file = hf_hub_download(
                repo_id=self.repo_id,
                filename=remote_path,
                repo_type="dataset",
                token=token,
                local_dir=self.temp_dir,
            )
            downloaded_path = Path(local_file)
                       stage = "load_and_resample"

            img, vol_xyz, valid_xyz = load_ctrate_hu(
                downloaded_path,
                slope=slope,
                intercept=intercept,
            )

            if not np.isfinite(vol_xyz).all():
                raise ValueError(
                    f"Non-finite HU values found in {volume_name}."
                )

            if not np.any(valid_xyz):
                raise ValueError(
                    f"No valid voxels found in {volume_name}."
                )

            intensity_min = float(np.min(vol_xyz[valid_xyz]))
            intensity_max = float(np.max(vol_xyz[valid_xyz]))

            if intensity_min < -2000 or intensity_max > 5000:
                raise ValueError(
                    f"Implausible HU range [{intensity_min:.1f}, "
                    f"{intensity_max:.1f}] for {volume_name}."
                )

            logger.info(
                "%s: HU range=[%.1f, %.1f], shape=%s",
                volume_name,
                intensity_min,
                intensity_max,
                vol_xyz.shape,
            )

            vol_zyx = np.transpose(vol_xyz, (2, 1, 0))
            valid_zyx = np.transpose(valid_xyz, (2, 1, 0))
            spacing_zyx = (1.0, 1.0, 1.0)
            stage = "segmentation"
            mask = (
                self.segmenter.segment_lobes(vol_zyx, spacing_zyx)
                if self.use_lobe_segmentation
                else self.segmenter.segment(vol_zyx, spacing_zyx)
            )
            if mask.shape != vol_zyx.shape:
                raise ValueError(
                    f"Mask shape {mask.shape} does not match CT shape {vol_zyx.shape}."
                )
            if not np.isfinite(mask).all() or not np.any(mask > 0):
                raise ValueError("Lung segmentation returned an empty or non-finite mask.")
            lung_mask = (mask > 0).astype(np.uint8)
            mask_fraction = float(np.mean(lung_mask > 0))
            if not 0.01 <= mask_fraction <= 0.65:
                raise ValueError(
                    f"Implausible lung-mask fraction {mask_fraction:.3f}; expected 0.01–0.65. "
                    "Inspect HU scaling and threshold segmentation before accepting this scan."
                )
            touches_xy_border = bool(
                np.any(lung_mask[:, 0, :])
                or np.any(lung_mask[:, -1, :])
                or np.any(lung_mask[:, :, 0])
                or np.any(lung_mask[:, :, -1])
            )
            if touches_xy_border:
                raise ValueError(
                    "Lung mask touches a lateral image border; possible exterior-air leakage. "
                    "Rejecting this scan rather than extracting potentially invalid patches."
                )
            logger.info(
                "%s: lung-mask fraction=%.3f, voxels=%d",
                volume_name,
                mask_fraction,
                int(np.count_nonzero(lung_mask)),
            )
            stage = "patch_extraction"
           patches, records = self.sampler.extract_patches(
    volume_insp=vol_zyx,
    volume_exp=None,
    lung_mask=lung_mask,
    lobe_mask=mask if self.use_lobe_segmentation else None,
    valid_mask=valid_zyx,
    max_patches_per_scan=self.max_patches_per_scan,
    patient_id=patient_id,
    scan_id=volume_name.removesuffix(".nii.gz"),
)

            if len(records) == 0:
                logger.warning(
                    "%s: patch extraction returned zero patches; check mask coverage and sampler thresholds.",
                    volume_name,
                )
                return 0
            stage = "write_shards"
            writer.add_patches(patches, records)
            logger.info("Successfully processed %s: %d patches saved.", volume_name, len(records))
            return len(records)

        except Exception as exc:
            logger.exception(
                "CT-RATE volume failed: volume=%s stage=%s exception=%s",
                volume_name,
                stage,
                type(exc).__name__,
            )
            raise RuntimeError(
                f"CT-RATE volume {volume_name!r} failed at stage={stage} "
                f"({type(exc).__name__}): {exc}"
            ) from exc

        finally:
            if downloaded_path and downloaded_path.exists():
                try:
                    downloaded_path.unlink()
                    logger.debug(f"Deleted temporary raw NIfTI: {downloaded_path.name}")
                except Exception as exc:
                    logger.warning(f"Could not delete temporary file {downloaded_path}: {exc}")
