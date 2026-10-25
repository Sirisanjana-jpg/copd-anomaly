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
from nibabel.processing import resample_to_output
import pandas as pd

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
    tier_cfg = TIERS.get(tier, TIERS["lean"])
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
    ) -> None:
        self.repo_id = repo_id
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.patch_size = patch_size
        self.max_patches_per_scan = max_patches_per_scan
        self.sampler = UniformLungPatchSampler(patch_size=patch_size)
        self.segmenter = LungSegmenter(backend="threshold")

    def stream_and_process_scan(
        self,
        volume_name: str,
        patient_id: str,
        writer: ShardedPatchWriter,
    ) -> int:
        """Streams a single scan and extracts patches while enforcing cleanup."""
        token = os.environ.get("HF_TOKEN") or os.environ.get("HF_READ")
        downloaded_path: Path | None = None

        try:
            from huggingface_hub import hf_hub_download

            logger.info(f"Downloading {volume_name} from {self.repo_id}...")
            subfolder = "train" if "train" in volume_name else "valid"
            remote_path = f"dataset/{subfolder}/{volume_name}"

            local_file = hf_hub_download(
                repo_id=self.repo_id,
                filename=remote_path,
                repo_type="dataset",
                token=token,
                local_dir=self.temp_dir,
            )
            downloaded_path = Path(local_file)

            # Normalize orientation before resampling to an isotropic 1 mm grid.
            # Interpolation is linear for intensities; segmentation is derived after resampling.
            img = nib.as_closest_canonical(nib.load(str(downloaded_path)))
            img = resample_to_output(img, voxel_sizes=(1.0, 1.0, 1.0), order=1)
            vol_xyz = img.get_fdata(dtype=np.float32)
            if not np.isfinite(vol_xyz).all():
                raise ValueError(f"Non-finite CT intensities found in {volume_name}.")
            intensity_min = float(np.min(vol_xyz))
            intensity_max = float(np.max(vol_xyz))
            if intensity_min >= -200.0 or intensity_min < -5000.0 or intensity_max > 10000.0:
                raise ValueError(
                    f"Intensity range [{intensity_min:.1f}, {intensity_max:.1f}] for {volume_name} "
                    "does not look like HU-scaled CT. Verify NIfTI rescale slope/intercept."
                )
            logger.info(
                "%s: isotropic 1 mm grid shape=%s, intensity range=[%.1f, %.1f] HU",
                volume_name, vol_xyz.shape, intensity_min, intensity_max,
            )
            vol_zyx = np.transpose(vol_xyz, (2, 1, 0))
            spacing_zyx = (1.0, 1.0, 1.0)

            mask = self.segmenter.segment(vol_zyx, spacing_zyx)
            patches, records = self.sampler.extract_patches(
                volume_insp=vol_zyx,
                volume_exp=None,
                lung_mask=mask,
                max_patches_per_scan=self.max_patches_per_scan,
                patient_id=patient_id,
                scan_id=volume_name.split(".")[0],
            )

            if len(records) > 0:
                writer.add_patches(patches, records)
                logger.info(f"Successfully processed {volume_name}: {len(records)} patches saved.")
            return len(records)

        except Exception as exc:
            logger.error(f"Failed to process remote volume {volume_name}: {exc}")
            raise RuntimeError(f"Failed to process remote volume {volume_name}") from exc

        finally:
            if downloaded_path and downloaded_path.exists():
                try:
                    downloaded_path.unlink()
                    logger.debug(f"Deleted temporary raw NIfTI: {downloaded_path.name}")
                except Exception as exc:
                    logger.warning(f"Could not delete temporary file {downloaded_path}: {exc}")
