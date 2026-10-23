"""
Extract 3D volumetric patches and generate uint8 sharded dataset.

Reads processed NIfTI scans and lung masks, extracts patches with >=70% lung parenchyma,
and saves them into compact uint8 .npy shards with a master parquet index.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import nibabel as nib
import numpy as np
import yaml

from copd_anomaly.patches.dataset import ShardedPatchWriter
from copd_anomaly.patches.sampler import UniformLungPatchSampler
from copd_anomaly.utils.helpers import get_logger

logger = get_logger("make_patches")


def extract_and_shard_patches(
    nifti_dir: Path,
    masks_dir: Path,
    output_dir: Path,
    patch_size: int = 50,
    max_patches_per_scan: int = 100,
    shard_size: int = 1000,
    cases: list[str] | None = None,
) -> Path:
    sampler = UniformLungPatchSampler(patch_size=patch_size, min_lung_fraction=0.70)
    writer = ShardedPatchWriter(output_dir=output_dir, shard_size=shard_size)

    nifti_files = sorted(nifti_dir.glob("*_insp.nii.gz"))
    if not nifti_files:
        logger.warning(f"No inspiratory scans found in {nifti_dir}")

    total_scans_processed = 0

    for nifti_path in nifti_files:
        scan_id = nifti_path.name.replace(".nii.gz", "")
        case_id = scan_id.split("_")[0]
        if cases and case_id not in cases:
            continue

        mask_path = masks_dir / f"{scan_id}_mask.nii.gz"
        if not mask_path.exists():
            logger.warning(f"Mask not found for {scan_id}; skipping.")
            continue

        logger.info(f"Extracting patches from {scan_id}...")
        img_insp = nib.load(str(nifti_path))
        vol_insp = np.transpose(img_insp.get_fdata().astype(np.float32), (2, 1, 0))

        img_mask = nib.load(str(mask_path))
        mask = np.transpose(img_mask.get_fdata().astype(np.uint8), (2, 1, 0))

        # Check for registered exhale scan
        registered_exp_path = Path("data/processed/registered") / f"{case_id}_exp_warped.nii.gz"
        vol_exp = None
        if registered_exp_path.exists():
            vol_exp = np.transpose(
                nib.load(str(registered_exp_path)).get_fdata().astype(np.float32), (2, 1, 0)
            )

        patches, records = sampler.extract_patches(
            volume_insp=vol_insp,
            volume_exp=vol_exp,
            lung_mask=mask,
            lobe_mask=None,
            max_patches_per_scan=max_patches_per_scan,
            patient_id=case_id,
            scan_id=scan_id,
        )

        if len(records) > 0:
            writer.add_patches(patches, records)
            total_scans_processed += 1

    manifest_path = writer.close()
    logger.info(f"Finished patch extraction across {total_scans_processed} scans.")
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract 3D patches into sharded dataset.")
    parser.add_argument("--nifti-dir", type=Path, default=Path("data/processed/nifti"))
    parser.add_argument("--masks-dir", type=Path, default=Path("data/processed/masks"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/patches"))
    parser.add_argument("--patch-size", type=int, default=50)
    parser.add_argument("--max-patches", type=int, default=100)
    parser.add_argument("--shard-size", type=int, default=1000)
    parser.add_argument("--cases", nargs="*", default=None)
    args = parser.parse_args()

    extract_and_shard_patches(
        nifti_dir=args.nifti_dir,
        masks_dir=args.masks_dir,
        output_dir=args.output_dir,
        patch_size=args.patch_size,
        max_patches_per_scan=args.max_patches,
        shard_size=args.shard_size,
        cases=args.cases,
    )


if __name__ == "__main__":
    main()
