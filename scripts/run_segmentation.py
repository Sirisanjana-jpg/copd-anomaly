"""
Execute lung parenchyma and lobe segmentation across processed NIfTI scans.

Saves binary/multilabel masks to data/processed/masks/{scan_id}_mask.nii.gz.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import nibabel as nib
import numpy as np

from copd_anomaly.segmentation.segmenter import LungSegmenter
from copd_anomaly.utils.helpers import get_logger

logger = get_logger("run_segmentation")


def segment_scans(
    nifti_dir: Path,
    output_mask_dir: Path,
    backend: str = "auto",
    cases: list[str] | None = None,
) -> None:
    output_mask_dir.mkdir(parents=True, exist_ok=True)
    segmenter = LungSegmenter(backend=backend)

    nifti_files = sorted(nifti_dir.glob("*.nii.gz"))
    if not nifti_files:
        logger.warning(f"No NIfTI scans found in {nifti_dir}")
        return

    for nifti_path in nifti_files:
        scan_id = nifti_path.name.replace(".nii.gz", "")
        case_id = scan_id.split("_")[0]
        if cases and case_id not in cases:
            continue

        mask_out = output_mask_dir / f"{scan_id}_mask.nii.gz"
        if mask_out.exists() and mask_out.stat().st_size > 0:
            logger.info(f"Mask already exists: {mask_out.name}. Skipping.")
            continue

        logger.info(f"Segmenting {scan_id}...")
        img = nib.load(str(nifti_path))
        # NIfTI is (X, Y, Z); transpose to (Z, Y, X) for segmenter
        vol_xyz = img.get_fdata().astype(np.float32)
        vol_zyx = np.transpose(vol_xyz, (2, 1, 0))
        zooms_xyz = img.header.get_zooms()[:3]
        spacing_zyx = (float(zooms_xyz[2]), float(zooms_xyz[1]), float(zooms_xyz[0]))

        mask_zyx = segmenter.segment(vol_zyx, spacing_zyx)
        mask_xyz = np.transpose(mask_zyx, (2, 1, 0))

        mask_nib = nib.Nifti1Image(mask_xyz, img.affine, img.header)
        nib.save(mask_nib, str(mask_out))
        voxels = int(np.sum(mask_zyx > 0))
        vol_ml = voxels * np.prod(spacing_zyx) / 1000.0
        logger.info(f"  Saved mask: {mask_out.name} ({voxels:,} voxels, {vol_ml:.1f} mL)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run lung segmentation.")
    parser.add_argument("--nifti-dir", type=Path, default=Path("data/processed/nifti"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/masks"))
    parser.add_argument("--backend", type=str, default="auto", choices=["auto", "threshold", "totalsegmentator", "lungmask"])
    parser.add_argument("--cases", nargs="*", default=None, help="Optional cases to filter.")
    args = parser.parse_args()

    segment_scans(args.nifti_dir, args.output_dir, backend=args.backend, cases=args.cases)


if __name__ == "__main__":
    main()
