"""Generate validated five-lobe masks from real NIfTI CT volumes.

Requires the optional TotalSegmentator backend. No threshold or synthetic fallback is
permitted: this command fails if a full five-lobe mask cannot be generated.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to

from copd_anomaly.segmentation.segmenter import LungSegmenter, validate_lobe_mask
from copd_anomaly.utils.helpers import get_logger

logger = get_logger("segment_lobes")

LOBE_LABELS = {
    1: "right_upper_lobe",
    2: "right_middle_lobe",
    3: "right_lower_lobe",
    4: "left_upper_lobe",
    5: "left_lower_lobe",
}


def segment_directory(
    input_dir: Path, output_dir: Path, cases: list[str] | None = None
) -> list[Path]:
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input NIfTI directory does not exist: {input_dir}")
    scans = sorted(input_dir.glob("*_insp.nii.gz"))
    if cases:
        scans = [path for path in scans if path.name.split("_")[0] in cases]
    if not scans:
        raise FileNotFoundError(
            f"No *_insp.nii.gz scans found under {input_dir} for cases={cases}."
        )

    segmenter = LungSegmenter(backend="totalsegmentator")
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for scan_path in scans:
        logger.info("Segmenting five lobes for %s", scan_path.name)
        original = nib.load(str(scan_path))
        canonical = nib.as_closest_canonical(original)
        volume_xyz = canonical.get_fdata(dtype=np.float32)
        if not np.isfinite(volume_xyz).all():
            raise ValueError(f"Non-finite CT values in {scan_path}.")
        zooms_xyz = canonical.header.get_zooms()[:3]
        spacing_zyx = (float(zooms_xyz[2]), float(zooms_xyz[1]), float(zooms_xyz[0]))
        volume_zyx = np.transpose(volume_xyz, (2, 1, 0))
        labels_zyx = segmenter.segment_lobes(volume_zyx, spacing_zyx)
        labels_xyz = np.transpose(
            validate_lobe_mask(labels_zyx, expected_shape=volume_zyx.shape), (2, 1, 0)
        )
        canonical_mask = nib.Nifti1Image(labels_xyz, canonical.affine, canonical.header.copy())
        # Map discrete labels back to the exact source scan grid; nearest-neighbour only.
        original_grid_mask = resample_from_to(
            canonical_mask, (original.shape[:3], original.affine), order=0, mode="constant", cval=0
        )
        output_labels = validate_lobe_mask(
            np.transpose(original_grid_mask.get_fdata(dtype=np.float32), (2, 1, 0)),
            expected_shape=(original.shape[2], original.shape[1], original.shape[0]),
        )
        unique = set(np.unique(output_labels).astype(int)) - {0}
        if unique != {1, 2, 3, 4, 5}:
            raise ValueError(
                f"Reoriented lobe mask for {scan_path.name} lost labels: found {sorted(unique)}."
            )
        output_path = output_dir / scan_path.name.replace("_insp.nii.gz", "_lobes.nii.gz")
        output_img = nib.Nifti1Image(
            np.transpose(output_labels, (2, 1, 0)), original.affine, original.header.copy()
        )
        output_img.set_data_dtype(np.uint8)
        nib.save(output_img, str(output_path))
        written.append(output_path)
        logger.info("Saved validated lobe mask %s; labels=%s", output_path, sorted(unique))

    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate true five-lobe masks for real CT scans.")
    parser.add_argument("--nifti-dir", type=Path, default=Path("data/processed/nifti"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/lobes"))
    parser.add_argument("--cases", nargs="*", default=None)
    args = parser.parse_args()
    segment_directory(args.nifti_dir, args.output_dir, args.cases)


if __name__ == "__main__":
    main()
