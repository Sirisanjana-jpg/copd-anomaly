"""
Extract 3D volumetric patches and generate uint8 sharded datasets.

Only emits a manifest when real scans, masks, and eligible patches are present.
The current patching path does not assign five-lobe labels; lobe-wise features must
remain disabled until validated lobe masks are connected.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np

from copd_anomaly.patches.dataset import ShardedPatchWriter
from copd_anomaly.patches.sampler import UniformLungPatchSampler
from copd_anomaly.segmentation.segmenter import validate_lobe_mask
from copd_anomaly.utils.helpers import get_logger

logger = get_logger("make_patches")


def extract_and_shard_patches(
    nifti_dir: Path,
    masks_dir: Path,
    output_dir: Path,
    lobe_masks_dir: Path | None = None,
    patch_size: int = 50,
    max_patches_per_scan: int = 100,
    shard_size: int = 1000,
    cases: list[str] | None = None,
) -> Path:
    if patch_size < 1 or max_patches_per_scan < 1 or shard_size < 1:
        raise ValueError("patch_size, max_patches_per_scan and shard_size must be positive.")
    if not nifti_dir.is_dir():
        raise FileNotFoundError(f"NIfTI input directory not found: {nifti_dir}")
    if not masks_dir.is_dir():
        raise FileNotFoundError(f"Lung-mask directory not found: {masks_dir}")
    if lobe_masks_dir is not None and not lobe_masks_dir.is_dir():
        raise FileNotFoundError(f"Lobe-mask directory not found: {lobe_masks_dir}")

    nifti_files = sorted(nifti_dir.glob("*_insp.nii.gz"))
    if cases:
        nifti_files = [path for path in nifti_files if path.name.split("_")[0] in cases]
    if not nifti_files:
        raise FileNotFoundError(
            f"No inspiratory NIfTI scans matched in {nifti_dir} for cases={cases}."
        )

    existing_manifest = output_dir / "patch_manifest.parquet"
    if existing_manifest.exists() or (output_dir / "patch_manifest.csv").exists():
        raise FileExistsError(
            f"A patch manifest already exists under {output_dir}. Choose a fresh output directory "
            "to avoid mixing old and new shards."
        )
    sampler = UniformLungPatchSampler(patch_size=patch_size, min_lung_fraction=0.70)
    writer = ShardedPatchWriter(output_dir=output_dir, shard_size=shard_size)
    total_scans_processed = 0
    total_patches = 0
    missing_masks = []

    for nifti_path in nifti_files:
        scan_id = nifti_path.name.replace(".nii.gz", "")
        case_id = scan_id.split("_")[0]
        mask_path = masks_dir / f"{scan_id}_mask.nii.gz"
        if not mask_path.is_file():
            missing_masks.append(scan_id)
            logger.warning("Lung mask not found for %s; skipping this scan.", scan_id)
            continue

        logger.info("Extracting patches from %s...", scan_id)
        img_insp = nib.load(str(nifti_path))
        vol_insp = np.transpose(img_insp.get_fdata(dtype=np.float32), (2, 1, 0))
        img_mask = nib.load(str(mask_path))
        mask = np.transpose(img_mask.get_fdata(dtype=np.float32), (2, 1, 0))
        if vol_insp.shape != mask.shape or not np.allclose(
            img_insp.affine, img_mask.affine, rtol=0.0, atol=1e-4
        ):
            raise ValueError(
                f"Volume/mask geometry mismatch for {scan_id}: volume shape={vol_insp.shape}, "
                f"mask shape={mask.shape}. Check image affines and resampling before patch extraction."
            )
        if not np.isfinite(vol_insp).all() or not np.isfinite(mask).all():
            raise ValueError(f"Non-finite voxels found in volume or mask for {scan_id}.")
        mask = (mask > 0).astype(np.uint8)
        if not mask.any():
            logger.warning("Lung mask for %s is empty; skipping scan.", scan_id)
            continue

        lobe_mask = None
        if lobe_masks_dir is not None:
            lobe_path = lobe_masks_dir / f"{scan_id.replace('_insp', '')}_lobes.nii.gz"
            if not lobe_path.is_file():
                raise FileNotFoundError(
                    f"Requested lobe mask missing for {scan_id}: {lobe_path}. "
                    "Do not silently continue a lobe-wise dataset without anatomical labels."
                )
            lobe_img = nib.load(str(lobe_path))
            lobe_values = np.transpose(lobe_img.get_fdata(dtype=np.float32), (2, 1, 0))
            if lobe_values.shape != vol_insp.shape or not np.allclose(
                lobe_img.affine, img_insp.affine, rtol=0.0, atol=1e-4
            ):
                raise ValueError(
                    f"Lobe-mask geometry mismatch for {scan_id}: lobe shape={lobe_values.shape}, "
                    f"CT shape={vol_insp.shape}. Re-run segmentation on the exact CT grid."
                )
            lobe_mask = validate_lobe_mask(lobe_values, expected_shape=vol_insp.shape)
            outside_fraction = float(np.mean((lobe_mask > 0) & (mask == 0)))
            if outside_fraction > 0.01:
                raise ValueError(
                    f"Lobe mask for {scan_id} extends outside the lung mask in "
                    f"{outside_fraction:.2%} of voxels; review segmentation alignment."
                )

        registered_exp_path = Path("data/processed/registered") / f"{case_id}_exp_warped.nii.gz"
        vol_exp = None
        if registered_exp_path.is_file():
            exp_img = nib.load(str(registered_exp_path))
            vol_exp = np.transpose(exp_img.get_fdata(dtype=np.float32), (2, 1, 0))
            if vol_exp.shape != vol_insp.shape or not np.allclose(
                exp_img.affine, img_insp.affine, rtol=0.0, atol=1e-4
            ):
                raise ValueError(
                    f"Registered expiratory volume geometry mismatch for {case_id}: "
                    f"{vol_exp.shape} vs {vol_insp.shape}; verify shape and affine alignment."
                )

        # A lobe mask is optional for global anomaly modelling, but when requested it
        # must be a validated five-label mask aligned to the CT; no lobe IDs are fabricated.
        patches, records = sampler.extract_patches(
            volume_insp=vol_insp,
            volume_exp=vol_exp,
            lung_mask=mask,
            lobe_mask=lobe_mask,
            max_patches_per_scan=max_patches_per_scan,
            patient_id=case_id,
            scan_id=scan_id,
        )
        if len(records) == 0:
            logger.warning("No eligible patches extracted from %s.", scan_id)
            continue
        writer.add_patches(patches, records)
        total_scans_processed += 1
        total_patches += len(records)

    if total_patches == 0:
        raise RuntimeError(
            "Patch extraction produced zero patches; no manifest was written. "
            f"Scans considered={len(nifti_files)}, missing masks={missing_masks}. "
            "Inspect mask alignment, mask contents, patch size, and lung-fraction threshold."
        )

    manifest_path = writer.close()
    if not manifest_path.is_file():
        raise RuntimeError(f"Patch writer did not create expected manifest: {manifest_path}")
    logger.info(
        "Wrote %d real patches from %d scans to %s. Lobe IDs are not available in this pipeline.",
        total_patches,
        total_scans_processed,
        manifest_path,
    )
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract real 3D patches into sharded datasets.")
    parser.add_argument("--nifti-dir", type=Path, default=Path("data/processed/nifti"))
    parser.add_argument("--masks-dir", type=Path, default=Path("data/processed/masks"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/patches"))
    parser.add_argument(
        "--lobe-masks-dir",
        type=Path,
        default=None,
        help="Optional directory of validated *_lobes.nii.gz five-label masks.",
    )
    parser.add_argument("--patch-size", type=int, default=50)
    parser.add_argument("--max-patches", type=int, default=100)
    parser.add_argument("--shard-size", type=int, default=1000)
    parser.add_argument("--cases", nargs="*", default=None)
    args = parser.parse_args()
    extract_and_shard_patches(
        nifti_dir=args.nifti_dir,
        masks_dir=args.masks_dir,
        output_dir=args.output_dir,
        lobe_masks_dir=args.lobe_masks_dir,
        patch_size=args.patch_size,
        max_patches_per_scan=args.max_patches,
        shard_size=args.shard_size,
        cases=args.cases,
    )


if __name__ == "__main__":
    main()
