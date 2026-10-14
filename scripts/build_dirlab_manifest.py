"""
Build DIR-Lab master manifest and convert raw scans to NIfTI.

Creates:
1. data/processed/nifti/{case_id}_{phase}.nii.gz
2. data/processed/dirlab_manifest.parquet
3. data/processed/dirlab_manifest.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

from copd_anomaly.data.dirlab_loader import DIRLabLoader
from copd_anomaly.utils.helpers import get_logger

logger = get_logger("build_dirlab_manifest")


def build_manifest(
    zips_dir: Path,
    config_path: Path,
    interim_dir: Path,
    output_nifti_dir: Path,
    manifest_parquet_path: Path,
) -> pd.DataFrame:
    loader = DIRLabLoader(config_path)
    with open(config_path, encoding="utf-8") as f:
        cases_config = yaml.safe_load(f).get("cases", {})

    records = []

    for case_id, case_info in sorted(cases_config.items()):
        zip_path = zips_dir / f"{case_id}.zip"
        if not zip_path.exists():
            logger.warning(f"Zip archive not found for {case_id}: {zip_path}. Skipping.")
            continue

        logger.info(f"Processing case: {case_id}...")
        case_interim = interim_dir / case_id
        extracted = loader.extract_case(zip_path, case_interim)

        dims_xyz = case_info["dims"]
        spacing_xyz = case_info["spacing_mm"]  # [sx, sy, sz]
        spacing_zyx = (spacing_xyz[2], spacing_xyz[1], spacing_xyz[0])

        for phase in ["insp", "exp"]:
            img_key = f"{phase}_img"
            lm_key = f"{phase}_landmarks"

            if img_key not in extracted:
                logger.error(f"Missing {img_key} for {case_id}")
                continue

            img_path = extracted[img_key]
            # Check for alternate dims (e.g. copd2)
            actual_size = img_path.stat().st_size
            primary_size = 2 * dims_xyz[0] * dims_xyz[1] * dims_xyz[2]
            target_dims = dims_xyz

            if actual_size != primary_size and "dims_alt" in case_info:
                alt_dims = case_info["dims_alt"]
                alt_size = 2 * alt_dims[0] * alt_dims[1] * alt_dims[2]
                if actual_size == alt_size:
                    target_dims = alt_dims
                    logger.info(f"Using alternate dims {alt_dims} for {case_id}")

            # Read and convert to HU
            vol_hu = loader.read_raw_volume(img_path, target_dims)
            shape_zyx = vol_hu.shape

            # Verify HU offset
            is_valid_hu, median_hu = loader.verify_hu_offset(vol_hu)
            logger.info(
                f"  {case_id}_{phase} median lung HU: {median_hu:.1f} (valid: {is_valid_hu})"
            )

            # Save NIfTI if not already present
            nifti_path = output_nifti_dir / f"{case_id}_{phase}.nii.gz"
            if not nifti_path.exists() or nifti_path.stat().st_size == 0:
                loader.save_nifti(vol_hu, spacing_zyx, nifti_path)
            else:
                logger.info(f"  NIfTI already exists: {nifti_path.name}")

            # Landmarks handling
            lm_path = extracted.get(lm_key)
            base_idx = None
            in_lung_frac = None
            if lm_path:
                lung_mask = loader.crude_lung_mask(vol_hu)
                _, base_idx, in_lung_frac = loader.load_landmarks(lm_path, shape_zyx, lung_mask)
                del lung_mask

            del vol_hu
            import gc
            gc.collect()

            records.append(
                {
                    "dataset": "dirlab",
                    "patient_id": case_id,
                    "scan_id": f"{case_id}_{phase}",
                    "phase": phase,
                    "split": "dirlab_all",
                    "role": "evaluation",
                    "path": str(nifti_path),
                    "spacing_z_mm": float(spacing_zyx[0]),
                    "spacing_y_mm": float(spacing_zyx[1]),
                    "spacing_x_mm": float(spacing_zyx[2]),
                    "dim_z": int(shape_zyx[0]),
                    "dim_y": int(shape_zyx[1]),
                    "dim_x": int(shape_zyx[2]),
                    "hu_offset_verified": bool(is_valid_hu),
                    "lung_median_hu": float(median_hu),
                    "landmarks_path": str(lm_path) if lm_path else None,
                    "landmark_index_base": int(base_idx) if base_idx is not None else None,
                    "landmark_in_lung_frac": float(in_lung_frac)
                    if in_lung_frac is not None
                    else None,
                }
            )

    if not records:
        raise RuntimeError("No records were successfully generated!")

    df = pd.DataFrame(records)
    manifest_parquet_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(manifest_parquet_path, index=False)
    csv_path = manifest_parquet_path.with_suffix(".csv")
    df.to_csv(csv_path, index=False)

    logger.info(f"Manifest written to {manifest_parquet_path} ({len(df)} scans)")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Build DIR-Lab dataset manifest.")
    parser.add_argument("--zips-dir", type=Path, default=Path("data/raw_zips"))
    parser.add_argument("--config", type=Path, default=Path("configs/data/dirlab_cases.yaml"))
    parser.add_argument("--interim-dir", type=Path, default=Path("data/interim/dirlab_raw"))
    parser.add_argument("--output-nifti-dir", type=Path, default=Path("data/processed/nifti"))
    parser.add_argument(
        "--manifest-parquet",
        type=Path,
        default=Path("data/processed/dirlab_manifest.parquet"),
    )
    args = parser.parse_args()

    try:
        build_manifest(
            args.zips_dir,
            args.config,
            args.interim_dir,
            args.output_nifti_dir,
            args.manifest_parquet,
        )
        sys.exit(0)
    except Exception as e:
        logger.error(f"Failed to build manifest: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
