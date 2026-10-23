"""
Execute deformable registration on DIR-Lab paired breath-hold CT scans.

Calculates Target Registration Error (TRE) in mm using expert landmarks,
evaluates Parametric Response Mapping (PRM) on registered pairs,
and outputs structured reports to reports/.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import nibabel as nib
import numpy as np
import pandas as pd
import yaml

from copd_anomaly.data.dirlab_loader import DIRLabLoader
from copd_anomaly.features.prm import compute_prm_stats
from copd_anomaly.registration.registrar import SimpleITKRegistrar, evaluate_tre
from copd_anomaly.segmentation.segmenter import LungSegmenter
from copd_anomaly.utils.helpers import get_logger

logger = get_logger("run_registration")


def register_pairs(
    nifti_dir: Path,
    masks_dir: Path,
    output_dir: Path,
    reports_dir: Path,
    config_path: Path,
    fast_mode: bool = False,
    cases: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    with open(config_path, encoding="utf-8") as f:
        cases_config = yaml.safe_load(f).get("cases", {})

    loader = DIRLabLoader(config_path)
    registrar = SimpleITKRegistrar(fast_mode=fast_mode)
    segmenter = LungSegmenter(backend="threshold")

    tre_records = []
    prm_records = []

    for case_id in sorted(cases_config.keys()):
        if cases and case_id not in cases:
            continue

        insp_path = nifti_dir / f"{case_id}_insp.nii.gz"
        exp_path = nifti_dir / f"{case_id}_exp.nii.gz"

        if not insp_path.exists() or not exp_path.exists():
            logger.warning(f"Missing NIfTI pairs for {case_id}; skipping.")
            continue

        logger.info(f"=== Registering {case_id} (fast_mode={fast_mode}) ===")
        img_insp = nib.load(str(insp_path))
        img_exp = nib.load(str(exp_path))

        vol_insp = np.transpose(img_insp.get_fdata().astype(np.float32), (2, 1, 0))
        vol_exp = np.transpose(img_exp.get_fdata().astype(np.float32), (2, 1, 0))
        zooms_xyz = img_insp.header.get_zooms()[:3]
        spacing_zyx = (float(zooms_xyz[2]), float(zooms_xyz[1]), float(zooms_xyz[0]))

        # Load or generate masks
        insp_mask_path = masks_dir / f"{case_id}_insp_mask.nii.gz"
        if insp_mask_path.exists():
            mask_insp = np.transpose(nib.load(str(insp_mask_path)).get_fdata().astype(np.uint8), (2, 1, 0))
        else:
            mask_insp = segmenter.segment(vol_insp, spacing_zyx)

        exp_mask_path = masks_dir / f"{case_id}_exp_mask.nii.gz"
        if exp_mask_path.exists():
            mask_exp = np.transpose(nib.load(str(exp_mask_path)).get_fdata().astype(np.uint8), (2, 1, 0))
        else:
            mask_exp = segmenter.segment(vol_exp, spacing_zyx)

        # Execute registration
        reg_result = registrar.register(
            fixed_volume=vol_insp,
            moving_volume=vol_exp,
            fixed_mask=mask_insp,
            moving_mask=mask_exp,
            spacing_zyx=spacing_zyx,
        )

        # Save warped volume
        warped_out = output_dir / f"{case_id}_exp_warped.nii.gz"
        warped_xyz = np.transpose(reg_result.warped_volume, (2, 1, 0))
        warped_nib = nib.Nifti1Image(warped_xyz, img_insp.affine, img_insp.header)
        nib.save(warped_nib, str(warped_out))
        logger.info(f"Saved warped volume: {warped_out.name}")

        # Landmark TRE evaluation
        interim_dir = Path("data/interim/dirlab_raw") / case_id
        insp_lm_file = list(interim_dir.glob("*_iBH_*.txt"))
        exp_lm_file = list(interim_dir.glob("*_eBH_*.txt"))

        if insp_lm_file and exp_lm_file:
            pts_insp, _, _ = loader.load_landmarks(insp_lm_file[0], vol_insp.shape, mask_insp)
            pts_exp, _, _ = loader.load_landmarks(exp_lm_file[0], vol_exp.shape, mask_exp)

            pts_exp_warped = registrar.transform_points(pts_exp, reg_result.transform, spacing_zyx)
            tre_stats = evaluate_tre(pts_insp, pts_exp, pts_exp_warped, spacing_zyx)
            tre_stats["case_id"] = case_id
            tre_records.append(tre_stats)
            logger.info(
                f"  TRE {case_id}: Initial {tre_stats['initial_mean_tre_mm']:.2f} ± {tre_stats['initial_std_tre_mm']:.2f} mm -> "
                f"Final {tre_stats['mean_tre_mm']:.2f} ± {tre_stats['std_tre_mm']:.2f} mm "
                f"({tre_stats['tre_reduction_pct']:.1f}% reduction)"
            )

        # PRM computation
        prm_stats = compute_prm_stats(vol_insp, reg_result.warped_volume, mask_insp, spacing_zyx=spacing_zyx)
        prm_stats["case_id"] = case_id
        prm_records.append(prm_stats)
        logger.info(
            f"  PRM {case_id}: Normal {prm_stats.get('prm_norm_pct', 0):.1f}%, "
            f"fSAD {prm_stats.get('prm_fsad_pct', 0):.1f}%, "
            f"Emph {prm_stats.get('prm_emph_pct', 0):.1f}%"
        )

    df_tre = pd.DataFrame(tre_records)
    if not df_tre.empty:
        tre_csv = reports_dir / "dirlab_registration_tre.csv"
        df_tre.to_csv(tre_csv, index=False)
        logger.info(f"Saved TRE report: {tre_csv}")

    df_prm = pd.DataFrame(prm_records)
    if not df_prm.empty:
        prm_csv = reports_dir / "dirlab_prm_stats.csv"
        df_prm.to_csv(prm_csv, index=False)
        logger.info(f"Saved PRM report: {prm_csv}")

    return df_tre, df_prm


def main() -> None:
    parser = argparse.ArgumentParser(description="Run deformable registration on paired scans.")
    parser.add_argument("--nifti-dir", type=Path, default=Path("data/processed/nifti"))
    parser.add_argument("--masks-dir", type=Path, default=Path("data/processed/masks"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/registered"))
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    parser.add_argument("--config", type=Path, default=Path("configs/data/dirlab_cases.yaml"))
    parser.add_argument("--fast-mode", action="store_true", help="Run with minimal iterations for testing.")
    parser.add_argument("--cases", nargs="*", default=None, help="Filter cases.")
    args = parser.parse_args()

    register_pairs(
        args.nifti_dir,
        args.masks_dir,
        args.output_dir,
        args.reports_dir,
        args.config,
        fast_mode=args.fast_mode,
        cases=args.cases,
    )


if __name__ == "__main__":
    main()
