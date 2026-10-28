"""Extract one numeric feature row per scan and anatomical lobe from CT + lobe masks.

This bridges segmented CT-RATE volumes to the weakly supervised five-lobe MIL model.
CT-RATE report labels supervise scan-level predictions; they do not verify lobe-level
localization. Do not use the 10-case DIR-Lab COPDgene set as a multi-label training
cohort because it does not provide the required 18 finding labels.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

LOBE_LABELS = {
    1: "RUL",
    2: "RML",
    3: "RLL",
    4: "LUL",
    5: "LLL",
}


def _scan_id(path: Path) -> str:
    name = path.name
    for suffix in (".nii.gz", ".nii"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def _region_features(
    values: np.ndarray, spacing_xyz: tuple[float, float, float]
) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float32)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("Empty lobe region encountered.")
    return {
        "voxel_count": int(values.size),
        "volume_ml": float(values.size * np.prod(spacing_xyz) / 1000.0),
        "mean_hu": float(np.mean(values)),
        "median_hu": float(np.median(values)),
        "p15_hu": float(np.percentile(values, 15)),
        "std_hu": float(np.std(values)),
        "low_attenuation_pct_le_minus950": float(np.mean(values <= -950.0) * 100.0),
        "low_attenuation_pct_le_minus910": float(np.mean(values <= -910.0) * 100.0),
    }


def build_lobe_feature_table(ct_dir: Path, lobe_dir: Path, output_csv: Path) -> pd.DataFrame:
    if not ct_dir.is_dir():
        raise FileNotFoundError(f"CT directory does not exist: {ct_dir}")
    if not lobe_dir.is_dir():
        raise FileNotFoundError(f"Lobe-mask directory does not exist: {lobe_dir}")
    ct_paths = sorted(
        p
        for p in ct_dir.rglob("*")
        if p.is_file()
        and (p.name.endswith(".nii.gz") or p.name.endswith(".nii"))
        and not p.name.endswith("_lobes.nii.gz")
    )
    if not ct_paths:
        raise FileNotFoundError(f"No NIfTI CT scans found under {ct_dir}")
    masks_by_name = {p.name: p for p in lobe_dir.rglob("*_lobes.nii.gz") if p.is_file()}
    records: list[dict[str, object]] = []
    seen_ids: set[str] = set()
    for ct_path in ct_paths:
        scan_id = _scan_id(ct_path)
        if scan_id in seen_ids:
            raise ValueError(f"Duplicate scan ID {scan_id}; use unique CT volume filenames.")
        seen_ids.add(scan_id)
        mask_name = f"{scan_id}_lobes.nii.gz"
        mask_path = masks_by_name.get(mask_name)
        if mask_path is None:
            raise FileNotFoundError(
                f"Missing lobe mask for {ct_path}: expected a file named {mask_name}."
            )
        ct_img, mask_img = nib.load(str(ct_path)), nib.load(str(mask_path))
        if ct_img.shape[:3] != mask_img.shape[:3] or not np.allclose(
            ct_img.affine, mask_img.affine, rtol=0.0, atol=1e-3
        ):
            raise ValueError(f"CT/mask geometry mismatch for scan {scan_id}.")
        ct = ct_img.get_fdata(dtype=np.float32)
        mask_float = mask_img.get_fdata(dtype=np.float32)
        if (
            ct.ndim != 3
            or mask_float.ndim != 3
            or not np.isfinite(ct).all()
            or not np.isfinite(mask_float).all()
        ):
            raise ValueError(f"Invalid 3D CT or mask values for scan {scan_id}.")
        if not np.equal(mask_float, np.round(mask_float)).all():
            raise ValueError(f"Non-integer lobe labels for scan {scan_id}.")
        mask = mask_float.astype(np.uint8)
        present = set(np.unique(mask).astype(int)) - {0}
        if present != set(LOBE_LABELS):
            raise ValueError(
                f"Scan {scan_id} must contain lobe labels 1-5; found {sorted(present)}."
            )
        spacing = tuple(float(v) for v in ct_img.header.get_zooms()[:3])
        relative_parts = ct_path.relative_to(ct_dir).parts
        if len(relative_parts) >= 3:
            patient_id = relative_parts[-3]
        elif len(relative_parts) == 2:
            patient_id = relative_parts[0]
        else:
            patient_id = scan_id
        for label_id, lobe_name in LOBE_LABELS.items():
            row: dict[str, object] = {
                "VolumeName": scan_id,
                "patient_id": patient_id,
                "lobe_id": lobe_name,
                "ct_path": str(ct_path),
            }
            row.update(_region_features(ct[mask == label_id], spacing))
            records.append(row)
    frame = pd.DataFrame(records)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_csv, index=False)
    return frame


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ct-dir", type=Path, required=True, help="CT-RATE NIfTI root; scanned recursively."
    )
    parser.add_argument(
        "--lobe-dir", type=Path, required=True, help="Directory containing *_lobes.nii.gz masks."
    )
    parser.add_argument(
        "--output-csv", type=Path, default=Path("outputs/real/reports/ct_lobe_feature_table.csv")
    )
    args = parser.parse_args()
    frame = build_lobe_feature_table(args.ct_dir, args.lobe_dir, args.output_csv)
    print(
        f"Wrote {len(frame)} scan/lobe rows for {frame['VolumeName'].nunique()} scans to {args.output_csv}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
