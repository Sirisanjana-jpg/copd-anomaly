"""
Volumetric Anomaly Heatmap Reconstruction and Slice Visualization.
"""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np

from copd_anomaly.interfaces import PatchRecord
from copd_anomaly.utils.helpers import get_logger

logger = get_logger("heatmaps")


def reconstruct_anomaly_volume(
    shape_zyx: tuple[int, int, int],
    records: list[PatchRecord],
    patch_scores: np.ndarray,
    patch_size: int = 50,
) -> np.ndarray:
    """
    Projects discrete patch anomaly scores back into a continuous 3D voxel heatmap.

    Returns:
        anomaly_vol: (Z, Y, X) float32 array.
    """
    z_dim, y_dim, x_dim = shape_zyx
    half_p = patch_size // 2

    accum_vol = np.zeros(shape_zyx, dtype=np.float32)
    count_vol = np.zeros(shape_zyx, dtype=np.float32)

    for rec, score in zip(records, patch_scores):
        cz, cy, cx = rec.center_zyx
        z0, z1 = max(0, cz - half_p), min(z_dim, cz + half_p)
        y0, y1 = max(0, cy - half_p), min(y_dim, cy + half_p)
        x0, x1 = max(0, cx - half_p), min(x_dim, cx + half_p)

        accum_vol[z0:z1, y0:y1, x0:x1] += float(score)
        count_vol[z0:z1, y0:y1, x0:x1] += 1.0

    mask = count_vol > 0
    anomaly_vol = np.zeros(shape_zyx, dtype=np.float32)
    anomaly_vol[mask] = accum_vol[mask] / count_vol[mask]

    return anomaly_vol


def save_anomaly_heatmap_nifti(
    anomaly_vol_zyx: np.ndarray,
    spacing_zyx: tuple[float, float, float],
    output_path: Path,
) -> None:
    """Saves reconstructed anomaly volume as NIfTI."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sz, sy, sx = spacing_zyx
    affine = np.diag([sx, sy, sz, 1.0])
    vol_xyz = np.transpose(anomaly_vol_zyx, (2, 1, 0))
    nib_img = nib.Nifti1Image(vol_xyz, affine)
    nib.save(nib_img, str(output_path))
    logger.info(f"Saved anomaly heatmap to {output_path}")
