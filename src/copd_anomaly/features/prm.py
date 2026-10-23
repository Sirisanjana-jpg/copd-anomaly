"""
Parametric Response Mapping (PRM) and Low Attenuation Area (LAA) analysis.

Classifies co-registered inspiratory and expiratory thoracic CT voxels into:
- PRM_norm: Normal lung parenchyma (Insp >= -950 HU, Exp >= -856 HU)
- PRM_fsad: Functional Small Airway Disease / Gas Trapping (Insp >= -950 HU, Exp < -856 HU)
- PRM_emph: Emphysema (Insp < -950 HU, Exp < -856 HU)
- PRM_pd:   Parenchymal disease / high attenuation (Insp >= -700 HU)
- LAA-950 (Insp) and LAA-856 (Exp)

Provides global and lobe-resolved quantitative summaries.
"""

from __future__ import annotations

import numpy as np

from copd_anomaly.utils.helpers import get_logger

logger = get_logger("prm")

LOBE_NAMES = {
    1: "rul",  # Right Upper Lobe
    2: "rml",  # Right Middle Lobe
    3: "rll",  # Right Lower Lobe
    4: "lul",  # Left Upper Lobe
    5: "lll",  # Left Lower Lobe
}


def compute_prm_map(
    vol_insp: np.ndarray,
    vol_exp_warped: np.ndarray,
    lung_mask: np.ndarray,
    insp_emph_thresh: float = -950.0,
    exp_trap_thresh: float = -856.0,
) -> np.ndarray:
    """
    Computes integer categorical PRM map:
        0 = Background / non-lung
        1 = Normal parenchyma
        2 = Functional Small Airway Disease (fSAD / Gas Trapping)
        3 = Emphysema
        4 = Parenchymal Disease / High Attenuation

    Args:
        vol_insp: (Z, Y, X) float32 in HU.
        vol_exp_warped: (Z, Y, X) float32 in HU registered to insp.
        lung_mask: (Z, Y, X) binary lung parenchyma mask.

    Returns:
        prm_map: (Z, Y, X) uint8.
    """
    prm_map = np.zeros_like(lung_mask, dtype=np.uint8)
    in_lung = lung_mask > 0

    insp_low = vol_insp < insp_emph_thresh
    exp_low = vol_exp_warped < exp_trap_thresh

    # 3 = Emphysema (Insp < -950, Exp < -856)
    emph_voxels = in_lung & insp_low & exp_low
    prm_map[emph_voxels] = 3

    # 2 = fSAD (Insp >= -950, Exp < -856)
    fsad_voxels = in_lung & (~insp_low) & exp_low
    prm_map[fsad_voxels] = 2

    # 1 = Normal (Insp >= -950, Exp >= -856)
    norm_voxels = in_lung & (~insp_low) & (~exp_low)
    prm_map[norm_voxels] = 1

    # 4 = High Attenuation / PD (Insp >= -700 inside lung)
    pd_voxels = in_lung & (vol_insp >= -700.0)
    prm_map[pd_voxels] = 4

    return prm_map


def compute_prm_stats(
    vol_insp: np.ndarray,
    vol_exp_warped: np.ndarray,
    lung_mask: np.ndarray,
    lobe_mask: np.ndarray | None = None,
    spacing_zyx: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> dict[str, float]:
    """
    Computes whole-lung and lobe-wise PRM and LAA percentages.

    Returns:
        Dictionary of quantitative metrics.
    """
    sz, sy, sx = spacing_zyx
    voxel_volume_ml = (sz * sy * sx) / 1000.0  # mm3 to mL

    in_lung = lung_mask > 0
    total_lung_voxels = int(np.sum(in_lung))

    if total_lung_voxels == 0:
        logger.warning("Empty lung mask encountered in compute_prm_stats.")
        return {}

    prm_map = compute_prm_map(vol_insp, vol_exp_warped, lung_mask)

    n_norm = int(np.sum(prm_map == 1))
    n_fsad = int(np.sum(prm_map == 2))
    n_emph = int(np.sum(prm_map == 3))
    n_pd = int(np.sum(prm_map == 4))

    # Classical LAA metrics
    laa950_voxels = int(np.sum(in_lung & (vol_insp < -950.0)))
    laa856_voxels = int(np.sum(in_lung & (vol_exp_warped < -856.0)))

    results: dict[str, float] = {
        "lung_volume_ml": total_lung_voxels * voxel_volume_ml,
        "prm_norm_pct": (n_norm / total_lung_voxels) * 100.0,
        "prm_fsad_pct": (n_fsad / total_lung_voxels) * 100.0,
        "prm_emph_pct": (n_emph / total_lung_voxels) * 100.0,
        "prm_pd_pct": (n_pd / total_lung_voxels) * 100.0,
        "laa950_insp_pct": (laa950_voxels / total_lung_voxels) * 100.0,
        "laa856_exp_pct": (laa856_voxels / total_lung_voxels) * 100.0,
    }

    # Per-lobe breakdown if lobe mask is available
    if lobe_mask is not None and np.any(lobe_mask > 0):
        lobe_emph_pcts = []
        lobe_fsad_pcts = []

        upper_voxels_total = 0
        upper_abnormal_voxels = 0
        lower_voxels_total = 0
        lower_abnormal_voxels = 0

        right_voxels_total = 0
        right_abnormal_voxels = 0
        left_voxels_total = 0
        left_abnormal_voxels = 0

        for lobe_id, lobe_name in LOBE_NAMES.items():
            in_lobe = (lobe_mask == lobe_id) & in_lung
            n_lobe_voxels = int(np.sum(in_lobe))

            if n_lobe_voxels > 0:
                l_norm = int(np.sum(in_lobe & (prm_map == 1)))
                l_fsad = int(np.sum(in_lobe & (prm_map == 2)))
                l_emph = int(np.sum(in_lobe & (prm_map == 3)))
                l_laa950 = int(np.sum(in_lobe & (vol_insp < -950.0)))

                fsad_p = (l_fsad / n_lobe_voxels) * 100.0
                emph_p = (l_emph / n_lobe_voxels) * 100.0
                lobe_fsad_pcts.append(fsad_p)
                lobe_emph_pcts.append(emph_p)

                results[f"{lobe_name}_volume_ml"] = n_lobe_voxels * voxel_volume_ml
                results[f"{lobe_name}_prm_norm_pct"] = (l_norm / n_lobe_voxels) * 100.0
                results[f"{lobe_name}_prm_fsad_pct"] = fsad_p
                results[f"{lobe_name}_prm_emph_pct"] = emph_p
                results[f"{lobe_name}_laa950_pct"] = (l_laa950 / n_lobe_voxels) * 100.0

                # Spatial asymmetry groupings:
                # Upper lobes: RUL (1), LUL (4)
                # Lower lobes: RLL (3), LLL (5)
                # Right lobes: RUL (1), RML (2), RLL (3)
                # Left lobes: LUL (4), LLL (5)
                abnormal_count = l_fsad + l_emph
                if lobe_id in (1, 4):
                    upper_voxels_total += n_lobe_voxels
                    upper_abnormal_voxels += abnormal_count
                elif lobe_id in (3, 5):
                    lower_voxels_total += n_lobe_voxels
                    lower_abnormal_voxels += abnormal_count

                if lobe_id in (1, 2, 3):
                    right_voxels_total += n_lobe_voxels
                    right_abnormal_voxels += abnormal_count
                elif lobe_id in (4, 5):
                    left_voxels_total += n_lobe_voxels
                    left_abnormal_voxels += abnormal_count

        if upper_voxels_total > 0 and lower_voxels_total > 0:
            upper_rate = upper_abnormal_voxels / upper_voxels_total
            lower_rate = lower_abnormal_voxels / lower_voxels_total
            results["upper_lower_ratio"] = float(upper_rate / (lower_rate + 1e-6))

        if right_voxels_total > 0 and left_voxels_total > 0:
            right_rate = right_abnormal_voxels / right_voxels_total
            left_rate = left_abnormal_voxels / left_voxels_total
            results["left_right_asymmetry"] = float(abs(right_rate - left_rate))

        if len(lobe_emph_pcts) == 5:
            results["heterogeneity_index"] = float(np.std(lobe_emph_pcts))

    return results
