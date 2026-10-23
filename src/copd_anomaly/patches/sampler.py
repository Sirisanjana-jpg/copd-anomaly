"""
3D Patch Sampler for thoracic CT volumes.

Extracts fixed-size 3D volumetric patches (e.g. 50x50x50) from inspiratory (and co-registered expiratory) scans,
restricted to the segmented lung parenchyma (min_lung_fraction >= 0.70).
Converts Hounsfield intensities to compact uint8 representations within [-1024, 0] HU.
"""

from __future__ import annotations

import numpy as np

from copd_anomaly.interfaces import PatchRecord, PatchSampler
from copd_anomaly.utils.helpers import get_logger

logger = get_logger("patch_sampler")


def window_and_to_uint8(
    volume_hu: np.ndarray,
    window_hu: tuple[float, float] = (-1024.0, 0.0),
) -> np.ndarray:
    """
    Windows HU volume to [min_hu, max_hu] and scales linearly to [0, 255] uint8.
    """
    min_hu, max_hu = window_hu
    clipped = np.clip(volume_hu, min_hu, max_hu)
    normalized = (clipped - min_hu) / (max_hu - min_hu + 1e-8) * 255.0
    return np.round(normalized).astype(np.uint8)


def uint8_to_float32(
    volume_uint8: np.ndarray,
) -> np.ndarray:
    """
    Scales uint8 [0, 255] to float32 [0.0, 1.0].
    """
    return volume_uint8.astype(np.float32) / 255.0


class UniformLungPatchSampler(PatchSampler):
    """
    Extracts 3D volumetric patches uniformly or semi-randomly across lung parenchyma.
    """

    def __init__(
        self,
        patch_size: int = 50,
        min_lung_fraction: float = 0.70,
        window_hu: tuple[float, float] = (-1024.0, 0.0),
    ) -> None:
        self.patch_size = patch_size
        self.min_lung_fraction = min_lung_fraction
        self.window_hu = window_hu

    def extract_patches(
        self,
        volume_insp: np.ndarray,
        volume_exp: np.ndarray | None,
        lung_mask: np.ndarray,
        lobe_mask: np.ndarray | None = None,
        patch_size: int | None = None,
        overlap: float = 0.2,
        min_lung_fraction: float | None = None,
        max_patches_per_scan: int = 100,
        patient_id: str = "unknown",
        scan_id: str = "unknown",
        seed: int = 42,
    ) -> tuple[np.ndarray, list[PatchRecord]]:
        """
        Extracts 3D patches from lung parenchyma.

        Returns:
            patches: np.ndarray of shape (N, C, P, P, P) in uint8, where C is 1 or 2.
            records: list of PatchRecord metadata.
        """
        ps = patch_size or self.patch_size
        mlf = min_lung_fraction if min_lung_fraction is not None else self.min_lung_fraction

        z_dim, y_dim, x_dim = volume_insp.shape
        half_p = ps // 2

        # Convert to uint8 representation
        uint8_insp = window_and_to_uint8(volume_insp, self.window_hu)
        uint8_exp = window_and_to_uint8(volume_exp, self.window_hu) if volume_exp is not None else None

        channels = 2 if uint8_exp is not None else 1

        # Grid candidate centers with stride based on overlap
        stride = max(1, int(ps * (1.0 - overlap)))
        z_centers = list(range(half_p, z_dim - half_p, stride))
        y_centers = list(range(half_p, y_dim - half_p, stride))
        x_centers = list(range(half_p, x_dim - half_p, stride))

        candidates = []
        for z in z_centers:
            for y in y_centers:
                for x in x_centers:
                    # Quick center check
                    if lung_mask[z, y, x] == 0:
                        continue
                    candidates.append((z, y, x))

        rng = np.random.RandomState(seed)
        rng.shuffle(candidates)

        extracted_patches: list[np.ndarray] = []
        records: list[PatchRecord] = []

        total_patch_voxels = ps * ps * ps

        for z, y, x in candidates:
            if len(records) >= max_patches_per_scan:
                break

            z_start, z_end = z - half_p, z + half_p
            y_start, y_end = y - half_p, y + half_p
            x_start, x_end = x - half_p, x + half_p

            sub_mask = lung_mask[z_start:z_end, y_start:y_end, x_start:x_end]
            lung_voxels = int(np.sum(sub_mask > 0))
            lung_frac = lung_voxels / float(total_patch_voxels)

            if lung_frac < mlf:
                continue

            # Compute LAA-950 fraction on original HU values
            sub_hu = volume_insp[z_start:z_end, y_start:y_end, x_start:x_end]
            laa950_count = int(np.sum((sub_hu < -950.0) & (sub_mask > 0)))
            laa950_frac = laa950_count / float(max(1, lung_voxels))

            # Determine dominant lobe ID if lobe mask is present
            dominant_lobe: int | None = None
            if lobe_mask is not None:
                sub_lobe = lobe_mask[z_start:z_end, y_start:y_end, x_start:x_end]
                lobe_voxels = sub_lobe[sub_lobe > 0]
                if len(lobe_voxels) > 0:
                    counts = np.bincount(lobe_voxels)
                    dominant_lobe = int(np.argmax(counts))

            # Slice uint8 patch
            patch_insp = uint8_insp[z_start:z_end, y_start:y_end, x_start:x_end]
            if uint8_exp is not None:
                patch_exp = uint8_exp[z_start:z_end, y_start:y_end, x_start:x_end]
                patch_tensor = np.stack([patch_insp, patch_exp], axis=0)  # Shape (2, P, P, P)
            else:
                patch_tensor = patch_insp[np.newaxis, ...]  # Shape (1, P, P, P)

            patch_idx = len(records)
            rec = PatchRecord(
                patch_id=f"{scan_id}_p{patch_idx:04d}",
                patient_id=patient_id,
                scan_id=scan_id,
                center_zyx=(int(z), int(y), int(x)),
                lobe_id=dominant_lobe,
                lung_fraction=float(lung_frac),
                volume_weight=1.0,
                laa950_fraction=float(laa950_frac),
            )

            extracted_patches.append(patch_tensor)
            records.append(rec)

        if not extracted_patches:
            logger.warning(f"No patches extracted for {scan_id} with min_lung_fraction={mlf}!")
            return np.zeros((0, channels, ps, ps, ps), dtype=np.uint8), []

        patches_array = np.stack(extracted_patches, axis=0)
        logger.info(f"Extracted {len(records)} patches for {scan_id} (shape: {patches_array.shape})")
        return patches_array, records
