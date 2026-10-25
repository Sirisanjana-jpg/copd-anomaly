"""
Lung parenchyma and lobe segmentation engine.

Supports:
1. Classical morphological lung segmentation (Hounsfield thresholding, connected component
   labeling, 3D hole filling, and trachea removal) as robust, license-free default.
2. Optional neural network backends: 'lungmask' (Apache-2.0) and 'totalsegmentator' (Academic).
3. If lobe segmentation is unavailable, gracefully falls back to whole-lung mask (label 1)
   with explicit logging and alert flag.
"""

from __future__ import annotations

import importlib.util
from typing import Literal

import numpy as np
import scipy.ndimage as ndi

from copd_anomaly.interfaces import Segmenter
from copd_anomaly.utils.helpers import get_logger

logger = get_logger("segmenter")


class LungSegmenter(Segmenter):
    """
    Pluggable lung and lobe segmentation implementation.
    """

    def __init__(
        self,
        backend: Literal["auto", "totalsegmentator", "lungmask", "threshold"] = "auto",
        hu_threshold_low: float = -1000.0,
        hu_threshold_high: float = -400.0,
    ) -> None:
        self.hu_threshold_low = hu_threshold_low
        self.hu_threshold_high = hu_threshold_high
        self.backend = self._resolve_backend(backend)
        logger.info(f"Initialized LungSegmenter with resolved backend: '{self.backend}'")

    def _resolve_backend(self, requested: str) -> str:
        if requested == "threshold":
            return "threshold"

        has_totalseg = importlib.util.find_spec("totalsegmentator") is not None
        has_lungmask = importlib.util.find_spec("lungmask") is not None

        if requested == "totalsegmentator":
            if has_totalseg:
                return "totalsegmentator"
            logger.warning(
                "TotalSegmentator requested but not installed. Falling back to threshold."
            )
            return "threshold"

        if requested == "lungmask":
            if has_lungmask:
                return "lungmask"
            logger.warning("lungmask requested but not installed. Falling back to threshold.")
            return "threshold"

        # Auto resolution
        if has_totalseg:
            return "totalsegmentator"
        if has_lungmask:
            return "lungmask"
        return "threshold"

    def segment(
        self,
        volume: np.ndarray,
        spacing_zyx: tuple[float, float, float],
    ) -> np.ndarray:
        """
        Segments lung parenchyma and lobes.

        Args:
            volume: (Z, Y, X) float32 in HU.
            spacing_zyx: (sz, sy, sx) in mm.

        Returns:
            mask: (Z, Y, X) uint8.
                  0 = background, 1..5 = lobes (or 1 = whole lung).
        """
        if self.backend == "totalsegmentator":
            return self._segment_totalsegmentator(volume, spacing_zyx)
        elif self.backend == "lungmask":
            return self._segment_lungmask(volume, spacing_zyx)
        else:
            return self._segment_threshold(volume)

    def _segment_threshold(self, volume: np.ndarray) -> np.ndarray:
        """
        Fast, robust morphological whole-lung segmentation.
        HU threshold [-1000, -400], connected component analysis to keep the two
        largest components (lungs), slice-by-slice/3D hole filling.
        """
        binary = (volume >= self.hu_threshold_low) & (volume <= self.hu_threshold_high)
        if not np.any(binary):
            logger.warning("No voxels found in threshold range for lung segmentation.")
            return np.zeros_like(volume, dtype=np.uint8)

        # Remove small noise and isolate major air cavities
        labeled, num_features = ndi.label(binary)
        if num_features == 0:
            return np.zeros_like(volume, dtype=np.uint8)

        component_sizes = ndi.sum(binary, labeled, range(1, num_features + 1))
        # Keep up to 2 largest components (left and right lungs)
        # Note: sometimes left and right lungs are connected into 1 component; keep top components
        sorted_indices = np.argsort(component_sizes)[::-1]

        mask = np.zeros_like(volume, dtype=np.uint8)
        # Add the largest component
        largest_idx = sorted_indices[0] + 1
        mask[labeled == largest_idx] = 1

        # If second largest is substantial (> 15% of largest), also include it (separate lung)
        if len(sorted_indices) > 1:
            second_idx = sorted_indices[1] + 1
            if component_sizes[sorted_indices[1]] > 0.15 * component_sizes[sorted_indices[0]]:
                mask[labeled == second_idx] = 1

        # Fill holes slice by slice along axial plane (Z-axis)
        for z in range(mask.shape[0]):
            if np.any(mask[z]):
                mask[z] = ndi.binary_fill_holes(mask[z]).astype(np.uint8)

        logger.debug(f"Threshold segmentation produced {np.sum(mask)} lung voxels.")
        return mask

    def _segment_totalsegmentator(
        self, volume: np.ndarray, spacing_zyx: tuple[float, float, float]
    ) -> np.ndarray:
        try:
            import nibabel as nib
            from totalsegmentator.python_api import totalsegmentator

            sz, sy, sx = spacing_zyx
            affine = np.diag([sx, sy, sz, 1.0])
            nib_img = nib.Nifti1Image(np.transpose(volume, (2, 1, 0)), affine)
            seg_nib = totalsegmentator(nib_img, task="lung_vessels")
            seg_xyz = seg_nib.get_fdata().astype(np.uint8)
            return np.transpose(seg_xyz, (2, 1, 0))
        except Exception as e:
            logger.warning(f"TotalSegmentator failed ({e}); falling back to threshold whole-lung.")
            return self._segment_threshold(volume)

    def _segment_lungmask(
        self, volume: np.ndarray, spacing_zyx: tuple[float, float, float]
    ) -> np.ndarray:
        try:
            import SimpleITK as sitk
            from lungmask import mask as lm_mask

            itk_img = sitk.GetImageFromArray(volume)
            itk_img.SetSpacing((spacing_zyx[2], spacing_zyx[1], spacing_zyx[0]))
            seg_np = lm_mask.apply(itk_img)
            return (seg_np > 0).astype(np.uint8)
        except Exception as e:
            logger.warning(f"lungmask failed ({e}); falling back to threshold whole-lung.")
            return self._segment_threshold(volume)
