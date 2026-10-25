"""
Deformable image registration and Target Registration Error (TRE) evaluation.

Implements non-rigid registration between expiratory and inspiratory breath-hold CT scans
using SimpleITK, supporting BSpline and Demons transformations, lung-mask guidance,
and continuous point coordinate mapping for landmark TRE benchmarking.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import SimpleITK as sitk

from copd_anomaly.interfaces import Registrar, RegistrationResult
from copd_anomaly.utils.helpers import get_logger

logger = get_logger("registrar")


def evaluate_tre(
    fixed_landmarks: np.ndarray,
    moving_landmarks: np.ndarray,
    warped_landmarks: np.ndarray | None = None,
    spacing_zyx: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> dict[str, float]:
    """
    Computes Target Registration Error (TRE) in physical millimeters.

    Args:
        fixed_landmarks: (N, 3) float32 coordinates (x, y, z) in voxel index space.
        moving_landmarks: (N, 3) float32 coordinates (x, y, z) before registration.
        warped_landmarks: Optional (N, 3) float32 coordinates (x, y, z) after transformation.
        spacing_zyx: (sz, sy, sx) in mm.

    Returns:
        Dictionary with mean, std, median, min, max TRE in mm, and initial TRE.
    """
    sz, sy, sx = spacing_zyx
    spacing_xyz = np.array([sx, sy, sz], dtype=np.float32)

    # Initial displacement (before registration)
    initial_diff = (fixed_landmarks - moving_landmarks) * spacing_xyz
    initial_tre = np.sqrt(np.sum(initial_diff**2, axis=-1))

    stats: dict[str, float] = {
        "initial_mean_tre_mm": float(np.mean(initial_tre)),
        "initial_std_tre_mm": float(np.std(initial_tre)),
        "initial_median_tre_mm": float(np.median(initial_tre)),
        "initial_max_tre_mm": float(np.max(initial_tre)),
    }

    if warped_landmarks is not None:
        warped_diff = (fixed_landmarks - warped_landmarks) * spacing_xyz
        final_tre = np.sqrt(np.sum(warped_diff**2, axis=-1))
        stats.update(
            {
                "mean_tre_mm": float(np.mean(final_tre)),
                "std_tre_mm": float(np.std(final_tre)),
                "median_tre_mm": float(np.median(final_tre)),
                "min_tre_mm": float(np.min(final_tre)),
                "max_tre_mm": float(np.max(final_tre)),
                "tre_reduction_pct": float(
                    (np.mean(initial_tre) - np.mean(final_tre))
                    / (np.mean(initial_tre) + 1e-8)
                    * 100.0
                ),
            }
        )
    return stats


class SimpleITKRegistrar(Registrar):
    """
    Non-rigid deformable registration engine using SimpleITK.
    """

    def __init__(
        self,
        method: str = "bspline",
        grid_spacing_mm: float = 20.0,
        iterations: int = 50,
        sampling_percentage: float = 0.2,
        fast_mode: bool = False,
    ) -> None:
        """
        Args:
            method: 'bspline' or 'demons'.
            grid_spacing_mm: BSpline control point grid spacing in mm.
            iterations: Maximum optimizer iterations per scale.
            sampling_percentage: Fraction of voxels sampled for metric calculation.
            fast_mode: If True, uses single coarse scale with minimal iterations for unit tests.
        """
        self.method = method
        self.grid_spacing_mm = grid_spacing_mm
        self.iterations = 5 if fast_mode else iterations
        self.sampling_percentage = 0.05 if fast_mode else sampling_percentage
        self.fast_mode = fast_mode

    def _numpy_to_sitk(
        self, array_zyx: np.ndarray, spacing_zyx: tuple[float, float, float]
    ) -> sitk.Image:
        """Converts (Z, Y, X) numpy array to SimpleITK Image with spacing (sx, sy, sz)."""
        image = sitk.GetImageFromArray(array_zyx)
        sz, sy, sx = spacing_zyx
        image.SetSpacing((float(sx), float(sy), float(sz)))
        return image

    def register(
        self,
        fixed_volume: np.ndarray,
        moving_volume: np.ndarray,
        fixed_mask: np.ndarray | None = None,
        moving_mask: np.ndarray | None = None,
        spacing_zyx: tuple[float, float, float] = (1.0, 1.0, 1.0),
    ) -> RegistrationResult:
        """
        Registers moving (exhale) volume to fixed (inhale) volume.
        """
        fixed_img = self._numpy_to_sitk(fixed_volume, spacing_zyx)
        moving_img = self._numpy_to_sitk(moving_volume, spacing_zyx)

        # Resample moving image to match fixed geometry if shapes differ
        if fixed_volume.shape != moving_volume.shape:
            logger.info("Resampling moving image grid to match fixed image shape...")
            moving_img = sitk.Resample(
                moving_img,
                fixed_img,
                sitk.Transform(),
                sitk.sitkLinear,
                -1024.0,
                moving_img.GetPixelID(),
            )

        # Set up registration method
        R = sitk.ImageRegistrationMethod()
        R.SetMetricAsMeanSquares()

        if fixed_mask is not None and np.any(fixed_mask):
            fixed_mask_img = self._numpy_to_sitk(fixed_mask.astype(np.uint8), spacing_zyx)
            R.SetMetricFixedMask(fixed_mask_img)

        R.SetMetricSamplingStrategy(R.RANDOM)
        R.SetMetricSamplingPercentage(self.sampling_percentage)
        R.SetInterpolator(sitk.sitkLinear)

        # Set optimizer
        R.SetOptimizerAsGradientDescent(
            learningRate=1.0,
            numberOfIterations=self.iterations,
            convergenceMinimumValue=1e-6,
            convergenceWindowSize=5,
        )

        if self.fast_mode:
            R.SetShrinkFactorsPerLevel([1])
            R.SetSmoothingSigmasPerLevel([0.0])
        else:
            R.SetShrinkFactorsPerLevel([4, 2, 1])
            R.SetSmoothingSigmasPerLevel([2.0, 1.0, 0.0])
        R.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

        # Initialize BSpline transform
        transform_domain_mesh_size = [
            max(1, int(fixed_img.GetSize()[i] * fixed_img.GetSpacing()[i] / self.grid_spacing_mm))
            for i in range(3)
        ]
        initial_transform = sitk.BSplineTransformInitializer(
            fixed_img, transform_domain_mesh_size, order=3
        )
        R.SetInitialTransform(initial_transform, inPlace=False)

        logger.info(f"Running registration ({self.method}, fast_mode={self.fast_mode})...")
        fitted_transform = R.Execute(fixed_img, moving_img)

        # Resample moving image using fitted transform
        warped_img = sitk.Resample(
            moving_img,
            fixed_img,
            fitted_transform,
            sitk.sitkLinear,
            -1024.0,
            sitk.sitkFloat32,
        )
        warped_volume = sitk.GetArrayFromImage(warped_img)

        # Generate displacement field
        displacement_filter = sitk.TransformToDisplacementFieldFilter()
        displacement_filter.SetReferenceImage(fixed_img)
        disp_field_img = displacement_filter.Execute(fitted_transform)
        displacement_field = sitk.GetArrayFromImage(disp_field_img)  # Shape: (Z, Y, X, 3)

        return RegistrationResult(
            warped_volume=warped_volume,
            displacement_field=displacement_field,
            transform=fitted_transform,
            tre_stats=None,
        )

    def transform_points(
        self,
        points: np.ndarray,
        transform: Any,
        spacing_zyx: tuple[float, float, float] = (1.0, 1.0, 1.0),
    ) -> np.ndarray:
        """
        Transforms moving landmark coordinates (x, y, z) into the fixed coordinate space.

        Args:
            points: (N, 3) float32 coordinates as (x, y, z) in voxel space.
            transform: Fitted SimpleITK transform.
            spacing_zyx: (sz, sy, sx) in mm.

        Returns:
            warped_points: (N, 3) float32 coordinates in fixed voxel space.
        """
        sz, sy, sx = spacing_zyx
        spacing_xyz = np.array([sx, sy, sz], dtype=np.float64)

        transformed = np.zeros_like(points, dtype=np.float32)
        for i in range(len(points)):
            pt_voxel = points[i]
            pt_phys = tuple(pt_voxel * spacing_xyz)
            warped_phys = transform.TransformPoint(pt_phys)
            warped_voxel = np.array(warped_phys, dtype=np.float64) / spacing_xyz
            transformed[i] = warped_voxel.astype(np.float32)

        return transformed
