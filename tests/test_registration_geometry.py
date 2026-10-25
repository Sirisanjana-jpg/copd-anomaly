"""Registration coordinate tests for nonzero origins and oriented images."""

from __future__ import annotations

import numpy as np
import SimpleITK as sitk

from copd_anomaly.registration.registrar import SimpleITKRegistrar


def test_transform_points_preserves_origin_direction_and_spacing() -> None:
    registrar = SimpleITKRegistrar(fast_mode=True)
    transform = sitk.TranslationTransform(3, (1.0, 0.0, 0.0))
    points = np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32)
    direction = (0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0)

    result = registrar.transform_points(
        points,
        transform,
        spacing_zyx=(3.0, 1.0, 2.0),
        origin_xyz=(10.0, 20.0, 30.0),
        direction_xyz=direction,
    )

    assert np.allclose(result, [[1.0, 1.0, 3.0]], atol=1e-5)


def test_transform_points_rejects_nonorthonormal_direction() -> None:
    registrar = SimpleITKRegistrar(fast_mode=True)
    transform = sitk.Transform(3, sitk.sitkIdentity)
    with np.testing.assert_raises_regex(ValueError, "orthonormal"):
        registrar.transform_points(
            np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32),
            transform,
            direction_xyz=(2.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        )
