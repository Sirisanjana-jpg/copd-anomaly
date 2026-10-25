"""Segmentation safety tests on synthetic geometric fixtures."""

from __future__ import annotations

import numpy as np

from copd_anomaly.segmentation.segmenter import LungSegmenter


def test_threshold_segmentation_discards_lateral_exterior_air() -> None:
    volume = np.zeros((40, 40, 40), dtype=np.float32)
    volume[:, :2, :] = -1000.0
    volume[:, -2:, :] = -1000.0
    volume[:, :, :2] = -1000.0
    volume[:, :, -2:] = -1000.0
    z, y, x = np.ogrid[:40, :40, :40]
    right_lung = ((z - 20) / 10) ** 2 + ((y - 17) / 6) ** 2 + ((x - 13) / 5) ** 2 <= 1
    left_lung = ((z - 20) / 10) ** 2 + ((y - 17) / 6) ** 2 + ((x - 27) / 5) ** 2 <= 1
    volume[right_lung | left_lung] = -800.0

    mask = LungSegmenter(backend="threshold").segment(volume, (1.0, 1.0, 1.0))

    assert mask.shape == volume.shape
    assert mask.any()
    assert not mask[:, 0, :].any()
    assert not mask[:, -1, :].any()
    assert not mask[:, :, 0].any()
    assert not mask[:, :, -1].any()
