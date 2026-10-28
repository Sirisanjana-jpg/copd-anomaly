"""Safety tests for anatomical lobe-mask validation."""

from __future__ import annotations

import numpy as np
import pytest

from copd_anomaly.segmentation.segmenter import validate_lobe_mask


def test_valid_five_lobe_mask_preserves_labels() -> None:
    mask = np.zeros((5, 5, 5), dtype=np.uint8)
    for label in range(1, 6):
        mask[label - 1, 0, 0] = label
    validated = validate_lobe_mask(mask, expected_shape=(5, 5, 5))
    assert validated.dtype == np.uint8
    assert set(np.unique(validated)) == {0, 1, 2, 3, 4, 5}


def test_lobe_mask_rejects_binary_mask_as_complete_lobe_mask() -> None:
    with pytest.raises(ValueError, match="Expected all five lobe labels"):
        validate_lobe_mask(np.ones((4, 4, 4), dtype=np.uint8))


def test_lobe_mask_rejects_unknown_labels() -> None:
    mask = np.zeros((3, 3, 3), dtype=np.uint8)
    mask[1, 1, 1] = 6
    with pytest.raises(ValueError, match="Unsupported lobe labels"):
        validate_lobe_mask(mask, require_all=False)


def test_lobe_mask_rejects_shape_mismatch_and_fractional_labels() -> None:
    with pytest.raises(ValueError, match="does not match expected shape"):
        validate_lobe_mask(np.zeros((2, 2, 2)), expected_shape=(3, 3, 3))
    fractional = np.zeros((3, 3, 3), dtype=np.float32)
    fractional[0, 0, 0] = 1.5
    with pytest.raises(ValueError, match="non-integer"):
        validate_lobe_mask(fractional, require_all=False)
