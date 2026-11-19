import numpy as np

from prs_med15.data.geometry import (VolumeTransformRecord, apply_transform, apply_volume_transform,
                                     invert_transform, invert_volume_transform,
                                     make_letterbox_transform, physical_grid)


def test_letterbox_and_exact_nearest_roundtrip():
    mask = np.zeros((2, 4, 6), np.uint8)
    mask[:, 1:3, 2:5] = 1
    transform = make_letterbox_transform((4, 6), (12, 12), scale=2)
    encoded = apply_transform(mask, transform, order=0)
    assert encoded.shape == (2, 12, 12)
    assert np.array_equal(invert_transform(encoded, transform, order=0), mask)
    assert transform.pad == (2, 2, 0, 0)


def test_physical_grid_unknown_and_known():
    centres, extents, unknown = physical_grid(None, (2, 2, 2), (4, 4, 4))
    assert unknown and not centres.any() and not extents.any()
    centres, extents, unknown = physical_grid(np.diag([2, 3, 4, 1]), (2, 2, 2), (4, 4, 4))
    assert not unknown and tuple(extents[0, 0, 0]) == (4, 6, 8)


def test_volume_transform_roundtrip_nearest():
    raw = np.zeros((3, 4, 6), np.uint8)
    raw[1:, 1:3, 2:5] = 1
    letterbox = make_letterbox_transform((8, 12), (16, 16), scale=1)
    transform = VolumeTransformRecord((3, 4, 6), (6, 8, 12), (6, 16, 16), (2, 2, 2),
                                      ((0, 3), (0, 4), (0, 6)), letterbox)
    processed = apply_volume_transform(raw, transform)
    assert processed.shape == (6, 16, 16)
    assert np.array_equal(invert_volume_transform(processed, transform), raw)
