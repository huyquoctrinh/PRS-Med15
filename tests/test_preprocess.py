import numpy as np

from prs_med15.data.preprocess import (PreprocessConfig, normalize_ct, normalize_mr,
                                      preprocess_native_2d, preprocess_volume,
                                      split_slabs)


def test_normalization_preprocessing_and_slabs():
    assert np.allclose(normalize_ct(np.array([-1000, 0, 1000])), [0, .5, 1])
    assert np.isfinite(normalize_mr(np.ones((2, 2)))).all()
    image = np.linspace(-1000, 1000, 5 * 4 * 6).reshape(5, 4, 6)
    label = np.zeros_like(image, dtype=np.int16)
    label[3:, 1:3, 2:4] = 1
    out = preprocess_volume(image, (2, 1, 1), "CT", PreprocessConfig(image_size=12, d_max=2), label)
    assert out.image.shape == (3, 5, 12, 12)
    assert out.label.dtype == np.int16
    slabs = split_slabs(out, {"target": out.label == 1}, 2)
    assert [(x.start, x.stop) for x in slabs] == [(0, 2), (2, 4), (4, 5)]
    assert [x.visibility["target"] for x in slabs] == [False, True, True]


def test_native_rgb_letterbox():
    rgb = np.ones((4, 8, 3), np.float32)
    image, valid, transform = preprocess_native_2d(rgb, 16)
    assert image.shape == (3, 1, 16, 16) and valid.shape == (1, 16, 16)
    assert transform.pad == (4, 4, 0, 0)
    assert not valid[:, :4].any() and valid[:, 4:12].all()
