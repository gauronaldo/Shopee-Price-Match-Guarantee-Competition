from __future__ import annotations

import numpy as np
import torch

from shopee_match.training.image_data import ImagePreprocessor, resize_and_pad_rgb


def test_resize_and_pad_preserves_aspect_and_centers_content() -> None:
    image = np.full((10, 20, 3), 255, dtype=np.uint8)
    output = resize_and_pad_rgb(image, 20, pad_value=0)

    assert output.shape == (20, 20, 3)
    assert np.all(output[:5] == 0)
    assert np.all(output[5:15] == 255)
    assert np.all(output[15:] == 0)


def test_validation_preprocessing_is_deterministic() -> None:
    image = np.arange(12 * 18 * 3, dtype=np.uint8).reshape(12, 18, 3)
    transform = ImagePreprocessor(32, training=False, seed=5)

    first = transform(image, sample_index=1)
    transform.set_epoch(9)
    second = transform(image, sample_index=1)

    assert first.shape == (3, 32, 32)
    assert first.dtype == torch.float32
    assert torch.equal(first, second)
    assert float(first.min()) >= -1.0
    assert float(first.max()) <= 1.0


def test_training_preprocessing_repeats_within_epoch() -> None:
    image = np.full((20, 30, 3), 128, dtype=np.uint8)
    transform = ImagePreprocessor(32, training=True, seed=8)

    assert torch.equal(transform(image, 2), transform(image, 2))
    transform.set_epoch(1)
    assert not torch.equal(
        transform(image, 2), ImagePreprocessor(32, training=True, seed=8)(image, 2)
    )
