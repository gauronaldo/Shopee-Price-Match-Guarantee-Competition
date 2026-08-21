from __future__ import annotations

from pathlib import Path

import torch

from shopee_match.evaluation.pretrained_benchmark import PretrainedImageDataset
from shopee_match.evaluation.protocol import CorpusItem, EvaluationSplit


def test_pretrained_dataset_decodes_rgb_and_applies_frozen_transform() -> None:
    image_dir = Path("tests/fixtures/smoke/train_images")
    item = CorpusItem(
        posting_id="p1",
        image="blue_front.ppm",
        title="blue product",
        image_phash="0000000000000000",
    )
    split = EvaluationSplit((item,), {"p1": "group1"})
    observed: list[tuple[int, ...]] = []

    def transform(image: torch.Tensor) -> torch.Tensor:
        observed.append(tuple(image.shape))
        return torch.nn.functional.interpolate(
            image.unsqueeze(0), size=(240, 240), mode="bilinear", align_corners=False
        ).squeeze(0)

    dataset = PretrainedImageDataset(split, image_dir, transform)
    row = dataset[0]
    assert observed == [(3, 4, 4)]
    assert row["posting_id"] == "p1"
    assert isinstance(row["image"], torch.Tensor)
    assert row["image"].shape == (3, 240, 240)
    assert torch.isfinite(row["image"]).all()
