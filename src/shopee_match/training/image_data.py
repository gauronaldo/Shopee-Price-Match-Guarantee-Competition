"""Leakage-safe image loading, conservative augmentation, and product-aware batches."""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Iterator, Sequence
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, Sampler

from shopee_match.errors import DataValidationError
from shopee_match.evaluation.protocol import CorpusItem, EvaluationSplit


def resize_and_pad_rgb(image: np.ndarray, output_size: int, pad_value: int = 127) -> np.ndarray:
    """Aspect-preserving resize onto a deterministic square canvas."""
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must be an RGB array with shape [height, width, 3]")
    if output_size <= 0:
        raise ValueError("output_size must be positive")
    height, width = image.shape[:2]
    if height <= 0 or width <= 0:
        raise ValueError("image dimensions must be positive")
    scale = output_size / max(height, width)
    resized_width = max(1, min(output_size, round(width * scale)))
    resized_height = max(1, min(output_size, round(height * scale)))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=interpolation)
    canvas = np.full((output_size, output_size, 3), pad_value, dtype=np.uint8)
    top = (output_size - resized_height) // 2
    left = (output_size - resized_width) // 2
    canvas[top : top + resized_height, left : left + resized_width] = resized
    return canvas


class ImagePreprocessor:
    """OpenCV preprocessing with deterministic per-sample training randomness."""

    normalization_policy = "rgb_[0,1]_then_(x-0.5)/0.5; fixed_non_pretrained"

    def __init__(self, image_size: int, *, training: bool, seed: int) -> None:
        if image_size < 16:
            raise ValueError("image_size must be at least 16")
        self.image_size = image_size
        self.training = training
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        self.epoch = epoch

    def _augment(self, image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        height, width = image.shape[:2]
        if float(rng.random()) < 0.5:
            image = np.ascontiguousarray(image[:, ::-1])

        angle = float(rng.uniform(-5.0, 5.0))
        shift_x = float(rng.uniform(-0.02, 0.02) * width)
        shift_y = float(rng.uniform(-0.02, 0.02) * height)
        matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
        matrix[:, 2] += (shift_x, shift_y)
        image = cv2.warpAffine(
            image,
            matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )

        contrast = float(rng.uniform(0.92, 1.08))
        brightness = float(rng.uniform(-0.06, 0.06) * 255)
        image = np.clip(image.astype(np.float32) * contrast + brightness, 0, 255)
        noise_sigma = float(rng.uniform(0.0, 2.5))
        if noise_sigma > 0:
            image += rng.normal(0.0, noise_sigma, size=image.shape).astype(np.float32)
        return np.clip(image, 0, 255).astype(np.uint8)

    def __call__(self, image: np.ndarray, sample_index: int) -> Tensor:
        if self.training:
            sequence = np.random.SeedSequence([self.seed, self.epoch, sample_index])
            image = self._augment(image, np.random.default_rng(sequence))
        square = resize_and_pad_rgb(image, self.image_size)
        array = square.astype(np.float32) / 255.0
        array = (array - 0.5) / 0.5
        return torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1)))


class ProductImageDataset(Dataset[dict[str, Tensor | str]]):
    """Images selected by a frozen split, with train-local numeric labels."""

    def __init__(
        self,
        split: EvaluationSplit,
        image_dir: Path,
        preprocessor: ImagePreprocessor,
        label_to_index: dict[str, int],
    ) -> None:
        self.items = split.items
        self.image_dir = image_dir
        self.preprocessor = preprocessor
        self.labels = tuple(split.label_by_id[item.posting_id] for item in self.items)
        unknown = sorted(set(self.labels) - set(label_to_index))
        if unknown:
            raise DataValidationError(
                "Numeric label mapping does not cover this split; build a split-local mapping"
            )
        self.label_indices = tuple(label_to_index[label] for label in self.labels)

    @classmethod
    def for_split(
        cls,
        split: EvaluationSplit,
        image_dir: Path,
        preprocessor: ImagePreprocessor,
    ) -> ProductImageDataset:
        labels = sorted(set(split.label_by_id.values()))
        return cls(
            split, image_dir, preprocessor, {label: index for index, label in enumerate(labels)}
        )

    def set_epoch(self, epoch: int) -> None:
        self.preprocessor.set_epoch(epoch)

    def __len__(self) -> int:
        return len(self.items)

    def _decode_rgb(self, item: CorpusItem) -> np.ndarray:
        path = self.image_dir / item.image
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise DataValidationError(
                f"Cannot decode image for posting_id={item.posting_id}: {path}"
            )
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def __getitem__(self, index: int) -> dict[str, Tensor | str]:
        item = self.items[index]
        image = self.preprocessor(self._decode_rgb(item), index)
        return {
            "image": image,
            "label": torch.tensor(self.label_indices[index], dtype=torch.long),
            "posting_id": item.posting_id,
        }


class ProductBatchSampler(Sampler[list[int]]):
    """Deterministic P x K sampler with within-product replacement when necessary."""

    def __init__(
        self,
        labels: Sequence[str],
        *,
        products_per_batch: int,
        samples_per_product: int,
        batches_per_epoch: int,
        seed: int,
    ) -> None:
        if products_per_batch < 2:
            raise ValueError("products_per_batch must be at least two")
        if samples_per_product < 2:
            raise ValueError("samples_per_product must be at least two")
        if batches_per_epoch <= 0:
            raise ValueError("batches_per_epoch must be positive")
        grouped: dict[str, list[int]] = defaultdict(list)
        for index, label in enumerate(labels):
            grouped[label].append(index)
        if len(grouped) < products_per_batch:
            raise ValueError("products_per_batch exceeds the number of train product groups")
        self.groups = {label: tuple(indices) for label, indices in sorted(grouped.items())}
        self.products_per_batch = products_per_batch
        self.samples_per_product = samples_per_product
        self.batches_per_epoch = batches_per_epoch
        self.seed = seed
        self.epoch = 0

    @property
    def batch_size(self) -> int:
        return self.products_per_batch * self.samples_per_product

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        self.epoch = epoch

    def __len__(self) -> int:
        return self.batches_per_epoch

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(f"{self.seed}:{self.epoch}")
        labels = list(self.groups)
        for _ in range(self.batches_per_epoch):
            selected_labels = rng.sample(labels, self.products_per_batch)
            batch: list[int] = []
            for label in selected_labels:
                indices = self.groups[label]
                if len(indices) >= self.samples_per_product:
                    batch.extend(rng.sample(list(indices), self.samples_per_product))
                else:
                    batch.extend(rng.choices(indices, k=self.samples_per_product))
            yield batch
