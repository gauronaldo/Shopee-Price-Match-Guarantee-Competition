"""Deterministic product-aware sampling shared by unimodal and multimodal training."""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Iterator, Sequence

from torch.utils.data import Sampler


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
