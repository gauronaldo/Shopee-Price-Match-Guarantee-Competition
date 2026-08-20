from __future__ import annotations

from collections import Counter

from shopee_match.training.sampling import ProductBatchSampler


def test_product_batch_sampler_is_reproducible_and_has_positive_pairs() -> None:
    labels = ["a", "a", "b", "b", "b", "c", "c", "d", "d"]
    first = ProductBatchSampler(
        labels,
        products_per_batch=3,
        samples_per_product=2,
        batches_per_epoch=4,
        seed=2026,
    )
    second = ProductBatchSampler(
        labels,
        products_per_batch=3,
        samples_per_product=2,
        batches_per_epoch=4,
        seed=2026,
    )

    batches = list(first)
    assert batches == list(second)
    assert len(batches) == 4
    for batch in batches:
        counts = Counter(labels[index] for index in batch)
        assert len(batch) == 6
        assert sorted(counts.values()) == [2, 2, 2]

    first.set_epoch(1)
    assert list(first) != batches


def test_product_batch_sampler_replaces_within_small_group() -> None:
    labels = ["a", "b", "b"]
    sampler = ProductBatchSampler(
        labels,
        products_per_batch=2,
        samples_per_product=2,
        batches_per_epoch=1,
        seed=3,
    )
    batch = next(iter(sampler))

    assert Counter(labels[index] for index in batch) == {"a": 2, "b": 2}
