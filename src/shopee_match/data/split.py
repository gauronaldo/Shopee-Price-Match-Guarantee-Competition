"""Leakage-safe connected-component split generation and integrity checks."""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, cast

from shopee_match.data.config import SplitConfig
from shopee_match.data.models import Component, ImageRecord, Listing, SplitName
from shopee_match.errors import DataValidationError

SPLITS: tuple[SplitName, ...] = ("train", "validation", "test")


class _UnionFind:
    def __init__(self, items: Iterable[str]) -> None:
        self.parent = {item: item for item in items}

    def find(self, item: str) -> str:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            keep, attach = sorted((left_root, right_root))
            self.parent[attach] = keep


@dataclass(frozen=True, slots=True)
class SplitBundle:
    component_assignments: dict[str, SplitName]
    label_assignments: dict[str, SplitName]
    component_ids: dict[str, str]
    components: tuple[Component, ...]
    summary: dict[str, Any]


def _union_bucket(union_find: _UnionFind, labels: set[str]) -> None:
    ordered = sorted(labels)
    for label in ordered[1:]:
        union_find.union(ordered[0], label)


def _size_band(max_label_size: int) -> str:
    if max_label_size <= 2:
        return "2"
    if max_label_size <= 5:
        return "3_to_5"
    if max_label_size <= 9:
        return "6_to_9"
    return "10_plus"


def build_components(
    listings: list[Listing], images: dict[str, ImageRecord], config: SplitConfig
) -> tuple[tuple[Component, ...], dict[str, str]]:
    """Link labels by exact visual identity evidence before any split assignment."""
    labels = {item.label_group for item in listings}
    union_find = _UnionFind(labels)
    by_image: dict[str, set[str]] = defaultdict(set)
    by_phash: dict[str, set[str]] = defaultdict(set)
    by_sha: dict[str, set[str]] = defaultdict(set)
    label_counts = Counter(item.label_group for item in listings)
    for item in listings:
        by_image[item.image].add(item.label_group)
        by_phash[item.image_phash].add(item.label_group)
        image = images.get(item.image)
        if image is not None:
            by_sha[image.sha256].add(item.label_group)
    if config.link_exact_image_reference:
        for bucket in by_image.values():
            _union_bucket(union_find, bucket)
    if config.link_exact_sha256:
        for bucket in by_sha.values():
            _union_bucket(union_find, bucket)
    if config.link_exact_phash:
        for bucket in by_phash.values():
            _union_bucket(union_find, bucket)

    root_labels: dict[str, list[str]] = defaultdict(list)
    for label in sorted(labels):
        root_labels[union_find.find(label)].append(label)
    label_to_component: dict[str, str] = {}
    components: list[Component] = []
    for component_labels in root_labels.values():
        stable_material = "\n".join(sorted(component_labels)).encode()
        component_id = f"component_{hashlib.sha256(stable_material).hexdigest()[:16]}"
        row_count = sum(label_counts[label] for label in component_labels)
        band = _size_band(max(label_counts[label] for label in component_labels))
        components.append(Component(component_id, tuple(component_labels), row_count, band))
        for label in component_labels:
            label_to_component[label] = component_id
    return tuple(sorted(components, key=lambda item: item.component_id)), label_to_component


def assign_components(
    components: tuple[Component, ...], config: SplitConfig
) -> dict[str, SplitName]:
    """Assign whole components with deterministic size-stratified normalized load balancing."""
    fractions = {
        "train": config.train_fraction,
        "validation": config.validation_fraction,
        "test": config.test_fraction,
    }
    total_rows = sum(item.row_count for item in components)
    band_totals = Counter(item.size_band for item in components)
    assigned_rows: Counter[str] = Counter()
    assigned_bands: dict[str, Counter[str]] = defaultdict(Counter)
    assignments: dict[str, SplitName] = {}

    def stable_order(item: Component) -> tuple[int, str]:
        digest = hashlib.sha256(f"{config.seed}:{item.component_id}".encode()).hexdigest()
        return (-item.row_count, digest)

    for component in sorted(components, key=stable_order):
        scored: list[tuple[float, int, SplitName]] = []
        for index, split in enumerate(SPLITS):
            row_target = total_rows * fractions[split]
            band_target = band_totals[component.size_band] * fractions[split]
            row_load = assigned_rows[split] / row_target
            band_load = assigned_bands[component.size_band][split] / max(band_target, 1e-12)
            scored.append((0.7 * row_load + 0.3 * band_load, index, split))
        selected = min(scored)[2]
        assignments[component.component_id] = selected
        assigned_rows[selected] += component.row_count
        assigned_bands[component.size_band][selected] += 1
    if set(assignments.values()) != set(SPLITS):
        raise DataValidationError("Component assignment produced an empty split")
    return assignments


def _cross_split_buckets(
    listings: list[Listing], label_assignments: dict[str, SplitName], key: str
) -> int:
    buckets: dict[str, set[SplitName]] = defaultdict(set)
    for item in listings:
        buckets[cast(str, getattr(item, key))].add(label_assignments[item.label_group])
    return sum(len(splits) > 1 for splits in buckets.values())


def create_split(
    listings: list[Listing], images: dict[str, ImageRecord], config: SplitConfig
) -> SplitBundle:
    """Build, assign, and verify leakage-safe super-components."""
    components, label_to_component = build_components(listings, images, config)
    component_assignments = assign_components(components, config)
    label_assignments = {
        label: component_assignments[component_id]
        for label, component_id in label_to_component.items()
    }
    listing_counts = Counter(label_assignments[item.label_group] for item in listings)
    group_splits: dict[str, set[SplitName]] = defaultdict(set)
    sha_splits: dict[str, set[SplitName]] = defaultdict(set)
    for item in listings:
        split = label_assignments[item.label_group]
        group_splits[item.label_group].add(split)
        image = images.get(item.image)
        if image is not None:
            sha_splits[image.sha256].add(split)
    integrity = {
        "label_groups_cross_split": sum(len(value) > 1 for value in group_splits.values()),
        "image_references_cross_split": _cross_split_buckets(listings, label_assignments, "image"),
        "exact_phashes_cross_split": _cross_split_buckets(
            listings, label_assignments, "image_phash"
        ),
        "sha256_cross_split": sum(len(value) > 1 for value in sha_splits.values()),
    }
    if any(integrity.values()):
        raise DataValidationError(f"Split integrity failure: {integrity}")
    component_sizes = [len(item.label_groups) for item in components]
    summary: dict[str, Any] = {
        "strategy_version": config.strategy_version,
        "seed": config.seed,
        "listings": dict(sorted(listing_counts.items())),
        "label_groups": dict(sorted(Counter(label_assignments.values()).items())),
        "components": dict(sorted(Counter(component_assignments.values()).items())),
        "super_components_total": len(components),
        "multi_label_components": sum(size > 1 for size in component_sizes),
        "max_labels_per_component": max(component_sizes, default=0),
        "max_rows_per_component": max((item.row_count for item in components), default=0),
        "integrity": integrity,
    }
    return SplitBundle(
        component_assignments,
        label_assignments,
        label_to_component,
        components,
        summary,
    )
