"""Metadata, label, pHash, title, and image-quality audit calculations."""

from __future__ import annotations

import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from shopee_match.data.models import Finding, ImageRecord, Listing


@dataclass(frozen=True, slots=True)
class AuditBundle:
    summary: dict[str, Any]
    findings: tuple[Finding, ...]
    near_phash_pairs: tuple[tuple[str, str, int], ...]


def normalize_title(title: str) -> str:
    """Normalize only for audit comparisons; never replace preserved raw titles."""
    return " ".join(unicodedata.normalize("NFKC", title).casefold().split())


def _quantiles(values: Iterable[float], digits: int | None = None) -> dict[str, float | int]:
    ordered = sorted(values)
    if not ordered:
        return {}

    def at(fraction: float) -> float | int:
        value = ordered[round((len(ordered) - 1) * fraction)]
        return round(value, digits) if digits is not None else value

    return {
        "min": at(0),
        "p25": at(0.25),
        "median": at(0.5),
        "p75": at(0.75),
        "p95": at(0.95),
        "p99": at(0.99),
        "max": at(1),
    }


class _BKNode:
    __slots__ = ("children", "value")

    def __init__(self, value: int) -> None:
        self.value = value
        self.children: dict[int, _BKNode] = {}


def find_near_phashes(
    phash_labels: dict[str, set[str]], max_distance: int
) -> tuple[tuple[str, str, int], ...]:
    """Find unique cross-label pHash pairs using a deterministic BK-tree."""
    integer_to_hex = {int(value, 16): value for value in phash_labels}
    root: _BKNode | None = None
    pairs: list[tuple[str, str, int]] = []
    for value in sorted(integer_to_hex):
        if root is None:
            root = _BKNode(value)
            continue
        stack = [root]
        while stack:
            node = stack.pop()
            distance = (value ^ node.value).bit_count()
            if distance <= max_distance:
                left_hex, right_hex = integer_to_hex[node.value], integer_to_hex[value]
                if len(phash_labels[left_hex] | phash_labels[right_hex]) > 1:
                    pairs.append((left_hex, right_hex, distance))
            low, high = distance - max_distance, distance + max_distance
            stack.extend(child for edge, child in node.children.items() if low <= edge <= high)
        node = root
        while True:
            distance = (value ^ node.value).bit_count()
            child = node.children.get(distance)
            if child is None:
                node.children[distance] = _BKNode(value)
                break
            node = child
    return tuple(pairs)


def audit_dataset(
    listings: list[Listing],
    images: dict[str, ImageRecord],
    image_stats: dict[str, Any],
    near_phash_distance: int,
    initial_findings: list[Finding],
) -> AuditBundle:
    """Compute aggregate audit statistics and actionable suspicious-data findings."""
    groups: dict[str, list[Listing]] = defaultdict(list)
    image_rows: dict[str, list[Listing]] = defaultdict(list)
    phash_rows: dict[str, list[Listing]] = defaultdict(list)
    title_rows: dict[str, list[Listing]] = defaultdict(list)
    sha_labels: dict[str, set[str]] = defaultdict(set)
    for item in listings:
        groups[item.label_group].append(item)
        image_rows[item.image].append(item)
        phash_rows[item.image_phash].append(item)
        title_rows[normalize_title(item.title)].append(item)
        image = images.get(item.image)
        if image is not None:
            sha_labels[image.sha256].add(item.label_group)

    group_sizes = [len(items) for items in groups.values()]
    image_cross = [
        items for items in image_rows.values() if len({item.label_group for item in items}) > 1
    ]
    phash_cross = [
        items for items in phash_rows.values() if len({item.label_group for item in items}) > 1
    ]
    title_cross = [
        items for items in title_rows.values() if len({item.label_group for item in items}) > 1
    ]
    phash_labels = {
        value: {item.label_group for item in items} for value, items in phash_rows.items()
    }
    near_pairs = find_near_phashes(phash_labels, near_phash_distance)
    findings = list(initial_findings)
    warning_specs = (
        ("exact_image_cross_label", "Exact image reference spans labels", len(image_cross)),
        (
            "exact_sha_cross_label",
            "Exact image bytes span labels",
            sum(len(labels) > 1 for labels in sha_labels.values()),
        ),
        ("exact_phash_cross_label", "Exact pHash spans labels", len(phash_cross)),
        (
            "near_phash_cross_label",
            f"Cross-label pHash pairs have Hamming distance <= {near_phash_distance}",
            len(near_pairs),
        ),
        ("normalized_title_cross_label", "Normalized title spans labels", len(title_cross)),
    )
    for code, message, count in warning_specs:
        if count:
            findings.append(Finding("warning", code, message, count))

    widths = [item.width for item in images.values()]
    heights = [item.height for item in images.values()]
    aspects = [item.width / item.height for item in images.values()]
    file_bytes = [item.file_bytes for item in images.values()]
    script_presence: Counter[str] = Counter()
    for item in listings:
        scripts = set()
        for character in item.title:
            point = ord(character)
            if character.isdigit():
                scripts.add("digits")
            elif 0x4E00 <= point <= 0x9FFF:
                scripts.add("cjk")
            elif 0x3040 <= point <= 0x30FF:
                scripts.add("japanese")
            elif 0xAC00 <= point <= 0xD7AF:
                scripts.add("hangul")
            elif 0x0E00 <= point <= 0x0E7F:
                scripts.add("thai")
            elif 0x0400 <= point <= 0x04FF:
                scripts.add("cyrillic")
            elif 0x0600 <= point <= 0x06FF:
                scripts.add("arabic")
            elif "LATIN" in unicodedata.name(character, ""):
                scripts.add("latin")
        script_presence.update(scripts or {"other"})

    group_bands = {
        "2": sum(size == 2 for size in group_sizes),
        "3_to_5": sum(3 <= size <= 5 for size in group_sizes),
        "6_to_9": sum(6 <= size <= 9 for size in group_sizes),
        "10_plus": sum(size >= 10 for size in group_sizes),
    }
    summary: dict[str, Any] = {
        "status": (
            "blocked"
            if any(item.severity == "critical" for item in findings)
            else "passed_with_warnings"
            if findings
            else "passed"
        ),
        "listings": len(listings),
        "label_groups": len(groups),
        "group_size": _quantiles(group_sizes),
        "group_count_bands": group_bands,
        "singleton_groups": sum(size == 1 for size in group_sizes),
        "title_characters": _quantiles(len(item.title) for item in listings),
        "title_tokens": _quantiles(len(item.title.split()) for item in listings),
        "title_script_presence": dict(sorted(script_presence.items())),
        "titles_with_digits": sum(any(char.isdigit() for char in item.title) for item in listings),
        "non_ascii_titles": sum(not item.title.isascii() for item in listings),
        "unique_images": len(image_rows),
        "repeated_image_reference_buckets": sum(len(items) > 1 for items in image_rows.values()),
        "exact_image_cross_label_buckets": len(image_cross),
        "unique_phashes": len(phash_rows),
        "repeated_phash_buckets": sum(len(items) > 1 for items in phash_rows.values()),
        "exact_phash_cross_label_buckets": len(phash_cross),
        "near_phash_cross_label_pairs": len(near_pairs),
        "near_phash_distance_distribution": dict(
            sorted(Counter(distance for _, _, distance in near_pairs).items())
        ),
        "unique_normalized_titles": len(title_rows),
        "normalized_title_cross_label_buckets": len(title_cross),
        "images": {
            **image_stats,
            "width": _quantiles(widths),
            "height": _quantiles(heights),
            "aspect_ratio": _quantiles(aspects, 3),
            "file_bytes": _quantiles(file_bytes),
            "small_min_side_lt_64": sum(
                min(item.width, item.height) < 64 for item in images.values()
            ),
            "extreme_aspect_ratio": sum(
                item.width / item.height < 0.25 or item.width / item.height > 4
                for item in images.values()
            ),
            "unique_sha256": len({item.sha256 for item in images.values()}),
            "sha_cross_label_buckets": sum(len(labels) > 1 for labels in sha_labels.values()),
        },
        "findings": [
            {
                "severity": item.severity,
                "code": item.code,
                "message": item.message,
                "count": item.count,
            }
            for item in findings
        ],
    }
    return AuditBundle(summary, tuple(findings), near_pairs)
