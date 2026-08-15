from __future__ import annotations

from pathlib import Path

from shopee_match.data.config import SplitConfig
from shopee_match.data.models import ImageRecord, Listing
from shopee_match.data.split import build_components, create_split


def split_config() -> SplitConfig:
    return SplitConfig(
        "test.strategy.v1",
        Path("split.jsonl"),
        Path("summary.json"),
        42,
        0.8,
        0.1,
        0.1,
        "label_group",
        True,
        True,
        True,
    )


def test_exact_phash_links_different_labels_into_one_component() -> None:
    listings = [
        Listing(2, "p1", "a.jpg", "0000000000000000", "Product one", "g1"),
        Listing(3, "p2", "b.jpg", "0000000000000000", "Product one", "g2"),
        Listing(4, "p3", "c.jpg", "ffffffffffffffff", "Other", "g3"),
    ]
    images = {
        item.image: ImageRecord(item.image, Path(item.image), item.posting_id * 32, 100, 100, 10)
        for item in listings
    }

    components, mapping = build_components(listings, images, split_config())

    assert len(components) == 2
    assert mapping["g1"] == mapping["g2"]
    assert mapping["g1"] != mapping["g3"]


def test_split_is_deterministic_and_group_disjoint() -> None:
    listings: list[Listing] = []
    images: dict[str, ImageRecord] = {}
    for group_index in range(30):
        for item_index in range(2 + group_index % 4):
            posting_id = f"p{group_index:02d}_{item_index}"
            image = f"{posting_id}.jpg"
            phash = f"{group_index * 100 + item_index:016x}"
            listings.append(Listing(2, posting_id, image, phash, posting_id, f"g{group_index:02d}"))
            images[image] = ImageRecord(image, Path(image), posting_id.ljust(64, "0"), 100, 100, 10)

    first = create_split(listings, images, split_config())
    second = create_split(listings, images, split_config())

    assert first.label_assignments == second.label_assignments
    assert set(first.label_assignments.values()) == {"train", "validation", "test"}
    assert not any(first.summary["integrity"].values())
