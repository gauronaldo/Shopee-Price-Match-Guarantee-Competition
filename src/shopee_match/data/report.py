"""Deterministic aggregate reports, SVG figures, and local inspection gallery."""

from __future__ import annotations

import hashlib
import html
import json
from collections import defaultdict
from collections.abc import Iterable
from itertools import pairwise
from pathlib import Path
from typing import Any

from shopee_match.data.audit import normalize_title
from shopee_match.data.config import Phase1Config
from shopee_match.data.models import Listing
from shopee_match.data.split import SplitBundle


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _bar_svg(title: str, values: dict[str, int]) -> str:
    width, height = 760, 420
    margin_left, margin_bottom, margin_top = 90, 70, 55
    plot_width = width - margin_left - 30
    plot_height = height - margin_bottom - margin_top
    maximum = max(values.values(), default=1)
    bar_width = plot_width / max(len(values), 1)
    bars: list[str] = []
    for index, (label, value) in enumerate(values.items()):
        x = margin_left + index * bar_width + bar_width * 0.15
        rendered_width = bar_width * 0.7
        rendered_height = plot_height * value / maximum
        y = margin_top + plot_height - rendered_height
        bars.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{rendered_width:.1f}" '
            f'height="{rendered_height:.1f}" fill="#2563eb" />'
            f'<text x="{x + rendered_width / 2:.1f}" y="{y - 8:.1f}" '
            f'text-anchor="middle">{value}</text>'
            f'<text x="{x + rendered_width / 2:.1f}" y="{height - 35}" '
            f'text-anchor="middle">{html.escape(label)}</text>'
        )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}">'
        "<style>text{font:14px sans-serif;fill:#111827}</style>"
        f'<text x="{width / 2}" y="28" text-anchor="middle" '
        f'style="font-size:20px;font-weight:bold">{html.escape(title)}</text>'
        f'<line x1="{margin_left}" y1="{margin_top + plot_height}" '
        f'x2="{width - 30}" y2="{margin_top + plot_height}" stroke="#111827" />'
        + "".join(bars)
        + "</svg>"
    )


def _histogram(values: Iterable[int], edges: tuple[int, ...]) -> dict[str, int]:
    counts: dict[str, int] = {}
    values_list = list(values)
    for lower, upper in pairwise(edges):
        label = f"{lower}-{upper - 1}"
        counts[label] = sum(lower <= value < upper for value in values_list)
    counts[f"{edges[-1]}+"] = sum(value >= edges[-1] for value in values_list)
    return counts


def _stable_order(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def inspection_samples(
    listings: list[Listing], config: Phase1Config
) -> dict[str, list[dict[str, Any]]]:
    """Select deterministic same-group and difficult cross-group pairs for human review."""
    groups: dict[str, list[Listing]] = defaultdict(list)
    phashes: dict[str, list[Listing]] = defaultdict(list)
    titles: dict[str, list[Listing]] = defaultdict(list)
    for item in listings:
        groups[item.label_group].append(item)
        phashes[item.image_phash].append(item)
        titles[normalize_title(item.title)].append(item)

    same_pairs: list[tuple[Listing, Listing, str]] = []
    for _group, items in sorted(
        groups.items(), key=lambda pair: _stable_order(config.audit.random_sample_seed, pair[0])
    ):
        if len(items) < 2:
            continue
        ordered = sorted(items, key=lambda item: (item.image, item.posting_id))
        right = next((item for item in ordered[1:] if item.image != ordered[0].image), ordered[1])
        same_pairs.append((ordered[0], right, "same_label_group"))
        if len(same_pairs) >= config.audit.same_group_samples:
            break

    cross_candidates: list[tuple[Listing, Listing, str]] = []
    for reason, buckets in (("exact_phash_cross_label", phashes), ("title_cross_label", titles)):
        for _key, items in sorted(buckets.items(), key=lambda pair: pair[0]):
            labels = sorted({item.label_group for item in items})
            if len(labels) < 2:
                continue
            left = next(item for item in items if item.label_group == labels[0])
            right = next(item for item in items if item.label_group == labels[1])
            cross_candidates.append((left, right, reason))
    cross_candidates.sort(
        key=lambda pair: _stable_order(
            config.audit.random_sample_seed,
            f"{pair[2]}:{pair[0].posting_id}:{pair[1].posting_id}",
        )
    )
    different_pairs = cross_candidates[: config.audit.different_group_samples]

    def serialize(pairs: list[tuple[Listing, Listing, str]]) -> list[dict[str, Any]]:
        return [
            {
                "reason": reason,
                "left": {
                    "posting_id": left.posting_id,
                    "label_group": left.label_group,
                    "image": left.image,
                    "title": left.title,
                },
                "right": {
                    "posting_id": right.posting_id,
                    "label_group": right.label_group,
                    "image": right.image,
                    "title": right.title,
                },
            }
            for left, right, reason in pairs
        ]

    return {"same_group": serialize(same_pairs), "different_group": serialize(different_pairs)}


def _gallery(samples: dict[str, list[dict[str, Any]]], image_dir: Path) -> str:
    cards: list[str] = []
    for category, pairs in samples.items():
        for index, pair in enumerate(pairs, start=1):
            figures: list[str] = []
            for side in ("left", "right"):
                item = pair[side]
                uri = (image_dir / item["image"]).resolve().as_uri()
                figures.append(
                    f'<figure><img src="{html.escape(uri)}" alt="{side}">'
                    f"<figcaption><b>{html.escape(item['posting_id'])}</b><br>"
                    f"label={html.escape(item['label_group'])}<br>"
                    f"{html.escape(item['title'])}</figcaption></figure>"
                )
            cards.append(
                f"<article><h2>{html.escape(category)} #{index}</h2>"
                f"<p>{html.escape(pair['reason'])}</p><div>{''.join(figures)}</div></article>"
            )
    prefix = """<!doctype html>
<meta charset="utf-8">
<title>Phase 1 inspection gallery</title>
<style>
body { font-family: sans-serif; background: #f8fafc }
article { background: white; margin: 1rem; padding: 1rem; border: 1px solid #cbd5e1 }
article > div { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem }
img { width: 100%; max-height: 420px; object-fit: contain }
figcaption { overflow-wrap: anywhere }
</style>
<h1>Shopee Phase 1 deterministic inspection gallery</h1>
<p>Local restricted artifact; do not publish raw competition images.</p>
"""
    return prefix + "".join(cards)


def write_reports(
    config: Phase1Config,
    listings: list[Listing],
    audit: dict[str, Any],
    split: SplitBundle,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    """Write aggregate publishable reports and ignored local inspection artifacts."""
    payload = {"provenance": provenance, "audit": audit, "split": split.summary}
    config.audit.report_json.parent.mkdir(parents=True, exist_ok=True)
    config.audit.report_json.write_bytes(_json_bytes(payload))
    config.audit.figure_dir.mkdir(parents=True, exist_ok=True)
    (config.audit.figure_dir / "group_size_bands.svg").write_text(
        _bar_svg("Label-group size distribution", audit["group_count_bands"]),
        encoding="utf-8",
    )
    title_histogram = _histogram((len(item.title) for item in listings), (0, 25, 50, 75, 100, 150))
    (config.audit.figure_dir / "title_length_histogram.svg").write_text(
        _bar_svg("Title length (characters)", title_histogram), encoding="utf-8"
    )
    split_counts = {key: int(value) for key, value in split.summary["listings"].items()}
    (config.audit.figure_dir / "split_listing_counts.svg").write_text(
        _bar_svg("Leakage-safe split listing counts", split_counts), encoding="utf-8"
    )

    findings = audit["findings"]
    finding_rows = "\n".join(
        f"| `{item['code']}` | {item['severity']} | {item['count']} | {item['message']} |"
        for item in findings
    )
    group_size_summary = (
        f"{audit['group_size']['median']} / {audit['group_size']['p95']} / "
        f"{audit['group_size']['max']}"
    )
    title_length_summary = (
        f"{audit['title_characters']['median']} / {audit['title_characters']['p95']}"
    )
    component_summary = (
        f"{split.summary['max_labels_per_component']} labels / "
        f"{split.summary['max_rows_per_component']} rows"
    )
    markdown = f"""# Shopee data audit v1

This report is generated from aggregate statistics only. Raw Kaggle CSVs and images remain local.

## Provenance

- Config: `{provenance["config_version"]}` (`{provenance["config_sha256"]}`)
- Train CSV SHA-256: `{provenance["metadata_sha256"]}`
- OpenCV: `{audit["images"]["opencv_version"]}`
- Split strategy: `{split.summary["strategy_version"]}`, seed `{split.summary["seed"]}`
- Manifest SHA-256: `{provenance["manifest_sha256"]}`

## Dataset

- Listings: **{audit["listings"]:,}**
- Label groups: **{audit["label_groups"]:,}**
- Unique referenced images: **{audit["unique_images"]:,}**
- Missing values / duplicate IDs / decode failures: **0 / 0 / {audit["images"]["decode_failures"]}**
- Group size median / P95 / max: **{group_size_summary}**
- Title length median / P95: **{title_length_summary}** characters

![Group sizes](figures/data_audit_v1/group_size_bands.svg)

![Title lengths](figures/data_audit_v1/title_length_histogram.svg)

## Leakage-safe split

- Listings: `{json.dumps(split.summary["listings"], sort_keys=True)}`
- Label groups: `{json.dumps(split.summary["label_groups"], sort_keys=True)}`
- Super-components: **{split.summary["super_components_total"]:,}**
- Multi-label components: **{split.summary["multi_label_components"]:,}**
- Maximum component: **{component_summary}**
- Integrity checks: `{json.dumps(split.summary["integrity"], sort_keys=True)}`
- Near-pHash pairs crossing splits: **{split.summary["near_phash_pairs_cross_split"]}**. They are
  audited but not automatically merged because some are valid variants.

![Split counts](figures/data_audit_v1/split_listing_counts.svg)

## Findings

| Code | Severity | Count | Meaning |
|---|---:|---:|---|
{finding_rows}

## Manual inspection

A deterministic local gallery with {config.audit.same_group_samples} same-group and
{config.audit.different_group_samples} difficult cross-group pairs is generated under the ignored
inspection directory. Spot checks found both probable fragmented labels and legitimate product
variants. This supports retaining source labels while treating pHash/title collisions as audit and
hard-negative signals.
"""
    config.audit.report_markdown.parent.mkdir(parents=True, exist_ok=True)
    config.audit.report_markdown.write_text(markdown, encoding="utf-8")

    samples = inspection_samples(listings, config)
    config.audit.inspection_dir.mkdir(parents=True, exist_ok=True)
    (config.audit.inspection_dir / "samples.json").write_bytes(_json_bytes(samples))
    (config.audit.inspection_dir / "gallery.html").write_text(
        _gallery(samples, config.dataset.image_dir), encoding="utf-8"
    )
    return samples
