"""Deterministic validation-only diagnostics for multimodal retrieval."""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

from shopee_match.evaluation.protocol import EvaluationSplit, Ranking

DIGIT_PATTERN = re.compile(r"\d+(?:[.,]\d+)?")
UNIT_PATTERN = re.compile(
    r"\b(?:kg|g|mg|l|ml|cm|mm|m|gb|tb|mah|w|v|pcs|pc|pack|size|sz)\b",
    flags=re.IGNORECASE,
)


def _identity_tokens(title: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    lowered = title.casefold()
    return tuple(DIGIT_PATTERN.findall(lowered)), tuple(UNIT_PATTERN.findall(lowered))


def _top1_correct(ranking: Ranking, query_id: str, positives: set[str]) -> bool:
    return bool(ranking[query_id]) and ranking[query_id][0].posting_id in positives


def _contains_positive(ranking: Ranking, query_id: str, positives: set[str]) -> bool:
    return any(candidate.posting_id in positives for candidate in ranking[query_id])


def build_multimodal_failure_review(
    rankings: dict[str, Ranking],
    split: EvaluationSplit,
    *,
    limit_per_bucket: int = 40,
) -> dict[str, Any]:
    """Categorize modality disagreements and fusion errors without touching test data."""
    required = {"image", "text", "simple_fusion", "learned_fusion", "pair_head"}
    if set(rankings) != required:
        raise ValueError(f"rankings must contain exactly {sorted(required)}")
    if limit_per_bucket <= 0:
        raise ValueError("limit_per_bucket must be positive")
    query_ids = set(split.label_by_id)
    if any(set(ranking) != query_ids for ranking in rankings.values()):
        raise ValueError("every ranking must align with the evaluation split")

    items = {item.posting_id: item for item in split.items}
    members: dict[str, set[str]] = defaultdict(set)
    for posting_id, label in split.label_by_id.items():
        members[label].add(posting_id)

    bucket_names = (
        "pair_top1_false_match",
        "pair_retrieval_miss",
        "pair_head_regression",
        "pair_head_rescue",
        "image_rescue",
        "text_rescue",
        "modality_disagreement",
        "variant_token_conflict",
    )
    counts = {name: 0 for name in bucket_names}
    samples: dict[str, list[dict[str, Any]]] = {name: [] for name in bucket_names}

    for query_id in sorted(query_ids):
        positives = members[split.label_by_id[query_id]] - {query_id}
        correct = {
            name: _top1_correct(ranking, query_id, positives) for name, ranking in rankings.items()
        }
        pair_top = rankings["pair_head"][query_id][0]
        query_digits, query_units = _identity_tokens(items[query_id].title)
        candidate_digits, candidate_units = _identity_tokens(items[pair_top.posting_id].title)
        variant_conflict = bool(
            (query_digits or candidate_digits or query_units or candidate_units)
            and (query_digits != candidate_digits or query_units != candidate_units)
        )
        active = {
            "pair_top1_false_match": not correct["pair_head"],
            "pair_retrieval_miss": not _contains_positive(
                rankings["pair_head"], query_id, positives
            ),
            "pair_head_regression": correct["simple_fusion"] and not correct["pair_head"],
            "pair_head_rescue": not correct["simple_fusion"] and correct["pair_head"],
            "image_rescue": correct["image"] and not correct["text"],
            "text_rescue": correct["text"] and not correct["image"],
            "modality_disagreement": (
                rankings["image"][query_id][0].posting_id
                != rankings["text"][query_id][0].posting_id
            ),
            "variant_token_conflict": not correct["pair_head"] and variant_conflict,
        }
        record = {
            "query_id": query_id,
            "query_title": items[query_id].title,
            "positive_titles": [items[value].title for value in sorted(positives)[:5]],
            "top_candidates": {
                name: {
                    "posting_id": ranking[query_id][0].posting_id,
                    "title": items[ranking[query_id][0].posting_id].title,
                    "score": ranking[query_id][0].score,
                    "is_match": correct[name],
                }
                for name, ranking in rankings.items()
            },
            "query_digits": query_digits,
            "query_units": query_units,
            "pair_candidate_digits": candidate_digits,
            "pair_candidate_units": candidate_units,
            "manual_category": None,
            "manual_notes": None,
        }
        for bucket, is_active in active.items():
            if not is_active:
                continue
            counts[bucket] += 1
            if len(samples[bucket]) < limit_per_bucket:
                samples[bucket].append(record)
    return {
        "split": "validation",
        "queries": len(query_ids),
        "counts": counts,
        "samples": samples,
        "categories_are_overlapping": True,
        "test_accessed": False,
    }
