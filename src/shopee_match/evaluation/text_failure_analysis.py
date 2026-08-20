"""Local-only review records for scratch text retrieval errors."""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

from shopee_match.evaluation.protocol import EvaluationSplit, Ranking
from shopee_match.features.text import normalize_title

DIGIT_PATTERN = re.compile(r"\d+(?:[.,]\d+)?")
UNIT_PATTERN = re.compile(
    r"\b(?:kg|g|mg|l|ml|cm|mm|m|gb|tb|mah|w|v|pcs|pc|pack|size|sz)\b",
    flags=re.IGNORECASE,
)


def _identity_tokens(title: str) -> dict[str, list[str]]:
    normalized = normalize_title(title)
    return {
        "digits": DIGIT_PATTERN.findall(normalized),
        "units": UNIT_PATTERN.findall(normalized),
    }


def build_text_failure_review(
    ranking: Ranking, split: EvaluationSplit, *, limit_per_bucket: int = 40
) -> dict[str, list[dict[str, Any]]]:
    """Create deterministic title-rich records for local manual categorization."""
    if limit_per_bucket <= 0:
        raise ValueError("limit_per_bucket must be positive")
    labels = split.label_by_id
    items = {item.posting_id: item for item in split.items}
    members: dict[str, list[str]] = defaultdict(list)
    for posting_id, label in labels.items():
        members[label].append(posting_id)
    result: dict[str, list[dict[str, Any]]] = {
        "top1_false_match": [],
        "retrieval_miss": [],
        "top1_success": [],
    }
    for query_id in sorted(ranking):
        candidates = ranking[query_id]
        if not candidates:
            continue
        positives = set(members[labels[query_id]]) - {query_id}
        top = candidates[0]
        positive_candidates = [
            candidate for candidate in candidates if candidate.posting_id in positives
        ]
        top_correct = top.posting_id in positives
        base_record: dict[str, Any] = {
            "query_id": query_id,
            "query_title": items[query_id].title,
            "query_normalized": normalize_title(items[query_id].title),
            "query_identity_tokens": _identity_tokens(items[query_id].title),
            "candidate_id": top.posting_id,
            "candidate_title": items[top.posting_id].title,
            "candidate_normalized": normalize_title(items[top.posting_id].title),
            "candidate_identity_tokens": _identity_tokens(items[top.posting_id].title),
            "candidate_score": top.score,
            "best_positive_rank": (
                next(
                    index
                    for index, candidate in enumerate(candidates, start=1)
                    if candidate.posting_id in positives
                )
                if positive_candidates
                else None
            ),
            "best_positive_score": positive_candidates[0].score if positive_candidates else None,
            "positive_titles": [items[posting_id].title for posting_id in sorted(positives)[:5]],
            "manual_category": None,
            "manual_notes": None,
        }
        bucket = "top1_success" if top_correct else "top1_false_match"
        if len(result[bucket]) < limit_per_bucket:
            result[bucket].append(dict(base_record))
        if not positive_candidates and len(result["retrieval_miss"]) < limit_per_bucket:
            result["retrieval_miss"].append(dict(base_record))
    return result
