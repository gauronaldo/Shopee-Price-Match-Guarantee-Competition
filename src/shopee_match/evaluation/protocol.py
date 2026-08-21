"""Leakage-safe data loading and retrieval metrics."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from shopee_match.errors import DataValidationError


@dataclass(frozen=True)
class CorpusItem:
    """Retriever-visible fields; ground-truth labels are deliberately excluded."""

    posting_id: str
    image: str
    image_phash: str
    title: str


@dataclass(frozen=True)
class EvaluationSplit:
    """Candidate corpus and separately held evaluation truth."""

    items: tuple[CorpusItem, ...]
    label_by_id: dict[str, str]


@dataclass(frozen=True)
class ScoredCandidate:
    posting_id: str
    score: float


Ranking = dict[str, list[ScoredCandidate]]


def load_splits(metadata_csv: Path, manifest_path: Path) -> dict[str, EvaluationSplit]:
    """Load and cross-check metadata against the frozen row-level manifest."""
    with metadata_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_id = {row["posting_id"]: row for row in rows}
    if len(by_id) != len(rows):
        raise DataValidationError("Metadata contains duplicate posting_id values")

    manifest: dict[str, str] = {}
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                record = json.loads(line)
                posting_id = str(record["posting_id"])
                split = str(record["split"])
            except (json.JSONDecodeError, KeyError, TypeError) as error:
                raise DataValidationError(
                    f"Invalid split manifest record at line {line_number}"
                ) from error
            if posting_id in manifest:
                raise DataValidationError(f"Duplicate manifest posting_id: {posting_id}")
            if split not in {"train", "validation", "test"}:
                raise DataValidationError(f"Unknown split {split!r}")
            manifest[posting_id] = split
    if set(manifest) != set(by_id):
        raise DataValidationError("Metadata and manifest posting_id sets differ")

    items_by_split: dict[str, list[CorpusItem]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    labels_by_split: dict[str, dict[str, str]] = {
        "train": {},
        "validation": {},
        "test": {},
    }
    label_split: dict[str, str] = {}
    for posting_id in sorted(by_id):
        row = by_id[posting_id]
        split = manifest[posting_id]
        label = row["label_group"]
        previous = label_split.setdefault(label, split)
        if previous != split:
            raise DataValidationError(f"label_group {label} crosses splits")
        items_by_split[split].append(
            CorpusItem(
                posting_id=posting_id,
                image=row["image"],
                image_phash=row["image_phash"],
                title=row["title"],
            )
        )
        labels_by_split[split][posting_id] = label
    return {
        split: EvaluationSplit(tuple(items), labels_by_split[split])
        for split, items in items_by_split.items()
    }


def load_named_split(metadata_csv: Path, manifest_path: Path, split_name: str) -> EvaluationSplit:
    """Load only one named split so validation workflows never retain test labels."""
    if split_name not in {"train", "validation", "test"}:
        raise ValueError("split_name must be train, validation, or test")
    selected_ids: set[str] = set()
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                record = json.loads(line)
                posting_id = str(record["posting_id"])
                split = str(record["split"])
            except (json.JSONDecodeError, KeyError, TypeError) as error:
                raise DataValidationError(
                    f"Invalid split manifest record at line {line_number}"
                ) from error
            if split == split_name:
                if posting_id in selected_ids:
                    raise DataValidationError(f"Duplicate manifest posting_id: {posting_id}")
                selected_ids.add(posting_id)
    if not selected_ids:
        raise DataValidationError(f"Split {split_name!r} is empty")

    items: list[CorpusItem] = []
    labels: dict[str, str] = {}
    with metadata_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            posting_id = row["posting_id"]
            if posting_id not in selected_ids:
                continue
            if posting_id in labels:
                raise DataValidationError(f"Metadata contains duplicate posting_id: {posting_id}")
            labels[posting_id] = row["label_group"]
            items.append(
                CorpusItem(
                    posting_id=posting_id,
                    image=row["image"],
                    image_phash=row["image_phash"],
                    title=row["title"],
                )
            )
    if set(labels) != selected_ids:
        raise DataValidationError(f"Metadata is missing rows from split {split_name!r}")
    items.sort(key=lambda item: item.posting_id)
    return EvaluationSplit(tuple(items), labels)


def _relevant_by_query(labels: dict[str, str]) -> dict[str, set[str]]:
    members: dict[str, set[str]] = {}
    for posting_id, label in labels.items():
        members.setdefault(label, set()).add(posting_id)
    return {posting_id: members[label] - {posting_id} for posting_id, label in labels.items()}


def _validate_ranking(ranking: Ranking, label_by_id: dict[str, str]) -> None:
    if set(ranking) != set(label_by_id):
        raise ValueError("Ranking must contain every and only evaluated query")
    known_ids = set(label_by_id)
    for query_id, candidates in ranking.items():
        candidate_ids = [candidate.posting_id for candidate in candidates]
        if query_id in candidate_ids or len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError(f"Invalid candidates for query {query_id}")
        if not set(candidate_ids) <= known_ids:
            raise ValueError(f"Unknown candidate for query {query_id}")
        if any(not math.isfinite(candidate.score) for candidate in candidates):
            raise ValueError(f"Non-finite score for query {query_id}")
        expected = sorted(candidates, key=lambda item: (-item.score, item.posting_id))
        if candidates != expected:
            raise ValueError(f"Candidates are not deterministically ranked for query {query_id}")


def retrieval_metrics(
    ranking: Ranking,
    label_by_id: dict[str, str],
    recall_at: tuple[int, ...],
    average_precision_at: int,
) -> dict[str, float]:
    """Compute macro query retrieval metrics over every query in the split."""
    relevant = _relevant_by_query(label_by_id)
    _validate_ranking(ranking, label_by_id)
    accumulators = {f"hit_rate@{k}": 0.0 for k in recall_at}
    accumulators.update({f"recall@{k}": 0.0 for k in recall_at})
    accumulators.update({f"precision@{k}": 0.0 for k in recall_at})
    accumulators.update({f"f1@{k}": 0.0 for k in recall_at})
    mean_ap = 0.0
    for query_id in sorted(ranking):
        positives = relevant[query_id]
        if not positives:
            raise ValueError(f"Query {query_id} has no positive in its evaluation split")
        candidate_ids = [candidate.posting_id for candidate in ranking[query_id]]
        for k in recall_at:
            found = len(positives.intersection(candidate_ids[:k]))
            denominator = min(k, len(label_by_id) - 1)
            precision = found / denominator if denominator else 0.0
            recall = found / len(positives)
            accumulators[f"hit_rate@{k}"] += float(found > 0)
            accumulators[f"recall@{k}"] += recall
            accumulators[f"precision@{k}"] += precision
            accumulators[f"f1@{k}"] += (
                2 * precision * recall / (precision + recall) if precision + recall else 0.0
            )
        hits = 0
        precision_sum = 0.0
        for rank, candidate_id in enumerate(candidate_ids[:average_precision_at], start=1):
            if candidate_id in positives:
                hits += 1
                precision_sum += hits / rank
        mean_ap += precision_sum / min(len(positives), average_precision_at)
    query_count = len(ranking)
    return {
        **{name: value / query_count for name, value in accumulators.items()},
        f"map@{average_precision_at}": mean_ap / query_count,
        "queries": float(query_count),
    }


def pair_metrics_at_threshold(
    ranking: Ranking, label_by_id: dict[str, str], threshold: float
) -> dict[str, float]:
    """Score directed retrieved pairs; unretrieved positives count as false negatives."""
    _validate_ranking(ranking, label_by_id)
    relevant = _relevant_by_query(label_by_id)
    true_positive = false_positive = 0
    for query_id, candidates in ranking.items():
        for candidate in candidates:
            if candidate.score < threshold:
                continue
            if candidate.posting_id in relevant[query_id]:
                true_positive += 1
            else:
                false_positive += 1
    total_positive = sum(len(value) for value in relevant.values())
    false_negative = total_positive - true_positive
    precision = true_positive / (true_positive + false_positive) if true_positive else 0.0
    recall = true_positive / total_positive if total_positive else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positive": float(true_positive),
        "false_positive": float(false_positive),
        "false_negative": float(false_negative),
    }


def select_threshold(ranking: Ranking, label_by_id: dict[str, str]) -> dict[str, float]:
    """Select an exact validation threshold in O(retrieved-pairs log retrieved-pairs)."""
    _validate_ranking(ranking, label_by_id)
    relevant = _relevant_by_query(label_by_id)
    scored_labels = sorted(
        (
            (candidate.score, candidate.posting_id in relevant[query_id])
            for query_id, candidates in ranking.items()
            for candidate in candidates
        ),
        key=lambda item: -item[0],
    )
    if not scored_labels:
        raise ValueError("Cannot select a threshold from an empty ranking")
    total_positive = sum(len(value) for value in relevant.values())
    true_positive = false_positive = 0
    best: dict[str, float] | None = None
    index = 0
    while index < len(scored_labels):
        threshold = scored_labels[index][0]
        while index < len(scored_labels) and scored_labels[index][0] == threshold:
            if scored_labels[index][1]:
                true_positive += 1
            else:
                false_positive += 1
            index += 1
        precision = true_positive / (true_positive + false_positive) if true_positive else 0.0
        recall = true_positive / total_positive if total_positive else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        result = {
            "threshold": threshold,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "true_positive": float(true_positive),
            "false_positive": float(false_positive),
            "false_negative": float(total_positive - true_positive),
        }
        if best is None or (f1, precision, threshold) > (
            best["f1"],
            best["precision"],
            best["threshold"],
        ):
            best = result
    if best is None:  # pragma: no cover - guarded by the non-empty check
        raise AssertionError("Threshold selection did not evaluate a score")
    return best


def precision_at_minimum_recall(
    ranking: Ranking,
    label_by_id: dict[str, str],
    minimum_recall: float,
) -> dict[str, float]:
    """Find the most precise retrieved-pair threshold that preserves a recall target."""
    if not 0 < minimum_recall <= 1:
        raise ValueError("minimum_recall must be inside (0, 1]")
    _validate_ranking(ranking, label_by_id)
    relevant = _relevant_by_query(label_by_id)
    scored_labels = sorted(
        (
            (candidate.score, candidate.posting_id in relevant[query_id])
            for query_id, candidates in ranking.items()
            for candidate in candidates
        ),
        key=lambda item: -item[0],
    )
    total_positive = sum(len(value) for value in relevant.values())
    true_positive = false_positive = 0
    best: dict[str, float] | None = None
    index = 0
    while index < len(scored_labels):
        threshold = scored_labels[index][0]
        while index < len(scored_labels) and scored_labels[index][0] == threshold:
            if scored_labels[index][1]:
                true_positive += 1
            else:
                false_positive += 1
            index += 1
        recall = true_positive / total_positive if total_positive else 0.0
        if recall < minimum_recall:
            continue
        precision = true_positive / (true_positive + false_positive) if true_positive else 0.0
        result = {
            "threshold": threshold,
            "precision": precision,
            "recall": recall,
            "true_positive": float(true_positive),
            "false_positive": float(false_positive),
            "false_negative": float(total_positive - true_positive),
            "minimum_recall": minimum_recall,
        }
        if best is None or (precision, threshold) > (best["precision"], best["threshold"]):
            best = result
    if best is None:
        maximum_recall = true_positive / total_positive if total_positive else 0.0
        raise ValueError(
            f"Retrieved candidates cannot reach minimum recall {minimum_recall}; "
            f"maximum is {maximum_recall}"
        )
    return best


def ranking_to_json(ranking: Ranking) -> dict[str, list[dict[str, Any]]]:
    return {
        query: [
            {"posting_id": candidate.posting_id, "score": candidate.score}
            for candidate in candidates
        ]
        for query, candidates in ranking.items()
    }
