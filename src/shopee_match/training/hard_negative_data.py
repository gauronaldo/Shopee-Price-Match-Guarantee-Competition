"""Deterministic hard-negative filtering, manifests, and batch sampling."""

from __future__ import annotations

import json
import math
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from shopee_match.errors import DataValidationError
from shopee_match.evaluation.protocol import CorpusItem
from shopee_match.features.text import normalize_title

IDENTITY_TOKEN = re.compile(
    r"\d+(?:[.,]\d+)?|(?:ml|cl|dl|l|mg|g|kg|mm|cm|m|gb|tb|pcs?|pack|set|x)",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class MiningCandidate:
    """A cross-listing candidate produced by exact train retrieval."""

    query_index: int
    candidate_index: int
    cosine_similarity: float
    pair_probability: float


@dataclass(frozen=True, slots=True)
class MinedHardNegative:
    """Canonical, symmetric, cross-label pair persisted in the Phase 6 manifest."""

    left_posting_id: str
    right_posting_id: str
    left_label_group: str
    right_label_group: str
    cosine_similarity: float
    pair_probability: float
    variant_conflict: bool
    left_identity_tokens: tuple[str, ...]
    right_identity_tokens: tuple[str, ...]


@dataclass(slots=True)
class MiningSelectionStats:
    candidates_seen: int = 0
    excluded_same_label: int = 0
    excluded_probability: int = 0
    excluded_same_phash: int = 0
    excluded_exact_title: int = 0
    eligible: int = 0
    selected_before_symmetric_dedup: int = 0
    symmetric_duplicates_removed: int = 0
    variant_quota_removed: int = 0

    def add(self, other: MiningSelectionStats) -> None:
        """Accumulate counters from a query or retrieval block."""
        for field in (
            "candidates_seen",
            "excluded_same_label",
            "excluded_probability",
            "excluded_same_phash",
            "excluded_exact_title",
            "eligible",
            "selected_before_symmetric_dedup",
            "symmetric_duplicates_removed",
            "variant_quota_removed",
        ):
            setattr(self, field, getattr(self, field) + getattr(other, field))


def identity_tokens(title: str) -> tuple[str, ...]:
    """Return sorted digits and quantity/unit markers retained by normalization."""
    return tuple(sorted(set(IDENTITY_TOKEN.findall(normalize_title(title)))))


def has_variant_conflict(left_title: str, right_title: str) -> bool:
    """Flag different non-empty identity-token sets as a likely variant distinction."""
    left = identity_tokens(left_title)
    right = identity_tokens(right_title)
    return bool(left or right) and left != right


def _to_mined_pair(
    candidate: MiningCandidate,
    items: tuple[CorpusItem, ...],
    label_by_id: dict[str, str],
) -> MinedHardNegative:
    query = items[candidate.query_index]
    neighbour = items[candidate.candidate_index]
    left, right = sorted((query, neighbour), key=lambda item: item.posting_id)
    left_tokens = identity_tokens(left.title)
    right_tokens = identity_tokens(right.title)
    return MinedHardNegative(
        left_posting_id=left.posting_id,
        right_posting_id=right.posting_id,
        left_label_group=label_by_id[left.posting_id],
        right_label_group=label_by_id[right.posting_id],
        cosine_similarity=candidate.cosine_similarity,
        pair_probability=candidate.pair_probability,
        variant_conflict=bool(left_tokens or right_tokens) and left_tokens != right_tokens,
        left_identity_tokens=left_tokens,
        right_identity_tokens=right_tokens,
    )


def select_hard_negatives(
    candidates_by_query: list[list[MiningCandidate]],
    items: tuple[CorpusItem, ...],
    label_by_id: dict[str, str],
    *,
    negatives_per_query: int,
    minimum_pair_probability: float,
    maximum_pair_probability: float,
    exclude_same_phash: bool,
    exclude_exact_normalized_title: bool,
    variant_priority_fraction: float,
) -> tuple[list[MinedHardNegative], MiningSelectionStats]:
    """Filter label-noise risks, prefer variant conflicts, and deduplicate symmetric pairs."""
    if len(candidates_by_query) != len(items):
        raise ValueError("candidates_by_query must align with items")
    if negatives_per_query <= 0 or not 0 <= variant_priority_fraction <= 1:
        raise ValueError("invalid hard-negative selection bounds")
    stats = MiningSelectionStats()
    selected: list[MinedHardNegative] = []
    for query_index, candidates in enumerate(candidates_by_query):
        query_selected, query_stats = select_query_hard_negatives(
            query_index,
            candidates,
            items,
            label_by_id,
            negatives_per_query=negatives_per_query,
            minimum_pair_probability=minimum_pair_probability,
            maximum_pair_probability=maximum_pair_probability,
            exclude_same_phash=exclude_same_phash,
            exclude_exact_normalized_title=exclude_exact_normalized_title,
            variant_priority_fraction=variant_priority_fraction,
        )
        selected.extend(query_selected)
        stats.add(query_stats)

    result, duplicates = deduplicate_hard_negatives(selected)
    stats.symmetric_duplicates_removed = duplicates
    result, quota_removed = cap_variant_share(result, variant_priority_fraction)
    stats.variant_quota_removed = quota_removed
    return result, stats


def deduplicate_hard_negatives(
    selected: list[MinedHardNegative],
) -> tuple[list[MinedHardNegative], int]:
    """Collapse symmetric duplicates, retaining the strongest deterministic record."""
    by_pair: dict[tuple[str, str], MinedHardNegative] = {}
    for pair in selected:
        key = (pair.left_posting_id, pair.right_posting_id)
        previous = by_pair.get(key)
        if previous is None or (
            pair.pair_probability,
            pair.cosine_similarity,
        ) > (
            previous.pair_probability,
            previous.cosine_similarity,
        ):
            by_pair[key] = pair
    result = [by_pair[key] for key in sorted(by_pair)]
    return result, len(selected) - len(result)


def cap_variant_share(
    pairs: list[MinedHardNegative], maximum_fraction: float
) -> tuple[list[MinedHardNegative], int]:
    """Enforce a global variant share after dedup while retaining the hardest pairs."""
    if not 0 <= maximum_fraction <= 1:
        raise ValueError("maximum_fraction must be inside [0, 1]")
    if maximum_fraction == 1:
        return pairs, 0
    ordinary = [pair for pair in pairs if not pair.variant_conflict]
    variants = [pair for pair in pairs if pair.variant_conflict]
    maximum_variants = math.floor(
        maximum_fraction * len(ordinary) / max(1e-12, 1 - maximum_fraction)
    )
    variants.sort(
        key=lambda pair: (
            -pair.pair_probability,
            -pair.cosine_similarity,
            pair.left_posting_id,
            pair.right_posting_id,
        )
    )
    retained = ordinary + variants[:maximum_variants]
    retained.sort(key=lambda pair: (pair.left_posting_id, pair.right_posting_id))
    return retained, len(variants) - min(len(variants), maximum_variants)


def select_query_hard_negatives(
    query_index: int,
    candidates: list[MiningCandidate],
    items: tuple[CorpusItem, ...],
    label_by_id: dict[str, str],
    *,
    negatives_per_query: int,
    minimum_pair_probability: float,
    maximum_pair_probability: float,
    exclude_same_phash: bool,
    exclude_exact_normalized_title: bool,
    variant_priority_fraction: float,
) -> tuple[list[MinedHardNegative], MiningSelectionStats]:
    """Select a bounded deterministic set for one train query."""
    query = items[query_index]
    stats = MiningSelectionStats(candidates_seen=len(candidates))
    eligible: list[MiningCandidate] = []
    for candidate in candidates:
        neighbour = items[candidate.candidate_index]
        if label_by_id[query.posting_id] == label_by_id[neighbour.posting_id]:
            stats.excluded_same_label += 1
            continue
        if not minimum_pair_probability <= candidate.pair_probability <= maximum_pair_probability:
            stats.excluded_probability += 1
            continue
        if exclude_same_phash and query.image_phash == neighbour.image_phash:
            stats.excluded_same_phash += 1
            continue
        if (
            exclude_exact_normalized_title
            and normalize_title(query.title) == normalize_title(neighbour.title)
        ):
            stats.excluded_exact_title += 1
            continue
        eligible.append(candidate)
    stats.eligible = len(eligible)
    eligible.sort(
        key=lambda row: (
            -row.pair_probability,
            -row.cosine_similarity,
            items[row.candidate_index].posting_id,
        )
    )
    variant_keys = {
        (row.query_index, row.candidate_index)
        for row in eligible
        if has_variant_conflict(query.title, items[row.candidate_index].title)
    }
    variants = [
        row for row in eligible if (row.query_index, row.candidate_index) in variant_keys
    ]
    ordinary = [
        row for row in eligible if (row.query_index, row.candidate_index) not in variant_keys
    ]
    variant_limit = round(negatives_per_query * variant_priority_fraction)
    chosen = variants[:variant_limit]
    chosen.extend(ordinary[: negatives_per_query - len(chosen)])
    result = [_to_mined_pair(row, items, label_by_id) for row in chosen]
    stats.selected_before_symmetric_dedup = len(result)
    return result, stats


def hard_negative_jsonl(pairs: list[MinedHardNegative]) -> str:
    """Serialize a stable JSONL manifest with no user-specific paths."""
    return "".join(
        json.dumps(asdict(pair), ensure_ascii=False, sort_keys=True) + "\n" for pair in pairs
    )


def load_hard_negative_manifest(path: Path) -> list[MinedHardNegative]:
    """Load and strictly validate the deterministic Phase 6 JSONL manifest."""
    pairs: list[MinedHardNegative] = []
    seen: set[tuple[str, str]] = set()
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise DataValidationError(f"Cannot read hard-negative manifest: {path}") from exc
    for line_number, line in enumerate(lines, start=1):
        try:
            raw: dict[str, Any] = json.loads(line)
            raw["left_identity_tokens"] = tuple(raw["left_identity_tokens"])
            raw["right_identity_tokens"] = tuple(raw["right_identity_tokens"])
            pair = MinedHardNegative(**raw)
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise DataValidationError(
                f"Invalid hard-negative record at line {line_number}"
            ) from exc
        key = (pair.left_posting_id, pair.right_posting_id)
        if pair.left_posting_id >= pair.right_posting_id or key in seen:
            raise DataValidationError("Hard-negative manifest is not canonical and unique")
        if pair.left_label_group == pair.right_label_group:
            raise DataValidationError("Hard-negative manifest contains a positive pair")
        if not 0 <= pair.pair_probability <= 1:
            raise DataValidationError("Hard-negative pair probability is outside [0, 1]")
        seen.add(key)
        pairs.append(pair)
    if not pairs:
        raise DataValidationError("Hard-negative manifest is empty")
    if [(pair.left_posting_id, pair.right_posting_id) for pair in pairs] != sorted(seen):
        raise DataValidationError("Hard-negative manifest records are not sorted")
    return pairs


class HardNegativeBatchProvider:
    """Map mined IDs to train rows and sample pairs reproducibly per optimization step."""

    def __init__(
        self,
        pairs: list[MinedHardNegative],
        posting_ids: tuple[str, ...],
        labels: tuple[str, ...],
        *,
        seed: int,
    ) -> None:
        index_by_id = {posting_id: index for index, posting_id in enumerate(posting_ids)}
        if len(index_by_id) != len(posting_ids):
            raise DataValidationError("Train cache contains duplicate posting IDs")
        self.index_pairs: tuple[tuple[int, int], ...] = tuple(
            (index_by_id[pair.left_posting_id], index_by_id[pair.right_posting_id])
            for pair in pairs
            if pair.left_posting_id in index_by_id and pair.right_posting_id in index_by_id
        )
        if len(self.index_pairs) != len(pairs):
            raise DataValidationError("Mined pair IDs do not exactly match the train cache")
        if any(labels[left] == labels[right] for left, right in self.index_pairs):
            raise DataValidationError("Mined train pair unexpectedly has the same label")
        self.seed = seed

    def sample(self, epoch: int, batch_index: int, count: int) -> tuple[Tensor, Tensor]:
        """Return CPU index tensors; sampling is stable for an epoch/batch coordinate."""
        if epoch < 0 or batch_index < 0 or count <= 0:
            raise ValueError("invalid hard-negative batch coordinate")
        rng = random.Random(f"{self.seed}:{epoch}:{batch_index}")
        if len(self.index_pairs) >= count:
            chosen = rng.sample(self.index_pairs, count)
        else:
            chosen = rng.choices(self.index_pairs, k=count)
        return (
            torch.tensor([left for left, _ in chosen], dtype=torch.long),
            torch.tensor([right for _, right in chosen], dtype=torch.long),
        )
