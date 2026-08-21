"""Label-blind pair scoring and conservative reciprocal-neighbour clustering."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

from shopee_match.evaluation.protocol import CorpusItem, Ranking
from shopee_match.models import LearnedMultimodalFusion
from shopee_match.training.hard_negative_data import has_variant_conflict

FloatArray = NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class ScoredPair:
    """One deterministic undirected candidate pair with directional retrieval ranks."""

    left_posting_id: str
    right_posting_id: str
    left_index: int
    right_index: int
    cosine_similarity: float
    pair_probability: float
    left_rank: int
    right_rank: int
    variant_conflict: bool


@dataclass(frozen=True, slots=True)
class ClusterAssignment:
    """Label-blind entity assignment and review evidence for one listing."""

    posting_id: str
    entity_id: str
    cluster_size: int
    cluster_confidence: float
    manual_review: bool


@dataclass(frozen=True, slots=True)
class GraphDiagnostics:
    candidate_pairs: int
    below_probability: int
    non_reciprocal: int
    variant_conflict_rejected: int
    eligible_edges: int
    accepted_merges: int
    size_rejections: int
    consistency_rejections: int
    clusters: int
    singleton_clusters: int
    manual_review_clusters: int


class _UnionFind:
    def __init__(self, posting_ids: tuple[str, ...]) -> None:
        self.parent = list(range(len(posting_ids)))
        self.members: dict[int, set[int]] = {index: {index} for index in range(len(posting_ids))}
        self.posting_ids = posting_ids

    def find(self, node: int) -> int:
        root = node
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[node] != node:
            parent = self.parent[node]
            self.parent[node] = root
            node = parent
        return root

    def union(self, left: int, right: int) -> int:
        left_root, right_root = self.find(left), self.find(right)
        if left_root == right_root:
            return left_root
        left_key = min(self.posting_ids[index] for index in self.members[left_root])
        right_key = min(self.posting_ids[index] for index in self.members[right_root])
        keep, drop = (left_root, right_root) if left_key <= right_key else (right_root, left_root)
        self.parent[drop] = keep
        self.members[keep].update(self.members.pop(drop))
        return keep


def score_candidate_pairs(
    model: LearnedMultimodalFusion,
    posting_ids: tuple[str, ...],
    items: tuple[CorpusItem, ...],
    embeddings: FloatArray,
    ranking: Ranking,
    device: torch.device,
    *,
    batch_size: int,
) -> list[ScoredPair]:
    """Deduplicate directed candidates and score each undirected pair exactly once."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if len(posting_ids) != len(items) or embeddings.shape[0] != len(posting_ids):
        raise ValueError("posting IDs, items, and embeddings must align")
    index_by_id = {posting_id: index for index, posting_id in enumerate(posting_ids)}
    item_by_id = {item.posting_id: item for item in items}
    directed: dict[tuple[int, int], tuple[int, float]] = {}
    undirected: set[tuple[int, int]] = set()
    for query_id in posting_ids:
        query_index = index_by_id[query_id]
        for rank, candidate in enumerate(ranking[query_id], start=1):
            candidate_index = index_by_id[candidate.posting_id]
            directed[(query_index, candidate_index)] = (rank, candidate.score)
            pair_key = (
                (query_index, candidate_index)
                if query_index < candidate_index
                else (candidate_index, query_index)
            )
            undirected.add(pair_key)
    pair_indices = sorted(
        undirected,
        key=lambda pair: (posting_ids[pair[0]], posting_ids[pair[1]]),
    )
    tensor = torch.from_numpy(embeddings.astype(np.float32, copy=False))
    probabilities: list[float] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(pair_indices), batch_size):
            chunk = pair_indices[start : start + batch_size]
            left_tensor = tensor[[pair[0] for pair in chunk]].to(device)
            right_tensor = tensor[[pair[1] for pair in chunk]].to(device)
            probabilities.extend(
                torch.sigmoid(model.pair_logits(left_tensor, right_tensor)).cpu().tolist()
            )
    missing_rank = len(posting_ids) + 1
    result: list[ScoredPair] = []
    for (left_index, right_index), probability in zip(pair_indices, probabilities, strict=True):
        cosine = float(embeddings[left_index] @ embeddings[right_index])
        left_direction = directed.get((left_index, right_index), (missing_rank, cosine))
        right_direction = directed.get((right_index, left_index), (missing_rank, cosine))
        left_id, right_id = posting_ids[left_index], posting_ids[right_index]
        result.append(
            ScoredPair(
                left_posting_id=left_id,
                right_posting_id=right_id,
                left_index=left_index,
                right_index=right_index,
                cosine_similarity=max(left_direction[1], right_direction[1]),
                pair_probability=float(probability),
                left_rank=left_direction[0],
                right_rank=right_direction[0],
                variant_conflict=has_variant_conflict(
                    item_by_id[left_id].title, item_by_id[right_id].title
                ),
            )
        )
    return result


def eligible_pairs(
    pairs: list[ScoredPair],
    *,
    pair_probability_threshold: float,
    reciprocal_rank: int,
    variant_conflict_override_probability: float,
) -> tuple[list[ScoredPair], dict[str, int]]:
    """Apply label-blind edge gates and return deterministic strongest-first edges."""
    counters = {
        "below_probability": 0,
        "non_reciprocal": 0,
        "variant_conflict_rejected": 0,
    }
    result: list[ScoredPair] = []
    for pair in pairs:
        if pair.pair_probability < pair_probability_threshold:
            counters["below_probability"] += 1
        elif max(pair.left_rank, pair.right_rank) > reciprocal_rank:
            counters["non_reciprocal"] += 1
        elif (
            pair.variant_conflict and pair.pair_probability < variant_conflict_override_probability
        ):
            counters["variant_conflict_rejected"] += 1
        else:
            result.append(pair)
    result.sort(
        key=lambda pair: (
            -pair.pair_probability,
            -pair.cosine_similarity,
            pair.left_posting_id,
            pair.right_posting_id,
        )
    )
    return result, counters


def _cross_component_coverage(
    left_members: set[int],
    right_members: set[int],
    adjacency: dict[int, set[int]],
) -> float:
    left_covered = sum(bool(adjacency[node] & right_members) for node in left_members)
    right_covered = sum(bool(adjacency[node] & left_members) for node in right_members)
    return min(left_covered / len(left_members), right_covered / len(right_members))


def build_conservative_clusters(
    posting_ids: tuple[str, ...],
    pairs: list[ScoredPair],
    *,
    pair_probability_threshold: float,
    reciprocal_rank: int,
    cross_component_minimum_coverage: float,
    variant_conflict_override_probability: float,
    maximum_cluster_size: int,
    manual_review_margin: float,
) -> tuple[list[ClusterAssignment], GraphDiagnostics]:
    """Build connected components while blocking weak transitive component bridges."""
    eligible, counters = eligible_pairs(
        pairs,
        pair_probability_threshold=pair_probability_threshold,
        reciprocal_rank=reciprocal_rank,
        variant_conflict_override_probability=variant_conflict_override_probability,
    )
    adjacency: dict[int, set[int]] = defaultdict(set)
    for pair in eligible:
        adjacency[pair.left_index].add(pair.right_index)
        adjacency[pair.right_index].add(pair.left_index)
    union_find = _UnionFind(posting_ids)
    accepted: list[ScoredPair] = []
    size_rejections = 0
    consistency_rejections = 0
    for pair in eligible:
        left_root = union_find.find(pair.left_index)
        right_root = union_find.find(pair.right_index)
        if left_root == right_root:
            continue
        left_members = union_find.members[left_root]
        right_members = union_find.members[right_root]
        if len(left_members) + len(right_members) > maximum_cluster_size:
            size_rejections += 1
            continue
        coverage = _cross_component_coverage(left_members, right_members, adjacency)
        if coverage < cross_component_minimum_coverage:
            consistency_rejections += 1
            continue
        union_find.union(left_root, right_root)
        accepted.append(pair)

    components = sorted(
        union_find.members.values(),
        key=lambda members: min(posting_ids[index] for index in members),
    )
    accepted_by_component: dict[int, list[ScoredPair]] = defaultdict(list)
    root_by_node = {index: union_find.find(index) for index in range(len(posting_ids))}
    for pair in accepted:
        accepted_by_component[root_by_node[pair.left_index]].append(pair)

    assignments: list[ClusterAssignment] = []
    manual_review_clusters = 0
    for entity_number, members in enumerate(components, start=1):
        root = root_by_node[next(iter(members))]
        component_edges = accepted_by_component[root]
        confidence = min((pair.pair_probability for pair in component_edges), default=0.0)
        has_low_confidence_variant = any(
            pair.variant_conflict
            and pair.pair_probability < pair_probability_threshold + 2 * manual_review_margin
            for pair in component_edges
        )
        manual_review = len(members) > 1 and (
            confidence < pair_probability_threshold + manual_review_margin
            or has_low_confidence_variant
        )
        manual_review_clusters += int(manual_review)
        entity_id = f"entity_{entity_number:06d}"
        for index in sorted(members, key=lambda value: posting_ids[value]):
            assignments.append(
                ClusterAssignment(
                    posting_id=posting_ids[index],
                    entity_id=entity_id,
                    cluster_size=len(members),
                    cluster_confidence=confidence,
                    manual_review=manual_review,
                )
            )
    assignments.sort(key=lambda row: row.posting_id)
    diagnostics = GraphDiagnostics(
        candidate_pairs=len(pairs),
        below_probability=counters["below_probability"],
        non_reciprocal=counters["non_reciprocal"],
        variant_conflict_rejected=counters["variant_conflict_rejected"],
        eligible_edges=len(eligible),
        accepted_merges=len(accepted),
        size_rejections=size_rejections,
        consistency_rejections=consistency_rejections,
        clusters=len(components),
        singleton_clusters=sum(len(component) == 1 for component in components),
        manual_review_clusters=manual_review_clusters,
    )
    return assignments, diagnostics


def scored_pair_payload(pair: ScoredPair) -> dict[str, Any]:
    """Return a stable JSON-serializable pair record."""
    return {
        "left_posting_id": pair.left_posting_id,
        "right_posting_id": pair.right_posting_id,
        "cosine_similarity": pair.cosine_similarity,
        "pair_probability": pair.pair_probability,
        "left_rank": pair.left_rank,
        "right_rank": pair.right_rank,
        "variant_conflict": pair.variant_conflict,
    }
