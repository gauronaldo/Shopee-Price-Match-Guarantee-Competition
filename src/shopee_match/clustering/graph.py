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
    singleton_attachment_attempts: int
    singleton_attachments: int
    singleton_attachment_ambiguous: int
    singleton_attachment_insufficient_support: int
    singleton_attachment_size_rejections: int
    fragment_attachment_attempts: int
    fragment_attachments: int
    fragment_attachment_ambiguous: int
    fragment_attachment_insufficient_support: int
    fragment_attachment_size_rejections: int
    fragment_attachment_variant_conflict_rejections: int
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


def _attach_supported_singletons(
    union_find: _UnionFind,
    posting_ids: tuple[str, ...],
    eligible: list[ScoredPair],
    accepted: list[ScoredPair],
    *,
    minimum_support: int,
    target_margin: float,
    maximum_cluster_size: int,
) -> dict[str, int]:
    """Attach a core singleton only when distinct members support one established component.

    Component membership is snapshotted before attachment. Consequently, newly attached
    singletons cannot provide evidence for later attachments and two singleton chains cannot
    bootstrap a cluster without support from an established core component.
    """
    core_root_by_node = {index: union_find.find(index) for index in range(len(posting_ids))}
    core_members = {root: frozenset(members) for root, members in union_find.members.items()}
    singleton_nodes = sorted(
        (next(iter(members)) for members in core_members.values() if len(members) == 1),
        key=lambda index: posting_ids[index],
    )
    support_by_singleton: dict[int, dict[int, list[ScoredPair]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for pair in eligible:
        left_root = core_root_by_node[pair.left_index]
        right_root = core_root_by_node[pair.right_index]
        if left_root == right_root:
            continue
        if len(core_members[left_root]) == 1 and len(core_members[right_root]) > 1:
            support_by_singleton[pair.left_index][right_root].append(pair)
        if len(core_members[right_root]) == 1 and len(core_members[left_root]) > 1:
            support_by_singleton[pair.right_index][left_root].append(pair)

    counters = {
        "attempts": 0,
        "attachments": 0,
        "ambiguous": 0,
        "insufficient_support": 0,
        "size_rejections": 0,
    }
    for singleton in singleton_nodes:
        target_evidence = support_by_singleton.get(singleton)
        if not target_evidence:
            continue
        counters["attempts"] += 1
        candidates: list[tuple[int, float, float, str, int, list[ScoredPair]]] = []
        for target_root, evidence in target_evidence.items():
            ordered = sorted(
                evidence,
                key=lambda pair: (
                    -pair.pair_probability,
                    -pair.cosine_similarity,
                    pair.left_posting_id,
                    pair.right_posting_id,
                ),
            )
            if len(ordered) < minimum_support:
                continue
            required = ordered[:minimum_support]
            target_key = min(posting_ids[index] for index in core_members[target_root])
            candidates.append(
                (
                    len(ordered),
                    float(np.mean([pair.pair_probability for pair in required])),
                    min(pair.pair_probability for pair in required),
                    target_key,
                    target_root,
                    required,
                )
            )
        if not candidates:
            counters["insufficient_support"] += 1
            continue
        candidates.sort(key=lambda row: (-row[0], -row[1], -row[2], row[3]))
        best = candidates[0]
        if (
            len(candidates) > 1
            and candidates[1][0] == best[0]
            and best[1] - candidates[1][1] < target_margin
        ):
            counters["ambiguous"] += 1
            continue
        current_target_root = union_find.find(best[4])
        if 1 + len(union_find.members[current_target_root]) > maximum_cluster_size:
            counters["size_rejections"] += 1
            continue
        union_find.union(singleton, current_target_root)
        # The weakest required support edge is the conservative confidence of this merge.
        accepted.append(best[5][-1])
        counters["attachments"] += 1
    return counters


def _attach_supported_fragments(
    union_find: _UnionFind,
    posting_ids: tuple[str, ...],
    eligible: list[ScoredPair],
    accepted: list[ScoredPair],
    *,
    maximum_source_size: int,
    minimum_target_size: int,
    minimum_support: int,
    minimum_source_coverage: float,
    minimum_target_support: int,
    target_margin: float,
    maximum_cluster_size: int,
    reject_variant_conflicts: bool,
) -> dict[str, int]:
    """Attach a small frozen component to a larger component with multi-node support.

    All component membership and cross-component evidence are snapshotted before this pass.
    Attached fragments therefore cannot create evidence for later attachments. A component that
    has already been attached as a source is also prevented from acting as an intermediate target.
    """
    root_by_node = {index: union_find.find(index) for index in range(len(posting_ids))}
    frozen_members = {root: frozenset(members) for root, members in union_find.members.items()}
    evidence_by_source: dict[int, dict[int, list[ScoredPair]]] = defaultdict(
        lambda: defaultdict(list)
    )
    variant_conflict_rejections = 0
    for pair in eligible:
        left_root = root_by_node[pair.left_index]
        right_root = root_by_node[pair.right_index]
        if left_root == right_root:
            continue
        left_size = len(frozen_members[left_root])
        right_size = len(frozen_members[right_root])
        left_key = min(posting_ids[index] for index in frozen_members[left_root])
        right_key = min(posting_ids[index] for index in frozen_members[right_root])
        if (
            1 < left_size <= maximum_source_size
            and right_size >= minimum_target_size
            and (left_size < right_size or (left_size == right_size and left_key < right_key))
        ):
            source_root, target_root = left_root, right_root
        elif (
            1 < right_size <= maximum_source_size
            and left_size >= minimum_target_size
            and (right_size < left_size or (right_size == left_size and right_key < left_key))
        ):
            source_root, target_root = right_root, left_root
        else:
            continue
        if reject_variant_conflicts and pair.variant_conflict:
            variant_conflict_rejections += 1
            continue
        evidence_by_source[source_root][target_root].append(pair)

    counters = {
        "attempts": 0,
        "attachments": 0,
        "ambiguous": 0,
        "insufficient_support": 0,
        "size_rejections": 0,
        "variant_conflict_rejections": variant_conflict_rejections,
    }
    attached_source_roots: set[int] = set()
    source_roots = sorted(
        evidence_by_source,
        key=lambda root: (
            len(frozen_members[root]),
            min(posting_ids[index] for index in frozen_members[root]),
        ),
    )
    for source_root in source_roots:
        current_source_root = union_find.find(source_root)
        if len(union_find.members[current_source_root]) != len(frozen_members[source_root]):
            continue
        counters["attempts"] += 1
        candidates: list[tuple[float, int, int, float, float, str, int, list[ScoredPair]]] = []
        for target_root, evidence in evidence_by_source[source_root].items():
            if target_root in attached_source_roots:
                continue
            source_nodes: set[int] = set()
            target_nodes: set[int] = set()
            for pair in evidence:
                if pair.left_index in frozen_members[source_root]:
                    source_nodes.add(pair.left_index)
                    target_nodes.add(pair.right_index)
                else:
                    source_nodes.add(pair.right_index)
                    target_nodes.add(pair.left_index)
            source_coverage = len(source_nodes) / len(frozen_members[source_root])
            if (
                len(evidence) < minimum_support
                or source_coverage < minimum_source_coverage
                or len(target_nodes) < minimum_target_support
            ):
                continue
            ordered = sorted(
                evidence,
                key=lambda pair: (
                    -pair.pair_probability,
                    -pair.cosine_similarity,
                    pair.left_posting_id,
                    pair.right_posting_id,
                ),
            )
            target_key = min(posting_ids[index] for index in frozen_members[target_root])
            candidates.append(
                (
                    source_coverage,
                    len(target_nodes),
                    len(ordered),
                    float(np.mean([pair.pair_probability for pair in ordered])),
                    min(pair.pair_probability for pair in ordered),
                    target_key,
                    target_root,
                    ordered,
                )
            )
        if not candidates:
            counters["insufficient_support"] += 1
            continue
        candidates.sort(key=lambda row: (-row[0], -row[1], -row[2], -row[3], -row[4], row[5]))
        best = candidates[0]
        if len(candidates) > 1 and best[3] - candidates[1][3] < target_margin:
            counters["ambiguous"] += 1
            continue
        current_target_root = union_find.find(best[6])
        if len(union_find.members[current_source_root]) + len(
            union_find.members[current_target_root]
        ) > maximum_cluster_size:
            counters["size_rejections"] += 1
            continue
        union_find.union(current_source_root, current_target_root)
        accepted.append(min(best[7], key=lambda pair: pair.pair_probability))
        attached_source_roots.add(source_root)
        counters["attachments"] += 1
    return counters


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
    singleton_attachment: bool = False,
    singleton_probability_threshold: float | None = None,
    singleton_reciprocal_rank: int | None = None,
    singleton_minimum_support: int = 2,
    singleton_target_margin: float = 0.0,
    fragment_attachment: bool = False,
    fragment_probability_threshold: float | None = None,
    fragment_reciprocal_rank: int | None = None,
    fragment_maximum_source_size: int = 3,
    fragment_minimum_target_size: int = 3,
    fragment_minimum_support: int = 2,
    fragment_minimum_source_coverage: float = 1.0,
    fragment_minimum_target_support: int = 2,
    fragment_target_margin: float = 0.0,
    fragment_reject_variant_conflicts: bool = True,
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

    attachment_counters = {
        "attempts": 0,
        "attachments": 0,
        "ambiguous": 0,
        "insufficient_support": 0,
        "size_rejections": 0,
    }
    if singleton_attachment:
        if singleton_probability_threshold is None or singleton_reciprocal_rank is None:
            raise ValueError("singleton attachment requires a probability threshold and rank")
        if singleton_minimum_support < 2:
            raise ValueError("singleton attachment requires at least two independent supports")
        attachment_eligible, _ = eligible_pairs(
            pairs,
            pair_probability_threshold=singleton_probability_threshold,
            reciprocal_rank=singleton_reciprocal_rank,
            variant_conflict_override_probability=variant_conflict_override_probability,
        )
        attachment_counters = _attach_supported_singletons(
            union_find,
            posting_ids,
            attachment_eligible,
            accepted,
            minimum_support=singleton_minimum_support,
            target_margin=singleton_target_margin,
            maximum_cluster_size=maximum_cluster_size,
        )

    fragment_counters = {
        "attempts": 0,
        "attachments": 0,
        "ambiguous": 0,
        "insufficient_support": 0,
        "size_rejections": 0,
        "variant_conflict_rejections": 0,
    }
    if fragment_attachment:
        if fragment_probability_threshold is None or fragment_reciprocal_rank is None:
            raise ValueError("fragment attachment requires a probability threshold and rank")
        if fragment_maximum_source_size < 2:
            raise ValueError("fragment maximum source size must be at least two")
        if fragment_minimum_target_size < 2:
            raise ValueError("fragment minimum target size must be at least two")
        if fragment_minimum_support < 2 or fragment_minimum_target_support < 2:
            raise ValueError("fragment attachment requires at least two independent supports")
        if not 0.0 < fragment_minimum_source_coverage <= 1.0:
            raise ValueError("fragment source coverage must be inside (0, 1]")
        fragment_eligible, _ = eligible_pairs(
            pairs,
            pair_probability_threshold=fragment_probability_threshold,
            reciprocal_rank=fragment_reciprocal_rank,
            variant_conflict_override_probability=variant_conflict_override_probability,
        )
        fragment_counters = _attach_supported_fragments(
            union_find,
            posting_ids,
            fragment_eligible,
            accepted,
            maximum_source_size=fragment_maximum_source_size,
            minimum_target_size=fragment_minimum_target_size,
            minimum_support=fragment_minimum_support,
            minimum_source_coverage=fragment_minimum_source_coverage,
            minimum_target_support=fragment_minimum_target_support,
            target_margin=fragment_target_margin,
            maximum_cluster_size=maximum_cluster_size,
            reject_variant_conflicts=fragment_reject_variant_conflicts,
        )

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
        singleton_attachment_attempts=attachment_counters["attempts"],
        singleton_attachments=attachment_counters["attachments"],
        singleton_attachment_ambiguous=attachment_counters["ambiguous"],
        singleton_attachment_insufficient_support=attachment_counters["insufficient_support"],
        singleton_attachment_size_rejections=attachment_counters["size_rejections"],
        fragment_attachment_attempts=fragment_counters["attempts"],
        fragment_attachments=fragment_counters["attachments"],
        fragment_attachment_ambiguous=fragment_counters["ambiguous"],
        fragment_attachment_insufficient_support=fragment_counters["insufficient_support"],
        fragment_attachment_size_rejections=fragment_counters["size_rejections"],
        fragment_attachment_variant_conflict_rejections=fragment_counters[
            "variant_conflict_rejections"
        ],
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
