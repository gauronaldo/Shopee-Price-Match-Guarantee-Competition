"""Exact multimodal score fusion and learned pair-head reranking."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import torch
from numpy.typing import NDArray

from shopee_match.evaluation.embedding_retrieval import rank_cosine_embeddings
from shopee_match.evaluation.protocol import Ranking, ScoredCandidate, retrieval_metrics
from shopee_match.models import LearnedMultimodalFusion

FloatArray = NDArray[np.floating[Any]]


def rank_simple_score_fusion(
    posting_ids: tuple[str, ...],
    image_embeddings: FloatArray,
    text_embeddings: FloatArray,
    *,
    image_weight: float,
    candidate_k: int,
) -> Ranking:
    """Rank a validation split by a weighted sum of modality cosine scores."""
    if not 0 <= image_weight <= 1:
        raise ValueError("image_weight must be inside [0, 1]")
    if image_embeddings.shape != text_embeddings.shape:
        raise ValueError("image and text embedding matrices must have equal shapes")
    if image_embeddings.ndim != 2 or image_embeddings.shape[0] != len(posting_ids):
        raise ValueError("embedding matrices must have shape [len(posting_ids), dimension]")

    def normalized(values: FloatArray) -> FloatArray:
        norms = np.linalg.norm(values, axis=1, keepdims=True)
        if not np.isfinite(values).all() or np.any(norms <= 0):
            raise ValueError("modality embeddings must be finite and non-zero")
        return cast(FloatArray, values / norms)

    image = normalized(image_embeddings)
    text = normalized(text_embeddings)
    scores = image_weight * (image @ image.T) + (1 - image_weight) * (text @ text.T)
    identifiers = np.asarray(posting_ids, dtype=str)
    limit = min(candidate_k, max(0, len(posting_ids) - 1))
    ranking: Ranking = {}
    for query_index, query_id in enumerate(posting_ids):
        row = scores[query_index].copy()
        row[query_index] = -np.inf
        order = np.lexsort((identifiers, -row))
        candidates = [index for index in order if index != query_index][:limit]
        ranking[query_id] = [
            ScoredCandidate(posting_ids[index], float(row[index])) for index in candidates
        ]
    return ranking


def select_simple_score_fusion(
    posting_ids: tuple[str, ...],
    image_embeddings: FloatArray,
    text_embeddings: FloatArray,
    label_by_id: dict[str, str],
    *,
    image_weights: tuple[float, ...],
    candidate_k: int,
    recall_at: tuple[int, ...],
    average_precision_at: int,
) -> tuple[float, Ranking, dict[str, float], list[dict[str, float]]]:
    """Select a simple fusion weight on validation mAP with deterministic ties."""
    trials: list[dict[str, float]] = []
    selected: tuple[float, Ranking, dict[str, float]] | None = None
    metric_name = f"map@{average_precision_at}"
    for weight in image_weights:
        ranking = rank_simple_score_fusion(
            posting_ids,
            image_embeddings,
            text_embeddings,
            image_weight=weight,
            candidate_k=candidate_k,
        )
        metrics = retrieval_metrics(ranking, label_by_id, recall_at, average_precision_at)
        trials.append({"image_weight": weight, **metrics})
        candidate = (weight, ranking, metrics)
        if selected is None or (metrics[metric_name], -abs(weight - 0.5), -weight) > (
            selected[2][metric_name],
            -abs(selected[0] - 0.5),
            -selected[0],
        ):
            selected = candidate
    if selected is None:
        raise ValueError("simple fusion requires at least one image weight")
    return selected[0], selected[1], selected[2], trials


def rerank_with_pair_head(
    model: LearnedMultimodalFusion,
    posting_ids: tuple[str, ...],
    joint_embeddings: FloatArray,
    candidate_ranking: Ranking,
    device: torch.device,
    *,
    batch_size: int = 4096,
) -> Ranking:
    """Rerank fixed candidates by symmetric pair probability without changing recall ceiling."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if joint_embeddings.ndim != 2 or joint_embeddings.shape[0] != len(posting_ids):
        raise ValueError("joint_embeddings must align with posting_ids")
    index_by_id = {posting_id: index for index, posting_id in enumerate(posting_ids)}
    pairs = [
        (query_id, candidate.posting_id)
        for query_id in posting_ids
        for candidate in candidate_ranking[query_id]
    ]
    embeddings = torch.from_numpy(joint_embeddings.astype(np.float32, copy=False))
    probabilities: list[float] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(pairs), batch_size):
            chunk = pairs[start : start + batch_size]
            left = embeddings[[index_by_id[left_id] for left_id, _ in chunk]].to(device)
            right = embeddings[[index_by_id[right_id] for _, right_id in chunk]].to(device)
            probabilities.extend(torch.sigmoid(model.pair_logits(left, right)).cpu().tolist())
    reranked: Ranking = {posting_id: [] for posting_id in posting_ids}
    for (query_id, candidate_id), score in zip(pairs, probabilities, strict=True):
        reranked[query_id].append(ScoredCandidate(candidate_id, float(score)))
    for query_id in reranked:
        reranked[query_id].sort(key=lambda candidate: (-candidate.score, candidate.posting_id))
    return reranked


def unimodal_rankings(
    posting_ids: tuple[str, ...],
    image_embeddings: FloatArray,
    text_embeddings: FloatArray,
    candidate_k: int,
) -> tuple[Ranking, Ranking]:
    """Return exact image-only and text-only rankings for Phase 5 ablations."""
    return (
        rank_cosine_embeddings(posting_ids, image_embeddings, candidate_k),
        rank_cosine_embeddings(posting_ids, text_embeddings, candidate_k),
    )
