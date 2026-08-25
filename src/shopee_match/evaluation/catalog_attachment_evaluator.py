"""Development-only evaluation for static catalog attachment and new-entity detection."""

from __future__ import annotations

import json
import logging
import os
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from shopee_match.errors import ConfigurationError, DataValidationError, OutputConflictError
from shopee_match.evaluation.catalog_attachment_config import (
    CatalogAttachmentComparison,
    CatalogAttachmentConfig,
    CatalogAttachmentSafety,
    load_catalog_attachment_config,
)
from shopee_match.evaluation.catalog_attachment_protocol import (
    CatalogRole,
    load_catalog_roles,
    validate_catalog_roles,
)
from shopee_match.evaluation.protocol import (
    CorpusItem,
    EvaluationSplit,
    Ranking,
    ScoredCandidate,
    load_named_split,
)
from shopee_match.features.text import CharTfidfModel
from shopee_match.hashing import canonical_text_sha256, sha256_file
from shopee_match.models import LearnedMultimodalFusion, ScratchResidualImageEncoder, ScratchTextCNN
from shopee_match.reproducibility import seed_everything
from shopee_match.retrieval.benchmark import load_phase6_model
from shopee_match.retrieval.hybrid import reciprocal_rank_fusion
from shopee_match.retrieval.vector_index import ExactCosineIndex, search_result_to_ranking
from shopee_match.training.hard_negative_data import has_variant_conflict
from shopee_match.training.image_data import ImagePreprocessor
from shopee_match.training.joint_recall_trainer import load_joint_recall_config
from shopee_match.training.multimodal_data import load_frozen_encoders
from shopee_match.training.multimodal_trainer import _git_state, _resolve_device
from shopee_match.training.text_data import CharacterVocabulary

LOGGER = logging.getLogger(__name__)
FloatArray = NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class PairEvidence:
    query_id: str
    candidate_id: str
    retrieval_rank: int
    pair_probability: float
    variant_conflict: bool


def _write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _load_models(
    config: CatalogAttachmentConfig,
    device: torch.device,
) -> tuple[
    ScratchResidualImageEncoder,
    ScratchTextCNN,
    LearnedMultimodalFusion,
    CharacterVocabulary,
    int,
    Path,
    int,
]:
    phase7 = config.hybrid.source.experiment
    phase6 = phase7.source.experiment
    multimodal = phase6.source.experiment
    model = load_phase6_model(phase7, device)
    image_model, text_model, vocabulary, maximum_length = load_frozen_encoders(multimodal, device)
    if config.model_variant == "full_joint_candidate":
        if config.full_joint_config is None or config.full_joint_checkpoint is None:
            raise ConfigurationError("Full-joint candidate sources are incomplete")
        joint_config = load_joint_recall_config(config.full_joint_config)
        if joint_config.checkpoint_output != config.full_joint_checkpoint:
            raise ConfigurationError("Candidate checkpoint differs from its training config")
        payload = torch.load(config.full_joint_checkpoint, map_location="cpu", weights_only=False)
        if (
            payload.get("checkpoint_version") != "multimodal.full_joint_pair_recall.v1"
            or payload.get("config_sha256") != canonical_text_sha256(config.full_joint_config)
            or not isinstance(payload.get("model_state"), dict)
        ):
            raise ConfigurationError("Full-joint checkpoint contract is invalid")
        state = cast(dict[str, Any], payload["model_state"])
        image_model.load_state_dict(state["image_model"])
        text_model.load_state_dict(state["text_model"])
        model.load_state_dict(state["fusion_model"])
    image_model.eval().to(device)
    text_model.eval().to(device)
    model.eval().to(device)
    return (
        image_model,
        text_model,
        model,
        vocabulary,
        maximum_length,
        multimodal.data.image_dir,
        multimodal.frozen.image_config.training_experiment.image_size,
    )


def _extract_embeddings(
    split: EvaluationSplit,
    *,
    image_dir: Path,
    image_size: int,
    image_model: ScratchResidualImageEncoder,
    text_model: ScratchTextCNN,
    fusion_model: LearnedMultimodalFusion,
    vocabulary: CharacterVocabulary,
    maximum_length: int,
    device: torch.device,
    batch_size: int,
) -> tuple[tuple[str, ...], FloatArray]:
    preprocessor = ImagePreprocessor(image_size, training=False, seed=0)
    identifiers: list[str] = []
    chunks: list[FloatArray] = []
    with torch.inference_mode():
        for start in range(0, len(split.items), batch_size):
            items = split.items[start : start + batch_size]
            images: list[Tensor] = []
            token_ids: list[Tensor] = []
            lengths: list[Tensor] = []
            for offset, item in enumerate(items):
                image = cv2.imread(str(image_dir / item.image), cv2.IMREAD_COLOR)
                if image is None:
                    raise DataValidationError(
                        f"Cannot decode catalog-evaluation image: {item.image}"
                    )
                rgb = cast(NDArray[np.uint8], cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                images.append(preprocessor(rgb, start + offset))
                tokens, length = vocabulary.encode(item.title, maximum_length)
                token_ids.append(tokens)
                lengths.append(length)
            image_embeddings = image_model(torch.stack(images).to(device))
            text_embeddings = text_model(
                torch.stack(token_ids).to(device),
                torch.stack(lengths).to(device),
            )
            joint = fusion_model(image_embeddings, text_embeddings)
            chunks.append(cast(FloatArray, joint.cpu().numpy()).astype(np.float32, copy=False))
            identifiers.extend(item.posting_id for item in items)
    embeddings = np.concatenate(chunks)
    if not np.isfinite(embeddings).all() or not np.allclose(
        np.linalg.norm(embeddings, axis=1), 1.0, atol=1e-5
    ):
        raise DataValidationError("Catalog-attachment embeddings are invalid")
    return tuple(identifiers), embeddings


def cross_text_ranking(
    model: CharTfidfModel,
    queries: tuple[CorpusItem, ...],
    references: tuple[CorpusItem, ...],
    top_k: int,
) -> Ranking:
    reference_vectors = {item.posting_id: model.transform_one(item.title) for item in references}
    postings: dict[int, list[tuple[str, float]]] = defaultdict(list)
    for posting_id, vector in reference_vectors.items():
        for feature, weight in vector.items():
            postings[feature].append((posting_id, weight))
    reference_ids = sorted(reference_vectors)
    ranking: Ranking = {}
    for query in queries:
        scores: dict[str, float] = defaultdict(float)
        for feature, query_weight in model.transform_one(query.title).items():
            for candidate_id, candidate_weight in postings[feature]:
                scores[candidate_id] += query_weight * candidate_weight
        ordered = sorted(scores.items(), key=lambda row: (-row[1], row[0]))[:top_k]
        selected = {identifier for identifier, _score in ordered}
        ordered.extend(
            (identifier, 0.0) for identifier in reference_ids if identifier not in selected
        )
        ranking[query.posting_id] = [
            ScoredCandidate(identifier, float(score)) for identifier, score in ordered[:top_k]
        ]
    return ranking


def cross_phash_ranking(
    queries: tuple[CorpusItem, ...],
    references: tuple[CorpusItem, ...],
    top_k: int,
) -> Ranking:
    ordered_references = sorted(references, key=lambda item: item.posting_id)
    ranking: Ranking = {}
    for query in sorted(queries, key=lambda item: item.posting_id):
        candidates = sorted(
            ordered_references,
            key=lambda item: (
                (int(query.image_phash, 16) ^ int(item.image_phash, 16)).bit_count(),
                item.posting_id,
            ),
        )[:top_k]
        ranking[query.posting_id] = [
            ScoredCandidate(
                item.posting_id,
                1.0 - (int(query.image_phash, 16) ^ int(item.image_phash, 16)).bit_count() / 64.0,
            )
            for item in candidates
        ]
    return ranking


def _hybrid_ranking(
    config: CatalogAttachmentConfig,
    train: EvaluationSplit,
    development: EvaluationSplit,
    posting_ids: tuple[str, ...],
    embeddings: FloatArray,
    roles: tuple[CatalogRole, ...],
) -> tuple[Ranking, tuple[CorpusItem, ...], tuple[CorpusItem, ...]]:
    role_by_id = {row.posting_id: row.role for row in roles}
    references = tuple(
        item for item in development.items if role_by_id[item.posting_id] == "catalog_reference"
    )
    queries = tuple(
        item for item in development.items if role_by_id[item.posting_id] != "catalog_reference"
    )
    index_by_id = {identifier: index for index, identifier in enumerate(posting_ids)}
    reference_ids = tuple(item.posting_id for item in references)
    query_ids = tuple(item.posting_id for item in queries)
    reference_embeddings = embeddings[[index_by_id[identifier] for identifier in reference_ids]]
    query_embeddings = embeddings[[index_by_id[identifier] for identifier in query_ids]]
    source_k = min(len(reference_ids), max(config.policy.candidate_k, 50))
    index = ExactCosineIndex(reference_ids, reference_embeddings)
    indices, scores = index.search(
        query_embeddings,
        source_k,
        query_ids=query_ids,
        block_size=512,
    )
    dense = search_result_to_ranking(query_ids, reference_ids, indices, scores)
    tfidf = CharTfidfModel.fit(
        train.items,
        config.hybrid.tfidf.ngram_range,
        config.hybrid.tfidf.max_features,
    )
    text = cross_text_ranking(tfidf, queries, references, source_k)
    phash = cross_phash_ranking(queries, references, min(source_k, 20))
    ranking = reciprocal_rank_fusion(
        {"dense": dense, "tfidf": text, "phash": phash},
        weights={
            "dense": config.hybrid.fusion.dense_weight,
            "tfidf": config.hybrid.fusion.tfidf_weight,
            "phash": config.hybrid.fusion.phash_weight,
        },
        rrf_constant=config.hybrid.fusion.rrf_constant,
        top_k=config.policy.candidate_k,
    )
    return ranking, references, queries


def retrieval_metrics(
    ranking: Ranking,
    queries: tuple[CorpusItem, ...],
    references: tuple[CorpusItem, ...],
    label_by_id: dict[str, str],
    k_values: tuple[int, ...],
) -> dict[str, Any]:
    reference_by_label = {label_by_id[item.posting_id]: item.posting_id for item in references}
    known = [item for item in queries if label_by_id[item.posting_id] in reference_by_label]
    metrics: dict[str, Any] = {"known_queries": len(known)}
    reciprocal_ranks: list[float] = []
    for query in known:
        target = reference_by_label[label_by_id[query.posting_id]]
        identifiers = [candidate.posting_id for candidate in ranking[query.posting_id]]
        reciprocal_ranks.append(
            1.0 / (identifiers.index(target) + 1) if target in identifiers else 0.0
        )
    metrics["mrr"] = float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0
    for k in k_values:
        hits = 0
        for query in known:
            target = reference_by_label[label_by_id[query.posting_id]]
            if target in {row.posting_id for row in ranking[query.posting_id][:k]}:
                hits += 1
        metrics[f"recall@{k}"] = hits / len(known) if known else 0.0
    return metrics


def validate_ranking_contract(
    ranking: Ranking,
    queries: tuple[CorpusItem, ...],
    references: tuple[CorpusItem, ...],
    candidate_k: int,
) -> dict[str, int]:
    query_ids = {item.posting_id for item in queries}
    reference_ids = {item.posting_id for item in references}
    if query_ids & reference_ids:
        raise DataValidationError("Catalog queries overlap reference listings")
    if set(ranking) != query_ids:
        raise DataValidationError("Catalog ranking does not cover exactly the query set")
    minimum = candidate_k
    maximum = 0
    for query_id, candidates in ranking.items():
        identifiers = [row.posting_id for row in candidates]
        if (
            len(identifiers) > candidate_k
            or len(set(identifiers)) != len(identifiers)
            or not set(identifiers) <= reference_ids
            or query_id in identifiers
        ):
            raise DataValidationError("Catalog ranking violates candidate isolation or budget")
        minimum = min(minimum, len(identifiers))
        maximum = max(maximum, len(identifiers))
    return {
        "query_reference_id_overlap": 0,
        "candidate_outside_reference_count": 0,
        "self_candidate_count": 0,
        "duplicate_candidate_count": 0,
        "minimum_candidates_per_query": minimum,
        "maximum_candidates_per_query": maximum,
    }


def _score_pairs(
    model: LearnedMultimodalFusion,
    posting_ids: tuple[str, ...],
    embeddings: FloatArray,
    ranking: Ranking,
    item_by_id: dict[str, CorpusItem],
    device: torch.device,
    batch_size: int,
) -> dict[str, tuple[PairEvidence, ...]]:
    index_by_id = {identifier: index for index, identifier in enumerate(posting_ids)}
    flat = [
        (query_id, candidate.posting_id, rank)
        for query_id, candidates in sorted(ranking.items())
        for rank, candidate in enumerate(candidates, start=1)
    ]
    tensor = torch.from_numpy(embeddings.astype(np.float32, copy=False))
    probabilities: list[float] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(flat), batch_size):
            chunk = flat[start : start + batch_size]
            left = tensor[[index_by_id[row[0]] for row in chunk]].to(device)
            right = tensor[[index_by_id[row[1]] for row in chunk]].to(device)
            probabilities.extend(torch.sigmoid(model.pair_logits(left, right)).cpu().tolist())
    evidence: dict[str, list[PairEvidence]] = defaultdict(list)
    for (query_id, candidate_id, rank), probability in zip(flat, probabilities, strict=True):
        evidence[query_id].append(
            PairEvidence(
                query_id,
                candidate_id,
                rank,
                float(probability),
                has_variant_conflict(item_by_id[query_id].title, item_by_id[candidate_id].title),
            )
        )
    return {
        query_id: tuple(
            sorted(
                rows, key=lambda row: (-row.pair_probability, row.retrieval_rank, row.candidate_id)
            )
        )
        for query_id, rows in evidence.items()
    }


def attachment_metrics(
    evidence: dict[str, tuple[PairEvidence, ...]],
    roles: tuple[CatalogRole, ...],
    label_by_id: dict[str, str],
    *,
    threshold: float,
    manual_review_margin: float,
    target_margin: float,
    variant_conflict_override_probability: float,
) -> dict[str, float]:
    role_by_id = {row.posting_id: row.role for row in roles}
    query_ids = sorted(
        identifier for identifier, role in role_by_id.items() if role != "catalog_reference"
    )
    known_ids = {
        identifier for identifier in query_ids if role_by_id[identifier] == "known_entity_query"
    }
    new_ids = set(query_ids) - known_ids
    auto_attachments = correct_attachments = false_attachments = 0
    new_correct = new_false_attach = manual_reviews = 0
    for query_id in query_ids:
        rows = [
            row
            for row in evidence[query_id]
            if not row.variant_conflict
            or row.pair_probability >= variant_conflict_override_probability
        ]
        top = rows[0] if rows else None
        second = rows[1] if len(rows) > 1 else None
        near_boundary = (
            top is not None and abs(top.pair_probability - threshold) <= manual_review_margin
        )
        ambiguous = (
            top is not None
            and second is not None
            and top.pair_probability >= threshold
            and second.pair_probability >= threshold
            and top.pair_probability - second.pair_probability < target_margin
        )
        if near_boundary or ambiguous:
            manual_reviews += 1
            continue
        if top is None or top.pair_probability < threshold:
            if query_id in new_ids:
                new_correct += 1
            continue
        auto_attachments += 1
        if query_id in known_ids and label_by_id[top.candidate_id] == label_by_id[query_id]:
            correct_attachments += 1
        else:
            false_attachments += 1
            if query_id in new_ids:
                new_false_attach += 1
    precision = correct_attachments / auto_attachments if auto_attachments else 0.0
    recall = correct_attachments / len(known_ids) if known_ids else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    new_recall = new_correct / len(new_ids) if new_ids else 0.0
    balanced = (f1 + new_recall) / 2
    total = len(query_ids)
    return {
        "threshold": threshold,
        "known_queries": float(len(known_ids)),
        "new_entity_queries": float(len(new_ids)),
        "auto_attachments": float(auto_attachments),
        "correct_attachments": float(correct_attachments),
        "false_attachments": float(false_attachments),
        "attachment_precision": precision,
        "attachment_recall": recall,
        "attachment_f1": f1,
        "new_entity_detection_recall": new_recall,
        "new_entity_false_attachment_rate": new_false_attach / len(new_ids) if new_ids else 0.0,
        "overall_false_attachment_rate": false_attachments / total if total else 0.0,
        "manual_review_rate": manual_reviews / total if total else 0.0,
        "automatic_coverage": 1.0 - manual_reviews / total if total else 0.0,
        "balanced_score": balanced,
    }


def passes_safety(metrics: dict[str, float], safety: CatalogAttachmentSafety) -> bool:
    return (
        metrics["attachment_precision"] >= safety.minimum_attachment_precision
        and metrics["new_entity_detection_recall"] >= safety.minimum_new_entity_detection_recall
        and metrics["new_entity_false_attachment_rate"]
        <= safety.maximum_new_entity_false_attachment_rate
        and metrics["overall_false_attachment_rate"] <= safety.maximum_overall_false_attachment_rate
        and metrics["manual_review_rate"] <= safety.maximum_manual_review_rate
    )


def passes_comparison(
    metrics: dict[str, float], comparison: CatalogAttachmentComparison | None
) -> bool:
    if comparison is None:
        return True
    baseline = comparison.baseline_metrics["selection"]["selected"]
    return bool(
        metrics["attachment_recall"] - baseline["attachment_recall"]
        >= comparison.minimum_attachment_recall_delta
        and metrics["attachment_f1"] - baseline["attachment_f1"]
        >= comparison.minimum_attachment_f1_delta
        and baseline["attachment_precision"] - metrics["attachment_precision"]
        <= comparison.maximum_attachment_precision_drop
        and baseline["new_entity_detection_recall"] - metrics["new_entity_detection_recall"]
        <= comparison.maximum_new_entity_detection_recall_drop
        and metrics["new_entity_false_attachment_rate"]
        - baseline["new_entity_false_attachment_rate"]
        <= comparison.maximum_new_entity_false_attachment_rate_increase
        and metrics["overall_false_attachment_rate"] - baseline["overall_false_attachment_rate"]
        <= comparison.maximum_overall_false_attachment_rate_increase
        and metrics["manual_review_rate"] - baseline["manual_review_rate"]
        <= comparison.maximum_manual_review_rate_increase
    )


def comparison_delta(
    selected: dict[str, float], comparison: CatalogAttachmentComparison | None
) -> dict[str, float] | None:
    if comparison is None:
        return None
    baseline = comparison.baseline_metrics["selection"]["selected"]
    names = (
        "attachment_precision",
        "attachment_recall",
        "attachment_f1",
        "new_entity_detection_recall",
        "new_entity_false_attachment_rate",
        "overall_false_attachment_rate",
        "manual_review_rate",
    )
    return {name: selected[name] - float(baseline[name]) for name in names}


def _report(run: dict[str, Any]) -> str:
    selected = run["selection"]["selected"]
    retrieval = run["retrieval"]
    rows = "\n".join(
        f"| Recall@{k} | {retrieval[f'recall@{k}']:.5f} |"
        for k in run["evaluation"]["metric_k_values"]
    )
    return f"""# Catalog Attachment Development Evaluation

Status: **{run["status"]}**. Model: **{run["model_variant"]}**.

The protocol evaluates static attachment to an existing catalog, explicit new-entity detection,
and manual-review behavior. Internal confirmation was not accessed.

## Retrieval

| Metric | Development |
|---|---:|
{rows}
| MRR | {retrieval["mrr"]:.5f} |

## Selected operating point

| Metric | Value |
|---|---:|
| Pair threshold | {selected["threshold"]:.2f} |
| Attachment precision | {selected["attachment_precision"]:.5f} |
| Attachment recall | {selected["attachment_recall"]:.5f} |
| Attachment F1 | {selected["attachment_f1"]:.5f} |
| New-entity detection recall | {selected["new_entity_detection_recall"]:.5f} |
| New-entity false-attachment rate | {selected["new_entity_false_attachment_rate"]:.5f} |
| Overall false-attachment rate | {selected["overall_false_attachment_rate"]:.5f} |
| Manual-review rate | {selected["manual_review_rate"]:.5f} |

This is development-only evidence. Thresholds and gates must be frozen before confirmation.
"""


def run_catalog_attachment_evaluation(config_path: Path) -> dict[str, object]:
    config = load_catalog_attachment_config(config_path)
    existing = [
        str(path) for path in (config.artifacts.metrics, config.artifacts.report) if path.exists()
    ]
    if existing:
        raise OutputConflictError(
            "Refusing to overwrite catalog-attachment outputs: " + ", ".join(existing)
        )
    commit, dirty = _git_state()
    if dirty:
        raise DataValidationError("Catalog-attachment evaluation requires a clean Git worktree")
    started = time.perf_counter()
    seed_everything(config.seed, deterministic=True)
    device = _resolve_device(config.runtime.device)
    train = load_named_split(config.metadata_csv, config.source_manifest, "train")
    development = load_named_split(config.metadata_csv, config.source_manifest, "validation")
    roles = load_catalog_roles(config.role_manifest, partition="development")
    validate_catalog_roles(development, roles)
    (
        image_model,
        text_model,
        fusion_model,
        vocabulary,
        maximum_length,
        image_dir,
        image_size,
    ) = _load_models(config, device)
    posting_ids, embeddings = _extract_embeddings(
        development,
        image_dir=image_dir,
        image_size=image_size,
        image_model=image_model,
        text_model=text_model,
        fusion_model=fusion_model,
        vocabulary=vocabulary,
        maximum_length=maximum_length,
        device=device,
        batch_size=config.runtime.extraction_batch_size,
    )
    ranking, references, queries = _hybrid_ranking(
        config, train, development, posting_ids, embeddings, roles
    )
    ranking_audit = validate_ranking_contract(
        ranking, queries, references, config.policy.candidate_k
    )
    retrieval = retrieval_metrics(
        ranking,
        queries,
        references,
        development.label_by_id,
        config.policy.metric_k_values,
    )
    evidence = _score_pairs(
        fusion_model,
        posting_ids,
        embeddings,
        ranking,
        {item.posting_id: item for item in development.items},
        device,
        config.runtime.pair_batch_size,
    )
    trials = [
        attachment_metrics(
            evidence,
            roles,
            development.label_by_id,
            threshold=threshold,
            manual_review_margin=config.policy.manual_review_margin,
            target_margin=config.policy.target_margin,
            variant_conflict_override_probability=(
                config.policy.variant_conflict_override_probability
            ),
        )
        for threshold in config.policy.thresholds
    ]
    safe_trials = [trial for trial in trials if passes_safety(trial, config.safety)]
    accepted_trials = [
        trial for trial in safe_trials if passes_comparison(trial, config.comparison)
    ]
    selected = max(
        accepted_trials or safe_trials or trials,
        key=lambda row: (
            row["balanced_score"],
            row["attachment_f1"],
            row["attachment_precision"],
            -row["threshold"],
        ),
    )
    if accepted_trials:
        status = "development_policy_selected"
    elif safe_trials:
        status = "development_candidate_rejected"
    else:
        status = "no_development_policy_passed"
    run: dict[str, Any] = {
        "pipeline_version": "catalog_attachment.evaluation.v4",
        "status": status,
        "model_variant": config.model_variant,
        "provenance": {
            "git_commit": commit,
            "git_dirty": False,
            "config_sha256": canonical_text_sha256(config.config_path),
            "role_manifest_sha256": sha256_file(config.role_manifest),
        },
        "data": {
            "train_listings_for_statistics": len(train.items),
            "development_listings": len(development.items),
            "catalog_references": len(references),
            "queries": len(queries),
            "confirmation_accessed": False,
            "historical_test_accessed": False,
            "label_group_used_by_retrieval": False,
            "train_development_label_overlap": len(
                set(train.label_by_id.values()) & set(development.label_by_id.values())
            ),
        },
        "evaluation": {
            "candidate_k": config.policy.candidate_k,
            "metric_k_values": config.policy.metric_k_values,
            "objective": "maximize_attachment_f1_and_new_entity_detection_subject_to_safety",
        },
        "retrieval": retrieval,
        "ranking_contract": ranking_audit,
        "safety": asdict(config.safety),
        "selection": {
            "passes_absolute_safety": bool(safe_trials),
            "passes_comparison_gate": bool(accepted_trials),
            "selected": selected,
            "trials": trials,
            "comparison_delta": comparison_delta(selected, config.comparison),
        },
        "runtime_seconds": time.perf_counter() - started,
    }
    _write_atomic(config.artifacts.metrics, json.dumps(run, indent=2, sort_keys=True) + "\n")
    _write_atomic(config.artifacts.report, _report(run))
    return {
        "status": status,
        "model_variant": config.model_variant,
        "metrics": str(config.artifacts.metrics),
        "selected": selected,
        "confirmation_accessed": False,
    }
