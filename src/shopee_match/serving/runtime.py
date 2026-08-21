"""Frozen end-to-end inference runtime for the local product-matching demo."""

from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import Lock
from typing import Any, cast

import cv2
import numpy as np
import torch
from numpy.typing import NDArray

from shopee_match.errors import ConfigurationError, ContractError, DataValidationError
from shopee_match.evaluation.protocol import CorpusItem
from shopee_match.retrieval.benchmark import load_phase6_model
from shopee_match.retrieval.vector_index import ExactCosineIndex, FaissHnswIndex
from shopee_match.serving.config import DemoConfig, DemoPolicyConfig, load_demo_config
from shopee_match.training.hard_negative_data import has_variant_conflict
from shopee_match.training.image_data import ImagePreprocessor
from shopee_match.training.multimodal_data import load_frozen_encoders

FloatArray = NDArray[np.float32]
UInt8Image = NDArray[np.uint8]


@dataclass(frozen=True, slots=True)
class CandidateEvidence:
    posting_id: str
    title: str
    rank: int
    image_similarity: float | None
    title_similarity: float | None
    joint_similarity: float | None
    match_probability: float | None
    reciprocal_rank: int | None
    variant_conflict: bool | None
    accepted_match: bool | None
    decision_reason: str
    entity_id: str
    cluster_size: int
    cluster_confidence: float
    cluster_manual_review: bool


@dataclass(frozen=True, slots=True)
class MatchPrediction:
    status: str
    query_mode: str
    predicted_entity_id: str | None
    confident_match: bool
    manual_review: bool
    decision_summary: str
    device: str
    index_backend: str
    candidate_k: int
    returned_k: int
    latency_ms: float
    candidates: tuple[CandidateEvidence, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class _EntityAssignment:
    entity_id: str
    cluster_size: int
    cluster_confidence: float
    manual_review: bool


def _resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise ConfigurationError("runtime.device=cuda but CUDA is not available")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _decode_rgb(image_bytes: bytes) -> UInt8Image:
    encoded = np.frombuffer(image_bytes, dtype=np.uint8)
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if decoded is None:
        raise ContractError("Uploaded file is not a decodable image")
    return cast(UInt8Image, cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB))


def _load_embedding_arrays(
    config: DemoConfig,
) -> tuple[tuple[str, ...], FloatArray, FloatArray, FloatArray]:
    try:
        with np.load(config.source.modality_embeddings_path, allow_pickle=False) as payload:
            modality_ids = tuple(str(value) for value in payload["posting_ids"].tolist())
            image = payload["image_embeddings"].astype(np.float32, copy=True)
            text = payload["text_embeddings"].astype(np.float32, copy=True)
        joint_path = config.source.experiment.source.embedding_cache_path
        with np.load(joint_path, allow_pickle=False) as payload:
            joint_ids = tuple(str(value) for value in payload["posting_ids"].tolist())
            joint = payload["embeddings"].astype(np.float32, copy=True)
    except (OSError, KeyError, ValueError) as exc:
        raise DataValidationError("Cannot load aligned demo catalog embeddings") from exc
    if modality_ids != joint_ids or image.shape[0] != text.shape[0] or len(joint) != len(joint_ids):
        raise DataValidationError("Demo modality and joint embedding caches are not aligned")
    arrays = (image, text, joint)
    if any(array.ndim != 2 or not np.isfinite(array).all() for array in arrays):
        raise DataValidationError("Demo embedding cache contains invalid arrays")
    if any(not np.allclose(np.linalg.norm(array, axis=1), 1.0, atol=1e-5) for array in arrays):
        raise DataValidationError("Demo embeddings must be L2-normalized")
    return modality_ids, image, text, joint


def _load_assignments(path: Path) -> dict[str, _EntityAssignment]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except OSError as exc:
        raise DataValidationError(f"Cannot read demo entity assignments: {path}") from exc
    assignments: dict[str, _EntityAssignment] = {}
    try:
        for row in rows:
            posting_id = row["posting_id"]
            if posting_id in assignments:
                raise DataValidationError(f"Duplicate entity assignment: {posting_id}")
            assignments[posting_id] = _EntityAssignment(
                entity_id=row["entity_id"],
                cluster_size=int(row["cluster_size"]),
                cluster_confidence=float(row["cluster_confidence"]),
                manual_review=row["manual_review"].strip().lower() == "true",
            )
    except (KeyError, TypeError, ValueError) as exc:
        raise DataValidationError("Demo entity assignment schema is invalid") from exc
    return assignments


def _load_label_blind_catalog(
    metadata_csv: Path, manifest_path: Path, split_name: str
) -> tuple[CorpusItem, ...]:
    selected_ids: set[str] = set()
    try:
        with manifest_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                record = json.loads(line)
                if record.get("split") == split_name:
                    posting_id = str(record["posting_id"])
                    if posting_id in selected_ids:
                        raise DataValidationError(
                            f"Duplicate demo manifest posting_id at line {line_number}"
                        )
                    selected_ids.add(posting_id)
    except (OSError, AttributeError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise DataValidationError("Cannot read the frozen demo split manifest") from exc
    if not selected_ids:
        raise DataValidationError(f"Demo catalog split is empty: {split_name}")

    items: list[CorpusItem] = []
    seen: set[str] = set()
    try:
        with metadata_csv.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                posting_id = row["posting_id"]
                if posting_id not in selected_ids:
                    continue
                if posting_id in seen:
                    raise DataValidationError(f"Duplicate demo metadata posting_id: {posting_id}")
                seen.add(posting_id)
                items.append(
                    CorpusItem(
                        posting_id=posting_id,
                        image=row["image"],
                        image_phash=row["image_phash"],
                        title=row["title"],
                    )
                )
    except (OSError, KeyError) as exc:
        raise DataValidationError("Cannot read label-blind demo catalog fields") from exc
    if seen != selected_ids:
        raise DataValidationError("Demo metadata does not cover the selected catalog split")
    return tuple(sorted(items, key=lambda item: item.posting_id))


def _apply_pair_policy(
    policy: DemoPolicyConfig,
    probability: float,
    forward_rank: int,
    reverse_rank: int,
    variant_conflict: bool,
) -> tuple[bool, str]:
    if probability < policy.pair_probability_threshold:
        return False, "below_probability_threshold"
    if max(forward_rank, reverse_rank) > policy.reciprocal_rank:
        return False, "not_reciprocal_top_k"
    if variant_conflict and probability < policy.variant_conflict_override_probability:
        return False, "variant_conflict"
    return True, "accepted_by_frozen_policy"


class DemoRuntime:
    """Load models once and serve label-blind product-match predictions."""

    def __init__(self, config: DemoConfig) -> None:
        self.config = config
        self.device = _resolve_device(config.runtime.device)
        entity_config = config.source.experiment
        phase7 = entity_config.source.experiment
        multimodal = phase7.source.experiment.source.experiment
        catalog_items = _load_label_blind_catalog(
            multimodal.data.metadata_csv, multimodal.data.split_manifest, "validation"
        )
        self.posting_ids, self.image_embeddings, self.text_embeddings, self.joint_embeddings = (
            _load_embedding_arrays(config)
        )
        expected_ids = tuple(item.posting_id for item in catalog_items)
        if self.posting_ids != expected_ids:
            raise DataValidationError("Demo catalog does not align with the validation manifest")
        self.item_by_id = {item.posting_id: item for item in catalog_items}
        self.assignment_by_id = _load_assignments(config.source.entity_assignments_path)
        if set(self.assignment_by_id) != set(self.posting_ids):
            raise DataValidationError("Demo entity assignments do not cover the catalog exactly")

        self.image_model, self.text_model, self.vocabulary, self.maximum_title_length = (
            load_frozen_encoders(multimodal, self.device)
        )
        self.fusion_model = load_phase6_model(phase7, self.device)
        self.image_preprocessor = ImagePreprocessor(
            multimodal.frozen.image_config.training_experiment.image_size,
            training=False,
            seed=phase7.seed,
        )
        if config.index.backend == "faiss_hnsw":
            self.index: ExactCosineIndex | FaissHnswIndex = FaissHnswIndex(
                self.posting_ids,
                self.joint_embeddings,
                m=config.index.m,
                ef_construction=config.index.ef_construction,
                ef_search=config.index.ef_search,
                threads=config.index.threads,
                rerank_buffer=config.index.rerank_buffer,
            )
        else:
            self.index = ExactCosineIndex(self.posting_ids, self.joint_embeddings)
        self.image_index = ExactCosineIndex(self.posting_ids, self.image_embeddings)
        self.text_index = ExactCosineIndex(self.posting_ids, self.text_embeddings)
        self._inference_lock = Lock()

    @classmethod
    def load(cls, config_path: Path) -> DemoRuntime:
        return cls(load_demo_config(config_path))

    @property
    def catalog_size(self) -> int:
        return len(self.posting_ids)

    @property
    def index_backend(self) -> str:
        return self.index.backend

    def health(self) -> dict[str, Any]:
        return {
            "status": "ready",
            "device": str(self.device),
            "index_backend": self.index_backend,
            "catalog_split": "validation",
            "catalog_size": self.catalog_size,
            "candidate_k": self.config.policy.candidate_k,
            "pair_probability_threshold": self.config.policy.pair_probability_threshold,
            "supported_query_modes": ["multimodal", "image_only", "text_only"],
            "ground_truth_used_for_inference": False,
        }

    def catalog_image_path(self, posting_id: str) -> Path:
        item = self.item_by_id.get(posting_id)
        if item is None:
            raise ContractError(f"Unknown catalog posting_id: {posting_id}")
        phase7 = self.config.source.experiment.source.experiment
        multimodal = phase7.source.experiment.source.experiment
        path = multimodal.data.image_dir / item.image
        if not path.is_file():
            raise DataValidationError(f"Catalog image is unavailable: {path}")
        return path

    def _encode_image(self, image_bytes: bytes) -> FloatArray:
        if not image_bytes:
            raise ContractError("An image is required")
        if len(image_bytes) > self.config.runtime.maximum_upload_bytes:
            raise ContractError("Uploaded image exceeds the configured size limit")
        image = self.image_preprocessor(_decode_rgb(image_bytes), 0).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            embedding = self.image_model(image)
        return cast(FloatArray, embedding.cpu().numpy()).astype(np.float32, copy=False)

    def _encode_text(self, title: str) -> FloatArray:
        normalized_title = title.strip()
        if not normalized_title:
            raise ContractError("A non-empty product title is required")
        if len(normalized_title) > 500:
            raise ContractError("Product title must contain at most 500 characters")
        token_ids, length = self.vocabulary.encode(normalized_title, self.maximum_title_length)
        with torch.inference_mode():
            embedding = self.text_model(
                token_ids.unsqueeze(0).to(self.device), length.unsqueeze(0).to(self.device)
            )
        return cast(FloatArray, embedding.cpu().numpy()).astype(np.float32, copy=False)

    def _encode_multimodal(
        self, image_bytes: bytes, title: str
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        image_embedding = self._encode_image(image_bytes)
        text_embedding = self._encode_text(title)
        with torch.inference_mode():
            joint_embedding = self.fusion_model(
                torch.from_numpy(image_embedding).to(self.device),
                torch.from_numpy(text_embedding).to(self.device),
            )
        return (
            image_embedding,
            text_embedding,
            cast(FloatArray, joint_embedding.cpu().numpy()).astype(np.float32, copy=False),
        )

    def _reverse_rank(self, candidate_index: int, query_embedding: FloatArray) -> int:
        query_score = float(self.joint_embeddings[candidate_index] @ query_embedding[0])
        corpus_scores = self.joint_embeddings @ self.joint_embeddings[candidate_index]
        corpus_scores[candidate_index] = -np.inf
        return 1 + int(np.count_nonzero(corpus_scores > query_score))

    def _decision(
        self, probability: float, forward_rank: int, reverse_rank: int, variant_conflict: bool
    ) -> tuple[bool, str]:
        return _apply_pair_policy(
            self.config.policy,
            probability,
            forward_rank,
            reverse_rank,
            variant_conflict,
        )

    def _unimodal_prediction(
        self,
        mode: str,
        query_embedding: FloatArray,
        requested_k: int,
        started: float,
    ) -> MatchPrediction:
        if mode == "image_only":
            index = self.image_index
        elif mode == "text_only":
            index = self.text_index
        else:  # pragma: no cover - internal invariant
            raise AssertionError("unsupported unimodal query mode")
        indices, scores = index.search(
            query_embedding, self.config.policy.candidate_k, block_size=512
        )
        candidates: list[CandidateEvidence] = []
        for offset, candidate_index in enumerate(indices[0, :requested_k].tolist()):
            posting_id = self.posting_ids[candidate_index]
            item = self.item_by_id[posting_id]
            assignment = self.assignment_by_id[posting_id]
            similarity = float(scores[0, offset])
            candidates.append(
                CandidateEvidence(
                    posting_id=posting_id,
                    title=item.title,
                    rank=offset + 1,
                    image_similarity=similarity if mode == "image_only" else None,
                    title_similarity=similarity if mode == "text_only" else None,
                    joint_similarity=None,
                    match_probability=None,
                    reciprocal_rank=None,
                    variant_conflict=None,
                    accepted_match=None,
                    decision_reason=f"{mode}_retrieval_only",
                    entity_id=assignment.entity_id,
                    cluster_size=assignment.cluster_size,
                    cluster_confidence=assignment.cluster_confidence,
                    cluster_manual_review=assignment.manual_review,
                )
            )
        modality = "image" if mode == "image_only" else "title"
        return MatchPrediction(
            status="retrieval_only",
            query_mode=mode,
            predicted_entity_id=None,
            confident_match=False,
            manual_review=False,
            decision_summary=(
                f"Candidates are ranked by {modality} similarity only. Add the missing modality "
                "to enable calibrated pair scoring and entity assignment."
            ),
            device=str(self.device),
            index_backend=index.backend,
            candidate_k=self.config.policy.candidate_k,
            returned_k=requested_k,
            latency_ms=(time.perf_counter() - started) * 1000,
            candidates=tuple(candidates),
        )

    def match(
        self,
        image_bytes: bytes | None,
        title: str | None,
        top_k: int | None = None,
    ) -> MatchPrediction:
        requested_k = top_k or self.config.policy.default_top_k
        if not 1 <= requested_k <= self.config.policy.maximum_top_k:
            raise ContractError(f"top_k must be between 1 and {self.config.policy.maximum_top_k}")
        has_image = bool(image_bytes)
        normalized_title = title.strip() if title is not None else ""
        has_title = bool(normalized_title)
        if not has_image and not has_title:
            raise ContractError("Provide a product image, a product title, or both")
        started = time.perf_counter()
        with self._inference_lock:
            if has_image and not has_title:
                assert image_bytes is not None
                return self._unimodal_prediction(
                    "image_only", self._encode_image(image_bytes), requested_k, started
                )
            if has_title and not has_image:
                return self._unimodal_prediction(
                    "text_only", self._encode_text(normalized_title), requested_k, started
                )
            assert image_bytes is not None
            image, text, joint = self._encode_multimodal(image_bytes, normalized_title)
            if isinstance(self.index, ExactCosineIndex):
                indices, scores = self.index.search(
                    joint, self.config.policy.candidate_k, block_size=512
                )
            else:
                indices, scores = self.index.search(joint, self.config.policy.candidate_k)
            selected_indices = indices[0]
            candidate_joint = torch.from_numpy(self.joint_embeddings[selected_indices]).to(
                self.device
            )
            query_joint = torch.from_numpy(joint).to(self.device).expand_as(candidate_joint)
            with torch.inference_mode():
                probabilities = (
                    torch.sigmoid(self.fusion_model.pair_logits(query_joint, candidate_joint))
                    .cpu()
                    .numpy()
                )

        candidates: list[CandidateEvidence] = []
        accepted_entity_ids: list[str] = []
        near_boundary = False
        accepted_cluster_review = False
        for offset, candidate_index in enumerate(selected_indices.tolist()):
            rank = offset + 1
            posting_id = self.posting_ids[candidate_index]
            item: CorpusItem = self.item_by_id[posting_id]
            assignment = self.assignment_by_id[posting_id]
            probability = float(probabilities[offset])
            reverse_rank = self._reverse_rank(candidate_index, joint)
            variant_conflict = has_variant_conflict(normalized_title, item.title)
            accepted, reason = self._decision(probability, rank, reverse_rank, variant_conflict)
            if (
                max(rank, reverse_rank) <= self.config.policy.reciprocal_rank
                and abs(probability - self.config.policy.pair_probability_threshold)
                <= self.config.policy.manual_review_margin
            ):
                near_boundary = True
            if accepted:
                accepted_entity_ids.append(assignment.entity_id)
                accepted_cluster_review = accepted_cluster_review or assignment.manual_review
            if rank <= requested_k:
                candidates.append(
                    CandidateEvidence(
                        posting_id=posting_id,
                        title=item.title,
                        rank=rank,
                        image_similarity=float(image[0] @ self.image_embeddings[candidate_index]),
                        title_similarity=float(text[0] @ self.text_embeddings[candidate_index]),
                        joint_similarity=float(scores[0, offset]),
                        match_probability=probability,
                        reciprocal_rank=reverse_rank,
                        variant_conflict=variant_conflict,
                        accepted_match=accepted,
                        decision_reason=reason,
                        entity_id=assignment.entity_id,
                        cluster_size=assignment.cluster_size,
                        cluster_confidence=assignment.cluster_confidence,
                        cluster_manual_review=assignment.manual_review,
                    )
                )

        unique_entities = tuple(dict.fromkeys(accepted_entity_ids))
        confident = bool(unique_entities)
        manual_review = len(unique_entities) > 1 or near_boundary or accepted_cluster_review
        if not confident:
            summary = "No candidate passed the frozen match policy"
        elif len(unique_entities) > 1:
            summary = "Accepted candidates disagree on entity; manual review is required"
        else:
            summary = "At least one reciprocal candidate passed the frozen match policy"
        return MatchPrediction(
            status="complete",
            query_mode="multimodal",
            predicted_entity_id=unique_entities[0] if unique_entities else None,
            confident_match=confident,
            manual_review=manual_review,
            decision_summary=summary,
            device=str(self.device),
            index_backend=self.index_backend,
            candidate_k=self.config.policy.candidate_k,
            returned_k=requested_k,
            latency_ms=(time.perf_counter() - started) * 1000,
            candidates=tuple(candidates),
        )
