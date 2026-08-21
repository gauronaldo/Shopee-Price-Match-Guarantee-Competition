"""Exact cosine and FAISS HNSW indexes behind one deterministic search contract."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from shopee_match.errors import DataValidationError
from shopee_match.evaluation.protocol import Ranking, ScoredCandidate

FloatArray = NDArray[np.float32]
IntArray = NDArray[np.int64]


def normalize_embeddings(values: FloatArray) -> FloatArray:
    """Return contiguous finite float32 unit vectors."""
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2 or not len(array) or not np.isfinite(array).all():
        raise ValueError("embeddings must be a non-empty finite matrix")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise ValueError("embeddings must have non-zero norm")
    return np.ascontiguousarray(array / norms, dtype=np.float32)


def _validate_identifiers(posting_ids: tuple[str, ...], rows: int) -> None:
    if len(posting_ids) != rows or len(set(posting_ids)) != rows:
        raise ValueError("posting_ids must be unique and align with embeddings")


def search_result_to_ranking(
    query_ids: tuple[str, ...],
    corpus_ids: tuple[str, ...],
    indices: IntArray,
    scores: FloatArray,
) -> Ranking:
    """Convert fixed-width search arrays into the shared ranking representation."""
    if indices.shape != scores.shape or indices.shape[0] != len(query_ids):
        raise ValueError("search arrays must align with query_ids")
    ranking: Ranking = {}
    for row, query_id in enumerate(query_ids):
        candidates = [
            ScoredCandidate(corpus_ids[int(index)], float(score))
            for index, score in zip(indices[row], scores[row], strict=True)
        ]
        expected = sorted(candidates, key=lambda item: (-item.score, item.posting_id))
        if candidates != expected:
            raise DataValidationError("Index output is not deterministically sorted")
        ranking[query_id] = candidates
    return ranking


class ExactCosineIndex:
    """Blockwise exact inner-product search over normalized listing embeddings."""

    backend = "exact_cosine"

    def __init__(self, posting_ids: tuple[str, ...], embeddings: FloatArray) -> None:
        self.vectors = normalize_embeddings(embeddings)
        _validate_identifiers(posting_ids, len(self.vectors))
        self.posting_ids = posting_ids
        self.index_by_id = {posting_id: index for index, posting_id in enumerate(posting_ids)}
        self._id_array = np.asarray(posting_ids, dtype=str)

    @property
    def dimension(self) -> int:
        return int(self.vectors.shape[1])

    @property
    def estimated_memory_bytes(self) -> int:
        identifier_bytes = sum(len(value.encode("utf-8")) for value in self.posting_ids)
        return int(self.vectors.nbytes + identifier_bytes)

    def search(
        self,
        queries: FloatArray,
        candidate_k: int,
        *,
        query_ids: tuple[str, ...] | None = None,
        block_size: int = 512,
    ) -> tuple[IntArray, FloatArray]:
        """Search exactly, excluding a matching corpus ID when query_ids are supplied."""
        normalized = normalize_embeddings(queries)
        if normalized.shape[1] != self.dimension:
            raise ValueError("query embedding dimension differs from the index")
        if candidate_k <= 0 or block_size <= 0:
            raise ValueError("candidate_k and block_size must be positive")
        if query_ids is not None and len(query_ids) != len(normalized):
            raise ValueError("query_ids must align with query embeddings")
        available = len(self.posting_ids) - (1 if query_ids is not None else 0)
        if candidate_k > available:
            raise ValueError("candidate_k exceeds the searchable corpus")
        all_indices = np.empty((len(normalized), candidate_k), dtype=np.int64)
        all_scores = np.empty((len(normalized), candidate_k), dtype=np.float32)
        for start in range(0, len(normalized), block_size):
            stop = min(start + block_size, len(normalized))
            similarities = normalized[start:stop] @ self.vectors.T
            for local_row, query_row in enumerate(range(start, stop)):
                row = similarities[local_row]
                if query_ids is not None:
                    own_index = self.index_by_id.get(query_ids[query_row])
                    if own_index is not None:
                        row[own_index] = -np.inf
                order = np.lexsort((self._id_array, -row))[:candidate_k]
                all_indices[query_row] = order
                all_scores[query_row] = row[order]
        return all_indices, all_scores

    def save(self, path: Path) -> None:
        """Serialize exact vectors and corpus IDs atomically."""
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                posting_ids=np.asarray(self.posting_ids, dtype=str),
                embeddings=self.vectors,
                backend=np.asarray([self.backend], dtype=str),
            )
        temporary.replace(path)

    @classmethod
    def load(cls, path: Path) -> ExactCosineIndex:
        """Load a serialized exact index and revalidate its contents."""
        try:
            with np.load(path, allow_pickle=False) as payload:
                backend = str(payload["backend"][0])
                posting_ids = tuple(str(value) for value in payload["posting_ids"].tolist())
                embeddings = payload["embeddings"].astype(np.float32, copy=True)
        except (OSError, KeyError, ValueError) as exc:
            raise DataValidationError(f"Cannot load exact index: {path}") from exc
        if backend != cls.backend:
            raise DataValidationError("Serialized exact index backend is unsupported")
        if (
            embeddings.ndim != 2
            or not len(embeddings)
            or not np.isfinite(embeddings).all()
            or not np.allclose(np.linalg.norm(embeddings, axis=1), 1.0, rtol=1e-5, atol=1e-6)
        ):
            raise DataValidationError("Serialized exact index vectors are not normalized")

        # Preserve the serialized float32 vectors byte-for-byte. Calling __init__
        # would normalize them a second time; a one-ULP change can reorder an exact
        # tie at the Top-K boundary and violate the index round-trip contract.
        instance = cls.__new__(cls)
        instance.vectors = np.ascontiguousarray(embeddings, dtype=np.float32)
        _validate_identifiers(posting_ids, len(instance.vectors))
        instance.posting_ids = posting_ids
        instance.index_by_id = {posting_id: index for index, posting_id in enumerate(posting_ids)}
        instance._id_array = np.asarray(posting_ids, dtype=str)
        return instance


class FaissHnswIndex:
    """CPU FAISS HNSW index using cosine-equivalent normalized inner product."""

    backend = "faiss_hnsw_flat"

    def __init__(
        self,
        posting_ids: tuple[str, ...],
        embeddings: FloatArray,
        *,
        m: int,
        ef_construction: int,
        ef_search: int,
        threads: int = 1,
        rerank_buffer: int = 32,
    ) -> None:
        vectors = normalize_embeddings(embeddings)
        _validate_identifiers(posting_ids, len(vectors))
        if min(m, ef_construction, ef_search, threads) <= 0 or rerank_buffer < 0:
            raise ValueError("FAISS HNSW parameters must be positive")
        faiss = self._faiss()
        faiss.omp_set_num_threads(threads)
        index = faiss.IndexHNSWFlat(vectors.shape[1], m, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = ef_construction
        index.hnsw.efSearch = ef_search
        index.add(vectors)
        self._index = index
        self.posting_ids = posting_ids
        self.index_by_id = {posting_id: index for index, posting_id in enumerate(posting_ids)}
        self.dimension = int(vectors.shape[1])
        self.m = m
        self.ef_construction = ef_construction
        self.threads = threads
        self.rerank_buffer = rerank_buffer

    @staticmethod
    def _faiss() -> Any:
        try:
            return importlib.import_module("faiss")
        except ImportError as exc:
            raise DataValidationError(
                "FAISS is unavailable; install the project retrieval extra"
            ) from exc

    @property
    def ef_search(self) -> int:
        return int(self._index.hnsw.efSearch)

    @ef_search.setter
    def ef_search(self, value: int) -> None:
        if value <= 0:
            raise ValueError("ef_search must be positive")
        self._index.hnsw.efSearch = value

    def search(
        self,
        queries: FloatArray,
        candidate_k: int,
        *,
        query_ids: tuple[str, ...] | None = None,
    ) -> tuple[IntArray, FloatArray]:
        """Search HNSW and deterministically sort returned candidates by score then ID."""
        normalized = normalize_embeddings(queries)
        if normalized.shape[1] != self.dimension:
            raise ValueError("query embedding dimension differs from the index")
        if candidate_k <= 0 or (query_ids is not None and len(query_ids) != len(normalized)):
            raise ValueError("invalid FAISS search arguments")
        extra = 1 if query_ids is not None else 0
        requested = min(len(self.posting_ids), candidate_k + extra + self.rerank_buffer)
        if candidate_k > len(self.posting_ids) - extra:
            raise ValueError("candidate_k exceeds the searchable corpus")
        faiss = self._faiss()
        faiss.omp_set_num_threads(self.threads)
        raw_scores, raw_indices = self._index.search(normalized, requested)
        output_indices = np.empty((len(normalized), candidate_k), dtype=np.int64)
        output_scores = np.empty((len(normalized), candidate_k), dtype=np.float32)
        for row in range(len(normalized)):
            own_id = query_ids[row] if query_ids is not None else None
            candidates = [
                (int(index), float(score))
                for index, score in zip(raw_indices[row], raw_scores[row], strict=True)
                if int(index) >= 0 and self.posting_ids[int(index)] != own_id
            ]
            candidates.sort(key=lambda item: (-item[1], self.posting_ids[item[0]]))
            if len(candidates) < candidate_k:
                raise DataValidationError("FAISS returned fewer candidates than requested")
            selected = candidates[:candidate_k]
            output_indices[row] = [index for index, _score in selected]
            output_scores[row] = [score for _index, score in selected]
        return output_indices, output_scores

    @property
    def estimated_memory_bytes(self) -> int:
        serialized = self._faiss().serialize_index(self._index)
        identifier_bytes = sum(len(value.encode("utf-8")) for value in self.posting_ids)
        return int(serialized.nbytes + identifier_bytes)

    def save(self, index_path: Path, metadata_path: Path) -> None:
        """Persist the FAISS graph and ID/config metadata."""
        index_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_index = index_path.with_name(f".{index_path.name}.tmp")
        temporary_metadata = metadata_path.with_name(f".{metadata_path.name}.tmp")
        faiss = self._faiss()
        faiss.write_index(self._index, str(temporary_index))
        temporary_metadata.write_text(
            json.dumps(
                {
                    "backend": self.backend,
                    "posting_ids": self.posting_ids,
                    "m": self.m,
                    "ef_construction": self.ef_construction,
                    "ef_search": self.ef_search,
                    "threads": self.threads,
                    "rerank_buffer": self.rerank_buffer,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        temporary_index.replace(index_path)
        temporary_metadata.replace(metadata_path)

    @classmethod
    def load(cls, index_path: Path, metadata_path: Path) -> FaissHnswIndex:
        """Load a persisted HNSW graph without rebuilding it."""
        try:
            metadata = cast(dict[str, Any], json.loads(metadata_path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError) as exc:
            raise DataValidationError("Cannot load FAISS index metadata") from exc
        if metadata.get("backend") != cls.backend:
            raise DataValidationError("Serialized FAISS backend is unsupported")
        faiss = cls._faiss()
        instance = cls.__new__(cls)
        instance._index = faiss.read_index(str(index_path))
        instance.posting_ids = tuple(str(value) for value in metadata["posting_ids"])
        instance.index_by_id = {
            posting_id: index for index, posting_id in enumerate(instance.posting_ids)
        }
        instance.dimension = int(instance._index.d)
        instance.m = int(metadata["m"])
        instance.ef_construction = int(metadata["ef_construction"])
        instance.threads = int(metadata["threads"])
        instance.rerank_buffer = int(metadata["rerank_buffer"])
        instance.ef_search = int(metadata["ef_search"])
        _validate_identifiers(instance.posting_ids, int(instance._index.ntotal))
        return instance
