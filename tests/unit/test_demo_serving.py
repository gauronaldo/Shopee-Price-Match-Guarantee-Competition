from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

from fastapi.testclient import TestClient

from shopee_match.serving.api import create_app
from shopee_match.serving.cli import _parser
from shopee_match.serving.config import DemoPolicyConfig
from shopee_match.serving.runtime import (
    CandidateEvidence,
    DemoRuntime,
    MatchPrediction,
    _apply_pair_policy,
    _load_label_blind_catalog,
)


class _FakeRuntime:
    def __init__(self, image_path: Path) -> None:
        self.config = SimpleNamespace(
            runtime=SimpleNamespace(maximum_upload_bytes=1024, maximum_batch_size=2)
        )
        self.image_path = image_path

    def health(self) -> dict[str, Any]:
        return {
            "status": "ready",
            "device": "cpu",
            "index_backend": "exact_cosine",
            "catalog_split": "validation",
            "catalog_size": 3,
            "candidate_k": 2,
            "pair_probability_threshold": 0.16,
            "supported_query_modes": ["multimodal", "image_only", "text_only"],
            "ground_truth_used_for_inference": False,
        }

    def match(
        self, image_bytes: bytes | None, title: str | None, top_k: int | None
    ) -> MatchPrediction:
        assert top_k == 1
        if image_bytes is not None and title:
            query_mode = "multimodal"
        elif image_bytes is not None:
            query_mode = "image_only"
        elif title:
            query_mode = "text_only"
        else:
            raise AssertionError("test request must contain at least one modality")
        multimodal = query_mode == "multimodal"
        candidate = CandidateEvidence(
            posting_id="candidate-1",
            title="sample title",
            rank=1,
            image_similarity=0.8 if query_mode != "text_only" else None,
            title_similarity=0.9 if query_mode != "image_only" else None,
            joint_similarity=0.85 if multimodal else None,
            match_probability=0.91 if multimodal else None,
            reciprocal_rank=1 if multimodal else None,
            variant_conflict=False if multimodal else None,
            accepted_match=True if multimodal else None,
            decision_reason=("accepted_by_frozen_policy" if multimodal else "retrieval_only"),
            entity_id="entity-1",
            cluster_size=2,
            cluster_confidence=0.91,
            cluster_manual_review=False,
        )
        return MatchPrediction(
            status="complete" if multimodal else "retrieval_only",
            query_mode=query_mode,
            predicted_entity_id="entity-1" if multimodal else None,
            confident_match=multimodal,
            manual_review=False,
            decision_summary="accepted",
            device="cpu",
            index_backend="exact_cosine",
            candidate_k=2,
            returned_k=1,
            latency_ms=1.5,
            candidates=(candidate,),
        )

    def catalog_image_path(self, posting_id: str) -> Path:
        assert posting_id == "candidate-1"
        return self.image_path


def _client(tmp_path: Path) -> TestClient:
    image_path = tmp_path / "candidate.jpg"
    image_path.write_bytes(b"catalog-image")
    runtime = cast(DemoRuntime, _FakeRuntime(image_path))
    return TestClient(create_app(runtime))


def test_health_and_match_contract(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["ground_truth_used_for_inference"] is False

        response = client.post(
            "/api/v1/match",
            files={"image": ("query.jpg", b"image-bytes", "image/jpeg")},
            data={"title": "sample title", "top_k": "1"},
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["predicted_entity_id"] == "entity-1"
        assert payload["candidates"][0]["accepted_match"] is True


def test_single_endpoint_accepts_either_modality(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        text_response = client.post("/api/v1/match", data={"title": "sample title", "top_k": "1"})
        assert text_response.status_code == 200
        assert text_response.json()["query_mode"] == "text_only"
        assert text_response.json()["candidates"][0]["match_probability"] is None

        image_response = client.post(
            "/api/v1/match",
            files={"image": ("query.jpg", b"image-bytes", "image/jpeg")},
            data={"top_k": "1"},
        )
        assert image_response.status_code == 200
        assert image_response.json()["query_mode"] == "image_only"
        assert image_response.json()["candidates"][0]["match_probability"] is None


def test_batch_rejects_misaligned_inputs(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        response = client.post(
            "/api/v1/match/batch",
            files=[
                ("images", ("query.jpg", b"image-bytes", "image/jpeg")),
                ("titles", (None, "sample title")),
                ("titles", (None, "extra title")),
            ],
        )
        assert response.status_code == 400
        assert "equal length" in response.json()["detail"]


def test_catalog_image_endpoint(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        response = client.get("/api/v1/catalog/candidate-1/image")
        assert response.status_code == 200
        assert response.content == b"catalog-image"


def test_frozen_pair_policy_requires_probability_and_reciprocity() -> None:
    policy = DemoPolicyConfig(
        candidate_k=50,
        default_top_k=10,
        maximum_top_k=20,
        pair_probability_threshold=0.16,
        reciprocal_rank=5,
        variant_conflict_override_probability=0.15,
        manual_review_margin=0.02,
    )
    assert _apply_pair_policy(policy, 0.8, 1, 2, False) == (
        True,
        "accepted_by_frozen_policy",
    )
    assert _apply_pair_policy(policy, 0.15, 1, 2, False) == (
        False,
        "below_probability_threshold",
    )
    assert _apply_pair_policy(policy, 0.8, 1, 6, False) == (
        False,
        "not_reciprocal_top_k",
    )


def test_catalog_loader_exposes_no_ground_truth(tmp_path: Path) -> None:
    metadata = tmp_path / "metadata.csv"
    metadata.write_text(
        "posting_id,image,image_phash,title,label_group\n"
        "p1,one.jpg,hash-one,First product,secret-label\n"
        "p2,two.jpg,hash-two,Second product,another-label\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        '{"posting_id":"p1","split":"validation"}\n{"posting_id":"p2","split":"train"}\n',
        encoding="utf-8",
    )
    items = _load_label_blind_catalog(metadata, manifest, "validation")
    assert len(items) == 1
    assert items[0].posting_id == "p1"
    assert not hasattr(items[0], "label_group")


def test_launch_cli_has_one_command_defaults() -> None:
    arguments = _parser().parse_args(["launch"])
    assert arguments.command == "launch"
    assert arguments.api_port == 8000
    assert arguments.ui_port == 8501
