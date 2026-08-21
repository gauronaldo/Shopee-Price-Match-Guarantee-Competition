from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from shopee_match.errors import ConfigurationError
from shopee_match.evaluation.multimodal_failure_analysis import (
    build_multimodal_failure_review,
)
from shopee_match.evaluation.multimodal_retrieval import (
    rank_simple_score_fusion,
    select_simple_score_fusion,
)
from shopee_match.evaluation.protocol import (
    CorpusItem,
    EvaluationSplit,
    ScoredCandidate,
    retrieval_metrics,
)
from shopee_match.models import (
    LearnedMultimodalFusion,
    MultimodalFusionSpec,
    balanced_pair_indices,
)
from shopee_match.training.multimodal_config import (
    load_multimodal_experiment_config,
    load_multimodal_model_config,
)
from shopee_match.training.multimodal_data import CachedMultimodalDataset
from shopee_match.training.multimodal_evaluator import ensure_frozen_test_output_absent


def _spec() -> MultimodalFusionSpec:
    return MultimodalFusionSpec(
        image_embedding_dim=8,
        text_embedding_dim=8,
        fusion_hidden_dim=16,
        joint_embedding_dim=6,
        pair_hidden_dim=8,
        dropout=0.0,
    )


def test_multimodal_fusion_normalizes_embeddings_and_scores_pairs_symmetrically() -> None:
    torch.manual_seed(7)
    model = LearnedMultimodalFusion(_spec()).eval()
    image = torch.randn(5, 8)
    text = torch.randn(5, 8)

    joint = model(image, text)
    assert joint.shape == (5, 6)
    assert torch.allclose(torch.linalg.vector_norm(joint, dim=1), torch.ones(5), atol=1e-6)
    assert torch.allclose(
        model.pair_logits(joint[:3], joint[2:5]),
        model.pair_logits(joint[2:5], joint[:3]),
        atol=1e-7,
    )


def test_multimodal_fusion_propagates_gradients_through_both_heads() -> None:
    torch.manual_seed(11)
    model = LearnedMultimodalFusion(_spec())
    image = torch.randn(6, 8)
    text = torch.randn(6, 8)
    joint = model(image, text)
    loss = joint.square().mean() + model.pair_logits(joint[:3], joint[3:]).square().mean()
    loss.backward()
    assert all(parameter.grad is not None for parameter in model.parameters())
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters())


def test_residual_fusion_initializes_to_the_configured_simple_score() -> None:
    spec = MultimodalFusionSpec(
        image_embedding_dim=2,
        text_embedding_dim=2,
        fusion_hidden_dim=8,
        joint_embedding_dim=4,
        pair_hidden_dim=4,
        dropout=0.0,
        fusion_mode="residual_score_preserving",
        base_image_weight=0.4,
        residual_scale=0.1,
    )
    model = LearnedMultimodalFusion(spec).eval()
    image = torch.tensor([[1.0, 0.0], [0.6, 0.8]])
    text = torch.tensor([[0.0, 1.0], [0.8, 0.6]])
    joint = model(image, text)
    expected = 0.4 * (image[0] @ image[1]) + 0.6 * (text[0] @ text[1])
    assert float((joint[0] @ joint[1]).detach()) == pytest.approx(float(expected), abs=1e-6)


def test_balanced_pair_indices_are_deterministic_and_bounded() -> None:
    labels = torch.tensor([0, 0, 1, 1, 2, 2])
    first = balanced_pair_indices(
        labels, maximum_negative_ratio=2, generator=torch.Generator().manual_seed(19)
    )
    second = balanced_pair_indices(
        labels, maximum_negative_ratio=2, generator=torch.Generator().manual_seed(19)
    )
    assert all(torch.equal(left, right) for left, right in zip(first, second, strict=True))
    targets = first[2]
    assert int(targets.sum()) == 3
    assert int((targets == 0).sum()) == 6


def test_simple_score_fusion_selects_complementary_modalities() -> None:
    posting_ids = ("a1", "a2", "b1", "b2")
    labels = {"a1": "a", "a2": "a", "b1": "b", "b2": "b"}
    image = np.asarray([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=np.float32)
    text = np.asarray([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=np.float32)
    weight, ranking, metrics, trials = select_simple_score_fusion(
        posting_ids,
        image,
        text,
        labels,
        image_weights=(0.0, 0.5, 1.0),
        candidate_k=2,
        recall_at=(1, 2),
        average_precision_at=2,
    )
    assert weight == 0.5
    assert metrics["map@2"] == pytest.approx(1.0)
    assert len(trials) == 3
    direct = rank_simple_score_fusion(posting_ids, image, text, image_weight=weight, candidate_k=2)
    assert retrieval_metrics(direct, labels, (1, 2), 2)["recall@1"] == pytest.approx(1.0)
    assert ranking == direct


def test_multimodal_spec_rejects_mismatched_modality_dimensions() -> None:
    with pytest.raises(ValueError, match="equal dimensions"):
        MultimodalFusionSpec(8, 7, 16, 6, 8, 0.0).validate()


def test_multimodal_failure_review_separates_rescues_and_regressions() -> None:
    posting_ids = ("a1", "a2", "b1", "b2")
    labels = {"a1": "a", "a2": "a", "b1": "b", "b2": "b"}
    split = EvaluationSplit(
        tuple(
            CorpusItem(value, f"{value}.jpg", value, f"product {value[-1]} 10ml")
            for value in posting_ids
        ),
        labels,
    )

    def ranking(top_by_query: dict[str, str]) -> dict[str, list[ScoredCandidate]]:
        return {
            query: [
                ScoredCandidate(top_by_query[query], 0.9),
                ScoredCandidate(
                    next(
                        value for value in posting_ids if value not in {query, top_by_query[query]}
                    ),
                    0.5,
                ),
            ]
            for query in posting_ids
        }

    correct = {"a1": "a2", "a2": "a1", "b1": "b2", "b2": "b1"}
    wrong = {"a1": "b1", "a2": "b2", "b1": "a1", "b2": "a2"}
    review = build_multimodal_failure_review(
        {
            "image": ranking(correct),
            "text": ranking(wrong),
            "simple_fusion": ranking(correct),
            "learned_fusion": ranking(correct),
            "pair_head": ranking(wrong),
        },
        split,
    )
    assert review["counts"]["image_rescue"] == 4
    assert review["counts"]["pair_head_regression"] == 4
    assert review["counts"]["pair_head_rescue"] == 0
    assert review["test_accessed"] is False


def test_multimodal_model_config_accepts_only_repository_random_initialization(
    tmp_path: Path,
) -> None:
    path = tmp_path / "model.yaml"
    path.write_text(
        """config_version: phase5.scratch_multimodal_model.v1
model:
  name: learned_multimodal_fusion
  source: repository
  initialization: random
  image_embedding_dim: 8
  text_embedding_dim: 8
  fusion_hidden_dim: 16
  joint_embedding_dim: 6
  pair_hidden_dim: 8
  dropout: 0.0
""",
        encoding="utf-8",
    )
    assert load_multimodal_model_config(path).joint_embedding_dim == 6
    invalid = path.read_text(encoding="utf-8").replace("random", "pretrained")
    path.write_text(invalid, encoding="utf-8")
    with pytest.raises(ConfigurationError, match="randomly initialized"):
        load_multimodal_model_config(path)


def test_cached_multimodal_dataset_preserves_alignment(tmp_path: Path) -> None:
    path = tmp_path / "validation.npz"
    np.savez_compressed(
        path,
        posting_ids=np.asarray(["p1", "p2"]),
        labels=np.asarray(["g", "g"]),
        image_embeddings=np.eye(2, dtype=np.float32),
        text_embeddings=np.eye(2, dtype=np.float32),
    )
    dataset = CachedMultimodalDataset(path)
    assert dataset.posting_ids == ("p1", "p2")
    assert dataset.labels == ("g", "g")
    assert dataset[1]["posting_id"] == "p2"


def test_frozen_multimodal_evaluation_refuses_to_overwrite_output(tmp_path: Path) -> None:
    report = tmp_path / "report.md"
    artifact_root = tmp_path / "evaluation"
    artifact_root.mkdir()
    (artifact_root / "metrics.json").write_text("{}", encoding="utf-8")
    config = SimpleNamespace(artifact_root=artifact_root, report_path=report)
    with pytest.raises(ConfigurationError, match="refusing to rerun"):
        ensure_frozen_test_output_absent(config)  # type: ignore[arg-type]


def test_multimodal_experiment_rejects_test_access_and_encoder_updates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = SimpleNamespace(
        metadata_csv=Path("data/raw/train.csv"),
        split_manifest=Path("data/splits/manifest.jsonl"),
    )
    image_training = SimpleNamespace(
        data=SimpleNamespace(**vars(data), image_dir=Path("data/raw/train_images")),
        model_spec=SimpleNamespace(embedding_dim=8),
    )
    text_training = SimpleNamespace(data=data, model_spec=SimpleNamespace(embedding_dim=8))
    import shopee_match.training.multimodal_config as config_module

    monkeypatch.setattr(config_module, "_verify_frozen_config", lambda *_args: None)
    monkeypatch.setattr(
        config_module,
        "load_frozen_image_test_config",
        lambda _path: SimpleNamespace(training_experiment=image_training),
    )
    monkeypatch.setattr(
        config_module,
        "load_frozen_text_test_config",
        lambda _path: SimpleNamespace(training_experiment=text_training),
    )
    monkeypatch.setattr(config_module, "load_multimodal_model_config", lambda _path: _spec())
    config_path = tmp_path / "experiment.yaml"
    template = f"""config_version: phase5.multimodal_experiment.v1
seed: 7
data:
  metadata_csv: data/raw/train.csv
  split_manifest: data/splits/manifest.jsonl
  image_dir: data/raw/train_images
frozen_encoders:
  image_evaluation_config: configs/image.yaml
  image_evaluation_config_sha256: "{"0" * 64}"
  text_evaluation_config: configs/text.yaml
  text_evaluation_config_sha256: "{"1" * 64}"
cache:
  root: artifacts/cache
  batch_size: 4
  num_workers: 0
model_config: configs/model.yaml
training:
  device: cpu
  epochs: 1
  products_per_batch: 2
  samples_per_product: 2
  batches_per_epoch: 1
  learning_rate: 0.001
  weight_decay: 0.0
  minimum_learning_rate: 0.0
  gradient_clip_norm: 1.0
  early_stopping_patience: 1
  deterministic: true
  freeze_image_encoder: true
  freeze_text_encoder: true
loss:
  supervised_contrastive_weight: 1.0
  pair_bce_weight: 0.5
  temperature: 0.07
  maximum_negative_ratio: 2
evaluation:
  tune_split: validation
  final_split: test
  evaluate_test: false
  candidate_pool: full_split
  exclude_query_itself: true
  recall_at: [1, 2]
  average_precision_at: 2
  candidate_k: 2
  checkpoint_metric: map@2
  simple_fusion_image_weights: [0.0, 0.5, 1.0]
artifacts:
  root: artifacts/run
  report: reports/run.md
"""
    config_path.write_text(template, encoding="utf-8")
    assert load_multimodal_experiment_config(config_path).evaluation.checkpoint_target == (
        "learned_fusion"
    )
    config_path.write_text(
        template.replace("evaluate_test: false", "evaluate_test: true"), encoding="utf-8"
    )
    with pytest.raises(ConfigurationError, match="keep test disabled"):
        load_multimodal_experiment_config(config_path)
    config_path.write_text(
        template.replace("freeze_image_encoder: true", "freeze_image_encoder: false"),
        encoding="utf-8",
    )
    with pytest.raises(ConfigurationError, match="requires both encoders frozen"):
        load_multimodal_experiment_config(config_path)
