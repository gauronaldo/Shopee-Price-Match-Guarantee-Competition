"""Full joint multimodal fine-tuning for conservative pair-recall improvement."""

from __future__ import annotations

import copy
import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray
from torch import Tensor
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from shopee_match.clustering.graph import build_conservative_clusters, score_candidate_pairs
from shopee_match.clustering.hybrid_entity_config import load_hybrid_entity_config
from shopee_match.clustering.metrics import clustering_metrics
from shopee_match.errors import ConfigurationError, DataValidationError, OutputConflictError
from shopee_match.evaluation.protocol import EvaluationSplit, load_named_split
from shopee_match.features.image import rank_phash
from shopee_match.features.text import CharTfidfModel
from shopee_match.hashing import canonical_text_sha256, sha256_file
from shopee_match.reproducibility import seed_everything
from shopee_match.retrieval.benchmark import load_phase6_model
from shopee_match.retrieval.hybrid import reciprocal_rank_fusion
from shopee_match.retrieval.hybrid_config import load_hybrid_retrieval_config
from shopee_match.retrieval.vector_index import ExactCosineIndex, search_result_to_ranking
from shopee_match.training.hard_negative_config import load_hard_negative_experiment_config
from shopee_match.training.hard_negative_data import load_hard_negative_manifest
from shopee_match.training.hard_positive_trainer import (
    MixedPairBatchProvider,
    _extract_joint_embeddings,
    _score_index_pairs,
    positive_pairs,
    select_hard_positive_pairs,
)
from shopee_match.training.image_data import ImagePreprocessor
from shopee_match.training.multimodal_data import load_cached_multimodal_split, load_frozen_encoders
from shopee_match.training.multimodal_trainer import _git_state, _resolve_device
from shopee_match.training.text_config import (
    _mapping,
    _number,
    _only_keys,
    _positive_int,
    _read_yaml,
    _relative_path,
    _typed,
)

LOGGER = logging.getLogger(__name__)
FloatArray = NDArray[np.float32]
IndexPair = tuple[int, int]


@dataclass(frozen=True, slots=True)
class JointRecallConfig:
    seed: int
    phase6_config: Path
    checkpoint: Path
    hard_negative_manifest: Path
    hybrid_config: Path
    entity_config: Path
    protocol_manifest: Path
    device: str
    epochs: int
    batches_per_epoch: int
    pair_batch_size: int
    num_workers: int
    learning_rates: tuple[float, float, float, float]
    weight_decay: float
    gradient_clip_norm: float
    early_stopping_patience: int
    pair_fractions: tuple[float, float, float, float]
    negative_distillation_weight: float
    hard_positive_limit: int
    hard_negative_limit: int
    require_different_phash: bool
    candidate_k: int
    thresholds: tuple[float, ...]
    reciprocal_rank: int
    gates: dict[str, float]
    root: Path
    checkpoint_output: Path
    metrics: Path
    report: Path
    config_path: Path


@dataclass(frozen=True, slots=True)
class _PairFractions:
    hard_positive_fraction: float
    hard_negative_fraction: float
    random_positive_fraction: float
    random_negative_fraction: float


def _verified(raw: dict[str, Any], name: str, *, text: bool = False) -> Path:
    path = _relative_path(raw[name], f"source.{name}")
    expected = _typed(raw[f"{name}_sha256"], str, f"source.{name}_sha256").lower()
    actual = canonical_text_sha256(path) if text else sha256_file(path)
    if actual != expected:
        raise ConfigurationError(f"Full-joint source hash mismatch for {path}")
    return path


def load_joint_recall_config(path: Path) -> JointRecallConfig:
    root = _read_yaml(path, "full joint pair-recall config")
    _only_keys(
        root,
        {
            "config_version",
            "seed",
            "source",
            "data",
            "mining",
            "training",
            "selection",
            "artifacts",
        },
        "config",
    )
    if root["config_version"] != "multimodal.full_joint_pair_recall.v1":
        raise ConfigurationError("Unsupported full-joint pair-recall version")
    source = _mapping(root["source"], "source")
    names = {
        "phase6_config",
        "checkpoint",
        "hard_negative_manifest",
        "hybrid_config",
        "entity_config",
        "protocol_manifest",
    }
    _only_keys(source, names | {f"{name}_sha256" for name in names}, "source")
    phase6 = _verified(source, "phase6_config", text=True)
    checkpoint = _verified(source, "checkpoint")
    hard_negatives = _verified(source, "hard_negative_manifest")
    hybrid = _verified(source, "hybrid_config", text=True)
    entity = _verified(source, "entity_config", text=True)
    protocol = _verified(source, "protocol_manifest")
    data = _mapping(root["data"], "data")
    _only_keys(data, {"train_split", "development_split", "evaluate_internal_confirmation"}, "data")
    if data != {
        "train_split": "train",
        "development_split": "validation",
        "evaluate_internal_confirmation": False,
    }:
        raise ConfigurationError("Full-joint training must disable internal confirmation")
    mining = _mapping(root["mining"], "mining")
    _only_keys(
        mining, {"hard_positive_limit", "hard_negative_limit", "require_different_phash"}, "mining"
    )
    training = _mapping(root["training"], "training")
    training_keys = {
        "device",
        "epochs",
        "batches_per_epoch",
        "pair_batch_size",
        "num_workers",
        "image_encoder_learning_rate",
        "text_encoder_learning_rate",
        "fusion_learning_rate",
        "pair_head_learning_rate",
        "weight_decay",
        "gradient_clip_norm",
        "early_stopping_patience",
        "hard_positive_fraction",
        "hard_negative_fraction",
        "random_positive_fraction",
        "random_negative_fraction",
        "negative_distillation_weight",
    }
    _only_keys(training, training_keys, "training")
    fractions = tuple(
        float(training[name])
        for name in (
            "hard_positive_fraction",
            "hard_negative_fraction",
            "random_positive_fraction",
            "random_negative_fraction",
        )
    )
    if min(fractions) <= 0 or abs(sum(fractions) - 1.0) > 1e-9:
        raise ConfigurationError("Full-joint pair fractions must be positive and sum to one")
    selection = _mapping(root["selection"], "selection")
    selection_keys = {
        "candidate_k",
        "pair_probability_thresholds",
        "reciprocal_rank",
        "minimum_pairwise_precision",
        "minimum_pairwise_recall_delta",
        "minimum_pairwise_f1_delta",
        "maximum_false_merge_pair_rate",
        "maximum_false_split_group_rate",
    }
    _only_keys(selection, selection_keys, "selection")
    thresholds = tuple(
        _number(value, f"selection.pair_probability_thresholds[{index}]")
        for index, value in enumerate(cast(list[object], selection["pair_probability_thresholds"]))
    )
    if not thresholds or tuple(sorted(set(thresholds))) != thresholds:
        raise ConfigurationError("Pair thresholds must be sorted and unique")
    artifacts = _mapping(root["artifacts"], "artifacts")
    _only_keys(artifacts, {"root", "checkpoint", "metrics", "report"}, "artifacts")
    output_root = _relative_path(artifacts["root"], "artifacts.root")
    outputs = tuple(
        _relative_path(artifacts[name], f"artifacts.{name}")
        for name in ("checkpoint", "metrics", "report")
    )
    if any(output.parent != output_root for output in outputs):
        raise ConfigurationError("Full-joint outputs must live directly under artifacts.root")
    lrs = tuple(
        _number(training[name], f"training.{name}")
        for name in (
            "image_encoder_learning_rate",
            "text_encoder_learning_rate",
            "fusion_learning_rate",
            "pair_head_learning_rate",
        )
    )
    gates = {
        name: float(selection[name])
        for name in selection_keys
        if name.startswith("minimum_") or name.startswith("maximum_")
    }
    device = _typed(training["device"], str, "training.device")
    if device not in {"cpu", "cuda", "auto"}:
        raise ConfigurationError("Invalid full-joint device")
    return JointRecallConfig(
        int(root["seed"]),
        phase6,
        checkpoint,
        hard_negatives,
        hybrid,
        entity,
        protocol,
        device,
        _positive_int(training["epochs"], "training.epochs"),
        _positive_int(training["batches_per_epoch"], "training.batches_per_epoch"),
        _positive_int(training["pair_batch_size"], "training.pair_batch_size"),
        int(training["num_workers"]),
        cast(tuple[float, float, float, float], lrs),
        float(training["weight_decay"]),
        float(training["gradient_clip_norm"]),
        _positive_int(training["early_stopping_patience"], "training.early_stopping_patience"),
        cast(tuple[float, float, float, float], fractions),
        float(training["negative_distillation_weight"]),
        _positive_int(mining["hard_positive_limit"], "mining.hard_positive_limit"),
        _positive_int(mining["hard_negative_limit"], "mining.hard_negative_limit"),
        bool(mining["require_different_phash"]),
        _positive_int(selection["candidate_k"], "selection.candidate_k"),
        thresholds,
        _positive_int(selection["reciprocal_rank"], "selection.reciprocal_rank"),
        gates,
        output_root,
        outputs[0],
        outputs[1],
        outputs[2],
        path,
    )


class _JointPairDataset(Dataset[dict[str, Tensor]]):
    def __init__(
        self,
        split: EvaluationSplit,
        image_dir: Path,
        vocabulary: Any,
        maximum_length: int,
        pairs: list[IndexPair],
        targets: Tensor,
        preprocessor: ImagePreprocessor,
    ) -> None:
        self.split, self.image_dir, self.vocabulary = split, image_dir, vocabulary
        self.maximum_length, self.pairs, self.targets, self.preprocessor = (
            maximum_length,
            pairs,
            targets,
            preprocessor,
        )

    def __len__(self) -> int:
        return len(self.pairs)

    def _listing(self, index: int, sample_index: int) -> tuple[Tensor, Tensor, Tensor]:
        item = self.split.items[index]
        image = cv2.imread(str(self.image_dir / item.image), cv2.IMREAD_COLOR)
        if image is None:
            raise DataValidationError(f"Cannot decode joint-training image: {item.image}")
        rgb = cast(NDArray[np.uint8], cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        token_ids, length = self.vocabulary.encode(item.title, self.maximum_length)
        return self.preprocessor(rgb, sample_index), token_ids, length

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        left, right = self.pairs[index]
        left_image, left_tokens, left_length = self._listing(left, index * 2)
        right_image, right_tokens, right_length = self._listing(right, index * 2 + 1)
        return {
            "left_image": left_image,
            "left_tokens": left_tokens,
            "left_length": left_length,
            "right_image": right_image,
            "right_tokens": right_tokens,
            "right_length": right_length,
            "left_index": torch.tensor(left),
            "right_index": torch.tensor(right),
            "target": self.targets[index],
        }


def _extract_raw(
    image_model: Any,
    text_model: Any,
    fusion: Any,
    split: EvaluationSplit,
    image_dir: Path,
    vocabulary: Any,
    maximum_length: int,
    image_size: int,
    device: torch.device,
    batch_size: int = 64,
) -> tuple[tuple[str, ...], FloatArray]:
    preprocessor = ImagePreprocessor(image_size, training=False, seed=0)
    posting_ids: list[str] = []
    arrays: list[FloatArray] = []
    image_model.eval()
    text_model.eval()
    fusion.eval()
    with torch.inference_mode():
        for start in range(0, len(split.items), batch_size):
            items = split.items[start : start + batch_size]
            images, tokens, lengths = [], [], []
            for offset, item in enumerate(items):
                image = cv2.imread(str(image_dir / item.image), cv2.IMREAD_COLOR)
                if image is None:
                    raise DataValidationError(f"Cannot decode evaluation image: {item.image}")
                rgb = cast(NDArray[np.uint8], cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                images.append(preprocessor(rgb, start + offset))
                token_ids, length = vocabulary.encode(item.title, maximum_length)
                tokens.append(token_ids)
                lengths.append(length)
            image_embedding = image_model(torch.stack(images).to(device))
            text_embedding = text_model(
                torch.stack(tokens).to(device), torch.stack(lengths).to(device)
            )
            arrays.append(cast(FloatArray, fusion(image_embedding, text_embedding).cpu().numpy()))
            posting_ids.extend(item.posting_id for item in items)
    return tuple(posting_ids), np.concatenate(arrays)


def _ranking(
    posting_ids: tuple[str, ...],
    embeddings: FloatArray,
    split: EvaluationSplit,
    hybrid: Any,
    tfidf_model: CharTfidfModel,
    candidate_k: int,
) -> Any:
    index = ExactCosineIndex(posting_ids, embeddings)
    indices, scores = index.search(embeddings, candidate_k, query_ids=posting_ids, block_size=512)
    dense = search_result_to_ranking(posting_ids, posting_ids, indices, scores)
    tfidf = tfidf_model.rank(split.items, candidate_k)
    phash = rank_phash(split.items, hybrid.fusion.phash_candidate_k)
    return reciprocal_rank_fusion(
        {"dense": dense, "tfidf": tfidf, "phash": phash},
        weights={
            "dense": hybrid.fusion.dense_weight,
            "tfidf": hybrid.fusion.tfidf_weight,
            "phash": hybrid.fusion.phash_weight,
        },
        rrf_constant=hybrid.fusion.rrf_constant,
        top_k=candidate_k,
    )


def _select_graph(
    config: JointRecallConfig,
    model: Any,
    posting_ids: tuple[str, ...],
    embeddings: FloatArray,
    split: EvaluationSplit,
    ranking: Any,
    device: torch.device,
    recovery: Any,
    baseline: dict[str, Any] | None,
) -> dict[str, Any]:
    pairs = score_candidate_pairs(
        model, posting_ids, split.items, embeddings, ranking, device, batch_size=8192
    )
    trials: list[dict[str, Any]] = []
    for threshold in config.thresholds:
        assignments, graph = build_conservative_clusters(
            posting_ids,
            pairs,
            pair_probability_threshold=threshold,
            reciprocal_rank=config.reciprocal_rank,
            cross_component_minimum_coverage=1.0,
            variant_conflict_override_probability=recovery.variant_conflict_override_probability,
            maximum_cluster_size=recovery.maximum_cluster_size,
            manual_review_margin=recovery.manual_review_margin,
            singleton_attachment=True,
            singleton_probability_threshold=max(threshold, 0.18),
            singleton_reciprocal_rank=50,
            singleton_minimum_support=2,
            singleton_target_margin=0.02,
        )
        trials.append(
            {
                "threshold": threshold,
                "clustering": clustering_metrics(assignments, split.label_by_id),
                "graph": asdict(graph),
            }
        )
    if baseline is None:
        return max(
            trials,
            key=lambda row: (
                row["clustering"]["pairwise"]["f1"],
                row["clustering"]["pairwise"]["precision"],
            ),
        )
    base_cluster = baseline["clustering"]
    eligible = [
        row
        for row in trials
        if (
            row["clustering"]["pairwise"]["precision"] >= config.gates["minimum_pairwise_precision"]
            and row["clustering"]["pairwise"]["recall"] - base_cluster["pairwise"]["recall"]
            >= config.gates["minimum_pairwise_recall_delta"]
            and row["clustering"]["pairwise"]["f1"] - base_cluster["pairwise"]["f1"]
            >= config.gates["minimum_pairwise_f1_delta"]
            and row["clustering"]["false_merge_pair_rate"]
            <= config.gates["maximum_false_merge_pair_rate"]
            and row["clustering"]["false_split_group_rate"]
            <= config.gates["maximum_false_split_group_rate"]
        )
    ]
    selected = max(
        eligible or trials,
        key=lambda row: (
            row["clustering"]["pairwise"]["f1"],
            row["clustering"]["pairwise"]["recall"],
        ),
    )
    selected["passes_all_gates"] = bool(eligible)
    return selected


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _save_checkpoint(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def run_joint_recall_training(
    config_path: Path, *, progress_updates_per_epoch: int = 4
) -> dict[str, object]:
    if progress_updates_per_epoch < 0:
        raise ValueError("progress_updates_per_epoch must be nonnegative")
    config = load_joint_recall_config(config_path)
    existing = [
        str(path)
        for path in (config.checkpoint_output, config.metrics, config.report)
        if path.exists()
    ]
    if existing:
        raise OutputConflictError(
            "Refusing to overwrite full-joint outputs: " + ", ".join(existing)
        )
    commit, dirty = _git_state()
    if dirty:
        raise DataValidationError("Full-joint training requires a clean Git worktree")
    started = time.perf_counter()
    seed_everything(config.seed, deterministic=True)
    device = _resolve_device(config.device)
    phase6 = load_hard_negative_experiment_config(config.phase6_config)
    multimodal = phase6.source.experiment
    hybrid = load_hybrid_retrieval_config(config.hybrid_config)
    if config.checkpoint != hybrid.source.experiment.source.checkpoint_path:
        raise ConfigurationError(
            "Full-joint checkpoint differs from the frozen candidate-retrieval source"
        )
    train = load_named_split(
        multimodal.data.metadata_csv,
        config.protocol_manifest,
        "train",
    )
    development = load_named_split(
        multimodal.data.metadata_csv,
        config.protocol_manifest,
        "validation",
    )
    model = load_phase6_model(hybrid.source.experiment, device)
    image_model, text_model, vocabulary, maximum_length = load_frozen_encoders(multimodal, device)
    image_model.requires_grad_(True)
    text_model.requires_grad_(True)
    image_size = multimodal.frozen.image_config.training_experiment.image_size
    cache = load_cached_multimodal_split(multimodal, "train")
    if tuple(item.posting_id for item in train.items) != cache.posting_ids:
        raise DataValidationError(
            "Pair-recall protocol train split differs from the frozen training cache"
        )
    source_joint = _extract_joint_embeddings(model, cache, device, batch_size=8192)
    all_positives = positive_pairs(cache.labels, tuple(range(len(cache))))
    if config.require_different_phash:
        item_by_id = {item.posting_id: item for item in train.items}
        all_positives = [
            pair
            for pair in all_positives
            if item_by_id[cache.posting_ids[pair[0]]].image_phash
            != item_by_id[cache.posting_ids[pair[1]]].image_phash
        ]
    probabilities = _score_index_pairs(
        model, source_joint, all_positives, batch_size=8192, device=device
    )
    hard_positives = select_hard_positive_pairs(
        all_positives, probabilities, limit=min(config.hard_positive_limit, len(all_positives))
    )
    index_by_id = {posting_id: index for index, posting_id in enumerate(cache.posting_ids)}
    hard_negatives = [
        (index_by_id[pair.left_posting_id], index_by_id[pair.right_posting_id])
        for pair in load_hard_negative_manifest(config.hard_negative_manifest)
        if pair.left_posting_id in index_by_id and pair.right_posting_id in index_by_id
    ][: config.hard_negative_limit]
    training_proxy = _PairFractions(*config.pair_fractions)
    provider = MixedPairBatchProvider(
        hard_positives=hard_positives,
        hard_negatives=hard_negatives,
        random_positives=positive_pairs(cache.labels, tuple(range(len(cache)))),
        allowed_indices=tuple(range(len(cache))),
        labels=cache.labels,
        seed=config.seed,
    )
    tfidf_model = CharTfidfModel.fit(
        train.items, hybrid.tfidf.ngram_range, hybrid.tfidf.max_features
    )
    posting_ids, base_embeddings = _extract_raw(
        image_model,
        text_model,
        model,
        development,
        multimodal.data.image_dir,
        vocabulary,
        maximum_length,
        image_size,
        device,
    )
    base_ranking = _ranking(
        posting_ids, base_embeddings, development, hybrid, tfidf_model, config.candidate_k
    )
    entity_config = load_hybrid_entity_config(config.entity_config)
    recovery_policy = entity_config.source.recovery.selection
    baseline = _select_graph(
        config,
        model,
        posting_ids,
        base_embeddings,
        development,
        base_ranking,
        device,
        recovery_policy,
        None,
    )
    teacher = copy.deepcopy(model.pair_head).requires_grad_(False).eval()
    optimizer = AdamW(
        [
            {"params": image_model.parameters(), "lr": config.learning_rates[0]},
            {"params": text_model.parameters(), "lr": config.learning_rates[1]},
            {"params": model.fusion.parameters(), "lr": config.learning_rates[2]},
            {"params": model.pair_head.parameters(), "lr": config.learning_rates[3]},
        ],
        weight_decay=config.weight_decay,
    )
    best_state = None
    best_epoch = -1
    best_development = baseline
    history = []
    stale = 0
    for epoch in range(config.epochs):
        pairs: list[IndexPair] = []
        target_parts: list[Tensor] = []
        for batch_index in range(config.batches_per_epoch):
            left, right, targets = provider.sample(
                epoch=epoch,
                batch_index=batch_index,
                batch_size=config.pair_batch_size,
                config=cast(Any, training_proxy),
            )
            pairs.extend(zip(left.tolist(), right.tolist(), strict=True))
            target_parts.append(targets)
        targets = torch.cat(target_parts)
        preprocessor = ImagePreprocessor(image_size, training=True, seed=config.seed)
        preprocessor.set_epoch(epoch)
        loader = DataLoader(
            _JointPairDataset(
                train,
                multimodal.data.image_dir,
                vocabulary,
                maximum_length,
                pairs,
                targets,
                preprocessor,
            ),
            batch_size=config.pair_batch_size,
            shuffle=False,
            num_workers=config.num_workers,
        )
        image_model.train()
        text_model.train()
        model.train()
        total_loss = 0.0
        milestones = (
            {
                max(1, round(len(loader) * step / progress_updates_per_epoch))
                for step in range(1, progress_updates_per_epoch + 1)
            }
            if progress_updates_per_epoch
            else set()
        )
        for loader_index, batch in enumerate(loader, start=1):
            optimizer.zero_grad(set_to_none=True)
            left_image = image_model(batch["left_image"].to(device))
            right_image = image_model(batch["right_image"].to(device))
            left_text = text_model(batch["left_tokens"].to(device), batch["left_length"].to(device))
            right_text = text_model(
                batch["right_tokens"].to(device), batch["right_length"].to(device)
            )
            left_joint = model(left_image, left_text)
            right_joint = model(right_image, right_text)
            logits = model.pair_logits(left_joint, right_joint)
            target = batch["target"].to(device)
            loss = F.binary_cross_entropy_with_logits(logits, target)
            negative = target == 0
            if negative.any():
                with torch.no_grad():
                    teacher_left = source_joint[batch["left_index"]].to(device)
                    teacher_right = source_joint[batch["right_index"]].to(device)
                    teacher_logits = teacher(
                        torch.cat(
                            (
                                teacher_left * teacher_right,
                                torch.abs(teacher_left - teacher_right),
                            ),
                            dim=1,
                        )
                    ).squeeze(1)
                loss = loss + config.negative_distillation_weight * F.mse_loss(
                    torch.sigmoid(logits[negative]), torch.sigmoid(teacher_logits[negative])
                )
            loss.backward()  # type: ignore[no-untyped-call]
            torch.nn.utils.clip_grad_norm_(
                list(image_model.parameters())
                + list(text_model.parameters())
                + list(model.parameters()),
                config.gradient_clip_norm,
            )
            optimizer.step()
            total_loss += float(loss.detach().cpu())
            if loader_index in milestones:
                LOGGER.info(
                    "joint epoch %d/%d training: %d/%d batches (%d%%)",
                    epoch + 1,
                    config.epochs,
                    loader_index,
                    len(loader),
                    round(100 * loader_index / len(loader)),
                )
        posting_ids, embeddings = _extract_raw(
            image_model,
            text_model,
            model,
            development,
            multimodal.data.image_dir,
            vocabulary,
            maximum_length,
            image_size,
            device,
        )
        ranking = _ranking(
            posting_ids, embeddings, development, hybrid, tfidf_model, config.candidate_k
        )
        selected = _select_graph(
            config,
            model,
            posting_ids,
            embeddings,
            development,
            ranking,
            device,
            recovery_policy,
            baseline,
        )
        cluster = selected["clustering"]
        history.append(
            {
                "epoch": epoch,
                "train_loss": total_loss / len(loader),
                "threshold": selected["threshold"],
                "pairwise_precision": cluster["pairwise"]["precision"],
                "pairwise_recall": cluster["pairwise"]["recall"],
                "pairwise_f1": cluster["pairwise"]["f1"],
                "false_merge_pair_rate": cluster["false_merge_pair_rate"],
                "passes_all_gates": selected.get("passes_all_gates", False),
            }
        )
        LOGGER.info(
            "joint epoch %d/%d loss=%.5f P=%.5f R=%.5f F1=%.5f false_merge=%.5f gates=%s",
            epoch + 1,
            config.epochs,
            history[-1]["train_loss"],
            cluster["pairwise"]["precision"],
            cluster["pairwise"]["recall"],
            cluster["pairwise"]["f1"],
            cluster["false_merge_pair_rate"],
            selected.get("passes_all_gates", False),
        )
        if (
            selected.get("passes_all_gates", False)
            and cluster["pairwise"]["f1"] > best_development["clustering"]["pairwise"]["f1"]
        ):
            best_epoch = epoch
            best_development = selected
            stale = 0
            best_state = {
                "image_model": copy.deepcopy(image_model.state_dict()),
                "text_model": copy.deepcopy(text_model.state_dict()),
                "fusion_model": copy.deepcopy(model.state_dict()),
            }
        elif best_state is not None:
            stale += 1
        if best_state is not None and stale >= config.early_stopping_patience:
            break
    status = (
        "accepted_development_only" if best_state is not None else "not_accepted_development_only"
    )
    config.root.mkdir(parents=True, exist_ok=True)
    if best_state is not None:
        payload: dict[str, object] = {
            "checkpoint_version": "multimodal.full_joint_pair_recall.v1",
            "model_state": best_state,
            "best_epoch": best_epoch,
            "selected_policy": best_development,
            "config_sha256": canonical_text_sha256(config.config_path),
        }
        _save_checkpoint(config.checkpoint_output, payload)
    run = {
        "pipeline_version": "multimodal.full_joint_pair_recall.v1",
        "status": status,
        "provenance": {
            "git_commit": commit,
            "git_dirty": False,
            "config_sha256": canonical_text_sha256(config.config_path),
            "protocol_manifest_sha256": sha256_file(config.protocol_manifest),
        },
        "data": {
            "train_listings": len(train.items),
            "development_listings": len(development.items),
            "confirmation_accessed": False,
            "historical_test_accessed": False,
        },
        "training": {
            "learning_rates": config.learning_rates,
            "history": history,
            "best_epoch": best_epoch,
        },
        "safety_gates": config.gates,
        "baseline": baseline,
        "selected": best_development,
        "runtime_seconds": time.perf_counter() - started,
    }
    _write(config.metrics, json.dumps(run, indent=2, sort_keys=True) + "\n")
    _write(
        config.report,
        (
            "# Full Joint Pair-Recall Training\n\n"
            f"Status: **{status}**.\n\n"
            "The run used only the pair-recall v3 train and development partitions. "
            "Internal confirmation and historical test labels were not loaded.\n"
        ),
    )
    return {
        "status": status,
        "best_epoch": best_epoch,
        "metrics": str(config.metrics),
        "confirmation_accessed": False,
    }
