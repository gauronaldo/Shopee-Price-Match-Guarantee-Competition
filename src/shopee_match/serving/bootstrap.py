"""Build the local validation-catalog demo from released weights without training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import yaml

from shopee_match.clustering.benchmark import run_entity_resolution_benchmark
from shopee_match.data.pipeline import prepare_dataset
from shopee_match.errors import ConfigurationError, DataValidationError
from shopee_match.hashing import sha256_file
from shopee_match.retrieval.benchmark import run_candidate_retrieval_benchmark
from shopee_match.serving.config import load_demo_config
from shopee_match.serving.release_artifacts import load_model_release, model_release_status
from shopee_match.training.multimodal_data import prepare_frozen_multimodal_split_cache

DEFAULT_RELEASE_MANIFEST = Path("configs/serving/model_release.yaml")
DEFAULT_RUNTIME_ROOT = Path("artifacts/demo/runtime")
DEFAULT_DEMO_CONFIG = DEFAULT_RUNTIME_ROOT / "demo.yaml"


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigurationError(f"Cannot read bootstrap source config: {path}") from exc
    if not isinstance(value, dict):
        raise ConfigurationError(f"Bootstrap source config must be a mapping: {path}")
    return cast(dict[str, Any], value)


def _write_yaml(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = yaml.safe_dump(value, sort_keys=False, allow_unicode=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(rendered, encoding="utf-8", newline="\n")
    temporary.replace(path)


def _write_bootstrap_data_config(runtime_root: Path) -> Path:
    source = _load_yaml(Path("configs/data/shopee.yaml"))
    audit_root = runtime_root / "data_audit"
    audit = cast(dict[str, Any], source["audit"])
    audit["report_json"] = (audit_root / "audit.json").as_posix()
    audit["report_markdown"] = (audit_root / "audit.md").as_posix()
    audit["figure_dir"] = (audit_root / "figures").as_posix()
    audit["inspection_dir"] = (audit_root / "inspection").as_posix()
    output = runtime_root / "configs" / "data.yaml"
    _write_yaml(output, source)
    return output


def _ensure_split(runtime_root: Path, expected_sha256: str) -> dict[str, object]:
    manifest = Path("data/splits/shopee_group_split_v1.jsonl")
    if not manifest.is_file():
        data_config = _write_bootstrap_data_config(runtime_root)
        result = prepare_dataset(data_config)
        state = "created"
    else:
        result = {"manifest": str(manifest)}
        state = "reused"
    actual = sha256_file(manifest) if manifest.is_file() else ""
    if actual != expected_sha256:
        raise DataValidationError(
            "The generated split differs from the split used to train the released models"
        )
    return {"status": state, "sha256": actual, **result}


def _candidate_config(runtime_root: Path, device: str) -> Path:
    config = _load_yaml(Path("configs/experiment/candidate_retrieval_benchmark.yaml"))
    config["config_version"] = "demo.candidate_retrieval.v1"
    cast(dict[str, Any], config["source"]).pop("mined_manifest")
    cast(dict[str, Any], config["embedding"])["device"] = device
    artifact_root = runtime_root / "candidate_retrieval"
    artifacts = cast(dict[str, Any], config["artifacts"])
    artifacts.update(
        {
            "root": artifact_root.as_posix(),
            "embedding_cache": (artifact_root / "embeddings.npz").as_posix(),
            "exact_index": (artifact_root / "exact_index.npz").as_posix(),
            "faiss_index": (artifact_root / "hnsw.faiss").as_posix(),
            "faiss_metadata": (artifact_root / "hnsw.metadata.json").as_posix(),
            "metrics": (artifact_root / "metrics.json").as_posix(),
            "review": (artifact_root / "failure_review.json").as_posix(),
            "report": (runtime_root / "reports" / "candidate_retrieval.md").as_posix(),
        }
    )
    output = runtime_root / "configs" / "candidate_retrieval.yaml"
    _write_yaml(output, config)
    return output


def _entity_config(runtime_root: Path, candidate_config: Path, device: str) -> Path:
    config = _load_yaml(Path("configs/experiment/entity_resolution_benchmark.yaml"))
    candidate = _load_yaml(candidate_config)
    candidate_artifacts = cast(dict[str, Any], candidate["artifacts"])
    source = cast(dict[str, Any], config["source"])
    source.update(
        {
            "phase7_config": candidate_config.as_posix(),
            "phase7_config_sha256": sha256_file(candidate_config),
            "phase7_metrics": candidate_artifacts["metrics"],
            "phase7_metrics_sha256": sha256_file(Path(candidate_artifacts["metrics"])),
            "embedding_cache": candidate_artifacts["embedding_cache"],
            "embedding_cache_sha256": sha256_file(Path(candidate_artifacts["embedding_cache"])),
        }
    )
    cast(dict[str, Any], config["pair_scoring"])["device"] = device
    artifact_root = runtime_root / "entity_resolution"
    artifacts = cast(dict[str, Any], config["artifacts"])
    artifacts.update(
        {
            "root": artifact_root.as_posix(),
            "scored_pairs": (artifact_root / "scored_pairs.jsonl").as_posix(),
            "assignments": (artifact_root / "entity_assignments.csv").as_posix(),
            "metrics": (artifact_root / "metrics.json").as_posix(),
            "review": (artifact_root / "failure_review.json").as_posix(),
            "report": (runtime_root / "reports" / "entity_resolution.md").as_posix(),
        }
    )
    output = runtime_root / "configs" / "entity_resolution.yaml"
    _write_yaml(output, config)
    return output


def _demo_config(runtime_root: Path, entity_config: Path) -> Path:
    config = _load_yaml(Path("configs/serving/demo.yaml"))
    entity = _load_yaml(entity_config)
    entity_artifacts = cast(dict[str, Any], entity["artifacts"])
    modality_embeddings = Path("artifacts/multimodal_fusion/frozen_encoder_cache/validation.npz")
    source = cast(dict[str, Any], config["source"])
    source.update(
        {
            "entity_config": entity_config.as_posix(),
            "entity_config_sha256": sha256_file(entity_config),
            "entity_metrics": entity_artifacts["metrics"],
            "entity_metrics_sha256": sha256_file(Path(entity_artifacts["metrics"])),
            "modality_embeddings": modality_embeddings.as_posix(),
            "modality_embeddings_sha256": sha256_file(modality_embeddings),
            "entity_assignments": entity_artifacts["assignments"],
            "entity_assignments_sha256": sha256_file(Path(entity_artifacts["assignments"])),
        }
    )
    output = runtime_root / "demo.yaml"
    _write_yaml(output, config)
    return output


def bootstrap_demo(
    *,
    release_manifest: Path = DEFAULT_RELEASE_MANIFEST,
    runtime_root: Path = DEFAULT_RUNTIME_ROOT,
    device: str = "auto",
) -> dict[str, object]:
    """Create catalog embeddings and clusters using frozen weights; never train or access test."""
    if device not in {"auto", "cpu", "cuda"}:
        raise ConfigurationError("bootstrap device must be auto, cpu, or cuda")
    release = load_model_release(release_manifest)
    model_status = model_release_status(release_manifest)
    if model_status["status"] != "ready":
        raise DataValidationError("Model artifacts are missing; run download-models first")

    if (runtime_root / "demo.yaml").is_file():
        try:
            load_demo_config(runtime_root / "demo.yaml")
        except (ConfigurationError, DataValidationError, OSError, ValueError):
            pass
        else:
            return {
                "status": "ready",
                "bootstrap": "reused",
                "training_performed": False,
                "test_accessed": False,
                "config": str(runtime_root / "demo.yaml"),
            }

    split = _ensure_split(runtime_root, release.split_manifest_sha256)
    cache = prepare_frozen_multimodal_split_cache(
        Path("configs/experiment/multimodal_embedding_training.yaml"),
        "validation",
        requested_device=device,
    )
    candidate_config = _candidate_config(runtime_root, device)
    candidate = run_candidate_retrieval_benchmark(candidate_config)
    entity_config = _entity_config(runtime_root, candidate_config, device)
    entity = run_entity_resolution_benchmark(entity_config)
    demo_config = _demo_config(runtime_root, entity_config)
    load_demo_config(demo_config)
    result = {
        "status": "ready",
        "bootstrap": "created",
        "training_performed": False,
        "test_accessed": False,
        "release": release.tag,
        "split": split,
        "cache": cache,
        "candidate_retrieval": candidate,
        "entity_resolution": entity,
        "config": str(demo_config),
    }
    summary = runtime_root / "bootstrap_summary.json"
    summary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result
