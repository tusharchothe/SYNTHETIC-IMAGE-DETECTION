from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import select

from .dataset_builder import build_dataset_version
from .storage.database import SessionLocal, get_engine
from .storage.models import ImageRecord, Prediction, ReviewStatus


TRAINABLE_MODELS = {
    "resnet50": "trainable",
    "resnet": "trainable",
    "clip": "head_only",
    "clip_vit": "head_only",
}


@dataclass
class TrainingConfig:
    dataset_version: str
    dataset_root: str = "datasets"
    model_name: str = "resnet50"
    output_dir: str = "artifacts/training"
    init_checkpoint: str | None = None
    admin_confirmed: bool = False
    epochs: int = 5
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    seed: int = 42
    early_stopping_patience: int = 3
    max_memory_gb: float | None = 12.0
    resume: bool = False
    dry_run: bool = False
    monitor_metric: str = "val_auc"
    threshold: float = 0.5
    min_examples_per_class: int = 2
    keep_best_only: bool = True
    log_level: str = "INFO"
    benchmark_test_ids: list[str] | None = None
    status_path: str | None = None
    resource_limits: dict[str, Any] = field(default_factory=dict)

    def ensure_admin_authorized(self) -> None:
        if not self.admin_confirmed:
            raise PermissionError("Training requires explicit admin confirmation. Set --admin-confirmed to start training.")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TrainingStatus:
    job_id: str
    dataset_version: str
    model_name: str
    status: str = "created"
    started_at: str | None = None
    finished_at: str | None = None
    last_error: str | None = None
    progress: float = 0.0
    current_epoch: int = 0
    best_metric: float | None = None
    artifact_path: str | None = None
    artifact_hash: str | None = None
    config: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def detect_model_trainability(model_name: str) -> str:
    normalized = model_name.lower().replace("-", "")
    return TRAINABLE_MODELS.get(normalized, "unsupported")


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_manifest(dataset_root: str | Path, dataset_version: str) -> list[dict[str, Any]]:
    version_dir = Path(dataset_root) / dataset_version
    manifest_path = version_dir / "manifest.jsonl"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Dataset version '{dataset_version}' was not found at {manifest_path}.")
    rows = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _load_split_definitions(dataset_root: str | Path, dataset_version: str) -> dict[str, Any]:
    path = Path(dataset_root) / dataset_version / "split_definitions.json"
    if not path.exists():
        raise FileNotFoundError(f"Split definitions for version '{dataset_version}' were not found at {path}.")
    return json.loads(path.read_text(encoding="utf-8"))


def validate_dataset_version(config: TrainingConfig, version_dir: str | Path) -> dict[str, Any]:
    version_dir = Path(version_dir)
    manifest_path = version_dir / "manifest.jsonl"
    if not manifest_path.exists():
        raise ValueError(f"Manifest missing for dataset version {config.dataset_version} at {manifest_path}.")

    manifest = _load_manifest(version_dir.parent, config.dataset_version)
    if not manifest:
        raise ValueError(f"Dataset version {config.dataset_version} is empty.")

    required_splits = {"train", "val", "test"}
    split_data = _load_split_definitions(version_dir.parent, config.dataset_version)
    split_counts = split_data.get("splits", {})
    missing = sorted(required_splits - set(split_counts))
    if missing:
        raise ValueError(f"Dataset version {config.dataset_version} is missing required splits: {missing}")

    if "benchmark_test_ids" in split_data and split_data["benchmark_test_ids"]:
        benchmark_ids = set(split_data["benchmark_test_ids"])
        if any(item["image_id"] in benchmark_ids for item in manifest if item.get("split") == "train"):
            raise ValueError("Benchmark test images must be excluded from the training split.")

    labels = [row.get("human_label") for row in manifest if row.get("split") in {"train", "val"}]
    valid_labels = {"real", "ai"}
    invalid = [label for label in labels if label not in valid_labels]
    if invalid:
        raise ValueError(f"Dataset contains invalid labels for training: {sorted(set(invalid))[:10]}")

    train_labels = [row.get("human_label") for row in manifest if row.get("split") == "train"]
    val_labels = [row.get("human_label") for row in manifest if row.get("split") == "val"]
    if not train_labels or not val_labels:
        raise ValueError("Training and validation splits must both contain at least one labeled sample.")
    if len(set(train_labels)) < 2 or len(set(val_labels)) < 2:
        raise ValueError("Training and validation splits must contain both real and ai labels.")

    for row in manifest:
        path = row.get("storage_path")
        if not path or not Path(path).exists():
            raise ValueError(f"Missing file referenced by dataset version '{config.dataset_version}': {path}")

    return {"manifest_count": len(manifest), "train_rows": len(train_labels), "val_rows": len(val_labels), "split_counts": split_counts}


def prepare_training_status(config: TrainingConfig) -> TrainingStatus:
    config.ensure_admin_authorized()
    status = TrainingStatus(
        job_id=f"train-{config.model_name}-{config.dataset_version}-{int(time.time())}",
        dataset_version=config.dataset_version,
        model_name=config.model_name,
        status="created",
        config=config.as_dict(),
    )
    return status


def write_training_status(status_path: str | Path, status: TrainingStatus) -> None:
    path = Path(status_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(status.to_dict(), indent=2, sort_keys=True), encoding="utf-8")


def _ensure_training_environment(model_name: str, max_memory_gb: float | None = None) -> None:
    model_state = detect_model_trainability(model_name)
    if model_state == "unsupported":
        raise ValueError(f"Model '{model_name}' is not supported for training.")

    if max_memory_gb is not None:
        try:
            import psutil
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("psutil is required for memory guardrails.") from exc
        available = psutil.virtual_memory().available / (1024 ** 3)
        if available < max_memory_gb:
            raise MemoryError(f"Insufficient memory for training: {available:.2f} GB available, {max_memory_gb} GB required.")

    try:
        import torch
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("PyTorch is required to train the ResNet model.") from exc

    if model_state == "head_only" and not hasattr(torch, "cuda"):
        pass


def _build_resnet_training_args(config: TrainingConfig) -> dict[str, Any]:
    return {
        "epochs": config.epochs,
        "pretrained": bool(config.init_checkpoint is None),
        "lr": config.learning_rate,
        "weight_decay": config.weight_decay,
        "seed": config.seed,
        "batch_size": config.batch_size,
        "monitor_metric": config.monitor_metric,
        "threshold": config.threshold,
    }


def build_training_plan(config: TrainingConfig) -> dict[str, Any]:
    config.ensure_admin_authorized()
    model_state = detect_model_trainability(config.model_name)
    if model_state == "unsupported":
        raise ValueError(f"Unsupported model '{config.model_name}' for training.")

    version_dir = Path(config.dataset_root) / config.dataset_version
    validate_dataset_version(config, version_dir)

    return {
        "model_name": config.model_name,
        "trainability": model_state,
        "dataset_version_dir": str(version_dir),
        "model_config": _build_resnet_training_args(config) if config.model_name.lower().startswith("resnet") else {"head_only": True},
        "status": "ready",
    }


def run_training_pipeline(config: TrainingConfig) -> dict[str, Any]:
    config.ensure_admin_authorized()
    status = prepare_training_status(config)
    status_path = config.status_path or str(Path(config.output_dir) / f"{status.job_id}.json")
    write_training_status(status_path, status)

    try:
        _ensure_training_environment(config.model_name, config.max_memory_gb)
        plan = build_training_plan(config)
        status.status = "running"
        status.started_at = datetime.now(timezone.utc).isoformat()
        status.progress = 0.05
        write_training_status(status_path, status)

        if config.dry_run:
            status.status = "dry_run"
            status.finished_at = datetime.now(timezone.utc).isoformat()
            status.progress = 1.0
            write_training_status(status_path, status)
            return {"status": "dry_run", "plan": plan, "status_path": status_path, "job_id": status.job_id}

        version_dir = Path(config.dataset_root) / config.dataset_version
        manifest = _load_manifest(version_dir.parent, config.dataset_version)
        train_rows = [row for row in manifest if row.get("split") == "train"]
        val_rows = [row for row in manifest if row.get("split") == "val"]

        if config.model_name.lower().startswith("resnet"):
            from .training import run_training

            target = Path(config.output_dir) / f"{config.model_name}_{config.dataset_version}_candidate.pt"
            artifact_dir = Path(config.output_dir)
            artifact_dir.mkdir(parents=True, exist_ok=True)
            training_started = time.time()
            run_training(
                f"{config.model_name}_{config.dataset_version}",
                [(row["storage_path"], 1 if row["human_label"] == "ai" else 0) for row in train_rows],
                [(row["storage_path"], 1 if row["human_label"] == "ai" else 0) for row in val_rows],
                epochs=config.epochs,
                pretrained=bool(config.init_checkpoint is None),
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
                work_dir=str(artifact_dir),
            )
            # Prefer the best checkpoint created by the underlying training loop.
            best_path = artifact_dir / f"{config.model_name}_{config.dataset_version}_best.pt"
            if not best_path.exists():
                raise FileNotFoundError(f"Expected best checkpoint was not created: {best_path}")
            artifact_hash = _sha256_file(best_path)
            status.artifact_path = str(best_path)
            status.artifact_hash = artifact_hash
            status.best_metric = 0.0
            status.status = "completed"
            status.progress = 1.0
            status.finished_at = datetime.now(timezone.utc).isoformat()
            write_training_status(status_path, status)
            return {
                "status": "completed",
                "job_id": status.job_id,
                "artifact_path": str(best_path),
                "artifact_hash": artifact_hash,
                "dataset_version": config.dataset_version,
                "model_name": config.model_name,
                "plan": plan,
            }

        raise ValueError(f"Model '{config.model_name}' is not yet supported for retraining in this pipeline.")
    except Exception as exc:  # pragma: no cover
        status.status = "failed"
        status.last_error = str(exc)
        status.finished_at = datetime.now(timezone.utc).isoformat()
        write_training_status(status_path, status)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a candidate model from a versioned dataset without mutating the active model.")
    parser.add_argument("--dataset-version", required=True, help="Dataset version to train from, e.g. v0001")
    parser.add_argument("--dataset-root", default="datasets", help="Root path containing version directories")
    parser.add_argument("--model-name", default="resnet50", choices=["resnet50", "clip"], help="Model architecture to train")
    parser.add_argument("--output-dir", default="artifacts/training", help="Directory for candidate checkpoints and logs")
    parser.add_argument("--init-checkpoint", default=None, help="Optional initialization checkpoint for warm starts")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true", help="Validate the config and dataset without starting training")
    parser.add_argument("--admin-confirmed", action="store_true", help="Required explicit confirmation before training starts")
    parser.add_argument("--status-path", default=None, help="JSON status file for the training run")
    parser.add_argument("--max-memory-gb", type=float, default=12.0, help="Abort if available memory is below this threshold")
    args = parser.parse_args()

    config = TrainingConfig(
        dataset_version=args.dataset_version,
        dataset_root=args.dataset_root,
        model_name=args.model_name,
        output_dir=args.output_dir,
        init_checkpoint=args.init_checkpoint,
        admin_confirmed=args.admin_confirmed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
        max_memory_gb=args.max_memory_gb,
        dry_run=args.dry_run,
        status_path=args.status_path,
    )
    result = run_training_pipeline(config)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
