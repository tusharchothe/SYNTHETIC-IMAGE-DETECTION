import json
from pathlib import Path

import pytest

from v1.training_pipeline import TrainingConfig, detect_model_trainability, validate_dataset_version


def _write_version(tmp_path: Path, version: str, rows):
    base = tmp_path / version
    base.mkdir(parents=True, exist_ok=True)
    (base / "manifest.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    (base / "split_definitions.json").write_text(json.dumps({"splits": {"train": 1, "val": 1, "test": 1}}, indent=2), encoding="utf-8")
    return base


def test_detect_model_trainability_distinguishes_resnet_and_clip(tmp_path):
    assert detect_model_trainability("resnet50") == "trainable"
    assert detect_model_trainability("clip") == "head_only"


def test_validate_dataset_version_rejects_missing_training_labels(tmp_path):
    row = {
        "image_id": "img-1",
        "storage_path": str(tmp_path / "img.png"),
        "human_label": "real",
        "split": "train",
    }
    version_dir = _write_version(tmp_path, "v0001", [row])
    config = TrainingConfig(dataset_version="v0001", dataset_root=str(tmp_path), model_name="resnet50", admin_confirmed=True)
    with pytest.raises(ValueError):
        validate_dataset_version(config, version_dir)


def test_validate_dataset_version_requires_train_and_validation_split(tmp_path):
    rows = [
        {"image_id": "img-1", "storage_path": str(tmp_path / "img1.png"), "human_label": "real", "split": "train"},
        {"image_id": "img-2", "storage_path": str(tmp_path / "img2.png"), "human_label": "ai", "split": "train"},
    ]
    _write_version(tmp_path, "v0001", rows)
    config = TrainingConfig(dataset_version="v0001", dataset_root=str(tmp_path), model_name="resnet50", admin_confirmed=True)
    with pytest.raises(ValueError):
        validate_dataset_version(config, tmp_path / "v0001")


def test_training_config_requires_explicit_admin_action():
    config = TrainingConfig(dataset_version="v0001", dataset_root="datasets", model_name="resnet50", admin_confirmed=False)
    with pytest.raises(PermissionError):
        config.ensure_admin_authorized()
