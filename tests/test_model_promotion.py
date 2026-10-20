import json
from pathlib import Path

import numpy as np
import pytest

from v1.evaluation_pipeline import EvaluationCriteria, evaluate_candidate_scores
from v1.model_registry import ModelRegistry


def _make_manifest(tmp_path: Path):
    manifest = [
        {"image_id": "a1", "storage_path": str(tmp_path / "a1.png"), "human_label": "real", "generator": "sd14", "source": "authentic_a", "transformation": "none", "quality_category": "standard"},
        {"image_id": "a2", "storage_path": str(tmp_path / "a2.png"), "human_label": "real", "generator": "sd14", "source": "authentic_a", "transformation": "none", "quality_category": "standard"},
        {"image_id": "a3", "storage_path": str(tmp_path / "a3.png"), "human_label": "ai", "generator": "sd14", "source": "synthetic_a", "transformation": "none", "quality_category": "standard"},
        {"image_id": "a4", "storage_path": str(tmp_path / "a4.png"), "human_label": "ai", "generator": "sd14", "source": "synthetic_a", "transformation": "none", "quality_category": "standard"},
        {"image_id": "b1", "storage_path": str(tmp_path / "b1.png"), "human_label": "real", "generator": "midjourney", "source": "authentic_b", "transformation": "jpeg", "quality_category": "degraded"},
        {"image_id": "b2", "storage_path": str(tmp_path / "b2.png"), "human_label": "real", "generator": "midjourney", "source": "authentic_b", "transformation": "jpeg", "quality_category": "degraded"},
        {"image_id": "b3", "storage_path": str(tmp_path / "b3.png"), "human_label": "ai", "generator": "midjourney", "source": "synthetic_b", "transformation": "jpeg", "quality_category": "degraded"},
        {"image_id": "b4", "storage_path": str(tmp_path / "b4.png"), "human_label": "ai", "generator": "midjourney", "source": "synthetic_b", "transformation": "jpeg", "quality_category": "degraded"},
    ]
    return manifest


def test_evaluate_candidate_scores_rejects_high_authentic_fpr():
    manifest = _make_manifest(Path("/tmp"))
    candidate_scores = np.array([0.95, 0.92, 0.8, 0.7, 0.96, 0.91, 0.74, 0.78])
    production_scores = np.array([0.88, 0.9, 0.82, 0.79, 0.9, 0.86, 0.75, 0.8])
    labels = np.array([0, 0, 1, 1, 0, 0, 1, 1])
    policy = EvaluationCriteria(max_authentic_false_positive_rate=0.1, min_auc=0.9)

    report = evaluate_candidate_scores(manifest, labels, candidate_scores, production_scores, threshold=0.5, criteria=policy)
    assert report["passed"] is False
    assert any("authentic_false_positive_rate" in reason for reason in report["rejection_reasons"])


def test_model_registry_promotes_candidate_after_successful_evaluation(tmp_path):
    model_name = "resnet50"
    registry = ModelRegistry(root_dir=tmp_path / "registry")
    active = tmp_path / "production.pt"
    candidate = tmp_path / "candidate.pt"
    active.write_bytes(b"active-model")
    candidate.write_bytes(b"candidate-model")

    registry.set_active_model(model_name, active)
    manifest = _make_manifest(tmp_path)
    labels = np.array([0, 0, 1, 1, 0, 0, 1, 1])
    candidate_scores = np.array([0.1, 0.15, 0.9, 0.85, 0.2, 0.18, 0.8, 0.9])
    production_scores = np.array([0.12, 0.14, 0.88, 0.84, 0.21, 0.17, 0.78, 0.85])

    report = evaluate_candidate_scores(manifest, labels, candidate_scores, production_scores, threshold=0.5, criteria=EvaluationCriteria())
    record = registry.promote_candidate(
        model_name=model_name,
        candidate_path=candidate,
        evaluation_report=report,
        approver="admin-user",
        authorizer=lambda user: user == "admin-user",
    )

    assert record["approved"] is True
    assert registry.get_active_path(model_name).exists()
    assert registry.get_backup_path(model_name).exists()


def test_model_registry_rejects_candidate_when_guardrail_is_violated(tmp_path):
    model_name = "resnet50"
    registry = ModelRegistry(root_dir=tmp_path / "registry")
    active = tmp_path / "production.pt"
    candidate = tmp_path / "candidate.pt"
    active.write_bytes(b"active-model")
    candidate.write_bytes(b"candidate-model")
    registry.set_active_model(model_name, active)

    manifest = _make_manifest(tmp_path)
    labels = np.array([0, 0, 1, 1, 0, 0, 1, 1])
    candidate_scores = np.array([0.8, 0.85, 0.6, 0.65, 0.7, 0.72, 0.5, 0.55])
    production_scores = np.array([0.1, 0.09, 0.9, 0.91, 0.12, 0.1, 0.88, 0.9])

    report = evaluate_candidate_scores(manifest, labels, candidate_scores, production_scores, threshold=0.5, criteria=EvaluationCriteria(min_auc=0.95, generator_guardrails={"sd14": {"min_auc": 0.99}}))
    assert report["passed"] is False

    with pytest.raises(ValueError):
        registry.promote_candidate(
            model_name=model_name,
            candidate_path=candidate,
            evaluation_report=report,
            approver="admin-user",
            authorizer=lambda user: user == "admin-user",
        )
