import io

from fastapi.testclient import TestClient
from PIL import Image
from sqlalchemy import create_engine

import v1.api as api
from v1.storage.database import Base, SessionLocal
from v1.storage.service import ImageStorageService


def _png_bytes(size=(32, 32), color=(255, 0, 0)):
    image = Image.new("RGB", size=size, color=color)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _fake_inference(model_name, image_bytes):
    from v1.services.inference_service import InferenceResult

    score = 0.91 if model_name == "resnet" else 0.74
    return InferenceResult(
        model_name="resnet50" if model_name == "resnet" else "clip",
        model_version="review-test-v1",
        predicted_class="ai" if score >= 0.5 else "real",
        raw_score=score,
        threshold=0.5,
        score_kind="raw_model_score",
        calibrated=False,
        preprocessing_version="test-preprocess-v1",
        inference_timestamp="2026-01-01T00:00:00Z",
        warnings=[],
    )


def test_review_queue_and_submission_work(tmp_path, monkeypatch):
    monkeypatch.setenv("AIDETECT_ALLOWED_REVIEWERS", "reviewer-1")
    monkeypatch.setenv("AIDETECT_REVIEWER_TOKEN", "secret")
    engine = create_engine(f"sqlite:///{tmp_path / 'review_queue.db'}")
    Base.metadata.create_all(bind=engine)
    SessionLocal.configure(bind=engine)

    storage = ImageStorageService(session_factory=lambda: api.SessionLocal(), storage_dir=str(tmp_path / "storage"))
    monkeypatch.setattr(api, "storage_service", storage)
    monkeypatch.setattr(api, "run_prediction_for_model", _fake_inference)

    client = TestClient(api.app)
    upload = client.post(
        "/images/upload",
        files={"file": ("review_sample.png", _png_bytes(), "image/png")},
        data={"source": "manual"},
    )
    image_id = upload.json()["image_id"]

    queue = client.get(
        "/reviews/queue",
        params={"page": 1, "limit": 10, "status": "pending"},
        headers={"X-Reviewer-Id": "reviewer-1"},
    )
    assert queue.status_code == 200
    payload = queue.json()
    assert payload["total"] >= 1

    submit = client.post(
        "/reviews/submit",
        json={
            "image_id": image_id,
            "decision": "real",
            "confidence": 0.84,
            "evidence_source": "manual_inspection",
            "source_notes": "The image looks authentic.",
        },
        headers={"X-Reviewer-Id": "reviewer-1", "Authorization": "Bearer secret"},
    )
    assert submit.status_code == 200
    body = submit.json()
    assert body["decision"] == "real"

    duplicate = client.post(
        "/reviews/submit",
        json={
            "image_id": image_id,
            "decision": "real",
            "confidence": 0.84,
            "evidence_source": "manual_inspection",
            "source_notes": "The image looks authentic.",
        },
        headers={"X-Reviewer-Id": "reviewer-1", "Authorization": "Bearer secret"},
    )
    assert duplicate.status_code == 409
