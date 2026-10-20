import io

from fastapi.testclient import TestClient
from PIL import Image
from sqlalchemy import create_engine

import v1.api as api
from v1.storage.database import Base, SessionLocal
from v1.storage.models import ReviewStatus
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
        model_name=model_name,
        model_version="test-version",
        predicted_class="ai" if score >= 0.5 else "real",
        raw_score=score,
        threshold=0.5,
        score_kind="raw_model_score",
        calibrated=False,
        preprocessing_version="test-preprocess-v1",
        inference_timestamp="2026-01-01T00:00:00Z",
        warnings=[],
    )


def test_upload_image_persists_and_returns_predictions(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'api_upload.db'}")
    Base.metadata.create_all(bind=engine)
    SessionLocal.configure(bind=engine)

    storage = ImageStorageService(session_factory=lambda: api.SessionLocal(), storage_dir=str(tmp_path / "storage"))
    monkeypatch.setattr(api, "storage_service", storage)
    monkeypatch.setattr(api, "run_prediction_for_model", _fake_inference)

    client = TestClient(api.app)
    payload = _png_bytes()
    response = client.post(
        "/images/upload",
        files={"file": ("sample.png", payload, "image/png")},
        data={"source": "manual"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["image_id"]
    assert body["review_status"] == ReviewStatus.PENDING.value
    assert len(body["predictions"]) == 2
    assert body["predictions"][0]["score_kind"] == "raw_model_score"

    history = client.get(f"/images/{body['image_id']}/history")
    assert history.status_code == 200
    history_body = history.json()
    assert len(history_body["predictions"]) == 2

    listed = client.get("/images?page=1&limit=10")
    assert listed.status_code == 200
    assert listed.json()["total"] == 1


def test_upload_rejects_invalid_files(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'api_invalid.db'}")
    Base.metadata.create_all(bind=engine)
    SessionLocal.configure(bind=engine)

    storage = ImageStorageService(session_factory=lambda: api.SessionLocal(), storage_dir=str(tmp_path / "storage"))
    monkeypatch.setattr(api, "storage_service", storage)

    client = TestClient(api.app)
    response = client.post("/images/upload", files={"file": ("bad.bin", b"not-a-real-image", "application/octet-stream")})
    assert response.status_code == 400


def test_model_failure_is_reported_safely(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'api_model_fail.db'}")
    Base.metadata.create_all(bind=engine)
    SessionLocal.configure(bind=engine)

    storage = ImageStorageService(session_factory=lambda: api.SessionLocal(), storage_dir=str(tmp_path / "storage"))
    monkeypatch.setattr(api, "storage_service", storage)

    def failing_inference(model_name, image_bytes):
        raise RuntimeError("mock model failure")

    monkeypatch.setattr(api, "run_prediction_for_model", failing_inference)

    client = TestClient(api.app)
    response = client.post("/images/upload", files={"file": ("sample.png", _png_bytes(), "image/png")})
    assert response.status_code == 200
    body = response.json()
    assert body["predictions"] == []
    assert any("model failed" in warning.lower() for warning in body["input_warnings"])


def test_legacy_predict_endpoint_returns_raw_score_semantics(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'api_legacy.db'}")
    Base.metadata.create_all(bind=engine)
    SessionLocal.configure(bind=engine)
    monkeypatch.setattr(api, "run_prediction_for_model", _fake_inference)

    client = TestClient(api.app)
    response = client.post("/predict", files={"file": ("sample.png", _png_bytes(), "image/png")}, data={"model_type": "resnet"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["score_kind"] == "raw_model_score"
    assert payload["calibrated"] is False
    assert payload["label"] in {"ai", "real"}


def test_gradio_save_to_review_queue_uploads_image(monkeypatch):
    import v1.gradio_app as gradio_app

    captured = {}

    class FakeResponse:
        status_code = 200

        def json(self):
            return {"image_id": "img-123", "review_status": "pending", "predictions": [{"model_name": "resnet"}]}

    def fake_post(url, files=None, data=None, headers=None, timeout=None):
        captured["url"] = url
        captured["files"] = files
        captured["data"] = data
        captured["headers"] = headers
        return FakeResponse()

    monkeypatch.setattr(gradio_app.httpx, "post", fake_post)

    image = Image.new("RGB", (32, 32), color=(255, 0, 0))
    message = gradio_app.save_image_to_review_queue(
        "http://localhost:8000",
        "reviewer-1",
        "241005",
        image,
        source="manual",
    )

    assert message == "Image saved to review queue: img-123"
    assert captured["url"] == "http://localhost:8000/images/upload"
    assert captured["data"] == {"source": "manual"}
    assert captured["headers"]["X-Reviewer-Id"] == "reviewer-1"
    assert captured["headers"]["Authorization"] == "Bearer 241005"
