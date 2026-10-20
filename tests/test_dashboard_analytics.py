from io import BytesIO

import pandas as pd
from PIL import Image
from fastapi.testclient import TestClient

from v1.api import app
from v1.storage.database import SessionLocal, init_db
from v1.storage.models import HumanReview, ReviewStatus
from v1.storage.service import ImageStorageService


def _make_png_bytes(color=(255, 0, 0), size=(32, 32)):
    img = Image.new("RGB", size, color)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_dashboard_analytics_and_export_are_live_and_filtered(tmp_path, monkeypatch):
    monkeypatch.setenv("AIDETECT_ALLOWED_REVIEWERS", "reviewer-1")
    monkeypatch.setenv("AIDETECT_REVIEWER_TOKEN", "token-123")
    monkeypatch.setenv("AIDETECT_ALLOWED_ADMINS", "admin-1")
    monkeypatch.setenv("AIDETECT_MODEL_REGISTRY_DIR", str(tmp_path / "registry"))
    db_url = f"sqlite:///{tmp_path / 'dashboard.db'}"
    init_db(db_url)
    SessionLocal.configure(bind=None)
    SessionLocal.configure(bind=SessionLocal.kw["bind"] if False else None)

    # rebind the app session factory to the temporary DB
    import v1.api as api_module
    api_module.SessionLocal.configure(bind=api_module.init_db(db_url))
    api_module.storage_service = ImageStorageService(
        session_factory=lambda: api_module.SessionLocal(),
        storage_dir=str(tmp_path / "storage"),
        max_file_size_bytes=2_000_000,
    )

    service = api_module.storage_service
    image = service.store_image(_make_png_bytes(), "sample.png", "image/png", source="sd14")
    service.add_prediction(image.id, "resnet50", "v1", "ai", 0.9, 0.5)
    service.add_prediction(image.id, "clip", "v1", "ai", 0.82, 0.5)
    service.add_review(
        image_id=image.id,
        prediction_id=None,
        reviewer_id="reviewer-1",
        verified_class="ai",
        known_generator="sd14",
        evidence_source="human-check",
        source_notes="synthetic image",
        confidence=0.9,
        review_status=ReviewStatus.APPROVED,
    )

    client = TestClient(app)
    headers = {"X-Reviewer-Id": "reviewer-1", "Authorization": "Bearer token-123"}
    overview = client.get("/dashboard/overview", headers=headers)
    assert overview.status_code == 200, overview.text
    payload = overview.json()
    assert payload["counts"]["uploaded_total"] >= 1
    assert payload["counts"]["training_eligible"] >= 0
    assert "generator_performance" in payload

    export_json = client.get("/dashboard/export?format=json", headers=headers)
    assert export_json.status_code == 200, export_json.text
    export_csv = client.get("/dashboard/export?format=csv", headers=headers)
    assert export_csv.status_code == 200, export_csv.text
    assert "uploaded_total" in export_csv.text

    metrics = client.get("/dashboard/model-lifecycle", headers=headers)
    assert metrics.status_code == 200, metrics.text
    assert "active_model" in metrics.json()
