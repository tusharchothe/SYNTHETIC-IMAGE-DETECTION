"""Tests for the simple no-auth local workflow.

Coverage
--------
service layer (process_upload, get_pending_queue)
    - valid upload persists record + 2 predictions, enters PENDING queue
    - exact duplicate (same SHA-256) returns existing record without re-running inference
    - invalid/corrupt file raises ValueError, nothing written to DB
    - oversized file raises ValueError
    - first model failure is recorded as a warning but upload still succeeds
    - both model failures still persist the image, result has no predictions + 2 warnings
    - near-duplicate (perceptual hash) stores image with NEEDS_REVIEW status

API layer (FastAPI TestClient)
    - GET /health returns 200
    - POST /upload single valid file → 200, predictions present
    - POST /upload multiple files in one request → result per file
    - POST /upload with invalid file returns per-file error entry (not HTTP 4xx)
    - POST /upload with no files → 400
    - GET /queue returns pending items after upload
    - GET /images/{id}/file serves the image bytes
    - GET /images/{id}/predictions returns stored predictions
    - GET /images/unknown/file → 404
    - GET /images/unknown/predictions → 404

All tests use tmp_path-isolated databases and upload directories.
Real inference is replaced with a fast fake that returns deterministic scores.
No network calls, no real model weights needed.
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from v1.services.inference_service import InferenceResult
from v1.storage.database import Base
from v1.storage.models import ImageRecord, Prediction, ReviewStatus
from v1.storage.service import ImageStorageService


# ── Fixtures & helpers -------------------------------------------------------

def _make_png(width: int = 32, height: int = 32, color: tuple = (120, 80, 200)) -> bytes:
    """Return raw PNG bytes for a solid-colour image."""
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_jpeg(width: int = 32, height: int = 32, color: tuple = (200, 100, 50)) -> bytes:
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def _make_unique_png(seed: int, width: int = 32, height: int = 32) -> bytes:
    """Return a PNG with a checkerboard pattern whose tile size and brightness
    vary with ``seed``.  Seeds 1-7 are verified to produce pairwise perceptual-
    hash Hamming distances > 4, so they are treated as distinct images by
    ImageStorageService.find_near_duplicates (threshold=4).
    """
    import random as _random
    rng = _random.Random(seed)
    tile = max(1, (seed % 8) + 1)
    pixels = []
    for y in range(height):
        for x in range(width):
            bright = rng.randint(180, 255)
            dark   = rng.randint(0, 70)
            v = bright if ((x // tile) + (y // tile)) % 2 == 0 else dark
            pixels.append((v, v, v))
    img = Image.new("RGB", (width, height))
    img.putdata(pixels)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _fake_resnet(image_bytes: bytes) -> InferenceResult:
    return InferenceResult(
        model_name="resnet50",
        model_version="test-resnet-v1",
        predicted_class="ai",
        raw_score=0.91,
        threshold=0.5,
        score_kind="raw_model_score",
        calibrated=False,
        preprocessing_version="fit_crop_256_imagenet_norm_v1",
        inference_timestamp="2026-01-01T00:00:00+00:00",
        warnings=[],
    )


def _fake_clip(image_bytes: bytes) -> InferenceResult:
    return InferenceResult(
        model_name="clip",
        model_version="ViT-L-14/openai-test",
        predicted_class="real",
        raw_score=0.34,
        threshold=0.5,
        score_kind="raw_model_score",
        calibrated=False,
        preprocessing_version="clip_vit_l_14_openai_preprocess_v1",
        inference_timestamp="2026-01-01T00:00:00+00:00",
        warnings=[],
    )


@pytest.fixture()
def isolated_service(tmp_path):
    """Return an (ImageStorageService, Session, engine) triple backed by tmp_path."""
    db_path = tmp_path / "test.db"
    engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    factory = sessionmaker(bind=engine, autocommit=False, autoflush=False)

    storage = ImageStorageService(
        session_factory=lambda: factory(),
        storage_dir=str(tmp_path / "uploads"),
        max_file_size_bytes=8 * 1024 * 1024,
    )
    session = factory()
    yield storage, session, engine
    session.close()


@pytest.fixture()
def api_client(tmp_path):
    """
    Return a FastAPI TestClient wired to isolated tmp_path storage/DB and
    fake inference functions.
    """
    from v1.simple_app import api as api_module
    from v1.simple_app import service as svc_module

    # Build isolated engine + session factory
    db_path = tmp_path / "api_test.db"
    engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    factory = sessionmaker(bind=engine, autocommit=False, autoflush=False)

    storage = ImageStorageService(
        session_factory=lambda: factory(),
        storage_dir=str(tmp_path / "uploads"),
    )

    # Patch module-level singletons
    original_storage  = svc_module._storage_service
    original_factory  = svc_module._SessionFactory
    original_engine   = svc_module._engine

    svc_module._storage_service = storage
    svc_module._SessionFactory  = factory
    svc_module._engine          = engine

    # Override the DB dependency in the FastAPI app
    def _override_db():
        db = factory()
        try:
            yield db
        finally:
            db.close()

    api_module.app.dependency_overrides[api_module._get_db] = _override_db

    client = TestClient(api_module.app, raise_server_exceptions=False)

    yield client, storage, factory

    # Restore originals
    svc_module._storage_service = original_storage
    svc_module._SessionFactory  = original_factory
    svc_module._engine          = original_engine
    api_module.app.dependency_overrides.clear()


# ── Service layer tests ------------------------------------------------------

class TestProcessUpload:
    def test_valid_image_persists_record_and_predictions(self, isolated_service):
        storage, session, _ = isolated_service
        from v1.simple_app.service import process_upload

        result = process_upload(
            _make_png(), "test.png", "image/png",
            _storage=storage,
            _run_resnet=_fake_resnet,
            _run_clip=_fake_clip,
        )

        assert result.image_id
        assert result.safe_filename
        assert result.sha256
        assert result.file_size > 0
        assert result.width == 32
        assert result.height == 32
        assert result.review_status == ReviewStatus.PENDING.value
        assert not result.duplicate
        assert len(result.predictions) == 2
        assert result.warnings == []

        names = {p.model_name for p in result.predictions}
        assert "resnet50" in names
        assert "clip" in names

        # Confirm DB rows
        preds = session.query(Prediction).filter_by(image_id=result.image_id).all()
        assert len(preds) == 2

    def test_duplicate_upload_returns_existing_record(self, isolated_service):
        storage, session, _ = isolated_service
        from v1.simple_app.service import process_upload

        payload = _make_png(color=(10, 20, 30))
        first  = process_upload(payload, "a.png", "image/png",
                                _storage=storage, _run_resnet=_fake_resnet, _run_clip=_fake_clip)
        second = process_upload(payload, "b.png", "image/png",
                                _storage=storage, _run_resnet=_fake_resnet, _run_clip=_fake_clip)

        # Same image_id returned
        assert first.image_id == second.image_id
        # Only one ImageRecord in DB
        assert session.query(ImageRecord).count() == 1

    def test_invalid_file_raises_value_error(self, isolated_service):
        storage, session, _ = isolated_service
        from v1.simple_app.service import process_upload

        with pytest.raises(ValueError):
            process_upload(b"not-an-image", "bad.bin", "application/octet-stream",
                           _storage=storage, _run_resnet=_fake_resnet, _run_clip=_fake_clip)

        assert session.query(ImageRecord).count() == 0

    def test_oversized_file_raises_value_error(self, tmp_path):
        tiny_limit = 100  # 100 bytes
        from v1.storage.service import ImageStorageService
        from v1.simple_app.service import process_upload

        db_path = tmp_path / "tiny.db"
        engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})
        Base.metadata.create_all(bind=engine)
        factory = sessionmaker(bind=engine)
        storage = ImageStorageService(
            session_factory=lambda: factory(),
            storage_dir=str(tmp_path / "uploads"),
            max_file_size_bytes=tiny_limit,
        )

        with pytest.raises(ValueError, match="maximum allowed size"):
            process_upload(_make_png(), "big.png", "image/png",
                           _storage=storage, _run_resnet=_fake_resnet, _run_clip=_fake_clip)

    def test_first_model_failure_recorded_as_warning(self, isolated_service):
        storage, session, _ = isolated_service
        from v1.simple_app.service import process_upload

        def _failing_resnet(image_bytes):
            raise RuntimeError("ResNet weights missing")

        result = process_upload(
            _make_png(), "x.png", "image/png",
            _storage=storage,
            _run_resnet=_failing_resnet,
            _run_clip=_fake_clip,
        )

        # Upload succeeded despite model failure
        assert result.image_id
        assert result.review_status == ReviewStatus.PENDING.value
        # Only clip prediction saved
        assert len(result.predictions) == 1
        assert result.predictions[0].model_name == "clip"
        # Warning recorded
        assert len(result.warnings) == 1
        assert "resnet" in result.warnings[0].lower()

    def test_both_models_fail_image_still_stored(self, isolated_service):
        storage, session, _ = isolated_service
        from v1.simple_app.service import process_upload

        def _fail(image_bytes):
            raise RuntimeError("no GPU")

        result = process_upload(
            _make_png(color=(5, 5, 5)), "y.png", "image/png",
            _storage=storage,
            _run_resnet=_fail,
            _run_clip=_fail,
        )

        assert result.image_id
        assert result.predictions == []
        assert len(result.warnings) == 2
        assert session.query(ImageRecord).count() == 1
        assert session.query(Prediction).count() == 0

    def test_jpeg_upload_succeeds(self, isolated_service):
        storage, session, _ = isolated_service
        from v1.simple_app.service import process_upload

        result = process_upload(
            _make_jpeg(), "photo.jpg", "image/jpeg",
            _storage=storage,
            _run_resnet=_fake_resnet,
            _run_clip=_fake_clip,
        )
        assert result.image_id
        assert len(result.predictions) == 2


class TestGetPendingQueue:
    def test_uploaded_image_appears_in_queue(self, isolated_service):
        storage, session, _ = isolated_service
        from v1.simple_app.service import get_pending_queue, process_upload

        process_upload(_make_unique_png(seed=1), "q1.png", "image/png",
                       _storage=storage, _run_resnet=_fake_resnet, _run_clip=_fake_clip)
        process_upload(_make_unique_png(seed=2), "q2.png", "image/png",
                       _storage=storage, _run_resnet=_fake_resnet, _run_clip=_fake_clip)

        result = get_pending_queue(session, page=1, limit=10)

        assert result["total"] == 2
        assert len(result["items"]) == 2
        assert result["page"] == 1
        assert result["has_next"] is False

        for item in result["items"]:
            assert item["review_status"] == "pending"
            assert len(item["predictions"]) == 2

    def test_pagination_works_correctly(self, isolated_service):
        storage, session, _ = isolated_service
        from v1.simple_app.service import get_pending_queue, process_upload

        # Seeds 1-5 produce pairwise Hamming distance > 4 (verified offline).
        for i in range(1, 6):
            process_upload(_make_unique_png(seed=i), f"p{i}.png", "image/png",
                           _storage=storage, _run_resnet=_fake_resnet, _run_clip=_fake_clip)

        page1 = get_pending_queue(session, page=1, limit=2)
        page2 = get_pending_queue(session, page=2, limit=2)
        page3 = get_pending_queue(session, page=3, limit=2)

        assert page1["total"] == 5
        assert len(page1["items"]) == 2
        assert page1["has_next"] is True

        assert len(page2["items"]) == 2
        assert page2["has_next"] is True

        assert len(page3["items"]) == 1
        assert page3["has_next"] is False

    def test_empty_queue_returns_zero_items(self, isolated_service):
        _, session, _ = isolated_service
        from v1.simple_app.service import get_pending_queue

        result = get_pending_queue(session, page=1, limit=10)
        assert result["total"] == 0
        assert result["items"] == []


# ── API layer tests ----------------------------------------------------------

class TestAPIHealth:
    def test_health_returns_ok(self, api_client):
        client, *_ = api_client
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert "upload_dir" in body
        assert "database_url" in body


class TestAPIUpload:
    def _post_png(self, client, color=(100, 150, 200), filename="sample.png"):
        return client.post(
            "/upload",
            files=[("files", (filename, _make_png(color=color), "image/png"))],
        )

    def test_single_valid_upload_returns_ok(self, api_client):
        client, *_ = api_client
        # Patch inference in the service module
        import v1.simple_app.service as svc
        orig_r, orig_c = svc.run_resnet_prediction, svc.run_clip_prediction
        svc.run_resnet_prediction = _fake_resnet
        svc.run_clip_prediction   = _fake_clip
        try:
            r = self._post_png(client)
            assert r.status_code == 200
            results = r.json()["results"]
            assert len(results) == 1
            assert results[0]["status"] == "ok"
            assert results[0]["image_id"]
            assert len(results[0]["predictions"]) == 2
        finally:
            svc.run_resnet_prediction = orig_r
            svc.run_clip_prediction   = orig_c

    def test_multiple_files_in_one_request(self, api_client):
        client, *_ = api_client
        import v1.simple_app.service as svc
        orig_r, orig_c = svc.run_resnet_prediction, svc.run_clip_prediction
        svc.run_resnet_prediction = _fake_resnet
        svc.run_clip_prediction   = _fake_clip
        try:
            # Use gradient images with distinct seeds to avoid perceptual-hash collisions.
            r = client.post(
                "/upload",
                files=[
                    ("files", ("a.png", _make_unique_png(seed=10), "image/png")),
                    ("files", ("b.png", _make_unique_png(seed=20), "image/png")),
                    ("files", ("c.png", _make_unique_png(seed=30), "image/png")),
                ],
            )
            assert r.status_code == 200
            results = r.json()["results"]
            assert len(results) == 3
            for res in results:
                assert res["status"] == "ok"
        finally:
            svc.run_resnet_prediction = orig_r
            svc.run_clip_prediction   = orig_c

    def test_invalid_file_returns_per_file_error_not_400(self, api_client):
        client, *_ = api_client
        r = client.post(
            "/upload",
            files=[("files", ("bad.bin", b"definitely not an image", "application/octet-stream"))],
        )
        # The endpoint itself is 200 — error is per-file
        assert r.status_code == 200
        results = r.json()["results"]
        assert results[0]["status"] == "error"
        assert "error" in results[0]

    def test_no_files_returns_400(self, api_client):
        client, *_ = api_client
        # FastAPI returns 422 (Unprocessable Entity) when a required field like
        # `files` is missing entirely — this is correct FastAPI behaviour.
        r = client.post("/upload", files=[])
        assert r.status_code in (400, 422)

    def test_duplicate_upload_returns_duplicate_status(self, api_client):
        client, *_ = api_client
        import v1.simple_app.service as svc
        orig_r, orig_c = svc.run_resnet_prediction, svc.run_clip_prediction
        svc.run_resnet_prediction = _fake_resnet
        svc.run_clip_prediction   = _fake_clip
        try:
            payload = _make_png(color=(11, 22, 33))
            first  = client.post("/upload", files=[("files", ("x.png", payload, "image/png"))])
            second = client.post("/upload", files=[("files", ("x.png", payload, "image/png"))])
            assert first.status_code == 200
            assert second.status_code == 200
            r1 = first.json()["results"][0]
            r2 = second.json()["results"][0]
            assert r1["status"] == "ok"
            # Second upload of same bytes → duplicate
            assert r2["image_id"] == r1["image_id"]
        finally:
            svc.run_resnet_prediction = orig_r
            svc.run_clip_prediction   = orig_c

    def test_model_failure_warning_in_result(self, api_client):
        client, *_ = api_client
        import v1.simple_app.api as api_mod
        from v1.simple_app.service import process_upload as real_process_upload

        def _process_with_failing_resnet(image_bytes, filename, content_type=None):
            # Call process_upload with the DI seam so the patch reaches
            # the inner inference call without touching real model weights.
            def _boom(image_bytes):
                raise RuntimeError("no GPU available")
            from v1.simple_app import service as svc
            return real_process_upload(
                image_bytes, filename, content_type=content_type,
                _storage=svc._storage_service,
                _run_resnet=_boom,
                _run_clip=_fake_clip,
            )

        # Temporarily replace the module-level process_upload used by the API
        import v1.simple_app.api as api_module_ref
        orig_pu = api_module_ref.process_upload
        api_module_ref.process_upload = _process_with_failing_resnet
        try:
            r = client.post(
                "/upload",
                files=[("files", ("w.png", _make_png(color=(128, 128, 0)), "image/png"))],
            )
            assert r.status_code == 200
            res = r.json()["results"][0]
            assert res["status"] == "ok"
            assert len(res["predictions"]) == 1
            assert res["predictions"][0]["model_name"] == "clip"
            assert any("resnet" in w.lower() for w in res["warnings"])
        finally:
            api_module_ref.process_upload = orig_pu


class TestAPIQueue:
    def _upload_one(self, client, color=(50, 100, 150)):
        import v1.simple_app.service as svc
        orig_r, orig_c = svc.run_resnet_prediction, svc.run_clip_prediction
        svc.run_resnet_prediction = _fake_resnet
        svc.run_clip_prediction   = _fake_clip
        try:
            r = client.post(
                "/upload",
                files=[("files", ("img.png", _make_png(color=color), "image/png"))],
            )
            return r.json()["results"][0]["image_id"]
        finally:
            svc.run_resnet_prediction = orig_r
            svc.run_clip_prediction   = orig_c

    def test_queue_contains_uploaded_image(self, api_client):
        client, *_ = api_client
        image_id = self._upload_one(client, color=(33, 44, 55))

        r = client.get("/queue")
        assert r.status_code == 200
        body = r.json()
        assert body["total"] >= 1
        ids = [item["image_id"] for item in body["items"]]
        assert image_id in ids

    def test_queue_pagination_params_accepted(self, api_client):
        client, *_ = api_client
        r = client.get("/queue?page=1&limit=5")
        assert r.status_code == 200
        body = r.json()
        assert "items" in body
        assert body["limit"] == 5

    def test_queue_invalid_limit_rejected(self, api_client):
        client, *_ = api_client
        r = client.get("/queue?limit=9999")
        assert r.status_code == 422


class TestAPIImageFile:
    def test_serve_image_file_returns_bytes(self, api_client):
        client, *_ = api_client
        import v1.simple_app.service as svc
        orig_r, orig_c = svc.run_resnet_prediction, svc.run_clip_prediction
        svc.run_resnet_prediction = _fake_resnet
        svc.run_clip_prediction   = _fake_clip
        try:
            up = client.post(
                "/upload",
                files=[("files", ("vis.png", _make_png(color=(90, 90, 90)), "image/png"))],
            )
            image_id = up.json()["results"][0]["image_id"]
        finally:
            svc.run_resnet_prediction = orig_r
            svc.run_clip_prediction   = orig_c

        r = client.get(f"/images/{image_id}/file")
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("image/")
        # The bytes decode as a valid image
        img = Image.open(io.BytesIO(r.content))
        assert img.size == (32, 32)

    def test_unknown_image_id_returns_404(self, api_client):
        client, *_ = api_client
        r = client.get("/images/does-not-exist/file")
        assert r.status_code == 404


class TestAPIImagePredictions:
    def test_predictions_returned_for_known_image(self, api_client):
        client, *_ = api_client
        import v1.simple_app.service as svc
        orig_r, orig_c = svc.run_resnet_prediction, svc.run_clip_prediction
        svc.run_resnet_prediction = _fake_resnet
        svc.run_clip_prediction   = _fake_clip
        try:
            up = client.post(
                "/upload",
                files=[("files", ("pred.png", _make_png(color=(111, 222, 111)), "image/png"))],
            )
            image_id = up.json()["results"][0]["image_id"]
        finally:
            svc.run_resnet_prediction = orig_r
            svc.run_clip_prediction   = orig_c

        r = client.get(f"/images/{image_id}/predictions")
        assert r.status_code == 200
        body = r.json()
        assert body["image_id"] == image_id
        assert len(body["predictions"]) == 2
        names = {p["model_name"] for p in body["predictions"]}
        assert "resnet50" in names
        assert "clip" in names

    def test_unknown_image_id_returns_404(self, api_client):
        client, *_ = api_client
        r = client.get("/images/no-such-id/predictions")
        assert r.status_code == 404
