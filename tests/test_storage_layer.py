import io
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from v1.storage.database import Base, init_db
from v1.storage.models import HumanReview, ImageRecord, Prediction, ReviewStatus
from v1.storage.service import ImageStorageService


@pytest.fixture
def session_db(tmp_path):
    db_path = tmp_path / "test_storage.db"
    engine = create_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(bind=engine)
    session = Session(engine)
    yield session, engine
    session.close()


def _make_png_bytes(width=32, height=32, color=(255, 0, 0)):
    from PIL import Image

    image = Image.new("RGB", (width, height), color=color)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_image_record_id_and_file_metadata_are_persisted(session_db):
    session, engine = session_db
    service = ImageStorageService(session_factory=lambda: session, storage_dir=str(tmp_path := Path("/tmp/ai_detector_storage_test")))
    image_bytes = _make_png_bytes()
    record = service.store_image(image_bytes, "sample.png", content_type="image/png", source="manual")

    assert record.id
    assert record.file_size > 0
    assert record.width == 32
    assert record.height == 32
    assert record.file_format == "PNG"
    assert record.sha256
    assert record.perceptual_hash
    assert record.review_status == ReviewStatus.PENDING


def test_duplicate_upload_is_detected_by_hash(session_db):
    session, engine = session_db
    storage_dir = Path("/tmp/ai_detector_storage_test_duplicates")
    service = ImageStorageService(session_factory=lambda: session, storage_dir=str(storage_dir))
    payload = _make_png_bytes()

    first = service.store_image(payload, "sample.png", content_type="image/png", source="manual")
    second = service.store_image(payload, "sample_copy.png", content_type="image/png", source="manual")

    assert first.id == second.id
    assert session.query(ImageRecord).count() == 1


def test_invalid_file_is_rejected(session_db):
    session, engine = session_db
    service = ImageStorageService(session_factory=lambda: session, storage_dir=str(Path("/tmp/ai_detector_storage_invalid")))

    with pytest.raises(ValueError):
        service.store_image(b"not-a-real-image", "bad.bin", content_type="application/octet-stream", source="manual")

    with pytest.raises(ValueError):
        service.store_image(b"x" * 2000000, "too_big.png", content_type="image/png", source="manual")


def test_prediction_and_review_history_are_preserved(session_db):
    session, engine = session_db
    storage_dir = Path("/tmp/ai_detector_storage_reviews")
    service = ImageStorageService(session_factory=lambda: session, storage_dir=str(storage_dir))
    payload = _make_png_bytes()

    image = service.store_image(payload, "reviewed.png", content_type="image/png", source="manual")
    pred1 = service.add_prediction(
        image_id=image.id,
        model_name="resnet50",
        model_version="v1",
        predicted_class="ai",
        raw_score=0.91,
        threshold=0.5,
    )
    review1 = service.add_review(
        image_id=image.id,
        prediction_id=pred1.id,
        reviewer_id="reviewer-1",
        verified_class="real",
        known_generator="sd14",
        evidence_source="manual_inspection",
        source_notes="The image appears photographic and not synthetic.",
        confidence=0.86,
        review_status=ReviewStatus.APPROVED,
    )
    pred2 = service.add_prediction(
        image_id=image.id,
        model_name="clip",
        model_version="v2",
        predicted_class="ai",
        raw_score=0.78,
        threshold=0.5,
    )
    review2 = service.add_review(
        image_id=image.id,
        prediction_id=pred2.id,
        reviewer_id="reviewer-2",
        verified_class="real",
        known_generator="sd14",
        evidence_source="external_metadata",
        source_notes="A second review confirms the label.",
        confidence=0.92,
        review_status=ReviewStatus.APPROVED,
    )

    assert session.query(Prediction).filter_by(image_id=image.id).count() == 2
    assert session.query(HumanReview).filter_by(image_id=image.id).count() == 2
    assert pred1.id != pred2.id
    assert review1.id != review2.id
    assert image.review_status == ReviewStatus.PENDING
