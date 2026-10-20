import io
from pathlib import Path

from PIL import Image
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from v1.storage.database import Base
from v1.storage.models import HumanReview, ImageRecord, Prediction, ReviewStatus
from v1.dataset_builder import build_dataset_version


def _write_png(path: Path, color=(255, 0, 0)):
    image = Image.new("RGB", (16, 16), color=color)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    path.write_bytes(buffer.getvalue())


def _session(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'dataset_builder.db'}")
    Base.metadata.create_all(bind=engine)
    return sessionmaker(bind=engine)()


def test_dataset_builder_excludes_unreviewed_uncertain_and_missing_files(tmp_path):
    session = _session(tmp_path)
    image_dir = tmp_path / "images"
    image_dir.mkdir()

    good_path = image_dir / "good.png"
    _write_png(good_path)
    good = ImageRecord(
        storage_path=str(good_path),
        safe_filename=good_path.name,
        sha256="a" * 64,
        perceptual_hash="0" * 64,
        file_size=good_path.stat().st_size,
        width=16,
        height=16,
        file_format="PNG",
        mime_type="image/png",
        source="source_a",
        review_status=ReviewStatus.APPROVED,
    )
    session.add(good)
    session.flush()

    session.add(Prediction(image_id=good.id, model_name="resnet", model_version="v1", predicted_class="real", raw_score=0.2, threshold=0.5, preprocessing_version="p1"))
    session.add(Prediction(image_id=good.id, model_name="clip", model_version="v1", predicted_class="real", raw_score=0.31, threshold=0.5, preprocessing_version="p1"))
    session.add(HumanReview(image_id=good.id, reviewer_id="r1", verified_class="real", confidence=0.9, review_status=ReviewStatus.APPROVED, evidence_source="manual"))

    uncertain = ImageRecord(
        storage_path=str(image_dir / "uncertain.png"),
        safe_filename="uncertain.png",
        sha256="b" * 64,
        perceptual_hash="1" * 64,
        file_size=10,
        width=16,
        height=16,
        file_format="PNG",
        mime_type="image/png",
        source="source_b",
        review_status=ReviewStatus.NEEDS_REVIEW,
    )
    session.add(uncertain)
    session.flush()
    session.add(HumanReview(image_id=uncertain.id, reviewer_id="r2", verified_class="uncertain", confidence=0.4, review_status=ReviewStatus.NEEDS_REVIEW, evidence_source="manual"))

    rejected = ImageRecord(
        storage_path=str(image_dir / "rejected.png"),
        safe_filename="rejected.png",
        sha256="c" * 64,
        perceptual_hash="2" * 64,
        file_size=12,
        width=16,
        height=16,
        file_format="PNG",
        mime_type="image/png",
        source="source_c",
        review_status=ReviewStatus.REJECTED,
    )
    session.add(rejected)
    session.flush()
    session.add(HumanReview(image_id=rejected.id, reviewer_id="r3", verified_class="real", confidence=0.7, review_status=ReviewStatus.REJECTED, evidence_source="manual"))

    missing = ImageRecord(
        storage_path=str(image_dir / "missing.png"),
        safe_filename="missing.png",
        sha256="d" * 64,
        perceptual_hash="3" * 64,
        file_size=12,
        width=16,
        height=16,
        file_format="PNG",
        mime_type="image/png",
        source="source_d",
        review_status=ReviewStatus.APPROVED,
    )
    session.add(missing)
    session.flush()
    session.add(HumanReview(image_id=missing.id, reviewer_id="r4", verified_class="ai", confidence=0.8, review_status=ReviewStatus.APPROVED, evidence_source="manual"))

    session.commit()
    result = build_dataset_version(session=session, output_dir=tmp_path / "datasets", version="v0001", dry_run=False)

    assert len(result["manifest"]) == 1
    assert result["report"]["excluded_counts"]["uncertain_label"] >= 1
    assert result["report"]["excluded_counts"]["missing_file"] >= 1
    assert result["report"]["excluded_counts"]["rejected_or_ineligible"] >= 1


def test_dataset_builder_deduplicates_identical_and_near_duplicate_images(tmp_path):
    session = _session(tmp_path)
    image_dir = tmp_path / "images"
    image_dir.mkdir()

    first_path = image_dir / "dup_1.png"
    second_path = image_dir / "dup_2.png"
    _write_png(first_path, color=(255, 0, 0))
    _write_png(second_path, color=(255, 0, 0))

    first = ImageRecord(
        storage_path=str(first_path),
        safe_filename=first_path.name,
        sha256="e" * 64,
        perceptual_hash="0" * 64,
        file_size=first_path.stat().st_size,
        width=16,
        height=16,
        file_format="PNG",
        mime_type="image/png",
        source="family_shared",
        review_status=ReviewStatus.APPROVED,
    )
    second = ImageRecord(
        storage_path=str(second_path),
        safe_filename=second_path.name,
        sha256="f" * 64,
        perceptual_hash="0" * 64,
        file_size=second_path.stat().st_size,
        width=16,
        height=16,
        file_format="PNG",
        mime_type="image/png",
        source="family_shared",
        review_status=ReviewStatus.APPROVED,
    )
    session.add_all([first, second])
    session.flush()
    session.add(Prediction(image_id=first.id, model_name="resnet", model_version="v1", predicted_class="ai", raw_score=0.9, threshold=0.5, preprocessing_version="p1"))
    session.add(Prediction(image_id=second.id, model_name="resnet", model_version="v1", predicted_class="ai", raw_score=0.9, threshold=0.5, preprocessing_version="p1"))
    session.add(HumanReview(image_id=first.id, reviewer_id="r1", verified_class="ai", confidence=0.9, review_status=ReviewStatus.APPROVED, evidence_source="manual"))
    session.add(HumanReview(image_id=second.id, reviewer_id="r2", verified_class="ai", confidence=0.9, review_status=ReviewStatus.APPROVED, evidence_source="manual"))
    session.commit()

    result = build_dataset_version(session=session, output_dir=tmp_path / "datasets", version="v0001", dry_run=False)

    assert result["report"]["duplicate_total"] >= 1
    assert len(result["manifest"]) == 1


def test_dataset_builder_enforces_group_leakage_prevention(tmp_path):
    session = _session(tmp_path)
    image_dir = tmp_path / "images"
    image_dir.mkdir()

    for idx in range(4):
        path = image_dir / f"family_{idx}.png"
        _write_png(path)
        rec = ImageRecord(
            storage_path=str(path),
            safe_filename=path.name,
            sha256=f"{idx:064x}" * 1,
            perceptual_hash=f"{idx:064b}",
            file_size=path.stat().st_size,
            width=16,
            height=16,
            file_format="PNG",
            mime_type="image/png",
            source="family_a",
            review_status=ReviewStatus.APPROVED,
        )
        session.add(rec)
        session.flush()
        session.add(Prediction(image_id=rec.id, model_name="resnet", model_version="v1", predicted_class="real", raw_score=0.1, threshold=0.5, preprocessing_version="p1"))
        session.add(HumanReview(image_id=rec.id, reviewer_id=f"r{idx}", verified_class="real", confidence=0.9, review_status=ReviewStatus.APPROVED, evidence_source="manual"))

    session.commit()
    result = build_dataset_version(session=session, output_dir=tmp_path / "datasets", version="v0001", dry_run=False)

    by_group = {}
    for row in result["manifest"]:
        by_group.setdefault(row["group_key"], set()).add(row["split"])
    assert all(len(v) == 1 for v in by_group.values())


def test_dataset_builder_requires_second_review_when_configured(tmp_path):
    session = _session(tmp_path)
    image_dir = tmp_path / "images"
    image_dir.mkdir()

    path = image_dir / "single_review.png"
    _write_png(path)
    rec = ImageRecord(
        storage_path=str(path),
        safe_filename=path.name,
        sha256="g" * 64,
        perceptual_hash="7" * 64,
        file_size=path.stat().st_size,
        width=16,
        height=16,
        file_format="PNG",
        mime_type="image/png",
        source="single_source",
        review_status=ReviewStatus.APPROVED,
    )
    session.add(rec)
    session.flush()
    session.add(Prediction(image_id=rec.id, model_name="resnet", model_version="v1", predicted_class="ai", raw_score=0.9, threshold=0.5, preprocessing_version="p1"))
    session.add(HumanReview(image_id=rec.id, reviewer_id="r1", verified_class="ai", confidence=0.8, review_status=ReviewStatus.APPROVED, evidence_source="manual"))
    session.commit()

    result = build_dataset_version(session=session, output_dir=tmp_path / "datasets", version="v0001", dry_run=False, require_second_review=True)
    assert len(result["manifest"]) == 0
    assert result["report"]["excluded_counts"]["needs_second_review"] >= 1
