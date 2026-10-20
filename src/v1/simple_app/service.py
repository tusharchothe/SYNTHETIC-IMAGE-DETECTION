"""Core service layer for the simple no-auth local workflow.

Responsibilities
----------------
- Validate and store an uploaded image under data/uploads/.
- Run both the ResNet-50 and CLIP inference pipelines.
- Persist the image record, predictions, and inference timestamps to SQLite.
- Return a structured result that the API layer can serialise directly.

Design rules
------------
- No authentication or role checks anywhere in this file.
- Inference functions (run_resnet_prediction, run_clip_prediction) are called
  unchanged — no alterations to preprocessing or model architecture.
- ImageStorageService is reused for deduplication, perceptual hashing, and
  file writing.  Storage directory is overridden to data/uploads/ via config.
- Every error surface (invalid image, duplicate, model failure, DB failure)
  is handled gracefully: a model failure records a warning but does not
  abort the upload.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from v1.services.inference_service import (
    InferenceResult,
    run_clip_prediction,
    run_resnet_prediction,
)
from v1.storage.database import Base
from v1.storage.models import ImageRecord, Prediction, ReviewStatus
from v1.storage.service import ImageStorageService

from .config import DATABASE_URL, MAX_FILE_SIZE_BYTES, UPLOAD_DIR


# ── Database bootstrap -------------------------------------------------------

def _make_engine(database_url: str = DATABASE_URL):
    """Create (and initialise) the SQLite engine for the simple workflow."""
    engine = create_engine(database_url, connect_args={"check_same_thread": False})
    # Ensure data/ directory exists before SQLite creates the file there.
    if database_url.startswith("sqlite:///"):
        db_path = Path(database_url.removeprefix("sqlite:///"))
        db_path.parent.mkdir(parents=True, exist_ok=True)
    Base.metadata.create_all(bind=engine)
    return engine


def _make_session_factory(engine) -> sessionmaker:
    return sessionmaker(autocommit=False, autoflush=False, bind=engine, future=True)


# Module-level singletons — tests override these via dependency injection.
_engine = _make_engine()
_SessionFactory = _make_session_factory(_engine)

_storage_service = ImageStorageService(
    session_factory=lambda: _SessionFactory(),
    storage_dir=str(UPLOAD_DIR),
    max_file_size_bytes=MAX_FILE_SIZE_BYTES,
)


def get_session() -> Session:
    """Yield a new database session (used as a FastAPI dependency)."""
    db = _SessionFactory()
    try:
        yield db
    finally:
        db.close()


# ── Result dataclass ---------------------------------------------------------

@dataclass
class PredictionSummary:
    model_name: str
    model_version: str
    predicted_class: str
    raw_score: float
    threshold: float
    preprocessing_version: str
    inference_timestamp: str
    warnings: list[str] = field(default_factory=list)


@dataclass
class UploadResult:
    """Structured result returned by process_upload for every image."""
    image_id: str
    safe_filename: str
    storage_path: str
    sha256: str
    file_size: int
    width: int
    height: int
    uploaded_at: str
    review_status: str
    # True when a previously stored copy was returned instead of a new record.
    duplicate: bool
    predictions: list[PredictionSummary] = field(default_factory=list)
    # Non-fatal warnings (e.g. one model failed but the other succeeded).
    warnings: list[str] = field(default_factory=list)


# ── Main entry point ---------------------------------------------------------

def process_upload(
    image_bytes: bytes,
    filename: str | None,
    content_type: str | None = None,
    *,
    # Dependency-injection seams used by tests to avoid hitting real models.
    _storage: ImageStorageService | None = None,
    _run_resnet: Any = run_resnet_prediction,
    _run_clip: Any = run_clip_prediction,
) -> UploadResult:
    """Validate, store, infer, and persist a single image upload.

    Parameters
    ----------
    image_bytes:
        Raw bytes of the uploaded file.
    filename:
        Original filename supplied by the client (used for extension
        validation and safe-name generation; may be None).
    content_type:
        MIME type reported by the client (optional, used for validation).
    _storage / _run_resnet / _run_clip:
        Injection points for unit tests — do not use in production.

    Returns
    -------
    UploadResult
        Always returns a result.  If a model fails its warning is recorded
        in ``result.warnings`` and the prediction list may be partial.

    Raises
    ------
    ValueError
        For invalid/corrupt images, unsupported formats, oversized files, or
        disallowed MIME types.  Callers should map these to HTTP 400.
    RuntimeError
        For unexpected database or filesystem failures.  Callers should map
        these to HTTP 500.
    """
    storage = _storage or _storage_service
    warnings: list[str] = []

    # ── 1. Store image (validates format, checks duplicates, writes file) ----
    # ValueError is raised for invalid inputs; let it propagate to the caller.
    image_record: ImageRecord = storage.store_image(
        image_bytes,
        filename,
        content_type=content_type,
        source="simple_upload",
    )

    is_duplicate = (
        image_record.review_status != ReviewStatus.PENDING
        or _is_existing_record(image_record, image_bytes)
    )

    # ── 2. Run inference pipelines ------------------------------------------
    predictions: list[PredictionSummary] = []

    for model_label, run_fn in (("resnet", _run_resnet), ("clip", _run_clip)):
        try:
            result: InferenceResult = run_fn(image_bytes)
            _persist_prediction(storage, image_record.id, result)
            predictions.append(
                PredictionSummary(
                    model_name=result.model_name,
                    model_version=result.model_version,
                    predicted_class=result.predicted_class,
                    raw_score=result.raw_score,
                    threshold=result.threshold,
                    preprocessing_version=result.preprocessing_version,
                    inference_timestamp=result.inference_timestamp,
                    warnings=list(result.warnings or []),
                )
            )
        except Exception as exc:
            warnings.append(f"{model_label} model failed: {exc}")

    # ── 3. Build and return result -------------------------------------------
    return UploadResult(
        image_id=image_record.id,
        safe_filename=image_record.safe_filename,
        storage_path=image_record.storage_path,
        sha256=image_record.sha256,
        file_size=image_record.file_size,
        width=image_record.width,
        height=image_record.height,
        uploaded_at=image_record.uploaded_at.isoformat(),
        review_status=image_record.review_status.value,
        duplicate=_detect_duplicate(image_bytes, image_record),
        predictions=predictions,
        warnings=warnings,
    )


# ── Queue helpers ------------------------------------------------------------

def get_pending_queue(
    db: Session,
    *,
    page: int = 1,
    limit: int = 20,
) -> dict:
    """Return a paginated list of images with PENDING review status.

    Each item includes the image metadata and its latest predictions so the
    UI can show inference results alongside the queued images.
    """
    page = max(1, page)
    limit = max(1, min(limit, 100))

    stmt = (
        select(ImageRecord)
        .where(ImageRecord.review_status == ReviewStatus.PENDING)
        .order_by(ImageRecord.uploaded_at.desc())
    )
    all_records = db.execute(stmt).scalars().all()
    total = len(all_records)
    start = (page - 1) * limit
    paged = all_records[start : start + limit]

    items = []
    for record in paged:
        preds = db.execute(
            select(Prediction)
            .where(Prediction.image_id == record.id)
            .order_by(Prediction.inference_timestamp.asc())
        ).scalars().all()

        items.append({
            "image_id": record.id,
            "safe_filename": record.safe_filename,
            "storage_path": record.storage_path,
            "sha256": record.sha256,
            "file_size": record.file_size,
            "width": record.width,
            "height": record.height,
            "uploaded_at": record.uploaded_at.isoformat(),
            "review_status": record.review_status.value,
            "predictions": [
                {
                    "model_name": p.model_name,
                    "model_version": p.model_version,
                    "predicted_class": p.predicted_class,
                    "raw_score": float(p.raw_score),
                    "threshold": float(p.threshold),
                    "preprocessing_version": p.preprocessing_version,
                    "inference_timestamp": p.inference_timestamp.isoformat(),
                }
                for p in preds
            ],
        })

    return {
        "items": items,
        "total": total,
        "page": page,
        "limit": limit,
        "has_next": (start + limit) < total,
    }


# ── Internal helpers ---------------------------------------------------------

def _persist_prediction(
    storage: ImageStorageService,
    image_id: str,
    result: InferenceResult,
) -> None:
    """Write a single InferenceResult to the predictions table."""
    storage.add_prediction(
        image_id=image_id,
        model_name=result.model_name,
        model_version=result.model_version,
        predicted_class=result.predicted_class,
        raw_score=result.raw_score,
        threshold=result.threshold,
        preprocessing_version=result.preprocessing_version,
    )


def _detect_duplicate(image_bytes: bytes, record: ImageRecord) -> bool:
    """Return True when the image record already existed before this upload."""
    import hashlib
    return hashlib.sha256(image_bytes).hexdigest() == record.sha256 and (
        # If the record has predictions it was processed in a prior upload.
        # We use review_status as a proxy: a brand-new PENDING record with
        # no predictions was just created; everything else is a duplicate hit.
        record.review_status in {
            ReviewStatus.NEEDS_REVIEW,
            ReviewStatus.APPROVED,
            ReviewStatus.REJECTED,
            ReviewStatus.DUPLICATE,
        }
    )


def _is_existing_record(record: ImageRecord, image_bytes: bytes) -> bool:
    """Secondary duplicate check comparing SHA-256."""
    import hashlib
    return hashlib.sha256(image_bytes).hexdigest() == record.sha256


def submit_review(
    db: Session,
    image_id: str,
    verified_class: str,
    confidence: float = 1.0,
    reviewer_id: str = "local_reviewer",
    known_generator: str | None = None,
    evidence_source: str | None = None,
    source_notes: str | None = None,
) -> dict:
    """Submit a human review for an image.

    Parameters
    ----------
    db:
        Active database session.
    image_id:
        ID of the image being reviewed.
    verified_class:
        Human label: "real", "ai", or "uncertain".
    confidence:
        Confidence score 0-1.
    reviewer_id:
        Identifier for the reviewer (no-auth default: "local_reviewer").
    known_generator:
        Optional: known generator name if AI-generated.
    evidence_source:
        Optional: source of evidence for the label.
    source_notes:
        Optional: free-text notes.

    Returns
    -------
    dict
        Review record summary.

    Raises
    ------
    ValueError
        If image not found or invalid label.
    """
    from v1.storage.models import HumanReview

    # Validate image exists
    record: ImageRecord | None = db.get(ImageRecord, image_id)
    if record is None:
        raise ValueError(f"Image '{image_id}' not found.")

    # Validate label
    valid_labels = {"real", "ai", "uncertain"}
    if verified_class.lower() not in valid_labels:
        raise ValueError(f"Invalid label '{verified_class}'. Must be one of: {valid_labels}")

    # Create review record
    review = HumanReview(
        image_id=image_id,
        reviewer_id=reviewer_id,
        verified_class=verified_class.lower(),
        known_generator=known_generator,
        evidence_source=evidence_source,
        source_notes=source_notes,
        confidence=max(0.0, min(1.0, confidence)),
        review_status=ReviewStatus.APPROVED,
    )
    db.add(review)

    # Update image review status
    if verified_class.lower() == "uncertain":
        record.review_status = ReviewStatus.NEEDS_REVIEW
    else:
        record.review_status = ReviewStatus.APPROVED

    db.commit()
    db.refresh(review)

    return {
        "review_id": review.id,
        "image_id": image_id,
        "verified_class": review.verified_class,
        "confidence": review.confidence,
        "reviewer_id": review.reviewer_id,
        "review_status": review.review_status.value,
        "reviewed_at": review.reviewed_at.isoformat(),
    }


def build_dataset(
    output_dir: str | Path,
    version: str,
    *,
    dry_run: bool = False,
    require_second_review: bool = False,
) -> dict:
    """Build a versioned dataset from reviewed images.

    Parameters
    ----------
    output_dir:
        Directory to store dataset versions.
    version:
        Version identifier (e.g., "v0001").
    dry_run:
        If True, validate but don't write files.
    require_second_review:
        If True, require at least 2 approved reviews per image.

    Returns
    -------
    dict
        Dataset build summary with manifest count and report.

    Raises
    ------
    FileExistsError
        If version already exists.
    ValueError
        If dataset criteria not met.
    """
    from v1.dataset_builder import build_dataset_version

    with _SessionFactory() as session:
        result = build_dataset_version(
            session=session,
            output_dir=output_dir,
            version=version,
            dry_run=dry_run,
            require_second_review=require_second_review,
        )

    return {
        "version": result["version"],
        "version_dir": result["version_dir"],
        "manifest_count": len(result["manifest"]),
        "report": result["report"],
        "dry_run": dry_run,
    }


def run_training(
    dataset_version: str,
    model_name: str = "resnet50",
    *,
    dataset_root: str = "datasets",
    output_dir: str = "artifacts/training",
    epochs: int = 5,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    dry_run: bool = False,
) -> dict:
    """Run model training on a dataset version.

    Parameters
    ----------
    dataset_version:
        Dataset version to train on (e.g., "v0001").
    model_name:
        Model architecture: "resnet50" or "clip".
    dataset_root:
        Root directory containing dataset versions.
    output_dir:
        Output directory for training artifacts.
    epochs:
        Number of training epochs.
    batch_size:
        Training batch size.
    learning_rate:
        Learning rate.
    dry_run:
        If True, validate but don't run training.

    Returns
    -------
    dict
        Training result with job ID and artifact path.

    Raises
    ------
    ValueError
        If dataset invalid or model unsupported.
    PermissionError
        If admin confirmation required.
    """
    from v1.training_pipeline import TrainingConfig, run_training_pipeline

    config = TrainingConfig(
        dataset_version=dataset_version,
        dataset_root=dataset_root,
        model_name=model_name,
        output_dir=output_dir,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        dry_run=dry_run,
        admin_confirmed=True,  # No-auth workflow: implicitly confirmed
    )

    result = run_training_pipeline(config)
    return result


def list_datasets(dataset_root: str | Path = "datasets") -> dict:
    """List available dataset versions.

    Parameters
    ----------
    dataset_root:
        Root directory containing dataset versions.

    Returns
    -------
    dict
        List of dataset versions with metadata.
    """
    dataset_root = Path(dataset_root)
    versions_index = dataset_root / "versions.json"

    if not versions_index.exists():
        return {"versions": []}

    import json
    return json.loads(versions_index.read_text(encoding="utf-8"))


def list_training_runs(training_dir: str | Path = "artifacts/training") -> list[dict]:
    """List completed training runs.

    Parameters
    ----------
    training_dir:
        Directory containing training artifacts.

    Returns
    -------
    list[dict]
        List of training run summaries.
    """
    import json
    from glob import glob

    training_dir = Path(training_dir)
    if not training_dir.exists():
        return []

    runs = []
    for status_file in glob(str(training_dir / "train-*.json")):
        try:
            status = json.loads(Path(status_file).read_text(encoding="utf-8"))
            runs.append({
                "job_id": status.get("job_id"),
                "dataset_version": status.get("dataset_version"),
                "model_name": status.get("model_name"),
                "status": status.get("status"),
                "started_at": status.get("started_at"),
                "finished_at": status.get("finished_at"),
                "artifact_path": status.get("artifact_path"),
            })
        except Exception:
            continue

    return sorted(runs, key=lambda x: x.get("started_at", ""), reverse=True)


__all__ = [
    "UploadResult",
    "PredictionSummary",
    "process_upload",
    "get_pending_queue",
    "submit_review",
    "build_dataset",
    "run_training",
    "list_datasets",
    "list_training_runs",
    "get_session",
    "_make_engine",
    "_make_session_factory",
    "_storage_service",
]
