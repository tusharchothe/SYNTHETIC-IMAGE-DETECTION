"""FastAPI backend for the simple no-auth local workflow.

Endpoints
---------
GET  /health                         – liveness check
POST /upload                         – accept one or more images
GET  /queue                          – paginated pending-review queue
GET  /images/{image_id}/file         – serve the raw image bytes
GET  /images/{image_id}/predictions  – prediction history for one image

No authentication, no role checks.  All errors are returned as JSON with
appropriate HTTP status codes.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Generator

from fastapi import Depends, FastAPI, File, HTTPException, Query, UploadFile, status
from fastapi.responses import FileResponse
from PIL import Image
from sqlalchemy import select
from sqlalchemy.orm import Session

from v1.storage.models import ImageRecord, Prediction

from .service import (
    UploadResult,
    build_dataset,
    get_pending_queue,
    get_session,
    list_datasets,
    list_training_runs,
    process_upload,
    run_training,
    submit_review,
    _storage_service,
)

# ---------------------------------------------------------------------------
app = FastAPI(
    title="AI Image Detector – Simple Local Workflow",
    version="1.0.0",
    description=(
        "No-auth local workflow. "
        "Upload images, run ResNet-50 and CLIP inference, "
        "browse the pending-review queue."
    ),
)


# ── Dependency: DB session ---------------------------------------------------

def _get_db() -> Generator[Session, None, None]:
    yield from get_session()


# ── Health -------------------------------------------------------------------

@app.get("/health", tags=["meta"])
def health() -> dict:
    """Return service status and active configuration paths."""
    from .config import DATABASE_URL, UPLOAD_DIR
    return {
        "status": "ok",
        "upload_dir": str(UPLOAD_DIR),
        "database_url": DATABASE_URL,
    }


# ── Upload -------------------------------------------------------------------

@app.post("/upload", status_code=status.HTTP_200_OK, tags=["images"])
async def upload_images(
    files: list[UploadFile] = File(..., description="One or more image files (PNG / JPEG / WebP, max 8 MB each)"),
) -> dict:
    """Upload one or more images.

    For each file the endpoint will:
    1. Validate format and size.
    2. Deduplicate by SHA-256.
    3. Write to ``data/uploads/`` with a UUID-prefixed safe filename.
    4. Run ResNet-50 and CLIP inference.
    5. Persist image record and predictions to SQLite.
    6. Add successfully processed images to the pending-review queue.

    Returns a list of per-file results.  A per-file error does **not** abort
    the remaining files.
    """
    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No files provided.",
        )

    results = []
    for upload in files:
        filename = upload.filename or "upload"
        content_type = upload.content_type or ""
        try:
            image_bytes = await upload.read()
        except Exception as exc:
            results.append({
                "filename": filename,
                "status": "error",
                "error": f"Could not read file: {exc}",
            })
            continue

        # Basic PIL sanity check before handing off to service
        try:
            Image.open(io.BytesIO(image_bytes)).verify()
        except Exception:
            results.append({
                "filename": filename,
                "status": "error",
                "error": "File is not a valid image.",
            })
            continue

        try:
            result: UploadResult = process_upload(
                image_bytes,
                filename,
                content_type=content_type,
            )
            entry = {
                "filename": filename,
                "status": "duplicate" if result.duplicate else "ok",
                "image_id": result.image_id,
                "safe_filename": result.safe_filename,
                "sha256": result.sha256,
                "file_size": result.file_size,
                "width": result.width,
                "height": result.height,
                "uploaded_at": result.uploaded_at,
                "review_status": result.review_status,
                "predictions": [
                    {
                        "model_name": p.model_name,
                        "model_version": p.model_version,
                        "predicted_class": p.predicted_class,
                        "raw_score": round(p.raw_score, 6),
                        "threshold": p.threshold,
                        "preprocessing_version": p.preprocessing_version,
                        "inference_timestamp": p.inference_timestamp,
                        "warnings": p.warnings,
                    }
                    for p in result.predictions
                ],
                "warnings": result.warnings,
            }
        except ValueError as exc:
            # Invalid image, oversized file, unsupported format, etc.
            entry = {
                "filename": filename,
                "status": "error",
                "error": str(exc),
            }
        except Exception as exc:
            # Unexpected DB / filesystem failure.
            entry = {
                "filename": filename,
                "status": "error",
                "error": f"Internal error: {exc}",
            }

        results.append(entry)

    return {"results": results}


# ── Pending-review queue -----------------------------------------------------

@app.get("/queue", tags=["queue"])
def get_queue(
    page: int = Query(1, ge=1, description="Page number (1-based)"),
    limit: int = Query(20, ge=1, le=100, description="Items per page"),
    db: Session = Depends(_get_db),
) -> dict:
    """Return a paginated list of images awaiting review.

    Only images with ``review_status = pending`` are included.
    Each item carries its latest inference results so the UI can show
    model predictions alongside the queued image.
    """
    return get_pending_queue(db, page=page, limit=limit)


# ── Serve image file ---------------------------------------------------------

@app.get("/images/{image_id}/file", tags=["images"])
def serve_image_file(
    image_id: str,
    db: Session = Depends(_get_db),
) -> FileResponse:
    """Serve the raw bytes of a stored image.

    Using this endpoint instead of a filesystem path means the UI works
    correctly even when run on a remote host.
    """
    record: ImageRecord | None = db.get(ImageRecord, image_id)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Image '{image_id}' not found.",
        )

    file_path = Path(record.storage_path)
    if not file_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Image file missing from disk.",
        )

    _fmt_to_mime = {
        "JPEG": "image/jpeg",
        "PNG": "image/png",
        "WEBP": "image/webp",
    }
    media_type = _fmt_to_mime.get(record.file_format.upper(), "application/octet-stream")
    return FileResponse(
        path=str(file_path),
        media_type=media_type,
        filename=record.safe_filename,
    )


# ── Prediction history for one image ----------------------------------------

@app.get("/images/{image_id}/predictions", tags=["images"])
def get_predictions(
    image_id: str,
    db: Session = Depends(_get_db),
) -> dict:
    """Return all stored predictions for a single image."""
    record: ImageRecord | None = db.get(ImageRecord, image_id)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Image '{image_id}' not found.",
        )

    preds = db.execute(
        select(Prediction)
        .where(Prediction.image_id == image_id)
        .order_by(Prediction.inference_timestamp.asc())
    ).scalars().all()

    return {
        "image_id": image_id,
        "predictions": [
            {
                "prediction_id": p.id,
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
    }


# ── Submit review -----------------------------------------------------------

@app.post("/reviews/submit", status_code=status.HTTP_201_CREATED, tags=["reviews"])
def submit_review_endpoint(
    image_id: str = Query(..., description="Image ID to review"),
    verified_class: str = Query(..., description="Human label: real, ai, or uncertain"),
    confidence: float = Query(1.0, ge=0.0, le=1.0, description="Confidence score 0-1"),
    known_generator: str | None = Query(None, description="Known generator name if AI"),
    evidence_source: str | None = Query(None, description="Source of evidence"),
    source_notes: str | None = Query(None, description="Free-text notes"),
    reviewer_id: str = Query("local_reviewer", description="Reviewer identifier"),
    db: Session = Depends(_get_db),
) -> dict:
    """Submit a human review for an image.

    This endpoint stores the human label separately from model predictions,
    maintaining audit trails and preventing data leakage.

    Valid labels:
    - ``real``: Verified as real/natural image
    - ``ai``: Verified as AI-generated
    - ``uncertain``: Unsure or disputed

    After submission, the image's review_status is updated:
    - ``uncertain`` → NEEDS_REVIEW
    - ``real`` or ``ai`` → APPROVED
    """
    try:
        result = submit_review(
            db=db,
            image_id=image_id,
            verified_class=verified_class,
            confidence=confidence,
            reviewer_id=reviewer_id,
            known_generator=known_generator,
            evidence_source=evidence_source,
            source_notes=source_notes,
        )
        return result
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error: {exc}",
        )


# ── Dataset management -------------------------------------------------------

@app.post("/datasets/build", status_code=status.HTTP_201_CREATED, tags=["datasets"])
def build_dataset_endpoint(
    version: str = Query(..., description="Dataset version identifier, e.g. v0001"),
    output_dir: str = Query("datasets", description="Output directory for dataset versions"),
    dry_run: bool = Query(False, description="Validate without writing files"),
    require_second_review: bool = Query(False, description="Require 2+ approved reviews per image"),
) -> dict:
    """Build a versioned dataset from reviewed images.

    Only images with approved reviews and valid binary labels (real/ai) are included.
    Images with uncertain labels, duplicates, or missing files are excluded.

    The dataset is split deterministically into train/val/calibration/test sets
    while preventing leakage across related image groups (same source or perceptual hash).

    Returns a manifest file, split definitions, and a detailed report.
    """
    try:
        result = build_dataset(
            output_dir=output_dir,
            version=version,
            dry_run=dry_run,
            require_second_review=require_second_review,
        )
        return result
    except FileExistsError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error: {exc}",
        )


@app.get("/datasets", tags=["datasets"])
def list_datasets_endpoint(
    dataset_root: str = Query("datasets", description="Root directory for datasets"),
) -> dict:
    """List all available dataset versions."""
    try:
        return list_datasets(dataset_root=dataset_root)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error: {exc}",
        )


# ── Training management ------------------------------------------------------

@app.post("/training/run", status_code=status.HTTP_202_ACCEPTED, tags=["training"])
def run_training_endpoint(
    dataset_version: str = Query(..., description="Dataset version to train on, e.g. v0001"),
    model_name: str = Query("resnet50", description="Model architecture: resnet50 or clip"),
    dataset_root: str = Query("datasets", description="Root directory for datasets"),
    output_dir: str = Query("artifacts/training", description="Output directory for training artifacts"),
    epochs: int = Query(5, ge=1, le=100, description="Number of training epochs"),
    batch_size: int = Query(32, ge=1, le=256, description="Training batch size"),
    learning_rate: float = Query(1e-4, gt=0.0, le=1.0, description="Learning rate"),
    dry_run: bool = Query(False, description="Validate without running training"),
) -> dict:
    """Run model training on a dataset version.

    This is an explicit action that requires a valid dataset version.
    Training runs in the current process and may take several minutes.

    Returns a job ID and artifact path upon completion.
    For long-running training, consider running this via a background worker.
    """
    try:
        result = run_training(
            dataset_version=dataset_version,
            model_name=model_name,
            dataset_root=dataset_root,
            output_dir=output_dir,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            dry_run=dry_run,
        )
        return result
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error: {exc}",
        )


@app.get("/training/runs", tags=["training"])
def list_training_runs_endpoint(
    training_dir: str = Query("artifacts/training", description="Training artifacts directory"),
) -> dict:
    """List all training runs with their status and artifacts."""
    try:
        runs = list_training_runs(training_dir=training_dir)
        return {"runs": runs, "total": len(runs)}
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error: {exc}",
        )


__all__ = ["app"]
