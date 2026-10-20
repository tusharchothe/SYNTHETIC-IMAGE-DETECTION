from __future__ import annotations

import csv
import io
import json
import os
from collections import Counter, defaultdict
from datetime import datetime
from math import sqrt
from pathlib import Path
from typing import Generator

from fastapi import Depends, FastAPI, File, Header, HTTPException, Query, UploadFile, status
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from PIL import Image
from sqlalchemy import select
from sqlalchemy.orm import Session

from .config import ROOT
from .model_registry import ModelRegistry
from .schemas import (
    ImageListItem,
    ListImagesResponse,
    PredictionHistoryEntry,
    PredictionHistoryResponse,
    PredictionResponse,
    ReviewDecisionRequest,
    ReviewPredictionSummary,
    ReviewQueueItem,
    ReviewQueueResponse,
    ReviewSubmissionResponse,
    UploadImageResponse,
)
from .services.inference_service import run_prediction_for_model
from .storage.config import DEFAULT_DATABASE_URL, DEFAULT_STORAGE_DIR, MAX_IMAGE_SIZE_BYTES
from .storage.database import SessionLocal, init_db
from .storage.models import HumanReview, ImageRecord, Prediction, ReviewStatus
from .storage.service import ImageStorageService
from .training_pipeline import TrainingConfig, prepare_training_status, write_training_status

MODEL_PATH = Path(os.environ.get("AIDETECT_MODEL_PATH", ROOT / "resnet50_sd14_best.pt"))
CLIP_MODEL_PATH = Path(os.environ.get("AIDETECT_CLIP_MODEL_PATH", ROOT / "clip_probe.joblib"))

engine = init_db(DEFAULT_DATABASE_URL)
SessionLocal.configure(bind=engine)

app = FastAPI(title="AI Image Detector API", version="1.0.0")

storage_service = ImageStorageService(
    session_factory=lambda: SessionLocal(),
    storage_dir=str(DEFAULT_STORAGE_DIR),
    max_file_size_bytes=MAX_IMAGE_SIZE_BYTES,
)


def _allowed_admins() -> set[str]:
    raw = os.getenv("AIDETECT_ALLOWED_ADMINS", "")
    return {item.strip() for item in raw.split(",") if item.strip()}


def require_admin(
    x_reviewer_id: str | None = Header(default=None, alias="X-Reviewer-Id"),
    x_admin_id: str | None = Header(default=None, alias="X-Admin-Id"),
    authorization: str | None = Header(default=None),
) -> str:
    allowed = _allowed_admins()
    if not allowed:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin access is not configured for this environment.")

    # X-Admin-Id (or X-Reviewer-Id as a fallback) is always required for a
    # traceable audit trail. A Bearer token only *authorises* the request.
    admin_id = x_admin_id or x_reviewer_id
    if not admin_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing admin identity. Provide the X-Admin-Id header.",
        )

    # If a Bearer token is present, verify it before checking the allowlist.
    if authorization and authorization.lower().startswith("bearer "):
        token = authorization.split(" ", 1)[1].strip()
        expected = os.getenv("AIDETECT_ADMIN_TOKEN", "")
        if expected and token != expected:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid admin token.",
            )

    if admin_id not in allowed:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=f"Admin '{admin_id}' is not authorized.")
    return admin_id


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def _allowed_reviewers() -> set[str]:
    raw = os.getenv("AIDETECT_ALLOWED_REVIEWERS", "")
    return {item.strip() for item in raw.split(",") if item.strip()}


def require_reviewer(
    x_reviewer_id: str | None = Header(default=None, alias="X-Reviewer-Id"),
    authorization: str | None = Header(default=None),
) -> str:
    allowed = _allowed_reviewers()
    if not allowed:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Review access is not configured for this environment.")

    # X-Reviewer-Id is always required for a traceable audit trail.
    # A Bearer token only *authorises* the request — it does not substitute
    # for the reviewer's identity.
    reviewer_id = x_reviewer_id
    if not reviewer_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing reviewer identity. Provide the X-Reviewer-Id header.",
        )

    # If a Bearer token is present, verify it before checking the allowlist.
    if authorization and authorization.lower().startswith("bearer "):
        token = authorization.split(" ", 1)[1].strip()
        expected = os.getenv("AIDETECT_REVIEWER_TOKEN", "")
        if expected and token != expected:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid reviewer token.",
            )

    if reviewer_id not in allowed:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=f"Reviewer '{reviewer_id}' is not authorized.")
    return reviewer_id


def _normalize_label(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    return normalized if normalized in {"real", "ai", "uncertain"} else None


def _human_label_for_image(image: ImageRecord) -> str | None:
    counts: Counter[str] = Counter()
    for review in image.reviews or []:
        if review.review_status != ReviewStatus.APPROVED:
            continue
        normalized = _normalize_label(review.verified_class)
        if normalized in {"real", "ai"}:
            counts[normalized] += 1
    if not counts:
        return None
    return counts.most_common(1)[0][0]


def _known_generator_for_image(image: ImageRecord) -> str | None:
    for review in sorted((image.reviews or []), key=lambda item: item.reviewed_at, reverse=True):
        if review.known_generator:
            return review.known_generator
    return image.source


def _prediction_summary(prediction: Prediction | None) -> ReviewPredictionSummary | None:
    if prediction is None:
        return None
    return ReviewPredictionSummary(
        model_name=prediction.model_name,
        model_version=prediction.model_version,
        predicted_class=prediction.predicted_class,
        raw_score=float(prediction.raw_score),
        threshold=float(prediction.threshold),
        preprocessing_version=prediction.preprocessing_version,
        inference_timestamp=prediction.inference_timestamp.isoformat(),
    )


def _binomial_ci(proportion: float, sample_count: int) -> tuple[float, float]:
    if sample_count <= 0:
        return (0.0, 0.0)
    p = max(0.0, min(1.0, float(proportion)))
    se = sqrt(max(p * (1.0 - p), 0.0) / sample_count)
    z = 1.96
    margin = z * se
    return (max(0.0, p - margin), min(1.0, p + margin))


def _latest_predictions_by_model(image: ImageRecord) -> dict[str, Prediction]:
    latest: dict[str, Prediction] = {}
    for prediction in image.predictions or []:
        model_name = (prediction.model_name or "").lower()
        current = latest.get(model_name)
        if current is None or prediction.inference_timestamp >= current.inference_timestamp:
            latest[model_name] = prediction
    return latest


def _metric_row_for_group(rows: list[dict[str, object]]) -> dict[str, object]:
    if not rows:
        return {"sample_count": 0, "accuracy": 0.0, "false_positive_rate": 0.0, "false_negative_rate": 0.0, "confidence_interval_95": [0.0, 0.0], "warning": "No samples were available."}

    correct = sum(1 for row in rows if row["prediction"] == row["human_label"])
    false_positives = sum(1 for row in rows if row["human_label"] == "real" and row["prediction"] == "ai")
    false_negatives = sum(1 for row in rows if row["human_label"] == "ai" and row["prediction"] == "real")
    sample_count = len(rows)
    accuracy = correct / sample_count if sample_count else 0.0
    fp_rate = false_positives / max(sum(1 for row in rows if row["human_label"] == "real"), 1)
    fn_rate = false_negatives / max(sum(1 for row in rows if row["human_label"] == "ai"), 1)
    warning = "Metric is based on too few samples (<10)." if sample_count < 10 else ""
    return {
        "sample_count": sample_count,
        "accuracy": round(accuracy, 4),
        "false_positive_rate": round(fp_rate, 4),
        "false_negative_rate": round(fn_rate, 4),
        "confidence_interval_95": [round(value, 4) for value in _binomial_ci(accuracy, sample_count)],
        "warning": warning,
    }


def _build_dashboard_payload(db: Session, *, start_date: str | None = None, end_date: str | None = None, model_version: str | None = None, generator: str | None = None, source: str | None = None, label: str | None = None, review_status: str | None = None) -> dict:
    rows: list[dict[str, object]] = []
    images = db.execute(select(ImageRecord).order_by(ImageRecord.uploaded_at.desc())).scalars().all()
    for image in images:
        if start_date:
            start_dt = datetime.fromisoformat(start_date)
            if image.uploaded_at < start_dt:
                continue
        if end_date:
            end_dt = datetime.fromisoformat(end_date)
            if image.uploaded_at > end_dt:
                continue

        human_label = _human_label_for_image(image)
        if label and human_label != label:
            continue
        if review_status and image.review_status.value != review_status:
            continue
        if source and (image.source or "") != source:
            continue
        generator_value = _known_generator_for_image(image)
        if generator and generator_value != generator:
            continue

        latest = _latest_predictions_by_model(image)
        resnet_pred = latest.get("resnet50") or latest.get("resnet")
        clip_pred = latest.get("clip")
        resnet_label = _normalize_label(resnet_pred.predicted_class) if resnet_pred else None
        clip_label = _normalize_label(clip_pred.predicted_class) if clip_pred else None
        prediction = None
        if resnet_label and clip_label and resnet_label != clip_label:
            prediction = "model_disagreement"
        elif resnet_label:
            prediction = resnet_label
        elif clip_label:
            prediction = clip_label
        if model_version:
            if not any(pred.model_version == model_version for pred in (resnet_pred, clip_pred) if pred is not None):
                continue

        rows.append(
            {
                "image_id": image.id,
                "storage_path": image.storage_path,
                "uploaded_at": image.uploaded_at.isoformat(),
                "source": image.source,
                "generator": generator_value,
                "human_label": human_label,
                "prediction": prediction,
                "resnet_prediction": resnet_label,
                "clip_prediction": clip_label,
                "review_status": image.review_status.value,
                "model_version": model_version or (resnet_pred.model_version if resnet_pred else clip_pred.model_version if clip_pred else None),
                "is_duplicate": bool(image.duplicate_of_id),
            }
        )

    counts = {
        "uploaded_total": len(rows),
        "pending": sum(1 for row in rows if row["review_status"] == "pending"),
        "needs_review": sum(1 for row in rows if row["review_status"] == "needs_review"),
        "approved": sum(1 for row in rows if row["review_status"] == "approved"),
        "rejected": sum(1 for row in rows if row["review_status"] == "rejected"),
        "duplicate": sum(1 for row in rows if row["review_status"] == "duplicate" or row["is_duplicate"]),
        "uncertain_label": sum(1 for row in rows if row["human_label"] == "uncertain"),
        "training_eligible": 0,
        "verified_label_counts": dict(Counter(row["human_label"] for row in rows if row["human_label"] in {"real", "ai"})),
        "false_positive_count": sum(1 for row in rows if row["human_label"] == "real" and row["prediction"] in {"ai", "model_disagreement"}),
        "false_negative_count": sum(1 for row in rows if row["human_label"] == "ai" and row["prediction"] in {"real", "model_disagreement"}),
        "model_disagreement_count": sum(1 for row in rows if row["prediction"] == "model_disagreement"),
    }

    dataset_versions = _load_dataset_versions(Path(os.getenv("AIDETECT_DATASET_ROOT", ROOT / "datasets")))
    counts["training_eligible"] = sum(1 for item in dataset_versions if item.get("eligible_total", 0) > 0)

    generator_groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    source_groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        if row["generator"]:
            generator_groups[str(row["generator"])].append(row)
        if row["source"]:
            source_groups[str(row["source"])].append(row)

    generator_performance = {name: _metric_row_for_group(rows_for_group) for name, rows_for_group in sorted(generator_groups.items())}
    source_performance = {name: _metric_row_for_group(rows_for_group) for name, rows_for_group in sorted(source_groups.items())}
    for name, metrics in generator_performance.items():
        if metrics["sample_count"] < 10:
            metrics["warning"] = "Metric is based on too few samples (<10)."
    for name, metrics in source_performance.items():
        if metrics["sample_count"] < 10:
            metrics["warning"] = "Metric is based on too few samples (<10)."

    warnings = []
    if counts["uploaded_total"] < 10:
        warnings.append("Dashboard is based on fewer than 10 uploaded items; confidence is limited.")
    for name, metrics in generator_performance.items():
        if metrics["sample_count"] < 10:
            warnings.append(f"Generator '{name}' has only {metrics['sample_count']} samples; accuracy is low-confidence.")
    for name, metrics in source_performance.items():
        if metrics["sample_count"] < 10:
            warnings.append(f"Source '{name}' has only {metrics['sample_count']} samples; accuracy is low-confidence.")

    return {
        "filters": {
            "start_date": start_date,
            "end_date": end_date,
            "model_version": model_version,
            "generator": generator,
            "source": source,
            "label": label,
            "review_status": review_status,
        },
        "counts": counts,
        "generator_performance": generator_performance,
        "source_performance": source_performance,
        "dataset_history": dataset_versions,
        "warnings": warnings,
        "rows": [
            {
                "image_id": row["image_id"],
                "generator": row["generator"],
                "source": row["source"],
                "human_label": row["human_label"],
                "prediction": row["prediction"],
                "review_status": row["review_status"],
                "uploaded_at": row["uploaded_at"],
            }
            for row in rows
        ],
    }


def _load_dataset_versions(dataset_root: str | Path) -> list[dict[str, object]]:
    root = Path(dataset_root)
    if not root.exists():
        return []
    versions: list[dict[str, object]] = []
    for version_dir in sorted(root.iterdir()):
        report_path = version_dir / "dataset_report.json"
        if report_path.exists():
            payload = json.loads(report_path.read_text(encoding="utf-8"))
            versions.append({
                "version": version_dir.name,
                "eligible_total": payload.get("eligible_total", 0),
                "class_balance": payload.get("class_balance", {}),
                "source_balance": payload.get("source_balance", {}),
                "generator_balance": payload.get("generator_balance", {}),
                "excluded_counts": payload.get("excluded_counts", {}),
                "generated_at": payload.get("generated_at"),
            })
    return versions


def _model_registry_path() -> Path:
    return Path(os.getenv("AIDETECT_MODEL_REGISTRY_DIR", str(ROOT / "artifacts" / "registry")))


def _read_registry_state() -> dict:
    registry_path = _model_registry_path() / "registry.json"
    if not registry_path.exists():
        return {"models": {}}
    try:
        return json.loads(registry_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"models": {}}


def _read_training_runs() -> list[dict]:
    candidate_dirs = [Path(os.getenv("AIDETECT_TRAINING_DIR", str(ROOT / "artifacts" / "training"))), Path(os.getenv("AIDETECT_WORK_DIR", str(ROOT / "artifacts"))) / "training"]
    runs: list[dict] = []
    for directory in candidate_dirs:
        if not directory.exists():
            continue
        for file_path in sorted(directory.glob("**/*.json")):
            try:
                content = json.loads(file_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            if isinstance(content, dict) and ("job_id" in content or "status" in content or "model_name" in content):
                runs.append({"path": str(file_path), **content})
    deduped: dict[str, dict] = {}
    for run in runs:
        deduped[run.get("job_id") or run.get("path", str(run))] = run
    return list(deduped.values())


def _queue_filter_matches(image: ImageRecord, predictions: list[Prediction], filter_name: str) -> bool:
    if filter_name in {"pending", "reviewed"}:
        if filter_name == "pending":
            return image.review_status in {ReviewStatus.PENDING, ReviewStatus.NEEDS_REVIEW}
        return image.review_status in {ReviewStatus.APPROVED, ReviewStatus.REJECTED, ReviewStatus.DUPLICATE}

    summary_by_model = {pred.model_name: pred for pred in predictions}
    resnet = summary_by_model.get("resnet50") or summary_by_model.get("resnet")
    clip = summary_by_model.get("clip")
    if not resnet or not clip:
        return False

    if filter_name == "model_disagreement":
        return resnet.predicted_class != clip.predicted_class
    if filter_name == "low_confidence":
        return abs(float(resnet.raw_score) - 0.5) <= 0.15 and abs(float(clip.raw_score) - 0.5) <= 0.15
    if filter_name == "suspected_false_positive":
        return (resnet.predicted_class == "ai" and float(resnet.raw_score) < 0.65) or (clip.predicted_class == "ai" and float(clip.raw_score) < 0.65)
    if filter_name == "suspected_false_negative":
        return (resnet.predicted_class == "real" and float(resnet.raw_score) > 0.35) or (clip.predicted_class == "real" and float(clip.raw_score) > 0.35)
    return True


@app.get("/dashboard/overview")
def dashboard_overview(
    db: Session = Depends(get_db),
    start_date: str | None = None,
    end_date: str | None = None,
    model_version: str | None = None,
    generator: str | None = None,
    source: str | None = None,
    label: str | None = None,
    review_status: str | None = None,
    _reviewer: str = Depends(require_reviewer),
):
    payload = _build_dashboard_payload(
        db,
        start_date=start_date,
        end_date=end_date,
        model_version=model_version,
        generator=generator,
        source=source,
        label=label,
        review_status=review_status,
    )
    return payload


@app.get("/dashboard/export")
def export_dashboard_report(
    db: Session = Depends(get_db),
    format: str = Query("json", pattern="^(json|csv)$"),
    start_date: str | None = None,
    end_date: str | None = None,
    model_version: str | None = None,
    generator: str | None = None,
    source: str | None = None,
    label: str | None = None,
    review_status: str | None = None,
    _reviewer: str = Depends(require_reviewer),
):
    payload = _build_dashboard_payload(
        db,
        start_date=start_date,
        end_date=end_date,
        model_version=model_version,
        generator=generator,
        source=source,
        label=label,
        review_status=review_status,
    )
    if format == "csv":
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["metric", "value"])
        for key, value in payload["counts"].items():
            writer.writerow([key, json.dumps(value, sort_keys=True)])
        return PlainTextResponse(content=output.getvalue(), media_type="text/csv", headers={"Content-Disposition": "attachment; filename=dashboard-export.csv"})
    return JSONResponse(content={"generated_at": datetime.utcnow().isoformat(), "report": payload}, media_type="application/json")


@app.get("/dashboard/model-lifecycle")
def model_lifecycle_dashboard(db: Session = Depends(get_db), _reviewer: str = Depends(require_reviewer)):
    registry = _read_registry_state()
    models = registry.get("models", {})
    active_model_name = None
    active_model = None
    model_history = []
    for name, payload in models.items():
        if payload.get("active_model"):
            active_model_name = name
            active_model = {"name": name, "path": payload.get("active_model"), "version": payload.get("current_version"), "updated_at": payload.get("updated_at"), "evaluation_report": payload.get("history", [{}])[-1].get("evaluation_report") if payload.get("history") else None}
            model_history = payload.get("history", [])
            break

    return {
        "active_model": active_model,
        "promotion_history": model_history,
        "training_runs": _read_training_runs(),
        "rollback_available": bool(model_history),
        "guardrails": {"require_admin_for_destructive_actions": True, "require_admin_for_training": True, "require_admin_for_promotion": True},
    }


@app.post("/dashboard/model-lifecycle/promote")
def promote_model_candidate(
    payload: dict,
    db: Session = Depends(get_db),
    admin_id: str = Depends(require_admin),
):
    model_name = str(payload.get("model_name") or "resnet50")
    candidate_path = str(payload.get("candidate_path") or "")
    evaluation_report = payload.get("evaluation_report") or _build_dashboard_payload(db)["counts"]
    registry = ModelRegistry(root_dir=_model_registry_path())
    record = registry.promote_candidate(
        model_name=model_name,
        candidate_path=candidate_path,
        evaluation_report=evaluation_report,
        approver=admin_id,
        authorizer=lambda user: user == admin_id,
        model_version=str(payload.get("model_version") or "candidate"),
    )
    return {"status": "promoted", "record": record, "authorized_by": admin_id}


@app.post("/dashboard/model-lifecycle/rollback")
def rollback_model_candidate(
    model_name: str,
    admin_id: str = Depends(require_admin),
):
    registry = ModelRegistry(root_dir=_model_registry_path())
    record = registry.rollback(model_name=model_name, approver=admin_id, authorizer=lambda user: user == admin_id)
    return {"status": "rolled_back", "record": record, "authorized_by": admin_id}


@app.post("/admin/training/run")
def admin_training_run(
    payload: dict,
    admin_id: str = Depends(require_admin),
):
    training_config = TrainingConfig(
        dataset_version=str(payload.get("dataset_version") or "latest"),
        dataset_root=str(payload.get("dataset_root") or str(ROOT / "datasets")),
        model_name=str(payload.get("model_name") or "resnet50"),
        output_dir=str(payload.get("output_dir") or str(ROOT / "artifacts" / "training")),
        admin_confirmed=True,
        epochs=int(payload.get("epochs") or 5),
        batch_size=int(payload.get("batch_size") or 32),
        learning_rate=float(payload.get("learning_rate") or 1e-4),
        max_memory_gb=float(payload.get("max_memory_gb") or 12.0),
        dry_run=bool(payload.get("dry_run") or False),
    )
    status = prepare_training_status(training_config)
    status_path = payload.get("status_path") or str(Path(training_config.output_dir) / f"{status.job_id}.json")
    write_training_status(status_path, status)
    return {"status": "training_queued", "job_id": status.job_id, "authorized_by": admin_id, "status_path": status_path}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_path": str(MODEL_PATH),
        "clip_model_path": str(CLIP_MODEL_PATH),
        "database_url": DEFAULT_DATABASE_URL,
        "storage_dir": str(DEFAULT_STORAGE_DIR),
    }


@app.post("/images/upload", response_model=UploadImageResponse)
async def upload_image(
    file: UploadFile = File(...),
    source: str | None = None,
    db: Session = Depends(get_db),
):
    if not file.filename:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Please upload an image file.")

    content = await file.read()
    warnings: list[str] = []

    try:
        Image.open(io.BytesIO(content)).verify()
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded content is not a valid image.") from exc

    try:
        image_record = storage_service.store_image(content, file.filename, file.content_type, source=source)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Database or storage error: {exc}") from exc

    result_entries: list[PredictionResponse] = []
    for model_name in ("resnet", "clip"):
        try:
            inference = run_prediction_for_model(model_name, content)
            prediction = storage_service.add_prediction(
                image_id=image_record.id,
                model_name=inference.model_name,
                model_version=inference.model_version,
                predicted_class=inference.predicted_class,
                raw_score=inference.raw_score,
                threshold=inference.threshold,
                preprocessing_version=inference.preprocessing_version,
            )
            result_entries.append(
                PredictionResponse(
                    model_name=inference.model_name,
                    model_version=inference.model_version,
                    predicted_class=inference.predicted_class,
                    raw_score=inference.raw_score,
                    threshold=inference.threshold,
                    score_kind=inference.score_kind,
                    calibrated=inference.calibrated,
                    preprocessing_version=inference.preprocessing_version,
                    inference_timestamp=inference.inference_timestamp,
                    warnings=inference.warnings or [],
                )
            )
        except Exception as exc:  # model or DB failure should not crash upload
            warnings.append(f"{model_name} model failed: {exc}")

    if not result_entries:
        image_record.review_status = ReviewStatus.NEEDS_REVIEW
        db.add(image_record)
        db.commit()

    return UploadImageResponse(
        image_id=image_record.id,
        storage_path=image_record.storage_path,
        review_status=image_record.review_status.value,
        input_warnings=warnings,
        predictions=result_entries,
    )


_QUEUE_STATUS_VALUES = {
    "pending",
    "reviewed",
    "model_disagreement",
    "low_confidence",
    "suspected_false_positive",
    "suspected_false_negative",
}

@app.get("/reviews/queue", response_model=ReviewQueueResponse)
def get_review_queue(
    page: int = Query(1, ge=1),
    limit: int = Query(10, ge=1, le=50),
    status: str = Query(
        "pending",
        description="Filter: pending | reviewed | model_disagreement | low_confidence | suspected_false_positive | suspected_false_negative",
        pattern="^(pending|reviewed|model_disagreement|low_confidence|suspected_false_positive|suspected_false_negative)$",
    ),
    db: Session = Depends(get_db),
    _reviewer: str = Depends(require_reviewer),
):
    records = db.execute(select(ImageRecord).order_by(ImageRecord.uploaded_at.desc())).scalars().all()
    items: list[ReviewQueueItem] = []
    for record in records:
        predictions = db.execute(
            select(Prediction).where(Prediction.image_id == record.id).order_by(Prediction.inference_timestamp.asc())
        ).scalars().all()
        if not _queue_filter_matches(record, predictions, status):
            continue

        summary_by_model = {pred.model_name: pred for pred in predictions}
        resnet = summary_by_model.get("resnet50") or summary_by_model.get("resnet")
        clip = summary_by_model.get("clip")
        resnet_val = _prediction_summary(resnet)
        clip_val = _prediction_summary(clip)
        model_disagreement = bool(resnet and clip and resnet.predicted_class != clip.predicted_class)
        low_confidence = bool(
            resnet and clip and abs(float(resnet.raw_score) - 0.5) <= 0.15 and abs(float(clip.raw_score) - 0.5) <= 0.15
        )
        suspected_false_positive = bool(
            (resnet and resnet.predicted_class == "ai" and float(resnet.raw_score) < 0.65)
            or (clip and clip.predicted_class == "ai" and float(clip.raw_score) < 0.65)
        )
        suspected_false_negative = bool(
            (resnet and resnet.predicted_class == "real" and float(resnet.raw_score) > 0.35)
            or (clip and clip.predicted_class == "real" and float(clip.raw_score) > 0.35)
        )

        items.append(
            ReviewQueueItem(
                image_id=record.id,
                storage_path=record.storage_path,
                safe_filename=record.safe_filename,
                uploaded_at=record.uploaded_at.isoformat(),
                review_status=record.review_status.value,
                resnet=resnet_val,
                clip=clip_val,
                model_disagreement=model_disagreement,
                low_confidence=low_confidence,
                suspected_false_positive=suspected_false_positive,
                suspected_false_negative=suspected_false_negative,
            )
        )

    total = len(items)
    start = (page - 1) * limit
    end = start + limit
    return ReviewQueueResponse(items=items[start:end], total=total, page=page, limit=limit, has_next=(end < total))


@app.post("/reviews/submit", response_model=ReviewSubmissionResponse)
def submit_review(
    payload: ReviewDecisionRequest,
    reviewer_id: str = Depends(require_reviewer),
    db: Session = Depends(get_db),
):
    image = db.get(ImageRecord, payload.image_id)
    if image is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Image not found.")

    decision = payload.decision.lower()
    if decision == "uncertain":
        verified_class = "uncertain"
        review_status = ReviewStatus.NEEDS_REVIEW
    else:
        verified_class = decision
        review_status = ReviewStatus.APPROVED

    if decision == "ai" and (payload.known_generator is None or payload.known_generator.strip() == ""):
        payload.known_generator = "Unknown"

    duplicate = db.execute(
        select(HumanReview).where(
            HumanReview.image_id == payload.image_id,
            HumanReview.reviewer_id == reviewer_id,
            HumanReview.verified_class == verified_class,
            HumanReview.confidence == payload.confidence,
            HumanReview.evidence_source == payload.evidence_source,
            HumanReview.source_notes == payload.source_notes,
        )
    ).scalars().first()
    if duplicate is not None:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="This exact review has already been submitted; create a new review to amend a prior decision.")

    review = storage_service.add_review(
        image_id=payload.image_id,
        prediction_id=payload.prediction_id,
        reviewer_id=reviewer_id,
        verified_class=verified_class,
        known_generator=payload.known_generator,
        evidence_source=payload.evidence_source,
        source_notes=payload.source_notes,
        confidence=payload.confidence,
        review_status=review_status,
    )

    image.review_status = review_status
    db.add(image)
    db.commit()

    return ReviewSubmissionResponse(
        image_id=image.id,
        reviewer_id=reviewer_id,
        decision=verified_class,
        review_status=review_status.value,
        review_id=review.id,
        message="Review submitted successfully.",
    )


@app.get("/images/{image_id}/file")
def get_image_file(image_id: str, db: Session = Depends(get_db)):
    """Serve the raw image bytes for a stored image record.

    Using this endpoint instead of a local filesystem path allows Gradio (and
    any other client) to display previews even when it runs on a different host
    from the API server.
    """
    image = db.get(ImageRecord, image_id)
    if image is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Image not found.")

    file_path = Path(image.storage_path)
    if not file_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Image file is not present on disk.",
        )

    # Derive a safe media type from the stored format
    fmt_map = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp"}
    media_type = fmt_map.get(image.file_format.upper(), "application/octet-stream")

    return FileResponse(
        path=str(file_path),
        media_type=media_type,
        filename=image.safe_filename,
    )


@app.get("/images/{image_id}/history", response_model=PredictionHistoryResponse)
def get_prediction_history(image_id: str, db: Session = Depends(get_db)):
    image = db.get(ImageRecord, image_id)
    if image is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Image not found.")

    predictions = db.execute(
        select(Prediction).where(Prediction.image_id == image_id).order_by(Prediction.inference_timestamp.asc())
    ).scalars().all()

    entries = [
        PredictionHistoryEntry(
            prediction_id=pred.id,
            image_id=pred.image_id,
            model_name=pred.model_name,
            model_version=pred.model_version,
            predicted_class=pred.predicted_class,
            raw_score=pred.raw_score,
            threshold=pred.threshold,
            calibration_status="not_calibrated",
            preprocessing_version=pred.preprocessing_version,
            inference_timestamp=pred.inference_timestamp.isoformat(),
        )
        for pred in predictions
    ]

    return PredictionHistoryResponse(image_id=image_id, predictions=entries)


@app.get("/images", response_model=ListImagesResponse)
def list_images(
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    review_status: str | None = None,
    source: str | None = None,
    db: Session = Depends(get_db),
):
    stmt = select(ImageRecord)
    if review_status:
        try:
            review_value = ReviewStatus(review_status)
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid review_status: {review_status}") from exc
        stmt = stmt.where(ImageRecord.review_status == review_value)
    if source:
        stmt = stmt.where(ImageRecord.source == source)

    total = db.execute(stmt.order_by(ImageRecord.uploaded_at.desc())).scalars().all()
    paged = total[(page - 1) * limit : page * limit]

    items = [
        ImageListItem(
            id=record.id,
            storage_path=record.storage_path,
            safe_filename=record.safe_filename,
            sha256=record.sha256,
            file_size=record.file_size,
            width=record.width,
            height=record.height,
            file_format=record.file_format,
            mime_type=record.mime_type,
            uploaded_at=record.uploaded_at.isoformat(),
            review_status=record.review_status.value,
            source=record.source,
        )
        for record in paged
    ]

    return ListImagesResponse(
        items=items,
        total=len(total),
        page=page,
        limit=limit,
        has_next=(page * limit) < len(total),
    )


@app.post("/predict")
async def predict(file: UploadFile = File(...), model_type: str = "resnet"):
    if not file.filename:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Please upload an image file.")

    content = await file.read()
    try:
        Image.open(io.BytesIO(content)).verify()
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded content is not a valid image.") from exc

    model_type = model_type.lower()
    if model_type not in {"resnet", "clip"}:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="model_type must be 'resnet' or 'clip'.")

    try:
        inference = run_prediction_for_model(model_type, content)
        score = inference.raw_score
        label = inference.predicted_class
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc

    return {
        "model_type": model_type,
        "label": label,
        "raw_score": round(score, 6),
        "threshold": inference.threshold,
        "score_kind": inference.score_kind,
        "calibrated": inference.calibrated,
        "preprocessing_version": inference.preprocessing_version,
        "inference_timestamp": inference.inference_timestamp,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
