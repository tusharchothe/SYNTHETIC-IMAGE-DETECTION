from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class PredictionResponse(BaseModel):
    model_name: str
    model_version: str
    predicted_class: str
    raw_score: float
    threshold: float
    score_kind: Literal["raw_model_score"] = "raw_model_score"
    calibrated: bool = False
    preprocessing_version: str
    inference_timestamp: str
    warnings: list[str] = Field(default_factory=list)


class UploadImageResponse(BaseModel):
    image_id: str
    storage_path: str
    review_status: str
    input_warnings: list[str] = Field(default_factory=list)
    predictions: list[PredictionResponse]


class PredictionHistoryEntry(BaseModel):
    prediction_id: str
    image_id: str
    model_name: str
    model_version: str
    predicted_class: str
    raw_score: float
    threshold: float
    calibration_status: str = "not_calibrated"
    preprocessing_version: str
    inference_timestamp: str


class PredictionHistoryResponse(BaseModel):
    image_id: str
    predictions: list[PredictionHistoryEntry]


class ImageListItem(BaseModel):
    id: str
    storage_path: str
    safe_filename: str
    sha256: str
    file_size: int
    width: int
    height: int
    file_format: str
    mime_type: str
    uploaded_at: str
    review_status: str
    source: str | None = None


class ListImagesResponse(BaseModel):
    items: list[ImageListItem]
    total: int
    page: int
    limit: int
    has_next: bool


class ReviewDecisionRequest(BaseModel):
    image_id: str
    decision: Literal["real", "ai", "uncertain"]
    known_generator: str | None = None
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    evidence_source: str | None = None
    source_notes: str | None = None
    prediction_id: str | None = None


class ReviewPredictionSummary(BaseModel):
    model_name: str
    model_version: str
    predicted_class: str
    raw_score: float
    threshold: float
    preprocessing_version: str
    inference_timestamp: str


class ReviewQueueItem(BaseModel):
    image_id: str
    storage_path: str
    safe_filename: str
    uploaded_at: str
    review_status: str
    resnet: ReviewPredictionSummary | None = None
    clip: ReviewPredictionSummary | None = None
    model_disagreement: bool = False
    low_confidence: bool = False
    suspected_false_positive: bool = False
    suspected_false_negative: bool = False


class ReviewQueueResponse(BaseModel):
    items: list[ReviewQueueItem]
    total: int
    page: int
    limit: int
    has_next: bool


class ReviewSubmissionResponse(BaseModel):
    image_id: str
    reviewer_id: str
    decision: str
    review_status: str
    review_id: str
    message: str
