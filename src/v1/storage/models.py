from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum as PyEnum
from typing import Optional

from sqlalchemy import DateTime, Enum, Float, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .database import Base


class ReviewStatus(str, PyEnum):
    PENDING = "pending"
    NEEDS_REVIEW = "needs_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    DUPLICATE = "duplicate"


class ImageRecord(Base):
    __tablename__ = "image_records"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: str(uuid.uuid4()))
    storage_path: Mapped[str] = mapped_column(String(512), unique=True, nullable=False)
    original_filename: Mapped[str | None] = mapped_column(String(255), nullable=True)
    safe_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    sha256: Mapped[str] = mapped_column(String(128), unique=True, nullable=False, index=True)
    perceptual_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    uploaded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)
    file_size: Mapped[int] = mapped_column(Integer, nullable=False)
    width: Mapped[int] = mapped_column(Integer, nullable=False)
    height: Mapped[int] = mapped_column(Integer, nullable=False)
    file_format: Mapped[str] = mapped_column(String(32), nullable=False)
    mime_type: Mapped[str] = mapped_column(String(64), nullable=False)
    source: Mapped[str | None] = mapped_column(String(255), nullable=True)
    review_status: Mapped[ReviewStatus] = mapped_column(Enum(ReviewStatus, name="review_status"), default=ReviewStatus.PENDING, nullable=False, index=True)
    duplicate_of_id: Mapped[str | None] = mapped_column(String(64), ForeignKey("image_records.id"), nullable=True)

    predictions: Mapped[list["Prediction"]] = relationship(back_populates="image", cascade="all, delete-orphan")
    reviews: Mapped[list["HumanReview"]] = relationship(back_populates="image", cascade="all, delete-orphan")

    __table_args__ = (
        UniqueConstraint("sha256", name="uq_image_sha256"),
        UniqueConstraint("storage_path", name="uq_image_storage_path"),
    )


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: str(uuid.uuid4()))
    image_id: Mapped[str] = mapped_column(String(64), ForeignKey("image_records.id"), nullable=False, index=True)
    model_name: Mapped[str] = mapped_column(String(128), nullable=False)
    model_version: Mapped[str] = mapped_column(String(128), nullable=False)
    predicted_class: Mapped[str] = mapped_column(String(32), nullable=False)
    raw_score: Mapped[float] = mapped_column(Float, nullable=False)
    threshold: Mapped[float] = mapped_column(Float, nullable=False)
    preprocessing_version: Mapped[str] = mapped_column(String(256), nullable=False, default="unknown")
    inference_timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)

    image: Mapped[ImageRecord] = relationship(back_populates="predictions")


class HumanReview(Base):
    __tablename__ = "human_reviews"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: str(uuid.uuid4()))
    image_id: Mapped[str] = mapped_column(String(64), ForeignKey("image_records.id"), nullable=False, index=True)
    prediction_id: Mapped[str | None] = mapped_column(String(64), ForeignKey("predictions.id"), nullable=True, index=True)
    reviewer_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    verified_class: Mapped[str] = mapped_column(String(32), nullable=False)
    known_generator: Mapped[str | None] = mapped_column(String(128), nullable=True)
    evidence_source: Mapped[str | None] = mapped_column(String(255), nullable=True)
    source_notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    review_status: Mapped[ReviewStatus] = mapped_column(Enum(ReviewStatus, name="review_status"), default=ReviewStatus.PENDING, nullable=False, index=True)
    reviewed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)

    image: Mapped[ImageRecord] = relationship(back_populates="reviews")
    prediction: Mapped[Optional[Prediction]] = relationship()


__all__ = ["Base", "ImageRecord", "Prediction", "HumanReview", "ReviewStatus"]
