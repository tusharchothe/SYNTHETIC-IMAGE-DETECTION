from __future__ import annotations

import hashlib
import io
import os
import re
import uuid
from pathlib import Path
from typing import Iterable

from PIL import Image, UnidentifiedImageError
from sqlalchemy import select
from sqlalchemy.orm import Session

from .config import ALLOWED_IMAGE_EXTENSIONS, ALLOWED_MIME_TYPES, MAX_IMAGE_SIZE_BYTES, DEFAULT_STORAGE_DIR
from .models import HumanReview, ImageRecord, Prediction, ReviewStatus


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _safe_name(raw_name: str | None, digest: str, extension: str) -> str:
    sanitized = (raw_name or "upload").strip()
    sanitized = os.path.basename(sanitized)
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", sanitized).strip("._") or "upload"
    safe_stem = sanitized.rsplit(".", 1)[0] if "." in sanitized else sanitized
    safe_stem = re.sub(r"[^A-Za-z0-9_-]+", "_", safe_stem)[:80] or "image"
    return f"{safe_stem}_{digest[:12]}{extension.lower()}"


def _perceptual_hash(image: Image.Image, size: int = 8) -> str:
    gray = image.convert("L").resize((size, size), Image.Resampling.LANCZOS)
    pixels = list(gray.getdata())
    avg = sum(pixels) / len(pixels)
    bits = "".join("1" if p >= avg else "0" for p in pixels)
    return bits


def _hamming_distance(a: str, b: str) -> int:
    if len(a) != len(b):
        raise ValueError("Perceptual hashes must be the same length.")
    return sum(ch1 != ch2 for ch1, ch2 in zip(a, b))


class ImageStorageService:
    def __init__(self, session_factory, storage_dir: str | None = None, max_file_size_bytes: int = MAX_IMAGE_SIZE_BYTES):
        self.session_factory = session_factory
        self.storage_dir = Path(storage_dir or DEFAULT_STORAGE_DIR)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.max_file_size_bytes = max_file_size_bytes

    def _session(self) -> Session:
        return self.session_factory()

    def _decode_and_validate(self, image_bytes: bytes, uploaded_filename: str | None = None, content_type: str | None = None):
        if not image_bytes:
            raise ValueError("Uploaded image is empty.")
        if len(image_bytes) > self.max_file_size_bytes:
            raise ValueError(f"Image exceeds the maximum allowed size of {self.max_file_size_bytes} bytes.")

        if content_type and content_type not in ALLOWED_MIME_TYPES:
            raise ValueError(f"Image MIME type '{content_type}' is not allowed.")

        try:
            image = Image.open(io.BytesIO(image_bytes))
            image.load()
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise ValueError("Uploaded content is not a valid image or could not be decoded.") from exc

        if image.format is None:
            raise ValueError("Image format could not be detected.")

        fmt = str(image.format).upper()
        allowed_formats = {"PNG", "JPEG", "WEBP"}
        if fmt not in allowed_formats:
            raise ValueError(f"File format '{fmt}' is not supported.")

        if uploaded_filename:
            ext = Path(uploaded_filename).suffix.lower()
            if ext not in ALLOWED_IMAGE_EXTENSIONS:
                raise ValueError(f"Uploaded filename extension '{ext}' is not allowed.")

        width, height = image.size
        if width <= 0 or height <= 0:
            raise ValueError("Image dimensions are invalid.")

        return image, fmt

    def _find_existing_duplicate(self, session: Session, sha256_hash: str):
        return session.execute(select(ImageRecord).where(ImageRecord.sha256 == sha256_hash)).scalar_one_or_none()

    def find_near_duplicates(self, session: Session, perceptual_hash: str, threshold: int = 4):
        candidates = session.execute(select(ImageRecord).where(ImageRecord.perceptual_hash.isnot(None))).scalars().all()
        matches = []
        for candidate in candidates:
            distance = _hamming_distance(candidate.perceptual_hash, perceptual_hash)
            if distance <= threshold:
                matches.append(candidate)
        return matches

    def store_image(self, image_bytes: bytes, uploaded_filename: str | None, content_type: str | None = None, source: str | None = None):
        image, file_format = self._decode_and_validate(image_bytes, uploaded_filename, content_type)
        digest = _sha256(image_bytes)

        session = self._session()
        try:
            existing = self._find_existing_duplicate(session, digest)
            if existing is not None:
                return existing

            width, height = image.size
            perceptual = _perceptual_hash(image)
            safe_name = _safe_name(uploaded_filename, digest, f".{file_format.lower()}")
            storage_path = self.storage_dir / safe_name
            counter = 1
            while storage_path.exists():
                stem = Path(safe_name).stem
                suffix = Path(safe_name).suffix
                storage_path = self.storage_dir / f"{stem}_{counter}{suffix}"
                counter += 1

            storage_path.write_bytes(image_bytes)

            record = ImageRecord(
                id=str(uuid.uuid4()),
                storage_path=str(storage_path),
                original_filename=uploaded_filename,
                safe_filename=storage_path.name,
                sha256=digest,
                perceptual_hash=perceptual,
                file_size=len(image_bytes),
                width=width,
                height=height,
                file_format=file_format,
                mime_type=content_type or "image/unknown",
                source=source,
                review_status=ReviewStatus.PENDING,
            )

            near_duplicates = self.find_near_duplicates(session, perceptual)
            if near_duplicates:
                record.review_status = ReviewStatus.NEEDS_REVIEW

            session.add(record)
            session.commit()
            session.refresh(record)
            return record
        finally:
            session.close()

    def add_prediction(self, image_id: str, model_name: str, model_version: str, predicted_class: str, raw_score: float, threshold: float, preprocessing_version: str = "unknown"):
        session = self._session()
        try:
            prediction = Prediction(
                id=str(uuid.uuid4()),
                image_id=image_id,
                model_name=model_name,
                model_version=model_version,
                predicted_class=predicted_class,
                raw_score=float(raw_score),
                threshold=float(threshold),
                preprocessing_version=preprocessing_version,
            )
            session.add(prediction)
            session.commit()
            session.refresh(prediction)
            return prediction
        finally:
            session.close()

    def add_review(self, image_id: str, prediction_id: str | None, reviewer_id: str, verified_class: str, known_generator: str | None = None, evidence_source: str | None = None, source_notes: str | None = None, confidence: float = 0.0, review_status: ReviewStatus = ReviewStatus.APPROVED):
        session = self._session()
        try:
            image = session.get(ImageRecord, image_id)
            if image is None:
                raise ValueError(f"Image {image_id} does not exist.")

            review = HumanReview(
                id=str(uuid.uuid4()),
                image_id=image_id,
                prediction_id=prediction_id,
                reviewer_id=reviewer_id,
                verified_class=verified_class,
                known_generator=known_generator,
                evidence_source=evidence_source,
                source_notes=source_notes,
                confidence=float(confidence),
                review_status=review_status,
            )
            session.add(review)
            image.review_status = review_status
            session.commit()
            session.refresh(review)
            return review
        finally:
            session.close()


__all__ = ["ImageStorageService"]
