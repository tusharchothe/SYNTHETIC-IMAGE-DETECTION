from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_STORAGE_DIR = Path(os.getenv("AIDETECT_STORAGE_DIR", str(ROOT / "storage")))
DEFAULT_DATABASE_URL = os.getenv(
    "AIDETECT_DATABASE_URL",
    f"sqlite:///{ROOT / 'ai_detector.db'}",
)
MAX_IMAGE_SIZE_BYTES = int(os.getenv("AIDETECT_MAX_IMAGE_SIZE_BYTES", str(8 * 1024 * 1024)))
ALLOWED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
ALLOWED_MIME_TYPES = {"image/png", "image/jpeg", "image/webp"}
DATABASE_ECHO = os.getenv("AIDETECT_DATABASE_ECHO", "false").lower() == "true"

__all__ = [
    "ROOT",
    "DEFAULT_STORAGE_DIR",
    "DEFAULT_DATABASE_URL",
    "MAX_IMAGE_SIZE_BYTES",
    "ALLOWED_IMAGE_EXTENSIONS",
    "ALLOWED_MIME_TYPES",
    "DATABASE_ECHO",
]
