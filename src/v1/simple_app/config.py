"""Configuration for the simple no-auth local workflow.

All paths default to data/ inside the project root so the workflow is
completely self-contained.  Every value can be overridden with an environment
variable without touching code.
"""

from __future__ import annotations

import os
from pathlib import Path

# ── Project root (three levels up from this file: simple_app → v1 → src → root)
_ROOT = Path(__file__).resolve().parents[3]

# ── Load .env if present (reuse existing loader so values are consistent)
try:
    from v1.config import _load_default_env
    _load_default_env()
except Exception:  # pragma: no cover
    pass

# ── Storage ------------------------------------------------------------------
# Images are written to data/uploads/ by default, separate from the full
# workflow's storage/ directory so the two modes don't collide.
UPLOAD_DIR = Path(os.getenv("SIMPLE_UPLOAD_DIR", str(_ROOT / "data" / "uploads")))

# ── Database -----------------------------------------------------------------
# Separate SQLite database so this workflow is fully isolated from the main DB.
DATABASE_URL = os.getenv(
    "SIMPLE_DATABASE_URL",
    f"sqlite:///{_ROOT / 'data' / 'simple_app.db'}",
)

# ── File validation ----------------------------------------------------------
MAX_FILE_SIZE_BYTES: int = int(
    os.getenv("SIMPLE_MAX_FILE_SIZE_BYTES", str(8 * 1024 * 1024))  # 8 MB
)
ALLOWED_EXTENSIONS: frozenset[str] = frozenset({".png", ".jpg", ".jpeg", ".webp"})
ALLOWED_MIME_TYPES: frozenset[str] = frozenset({"image/png", "image/jpeg", "image/webp"})

# ── Server -------------------------------------------------------------------
API_HOST: str = os.getenv("SIMPLE_API_HOST", "0.0.0.0")
API_PORT: int = int(os.getenv("SIMPLE_API_PORT", "8100"))

UI_HOST: str = os.getenv("SIMPLE_UI_HOST", "0.0.0.0")
UI_PORT: int = int(os.getenv("SIMPLE_UI_PORT", "7900"))

# Internal base URL that the UI uses to talk to the API.
# When both run on the same machine this is always localhost:<API_PORT>.
API_BASE_URL: str = os.getenv("SIMPLE_API_BASE_URL", f"http://localhost:{API_PORT}")

__all__ = [
    "_ROOT",
    "UPLOAD_DIR",
    "DATABASE_URL",
    "MAX_FILE_SIZE_BYTES",
    "ALLOWED_EXTENSIONS",
    "ALLOWED_MIME_TYPES",
    "API_HOST",
    "API_PORT",
    "UI_HOST",
    "UI_PORT",
    "API_BASE_URL",
]
