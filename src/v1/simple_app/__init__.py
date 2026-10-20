"""Simple no-auth local workflow for AI image detection.

Provides a self-contained FastAPI backend and Gradio UI that:
- Accept one or more image uploads
- Store images under data/uploads/ with unique IDs
- Run ResNet-50 and CLIP inference pipelines
- Persist everything in a local SQLite database
- Maintain a pending-review queue

No authentication, user roles, cloud infrastructure, or automatic retraining.
Existing model architectures and preprocessing are reused unchanged.
"""

__all__ = ["api", "config", "service", "ui"]
