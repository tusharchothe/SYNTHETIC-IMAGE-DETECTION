from __future__ import annotations

import io
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import torch
from PIL import Image

from v1.config import ROOT
from v1.models import fit_crop

MODEL_PATH = Path(os.environ.get("AIDETECT_MODEL_PATH", ROOT / "resnet50_sd14_best.pt"))
CLIP_MODEL_PATH = Path(os.environ.get("AIDETECT_CLIP_MODEL_PATH", ROOT / "clip_probe.joblib"))

RESNET_THRESHOLD = 0.5
CLIP_THRESHOLD = 0.5


@dataclass(frozen=True)
class InferenceResult:
    model_name: str
    model_version: str
    predicted_class: str
    raw_score: float
    threshold: float
    score_kind: str
    calibrated: bool
    preprocessing_version: str
    inference_timestamp: str
    warnings: list[str] | None = None

    @property
    def decision_label(self) -> str:
        return "ai" if self.raw_score >= self.threshold else "real"


@lru_cache(maxsize=1)
def _load_resnet_model_cached():
    from v1.models import make_model

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"ResNet checkpoint not found at {MODEL_PATH}")

    model = make_model(pretrained=False)
    state = torch.load(str(MODEL_PATH), map_location="cpu")
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.eval()
    return model


@lru_cache(maxsize=1)
def _load_clip_model_cached():
    if not CLIP_MODEL_PATH.exists():
        raise FileNotFoundError(f"CLIP probe model not found at {CLIP_MODEL_PATH}")

    import open_clip

    clf = joblib.load(str(CLIP_MODEL_PATH))
    clip_model, _, preprocess = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai")
    clip_model = clip_model.eval().half().cuda() if torch.cuda.is_available() else clip_model.eval()
    return clf, clip_model, preprocess


def _as_resnet_score(image_bytes: bytes) -> tuple[float, list[str]]:
    warnings: list[str] = []
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = fit_crop(image, 256)
        array = np.array(image).astype("float32") / 255.0
        tensor = torch.tensor(array).permute(2, 0, 1).unsqueeze(0)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        tensor = (tensor - mean) / std

        model = _load_resnet_model_cached()
        device = next(model.parameters()).device
        tensor = tensor.to(device)
        with torch.no_grad():
            score = torch.sigmoid(model(tensor).squeeze(1)).cpu().item()
        return float(score), warnings
    except Exception as exc:  # pragma: no cover - allowed to surface as a model failure warning
        warnings.append(f"ResNet inference failed: {exc}")
        raise RuntimeError("ResNet inference failed.") from exc


def _as_clip_score(image_bytes: bytes) -> tuple[float, list[str]]:
    warnings: list[str] = []
    try:
        clf, clip_model, preprocess = _load_clip_model_cached()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        x = preprocess(image).unsqueeze(0)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        x = x.to(device)
        with torch.no_grad():
            feats = clip_model.encode_image(x)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            probs = clf.predict_proba(feats.float().cpu().numpy())[:, 1]
        return float(probs[0]), warnings
    except Exception as exc:  # pragma: no cover
        warnings.append(f"CLIP inference failed: {exc}")
        raise RuntimeError("CLIP inference failed.") from exc


def run_resnet_prediction(image_bytes: bytes) -> InferenceResult:
    score, warnings = _as_resnet_score(image_bytes)
    return InferenceResult(
        model_name="resnet50",
        model_version=MODEL_PATH.name,
        predicted_class="ai" if score >= RESNET_THRESHOLD else "real",
        raw_score=float(score),
        threshold=RESNET_THRESHOLD,
        score_kind="raw_model_score",
        calibrated=False,
        preprocessing_version="fit_crop_256_imagenet_norm_v1",
        inference_timestamp=datetime.now(timezone.utc).isoformat(),
        warnings=warnings,
    )


def run_clip_prediction(image_bytes: bytes) -> InferenceResult:
    score, warnings = _as_clip_score(image_bytes)
    return InferenceResult(
        model_name="clip",
        model_version="ViT-L-14/openai",
        predicted_class="ai" if score >= CLIP_THRESHOLD else "real",
        raw_score=float(score),
        threshold=CLIP_THRESHOLD,
        score_kind="raw_model_score",
        calibrated=False,
        preprocessing_version="clip_vit_l_14_openai_preprocess_v1",
        inference_timestamp=datetime.now(timezone.utc).isoformat(),
        warnings=warnings,
    )


def run_prediction_for_model(model_name: str, image_bytes: bytes) -> InferenceResult:
    if model_name == "resnet":
        return run_resnet_prediction(image_bytes)
    if model_name == "clip":
        return run_clip_prediction(image_bytes)
    raise ValueError(f"Unsupported model: {model_name}")


__all__ = [
    "InferenceResult",
    "run_resnet_prediction",
    "run_clip_prediction",
    "run_prediction_for_model",
    "RESNET_THRESHOLD",
    "CLIP_THRESHOLD",
]
