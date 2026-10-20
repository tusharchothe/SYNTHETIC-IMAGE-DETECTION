from __future__ import annotations

from pathlib import Path

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

from PIL import Image

try:
    from .models import fit_crop, make_model
except Exception:  # pragma: no cover
    fit_crop = None
    make_model = None


def load_model_from_checkpoint(checkpoint_path: str | Path, pretrained: bool = True):
    if torch is None:
        raise ImportError("PyTorch is required to load a detector checkpoint.")
    model = make_model(pretrained=pretrained)
    state = torch.load(str(checkpoint_path), map_location="cpu")
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    return model.eval()


def score_image(model, image_path: str | Path, threshold: float = 0.5):
    image = Image.open(image_path).convert("RGB")
    image = fit_crop(image, 256)
    tensor = torch.tensor((__import__("numpy").array(image).astype("float32") / 255.0)).permute(2, 0, 1)
    tensor = tensor.unsqueeze(0)
    tensor = (tensor - torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)) / torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    device = next(model.parameters()).device
    tensor = tensor.to(device)
    with torch.no_grad():
        prob = torch.sigmoid(model(tensor).squeeze(1)).cpu().item()
    return prob, prob >= threshold


def score_label(model, image_path: str | Path):
    prob, is_ai = score_image(model, image_path)
    label = "ai" if is_ai else "real"
    return {"probability": prob, "label": label}


__all__ = ["load_model_from_checkpoint", "score_image", "score_label"]
