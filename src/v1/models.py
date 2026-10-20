from __future__ import annotations

import io
import os
from typing import Callable, Iterable, List, Tuple

import numpy as np

try:
    import torch
    from torch.utils.data import DataLoader, Dataset
except Exception:  # pragma: no cover
    torch = None
    DataLoader = None
    Dataset = object

if Dataset is None:
    Dataset = object

try:
    import torchvision.transforms as T
    from torchvision.models import ResNet50_Weights, resnet50
except Exception:  # pragma: no cover
    T = None
    ResNet50_Weights = None
    resnet50 = None

from PIL import Image, ImageFilter

from .config import CLIP_IMG_SIZE, IMG_SIZE


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def fit_crop(image: Image.Image, size: int) -> Image.Image:
    width, height = image.size
    if width < size or height < size:
        pad_color = (0, 0, 0)
        new_w = max(width, size)
        new_h = max(height, size)
        canvas = Image.new("RGB", (new_w, new_h), color=pad_color)
        offset_x = (new_w - width) // 2
        offset_y = (new_h - height) // 2
        canvas.paste(image, (offset_x, offset_y))
        image = canvas
        width, height = image.size

    scale = size / min(width, height)
    resized = image.resize((max(1, int(round(width * scale))), max(1, int(round(height * scale)))), Image.BILINEAR)
    left = max(0, (resized.width - size) // 2)
    top = max(0, (resized.height - size) // 2)
    return resized.crop((left, top, left + size, top + size))


def rand_jpeg(image: Image.Image, p: float = 0.5, q: int = 90):
    if np.random.rand() < p:
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=q)
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")
    return image


def jpeg_deg(q: int) -> Callable[[Image.Image], Image.Image]:
    def _apply(image: Image.Image) -> Image.Image:
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=q)
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")

    return _apply


def blur_deg(sigma: float) -> Callable[[Image.Image], Image.Image]:
    return lambda image: image.filter(ImageFilter.GaussianBlur(sigma))


def lowres_deg(size: int) -> Callable[[Image.Image], Image.Image]:
    def _apply(image: Image.Image) -> Image.Image:
        return image.resize((size, size), Image.BILINEAR).resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)

    return _apply


def make_model(pretrained: bool = True):
    if torch is None:
        raise ImportError("PyTorch is required to train or use the detector model.")
    if resnet50 is None or ResNet50_Weights is None:
        raise ImportError("A compatible torchvision installation is required for ResNet model creation. Install matching torch/torchvision versions.")
    weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    model = resnet50(weights=weights)
    model.fc = torch.nn.Linear(model.fc.in_features, 1)
    return model


class ImagePairDataset(Dataset if Dataset is not None else object):
    def __init__(self, items, transform=None):
        self.items = items
        self.transform = transform or self._default_transform()

    @staticmethod
    def _default_transform():
        return T.Compose(
            [
                T.Resize((IMG_SIZE, IMG_SIZE)),
                T.ToTensor(),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        path, label = self.items[index]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        y = torch.tensor(float(label), dtype=torch.float32)
        return image, y


def loader(items, train: bool = False, batch_size: int = 64, num_workers: int = 2):
    if torch is None or DataLoader is None:
        raise ImportError("PyTorch is required for data loading.")
    if T is None:
        raise ImportError("A compatible torchvision installation is required for data loading and transforms.")
    transform = None
    if train:
        transform = T.Compose(
            [
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(p=0.05),
                T.Resize((IMG_SIZE, IMG_SIZE)),
                T.ToTensor(),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )
    else:
        transform = T.Compose(
            [
                T.Resize((IMG_SIZE, IMG_SIZE)),
                T.ToTensor(),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

    dataset = ImagePairDataset(items, transform=transform)
    shuffle = train
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=torch.cuda.is_available())


def predict(model, items, degrade=None, batch_size: int = 128):
    if torch is None or DataLoader is None:
        raise ImportError("PyTorch is required for prediction.")
    if T is None:
        raise ImportError("A compatible torchvision installation is required for prediction transforms.")
    device = next(model.parameters()).device
    eval_transform = T.Compose(
        [
            T.Resize((IMG_SIZE, IMG_SIZE)),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    def transform(image):
        image = image.convert("RGB")
        if degrade is not None:
            image = degrade(image)
        return eval_transform(image)

    preds, labels = [], []
    model.eval()
    with torch.no_grad():
        for batch in DataLoader(ImagePairDataset(items, transform=transform), batch_size=batch_size, shuffle=False, num_workers=2):
            x, y = batch
            x = x.to(device)
            out = model(x).squeeze(1)
            preds.append(torch.sigmoid(out).detach().cpu().numpy())
            labels.append(y.detach().cpu().numpy())

    return np.concatenate(preds), np.concatenate(labels)


def metrics(scores, labels):
    scores = np.asarray(scores, dtype=float).ravel()
    labels = np.asarray(labels, dtype=int).ravel()

    if len(np.unique(labels)) < 2:
        return {"acc": float("nan"), "auc": float("nan"), "tpr_at_1fpr": float("nan"), "real_flagged": float("nan")}

    acc = float(((scores >= 0.5).astype(int) == labels).mean())
    from sklearn.metrics import roc_auc_score, roc_curve

    auc = float(roc_auc_score(labels, scores))
    fpr, tpr, _ = roc_curve(labels, scores)
    idx = np.argmin(np.abs(fpr - 0.01))
    tpr_at_1fpr = float(tpr[idx] * 100.0)
    real_flagged = float(((scores > 0.5) & (labels == 0)).sum() / max((labels == 0).sum(), 1))
    return {
        "acc": acc,
        "auc": auc,
        "tpr_at_1fpr": tpr_at_1fpr,
        "real_flagged": real_flagged * 100.0,
    }


__all__ = [
    "fit_crop",
    "rand_jpeg",
    "jpeg_deg",
    "blur_deg",
    "lowres_deg",
    "make_model",
    "loader",
    "predict",
    "metrics",
    "ImagePairDataset",
]
