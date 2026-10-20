from __future__ import annotations

import os
from pathlib import Path

try:
    import pandas as pd
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover
    pd = None
    torch = None
    nn = None

from .config import WORK_DIR

try:
    from .models import loader, metrics, predict, make_model
except ImportError:  # pragma: no cover
    loader = None
    metrics = None
    predict = None
    make_model = None


def run_training(tag: str, items, val_items=None, epochs: int = 5, pretrained: bool = True, lr: float = 1e-4, weight_decay: float = 1e-4, work_dir: str | os.PathLike | None = None):
    if torch is None or nn is None or pd is None or loader is None or metrics is None or predict is None or make_model is None:
        raise ImportError("PyTorch and pandas are required to train the detector.")
    work_dir = Path(work_dir) if work_dir is not None else WORK_DIR
    ckpt_dir = work_dir / "ckpt"
    results_dir = work_dir / "results"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = make_model(pretrained=pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda") if torch.cuda.is_available() else None
    loss_fn = nn.BCEWithLogitsLoss()

    ck_path = ckpt_dir / f"{tag}_last.pt"
    best_ck_path = ckpt_dir / f"{tag}_best.pt"
    start_epoch = 0
    best_auc = -1.0
    history = []

    if ck_path.exists():
        state = torch.load(ck_path, map_location=device)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        if scaler is not None:
            scaler.load_state_dict(state["scaler"])
        start_epoch = state.get("epoch", 0) + 1
        best_auc = state.get("best_auc", -1.0)
        history = state.get("history", [])
        print(f"resumed from epoch {start_epoch}")

    train_loader = loader(items, train=True, batch_size=64)
    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss = 0.0
        total_count = 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            if torch.cuda.is_available():
                with torch.autocast(device_type="cuda"):
                    logits = model(x).squeeze(1)
                    loss = loss_fn(logits, y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(x).squeeze(1)
                loss = loss_fn(logits, y)
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * len(y)
            total_count += len(y)

        scheduler.step()

        if val_items is not None:
            scores, labels = predict(model, val_items)
            metrics_dict = metrics(scores, labels)
        else:
            metrics_dict = {"acc": None, "auc": None, "tpr_at_1fpr": None, "real_flagged": None}

        row = {"epoch": epoch + 1, "train_loss": round(total_loss / max(total_count, 1), 4), **metrics_dict}
        history.append(row)
        print(row)

        if metrics_dict.get("auc") is not None and metrics_dict["auc"] > best_auc:
            best_auc = metrics_dict["auc"]
            torch.save(model.state_dict(), best_ck_path)

        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict() if scaler is not None else None,
                "epoch": epoch,
                "best_auc": best_auc,
                "history": history,
            },
            ck_path,
        )

    if best_ck_path.exists():
        model.load_state_dict(torch.load(best_ck_path, map_location=device))
    pd.DataFrame(history).to_csv(results_dir / f"{tag}_history.csv", index=False)
    return model


__all__ = ["run_training"]
