from __future__ import annotations

import argparse

from .config import WORK_DIR


def train():
    from .data import build_dataset_lists
    from .training import run_training
    parser = argparse.ArgumentParser(description="Train the AI image detector.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--work-dir", type=str, default=str(WORK_DIR))
    args = parser.parse_args()

    dataset = build_dataset_lists()
    run_training("resnet50_sd14", dataset["train"], dataset["val"], epochs=args.epochs, pretrained=args.pretrained, lr=args.lr, work_dir=args.work_dir)


def evaluate():
    from .data import build_dataset_lists
    from .inference import load_model_from_checkpoint
    from .models import metrics, predict

    parser = argparse.ArgumentParser(description="Evaluate a trained detector on a dataset split.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--generator", type=str, default="sd14")
    args = parser.parse_args()

    dataset = build_dataset_lists()
    model = load_model_from_checkpoint(args.checkpoint, pretrained=False)
    items = dataset[f"test_{args.generator}"] if args.split == "test" else dataset[args.split]
    scores, labels = predict(model, items)
    print(metrics(scores, labels))


def predict():
    from .inference import load_model_from_checkpoint, score_label

    parser = argparse.ArgumentParser(description="Run inference on a single image.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
    args = parser.parse_args()

    model = load_model_from_checkpoint(args.checkpoint, pretrained=False)
    result = score_label(model, args.image)
    print(result)


def dataset_builder():
    from .dataset_builder import main as builder_main

    raise SystemExit(builder_main())


def main():
    parser = argparse.ArgumentParser(description="AI image detection project")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("train", help="train the detector").set_defaults(func=train)
    subparsers.add_parser("evaluate", help="evaluate the detector").set_defaults(func=evaluate)
    subparsers.add_parser("predict", help="score an image").set_defaults(func=predict)
    subparsers.add_parser("dataset-builder", help="build a versioned dataset manifest").set_defaults(func=dataset_builder)
    args = parser.parse_args()
    args.func()


if __name__ == "__main__":
    main()
