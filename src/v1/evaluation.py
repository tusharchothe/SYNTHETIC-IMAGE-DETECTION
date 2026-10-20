from __future__ import annotations

from typing import Dict, Iterable, List

import pandas as pd

from .config import GENS, ZIPS
from .models import metrics, predict


def evaluate_model(model, items):
    scores, labels = predict(model, items)
    return metrics(scores, labels)


def evaluate_generator_suite(model, dataset_map: Dict[str, list], generators: Iterable[str] | None = None):
    generators = list(generators) if generators is not None else list(GENS.keys())
    rows = []
    for generator in generators:
        key = f"test_{generator}"
        if key not in dataset_map:
            continue
        scores, labels = predict(model, dataset_map[key])
        row = {"generator": generator, **metrics(scores, labels)}
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_generator_results(results: pd.DataFrame):
    return results.pivot_table(index="generator", values=["acc", "auc", "tpr_at_1fpr", "real_flagged"], aggfunc="mean")


__all__ = ["evaluate_model", "evaluate_generator_suite", "summarize_generator_results"]
