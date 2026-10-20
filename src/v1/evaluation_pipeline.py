from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Iterable

import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score


@dataclass
class EvaluationCriteria:
    max_authentic_false_positive_rate: float = 0.05
    min_accuracy: float = 0.8
    min_precision: float = 0.7
    min_recall: float = 0.7
    min_f1: float = 0.7
    min_auc: float = 0.8
    min_average_precision: float = 0.75
    min_delta_auc: float = 0.0
    max_auc_drop: float = 0.0
    generator_guardrails: dict[str, dict[str, float]] = field(default_factory=dict)
    require_calibration: bool = False
    evaluate_three_way: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_authentic_false_positive_rate": self.max_authentic_false_positive_rate,
            "min_accuracy": self.min_accuracy,
            "min_precision": self.min_precision,
            "min_recall": self.min_recall,
            "min_f1": self.min_f1,
            "min_auc": self.min_auc,
            "min_average_precision": self.min_average_precision,
            "min_delta_auc": self.min_delta_auc,
            "max_auc_drop": self.max_auc_drop,
            "generator_guardrails": self.generator_guardrails,
            "require_calibration": self.require_calibration,
            "evaluate_three_way": self.evaluate_three_way,
        }


def _bootstrap_ci(values: np.ndarray, metric_fn: Callable[[np.ndarray], float], ci_level: float = 0.95, n_boot: int = 200) -> tuple[float, float]:
    rng = np.random.default_rng(42)
    boot = []
    n = len(values)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot.append(metric_fn(values[idx]))
    low = np.quantile(boot, (1.0 - ci_level) / 2)
    high = np.quantile(boot, 1.0 - (1.0 - ci_level) / 2)
    return float(low), float(high)


def _metric_summary(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> dict[str, Any]:
    y_true = np.asarray(y_true, dtype=int).ravel()
    y_score = np.asarray(y_score, dtype=float).ravel()
    pred = (y_score >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    acc = accuracy_score(y_true, pred)
    precision = precision_score(y_true, pred, zero_division=0)
    recall = recall_score(y_true, pred, zero_division=0)
    f1 = f1_score(y_true, pred, zero_division=0)
    auc = roc_auc_score(y_true, y_score) if len(np.unique(y_true)) == 2 else None
    ap = average_precision_score(y_true, y_score) if len(np.unique(y_true)) == 2 else None
    fpr = fp / max(fp + tn, 1)
    tpr = tp / max(tp + fn, 1)

    ci = {
        "accuracy_95ci": _bootstrap_ci(y_true, lambda arr: accuracy_score(arr[:len(arr)//2], pred[:len(pred)//2]) if False else acc),
    }
    ci = {}
    ci["accuracy_95ci"] = _binomial_ci(acc, len(y_true))
    ci["precision_95ci"] = _binomial_ci(precision, max(tp + fp, 1))
    ci["recall_95ci"] = _binomial_ci(recall, max(tp + fn, 1))
    ci["f1_95ci"] = _binomial_ci(f1, len(y_true))
    if auc is not None:
        ci["auc_95ci"] = _bootstrap_ci(y_true, lambda arr: roc_auc_score(arr, y_score[:len(y_score)]), ci_level=0.95)
    if ap is not None:
        ci["ap_95ci"] = _bootstrap_ci(y_true, lambda arr: average_precision_score(arr, y_score[:len(y_score)]), ci_level=0.95)

    return {
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(auc) if auc is not None else None,
        "average_precision": float(ap) if ap is not None else None,
        "false_positive_rate": float(fpr),
        "true_positive_rate": float(tpr),
        "false_positive_count": int(fp),
        "false_negative_count": int(fn),
        "confidence_intervals_95": ci,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def _binomial_ci(score: float, denominator: int) -> tuple[float, float]:
    if denominator <= 0:
        return (0.0, 0.0)
    se = math.sqrt(max(score * (1.0 - score), 0.0) / denominator)
    z = 1.96
    margin = z * se
    return (max(0.0, score - margin), min(1.0, score + margin))


def _group_rows(rows: Iterable[dict[str, Any]], key_name: str) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        value = row.get(key_name)
        if value is None:
            value = "unknown"
        groups[str(value)].append(row)
    return groups


def _false_examples(rows: list[dict[str, Any]], y_true: np.ndarray, y_score: np.ndarray, threshold: float, label_name: str) -> list[dict[str, Any]]:
    examples = []
    for idx, row in enumerate(rows):
        pred = 1 if y_score[idx] >= threshold else 0
        actual = int(y_true[idx])
        if label_name == "false_positive" and actual == 0 and pred == 1:
            examples.append({"image_id": row.get("image_id"), "score": float(y_score[idx]), "path": row.get("storage_path"), "label": row.get("human_label")})
        if label_name == "false_negative" and actual == 1 and pred == 0:
            examples.append({"image_id": row.get("image_id"), "score": float(y_score[idx]), "path": row.get("storage_path"), "label": row.get("human_label")})
    return examples[:5]


def evaluate_candidate_scores(
    benchmark_rows: list[dict[str, Any]],
    labels: np.ndarray,
    candidate_scores: np.ndarray,
    production_scores: np.ndarray,
    *,
    threshold: float = 0.5,
    criteria: EvaluationCriteria | None = None,
    calibration_status: dict[str, Any] | None = None,
) -> dict[str, Any]:
    criteria = criteria or EvaluationCriteria()
    labels = np.asarray(labels, dtype=int).ravel()
    candidate_scores = np.asarray(candidate_scores, dtype=float).ravel()
    production_scores = np.asarray(production_scores, dtype=float).ravel()

    assert len(labels) == len(candidate_scores) == len(production_scores) == len(benchmark_rows), "Benchmark rows, labels, and scores must align exactly."

    candidate_summary = _metric_summary(labels, candidate_scores, threshold)
    production_summary = _metric_summary(labels, production_scores, threshold)
    summary = {
        "candidate": {"threshold": threshold, **candidate_summary},
        "production": {"threshold": threshold, **production_summary},
        "candidate_vs_production_delta": {
            "accuracy_delta": float(candidate_summary["accuracy"] - production_summary["accuracy"]),
            "precision_delta": float(candidate_summary["precision"] - production_summary["precision"]),
            "recall_delta": float(candidate_summary["recall"] - production_summary["recall"]),
            "f1_delta": float(candidate_summary["f1"] - production_summary["f1"]),
            "auc_delta": float((candidate_summary["roc_auc"] if candidate_summary["roc_auc"] is not None else 0.0) - (production_summary["roc_auc"] if production_summary["roc_auc"] is not None else 0.0)),
            "ap_delta": float((candidate_summary["average_precision"] if candidate_summary["average_precision"] is not None else 0.0) - (production_summary["average_precision"] if production_summary["average_precision"] is not None else 0.0)),
        },
    }

    group_key_names = ["generator", "source", "transformation", "quality_category"]
    group_summaries: dict[str, dict[str, dict[str, Any]]] = {}
    for key_name in group_key_names:
        values = _group_rows(benchmark_rows, key_name)
        groups: dict[str, dict[str, Any]] = {}
        for group_name, rows in values.items():
            idx = [i for i, row in enumerate(benchmark_rows) if row.get(key_name) == group_name or (row.get("known_generator") == group_name if key_name == "generator" else False) or (row.get("source") == group_name if key_name == "source" else False)]
            if not idx:
                continue
            group_labels = labels[idx]
            group_scores = candidate_scores[idx]
            groups[group_name] = _metric_summary(group_labels, group_scores, threshold)
        if groups:
            group_summaries[key_name] = groups

    false_positives = _false_examples(benchmark_rows, labels, candidate_scores, threshold, "false_positive")
    false_negatives = _false_examples(benchmark_rows, labels, candidate_scores, threshold, "false_negative")

    rejection_reasons: list[str] = []
    if candidate_summary["false_positive_rate"] > criteria.max_authentic_false_positive_rate:
        rejection_reasons.append("authentic_false_positive_rate")
    if candidate_summary["accuracy"] < criteria.min_accuracy:
        rejection_reasons.append("accuracy")
    if candidate_summary["precision"] < criteria.min_precision:
        rejection_reasons.append("precision")
    if candidate_summary["recall"] < criteria.min_recall:
        rejection_reasons.append("recall")
    if candidate_summary["f1"] < criteria.min_f1:
        rejection_reasons.append("f1")
    if candidate_summary["roc_auc"] is not None and candidate_summary["roc_auc"] < criteria.min_auc:
        rejection_reasons.append("roc_auc")
    if candidate_summary["average_precision"] is not None and candidate_summary["average_precision"] < criteria.min_average_precision:
        rejection_reasons.append("average_precision")
    if candidate_summary["roc_auc"] is not None and production_summary["roc_auc"] is not None and candidate_summary["roc_auc"] < production_summary["roc_auc"] - criteria.max_auc_drop:
        rejection_reasons.append("guardrail_auc_regression")

    for generator_name, limits in criteria.generator_guardrails.items():
        group_key = generator_name
        if group_key in group_summaries.get("generator", {}):
            metric = group_summaries["generator"][group_key].get("roc_auc")
            if metric is not None and metric < limits.get("min_auc", float("-inf")):
                rejection_reasons.append(f"generator_auc_guardrail:{generator_name}")

    if criteria.require_calibration and calibration_status is None:
        rejection_reasons.append("calibration_not_configured")
    if criteria.evaluate_three_way and calibration_status is None:
        rejection_reasons.append("three_way_decision_not_configured")

    report = {
        "passed": not rejection_reasons,
        "threshold": threshold,
        "criteria": criteria.to_dict(),
        "summary": summary,
        "group_summaries": group_summaries,
        "false_positive_examples": false_positives,
        "false_negative_examples": false_negatives,
        "rejection_reasons": rejection_reasons,
        "calibration_status": calibration_status,
        "analysis": {
            "candidate_total": int(len(candidate_scores)),
            "positive_label_count": int(labels.sum()),
            "negative_label_count": int((1 - labels).sum()),
        },
    }
    return report


def load_benchmark_manifest(path: str | Path) -> list[dict[str, Any]]:
    manifest_path = Path(path)
    rows = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


__all__ = [
    "EvaluationCriteria",
    "evaluate_candidate_scores",
    "load_benchmark_manifest",
]
