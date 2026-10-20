from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from sqlalchemy import select
from sqlalchemy.orm import Session

from .storage.database import get_engine
from .storage.models import HumanReview, ImageRecord, Prediction, ReviewStatus

VALID_LABELS = {"real", "ai"}
EXCLUDED_LABELS = {"uncertain"}
DEFAULT_SPLITS = {"train": 0.7, "val": 0.15, "calibration": 0.1, "test": 0.05}


def _maybe_label(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if normalized in VALID_LABELS or normalized in EXCLUDED_LABELS:
        return normalized
    return None


def _most_common_label(reviews: Iterable[HumanReview]) -> str | None:
    labels = [_maybe_label(review.verified_class) for review in reviews if review.review_status == ReviewStatus.APPROVED]
    labels = [label for label in labels if label is not None]
    if not labels:
        return None
    counts = Counter(labels)
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


def _review_record_summary(image: ImageRecord) -> dict[str, Any]:
    reviews = list(image.reviews or [])
    approved = [review for review in reviews if review.review_status == ReviewStatus.APPROVED]
    return {
        "review_count": len(reviews),
        "approved_review_count": len(approved),
        "reviewer_ids": sorted({review.reviewer_id for review in reviews}),
        "approved_reviewers": sorted({review.reviewer_id for review in approved}),
        "newest_review": max((review.reviewed_at for review in reviews), default=None),
        "latest_evidence": approved[-1].evidence_source if approved else None,
        "latest_notes": approved[-1].source_notes if approved else None,
        "latest_confidence": approved[-1].confidence if approved else 0.0,
        "latest_known_generator": approved[-1].known_generator if approved else None,
    }


def _latest_prediction_by_model(image: ImageRecord) -> dict[str, Prediction]:
    latest: dict[str, Prediction] = {}
    for prediction in image.predictions or []:
        model_name = prediction.model_name.lower()
        current = latest.get(model_name)
        if current is None or prediction.inference_timestamp >= current.inference_timestamp:
            latest[model_name] = prediction
    return latest


def _case_type(human_label: str, resnet_pred: str | None, clip_pred: str | None) -> str:
    predictions = [resnet_pred, clip_pred]
    if human_label is None:
        return "uncertainty_case"
    if not resnet_pred and not clip_pred:
        return "uncertainty_case"
    if resnet_pred and clip_pred and resnet_pred != clip_pred:
        return "model_disagreement"
    if resnet_pred == human_label and (not clip_pred or clip_pred == human_label):
        return "correct_prediction"
    if clip_pred == human_label and (not resnet_pred or resnet_pred == human_label):
        return "correct_prediction"
    if resnet_pred == "ai" and human_label == "real":
        return "false_positive"
    if clip_pred == "ai" and human_label == "real":
        return "false_positive"
    if resnet_pred == "real" and human_label == "ai":
        return "false_negative"
    if clip_pred == "real" and human_label == "ai":
        return "false_negative"
    return "uncertainty_case"


def _build_group_key(image: ImageRecord) -> str:
    if image.source:
        return f"source:{image.source.strip()}"
    if image.perceptual_hash:
        return f"phash:{image.perceptual_hash[:12]}"
    return f"sha:{image.sha256[:12]}"


def _assign_group_splits(groups: dict[str, list[dict[str, Any]]], benchmark_ids: set[str] | None = None) -> dict[str, str]:
    ordered_names = sorted(groups.keys())
    split_order = ["train", "val", "calibration", "test"]
    split_map: dict[str, str] = {}

    benchmark_ids = benchmark_ids or set()
    if benchmark_ids:
        for group_name, items in groups.items():
            if any(item["image_id"] in benchmark_ids for item in items):
                split_map[group_name] = "benchmark_test"

    remaining = [group_name for group_name in ordered_names if group_name not in split_map]
    if not remaining:
        return split_map

    total_groups = len(remaining)
    ratios = DEFAULT_SPLITS
    target_counts = {
        split_name: max(1, round(total_groups * ratios[split_name])) if split_name != "test" else max(1, round(total_groups * ratios[split_name]))
        for split_name in split_order
    }
    current = {split_name: 0 for split_name in split_order}
    for group_name in remaining:
        group_index = ordered_names.index(group_name)
        base = (group_index % len(split_order))
        pick = split_order[base]
        while current[pick] >= target_counts[pick] and pick != "test":
            pick = split_order[(split_order.index(pick) + 1) % len(split_order)]
        split_map[group_name] = pick
        current[pick] += 1

    # fill any remaining empty splits by deterministic assignment
    for split_name in split_order:
        if current[split_name] == 0 and remaining:
            first_group = remaining[0]
            split_map[first_group] = split_name
            remaining.pop(0)

    return split_map


def _select_eligible_records(session: Session, require_second_review: bool = False) -> tuple[list[dict[str, Any]], dict[str, int]]:
    excluded_counts: dict[str, int] = defaultdict(int)
    records: list[dict[str, Any]] = []

    images = session.execute(select(ImageRecord)).scalars().all()
    for image in images:
        if image.review_status not in {ReviewStatus.APPROVED, ReviewStatus.REJECTED, ReviewStatus.DUPLICATE, ReviewStatus.NEEDS_REVIEW, ReviewStatus.PENDING}:
            excluded_counts["rejected_or_ineligible"] += 1
            continue

        reviews = list(image.reviews or [])
        if not reviews:
            excluded_counts["unreviewed"] += 1
            continue

        if any(_maybe_label(review.verified_class) == "uncertain" for review in reviews):
            excluded_counts["uncertain_label"] += 1
            continue

        if any(_maybe_label(review.verified_class) is None for review in reviews):
            excluded_counts["disputed_label"] += 1
            continue

        if any(review.review_status in {ReviewStatus.REJECTED, ReviewStatus.DUPLICATE} for review in reviews):
            excluded_counts["rejected_or_ineligible"] += 1
            continue

        if any(review.review_status == ReviewStatus.NEEDS_REVIEW for review in reviews):
            excluded_counts["needs_review"] += 1
            continue

        storage_path = Path(image.storage_path)
        if not storage_path.exists():
            excluded_counts["missing_file"] += 1
            continue

        approved_reviews = [review for review in reviews if review.review_status == ReviewStatus.APPROVED]
        if not approved_reviews:
            excluded_counts["unreviewed"] += 1
            continue

        normalized_labels = [_maybe_label(review.verified_class) for review in approved_reviews]
        normalized_labels = [label for label in normalized_labels if label is not None]
        if not normalized_labels:
            excluded_counts["uncertain_label"] += 1
            continue

        if any(_maybe_label(review.verified_class) == "uncertain" for review in approved_reviews):
            excluded_counts["uncertain_label"] += 1
            continue

        label_counts = Counter(normalized_labels)
        final_label = sorted(label_counts.items(), key=lambda item: (-item[1], item[0]))[0][0]
        if len({review.reviewer_id for review in approved_reviews}) < 2 and require_second_review:
            excluded_counts["needs_second_review"] += 1
            continue

        if len(label_counts) > 1:
            excluded_counts["disputed_label"] += 1
            continue

        latest_predictions = _latest_prediction_by_model(image)
        resnet_pred = _maybe_label(latest_predictions.get("resnet", Prediction(model_name="resnet", model_version="", predicted_class="", raw_score=0.0, threshold=0.5, preprocessing_version="unknown")).predicted_class)
        clip_pred = _maybe_label(latest_predictions.get("clip", Prediction(model_name="clip", model_version="", predicted_class="", raw_score=0.0, threshold=0.5, preprocessing_version="unknown")).predicted_class)

        if not resnet_pred and not clip_pred:
            excluded_counts["missing_predictions"] += 1
            continue

        case_type = _case_type(final_label, resnet_pred, clip_pred)

        approved_review = approved_reviews[-1]
        records.append(
            {
                "image_id": image.id,
                "storage_path": str(image.storage_path),
                "sha256": image.sha256,
                "perceptual_hash": image.perceptual_hash,
                "source": image.source,
                "known_generator": approved_review.known_generator,
                "group_key": _build_group_key(image),
                "human_label": final_label,
                "review_status": image.review_status.value,
                "review_evidence_source": approved_review.evidence_source,
                "review_confidence": approved_review.confidence,
                "review_notes": approved_review.source_notes,
                "case_type": case_type,
                "resnet_prediction": resnet_pred,
                "clip_prediction": clip_pred,
                "resnet_score": latest_predictions.get("resnet").raw_score if "resnet" in latest_predictions else None,
                "clip_score": latest_predictions.get("clip").raw_score if "clip" in latest_predictions else None,
                "duplicate_reason": None,
                "reason": f"approved_review:{approved_review.review_status.value};case:{case_type}",
                "review_summary": _review_record_summary(image),
            }
        )

    return records, dict(excluded_counts)


def _deduplicate_records(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    deduped: list[dict[str, Any]] = []
    counts: dict[str, int] = defaultdict(int)
    seen_sha: set[str] = set()
    seen_phash: set[str] = set()

    for record in records:
        sha = record["sha256"]
        phash = record["perceptual_hash"]

        if sha in seen_sha:
            counts["duplicate_sha256"] += 1
            record["duplicate_reason"] = "sha256"
            continue
        if phash in seen_phash:
            counts["duplicate_perceptual"] += 1
            record["duplicate_reason"] = "perceptual_hash"
            continue

        seen_sha.add(sha)
        seen_phash.add(phash)
        deduped.append(record)

    return deduped, dict(counts)


def _report_balances(rows: list[dict[str, Any]], excluded_counts: dict[str, int], duplicate_counts: dict[str, int]) -> dict[str, Any]:
    class_balance = dict(Counter(row["human_label"] for row in rows))
    source_balance = dict(Counter(row["source"] or "unknown" for row in rows))
    generator_balance = dict(Counter(row["known_generator"] or "unknown" for row in rows))
    case_balance = dict(Counter(row["case_type"] for row in rows))

    exclusion_keys = [
        "uncertain_label",
        "rejected_or_ineligible",
        "needs_review",
        "unreviewed",
        "missing_file",
        "missing_predictions",
        "needs_second_review",
        "disputed_label",
    ]
    normalized_excluded = {key: int(excluded_counts.get(key, 0)) for key in exclusion_keys}

    return {
        "dataset_version": None,
        "eligible_total": len(rows),
        "class_balance": class_balance,
        "source_balance": source_balance,
        "generator_balance": generator_balance,
        "case_balance": case_balance,
        "duplicate_total": sum(duplicate_counts.values()),
        "duplicate_counts": duplicate_counts,
        "excluded_counts": normalized_excluded,
        "uncertain_labels": normalized_excluded.get("uncertain_label", 0),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def build_dataset_version(
    session: Session,
    output_dir: str | Path,
    version: str,
    *,
    dry_run: bool = False,
    allow_overwrite: bool = False,
    require_second_review: bool = False,
    benchmark_test_set: Iterable[str] | None = None,
    split_ratios: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Build a versioned dataset manifest from reviewed images.

    The function selects approved, binary-labeled images, deduplicates them by SHA-256 and
    perceptual hash, and assigns groups to a deterministic split without leaking across
    related images or families.
    """
    output_dir = Path(output_dir)
    version_dir = output_dir / str(version)
    if version_dir.exists() and not allow_overwrite and not dry_run:
        raise FileExistsError(f"Dataset version '{version}' already exists under {output_dir}. Use a new version or allow_overwrite=True.")

    if split_ratios is None:
        split_ratios = DEFAULT_SPLITS.copy()

    if benchmark_test_set is None:
        benchmark_test_set = set()
    else:
        benchmark_test_set = set(str(item) for item in benchmark_test_set)

    records, excluded_counts = _select_eligible_records(session, require_second_review=require_second_review)
    deduped, duplicate_counts = _deduplicate_records(records)

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in deduped:
        groups[item["group_key"]].append(item)

    split_names = ["train", "val", "calibration", "test"]
    split_map = _assign_group_splits(groups, benchmark_ids=benchmark_test_set)

    manifest: list[dict[str, Any]] = []
    split_counts: dict[str, int] = {split_name: 0 for split_name in split_names + ["benchmark_test"]}

    for item in deduped:
        item_group = item["group_key"]
        split_name = split_map.get(item_group, "train")
        if item["image_id"] in benchmark_test_set:
            split_name = "benchmark_test"
        item["split"] = split_name
        split_counts[split_name] = split_counts.get(split_name, 0) + 1

        manifest.append(
            {
                "image_id": item["image_id"],
                "storage_path": item["storage_path"],
                "sha256": item["sha256"],
                "perceptual_hash": item["perceptual_hash"],
                "source": item["source"],
                "known_generator": item["known_generator"],
                "human_label": item["human_label"],
                "case_type": item["case_type"],
                "reason": item["reason"],
                "group_key": item_group,
                "split": split_name,
                "review_confidence": item["review_confidence"],
                "review_evidence_source": item["review_evidence_source"],
                "review_notes": item["review_notes"],
                "resnet_prediction": item["resnet_prediction"],
                "clip_prediction": item["clip_prediction"],
                "resnet_score": item["resnet_score"],
                "clip_score": item["clip_score"],
            }
        )

    report = _report_balances(manifest, excluded_counts, duplicate_counts)
    report["dataset_version"] = version
    report["split_counts"] = split_counts
    report["split_ratios"] = split_ratios
    report["benchmark_test_ids"] = sorted(benchmark_test_set)

    if dry_run:
        return {"version": version, "version_dir": str(version_dir), "manifest": manifest, "split_definitions": split_counts, "report": report}

    version_dir.mkdir(parents=True, exist_ok=False)

    manifest_path = version_dir / "manifest.jsonl"
    split_path = version_dir / "split_definitions.json"
    report_path = version_dir / "dataset_report.json"
    versions_index = output_dir / "versions.json"

    with manifest_path.open("w", encoding="utf-8") as handle:
        for row in manifest:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    split_payload = {
        "version": version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "splits": split_counts,
        "benchmark_test_ids": sorted(benchmark_test_set),
    }
    split_path.write_text(json.dumps(split_payload, indent=2, sort_keys=True), encoding="utf-8")

    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    versions_index_payload: dict[str, Any] = {"versions": []}
    if versions_index.exists():
        versions_index_payload = json.loads(versions_index.read_text(encoding="utf-8"))
    versions_list = versions_index_payload.setdefault("versions", [])
    if not any(item.get("version") == version for item in versions_list):
        versions_list.append({"version": version, "path": str(version_dir), "created_at": datetime.now(timezone.utc).isoformat()})
    versions_index.write_text(json.dumps(versions_index_payload, indent=2, sort_keys=True), encoding="utf-8")

    return {"version": version, "version_dir": str(version_dir), "manifest": manifest, "split_definitions": split_payload, "report": report}


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a versioned dataset from reviewed AI image records.")
    parser.add_argument("--version", required=True, help="Version identifier, e.g. v0001")
    parser.add_argument("--output-dir", default="datasets", help="Directory that stores dataset versions")
    parser.add_argument("--db-url", default=None, help="Optional database URL override")
    parser.add_argument("--dry-run", action="store_true", help="Print the dataset plan without writing files")
    parser.add_argument("--require-second-review", action="store_true", help="Require a second approved review before an image is eligible")
    parser.add_argument("--benchmark-test-set", default=None, help="Optional CSV/JSON list of image IDs to hold out as fixed benchmark tests")
    parser.add_argument("--allow-overwrite", action="store_true", help="Allow rewriting a dataset version dir")
    args = parser.parse_args()

    engine = get_engine(args.db_url)
    with Session(engine) as session:
        benchmark_ids: set[str] = set()
        if args.benchmark_test_set:
            path = Path(args.benchmark_test_set)
            if path.suffix.lower() == ".json":
                benchmark_ids = set(json.loads(path.read_text(encoding="utf-8")))
            else:
                benchmark_ids = {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}

        result = build_dataset_version(
            session=session,
            output_dir=args.output_dir,
            version=args.version,
            dry_run=args.dry_run,
            allow_overwrite=args.allow_overwrite,
            require_second_review=args.require_second_review,
            benchmark_test_set=benchmark_ids,
        )

    print(json.dumps({
        "version": result["version"],
        "version_dir": result["version_dir"],
        "manifest_rows": len(result["manifest"]),
        "split_definitions": result["split_definitions"],
        "report": result["report"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
