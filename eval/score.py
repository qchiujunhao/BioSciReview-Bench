"""Unified scoring for BioSciReview-Bench agent predictions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
CATEGORY_TO_CODE = {
    "B1": "B1",
    "B2": "B2",
    "B3": "B3",
    "B4": "B4",
    "B5": "B5",
    "B6": "B6",
    "B1_normalization": "B1",
    "B2_statistical_test": "B2",
    "B3_multiple_testing": "B3",
    "B4_sample_label": "B4",
    "B5_data_leakage": "B5",
    "B6_filter_threshold": "B6",
    "normalization": "B1",
    "statistical_test": "B2",
    "multiple_testing": "B3",
    "sample_label": "B4",
    "data_leakage": "B5",
    "filter_threshold": "B6",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def normalize_category(category: Any) -> str | None:
    if category is None:
        return None
    if isinstance(category, str):
        return CATEGORY_TO_CODE.get(category.strip(), CATEGORY_TO_CODE.get(category.strip().lower(), category.strip()))
    return str(category)


def normalize_truth(row: dict[str, Any]) -> dict[str, Any]:
    rendered_bug_line = row.get("rendered_bug_line")
    if rendered_bug_line is None and row.get("bug_present"):
        rendered_bug_line = row.get("bug_line")
    return {
        **row,
        "sample_id": row.get("neutral_id") or row.get("id"),
        "pair_id": row.get("pair_id") or (row.get("id", "").rsplit("_", 1)[0] if row.get("id") else None),
        "bug_present": bool(row.get("bug_present")),
        "bug_category": normalize_category(row.get("bug_category")),
        "rendered_bug_line": rendered_bug_line,
    }


def normalize_prediction(row: dict[str, Any]) -> dict[str, Any]:
    bug_present = row.get("bug_present")
    if not isinstance(bug_present, bool):
        bug_present = False
    return {
        **row,
        "sample_id": row.get("neutral_id") or row.get("id") or row.get("sample_id"),
        "bug_present": bug_present,
        "bug_line": row.get("bug_line") if isinstance(row.get("bug_line"), int) else None,
        "bug_category": normalize_category(row.get("bug_category")),
    }


def load_truth(path: Path) -> list[dict[str, Any]]:
    return [normalize_truth(row) for row in read_jsonl(path)]


def load_predictions(path: Path) -> dict[str, dict[str, Any]]:
    rows = read_jsonl(path)
    predictions: dict[str, dict[str, Any]] = {}
    for row in rows:
        pred = normalize_prediction(row)
        sample_id = pred.get("sample_id")
        if not sample_id:
            raise ValueError(f"Prediction row missing neutral_id/id/sample_id: {row}")
        predictions[sample_id] = pred
    return predictions


def precision_recall_f1(tp: int, fp: int, fn: int) -> dict[str, float | int]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def detection_counts(truth_rows: list[dict[str, Any]], predictions: dict[str, dict[str, Any]]) -> dict[str, int]:
    tp = fp = tn = fn = 0
    for truth in truth_rows:
        pred = predictions.get(truth["sample_id"], {})
        truth_bug = bool(truth["bug_present"])
        pred_bug = bool(pred.get("bug_present"))
        if truth_bug and pred_bug:
            tp += 1
        elif truth_bug and not pred_bug:
            fn += 1
        elif not truth_bug and pred_bug:
            fp += 1
        else:
            tn += 1
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def joint_counts(
    truth_rows: list[dict[str, Any]],
    predictions: dict[str, dict[str, Any]],
    ok: Callable[[dict[str, Any], dict[str, Any]], bool],
) -> dict[str, int]:
    tp = fp = tn = fn = 0
    for truth in truth_rows:
        pred = predictions.get(truth["sample_id"], {})
        truth_bug = bool(truth["bug_present"])
        pred_bug = bool(pred.get("bug_present"))
        if truth_bug and pred_bug and ok(truth, pred):
            tp += 1
        elif truth_bug:
            fn += 1
            if pred_bug:
                fp += 1
        elif pred_bug:
            fp += 1
        else:
            tn += 1
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def line_exact(truth: dict[str, Any], pred: dict[str, Any]) -> bool:
    return isinstance(pred.get("bug_line"), int) and pred["bug_line"] == truth.get("rendered_bug_line")


def line_pm(truth: dict[str, Any], pred: dict[str, Any], tolerance: int = 3) -> bool:
    return (
        isinstance(pred.get("bug_line"), int)
        and isinstance(truth.get("rendered_bug_line"), int)
        and abs(pred["bug_line"] - truth["rendered_bug_line"]) <= tolerance
    )


def category_ok(truth: dict[str, Any], pred: dict[str, Any]) -> bool:
    return normalize_category(pred.get("bug_category")) == normalize_category(truth.get("bug_category"))


def metric_from_counts(prefix: str, counts: dict[str, int]) -> dict[str, float | int]:
    prf = precision_recall_f1(counts["tp"], counts["fp"], counts["fn"])
    return {
        f"{prefix}_precision": prf["precision"],
        f"{prefix}_recall": prf["recall"],
        f"{prefix}_f1": prf["f1"],
        f"{prefix}_tp": counts["tp"],
        f"{prefix}_fp": counts["fp"],
        f"{prefix}_tn": counts["tn"],
        f"{prefix}_fn": counts["fn"],
    }


def score_records(
    truth_rows: list[dict[str, Any]],
    predictions: dict[str, dict[str, Any]],
    tolerance: int = 3,
    prediction_inventory: dict[str, Any] | None = None,
) -> dict[str, Any]:
    parse_errors = sum(1 for pred in predictions.values() if pred.get("parse_error") or pred.get("schema_errors"))

    detection = detection_counts(truth_rows, predictions)
    joint_exact = joint_counts(truth_rows, predictions, line_exact)
    joint_pm = joint_counts(truth_rows, predictions, lambda truth, pred: line_pm(truth, pred, tolerance))
    full_exact = joint_counts(truth_rows, predictions, lambda truth, pred: line_exact(truth, pred) and category_ok(truth, pred))
    full_pm = joint_counts(
        truth_rows,
        predictions,
        lambda truth, pred: line_pm(truth, pred, tolerance) and category_ok(truth, pred),
    )

    loc_total = loc_ok = 0
    cat_total = cat_ok_count = 0
    for truth in truth_rows:
        if not truth["bug_present"]:
            continue
        pred = predictions.get(truth["sample_id"], {})
        if pred.get("bug_present") and isinstance(pred.get("bug_line"), int):
            loc_total += 1
            if line_pm(truth, pred, tolerance):
                loc_ok += 1
        if pred.get("bug_present") and pred.get("bug_category") is not None:
            cat_total += 1
            if category_ok(truth, pred):
                cat_ok_count += 1

    metrics: dict[str, Any] = {
        "n_truth": len(truth_rows),
        "n_predictions": len(predictions),
        "n_missing_predictions": 0,
        "n_extra_predictions": 0,
        "n_duplicate_predictions": 0,
        "parse_or_schema_errors": parse_errors,
        "tolerance": tolerance,
        "conditional_loc_pm3_accuracy": loc_ok / loc_total if loc_total else 0.0,
        "conditional_loc_pm3_n": loc_total,
        "category_accuracy": cat_ok_count / cat_total if cat_total else 0.0,
        "category_n": cat_total,
    }
    if prediction_inventory:
        for key in (
            "n_truth",
            "n_predictions",
            "n_missing_predictions",
            "n_extra_predictions",
            "n_duplicate_predictions",
        ):
            metrics[key] = prediction_inventory.get(key, metrics.get(key, 0))
    metrics.update(metric_from_counts("detection", detection))
    metrics.update(metric_from_counts("joint_loc_exact", joint_exact))
    metrics.update(metric_from_counts("joint_loc_pm3", joint_pm))
    metrics.update(metric_from_counts("full_diagnosis_exact", full_exact))
    metrics.update(metric_from_counts("full_diagnosis_pm3", full_pm))
    return metrics


def score_subset(
    truth_rows: list[dict[str, Any]],
    predictions: dict[str, dict[str, Any]],
    field: str,
    value: Any,
) -> dict[str, Any]:
    subset = [row for row in truth_rows if row.get(field) == value]
    metrics = score_records(subset, predictions)
    metrics[field] = value
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--only-predicted", action="store_true")
    args = parser.parse_args()

    truth = load_truth(args.truth)
    prediction_rows = read_jsonl(args.predictions)
    predictions = load_predictions(args.predictions)

    if args.only_predicted:
        truth = [row for row in truth if row["sample_id"] in predictions]
    from eval.completeness import prediction_inventory

    inventory = prediction_inventory(
        prediction_rows,
        {row["sample_id"] for row in truth},
        expected_count=len(truth),
    )
    metrics = score_records(truth, predictions, prediction_inventory=inventory)
    text = json.dumps(metrics, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
