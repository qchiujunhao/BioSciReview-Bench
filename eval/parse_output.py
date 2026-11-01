"""Parse agent output files into normalized JSON predictions."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


JSON_RE = re.compile(r"\{.*\}", re.DOTALL)
VALID_CATEGORIES = {"B1", "B2", "B3", "B4", "B5", "B6", None}
CATEGORY_TO_CODE = {
    "b1": "B1",
    "b2": "B2",
    "b3": "B3",
    "b4": "B4",
    "b5": "B5",
    "b6": "B6",
    "normalization": "B1",
    "b1_normalization": "B1",
    "statistical_test": "B2",
    "b2_statistical_test": "B2",
    "multiple_testing": "B3",
    "b3_multiple_testing": "B3",
    "sample_label": "B4",
    "b4_sample_label": "B4",
    "data_leakage": "B5",
    "b5_data_leakage": "B5",
    "filter_threshold": "B6",
    "b6_filter_threshold": "B6",
}
PREDICTION_KEYS = {"bug_present", "bug_line", "bug_category", "confidence", "rationale"}
EXPLANATION_KEYS = ("rationale", "explanation", "reasoning")


def extract_json(text: str) -> dict[str, Any]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        match = JSON_RE.search(text)
        if not match:
            raise
        value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise ValueError("Top-level JSON value is not an object")
    return value


def unwrap_prediction(record: dict[str, Any]) -> dict[str, Any]:
    if PREDICTION_KEYS.intersection(record):
        return record

    for key in ("result", "response", "content", "text", "message"):
        value = record.get(key)
        if isinstance(value, str):
            try:
                nested = extract_json(value)
            except Exception:
                continue
            if PREDICTION_KEYS.intersection(nested):
                return nested
        if isinstance(value, dict) and PREDICTION_KEYS.intersection(value):
            return value

    structured = record.get("structured_output")
    if isinstance(structured, dict) and PREDICTION_KEYS.intersection(structured):
        return structured

    if isinstance(record.get("messages"), list):
        for message in reversed(record["messages"]):
            if isinstance(message, dict):
                try:
                    return unwrap_prediction(message)
                except ValueError:
                    pass

    raise ValueError("No prediction JSON object found")


def normalize_category(category: Any) -> str | None:
    if category is None:
        return None
    if isinstance(category, str):
        key = category.strip().lower()
        if key in {"", "none", "null", "no_bug", "clean"}:
            return None
        return CATEGORY_TO_CODE.get(key, category.strip())
    return str(category)


def normalize_bug_line(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.lower() in {"", "none", "null", "n/a", "na"}:
            return None
        if re.fullmatch(r"\d+", stripped):
            return int(stripped)
    return None


def schema_errors(prediction: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if not isinstance(prediction.get("bug_present"), bool):
        errors.append("bug_present must be boolean")
    if prediction.get("bug_line") is not None and not isinstance(prediction.get("bug_line"), int):
        errors.append("bug_line must be integer or null")
    if prediction.get("bug_category") not in VALID_CATEGORIES:
        errors.append("bug_category must be B1-B6 or null")
    confidence = prediction.get("confidence")
    if confidence is not None and (
        not isinstance(confidence, (int, float)) or isinstance(confidence, bool) or confidence < 0 or confidence > 1
    ):
        errors.append("confidence must be a number from 0 to 1 or null")
    if not isinstance(prediction.get("rationale"), str):
        errors.append("rationale must be string")
    if prediction.get("bug_present") is False:
        if prediction.get("bug_line") is not None:
            errors.append("bug_line must be null when bug_present is false")
        if prediction.get("bug_category") is not None:
            errors.append("bug_category must be null when bug_present is false")
    if prediction.get("bug_present") is True:
        if prediction.get("bug_line") is None:
            errors.append("bug_line is required when bug_present is true")
        if prediction.get("bug_category") is None:
            errors.append("bug_category is required when bug_present is true")
    return errors


def normalize_prediction(record: dict[str, Any]) -> dict[str, Any]:
    bug_present = record.get("bug_present")
    if isinstance(bug_present, str):
        bug_present = bug_present.strip().lower() in {"true", "yes", "1", "bug", "buggy"}
    elif not isinstance(bug_present, bool):
        bug_present = None

    category = normalize_category(record.get("bug_category"))
    line = normalize_bug_line(record.get("bug_line"))
    if bug_present is False:
        category = None
        line = None

    rationale = ""
    for key in EXPLANATION_KEYS:
        if isinstance(record.get(key), str):
            rationale = record[key]
            break

    confidence = record.get("confidence")
    if isinstance(confidence, str):
        try:
            confidence = float(confidence)
        except ValueError:
            confidence = None

    prediction = {
        "bug_present": bug_present,
        "bug_line": line,
        "bug_category": category,
        "confidence": confidence if isinstance(confidence, (int, float)) and not isinstance(confidence, bool) else None,
        "rationale": rationale,
    }
    errors = schema_errors(prediction)
    if errors:
        prediction["schema_errors"] = errors
    return prediction


def parse_text(text: str) -> dict[str, Any]:
    try:
        prediction = normalize_prediction(unwrap_prediction(extract_json(text)))
    except Exception as exc:
        return {
            "bug_present": None,
            "bug_line": None,
            "bug_category": None,
            "confidence": None,
            "rationale": "",
            "parse_error": str(exc),
            "raw_text": text,
        }
    return prediction


def parse_file(path: Path) -> dict[str, Any]:
    return parse_text(path.read_text(errors="replace"))


def load_neutral_mapping(manifest_path: Path | None) -> dict[str, str]:
    if manifest_path is None:
        return {}
    mapping: dict[str, str] = {}
    with manifest_path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            mapping[row["neutral_id"]] = row.get("original_id", row["neutral_id"])
    return mapping


def output_name_for(path: Path, neutral_mapping: dict[str, str]) -> str:
    if not neutral_mapping:
        return path.name
    for neutral_id, original_id in sorted(neutral_mapping.items(), key=lambda item: len(item[0]), reverse=True):
        suffix = f"_{neutral_id}"
        if path.stem.endswith(suffix):
            model_prefix = path.stem[: -len(suffix)]
            return f"{model_prefix}_{original_id}.json"
        if path.stem == neutral_id:
            return f"{original_id}.json"
    return path.name


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--manifest", type=Path, help="Rendered manifest for neutral-id to original-id mapping.")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    neutral_mapping = load_neutral_mapping(args.manifest)
    for path in sorted(args.input_dir.glob("*.json")):
        if path.name.endswith(".run_meta.json") or path.name.endswith(".raw.json"):
            continue
        parsed = parse_file(path)
        (args.output_dir / output_name_for(path, neutral_mapping)).write_text(json.dumps(parsed, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
