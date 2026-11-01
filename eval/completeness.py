"""Full-run prediction completeness checks."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from eval.score import normalize_prediction, normalize_truth
from eval.utils import AGENT_KEYS, ground_truth_path, load_agent_matrix, read_jsonl, run_dir


EXPECTED_FULL_PREDICTIONS = 200
NEUTRAL_ID_RE = re.compile(r"^sample_\d{5}$")


class CompletenessError(ValueError):
    """Raised when full-run prediction files are missing or incomplete."""


def load_truth_rows(run_name: str | None = None) -> list[dict[str, Any]]:
    return [normalize_truth(row) for row in read_jsonl(ground_truth_path(run_name))]


def truth_ids(truth_rows: list[dict[str, Any]]) -> list[str]:
    return [row["sample_id"] for row in truth_rows]


def read_prediction_rows(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    issues: list[str] = []
    if not path.exists():
        return rows, [f"missing parsed file: {path}"]
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                issues.append(f"{path}: line {line_number}: invalid JSON: {exc}")
                continue
            if not isinstance(value, dict):
                issues.append(f"{path}: line {line_number}: top-level value is not an object")
                continue
            rows.append(value)
    return rows, issues


def prediction_inventory(
    rows: list[dict[str, Any]],
    expected_ids: set[str],
    expected_count: int = EXPECTED_FULL_PREDICTIONS,
) -> dict[str, Any]:
    ids = [row.get("neutral_id") or row.get("id") or row.get("sample_id") for row in rows]
    malformed_ids = [value for value in ids if not isinstance(value, str) or not NEUTRAL_ID_RE.fullmatch(value)]
    valid_ids = [value for value in ids if isinstance(value, str)]
    counts = Counter(valid_ids)
    duplicate_ids = sorted(sample_id for sample_id, count in counts.items() if count > 1)
    id_set = set(valid_ids)
    missing_ids = sorted(expected_ids - id_set)
    extra_ids = sorted(id_set - expected_ids)
    parse_or_schema_error_ids = sorted(
        str(row.get("neutral_id") or row.get("id") or row.get("sample_id") or f"row_{index}")
        for index, row in enumerate(rows, start=1)
        if row.get("parse_status") == "failed" or row.get("parse_error") or row.get("schema_errors")
    )
    return {
        "n_truth": expected_count,
        "n_predictions": len(rows),
        "n_missing_predictions": len(missing_ids),
        "n_extra_predictions": len(extra_ids),
        "n_duplicate_predictions": sum(count - 1 for count in counts.values() if count > 1),
        "n_malformed_neutral_ids": len(malformed_ids),
        "n_parse_or_schema_errors": len(parse_or_schema_error_ids),
        "missing_ids": missing_ids,
        "extra_ids": extra_ids,
        "duplicate_ids": duplicate_ids,
        "malformed_neutral_ids": malformed_ids,
        "parse_or_schema_error_ids": parse_or_schema_error_ids,
    }


def validate_prediction_rows(
    agent_key: str,
    rows: list[dict[str, Any]],
    expected_ids: set[str],
    expected_count: int = EXPECTED_FULL_PREDICTIONS,
    strict: bool = True,
    initial_issues: list[str] | None = None,
) -> dict[str, Any]:
    inventory = prediction_inventory(rows, expected_ids, expected_count=expected_count)
    issues = list(initial_issues or [])
    if len(rows) != expected_count:
        issues.append(f"{agent_key}: parsed file has {len(rows)} rows, expected {expected_count}")
    if inventory["n_malformed_neutral_ids"]:
        issues.append(f"{agent_key}: malformed neutral_id values: {inventory['malformed_neutral_ids'][:20]}")
    if inventory["n_duplicate_predictions"]:
        issues.append(f"{agent_key}: duplicate neutral_id values: {inventory['duplicate_ids'][:20]}")
    if inventory["n_missing_predictions"]:
        issues.append(f"{agent_key}: missing prediction ids: {inventory['missing_ids'][:20]}")
    if inventory["n_extra_predictions"]:
        issues.append(f"{agent_key}: extra prediction ids: {inventory['extra_ids'][:20]}")
    if inventory["n_parse_or_schema_errors"]:
        issues.append(f"{agent_key}: parse/schema failures: {inventory['parse_or_schema_error_ids'][:20]}")

    predictions: dict[str, dict[str, Any]] = {}
    for row in rows:
        pred = normalize_prediction(row)
        sample_id = pred.get("sample_id")
        if isinstance(sample_id, str):
            predictions[sample_id] = pred

    if strict and issues:
        raise CompletenessError("\n".join(issues))

    return {
        "agent_key": agent_key,
        "complete": not issues,
        "issues": issues,
        "inventory": inventory,
        "predictions": predictions,
        "rows": rows,
    }


def validate_prediction_file(
    agent_key: str,
    path: Path,
    expected_ids: set[str],
    expected_count: int = EXPECTED_FULL_PREDICTIONS,
    strict: bool = True,
) -> dict[str, Any]:
    rows, read_issues = read_prediction_rows(path)
    return validate_prediction_rows(
        agent_key,
        rows,
        expected_ids,
        expected_count=expected_count,
        strict=strict,
        initial_issues=read_issues,
    )


def validate_all_prediction_files(
    run_name: str,
    matrix_path: Path,
    allow_partial: bool = False,
) -> dict[str, Any]:
    matrix = load_agent_matrix(matrix_path)
    truth = load_truth_rows(run_name)
    expected_ids = set(truth_ids(truth))
    issues: list[str] = []
    if len(truth) != EXPECTED_FULL_PREDICTIONS:
        issues.append(f"rendered ground truth has {len(truth)} rows, expected {EXPECTED_FULL_PREDICTIONS}")
    if len(expected_ids) != len(truth):
        issues.append("rendered ground truth has duplicate neutral_ids")
    if set(matrix) != set(AGENT_KEYS):
        issues.append(f"agent matrix keys are {sorted(matrix)}, expected {list(AGENT_KEYS)}")

    agents: dict[str, Any] = {}
    for agent_key in AGENT_KEYS:
        path = run_dir(run_name) / "parsed" / f"{agent_key}.jsonl"
        result = validate_prediction_file(
            agent_key,
            path,
            expected_ids,
            expected_count=len(truth),
            strict=False,
        )
        agents[agent_key] = result
        issues.extend(result["issues"])

    if issues and not allow_partial:
        raise CompletenessError("\n".join(issues))

    return {
        "run": run_name,
        "truth": truth,
        "expected_ids": expected_ids,
        "agents": agents,
        "issues": issues,
        "complete": not issues,
    }
