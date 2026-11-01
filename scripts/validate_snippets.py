"""Validate BioSciReview-Bench snippet pairs.

Checks:
1. Every annotated snippet runs without error.
2. Every buggy/correct pair produces different stdout.
3. Runtime stderr is reported for manual inspection.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ANNOTATIONS = ROOT / "data" / "annotations.jsonl"
PYTHON = os.environ.get("BIOSCI_PYTHON", sys.executable)
RSCRIPT = os.environ.get("RSCRIPT", "Rscript")


def load_annotations() -> dict[str, dict]:
    annotations: dict[str, dict] = {}
    with ANNOTATIONS.open() as handle:
        for line_num, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if record["id"] in annotations:
                raise ValueError(f"Duplicate annotation ID on line {line_num}: {record['id']}")
            annotations[record["id"]] = record
    return annotations


def pair_members(annotations: dict[str, dict]) -> dict[str, dict[str, str]]:
    pairs: dict[str, dict[str, str]] = {}
    for snippet_id, record in annotations.items():
        pair_id = record.get("pair_id")
        if not pair_id:
            continue
        bucket = pairs.setdefault(pair_id, {})
        key = "buggy" if record.get("bug_present") else "correct"
        if key in bucket:
            raise ValueError(f"Pair {pair_id} has more than one {key} snippet")
        bucket[key] = snippet_id
    return pairs


def command_for(path: Path) -> list[str]:
    if path.suffix == ".py":
        return [PYTHON, str(path)]
    if path.suffix.lower() == ".r":
        return [RSCRIPT, str(path)]
    raise ValueError(f"Unsupported snippet type: {path}")


def run_snippet(path: Path, timeout: int = 30) -> dict:
    try:
        result = subprocess.run(
            command_for(path),
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {"success": False, "stdout": "", "stderr": "TIMEOUT", "returncode": None}
    except FileNotFoundError as exc:
        return {"success": False, "stdout": "", "stderr": str(exc), "returncode": None}

    return {
        "success": result.returncode == 0,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "returncode": result.returncode,
    }


def validate_pairs(annotations: dict[str, dict]) -> list[str]:
    issues: list[str] = []
    results: dict[str, dict] = {}

    for snippet_id, record in annotations.items():
        path = ROOT / record["path"]
        if not path.exists():
            issues.append(f"{snippet_id}: missing file {path}")
            continue
        results[snippet_id] = run_snippet(path)
        if not results[snippet_id]["success"]:
            stderr = results[snippet_id]["stderr"].strip()[:500]
            issues.append(f"{snippet_id}: failed to run: {stderr}")
        elif results[snippet_id]["stderr"].strip():
            stderr = results[snippet_id]["stderr"].strip()[:500]
            issues.append(f"{snippet_id}: wrote to stderr: {stderr}")

    for pair_id, members in pair_members(annotations).items():
        correct_id = members.get("correct")
        buggy_id = members.get("buggy")
        if not correct_id or not buggy_id:
            issues.append(f"{pair_id}: expected one correct and one buggy snippet")
            continue
        if buggy_id not in results or correct_id not in results:
            continue
        buggy = results[buggy_id]
        correct = results[correct_id]
        if buggy["success"] and correct["success"] and buggy["stdout"] == correct["stdout"]:
            issues.append(f"{buggy_id}: stdout matches paired correct snippet {correct_id}")

    return issues


def main() -> int:
    annotations = load_annotations()
    issues = validate_pairs(annotations)

    if issues:
        print("Validation failed:")
        for issue in issues:
            print(f"- {issue}")
        return 1

    print(f"Validation passed for {len(annotations)} snippets.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
