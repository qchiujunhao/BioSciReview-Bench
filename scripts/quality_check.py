"""Quality checks for BioSciReview-Bench annotations and snippets."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ANNOTATIONS = ROOT / "data" / "annotations.jsonl"
DOMAINS = {
    "D1_bulk_rnaseq",
    "D2_single_cell",
    "D3_variant_genomics",
    "D4_ml_clinical",
    "D5_statistical_genomics",
}
CATEGORIES = {
    "B1",
    "B2",
    "B3",
    "B4",
    "B5",
    "B6",
}
SEVERITIES = {"minor", "moderate", "conclusion_changing"}
REQUIRED = {
    "id",
    "pair_id",
    "language",
    "domain",
    "bug_present",
    "bug_category",
    "bug_line",
    "bug_description",
    "severity",
    "expected_fix",
    "expert_rationale",
    "task_description",
    "libraries",
    "lines_of_code",
    "source_inspiration",
    "path",
}


def load_annotations() -> list[dict]:
    records = []
    with ANNOTATIONS.open() as handle:
        for line_num, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            record["_line_num"] = line_num
            records.append(record)
    return records


def count_lines(path: Path) -> int:
    with path.open() as handle:
        return sum(1 for _ in handle)


def check_record(record: dict, pair_ids: set[str]) -> list[str]:
    issues: list[str] = []
    prefix = f"{record.get('id', '<missing id>')}:"
    missing = REQUIRED - set(record)
    if missing:
        issues.append(f"{prefix} missing fields: {sorted(missing)}")

    if record.get("language") not in {"python", "r"}:
        issues.append(f"{prefix} invalid language {record.get('language')}")
    if record.get("domain") not in DOMAINS:
        issues.append(f"{prefix} invalid domain {record.get('domain')}")

    path = ROOT / str(record.get("path", ""))
    if not path.exists():
        issues.append(f"{prefix} file does not exist: {path}")
        return issues

    actual_lines = count_lines(path)
    if record.get("lines_of_code") != actual_lines:
        issues.append(
            f"{prefix} annotated lines_of_code={record.get('lines_of_code')} but file has {actual_lines}"
        )
    if record.get("split") == "long_context":
        if not 100 <= actual_lines <= 220:
            issues.append(f"{prefix} long-context line count outside 100-220: {actual_lines}")
    elif record.get("split") == "semantic_stress":
        if not 45 <= actual_lines <= 120:
            issues.append(f"{prefix} semantic-stress line count outside 45-120: {actual_lines}")
    elif not 15 <= actual_lines <= 50:
        issues.append(f"{prefix} line count outside 15-50: {actual_lines}")

    bug_present = record.get("bug_present")
    if bug_present:
        if record.get("pair_id") not in pair_ids:
            issues.append(f"{prefix} pair_id missing or unknown: {record.get('pair_id')}")
        if record.get("bug_category") not in CATEGORIES:
            issues.append(f"{prefix} invalid bug_category {record.get('bug_category')}")
        if record.get("severity") not in SEVERITIES:
            issues.append(f"{prefix} invalid severity {record.get('severity')}")
        bug_line = record.get("bug_line")
        if not isinstance(bug_line, int) or not 1 <= bug_line <= actual_lines:
            issues.append(f"{prefix} invalid bug_line {bug_line}")
    else:
        for field in ("bug_category", "bug_line", "bug_description", "severity", "expected_fix"):
            if record.get(field) is not None:
                issues.append(f"{prefix} correct snippet has non-null {field}")

    return issues


def main() -> int:
    records = load_annotations()
    ids = [record["id"] for record in records]
    issues: list[str] = []

    duplicate_ids = [item for item, count in Counter(ids).items() if count > 1]
    if duplicate_ids:
        issues.append(f"Duplicate IDs: {duplicate_ids}")

    pair_ids = {record["pair_id"] for record in records if record.get("pair_id")}
    for record in records:
        issues.extend(check_record(record, pair_ids))

    annotated_paths = {Path(record["path"]) for record in records}
    snippet_paths = {
        path.relative_to(ROOT)
        for path in (ROOT / "data" / "snippets").rglob("*")
        if path.is_file() and path.suffix in {".py", ".R", ".r"}
    }
    unannotated = sorted(snippet_paths - annotated_paths)
    missing_files = sorted(annotated_paths - snippet_paths)
    if unannotated:
        issues.append(f"Unannotated snippet files present: {[str(path) for path in unannotated[:10]]}")
    if missing_files:
        issues.append(f"Annotated snippet files missing: {[str(path) for path in missing_files[:10]]}")

    buggy = [record for record in records if record.get("bug_present")]
    category_counts = Counter(record["bug_category"] for record in buggy)
    domain_counts = Counter(record["domain"] for record in records)

    if issues:
        print("Quality check failed:")
        for issue in issues:
            print(f"- {issue}")
        return 1

    print(f"Quality check passed for {len(records)} snippets.")
    print("Bug categories:", dict(sorted(category_counts.items())))
    print("Domains:", dict(sorted(domain_counts.items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
