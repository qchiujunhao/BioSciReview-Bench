"""Validate BioSciReview-Bench dataset balance and pairing."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ANNOTATIONS = ROOT / "data" / "annotations.jsonl"

DOMAIN_TO_SHORT = {
    "D1_bulk_rnaseq": "D1",
    "D2_single_cell": "D2",
    "D3_variant_genomics": "D3",
    "D4_ml_clinical": "D4",
    "D5_statistical_genomics": "D5",
}
EXPECTED_DOMAINS = {"D1": 20, "D2": 20, "D3": 20, "D4": 20, "D5": 20}
EXPECTED_LANGUAGES = {"python": 50, "r": 50}
EXPECTED_CATEGORIES = {"B1": 16, "B2": 15, "B3": 17, "B4": 18, "B5": 17, "B6": 17}


def load_records(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def short_domain(domain: str) -> str:
    return DOMAIN_TO_SHORT.get(domain, domain)


def validate_counts(records: list[dict]) -> list[str]:
    issues: list[str] = []
    pairs: dict[str, list[dict]] = defaultdict(list)
    ids: set[str] = set()

    for index, record in enumerate(records, start=1):
        snippet_id = record.get("id")
        pair_id = record.get("pair_id")
        if not snippet_id:
            issues.append(f"line {index}: missing id")
        elif snippet_id in ids:
            issues.append(f"{snippet_id}: duplicate id")
        ids.add(snippet_id)

        if not pair_id:
            issues.append(f"{snippet_id or f'line {index}'}: missing pair_id")
        else:
            pairs[pair_id].append(record)

        snippet_path = ROOT / record.get("path", "")
        if not snippet_path.exists():
            issues.append(f"{snippet_id}: missing snippet file {record.get('path')}")

    if len(pairs) != 100:
        issues.append(f"n_pairs={len(pairs)}, expected 100")
    if len(records) != 200:
        issues.append(f"n_snippets={len(records)}, expected 200")

    correct_count = sum(1 for record in records if not record.get("bug_present"))
    buggy_count = sum(1 for record in records if record.get("bug_present"))
    if correct_count != 100:
        issues.append(f"n_correct={correct_count}, expected 100")
    if buggy_count != 100:
        issues.append(f"n_buggy={buggy_count}, expected 100")

    for pair_id, members in sorted(pairs.items()):
        if len(members) != 2:
            issues.append(f"{pair_id}: expected 2 snippets, found {len(members)}")
            continue
        flags = Counter(bool(member.get("bug_present")) for member in members)
        if flags[False] != 1 or flags[True] != 1:
            issues.append(f"{pair_id}: expected one correct and one buggy snippet")
        domains = {short_domain(member.get("domain", "")) for member in members}
        languages = {member.get("language") for member in members}
        if len(domains) != 1:
            issues.append(f"{pair_id}: pair has mixed domains {sorted(domains)}")
        if len(languages) != 1:
            issues.append(f"{pair_id}: pair has mixed languages {sorted(languages)}")

    domain_counts = Counter(short_domain(members[0].get("domain", "")) for members in pairs.values() if members)
    language_counts = Counter(members[0].get("language") for members in pairs.values() if members)
    category_counts = Counter(record.get("bug_category") for record in records if record.get("bug_present"))

    if dict(sorted(domain_counts.items())) != EXPECTED_DOMAINS:
        issues.append(f"domain counts={dict(sorted(domain_counts.items()))}, expected {EXPECTED_DOMAINS}")
    if dict(sorted(language_counts.items())) != EXPECTED_LANGUAGES:
        issues.append(f"language pair counts={dict(sorted(language_counts.items()))}, expected {EXPECTED_LANGUAGES}")
    if dict(sorted(category_counts.items())) != EXPECTED_CATEGORIES:
        issues.append(f"bug category counts={dict(sorted(category_counts.items()))}, expected {EXPECTED_CATEGORIES}")

    return issues


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, default=ANNOTATIONS)
    args = parser.parse_args()

    path = args.annotations if args.annotations.is_absolute() else ROOT / args.annotations
    issues = validate_counts(load_records(path))
    if issues:
        print("Dataset count validation failed:")
        for issue in issues:
            print(f"- {issue}")
        return 1

    print("Dataset count validation passed: 100 pairs, 200 snippets, balanced distribution.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
