"""Validate rendered evaluation inputs for model-facing answer leakage."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

NEUTRAL_FILENAME_RE = re.compile(r"^(?:snippet|sample)_\d{5}\.(?:py|R)$")
ORIGINAL_ID_RE = re.compile(r"\b(?:PY|R)_D[1-5]_B[1-6]_\d{3}_(?:correct|buggy)\b")

PROMPT_FORBIDDEN_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("original snippet id", ORIGINAL_ID_RE),
    ("original snippets path", re.compile(r"data/snippets|data/releases/.*/snippets", re.IGNORECASE)),
    ("correct/buggy path segment", re.compile(r"[/\\](?:correct|buggy)[/\\]", re.IGNORECASE)),
    ("maintainer bug line marker", re.compile(r"\bBUG\s+LINE\b|<--\s*BUG", re.IGNORECASE)),
    ("fixme marker", re.compile(r"\bFIXME\b", re.IGNORECASE)),
    ("buggy label", re.compile(r"\bbuggy\b", re.IGNORECASE)),
]

CODE_EXTRA_FORBIDDEN_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "maintainer comment",
        re.compile(
            r"^\s*#+\s*(?:TASK|BUG|BUG\s+LINE|EXPECTED|HINT|MAINTAINER|INTERNAL|GOLD)\b",
            re.IGNORECASE | re.MULTILINE,
        ),
    ),
]

SCHEMA_FORBIDDEN_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("original snippet id", ORIGINAL_ID_RE),
    ("original snippets path", re.compile(r"data/snippets|data/releases/.*/snippets", re.IGNORECASE)),
    ("correct/buggy path segment", re.compile(r"[/\\](?:correct|buggy)[/\\]", re.IGNORECASE)),
    ("maintainer bug line marker", re.compile(r"\bBUG\s+LINE\b|<--\s*BUG", re.IGNORECASE)),
    ("fixme marker", re.compile(r"\bFIXME\b", re.IGNORECASE)),
]


def load_manifest(rendered_dir: Path) -> list[dict]:
    manifest_path = rendered_dir / "manifest_private.jsonl"
    if not manifest_path.exists():
        manifest_path = rendered_dir / "manifest.jsonl"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    with manifest_path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def resolve_rendered_path(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def scan_text(label: str, text: str, patterns: list[tuple[str, re.Pattern[str]]]) -> list[str]:
    issues: list[str] = []
    for name, pattern in patterns:
        match = pattern.search(text)
        if match:
            excerpt = text[max(0, match.start() - 40) : match.end() + 40].replace("\n", "\\n")
            issues.append(f"{label}: {name}: {excerpt}")
    return issues


def validate_rendered_dir(rendered_dir: Path, schema_path: Path | None = None) -> list[str]:
    issues: list[str] = []
    for row in load_manifest(rendered_dir):
        neutral_filename = row["neutral_filename"]
        if not NEUTRAL_FILENAME_RE.fullmatch(neutral_filename):
            issues.append(f"{row['neutral_id']}: non-neutral filename {neutral_filename!r}")
        if row["neutral_id"] not in neutral_filename:
            issues.append(f"{row['neutral_id']}: neutral id not reflected in filename {neutral_filename!r}")

        prompt_path = resolve_rendered_path(row["rendered_prompt_path"])
        code_path = resolve_rendered_path(row["rendered_code_path"])
        prompt_text = prompt_path.read_text()
        code_text = code_path.read_text()

        issues.extend(scan_text(str(prompt_path), prompt_text, PROMPT_FORBIDDEN_PATTERNS))
        issues.extend(scan_text(str(code_path), code_text, PROMPT_FORBIDDEN_PATTERNS))
        issues.extend(scan_text(str(code_path), code_text, CODE_EXTRA_FORBIDDEN_PATTERNS))

        original_id = row["original_id"]
        source_path = row["source_path"]
        for model_facing_label, text in ((str(prompt_path), prompt_text), (str(code_path), code_text)):
            if original_id in text:
                issues.append(f"{model_facing_label}: contains original id {original_id}")
            if source_path in text:
                issues.append(f"{model_facing_label}: contains original source path {source_path}")

    if schema_path:
        schema = schema_path if schema_path.is_absolute() else ROOT / schema_path
        issues.extend(scan_text(str(schema), schema.read_text(), SCHEMA_FORBIDDEN_PATTERNS))

    return issues


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("rendered_dir", type=Path)
    parser.add_argument("--schema", type=Path)
    args = parser.parse_args()

    rendered_dir = args.rendered_dir if args.rendered_dir.is_absolute() else ROOT / args.rendered_dir
    issues = validate_rendered_dir(rendered_dir, args.schema)
    if issues:
        print("Input validation failed:")
        for issue in issues:
            print(f"- {issue}")
        return 1
    print(f"Input validation passed for {rendered_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
