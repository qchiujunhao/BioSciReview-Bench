"""Render leakage-safe inputs for BioSciReview-Bench agent evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import shutil
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ANNOTATIONS = ROOT / "data" / "annotations.jsonl"

CATEGORY_DEFINITIONS = {
    "B1": "Incorrect normalization, scaling, or transformation.",
    "B2": "Using the wrong statistical test for the data structure.",
    "B3": "Missing or incorrect multiple-testing correction.",
    "B4": "Sample mismatch, label swap, or metadata merge error.",
    "B5": "Information leakage between training and evaluation data.",
    "B6": "Incorrect filtering, thresholding, or quality-control criteria.",
}

CATEGORY_TO_CODE = {
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

ID_RE = re.compile(r"\b(?:PY|R)_D[1-5]_B[1-6]_\d{3}_(?:correct|buggy)\b")
FULL_LINE_MAINTAINER_RE = re.compile(
    r"^\s*#+\s*(?:TASK|BUG|BUG\s+LINE|EXPECTED|HINT|MAINTAINER|INTERNAL|GOLD)\s*:?.*$",
    re.IGNORECASE,
)
INLINE_MAINTAINER_RE = re.compile(
    r"^\s*#+\s*(?:BUG(?:\s+LINE)?|<--\s*BUG)\b.*$",
    re.IGNORECASE,
)

TEXT_REPLACEMENTS: list[tuple[re.Pattern[str], str]] = [
    (ID_RE, "neutral_snippet"),
    (re.compile(r"data/snippets/[^\s'\"`]+", re.IGNORECASE), "neutral_snippet"),
    (re.compile(r"\bsnippets/(?:python|r)/(?:correct|buggy)/[^\s'\"`]+", re.IGNORECASE), "neutral_snippet"),
    (re.compile(r"\bB[1-6](?:_[A-Za-z0-9_]+)?\b"), "analysis_category"),
    (re.compile(r"\bdata_leakage\b", re.IGNORECASE), "heldout_evaluation"),
    (re.compile(r"\bmultiple_testing\b", re.IGNORECASE), "multi_hypothesis"),
    (re.compile(r"\bsample_label\b", re.IGNORECASE), "metadata_alignment"),
    (re.compile(r"\bfilter_threshold\b", re.IGNORECASE), "record_selection"),
    (re.compile(r"\bstatistical_test\b", re.IGNORECASE), "comparison_method"),
    (re.compile(r"\bnormalization\b", re.IGNORECASE), "scaling"),
    (re.compile(r"\bmultiple[- ]testing\b", re.IGNORECASE), "multi-hypothesis"),
    (re.compile(r"\bbuggy\b", re.IGNORECASE), "variant"),
    (re.compile(r"\bcorrect\b", re.IGNORECASE), "reference"),
]


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_annotations(path: Path = ANNOTATIONS) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open() as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def normalize_category(category: str | None) -> str | None:
    if category is None:
        return None
    return CATEGORY_TO_CODE.get(category, category)


def neutralize_text(text: str) -> str:
    neutral = text
    for pattern, replacement in TEXT_REPLACEMENTS:
        neutral = pattern.sub(replacement, neutral)
    return neutral


def is_full_line_maintainer_comment(line: str) -> bool:
    return bool(FULL_LINE_MAINTAINER_RE.match(line))


def strip_inline_maintainer_marker(line: str) -> str:
    comment_index = line.find("#")
    if comment_index < 0:
        return line

    before = line[:comment_index]
    comment = line[comment_index:]
    if before.strip() and INLINE_MAINTAINER_RE.match(comment):
        return before.rstrip()
    return line


def clean_code_and_map(source: str, language: str) -> tuple[str, dict[int, int | None]]:
    """Remove maintainer metadata while preserving a source-to-rendered line map."""

    del language  # R and Python both use # comments in this benchmark.
    source_to_rendered: dict[int, int | None] = {}
    kept: list[tuple[int, str]] = []

    for source_line_number, raw_line in enumerate(source.splitlines(), start=1):
        if is_full_line_maintainer_comment(raw_line):
            source_to_rendered[source_line_number] = None
            continue

        cleaned = neutralize_text(strip_inline_maintainer_marker(raw_line))
        kept.append((source_line_number, cleaned))

    while kept and not kept[0][1].strip():
        source_to_rendered[kept[0][0]] = None
        kept.pop(0)
    while kept and not kept[-1][1].strip():
        source_to_rendered[kept[-1][0]] = None
        kept.pop()

    rendered_lines: list[str] = []
    for rendered_line_number, (source_line_number, line) in enumerate(kept, start=1):
        source_to_rendered[source_line_number] = rendered_line_number
        rendered_lines.append(line)

    rendered = "\n".join(rendered_lines)
    if rendered:
        rendered += "\n"
    return rendered, source_to_rendered


def clean_code(source: str) -> str:
    """Backward-compatible wrapper returning only rendered code."""

    return clean_code_and_map(source, "python")[0]


def clean_task(task: str) -> str:
    return neutralize_text(task).strip()


def language_for_path(path: Path) -> str:
    return "r" if path.suffix == ".R" else "python"


def select_records(
    records: list[dict[str, Any]],
    snippet_list: Path | None,
    snippet_filter: str,
    max_snippets: int,
) -> list[dict[str, Any]]:
    by_id = {record["id"]: record for record in records}
    if snippet_list:
        selected: list[dict[str, Any]] = []
        for raw_line in snippet_list.read_text().splitlines():
            snippet_id = raw_line.strip()
            if not snippet_id or snippet_id.startswith("#"):
                continue
            if snippet_id not in by_id:
                raise ValueError(f"Missing snippet listed in {snippet_list}: {snippet_id}")
            selected.append(by_id[snippet_id])
    else:
        selected = sorted(records, key=lambda record: (record.get("pair_id", ""), record["id"]))

    if snippet_filter:
        selected = [
            record
            for record in selected
            if snippet_filter in record["id"] or snippet_filter in record.get("path", "")
        ]

    if max_snippets > 0:
        selected = selected[:max_snippets]
    return selected


def assign_neutral_names(records: list[dict[str, Any]], seed: int) -> dict[str, str]:
    names: dict[str, str] = {}
    used: set[int] = set()
    for record in records:
        rng = random.Random(f"{seed}:{record['id']}")
        number = rng.randrange(100000)
        while number in used:
            number = (number + 1) % 100000
        used.add(number)
        names[record["id"]] = f"sample_{number:05d}"
    return names


def numbered_code(code: str) -> str:
    lines = code.splitlines()
    width = max(1, len(str(len(lines))))
    return "\n".join(f"{index:>{width}} | {line}" for index, line in enumerate(lines, start=1))


def render_prompt(
    agent_instructions: str,
    prompt_template: str,
    language: str,
    code: str,
    task: str,
) -> str:
    rendered = (
        prompt_template.replace("{language}", language)
        .replace("{code}", numbered_code(code))
        .replace("{task_description}", task)
        .replace(
            "{bug_categories}",
            "\n".join(f"{code}: {description}" for code, description in CATEGORY_DEFINITIONS.items()),
        )
    )
    prefix = agent_instructions.strip()
    body = rendered.strip()
    return f"{prefix}\n\n{body}\n" if prefix else f"{body}\n"


def format_manifest_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def public_manifest_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "index": row["index"],
        "neutral_id": row["neutral_id"],
        "neutral_filename": row["neutral_filename"],
        "language": row["language"],
        "rendered_code_path": row["rendered_code_path"],
        "rendered_prompt_path": row["rendered_prompt_path"],
        "code_sha256": row["code_sha256"],
        "prompt_sha256": row["prompt_sha256"],
    }


def ground_truth_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "neutral_id": row["neutral_id"],
        "original_id": row["original_id"],
        "pair_id": row["pair_id"],
        "language": row["language"],
        "domain": row["domain"],
        "bug_present": row["bug_present"],
        "bug_category": row["bug_category"],
        "source_bug_line": row["source_bug_line"],
        "rendered_bug_line": row["rendered_bug_line"],
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + ("\n" if rows else ""))


def render_dataset(
    output_dir: Path,
    prompt_file: Path,
    agent_instructions_file: Path | None = None,
    snippet_list: Path | None = None,
    snippet_filter: str = "",
    max_snippets: int = 0,
    seed: int = 20260613,
    fresh: bool = False,
) -> list[dict[str, Any]]:
    if fresh and output_dir.exists():
        shutil.rmtree(output_dir)
    snippets_dir = output_dir / "snippets"
    prompts_dir = output_dir / "prompts"
    snippets_dir.mkdir(parents=True, exist_ok=True)
    prompts_dir.mkdir(parents=True, exist_ok=True)

    records = select_records(load_annotations(), snippet_list, snippet_filter, max_snippets)
    neutral_names = assign_neutral_names(records, seed)
    prompt_template = prompt_file.read_text()
    agent_instructions = agent_instructions_file.read_text() if agent_instructions_file else ""

    manifest: list[dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        source_path = ROOT / record["path"]
        source_code = source_path.read_text()
        language = record.get("language") or language_for_path(source_path)
        extension = ".R" if language == "r" else ".py"
        neutral_id = neutral_names[record["id"]]
        neutral_filename = f"{neutral_id}{extension}"
        rendered_code, source_to_rendered_line = clean_code_and_map(source_code, language)
        rendered_task = clean_task(record["task_description"])

        source_bug_line = record.get("bug_line")
        rendered_bug_line: int | None = None
        if record.get("bug_present"):
            if not isinstance(source_bug_line, int):
                raise ValueError(f"{record['id']}: buggy record has non-integer bug_line {source_bug_line!r}")
            rendered_bug_line = source_to_rendered_line.get(source_bug_line)
            if rendered_bug_line is None:
                raise ValueError(
                    f"{record['id']}: source bug_line {source_bug_line} was deleted by renderer"
                )

        rendered_prompt = render_prompt(
            agent_instructions=agent_instructions,
            prompt_template=prompt_template,
            language=language,
            code=rendered_code,
            task=rendered_task,
        )

        code_path = snippets_dir / neutral_filename
        prompt_path = prompts_dir / f"{neutral_id}.txt"
        code_path.write_text(rendered_code)
        prompt_path.write_text(rendered_prompt)

        manifest.append(
            {
                "index": index,
                "original_id": record["id"],
                "pair_id": record["pair_id"],
                "neutral_id": neutral_id,
                "neutral_filename": neutral_filename,
                "language": language,
                "domain": record["domain"],
                "bug_present": bool(record["bug_present"]),
                "bug_category": normalize_category(record.get("bug_category")),
                "source_bug_line": source_bug_line,
                "rendered_bug_line": rendered_bug_line,
                "source_path": record["path"],
                "rendered_code_path": format_manifest_path(code_path),
                "rendered_prompt_path": format_manifest_path(prompt_path),
                "task_description": rendered_task,
                "source_to_rendered_line": source_to_rendered_line,
                "code_sha256": sha256_text(rendered_code),
                "prompt_sha256": sha256_text(rendered_prompt),
            }
        )

    public_manifest = [public_manifest_row(row) for row in manifest]
    rendered_ground_truth = [ground_truth_row(row) for row in manifest]
    write_jsonl(output_dir / "manifest_private.jsonl", manifest)
    write_jsonl(output_dir / "manifest_public.jsonl", public_manifest)
    write_jsonl(output_dir / "rendered_ground_truth.jsonl", rendered_ground_truth)
    write_jsonl(output_dir / "manifest.jsonl", manifest)

    config = {
        "prompt_file": format_manifest_path(prompt_file),
        "agent_instructions_file": format_manifest_path(agent_instructions_file) if agent_instructions_file else None,
        "snippet_list": format_manifest_path(snippet_list) if snippet_list else None,
        "snippet_filter": snippet_filter,
        "max_snippets": max_snippets,
        "seed": seed,
        "count": len(manifest),
        "line_numbering": "rendered_prompt_display_lines",
        "renderer": "clean_code_and_map",
    }
    (output_dir / "render_config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    return manifest


def load_manifest(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def print_manifest_tsv(manifest_path: Path) -> None:
    for row in load_manifest(manifest_path):
        print(
            "\t".join(
                [
                    row["neutral_id"],
                    row["neutral_filename"],
                    row["language"],
                    row["rendered_prompt_path"],
                    row["rendered_code_path"],
                ]
            )
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prompt-file", type=Path, default=ROOT / "eval" / "prompts" / "baseline.txt")
    parser.add_argument("--agent-instructions", type=Path)
    parser.add_argument("--snippet-list", type=Path)
    parser.add_argument("--snippet-filter", default="")
    parser.add_argument("--max-snippets", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260613)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--print-manifest-tsv", action="store_true")
    args = parser.parse_args()

    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    if args.print_manifest_tsv:
        print_manifest_tsv(output_dir / "manifest_private.jsonl")
        return 0

    render_dataset(
        output_dir=output_dir,
        prompt_file=args.prompt_file if args.prompt_file.is_absolute() else ROOT / args.prompt_file,
        agent_instructions_file=(
            args.agent_instructions
            if not args.agent_instructions or args.agent_instructions.is_absolute()
            else ROOT / args.agent_instructions
        ),
        snippet_list=args.snippet_list if not args.snippet_list or args.snippet_list.is_absolute() else ROOT / args.snippet_list,
        snippet_filter=args.snippet_filter,
        max_snippets=args.max_snippets,
        seed=args.seed,
        fresh=args.fresh,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
