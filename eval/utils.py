"""Shared paths, file helpers and agent configuration."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = ROOT / "results"
AGENT_KEYS = (
    "codex_gpt54",
    "codex_gpt55",
    "claude_code_sonnet46",
    "claude_code_opus48",
    "claude_code_deepseek_flash",
    "claude_code_deepseek_pro",
)


def resolve(path: Path | str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def resolve_run(run_name: str | None = None) -> str:
    if run_name:
        return run_name
    runs = sorted(path.parent.parent.name for path in RESULTS_ROOT.glob("*/rendered/rendered_ground_truth.jsonl"))
    if len(runs) == 1:
        return runs[0]
    if len(runs) > 1:
        raise ValueError("Multiple result directories found; select one with --run NAME.")
    return "benchmark"


def run_dir(run_name: str | None = None) -> Path:
    return RESULTS_ROOT / resolve_run(run_name)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + ("\n" if rows else ""))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def clean_dir(path: Path, fresh: bool = False) -> None:
    if fresh and path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def load_agent_matrix(path: Path) -> dict[str, dict[str, str]]:
    """Load the simple checked-in agent matrix YAML without requiring PyYAML."""

    matrix: dict[str, dict[str, str]] = {}
    current_key: str | None = None
    for raw_line in path.read_text().splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        if not line.startswith(" ") and line.endswith(":"):
            current_key = line[:-1].strip()
            matrix[current_key] = {}
            continue
        if current_key is None or ":" not in line:
            raise ValueError(f"Unsupported matrix line: {raw_line!r}")
        key, value = line.strip().split(":", 1)
        matrix[current_key][key.strip()] = value.strip().strip("\"'")

    missing = [key for key in AGENT_KEYS if key not in matrix]
    extra = [key for key in matrix if key not in AGENT_KEYS]
    if missing or extra:
        raise ValueError(f"Agent matrix keys mismatch; missing={missing}, extra={extra}")
    return matrix


def rendered_dir_for(run_name: str) -> Path:
    return run_dir(run_name) / "rendered"


def ground_truth_path(run_name: str) -> Path:
    return rendered_dir_for(run_name) / "rendered_ground_truth.jsonl"


def private_manifest_path(run_name: str) -> Path:
    return rendered_dir_for(run_name) / "manifest_private.jsonl"
