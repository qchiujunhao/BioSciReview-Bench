"""Validate agent run workspaces, metadata, and raw outputs."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.create_sample_workspace import ALLOWED_FILES, workspace_path
from eval.utils import (
    resolve_run,
    ROOT,
    load_agent_matrix,
    private_manifest_path,
    read_jsonl,
    sha256_file,
    run_dir,
    write_json,
)


ID_RE = re.compile(r"\b(?:PY|R)_D[1-5]_B[1-6]_\d{3}(?:_(?:correct|buggy))?\b")
FORBIDDEN_FILE_RE = re.compile(
    r"annotations|manifest|results|paper|data|CLAUDE\.md|AGENTS\.md|\.codex|\.claude|mcp",
    re.IGNORECASE,
)
FORBIDDEN_LOG_RE = re.compile(
    r"web_search(?!_requests)|network access|mcp__|slash command|CLAUDE\.md|AGENTS\.md|annotations\.jsonl|manifest_private|data/snippets|/paper|failed to load skill|codex_core_plugins|\.codex/\.tmp|/plugins/",
    re.IGNORECASE,
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text()) if path.exists() else {}


def validate_workspace(run_name: str, agent_key: str, row: dict[str, Any], schema_sha: str) -> list[str]:
    issues: list[str] = []
    workspace = workspace_path(run_name, agent_key, row["neutral_id"])
    if not workspace.exists():
        return [f"{agent_key}/{row['neutral_id']}: missing workspace {workspace}"]
    files = sorted(path.name for path in workspace.iterdir() if path.is_file())
    dirs = sorted(path.name for path in workspace.iterdir() if path.is_dir())
    unexpected = [name for name in files if name not in ALLOWED_FILES]
    if unexpected:
        issues.append(f"{agent_key}/{row['neutral_id']}: unexpected workspace files {unexpected}")
    if dirs:
        issues.append(f"{agent_key}/{row['neutral_id']}: unexpected workspace directories {dirs}")
    snippet_files = [name for name in files if name in {"snippet.py", "snippet.R"}]
    if len(snippet_files) != 1:
        issues.append(f"{agent_key}/{row['neutral_id']}: expected exactly one snippet.py or snippet.R")
    for name in files + dirs:
        if FORBIDDEN_FILE_RE.search(name):
            issues.append(f"{agent_key}/{row['neutral_id']}: forbidden workspace path name {name}")
        if ID_RE.search(name):
            issues.append(f"{agent_key}/{row['neutral_id']}: original id appears in workspace file name {name}")

    all_text = "\n".join(path.read_text(errors="replace") for path in workspace.iterdir() if path.is_file())
    if ID_RE.search(all_text):
        issues.append(f"{agent_key}/{row['neutral_id']}: original id or pair id appears in workspace text")
    if "manifest_private" in all_text or "annotations.jsonl" in all_text:
        issues.append(f"{agent_key}/{row['neutral_id']}: annotation or private manifest reference in workspace")
    if sha256_file(workspace / "prompt.txt") != row["prompt_sha256"]:
        issues.append(f"{agent_key}/{row['neutral_id']}: prompt hash does not match rendered manifest")
    if sha256_file(workspace / "prediction_schema.json") != schema_sha:
        issues.append(f"{agent_key}/{row['neutral_id']}: schema hash does not match expected schema")
    return issues


def validate_run_metadata(
    run_name: str,
    agent_key: str,
    config: dict[str, str],
    row: dict[str, Any],
    allow_dry_run: bool,
) -> list[str]:
    issues: list[str] = []
    raw_dir = run_dir(run_name) / "raw" / config["result_dir"]
    meta_path = raw_dir / f"{row['neutral_id']}.run_meta.json"
    meta = load_json(meta_path)
    if not meta:
        return [f"{agent_key}/{row['neutral_id']}: missing run metadata"]
    dry_run = bool(meta.get("dry_run"))
    skip_execution_checks = dry_run and allow_dry_run

    if not skip_execution_checks:
        raw_path = raw_dir / f"{row['neutral_id']}.raw.json"
        if not raw_path.exists():
            issues.append(f"{agent_key}/{row['neutral_id']}: missing raw parsed output")
        else:
            raw = load_json(raw_path)
            if raw.get("parse_status") != "ok":
                issues.append(f"{agent_key}/{row['neutral_id']}: parsed output status is {raw.get('parse_status')!r}")
            if raw.get("parse_error") or raw.get("schema_errors"):
                issues.append(f"{agent_key}/{row['neutral_id']}: parsed output has parse/schema errors")

        if not meta.get("success"):
            issues.append(f"{agent_key}/{row['neutral_id']}: invocation was not successful: {meta.get('failure_reason')}")
        if meta.get("status") != "completed":
            issues.append(f"{agent_key}/{row['neutral_id']}: run status is {meta.get('status')!r}")
        if meta.get("exit_code") != 0:
            issues.append(f"{agent_key}/{row['neutral_id']}: exit_code is {meta.get('exit_code')!r}")

    if meta.get("isolation_mode") != "temporary_runtime_outside_repo":
        issues.append(f"{agent_key}/{row['neutral_id']}: unexpected isolation mode {meta.get('isolation_mode')!r}")
    actual_cwd = str(meta.get("actual_cwd", ""))
    if not actual_cwd:
        issues.append(f"{agent_key}/{row['neutral_id']}: missing actual_cwd metadata")
    elif str(ROOT) in actual_cwd:
        issues.append(f"{agent_key}/{row['neutral_id']}: actual_cwd is inside repository: {actual_cwd}")
    runtime_audit = meta.get("runtime_audit", {})
    for key in (
        "repo_root_visible_by_parent_traversal",
        "runtime_inside_repo",
        "data_visible_by_parent_traversal",
        "results_visible_by_parent_traversal",
        "annotations_visible",
        "manifest_private_visible",
        "manifest_public_visible",
        "rendered_ground_truth_visible",
        "original_id_visible",
        "source_path_visible",
        "pair_id_visible",
        "correct_or_buggy_label_visible",
        "claude_or_codex_config_visible",
        "paper_files_visible",
    ):
        if runtime_audit.get(key):
            issues.append(f"{agent_key}/{row['neutral_id']}: runtime audit violation {key}={runtime_audit.get(key)!r}")
    if runtime_audit.get("unexpected_files"):
        issues.append(f"{agent_key}/{row['neutral_id']}: unexpected runtime files {runtime_audit['unexpected_files']}")
    if runtime_audit.get("unexpected_dirs"):
        issues.append(f"{agent_key}/{row['neutral_id']}: unexpected runtime directories {runtime_audit['unexpected_dirs']}")
    runtime_files = sorted(meta.get("workspace_file_listing", []))
    unexpected_runtime_files = [name for name in runtime_files if name not in ALLOWED_FILES]
    snippet_files = [name for name in runtime_files if name in {"snippet.py", "snippet.R"}]
    for required in ("prompt.txt", "prediction_schema.json", "README.txt"):
        if required not in runtime_files:
            issues.append(f"{agent_key}/{row['neutral_id']}: runtime workspace missing {required}")
    if unexpected_runtime_files:
        issues.append(f"{agent_key}/{row['neutral_id']}: unexpected runtime workspace files {unexpected_runtime_files}")
    if len(snippet_files) != 1:
        issues.append(f"{agent_key}/{row['neutral_id']}: runtime workspace expected exactly one snippet file")

    if not skip_execution_checks:
        for path in sorted(raw_dir.glob(f"{row['neutral_id']}*.log")):
            match = FORBIDDEN_LOG_RE.search(path.read_text(errors="replace"))
            if match:
                issues.append(f"{agent_key}/{row['neutral_id']}: forbidden log pattern in {path.name}: {match.group(0)}")

    if config["harness"] == "claude_code":
        default_claude_code_auth = config["provider"] == "anthropic" and meta.get("claude_auth_mode") == "default_claude_code"
        expected_home_policy = "default_claude_code_auth" if default_claude_code_auth else "temporary_claude_home_auth_only"
        expected_home_path_type = "default_user_home" if default_claude_code_auth else "temporary"
        if meta.get("home_policy") != expected_home_policy:
            issues.append(f"{agent_key}/{row['neutral_id']}: Claude home policy is {meta.get('home_policy')!r}")
        if meta.get("home_path_type") != expected_home_path_type:
            issues.append(
                f"{agent_key}/{row['neutral_id']}: Claude home path type is {meta.get('home_path_type')!r}, expected {expected_home_path_type!r}"
            )
        if meta.get("copied_user_claude_dir") is not False:
            issues.append(f"{agent_key}/{row['neutral_id']}: Claude metadata does not prove user dir was not copied")
        if meta.get("mcp_enabled") is not False:
            issues.append(f"{agent_key}/{row['neutral_id']}: Claude MCP metadata is not disabled")
        if meta.get("session_persistence") is not False:
            issues.append(f"{agent_key}/{row['neutral_id']}: Claude session persistence metadata is not disabled")
        if meta.get("tools_policy") != "disabled":
            issues.append(f"{agent_key}/{row['neutral_id']}: Claude tools policy is not disabled")
        if meta.get("project_instructions_visible") is not False:
            issues.append(f"{agent_key}/{row['neutral_id']}: Claude project instructions visibility metadata is not false")
        if not default_claude_code_auth and "--bare" not in meta.get("command_argv_redacted", []):
            issues.append(f"{agent_key}/{row['neutral_id']}: Claude command missing --bare")

    if config["provider"] == "deepseek":
        if meta.get("provider") != "deepseek":
            issues.append(f"{agent_key}/{row['neutral_id']}: DeepSeek run metadata provider is not deepseek")
        if meta.get("gateway") != "anthropic_compatible":
            issues.append(f"{agent_key}/{row['neutral_id']}: DeepSeek gateway is not Anthropic-compatible")
        if meta.get("requested_model") not in {"deepseek-v4-flash", "deepseek-v4-pro"}:
            issues.append(f"{agent_key}/{row['neutral_id']}: unexpected DeepSeek model {meta.get('requested_model')}")
        env = meta.get("environment_redacted", {})
        if env.get("ANTHROPIC_AUTH_TOKEN") == "<missing>" and not dry_run:
            issues.append(f"{agent_key}/{row['neutral_id']}: missing DeepSeek auth token metadata")
    elif config["provider"] == "anthropic":
        env = meta.get("environment_redacted", {})
        if "ANTHROPIC_BASE_URL" in env or "ANTHROPIC_MODEL" in env:
            issues.append(f"{agent_key}/{row['neutral_id']}: Anthropic run has DeepSeek gateway env metadata")
    elif config["provider"] == "openai":
        command = " ".join(meta.get("command_argv_redacted", []))
        if config["model"] not in command:
            issues.append(f"{agent_key}/{row['neutral_id']}: Codex command missing requested model")
        if "read-only" not in command:
            issues.append(f"{agent_key}/{row['neutral_id']}: Codex command missing read-only sandbox")
        if re.search(r"\b(web|search)\b", command, re.IGNORECASE):
            issues.append(f"{agent_key}/{row['neutral_id']}: Codex command appears to enable web/search")
        if meta.get("home_policy") != "temporary_codex_home_auth_only":
            issues.append(f"{agent_key}/{row['neutral_id']}: Codex home policy is {meta.get('home_policy')!r}")
    return issues


def validate_runs(run_name: str, matrix_path: Path, allow_dry_run: bool = False) -> dict[str, Any]:
    matrix = load_agent_matrix(matrix_path)
    manifest = read_jsonl(private_manifest_path(run_name))
    schema_sha = sha256_file(ROOT / "eval" / "prediction_schema.json")
    issues: list[str] = []
    audited = 0
    for agent_key, config in matrix.items():
        for row in manifest:
            issues.extend(validate_workspace(run_name, agent_key, row, schema_sha))
            issues.extend(validate_run_metadata(run_name, agent_key, config, row, allow_dry_run=allow_dry_run))
            audited += 1
    return {"run": run_name, "audited_invocations": audited, "issue_count": len(issues), "issues": issues}


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Agent Run Validation",
        "",
        f"Run: `{report['run']}`",
        f"Audited invocations: {report['audited_invocations']}",
        f"Issues: {report['issue_count']}",
        "",
    ]
    if report["issues"]:
        lines.append("## Issues")
        lines.extend(f"- {issue}" for issue in report["issues"][:500])
    else:
        lines.append("No serious workspace or run-metadata violations found.")
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", dest="run_name", default=None)
    parser.add_argument("--matrix", type=Path, default=ROOT / "eval" / "agent_matrix.yaml")
    parser.add_argument("--allow-dry-run", action="store_true")
    args = parser.parse_args()
    args.run_name = resolve_run(args.run_name)

    report = validate_runs(args.run_name, args.matrix, allow_dry_run=args.allow_dry_run)
    analysis_dir = run_dir(args.run_name) / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    write_json(analysis_dir / "run_validation.json", report)
    write_markdown(report, analysis_dir / "run_validation.md")
    if report["issues"]:
        print(f"Agent-run validation failed with {report['issue_count']} issues.")
        for issue in report["issues"][:50]:
            print(f"- {issue}")
        return 1
    print(f"Agent-run validation passed for {args.run_name}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
