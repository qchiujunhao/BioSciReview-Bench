"""Run or dry-run the coding-agent matrix on isolated sample workspaces."""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.create_sample_workspace import create_workspaces, workspace_path
from eval.parse_output import parse_text
from eval.utils import (
    resolve_run,
    ROOT,
    load_agent_matrix,
    private_manifest_path,
    read_jsonl,
    rendered_dir_for,
    resolve,
    sha256_file,
    sha256_text,
    run_dir,
    write_json,
    write_jsonl,
)


REAL_HOME = Path.home()
RUNTIME_ROOT = REAL_HOME / ".cache" / "bioscireview_agent_runtime"
AGENT_HOME_ROOT = REAL_HOME / ".cache" / "bioscireview_agent_homes"
ALLOWED_RUNTIME_FILES = {"prompt.txt", "snippet.py", "snippet.R", "prediction_schema.json", "README.txt"}


def utc_now() -> str:
    return dt.datetime.now(dt.UTC).isoformat().replace("+00:00", "Z")


def run_capture(command: list[str], timeout: int = 20) -> dict[str, Any]:
    executable = shutil.which(command[0])
    if not executable:
        return {"available": False, "command": command, "stdout": "", "stderr": "", "returncode": None}
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
    except Exception as exc:
        return {"available": True, "command": command, "stdout": "", "stderr": str(exc), "returncode": None}
    return {
        "available": True,
        "command": command,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "returncode": result.returncode,
        "sha256": sha256_text(result.stdout + result.stderr),
    }


def collect_cli_metadata(harness: str) -> dict[str, Any]:
    if harness == "codex_cli":
        return {
            "codex_version": run_capture(["codex", "--version"]),
            "codex_help": run_capture(["codex", "--help"]),
            "codex_exec_help": run_capture(["codex", "exec", "--help"]),
        }
    if harness == "claude_code":
        return {
            "claude_version": run_capture(["claude", "--version"]),
            "claude_help": run_capture(["claude", "--help"]),
        }
    return {}


@contextlib.contextmanager
def isolated_codex_home(run_name: str, run_id: str, agent_key: str, neutral_id: str):
    """Provide Codex CLI with auth only, excluding user config, plugins, skills, and memory."""

    root = AGENT_HOME_ROOT / run_name / run_id / "codex" / agent_key
    root.mkdir(parents=True, exist_ok=True)
    os.chmod(root, 0o700)
    with tempfile.TemporaryDirectory(prefix=f"{neutral_id}_", dir=root) as tmp:
        tmp_path = Path(tmp)
        auth_src = REAL_HOME / ".codex" / "auth.json"
        if auth_src.exists():
            auth_dst = tmp_path / "auth.json"
            shutil.copyfile(auth_src, auth_dst)
            os.chmod(auth_dst, 0o600)
        yield str(tmp_path)


@contextlib.contextmanager
def isolated_claude_home(run_name: str, run_id: str, agent_key: str, neutral_id: str):
    """Provide Claude Code with an empty temporary HOME and no copied user config."""

    root = AGENT_HOME_ROOT / run_name / run_id / "claude" / agent_key
    root.mkdir(parents=True, exist_ok=True)
    os.chmod(root, 0o700)
    with tempfile.TemporaryDirectory(prefix=f"{neutral_id}_", dir=root) as tmp:
        tmp_path = Path(tmp)
        os.chmod(tmp_path, 0o700)
        yield str(tmp_path)


def smoke_neutral_ids(manifest: list[dict[str, Any]]) -> list[str]:
    by_pair: dict[str, list[dict[str, Any]]] = {}
    for row in manifest:
        by_pair.setdefault(row["pair_id"], []).append(row)

    selected_pairs: list[str] = []

    def add_pair(pair_id: str) -> None:
        if pair_id not in selected_pairs:
            selected_pairs.append(pair_id)

    for target in ("python", "r"):
        for pair_id, rows in sorted(by_pair.items()):
            if rows[0]["language"] == target:
                add_pair(pair_id)
                break
    for target in ("D1_bulk_rnaseq", "D2_single_cell", "D3_variant_genomics", "D4_ml_clinical", "D5_statistical_genomics"):
        for pair_id, rows in sorted(by_pair.items()):
            if rows[0]["domain"] == target:
                add_pair(pair_id)
                break
    for target in ("B1", "B2", "B3", "B4", "B5", "B6"):
        for pair_id, rows in sorted(by_pair.items()):
            if any(row.get("bug_category") == target for row in rows):
                add_pair(pair_id)
                break

    longest_pairs = sorted(
        by_pair,
        key=lambda pair_id: max(row.get("source_bug_line") or 0 for row in by_pair[pair_id]),
        reverse=True,
    )
    for pair_id in longest_pairs:
        if len(selected_pairs) >= 15:
            break
        add_pair(pair_id)
    for pair_id in sorted(by_pair):
        if len(selected_pairs) >= 15:
            break
        add_pair(pair_id)

    neutral_ids: list[str] = []
    for pair_id in selected_pairs[:15]:
        neutral_ids.extend(row["neutral_id"] for row in sorted(by_pair[pair_id], key=lambda row: row["neutral_id"]))
    return neutral_ids[:30]


def command_for_agent(
    config: dict[str, str],
    workspace: Path,
    output_dir: Path,
    reasoning_effort: str | None = None,
    allow_claude_default_home: bool = False,
) -> tuple[list[str], dict[str, str], str | None]:
    model = config["model"]
    prediction_path = output_dir / "prediction.json"
    prompt_text = (workspace / "prompt.txt").read_text()
    if config["harness"] == "codex_cli":
        command = [
            "codex",
            "exec",
            "--disable",
            "plugins",
            "--skip-git-repo-check",
            "--ephemeral",
            "--ignore-user-config",
            "--ignore-rules",
            "--sandbox",
            "read-only",
            "--model",
            model,
            "--output-schema",
            str(workspace / "prediction_schema.json"),
            "--output-last-message",
            str(prediction_path),
            "-",
        ]
        if reasoning_effort:
            command[2:2] = ["-c", f'model_reasoning_effort="{reasoning_effort}"']
        return command, {}, prompt_text

    executable = "claude"
    if config["provider"] == "deepseek":
        executable = "claude-ds-v4flash" if model == "deepseek-v4-flash" else "claude-ds-v4pro"

    command = [
        executable,
        "--print",
        "--no-session-persistence",
        "--tools",
        "",
        "--disallowedTools",
        "*",
        "--disallowedTools",
        "mcp__*",
        "--strict-mcp-config",
        "--disable-slash-commands",
        "--model",
        model,
        "--output-format",
        "json",
        "--json-schema",
        (workspace / "prediction_schema.json").read_text(),
    ]
    if not (config["provider"] == "anthropic" and allow_claude_default_home):
        command.insert(1, "--bare")
    env: dict[str, str] = {"CLAUDE_CODE_SKIP_PROMPT_HISTORY": "1"}
    if not (config["provider"] == "anthropic" and allow_claude_default_home):
        env["CLAUDE_CODE_SIMPLE"] = "1"
    if config["provider"] == "anthropic" and not allow_claude_default_home:
        if os.environ.get("ANTHROPIC_API_KEY"):
            env["ANTHROPIC_API_KEY"] = os.environ["ANTHROPIC_API_KEY"]
    if config["provider"] == "deepseek":
        token = deepseek_token()
        deepseek_base_url = os.environ.get("DEEPSEEK_ANTHROPIC_BASE_URL", "https://api.deepseek.com/anthropic")
        env.update(
            {
                "DEEPSEEK_ENV_FILE": os.environ.get("DEEPSEEK_ENV_FILE", ""),
                "DEEPSEEK_API_KEY": token,
                "ANTHROPIC_AUTH_TOKEN": token,
                "DEEPSEEK_ANTHROPIC_BASE_URL": deepseek_base_url,
                "ANTHROPIC_BASE_URL": deepseek_base_url,
                "ANTHROPIC_MODEL": model,
            }
        )
    return command, env, prompt_text


def redacted_env(env: dict[str, str]) -> dict[str, str]:
    redacted: dict[str, str] = {}
    for key, value in env.items():
        if "TOKEN" in key or "KEY" in key:
            redacted[key] = "<redacted>" if value else "<missing>"
        elif value:
            redacted[key] = value
        else:
            redacted[key] = "<missing>"
    return redacted


def redacted_argv(command: list[str]) -> list[str]:
    redacted: list[str] = []
    for index, part in enumerate(command):
        if index > 0 and command[index - 1] == "--json-schema":
            redacted.append("<schema-json>")
        elif index == len(command) - 1 and part == "-":
            redacted.append("<prompt-stdin>")
        elif index == len(command) - 1:
            redacted.append("<prompt>")
        elif len(part) > 240:
            redacted.append("<long-arg>")
        else:
            redacted.append(part)
    return redacted


def base_process_env() -> dict[str, str]:
    keep = [
        "PATH",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
        "REQUESTS_CA_BUNDLE",
        "NODE_EXTRA_CA_CERTS",
        "TMPDIR",
        "USER",
        "LOGNAME",
        "SHELL",
    ]
    return {key: os.environ[key] for key in keep if os.environ.get(key)}


def read_deepseek_token_from_env_file(path: Path) -> str:
    if not path.exists():
        return ""
    for raw_line in path.read_text(errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line.removeprefix("export ").strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip() not in {"DEEPSEEK_API_KEY", "DEEPSEEK_AUTH_TOKEN", "ANTHROPIC_AUTH_TOKEN", "deepseek"}:
            continue
        value = value.strip().strip('"').strip("'")
        return value
    return ""


def deepseek_token() -> str:
    env_file = os.environ.get("DEEPSEEK_ENV_FILE")
    return (
        os.environ.get("DEEPSEEK_API_KEY")
        or os.environ.get("ANTHROPIC_AUTH_TOKEN")
        or (read_deepseek_token_from_env_file(Path(env_file)) if env_file else "")
    )


def credentials_issue(
    config: dict[str, str],
    extra_env: dict[str, str],
    allow_claude_default_home: bool = False,
) -> str | None:
    if config["harness"] == "codex_cli":
        if not (REAL_HOME / ".codex" / "auth.json").exists():
            return "missing_codex_auth_json"
        return None
    if config["provider"] == "anthropic":
        if allow_claude_default_home:
            return None
        if not extra_env.get("ANTHROPIC_API_KEY"):
            return "missing_anthropic_api_key_for_claude_bare_mode"
        return None
    if config["provider"] == "deepseek":
        if not extra_env.get("ANTHROPIC_BASE_URL"):
            return "missing_deepseek_anthropic_base_url"
        if not extra_env.get("ANTHROPIC_AUTH_TOKEN"):
            return "missing_deepseek_api_key"
        return None
    return f"unsupported_provider_{config['provider']}"


def provider_metadata_issue(config: dict[str, str], meta: dict[str, Any]) -> str | None:
    provider = config["provider"]
    model = config["model"]
    if meta.get("provider") != provider:
        return f"provider metadata {meta.get('provider')!r} != {provider!r}"
    if meta.get("requested_model") != model:
        return f"requested_model metadata {meta.get('requested_model')!r} != {model!r}"
    if provider == "anthropic":
        env = meta.get("environment_redacted", {})
        if "ANTHROPIC_BASE_URL" in env or "ANTHROPIC_MODEL" in env:
            return "anthropic row inherited DeepSeek gateway env metadata"
        if model not in {"claude-sonnet-4-6", "claude-opus-4-8"}:
            return f"unexpected Anthropic model {model}"
    if provider == "deepseek":
        if meta.get("gateway") != "anthropic_compatible":
            return "DeepSeek gateway metadata is not anthropic_compatible"
        if model not in {"deepseek-v4-flash", "deepseek-v4-pro"}:
            return f"unexpected DeepSeek model {model}"
        env = meta.get("environment_redacted", {})
        if env.get("ANTHROPIC_AUTH_TOKEN") != "<redacted>":
            return "DeepSeek auth token metadata is missing"
        if env.get("ANTHROPIC_BASE_URL") in {None, "<missing>"}:
            return "DeepSeek base URL metadata is missing"
    return None


def path_is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def file_hashes(path: Path) -> dict[str, str]:
    return {child.name: sha256_file(child) for child in sorted(path.iterdir()) if child.is_file()}


def runtime_visibility_audit(runtime_workspace: Path, row: dict[str, Any]) -> dict[str, Any]:
    runtime_workspace = runtime_workspace.resolve()
    runtime_files = sorted(path.name for path in runtime_workspace.iterdir() if path.is_file())
    runtime_dirs = sorted(path.name for path in runtime_workspace.iterdir() if path.is_dir())
    unexpected_files = [name for name in runtime_files if name not in ALLOWED_RUNTIME_FILES]
    all_text = "\n".join(path.read_text(errors="replace") for path in runtime_workspace.iterdir() if path.is_file())
    forbidden_terms = {
        "annotations_visible": "annotations.jsonl" in all_text,
        "manifest_private_visible": "manifest_private.jsonl" in all_text,
        "manifest_public_visible": "manifest_public.jsonl" in all_text,
        "rendered_ground_truth_visible": "rendered_ground_truth.jsonl" in all_text,
        "original_id_visible": row.get("original_id", "") in all_text if row.get("original_id") else False,
        "source_path_visible": row.get("source_path", "") in all_text if row.get("source_path") else False,
        "pair_id_visible": row.get("pair_id", "") in all_text if row.get("pair_id") else False,
        "correct_or_buggy_label_visible": bool(
            re.search(r"\bbuggy\b|[/\\](?:correct|buggy)[/\\]|_(?:correct|buggy)\b", all_text, re.I)
        ),
        "claude_or_codex_config_visible": ".claude" in all_text or ".codex" in all_text or "CLAUDE.md" in all_text or "AGENTS.md" in all_text,
    }
    root = ROOT.resolve()
    data_path = (ROOT / "data").resolve()
    results_path = (ROOT / "results").resolve()
    return {
        "isolation_mode": "temporary_runtime_outside_repo",
        "actual_cwd": str(runtime_workspace),
        "parent_dir": str(runtime_workspace.parent),
        "workspace_file_listing": runtime_files,
        "workspace_dir_listing": runtime_dirs,
        "workspace_file_hashes": file_hashes(runtime_workspace),
        "unexpected_files": unexpected_files,
        "unexpected_dirs": runtime_dirs,
        "repo_root_visible_by_parent_traversal": path_is_relative_to(root, runtime_workspace)
        or path_is_relative_to(runtime_workspace, root),
        "runtime_inside_repo": path_is_relative_to(runtime_workspace, root),
        "data_visible_by_parent_traversal": path_is_relative_to(data_path, runtime_workspace)
        or path_is_relative_to(runtime_workspace, data_path),
        "results_visible_by_parent_traversal": path_is_relative_to(results_path, runtime_workspace)
        or path_is_relative_to(runtime_workspace, results_path),
        "paper_files_visible": "paper/" in all_text.lower() or "/paper" in all_text.lower(),
        **forbidden_terms,
    }


def runtime_audit_issue(audit: dict[str, Any]) -> str | None:
    blocking_keys = [
        "unexpected_files",
        "unexpected_dirs",
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
    ]
    for key in blocking_keys:
        value = audit.get(key)
        if value:
            return f"runtime isolation violation: {key}={value!r}"
    return None


def create_invocation_runtime(
    run_name: str,
    run_id: str,
    agent_key: str,
    row: dict[str, Any],
    source_workspace: Path,
) -> dict[str, Any]:
    invocation_root = RUNTIME_ROOT / run_name / run_id / agent_key / row["neutral_id"]
    if invocation_root.exists():
        shutil.rmtree(invocation_root)
    runtime_workspace = invocation_root / "workspace"
    runtime_outputs = invocation_root / "outputs"
    runtime_workspace.mkdir(parents=True, exist_ok=True)
    runtime_outputs.mkdir(parents=True, exist_ok=True)
    os.chmod(invocation_root, 0o700)
    os.chmod(runtime_workspace, 0o700)
    os.chmod(runtime_outputs, 0o700)

    files = sorted(path.name for path in source_workspace.iterdir() if path.is_file())
    unexpected = [name for name in files if name not in ALLOWED_RUNTIME_FILES]
    if unexpected:
        raise ValueError(f"{source_workspace}: unexpected source workspace files {unexpected}")
    for name in files:
        shutil.copyfile(source_workspace / name, runtime_workspace / name)

    audit = runtime_visibility_audit(runtime_workspace, row)
    return {
        "invocation_root": invocation_root,
        "workspace": runtime_workspace,
        "outputs": runtime_outputs,
        "audit": audit,
    }


def replace_flag_value(command: list[str], flag: str, value: str) -> list[str]:
    updated = list(command)
    if flag in updated:
        index = updated.index(flag)
        if index + 1 >= len(updated):
            raise ValueError(f"{flag} has no value in command")
        updated[index + 1] = value
    return updated


def empty_invocation_summary(run_name: str, selected_agents: list[str], rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "run": run_name,
        "expected_invocations": len(selected_agents) * len(rows),
        "completed_invocations": 0,
        "failed_invocations": 0,
        "skipped_invocations": 0,
        "by_agent": {
            agent_key: {"expected": len(rows), "completed": 0, "failed": 0, "skipped": 0}
            for agent_key in selected_agents
        },
        "failed_items": [],
    }


def update_invocation_summary(summary: dict[str, Any], meta: dict[str, Any]) -> None:
    agent_key = meta["agent_key"]
    agent_summary = summary["by_agent"][agent_key]
    if meta.get("dry_run"):
        summary["skipped_invocations"] += 1
        agent_summary["skipped"] += 1
        return
    if meta.get("success"):
        summary["completed_invocations"] += 1
        agent_summary["completed"] += 1
        return
    summary["failed_invocations"] += 1
    agent_summary["failed"] += 1
    summary["failed_items"].append(
        {
            "agent_key": agent_key,
            "neutral_id": meta.get("neutral_id"),
            "status": meta.get("status"),
            "reason": meta.get("failure_reason") or meta.get("credential_issue") or "unknown_failure",
        }
    )


def full_success(summary: dict[str, Any]) -> bool:
    return (
        summary.get("expected_invocations") == summary.get("completed_invocations")
        and summary.get("failed_invocations") == 0
        and summary.get("skipped_invocations") == 0
    )


def run_one(
    run_name: str,
    run_id: str,
    agent_key: str,
    config: dict[str, str],
    row: dict[str, Any],
    dry_run: bool,
    reasoning_effort: str | None = None,
    retry: bool = True,
    keep_runtime: bool = False,
    allow_claude_default_home: bool = False,
) -> dict[str, Any]:
    source_workspace = workspace_path(run_name, agent_key, row["neutral_id"])
    raw_dir = run_dir(run_name) / "raw" / config["result_dir"]
    sample_raw_dir = raw_dir / row["neutral_id"]
    sample_raw_dir.mkdir(parents=True, exist_ok=True)
    effective_reasoning_effort = reasoning_effort if config["harness"] == "codex_cli" else None

    stdout_path = raw_dir / f"{row['neutral_id']}.stdout.log"
    stderr_path = raw_dir / f"{row['neutral_id']}.stderr.log"
    events_path = raw_dir / f"{row['neutral_id']}.events.log"
    prediction_path = sample_raw_dir / "prediction.json"
    retry_prediction_path = sample_raw_dir / "prediction.retry.json"
    meta_path = raw_dir / f"{row['neutral_id']}.run_meta.json"

    if not dry_run:
        if sample_raw_dir.exists():
            shutil.rmtree(sample_raw_dir)
        sample_raw_dir.mkdir(parents=True, exist_ok=True)
        stale_paths = [
            stdout_path,
            stderr_path,
            events_path,
            raw_dir / f"{row['neutral_id']}.retry.stdout.log",
            raw_dir / f"{row['neutral_id']}.retry.stderr.log",
            raw_dir / f"{row['neutral_id']}.raw.json",
        ]
        for stale_path in stale_paths:
            if stale_path.exists():
                stale_path.unlink()

    runtime = create_invocation_runtime(run_name, run_id, agent_key, row, source_workspace)
    runtime_workspace = runtime["workspace"]
    runtime_outputs = runtime["outputs"]
    runtime_prediction_path = runtime_outputs / "prediction.json"
    runtime_retry_prediction_path = runtime_outputs / "prediction.retry.json"
    command, extra_env, command_stdin = command_for_agent(
        config,
        runtime_workspace,
        runtime_outputs,
        effective_reasoning_effort,
        allow_claude_default_home=allow_claude_default_home,
    )

    workspace_files = sorted(path.name for path in runtime_workspace.iterdir() if path.is_file())
    command_argv_redacted = redacted_argv(command)
    use_default_claude_code_auth = config["provider"] == "anthropic" and allow_claude_default_home
    credential_problem = credentials_issue(config, extra_env, allow_claude_default_home=allow_claude_default_home)
    base_meta: dict[str, Any] = {
        "run": run_name,
        "run_id": run_id,
        "agent_key": agent_key,
        "agent_config": config,
        "neutral_id": row["neutral_id"],
        "run_date_utc": utc_now(),
        "dry_run": dry_run,
        "source_workspace": str(source_workspace),
        "workspace": str(runtime_workspace),
        "runtime_outputs": str(runtime_outputs),
        "runtime_invocation_root": str(runtime["invocation_root"]),
        "runtime_deleted": False,
        "isolation_mode": "temporary_runtime_outside_repo",
        "runtime_audit": runtime["audit"],
        "actual_cwd": str(runtime_workspace),
        "parent_dir": str(runtime_workspace.parent),
        "workspace_file_listing": workspace_files,
        "workspace_file_hashes": file_hashes(runtime_workspace),
        "prompt_sha256": sha256_file(runtime_workspace / "prompt.txt"),
        "schema_sha256": sha256_file(runtime_workspace / "prediction_schema.json"),
        "snippet_sha256": sha256_file(runtime_workspace / ("snippet.R" if row["language"] == "r" else "snippet.py")),
        "command_template": " ".join(shlex.quote(part) for part in command_argv_redacted),
        "command_argv_redacted": command_argv_redacted,
        "environment_redacted": redacted_env(extra_env),
        "requested_model": config["model"],
        "reasoning_effort": effective_reasoning_effort,
        "provider": config["provider"],
        "harness": config["harness"],
        "gateway": config.get("gateway"),
        "retry_count": 0,
        "home_policy": (
            "default_claude_code_auth"
            if use_default_claude_code_auth
            else
            "temporary_codex_home_auth_only"
            if config["harness"] == "codex_cli"
            else "temporary_claude_home_auth_only"
        ),
        "home_path_type": "default_user_home" if use_default_claude_code_auth else "temporary",
        "copied_user_claude_dir": False,
        "mcp_enabled": False,
        "session_persistence": False,
        "tools_policy": "disabled",
        "project_instructions_visible": False,
        "claude_auth_mode": "default_claude_code" if use_default_claude_code_auth else "bare_env_auth",
        "credential_issue": credential_problem,
        "success": False,
    }

    if dry_run:
        base_meta["exit_code"] = None
        base_meta["status"] = "dry_run"
        base_meta["success"] = True
        base_meta["failure_reason"] = None
        if not keep_runtime and runtime["invocation_root"].exists():
            shutil.rmtree(runtime["invocation_root"])
            base_meta["runtime_deleted"] = True
        write_json(meta_path, base_meta)
        return base_meta

    isolation_problem = runtime_audit_issue(runtime["audit"])
    if credential_problem or isolation_problem:
        base_meta["exit_code"] = None
        base_meta["status"] = "failed"
        base_meta["failure_reason"] = credential_problem or isolation_problem
        if not keep_runtime and runtime["invocation_root"].exists():
            shutil.rmtree(runtime["invocation_root"])
            base_meta["runtime_deleted"] = True
        write_json(meta_path, base_meta)
        return base_meta

    env = base_process_env()
    env.update(extra_env)

    if use_default_claude_code_auth:
        home_context = contextlib.nullcontext(str(REAL_HOME))
    elif config["harness"] == "codex_cli":
        home_context = isolated_codex_home(run_name, run_id, agent_key, row["neutral_id"])
    elif config["harness"] == "claude_code":
        home_context = isolated_claude_home(run_name, run_id, agent_key, row["neutral_id"])
    else:
        home_context = contextlib.nullcontext(None)

    with home_context as agent_home:
        if agent_home:
            env["HOME"] = agent_home
            if config["harness"] == "codex_cli" and not use_default_claude_code_auth:
                env["CODEX_HOME"] = agent_home
            base_meta["home_path"] = agent_home
        result = subprocess.run(
            command,
            cwd=runtime_workspace,
            input=command_stdin,
            capture_output=True,
            text=True,
            env=env,
            check=False,
        )
    stdout_path.write_text(result.stdout)
    stderr_path.write_text(result.stderr)
    events_path.write_text(result.stdout + "\n" + result.stderr)
    base_meta["exit_code"] = result.returncode
    if runtime_prediction_path.exists():
        shutil.copyfile(runtime_prediction_path, prediction_path)
    elif result.stdout.strip():
        prediction_path.write_text(result.stdout)

    parsed_source = prediction_path.read_text(errors="replace") if prediction_path.exists() else result.stdout
    parsed = parse_text(parsed_source)
    if retry and (parsed.get("parse_error") or parsed.get("schema_errors")):
        retry_prompt = (runtime_workspace / "prompt.txt").read_text() + "\nReturn valid JSON only. Do not include markdown.\n"
        retry_command = list(command)
        retry_stdin = command_stdin
        if retry_command[-1] == "-":
            retry_stdin = retry_prompt
        elif config["harness"] == "claude_code":
            retry_stdin = retry_prompt
        if use_default_claude_code_auth:
            retry_home_context = contextlib.nullcontext(str(REAL_HOME))
        elif config["harness"] == "codex_cli":
            retry_command = replace_flag_value(retry_command, "--output-last-message", str(runtime_retry_prediction_path))
            retry_home_context = isolated_codex_home(run_name, run_id, agent_key, row["neutral_id"])
        elif config["harness"] == "claude_code":
            retry_home_context = isolated_claude_home(run_name, run_id, agent_key, row["neutral_id"])
        else:
            retry_home_context = contextlib.nullcontext(None)
        with retry_home_context as retry_agent_home:
            if retry_agent_home:
                env["HOME"] = retry_agent_home
                if config["harness"] == "codex_cli" and not use_default_claude_code_auth:
                    env["CODEX_HOME"] = retry_agent_home
            retry_result = subprocess.run(
                retry_command,
                cwd=runtime_workspace,
                input=retry_stdin,
                capture_output=True,
                text=True,
                env=env,
                check=False,
            )
        (raw_dir / f"{row['neutral_id']}.retry.stdout.log").write_text(retry_result.stdout)
        (raw_dir / f"{row['neutral_id']}.retry.stderr.log").write_text(retry_result.stderr)
        if runtime_retry_prediction_path.exists():
            shutil.copyfile(runtime_retry_prediction_path, retry_prediction_path)
        elif retry_result.stdout.strip():
            retry_prediction_path.write_text(retry_result.stdout)
        retry_parse_source = retry_prediction_path.read_text(errors="replace") if retry_prediction_path.exists() else retry_result.stdout
        parsed = parse_text(retry_parse_source)
        base_meta["retry_count"] = 1
        base_meta["retry_exit_code"] = retry_result.returncode

    parsed["neutral_id"] = row["neutral_id"]
    parsed["parse_status"] = "ok" if not parsed.get("parse_error") and not parsed.get("schema_errors") else "failed"
    write_json(raw_dir / f"{row['neutral_id']}.raw.json", parsed)

    failure_reason = None
    if result.returncode != 0:
        failure_reason = f"nonzero_process_exit_{result.returncode}"
    elif not prediction_path.exists() and not result.stdout.strip():
        failure_reason = "missing_raw_output"
    elif parsed["parse_status"] != "ok":
        failure_reason = "json_parse_or_schema_failure_after_retry"
    else:
        failure_reason = provider_metadata_issue(config, base_meta)

    base_meta["status"] = "completed" if failure_reason is None else "failed"
    base_meta["failure_reason"] = failure_reason
    base_meta["success"] = failure_reason is None
    if not keep_runtime and runtime["invocation_root"].exists():
        shutil.rmtree(runtime["invocation_root"])
        base_meta["runtime_deleted"] = True
    write_json(meta_path, base_meta)
    return base_meta


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", dest="run_name", default=None)
    parser.add_argument("--matrix", type=Path, default=ROOT / "eval" / "agent_matrix.yaml")
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--agent-key", action="append")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-prepare-workspaces", action="store_true")
    parser.add_argument("--keep-runtime-workspaces", action="store_true")
    parser.add_argument("--allow-claude-default-home", action="store_true")
    parser.add_argument("--resume-existing", action="store_true")
    parser.add_argument("--stop-on-failure", action="store_true")
    parser.add_argument("--run-id")
    parser.add_argument(
        "--summary-path",
        type=Path,
        help="Optional run summary path. Defaults to run_metadata/full_invocation_summary.json.",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--reasoning-effort",
        choices=["minimal", "low", "medium", "high"],
        help="Codex CLI reasoning effort override recorded in run metadata.",
    )
    args = parser.parse_args()
    args.run_name = resolve_run(args.run_name)

    run_id = args.run_id or f"{dt.datetime.now(dt.UTC).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}"
    matrix_path = resolve(args.matrix)
    matrix = load_agent_matrix(matrix_path)
    selected_agents = [key for key in matrix if not args.agent_key or key in set(args.agent_key)]
    manifest = read_jsonl(private_manifest_path(args.run_name))
    neutral_ids = smoke_neutral_ids(manifest) if args.mode == "smoke" else [row["neutral_id"] for row in manifest]
    if args.limit:
        neutral_ids = neutral_ids[: args.limit]
    rows = [row for row in manifest if row["neutral_id"] in set(neutral_ids)]

    if not args.no_prepare_workspaces:
        create_workspaces(
            run_name=args.run_name,
            matrix_path=matrix_path,
            schema_path=ROOT / "eval" / "prediction_schema.json",
            neutral_ids=set(neutral_ids),
            agent_keys=set(selected_agents),
            fresh=False,
        )

    run_metadata_root = run_dir(args.run_name) / "run_metadata"
    summary_path = resolve(args.summary_path) if args.summary_path else run_metadata_root / "full_invocation_summary.json"
    summary = empty_invocation_summary(args.run_name, selected_agents, rows)
    summary["run_id"] = run_id
    summary["mode"] = args.mode
    summary["dry_run"] = args.dry_run
    summary["isolation_mode"] = "temporary_runtime_outside_repo"
    summary["runtime_root"] = str(RUNTIME_ROOT / args.run_name / run_id)
    for agent_key in selected_agents:
        config = matrix[agent_key]
        cli_metadata = collect_cli_metadata(config["harness"])
        write_json(
            run_metadata_root / f"{agent_key}.run_config.json",
            {
                "agent_key": agent_key,
                "agent_config": config,
                "run_id": run_id,
                "mode": args.mode,
                "dry_run": args.dry_run,
                "selected_sample_count": len(rows),
                "cli_metadata": cli_metadata,
                "isolation_mode": "temporary_runtime_outside_repo",
                "runtime_root": str(RUNTIME_ROOT / args.run_name / run_id / agent_key),
                "temporary_home_root": str(AGENT_HOME_ROOT / args.run_name / run_id),
                "keep_runtime_workspaces": args.keep_runtime_workspaces,
                "strict_completion": True,
                "allow_claude_default_home": args.allow_claude_default_home,
                "claude_bare_mode_required": config["harness"] == "claude_code"
                and not (config["provider"] == "anthropic" and args.allow_claude_default_home),
                "created_utc": utc_now(),
            },
        )
    write_json(summary_path, summary)

    parsed_by_agent: dict[str, list[dict[str, Any]]] = {agent_key: [] for agent_key in selected_agents}
    parsed_root = run_dir(args.run_name) / "parsed"
    if args.mode == "full" and not args.dry_run and not args.resume_existing:
        for agent_key in selected_agents:
            config = matrix[agent_key]
            raw_agent_dir = run_dir(args.run_name) / "raw" / config["result_dir"]
            if raw_agent_dir.exists():
                shutil.rmtree(raw_agent_dir)
            raw_agent_dir.mkdir(parents=True, exist_ok=True)
            parsed_path = parsed_root / f"{agent_key}.jsonl"
            if parsed_path.exists():
                parsed_path.unlink()
    for agent_key in selected_agents:
        config = matrix[agent_key]
        for index, row in enumerate(rows, start=1):
            raw_prediction = run_dir(args.run_name) / "raw" / config["result_dir"] / f"{row['neutral_id']}.raw.json"
            meta_path = run_dir(args.run_name) / "raw" / config["result_dir"] / f"{row['neutral_id']}.run_meta.json"
            if args.resume_existing and raw_prediction.exists() and meta_path.exists():
                existing_meta = json.loads(meta_path.read_text())
                existing_raw = json.loads(raw_prediction.read_text())
                if existing_meta.get("success") and existing_raw.get("parse_status") == "ok":
                    parsed_by_agent[agent_key].append(existing_raw)
                    write_jsonl(parsed_root / f"{agent_key}.jsonl", parsed_by_agent[agent_key])
                    update_invocation_summary(
                        summary,
                        {
                            "agent_key": agent_key,
                            "neutral_id": row["neutral_id"],
                            "dry_run": False,
                            "success": True,
                            "status": "completed_existing",
                        },
                    )
                    write_json(summary_path, summary)
                    print(
                        f"{utc_now()} {agent_key} {index}/{len(rows)} {row['neutral_id']} completed_existing",
                        flush=True,
                    )
                    continue
            meta = run_one(
                args.run_name,
                run_id,
                agent_key,
                config,
                row,
                dry_run=args.dry_run,
                reasoning_effort=args.reasoning_effort,
                keep_runtime=args.keep_runtime_workspaces,
                allow_claude_default_home=args.allow_claude_default_home,
            )
            update_invocation_summary(summary, meta)
            write_json(summary_path, summary)
            if meta.get("success") and raw_prediction.exists():
                parsed_by_agent[agent_key].append(__import__("json").loads(raw_prediction.read_text()))
                write_jsonl(parsed_root / f"{agent_key}.jsonl", parsed_by_agent[agent_key])
            elif args.dry_run:
                continue
            elif meta.get("success"):
                print(f"Missing parsed prediction for {agent_key} {row['neutral_id']}: {meta.get('status')}")
            print(
                f"{utc_now()} {agent_key} {index}/{len(rows)} {row['neutral_id']} {meta.get('status')} {meta.get('failure_reason') or ''}".rstrip(),
                flush=True,
            )
            if args.stop_on_failure and not args.dry_run and not meta.get("success"):
                for current_agent_key, parsed_rows in parsed_by_agent.items():
                    if parsed_rows:
                        write_jsonl(parsed_root / f"{current_agent_key}.jsonl", parsed_rows)
                write_json(summary_path, summary)
                print(
                    json.dumps(
                        {
                            "status": "stopped_on_failure",
                            "summary_path": str(summary_path),
                            "failed_item": summary["failed_items"][-1] if summary["failed_items"] else None,
                        },
                        indent=2,
                        sort_keys=True,
                    )
                )
                return 1

    for agent_key, parsed_rows in parsed_by_agent.items():
        if parsed_rows:
            write_jsonl(parsed_root / f"{agent_key}.jsonl", parsed_rows)

    write_json(summary_path, summary)
    if args.dry_run:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    if full_success(summary):
        completed = summary["completed_invocations"]
        print(f"Full run completed for {args.run_name}: {completed}/{completed} invocations succeeded.")
        return 0
    print(
        json.dumps(
            {
                "status": "failed" if summary["failed_invocations"] or summary["skipped_invocations"] else "incomplete",
                "summary_path": str(summary_path),
                "expected_invocations": summary["expected_invocations"],
                "completed_invocations": summary["completed_invocations"],
                "failed_invocations": summary["failed_invocations"],
                "skipped_invocations": summary["skipped_invocations"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 1 if summary["failed_invocations"] or summary["skipped_invocations"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
