"""Create isolated one-snippet workspaces for agent runs."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.utils import (
    resolve_run,
    ROOT,
    clean_dir,
    load_agent_matrix,
    private_manifest_path,
    read_jsonl,
    resolve,
    sha256_file,
    run_dir,
    write_json,
)


ALLOWED_FILES = {"prompt.txt", "snippet.py", "snippet.R", "prediction_schema.json", "README.txt"}


def workspace_path(run_name: str, agent_key: str, neutral_id: str) -> Path:
    return run_dir(run_name) / "workspaces" / agent_key / neutral_id


def create_workspace(
    row: dict,
    agent_key: str,
    run_name: str,
    schema_path: Path,
    fresh: bool = False,
) -> dict:
    workspace = workspace_path(run_name, agent_key, row["neutral_id"])
    clean_dir(workspace, fresh=fresh)

    for child in workspace.iterdir():
        if child.name not in ALLOWED_FILES:
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()

    prompt_src = resolve(row["rendered_prompt_path"])
    code_src = resolve(row["rendered_code_path"])
    extension = ".R" if row["language"] == "r" else ".py"
    code_dst = workspace / f"snippet{extension}"
    shutil.copyfile(prompt_src, workspace / "prompt.txt")
    shutil.copyfile(code_src, code_dst)
    shutil.copyfile(schema_path, workspace / "prediction_schema.json")
    (workspace / "README.txt").write_text(
        "BioSciReview-Bench isolated sample workspace.\n"
        "Review prompt.txt and snippet.py or snippet.R, then return one JSON prediction.\n"
        "This workspace intentionally contains only the current sample files.\n"
    )

    files = sorted(path.name for path in workspace.iterdir() if path.is_file())
    unexpected = [name for name in files if name not in ALLOWED_FILES]
    if unexpected:
        raise ValueError(f"{workspace}: unexpected files after creation: {unexpected}")

    return {
        "agent_key": agent_key,
        "neutral_id": row["neutral_id"],
        "workspace": str(workspace),
        "files": files,
        "prompt_sha256": sha256_file(workspace / "prompt.txt"),
        "schema_sha256": sha256_file(workspace / "prediction_schema.json"),
        "snippet_sha256": sha256_file(code_dst),
    }


def create_workspaces(
    run_name: str,
    matrix_path: Path,
    schema_path: Path,
    neutral_ids: set[str] | None = None,
    agent_keys: set[str] | None = None,
    fresh: bool = False,
) -> list[dict]:
    matrix = load_agent_matrix(matrix_path)
    manifest = read_jsonl(private_manifest_path(run_name))
    rows = [row for row in manifest if neutral_ids is None or row["neutral_id"] in neutral_ids]
    selected_agents = [key for key in matrix if agent_keys is None or key in agent_keys]
    records: list[dict] = []
    for agent_key in selected_agents:
        for row in rows:
            records.append(create_workspace(row, agent_key, run_name, schema_path, fresh=fresh))
    return records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", dest="run_name", default=None)
    parser.add_argument("--matrix", type=Path, default=ROOT / "eval" / "agent_matrix.yaml")
    parser.add_argument("--schema", type=Path, default=ROOT / "eval" / "prediction_schema.json")
    parser.add_argument("--neutral-id", action="append")
    parser.add_argument("--agent-key", action="append")
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--write-index", action="store_true")
    args = parser.parse_args()
    args.run_name = resolve_run(args.run_name)

    matrix_path = resolve(args.matrix)
    schema_path = resolve(args.schema)
    records = create_workspaces(
        run_name=args.run_name,
        matrix_path=matrix_path,
        schema_path=schema_path,
        neutral_ids=set(args.neutral_id) if args.neutral_id else None,
        agent_keys=set(args.agent_key) if args.agent_key else None,
        fresh=args.fresh,
    )
    if args.write_index:
        write_json(run_dir(args.run_name) / "workspaces" / "workspace_index.json", records)
    print(f"Created {len(records)} isolated workspaces for {args.run_name}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
