from __future__ import annotations

import json
import re
import shutil
import tempfile
import unittest
from pathlib import Path

from eval.validate_inputs import validate_rendered_dir
from eval.create_sample_workspace import create_workspace, workspace_path
from eval.render_inputs import clean_code_and_map, render_dataset


ROOT = Path(__file__).resolve().parents[1]


class LeakageSafeRenderTests(unittest.TestCase):
    def render_one(self, snippet_id: str, seed: int = 17) -> tuple[Path, dict]:
        (ROOT / "tmp").mkdir(exist_ok=True)
        tmp = tempfile.TemporaryDirectory(dir=ROOT / "tmp")
        self.addCleanup(tmp.cleanup)
        output_dir = Path(tmp.name) / "rendered"
        snippet_list = Path(tmp.name) / "snippets.txt"
        snippet_list.write_text(f"{snippet_id}\n")
        manifest = render_dataset(
            output_dir=output_dir,
            prompt_file=ROOT / "eval" / "prompts" / "baseline.txt",
            snippet_list=snippet_list,
            seed=seed,
            fresh=True,
        )
        return output_dir, manifest[0]

    def test_rendered_prompt_hides_original_id_and_paths(self) -> None:
        output_dir, row = self.render_one("PY_D4_B5_951_buggy")
        prompt = (ROOT / row["rendered_prompt_path"]).read_text()
        self.assertNotIn("PY_D4_B5_951_buggy", prompt)
        self.assertNotIn("data/snippets", prompt)
        self.assertNotRegex(prompt, r"[/\\](correct|buggy)[/\\]")
        self.assertFalse(validate_rendered_dir(output_dir, ROOT / "eval" / "prediction_schema.json"))

    def test_prompt_uses_numbered_rendered_lines_and_b_codes(self) -> None:
        _, row = self.render_one("PY_D4_B5_951_buggy")
        prompt = (ROOT / row["rendered_prompt_path"]).read_text()
        self.assertIn("Report the displayed rendered line number", prompt)
        self.assertRegex(prompt, r"(?m)^\s*1 \| ")
        self.assertIn("B5: Information leakage", prompt)

    def test_clean_code_deletes_maintainer_comments_without_hash_placeholders(self) -> None:
        source = (
            "# TASK: internal task\n"
            "## BUG: internal bug\n"
            "## HINT: internal hint\n"
            "\n"
            "x = y + 1  ## BUG LINE\n"
            "# ordinary scientific note\n"
            "z = x + 2\n"
        )
        code, mapping = clean_code_and_map(source, "python")
        self.assertEqual(code.splitlines()[0], "x = y + 1")
        self.assertNotRegex(code, r"(?m)^\s*#\s*$")
        self.assertNotIn("internal bug", code)
        self.assertNotIn("BUG LINE", code)
        self.assertIn("# ordinary scientific note", code)
        self.assertIsNone(mapping[1])
        self.assertIsNone(mapping[2])
        self.assertIsNone(mapping[3])
        self.assertEqual(mapping[5], 1)

    def test_buggy_record_has_rendered_bug_line(self) -> None:
        _, row = self.render_one("PY_D1_B1_971_buggy")
        self.assertIsInstance(row["rendered_bug_line"], int)
        self.assertIsInstance(row["source_to_rendered_line"][row["source_bug_line"]], int)
        code = (ROOT / row["rendered_code_path"]).read_text()
        self.assertNotIn("BUG LINE", code)
        self.assertNotIn("<-- BUG", code)

    def test_temporary_filename_is_neutral_and_seeded_random(self) -> None:
        _, first = self.render_one("R_D3_B4_951_correct", seed=17)
        _, second = self.render_one("R_D3_B4_951_correct", seed=18)
        self.assertRegex(first["neutral_filename"], r"^sample_\d{5}\.R$")
        self.assertNotIn("R_D3_B4_951_correct", first["neutral_filename"])
        self.assertNotIn("correct", first["neutral_filename"])
        self.assertNotIn("buggy", first["neutral_filename"])
        self.assertNotEqual(first["neutral_filename"], second["neutral_filename"])

    def test_manifest_private_and_public_split(self) -> None:
        output_dir, row = self.render_one("R_D3_B4_951_correct")
        private = json.loads((output_dir / "manifest_private.jsonl").read_text().strip())
        public = json.loads((output_dir / "manifest_public.jsonl").read_text().strip())
        self.assertEqual(private["original_id"], "R_D3_B4_951_correct")
        self.assertNotIn("original_id", public)
        self.assertNotIn("bug_present", public)
        prompt = (ROOT / row["rendered_prompt_path"]).read_text()
        code = (ROOT / row["rendered_code_path"]).read_text()
        self.assertNotIn(private["original_id"], prompt + code)

    def test_one_sample_workspace_contains_only_allowed_files(self) -> None:
        run_name = "test_workspace"
        rendered_dir, row = self.render_one("PY_D4_B5_951_buggy")
        workspace_root = ROOT / "results" / run_name
        self.addCleanup(lambda: shutil.rmtree(workspace_root, ignore_errors=True))
        create_workspace(
            row=row,
            agent_key="codex_gpt54",
            run_name=run_name,
            schema_path=ROOT / "eval" / "prediction_schema.json",
            fresh=True,
        )
        workspace = workspace_path(run_name, "codex_gpt54", row["neutral_id"])
        files = sorted(path.name for path in workspace.iterdir())
        self.assertEqual(files, ["README.txt", "prediction_schema.json", "prompt.txt", "snippet.py"])
        all_text = "\n".join(path.read_text(errors="replace") for path in workspace.iterdir())
        self.assertNotIn("PY_D4_B5_951_buggy", all_text)
        self.assertNotIn("manifest", all_text.lower())
        self.assertNotIn("annotations", all_text.lower())


if __name__ == "__main__":
    unittest.main()
