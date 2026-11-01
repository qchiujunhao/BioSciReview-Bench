from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from eval.analysis import write_final_summary
from eval.utils import ROOT, resolve_run


class AnalysisTests(unittest.TestCase):
    def setUp(self) -> None:
        (ROOT / "tmp").mkdir(exist_ok=True)
        temporary = tempfile.TemporaryDirectory(dir=ROOT / "tmp")
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)

    def test_summary_accepts_changed_rankings(self) -> None:
        rows = [
            {
                "agent_label": "Codex 5.4 high",
                "detection_precision": 0.9, "detection_recall": 0.9, "detection_f1": 0.9,
                "false_positives": 10, "false_negatives": 10,
                "joint_loc_pm3_f1": 0.6, "full_diagnosis_pm3_f1": 0.5,
                "full_diagnosis_exact_f1": 0.4,
            },
            {
                "agent_label": "Claude Sonnet 4.6",
                "detection_precision": 0.8, "detection_recall": 0.8, "detection_f1": 0.8,
                "false_positives": 20, "false_negatives": 20,
                "joint_loc_pm3_f1": 0.75, "full_diagnosis_pm3_f1": 0.7,
                "full_diagnosis_exact_f1": 0.65,
            },
        ]
        output = self.directory / "summary.md"
        write_final_summary(output, rows)
        summary = output.read_text()
        self.assertIn("Best detection result: Codex 5.4 high, Detection F1 = 0.900", summary)
        self.assertIn("Best Full Diagnosis@Exact: Claude Sonnet 4.6", summary)

    def test_selects_single_extracted_run_without_version_label(self) -> None:
        truth = self.directory / "published-results" / "rendered" / "rendered_ground_truth.jsonl"
        truth.parent.mkdir(parents=True)
        truth.touch()
        with patch("eval.utils.RESULTS_ROOT", self.directory):
            self.assertEqual(resolve_run(), "published-results")

    def test_multiple_runs_require_explicit_selection(self) -> None:
        for name in ("first", "second"):
            truth = self.directory / name / "rendered" / "rendered_ground_truth.jsonl"
            truth.parent.mkdir(parents=True)
            truth.touch()
        with patch("eval.utils.RESULTS_ROOT", self.directory):
            with self.assertRaisesRegex(ValueError, "--run"):
                resolve_run()
            self.assertEqual(resolve_run("second"), "second")


if __name__ == "__main__":
    unittest.main()
