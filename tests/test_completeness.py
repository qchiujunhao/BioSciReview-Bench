from __future__ import annotations

import unittest

from eval.completeness import CompletenessError, validate_prediction_rows


class CompletenessTests(unittest.TestCase):
    def rows(self, ids: list[str]) -> list[dict]:
        return [
            {
                "neutral_id": neutral_id,
                "bug_present": False,
                "bug_line": None,
                "bug_category": None,
                "confidence": 0.5,
                "rationale": "no bug",
                "parse_status": "ok",
            }
            for neutral_id in ids
        ]

    def test_complete_prediction_rows_pass(self) -> None:
        expected = {"sample_00001", "sample_00002"}
        result = validate_prediction_rows("agent", self.rows(sorted(expected)), expected, expected_count=2)
        self.assertTrue(result["complete"])
        self.assertEqual(result["inventory"]["n_missing_predictions"], 0)
        self.assertEqual(result["inventory"]["n_extra_predictions"], 0)
        self.assertEqual(result["inventory"]["n_duplicate_predictions"], 0)

    def test_duplicate_prediction_rows_fail(self) -> None:
        expected = {"sample_00001", "sample_00002"}
        with self.assertRaises(CompletenessError):
            validate_prediction_rows("agent", self.rows(["sample_00001", "sample_00001"]), expected, expected_count=2)

    def test_missing_and_extra_prediction_rows_fail(self) -> None:
        expected = {"sample_00001", "sample_00002"}
        with self.assertRaises(CompletenessError):
            validate_prediction_rows("agent", self.rows(["sample_00001", "sample_99999"]), expected, expected_count=2)

    def test_malformed_prediction_id_fails(self) -> None:
        expected = {"sample_00001"}
        with self.assertRaises(CompletenessError):
            validate_prediction_rows("agent", self.rows(["PY_D1_B1_001_buggy"]), expected, expected_count=1)


if __name__ == "__main__":
    unittest.main()
