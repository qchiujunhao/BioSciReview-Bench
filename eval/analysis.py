"""Calculate metrics, tables and confidence intervals from saved predictions."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.completeness import AGENT_KEYS, CompletenessError, validate_all_prediction_files
from eval.score import category_ok, line_exact, line_pm, load_truth, normalize_category, score_records
from eval.utils import resolve_run, ROOT, ground_truth_path, load_agent_matrix, run_dir, write_json, write_jsonl


ORDERED_AGENT_KEYS = AGENT_KEYS

METRIC_COLUMNS = [
    "detection_precision",
    "detection_recall",
    "detection_f1",
    "joint_loc_exact_f1",
    "joint_loc_pm3_f1",
    "full_diagnosis_exact_f1",
    "full_diagnosis_pm3_f1",
    "conditional_loc_pm3_accuracy",
    "category_accuracy",
]

BOOTSTRAP_METRICS = METRIC_COLUMNS

MAIN_COLUMNS = [
    "agent_key",
    "agent_config",
    "agent_harness",
    "provider",
    "backend_model",
    "agent_label",
    "detection_precision",
    "detection_recall",
    "detection_f1",
    "false_positives",
    "false_negatives",
    "joint_loc_exact_f1",
    "joint_loc_pm3_f1",
    "full_diagnosis_exact_f1",
    "full_diagnosis_pm3_f1",
    "conditional_loc_pm3_accuracy",
    "category_accuracy",
]


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def pctile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    pos = (len(ordered) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    w = pos - lo
    return ordered[lo] * (1 - w) + ordered[hi] * w


def fmt3(value: Any) -> str:
    return f"{float(value):.3f}"


def tex_escape(value: Any) -> str:
    return str(value).replace("&", r"\&").replace("_", r"\_")


def agent_label(agent_key: str, config: dict[str, str]) -> str:
    labels = {
        "codex_gpt54": "Codex 5.4 high",
        "codex_gpt55": "Codex 5.5 high",
        "claude_code_sonnet46": "Claude Sonnet 4.6",
        "claude_code_opus48": "Claude Opus 4.8",
        "claude_code_deepseek_flash": "DeepSeek V4 Flash through Claude Code",
        "claude_code_deepseek_pro": "DeepSeek V4 Pro through Claude Code",
    }
    return labels.get(agent_key, f"{config['harness']} + {config['model']}")


def main_row(agent_key: str, config: dict[str, str], metrics: dict[str, Any]) -> dict[str, Any]:
    row = {
        "agent_key": agent_key,
        "agent_config": agent_key,
        "agent_harness": config["harness"],
        "provider": config["provider"],
        "backend_model": config["model"],
        "agent_label": agent_label(agent_key, config),
        "false_positives": metrics["detection_fp"],
        "false_negatives": metrics["detection_fn"],
    }
    row.update({column: metrics[column] for column in METRIC_COLUMNS})
    return row


def group_rows(
    truth: list[dict[str, Any]],
    predictions_by_agent: dict[str, dict[str, dict[str, Any]]],
    matrix: dict[str, dict[str, str]],
    field: str,
    values: list[Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for agent_key in ORDERED_AGENT_KEYS:
        predictions = predictions_by_agent[agent_key]
        for value in values:
            subset = [row for row in truth if row.get(field) == value]
            metrics = score_records(subset, predictions)
            row = main_row(agent_key, matrix[agent_key], metrics)
            row[field] = value
            row["n"] = len(subset)
            rows.append(row)
    return rows


def case_row(agent_key: str, truth: dict[str, Any], pred: dict[str, Any]) -> dict[str, Any]:
    truth_bug = bool(truth.get("bug_present"))
    pred_bug = bool(pred.get("bug_present"))
    gold_line = truth.get("rendered_bug_line")
    pred_line = pred.get("bug_line") if isinstance(pred.get("bug_line"), int) else None
    delta = pred_line - gold_line if isinstance(pred_line, int) and isinstance(gold_line, int) else None
    exact = truth_bug and pred_bug and line_exact(truth, pred)
    within = truth_bug and pred_bug and line_pm(truth, pred, 3)
    cat_ok = truth_bug and pred_bug and category_ok(truth, pred)

    if truth_bug and pred_bug:
        detection = "TP"
    elif truth_bug:
        detection = "FN"
    elif pred_bug:
        detection = "FP"
    else:
        detection = "TN"

    if not truth_bug:
        failure_type = "false_positive" if pred_bug else "true_negative"
    elif not pred_bug:
        failure_type = "false_negative"
    elif pred_line is None:
        failure_type = "detected_but_no_line"
    elif not within:
        failure_type = "localized_outside_pm3"
    elif not cat_ok:
        failure_type = "within_pm3_wrong_category"
    elif not exact:
        failure_type = "within_pm3_correct_category_not_exact"
    else:
        failure_type = "full_exact_success"

    return {
        "agent_key": agent_key,
        "neutral_id": truth["sample_id"],
        "pair_id": truth["pair_id"],
        "language": truth.get("language"),
        "domain": truth.get("domain"),
        "truth_bug_present": truth_bug,
        "pred_bug_present": pred_bug,
        "detection_outcome": detection,
        "gold_rendered_bug_line": gold_line,
        "predicted_bug_line": pred_line,
        "line_exact": exact,
        "line_within_pm3": within,
        "line_delta": delta,
        "gold_bug_category": truth.get("bug_category"),
        "predicted_bug_category": normalize_category(pred.get("bug_category")),
        "category_correct": cat_ok,
        "joint_loc_exact_success": exact,
        "joint_loc_pm3_success": within,
        "full_diag_exact_success": exact and cat_ok,
        "full_diag_pm3_success": within and cat_ok,
        "failure_type": failure_type,
        "rationale": pred.get("rationale"),
    }


def diagnostic_rows(
    truth: list[dict[str, Any]],
    predictions_by_agent: dict[str, dict[str, dict[str, Any]]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for agent_key in ORDERED_AGENT_KEYS:
        predictions = predictions_by_agent[agent_key]
        for truth_row in truth:
            rows.append(case_row(agent_key, truth_row, predictions.get(truth_row["sample_id"], {})))
    return rows


def taxonomy_rows(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for agent_key in ORDERED_AGENT_KEYS:
        model_cases = [row for row in cases if row["agent_key"] == agent_key]
        counts = Counter(row["failure_type"] for row in model_cases)
        rows.append(
            {
                "agent_key": agent_key,
                "n_total": len(model_cases),
                "n_correct": sum(1 for row in model_cases if not row["truth_bug_present"]),
                "n_buggy": sum(1 for row in model_cases if row["truth_bug_present"]),
                "n_false_positive": counts["false_positive"],
                "n_false_negative": counts["false_negative"],
                "n_detected_but_no_line": counts["detected_but_no_line"],
                "n_localized_outside_pm3": counts["localized_outside_pm3"],
                "n_within_pm3_wrong_category": counts["within_pm3_wrong_category"],
                "n_within_pm3_correct_category_not_exact": counts["within_pm3_correct_category_not_exact"],
                "n_full_exact_success": counts["full_exact_success"],
                "n_joint_loc_pm3_success": sum(1 for row in model_cases if row["joint_loc_pm3_success"]),
                "n_full_diag_pm3_success": sum(1 for row in model_cases if row["full_diag_pm3_success"]),
            }
        )
    return rows


def category_confusion_rows(cases: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    counts: Counter[tuple[str, str, str]] = Counter()
    gold_totals: Counter[tuple[str, str]] = Counter()
    for row in cases:
        if not row["truth_bug_present"] or not row["pred_bug_present"]:
            continue
        gold = row["gold_bug_category"] or "missing"
        pred = row["predicted_bug_category"] or "missing"
        counts[(row["agent_key"], gold, pred)] += 1
        gold_totals[(row["agent_key"], gold)] += 1
    raw = [
        {"agent_key": agent, "gold_category": gold, "predicted_category": pred, "count": count}
        for (agent, gold, pred), count in sorted(counts.items())
    ]
    norm = [
        {
            **row,
            "gold_total": gold_totals[(row["agent_key"], row["gold_category"])],
            "proportion_within_gold": row["count"] / gold_totals[(row["agent_key"], row["gold_category"])],
        }
        for row in raw
    ]
    return raw, norm


def exact_vs_pm3_rows(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    columns = [
        "agent_key",
        "neutral_id",
        "pair_id",
        "language",
        "domain",
        "gold_bug_category",
        "predicted_bug_category",
        "gold_rendered_bug_line",
        "predicted_bug_line",
        "line_delta",
        "category_correct",
        "rationale",
    ]
    return [
        {column: row.get(column) for column in columns}
        for row in cases
        if row["truth_bug_present"] and row["pred_bug_present"] and row["line_within_pm3"] and not row["line_exact"]
    ]


def per_category_paper_rows(
    truth: list[dict[str, Any]],
    cases: list[dict[str, Any]],
    matrix: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    categories = ["B1", "B2", "B3", "B4", "B5", "B6"]
    buggy_by_category = {
        category: [row for row in truth if row.get("bug_present") and row.get("bug_category") == category]
        for category in categories
    }
    cases_by_agent_category: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in cases:
        if not row["truth_bug_present"]:
            continue
        cases_by_agent_category.setdefault((row["agent_key"], row["gold_bug_category"]), []).append(row)

    for category in categories:
        n_buggy = len(buggy_by_category[category])
        for agent_key in ORDERED_AGENT_KEYS:
            agent_cases = cases_by_agent_category.get((agent_key, category), [])
            detected_with_category = [
                row for row in agent_cases if row["pred_bug_present"] and row["predicted_bug_category"] is not None
            ]
            rows.append(
                {
                    "bug_category": category,
                    "agent_key": agent_key,
                    "agent_label": agent_label(agent_key, matrix[agent_key]),
                    "n_buggy": n_buggy,
                    "detection_recall": sum(row["detection_outcome"] == "TP" for row in agent_cases) / n_buggy
                    if n_buggy
                    else 0.0,
                    "joint_loc_pm3_success_rate": sum(row["joint_loc_pm3_success"] for row in agent_cases) / n_buggy
                    if n_buggy
                    else 0.0,
                    "full_diagnosis_pm3_success_rate": sum(row["full_diag_pm3_success"] for row in agent_cases) / n_buggy
                    if n_buggy
                    else 0.0,
                    "full_diagnosis_exact_success_rate": sum(row["full_diag_exact_success"] for row in agent_cases)
                    / n_buggy
                    if n_buggy
                    else 0.0,
                    "category_accuracy": sum(row["category_correct"] for row in detected_with_category)
                    / len(detected_with_category)
                    if detected_with_category
                    else 0.0,
                }
            )
    return rows


def overlap_and_oracle_rows(truth: list[dict[str, Any]], cases: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_case = {(row["agent_key"], row["neutral_id"]): row for row in cases}
    overlap: list[dict[str, Any]] = []
    metrics = [
        ("detection", "detection_success", lambda row: row["detection_outcome"] == "TP"),
        ("joint_loc_pm3", "joint_loc_pm3_success", lambda row: row["joint_loc_pm3_success"]),
        ("full_diagnosis_pm3", "full_diag_pm3_success", lambda row: row["full_diag_pm3_success"]),
        ("full_exact", "full_diag_exact_success", lambda row: row["full_diag_exact_success"]),
    ]
    oracle_counts = Counter()
    all_counts = Counter()
    total = 0
    for truth_row in truth:
        if not truth_row["bug_present"]:
            continue
        total += 1
        row: dict[str, Any] = {
            "neutral_id": truth_row["sample_id"],
            "pair_id": truth_row["pair_id"],
            "language": truth_row.get("language"),
            "domain": truth_row.get("domain"),
            "gold_category": truth_row.get("bug_category"),
        }
        for metric, _column, ok in metrics:
            succeeded = [agent for agent in ORDERED_AGENT_KEYS if ok(by_case[(agent, truth_row["sample_id"])])]
            failed = [agent for agent in ORDERED_AGENT_KEYS if agent not in succeeded]
            row[f"{metric}_succeeded_agents"] = ";".join(succeeded)
            row[f"{metric}_failed_agents"] = ";".join(failed)
            row[f"{metric}_by_all_agents"] = len(succeeded) == len(ORDERED_AGENT_KEYS)
            row[f"{metric}_by_at_least_one_agent"] = bool(succeeded)
            oracle_counts[metric] += bool(succeeded)
            all_counts[metric] += len(succeeded) == len(ORDERED_AGENT_KEYS)
        row["detected_by_all_agents"] = row["detection_by_all_agents"]
        row["missed_by_all_agents"] = not row["detection_by_at_least_one_agent"]
        row["localized_pm3_by_all_agents"] = row["joint_loc_pm3_by_all_agents"]
        row["localized_pm3_by_at_least_one_agent"] = row["joint_loc_pm3_by_at_least_one_agent"]
        row["full_diagnosis_pm3_by_all_agents"] = row["full_diagnosis_pm3_by_all_agents"]
        row["full_diagnosis_pm3_by_at_least_one_agent"] = row["full_diagnosis_pm3_by_at_least_one_agent"]
        row["full_exact_diagnosis_by_all_agents"] = row["full_exact_by_all_agents"]
        row["full_exact_diagnosis_by_at_least_one_agent"] = row["full_exact_by_at_least_one_agent"]
        overlap.append(row)

    oracle = [
        {
            "metric": metric,
            "total_buggy_cases": total,
            "oracle_any_agent_successes": oracle_counts[metric],
            "oracle_upper_bound_recall": oracle_counts[metric] / total if total else 0.0,
            "all_agents_successes": all_counts[metric],
            "all_agents_success_rate": all_counts[metric] / total if total else 0.0,
            "label": "oracle upper bound, not a deployable ensemble",
        }
        for metric, _column, _ok in metrics
    ]
    return overlap, oracle


def error_overlap_paper_rows(overlap: list[dict[str, Any]]) -> list[dict[str, Any]]:
    columns = [
        "neutral_id",
        "pair_id",
        "language",
        "domain",
        "gold_category",
        "detection_succeeded_agents",
        "detection_failed_agents",
        "joint_loc_pm3_succeeded_agents",
        "joint_loc_pm3_failed_agents",
        "full_diagnosis_pm3_succeeded_agents",
        "full_diagnosis_pm3_failed_agents",
        "full_diagnosis_exact_succeeded_agents",
        "full_diagnosis_exact_failed_agents",
    ]
    rows: list[dict[str, Any]] = []
    for row in overlap:
        paper_row = {column: row.get(column, "") for column in columns}
        paper_row["full_diagnosis_exact_succeeded_agents"] = row.get("full_exact_succeeded_agents", "")
        paper_row["full_diagnosis_exact_failed_agents"] = row.get("full_exact_failed_agents", "")
        rows.append(paper_row)
    return rows


def bootstrap_agent(
    truth: list[dict[str, Any]],
    predictions: dict[str, dict[str, Any]],
    samples: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    rng = random.Random(seed)
    by_pair: dict[str, list[dict[str, Any]]] = {}
    for row in truth:
        by_pair.setdefault(row["pair_id"], []).append(row)
    pair_ids = sorted(by_pair)
    point = score_records(truth, predictions)
    values = {metric: [] for metric in BOOTSTRAP_METRICS}
    for _ in range(samples):
        sampled_truth: list[dict[str, Any]] = []
        for _ in pair_ids:
            sampled_truth.extend(by_pair[rng.choice(pair_ids)])
        metrics = score_records(sampled_truth, predictions)
        for metric in BOOTSTRAP_METRICS:
            values[metric].append(float(metrics[metric]))
    return {
        metric: {
            "point_estimate": float(point[metric]),
            "lower_95": pctile(dist, 0.025),
            "upper_95": pctile(dist, 0.975),
        }
        for metric, dist in values.items()
    }


def write_main_table_tex(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Model & Detect F1 & FP/FN & Joint Loc@$\pm$3 F1 & Full Diagnosis@$\pm$3 F1 & Full Diagnosis@Exact F1 \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{tex_escape(row['agent_label'])} & {fmt3(row['detection_f1'])} & "
            f"{row['false_positives']}/{row['false_negatives']} & {fmt3(row['joint_loc_pm3_f1'])} & "
            f"{fmt3(row['full_diagnosis_pm3_f1'])} & {fmt3(row['full_diagnosis_exact_f1'])} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    path.write_text("\n".join(lines))


def write_final_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    best_detection = max(rows, key=lambda row: float(row["detection_f1"]))
    best_joint = max(rows, key=lambda row: float(row["joint_loc_pm3_f1"]))
    best_pm3 = max(rows, key=lambda row: float(row["full_diagnosis_pm3_f1"]))
    best_exact = max(rows, key=lambda row: float(row["full_diagnosis_exact_f1"]))

    def metric_range(metric: str) -> str:
        values = [float(row[metric]) for row in rows]
        return f"{min(values):.3f}-{max(values):.3f}"

    lines = [
        "# BioSciReview-Bench Results",
        "",
        "Before evaluation, we render each snippet into an annotation-free view. Maintainer annotations and construction metadata are removed, ordinary code comments and docstrings are preserved, and gold root-cause lines are remapped to the displayed rendered line numbers.",
        "",
        "Each agent invocation receives a single neutral workspace containing only the current snippet, prompt, and output schema. Workspaces use neutral filenames and do not contain annotations, pair metadata, other snippets, prior results, or source repository files.",
        "",
        "| Model | Detect P/R/F1 | FP/FN | Joint Loc@±3 F1 | Full Diagnosis@±3 F1 | Full Diagnosis@Exact F1 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['agent_label']} | {fmt3(row['detection_precision'])} / {fmt3(row['detection_recall'])} / {fmt3(row['detection_f1'])} "
            f"| {row['false_positives']} / {row['false_negatives']} | {fmt3(row['joint_loc_pm3_f1'])} "
            f"| {fmt3(row['full_diagnosis_pm3_f1'])} | {fmt3(row['full_diagnosis_exact_f1'])} |"
        )
    lines.extend(
        [
            "",
            f"Best detection result: {best_detection['agent_label']}, Detection F1 = {fmt3(best_detection['detection_f1'])}.",
            f"Best Joint Loc@±3: {best_joint['agent_label']}, Joint Loc@±3 F1 = {fmt3(best_joint['joint_loc_pm3_f1'])}.",
            f"Best Full Diagnosis@±3: {best_pm3['agent_label']}, Full Diagnosis@±3 F1 = {fmt3(best_pm3['full_diagnosis_pm3_f1'])}.",
            f"Best Full Diagnosis@Exact: {best_exact['agent_label']}, Full Diagnosis@Exact F1 = {fmt3(best_exact['full_diagnosis_exact_f1'])}.",
            "",
            f"Detection F1 range: {metric_range('detection_f1')}.",
            f"Joint Loc@±3 F1 range: {metric_range('joint_loc_pm3_f1')}.",
            f"Full Diagnosis@±3 F1 range: {metric_range('full_diagnosis_pm3_f1')}.",
            f"Full Diagnosis@Exact F1 range: {metric_range('full_diagnosis_exact_f1')}.",
            "",
            "Joint Loc@±3 F1 detects a buggy snippet and identifies a line within ±3 displayed rendered lines of the gold root-cause line. It does not require category correctness.",
            "Full Diagnosis@±3 F1 requires detection, line within ±3 displayed rendered lines, and correct bug category.",
            "Full Diagnosis@Exact F1 requires detection, exact displayed rendered root-cause line, and correct bug category.",
            "",
            "Small ranking differences should be interpreted with the pair-level bootstrap confidence intervals.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def generate(
    run_name: str,
    matrix_path: Path,
    bootstrap_samples: int = 10000,
    seed: int = 20260613,
) -> dict[str, Any]:
    root = run_dir(run_name)
    analysis_dir = root / "analysis"
    scores_dir = root / "scores"
    matrix = load_agent_matrix(matrix_path)
    truth = load_truth(ground_truth_path(run_name))
    completeness = validate_all_prediction_files(run_name, matrix_path, allow_partial=False)

    predictions_by_agent: dict[str, dict[str, dict[str, Any]]] = {}
    main_rows: list[dict[str, Any]] = []
    for agent_key in ORDERED_AGENT_KEYS:
        validation = completeness["agents"][agent_key]
        if validation["issues"]:
            raise CompletenessError("\n".join(validation["issues"]))
        predictions = validation["predictions"]
        predictions_by_agent[agent_key] = predictions
        metrics = score_records(truth, predictions, prediction_inventory=validation["inventory"])
        write_json(scores_dir / f"{agent_key}.metrics.json", metrics)
        main_rows.append(main_row(agent_key, matrix[agent_key], metrics))

    write_csv(analysis_dir / "main_table.csv", main_rows, MAIN_COLUMNS)
    write_main_table_tex(analysis_dir / "main_table.tex", main_rows)

    domains = sorted({row["domain"] for row in truth})
    languages = sorted({row["language"] for row in truth})
    categories = ["B1", "B2", "B3", "B4", "B5", "B6"]
    group_columns = ["agent_key", "agent_config", "agent_harness", "provider", "backend_model", "agent_label", "n", *METRIC_COLUMNS]
    write_csv(analysis_dir / "per_domain.csv", group_rows(truth, predictions_by_agent, matrix, "domain", domains), ["domain", *group_columns])
    write_csv(analysis_dir / "per_language.csv", group_rows(truth, predictions_by_agent, matrix, "language", languages), ["language", *group_columns])
    write_csv(analysis_dir / "per_category.csv", group_rows(truth, predictions_by_agent, matrix, "bug_category", categories), ["bug_category", *group_columns])

    cases = diagnostic_rows(truth, predictions_by_agent)
    write_jsonl(analysis_dir / "error_cases_by_model.jsonl", cases)
    write_csv(
        analysis_dir / "per_category_paper.csv",
        per_category_paper_rows(truth, cases, matrix),
        [
            "bug_category",
            "agent_key",
            "agent_label",
            "n_buggy",
            "detection_recall",
            "joint_loc_pm3_success_rate",
            "full_diagnosis_pm3_success_rate",
            "full_diagnosis_exact_success_rate",
            "category_accuracy",
        ],
    )
    write_csv(
        analysis_dir / "error_taxonomy_by_model.csv",
        taxonomy_rows(cases),
        [
            "agent_key",
            "n_total",
            "n_correct",
            "n_buggy",
            "n_false_positive",
            "n_false_negative",
            "n_detected_but_no_line",
            "n_localized_outside_pm3",
            "n_within_pm3_wrong_category",
            "n_within_pm3_correct_category_not_exact",
            "n_full_exact_success",
            "n_joint_loc_pm3_success",
            "n_full_diag_pm3_success",
        ],
    )
    confusion, confusion_norm = category_confusion_rows(cases)
    write_csv(analysis_dir / "category_confusion.csv", confusion, ["agent_key", "gold_category", "predicted_category", "count"])
    write_csv(
        analysis_dir / "category_confusion_normalized.csv",
        confusion_norm,
        ["agent_key", "gold_category", "predicted_category", "count", "gold_total", "proportion_within_gold"],
    )
    write_csv(
        analysis_dir / "exact_vs_pm3_cases.csv",
        exact_vs_pm3_rows(cases),
        [
            "agent_key",
            "neutral_id",
            "pair_id",
            "language",
            "domain",
            "gold_bug_category",
            "predicted_bug_category",
            "gold_rendered_bug_line",
            "predicted_bug_line",
            "line_delta",
            "category_correct",
            "rationale",
        ],
    )

    overlap, oracle = overlap_and_oracle_rows(truth, cases)
    overlap_columns = sorted({key for row in overlap for key in row})
    write_csv(analysis_dir / "error_overlap.csv", overlap, overlap_columns)
    write_csv(
        analysis_dir / "error_overlap_paper.csv",
        error_overlap_paper_rows(overlap),
        [
            "neutral_id",
            "pair_id",
            "language",
            "domain",
            "gold_category",
            "detection_succeeded_agents",
            "detection_failed_agents",
            "joint_loc_pm3_succeeded_agents",
            "joint_loc_pm3_failed_agents",
            "full_diagnosis_pm3_succeeded_agents",
            "full_diagnosis_pm3_failed_agents",
            "full_diagnosis_exact_succeeded_agents",
            "full_diagnosis_exact_failed_agents",
        ],
    )
    write_csv(
        analysis_dir / "oracle_upper_bound.csv",
        oracle,
        [
            "metric",
            "total_buggy_cases",
            "oracle_any_agent_successes",
            "oracle_upper_bound_recall",
            "all_agents_successes",
            "all_agents_success_rate",
            "label",
        ],
    )

    bootstrap_json: dict[str, Any] = {
        "metadata": {
            "bootstrap_unit": "pair_id",
            "n_pairs_per_resample": len({row["pair_id"] for row in truth}),
            "samples": bootstrap_samples,
            "seed": seed,
        },
        "agents": {},
    }
    bootstrap_rows: list[dict[str, Any]] = []
    for offset, agent_key in enumerate(ORDERED_AGENT_KEYS):
        agent_seed = seed + offset
        boot = bootstrap_agent(truth, predictions_by_agent[agent_key], bootstrap_samples, agent_seed)
        bootstrap_json["agents"][agent_key] = {"seed": agent_seed, "metrics": boot}
        for metric, values in boot.items():
            bootstrap_rows.append(
                {
                    "agent_key": agent_key,
                    "metric": metric,
                    "point_estimate": values["point_estimate"],
                    "lower_95": values["lower_95"],
                    "upper_95": values["upper_95"],
                    "bootstrap_unit": "pair_id",
                    "n_bootstrap": bootstrap_samples,
                    "seed": agent_seed,
                }
            )
    write_json(analysis_dir / "bootstrap_ci.json", bootstrap_json)
    write_csv(
        analysis_dir / "bootstrap_ci.csv",
        bootstrap_rows,
        ["agent_key", "metric", "point_estimate", "lower_95", "upper_95", "bootstrap_unit", "n_bootstrap", "seed"],
    )

    write_final_summary(analysis_dir / "final_results_summary.md", main_rows)
    write_json(
        analysis_dir / "generation_metadata.json",
        {
            "run": run_name,
            "source_root": root.relative_to(ROOT).as_posix(),
            "allow_partial": False,
            "bootstrap_samples": bootstrap_samples,
            "bootstrap_seed": seed,
            "bootstrap_base_seed": seed,
            "bootstrap_seed_policy": "base_seed + agent_index",
            "agents": list(ORDERED_AGENT_KEYS),
        },
    )

    return {
        "status": "complete",
        "run": run_name,
        "agents_scored": list(ORDERED_AGENT_KEYS),
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": seed,
        "allow_partial": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", dest="run_name", default=None)
    parser.add_argument("--matrix", type=Path, default=ROOT / "eval" / "agent_matrix.yaml")
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260613)
    args = parser.parse_args()
    args.run_name = resolve_run(args.run_name)

    try:
        result = generate(args.run_name, args.matrix, args.bootstrap_samples, args.seed)
    except CompletenessError as exc:
        print(json.dumps({"status": "failed", "reason": str(exc), "run": args.run_name}, indent=2, sort_keys=True))
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
