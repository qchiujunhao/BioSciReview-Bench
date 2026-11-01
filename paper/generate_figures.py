#!/usr/bin/env python3
"""Regenerate the paper's category heatmaps from scored results."""

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.utils import run_dir

ANALYSIS = None
FIGURES = ROOT / "paper" / "figs"
CATEGORIES = ("B1", "B2", "B3", "B4", "B5", "B6")
AGENTS = (
    ("codex_gpt54", "Codex 5.4"),
    ("codex_gpt55", "Codex 5.5"),
    ("claude_code_sonnet46", "Sonnet 4.6"),
    ("claude_code_opus48", "Opus 4.8"),
    ("claude_code_deepseek_flash", "DS-V4 Flash"),
    ("claude_code_deepseek_pro", "DS-V4 Pro"),
)
CATEGORY_LABELS = (
    "B1\nNorm/Scale",
    "B2\nStat. Test",
    "B3\nMult. Test",
    "B4\nSample Mix",
    "B5\nData Leak",
    "B6\nFilter/QC",
)


def read_csv(name):
    with (ANALYSIS / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def save_category_heatmap():
    rows = read_csv("per_category_paper.csv")
    values = {
        (row["bug_category"], row["agent_key"]):
        float(row["full_diagnosis_pm3_success_rate"])
        for row in rows
    }
    expected = {(category, agent) for category in CATEGORIES for agent, _ in AGENTS}
    assert set(values) == expected, "Per-category table is incomplete or has unexpected rows"
    matrix = np.array([[values[(category, agent)] for agent, _ in AGENTS]
                       for category in CATEGORIES])

    fig, ax = plt.subplots(figsize=(4.15, 2.5))
    image = ax.imshow(matrix, cmap="RdYlGn", vmin=0.65, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(AGENTS)), [label for _, label in AGENTS],
                  rotation=35, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(len(CATEGORIES)), CATEGORY_LABELS)
    ax.tick_params(length=4, width=0.8)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            ax.text(column, row, f"{value:.2f}", ha="center", va="center",
                    color="white" if value < 0.78 else "black", fontweight="bold",
                    fontsize=8)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.045, pad=0.025)
    colorbar.set_label("Full Diagnosis@$\\pm$3\nSuccess Rate")
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "fig_category_heatmap.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def save_confusion_heatmap():
    rows = read_csv("category_confusion_normalized.csv")
    agent_keys = tuple(agent for agent, _ in AGENTS)
    assert {row["agent_key"] for row in rows} == set(agent_keys), "Unexpected agent set"
    sparse = {
        (row["agent_key"], row["gold_category"], row["predicted_category"]):
        float(row["proportion_within_gold"])
        for row in rows
    }
    for agent in agent_keys:
        for gold in CATEGORIES:
            total = sum(sparse.get((agent, gold, predicted), 0.0)
                        for predicted in CATEGORIES)
            assert abs(total - 1.0) < 1e-9, f"Confusion row does not sum to one: {agent} {gold}"

    matrix = np.array([
        [sum(sparse.get((agent, gold, predicted), 0.0) for agent in agent_keys)
         / len(agent_keys) for predicted in CATEGORIES]
        for gold in CATEGORIES
    ])
    assert np.allclose(matrix.sum(axis=1), 1.0), "Aggregate confusion rows must sum to one"

    fig, ax = plt.subplots(figsize=(3.3, 2.75))
    image = ax.imshow(matrix, cmap="Blues", vmin=0.0, vmax=1.0, aspect="equal")
    ax.set_xticks(range(len(CATEGORIES)), CATEGORIES)
    ax.set_yticks(range(len(CATEGORIES)), CATEGORIES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Gold")
    ax.tick_params(length=4, width=0.8)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            if value >= 0.005:
                ax.text(column, row, f"{value:.2f}", ha="center", va="center",
                        color="white" if value > 0.5 else "black", fontsize=8)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.05, pad=0.035)
    colorbar.set_label("Avg. Proportion")
    fig.savefig(FIGURES / "fig_category_confusion.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main():
    global ANALYSIS
    parser = argparse.ArgumentParser()
    parser.add_argument("--run")
    args = parser.parse_args()
    ANALYSIS = run_dir(args.run) / "analysis"
    plt.rcParams.update({"font.family": "serif", "font.size": 9})
    save_category_heatmap()
    save_confusion_heatmap()


if __name__ == "__main__":
    main()
