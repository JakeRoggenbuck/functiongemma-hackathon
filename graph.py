#!/usr/bin/env python3
"""Plot comparisons across 2+ benchmark runs stored in results.txt."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


RUN_NAME_RE = re.compile(r"^Name:\s*(.+?)\s*$")
ROW_RE = re.compile(
    r"^\s*(\d+)\s*,\s*([a-zA-Z]+)\s*,\s*([a-zA-Z0-9_]+)\s*,\s*([0-9]*\.?[0-9]+)\s*,\s*([0-9]*\.?[0-9]+)\s*,\s*(.+?)\s*$"
)


def parse_results(path: Path) -> dict[str, list[dict[str, object]]]:
    runs: dict[str, list[dict[str, object]]] = {}
    current_run: str | None = None

    for lineno, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue

        name_match = RUN_NAME_RE.match(line)
        if name_match:
            current_run = name_match.group(1)
            runs.setdefault(current_run, [])
            continue

        row_match = ROW_RE.match(line)
        if row_match:
            if current_run is None:
                raise ValueError(f"Found data row before any run name (line {lineno}).")

            idx, difficulty, task, latency_ms, score, route = row_match.groups()
            runs[current_run].append(
                {
                    "index": int(idx),
                    "difficulty": difficulty.lower(),
                    "task": task,
                    "latency_ms": float(latency_ms),
                    "score": float(score),
                    "route": route.strip(),
                }
            )
            continue

        raise ValueError(f"Unrecognized line format at {lineno}: {raw_line!r}")

    if len(runs) < 2:
        raise ValueError("Need at least two runs in the input file.")
    return runs


def average(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def plot_comparison(runs: dict[str, list[dict[str, object]]], output_path: Path, show: bool) -> None:
    run_names = list(runs.keys())
    colors = plt.cm.tab10(np.linspace(0, 1, len(run_names)))

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    ax_latency, ax_score = axes[0]
    ax_diff_latency, ax_diff_score = axes[1]

    difficulties = ["easy", "medium", "hard"]
    x_positions = np.arange(len(difficulties))
    width = 0.8 / max(len(run_names), 1)

    for i, run_name in enumerate(run_names):
        rows = sorted(runs[run_name], key=lambda r: int(r["index"]))
        x = [int(r["index"]) for r in rows]
        latencies = [float(r["latency_ms"]) for r in rows]
        scores = [float(r["score"]) for r in rows]

        ax_latency.plot(x, latencies, marker="o", linewidth=1.5, markersize=3, label=run_name, color=colors[i])
        ax_score.plot(x, scores, marker="o", linewidth=1.5, markersize=3, label=run_name, color=colors[i])

        avg_latency_by_diff = [
            average([float(r["latency_ms"]) for r in rows if r["difficulty"] == diff]) for diff in difficulties
        ]
        avg_score_by_diff = [average([float(r["score"]) for r in rows if r["difficulty"] == diff]) for diff in difficulties]

        offset = (i - (len(run_names) - 1) / 2) * width
        ax_diff_latency.bar(x_positions + offset, avg_latency_by_diff, width=width, label=run_name, color=colors[i])
        ax_diff_score.bar(x_positions + offset, avg_score_by_diff, width=width, label=run_name, color=colors[i])

    ax_latency.set_title("Latency per Case")
    ax_latency.set_xlabel("Case Index")
    ax_latency.set_ylabel("Latency (ms)")
    ax_latency.grid(alpha=0.25)

    ax_score.set_title("Correctness Score per Case")
    ax_score.set_xlabel("Case Index")
    ax_score.set_ylabel("Score")
    ax_score.set_ylim(0.0, 1.05)
    ax_score.grid(alpha=0.25)

    ax_diff_latency.set_title("Average Latency by Difficulty")
    ax_diff_latency.set_xticks(x_positions, difficulties)
    ax_diff_latency.set_ylabel("Latency (ms)")
    ax_diff_latency.grid(axis="y", alpha=0.25)

    ax_diff_score.set_title("Average Score by Difficulty")
    ax_diff_score.set_xticks(x_positions, difficulties)
    ax_diff_score.set_ylabel("Score")
    ax_diff_score.set_ylim(0.0, 1.05)
    ax_diff_score.grid(axis="y", alpha=0.25)

    handles, labels = ax_latency.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(4, len(run_names)))
    fig.suptitle(f"Run Comparison ({len(run_names)} runs)", y=0.98, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_path, dpi=180)

    if show:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two or more benchmark runs from results.txt.")
    parser.add_argument("input", nargs="?", default="results.txt", help="Path to results text file.")
    parser.add_argument("-o", "--output", default="comparison.png", help="Output graph image path.")
    parser.add_argument("--show", action="store_true", help="Show plot window in addition to saving.")
    args = parser.parse_args()

    runs = parse_results(Path(args.input))
    plot_comparison(runs, Path(args.output), args.show)
    print(f"Saved graph to {args.output}")


if __name__ == "__main__":
    main()
