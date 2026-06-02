"""Write evaluation metrics to a markdown results file (CPU-only, no sim)."""

from __future__ import annotations

import os


def write_results_markdown(metrics: dict, path: str, title: str = "Results") -> None:
    """Write a metrics dict as a markdown table to path."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    lines = [f"# {title}", "", "| Metric | Value |", "| --- | ---: |"]
    for key, value in metrics.items():
        lines.append(f"| {key} | {value} |")
    with open(path, "w") as handle:
        handle.write("\n".join(lines) + "\n")
