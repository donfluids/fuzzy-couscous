#!/usr/bin/env python3
"""Build a per-rank-tier table and grouped bar chart from sweep_shapes.tsv."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load() -> list[dict]:
    rows = []
    with open(HERE / "sweep_shapes.tsv") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            try:
                row["ranks"] = int(row["ranks"])
                row["threads"] = int(row["threads"])
                row["wall_per_step_s"] = float(row["wall_per_step_s"])
            except ValueError:
                continue
            rows.append(row)
    return rows


def build_table(rows: list[dict]) -> str:
    by_R: dict[int, list[dict]] = {}
    for r in rows:
        by_R.setdefault(r["ranks"], []).append(r)

    sections: list[str] = []
    for R in sorted(by_R):
        group = by_R[R]
        # default = the row whose tag starts with "default"
        default = next((r for r in group if r["tag"].startswith("default")), group[0])
        T = group[0]["threads"]
        best = min(group, key=lambda r: r["wall_per_step_s"])
        sections.append(f"### R = {R}, T = {T}  (default = {default['dims']})\n")
        sections.append("| Decomposition | tag | s/step | vs default | vs best |")
        sections.append("|---|---|---|---|---|")
        for r in sorted(group, key=lambda r: r["wall_per_step_s"]):
            star = " ★" if r is best else ""
            vs_def = r["wall_per_step_s"] / default["wall_per_step_s"]
            vs_best = best["wall_per_step_s"] / r["wall_per_step_s"]
            sections.append(
                f"| {r['dims']}{star} | {r['tag']}"
                f" | {r['wall_per_step_s']:.3f} s"
                f" | {(vs_def - 1) * 100:+.1f} %"
                f" | {vs_best * 100:.1f} % |"
            )
        sections.append("")
    return "\n".join(sections)


def plot(rows: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    by_R: dict[int, list[dict]] = {}
    for r in rows:
        by_R.setdefault(r["ranks"], []).append(r)

    fig, axes = plt.subplots(1, len(by_R), figsize=(4.5 * len(by_R), 4.4),
                             sharey=True)
    if len(by_R) == 1:
        axes = [axes]
    overall_best = min(r["wall_per_step_s"] for r in rows)

    for ax, R in zip(axes, sorted(by_R)):
        group = sorted(by_R[R], key=lambda r: r["wall_per_step_s"])
        labels = [f"{r['dims']}\n({r['tag']})" for r in group]
        vals = [r["wall_per_step_s"] for r in group]
        colors = ["C2" if r["tag"].startswith("default") else "C0" for r in group]
        # mark the per-tier best
        best_idx = vals.index(min(vals))
        colors[best_idx] = "gold"
        ax.bar(np.arange(len(group)), vals, color=colors, edgecolor="black")
        ax.set_xticks(np.arange(len(group)))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_title(f"R = {R}, T = {group[0]['threads']}")
        ax.axhline(overall_best, color="grey", linestyle=":",
                   label=f"overall best ({overall_best:.3f} s)")
        ax.grid(axis="y", alpha=0.3)
        for i, v in enumerate(vals):
            ax.text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=8)
        ax.legend(loc="upper left", fontsize=8)
    axes[0].set_ylabel("Wall time per step (s)")
    fig.suptitle("Decomposition-shape sweep: 256³ Case 1, 192 cores",
                 y=1.02)
    fig.tight_layout()
    out = HERE / "shape_sweep.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"wrote {out}")


def main() -> int:
    rows = load()
    if not rows:
        print("no rows in sweep_shapes.tsv", file=sys.stderr)
        return 1

    table = build_table(rows)
    overall_best = min(rows, key=lambda r: r["wall_per_step_s"])
    text = (
        "# Decomposition-shape sweep — 192-core paper Case 1, 256³\n\n"
        "Same conditions as the partitioning sweeps. Within each rank tier "
        "we override `MPI_Dims_create` via `BLAST_DIMS=PX,PY,PZ`. "
        f"**Overall best: R={overall_best['ranks']}, T={overall_best['threads']}, "
        f"dims={overall_best['dims']} ({overall_best['tag']}) at "
        f"{overall_best['wall_per_step_s']:.3f} s/step.**\n\n"
        + table
    )
    (HERE / "results_shapes.md").write_text(text)
    print(text)
    plot(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
