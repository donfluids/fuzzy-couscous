#!/usr/bin/env python3
"""Strong-scaling figure for the paper supplementary.

Reads scaleout.tsv, computes speedup and parallel efficiency relative to
the smallest budget, and emits a publication-style two-panel figure.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load() -> list[dict]:
    rows = []
    with open(HERE / "scaleout.tsv") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            try:
                r["cores"] = int(r["cores"])
                r["ranks"] = int(r["ranks"])
                r["threads"] = int(r["threads"])
                r["wall_per_step_s"] = float(r["wall_per_step_s"])
            except ValueError:
                continue
            rows.append(r)
    rows.sort(key=lambda r: r["cores"])
    return rows


def build_table(rows: list[dict]) -> str:
    baseline = rows[0]
    base_t = baseline["wall_per_step_s"]
    base_c = baseline["cores"]
    lines = [
        f"| Cores | Layout | s/step | Speedup vs {base_c}c | Efficiency |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        speedup = base_t / r["wall_per_step_s"]
        ideal = r["cores"] / base_c
        eff = speedup / ideal
        lines.append(
            f"| {r['cores']:>3}"
            f" | {r['ranks']}×{r['threads']:<3}"
            f" | {r['wall_per_step_s']:>6.3f} s"
            f" | {speedup:>5.2f}×"
            f" | {eff * 100:>5.1f} % |"
        )
    return "\n".join(lines)


def plot(rows: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cores = [r["cores"] for r in rows]
    perstep = [r["wall_per_step_s"] for r in rows]
    base_t = perstep[0]
    base_c = cores[0]
    speedup = [base_t / t for t in perstep]
    ideal = [c / base_c for c in cores]
    efficiency = [s / i for s, i in zip(speedup, ideal)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.6))

    # Panel 1: speedup vs cores (log-log)
    ax1.plot(cores, ideal, "--", color="grey", label="ideal (linear)")
    ax1.plot(cores, speedup, "o-", color="C0", label="measured (8 MPI × T OMP)")
    ax1.set_xscale("log", base=2)
    ax1.set_yscale("log", base=2)
    ax1.set_xlabel("Total cores")
    ax1.set_ylabel(f"Speedup vs {base_c} cores")
    ax1.set_title("Strong scaling: 256³ Case 1")
    ax1.set_xticks(cores)
    ax1.set_xticklabels([str(c) for c in cores])
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend(loc="upper left")

    for c, s in zip(cores, speedup):
        ax1.annotate(f"{s:.2f}×", (c, s),
                     xytext=(6, -10), textcoords="offset points",
                     fontsize=8, color="C0")

    # Panel 2: parallel efficiency
    ax2.plot(cores, [e * 100 for e in efficiency], "o-", color="C3")
    ax2.axhline(100, color="grey", linestyle="--", label="100 %")
    ax2.axhline(50, color="grey", linestyle=":", alpha=0.5, label="50 %")
    ax2.set_xscale("log", base=2)
    ax2.set_ylim(0, 110)
    ax2.set_xlabel("Total cores")
    ax2.set_ylabel(f"Parallel efficiency vs {base_c} cores (%)")
    ax2.set_title("Strong-scaling efficiency")
    ax2.set_xticks(cores)
    ax2.set_xticklabels([str(c) for c in cores])
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend(loc="upper right")

    for c, e in zip(cores, efficiency):
        ax2.annotate(f"{e * 100:.0f}", (c, e * 100),
                     xytext=(0, 6), textcoords="offset points",
                     ha="center", fontsize=8, color="C3")

    fig.suptitle(
        "Single-node AMD EPYC 9965, layout = 8 MPI ranks × T threads",
        y=1.02,
    )
    fig.tight_layout()
    out = HERE / "strong_scaling.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


def main() -> int:
    rows = load()
    if not rows:
        print("no rows in scaleout.tsv", file=sys.stderr)
        return 1

    table = build_table(rows)
    # 96 → 192 production-regime cross-budget efficiency
    by_c = {r["cores"]: r for r in rows}
    if 96 in by_c and 192 in by_c:
        t96 = by_c[96]["wall_per_step_s"]
        t192 = by_c[192]["wall_per_step_s"]
        ratio = t96 / t192
        eff_prod = ratio / 2.0
        prod_note = (
            f"\n\n**96 → 192 cores (production regime):** "
            f"{t96:.3f} → {t192:.3f} s/step = "
            f"**{ratio:.2f}× speedup for 2× cores = "
            f"{eff_prod * 100:.0f} % strong-scaling efficiency**. This is the "
            "relevant number for paper runs — both points use 8 ranks with "
            "near-cubic decomposition and threads-per-rank that fill the "
            "available CCDs.\n"
        )
    else:
        prod_note = ""
    (HERE / "results_scaleout.md").write_text(
        "# Strong scaling — 256³ paper Case 1\n\n"
        "Single-node AMD EPYC 9965 (192 cores Zen5c, 12 NUMA × 16 cores), "
        "OpenMPI 4.1.6. Layout policy: **8 MPI ranks × T OpenMP threads**, "
        "where T = (total cores) / 8. The 8-rank Cartesian decomposition is "
        "the per-budget optimum identified in `results_c96.md` / "
        "`results_c192.md` / `results_shapes.md`. 30 step paper-Case-1 IC "
        "with all snapshot / spectra I/O disabled; s/step measured from "
        "the spdlog step-5 → step-30 timestamp delta on rank 0.\n\n"
        + table + "\n"
        + prod_note +
        "\nThe efficiency vs the 8-core baseline is pessimistic by design: "
        "at 8×1 each rank occupies a single core in a 16-core CCD, so the "
        "rank already has the full CCD memory channel to itself. Adding "
        "threads inside the same CCD (8×2 → 8×16) buys execution units but "
        "no extra bandwidth, so the curve only steepens once T exceeds 16 "
        "and ranks start spanning CCDs.\n"
    )
    print(table)
    plot(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
