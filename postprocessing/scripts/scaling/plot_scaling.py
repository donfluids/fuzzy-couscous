#!/usr/bin/env python3
"""Build a strong-scaling table + plot from sweep_summary.tsv."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


def load(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path) as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            try:
                ranks = int(row["ranks"])
                threads = int(row["threads"])
                wall = float(row["wall_total_s"])
                perstep = float(row["wall_per_step_s"])
            except (ValueError, KeyError):
                continue
            rows.append({
                "ranks": ranks,
                "threads": threads,
                "wall_total_s": wall,
                "wall_per_step_s": perstep,
            })
    rows.sort(key=lambda r: r["ranks"])
    return rows


def build_table(rows: list[dict]) -> tuple[str, list[dict]]:
    if not rows:
        return "(no rows)", []

    # R*T is constant (96 cores), so this is a *partitioning* study, not
    # classical strong scale-out. We report:
    #   - speedup vs 1×96 baseline (most threads / fewest ranks)
    #   - normalized throughput = best_s_per_step / this_s_per_step
    baseline = rows[0]
    base_t = baseline["wall_per_step_s"]
    best_t = min(r["wall_per_step_s"] for r in rows)

    base_label = f"vs {baseline['ranks']}×{baseline['threads']}"
    lines = [
        f"| Ranks × Threads | Wall (30 steps) | s/step | {base_label} | vs best |",
        "|---|---|---|---|---|",
    ]
    out_rows = []
    for r in rows:
        speedup = base_t / r["wall_per_step_s"]
        norm_best = best_t / r["wall_per_step_s"]
        marker = " ★" if r["wall_per_step_s"] == best_t else "  "
        lines.append(
            f"| {r['ranks']:>3} × {r['threads']:<3}{marker}"
            f" | {r['wall_total_s']:>7.2f} s"
            f" | {r['wall_per_step_s']:>6.3f} s"
            f" | {speedup:>5.2f}×"
            f" | {norm_best * 100:>5.1f} % |"
        )
        out_rows.append({**r, "speedup": speedup, "norm_best": norm_best})
    return "\n".join(lines), out_rows


def plot(rows: list[dict], out_png: Path, args) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ranks = [r["ranks"] for r in rows]
    perstep = [r["wall_per_step_s"] for r in rows]
    best_t = min(perstep)
    norm = [best_t / t for t in perstep]
    best_i = perstep.index(best_t)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    ax1.plot(ranks, norm, "o-", color="C0", label="measured")
    ax1.axhline(1.0, color="grey", linestyle="--",
                label="ideal (perfect partitioning)")
    ax1.scatter([ranks[best_i]], [norm[best_i]], s=140, marker="*",
                color="gold", edgecolor="black", zorder=5, label="optimum")
    ax1.set_xscale("log", base=2)
    ax1.set_ylim(0.6, 1.05)
    ax1.set_xlabel(f"MPI ranks (threads/rank = {args.total_cores} / ranks)")
    ax1.set_ylabel("Throughput / best throughput")
    ax1.set_title(f"Hybrid partitioning: 256³ Case 1, {args.total_cores} cores")
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend(loc="lower center")
    ax1.set_xticks(ranks)
    ax1.set_xticklabels([str(r) for r in ranks])

    ax2.plot(ranks, perstep, "o-", color="C3")
    ax2.scatter([ranks[best_i]], [best_t], s=140, marker="*",
                color="gold", edgecolor="black", zorder=5)
    ax2.set_xscale("log", base=2)
    ax2.set_xlabel("MPI ranks")
    ax2.set_ylabel("Wall time per step (s)")
    ax2.set_title("Time per step (lower is better)")
    ax2.grid(True, which="both", alpha=0.3)
    ax2.set_xticks(ranks)
    ax2.set_xticklabels([str(r) for r in ranks])
    for r, t in zip(ranks, perstep):
        ax2.annotate(f"{t:.2f}", (r, t), xytext=(0, 6),
                     textcoords="offset points",
                     ha="center", fontsize=8, color="C3")

    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    print(f"wrote {out_png}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--total-cores", type=int, default=96)
    p.add_argument("--tsv", default=None)
    p.add_argument("--out-md", default=None)
    p.add_argument("--out-csv", default=None)
    p.add_argument("--out-png", default=None)
    args = p.parse_args()

    tag = f"c{args.total_cores}"
    if args.tsv is None:
        args.tsv = f"sweep_summary_{tag}.tsv"
    if args.out_md is None:
        args.out_md = f"results_{tag}.md"
    if args.out_csv is None:
        args.out_csv = f"results_{tag}.csv"
    if args.out_png is None:
        args.out_png = f"strong_scaling_{tag}.png"

    here = Path(__file__).resolve().parent
    tsv = (here / args.tsv).resolve()
    rows = load(tsv)
    if not rows:
        print(f"no rows in {tsv}", file=sys.stderr)
        return 1

    table, enriched = build_table(rows)
    best = min(enriched, key=lambda r: r["wall_per_step_s"])
    numa_count = max(1, args.total_cores // 16)
    (here / args.out_md).write_text(
        f"# Strong scaling — paper Case 1, 256³, fixed {args.total_cores} cores\n\n"
        f"Single-node AMD EPYC 9965, cores 0..{args.total_cores - 1} (= {numa_count} of 12 NUMA "
        f"nodes, 16 cores per node), OpenMPI 4.1.6, `OMP_PROC_BIND=close`, "
        f"`OMP_PLACES=cores`. Each row holds **R × T = {args.total_cores} total physical "
        "cores** — partitioning study (best split of a fixed core budget "
        "between MPI ranks and OpenMP threads). 30-step paper-Case-1 IC "
        "(`scaling_case1_256.toml`) with all snapshot / spectra I/O "
        "disabled. s/step measured from the spdlog step-5 → step-30 "
        "timestamp delta on rank 0.\n\n"
        + table + "\n\n"
        f"**Optimum: {best['ranks']} ranks × {best['threads']} threads** at "
        f"{best['wall_per_step_s']:.3f} s/step.\n"
    )

    with open(here / args.out_csv, "w") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=["ranks", "threads", "wall_total_s",
                        "wall_per_step_s", "speedup", "norm_best"],
        )
        w.writeheader()
        for r in enriched:
            w.writerow(r)

    plot(enriched, here / args.out_png, args)
    print(table)
    return 0


if __name__ == "__main__":
    sys.exit(main())
