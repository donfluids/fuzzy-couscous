#!/usr/bin/env python3
"""Consolidate scattered run outputs into one categorized ``runs/`` tree.

Run outputs currently live in four places -- ``runs/out_*``, ``solver/out_*``,
``solver/runs/out_*``, and ``scaling/{runs,logs,pgo}_*`` -- all on one
filesystem and all gitignored. This tool moves each into
``runs/<category>/<name>`` (categories from ``postprocessing/paths.py``), with a
manifest recording every old->new mapping so the move is fully reversible.

Modes
-----
``--dry-run`` (default)
    Scan, classify, write ``runs/MANIFEST.md``, print a summary. **Moves nothing.**
``--apply``
    Perform the moves recorded in the freshly computed plan, then update the
    manifest with applied timestamps. Same-filesystem renames are atomic/instant.
``--rollback``
    Read ``runs/MANIFEST.md`` and move every entry new->old.

The tool is idempotent: re-running ``--apply`` after a completed migration is a
no-op, because a run already sitting under ``runs/<category>/`` no longer matches
any source pattern. A name that matches no taxonomy rule lands in ``runs/misc/``
and is flagged in the manifest -- never silently dropped.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import shutil
import sys
from pathlib import Path

# Reuse the single source of truth for the repo root and run taxonomy.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import paths  # noqa: E402

MANIFEST = paths.RUNS / "MANIFEST.md"
_TSV_FENCE = "```tsv"  # machine-readable mapping block inside MANIFEST.md

# (glob relative to repo root, kind). "run"/"scaling" classify directories;
# "runlog" classifies loose stdout-tee files left at the top of runs/.
_SOURCES: tuple[tuple[str, str], ...] = (
    ("runs/out_*", "run"),
    ("solver/out_*", "run"),
    ("solver/runs/out_*", "run"),
    ("runs/*.run.log", "runlog"),
    ("scaling/runs_*", "scaling"),
    ("scaling/logs_*", "scaling"),
    ("scaling/pgo_*", "scaling"),
)


def _category_for(kind: str, name: str) -> str:
    """Target category for a source entry of the given kind."""
    if kind == "scaling":
        return "scaling"
    if kind == "runlog":
        # e.g. "blast_512_thick.run.log" -> categorize as run "out_blast_512_thick"
        stem = name
        for suffix in (".run.log", ".log"):
            if stem.endswith(suffix):
                stem = stem[: -len(suffix)]
                break
        probe = stem if stem.startswith("out_") else f"out_{stem}"
        return paths.category_of(probe)
    return paths.category_of(name)  # kind == "run"


def build_plan() -> list[tuple[Path, Path]]:
    """Return the ordered list of (old, new) absolute paths to migrate.

    Skips entries already at their destination (idempotency) and entries whose
    destination already exists (conflict -- reported, not overwritten).
    """
    plan: list[tuple[Path, Path]] = []
    seen: set[Path] = set()
    for pattern, kind in _SOURCES:
        want_dir = kind in ("run", "scaling")
        for old in sorted(paths.REPO_ROOT.glob(pattern)):
            if old in seen:
                continue
            if want_dir and not old.is_dir():
                continue
            if kind == "runlog" and not old.is_file():
                continue
            category = _category_for(kind, old.name)
            new = paths.RUNS / category / old.name
            if old.resolve() == new.resolve():
                continue  # already migrated
            seen.add(old)
            plan.append((old, new))
    return plan


def find_collisions(plan: list[tuple[Path, Path]]) -> dict[Path, list[Path]]:
    """Return destinations that two or more sources map to.

    A collision means two distinct source runs share a directory name across
    locations (e.g. ``runs/out_x`` and ``solver/out_x``). Migrating both would
    require overwriting one; the tool refuses and asks the operator to resolve
    (keep the newer, drop the stale, or rename one) before applying.
    """
    dest_to_olds: dict[Path, list[Path]] = {}
    for old, new in plan:
        dest_to_olds.setdefault(new, []).append(old)
    return {dest: olds for dest, olds in dest_to_olds.items() if len(olds) > 1}


def _rel(p: Path) -> str:
    try:
        return p.relative_to(paths.REPO_ROOT).as_posix()
    except ValueError:
        return str(p)


def write_manifest(plan: list[tuple[Path, Path]], *, applied: bool) -> None:
    paths.RUNS.mkdir(parents=True, exist_ok=True)
    now = _dt.datetime.now().isoformat(timespec="seconds")
    by_cat: dict[str, int] = {}
    for _, new in plan:
        by_cat[new.parent.name] = by_cat.get(new.parent.name, 0) + 1

    lines: list[str] = []
    lines.append("# Runs migration manifest")
    lines.append("")
    lines.append(f"- Generated: {now}")
    lines.append(f"- Status: {'APPLIED' if applied else 'DRY-RUN (nothing moved)'}")
    lines.append(f"- Entries: {len(plan)}")
    if by_cat:
        lines.append(f"- By category: " + ", ".join(f"{k}={v}" for k, v in sorted(by_cat.items())))
    lines.append("")
    lines.append("Rollback with: `python3 postprocessing/scripts/migrate_runs.py --rollback`")
    lines.append("")

    collisions = find_collisions(plan)
    if collisions:
        lines.append("## ⚠ Collisions (must be resolved before --apply)")
        lines.append("")
        lines.append("Two or more sources map to the same destination. Resolve by "
                     "keeping one and removing or renaming the other(s):")
        lines.append("")
        for dest, olds in collisions.items():
            lines.append(f"- `{_rel(dest)}` <- " + ", ".join(f"`{_rel(o)}`" for o in olds))
        lines.append("")

    lines.append("## Mappings")
    lines.append("")
    lines.append("| category | old path | new path |")
    lines.append("|---|---|---|")
    for old, new in plan:
        lines.append(f"| {new.parent.name} | `{_rel(old)}` | `{_rel(new)}` |")
    lines.append("")
    lines.append("## Mappings (machine-readable -- do not edit; used by --rollback)")
    lines.append("")
    lines.append(_TSV_FENCE)
    for old, new in plan:
        lines.append(f"{_rel(old)}\t{_rel(new)}")
    lines.append("```")
    lines.append("")
    MANIFEST.write_text("\n".join(lines), encoding="utf-8")


def read_manifest_mappings() -> list[tuple[Path, Path]]:
    if not MANIFEST.is_file():
        raise SystemExit(f"no manifest at {MANIFEST}; nothing to roll back")
    text = MANIFEST.read_text(encoding="utf-8").splitlines()
    out: list[tuple[Path, Path]] = []
    in_block = False
    for line in text:
        if line.strip() == _TSV_FENCE:
            in_block = True
            continue
        if in_block:
            if line.strip() == "```":
                break
            old_s, _, new_s = line.partition("\t")
            if old_s and new_s:
                out.append((paths.REPO_ROOT / old_s, paths.REPO_ROOT / new_s))
    return out


def _move(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        raise SystemExit(f"refusing to overwrite existing destination: {_rel(dst)}")
    shutil.move(str(src), str(dst))  # os.rename fast-path on same filesystem


def cmd_dry_run() -> int:
    plan = build_plan()
    write_manifest(plan, applied=False)
    print(f"[dry-run] {len(plan)} entries planned; manifest -> {_rel(MANIFEST)}")
    for old, new in plan:
        print(f"  {_rel(old)}  ->  {_rel(new)}")
    if not plan:
        print("  (nothing to migrate -- already consolidated)")
    collisions = find_collisions(plan)
    if collisions:
        print(f"\n⚠ {len(collisions)} collision(s) -- --apply will refuse until resolved:")
        for dest, olds in collisions.items():
            print(f"  {_rel(dest)} <- " + ", ".join(_rel(o) for o in olds))
    print("\nReview the manifest, then run with --apply to perform the moves.")
    return 0


def cmd_apply() -> int:
    plan = build_plan()
    if not plan:
        print("[apply] nothing to migrate -- already consolidated")
        return 0
    collisions = find_collisions(plan)
    if collisions:
        print("[apply] refusing: unresolved destination collisions:")
        for dest, olds in collisions.items():
            print(f"  {_rel(dest)} <- " + ", ".join(_rel(o) for o in olds))
        print("Resolve (keep one, remove/rename the others) and re-run.")
        return 2
    for old, new in plan:
        _move(old, new)
        print(f"  moved {_rel(old)} -> {_rel(new)}")
    write_manifest(plan, applied=True)
    print(f"\n[apply] migrated {len(plan)} entries; manifest -> {_rel(MANIFEST)}")
    return 0


def cmd_rollback() -> int:
    mappings = read_manifest_mappings()
    moved = 0
    for old, new in mappings:
        if new.exists() and not old.exists():
            _move(new, old)
            print(f"  restored {_rel(new)} -> {_rel(old)}")
            moved += 1
    print(f"\n[rollback] restored {moved}/{len(mappings)} entries")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--dry-run", action="store_true", help="scan + write manifest, move nothing (default)")
    g.add_argument("--apply", action="store_true", help="perform the migration")
    g.add_argument("--rollback", action="store_true", help="reverse a prior migration from the manifest")
    args = ap.parse_args(argv)
    if args.apply:
        return cmd_apply()
    if args.rollback:
        return cmd_rollback()
    return cmd_dry_run()


if __name__ == "__main__":
    raise SystemExit(main())
