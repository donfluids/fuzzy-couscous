"""Shared path resolution for fuzzy-couscous post-processing.

Single source of truth for the repository root and the canonical data
locations (``runs/``, ``data/``, ``figs/``), plus the run-directory taxonomy.
Every analysis script imports from here so the pipeline is robust to
relocation: a script never hardcodes its own directory depth and never
hardcodes a run's category.

Before this module, scripts resolved the root with a fragile
``ROOT = Path(__file__).resolve().parent.parent`` and split run locations
between ``RUNS = ROOT/"runs"`` and ``RUNROOT = ROOT/"solver"``. Both break the
moment a script changes depth or a run is relocated. ``run_dir(name)`` hides
the categorization, so re-categorizing runs touches only this file.

Stdlib-only (``pathlib``) so it imports anywhere without dependencies.
"""

from __future__ import annotations

from pathlib import Path

__all__ = [
    "REPO_ROOT",
    "RUNS",
    "DATA",
    "FIGS",
    "CATEGORIES",
    "category_of",
    "run_dir",
]


def _find_repo_root(start: Path) -> Path:
    """Walk upward from ``start`` until a repository marker is found.

    A directory is the repo root if it contains a ``.git`` entry or a
    ``solver/CMakeLists.txt`` file. This makes the root independent of how deep
    the importing module sits (``postprocessing/``, ``postprocessing/scripts/``,
    ``postprocessing/tools/`` all resolve the same root). If no marker is found
    -- e.g. the tree was copied outside a checkout -- fall back to the historical
    ``<root>/<group>/<file>`` layout so the module still returns something usable.
    """
    here = start.resolve()
    for parent in (here, *here.parents):
        if (parent / ".git").exists() or (parent / "solver" / "CMakeLists.txt").is_file():
            return parent
    return here.parent.parent


REPO_ROOT = _find_repo_root(Path(__file__))
RUNS = REPO_ROOT / "runs"
DATA = REPO_ROOT / "data"
FIGS = REPO_ROOT / "figs"

#: Canonical run categories. ``scaling`` is populated by the runs migration from
#: the ``scaling/`` source location (its dirs are not ``out_*``-named);
#: ``misc`` is the documented fallback for names that match no rule.
CATEGORIES = ("blast", "tgv", "bhr", "chamber", "tnt", "scaling", "misc")

#: Ordered ``(token, category)`` rules; first match wins. Each rule matches the
#: run-directory *name* (e.g. ``"out_blast_128_cj_t5"``). A token that starts
#: with ``out_`` is matched as a name prefix; any other token is matched as a
#: substring. The ``_chamber`` substring rule is listed first so the closed-
#: chamber comparison set (``out_sf_chamber_*``, ``out_tg_chamber_*``,
#: ``out_tnt_chamber_*``) groups together under ``chamber/`` rather than being
#: split across ``tnt/`` and ``misc/``.
_RULES: tuple[tuple[str, str], ...] = (
    ("_chamber", "chamber"),
    ("out_tgv", "tgv"),
    ("out_tnt", "tnt"),
    ("out_blast", "blast"),
    ("out_bhr", "bhr"),
    ("out_rans", "bhr"),
)


def category_of(name: str) -> str:
    """Map a bare run-directory name to its category.

    >>> category_of("out_blast_128_budget_seed1")
    'blast'
    >>> category_of("out_tnt_chamber_128")   # _chamber wins over out_tnt
    'chamber'
    >>> category_of("out_tnt_freeair_128")
    'tnt'

    Raises ``ValueError`` if ``name`` is empty or contains a path separator --
    ``category_of`` classifies a single directory name, not a path. Use
    :func:`run_dir`, which accepts an optional sub-path. Unknown names map to
    ``"misc"`` rather than raising, so a stray run is never silently lost.
    """
    if not name:
        raise ValueError("run name must be a non-empty directory name")
    if "/" in name or "\\" in name:
        raise ValueError(f"category_of expects a bare name, got path {name!r}")
    for token, category in _RULES:
        if token.startswith("out_"):
            if name.startswith(token):
                return category
        elif token in name:
            return category
    return "misc"


def run_dir(name: str) -> Path:
    """Resolve a run name (optionally with a sub-path) under ``runs/<category>/``.

    >>> run_dir("out_blast_64_cj").relative_to(RUNS).as_posix()
    'blast/out_blast_64_cj'
    >>> run_dir("out_blast_64_cj_t5/blast_64_cj_t5_spectra.h5").name
    'blast_64_cj_t5_spectra.h5'

    The first path component is the run directory used for categorization; any
    remaining components are appended unchanged, so callers that previously did
    ``RUNROOT / "out_x/file.h5"`` can pass the same relative string.
    """
    parts = Path(name).parts
    if not parts:
        raise ValueError("empty run name")
    head, *rest = parts
    return RUNS.joinpath(category_of(head), head, *rest)
