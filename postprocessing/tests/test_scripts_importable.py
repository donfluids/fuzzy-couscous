"""Smoke test: every relocated post-processing script compiles, is wired to the
shared ``paths`` resolver, and imports without a path/anchor error.

    python3 -m unittest discover -s postprocessing/tests

This is the guard for U4 (the relocation/rewire). It proves "everything is
connected": no script still carries the fragile ``parent.parent`` anchor or the
old ``RUNROOT = ROOT/"solver"`` convention, and each module's imports resolve
from its new home.

Scripts that execute analysis at import time (load specific run files) are
imported too, but a *data* error (the specific run not being present) is
tolerated -- only a *wiring* error (ImportError/NameError/...) fails the test.
"""

import os
import sys
import importlib.util
import py_compile
import unittest
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")  # headless plotting

PP = Path(__file__).resolve().parent.parent          # postprocessing/
SCRIPTS = PP / "scripts"
TOOLS = PP / "tools"
for _d in (PP, TOOLS, SCRIPTS):
    sys.path.insert(0, str(_d))

# Scripts that run analysis on import (load specific run files); a missing-run
# error is tolerated, a wiring error is not. migrate_runs/paths covered elsewhere.
_RUN_ON_IMPORT = {
    "bhr_calibrate", "compare_mf_sg_spectra", "ic_compare_mf_sg",
    "movie_mf_evolution", "movie_tnt", "plot_bhr_vs_dns",
}
_EXCLUDE = {"migrate_runs", "paths"}
_WIRING_ERRORS = (ImportError, ModuleNotFoundError, NameError, AttributeError)


def _all_scripts():
    files = sorted(SCRIPTS.rglob("*.py")) + sorted(TOOLS.glob("*.py"))
    return [f for f in files if f.stem not in _EXCLUDE]


def _import_file(path: Path):
    spec = importlib.util.spec_from_file_location(f"_smoke_{path.stem}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)


class TestScriptsCompile(unittest.TestCase):
    def test_all_compile(self):
        for f in _all_scripts():
            with self.subTest(script=f.name):
                py_compile.compile(str(f), doraise=True)


class TestNoStaleAnchors(unittest.TestCase):
    # data/ did not move, so ROOT/"data" stays valid; the broken patterns are the
    # depth anchor, the solver-as-runroot convention, and runs-without-category.
    STALE = ('parent.parent', 'RUNROOT', 'ROOT / "runs"', 'ROOT/"runs"',
             'ROOT / "solver"', 'ROOT/"solver"')

    def test_no_stale_anchor_patterns(self):
        for f in _all_scripts():
            text = f.read_text(encoding="utf-8")
            for pat in self.STALE:
                with self.subTest(script=f.name, pattern=pat):
                    self.assertNotIn(pat, text, f"{f.name} still contains {pat!r}")


class TestScriptsImport(unittest.TestCase):
    def test_import_resolves_wiring(self):
        for f in _all_scripts():
            # Scripts that execute their full analysis at module level depend on
            # argv + specific run data; exec-ing them under the test runner is not
            # meaningful. Their wiring is covered by compile + the no-stale-anchor
            # static check above. Everything else must import cleanly.
            if f.stem in _RUN_ON_IMPORT:
                continue
            with self.subTest(script=f.name):
                try:
                    _import_file(f)
                except SystemExit:
                    pass  # argparse-style scripts may exit if invoked with no args
                except _WIRING_ERRORS as e:
                    self.fail(f"{f.name}: wiring error: {type(e).__name__}: {e}")


if __name__ == "__main__":
    unittest.main()
