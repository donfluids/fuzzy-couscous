"""Tests for the shared post-processing path resolver (``postprocessing/paths.py``).

Stdlib ``unittest`` so it runs without pytest:

    python3 -m unittest discover -s postprocessing/tests
    python3 postprocessing/tests/test_paths.py

It is also pytest-discoverable once pytest is installed.
"""

import sys
import tempfile
import unittest
from pathlib import Path

# Put postprocessing/ (the parent of tests/) on the path so `import paths` works
# whether run via unittest discovery from the repo root or as a direct script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import paths  # noqa: E402


class TestRepoRootResolution(unittest.TestCase):
    def test_repo_root_has_marker(self):
        # The resolved root must actually be the repo root.
        self.assertTrue((paths.REPO_ROOT / "solver" / "CMakeLists.txt").is_file())

    def test_canonical_dirs_under_root(self):
        self.assertEqual(paths.RUNS, paths.REPO_ROOT / "runs")
        self.assertEqual(paths.DATA, paths.REPO_ROOT / "data")
        self.assertEqual(paths.FIGS, paths.REPO_ROOT / "figs")

    def test_depth_independent_via_git_marker(self):
        # Synthesize a fake checkout and confirm the root is found from a deeply
        # nested file regardless of depth (proves postprocessing/scripts/ and
        # postprocessing/tools/ would resolve the same root).
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "fake_repo"
            (root / ".git").mkdir(parents=True)
            nested = root / "postprocessing" / "scripts"
            nested.mkdir(parents=True)
            found = paths._find_repo_root(nested / "deep_script.py")
            self.assertEqual(found, root.resolve())

    def test_repo_root_found_via_solver_marker(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "fake_repo"
            (root / "solver").mkdir(parents=True)
            (root / "solver" / "CMakeLists.txt").write_text("")
            nested = root / "postprocessing" / "tools"
            nested.mkdir(parents=True)
            found = paths._find_repo_root(nested / "util.py")
            self.assertEqual(found, root.resolve())

    def test_fallback_when_no_marker(self):
        with tempfile.TemporaryDirectory() as tmp:
            # No marker anywhere -> historical <root>/<group>/<file.py> fallback
            # (parent.parent), matching postprocessing/paths.py sitting one group
            # dir below the root.
            f = Path(tmp) / "group" / "mod.py"
            f.parent.mkdir(parents=True)
            self.assertEqual(paths._find_repo_root(f), Path(tmp).resolve())


class TestCategoryOf(unittest.TestCase):
    def test_blast(self):
        for name in (
            "out_blast_128_budget_seed1",
            "out_blast_64_cj",
            "out_blast_256_2fluid",
            "out_blast_512_thick",
            "out_blast_128_bhr",  # blast-driven bhr variant stays under blast/
        ):
            self.assertEqual(paths.category_of(name), "blast", name)

    def test_tgv(self):
        for name in (
            "out_tgv64_hyper2",
            "out_tgv128_hyper6_fd",
            "out_tgv256_hyper2_calib",
            "out_tgv64_hyper2.bak.20260517_162132",  # backup dir still tgv
        ):
            self.assertEqual(paths.category_of(name), "tgv", name)

    def test_chamber_precedes_tnt_and_others(self):
        # The closed-chamber comparison set groups together.
        self.assertEqual(paths.category_of("out_tnt_chamber_128"), "chamber")
        self.assertEqual(paths.category_of("out_tg_chamber_128"), "chamber")
        self.assertEqual(paths.category_of("out_sf_chamber_128"), "chamber")

    def test_tnt_non_chamber(self):
        self.assertEqual(paths.category_of("out_tnt_freeair_128"), "tnt")
        self.assertEqual(paths.category_of("out_tnt_smoke"), "tnt")

    def test_bhr_and_rans(self):
        self.assertEqual(paths.category_of("out_bhr_seed1"), "bhr")
        self.assertEqual(paths.category_of("out_rans_bhr"), "bhr")

    def test_scaling_hit_paper(self):
        self.assertEqual(paths.category_of("out_scaling_case1_256"), "scaling")
        self.assertEqual(paths.category_of("out_scaling_case1_64_pgo_train"), "scaling")
        self.assertEqual(paths.category_of("out_hit_forced_32"), "hit")
        self.assertEqual(paths.category_of("out_hit_forced_64"), "hit")
        # paper Case 1 is the closed-chamber production run -> chamber.
        self.assertEqual(paths.category_of("out_paper_case1"), "chamber")

    def test_unknown_falls_back_to_misc(self):
        self.assertEqual(paths.category_of("out_mpi_smoke"), "misc")
        self.assertEqual(paths.category_of("something_else"), "misc")

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            paths.category_of("")

    def test_path_separator_raises(self):
        with self.assertRaises(ValueError):
            paths.category_of("out_blast_64_cj/file.h5")


class TestRunDir(unittest.TestCase):
    def test_bare_name(self):
        self.assertEqual(
            paths.run_dir("out_blast_64_cj"),
            paths.RUNS / "blast" / "out_blast_64_cj",
        )

    def test_chamber_name(self):
        self.assertEqual(
            paths.run_dir("out_tnt_chamber_128"),
            paths.RUNS / "chamber" / "out_tnt_chamber_128",
        )

    def test_with_subpath(self):
        got = paths.run_dir("out_blast_64_cj_t5/blast_64_cj_t5_spectra.h5")
        self.assertEqual(
            got,
            paths.RUNS / "blast" / "out_blast_64_cj_t5" / "blast_64_cj_t5_spectra.h5",
        )

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            paths.run_dir("")


if __name__ == "__main__":
    unittest.main()
