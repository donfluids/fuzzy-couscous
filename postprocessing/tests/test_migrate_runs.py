"""Tests for the runs migration tool, exercised on a throwaway fixture tree.

    python3 -m unittest discover -s postprocessing/tests

Patches the path globals so nothing touches the real repo tree.
"""

import sys
import tempfile
import unittest
from pathlib import Path

_PP = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, _PP)
sys.path.insert(0, str(Path(_PP) / "scripts"))

import paths  # noqa: E402
import migrate_runs  # noqa: E402


def _touch(p: Path, text: str = "x") -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)


class MigrateRunsFixture(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        # Build a miniature of the real scatter.
        _touch(self.root / "runs" / "out_blast_64_cj" / "snap_000000.h5")
        _touch(self.root / "runs" / "blast_512_thick.run.log")          # loose log
        _touch(self.root / "solver" / "out_tgv64_hyper2" / "run_stats.csv")
        _touch(self.root / "solver" / "out_tnt_chamber_128" / "run_stats.csv")  # chamber
        _touch(self.root / "solver" / "runs" / "out_tnt_freeair_128" / "s.h5")  # tnt
        _touch(self.root / "scaling" / "runs_c96" / "tagA" / "s.h5")
        _touch(self.root / "scaling" / "logs_c96" / "a.log")
        _touch(self.root / "scaling" / "pgo_profile" / "p.gcda")
        _touch(self.root / "solver" / "out_weird_thing" / "x")          # -> misc
        # Repoint the path globals + manifest at the fixture.
        self._save = (paths.REPO_ROOT, paths.RUNS, migrate_runs.MANIFEST)
        paths.REPO_ROOT = self.root
        paths.RUNS = self.root / "runs"
        migrate_runs.MANIFEST = paths.RUNS / "MANIFEST.md"

    def tearDown(self):
        paths.REPO_ROOT, paths.RUNS, migrate_runs.MANIFEST = self._save
        self._tmp.cleanup()

    def _snapshot(self) -> set[str]:
        return {
            p.relative_to(self.root).as_posix()
            for p in self.root.rglob("*")
            if p.is_file() and p.name != "MANIFEST.md"
        }

    def test_plan_classifies_correctly(self):
        plan = {old.name: new.relative_to(paths.RUNS).as_posix() for old, new in migrate_runs.build_plan()}
        self.assertEqual(plan["out_blast_64_cj"], "blast/out_blast_64_cj")
        self.assertEqual(plan["out_tgv64_hyper2"], "tgv/out_tgv64_hyper2")
        self.assertEqual(plan["out_tnt_chamber_128"], "chamber/out_tnt_chamber_128")
        self.assertEqual(plan["out_tnt_freeair_128"], "tnt/out_tnt_freeair_128")
        self.assertEqual(plan["runs_c96"], "scaling/runs_c96")
        self.assertEqual(plan["logs_c96"], "scaling/logs_c96")
        self.assertEqual(plan["pgo_profile"], "scaling/pgo_profile")
        self.assertEqual(plan["blast_512_thick.run.log"], "blast/blast_512_thick.run.log")
        self.assertEqual(plan["out_weird_thing"], "misc/out_weird_thing")  # flagged, not lost

    def test_apply_then_rollback_round_trips(self):
        before = self._snapshot()
        migrate_runs.cmd_apply()
        after = self._snapshot()
        # Everything now lives under runs/<category>/ and nothing under solver/ or scaling/.
        self.assertTrue(all(a.startswith("runs/") for a in after), after)
        self.assertTrue(any(a.startswith("runs/blast/") for a in after))
        self.assertTrue(any(a.startswith("runs/scaling/") for a in after))
        migrate_runs.cmd_rollback()
        self.assertEqual(self._snapshot(), before)

    def test_apply_is_idempotent(self):
        migrate_runs.cmd_apply()
        after_first = self._snapshot()
        # Second apply must be a no-op (already consolidated).
        self.assertEqual(migrate_runs.build_plan(), [])
        migrate_runs.cmd_apply()
        self.assertEqual(self._snapshot(), after_first)

    def test_refuses_to_overwrite_existing_destination(self):
        # Pre-create a colliding destination.
        _touch(paths.RUNS / "blast" / "out_blast_64_cj" / "already_here.h5")
        with self.assertRaises(SystemExit):
            migrate_runs.cmd_apply()

    def test_detects_two_sources_one_destination(self):
        # Same run name in two locations -> one destination -> collision.
        _touch(self.root / "solver" / "out_blast_64_cj" / "newer.h5")
        plan = migrate_runs.build_plan()
        collisions = migrate_runs.find_collisions(plan)
        dest = paths.RUNS / "blast" / "out_blast_64_cj"
        self.assertIn(dest, collisions)
        self.assertEqual(len(collisions[dest]), 2)
        # apply must refuse (exit 2) and move nothing.
        before = self._snapshot()
        self.assertEqual(migrate_runs.cmd_apply(), 2)
        self.assertEqual(self._snapshot(), before)


if __name__ == "__main__":
    unittest.main()
