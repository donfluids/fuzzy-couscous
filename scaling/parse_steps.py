#!/usr/bin/env python3
"""Parse spdlog "step N" lines and report steady-state seconds/step.

Log lines look like:
    [12:34:56.789] [info] step      5 t=... dt=... ...

We compute (timestamp[last] - timestamp[window]) / (last_step - window).
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime, timedelta

ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
LINE_RE = re.compile(
    r"^\[(\d{2}):(\d{2}):(\d{2})\.(\d{3})\]\s+\[info\]\s+step\s+(\d+)\b"
)


def parse(path: str, window: int) -> tuple[float, int, int]:
    steps: list[tuple[datetime, int]] = []
    with open(path) as fh:
        for raw in fh:
            line = ANSI_RE.sub("", raw)
            m = LINE_RE.match(line)
            if not m:
                continue
            h, mi, s, ms, step = map(int, m.groups())
            # use today's date as anchor; we only need the delta
            dt = datetime(2000, 1, 1, h, mi, s, ms * 1000)
            steps.append((dt, step))

    if len(steps) < 2:
        raise RuntimeError(f"not enough step lines in {path}")

    # Pick the first entry whose step >= window, else fall back to first.
    start_idx = next(
        (i for i, (_, s) in enumerate(steps) if s >= window),
        0,
    )
    t_start, s_start = steps[start_idx]
    t_end, s_end = steps[-1]

    # Handle midnight wrap (extremely unlikely but cheap).
    delta = t_end - t_start
    if delta < timedelta(0):
        delta += timedelta(days=1)

    nsteps = s_end - s_start
    if nsteps <= 0:
        raise RuntimeError(
            f"non-positive step delta in {path}: start={s_start} end={s_end}"
        )
    return delta.total_seconds() / nsteps, s_start, s_end


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("log", help="solver log file (spdlog stdout)")
    p.add_argument("--window", type=int, default=5,
                   help="skip the first <window> steps as warmup")
    args = p.parse_args()
    try:
        rate, s_start, s_end = parse(args.log, args.window)
    except Exception as e:
        print(f"NaN", end="")
        print(f"  # {e}", file=sys.stderr)
        return 1
    print(f"{rate:.4f}", end="")
    print(f"  # window=[{s_start}, {s_end}]", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
