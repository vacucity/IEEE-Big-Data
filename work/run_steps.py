#!/usr/bin/env python
"""Run several work/ scripts one after another, each with its own log, stopping at the first failure.

Meant to be started as a detached process so long jobs survive the terminal that launched them:
  python work/run_steps.py --name chain "task1_gbm.py cvk --panels D7_I10_E" "task1_gbm.py fit --lr 0.05"
Logs: work/results/<name>_<i>.log; a final <name>.status file says done or which step failed.
"""
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("steps", nargs="+", help='each step: "<script.py> <args...>", relative to work/')
    args = ap.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    status = RESULTS / f"{args.name}.status"
    status.write_text("running\n")
    for i, step in enumerate(args.steps, 1):
        parts = shlex.split(step)
        log = RESULTS / f"{args.name}_{i}.log"
        t0 = time.time()
        with log.open("w") as fh:
            rc = subprocess.call([sys.executable, "-u", "-W", "ignore::RuntimeWarning", str(HERE / parts[0]), *parts[1:]],
                                 cwd=HERE.parent, stdout=fh, stderr=subprocess.STDOUT)
        with status.open("a") as fh:
            fh.write(f"step {i} rc={rc} {time.time() - t0:.0f}s: {step}\n")
        if rc != 0:
            status.open("a").write("failed\n")
            sys.exit(rc)
    with status.open("a") as fh:
        fh.write("done\n")


if __name__ == "__main__":
    main()
