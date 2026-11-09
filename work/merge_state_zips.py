#!/usr/bin/env python
"""Merge two state-only improvements that were both built on the same base submission.

  python work/merge_state_zips.py --base v56_t1_on54 --a v57_rs --b v57_blk --name v58_rs_blk

All three ZIPs must have identical row order.  For every line: if b differs from base, take b; else take a.
Asserts that a and b never change the same line, and that only state rows differ from the base.  The result is
validated with make_submission.validate and a manifest is written to work/results/<name>_manifest.json.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import zipfile
from itertools import zip_longest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_submission import validate  # noqa: E402

WORK = Path(__file__).resolve().parent
SUB = WORK / "submissions"


def lines(name):
    z = zipfile.ZipFile(SUB / f"{name}.zip")
    assert len(z.namelist()) == 1
    with z.open(z.namelist()[0]) as f:
        for raw in io.BufferedReader(f, 1 << 22):
            yield raw


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for blk in iter(lambda: f.read(1 << 22), b""):
            h.update(blk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--note", default="")
    args = ap.parse_args()
    out = SUB / f"{args.name}.zip"
    if out.exists():
        raise FileExistsError(out)
    stats = {"rows": 0, "from_a": 0, "from_b": 0, "a_by_task": {}, "b_by_task": {}}
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zout, \
            zout.open(f"{args.name}.csv", "w", force_zip64=True) as dst:
        for i, (l0, la, lb) in enumerate(zip_longest(lines(args.base), lines(args.a), lines(args.b))):
            assert l0 is not None and la is not None and lb is not None, f"line {i}: the three files differ in length"
            if i == 0:
                assert l0 == la == lb, "headers differ"
                dst.write(l0)
                continue
            stats["rows"] += 1
            ca, cb = la != l0, lb != l0
            assert not (ca and cb), f"line {i}: both {args.a} and {args.b} change it"
            task = l0.split(b",", 2)[1].decode()
            if ca:
                assert task == "state", f"line {i}: {args.a} changes a {task} row"
                stats["from_a"] += 1
                stats["a_by_task"][task] = stats["a_by_task"].get(task, 0) + 1
            if cb:
                assert task == "state", f"line {i}: {args.b} changes a {task} row"
                stats["from_b"] += 1
                stats["b_by_task"][task] = stats["b_by_task"].get(task, 0) + 1
            dst.write(lb if cb else la)
    with zipfile.ZipFile(out) as z:
        assert z.testzip() is None
    checks = validate(out)
    if not checks["ok"]:
        raise ValueError(f"validation failed: {checks}")
    manifest = dict(name=args.name, note=args.note, base=args.base, a=args.a, b=args.b,
                    inputs_sha256={n: sha256(SUB / f"{n}.zip") for n in (args.base, args.a, args.b)},
                    output=str(out), sha256=sha256(out), stats=stats, checks=checks, public_lb=None)
    (WORK / "results" / f"{args.name}_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("output", "sha256", "stats")}, indent=2))
    print(checks)


if __name__ == "__main__":
    main()
