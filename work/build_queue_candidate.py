"""Patch only queue rows of a validated submission, then validate the whole ZIP.

Example:
  python work/build_queue_candidate.py --name v27_candidate \
    --queue work/sub/queue_validation_v27.csv work/sub/queue_private_v27.csv

State and ODME rows are copied byte for byte from v26. No hidden labels are read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate import RELEASE, ROOT
from make_submission import validate

WORK = ROOT / "work"
KEYS = ["window_id", "timestamp", "link_id"]
HEADER = b"submission_id,task,speed_kmh,flow_vph,queue_pred,path_flow"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def build(args) -> None:
    if Path(args.name).name != args.name or not args.name:
        raise ValueError("name must be a filename without directories")
    out = WORK / "submissions" / f"{args.name}.zip"
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite {out}")
    candidate = pd.concat([pd.read_csv(p) for p in args.queue], ignore_index=True)
    if candidate[KEYS].isna().any().any() or candidate.duplicated(KEYS).any():
        raise ValueError("Null or duplicate queue keys")
    if not candidate.queue_pred.isin([0, 1]).all():
        raise ValueError("Queue predictions must be finite 0/1 values")
    cols = ["submission_id", "task", *KEYS]
    chunks = []
    key_dtypes = {"submission_id": np.int64, "task": str, **{k: str for k in KEYS}}
    for c in pd.read_csv(RELEASE / "submission_key.csv", usecols=cols, dtype=key_dtypes, chunksize=500_000):
        chunks.append(c[c.task == "queue"].copy())
    target = pd.concat(chunks, ignore_index=True)
    matched = target.merge(candidate, on=KEYS, how="outer", validate="one_to_one", indicator=True)
    if not matched._merge.eq("both").all():
        raise ValueError("Candidate must cover exactly all official queue keys in both splits")
    lookup = dict(zip(matched.submission_id.astype(np.int64), matched.queue_pred.astype(int)))
    changed = seen = total = 0
    source_nonqueue = hashlib.sha256()
    output_nonqueue = hashlib.sha256()
    changes = []
    with zipfile.ZipFile(args.base) as zin:
        if len(zin.namelist()) != 1:
            raise ValueError("Expected a single CSV member in the source archive")
        with zin.open(zin.namelist()[0]) as src, zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zout:
            with zout.open(f"{args.name}.csv", "w", force_zip64=True) as dst:
                header = src.readline()
                if header.strip() != HEADER:
                    raise ValueError("Unexpected source CSV schema")
                dst.write(header)
                for line in src:
                    total += 1
                    parts = line.rstrip(b"\r\n").split(b",")
                    if len(parts) != 6:
                        raise ValueError(f"Invalid source CSV row {total}")
                    if parts[1] == b"queue":
                        sid = int(parts[0])
                        pred = lookup.pop(sid)
                        seen += 1
                        if int(parts[4]) != pred:
                            changes.append({"submission_id": sid, "old": int(parts[4]), "new": pred})
                            changed += 1
                            parts[4] = str(pred).encode("ascii")
                            line = b",".join(parts) + b"\n"
                    else:
                        source_nonqueue.update(line)
                        output_nonqueue.update(line)
                    dst.write(line)
    if lookup or not seen:
        raise ValueError("Source submission did not contain all expected queue IDs")
    with zipfile.ZipFile(out) as z:
        if z.testzip() is not None:
            raise ValueError("ZIP integrity check failed")
    checks = validate(out)
    if not checks["ok"]:
        raise ValueError(f"Full submission validation failed: {checks}")
    change_frame = pd.DataFrame(changes, columns=["submission_id", "old", "new"])
    change_frame = change_frame.merge(target.drop(columns="task"), on="submission_id", how="left")
    change_path = WORK / "results" / f"{args.name}_changes.csv"
    change_frame.to_csv(change_path, index=False)
    manifest = {
        "name": args.name, "note": args.note,
        "baseline": str(args.base), "baseline_sha256": sha256(args.base),
        "queue_inputs": [{"path": str(p), "sha256": sha256(p)} for p in args.queue],
        "output": str(out), "sha256": sha256(out), "rows": total,
        "queue_rows": seen, "changed_queue_cells": changed,
        "state_odme_unchanged": source_nonqueue.hexdigest() == output_nonqueue.hexdigest(),
        "nonqueue_rows_sha256": source_nonqueue.hexdigest(),
        "checks": checks, "public_lb": None,
    }
    (WORK / "results" / f"{args.name}_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    log = WORK / "results" / "submissions.csv"
    pd.DataFrame([{
        "time": time.strftime("%Y-%m-%d %H:%M:%S"), "name": args.name, "note": args.note,
        "state": "work/sub/state_validation_vp.csv;work/sub/state_private_vp.csv",
        "queue": ";".join(map(str, args.queue)),
        "odme": "work/baseline/task4_validation/public_reference.csv;work/baseline/task4_private/public_reference.csv",
        "public_lb": "",
    }]).to_csv(log, mode="a", header=not log.exists(), index=False)
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--name", required=True)
    ap.add_argument("--base", type=Path, default=WORK / "submissions" / "v26_rnn2.zip")
    ap.add_argument("--queue", type=Path, nargs=2, required=True)
    ap.add_argument("--note", default="")
    build(ap.parse_args())
