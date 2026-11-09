"""Stage v85: recompute the decisions of the V85 history guard from the rebuilt V84 and frozen CPU probabilities.

The V85 history guard (built by a team member on V84; public 0.90345 as a single file, not selected) is half of
the final V87.  Its code and inputs ship unchanged in work/v85_history_guard/ (the team member's reproduction
hand-off; see its 先读_复现交接.md).  On the 40 + 40 official ongoing windows it
  1. replays the CPU ensemble of each window from frozen probabilities (work/v85_history_guard/replay/work/
     v85_project/work/results/v85_experiment/*.npz) and the selection-rule conditioning (20,000 conditional
     samples, seed crc32(window_id)) on the 'constr' windows;
  2. retrieves the 30 most similar training-period histories (June 2030 - February 2031, asserted < 2031-03-01) for
     the window's own history up to and including its origin T (12 five-minute steps, at most 16 spatial bins:
     normalised speed, queue share, coverage; clock time within +-60 min);
  3. proposes not-queued -> queued only where the CPU replay suggests it, V84 also has 0, at least 20 historical
     analogues observe the cell and their queue share exceeds the CPU probability by at least 0.30.
This stage
  a. writes V84's queue rows (submission_id, split, panel, window_id, timestamp, link_id, queue_pred) from the
     rebuilt work/submissions/v84_observation_flow.zip and asserts they equal the rows the team member used;
  b. runs the guard's audit script on them (clean_data/, kaggle_public/) and asserts that its three audit files equal
     the shipped originals (line endings ignored);
  c. derives the selected cells from the recomputed audit and asserts that they equal the official V85 change list
     and the V85 rows of the V87 change list; writes them for stage `final`.
With TFB_TOLERANT=1 (another GPU: V84's queue rows can differ in a few cells) a difference in a., b. or c., or a
failure of the audit script, is printed as TOLERATED and the stage writes the recorded official V85 cells instead
(the V86 cells are applied from their recorded list in the same way); manifest.json records which one was used.

  python work/build_v85_history_guard.py
Refuses to run if its output folder already holds files.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import sys
import time
import zipfile
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

import repro_check as RC

WORK = Path(__file__).resolve().parent
ROOT = WORK.parent
H = WORK / "v85_history_guard"
OUT = WORK / "results" / "v85_history_guard"
V84 = WORK / "submissions" / "v84_observation_flow.zip"
KEY = ROOT / "kaggle_public" / "submission_key.csv"
QUEUE_SHA_LF = "b5e146f086282da754324afdf5277cd1b998686b734a3d223e1095b95514068e"   # hand-off inputs/v84_queue.csv
AUDIT_FILES = ("official_gate_windows_audit.csv", "official_gate_cells_audit.csv", "official_gate_summary.json")
OFFICIAL_LIST = H / "inputs" / "V85_history_guard_changes.csv"
V87_LIST = WORK / "results" / "final_v86_v87" / "V87_changes.csv"
FIELDS = ("candidate", "submission_id", "split", "panel", "window_id", "timestamp", "link_id", "before", "after")


def sha_lf(p: Path) -> str:
    return hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def write_queue(path: Path) -> tuple[int, bool]:
    """V84's queue rows joined with the submission key (as v85_run_decisions.submitted() built them); returns the row
    count and whether they equal the team member's rows."""
    chunks = []
    with zipfile.ZipFile(V84) as zz, zz.open(zz.namelist()[0]) as f:
        for key, val in zip(pd.read_csv(KEY, chunksize=300000, low_memory=False), pd.read_csv(f, chunksize=300000)):
            assert (key.submission_id.to_numpy() == val.submission_id.to_numpy()).all()
            m = key.task == "queue"
            if m.any():
                a = key.loc[m, ["submission_id", "split", "panel", "window_id", "timestamp", "link_id"]].copy()
                a["queue_pred"] = val.loc[m, "queue_pred"].to_numpy()
                chunks.append(a)
    q = pd.concat(chunks)
    q.to_csv(path, index=False, lineterminator="\n")
    got = sha_lf(path)
    same = RC.soft(got == QUEUE_SHA_LF, f"V84 queue rows differ from the team member's ({got[:16]} != "
                                        f"{QUEUE_SHA_LF[:16]})")
    return len(q), same


def run_gate(queue: Path, audit: Path) -> dict:
    audit.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, V85_HANDOFF_CLEAN=str(ROOT / "clean_data"), V85_HANDOFF_RELEASE=str(ROOT / "kaggle_public"),
               V85_HANDOFF_AUDIT=str(audit), V85_HANDOFF_QUEUE=str(queue), TFB_EXPERIMENT="v85_experiment",
               PYTHONIOENCODING="utf-8", PYTHONDONTWRITEBYTECODE="1")
    subprocess.run([sys.executable, str(H / "replay" / "work" / "analog_gate_official_audit.py")], env=env, cwd=H,
                   check=True)
    cmp = {}
    for name in AUDIT_FILES:
        same = (audit / name).read_bytes().replace(b"\r\n", b"\n") == (H / "audit" / name).read_bytes().replace(b"\r\n", b"\n")
        cmp[name] = same
        print(f"audit {name}: {'identical to the original (line endings ignored)' if same else 'DIFFERS'}", flush=True)
    RC.soft(all(cmp.values()), f"recomputed V85 audit differs from the original: {cmp}")
    return cmp


def rows(p: Path) -> list[dict]:
    with p.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def select(audit: Path, queue: Path) -> list[dict]:
    """The gate of the audit applied again (valid_history >= 20 and probability_gap >= 0.30), mapped to V84 rows."""
    windows = {r["window_id"]: r for r in rows(audit / "official_gate_windows_audit.csv")}
    assert len(windows) == 80 and Counter(r["split"] for r in windows.values()) == {"validation": 40, "private": 40}
    cand = {}
    for r in rows(audit / "official_gate_cells_audit.csv"):
        w = windows[r["window_id"]]
        assert w["cls"] == "constr" and all(r[k] == w[k] for k in ("panel", "split", "date", "T"))
        assert (r["before"], r["after"]) == ("0", "1") and r["baseline_replay_mismatch"] == "False"
        valid, gap = int(r["valid_history"]), float(r["probability_gap"])
        assert abs(float(r["history_probability"]) - float(r["cpu_probability"]) - gap) <= 1e-7
        gated = valid >= 20 and gap >= 0.30
        assert r["gated"] == str(gated), r
        ts = datetime.fromisoformat(r["date"] + "T00:00:00+00:00") + timedelta(minutes=5 * int(r["T"]) +
                                                                                int(r["horizon_minutes"]))
        k = (r["split"], r["panel"], r["window_id"], r["link_id"], ts)
        assert k not in cand
        cand[k] = gated
    assert len(cand) == 12, len(cand)
    sid = {}
    for r in rows(queue):
        k = (r["split"], r["panel"], r["window_id"], r["link_id"],
             datetime.fromisoformat(r["timestamp"].replace("Z", "+00:00")))
        if k in cand:
            assert k not in sid and r["queue_pred"] == "0", (k, r["queue_pred"])
            sid[k] = int(r["submission_id"])
    assert sid.keys() == cand.keys()
    out = [dict(candidate="V85_history_guard", submission_id=str(sid[k]), split=k[0], panel=k[1], window_id=k[2],
                timestamp=str(k[4]), link_id=k[3], before="0", after="1") for k, g in cand.items() if g]
    assert Counter(r["split"] for r in out) == {"validation": 8, "private": 2}, Counter(r["split"] for r in out)
    return sorted(out, key=lambda r: int(r["submission_id"]))


def as_set(rs: list[dict]) -> set[tuple]:
    return {tuple(r[f] for f in FIELDS) for r in rs}


def main() -> None:
    old = [p for p in OUT.rglob("*") if p.is_file()] if OUT.exists() else []
    if old:                          # run_all.py creates the (empty) output folders before the stage starts
        raise FileExistsError(f"{OUT} already holds outputs, e.g. {old[0]}")
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    queue = OUT / "v84_queue.csv"
    n, queue_same = write_queue(queue)
    print(f"V84 queue rows: {n:,}, {'identical to' if queue_same else 'DIFFERENT from'} the team member's "
          "v84_queue.csv", flush=True)
    official_rows = rows(OFFICIAL_LIST)
    official = as_set(official_rows)
    in_v87 = as_set([r for r in rows(V87_LIST) if r["candidate"] == "V85_history_guard"])
    assert official == in_v87, "the official V85 list and the V85 rows of the V87 list differ"
    fallback, cmp = None, {}
    try:
        cmp = run_gate(queue, OUT / "gate")
        sel = select(OUT / "gate", queue)
        if not RC.soft(as_set(sel) == official, "recomputed V85 cells differ from the official V85 change list "
                       "(and so from the V85 rows of the V87 change list)"):
            fallback = "recomputed cells differ from the recorded list"
    except (AssertionError, subprocess.CalledProcessError) as e:
        if not RC.tolerant():
            raise
        fallback = f"recomputation stopped: {type(e).__name__}: {str(e)[:300]}"
        print(f"TOLERATED (TFB_TOLERANT=1): {fallback}", flush=True)
    if fallback:
        sel = sorted(({f: r[f] for f in FIELDS} for r in official_rows), key=lambda r: int(r["submission_id"]))
        print(f"TOLERATED (TFB_TOLERANT=1): writing the {len(sel)} recorded official V85 cells instead", flush=True)
    with (OUT / "V85_history_guard_changes.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n")
        w.writeheader()
        w.writerows(sel)
    man = dict(v84_sha256=hashlib.sha256(V84.read_bytes()).hexdigest(), queue_rows=n, queue_sha256_lf=QUEUE_SHA_LF,
               audit_identical=cmp, selected_cells=len(sel), by_split=dict(Counter(r["split"] for r in sel)),
               equals_official_v85_list=as_set(sel) == official, equals_v85_rows_of_v87_list=as_set(sel) == in_v87,
               v84_queue_identical=queue_same, tolerated_fallback_to_recorded_list=fallback,
               sources={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                        [Path(__file__), *sorted((H / "replay" / "work").glob("*.py"))]},
               seconds=round(time.time() - t0))
    (OUT / "manifest.json").write_text(json.dumps(man, indent=2), encoding="utf-8")
    print(json.dumps(man, indent=2), flush=True)


if __name__ == "__main__":
    main()
