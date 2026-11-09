"""Final submissions V86 and V87 = V84 (end of the model chain) + the recorded Task 2 queue guard flips.

The two selected Kaggle submissions of team vacucity are
  V87 = work/submissions/v87_combined.zip         (public 0.90368)  V84 + 104 queue_pred flips
  V86 = work/submissions/v86_spacetime_guard.zip  (public 0.90359)  V84 +  94 queue_pred flips
The flips come from two guard procedures run by a team member on top of V84 (report section "Final queue guards"):
  V86_spacetime_guard  94 cells (62 validation, 32 private): spatio-temporal / adjacent-history gating of V84's
                       queue cells; a proposed flip was kept only if a CPU replay agreed with the submitted base
                       (4 ambiguous flips were blocked)
  V85_history_guard    10 cells (8 validation, 2 private): not-queued -> queued only, accepted only where the
                       window's own history gave enough evidence for the matching persistent-congestion type
  V87 = union of the two (disjoint submission_ids).
The V85 cells are recomputed by stage v85 (build_v85_history_guard.py: the guard's own code on frozen CPU
probabilities, results/v85_history_guard/V85_history_guard_changes.csv).  The V86 decisions ship as a frozen change
list (work/results/final_v86_v87/V86_spacetime_guard_changes.csv, one row per flipped cell with its value before and
after); V87 = the V86 rows + the recomputed V85 rows, asserted equal to the frozen V87 list (V87_changes.csv).
This script applies them to the rebuilt V84 and asserts, cell by cell,
that every listed cell is a queue row of V84 whose queue_pred equals the recorded "before" value; every other
byte of V84 is copied unchanged.  The files are written the way the submitted ones were (one member
"submission.csv", deflate, fixed 1980 timestamp, no forced zip64), so a rebuild normally matches the submitted
SHA-256 exactly; run_all.py verify also compares by content.
With TFB_TOLERANT=1 (another GPU, where V84's queue rows can differ in a few cells) a listed cell whose V84 value is
not the recorded "before" is printed as TOLERATED and still set to the recorded "after" value (the submitted one).

  python work/build_final_v86_v87.py [--out-dir work/submissions]
Refuses to overwrite its outputs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
import zipfile
from pathlib import Path

import repro_check as RC

WORK = Path(__file__).resolve().parent
BASE = WORK / "submissions" / "v84_observation_flow.zip"
DEC = WORK / "results" / "final_v86_v87"
LISTS = {"v86_spacetime_guard": DEC / "V86_spacetime_guard_changes.csv", "v87_combined": DEC / "V87_changes.csv"}
V85_COMPUTED = WORK / "results" / "v85_history_guard" / "V85_history_guard_changes.csv"
LIST_SHA = {"v86_spacetime_guard": "e5e7df967778c3192061833cfd9da30c739b465c89e07c7dbf09d6143cef1f29",
            "v87_combined": "4a150b7bef98f9848c9e8f64c38077b2b38cb53ab543cf6a1d308a0a6c0a941d"}
SUBMITTED_SHA = {"v86_spacetime_guard": "9e1ba0523c88b6ad70aeccf95c41e64a3c1639558c9c4a555867c75fc7cf1867",
                 "v87_combined": "6bc4e383967c9dbf96d8b343879b8b729436f3118636619bb113bde635e3dff0"}
EXPECT = {"v86_spacetime_guard": {"V86_spacetime_guard": 94},
          "v87_combined": {"V86_spacetime_guard": 94, "V85_history_guard": 10}}
ROWS = 6980503


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def read_list(name: str) -> dict[int, dict]:
    p = LISTS[name]
    got = sha256(p)
    assert got == LIST_SHA[name], f"{p} sha256 {got} != recorded {LIST_SHA[name]}"
    rows = {}
    with p.open(encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            sid = int(r["submission_id"])
            assert sid not in rows, f"duplicate submission_id {sid} in {p.name}"
            assert (r["before"], r["after"]) in (("0", "1"), ("1", "0")), r
            rows[sid] = r
    counts: dict[str, int] = {}
    for r in rows.values():
        counts[r["candidate"]] = counts.get(r["candidate"], 0) + 1
    assert counts == EXPECT[name], (name, counts)
    return rows


def build(name: str, flips: dict[int, dict], dest: Path) -> dict:
    if dest.exists():
        raise FileExistsError(dest)
    seen = 0
    before_mismatch = 0
    by = {}
    with zipfile.ZipFile(BASE) as zin, zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as zout:
        with zin.open(zin.namelist()[0]) as fin, zout.open("submission.csv", "w") as fout:
            header = fin.readline()
            assert header.rstrip(b"\r\n") == b"submission_id,task,speed_kmh,flow_vph,queue_pred,path_flow"
            fout.write(header)
            n = 0
            for raw in fin:
                n += 1
                sid = int(raw.split(b",", 1)[0])
                r = flips.get(sid)
                if r is None:
                    fout.write(raw)
                    continue
                body = raw.rstrip(b"\r\n")
                parts = body.split(b",")
                assert len(parts) == 6 and parts[1] == b"queue", (sid, body)
                if not RC.soft(parts[4] == r["before"].encode(), f"{name}: submission_id {sid}: V84 queue_pred "
                               f"{parts[4].decode()} != recorded before {r['before']} (with TFB_TOLERANT=1 it is "
                               f"set to the recorded after {r['after']})"):
                    before_mismatch += 1
                parts[4] = r["after"].encode()
                fout.write(b",".join(parts) + raw[len(body):])
                seen += 1
                k = (r["candidate"], r["split"], f'{r["before"]}->{r["after"]}')
                by[k] = by.get(k, 0) + 1
    assert n == ROWS, n
    assert seen == len(flips), (seen, len(flips))
    with zipfile.ZipFile(dest) as z:
        assert z.testzip() is None
    out = dict(rows=n, flips=seen, flips_by=[dict(candidate=a, split=b, flip=c, cells=v) for (a, b, c), v in sorted(by.items())])
    if before_mismatch:                  # only reachable with TFB_TOLERANT=1
        out["tolerated_before_mismatches"] = before_mismatch
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(WORK / "submissions"))
    a = ap.parse_args()
    out = Path(a.out_dir)
    t0 = time.time()
    v86, v87 = read_list("v86_spacetime_guard"), read_list("v87_combined")
    v85 = {}
    with V85_COMPUTED.open(encoding="utf-8", newline="") as f:   # stage v85 output
        for r in csv.DictReader(f):
            v85[int(r["submission_id"])] = r
    assert len(v85) == 10 and not (set(v85) & set(v86)), "V85 cells: 10, disjoint from V86"
    rebuilt_v87 = {**v86, **v85}
    key = ("candidate", "split", "panel", "window_id", "timestamp", "link_id", "before", "after")
    assert set(rebuilt_v87) == set(v87) and all(tuple(rebuilt_v87[s][k] for k in key) == tuple(v87[s][k] for k in key)
                                                for s in v87), "V86 list + recomputed V85 cells != the V87 list"
    v87 = rebuilt_v87
    v86_in_87 = {s: r for s, r in v87.items() if r["candidate"] == "V86_spacetime_guard"}
    assert set(v86_in_87) == set(v86) and all(
        (v86_in_87[s]["before"], v86_in_87[s]["after"]) == (v86[s]["before"], v86[s]["after"]) for s in v86), \
        "the V86 rows of the V87 list must equal the V86 list"
    from make_submission import validate
    report = dict(base=BASE.name, base_sha256=sha256(BASE), lists={k: LIST_SHA[k] for k in LISTS}, files={})
    for name, flips in (("v86_spacetime_guard", v86), ("v87_combined", v87)):
        dest = out / f"{name}.zip"
        info = build(name, flips, dest)
        checks = validate(dest)
        assert checks["ok"], checks
        h = sha256(dest)
        report["files"][name] = dict(**info, checks=checks, sha256=h, submitted_sha256=SUBMITTED_SHA[name],
                                     bytes_identical_to_submitted=h == SUBMITTED_SHA[name])
        print(name, info["flips"], "flips;", "sha256 == submitted" if h == SUBMITTED_SHA[name] else
              f"sha256 {h[:16]} != submitted {SUBMITTED_SHA[name][:16]} (verify compares by content)", flush=True)
    report["sources"] = {Path(__file__).name: sha256(Path(__file__))}
    report["seconds"] = round(time.time() - t0)
    man = out.parent / "results" / "final_v86_v87_manifest.json" if out == WORK / "submissions" else \
        out / "final_v86_v87_manifest.json"
    if man.exists():
        raise FileExistsError(man)
    man.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
