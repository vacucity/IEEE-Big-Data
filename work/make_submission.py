#!/usr/bin/env python
"""Assemble the Kaggle upload file from per-task prediction files, validate it, and zip it.

Same result as the official src/merge_submissions.py, but memory-light: cells are matched on
integer keys and submission_key.csv is streamed, so it runs in well under 1 GB of RAM.

Each task takes one or more files in the official per-task format (validation and private can be
separate files):
  state  panel,timestamp,station_id,link_id,mask_regime,speed_kmh,flow_vph
  queue  window_id,timestamp,link_id,queue_pred
  odme   panel,departure_time,path_id,origin_zone,destination_zone,path_flow

Usage (from the "IEEE Big data" folder):
  python work/make_submission.py --name v1 \
      --state work/baseline/state_validation.csv work/baseline/state_private.csv \
      --queue work/sub/queue_validation.csv work/sub/queue_private.csv \
      --odme  work/baseline/task4_validation/public_reference.csv work/baseline/task4_private/public_reference.csv \
      --note  "T2 rules + T4 own-month counts"
Writes work/submissions/<name>.zip and appends a row to work/results/submissions.csv.
"""
from __future__ import annotations

import argparse
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import KEY_BASE, PANELS, RELEASE, REGIMES, RESULTS, ROOT, links, slot_of  # noqa: E402

OUT = ROOT / "work" / "submissions"
CHUNK = 500_000
REGIME_CODE = {r: i for i, r in enumerate(REGIMES)}
POS_OF = {p: dict(zip(links(p).link_id, links(p).pos)) for p in PANELS}


def cell_keys(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Integer key (slot * KEY_BASE + link position) per row, and a validity mask."""
    keys = np.full(len(frame), -1, np.int64)
    for p, idx in frame.groupby("panel").indices.items():
        pos = frame.link_id.iloc[idx].map(POS_OF.get(p, {}))
        ok = pos.notna().to_numpy()
        keys[idx[ok]] = slot_of(frame.timestamp.iloc[idx[ok]]) * KEY_BASE + pos[ok].astype(np.int64).to_numpy()
    return keys, keys >= 0


def load_state(paths: list[Path]) -> dict[str, tuple]:
    parts = []
    for path in paths:
        for c in pd.read_csv(path, chunksize=2_000_000, dtype={"panel": str, "link_id": str, "mask_regime": str},
                             usecols=["panel", "timestamp", "link_id", "mask_regime", "speed_kmh", "flow_vph"]):
            k, ok = cell_keys(c)
            if not ok.all():
                raise SystemExit(f"{path}: {int((~ok).sum())} rows with an unknown panel or link")
            parts.append(pd.DataFrame({"panel": c.panel.to_numpy(), "key": k,
                                       "regime": c.mask_regime.map(REGIME_CODE).to_numpy(),
                                       "speed": c.speed_kmh.to_numpy(float), "flow": c.flow_vph.to_numpy(float)}))
    d = pd.concat(parts, ignore_index=True)
    table = {}
    for p, g in d.groupby("panel"):
        g = g.sort_values("key")
        if g.key.duplicated().any():
            raise SystemExit(f"state: {int(g.key.duplicated().sum())} duplicate cells in {p}")
        table[p] = (g.key.to_numpy(), g.regime.to_numpy(), g.speed.to_numpy(), g.flow.to_numpy())
    return table


def load_queue(paths: list[Path]) -> pd.DataFrame:
    d = pd.concat([pd.read_csv(p, dtype={"window_id": str, "link_id": str}) for p in paths], ignore_index=True)
    bad = d.queue_pred[~d.queue_pred.isin([0, 1])]
    if len(bad):
        raise SystemExit(f"queue: {len(bad)} values that are not 0/1")
    d["slot"] = slot_of(d.timestamp)
    if d.duplicated(["window_id", "slot", "link_id"]).any():
        raise SystemExit("queue: duplicate (window_id, timestamp, link_id) rows")
    return d[["window_id", "slot", "link_id", "queue_pred"]]


def load_odme(paths: list[Path]) -> pd.DataFrame:
    d = pd.concat([pd.read_csv(p, dtype=str) for p in paths], ignore_index=True)
    d["path_flow"] = pd.to_numeric(d.path_flow)
    if d.duplicated(["panel", "departure_time", "path_id"]).any():
        raise SystemExit("odme: duplicate (panel, departure_time, path_id) rows")
    return d[["panel", "departure_time", "path_id", "path_flow"]]


def assemble(state: dict, queue: pd.DataFrame, odme: pd.DataFrame, out_csv: Path) -> dict:
    stats = {"state_missing": 0, "state_regime_mismatch": 0, "queue_missing": 0, "odme_missing": 0}
    first = True
    cols = ["submission_id", "task", "panel", "timestamp", "link_id", "mask_regime", "window_id",
            "departure_time", "path_id"]
    for key in pd.read_csv(RELEASE / "submission_key.csv", usecols=cols, dtype=str, chunksize=CHUNK):
        out = pd.DataFrame({"submission_id": key.submission_id.astype(np.int64), "task": key.task,
                            "speed_kmh": 0.0, "flow_vph": 0.0, "queue_pred": 0, "path_flow": 0.0})
        s = (key.task == "state").to_numpy()
        if s.any():
            ks = key[s]
            k, ok = cell_keys(ks)
            speed = np.full(len(ks), np.nan)
            flow = np.full(len(ks), np.nan)
            for p, idx in ks.groupby("panel").indices.items():
                if p not in state:
                    continue
                tk, treg, tsp, tfl = state[p]
                j = np.searchsorted(tk, k[idx]).clip(0, len(tk) - 1)
                hit = (tk[j] == k[idx]) & ok[idx]
                same = hit & (treg[j] == ks.mask_regime.iloc[idx].map(REGIME_CODE).to_numpy())
                stats["state_regime_mismatch"] += int((hit & ~same).sum())
                speed[idx[same]], flow[idx[same]] = tsp[j[same]], tfl[j[same]]
            miss = ~np.isfinite(speed) | ~np.isfinite(flow)
            stats["state_missing"] += int(miss.sum())
            out.loc[s, "speed_kmh"] = np.where(miss, 0.0, speed)
            out.loc[s, "flow_vph"] = np.where(miss, 0.0, flow)
        q = (key.task == "queue").to_numpy()
        if q.any():
            kq = key.loc[q, ["window_id", "timestamp", "link_id"]].copy()
            kq["slot"] = slot_of(kq.timestamp)
            m = kq.merge(queue, on=["window_id", "slot", "link_id"], how="left")
            stats["queue_missing"] += int(m.queue_pred.isna().sum())
            out.loc[q, "queue_pred"] = m.queue_pred.fillna(0).astype(int).to_numpy()
        o = (key.task == "odme").to_numpy()
        if o.any():
            m = key.loc[o, ["panel", "departure_time", "path_id"]].merge(odme, on=["panel", "departure_time", "path_id"],
                                                                        how="left")
            stats["odme_missing"] += int(m.path_flow.isna().sum())
            out.loc[o, "path_flow"] = m.path_flow.fillna(0.0).to_numpy()
        out.to_csv(out_csv, mode="w" if first else "a", header=first, index=False, float_format="%.6g")
        first = False
    return stats


def validate(out_csv: Path) -> dict:
    """The checks Kaggle and the evaluator apply, against sample_submission.csv."""
    cats = pd.CategoricalDtype(["odme", "queue", "state"])
    sub = pd.read_csv(out_csv, dtype={"task": cats})
    tmpl = pd.read_csv(RELEASE / "sample_submission.csv", usecols=["submission_id", "task"], dtype={"task": cats})
    st, q, o = sub.task == "state", sub.task == "queue", sub.task == "odme"
    same_len = len(sub) == len(tmpl)
    checks = {
        "rows_match": same_len,
        "ids_and_order_match": same_len and bool((sub.submission_id.to_numpy() == tmpl.submission_id.to_numpy()).all()),
        "tasks_match": same_len and bool((sub.task.cat.codes.to_numpy() == tmpl.task.cat.codes.to_numpy()).all()),
        "no_blank": not sub.isna().any().any(),
        "no_duplicate_id": not sub.submission_id.duplicated().any(),
        "queue_binary": bool(sub.loc[q, "queue_pred"].isin([0, 1]).all()),
        "path_flow_nonneg_finite": bool(np.isfinite(sub.loc[o, "path_flow"]).all() and (sub.loc[o, "path_flow"] >= 0).all()),
        "state_speed_positive": bool((sub.loc[st, "speed_kmh"] > 0).all()),
        "state_low_flow_share": round(float((sub.loc[st, "flow_vph"] < 50).mean()), 4),   # Task 3 voids S_FD above 0.20
    }
    checks["ok"] = all(v for k, v in checks.items() if k != "state_low_flow_share") and checks["state_low_flow_share"] <= 0.2
    return checks


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--state", nargs="+", type=Path, required=True)
    ap.add_argument("--queue", nargs="+", type=Path, required=True)
    ap.add_argument("--odme", nargs="+", type=Path, required=True)
    ap.add_argument("--note", default="")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    out_csv = OUT / f"{args.name}.csv"
    t0 = time.time()
    stats = assemble(load_state(args.state), load_queue(args.queue), load_odme(args.odme), out_csv)
    print("merge:", stats)
    checks = validate(out_csv)
    print("checks:", checks)
    if not checks["ok"] or any(stats.values()):
        raise SystemExit("NOT ready to upload: fix the issues above")
    zpath = OUT / f"{args.name}.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(out_csv, arcname=f"{args.name}.csv")
    out_csv.unlink()
    RESULTS.mkdir(parents=True, exist_ok=True)
    log = RESULTS / "submissions.csv"
    pd.DataFrame([{"time": time.strftime("%Y-%m-%d %H:%M:%S"), "name": args.name, "note": args.note,
                   "state": ";".join(map(str, args.state)), "queue": ";".join(map(str, args.queue)),
                   "odme": ";".join(map(str, args.odme)), "public_lb": ""}]).to_csv(
        log, mode="a", header=not log.exists(), index=False)
    print(f"ready: {zpath} ({zpath.stat().st_size / 1e6:.0f} MB, {time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
