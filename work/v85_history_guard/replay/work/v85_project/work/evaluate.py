#!/usr/bin/env python
"""Local evaluation for TrafficFlowBench, built on clean_data/ (see work/clean_data.py).

What each command scores, and how close it is to the leaderboard:
  t1  Task 1 S_state    exact replica of the official formula, on train (the only split with truth)
  t2  Task 2 S_queue    proxy on the 80 train windows: labels from speed_true <= 0.6 * free speed on
                        eligible cells (the leaderboard labels the noise-free state instead)
  t3  Task 3 S_physics  S_FD exact. S_LWR approximated: the organizer boundary flows are withheld, so
                        the conservation right-hand side is replaced by the observed dN, plus the
                        ~0.045 residual that an exact answer still carries on the leaderboard
  t4  Task 4 S_link     exact replica (the other 75% of S_ODME needs the withheld path flows)

Usage (from the "IEEE Big data" folder):
  python work/evaluate.py t1 --submission state_train.csv [--panels D12_I5_N]
  python work/evaluate.py t3 --submission state_train.csv --panels D12_I5_N
  python work/evaluate.py t2 --submission queue_train.csv
  python work/evaluate.py t4 --submission odme.csv --split validation
  python work/evaluate.py selftest        # checks the evaluators against known numbers

From Python, without a CSV round trip:
  from evaluate import t1_targets, score_t1_panel, t3_proxy_panel
"""
from __future__ import annotations

import argparse
import os
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
RELEASE = Path(os.environ["V85_HANDOFF_RELEASE"])
CLEAN = Path(os.environ["V85_HANDOFF_CLEAN"])
CACHE = ROOT / "work" / "cache"
RESULTS = ROOT / "work" / "results"

REGIMES = ("R1", "R2", "R3")
STEPS = 288
KEY_BASE = 1000                       # cell key = absolute 5-minute slot * KEY_BASE + link position
SPEED_NORM, FLOW_NORM = 25.0, 600.0   # Task 1 normalizers (km/h, veh/h/lane)
SPEED_W, FLOW_W = 0.54, 0.46
QUEUE_RATIO = 0.60
EMPTY_FLOOR_VPH, EMPTY_SHARE = 50.0, 0.20   # Task 3 empty-road guard
LWR_RESIDUAL_FLOOR = 0.045   # E_LWR of an exact answer on the leaderboard (S_physics ~0.96 with S_FD ~0.98)

_manifest = json.loads((RELEASE / "config" / "corridors.json").read_text(encoding="utf-8"))["panels"]
PANELS = [p["corridor_id"] for p in _manifest]
FAMILY = {p["corridor_id"]: p["family_id"] for p in _manifest}
T2_PANELS = [p for p in PANELS if not p.startswith("D12_I405")]


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------
def links(panel: str) -> pd.DataFrame:
    return pd.read_parquet(CLEAN / panel / "links.parquet").sort_values("pos").reset_index(drop=True)


def slot_of(text: pd.Series) -> np.ndarray:
    """Absolute 5-minute slot of timestamp text ('...T..Z' or '... +00:00')."""
    ts = pd.to_datetime(text, utc=True, format="ISO8601")
    return ts.values.astype("datetime64[m]").astype(np.int64) // 5


def aggregate(panel_scores: dict[str, float]) -> tuple[dict[str, float], float]:
    """Leaderboard aggregation: equal mean within a corridor family, then across families."""
    fam: dict[str, list[float]] = {}
    for p, s in panel_scores.items():
        fam.setdefault(FAMILY[p], []).append(s)
    fam_mean = {f: float(np.mean(v)) for f, v in sorted(fam.items())}
    return fam_mean, float(np.mean(list(fam_mean.values()))) if fam_mean else float("nan")


def log_result(task: str, split: str, submission: str, tag: str, overall: float, detail: dict) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    row = pd.DataFrame([{"time": time.strftime("%Y-%m-%d %H:%M:%S"), "task": task, "split": split,
                         "submission": submission, "tag": tag, "overall": round(overall, 6),
                         "detail": json.dumps(detail, default=float)}])
    path = RESULTS / "history.csv"
    row.to_csv(path, mode="a", header=not path.exists(), index=False)


# ---------------------------------------------------------------------------
# Task 1: target cells, exact S_state
# ---------------------------------------------------------------------------
def t1_targets(panel: str, split: str = "train") -> dict[str, np.ndarray]:
    """Task 1 target cells in clean_data order: key, regime code, and truth (train only). Cached."""
    src = CLEAN / panel / f"{split}.parquet"
    path = CACHE / f"t1_{panel}_{split}.npz"
    if path.exists() and path.stat().st_mtime >= src.stat().st_mtime:
        return dict(np.load(path))
    cols = ["ts", "pos", "mask_regime", "is_t1_target"] + (["speed_true", "flow_true"] if split == "train" else [])
    out = {"key": [], "regime": [], "speed_true": [], "flow_true": []}
    pf = pq.ParquetFile(src)
    for i in range(pf.num_row_groups):
        t = pf.read_row_group(i, columns=cols)
        m = t.column("is_t1_target").to_numpy()
        if not m.any():
            continue
        slot = t.column("ts").to_numpy().astype("datetime64[m]").astype(np.int64)[m] // 5
        out["key"].append(slot * KEY_BASE + t.column("pos").to_numpy()[m].astype(np.int64))
        regime = REGIMES.index(t.column("mask_regime")[0].as_py())
        out["regime"].append(np.full(int(m.sum()), regime, np.int8))
        for c in ("speed_true", "flow_true"):
            out[c].append(t.column(c).to_numpy()[m].astype(np.float32) if split == "train"
                          else np.full(int(m.sum()), np.nan, np.float32))
    res = {k: np.concatenate(v) for k, v in out.items()}
    assert np.all(np.diff(res["key"]) > 0), "target keys must be strictly increasing"
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(path, **res)
    return res


def score_t1_panel(panel: str, tg: dict, pred_speed: np.ndarray, pred_flow: np.ndarray) -> dict:
    """Official S_state for one panel. Predictions are aligned with t1_targets(panel)."""
    lanes = links(panel).lanes.to_numpy(float)[tg["key"] % KEY_BASE]
    ps = np.where(np.isfinite(pred_speed), pred_speed, 0.0)   # missing prediction = 0, as the scorer does
    pf = np.where(np.isfinite(pred_flow), pred_flow, 0.0)
    row = {"panel": panel, "missing": int((~np.isfinite(pred_speed) | ~np.isfinite(pred_flow)).sum())}
    scores = []
    for r, name in enumerate(REGIMES):
        m = (tg["regime"] == r) & np.isfinite(tg["speed_true"]) & np.isfinite(tg["flow_true"])
        if not m.any():
            rs = rq = float("nan")
            s = 0.0
        else:
            rs = float(np.sqrt(np.mean((ps[m] - tg["speed_true"][m]) ** 2)))
            rq = float(np.sqrt(np.mean(((pf[m] - tg["flow_true"][m]) / lanes[m]) ** 2)))
            s = SPEED_W * max(0.0, 1 - rs / SPEED_NORM) + FLOW_W * max(0.0, 1 - rq / FLOW_NORM)
        row.update({f"{name}_n": int(m.sum()), f"{name}_rmse_speed": rs, f"{name}_rmse_flow_lane": rq, f"{name}_S": s})
        scores.append(s)
    row["S_state"] = float(np.mean(scores))
    return row


def read_t1_csv(path: Path, split: str, panels: list[str]) -> tuple[dict, dict]:
    """Read a Task 1 submission in the official per-task format, aligned to the cached targets."""
    tgs = {p: t1_targets(p, split) for p in panels}
    preds = {p: (np.full(len(tgs[p]["key"]), np.nan, np.float64), np.full(len(tgs[p]["key"]), np.nan, np.float64))
             for p in panels}
    seen = {p: np.zeros(len(tgs[p]["key"]), np.int8) for p in panels}
    pos_of = {p: dict(zip(links(p).link_id, links(p).pos)) for p in panels}
    stats = {"rows": 0, "rows_other_panels": 0, "rows_not_target": 0, "regime_mismatch": 0, "duplicates": 0}
    for chunk in pd.read_csv(path, chunksize=2_000_000,
                             usecols=["panel", "timestamp", "link_id", "mask_regime", "speed_kmh", "flow_vph"]):
        stats["rows"] += len(chunk)
        for p, c in chunk.groupby("panel"):
            if p not in tgs:
                stats["rows_other_panels"] += len(c)
                continue
            tg = tgs[p]
            pos = c.link_id.map(pos_of[p])
            ok = pos.notna().to_numpy()
            key = slot_of(c.timestamp[ok]) * KEY_BASE + pos[ok].astype(np.int64).to_numpy()
            idx = np.searchsorted(tg["key"], key).clip(0, len(tg["key"]) - 1)
            hit = tg["key"][idx] == key
            reg = c.mask_regime[ok].map({r: i for i, r in enumerate(REGIMES)}).to_numpy()
            same = hit & (reg == tg["regime"][idx])
            stats["rows_not_target"] += int((~ok).sum() + (~hit).sum())
            stats["regime_mismatch"] += int((hit & ~same).sum())
            j = idx[same]
            stats["duplicates"] += int((seen[p][j] > 0).sum()) + (len(j) - len(np.unique(j)))
            seen[p][j] = 1
            preds[p][0][j] = c.speed_kmh[ok].to_numpy(float)[same]
            preds[p][1][j] = c.flow_vph[ok].to_numpy(float)[same]
    return preds, stats


# ---------------------------------------------------------------------------
# Task 3: S_FD exact, S_LWR proxy (train only)
# ---------------------------------------------------------------------------
def t3_proxy_panel(panel: str, tg: dict, pred_speed: np.ndarray, pred_flow: np.ndarray) -> dict:
    """Physics of the full eligible state (truth where released, predictions at targets), per regime."""
    L = links(panel)
    n_links = len(L)
    lanes = np.tile(L.lanes.to_numpy(float), STEPS)
    ffs = np.tile(L.free_speed_kmh.to_numpy(float), STEPS)
    cap = np.tile(L.capacity_vph.to_numpy(float), STEPS)
    length = np.tile(L.length_km.to_numpy(float), STEPS)
    kcrit = cap / np.maximum(ffs, 1.0)                        # recomputed, as the scorer does
    kjam = np.maximum(np.tile(L.k_jam.to_numpy(float), STEPS), 1.05 * kcrit)
    kcrit_l, kjam_l, cap_l = kcrit / lanes, kjam / lanes, cap / lanes
    w_l = cap_l / np.maximum(kjam_l - kcrit_l, 1e-9)
    acc = {r: dict.fromkeys(("fd_num", "fd_den", "sub_n", "sub_low", "lwr_num", "lwr_den", "trans"), 0.0)
           for r in range(3)}
    cols = ["ts", "pos", "mask_regime", "is_score_eligible", "is_t1_target", "speed_true", "flow_true",
            "on_ramp_n", "on_ramp_n_valid", "off_ramp_n", "off_ramp_n_valid"]
    pf = pq.ParquetFile(CLEAN / panel / "train.parquet")
    ptr = 0
    for i in range(pf.num_row_groups):
        t = pf.read_row_group(i, columns=cols)
        assert t.num_rows == STEPS * n_links, "expected one full day per row group"
        a = acc[REGIMES.index(t.column("mask_regime")[0].as_py())]
        elig = t.column("is_score_eligible").to_numpy()
        tgt = t.column("is_t1_target").to_numpy()
        v_true = t.column("speed_true").to_numpy().astype(float)
        q_true = t.column("flow_true").to_numpy().astype(float)
        k = int(tgt.sum())
        if k:
            slot = t.column("ts").to_numpy().astype("datetime64[m]").astype(np.int64)[tgt] // 5
            key = slot * KEY_BASE + t.column("pos").to_numpy()[tgt].astype(np.int64)
            assert np.array_equal(key, tg["key"][ptr:ptr + k]), "clean_data and target cache are out of sync"
        v, q = v_true.copy(), q_true.copy()
        v[tgt], q[tgt] = pred_speed[ptr:ptr + k], pred_flow[ptr:ptr + k]
        ptr += k
        use = elig & np.isfinite(v) & np.isfinite(q)            # the scorer drops cells it cannot use
        with np.errstate(invalid="ignore", divide="ignore"):
            q_l = q / lanes
            k_l = q / np.maximum(v, 1.0) / lanes
            q_fd = np.where(k_l <= kcrit_l, ffs * k_l, w_l * np.maximum(kjam_l - k_l, 0.0))
        a["fd_num"] += float(np.abs(q_l - q_fd)[use].sum())
        a["fd_den"] += float(np.abs(q_l)[use].sum())
        sub = tgt & use
        a["sub_n"] += int(sub.sum())
        a["sub_low"] += int((q[sub] < EMPTY_FLOOR_VPH).sum())
        # Conservation over five-minute transitions on the same link, both ends eligible, ramps valid at t.
        on_ok = (t.column("on_ramp_n").to_numpy() == 0) | (t.column("on_ramp_n_valid").to_numpy() > 0)
        off_ok = (t.column("off_ramp_n").to_numpy() == 0) | (t.column("off_ramp_n_valid").to_numpy() > 0)
        U = use.reshape(STEPS, n_links)
        ok = U[:-1] & U[1:] & (on_ok & off_ok).reshape(STEPS, n_links)[:-1]
        with np.errstate(invalid="ignore"):
            N_sub = (q / np.maximum(v, 1.0) * length).reshape(STEPS, n_links)
            N_true = (q_true / np.maximum(v_true, 1.0) * length).reshape(STEPS, n_links)
        d_sub, d_true = np.diff(N_sub, axis=0), np.diff(N_true, axis=0)
        a["lwr_num"] += float(np.abs(d_sub - d_true)[ok].sum())
        a["lwr_den"] += float(np.abs(d_true)[ok].sum())
        a["trans"] += int(ok.sum())
    row = {"panel": panel}
    phys = []
    for r, name in enumerate(REGIMES):
        a = acc[r]
        s_fd = max(0.0, 1 - a["fd_num"] / max(a["fd_den"], 1e-9))
        if a["sub_n"] and a["sub_low"] / a["sub_n"] > EMPTY_SHARE:
            s_fd = 0.0
        e_excess = a["lwr_num"] / max(a["lwr_den"], 1e-9)
        s_lwr = max(0.0, 1 - min(1.0, e_excess + LWR_RESIDUAL_FLOOR))
        s = (s_fd + 2 * s_lwr) / 3
        row.update({f"{name}_S_FD": s_fd, f"{name}_E_excess": e_excess, f"{name}_S_LWR": s_lwr, f"{name}_S": s})
        phys.append(s)
    row["S_physics_proxy"] = float(np.mean(phys))
    return row


# ---------------------------------------------------------------------------
# Task 2: proxy S_queue on the train windows
# ---------------------------------------------------------------------------
def t2_truth(panel: str) -> pd.DataFrame:
    """Train-window target cells with a proxy queue label. Cached."""
    path = CACHE / f"t2_truth_{panel}.parquet"
    src = CLEAN / panel / "train.parquet"
    if path.exists() and path.stat().st_mtime >= src.stat().st_mtime:
        return pd.read_parquet(path)
    tmpl = pd.read_csv(RELEASE / "task2" / panel / "train" / "sample_submission_queue.csv",
                       usecols=["window_id", "timestamp", "link_id"])
    win = pd.read_csv(RELEASE / "task2" / panel / "train" / "window_index.csv")
    L = links(panel)
    tmpl["slot"] = slot_of(tmpl.timestamp)
    tmpl["pos"] = tmpl.link_id.map(dict(zip(L.link_id, L.pos))).astype(np.int64)
    need = np.unique(tmpl.slot.to_numpy())
    parts = []
    pf = pq.ParquetFile(src)
    for i in range(pf.num_row_groups):
        t = pf.read_row_group(i, columns=["ts", "pos", "speed_true", "is_score_eligible"])
        slot = t.column("ts").to_numpy().astype("datetime64[m]").astype(np.int64) // 5
        m = np.isin(slot, need)
        if m.any():
            parts.append(pd.DataFrame({"slot": slot[m], "pos": t.column("pos").to_numpy()[m].astype(np.int64),
                                       "speed_true": t.column("speed_true").to_numpy()[m],
                                       "eligible": t.column("is_score_eligible").to_numpy()[m]}))
    truth = tmpl.merge(pd.concat(parts), on=["slot", "pos"], how="left")
    truth["eligible"] = truth.eligible.fillna(False).astype(bool) & truth.speed_true.notna()
    v_cut = L.v_cut_kmh.to_numpy()[truth.pos.to_numpy()]
    truth["queue_true"] = truth.eligible & (truth.speed_true <= v_cut)
    truth = truth.merge(win[["window_id", "condition"]], on="window_id", how="left")
    truth["panel"] = panel
    CACHE.mkdir(parents=True, exist_ok=True)
    truth.to_parquet(path, index=False)
    return truth


def score_t2(pred: pd.DataFrame, panels: list[str]) -> tuple[pd.DataFrame, dict]:
    """pred: window_id, slot, pos, queue_pred (0/1). Returns per-window IoU and per-panel scores."""
    rows = []
    for p in panels:
        tr = t2_truth(p)
        m = tr.merge(pred, on=["window_id", "slot", "pos"], how="left")
        m = m[m.eligible]
        yhat = m.queue_pred.fillna(0).astype(bool).to_numpy()        # missing row = 0, as the scorer does
        ytrue = m.queue_true.to_numpy()
        for w, idx in m.groupby("window_id").indices.items():
            inter = int((yhat[idx] & ytrue[idx]).sum())
            union = int((yhat[idx] | ytrue[idx]).sum())
            rows.append({"panel": p, "window_id": w, "condition": m.condition.iloc[idx[0]],
                         "IoU": 1.0 if union == 0 else inter / union, "true_cells": int(ytrue[idx].sum()),
                         "pred_cells": int(yhat[idx].sum())})
    detail = pd.DataFrame(rows)
    panel_scores = (detail.groupby(["panel", "condition"]).IoU.mean().groupby("panel").mean()).to_dict()
    return detail, panel_scores


def read_t2_csv(path: Path, panels: list[str]) -> pd.DataFrame:
    d = pd.read_csv(path, usecols=["window_id", "timestamp", "link_id", "queue_pred"])
    bad = d.queue_pred.dropna()
    bad = bad[~bad.isin([0, 1])]
    if len(bad):
        raise SystemExit(f"queue_pred must be 0/1; found {len(bad)} other values")
    if d.duplicated(["window_id", "timestamp", "link_id"]).any():
        raise SystemExit("duplicate (window_id, timestamp, link_id) rows")
    out = []
    for p in panels:
        L = links(p)
        x = d[d.window_id.str.startswith(f"Q2_{p}_")].copy()
        x["slot"] = slot_of(x.timestamp)
        x["pos"] = x.link_id.map(dict(zip(L.link_id, L.pos)))
        out.append(x.dropna(subset=["pos"]).astype({"pos": np.int64}))
    return pd.concat(out)[["window_id", "slot", "pos", "queue_pred"]]


# ---------------------------------------------------------------------------
# Task 4: exact S_link, plus the deviation from the prior as a diagnostic
# ---------------------------------------------------------------------------
def t4_operator(panel: str) -> tuple[list[str], list[str], np.ndarray]:
    net = RELEASE / "corridors" / panel / "network"
    paths = pd.read_csv(net / "path_set.csv", dtype={"path_id": str})
    inc = pd.read_csv(net / "path_link_incidence.csv", dtype=str)
    path_ids = paths.path_id.tolist()
    link_ids = inc.link_id.drop_duplicates().tolist()
    A = np.zeros((len(link_ids), len(path_ids)))
    A[inc.link_id.map({x: i for i, x in enumerate(link_ids)}).to_numpy(),
      inc.path_id.map({x: i for i, x in enumerate(path_ids)}).to_numpy()] = 1.0
    return path_ids, link_ids, A


def score_t4_panel(panel: str, split: str, sub: pd.DataFrame) -> dict:
    path_ids, link_ids, A = t4_operator(panel)
    base = RELEASE / "task4" / panel / split
    counts = pd.read_csv(base / "synthetic_link_counts.csv", dtype={"link_id": str})
    prior = pd.read_csv(base / "synthetic_weak_prior.csv", dtype={"path_id": str})
    li = {x: i for i, x in enumerate(link_ids)}
    scored = [x for x in counts.link_id if x in li]
    c = counts.set_index("link_id").reindex(scored)["count"].to_numpy(float)
    s = sub[(sub.panel == panel) & (sub.departure_time == str(prior.departure_time.iloc[0]))]
    row = {"panel": panel, "duplicates": int(s.duplicated("path_id").sum()),
           "missing": int(pd.Index(path_ids).difference(s.path_id).size)}
    f = s.drop_duplicates("path_id").set_index("path_id").reindex(path_ids)["path_flow"].to_numpy(float)
    row["invalid"] = int((~np.isfinite(f) | (f < 0)).sum())
    b = prior.set_index("path_id").reindex(path_ids)["path_flow"].fillna(0).to_numpy(float)
    if row["duplicates"] or row["missing"] or row["invalid"]:
        return {**row, "S_link": 0.0, "dev_ratio": float("nan")}
    loaded = A[[li[x] for x in scored]] @ f
    row["S_link"] = max(0.0, 1 - float(np.abs(loaded - c).sum()) / max(float(c.sum()), 1e-9))
    row["dev_ratio"] = float(np.abs(f - b).sum() / max(b.sum(), 1e-9))   # how far the solution moved from b
    return row


# ---------------------------------------------------------------------------
# command line
# ---------------------------------------------------------------------------
def print_table(rows: list[dict], cols: list[str], title: str) -> None:
    df = pd.DataFrame(rows)
    print(f"\n{title}")
    print(df[[c for c in cols if c in df.columns]].to_string(index=False, float_format=lambda x: f"{x:.4f}"))


def cmd_t1(args) -> float:
    panels = args.panels or PANELS
    preds, stats = read_t1_csv(args.submission, args.split, panels)
    print("read:", stats)
    rows = [score_t1_panel(p, t1_targets(p, args.split), *preds[p]) for p in panels]
    print_table(rows, ["panel", "R1_S", "R2_S", "R3_S", "R1_rmse_speed", "R3_rmse_speed",
                       "R1_rmse_flow_lane", "R3_rmse_flow_lane", "missing", "S_state"], "Task 1 S_state (exact)")
    fam, overall = aggregate({r["panel"]: r["S_state"] for r in rows})
    print(f"families: {fam}\nOverall S_state = {overall:.6f}" + ("" if len(panels) == 10 else f"  ({len(panels)} panels)"))
    log_result("t1", args.split, str(args.submission), args.tag, overall, {r["panel"]: r["S_state"] for r in rows})
    return overall


def cmd_t3(args) -> float:
    panels = args.panels or PANELS
    preds, _ = read_t1_csv(args.submission, "train", panels)
    rows = [t3_proxy_panel(p, t1_targets(p, "train"), *preds[p]) for p in panels]
    print_table(rows, ["panel", "R1_S_FD", "R1_E_excess", "R2_E_excess", "R3_E_excess", "R1_S_LWR", "S_physics_proxy"],
                "Task 3 physics proxy (S_FD exact, S_LWR approximate)")
    _, overall = aggregate({r["panel"]: r["S_physics_proxy"] for r in rows})
    print(f"Overall S_physics (proxy) = {overall:.6f}")
    log_result("t3_proxy", "train", str(args.submission), args.tag, overall, {r["panel"]: r["S_physics_proxy"] for r in rows})
    return overall


def cmd_t2(args) -> float:
    panels = [p for p in (args.panels or T2_PANELS) if p in T2_PANELS]
    detail, scores = score_t2(read_t2_csv(args.submission, panels), panels)
    by_cond = detail.groupby(["panel", "condition"]).IoU.mean().unstack()
    print("\nTask 2 S_queue proxy on train windows (IoU by condition)")
    print(by_cond.assign(panel_score=pd.Series(scores)).to_string(float_format=lambda x: f"{x:.4f}"))
    _, overall = aggregate(scores)
    print(f"Overall S_queue (train proxy) = {overall:.6f}")
    log_result("t2_proxy", "train", str(args.submission), args.tag, overall, scores)
    return overall


def cmd_t4(args) -> float:
    panels = args.panels or PANELS
    sub = pd.read_csv(args.submission, dtype={"panel": str, "departure_time": str, "path_id": str})
    rows = [score_t4_panel(p, args.split, sub) for p in panels]
    print_table(rows, ["panel", "missing", "duplicates", "invalid", "dev_ratio", "S_link"], f"Task 4 S_link ({args.split})")
    _, overall = aggregate({r["panel"]: r["S_link"] for r in rows})
    print(f"Overall S_link = {overall:.6f}   (S_link is 25% of S_ODME)")
    log_result("t4_link", args.split, str(args.submission), args.tag, overall, {r["panel"]: r["S_link"] for r in rows})
    return overall


def cmd_selftest(args) -> None:
    """Check each evaluator against numbers published by the organizers or known by construction."""
    p = "D12_I5_N"
    tg = t1_targets(p)
    perfect = score_t1_panel(p, tg, tg["speed_true"], tg["flow_true"])["S_state"]
    print(f"[t1] perfect answer on {p}: {perfect:.6f} (expect 1.0)")
    base = ROOT / "work" / "t1_baseline_train_D12_I5_N.csv"
    if base.exists():
        preds, _ = read_t1_csv(base, "train", [p])
        s = score_t1_panel(p, tg, *preds[p])["S_state"]
        print(f"[t1] official baseline on {p} train: {s:.6f} (official scorer printed 0.625461)")
        phys_b = t3_proxy_panel(p, tg, *preds[p])
        print(f"[t3] baseline physics proxy: {phys_b['S_physics_proxy']:.4f} "
              f"(leaderboard: naive 0.3242 on {p} validation), E_excess R1..R3 = "
              + ", ".join(f"{phys_b[f'{r}_E_excess']:.2f}" for r in REGIMES))
    phys_p = t3_proxy_panel(p, tg, tg["speed_true"], tg["flow_true"])
    print(f"[t3] perfect physics proxy: {phys_p['S_physics_proxy']:.4f} (leaderboard: perfect 0.9598), "
          f"S_FD R1 = {phys_p['R1_S_FD']:.4f}")
    q = ROOT / "work" / "baseline" / "queue_train.csv"
    if q.exists():
        _, scores = score_t2(read_t2_csv(q, T2_PANELS), T2_PANELS)
        print(f"[t2] persistence on train windows: {aggregate(scores)[1]:.4f} "
              f"(leaderboard persistence: 0.25-0.30 on validation)")
    o = ROOT / "work" / "baseline" / "task4_validation" / "baseline_submission.csv"
    if o.exists():
        sub = pd.read_csv(o, dtype={"panel": str, "departure_time": str, "path_id": str})
        print(f"[t4] baseline S_link on validation: {aggregate({x: score_t4_panel(x, 'validation', sub)['S_link'] for x in PANELS})[1]:.4f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("t1", "t2", "t3", "t4"):
        s = sub.add_parser(name)
        s.add_argument("--submission", type=Path, required=True)
        s.add_argument("--panels", nargs="*")
        s.add_argument("--split", default="train", choices=["train", "validation", "private"])
        s.add_argument("--tag", default="", help="label stored in work/results/history.csv")
    sub.add_parser("selftest")
    args = ap.parse_args()
    {"t1": cmd_t1, "t2": cmd_t2, "t3": cmd_t3, "t4": cmd_t4, "selftest": cmd_selftest}[args.cmd](args)


if __name__ == "__main__":
    main()
