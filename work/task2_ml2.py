#!/usr/bin/env python
"""Task 2, second ML version: window-weighted, more regularised LightGBM with lag features.

Changes against task2_ml.py (which stays as the reproducible v8 model):
  * Every training window counts equally (row weight = 1000 / rows in the window), as the leaderboard
    averages IoU per window.
  * Simpler, more regularised trees, which should lean less on corridor-specific memorisation:
      onset    15 leaves, L2 10, learning rate 0.04
      ongoing  31 leaves, depth 8, >=250 rows per leaf, L2 8, learning rate 0.045
  * Short lags of speed and flow (5, 10, 15, 30, 45, 60 minutes before T), recent-versus-earlier changes,
    history coverage, peak flow ratio and the corridor id.
  * Ongoing windows: at most WINDOWS_PER_DAY per corridor and day, 6 hours apart (like the official selector).
Information contract unchanged: features use only the 12-step history [T-60, T) and train-derived priors.

Usage (from the "IEEE Big data" folder):
  python work/task2_ml2.py build
  python work/task2_ml2.py cv                       # official train windows, both folds
  python work/task2_ml2.py predict --tag _v2         # final models -> work/sub/queue_<split>_ml2<tag>.csv
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import CACHE, CLEAN, RELEASE, ROOT, STEPS, T2_PANELS, links, slot_of  # noqa: E402
from task2_ml import (RADIUS_KM, ctx_of, hist_features, iou_scores, save_probs, threshold, to_frame)  # noqa: E402
from task2_rules import DEFAULTS as RULES, HIST_STEPS, HORIZON, onset_scores, predict_panel as rule_predict  # noqa: E402

DATA = CACHE / "t2_ml2"
OUT = ROOT / "work" / "sub"
SEED, WINDOWS_PER_DAY, MIN_SPACING = 2026, 1, 72
PARAMS = {
    "onset": {"objective": "binary", "learning_rate": 0.04, "num_leaves": 15, "min_data_in_leaf": 100,
              "lambda_l2": 10.0, "feature_fraction": 0.9, "num_threads": 16, "verbose": -1, "seed": SEED},
    "ongoing": {"objective": "binary", "learning_rate": 0.045, "num_leaves": 31, "max_depth": 8,
                "min_data_in_leaf": 250, "lambda_l2": 8.0, "feature_fraction": 0.85, "bagging_fraction": 0.85,
                "bagging_freq": 1, "max_bin": 127, "num_threads": 16, "verbose": -1, "seed": SEED},
}
ROUNDS, EARLY = 2000, 70
DROP = ["y", "w", "fold", "date", "kind", "panel"]


def features2(ctx, V, Q, O, tod_T: int, weekend: int, prior: np.ndarray, panel_no: int) -> dict:
    """task2_ml features plus short lags, recent-vs-earlier changes, coverage and peak flow."""
    f = hist_features(ctx, V, Q, O, tod_T, weekend, prior)
    with np.errstate(invalid="ignore", divide="ignore"):
        r, qc = V / ctx.ffs, Q / ctx.cap
        for lag in (1, 2, 3, 6, 9, 12):
            f[f"r_lag{lag}"], f[f"qc_lag{lag}"] = r[-lag], qc[-lag]
        mean = lambda a: np.where(np.isfinite(a).sum(0) > 0, np.nansum(a, 0) / np.maximum(np.isfinite(a).sum(0), 1), np.nan)
        f["r_recent_minus_prior"] = mean(r[-3:]) - mean(r[-6:-3])
        f["qc_recent_minus_prior"] = mean(qc[-3:]) - mean(qc[-6:-3])
        f["coverage"] = np.isfinite(V).mean(0)
        f["qc_max"] = np.where(np.isfinite(qc).any(0), np.nanmax(np.where(np.isfinite(qc), qc, -np.inf), 0), np.nan)
    f["panel_no"] = np.full(ctx.n, float(panel_no))
    return f


def frame(f, rows, step, y, fold, date, kind):
    df = to_frame(f, rows, step)
    df["y"], df["fold"], df["date"], df["kind"] = y.astype(np.int8), np.int8(fold), date, kind
    df["w"] = np.float32(1000.0 / max(len(rows), 1))       # per-window weight (per step for ongoing)
    return df


def build_panel(panel: str) -> pd.DataFrame:
    ctx = ctx_of(panel)
    n, pno = ctx.n, T2_PANELS.index(panel)
    ev = ctx.ev
    ev_tods = ev.drop_duplicates("event").groupby("date").tod.apply(list).to_dict()
    pf = pq.ParquetFile(CLEAN / panel / "train.parquet")
    parts, n_on, n_go = [], 0, 0
    for i in range(pf.num_row_groups):
        tb = pf.read_row_group(i, columns=["ts", "weekday", "speed_true", "flow_true", "occ_true", "is_score_eligible"])
        ts0 = pd.Timestamp(tb.column("ts")[0].as_py())
        date, weekday, fold = ts0.strftime("%Y-%m-%d"), int(tb.column("weekday")[0].as_py()), ts0.month % 2
        E = tb.column("is_score_eligible").to_numpy().reshape(STEPS, n)
        g = lambda c: np.where(E, tb.column(c).to_numpy().astype(float).reshape(STEPS, n), np.nan)
        V, Q, O = g("speed_true"), g("flow_true"), g("occ_true")
        with np.errstate(invalid="ignore"):
            Qd = E & (V <= ctx.vcut)
        sl = lambda T: slice(T - HIST_STEPS, T)
        for k in ev_tods.get(date, []):
            T = k - HORIZON
            prior = onset_scores(ev, n, k, weekday, date, RULES["sigma_h"], RULES["weekend"])
            f = features2(ctx, V[sl(T)], Q[sl(T)], O[sl(T)], T, int(weekday >= 5), prior, pno)
            rows = np.flatnonzero(E[k])
            parts.append(frame(f, rows, None, Qd[k, rows], fold, date, "onset"))
            n_on += 1
        cand = [T for T in range(HIST_STEPS, STEPS - HORIZON)
                if Qd[sl(T)].sum() >= 2 and Qd[sl(T)].sum(0).max() >= 2 and Qd[T + 1:T + 1 + HORIZON].any()
                and np.isfinite(V[sl(T)]).mean() >= 0.7]
        rng = np.random.default_rng(SEED + 1000 * pno + i)
        chosen = []
        for T in rng.permutation(cand):
            if all(abs(T - c) >= MIN_SPACING for c in chosen):
                chosen.append(int(T))
            if len(chosen) >= WINDOWS_PER_DAY:
                break
        for T in chosen:
            prior = onset_scores(ev, n, T + HORIZON, weekday, date, RULES["sigma_h"], RULES["weekend"])
            f = features2(ctx, V[sl(T)], Q[sl(T)], O[sl(T)], T, int(weekday >= 5), prior, pno)
            with np.errstate(invalid="ignore"):
                hq = (V[sl(T)] <= ctx.vcut).any(0)
            near = np.zeros(n, bool)
            for xq in ctx.x[hq]:
                near |= np.abs(ctx.x - xq) <= RADIUS_KM
            win_rows = sum(int((near & E[T + 1 + s]).sum()) for s in range(HORIZON))
            for s in range(HORIZON):
                rows = np.flatnonzero(near & E[T + 1 + s])
                df = frame(f, rows, s + 1, Qd[T + 1 + s, rows], fold, date, "ongoing")
                df["w"] = np.float32(1000.0 / max(win_rows, 1))
                parts.append(df)
            n_go += 1
    out = pd.concat(parts, ignore_index=True)
    out["panel"] = panel
    print(f"  {panel}: {n_on} onset events, {n_go} ongoing windows, {len(out):,} rows", flush=True)
    return out


def load(kind: str) -> pd.DataFrame:
    df = pd.concat([pd.read_parquet(DATA / f"{p}.parquet", filters=[("kind", "==", kind)]) for p in T2_PANELS],
                   ignore_index=True)
    if kind == "onset":
        df = df.drop(columns=["step"], errors="ignore")
    return df.astype({"date": "category", "kind": "category", "panel": "category"})


def fit(kind: str, train: pd.DataFrame, valid: pd.DataFrame | None, rounds: int | None = None, seed: int = SEED):
    feats = [c for c in train.columns if c not in DROP]
    params = {**PARAMS[kind], "seed": seed}
    dtr = lgb.Dataset(train[feats], train.y, weight=train.w)
    if valid is not None:
        dva = lgb.Dataset(valid[feats], valid.y, weight=valid.w, reference=dtr)
        b = lgb.train(params, dtr, ROUNDS, valid_sets=[dva], callbacks=[lgb.early_stopping(EARLY, verbose=False)])
        it = b.best_iteration
    else:
        b, it = lgb.train(params, dtr, rounds), rounds
    return (lambda X: b.predict(X[feats], num_iteration=it)), it, b


def window_probs2(panel: str, split: str, models: dict, by_fold: bool) -> list[dict]:
    """Like task2_ml.window_probs, with the v2 features built from window_history only."""
    ctx = ctx_of(panel)
    n, pno = ctx.n, T2_PANELS.index(panel)
    base = RELEASE / "task2" / panel / split
    win = pd.read_csv(base / "window_index.csv")
    hist = pd.read_parquet(base / "window_history.parquet")
    hist["pos"] = hist.link_id.map(dict(zip(links(panel).link_id, range(n))))
    hist["slot"] = slot_of(hist.timestamp)
    out = []
    for w in win.itertuples(index=False):
        T = int(slot_of(pd.Series([w.forecast_origin]))[0])
        origin = pd.Timestamp(w.forecast_origin)
        tod = origin.hour * 12 + origin.minute // 5
        h = hist[hist.window_id == w.window_id]
        ok = h.is_score_eligible.astype(bool) & h.speed_kmh.notna()
        si = (h.slot - (T - HIST_STEPS)).to_numpy()
        m = ok.to_numpy() & (si >= 0) & (si < HIST_STEPS)
        grids = {}
        for c in ("speed_kmh", "flow_vph", "occupancy"):
            grids[c] = np.full((HIST_STEPS, n), np.nan)
            grids[c][si[m], h.pos.to_numpy()[m]] = h[c].to_numpy()[m]
        prior = onset_scores(ctx.ev, n, tod + HORIZON, origin.weekday(), w.date if split == "train" else None,
                             RULES["sigma_h"], RULES["weekend"])
        f = features2(ctx, grids["speed_kmh"], grids["flow_vph"], grids["occupancy"], tod, int(origin.weekday() >= 5),
                      prior, pno)
        mdl = models[origin.month % 2] if by_fold else models
        prob = np.zeros((HORIZON, n))
        if w.condition == "queue_onset":
            prob[HORIZON - 1] = mdl["onset"](to_frame(f, np.arange(n), None))
        else:
            with np.errstate(invalid="ignore"):
                hq = (grids["speed_kmh"] <= ctx.vcut).any(0)
            near = np.zeros(n, bool)
            for xq in ctx.x[hq]:
                near |= np.abs(ctx.x - xq) <= RADIUS_KM
            idx = np.flatnonzero(near)
            for s in range(HORIZON if len(idx) else 0):
                prob[s, idx] = mdl["ongoing"](to_frame(f, idx, s + 1))
        out.append({"window_id": w.window_id, "condition": w.condition, "T": T, "prob": prob})
    return out


def cmd_build(args) -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    for p in args.panels:
        t0 = time.time()
        build_panel(p).to_parquet(DATA / f"{p}.parquet", index=False)
        print(f"    {time.time() - t0:.0f}s", flush=True)


def cmd_cv(args) -> None:
    rule = pd.concat([rule_predict(p, "train", RULES, loo=True) for p in T2_PANELS])
    base = iou_scores(rule[["window_id", "slot", "pos", "queue_pred"]], T2_PANELS)
    print(f"rules: overall {base['overall']:.4f} onset {base['queue_onset']:.4f} ongoing {base['queue_ongoing']:.4f}")
    models = {0: {}, 1: {}}
    for kind in ("onset", "ongoing"):
        d = load(kind)
        for k in (0, 1):
            fn, it, _ = fit(kind, d[d.fold != k], d[d.fold == k])
            models[k][kind] = fn
            print(f"  fold {k} {kind}: {it} rounds ({len(d):,} rows)", flush=True)
        del d
    probs = {p: window_probs2(p, "train", models, by_fold=True) for p in T2_PANELS}
    for th_on in (0.3, 0.4, 0.5, 0.55, 0.6):
        for th_go in (0.3, 0.4, 0.5):
            pred = pd.concat([threshold(probs[p], {"onset": th_on, "ongoing": th_go}) for p in T2_PANELS])
            sc = iou_scores(pred, T2_PANELS)
            print(f"  theta onset {th_on} ongoing {th_go}: overall {sc['overall']:.4f} onset {sc['queue_onset']:.4f} "
                  f"ongoing {sc['queue_ongoing']:.4f}", flush=True)


def cmd_predict(args) -> None:
    models = {}
    for kind in ("onset", "ongoing"):
        d = load(kind)
        _, i0, _ = fit(kind, d[d.fold != 0], d[d.fold == 0])
        _, i1, _ = fit(kind, d[d.fold != 1], d[d.fold == 1])
        models[kind], _, booster = fit(kind, d, None, int(1.1 * (i0 + i1) / 2))
        booster.save_model(str(ROOT / "work" / "models" / f"t2_ml2_{kind}{args.tag}.txt"))
        del d
    th = {"onset": args.theta_onset, "ongoing": args.theta_ongoing}
    for split in args.splits:
        parts = []
        for p in T2_PANELS:
            probs = window_probs2(p, split, models, by_fold=False)
            save_probs(probs, f"t2ml2{args.tag}", split, p)
            pr = threshold(probs, th)
            tmpl = pd.read_csv(RELEASE / "task2" / p / split / "sample_submission_queue.csv",
                               usecols=["window_id", "timestamp", "link_id"])
            tmpl["slot"] = slot_of(tmpl.timestamp)
            tmpl["pos"] = tmpl.link_id.map(dict(zip(links(p).link_id, range(len(links(p))))))
            parts.append(tmpl.merge(pr, on=["window_id", "slot", "pos"], how="left").fillna({"queue_pred": 0}))
        out = pd.concat(parts)
        out["queue_pred"] = out.queue_pred.astype(int)
        path = OUT / f"queue_{split}_ml2{args.tag}.csv"
        out[["window_id", "timestamp", "link_id", "queue_pred"]].to_csv(path, index=False)
        print(f"wrote {path} ({len(out):,} rows, {int(out.queue_pred.sum()):,} queued cells)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("build")
    s.add_argument("--panels", nargs="+", default=T2_PANELS)
    sub.add_parser("cv")
    s = sub.add_parser("predict")
    s.add_argument("--theta-onset", type=float, default=0.5)
    s.add_argument("--theta-ongoing", type=float, default=0.4)
    s.add_argument("--splits", nargs="+", default=["validation", "private"])
    s.add_argument("--tag", default="")
    args = ap.parse_args()
    {"build": cmd_build, "cv": cmd_cv, "predict": cmd_predict}[args.cmd](args)


if __name__ == "__main__":
    main()
