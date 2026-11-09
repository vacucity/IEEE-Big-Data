#!/usr/bin/env python
"""Task 2 machine-learning model: per (link, horizon step) queue classifiers, LightGBM vs XGBoost.

Training windows are generated from the 273 train days, mimicking the official selector:
  onset    every onset event (queue appears at k after a queue-free hour): history [k-18, k-7], one row per
           link, label = queued at k (= T+30). Only T+30 is predicted, as the selector guarantees.
  ongoing  origins every 30 minutes where the history holds an established queue and the horizon a queue;
           rows for links within 10 km of any queued link in the history, label per horizon step.
Features use only the 12-step history [T-60, T) - the same arrays the official window_history.parquet gives
for validation and private - plus link geometry and the train onset prior (computed leaving the day out).

Evaluation: two folds by month parity; the 80 official train windows of one fold are predicted by the
models fitted on the other fold, and scored with evaluate.py's proxy labels (IoU per window).

Usage (from the "IEEE Big data" folder):
  python work/task2_ml.py build      # generated training rows -> work/cache/t2_ml/<panel>.parquet
  python work/task2_ml.py cv         # LightGBM vs XGBoost vs rules on the official train windows
  python work/task2_ml.py predict --algo lgbm   # fit on all rows, write work/sub/queue_<split>_ml.csv
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from functools import lru_cache
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import CACHE, CLEAN, FAMILY, RELEASE, ROOT, STEPS, T2_PANELS, aggregate, links, slot_of, t2_truth  # noqa
from task2_rules import (DEFAULTS as RULES, HIST_STEPS, HORIZON, clusters, onset_events, onset_scores,  # noqa
                         predict_panel as rule_predict, travel_axis)

DATA = CACHE / "t2_ml"
OUT = ROOT / "work" / "sub"
RADIUS_KM, STRIDE, GAP_KM, SEED = 10.0, 6, 0.5, 2026
# The official selector drops windows that plain persistence already solves (maximum_persistence_iou 0.9) and
# windows with little history (minimum_history_coverage 0.7); training windows are filtered the same way.
MAX_PERSIST_IOU, MIN_HIST_COVERAGE = 1.0, 0.0   # filter tested (0.9 / 0.7): ongoing CV 0.8351 vs 0.8408 unfiltered -> off
MAX_ONGOING_WINDOWS = 400          # per panel, keeps the training set within a few hundred MB
LGB_PARAMS = {"objective": "binary", "learning_rate": 0.05, "num_leaves": 63, "min_data_in_leaf": 100,
              "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "verbose": -1,
              "num_threads": 16, "seed": SEED}
XGB_PARAMS = {"objective": "binary:logistic", "eval_metric": "logloss", "eta": 0.05, "max_depth": 8,
              "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 5, "tree_method": "hist",
              "nthread": 16, "seed": SEED}
ROUNDS, EARLY = 1500, 100


class Ctx:
    def __init__(self, panel: str):
        L = links(panel)
        self.panel, self.n = panel, len(L)
        self.x = travel_axis(panel)                       # km along travel, per pos
        self.order = np.argsort(self.x)                   # travel order -> pos
        self.vcut = L.v_cut_kmh.to_numpy(float)
        self.ffs = L.free_speed_kmh.to_numpy(float)
        self.lanes = L.lanes.to_numpy(float)
        self.cap = L.capacity_vph.to_numpy(float)
        rank = np.argsort(self.order)
        self.static = {"lanes": self.lanes, "ffs": self.ffs, "cap_l": self.cap / self.lanes,
                       "length": L.length_km.to_numpy(float), "pos_rel": rank / max(self.n - 1, 1),
                       "n_on": L.on_ramp_n.to_numpy(float), "n_off": L.off_ramp_n.to_numpy(float)}
        self.ev = onset_events(panel)


@lru_cache(maxsize=None)
def ctx_of(panel: str) -> Ctx:
    return Ctx(panel)


def neighbour(a: np.ndarray, ctx: Ctx, k: int) -> np.ndarray:
    """Value of the link k places upstream (k>0) or downstream (k<0) along travel."""
    f = a[ctx.order]
    out = np.full_like(f, np.nan, dtype=float)
    if k > 0:
        out[k:] = f[:-k]
    else:
        out[:k] = f[-k:]
    res = np.empty_like(out)
    res[ctx.order] = out
    return res


def hist_features(ctx: Ctx, V, Q, O, tod_T: int, weekend: int, prior: np.ndarray) -> dict[str, np.ndarray]:
    """Per-link features from a (12, n) history; NaN marks a missing or ineligible reading."""
    H, n = V.shape
    idx = np.arange(H)[:, None]
    valid = np.isfinite(V)
    li_grid = np.maximum.accumulate(np.where(valid, idx, -1), axis=0)          # last valid index up to t
    Vc = np.where(li_grid >= 0, np.take_along_axis(V, li_grid.clip(0), 0), np.nan)  # carried forward
    li = li_grid[-1]
    has = li >= 0
    pick = lambda A: np.where(has, A[li.clip(0), np.arange(n)], np.nan)
    v_last, q_last, o_last = pick(V), pick(Q), pick(O)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = V / ctx.ffs
        rc = Vc / ctx.ffs
        queued_c = Vc <= ctx.vcut
        q_hist = np.where(valid, V <= ctx.vcut, False)
        # slope of the speed ratio over the last 6 steps (least squares on valid points)
        t6, r6 = idx[-6:].astype(float), r[-6:]
        m6 = np.isfinite(r6)
        cnt = m6.sum(0)
        tm = np.where(m6, t6, 0).sum(0) / np.maximum(cnt, 1)
        rm = np.nansum(r6, 0) / np.maximum(cnt, 1)
        cov = np.where(m6, (t6 - tm) * (np.nan_to_num(r6) - rm), 0).sum(0)
        var = np.where(m6, (t6 - tm) ** 2, 0).sum(0)
        slope = np.where(cnt >= 3, cov / np.maximum(var, 1e-9), np.nan)
        feats = {"v_last_r": v_last / ctx.ffs, "age": np.where(has, H - li, H + 1).astype(float),
                 "r_mean": np.nanmean(np.where(valid, r, np.nan), 0) if valid.any() else np.full(n, np.nan),
                 "r_min": np.nanmin(np.where(valid, r, np.inf), 0), "r_3ago": rc[H - 4], "r_6ago": rc[H - 7],
                 "r_slope6": slope, "q_last_l": q_last / ctx.lanes, "q_cap": q_last / ctx.cap,
                 "q_mean_l": np.nanmean(np.where(valid, Q, np.nan), 0) / ctx.lanes, "o_last": o_last,
                 "o_mean": np.nanmean(np.where(valid, O, np.nan), 0),
                 "queued_last": queued_c[-1].astype(float), "n_q_hist": q_hist.sum(0).astype(float)}
    feats["r_min"] = np.where(np.isinf(feats["r_min"]), np.nan, feats["r_min"])
    for k, name in ((1, "up1"), (2, "up2"), (-1, "dn1"), (-2, "dn2")):
        feats[f"r_{name}"] = neighbour(rc[-1], ctx, k)
        feats[f"q_{name}"] = neighbour(queued_c[-1].astype(float), ctx, k)
    # geometry of the visible queue at the last step
    x, ql = ctx.x, queued_c[-1]
    xq = np.sort(x[ql])
    if len(xq):
        i = np.searchsorted(xq, x)
        up = np.where(i > 0, x - xq[np.clip(i - 1, 0, len(xq) - 1)], 50.0)
        j = np.searchsorted(xq, x, side="right")
        dn = np.where(j < len(xq), xq[np.clip(j, 0, len(xq) - 1)] - x, 50.0)
    else:
        up = dn = np.full(n, 50.0)
    inside = np.zeros(n)
    clen = np.zeros(n)
    tail_v = np.zeros(n)
    near = np.full(n, 1e9)
    for tail, head in clusters(x, ql, GAP_KM):
        ts, tt = [], []
        for t in range(H - 6, H):
            qi = queued_c[t] & (x >= tail - 3.0) & (x <= head)
            if qi.any():
                ts.append(t)
                tt.append(x[qi].min())
        v = np.polyfit(ts, tt, 1)[0] if len(ts) >= 3 else 0.0
        d = np.where(x < tail, tail - x, np.where(x > head, x - head, 0.0))
        closer = d < near
        near = np.where(closer, d, near)
        inside = np.where(closer & (d == 0), 1.0, inside)
        clen = np.where(closer, head - tail, clen)
        tail_v = np.where(closer, v, tail_v)
    feats.update({"dist_up_q": np.minimum(up, 50.0), "dist_dn_q": np.minimum(dn, 50.0), "in_cluster": inside,
                  "cluster_len": clen, "tail_v": tail_v, "dist_cluster": np.minimum(near, 50.0),
                  "nq_last": np.full(n, float(ql.sum())), "nq_6ago": np.full(n, float(queued_c[H - 7].sum())),
                  "tod_T": np.full(n, float(tod_T)), "weekend": np.full(n, float(weekend)), "prior": prior})
    for k, a in ctx.static.items():
        feats[k] = a
    return feats


def to_frame(feats: dict, rows: np.ndarray, step: int | None) -> pd.DataFrame:
    df = pd.DataFrame({k: np.asarray(v, float)[rows].astype(np.float32) for k, v in feats.items()})
    if step is not None:
        df["step"] = np.float32(step)
    return df


# ---------------------------------------------------------------------------
# training rows from the train split
# ---------------------------------------------------------------------------
def build_panel(panel: str) -> pd.DataFrame:
    """Two passes: onset rows and the list of qualifying ongoing origins, then features for a sample of them."""
    ctx = ctx_of(panel)
    n = ctx.n
    rng = np.random.default_rng(SEED + T2_PANELS.index(panel))
    ev = ctx.ev
    ev_tods = ev.drop_duplicates("event").groupby("date").tod.apply(list).to_dict()
    pf = pq.ParquetFile(CLEAN / panel / "train.parquet")

    def day(i):
        tb = pf.read_row_group(i, columns=["ts", "weekday", "speed_true", "flow_true", "occ_true", "is_score_eligible"])
        ts0 = pd.Timestamp(tb.column("ts")[0].as_py())
        E = tb.column("is_score_eligible").to_numpy().reshape(STEPS, n)
        g = lambda c: np.where(E, tb.column(c).to_numpy().astype(float).reshape(STEPS, n), np.nan)
        V = g("speed_true")
        with np.errstate(invalid="ignore"):
            Qd = E & (V <= ctx.vcut)
        return ts0.strftime("%Y-%m-%d"), int(tb.column("weekday")[0].as_py()), ts0.month % 2, E, V, g("flow_true"), g("occ_true"), Qd

    def frame(f, rows, step, y, fold, date, kind):
        df = to_frame(f, rows, step)
        df["y"], df["fold"], df["date"], df["kind"] = y.astype(np.int8), np.int8(fold), date, kind
        return df

    onset_parts, candidates = [], []
    for i in range(pf.num_row_groups):
        date, weekday, fold, E, V, Q, O, Qd = day(i)
        for k in ev_tods.get(date, []):
            T = k - HORIZON
            prior = onset_scores(ev, n, k, weekday, date, RULES["sigma_h"], RULES["weekend"])
            f = hist_features(ctx, V[T - HIST_STEPS:T], Q[T - HIST_STEPS:T], O[T - HIST_STEPS:T], T, int(weekday >= 5), prior)
            rows = np.flatnonzero(E[k])
            onset_parts.append(frame(f, rows, None, Qd[k, rows], fold, date, "onset"))
        for T in range(HIST_STEPS, STEPS - HORIZON, STRIDE):
            Hq = Qd[T - HIST_STEPS:T]
            if Hq.sum() >= 2 and Hq.sum(0).max() >= 2 and Qd[T + 1:T + 1 + HORIZON].any():
                if E[T - HIST_STEPS:T].mean() < MIN_HIST_COVERAGE:
                    continue
                el = E[T + 1:T + 1 + HORIZON]
                yt = Qd[T + 1:T + 1 + HORIZON][el]
                yp = np.broadcast_to(Qd[T - 1], (HORIZON, n))[el]    # official persistence: state at the last step
                union = (yt | yp).sum()
                if union and (yt & yp).sum() / union > MAX_PERSIST_IOU:
                    continue
                candidates.append((i, T))
    keep = sorted(candidates[j] for j in rng.permutation(len(candidates))[:MAX_ONGOING_WINDOWS])
    ongoing_parts = []
    for i in sorted({i for i, _ in keep}):
        date, weekday, fold, E, V, Q, O, Qd = day(i)
        for T in [T for j, T in keep if j == i]:
            prior = onset_scores(ev, n, T + HORIZON, weekday, date, RULES["sigma_h"], RULES["weekend"])
            f = hist_features(ctx, V[T - HIST_STEPS:T], Q[T - HIST_STEPS:T], O[T - HIST_STEPS:T], T, int(weekday >= 5), prior)
            with np.errstate(invalid="ignore"):
                hq = (V[T - HIST_STEPS:T] <= ctx.vcut).any(0)
            near = np.zeros(n, bool)
            for xq in ctx.x[hq]:
                near |= np.abs(ctx.x - xq) <= RADIUS_KM
            for s in range(HORIZON):
                rows = np.flatnonzero(near & E[T + 1 + s])
                ongoing_parts.append(frame(f, rows, s + 1, Qd[T + 1 + s, rows], fold, date, "ongoing"))
    out = pd.concat(onset_parts + ongoing_parts, ignore_index=True)
    out["panel"] = panel
    print(f"  {panel}: onset rows {sum(len(d) for d in onset_parts):,} ({len(onset_parts)} events), "
          f"ongoing rows {sum(len(d) for d in ongoing_parts):,} ({len(keep)} of {len(candidates)} windows)", flush=True)
    return out


def cmd_build(args) -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    for p in args.panels:
        t0 = time.time()
        build_panel(p).to_parquet(DATA / f"{p}.parquet", index=False)
        print(f"    {time.time() - t0:.0f}s", flush=True)


# ---------------------------------------------------------------------------
# official windows: features, prediction, IoU
# ---------------------------------------------------------------------------
def window_inputs(panel: str, split: str):
    """Features for every official window of a split, from window_history.parquet only."""
    ctx = ctx_of(panel)
    n = ctx.n
    base = RELEASE / "task2" / panel / split
    win = pd.read_csv(base / "window_index.csv")
    hist = pd.read_parquet(base / "window_history.parquet")
    pos_of = dict(zip(links(panel).link_id, range(n)))
    hist["pos"] = hist.link_id.map(pos_of)
    hist["slot"] = slot_of(hist.timestamp)
    out = []
    for w in win.itertuples(index=False):
        T = int(slot_of(pd.Series([w.forecast_origin]))[0])
        origin = pd.Timestamp(w.forecast_origin)
        tod = origin.hour * 12 + origin.minute // 5
        h = hist[hist.window_id == w.window_id]
        ok = h.is_score_eligible.astype(bool) & h.speed_kmh.notna()
        grids = {c: np.full((HIST_STEPS, n), np.nan) for c in ("speed_kmh", "flow_vph", "occupancy")}
        si = (h.slot - (T - HIST_STEPS)).to_numpy()
        m = ok.to_numpy() & (si >= 0) & (si < HIST_STEPS)
        for c in grids:
            grids[c][si[m], h.pos.to_numpy()[m]] = h[c].to_numpy()[m]
        prior = onset_scores(ctx.ev, n, tod + HORIZON, origin.weekday(), w.date if split == "train" else None,
                             RULES["sigma_h"], RULES["weekend"])
        f = hist_features(ctx, grids["speed_kmh"], grids["flow_vph"], grids["occupancy"], tod, int(origin.weekday() >= 5), prior)
        with np.errstate(invalid="ignore"):
            hq = (grids["speed_kmh"] <= ctx.vcut).any(0)
        near = np.zeros(n, bool)
        for xq in ctx.x[hq]:
            near |= np.abs(ctx.x - xq) <= RADIUS_KM
        out.append({"window_id": w.window_id, "condition": w.condition, "T": T, "month": origin.month,
                    "feats": f, "near": near})
    return out


@lru_cache(maxsize=None)
def cached_inputs(panel: str, split: str):
    return window_inputs(panel, split)


def window_probs(panel: str, split: str, models: dict, by_fold: bool) -> list[dict]:
    """Queue probabilities per official window: onset -> T+30 only, ongoing -> each step for nearby links."""
    n = ctx_of(panel).n
    out = []
    for w in cached_inputs(panel, split):
        m = models[w["month"] % 2] if by_fold else models
        prob = np.zeros((HORIZON, n))
        if w["condition"] == "queue_onset":
            prob[HORIZON - 1] = m["onset"](to_frame(w["feats"], np.arange(n), None))
        else:
            idx = np.flatnonzero(w["near"])
            for s in range(HORIZON if len(idx) else 0):
                prob[s, idx] = m["ongoing"](to_frame(w["feats"], idx, s + 1))
        out.append({**{k: w[k] for k in ("window_id", "condition", "T")}, "prob": prob})
    return out


PROBS = ROOT / "work" / "sub" / "probs"


def save_probs(probs: list[dict], name: str, split: str, panel: str) -> None:
    """Per-window probabilities, so thresholds and blends can be changed without refitting (see t2_blend.py)."""
    import pickle
    PROBS.mkdir(parents=True, exist_ok=True)
    with open(PROBS / f"{name}_{split}_{panel}.pkl", "wb") as fh:
        pickle.dump(probs, fh)


def threshold(probs: list[dict], th: dict) -> pd.DataFrame:
    rows = []
    for w in probs:
        n = w["prob"].shape[1]
        pred = np.zeros_like(w["prob"], dtype=np.int8)
        if w["condition"] == "queue_onset":
            p = w["prob"][HORIZON - 1]
            chosen = np.flatnonzero(p >= th["onset"])
            pred[HORIZON - 1, chosen if len(chosen) else [int(np.argmax(p))]] = 1
        else:
            pred = (w["prob"] >= th["ongoing"]).astype(np.int8)
        for s in range(HORIZON):
            rows.append(pd.DataFrame({"window_id": w["window_id"], "slot": w["T"] + s + 1, "pos": np.arange(n),
                                      "queue_pred": pred[s]}))
    return pd.concat(rows, ignore_index=True)


def iou_scores(pred: pd.DataFrame, panels: list[str]) -> dict:
    out = {}
    for p in panels:
        tr = t2_truth(p)
        m = tr.merge(pred, on=["window_id", "slot", "pos"], how="left")
        m = m[m.eligible]
        rows = []
        for (w, c), g in m.groupby(["window_id", "condition"]):
            y, yh = g.queue_true.to_numpy(bool), g.queue_pred.fillna(0).to_numpy(bool)
            u = (y | yh).sum()
            rows.append((c, 1.0 if u == 0 else (y & yh).sum() / u))
        d = pd.DataFrame(rows, columns=["cond", "iou"])
        out[p] = d.groupby("cond").iou.mean().to_dict()
    cond = {c: float(np.mean([out[p][c] for p in panels])) for c in ("queue_onset", "queue_ongoing")}
    _, overall = aggregate({p: np.mean(list(v.values())) for p, v in out.items()})
    return {"overall": overall, **cond}


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------
FEATS_DROP = ["y", "fold", "date", "kind", "panel"]


TUNED = ROOT / "work" / "models" / "t2_ongoing_params.json"   # written by task2_hpo.py


def lgb_params(kind: str, tuned: bool) -> dict:
    """The default LightGBM setting, or the random-search winner for the ongoing model."""
    if tuned and kind == "ongoing" and TUNED.exists():
        t = json.loads(TUNED.read_text())
        return {**LGB_PARAMS, **{k: t[k] for k in ("learning_rate", "num_leaves", "min_data_in_leaf",
                                                   "feature_fraction", "bagging_fraction", "lambda_l2")}}
    return LGB_PARAMS


def fit(algo: str, train: pd.DataFrame, valid: pd.DataFrame | None, rounds: int | None = None, params: dict | None = None):
    feats = [c for c in train.columns if c not in FEATS_DROP]
    if algo == "lgbm":
        dtr = lgb.Dataset(train[feats], train.y)
        if valid is not None:
            dva = lgb.Dataset(valid[feats], valid.y, reference=dtr)
            b = lgb.train(params or LGB_PARAMS, dtr, ROUNDS, valid_sets=[dva], callbacks=[lgb.early_stopping(EARLY, verbose=False)])
        else:
            b = lgb.train(params or LGB_PARAMS, dtr, rounds)
        return (lambda X: b.predict(X[feats])), b.best_iteration or rounds
    dtr = xgb.DMatrix(train[feats], label=train.y)
    if valid is not None:
        dva = xgb.DMatrix(valid[feats], label=valid.y)
        b = xgb.train(XGB_PARAMS, dtr, ROUNDS, evals=[(dva, "valid")], early_stopping_rounds=EARLY, verbose_eval=False)
        it = b.best_iteration + 1
    else:
        b = xgb.train(XGB_PARAMS, dtr, rounds)
        it = rounds
    return (lambda X: b.predict(xgb.DMatrix(X[feats]), iteration_range=(0, it))), it


def load_rows(panels: list[str], kind: str) -> pd.DataFrame:
    """Rows of one kind only (onset or ongoing), to keep memory low."""
    df = pd.concat([pd.read_parquet(DATA / f"{p}.parquet", filters=[("kind", "==", kind)]) for p in panels],
                   ignore_index=True)
    if kind == "onset":                 # onset rows carry no horizon step (always T+30)
        df = df.drop(columns=["step"], errors="ignore")
    return df.astype({"date": "category", "kind": "category", "panel": "category"})


def cmd_cv(args) -> None:
    rule = pd.concat([rule_predict(p, "train", RULES, loo=True) for p in T2_PANELS])
    base = iou_scores(rule[["window_id", "slot", "pos", "queue_pred"]], T2_PANELS)
    print(f"rules (current v4): overall {base['overall']:.4f}  onset {base['queue_onset']:.4f}  ongoing {base['queue_ongoing']:.4f}")
    for algo in args.algos:
        t0 = time.time()
        models = {0: {}, 1: {}}
        for kind in ("onset", "ongoing"):
            d = load_rows(T2_PANELS, kind)
            for k in (0, 1):
                fn, it = fit(algo, d[d.fold != k], d[d.fold == k], params=lgb_params(kind, args.tuned))
                models[k][kind] = fn
                print(f"  {algo} fold {k} {kind}: {it} rounds ({len(d):,} rows)", flush=True)
            del d
        probs = {p: window_probs(p, "train", models, by_fold=True) for p in T2_PANELS}
        for th_on in (0.2, 0.3, 0.4, 0.5):
            for th_go in (0.3, 0.4, 0.5, 0.6):
                pred = pd.concat([threshold(probs[p], {"onset": th_on, "ongoing": th_go}) for p in T2_PANELS])
                sc = iou_scores(pred, T2_PANELS)
                print(f"  {algo} theta onset {th_on} ongoing {th_go}: overall {sc['overall']:.4f}  "
                      f"onset {sc['queue_onset']:.4f}  ongoing {sc['queue_ongoing']:.4f}", flush=True)
        print(f"  {algo} done in {time.time() - t0:.0f}s", flush=True)


def cmd_predict(args) -> None:
    models = {}
    for kind in ("onset", "ongoing"):
        d = load_rows(T2_PANELS, kind)
        prm = lgb_params(kind, args.tuned)
        _, it0 = fit(args.algo, d[d.fold != 0], d[d.fold == 0], params=prm)
        _, it1 = fit(args.algo, d[d.fold != 1], d[d.fold == 1], params=prm)
        models[kind], _ = fit(args.algo, d, None, int(1.1 * (it0 + it1) / 2), params=prm)
    th = {"onset": args.theta_onset, "ongoing": args.theta_ongoing}
    OUT.mkdir(parents=True, exist_ok=True)
    for split in args.splits:
        parts = []
        for p in T2_PANELS:
            probs = window_probs(p, split, models, by_fold=False)
            save_probs(probs, f"t2ml_{args.algo}{args.tag}", split, p)
            pr = threshold(probs, th)
            tmpl = pd.read_csv(RELEASE / "task2" / p / split / "sample_submission_queue.csv",
                               usecols=["window_id", "timestamp", "link_id"])
            tmpl["slot"] = slot_of(tmpl.timestamp)
            tmpl["pos"] = tmpl.link_id.map(dict(zip(links(p).link_id, range(len(links(p))))))
            parts.append(tmpl.merge(pr, on=["window_id", "slot", "pos"], how="left").fillna({"queue_pred": 0}))
        out = pd.concat(parts)
        out["queue_pred"] = out.queue_pred.astype(int)
        path = OUT / f"queue_{split}_ml_{args.algo}{args.tag}.csv"
        out[["window_id", "timestamp", "link_id", "queue_pred"]].to_csv(path, index=False)
        print(f"wrote {path} ({len(out):,} rows, {int(out.queue_pred.sum()):,} queued cells)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("build")
    s.add_argument("--panels", nargs="+", default=T2_PANELS)
    s = sub.add_parser("cv")
    s.add_argument("--algos", nargs="+", default=["lgbm", "xgb"], choices=["lgbm", "xgb"])
    s.add_argument("--tuned", action="store_true", help="ongoing LightGBM with the task2_hpo.py winner")
    s = sub.add_parser("predict")
    s.add_argument("--algo", default="lgbm", choices=["lgbm", "xgb"])
    s.add_argument("--theta-onset", type=float, default=0.3)
    s.add_argument("--theta-ongoing", type=float, default=0.5)
    s.add_argument("--tag", default="", help="suffix for the output file names")
    s.add_argument("--splits", nargs="+", default=["validation", "private"])
    s.add_argument("--tuned", action="store_true", help="ongoing LightGBM with the task2_hpo.py winner")
    args = ap.parse_args()
    {"build": cmd_build, "cv": cmd_cv, "predict": cmd_predict}[args.cmd](args)


if __name__ == "__main__":
    main()
