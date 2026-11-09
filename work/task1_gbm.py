#!/usr/bin/env python
"""Task 1 second model: LightGBM corrections on top of the interpolation baseline (task1_interp tv_qb).

For every target cell the features describe what the same day shows around it: the nearest readings
before and after it in time and how far away they are, the nearest readings upstream and downstream
along the direction of travel, the neighbours' own temporal estimates, the usual level for that link
and time of day, the ramps, the link geometry and the time. Two regressors learn the residual of the
baseline: speed_true - speed_base and (flow_true - flow_base) / lanes.

Leakage guards
  * Profiles (usual level per link x time of day x weekday/weekend) come from the visible masked layer
    of train, never from the answers, so a train target never sees its own value.
  * Out-of-fold scoring: months alternate between two folds; each fold is predicted by a model fitted
    on the other one, so every train target gets an honest prediction for evaluate.py (T1 and T3).

Usage (from the "IEEE Big data" folder):
  python work/task1_gbm.py dataset            # sampled training rows -> work/cache/t1_gbm_sample.parquet
  python work/task1_gbm.py cv                 # two-fold out-of-fold T1 + T3 proxy vs the baseline
  python work/task1_gbm.py fit                # final models on all sampled rows -> work/models/
  python work/task1_gbm.py predict            # work/sub/state_<split>.csv for validation and private
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import (CACHE, CLEAN, PANELS, REGIMES, ROOT, STEPS, aggregate, links, log_result,  # noqa: E402
                      score_t1_panel, t1_targets, t3_proxy_panel)
from task1_interp import KM_PER_STEP_FLOW, OUT, blend, state_frame  # noqa: E402

MODELS = ROOT / "work" / "models"
SAMPLE = CACHE / "t1_gbm_sample.parquet"
SAMPLE_RATE = 0.04
SEED = 2026
SPEED_MIN, SPEED_MAX = 5.0, 120.75          # 120.75 is the release's clipping ceiling
PARAMS = {"objective": "regression", "learning_rate": 0.1, "num_leaves": 127, "min_data_in_leaf": 200,
          "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0,
          "num_threads": 16, "verbose": -1, "seed": SEED}
MAX_ROUNDS, EARLY_STOP = 2000, 100

DAY_COLS = ["ts", "tod", "weekday", "mask_regime", "pct_observed", "is_t1_target", "speed_obs", "flow_obs",
            "on_ramp_flow", "off_ramp_flow"]


# ---------------------------------------------------------------------------
# per-panel context: static arrays, travel direction, usual levels
# ---------------------------------------------------------------------------
class Panel:
    def __init__(self, panel: str):
        self.name = panel
        L = links(panel)
        self.n = len(L)
        report = json.loads((CLEAN / "cleaning_report.json").read_text(encoding="utf-8"))
        share = report[panel]["network"]["upstream_has_lower_milepost_share"]
        self.direction = 1 if share is None or share >= 0.5 else -1    # +1: traffic flows toward higher milepost
        self.perm = np.arange(self.n) if self.direction > 0 else np.arange(self.n)[::-1]   # pos -> flow order
        self.inv = np.argsort(self.perm)
        self.x_flow = (L.milepost.to_numpy(float) * self.direction)[self.perm]          # increasing along flow
        self.lanes = L.lanes.to_numpy(float)
        self.ffs = L.free_speed_kmh.to_numpy(float)
        self.static = {"lanes": self.lanes, "ffs": self.ffs, "cap_l": L.capacity_vph.to_numpy(float) / self.lanes,
                       "length": L.length_km.to_numpy(float), "kjam_l": L.k_jam.to_numpy(float) / self.lanes,
                       "pos_rel": self.inv / max(self.n - 1, 1), "n_on": L.on_ramp_n.to_numpy(float),
                       "n_off": L.off_ramp_n.to_numpy(float)}
        self.prof_v, self.prof_q = self.profiles()

    def profiles(self) -> tuple[np.ndarray, np.ndarray]:
        """Mean visible eligible speed / flow per (weekday-or-weekend, tod, link) from the train masked layer."""
        path = CACHE / f"t1_profile_{self.name}.npz"
        if path.exists():
            z = np.load(path)
            return z["v"], z["q"]
        sv, sq, cnt = (np.zeros((2, STEPS, self.n)) for _ in range(3))
        pf = pq.ParquetFile(CLEAN / self.name / "train.parquet")
        for i in range(pf.num_row_groups):
            tb = pf.read_row_group(i, columns=["weekday", "speed_clean", "flow_clean"])
            w = int(tb.column("weekday")[0].as_py() >= 5)
            v = tb.column("speed_clean").to_numpy().astype(float).reshape(STEPS, self.n)
            q = tb.column("flow_clean").to_numpy().astype(float).reshape(STEPS, self.n)
            ok = np.isfinite(v) & np.isfinite(q)
            sv[w] += np.where(ok, v, 0)
            sq[w] += np.where(ok, q, 0)
            cnt[w] += ok
        with np.errstate(invalid="ignore"):
            pv, pq_ = sv / cnt, sq / cnt
        # Fill rare empty slots from the other day type, then from the link mean.
        for a in (pv, pq_):
            a[np.isnan(a)] = np.broadcast_to(a[::-1], a.shape)[np.isnan(a)]
            a[np.isnan(a)] = np.broadcast_to(np.nanmean(a, axis=(0, 1)), a.shape)[np.isnan(a)]
        CACHE.mkdir(parents=True, exist_ok=True)
        np.savez(path, v=pv.astype(np.float32), q=pq_.astype(np.float32))
        return pv.astype(np.float32), pq_.astype(np.float32)


def nearest_along(G: np.ndarray, x: np.ndarray, axis: int):
    """Nearest visible value before / after each cell along an axis, with distances (NaN if none)."""
    G = np.moveaxis(G, axis, 0)
    m = G.shape[0]
    idx = np.arange(m)[:, None]
    vis = np.isfinite(G)
    prev = np.maximum.accumulate(np.where(vis, idx, -1), axis=0)
    nxt = np.minimum.accumulate(np.where(vis, idx, m)[::-1], axis=0)[::-1]
    hp, hn = prev >= 0, nxt < m
    pc, nc = prev.clip(0, m - 1), nxt.clip(0, m - 1)
    vp = np.where(hp, np.take_along_axis(G, pc, 0), np.nan)
    vn = np.where(hn, np.take_along_axis(G, nc, 0), np.nan)
    xs = x[:, None]
    dp = np.where(hp, xs - x[pc], np.nan)
    dn = np.where(hn, x[nc] - xs, np.nan)
    return [np.moveaxis(a, 0, axis) for a in (vp, vn, dp, dn, pc, nc)]


def lin(vp, vn, dp, dn):
    """Linear interpolation from the two nearest readings; one-sided where only one exists."""
    with np.errstate(invalid="ignore", divide="ignore"):
        both = vp + (vn - vp) * dp / (dp + dn)
    return np.where(np.isfinite(both), both, np.where(np.isfinite(vp), vp, vn))


def shift_flow(A: np.ndarray, ctx: Panel, k: int) -> np.ndarray:
    """Value of the link k steps upstream (k>0) or downstream (k<0) along the direction of travel."""
    F = A[:, ctx.perm]
    out = np.full_like(F, np.nan)
    if k > 0:
        out[:, k:] = F[:, :-k]
    else:
        out[:, :k] = F[:, -k:]
    return out[:, ctx.inv]


def day_features(ctx: Panel, tb, with_truth: bool):
    """Features for the target cells of one day, plus the baseline and (train) the residual targets."""
    n = ctx.n
    tgt = tb.column("is_t1_target").to_numpy()
    if not tgt.any():
        return None
    g = lambda c: tb.column(c).to_numpy().astype(float).reshape(STEPS, n)
    V, Q = g("speed_obs"), g("flow_obs")
    t = np.arange(STEPS, dtype=float)
    # temporal neighbours
    vp, vn, gp, gn, ip, inx = nearest_along(V, t, 0)
    qp, qn, _, _, _, _ = nearest_along(Q, t, 0)
    vt, qt = lin(vp, vn, gp, gn), lin(qp, qn, gp, gn)
    # spatial neighbours along the direction of travel
    Vf, Qf = V[:, ctx.perm], Q[:, ctx.perm]
    up_v, dn_v, du, dd, _, _ = nearest_along(Vf, ctx.x_flow, 1)
    up_q, dn_q, _, _, _, _ = nearest_along(Qf, ctx.x_flow, 1)
    back = lambda a: a[:, ctx.inv]
    up_v, dn_v, du, dd, up_q, dn_q = map(back, (up_v, dn_v, du, dd, up_q, dn_q))
    vs, qs = lin(up_v, dn_v, du, dd), lin(up_q, dn_q, du, dd)
    gs = np.fmin(du, dd)
    # baseline = task1_interp tv_qb
    base_v = np.where(np.isfinite(vt), vt, vs)
    base_q = blend(qt, np.fmin(gp, gn), qs, gs, KM_PER_STEP_FLOW)
    w = int(tb.column("weekday")[0].as_py() >= 5)
    pv, pq_ = ctx.prof_v[w].astype(float), ctx.prof_q[w].astype(float)
    P = g("pct_observed")
    lanes = np.broadcast_to(ctx.lanes, (STEPS, n))
    regime = REGIMES.index(tb.column("mask_regime")[0].as_py())
    feats = {
        "base_v": base_v, "base_q_l": base_q / lanes, "vt": vt, "qt_l": qt / lanes, "vs": vs, "qs_l": qs / lanes,
        "gp": gp, "gn": gn, "v_prev": vp, "v_next": vn, "q_prev_l": qp / lanes, "q_next_l": qn / lanes,
        "v_up": up_v, "v_dn": dn_v, "q_up_l": up_q / lanes, "q_dn_l": dn_q / lanes, "du": du, "dd": dd,
        "vt_up1": shift_flow(vt, ctx, 1), "vt_dn1": shift_flow(vt, ctx, -1),
        "vt_up2": shift_flow(vt, ctx, 2), "vt_dn2": shift_flow(vt, ctx, -2),
        "qt_up1_l": shift_flow(qt, ctx, 1) / lanes, "qt_dn1_l": shift_flow(qt, ctx, -1) / lanes,
        "prof_v": pv, "prof_q_l": pq_ / lanes,
        "v_prev_dev": vp - np.take_along_axis(pv, ip, 0), "v_next_dev": vn - np.take_along_axis(pv, inx, 0),
        "q_prev_dev_l": (qp - np.take_along_axis(pq_, ip, 0)) / lanes,
        "q_next_dev_l": (qn - np.take_along_axis(pq_, inx, 0)) / lanes,
        "pct": P, "pct_prev": np.take_along_axis(P, ip, 0), "pct_next": np.take_along_axis(P, inx, 0),
        "on_r_l": g("on_ramp_flow") / lanes, "off_r_l": g("off_ramp_flow") / lanes,
        "tod": np.broadcast_to(t[:, None], (STEPS, n)),
        "weekday": np.full((STEPS, n), float(tb.column("weekday")[0].as_py())),
        "regime": np.full((STEPS, n), float(regime)),
    }
    for k, a in ctx.static.items():
        feats[k] = np.broadcast_to(np.asarray(a, float), (STEPS, n))
    X = pd.DataFrame({k: np.asarray(a).ravel()[tgt].astype(np.float32) for k, a in feats.items()})
    out = {"X": X, "base_v": base_v.ravel()[tgt], "base_q": base_q.ravel()[tgt], "lanes": lanes.ravel()[tgt]}
    if with_truth:
        out["y_v"] = tb.column("speed_true").to_numpy()[tgt] - out["base_v"]
        out["y_q"] = (tb.column("flow_true").to_numpy()[tgt] - out["base_q"]) / out["lanes"]
    return out


def iter_days(panel: str, split: str, with_truth: bool):
    ctx = Panel(panel)
    cols = DAY_COLS + (["speed_true", "flow_true"] if with_truth else [])
    pf = pq.ParquetFile(CLEAN / panel / f"{split}.parquet")
    for i in range(pf.num_row_groups):
        tb = pf.read_row_group(i, columns=cols)
        month = pd.Timestamp(tb.column("ts")[0].as_py()).month
        d = day_features(ctx, tb, with_truth)
        if d is not None:
            yield i, month, d


def fold_of(month: int) -> int:
    return month % 2


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------
def cmd_dataset(args) -> None:
    parts = []
    for p in args.panels:
        t0 = time.time()
        for i, month, d in iter_days(p, "train", True):
            rng = np.random.default_rng(SEED + 1000 * PANELS.index(p) + i)
            keep = rng.random(len(d["X"])) < SAMPLE_RATE
            x = d["X"][keep].copy()
            x["y_v"], x["y_q"] = d["y_v"][keep].astype(np.float32), d["y_q"][keep].astype(np.float32)
            x["fold"] = np.int8(fold_of(month))
            parts.append(x)
        print(f"  {p}: {time.time() - t0:.0f}s, {sum(len(x) for x in parts):,} rows so far", flush=True)
    df = pd.concat(parts, ignore_index=True)
    df.to_parquet(SAMPLE, index=False)
    print(f"wrote {SAMPLE}: {len(df):,} rows x {df.shape[1]} cols")


def train_model(df: pd.DataFrame, target: str, valid: pd.DataFrame | None, rounds: int | None = None):
    feats = [c for c in df.columns if c not in ("y_v", "y_q", "fold")]
    dtrain = lgb.Dataset(df[feats], df[target], free_raw_data=True)
    if valid is not None:
        dvalid = lgb.Dataset(valid[feats], valid[target], reference=dtrain)
        model = lgb.train(PARAMS, dtrain, MAX_ROUNDS, valid_sets=[dvalid],
                          callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)])
    else:
        model = lgb.train(PARAMS, dtrain, rounds)
    return model


def predict_cells(models: dict, d: dict) -> tuple[np.ndarray, np.ndarray]:
    v = d["base_v"] + models["v"].predict(d["X"], num_threads=PARAMS["num_threads"])
    q = d["base_q"] + d["lanes"] * models["q"].predict(d["X"], num_threads=PARAMS["num_threads"])
    v = np.where(np.isfinite(v), v, d["base_v"])
    q = np.where(np.isfinite(q), q, d["base_q"])
    return np.clip(v, SPEED_MIN, SPEED_MAX), np.maximum(q, 0.0)


def cmd_cv(args) -> None:
    """Out-of-fold T1 + T3 proxy per panel. Resumable: fold models and per-panel results are saved."""
    MODELS.mkdir(parents=True, exist_ok=True)
    rounds_path = MODELS / "t1_gbm_cv_rounds.json"
    fold_file = lambda k, n: MODELS / f"t1_gbm_fold{k}_{n}.txt"
    if all(fold_file(k, n).exists() for k in (0, 1) for n in ("v", "q")) and rounds_path.exists():
        models = {k: {n: lgb.Booster(model_file=str(fold_file(k, n))) for n in ("v", "q")} for k in (0, 1)}
        print("  reusing saved fold models", flush=True)
    else:
        df = pd.read_parquet(SAMPLE)
        models, rounds = {}, {}
        for k in (0, 1):
            tr, va = df[df.fold != k], df[df.fold == k]
            models[k] = {}
            for name, target in (("v", "y_v"), ("q", "y_q")):
                t0 = time.time()
                m = train_model(tr, target, va)
                m.save_model(str(fold_file(k, name)), num_iteration=m.best_iteration)
                models[k][name] = m
                rounds[f"{k}_{name}"] = m.best_iteration
                print(f"  fold {k} {name}: best_iter={m.best_iteration} valid_l2={m.best_score['valid_0']['l2']:.3f} "
                      f"({time.time() - t0:.0f}s)", flush=True)
        del df
        rounds_path.write_text(json.dumps(rounds))
    done_path = ROOT / "work" / "results" / "t1_gbm_cv_panels.jsonl"
    done = {}
    if done_path.exists():
        done = {r["panel"]: r for r in map(json.loads, done_path.read_text().splitlines()) if r}
    rows = [done[p] for p in args.panels if p in done]
    for p in [p for p in args.panels if p not in done]:
        tg = t1_targets(p, "train")
        pv, pqf, bv, bq = [], [], [], []
        for i, month, d in iter_days(p, "train", False):
            v, q = predict_cells(models[fold_of(month)], d)      # the model that never saw this month
            pv.append(v); pqf.append(q); bv.append(d["base_v"]); bq.append(d["base_q"])
        pv, pqf, bv, bq = map(np.concatenate, (pv, pqf, bv, bq))
        r = {"panel": p}
        for tag, (v, q) in (("base", (bv, bq)), ("gbm", (pv, pqf))):
            r[f"{tag}_S_state"] = score_t1_panel(p, tg, v, q)["S_state"]
            r[f"{tag}_phys"] = t3_proxy_panel(p, tg, v, q)["S_physics_proxy"]
        rows.append(r)
        with done_path.open("a") as fh:
            fh.write(json.dumps(r) + "\n")
        print(f"  [{p}] T1 {r['base_S_state']:.4f} -> {r['gbm_S_state']:.4f}   T3 proxy {r['base_phys']:.4f} -> {r['gbm_phys']:.4f}",
              flush=True)
    res = pd.DataFrame(rows)
    for tag in ("base", "gbm"):
        s = aggregate(dict(zip(res.panel, res[f"{tag}_S_state"])))[1]
        ph = aggregate(dict(zip(res.panel, res[f"{tag}_phys"])))[1]
        print(f"{tag}: S_state={s:.4f}  S_physics_proxy={ph:.4f}  0.35*S+0.15*P={0.35 * s + 0.15 * ph:.4f}")
        if tag == "gbm":
            log_result("t1", "train", "task1_gbm:cv", f"lgbm residual, 2-fold OOF ({len(res)} panels)", s,
                       dict(zip(res.panel, res.gbm_S_state)))
            log_result("t3_proxy", "train", "task1_gbm:cv", f"lgbm residual, 2-fold OOF ({len(res)} panels)", ph,
                       dict(zip(res.panel, res.gbm_phys)))


def add_k_base(X: pd.DataFrame) -> pd.DataFrame:
    """The baseline density per lane, the starting point of the density model."""
    X = X.copy()
    X["k_base_l"] = (X.base_q_l / np.maximum(X.base_v, 1.0)).astype(np.float32)
    return X


def k_target(df: pd.DataFrame) -> pd.Series:
    """Residual of density per lane: k = q / v, the quantity Task 3's conservation check integrates."""
    v_t = df.base_v + df.y_v
    k_t = (df.base_q_l + df.y_q) / np.maximum(v_t, 1.0)
    return (k_t - df.base_q_l / np.maximum(df.base_v, 1.0)).astype(np.float32)


K_PARAMS = {**PARAMS, "objective": "regression_l1"}   # S_LWR sums absolute errors in N = k * L


def train_k_model(df: pd.DataFrame, valid: pd.DataFrame | None, rounds: int | None = None):
    feats = [c for c in df.columns if c not in ("y_v", "y_q", "fold")]
    dtrain = lgb.Dataset(add_k_base(df[feats]), k_target(df), weight=df.length, free_raw_data=True)
    if valid is None:
        return lgb.train(K_PARAMS, dtrain, rounds)
    dvalid = lgb.Dataset(add_k_base(valid[feats]), k_target(valid), weight=valid.length, reference=dtrain)
    return lgb.train(K_PARAMS, dtrain, MAX_ROUNDS, valid_sets=[dvalid],
                     callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)])


def combine(v: np.ndarray, q: np.ndarray, k: np.ndarray, w: tuple[float, float, float] | None):
    """Make (v, q) agree with a density estimate. w = None: q = k * v. Otherwise weighted least squares in
    log space: min w_v da^2 + w_q db^2 + w_k (log q' - log v' - log k)^2 over the adjustments da, db."""
    k = np.maximum(k, 1e-3)
    if w is None:
        return v, k * v
    wv, wq, wk = w
    a, b = np.log(np.maximum(v, 1.0)), np.log(np.maximum(q, 1.0))
    r = b - a - np.log(k)
    s = r / (1 + wk * (1 / wv + 1 / wq))
    return np.exp(a + wk * s / wv), np.exp(b - wk * s / wq)


def cmd_cvk(args) -> None:
    """Out-of-fold check of a density model and of ways to reconcile it with the speed and flow models."""
    fold_file = lambda k, n: MODELS / f"t1_gbm_fold{k}_{n}.txt"
    models = {k: {n: lgb.Booster(model_file=str(fold_file(k, n))) for n in ("v", "q")} for k in (0, 1)}
    if all(fold_file(k, "k").exists() for k in (0, 1)):
        for k in (0, 1):
            models[k]["k"] = lgb.Booster(model_file=str(fold_file(k, "k")))
    else:
        df = pd.read_parquet(SAMPLE)
        for k in (0, 1):
            t0 = time.time()
            m = train_k_model(df[df.fold != k], df[df.fold == k])
            m.save_model(str(fold_file(k, "k")), num_iteration=m.best_iteration)
            models[k]["k"] = m
            print(f"  fold {k} k: best_iter={m.best_iteration} valid_l1={m.best_score['valid_0']['l1']:.4f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
        del df
    variants = {"A_vq": "vq", "B_q=kv": None, "C_w111": (1, 1, 1), "C_w114": (1, 1, 4), "C_w411": (4, 1, 1)}
    rows = []
    for p in args.panels:
        tg = t1_targets(p, "train")
        acc = {name: ([], []) for name in variants}
        for i, month, d in iter_days(p, "train", False):
            m = models[fold_of(month)]
            v, q = predict_cells(m, d)
            kk = d["base_q"] / np.maximum(d["base_v"], 1.0) + d["lanes"] * m["k"].predict(add_k_base(d["X"]))
            for name, w in variants.items():
                vv, qq = (v, q) if w == "vq" else combine(v, q, kk, w)
                acc[name][0].append(np.clip(vv, SPEED_MIN, SPEED_MAX))
                acc[name][1].append(qq)
        for name, (vs, qs) in acc.items():
            vv, qq = np.concatenate(vs), np.concatenate(qs)
            s = score_t1_panel(p, tg, vv, qq)["S_state"]
            ph = t3_proxy_panel(p, tg, vv, qq)["S_physics_proxy"]
            rows.append({"panel": p, "variant": name, "S_state": s, "phys": ph, "obj": 0.35 * s + 0.15 * ph})
            print(f"  [{p}] {name:7s} T1={s:.4f} T3proxy={ph:.4f} 0.35T1+0.15T3={0.35 * s + 0.15 * ph:.4f}", flush=True)
    res = pd.DataFrame(rows)
    print(res.groupby("variant")[["S_state", "phys", "obj"]].mean().sort_values("obj", ascending=False)
          .to_string(float_format=lambda x: f"{x:.4f}"))


def cmd_fit(args) -> None:
    """Final models on all sampled rows. Rounds come from the CV (learning rate 0.1), rescaled when the final
    fit uses a smaller learning rate: half the step size, twice the trees."""
    global PARAMS
    df = pd.read_parquet(SAMPLE)
    cv = json.loads((MODELS / "t1_gbm_cv_rounds.json").read_text())
    scale = PARAMS["learning_rate"] / args.lr
    PARAMS = {**PARAMS, "learning_rate": args.lr}
    for name, target in (("v", "y_v"), ("q", "y_q")):
        if name not in args.models:
            continue
        rounds = int(1.1 * scale * np.mean([cv[f"0_{name}"], cv[f"1_{name}"]]))
        t0 = time.time()
        m = train_model(df, target, None, rounds)
        m.save_model(str(MODELS / f"t1_gbm_{name}.txt"))
        print(f"  saved t1_gbm_{name}.txt ({rounds} rounds at lr {args.lr}, {time.time() - t0:.0f}s)", flush=True)
    if "k" in args.models:
        # The L1 density model trains slowly; it keeps the CV learning rate (0.1) and the CV tree count.
        rounds = int(1.1 * np.mean([lgb.Booster(model_file=str(MODELS / f"t1_gbm_fold{k}_k.txt")).num_trees()
                                    for k in (0, 1)]))
        t0 = time.time()
        m = train_k_model(df, None, rounds)
        m.save_model(str(MODELS / "t1_gbm_k.txt"))
        print(f"  saved t1_gbm_k.txt ({rounds} rounds at lr {K_PARAMS['learning_rate']}, {time.time() - t0:.0f}s)",
              flush=True)


COMBINE_W = (1.0, 1.0, 1.0)   # C_w111: best 0.35*T1 + 0.15*T3 in the out-of-fold comparison (cmd_cvk)


def cmd_predict(args) -> None:
    models = {k: lgb.Booster(model_file=str(MODELS / f"t1_gbm_{k}.txt")) for k in ("v", "q")}
    use_k = args.combine and (MODELS / "t1_gbm_k.txt").exists()
    if use_k:
        models["k"] = lgb.Booster(model_file=str(MODELS / "t1_gbm_k.txt"))
    print(f"  speed + flow models{' + density model, combined with weights ' + str(COMBINE_W) if use_k else ''}")
    OUT.mkdir(parents=True, exist_ok=True)
    for split in args.splits:
        path = OUT / f"state_{split}.csv"
        for j, p in enumerate(args.panels):
            vs, qs = [], []
            for _, _, d in iter_days(p, split, False):
                v, q = predict_cells(models, d)
                if use_k:
                    k = d["base_q"] / np.maximum(d["base_v"], 1.0) + d["lanes"] * models["k"].predict(add_k_base(d["X"]))
                    v, q = combine(v, q, k, COMBINE_W)
                    v = np.clip(v, SPEED_MIN, SPEED_MAX)
                vs.append(v); qs.append(q)
            v, q = np.concatenate(vs), np.concatenate(qs)
            state_frame(p, split, v, q).to_csv(path, mode="w" if j == 0 else "a", header=j == 0, index=False)
            print(f"  [{split}] {p}: {len(v):,} rows", flush=True)
        print(f"wrote {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("dataset", "cv", "cvk", "fit", "predict"):
        s = sub.add_parser(name)
        s.add_argument("--panels", nargs="+", default=PANELS)
        if name == "predict":
            s.add_argument("--splits", nargs="+", default=["validation", "private"])
            s.add_argument("--no-combine", dest="combine", action="store_false",
                           help="skip the density model and use the speed and flow models as they are")
        if name == "fit":
            s.add_argument("--lr", type=float, default=0.05)
            s.add_argument("--models", nargs="+", default=["v", "q", "k"], choices=["v", "q", "k"])
    args = ap.parse_args()
    {"dataset": cmd_dataset, "cv": cmd_cv, "cvk": cmd_cvk, "fit": cmd_fit, "predict": cmd_predict}[args.cmd](args)


if __name__ == "__main__":
    main()
