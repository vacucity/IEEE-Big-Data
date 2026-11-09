"""V57 lane B: Task-1 state inside the official Task-2 blackouts (T, T+90] - a gap-cell residual GBM on top of the
production fill.  ONLY the Task-1 target rows inside official Task-2 blackouts change; every other state row, every
queue row and every ODME row stays byte-identical to V56 (work/submissions/v56_t1_on54.zip).

Background (work/t1_v57_blkruler.py, work/V57_BLACKOUT_README.md): inside an official blackout the production writes
the unstacked mean of the two U-Net tags.  Simulated official-like blackouts on the March and April visible layers
(replica-selector origins, every visible cell of the corridor hidden in rows T+1..T+18) give an in-gap speed RMSE of
4.7-10.7 km/h and flow RMSE of 62-106 veh/h/lane (outside: 1.3-2.2 / 25-39).  The ~2% blackout targets cost about
0.0039 (March) / 0.0042 (April) of the total score against a perfect fill (S_state 0.0066/0.0076, S_physics est.
0.0106/0.0100).  The U-Nets were trained with random 90-minute gaps; official windows are placed by the selector at
queue events (onset: a queue enters the horizon, mostly at T+30; ongoing: an established queue at T), and the gap
fill does not know that.

Method: LightGBM residual models (speed in km/h, flow per lane) on the eligible hidden gap cells of the simulations,
features = the U-Net outputs (tags, fold disagreement, neighbours, link minimum over the gap), the visible state at
T and right after T+90 (observations, U-Net residuals there, linear interpolation, queued share of the corridor),
the window condition (public window_index.csv label), k = steps after T, time of day, the day's mask rate and static
link data.  Fill = production + lambda * predicted residual (speed clipped to [5, 120.75], flow >= 0).

Steps (project root, anaconda python, OMP_NUM_THREADS=4):
  frame      real-frame bookkeeping: T1 target counts inside official blackouts by regime/condition, Task-3
             denominators per corridor-month-regime (results/t1_v57_blackout/frame_stats.csv, official_windows.csv)
  features   gap-cell features of the simulated windows (results/t1_v57_blkruler/sim_wide) -> cache parquet
  crossfit   pre-registered grid, cross-fitted on day parity (fit on one parity of both months and all corridors,
             predict the other), real-frame emulation of 0.35 dS_state + 0.15 dS_phys per corridor-month,
             selection rule (docstring of cmd_crossfit), nested month check, per-regime breakdown
  fit        final models on all simulated gap cells -> work/models/t1_blk_v57_{v,q}.txt
  official   U-Net outputs (CPU fp32, same code as the simulation) around the 160 official windows
  apply      features at the official blackout targets -> corrections -> work/sub/state_{split}_v57blk.csv
             (all state rows of v56_t1_on54.zip; only blackout target rows changed)
  package    t1_v52_monthbias package code (names patched) -> work/submissions/v57blk_state.zip (base v51_april.zip)
  audit      t1_v52_monthbias audit code (names patched): v57blk_state.zip vs v51_april.zip
  (then: python work/build_queue_candidate.py --name v57_blk --base work/submissions/v57blk_state.zip
         --queue work/sub/queue_validation_v54_onset.csv work/sub/queue_private_v54_onset.csv --note "...")
  compare    line-by-line v57_blk.zip vs v56_t1_on54.zip: only blackout target state rows differ

Rules: Task 1 is offline; only visible observations of March/April, the released window_index.csv (origin and
condition label of each official window) and the frozen production models are used.  No hidden label, no Task-2
prediction, no public-LB feedback.  The simulated origins (t1_v57_blkruler.py origins) only place T1 blackouts.
Nothing existing is overwritten (FileExistsError).
"""
from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "4")

import argparse
import json
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
import t1_v52_monthbias as M  # noqa: E402
from evaluate import (CLEAN, FAMILY, KEY_BASE, PANELS, REGIMES, RELEASE, STEPS, T2_PANELS, links,  # noqa: E402
                      t1_targets)

WORK = Path(__file__).resolve().parent
OUT = WORK / "results" / "t1_v57_blackout"
RULER = WORK / "results" / "t1_v57_blkruler"
SIM_WIDE = RULER / "sim_wide"
OFFICIAL = OUT / "official"
FEAT = OUT / "features"
MODELS = WORK / "models"
SPLITS = ("validation", "private")
V56_ZIP = WORK / "submissions" / "v56_t1_on54.zip"
V51_ZIP = WORK / "submissions" / "v51_april.zip"
NAME = "v57blk_state"
FULL_NAME = "v57_blk"
CSV = {s: WORK / "sub" / f"state_{s}_v57blk.csv" for s in SPLITS}
GAP = 18
K0, K1 = -6, GAP + 6
NK = K1 - K0 + 1
RATE = np.array([0.2, 0.3, 0.5])
W_STATE, W_PHYS = 0.35, 0.15
SPEED_MIN, SPEED_MAX = 5.0, 120.75
COMP_TAGS = ("_w48", "_w48lwr")
FEATS = ["k", "cond", "tod", "rate", "ffs", "vc_r", "lanes", "cap_l", "len", "xrel", "pv_r", "v_w48_r", "v_w48lwr_r",
         "pq_l", "q_w48_l", "q_w48lwr_l", "dv_tags", "dv_folds", "dq_folds", "pv_kprev", "pv_knext", "pv_min_link",
         "pv_up", "pv_dn", "e0v_r", "e1v_r", "e0q_l", "e1q_l", "d0", "d1", "r0v", "r1v", "r0q_l", "r1q_l", "e0v_up",
         "e0v_dn", "e1v_up", "e1v_dn", "lin_v_r", "lin_q_l", "qsh0", "qsh19", "pq_share", "qd0", "qd19"]
GBM_PARAMS = dict(objective="regression", learning_rate=0.05, num_leaves=63, min_data_in_leaf=500,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=10, num_threads=4,
                  verbose=-1, seed=57, deterministic=True, force_col_wise=True)
GBM_ROUNDS = 400


def write_new(path: Path, text: str) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.write_text(text, encoding="utf-8")


def g6(x: np.ndarray) -> np.ndarray:
    return np.array([float("%.6g" % v) for v in x])


# ---------------------------------------------------------------------------
# window tensors and gap-cell features (same code for simulated and official windows)
# ---------------------------------------------------------------------------
WIN_FLOAT = ["ov", "oq", "pv", "pq"] + [f"{c}{t}" for t in COMP_TAGS for c in "vq"] + \
            [f"{c}{t}f{f}" for t in COMP_TAGS for c in "vq" for f in (0, 1)]
WIN_BOOL = ["elig", "vis", "tgt", "hidden", "ramp_ok"]


def win_tensors(p: str, df: pd.DataFrame):
    """Long rows (day, T, k, pos, ...) -> arrays (W, NK, n) in pos order + window meta (one row per (day, T))."""
    n = len(links(p))
    df = df[(df.k >= K0) & (df.k <= K1)]
    w = df.day.astype(np.int64).to_numpy() * 1000 + df["T"].astype(np.int64).to_numpy()
    uw, wi = np.unique(w, return_inverse=True)
    ki = df.k.to_numpy(np.int64) - K0
    li = df.pos.to_numpy(np.int64)
    A = {}
    for c in WIN_FLOAT:
        a = np.full((len(uw), NK, n), np.nan, np.float32)
        a[wi, ki, li] = df[c].to_numpy(np.float32)
        A[c] = a
    for c in WIN_BOOL:
        a = np.zeros((len(uw), NK, n), bool)
        a[wi, ki, li] = df[c].to_numpy(bool)
        A[c] = a
    agg = dict(day=("day", "first"), T=("T", "first"), cond=("cond", "first"), regime=("regime", "first"))
    if "window_id" in df:
        agg["window_id"] = ("window_id", "first")
    meta = df.groupby(w).agg(**agg).loc[uw].reset_index(names="w")
    return A, meta


def _nearest(a, vis, rows):
    W, _, n = a.shape
    out = np.full((W, n), np.nan, np.float32)
    dist = np.full((W, n), np.nan, np.float32)
    for j, r in enumerate(rows):
        m = np.isnan(out) & vis[:, r, :]
        out[m] = a[:, r, :][m]
        dist[m] = j
    return out, dist


def gap_features(p: str, A: dict, meta: pd.DataFrame) -> pd.DataFrame:
    """Features of the cells to fill: eligible 'hidden' cells in rows k = 1..18 (simulation: visible cells hidden by
    the simulated blackout; official: Task-1 targets inside the official blackout)."""
    import t1_unet as U
    L = links(p)
    n = len(L)
    perm = U.static(p)["perm"]                # travel index -> pos
    inv = np.argsort(perm)                    # pos -> travel index
    ffs = L.free_speed_kmh.to_numpy(np.float32)
    cap = L.capacity_vph.to_numpy(np.float32)
    lanes = L.lanes.to_numpy(np.float32)
    vc = L.v_cut_kmh.to_numpy(np.float32)
    length = L.length_km.to_numpy(np.float32)
    W = len(meta)
    k0 = -K0
    vis = A["vis"] & ~A["hidden"]
    pre_rows = [k0 - j for j in range(0, 7)]
    post_rows = [k0 + 19 + j for j in range(0, 6)]
    e0v, d0 = _nearest(A["ov"], vis, pre_rows)
    e0q, _ = _nearest(A["oq"], vis, pre_rows)
    e1v, d1 = _nearest(A["ov"], vis, post_rows)
    e1q, _ = _nearest(A["oq"], vis, post_rows)

    def edge_res(ch, rows):
        out = np.full((W, n), np.nan, np.float32)
        for r in rows:
            m = np.isnan(out) & vis[:, r, :]
            out[m] = (A["o" + ch][:, r, :] - A["p" + ch][:, r, :])[m]
        return out
    r0v, r1v = edge_res("v", pre_rows), edge_res("v", post_rows)
    r0q, r1q = edge_res("q", pre_rows), edge_res("q", post_rows)
    with np.errstate(invalid="ignore"):
        qd0 = (A["ov"][:, k0] <= vc) & vis[:, k0]
        qd19 = (A["ov"][:, k0 + 19] <= vc) & vis[:, k0 + 19]
        pq_share = (A["pv"][:, k0 + 1:k0 + 19] <= vc).mean((1, 2))
    qsh0 = qd0.sum(1) / np.maximum(vis[:, k0].sum(1), 1)
    qsh19 = qd19.sum(1) / np.maximum(vis[:, k0 + 19].sum(1), 1)
    krange = np.arange(NK)[None, :, None]
    wi, ki, li = np.nonzero(A["hidden"] & A["elig"] & (krange >= k0 + 1) & (krange <= k0 + 18))
    k = ki - k0
    F = {"wi": wi, "k": k, "pos": li}
    F["cond"] = (meta.cond.to_numpy()[wi] == "onset").astype(np.int8)
    F["tod"] = (meta["T"].to_numpy()[wi] + k).astype(np.float32)
    F["rate"] = RATE[meta.regime.to_numpy()[wi]]
    F["ffs"] = ffs[li]
    F["vc_r"] = vc[li] / ffs[li]
    F["lanes"] = lanes[li]
    F["cap_l"] = cap[li] / lanes[li]
    F["len"] = length[li]
    F["xrel"] = inv[li] / max(n - 1, 1)
    for c in ("pv", "v_w48", "v_w48lwr"):
        F[c + "_r"] = A[c][wi, ki, li] / ffs[li]
    for c in ("pq", "q_w48", "q_w48lwr"):
        F[c + "_l"] = A[c][wi, ki, li] / lanes[li]
    F["dv_tags"] = (A["v_w48"] - A["v_w48lwr"])[wi, ki, li]
    F["dv_folds"] = (np.abs(A["v_w48f0"] - A["v_w48f1"]) + np.abs(A["v_w48lwrf0"] - A["v_w48lwrf1"]))[wi, ki, li]
    F["dq_folds"] = (np.abs(A["q_w48f0"] - A["q_w48f1"]) + np.abs(A["q_w48lwrf0"] - A["q_w48lwrf1"]))[wi, ki, li] / lanes[li]
    F["pv_kprev"] = A["pv"][wi, ki - 1, li] / ffs[li]
    F["pv_knext"] = A["pv"][wi, ki + 1, li] / ffs[li]
    F["pv_min_link"] = np.nanmin(A["pv"][:, k0 + 1:k0 + 19, :], axis=1)[wi, li] / ffs[li]
    ti = inv[li]
    up = perm[np.clip(ti - 1, 0, n - 1)]
    dn = perm[np.clip(ti + 1, 0, n - 1)]
    F["pv_up"] = A["pv"][wi, ki, up] / ffs[up]
    F["pv_dn"] = A["pv"][wi, ki, dn] / ffs[dn]
    F["e0v_r"] = e0v[wi, li] / ffs[li]
    F["e1v_r"] = e1v[wi, li] / ffs[li]
    F["e0q_l"] = e0q[wi, li] / lanes[li]
    F["e1q_l"] = e1q[wi, li] / lanes[li]
    F["d0"] = d0[wi, li]
    F["d1"] = d1[wi, li]
    F["r0v"] = r0v[wi, li]
    F["r1v"] = r1v[wi, li]
    F["r0q_l"] = r0q[wi, li] / lanes[li]
    F["r1q_l"] = r1q[wi, li] / lanes[li]
    F["e0v_up"] = e0v[wi, up] / ffs[up]
    F["e0v_dn"] = e0v[wi, dn] / ffs[dn]
    F["e1v_up"] = e1v[wi, up] / ffs[up]
    F["e1v_dn"] = e1v[wi, dn] / ffs[dn]
    with np.errstate(invalid="ignore", divide="ignore"):
        wl = np.clip((k + d0[wi, li]) / (19 + d0[wi, li] + d1[wi, li]), 0, 1)
    F["lin_v_r"] = (e0v[wi, li] + wl * (e1v[wi, li] - e0v[wi, li])) / ffs[li]
    F["lin_q_l"] = (e0q[wi, li] + wl * (e1q[wi, li] - e0q[wi, li])) / lanes[li]
    F["qsh0"] = qsh0[wi]
    F["qsh19"] = qsh19[wi]
    F["pq_share"] = pq_share[wi]
    F["qd0"] = qd0[wi, li].astype(np.int8)
    F["qd19"] = qd19[wi, li].astype(np.int8)
    F = pd.DataFrame(F)
    F["w"] = meta.w.to_numpy()[wi]
    F["day"] = meta.day.to_numpy()[wi]
    F["T"] = meta["T"].to_numpy()[wi]
    F["regime"] = meta.regime.to_numpy()[wi]
    F["condname"] = meta.cond.to_numpy()[wi]
    F["ov"] = A["ov"][wi, ki, li]
    F["oq"] = A["oq"][wi, ki, li]
    F["pv"] = A["pv"][wi, ki, li]
    F["pq"] = A["pq"][wi, ki, li]
    if "window_id" in meta:
        F["window_id"] = meta.window_id.to_numpy()[wi]
    return F


def sim_features(p: str, split: str) -> pd.DataFrame:
    path = FEAT / f"{p}_{split}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    A, meta = win_tensors(p, pd.read_parquet(SIM_WIDE / f"{p}_{split}_win.parquet"))
    F = gap_features(p, A, meta)
    FEAT.mkdir(parents=True, exist_ok=True)
    F.to_parquet(path, index=False)
    return F


def sim_tensors(p: str, split: str):
    return win_tensors(p, pd.read_parquet(SIM_WIDE / f"{p}_{split}_win.parquet"))


# ---------------------------------------------------------------------------
# real-frame bookkeeping and the evaluation of a fill on the simulated windows
# ---------------------------------------------------------------------------
def cmd_frame(args) -> None:
    """Per (panel, split, regime): T1 targets, targets inside official blackouts, eligible cells, Task-3
    denominators estimated from the visible cells (sum |q_lane| over eligible cells, sum |dN| over eligible
    ramp-valid transitions); per official window: day, regime, origin row T, condition, blackout targets."""
    OUT.mkdir(parents=True, exist_ok=True)
    for f in ("frame_stats.csv", "official_windows.csv"):
        if (OUT / f).exists():
            raise FileExistsError(OUT / f)
    rows, wrows = [], []
    for p in PANELS:
        L = links(p)
        n, lanes, length = len(L), L.lanes.to_numpy(float), L.length_km.to_numpy(float)
        for sp in SPLITS:
            pf = pq.ParquetFile(CLEAN / p / f"{sp}.parquet")
            cond_of = {}
            if p in T2_PANELS:
                wi = pd.read_csv(RELEASE / "task2" / p / sp / "window_index.csv")
                for _, w in wi.iterrows():
                    T = pd.Timestamp(w.forecast_origin)
                    cond_of[(str(w.date), T.hour * 12 + T.minute // 5)] = (
                        "onset" if w.condition == "queue_onset" else "ongoing", w.window_id)
            acc = {}
            for i in range(pf.num_row_groups):
                t = pf.read_row_group(i, columns=["date", "mask_regime", "is_t1_target", "is_t2_blackout",
                                                  "is_score_eligible", "speed_obs", "flow_obs"] + M.RAMP_COLS)
                g = lambda c: t.column(c).to_numpy(zero_copy_only=False).reshape(STEPS, n)
                date = str(t.column("date")[0].as_py())
                reg = REGIMES.index(str(t.column("mask_regime")[0].as_py()))
                tg, b, e = g("is_t1_target"), g("is_t2_blackout"), g("is_score_eligible")
                v, q = g("speed_obs").astype(float), g("flow_obs").astype(float)
                vis = np.isfinite(v) & np.isfinite(q)
                ramp_ok = (((g("on_ramp_n") == 0) | (g("on_ramp_n_valid") > 0)) &
                           ((g("off_ramp_n") == 0) | (g("off_ramp_n_valid") > 0)))
                ve = vis & e
                with np.errstate(invalid="ignore", divide="ignore"):
                    N = q / np.maximum(v, 1.0) * length[None]
                okv = ve[:-1] & ve[1:] & ramp_ok[:-1]
                oke = e[:-1] & e[1:] & ramp_ok[:-1]
                a = acc.setdefault(reg, dict(days=0, n_tgt=0, n_tgt_blk=0, n_elig=0, n_vis_elig=0, ql_sum_vis=0.0,
                                             dN_sum_vis=0.0, n_tr_vis=0, n_tr_elig=0))
                a["days"] += 1
                a["n_tgt"] += int(tg.sum())
                a["n_tgt_blk"] += int((tg & b).sum())
                a["n_elig"] += int(e.sum())
                a["n_vis_elig"] += int(ve.sum())
                a["ql_sum_vis"] += float(np.abs(q / lanes[None])[ve].sum())
                a["dN_sum_vis"] += float(np.abs(N[1:] - N[:-1])[okv].sum())
                a["n_tr_vis"] += int(okv.sum())
                a["n_tr_elig"] += int(oke.sum())
                rws = set(np.flatnonzero(b.any(1)).tolist())
                for s0 in sorted(rws):
                    if s0 - 1 in rws:
                        continue
                    s = s0
                    while s in rws:                      # adjacent blackouts would form one block of 18k rows
                        T = s - 1
                        cond, wid = cond_of[(date, T)]
                        assert b[s:s + GAP].all()
                        wrows.append(dict(panel=p, split=sp, window_id=wid, date=date, day=i, regime=reg, T=T,
                                          cond=cond, n_tgt_blk=int((tg & b)[s:s + GAP].sum()),
                                          n_elig_blk=int((e & b)[s:s + GAP].sum())))
                        s += GAP
            for reg, a in acc.items():
                rows.append(dict(panel=p, split=sp, regime=reg, **a))
        print("frame", p, flush=True)
    F = pd.DataFrame(rows)
    F["fd_den"] = F.ql_sum_vis / F.n_vis_elig * F.n_elig
    F["lwr_den"] = F.dN_sum_vis / F.n_tr_vis * F.n_tr_elig
    W = pd.DataFrame(wrows)
    assert len(W) == 160 and W.groupby(["split", "cond"]).size().eq(40).all()
    assert int(W.n_tgt_blk.sum()) == int(F.n_tgt_blk.sum())
    F.to_csv(OUT / "frame_stats.csv", index=False)
    W.to_csv(OUT / "official_windows.csv", index=False)
    print(F.groupby("split")[["n_tgt", "n_tgt_blk"]].sum(), flush=True)


def load_frame():
    return pd.read_csv(OUT / "frame_stats.csv"), pd.read_csv(OUT / "official_windows.csv")


_OM = {}


def outside_mse(p, split):
    """Per-regime MSE (speed, flow per lane) of the production chain on the hide-20% cells of t1_v52_monthbias."""
    if (p, split) not in _OM:
        C = pd.read_parquet(M.HIDE / f"{p}_{split}_cells.parquet", columns=["regime", "pos", "ov", "oq", "pv", "pq"])
        lanes = links(p).lanes.to_numpy(float)[C.pos.to_numpy()]
        ev2 = (C.pv.to_numpy(float) - C.ov.to_numpy(float)) ** 2
        eq2 = ((C.pq.to_numpy(float) - C.oq.to_numpy(float)) / lanes) ** 2
        r = C.regime.to_numpy()
        _OM[p, split] = {g: (float(ev2[r == g].mean()), float(eq2[r == g].mean())) for g in range(3) if (r == g).any()}
    return _OM[p, split]


def window_sums(p: str, A: dict, F: pd.DataFrame, v: np.ndarray, q: np.ndarray) -> pd.DataFrame:
    """Per simulated window (index wi): cells n, SSE speed, SSE flow/lane, FD numerator, LWR terms A, B, Ce.
    Real-frame emulation: inside an official blackout every eligible cell is a T1 target with probability r (the
    day's mask rate), otherwise the evaluator holds its truth; the LWR proxy residual of a transition a->b is
    |I_b e_b - I_a e_a|, e = N_fill - N_obs, N = q / max(v, 1) * length, so E[res] = r(1-r)(|e_a| + |e_b|) +
    r^2 |e_b - e_a| inside the gap (A, B) and r |e| at the two edges (Ce, the other cell a visible observation);
    transitions need both cells eligible and ramp_ok on the earlier cell (as evaluate.t3_proxy_panel)."""
    fd = M.static_fd(p)
    W, _, n = A["ov"].shape
    pos = F.pos.to_numpy(np.int64)
    wi = F.wi.to_numpy(np.int64)
    ki = F.k.to_numpy(np.int64) - K0
    ov, oq = F.ov.to_numpy(float), F.oq.to_numpy(float)
    ev2 = (v - ov) ** 2
    eq2 = ((q - oq) / fd["lanes"][pos]) ** 2
    fn, _ = M.fd_terms(v, q, pos, fd)
    E = np.full((W, NK, n), np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        E[wi, ki, pos] = (q / np.maximum(v, 1.0) - oq / np.maximum(ov, 1.0)) * fd["length"][pos]
    gapc = np.isfinite(E)
    edge = A["vis"] & A["elig"] & ~A["hidden"]
    rok = A["ramp_ok"]
    a, b = E[:, :-1], E[:, 1:]
    inside = gapc[:, :-1] & gapc[:, 1:] & rok[:, :-1]
    ein = edge[:, :-1] & gapc[:, 1:] & rok[:, :-1]
    eout = gapc[:, :-1] & edge[:, 1:] & rok[:, :-1]
    Aw = np.where(inside, np.abs(a) + np.abs(b), 0).sum((1, 2))
    Bw = np.where(inside, np.abs(b - a), 0).sum((1, 2))
    Cw = np.where(ein, np.abs(b), 0).sum((1, 2)) + np.where(eout, np.abs(a), 0).sum((1, 2))
    out = pd.DataFrame({"wi": wi, "ev2": ev2, "eq2": eq2, "fn": fn}).groupby("wi").agg(
        n=("ev2", "size"), sv=("ev2", "sum"), sq=("eq2", "sum"), fn=("fn", "sum"))
    out["A"], out["B"], out["Ce"] = Aw[out.index], Bw[out.index], Cw[out.index]
    first = F.groupby("wi")[["condname", "day", "regime"]].first().loc[out.index]
    out["cond"], out["day"], out["regime"] = first.condname.to_numpy(), first.day.to_numpy(), first.regime.to_numpy()
    return out


def real_frame(p, split, ws, FR, OW, pick=None):
    """Per regime: S_state, FD numerator share and LWR proxy E of the real frame when the official blackout targets
    carry the per-cell error of the simulated windows of the same condition (pick: optional dict cond -> array of
    row labels of ws, e.g. a bootstrap draw)."""
    ow = OW[(OW.panel == p) & (OW.split == split)]
    fr = FR[(FR.panel == p) & (FR.split == split)].set_index("regime")
    om = outside_mse(p, split)
    pc = {}
    for c, g in ws.groupby("cond"):
        if pick is not None:
            g = g.loc[pick[c]]
        nn = g.n.sum()
        pc[c] = {k: g[k].sum() / nn for k in ("sv", "sq", "fn", "A", "B", "Ce")}
    out = {}
    for reg in fr.index:
        f = fr.loc[reg]
        o = ow[ow.regime == reg]
        r = RATE[reg]
        mv, mq = om.get(reg, om[min(om)])
        n_out = f.n_tgt - f.n_tgt_blk
        sv, sq, fn, lw = n_out * mv, n_out * mq, 0.0, 0.0
        for cond, ntb, ne in zip(o.cond, o.n_tgt_blk, o.n_elig_blk):
            c = pc[cond]
            sv += ntb * c["sv"]
            sq += ntb * c["sq"]
            fn += r * ne * c["fn"]
            lw += ne * (r * (1 - r) * c["A"] + r * r * c["B"] + r * c["Ce"])
        rv, rq = np.sqrt(sv / f.n_tgt), np.sqrt(sq / f.n_tgt)
        out[reg] = dict(S=0.54 * max(0.0, 1 - rv / 25) + 0.46 * max(0.0, 1 - rq / 600), rv=rv, rq=rq,
                        FDn=fn / f.fd_den, E=lw / f.lwr_den)
    return out


def deltas(r0, r1):
    per = {}
    for reg in r0:
        dFD = -(r1[reg]["FDn"] - r0[reg]["FDn"])
        dL = -(r1[reg]["E"] - r0[reg]["E"])            # S_LWR = 1 - (E + floor), far from its clip
        per[reg] = dict(dS=r1[reg]["S"] - r0[reg]["S"], dFD=dFD, dLWR=dL, dphys=(dFD + 2 * dL) / 3)
    tot = {k: float(np.mean([per[r][k] for r in per])) for k in ("dS", "dFD", "dLWR", "dphys")}
    tot["dTotal"] = W_STATE * tot["dS"] + W_PHYS * tot["dphys"]
    return tot, per


def family_mean(d: dict) -> float:
    """Leaderboard aggregation over all ten corridors; D12_I405_* have no blackouts (delta 0)."""
    fam = {}
    for p in PANELS:
        fam.setdefault(FAMILY[p], []).append(d.get(p, 0.0))
    return float(np.mean([np.mean(v) for v in fam.values()]))


# ---------------------------------------------------------------------------
# crossfit: pre-registered grid and selection rule
# ---------------------------------------------------------------------------
LAM_V = (0.5, 0.625, 0.75, 0.875, 1.0)
LAM_Q = (0.0, 0.5, 0.75, 1.0)
N_BOOT = 300


def load_sim_all():
    parts, TEN = [], {}
    for p in T2_PANELS:
        for sp in SPLITS:
            F = sim_features(p, sp)
            F["panel"], F["split"] = p, sp
            parts.append(F)
            TEN[p, sp] = sim_tensors(p, sp)[0]
    ALL = pd.concat(parts, ignore_index=True)
    ALL["par"] = (ALL.day % 2).astype(np.int8)
    ALL["half"] = (ALL.day >= 15).astype(np.int8)
    ALL["sid"] = (ALL.split == "private").astype(np.int8)
    return ALL, TEN


def gbm_fit(X, y):
    import lightgbm as lgb
    return lgb.train(GBM_PARAMS, lgb.Dataset(X, y, feature_name=FEATS), GBM_ROUNDS)


def cross_predict(ALL, split_col, fit_mask_fn=None):
    """Residual predictions for every cell from models fitted on the other value of split_col (both months and all
    corridors pooled unless fit_mask_fn restricts the fitting rows)."""
    X = ALL[FEATS].to_numpy(np.float32)
    lanes = ALL.lanes.to_numpy(float)
    yv = ALL.ov.to_numpy(float) - ALL.pv.to_numpy(float)
    yq = (ALL.oq.to_numpy(float) - ALL.pq.to_numpy(float)) / lanes
    g = ALL[split_col].to_numpy()
    pv_, pq_ = np.full(len(ALL), np.nan), np.full(len(ALL), np.nan)
    for a in (0, 1):
        fit = g == a
        if fit_mask_fn is not None:
            fit &= fit_mask_fn(ALL)
        app = g != a
        pv_[app] = gbm_fit(X[fit], yv[fit]).predict(X[app], num_threads=4)
        pq_[app] = gbm_fit(X[fit], yq[fit]).predict(X[app], num_threads=4)
    return pv_, pq_


def fill(ALL, pv_, pq_, lv, lq):
    v = np.clip(ALL.pv.to_numpy(float) + lv * pv_, SPEED_MIN, SPEED_MAX)
    q = np.maximum(ALL.pq.to_numpy(float) + lq * pq_ * ALL.lanes.to_numpy(float), 0.0)
    return v, q


def evaluate_fill(ALL, TEN, v, q, FR, OW, base_ws, boot=None):
    rows, ws_all = [], {}
    for (p, sp), A in TEN.items():
        m = ((ALL.panel == p) & (ALL.split == sp)).to_numpy()
        F = ALL[m]
        ws1 = window_sums(p, A, F, v[m], q[m])
        ws_all[p, sp] = ws1
        tot, per = deltas(real_frame(p, sp, base_ws[p, sp], FR, OW), real_frame(p, sp, ws1, FR, OW))
        rows.append({"panel": p, "split": sp, **tot})
    return pd.DataFrame(rows), ws_all


def summarise_cm(R: pd.DataFrame) -> dict:
    out = {}
    for sp in SPLITS:
        g = R[R.split == sp]
        for k in ("dS", "dFD", "dLWR", "dphys", "dTotal"):
            out[f"{k}_family_{sp}"] = family_mean(dict(zip(g.panel, g[k])))
        out[f"n_pos_total_{sp}"] = int((g.dTotal > 0).sum())
        out[f"n_pos_state_{sp}"] = int((g.dS > 0).sum())
        out[f"dTotal_min_{sp}"] = float(g.dTotal.min())
    out["dTotal_family_both"] = (out["dTotal_family_validation"] + out["dTotal_family_private"]) / 2
    out["dS_family_both"] = (out["dS_family_validation"] + out["dS_family_private"]) / 2
    return out


def bootstrap(ALL, TEN, ws0, ws1, FR, OW, n_boot=N_BOOT, seed=57):
    """Realisation spread of the gain on 5 onset + 5 ongoing windows per corridor-month: draw that many simulated
    windows (with replacement, same draw for before/after) and recompute the family-mean dTotal."""
    rng = np.random.default_rng(seed)
    out = {sp: [] for sp in SPLITS}
    keys = list(TEN)
    for b in range(n_boot):
        acc = {sp: {} for sp in SPLITS}
        for (p, sp) in keys:
            w0 = ws0[p, sp]
            ow = OW[(OW.panel == p) & (OW.split == sp)]
            pick = {c: rng.choice(w0.index[w0.cond == c].to_numpy(), size=int((ow.cond == c).sum()), replace=True)
                    for c in ("onset", "ongoing")}
            tot, _ = deltas(real_frame(p, sp, w0, FR, OW, pick), real_frame(p, sp, ws1[p, sp], FR, OW, pick))
            acc[sp][p] = tot["dTotal"]
        for sp in SPLITS:
            out[sp].append(family_mean(acc[sp]))
    return {sp: {"mean": float(np.mean(v)), "sd": float(np.std(v)), "p05": float(np.percentile(v, 5)),
                 "p50": float(np.percentile(v, 50)), "p95": float(np.percentile(v, 95)),
                 "share_negative": float(np.mean(np.array(v) < 0))} for sp, v in out.items()}


def cmd_crossfit(args) -> None:
    """Pre-registered (written before this final run; exploration in the scratch folder used the same features
    and model, see the README):
      model     LightGBM residual regressions (speed km/h, flow veh/h/lane), FEATS, GBM_PARAMS, GBM_ROUNDS
      fitting   day parity cross-fit: fit on parity a of BOTH months and all eight T2 corridors, predict parity b
      grid      lambda_v in LAM_V x lambda_q in LAM_Q (fill = production + lambda * predicted residual)
      score     real-frame emulation per corridor-month (official blackout target counts, condition mix and
                mask rates); 0.35 dS_state + 0.15 dS_phys(est); family mean (D12_I405_* = 0)
      eligible  family dTotal > 0 and family dS_state > 0 in both months, dTotal > 0 in >= 7/8 corridors per month
      choice    best = max mean-over-months family dTotal among eligible configs; then, because the simulated
                history rows T-60..T-5 keep their Task-1 holes while the official history is fully visible (the
                production fill should be slightly better there, the MSE-optimal shrinkage on the R1 days of the
                simulation is 0.7-0.95), take the eligible config with the smallest lambda_v, then the smallest
                lambda_q, whose dTotal_both >= 0.95 x best
    Reported only: block cross-fit (days < 15 vs >= 15), nested month (fit on one month only), per-regime and
    per-condition MSE reduction, bootstrap of the realisation on 5+5 windows per corridor-month."""
    OUT.mkdir(parents=True, exist_ok=True)
    for f in ("crossfit_grid.csv", "crossfit_summary.csv", "crossfit_choice.json", "crossfit_corridors.csv"):
        if (OUT / f).exists():
            raise FileExistsError(OUT / f)
    t0 = time.time()
    FR, OW = load_frame()
    ALL, TEN = load_sim_all()
    print("simulated gap cells", len(ALL), "windows", ALL.groupby(["panel", "split"]).w.nunique().sum(),
          f"{time.time() - t0:.0f}s", flush=True)
    base_ws = {}
    for (p, sp), A in TEN.items():
        m = ((ALL.panel == p) & (ALL.split == sp)).to_numpy()
        F = ALL[m]
        base_ws[p, sp] = window_sums(p, A, F, F.pv.to_numpy(float), F.pq.to_numpy(float))
    pv_, pq_ = cross_predict(ALL, "par")
    np.savez_compressed(OUT / "crossfit_pred_parity.npz", pv=pv_, pq=pq_)
    print("parity cross-fit", f"{time.time() - t0:.0f}s", flush=True)
    grid, cm_rows = [], []
    for lv in LAM_V:
        for lq in LAM_Q:
            v, q = fill(ALL, pv_, pq_, lv, lq)
            R, _ = evaluate_fill(ALL, TEN, v, q, FR, OW, base_ws)
            name = f"v{lv}_q{lq}"
            R["cfg"] = name
            cm_rows.append(R)
            grid.append({"cfg": name, "lam_v": lv, "lam_q": lq, **summarise_cm(R)})
    G_ = pd.DataFrame(grid).set_index("cfg")
    G_["eligible"] = [all(r[f"dTotal_family_{s}"] > 0 and r[f"dS_family_{s}"] > 0 and r[f"n_pos_total_{s}"] >= 7
                          for s in SPLITS) for _, r in G_.iterrows()]
    el = G_[G_.eligible]
    if el.empty:
        write_new(OUT / "crossfit_choice.json", json.dumps({"chosen": None, "reason": "no eligible config"}, indent=2))
        print(G_.round(6).to_string())
        raise SystemExit("no eligible config")
    best = el.dTotal_family_both.max()
    near = el[el.dTotal_family_both >= 0.95 * best].sort_values(["lam_v", "lam_q"])
    chosen = near.index[0]
    lv, lq = float(near.lam_v.iloc[0]), float(near.lam_q.iloc[0])
    print(G_.round(6).to_string(), flush=True)
    print("best", el.dTotal_family_both.idxmax(), best, "-> chosen", chosen, flush=True)
    CM = pd.concat(cm_rows, ignore_index=True)
    # robustness at the chosen lambdas
    v, q = fill(ALL, pv_, pq_, lv, lq)
    R_par, ws_par = evaluate_fill(ALL, TEN, v, q, FR, OW, base_ws)
    rob = {"parity": summarise_cm(R_par)}
    bv_, bq_ = cross_predict(ALL, "half")
    rob["block_halves"] = summarise_cm(evaluate_fill(ALL, TEN, *fill(ALL, bv_, bq_, lv, lq), FR, OW, base_ws)[0])
    print("block cross-fit", f"{time.time() - t0:.0f}s", flush=True)
    for m_fit, nm in ((0, "fit_march_only"), (1, "fit_april_only")):
        nv_, nq_ = cross_predict(ALL, "par", lambda D, m_fit=m_fit: (D.sid == m_fit).to_numpy())
        rob[nm] = summarise_cm(evaluate_fill(ALL, TEN, *fill(ALL, nv_, nq_, lv, lq), FR, OW, base_ws)[0])
        print(nm, f"{time.time() - t0:.0f}s", flush=True)
    boot = bootstrap(ALL, TEN, base_ws, ws_par, FR, OW)
    # MSE reduction by (split, regime of the simulated day) and by (split, condition)
    D = pd.DataFrame({"split": ALL.split, "regime": ALL.regime, "cond": ALL.condname,
                      "e0": (ALL.pv - ALL.ov) ** 2, "e1": (v - ALL.ov) ** 2,
                      "f0": ((ALL.pq - ALL.oq) / ALL.lanes) ** 2, "f1": ((q - ALL.oq) / ALL.lanes) ** 2})
    def red(g):
        return pd.Series({"cells": len(g), "rmse_v0": np.sqrt(g.e0.mean()), "rmse_v1": np.sqrt(g.e1.mean()),
                          "mse_v_reduction": 1 - g.e1.sum() / g.e0.sum(), "rmse_q0": np.sqrt(g.f0.mean()),
                          "rmse_q1": np.sqrt(g.f1.mean()), "mse_q_reduction": 1 - g.f1.sum() / g.f0.sum()})
    by_reg = D.groupby(["split", "regime"]).apply(red)
    by_cond = D.groupby(["split", "cond"]).apply(red)
    by_cm = D.assign(panel=ALL.panel).groupby(["panel", "split"]).apply(red)
    G_.to_csv(OUT / "crossfit_summary.csv")
    CM.to_csv(OUT / "crossfit_grid.csv", index=False)
    R_par.merge(by_cm.reset_index(), on=["panel", "split"]).to_csv(OUT / "crossfit_corridors.csv", index=False)
    choice = {"rule": cmd_crossfit.__doc__, "chosen_name": chosen, "chosen": {"lam_v": lv, "lam_q": lq},
              "best_name": el.dTotal_family_both.idxmax(), "best_dTotal_both": float(best),
              "dS_family": {s: float(G_.loc[chosen, f"dS_family_{s}"]) for s in SPLITS},
              "dphys_family_est": {s: float(G_.loc[chosen, f"dphys_family_{s}"]) for s in SPLITS},
              "dTotal_family_est": {s: float(G_.loc[chosen, f"dTotal_family_{s}"]) for s in SPLITS},
              "n_positive_corridors": {s: int(G_.loc[chosen, f"n_pos_total_{s}"]) for s in SPLITS},
              "robustness": rob, "bootstrap_5plus5_windows": boot,
              "mse_reduction_by_regime": by_reg.reset_index().to_dict("records"),
              "mse_reduction_by_condition": by_cond.reset_index().to_dict("records"),
              "features": FEATS, "gbm_params": GBM_PARAMS, "gbm_rounds": GBM_ROUNDS,
              "seconds": round(time.time() - t0)}
    write_new(OUT / "crossfit_choice.json", json.dumps(choice, indent=2, default=float))
    print(json.dumps({k: choice[k] for k in ("chosen_name", "best_name", "dS_family", "dphys_family_est",
                                             "dTotal_family_est", "n_positive_corridors", "robustness",
                                             "bootstrap_5plus5_windows")}, indent=2, default=float), flush=True)
    print(by_reg.round(4).to_string(), "\n", by_cond.round(4).to_string(), flush=True)


# ---------------------------------------------------------------------------
# final models, official windows, apply
# ---------------------------------------------------------------------------
def cmd_fit(args) -> None:
    paths = {c: MODELS / f"t1_blk_v57_{c}.txt" for c in "vq"}
    for pth in paths.values():
        if pth.exists():
            raise FileExistsError(pth)
    ALL, _ = load_sim_all()
    X = ALL[FEATS].to_numpy(np.float32)
    yv = ALL.ov.to_numpy(float) - ALL.pv.to_numpy(float)
    yq = (ALL.oq.to_numpy(float) - ALL.pq.to_numpy(float)) / ALL.lanes.to_numpy(float)
    for c, y in (("v", yv), ("q", yq)):
        gbm_fit(X, y).save_model(str(paths[c]))
        print("saved", paths[c], M.sha256(paths[c]), flush=True)


def cmd_official(args) -> None:
    """U-Net outputs (both tags, both folds, CPU fp32 as in the simulation) on every official blackout day without
    extra hiding, rows T-6..T+24 of every cell around each official window -> official/{panel}_{split}_win.parquet.
    'hidden' = Task-1 targets inside the official blackout (the cells to fill)."""
    import t1_unet as U
    import task1_gbm as G
    FR, OW = load_frame()
    ch = M.Chain(args.threads)
    OFFICIAL.mkdir(parents=True, exist_ok=True)
    cols = list(dict.fromkeys(ch.B.COLS + M.RAMP_COLS))
    for p in T2_PANELS:
        for sp in SPLITS:
            path = OFFICIAL / f"{p}_{sp}_win.parquet"
            if path.exists():
                print("exists", path.name, flush=True)
                continue
            t0 = time.time()
            ctx, st = G.Panel(p), U.static(p)
            src, prof = U.open_split(p, sp), np.load(U.DATA / f"{p}_profiles.npy")
            pf = pq.ParquetFile(CLEAN / p / f"{sp}.parquet")
            n = ctx.n
            ow = OW[(OW.panel == p) & (OW.split == sp)]
            recs = []
            for day, g in ow.groupby("day"):
                tb = pf.read_row_group(int(day), columns=cols, use_threads=False)
                arr = lambda c: tb.column(c).to_numpy().reshape(STEPS, n)
                ov, oq = arr("speed_obs").astype(float), arr("flow_obs").astype(float)
                vis = np.isfinite(ov) & np.isfinite(oq)
                elig, tgt, blk = arr("is_score_eligible"), arr("is_t1_target"), arr("is_t2_blackout")
                ramp_ok = (((arr("on_ramp_n") == 0) | (arr("on_ramp_n_valid") > 0)) &
                           ((arr("off_ramp_n") == 0) | (arr("off_ramp_n_valid") > 0)))
                ud = U.Day(p, sp, int(day), src, st, prof)
                grid = {}
                for tag in ch.B.TAGS:
                    ps = [ch.unet(ch.nets[tag, f], ud, None) for f in (0, 1)]
                    for j, c in enumerate("vq"):
                        grid[f"{c}{tag}"] = np.mean([x[j] for x in ps], axis=0)[:, ctx.inv]
                        for f in (0, 1):
                            grid[f"{c}{tag}f{f}"] = ps[f][j][:, ctx.inv]
                pvv = .5 * (grid["v_w48"] + grid["v_w48lwr"])
                pqq = .5 * (grid["q_w48"] + grid["q_w48lwr"])
                for _, w in g.iterrows():
                    T = int(w["T"])
                    ks = np.arange(K0, K1 + 1)
                    tt = T + ks
                    okr = (tt >= 0) & (tt < STEPS)
                    ks, tt = ks[okr], tt[okr]
                    ti, li = np.repeat(tt, n), np.tile(np.arange(n), len(tt))
                    kk = np.repeat(ks, n)
                    inwin = (kk >= 1) & (kk <= GAP)
                    hidden = tgt[ti, li] & blk[ti, li] & inwin
                    assert not (vis[ti, li] & blk[ti, li]).any()
                    assert blk[ti, li][inwin].all() and not hidden[~inwin].any()
                    rec = {"day": np.full(len(ti), day, np.int16), "T": np.full(len(ti), T, np.int16),
                           "k": kk.astype(np.int8), "t": ti.astype(np.int16), "pos": li.astype(np.int16),
                           "cond": np.full(len(ti), w.cond), "regime": np.full(len(ti), ud.regime, np.int8),
                           "window_id": np.full(len(ti), w.window_id), "elig": elig[ti, li], "vis": vis[ti, li],
                           "tgt": tgt[ti, li], "hidden": hidden, "ramp_ok": ramp_ok[ti, li],
                           "ov": ov[ti, li].astype(np.float32), "oq": oq[ti, li].astype(np.float32),
                           "pv": pvv[ti, li].astype(np.float32), "pq": pqq[ti, li].astype(np.float32)}
                    for nm, a in grid.items():
                        rec[nm] = a[ti, li].astype(np.float32)
                    recs.append(pd.DataFrame(rec))
            D = pd.concat(recs, ignore_index=True)
            assert int(D.hidden.sum()) == int(ow.n_tgt_blk.sum()), (p, sp)
            D.to_parquet(path, index=False)
            print(p, sp, "official windows", len(ow), "blackout targets", int(D.hidden.sum()),
                  f"{time.time() - t0:.0f}s", flush=True)


def official_corrections(lv: float, lq: float):
    """{(split, panel): DataFrame(key, dv, dq, ...)} for the official blackout targets."""
    import lightgbm as lgb
    mv = lgb.Booster(model_file=str(MODELS / "t1_blk_v57_v.txt"))
    mq = lgb.Booster(model_file=str(MODELS / "t1_blk_v57_q.txt"))
    out = {}
    for p in T2_PANELS:
        for sp in SPLITS:
            D = pd.read_parquet(OFFICIAL / f"{p}_{sp}_win.parquet")
            A, meta = win_tensors(p, D)
            F = gap_features(p, A, meta)
            X = F[FEATS].to_numpy(np.float32)
            F["rv"] = mv.predict(X, num_threads=4)
            F["rq"] = mq.predict(X, num_threads=4)
            pf = pq.ParquetFile(CLEAN / p / f"{sp}.parquet")
            slot0 = {}
            for d in F.day.unique():
                ts = pf.read_row_group(int(d), columns=["ts"]).column(0).to_numpy()
                slot0[d] = int(ts.astype("datetime64[m]").astype(np.int64)[0] // 5)   # as evaluate.t1_targets
            t_abs = F.day.map(slot0).to_numpy(np.int64) + F["T"].to_numpy(np.int64) + F.k.to_numpy(np.int64)
            F["key"] = t_abs * KEY_BASE + F.pos.to_numpy(np.int64)
            F["lam_v"], F["lam_q"] = lv, lq
            out[sp, p] = F
    return out


def cmd_apply(args) -> None:
    import t1_assemble as TA
    choice = json.loads((OUT / "crossfit_choice.json").read_text())
    lv, lq = choice["chosen"]["lam_v"], choice["chosen"]["lam_q"]
    for pth in CSV.values():
        if pth.exists():
            raise FileExistsError(pth)
    for f in ("apply_changes.csv", "apply_summary.json", "apply_cells.parquet"):
        if (OUT / f).exists():
            raise FileExistsError(OUT / f)
    t0 = time.time()
    corr = official_corrections(lv, lq)
    K = M.read_key_state()
    with zipfile.ZipFile(V56_ZIP) as z, z.open(z.namelist()[0]) as f:
        Z = pd.read_csv(f, float_precision="round_trip", usecols=["submission_id", "task", "speed_kmh", "flow_vph"],
                        dtype={"submission_id": np.int64, "task": "category", "speed_kmh": str, "flow_vph": str})
    Z = Z[Z.task == "state"]
    assert np.array_equal(Z.submission_id.to_numpy(), K.submission_id.to_numpy())
    # every V56 state value is already %.6g text, so writing the CSV with %.6g reproduces it exactly
    sv_txt, sq_txt = Z.speed_kmh.to_numpy(), Z.flow_vph.to_numpy()
    K["v56"] = pd.to_numeric(Z.speed_kmh, errors="raise").to_numpy(float)
    K["q56"] = pd.to_numeric(Z.flow_vph, errors="raise").to_numpy(float)
    n_not6g = int(sum(("%.6g" % a) != b for a, b in zip(K.v56.to_numpy(), sv_txt))) + \
        int(sum(("%.6g" % a) != b for a, b in zip(K.q56.to_numpy(), sq_txt)))
    del Z, sv_txt, sq_txt
    assert n_not6g == 0, f"{n_not6g} V56 state values are not %.6g text; the CSV would change them"
    print("key + V56 state rows", len(K), "non-%.6g text:", n_not6g, f"{time.time() - t0:.0f}s", flush=True)
    rows, cells = [], []
    for split in SPLITS:
        S_ = K[K.split == split].reset_index(drop=True)
        v_all, q_all = S_.v56.to_numpy().copy(), S_.q56.to_numpy().copy()
        for p, idx in S_.groupby("panel", observed=True).indices.items():
            tg = t1_targets(p, split)
            kk = S_.key.to_numpy()[idx]
            j = np.searchsorted(tg["key"], kk)
            assert len(idx) == len(tg["key"]) and np.array_equal(tg["key"][j], kk), p
            gap = TA.target_blackout(p, split)[j].astype(bool)
            if p not in T2_PANELS:
                assert not gap.any()
                continue
            F = corr[split, p]
            assert len(F) == int(gap.sum()), (split, p, len(F), int(gap.sum()))
            loc = pd.Series(np.arange(len(idx)), index=kk)
            r = loc.reindex(F.key.to_numpy()).to_numpy()
            assert np.isfinite(r).all() and gap[r.astype(np.int64)].all()
            r = r.astype(np.int64)
            ii = idx[r]
            lanes = links(p).lanes.to_numpy(float)[F.pos.to_numpy(np.int64)]
            v0, q0 = v_all[ii], q_all[ii]
            v1 = g6(np.clip(v0 + lv * F.rv.to_numpy(), SPEED_MIN, SPEED_MAX))
            q1 = g6(np.maximum(q0 + lq * F.rq.to_numpy() * lanes, 0.0))
            v_all[ii], q_all[ii] = v1, q1
            ch = (v1 != v0) | (q1 != q0)
            cells.append(pd.DataFrame({"split": split, "panel": p, "key": F.key, "window_id": F.window_id,
                                       "k": F.k, "pos": F.pos, "cond": F.condname, "v56": v0, "q56": q0, "v57": v1,
                                       "q57": q1, "pv_cpu": F.pv, "pq_cpu": F.pq}))
            rows.append({"split": split, "panel": p, "targets": len(idx), "blackout_targets": int(gap.sum()),
                         "changed": int(ch.sum()),
                         "v56_minus_cpu_unet_rms": float(np.sqrt(np.mean((v0 - F.pv.to_numpy()) ** 2))),
                         "q56_minus_cpu_unet_rms": float(np.sqrt(np.mean((q0 - F.pq.to_numpy()) ** 2))),
                         "mean_dv": float(np.mean(v1 - v0)), "mean_abs_dv": float(np.mean(np.abs(v1 - v0))),
                         "max_abs_dv": float(np.max(np.abs(v1 - v0))), "mean_dq": float(np.mean(q1 - q0)),
                         "mean_abs_dq": float(np.mean(np.abs(q1 - q0))), "max_abs_dq": float(np.max(np.abs(q1 - q0)))})
            print(split, p, rows[-1], f"{time.time() - t0:.0f}s", flush=True)
        out = S_[["panel", "timestamp", "station_id", "link_id", "mask_regime"]].copy()
        out["speed_kmh"], out["flow_vph"] = v_all, q_all
        out.to_csv(CSV[split], index=False, float_format="%.6g")
        print("wrote", CSV[split], len(out), f"{time.time() - t0:.0f}s", flush=True)
        del S_, out
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "apply_changes.csv", index=False)
    C = pd.concat(cells, ignore_index=True)
    C.to_parquet(OUT / "apply_cells.parquet", index=False)
    by = C.assign(dv=C.v57 - C.v56, dq=C.q57 - C.q56).groupby(["split", "cond", "k"])[["dv", "dq"]].mean()
    summ = {"lam_v": lv, "lam_q": lq, "changes": rows, "non_6g_text_in_sample": n_not6g,
            "mean_correction_by_split_cond_k": by.reset_index().to_dict("records"),
            "outputs_sha256": {s: M.sha256(CSV[s]) for s in SPLITS}, "v56_zip_sha256": M.sha256(V56_ZIP),
            "models_sha256": {c: M.sha256(MODELS / f"t1_blk_v57_{c}.txt") for c in "vq"},
            "seconds": round(time.time() - t0)}
    write_new(OUT / "apply_summary.json", json.dumps(summ, indent=2, default=float))
    pd.set_option("display.width", 300)
    print(R.round(4).to_string(), flush=True)


# ---------------------------------------------------------------------------
# package / audit (t1_v52_monthbias code, names patched) and the line-by-line comparison with V56
# ---------------------------------------------------------------------------
def _new_state_arrays():
    K = M.read_key_state()
    sp = np.full(int(K.submission_id.max()) + 1, np.nan)
    fl = sp.copy()
    for s in SPLITS:
        d = pd.read_csv(CSV[s], float_precision="round_trip",
                        dtype={"panel": "category", "timestamp": "category", "station_id": "category",
                               "link_id": "category", "mask_regime": "category"})
        ks = K[K.split == s]
        assert len(d) == len(ks)
        for c in ("panel", "timestamp", "station_id", "link_id", "mask_regime"):
            assert np.array_equal(d[c].astype(str).to_numpy(), ks[c].astype(str).to_numpy()), (s, c)
        ids = ks.submission_id.to_numpy()
        sp[ids], fl[ids] = d.speed_kmh.to_numpy(), d.flow_vph.to_numpy()
        del d
    return sp, fl, K


def _patch():
    M.NAME = NAME
    M.OUT = OUT
    M.BASE_ZIP = V51_ZIP
    M.new_state_arrays = _new_state_arrays


def cmd_package(args) -> None:
    _patch()
    M.cmd_package(args)
    sup = OUT / "v57blk_state_manifest_supplement.json"
    write_new(sup, json.dumps({
        "note": "the manifest written by the imported t1_v52_monthbias package code carries the V52 note and "
                "state_inputs; the actual state inputs of v57blk_state.zip are listed here",
        "state_inputs": [{"path": str(CSV[s]), "sha256": M.sha256(CSV[s])} for s in SPLITS],
        "content": "all state rows = V56 (v56_t1_on54.zip) except the Task-1 targets inside official Task-2 "
                   "blackouts (V57 gap residual GBM); queue/ODME rows = v51_april.zip (replaced by V54 queues "
                   "in build_queue_candidate)",
        "output_sha256": M.sha256(WORK / "submissions" / f"{NAME}.zip")}, indent=2))


def cmd_audit(args) -> None:
    _patch()
    M.cmd_audit(args)


def cmd_compare(args) -> None:
    """v57_blk.zip vs v56_t1_on54.zip, line by line: queue/ODME lines byte-identical, state lines differ only at
    Task-1 targets inside official Task-2 blackouts, every state value equals the CSV value, %.6g text."""
    import t1_assemble as TA
    new = WORK / "submissions" / f"{FULL_NAME}.zip"
    path = OUT / "compare_v56.json"
    if path.exists():
        raise FileExistsError(path)
    sp, fl, K = _new_state_arrays()
    blk_id = np.zeros(len(sp), bool)
    for (split, p), idx in K.groupby(["split", "panel"], observed=True).indices.items():
        tg = t1_targets(p, split)
        kk = K.key.to_numpy()[idx]
        j = np.searchsorted(tg["key"], kk)
        blk_id[K.submission_id.to_numpy()[idx]] = TA.target_blackout(p, split)[j].astype(bool)
    split_arr = np.empty(len(sp), object)
    panel_arr = np.empty(len(sp), object)
    split_arr[K.submission_id.to_numpy()] = K.split.astype(str).to_numpy()
    panel_arr[K.submission_id.to_numpy()] = K.panel.astype(str).to_numpy()
    del K
    st = {"rows": 0, "state_rows": 0, "state_changed": 0, "state_changed_outside_blackout": 0,
          "blackout_state_rows": int(blk_id.sum()), "state_changed_speed": 0, "state_changed_flow": 0,
          "queue_rows": 0, "queue_changed": 0, "odme_rows": 0, "odme_changed": 0,
          "state_value_mismatch_vs_csv": 0, "state_text_not_6g": 0, "changed_by_split_panel": {},
          "max_abs_speed_change": 0.0, "max_abs_flow_change": 0.0}
    with zipfile.ZipFile(V56_ZIP) as za, zipfile.ZipFile(new) as zb:
        assert za.testzip() is None and zb.testzip() is None
        with za.open(za.namelist()[0]) as fa, zb.open(zb.namelist()[0]) as fb:
            assert fa.readline() == fb.readline()
            for la, lb in zip(fa, fb):
                st["rows"] += 1
                pa_, pb = la.rstrip(b"\r\n").split(b","), lb.rstrip(b"\r\n").split(b",")
                assert pa_[0] == pb[0] and pa_[1] == pb[1], st["rows"]
                if pa_[1] == b"state":
                    sid = int(pa_[0])
                    st["state_rows"] += 1
                    nv, nq = float(pb[2]), float(pb[3])
                    if nv != sp[sid] or nq != fl[sid]:
                        st["state_value_mismatch_vs_csv"] += 1
                    if la != lb:
                        if (b"%.6g" % nv) != pb[2] or (b"%.6g" % nq) != pb[3] or pa_[4:] != pb[4:]:
                            st["state_text_not_6g"] += 1
                        st["state_changed"] += 1
                        st["state_changed_outside_blackout"] += int(not blk_id[sid])
                        st["state_changed_speed"] += int(pa_[2] != pb[2])
                        st["state_changed_flow"] += int(pa_[3] != pb[3])
                        k = f"{split_arr[sid]}|{panel_arr[sid]}"
                        st["changed_by_split_panel"][k] = st["changed_by_split_panel"].get(k, 0) + 1
                        st["max_abs_speed_change"] = max(st["max_abs_speed_change"], abs(nv - float(pa_[2])))
                        st["max_abs_flow_change"] = max(st["max_abs_flow_change"], abs(nq - float(pa_[3])))
                elif pa_[1] == b"queue":
                    st["queue_rows"] += 1
                    st["queue_changed"] += int(la != lb)
                else:
                    st["odme_rows"] += 1
                    st["odme_changed"] += int(la != lb)
            assert fa.read() == b"" and fb.read() == b""
    st["changed_by_split"] = {s: sum(v for k, v in st["changed_by_split_panel"].items() if k.startswith(s))
                              for s in SPLITS}
    st["valid"] = bool(st["rows"] == 6980503 and st["queue_changed"] == 0 and st["odme_changed"] == 0 and
                       st["state_value_mismatch_vs_csv"] == 0 and st["state_text_not_6g"] == 0 and
                       st["state_changed_outside_blackout"] == 0 and 0 < st["state_changed"] <= st["blackout_state_rows"])
    st["v56_sha256"], st["v57_blk_sha256"] = M.sha256(V56_ZIP), M.sha256(new)
    write_new(path, json.dumps(st, indent=2))
    print(json.dumps(st, indent=2), flush=True)
    assert st["valid"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("frame", "crossfit", "fit", "apply", "package", "audit", "compare"):
        sub.add_parser(name)
    s = sub.add_parser("official")
    s.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    {"frame": cmd_frame, "crossfit": cmd_crossfit, "fit": cmd_fit, "official": cmd_official, "apply": cmd_apply,
     "package": cmd_package, "audit": cmd_audit, "compare": cmd_compare}[args.cmd](args)


if __name__ == "__main__":
    main()
