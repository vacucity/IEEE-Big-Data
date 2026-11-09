"""V65 Task 1 (lane E): LWR-consistent refinement of the congested non-blackout target cells, on top of V64.

V64 (v64_pool.zip, public 0.90173) = V62 with the 269,474 congested non-blackout T1 rows (V51 production
pv <= 1.15 v_cut, outside the official Task 2 blackouts) recomputed by the V61 L1-density model fitted on BOTH
months' hide cells (t1_v64_pooled.py / t1_v64_apply.py, CANDS["pooled31_l09"]).  The V61 density fit is per cell
(sum |N_new - N_obs|), but the Task 3 LWR term is a sum over five-minute transitions |dN_pred - dN_true| along time
on each link: a transition between a corrected cell and a visible cell ('edge') costs |e'| of that cell, a transition
between two corrected cells ('internal') costs |e'_{t+1} - e'_t|.  Real targets are hash-random with rate 0.2/0.3/0.5
(R1/R2/R3), the hide protocol hides 20% of the visible cells, so internal transitions are much more frequent among
real targets (R3 ~1/3 of target transitions) than among hide cells (~11%).  This lane tries to make consecutive
corrected cells mutually LWR-consistent.  The official boundary flows are organizer data projected onto the released
observations (rhs = true dN), so the LWR term couples cells only along time on the same link.

============================== PRE-REGISTERED CRITERION (frozen before any candidate was evaluated) ==============
Reference = V64 on the hide-20% frame of t1_v52_monthbias (both months): free hidden cells at the cross-fitted
V59_loc values (cache of t1_v61_congest), congested hidden cells (production pv <= 1.15 v_cut) at the V64 correction
cross-fitted exactly as t1_v64_pooled.cross / t1_v64_apply CANDS["pooled31_l09"] (both months pooled with the month
flag, odd/even-day folds of each corridor, 31 leaves, 1200 density / 600 split rounds, density step lambda 0.9,
alpha 0.75, split lam 0.5).  A candidate changes only the congested hidden cells and is cross-fitted on the same
day-parity folds (every value at a cell comes from models that never saw that cell's day).
Per corridor-month, relative to the V64 reference:
  dS        S_state proxy change, measured on the hide cells (t1_v52_monthbias.state_score)
  dS_rw     S_state change with the congested hidden cells reweighted to the real congested non-blackout target
            strata (t1_v61_congest.target_weights)
  dphys_est V61 real-frame scaling of the hide-frame S_FD / LWR change (t1_v52_monthbias.t3_delta)
  dphys_tr  same S_FD part; LWR part re-weighted to the REAL transition mix: per regime, the hide-frame residual
            change per transition of each class (E0 = corrected cell - visible cell, the corrected cell having no
            corrected neighbour on its other side; E1 = same but the corrected cell is the end of a run of corrected
            cells; IC = both ends corrected; IO = other end a non-corrected hidden / target cell) times the number of
            REAL-frame transitions of that class touching congested non-blackout targets, divided by the real-frame
            LWR denominator estimate lwr_den * (n_trans + n_trans_target) / n_trans (as t3_scales)
  dTotal_est = 0.35 dS + 0.15 dphys_est;  dTotal_tr = 0.35 dS + 0.15 dphys_tr
Family means (evaluate.FAMILY) per month.  ELIGIBLE iff in EACH month: family-mean dTotal_est > 0 and dTotal_tr > 0,
>= 8/10 corridors with dTotal_tr > 0, family-mean dS_rw >= 0; and the two-month mean of dTotal_tr >= 1e-5
(materiality: smaller gains are invisible at the 5-decimal leaderboard).  CHOICE: the eligible candidate with the
largest two-month mean dTotal_tr; candidates within 3e-6 of it -> the first in the registered grid order.
GATE (apply): real-frame S_FD family mean must not fall by more than 1e-5 vs V64 in either month.
No eligible candidate or a failed gate -> not built.  The rule is not relaxed after seeing results.
Reported with it: expected gain with the LWR part realised at 100% (congested cells, V61_cg) and at 40% (pessimistic,
V64-increment / free-flow levers): 0.35 dS_rw + 0.15 (dFD_est - 2 rho dE_tr) / 3, rho = 1.0 / 0.4.
=================================================================================================================

Subcommands (project root, anaconda python, OMP_NUM_THREADS=4):
  register   write results/t1_v65_lwrcong/prereg_criterion.json (the block above); refuses to overwrite
  structure  real-frame transition classes touching congested non-blackout targets vs hide-frame classes touching
             congested hidden cells, per split x corridor x regime (counts only, no candidate)
  baseline   V64 cross-fit on the hide cells (cached) + residual decomposition of V60-like / V64 by class (no candidate)
  grid       write results/t1_v65_lwrcong/prereg_grid.json (candidate grid, before crossfit)
  crossfit   evaluate the registered grid against V64, apply the criterion, write crossfit_choice.json
  apply / package / audit / compare  NOT implemented: the 2026-10-24 crossfit chose no candidate (none eligible,
             best two-month dTotal_tr +1.6e-6 < 1e-5, dS_rw < 0 in both months for every candidate) -> not built
Nothing existing is overwritten (FileExistsError).
"""
from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "4")

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
import t1_v52_monthbias as M  # noqa: E402
import t1_v57_restack as V  # noqa: E402
import t1_v59_local as LOC  # noqa: E402
import t1_v61_congest as C  # noqa: E402
import t1_v64_pooled as PO  # noqa: E402
from evaluate import CLEAN, FAMILY, KEY_BASE, PANELS, REGIMES, STEPS  # noqa: E402

WORK = Path(__file__).resolve().parent
OUT = WORK / "results" / "t1_v65_lwrcong"
CACHE = OUT / "cache"
SPLITS = M.SPLITS
SUB = WORK / "submissions"
V51_ZIP = SUB / "v51_april.zip"
V64_ZIP = SUB / "v64_pool.zip"
NAME = "v65lwr_state"
FULL_NAME = "v65_lwr"
CSV = {s: WORK / "sub" / f"state_{s}_v65lwr.csv" for s in SPLITS}
THREADS = 4
W_STATE, W_PHYS = 0.35, 0.15
CFG = C.CFGS["F600a75s"]
PRM31 = dict(C.PRM, num_leaves=31)
PRM31_S = dict(C.PRM_S, num_leaves=31)
R_D, R_S, LAM64 = 1200, 600, 0.9                 # V64 = t1_v64_apply.CANDS["pooled31_l09"]
CLASSES = ("E0", "E1", "IC", "IO")


def write_new(path: Path, text: str) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.write_text(text, encoding="utf-8")


def criterion_text() -> str:
    d = __doc__
    a = d.index("PRE-REGISTERED CRITERION")
    b = d.index("=====", d.index("Reported with it"))
    return d[a:b].strip()


def cmd_register(args) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    write_new(OUT / "prereg_criterion.json", json.dumps({
        "written": time.strftime("%Y-%m-%d %H:%M:%S"), "script": str(Path(__file__).resolve()),
        "criterion": criterion_text(),
        "thresholds": {"dTotal_est_family_each_month": "> 0", "dTotal_tr_family_each_month": "> 0",
                       "corridors_dTotal_tr_pos_each_month": ">= 8", "dS_rw_family_each_month": ">= 0",
                       "two_month_mean_dTotal_tr": ">= 1e-5", "tie_window": 3e-6,
                       "gate_real_frame_S_FD_family_change_each_month": ">= -1e-5"},
        "reference": "V64 pooled31_l09 cross-fitted on the hide frame",
        "status": "registered before any candidate evaluation"}, indent=2))
    print((OUT / "prereg_criterion.json").read_text())


# ---------------------------------------------------------------------------
# transition structure
# ---------------------------------------------------------------------------
def hide_classes(d, corr):
    """Class of every hide-frame transition (index into d['T']) w.r.t. the corrected hidden cells `corr` (bool over
    the hidden cells): -1 = untouched, 0 E0, 1 E1, 2 IC, 3 IO.  Neighbours: the hidden cells at t-1 / t+1 on the
    same link-day (in the hide frame only hidden cells can be changed)."""
    T = d["T"]
    h0, h1 = T.h0.to_numpy(), T.h1.to_numpy()
    c0 = (h0 >= 0) & corr[np.maximum(h0, 0)]
    c1 = (h1 >= 0) & corr[np.maximum(h1, 0)]
    nbp, nbn = corrected_neighbours_hide(d, corr)
    cls = np.full(len(T), -1, np.int8)
    ic = c0 & c1
    io = (c0 & (h1 >= 0) & ~c1) | (c1 & (h0 >= 0) & ~c0)
    e_from0 = c0 & (h1 < 0)            # corrected cell at t, visible at t+1: its other side is t-1
    e_from1 = c1 & (h0 < 0)            # corrected cell at t+1, visible at t: its other side is t+2
    hh0, hh1 = np.maximum(h0, 0), np.maximum(h1, 0)
    run0 = e_from0 & nbp[hh0]
    run1 = e_from1 & nbn[hh1]
    cls[e_from0 | e_from1] = 0
    cls[run0 | run1] = 1
    cls[io] = 3
    cls[ic] = 2
    return cls


def corrected_neighbours_hide(d, corr):
    """(has corrected hidden neighbour at t-1, at t+1) for every hidden cell."""
    Cc = d["C"]
    key = (Cc.day.to_numpy().astype(np.int64) * STEPS + Cc.t.to_numpy()) * KEY_BASE + Cc.pos.to_numpy()
    o = np.argsort(key)
    ks = key[o]
    cs = corr[o]

    def has(shift):
        kk = key + shift * KEY_BASE
        tt = Cc.t.to_numpy() + shift
        j = np.clip(np.searchsorted(ks, kk), 0, len(ks) - 1)
        return (ks[j] == kk) & cs[j] & (tt >= 0) & (tt < STEPS)
    return has(-1), has(1)


def real_counts(p, split):
    """Real-frame transitions touching congested non-blackout targets (CT) per regime and class, plus the totals
    used by t3_scales (n_trans visible-visible, n_trans_target), recomputed from the clean parquet."""
    fd = M.static_fd(p)
    n = fd["n"]
    LT = LOC.load_local(p, split, "T")
    pos = (LT["key"] % KEY_BASE).astype(np.int64)
    tt = ((LT["key"] // KEY_BASE) % STEPS).astype(np.int64)
    ct = ~(LT["v51"] > M.FREE_RATIO * fd["vc"][pos]) & ~LT["gap"]
    day = LT["day"].astype(np.int64)
    starts = np.searchsorted(day, np.arange(day.max() + 2))
    pf = pq.ParquetFile(CLEAN / p / f"{split}.parquet")
    acc = {}
    cols = ["mask_regime", "is_score_eligible", "is_t1_target", "speed_obs", "flow_obs"] + M.RAMP_COLS
    for dd in range(pf.num_row_groups):
        tb = pf.read_row_group(dd, columns=cols)
        r = REGIMES.index(tb.column("mask_regime")[0].as_py())
        g = lambda c: tb.column(c).to_numpy().reshape(STEPS, n)  # noqa: E731
        el, tgt = g("is_score_eligible"), g("is_t1_target")
        vis = np.isfinite(g("speed_obs").astype(float)) & np.isfinite(g("flow_obs").astype(float))
        ramp_ok = (((g("on_ramp_n") == 0) | (g("on_ramp_n_valid") > 0)) &
                   ((g("off_ramp_n") == 0) | (g("off_ramp_n_valid") > 0)))
        CT = np.zeros((STEPS, n), bool)
        a, b = starts[dd], starts[dd + 1]
        assert tgt.sum() == b - a, (p, split, dd)
        CT[tt[a:b][ct[a:b]], pos[a:b][ct[a:b]]] = True
        use = el & (vis | tgt)
        ok = use[:-1] & use[1:] & ramp_ok[:-1]
        ve = el & vis
        okv = ve[:-1] & ve[1:] & ramp_ok[:-1]
        c0, c1 = CT[:-1], CT[1:]
        t0, t1 = tgt[:-1], tgt[1:]
        nbp = np.zeros((STEPS, n), bool)
        nbp[1:] = CT[:-1]
        nbn = np.zeros((STEPS, n), bool)
        nbn[:-1] = CT[1:]
        e0 = ok & c0 & ~t1                       # corrected at t, visible at t+1; other side t-1
        e1 = ok & c1 & ~t0                       # corrected at t+1, visible at t; other side t+2
        run = (e0 & nbp[:-1]) | (e1 & nbn[1:])
        rec = acc.setdefault(r, {k: 0 for k in CLASSES + ("n_trans", "n_trans_target", "n_ct")})
        rec["IC"] += int((ok & c0 & c1).sum())
        rec["IO"] += int((ok & ((c0 & t1 & ~c1) | (c1 & t0 & ~c0))).sum())
        rec["E1"] += int(run.sum())
        rec["E0"] += int(((e0 | e1) & ~run).sum())
        rec["n_trans"] += int(okv.sum())
        rec["n_trans_target"] += int((ok & (t0 | t1)).sum())
        rec["n_ct"] += int(CT.sum())
    return acc


def cmd_structure(args) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "structure_real.csv"
    if path.exists():
        raise FileExistsError(path)
    rows = []
    t0 = time.time()
    for s in SPLITS:
        for p in PANELS:
            acc = real_counts(p, s)
            _, _, Dd = M.load_hide(p, s)
            for r, rec in acc.items():
                dm = Dd.regime == r
                rows.append({"split": s, "panel": p, "regime": r, **rec,
                             "hide_n_trans": int(Dd.n_trans[dm].sum()),
                             "hide_n_trans_target": int(Dd.n_trans_target[dm].sum())})
            print("structure", s, p, f"{time.time() - t0:.0f}s", flush=True)
    R = pd.DataFrame(rows)
    assert (R.n_trans == R.hide_n_trans).all() and (R.n_trans_target == R.hide_n_trans_target).all(), "frame counts"
    R.to_csv(path, index=False)
    print(R.groupby(["split", "regime"])[list(CLASSES)].sum().to_string())


def load_structure():
    R = pd.read_csv(OUT / "structure_real.csv")
    return {(r.split, r.panel, int(r.regime)): r for r in R.itertuples(index=False)}


# ---------------------------------------------------------------------------
# hide data + the V64 reference
# ---------------------------------------------------------------------------
def load_all():
    data = {s: C.load_month(s) for s in SPLITS}
    P = {s: C.pooled(data[s], CFG["feats"]) for s in SPLITS}
    Xs = [PO.with_month(P[s], i) for i, s in enumerate(SPLITS)]
    return data, P, Xs


def v64_crossfit(P, Xs):
    """Cross-fitted V64 density r_hat and split s_hat on the congested hide cells of both months (cached)."""
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / "v64cf.npz"
    if f.exists():
        z = np.load(f)
        return [z["rh0"], z["rh1"]], [z["sh0"], z["sh1"]]
    pars = [P[s]["par"] for s in SPLITS]
    rh = PO.cross(Xs, [P[s]["r"] for s in SPLITS], [P[s]["wd"] for s in SPLITS], pars, PRM31, R_D)
    sh = PO.cross(Xs, [P[s]["sstar"] for s in SPLITS], [P[s]["ws"] for s in SPLITS], pars, PRM31_S, R_S)
    np.savez(f, rh0=rh[0], rh1=rh[1], sh0=sh[0], sh1=sh[1])
    return rh, sh


def split_owner(P, arr, i):
    return arr[P["own"] == i]


def reference(data, P, rh, sh):
    """Attach the V64 hide-frame reference values / scores to every corridor-month dict."""
    for mi, s in enumerate(SPLITS):
        for i, p in enumerate(PANELS):
            d = data[s][p]
            r_, s_ = split_owner(P[s], rh[mi], i), split_owner(P[s], sh[mi], i)
            d["rh64"], d["sh64"] = r_, s_
            v, q = C.frame_values(d, r_, s_, CFG, LAM64)
            d["v64"], d["q64"] = v, q
            d["s64"] = M.state_score(d["C"], v, q, d["fd"])
            d["a64"] = M.t3_score(d["C"], d["T"], d["Dd"], v, q, d["fd"])
            d["cls"] = hide_classes(d, d["cong"])
            d["treg"] = d["T"].day.map(dict(zip(d["Dd"].day, d["Dd"].regime))).to_numpy()


def trans_res(d, v, q):
    """|dN_frame - dN_obs| per hide-frame transition (as t3_score)."""
    T = d["T"]
    Nc = C.dens(v, q, d["fd"], d["pos"])
    h0, h1 = T.h0.to_numpy(), T.h1.to_numpy()
    N0 = np.where(h0 >= 0, Nc[np.maximum(h0, 0)], T.N0.to_numpy())
    N1 = np.where(h1 >= 0, Nc[np.maximum(h1, 0)], T.N1.to_numpy())
    return np.abs((N1 - N0) - (T.N1.to_numpy() - T.N0.to_numpy()))


def dE_tr(d, res0, res1, S, s, p):
    """Transition-class reweighted real-frame LWR change (mean over the regimes present) + per-regime parts."""
    out, parts = [], {}
    Dd = d["Dd"]
    for r in sorted(set(Dd.regime)):
        dm = Dd.regime == r
        st = S[(s, p, int(r))]
        den = Dd.lwr_den[dm].sum() * (st.n_trans + st.n_trans_target) / max(st.n_trans, 1)
        m = d["treg"] == r
        dres = res1 - res0
        num, tot_n = 0.0, 0
        g_all = dres[m & (d["cls"] >= 0)].sum() / max(int((m & (d["cls"] >= 0)).sum()), 1)
        for k, c in enumerate(CLASSES):
            mc = m & (d["cls"] == k)
            nh = int(mc.sum())
            g = dres[mc].sum() / nh if nh > 0 else g_all
            nr = getattr(st, c)
            num += nr * g
            tot_n += nr
            parts[f"R{r + 1}_{c}_g"] = g
            parts[f"R{r + 1}_{c}_nh"] = nh
            parts[f"R{r + 1}_{c}_nr"] = nr
        e = num / max(den, 1e-9)
        parts[f"R{r + 1}_dE_tr"] = e
        out.append(e)
    return float(np.mean(out)), parts


def score_vs64(d, v, q, S, s, p, wrw):
    s1 = M.state_score(d["C"], v, q, d["fd"])
    a1 = M.t3_score(d["C"], d["T"], d["Dd"], v, q, d["fd"])
    t3 = M.t3_delta(d["a64"], a1, d["sc"])
    res0, res1 = trans_res(d, d["v64"], d["q64"]), trans_res(d, v, q)
    etr, parts = dE_tr(d, res0, res1, S, s, p)
    # per regime S_FD real-frame estimate (mean over regimes), as t3_delta
    dfd = t3["dFD_est"]
    dS = s1["S"] - d["s64"]["S"]
    sargs = (d["ov"], d["oq"], d["lanes"], d["reg"])
    S_rw = LOC.s_state_w(v, q, *sargs, wrw) - LOC.s_state_w(d["v64"], d["q64"], *sargs, wrw)
    dphys_tr = (dfd - 2 * etr) / 3
    out = {"dS": dS, "dS_rw": S_rw, "dFD_est": dfd, "dE_est": t3["dE_est"], "dE_tr": etr,
           "dphys_est": t3["dphys_est"], "dphys_tr": dphys_tr,
           "dTotal_est": W_STATE * dS + W_PHYS * t3["dphys_est"], "dTotal_tr": W_STATE * dS + W_PHYS * dphys_tr,
           "exp100": W_STATE * S_rw + W_PHYS * (dfd - 2 * 1.0 * etr) / 3,
           "exp40": W_STATE * S_rw + W_PHYS * (dfd - 2 * 0.4 * etr) / 3,
           "lwr_part_tr": W_PHYS * (-2 * etr) / 3, "fd_part": W_PHYS * dfd / 3}
    for k, c in enumerate(CLASSES):
        out[f"dres_{c}"] = float((res1 - res0)[d["cls"] == k].sum())
        out[f"res64_{c}"] = float(res0[d["cls"] == k].sum())
    out.update(parts)
    return out


def cmd_baseline(args) -> None:
    """V64 vs the V60-like hide reference (production congested values) per transition class: how much of V64's
    LWR gain sits in internal transitions, hide vs real mix.  No candidate is evaluated here."""
    path = OUT / "baseline_classes.csv"
    if path.exists():
        raise FileExistsError(path)
    t0 = time.time()
    S = load_structure()
    data, P, Xs = load_all()
    rh, sh = v64_crossfit(P, Xs)
    print("v64 crossfit", f"{time.time() - t0:.0f}s", flush=True)
    reference(data, P, rh, sh)
    rows = []
    for s in SPLITS:
        for p in PANELS:
            d = data[s][p]
            wrw, _ = C.target_weights(d, LOC.load_local(p, s, "T"), d["fd"])
            d["wrw"] = wrw
            # swap roles: score V60-like (bv, bq) against V64 -> sign-flip to get V64 vs V60
            r = score_vs64(d, d["bv"], d["bq"], S, s, p, wrw)
            rec = {"split": s, "panel": p, **{k: -v if isinstance(v, float) and (k.startswith("d") or
                                                                                  k.startswith("exp") or
                                                                                  k.endswith("_part_tr") or
                                                                                  k == "fd_part") else v
                                               for k, v in r.items()}}
            res64 = trans_res(d, d["v64"], d["q64"])
            res60 = trans_res(d, d["bv"], d["bq"])
            for k, c in enumerate(CLASSES):
                m = d["cls"] == k
                rec[f"n_{c}"] = int(m.sum())
                rec[f"res60_{c}"] = float(res60[m].sum())
                rec[f"res64_{c}"] = float(res64[m].sum())
            rows.append(rec)
    B = pd.DataFrame(rows)
    B.to_csv(path, index=False)
    pd.set_option("display.width", 250)
    for s in SPLITS:
        g = B[B.split == s].set_index("panel")
        print(s, {k: f"{M.family_mean(g[k].to_dict()) * 1e4:+.2f}e-4" for k in
                  ("dS", "dS_rw", "dE_est", "dE_tr", "dTotal_est", "dTotal_tr", "exp100", "exp40")}, flush=True)
        print(g[[f"n_{c}" for c in CLASSES] + [f"res60_{c}" for c in CLASSES] + [f"res64_{c}" for c in CLASSES]]
              .round(1).to_string(), flush=True)
    print(f"{time.time() - t0:.0f}s")


# ---------------------------------------------------------------------------
# candidates (identical operators at hide cells and real targets)
# ---------------------------------------------------------------------------
ALPHA = CFG["alpha"]
TA_PRM = dict(learning_rate=0.1, num_leaves=15, min_data_in_leaf=200, lambda_l2=10.0, num_threads=THREADS,
              verbose=-1, seed=65, deterministic=True, force_col_wise=True)
TA_EPS = 0.5           # IRLS floor of |residual| (vehicles)
U_CLIP = 1.0           # |log-density change| cap of every candidate (safety; V64 g is floored at 0.2)
GRID = {   # registered order = simplest first
    "L05": {"kind": "lap", "k": 0.05, "kv": 0.0},
    "L10": {"kind": "lap", "k": 0.10, "kv": 0.0},
    "L15": {"kind": "lap", "k": 0.15, "kv": 0.0},
    "LA10": {"kind": "lap", "k": 0.10, "kv": 0.05},
    "TA60": {"kind": "ta", "rounds": 60},
    "TA150": {"kind": "ta", "rounds": 150},
    "TA400": {"kind": "ta", "rounds": 400},
}
GRID_DOC = """Candidate grid (registered before the first evaluation; order = simplest first):
  L05 / L10 / L15  training-free Laplacian of the V64 density among CORRECTED temporal neighbours (t-1 / t+1 on the
                   same link-day that are corrected cells too: congested hidden cells in the hide frame, congested
                   non-blackout targets in the real frame): N' = N + k * sum_nb (N_nb - N) with the V64 values on the
                   right (one Jacobi step; a pair's difference is scaled by 1 - 2k, its mean is kept; isolated cells
                   unchanged); kappa k = 0.05 / 0.10 / 0.15
  LA10             L10 + run ends anchored to the visible neighbours: + kv * sum_vis (N_obs(t+-1) - N) over the
                   immediately adjacent visible cells (db == 1 / da == 1 of t1_v59_local), kv = 0.05
  TA60/150/400     transition-aware residual model on top of V64: LightGBM (15 leaves, min_data 200, lambda_l2 10,
                   lr 0.1, 60 / 150 / 400 rounds) of the log-density change u (v = v64 e^(-0.75 u), q = q64 e^(0.25 u))
                   with a custom objective = the linearised corridor-month dTotal_tr without S_FD: per cell the V61
                   marginal S_state weights a (v' - ov)^2 + b ((q' - oq)/lanes)^2, per hide transition touching a
                   corrected cell w |dN' - dN_obs| (IRLS-L1, floor 0.5 veh), w = 0.15 * 2/3 / R * (n_real_class /
                   n_hide_class) / real-frame LWR denominator of the corridor-month-regime.  Features = the V64
                   features (+ month flag, corridor id) + V64's own log-density correction + corrected-neighbour flags
                   at t-1 / t+1 and the log V64-density ratios to them.  The V64 values the TA model is trained on are
                   nested out-of-fold (inside each training parity, V64 refit on the (day // 2) halves); test cells use
                   the ordinary V64 cross-fit, real targets the full V64 model (same pattern as apply).
  All candidates: |u| capped at 1.0, v clipped to [5, 120.75] as V61; only the congested cells change."""


def nb_index(day, t, pos):
    """Index (within the given cell set) of the cell at t-1 / t+1 on the same link-day, -1 if not in the set."""
    key = (day.astype(np.int64) * STEPS + t.astype(np.int64)) * KEY_BASE + pos.astype(np.int64)
    o = np.argsort(key, kind="stable")
    ks = key[o]

    def find(shift):
        kk = key + shift * KEY_BASE
        j = np.clip(np.searchsorted(ks, kk), 0, len(ks) - 1)
        ok = (ks[j] == kk) & (t + shift >= 0) & (t + shift < STEPS)
        return np.where(ok, o[j], -1)
    return find(-1), find(1)


def vis_nb_density(L, lanes, length):
    """Observed density of the immediately adjacent visible cell before / after (nan if not adjacent)."""
    nb = np.where(L["db"] == 1, L["qb"] * lanes / np.maximum(L["vb"], 1.0) * length, np.nan)
    na = np.where(L["da"] == 1, L["qa"] * lanes / np.maximum(L["va"], 1.0) * length, np.nan)
    return nb, na


def lap_u(N, ip, in_, nvb, nva, k, kv):
    """Log-density change of the Laplacian candidates (V64 values N on the right)."""
    acc = np.zeros(len(N))
    for ix in (ip, in_):
        m = ix >= 0
        acc[m] += k * (N[ix[m]] - N[m])
    if kv > 0:
        for nv in (nvb, nva):
            m = np.isfinite(nv)
            acc[m] += kv * (nv[m] - N[m])
    Nn = np.maximum(N + acc, 0.2 * N)
    return np.log(Nn / N)


def apply_u(v64, q64, u):
    u = np.clip(u, -U_CLIP, U_CLIP)
    v = np.clip(v64 * np.exp(-ALPHA * u), M.SPEED_MIN, M.SPEED_MAX)
    q = np.maximum(q64 * np.exp((1 - ALPHA) * u), 0.0)
    return v, q


def ta_extra(N64, u0, ip, in_):
    """Extra TA features: V64's own log-density correction, corrected-neighbour flags and log density ratios."""
    f = [u0, (ip >= 0).astype(float), (in_ >= 0).astype(float)]
    for ix in (ip, in_):
        r = np.full(len(N64), np.nan)
        m = ix >= 0
        r[m] = np.log(np.maximum(N64[ix[m]], 1e-3) / np.maximum(N64[m], 1e-3))
        f.append(r)
    return np.column_stack(f)


def with_extra(X, E_):
    """Insert extra columns before the corridor id (last column, categorical)."""
    return np.column_stack([X[:, :-1], E_, X[:, -1:]])


class TAObjective:
    """Linearised dTotal_tr (without S_FD) of the log-density change u at the corrected training cells."""

    def __init__(self, cell, tr):
        self.c = cell          # dict of arrays over training cells: v64, q64, N64, ov, oq, lanes, a, b
        self.t = tr            # dict of arrays over training transitions: i0, i1 (cell index or -1), base, w
        g, h = self.raw(np.zeros(len(cell["v64"])))
        self.scale = 1.0 / max(float(np.mean(h)), 1e-30)

    def raw(self, u):
        c, t = self.c, self.t
        v = c["v64"] * np.exp(-ALPHA * u)
        ql = c["q64"] * np.exp((1 - ALPHA) * u) / c["lanes"]
        oql = c["oq"] / c["lanes"]
        g = 2 * c["a"] * (v - c["ov"]) * (-ALPHA * v) + 2 * c["b"] * (ql - oql) * ((1 - ALPHA) * ql)
        h = 2 * c["a"] * (ALPHA * v) ** 2 + 2 * c["b"] * ((1 - ALPHA) * ql) ** 2
        N = c["N64"] * np.exp(u)
        i0, i1 = t["i0"], t["i1"]
        m0, m1 = i0 >= 0, i1 >= 0
        res = t["base"].copy()
        res[m1] += N[i1[m1]]
        res[m0] -= N[i0[m0]]
        sg = np.sign(res) * t["w"]
        ir = t["w"] / np.maximum(np.abs(res), TA_EPS)
        n = len(u)
        g = g + np.bincount(i1[m1], weights=sg[m1] * N[i1[m1]], minlength=n) \
            - np.bincount(i0[m0], weights=sg[m0] * N[i0[m0]], minlength=n)
        h = h + np.bincount(i1[m1], weights=ir[m1] * N[i1[m1]] ** 2, minlength=n) \
            + np.bincount(i0[m0], weights=ir[m0] * N[i0[m0]] ** 2, minlength=n)
        return g, h

    def __call__(self, preds, ds):
        g, h = self.raw(preds)
        return g * self.scale, h * self.scale


def ta_train(X, cell, tr, rounds):
    import lightgbm as lgb
    obj = TAObjective(cell, tr)
    ds = lgb.Dataset(X, np.zeros(len(X)), categorical_feature=[X.shape[1] - 1], free_raw_data=True)
    return lgb.train(dict(TA_PRM, objective=obj), ds, num_boost_round=rounds)


# ---------------------------------------------------------------------------
# hide-frame bookkeeping for the candidates
# ---------------------------------------------------------------------------
def hide_cm(d):
    """Per corridor-month: corrected-cell neighbour indices and visible-neighbour densities (hide frame)."""
    c = d["cong"]
    Cc = d["C"]
    ci = np.flatnonzero(c)
    ip, in_ = nb_index(Cc.day.to_numpy()[ci], Cc.t.to_numpy()[ci], Cc.pos.to_numpy()[ci])
    Lc = {k: d["L"][k][ci] for k in ("db", "da", "vb", "qb", "va", "qa")}
    nvb, nva = vis_nb_density(Lc, d["lanes"][ci], d["fd"]["length"][d["pos"][ci]])
    return {"ci": ci, "ip": ip, "in": in_, "nvb": nvb, "nva": nva}


def class_factor(d, S, s, p):
    """Per hide transition: w = 0.15 * 2/3 / R * (n_real_class / n_hide_class) / real-frame LWR denominator."""
    Dd = d["Dd"]
    regs = sorted(set(Dd.regime))
    w = np.zeros(len(d["T"]))
    for r in regs:
        dm = Dd.regime == r
        st = S[(s, p, int(r))]
        den = Dd.lwr_den[dm].sum() * (st.n_trans + st.n_trans_target) / max(st.n_trans, 1)
        m = d["treg"] == r
        for k, cn in enumerate(CLASSES):
            mc = m & (d["cls"] == k)
            nh = int(mc.sum())
            if nh:
                w[mc] = W_PHYS * (2 / 3) / len(regs) * getattr(st, cn) / nh / den
    return w


def ta_data(data, P, sel, vals, S, wfac):
    """Training-cell and transition arrays (pooled over corridor-months) for the TA objective.
    sel[s]: bool over the pooled congested cells of month s (training cells); vals[s]: (v64, q64, N64) over the pooled
    congested cells of month s (the V64 values the model starts from)."""
    cell = {k: [] for k in ("v64", "q64", "N64", "ov", "oq", "lanes", "a", "b")}
    tr = {k: [] for k in ("i0", "i1", "base", "w")}
    off = 0
    for mi, s in enumerate(SPLITS):
        own = P[s]["own"]
        for i, p in enumerate(PANELS):
            d = data[s][p]
            m_own = own == i
            sl = sel[s][m_own]
            ci = d["cm"]["ci"]
            v64c, q64c, N64c = (x[m_own] for x in vals[s])
            # global index of every hidden cell that is a training corrected cell
            gi = np.full(len(d["pv"]), -1, np.int64)
            gi[ci[sl]] = off + np.arange(int(sl.sum()))
            off += int(sl.sum())
            for k, x in (("v64", v64c), ("q64", q64c), ("N64", N64c), ("ov", d["ov"][ci]), ("oq", d["oq"][ci]),
                         ("lanes", d["lanes"][ci]), ("a", d["wa"][ci]), ("b", d["wb"][ci])):
                cell[k].append(x[sl])
            # transitions touching a training corrected cell
            T = d["T"]
            h0, h1 = T.h0.to_numpy(), T.h1.to_numpy()
            g0 = np.where(h0 >= 0, gi[np.maximum(h0, 0)], -1)
            g1 = np.where(h1 >= 0, gi[np.maximum(h1, 0)], -1)
            use = (g0 >= 0) | (g1 >= 0)
            # fixed ends: visible obs or the reference value of a non-corrected hidden cell
            Nref = C.dens(d["v64"], d["q64"], d["fd"], d["pos"])
            N0f = np.where(h0 >= 0, Nref[np.maximum(h0, 0)], T.N0.to_numpy())
            N1f = np.where(h1 >= 0, Nref[np.maximum(h1, 0)], T.N1.to_numpy())
            N0f = np.where(g0 >= 0, 0.0, N0f)
            N1f = np.where(g1 >= 0, 0.0, N1f)
            base = (N1f - N0f) - (T.N1.to_numpy() - T.N0.to_numpy())
            # IC transitions with only one end in the training set cannot occur (same day = same fold)
            assert not (use & (d["cls"] == 2) & ((g0 < 0) | (g1 < 0))).any()
            tr["i0"].append(g0[use])
            tr["i1"].append(g1[use])
            tr["base"].append(base[use])
            tr["w"].append(wfac[s][p][use])
    cell = {k: np.concatenate(v) for k, v in cell.items()}
    tr = {k: np.concatenate(v) for k, v in tr.items()}
    return cell, tr


def v64_values(data, P, s, rh, sh):
    """(v64, q64, N64) over the pooled congested cells of month s from density / split predictions."""
    v, q, N = (np.zeros(len(rh)) for _ in range(3))
    own = P[s]["own"]
    for i, p in enumerate(PANELS):
        d = data[s][p]
        m = own == i
        c = d["cong"]
        vv, qq = C.corrected(d["pv"][c], d["pq"][c], rh[m], sh[m], CFG, LAM64)
        v[m], q[m] = vv, qq
        N[m] = C.dens(vv, qq, d["fd"], d["pos"][c])
    return v, q, N


def ta_features(data, P, Xs, s, mi, rh, vals):
    """TA design matrix over the pooled congested cells of month s, given V64 density predictions rh and values."""
    own = P[s]["own"]
    E_ = np.zeros((len(rh), 5))
    for i, p in enumerate(PANELS):
        d = data[s][p]
        m = own == i
        cm = d["cm"]
        u0 = np.log(np.maximum(1 + LAM64 * rh[m], 0.2))
        E_[m] = ta_extra(vals[2][m], u0, cm["ip"], cm["in"])
    return with_extra(Xs[mi], E_)


def nested_v64(P, Xs):
    """V64 density / split predictions for every congested hide cell from models that saw neither the cell's
    parity half nor its (day // 2) quarter: inside each training parity a, models on quarter b of parity a predict
    quarter 1-b of parity a.  Returns {a: ([rh_m0, rh_m1], [sh_m0, sh_m1])} valid on the cells of parity a."""
    f = CACHE / "v64nested.npz"
    if f.exists():
        z = np.load(f)
        return {a: ([z[f"rh{a}_0"], z[f"rh{a}_1"]], [z[f"sh{a}_0"], z[f"sh{a}_1"]]) for a in (0, 1)}
    X = np.concatenate(Xs)
    par = np.concatenate([P[s]["par"] for s in SPLITS])
    qtr = np.concatenate([P[s]["qtr"] for s in SPLITS])
    y = np.concatenate([P[s]["r"] for s in SPLITS])
    w = np.concatenate([P[s]["wd"] for s in SPLITS])
    ys = np.concatenate([P[s]["sstar"] for s in SPLITS])
    ws = np.concatenate([P[s]["ws"] for s in SPLITS])
    out, sav = {}, {}
    nm = [len(P[s]["r"]) for s in SPLITS]
    for a in (0, 1):
        rh, sh = np.full(len(y), np.nan), np.full(len(y), np.nan)
        for b in (0, 1):
            tr = (par == a) & (qtr == b)
            te = (par == a) & (qtr != b)
            md = C.train(X[tr], y[tr], w[tr], PRM31, R_D)
            rh[te] = md.predict(X[te], num_threads=THREADS)
            ms = C.train(X[tr], ys[tr], ws[tr], PRM31_S, R_S)
            sh[te] = ms.predict(X[te], num_threads=THREADS)
        out[a] = ([rh[:nm[0]], rh[nm[0]:]], [sh[:nm[0]], sh[nm[0]:]])
        for j in (0, 1):
            sav[f"rh{a}_{j}"], sav[f"sh{a}_{j}"] = out[a][0][j], out[a][1][j]
    np.savez(f, **sav)
    return out


def cmd_selftest(args) -> None:
    """Code-path check before registration of the grid: shapes, Laplacian on a toy pair, TA in-sample training loss
    on parity 0 for 30 rounds (no held-out cell is scored here)."""
    N = np.array([10.0, 14.0, 20.0])
    u = lap_u(N, np.array([-1, 0, -1]), np.array([1, -1, -1]), np.full(3, np.nan), np.full(3, np.nan), 0.1, 0.0)
    Nn = N * np.exp(u)
    assert abs((Nn[1] - Nn[0]) - 0.8 * 4.0) < 1e-9 and abs(Nn[:2].sum() - 24.0) < 1e-9 and Nn[2] == 20.0
    S = load_structure()
    data, P, Xs = load_all()
    rh64, sh64 = v64_crossfit(P, Xs)
    reference(data, P, rh64, sh64)
    prepare(data, P, S)
    wfac = {s: {p: class_factor(data[s][p], S, s, p) for p in PANELS} for s in SPLITS}
    vals = {s: v64_values(data, P, s, rh64[mi], sh64[mi]) for mi, s in enumerate(SPLITS)}
    for mi, s in enumerate(SPLITS):   # v64_values reproduces the reference frame values
        own = P[s]["own"]
        for i, p in enumerate(PANELS):
            d = data[s][p]
            assert np.allclose(vals[s][0][own == i], d["v64"][d["cong"]]) and                 np.allclose(vals[s][1][own == i], d["q64"][d["cong"]])
    sel = {s: P[s]["par"] == 0 for s in SPLITS}
    cell, tr = ta_data(data, P, sel, vals, S, wfac)
    X = np.concatenate([ta_features(data, P, Xs, s, mi, rh64[mi], vals[s])[sel[s]] for mi, s in enumerate(SPLITS)])
    obj = TAObjective(cell, tr)

    def loss(u):
        c = cell
        v = c["v64"] * np.exp(-ALPHA * u)
        ql = c["q64"] * np.exp((1 - ALPHA) * u) / c["lanes"]
        st = np.sum(c["a"] * (v - c["ov"]) ** 2 + c["b"] * (ql - c["oq"] / c["lanes"]) ** 2)
        Nn = c["N64"] * np.exp(u)
        res = tr["base"].copy()
        m0, m1 = tr["i0"] >= 0, tr["i1"] >= 0
        res[m1] += Nn[tr["i1"][m1]]
        res[m0] -= Nn[tr["i0"][m0]]
        return st, float(np.sum(tr["w"] * np.abs(res)))
    model = ta_train(X, cell, tr, 30)
    u = model.predict(X, num_threads=THREADS)
    print("cells", len(cell["v64"]), "transitions", len(tr["w"]), "X", X.shape, "scale", obj.scale)
    print("in-sample loss (state, lwr) at u=0", loss(np.zeros(len(u))), "after 30 rounds", loss(u),
          "u quantiles", np.quantile(u, [0.01, 0.5, 0.99]))


def cmd_grid(args) -> None:
    write_new(OUT / "prereg_grid.json", json.dumps({
        "written": time.strftime("%Y-%m-%d %H:%M:%S"), "grid": GRID, "order": list(GRID), "doc": GRID_DOC,
        "ta_params": TA_PRM, "ta_irls_floor": TA_EPS,
        "diagnostics_seen_before_registration": [
            "structure_real.csv (real vs hide transition class counts)",
            "baseline_classes.csv (V64 vs V60-like residual by class: V64 cuts E0/E1 residual ~6%, IC ~2%)",
            "on hide IC pairs only: corr of consecutive V64 errors 0.15-0.95; least-squares scale of the V64 pair "
            "difference vs the observed difference 0.43-1.54 (median 0.73 / 0.81); L1 of IC pair differences when "
            "the V64 difference is scaled by 0 / 0.5 / 0.75 / 0.9 / 1.25: median ratio 1.08-1.11 / 0.99-1.02 / "
            "0.98-1.00 / 0.99 / 1.03 (scale > 1 worse) -> grid of mild smoothing, no sharpening"],
        "criterion_file": "prereg_criterion.json"}, indent=2))
    print((OUT / "prereg_grid.json").read_text())


def cand_values(name, data, P, Xs, S, rh64, sh64):
    """Cross-fitted candidate (v, q) over the pooled congested cells of each month."""
    cfg = GRID[name]
    vals = {s: v64_values(data, P, s, rh64[mi], sh64[mi]) for mi, s in enumerate(SPLITS)}
    out = {}
    if cfg["kind"] == "lap":
        for mi, s in enumerate(SPLITS):
            own = P[s]["own"]
            u = np.zeros(len(rh64[mi]))
            for i, p in enumerate(PANELS):
                m = own == i
                cm = data[s][p]["cm"]
                u[m] = lap_u(vals[s][2][m], cm["ip"], cm["in"], cm["nvb"], cm["nva"], cfg["k"], cfg["kv"])
            out[s] = apply_u(vals[s][0], vals[s][1], u)
        return out
    # TA: train on parity a with nested V64 values, predict parity 1-a with the ordinary cross-fit V64 values
    NES = nested_v64(P, Xs)
    wfac = {s: {p: class_factor(data[s][p], S, s, p) for p in PANELS} for s in SPLITS}
    u = {s: np.zeros(len(rh64[mi])) for mi, s in enumerate(SPLITS)}
    for a in (0, 1):
        rhn, shn = NES[a]
        valn = {s: v64_values(data, P, s, rhn[mi], shn[mi]) for mi, s in enumerate(SPLITS)}
        sel = {s: P[s]["par"] == a for s in SPLITS}
        cell, tr = ta_data(data, P, sel, valn, S, wfac)
        Xtr = np.concatenate([ta_features(data, P, Xs, s, mi, rhn[mi], valn[s])[sel[s]] for mi, s in enumerate(SPLITS)])
        assert np.isfinite(rhn[0][sel[SPLITS[0]]]).all() and np.isfinite(rhn[1][sel[SPLITS[1]]]).all()
        model = ta_train(Xtr, cell, tr, cfg["rounds"])
        for mi, s in enumerate(SPLITS):
            te = P[s]["par"] != a
            Xte = ta_features(data, P, Xs, s, mi, rh64[mi], vals[s])[te]
            u[s][te] = model.predict(Xte, num_threads=THREADS)
        del cell, tr, Xtr, model
        gc.collect()
    for s in SPLITS:
        out[s] = apply_u(vals[s][0], vals[s][1], u[s])
    return out


def prepare(data, P, S):
    """Per corridor-month extras for the candidates (V64 reference must be attached)."""
    for s in SPLITS:
        qd = []
        for p in PANELS:
            d = data[s][p]
            d["cm"] = hide_cm(d)
            a, b, _ = C.cell_weights(d, d["v64"], d["q64"])
            d["wa"], d["wb"] = a, b
            d["wrw"], _ = C.target_weights(d, LOC.load_local(p, s, "T"), d["fd"])
            qd.append(((d["day"] // 2) % 2)[d["cong"]])
        P[s]["qtr"] = np.concatenate(qd)


def cmd_crossfit(args) -> None:
    for f in ("crossfit_grid.csv", "crossfit_summary.csv", "crossfit_choice.json"):
        if (OUT / f).exists():
            raise FileExistsError(OUT / f)
    assert (OUT / "prereg_criterion.json").exists() and (OUT / "prereg_grid.json").exists(), "register first"
    reg = json.loads((OUT / "prereg_grid.json").read_text())
    assert reg["grid"] == json.loads(json.dumps(GRID)), "grid changed after registration"
    t0 = time.time()
    S = load_structure()
    data, P, Xs = load_all()
    rh64, sh64 = v64_crossfit(P, Xs)
    reference(data, P, rh64, sh64)
    prepare(data, P, S)
    print("prepared", f"{time.time() - t0:.0f}s", flush=True)
    rows = []
    for name in GRID:
        cv = cand_values(name, data, P, Xs, S, rh64, sh64)
        for s in SPLITS:
            own = P[s]["own"]
            for i, p in enumerate(PANELS):
                d = data[s][p]
                m = own == i
                v, q = d["v64"].copy(), d["q64"].copy()
                c = d["cong"]
                v[c], q[c] = cv[s][0][m], cv[s][1][m]
                ch = (v[c] != d["v64"][c]) | (q[c] != d["q64"][c])
                rows.append({"cfg": name, "split": s, "panel": p, "n_cong": int(c.sum()), "n_changed": int(ch.sum()),
                             "rms_dv": float(np.sqrt(np.mean((v[c] - d["v64"][c]) ** 2))),
                             **score_vs64(d, v, q, S, s, p, d["wrw"])})
        G = pd.DataFrame([r for r in rows if r["cfg"] == name])
        print(name, {s: {k: f"{M.family_mean(G[G.split == s].set_index('panel')[k].to_dict()) * 1e4:+.3f}e-4"
                         for k in ("dS", "dS_rw", "dTotal_est", "dTotal_tr", "exp100", "exp40")} for s in SPLITS},
              f"{time.time() - t0:.0f}s", flush=True)
    G = pd.DataFrame(rows)
    G.to_csv(OUT / "crossfit_grid.csv", index=False)
    A = []
    for name in GRID:
        g = G[G.cfg == name]
        rec = {"cfg": name}
        for s in SPLITS:
            gg = g[g.split == s].set_index("panel")
            for k in ("dS", "dS_rw", "dFD_est", "dE_est", "dE_tr", "dphys_est", "dphys_tr", "dTotal_est",
                      "dTotal_tr", "exp100", "exp40", "lwr_part_tr", "fd_part"):
                rec[f"{k}_{s[:3]}"] = M.family_mean(gg[k].to_dict())
            rec[f"npos_tr_{s[:3]}"] = int((gg.dTotal_tr > 0).sum())
            rec[f"npos_est_{s[:3]}"] = int((gg.dTotal_est > 0).sum())
            rec[f"nposS_{s[:3]}"] = int((gg.dS > 0).sum())
        rec["dTot_tr_both"] = (rec["dTotal_tr_val"] + rec["dTotal_tr_pri"]) / 2
        ok = rec["dTot_tr_both"] >= 1e-5
        for m in ("val", "pri"):
            ok &= (rec[f"dTotal_est_{m}"] > 0) and (rec[f"dTotal_tr_{m}"] > 0) and (rec[f"npos_tr_{m}"] >= 8) \
                and (rec[f"dS_rw_{m}"] >= 0)
        rec["eligible"] = bool(ok)
        A.append(rec)
    A = pd.DataFrame(A).set_index("cfg")
    A.to_csv(OUT / "crossfit_summary.csv")
    choice, near = None, []
    if A.eligible.any():
        best = float(A[A.eligible].dTot_tr_both.max())
        near = [c for c in GRID if A.loc[c, "eligible"] and A.loc[c, "dTot_tr_both"] >= best - 3e-6]
        choice = near[0]
    nested = {}
    for sel, rep in (("val", "pri"), ("pri", "val")):
        okm = pd.Series(True, index=A.index)
        okm &= (A[f"dTotal_est_{sel}"] > 0) & (A[f"dTotal_tr_{sel}"] > 0) & (A[f"npos_tr_{sel}"] >= 8) & \
            (A[f"dS_rw_{sel}"] >= 0)
        E_ = A[okm]
        if len(E_):
            pk = E_[f"dTotal_tr_{sel}"].idxmax()
            nested[f"choose_on_{sel}"] = {"pick": pk, f"dTotal_tr_{rep}_of_pick": float(A.loc[pk, f"dTotal_tr_{rep}"])}
    out = {"criterion": json.loads((OUT / "prereg_criterion.json").read_text())["criterion"],
           "grid": GRID, "chosen_name": choice, "within_3e-6": near, "nested_month_check": nested,
           "chosen": {"name": choice, **GRID[choice]} if choice else None,
           "dS_family": {s: float(A.loc[choice, f"dS_{s[:3]}"]) for s in SPLITS} if choice else None,
           "summary": A.reset_index().to_dict("records"), "seconds": round(time.time() - t0)}
    write_new(OUT / "crossfit_choice.json", json.dumps(out, indent=2, default=float))
    pd.set_option("display.width", 320)
    show = [c for c in A.columns if any(c.startswith(k) for k in ("dS_", "dS_rw", "dE_tr", "dTotal", "exp", "npos_tr",
                                                                  "eligible", "dTot"))]
    print((A[show]).to_string())
    print("chosen", choice, near, nested, flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("register", "structure", "baseline", "selftest", "grid", "crossfit"):
        sub.add_parser(name)
    args = ap.parse_args()
    {"register": cmd_register, "structure": cmd_structure, "baseline": cmd_baseline, "selftest": cmd_selftest,
     "grid": cmd_grid, "crossfit": cmd_crossfit}[args.cmd](args)


if __name__ == "__main__":
    main()
