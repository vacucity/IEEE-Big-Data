"""V61 Task 1 (lane C): congested non-blackout target cells -- a pooled-month DENSITY specialist (state rows only).

V60 = v60_loc_blk.zip = V58 queue/ODME + V59_loc state on the free non-blackout targets + V59_blk state on the
official Task 2 blackout targets.  The remaining state rows -- targets the production state predicts congested
(V51 pv <= 1.15 v_cut) outside the blackouts, 269,474 rows -- are still the V51 production values (checked).
V61_cg changes ONLY those rows.

Why: on the hide-20% cells (t1_v52_monthbias.py collect) the congested cells are ~4% of the hidden cells but carry
~21% of the density L1 error, and in the corridor-month objective 0.35 S_state + 0.15 S_phys the LWR term of a
congested cell outweighs its speed term ~10x (density N = q/v*L is large and 1/v-sensitive there).  Mean-shift (L2)
corrections improve the speed RMSE a little but increase the L1 density error (V52/V56/V57/V59 lesson).  Here the
correction is fitted in density space with an L1 objective instead:
  r = N_obs / N_prod - 1 at the hidden congested cells, LightGBM objective l1 (leaf = weighted median), weights =
  (marginal LWR weight of the cell's corridor-month-regime) * N_prod, so the fit minimises the expected sum of
  |N_new - N_obs| -- the quantity the LWR proxy accumulates over five-minute transitions.
  Apply: g = max(1 + r_hat, 0.2); v = clip(pv / g^alpha, 5, 120.75); q = pq * g^(1 - alpha) (density = N_prod * g).
  Optional second stage ('s' configs): an L2 model of the iso-density move s (v and q scaled by the same e^s, density
  unchanged), target s* = -(a pv (pv-ov) + b ql (ql-oql)) / (a pv^2 + b ql^2), weights a pv^2 + b ql^2 (a, b = the
  marginal speed / per-lane flow MSE weights of the cell's regime), applied with lambda_s 0.5.
One model per month, pooled over the 10 corridors (corridor id = categorical feature), fitted on that month's hide
cells only (March rows only use March, April rows only April; Task 1 offline).  Features (all relative to the
production prediction, identical code at hide cells and real targets): position on the FD (pv/v_cut, pv/ffs,
k/k_crit, k/k_jam, q/cap), component density log-ratios and speed deviations (GBM, w48_vp, w48lwr_vp), and the
local visible context of t1_v59_local (nearest visible obs before / after on the same link and their interpolation
as density log-ratios and speed deviations, up/downstream links, second-nearest obs, distances, counts).

Subcommands (project root, anaconda python, OMP_NUM_THREADS=4):
  crossfit   pre-registered grid + choice + nested month check + lambda curve + per regime + transfer reweighting
  apply      fit the chosen config on all days of each month; write work/sub/state_{split}_v61cg.csv from the V60 state
             rows (only congested non-blackout targets change); real-frame S_FD gate; T-vs-H size comparison
  package    t1_v52_monthbias package code (imported, names patched) -> work/submissions/v61cg_state.zip
  audit      t1_v52_monthbias audit code (imported, names patched): v61cg_state.zip vs v51_april.zip
  compare    line-by-line v61_cg.zip (build_queue_candidate.py with the V54 queues) vs v60_loc_blk.zip
Outputs: work/results/t1_v61_congest/ (cache/ = cross-fitted V59_loc baseline on the hide cells), models
work/models/t1_cg_v61_<split>_{d,s}.txt.  Nothing existing is overwritten (FileExistsError).
See work/V61_CONGEST_README.md.
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
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import t1_v52_monthbias as M  # noqa: E402
import t1_v57_restack as V  # noqa: E402
import t1_v59_local as LOC  # noqa: E402
from evaluate import KEY_BASE, PANELS, REGIMES  # noqa: E402

WORK = Path(__file__).resolve().parent
OUT = WORK / "results" / "t1_v61_congest"
CACHE = OUT / "cache"
MODELS = WORK / "models"
SPLITS = M.SPLITS
V51_ZIP = WORK / "submissions" / "v51_april.zip"
V60_ZIP = WORK / "submissions" / "v60_loc_blk.zip"
NAME = "v61cg_state"
FULL_NAME = "v61_cg"
CSV = {s: WORK / "sub" / f"state_{s}_v61cg.csv" for s in SPLITS}
THREADS = 4
W_STATE, W_PHYS = 0.35, 0.15
V59_CFG = "Lvq_x_q3000"
CAP = 12
FEATS = ("relc", "relf", "kr", "kj", "qr", "lkg", "lku", "lkl", "dg", "du", "dl", "db", "da", "lkb", "lka", "lki",
         "dvb", "dva", "dvi", "lku_", "lkd_", "dvu", "dvd", "lku1", "lkd1", "lkb2", "lka2", "nsp", "n3", "reg")
NOLOC = ("relc", "relf", "kr", "kj", "qr", "lkg", "lku", "lkl", "dg", "du", "dl", "reg")
PRM = dict(objective="l1", learning_rate=0.05, num_leaves=15, min_data_in_leaf=200, lambda_l2=10.0,
           num_threads=THREADS, verbose=-1, seed=1, deterministic=True, force_col_wise=True)
PRM_S = dict(PRM, objective="regression", min_data_in_leaf=500)
CFGS = {
    "F300a1": {"feats": FEATS, "rounds": 300, "alpha": 1.0},
    "F300a75": {"feats": FEATS, "rounds": 300, "alpha": 0.75},
    "F600a1": {"feats": FEATS, "rounds": 600, "alpha": 1.0},
    "F600a75": {"feats": FEATS, "rounds": 600, "alpha": 0.75},
    "N600a75": {"feats": NOLOC, "rounds": 600, "alpha": 0.75},
    "F600a75s": {"feats": FEATS, "rounds": 600, "alpha": 0.75, "split": {"rounds": 300, "lam": 0.5}},
}
CPLX = ["F300a1", "F300a75", "N600a75", "F600a1", "F600a75", "F600a75s"]


def write_new(path: Path, text: str) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.write_text(text, encoding="utf-8")


# ---------------------------------------------------------------------------
# per-cell quantities (identical code at hide cells and real targets)
# ---------------------------------------------------------------------------
def kl(v, q, lanes):
    """per-lane density q / max(v, 1) / lanes (the Task 3 identity k = q / v)."""
    return q / lanes / np.maximum(v, 1.0)


def dens(v, q, fd, pos):
    """accumulation N = q / max(v, 1) * L of the LWR proxy."""
    return q / np.maximum(v, 1.0) * fd["length"][pos]


def cong_feats(d, L, names):
    """d: pv, pq (production), cv / cq (components GBM, w48, w48lwr), pos, lanes, fd, reg; L: raw local quantities
    of t1_v59_local (visibility of the frame: hide-run visibility at hide cells, real visibility at targets)."""
    fd, pos, lanes = d["fd"], d["pos"], d["lanes"]
    pv, pq_ = d["pv"], d["pq"]
    kp = kl(pv, pq_, lanes)

    def lk(v, ql):
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.log(np.maximum(ql, 1.0) / np.maximum(v, 1.0) / np.maximum(kp, 1e-3))

    okb, oka = L["db"] <= CAP, L["da"] <= CAP
    vb, va = np.where(okb, L["vb"], np.nan), np.where(oka, L["va"], np.nan)
    qb, qa = np.where(okb, L["qb"], np.nan), np.where(oka, L["qa"], np.nan)
    both = okb & oka
    db, da = np.minimum(L["db"], CAP + 1), np.minimum(L["da"], CAP + 1)
    w = np.where(both, da / np.maximum(db + da, 1), np.nan)
    vi, qi = w * vb + (1 - w) * va, w * qb + (1 - w) * qa
    ok2b, ok2a = L["db2"] <= CAP, L["da2"] <= CAP
    F = {"relc": lambda: pv / fd["vc"][pos], "relf": lambda: pv / fd["ffs"][pos],
         "kr": lambda: kp / fd["kcl"][pos], "kj": lambda: kp / fd["kjl"][pos],
         "qr": lambda: pq_ / lanes / (fd["kcl"][pos] * fd["ffs"][pos]),
         "lkg": lambda: lk(d["cv"][0], d["cq"][0] / lanes), "lku": lambda: lk(d["cv"][1], d["cq"][1] / lanes),
         "lkl": lambda: lk(d["cv"][2], d["cq"][2] / lanes),
         "dg": lambda: d["cv"][0] - pv, "du": lambda: d["cv"][1] - pv, "dl": lambda: d["cv"][2] - pv,
         "db": lambda: db, "da": lambda: da, "lkb": lambda: lk(vb, qb), "lka": lambda: lk(va, qa),
         "lki": lambda: lk(vi, qi), "dvb": lambda: vb - pv, "dva": lambda: va - pv, "dvi": lambda: vi - pv,
         "lku_": lambda: lk(L["vu"], L["qu"]), "lkd_": lambda: lk(L["vd"], L["qd"]),
         "dvu": lambda: L["vu"] - pv, "dvd": lambda: L["vd"] - pv,
         "lku1": lambda: lk(L["vu1"], L["qu1"]), "lkd1": lambda: lk(L["vd1"], L["qd1"]),
         "lkb2": lambda: lk(np.where(ok2b, L["vb2"], np.nan), np.where(ok2b, L["qb2"], np.nan)),
         "lka2": lambda: lk(np.where(ok2a, L["va2"], np.nan), np.where(ok2a, L["qa2"], np.nan)),
         "nsp": lambda: L["nsp"], "n3": lambda: L["n3"], "reg": lambda: np.asarray(d["reg"], float)}
    return np.column_stack([np.asarray(F[k](), float) for k in names])


def cell_weights(d, v, q):
    """Marginal weights of the corridor-month objective at the hide cells, linearised at (v, q):
    a per squared speed error, b per squared per-lane flow error, c per unit LWR residual (veh, real-frame scaled)."""
    reg = d["reg"]
    ev2 = (v - d["ov"]) ** 2
    eq2 = ((q - d["oq"]) / d["lanes"]) ** 2
    Dd, sc = d["Dd"], d["sc"]
    regs = [r for r in range(3) if (reg == r).any()]
    R = len(regs)
    a, b, c = np.zeros(3), np.zeros(3), np.zeros(3)
    for r in regs:
        m = reg == r
        n = m.sum()
        a[r] = W_STATE / R * 0.54 / 25 / (2 * n * np.sqrt(ev2[m].mean()))
        b[r] = W_STATE / R * 0.46 / 600 / (2 * n * np.sqrt(eq2[m].mean()))
        c[r] = W_PHYS / R * (2 / 3) * sc[r][1] / Dd.lwr_den[Dd.regime == r].sum()
    return a[reg], b[reg], c[reg]


# ---------------------------------------------------------------------------
# hide data with the V60-like baseline
# ---------------------------------------------------------------------------
def v59_baseline(p, s, d, L):
    """Cross-fitted V59_loc values at the hide cells (free cells changed, congested = production); cached."""
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"v59cf_{s}_{p}.npz"
    if f.exists():
        z = np.load(f)
        return z["v"], z["q"]
    v, q = LOC.crossfit(d, L, LOC.CFGS[V59_CFG])[1.0]
    ref = pd.read_csv(LOC.OUT / "crossfit_grid.csv")
    ref = ref[ref.cfg == V59_CFG].set_index(["split", "panel"])
    sc = V.score(d, v, q)
    assert abs(sc["dTotal_est"] - ref.loc[(s, p), "dTotal_est"]) < 1e-12, (p, s)
    assert np.array_equal(v[~d["free"]], d["pv"][~d["free"]]) and np.array_equal(q[~d["free"]], d["pq"][~d["free"]])
    np.savez(f, v=v, q=q)
    return v, q


def load_month(s):
    """{panel: dict} with the hide data, local quantities, V59 baseline, scores of the baseline and cell weights."""
    data = {}
    for p in PANELS:
        d, L = LOC.load(p, s)
        d["bv"], d["bq"] = v59_baseline(p, s, d, L)
        d["s0"] = M.state_score(d["C"], d["bv"], d["bq"], d["fd"])
        d["a0"] = M.t3_score(d["C"], d["T"], d["Dd"], d["bv"], d["bq"], d["fd"])
        d["cong"] = ~d["free"]
        d["L"] = L
        a, b, c = cell_weights(d, d["bv"], d["bq"])
        Np = dens(d["pv"], d["pq"], d["fd"], d["pos"])
        No = dens(d["ov"], d["oq"], d["fd"], d["pos"])
        d["r"] = No / np.maximum(Np, 1e-6) - 1
        d["wd"] = c * Np
        pql, oql = d["pq"] / d["lanes"], d["oq"] / d["lanes"]
        h = a * d["pv"] ** 2 + b * pql ** 2
        d["sstar"] = np.clip(-(a * d["pv"] * (d["pv"] - d["ov"]) + b * pql * (pql - oql)) / h, -0.5, 0.5)
        d["ws"] = h
        data[p] = d
    return data


def pooled(data, feats):
    """Congested hide cells of all corridors of the month: X (+ corridor id), targets, weights, parity, owner."""
    X, own = [], []
    for i, p in enumerate(PANELS):
        d = data[p]
        c = d["cong"]
        X.append(np.column_stack([cong_feats(d, d["L"], feats)[c], np.full(c.sum(), i)]))
        own.append(np.full(c.sum(), i))
    cat = lambda k: np.concatenate([data[p][k][data[p]["cong"]] for p in PANELS])  # noqa: E731
    wd, ws = cat("wd"), cat("ws")
    return {"X": np.concatenate(X), "r": cat("r"), "wd": wd / wd.mean(), "sstar": cat("sstar"), "ws": ws / ws.mean(),
            "par": cat("par"), "own": np.concatenate(own)}


def train(X, y, w, prm, rounds):
    import lightgbm as lgb
    ds = lgb.Dataset(X, y, weight=w, categorical_feature=[X.shape[1] - 1], free_raw_data=True)
    return lgb.train(prm, ds, num_boost_round=rounds)


def crossfit_density(P, rounds):
    """Odd/even-day cross-fitted density predictions r_hat on the pooled congested hide cells."""
    rh = np.zeros(len(P["r"]))
    for a in (0, 1):
        tr = P["par"] == a
        m = train(P["X"][tr], P["r"][tr], P["wd"][tr], PRM, rounds)
        rh[~tr] = m.predict(P["X"][~tr], num_threads=THREADS)
    return rh


def crossfit_split(P, rounds):
    """Odd/even-day cross-fitted iso-density split predictions s_hat."""
    sh = np.zeros(len(P["r"]))
    for a in (0, 1):
        tr = P["par"] == a
        m = train(P["X"][tr], P["sstar"][tr], P["ws"][tr], PRM_S, rounds)
        sh[~tr] = m.predict(P["X"][~tr], num_threads=THREADS)
    return sh


def corrected(pv, pq_, rh, sh, cfg, lam=1.0):
    """New (v, q) from the production (pv, pq) and the model outputs (same formula at hide cells and targets)."""
    g = np.maximum(1 + lam * rh, 0.2)
    al = cfg["alpha"]
    e = np.exp(cfg["split"]["lam"] * sh) if "split" in cfg else 1.0
    v = np.clip(pv / g ** al * e, M.SPEED_MIN, M.SPEED_MAX)
    q = np.maximum(pq_ * g ** (1 - al) * e, 0.0)
    return v, q


def frame_values(d, rh_c, sh_c, cfg, lam=1.0):
    """Hide-cell values: baseline everywhere, corrected values on the congested cells."""
    c = d["cong"]
    v, q = d["bv"].copy(), d["bq"].copy()
    v[c], q[c] = corrected(d["pv"][c], d["pq"][c], rh_c, sh_c, cfg, lam)
    return v, q


def score(d, v, q, w_rw=None):
    s1 = M.state_score(d["C"], v, q, d["fd"])
    a1 = M.t3_score(d["C"], d["T"], d["Dd"], v, q, d["fd"])
    t3 = M.t3_delta(d["a0"], a1, d["sc"])
    out = {"dS": s1["S"] - d["s0"]["S"], "dphys_est": t3["dphys_est"], "dFD_est": t3["dFD_est"], "dE_est": t3["dE_est"],
           "dE_hide": t3["dE"], "dTotal_est": W_STATE * (s1["S"] - d["s0"]["S"]) + W_PHYS * t3["dphys_est"]}
    for r in range(3):
        if f"R{r + 1}_E" in a1:
            out[f"R{r + 1}_dE_hide"] = a1[f"R{r + 1}_E"] - d["a0"][f"R{r + 1}_E"]
            out[f"R{r + 1}_rv_0"], out[f"R{r + 1}_rv_1"] = d["s0"][f"R{r + 1}_rv"], s1[f"R{r + 1}_rv"]
            out[f"R{r + 1}_rq_0"], out[f"R{r + 1}_rq_1"] = d["s0"][f"R{r + 1}_rq"], s1[f"R{r + 1}_rq"]
    if w_rw is not None:
        # transfer: hide cells reweighted to the real congested non-blackout targets' strata
        sargs = (d["ov"], d["oq"], d["lanes"], d["reg"])
        out["S_rw"] = LOC.s_state_w(v, q, *sargs, w_rw) - LOC.s_state_w(d["bv"], d["bq"], *sargs, w_rw)
        No = dens(d["ov"], d["oq"], d["fd"], d["pos"])
        c = d["cong"]
        d1 = np.abs(dens(v, q, d["fd"], d["pos"]) - No)[c] - np.abs(dens(d["bv"], d["bq"], d["fd"], d["pos"]) - No)[c]
        out["dL1N_rel"] = float(d1.sum() / np.abs(dens(d["bv"], d["bq"], d["fd"], d["pos"]) - No)[c].sum())
        ww = w_rw[c]
        out["dL1N_rel_rw"] = float(np.sum(ww * d1) / np.sum(ww * np.abs(dens(d["bv"], d["bq"], d["fd"], d["pos"]) - No)[c]))
    return out


def strata(L, reg, m):
    db = np.minimum(L["db"][m], 4).astype(int) - 1
    da = np.minimum(L["da"][m], 4).astype(int) - 1
    ns = np.digitize(L["nsp"][m], [2, 4])
    return ((reg[m] * 4 + db) * 4 + da) * 3 + ns


def target_weights(d, LT, fd):
    """Hide-cell weights (congested cells only) = share of the real congested non-blackout targets in the cell's
    stratum (regime x min(db,4) x min(da,4) x nsp bucket) / share of the congested hide cells, within the regime."""
    pos = (LT["key"] % KEY_BASE).astype(np.int64)
    cT = ~(LT["v51"] > M.FREE_RATIO * fd["vc"][pos]) & ~LT["gap"]
    sT = strata(LT, LT["regime"].astype(int), cT)
    c = d["cong"]
    sH = strata(d["L"], d["reg"], c)
    nS = 3 * 16 * 3
    nT = np.bincount(sT, minlength=nS).astype(float)
    nH = np.bincount(sH, minlength=nS).astype(float)
    regS = np.arange(nS) // 48
    shT = nT / np.maximum(np.bincount(regS, weights=nT, minlength=3)[regS], 1)
    shH = nH / np.maximum(np.bincount(regS, weights=nH, minlength=3)[regS], 1)
    wS = np.where(nH > 0, shT / np.maximum(shH, 1e-12), 0.0)
    w = np.ones(len(c))
    w[c] = wS[sH]
    unc = float(np.mean([shT[(regS == r) & (nH == 0)].sum() for r in range(3)]))
    return w, unc


# ---------------------------------------------------------------------------
# crossfit
# ---------------------------------------------------------------------------
def summarise(G: pd.DataFrame) -> pd.DataFrame:
    keys = ["dS", "dphys_est", "dFD_est", "dE_est", "dTotal_est", "S_rw", "dL1N_rel", "dL1N_rel_rw"]
    out = []
    for nm in G.cfg.unique():
        g = G[G.cfg == nm].set_index(["split", "panel"])
        rec = {"cfg": nm}
        for s in SPLITS:
            gg = g.loc[s]
            for k in keys:
                if k in gg and gg[k].notna().all():
                    rec[f"{k}_{s[:3]}"] = M.family_mean(gg[k].to_dict())
            rec[f"npos_{s[:3]}"] = int((gg.dTotal_est > 0).sum())
            rec[f"nposS_{s[:3]}"] = int((gg.dS > 0).sum())
            rec[f"nE_{s[:3]}"] = int((gg.dE_est < 0).sum())
            rec[f"min_dTotal_{s[:3]}"] = float(gg.dTotal_est.min())
            rec[f"lower_bound_{s[:3]}"] = W_STATE * rec[f"dS_{s[:3]}"] + W_PHYS * rec[f"dFD_est_{s[:3]}"] / 3
        rec["dTot_both"] = (rec["dTotal_est_val"] + rec["dTotal_est_pri"]) / 2
        out.append(rec)
    return pd.DataFrame(out).set_index("cfg")


def eligible(A, months=("val", "pri")):
    ok = pd.Series(True, index=A.index)
    for m in months:
        ok &= (A[f"dE_est_{m}"] < 0) & (A[f"dS_{m}"] >= 0) & (A[f"S_rw_{m}"] >= 0) & (A[f"npos_{m}"] >= 8)
    return ok


def cmd_crossfit(args) -> None:
    """Pre-registered rule (fixed before this grid was run; exploration in the lane-C scratch folder wf11/congest):
    reference = V60 on the hide frame: free hidden cells at the cross-fitted V59_loc values (Lvq_x_q3000, reproduced
    exactly), congested hidden cells (production pv <= 1.15 v_cut) at the production values (= V60 on the congested
    non-blackout targets).  Only the congested hidden cells change.  Every config is a pooled-month L1 density model
    (+ optional iso-density split model), cross-fitted on odd/even days (all corridors of the month pooled, each
    corridor's own day parity).  Maximise dTot_both = mean over March/April of the family-mean (0.35 dS_state + 0.15
    dS_phys real-frame estimate) relative to V60.  Eligible: in EACH month family-mean dE_est < 0 (LWR proxy not
    worse), family-mean dS_state >= 0, family-mean reweighted dS_state >= 0 (congested hide cells reweighted to the
    real congested non-blackout targets' regime x min(db,4) x min(da,4) x nsp-bucket shares) and >= 8/10 corridors
    with dTotal_est > 0.  Configs within 3e-6 of the max -> the simplest (order CPLX).  Gate (checked in apply):
    real-frame S_FD (family mean) must not fall by more than 1e-4 versus V60 in either month.  No eligible config ->
    not built."""
    OUT.mkdir(parents=True, exist_ok=True)
    for f in ("crossfit_grid.csv", "crossfit_summary.csv", "crossfit_choice.json", "crossfit_lambda.csv",
              "crossfit_transfer_weights.csv"):
        if (OUT / f).exists():
            raise FileExistsError(OUT / f)
    t0 = time.time()
    rows, lrows, transfer = [], [], []
    LAMS = (0.5, 0.75, 1.25, 1.5)
    for s in SPLITS:
        data = load_month(s)
        wrw = {}
        for p in PANELS:
            LT = LOC.load_local(p, s, "T")
            wrw[p], unc = target_weights(data[p], LT, data[p]["fd"])
            c = data[p]["cong"]
            transfer.append({"split": s, "panel": p, "uncovered_target_share": unc, "w_max": float(wrw[p][c].max()),
                             "w_ess_frac": float(wrw[p][c].sum() ** 2 / (wrw[p][c] ** 2).sum() / c.sum()),
                             "n_cong_hide": int(c.sum())})
            del LT
        Pc, dpred, spred = {}, {}, {}
        for nm, cfg in CFGS.items():
            fk = "F" if cfg["feats"] is FEATS else "N"
            if fk not in Pc:
                Pc[fk] = pooled(data, cfg["feats"])
            P = Pc[fk]
            if (fk, cfg["rounds"]) not in dpred:
                dpred[fk, cfg["rounds"]] = crossfit_density(P, cfg["rounds"])
            rh = dpred[fk, cfg["rounds"]]
            sh = np.zeros(len(rh))
            if "split" in cfg:
                if (fk, cfg["split"]["rounds"]) not in spred:
                    spred[fk, cfg["split"]["rounds"]] = crossfit_split(P, cfg["split"]["rounds"])
                sh = spred[fk, cfg["split"]["rounds"]]
            for i, p in enumerate(PANELS):
                d = data[p]
                m = P["own"] == i
                v, q = frame_values(d, rh[m], sh[m], cfg)
                rows.append({"panel": p, "split": s, "cfg": nm, **score(d, v, q, wrw[p])})
                for lam in LAMS:   # lambda curve on the density step (reported only)
                    v, q = frame_values(d, rh[m], sh[m], cfg, lam)
                    lrows.append({"panel": p, "split": s, "cfg": f"{nm}|lam{lam}", **score(d, v, q, wrw[p])})
            print("crossfit", s, nm, f"{time.time() - t0:.0f}s", flush=True)
        del data, Pc, dpred, spred
        gc.collect()
    G = pd.DataFrame(rows)
    A = summarise(G)
    A["eligible"] = eligible(A)
    best = float(A[A.eligible].dTot_both.max()) if A.eligible.any() else None
    choice, near = None, []
    if best is not None:
        near = [c for c in CPLX if A.loc[c, "eligible"] and A.loc[c, "dTot_both"] >= best - 3e-6]
        choice = near[0]
    nested = {}
    for sel, rep in (("val", "pri"), ("pri", "val")):
        E = A[eligible(A, (sel,))]
        if len(E):
            pk = E[f"dTotal_est_{sel}"].idxmax()
            nested[f"choose_on_{sel}"] = {"pick": pk, f"dTotal_{rep}_of_pick": float(A.loc[pk, f"dTotal_est_{rep}"]),
                                          f"dTotal_{rep}_of_choice": float(A.loc[choice, f"dTotal_est_{rep}"]) if choice else None}
    G.to_csv(OUT / "crossfit_grid.csv", index=False)
    A.to_csv(OUT / "crossfit_summary.csv")
    pd.DataFrame(transfer).to_csv(OUT / "crossfit_transfer_weights.csv", index=False)
    out = {"rule": cmd_crossfit.__doc__.strip(), "chosen_name": choice, "chosen": None, "max_eligible": best,
           "within_3e-6": near, "nested_month_check": nested, "summary": A.reset_index().to_dict("records")}
    if choice is not None:
        cfg = CFGS[choice]
        out["chosen"] = {"name": choice, "feats": list(cfg["feats"]), "rounds": cfg["rounds"], "alpha": cfg["alpha"],
                         "split": cfg.get("split"), "params": {k: v for k, v in PRM.items()},
                         "params_split": {k: v for k, v in PRM_S.items()}}
        out["dS_family"] = {s: float(A.loc[choice, f"dS_{s[:3]}"]) for s in SPLITS}
        Lm = pd.DataFrame(lrows)
        Lm.to_csv(OUT / "crossfit_lambda.csv", index=False)
        LA = summarise(pd.concat([Lm[Lm.cfg.str.startswith(choice + "|")], G[G.cfg == choice]]))
        out["lambda_curve"] = {k: {"dTot_both": float(LA.loc[k, "dTot_both"]), "dTotal_val": float(LA.loc[k, "dTotal_est_val"]),
                                   "dTotal_pri": float(LA.loc[k, "dTotal_est_pri"]), "dS_val": float(LA.loc[k, "dS_val"]),
                                   "dS_pri": float(LA.loc[k, "dS_pri"])} for k in LA.index}
        ch = G[G.cfg == choice].set_index(["split", "panel"])
        out["per_corridor_month"] = {s: {p: {k: float(ch.loc[(s, p), k]) for k in
                                             ("dS", "S_rw", "dphys_est", "dFD_est", "dE_est", "dE_hide", "dTotal_est",
                                              "dL1N_rel", "dL1N_rel_rw")}
                                         for p in PANELS} for s in SPLITS}
        reg = {}
        for s in SPLITS:
            for r in (1, 2, 3):
                g = ch.loc[s]
                if f"R{r}_rv_0" not in g:
                    continue
                g = g[g[f"R{r}_rv_0"].notna()]
                reg[f"{s}|R{r}"] = {"corridors": int(len(g)), "speed_rmse_0": float(g[f"R{r}_rv_0"].mean()),
                                    "speed_rmse_1": float(g[f"R{r}_rv_1"].mean()),
                                    "flow_rmse_lane_0": float(g[f"R{r}_rq_0"].mean()),
                                    "flow_rmse_lane_1": float(g[f"R{r}_rq_1"].mean()),
                                    "dE_hide_mean": float(g[f"R{r}_dE_hide"].mean()),
                                    "speed_rmse_improved": int((g[f"R{r}_rv_1"] < g[f"R{r}_rv_0"]).sum())}
        out["per_regime"] = reg
    out["seconds"] = round(time.time() - t0)
    write_new(OUT / "crossfit_choice.json", json.dumps(out, indent=2))
    pd.set_option("display.width", 320)
    show = [c for c in A.columns if not c.startswith("min_")]
    print((A[show] * 1).round(7).to_string())
    print(json.dumps({k: out.get(k) for k in ("chosen_name", "max_eligible", "within_3e-6", "nested_month_check",
                                              "lambda_curve", "per_regime")}, indent=2), flush=True)


def load_choice():
    return json.loads((OUT / "crossfit_choice.json").read_text())


def chosen_cfg():
    c = load_choice()
    if c["chosen_name"] is None:
        raise SystemExit("no eligible config: not built")
    return c["chosen_name"], CFGS[c["chosen_name"]]


# ---------------------------------------------------------------------------
# apply
# ---------------------------------------------------------------------------
def target_inputs(p, split, gf):
    """Real-target analogue of the hide dict (production = V51 state rows) in t1_targets order + local quantities."""
    tg, z = V.target_arrays(p, split, gf)
    LT = LOC.load_local(p, split, "T")
    assert np.array_equal(LT["key"], tg["key"])
    fd = M.static_fd(p)
    pos = (tg["key"] % KEY_BASE).astype(np.int64)
    dT = {"fd": fd, "pos": pos, "lanes": fd["lanes"][pos], "pv": LT["v51"], "pq": LT["q51"],
          "cv": [z["vgbm"], z["v_w48"], z["v_w48lwr"]], "cq": [z["qgbm"], z["q_w48"], z["q_w48lwr"]],
          "reg": LT["regime"].astype(np.int64)}
    dT["cong"] = ~(dT["pv"] > M.FREE_RATIO * fd["vc"][pos]) & ~LT["gap"]
    return tg, dT, LT


def wquant(x, w, qs):
    o = np.argsort(x)
    cw = np.cumsum(w[o]) / w.sum()
    return [float(np.interp(q, cw, x[o])) for q in qs]


def cmd_apply(args) -> None:
    cname, cfg = chosen_cfg()
    for pth in CSV.values():
        if pth.exists():
            raise FileExistsError(pth)
    mpaths = {s: (MODELS / f"t1_cg_v61_{s}_d.txt", MODELS / f"t1_cg_v61_{s}_s.txt") for s in SPLITS}
    outs = [OUT / f for f in ("apply_changes.csv", "apply_fd_frame.csv", "apply_summary.json", "apply_transfer.csv",
                              "cong_sids.npy")] + [x for s in SPLITS for x in mpaths[s]]
    for f in outs:
        if f.exists():
            raise FileExistsError(f)
    t0 = time.time()
    K = M.read_key_state()
    sid, v51, q51 = V.zip_state(V51_ZIP)
    assert np.array_equal(sid, K.submission_id.to_numpy())
    sid, v60, q60 = V.zip_state(V60_ZIP)
    assert np.array_equal(sid, K.submission_id.to_numpy())
    del sid
    print("key + V51 + V60 state rows", len(K), f"{time.time() - t0:.0f}s", flush=True)
    rows, fd_rows, trows, cong_sids = [], [], [], []
    qs = (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)
    for split in SPLITS:
        data = load_month(split)
        P = pooled(data, cfg["feats"])
        md = train(P["X"], P["r"], P["wd"], PRM, cfg["rounds"])
        md.save_model(str(mpaths[split][0]))
        ms = None
        if "split" in cfg:
            ms = train(P["X"], P["sstar"], P["ws"], PRM_S, cfg["split"]["rounds"])
            ms.save_model(str(mpaths[split][1]))
        rH_all = md.predict(P["X"], num_threads=THREADS)
        sH_all = ms.predict(P["X"], num_threads=THREADS) if ms is not None else np.zeros(len(rH_all))
        gf = V.gbm_csv(split)
        ks = (K.split == split).to_numpy()
        S_ = K[ks].reset_index(drop=True)
        V51v, V51q, V60v, V60q = v51[ks], q51[ks], v60[ks], q60[ks]
        v_all, q_all = V60v.copy(), V60q.copy()
        gidx = S_.groupby("panel", observed=True).indices
        for i, p in enumerate(PANELS):
            idx = gidx[p]
            fd = M.static_fd(p)
            tg, dT, LT = target_inputs(p, split, gf)
            kk = S_.key.to_numpy()[idx]
            j = np.searchsorted(tg["key"], kk)
            assert len(idx) == len(tg["key"]) and np.array_equal(tg["key"][j], kk), p
            reg = S_.mask_regime.iloc[idx].astype(str).map({r: k for k, r in enumerate(REGIMES)}).to_numpy()
            assert np.array_equal(reg, tg["regime"][j]), p
            # t1_targets order below
            a51, b51, a60, b60 = (np.empty(len(j)) for _ in range(4))
            a51[j], b51[j], a60[j], b60[j] = V51v[idx], V51q[idx], V60v[idx], V60q[idx]
            assert np.array_equal(a51, LT["v51"]) and np.array_equal(b51, LT["q51"])
            c = dT["cong"]
            # lineage: V60 keeps the V51 production values on every congested non-blackout target
            n_lineage = int(((a60[c] != a51[c]) | (b60[c] != b51[c])).sum())
            assert n_lineage == 0, (split, p, n_lineage)
            X = np.column_stack([cong_feats(dT, LT, cfg["feats"]), np.full(len(c), i)])[c]
            rT = md.predict(X, num_threads=THREADS)
            sT = ms.predict(X, num_threads=THREADS) if ms is not None else np.zeros(len(rT))
            vn, qn = corrected(a51[c], b51[c], rT, sT, cfg)
            v1, q1 = a60.copy(), b60.copy()
            v1[c], q1[c] = V.g6(vn), V.g6(qn)
            assert np.isfinite(v1).all() and np.isfinite(q1).all() and (v1 > 0).all() and (q1 >= 0).all()
            assert np.array_equal(v1[~c], a60[~c]) and np.array_equal(q1[~c], b60[~c])
            assert not (c & LT["gap"]).any()
            v_all[idx], q_all[idx] = v1[j], q1[j]
            cong_sids.append(LT["sid"][c])
            ch = (v1 != a60) | (q1 != b60)
            fr = M.fd_frame(p, split, tg, np.arange(len(j)), a60, b60, v1, q1, fd)
            for r_ in fr:
                fd_rows.append({"split": split, "panel": p, "regime": r_["regime"], "FD_V60": r_["FD0"],
                                "FD_V61": r_["FD1"], "low_share_V61": r_["low_share1"], "targets": r_["targets"]})
            # T vs H: the same full-month model at the hide cells (in-sample) and the real targets
            d = data[p]
            rH, sH = rH_all[P["own"] == i], sH_all[P["own"] == i]
            wH, _ = target_weights(d, LT, fd)
            wH = wH[d["cong"]]
            vH, qH = corrected(d["pv"][d["cong"]], d["pq"][d["cong"]], rH, sH, cfg)
            dvT, dvH = (v1 - a60)[c], vH - d["pv"][d["cong"]]
            dqT = ((q1 - b60) / dT["lanes"])[c]
            rec = {"split": split, "panel": p, "targets": len(idx), "congested_nonblackout": int(c.sum()),
                   "blackout": int(LT["gap"].sum()), "changed_vs_V60": int(ch.sum()),
                   "speed_changed_vs_V60": int((v1 != a60).sum()), "flow_changed_vs_V60": int((q1 != b60).sum()),
                   "lineage_mismatch_V60_V51_congested": n_lineage,
                   "T_mean_dv": float(dvT.mean()), "T_rms_dv": float(np.sqrt(np.mean(dvT ** 2))),
                   "H_mean_dv": float(np.sum(wH * dvH) / wH.sum()), "H_rms_dv": float(np.sqrt(np.sum(wH * dvH ** 2) / wH.sum())),
                   "T_mean_dq_lane": float(dqT.mean()), "T_rms_dq_lane": float(np.sqrt(np.mean(dqT ** 2))),
                   "n_abs_dv_gt3": int((np.abs(dvT) > 3).sum()), "n_abs_dv_gt10": int((np.abs(dvT) > 10).sum()),
                   "max_abs_dv": float(np.abs(dvT).max()), "max_abs_dq": float(np.abs(q1 - b60).max()),
                   "low_flow_share_after": float((q1 < 50).mean()),
                   "speed_min_after": float(v1.min()), "speed_max_after": float(v1.max())}
            rows.append(rec)
            tq, hq = wquant(rT, np.ones(len(rT)), qs), wquant(rH, wH, qs)
            trows.append({"split": split, "panel": p, "nT": int(c.sum()), "nH": int(len(rH)),
                          **{f"T_r_q{q}": a for q, a in zip(qs, tq)}, **{f"Hrw_r_q{q}": a for q, a in zip(qs, hq)},
                          "T_r_mean": float(rT.mean()), "Hrw_r_mean": float(np.sum(wH * rH) / wH.sum()),
                          "T_r_mabs": float(np.abs(rT).mean()), "Hrw_r_mabs": float(np.sum(wH * np.abs(rH)) / wH.sum()),
                          "T_share_r_gt_0.1": float((rT > 0.1).mean()), "Hrw_share_r_gt_0.1": float(np.sum(wH * (rH > 0.1)) / wH.sum())})
            print(split, p, rec["changed_vs_V60"], "changed vs V60", f"{time.time() - t0:.0f}s", flush=True)
            del tg, dT, LT
            gc.collect()
        out = S_[["panel", "timestamp", "station_id", "link_id", "mask_regime"]].copy()
        out["speed_kmh"], out["flow_vph"] = v_all, q_all
        out.to_csv(CSV[split], index=False, float_format="%.6g")
        print("wrote", CSV[split], len(out), f"{time.time() - t0:.0f}s", flush=True)
        del S_, out, gf, data, P
        gc.collect()
    np.save(OUT / "cong_sids.npy", np.sort(np.concatenate(cong_sids)))
    Rr = pd.DataFrame(rows)
    Rr.to_csv(OUT / "apply_changes.csv", index=False)
    Tr = pd.DataFrame(trows)
    Tr.to_csv(OUT / "apply_transfer.csv", index=False)
    F = pd.DataFrame(fd_rows)
    F.to_csv(OUT / "apply_fd_frame.csv", index=False)
    fdp = F.groupby(["split", "panel"])[["FD_V60", "FD_V61"]].mean()
    summ = {"config": load_choice()["chosen"], "config_name": cname, "changes": rows,
            "S_FD_real_frame_family_mean": {
                s: {k: M.family_mean(fdp.loc[s][k].to_dict()) for k in ("FD_V60", "FD_V61")} for s in SPLITS},
            "S_FD_real_frame_per_corridor": {s: (fdp.loc[s].FD_V61 - fdp.loc[s].FD_V60).to_dict() for s in SPLITS},
            "outputs_sha256": {s: M.sha256(CSV[s]) for s in SPLITS},
            "models_sha256": {str(x): M.sha256(x) for s in SPLITS for x in mpaths[s] if x.exists()},
            "v51_zip_sha256": M.sha256(V51_ZIP), "v60_zip_sha256": M.sha256(V60_ZIP),
            "seconds": round(time.time() - t0)}
    gate = True
    for s in SPLITS:
        x = summ["S_FD_real_frame_family_mean"][s]
        x["delta_V61_vs_V60"] = x["FD_V61"] - x["FD_V60"]
        gate &= x["delta_V61_vs_V60"] > -1e-4
    summ["gate_S_FD_real_frame_ok"] = bool(gate)
    write_new(OUT / "apply_summary.json", json.dumps(summ, indent=2))
    pd.set_option("display.width", 320)
    print(Rr.round(4).to_string(), flush=True)
    print(Tr.round(4).to_string(), flush=True)
    print(json.dumps({k: summ[k] for k in ("S_FD_real_frame_family_mean", "S_FD_real_frame_per_corridor",
                                           "gate_S_FD_real_frame_ok")}, indent=2), flush=True)
    assert gate, "real-frame S_FD gate failed"


# ---------------------------------------------------------------------------
# package / audit / compare
# ---------------------------------------------------------------------------
def _patch():
    V.NAME = NAME
    V.OUT = OUT
    V.CSV = CSV
    M.NAME = NAME
    M.OUT = OUT
    M.BASE_ZIP = V51_ZIP
    M.new_state_arrays = V._new_state_arrays


def cmd_package(args) -> None:
    cname, cfg = chosen_cfg()
    _patch()
    M.cmd_package(args)
    man = json.loads((WORK / "results" / f"{NAME}_manifest.json").read_text(encoding="utf-8"))
    sup = {"name": NAME, "zip_sha256": man["sha256"], "changed_state_rows_vs_V51": man["changed_state_rows"],
           "note": "state = V60 state rows (v60_loc_blk.zip) with the congested (V51 pv <= 1.15 v_cut) non-blackout "
                   "target rows replaced by the V61 pooled-month L1 density specialist (t1_v61_congest.py); free "
                   "non-blackout rows = V59_loc and Task 2 blackout rows = V59_blk (both unchanged from V60); "
                   "queue/ODME rows = V51 (v51_april.zip) byte-identical; the full candidate takes the V54 queues via "
                   "build_queue_candidate.py",
           "state_inputs": [{"path": str(CSV[s]), "sha256": M.sha256(CSV[s])} for s in SPLITS],
           "config_name": cname, "config": load_choice()["chosen"]}
    write_new(OUT / "v61cg_manifest_supplement.json", json.dumps(sup, indent=2, ensure_ascii=False))


def cmd_audit(args) -> None:
    _patch()
    M.cmd_audit(args)


def cmd_compare(args) -> None:
    """v61_cg.zip vs v60_loc_blk.zip: queue/ODME lines byte-identical; state lines differ only at congested
    non-blackout target rows (cong_sids.npy from apply) and exactly where the CSVs differ from V60 (per split-panel
    counts = apply); every differing value equals the CSV value, %.6g text."""
    new = WORK / "submissions" / f"{FULL_NAME}.zip"
    path = OUT / "compare_v60.json"
    if path.exists():
        raise FileExistsError(path)
    _patch()
    sp, fl, K = V._new_state_arrays()
    split_arr = np.empty(len(sp), object)
    panel_arr = np.empty(len(sp), object)
    split_arr[K.submission_id.to_numpy()] = K.split.astype(str).to_numpy()
    panel_arr[K.submission_id.to_numpy()] = K.panel.astype(str).to_numpy()
    del K
    cong = np.zeros(len(sp), bool)
    cong[np.load(OUT / "cong_sids.npy")] = True
    st = {"rows": 0, "state_rows": 0, "state_changed": 0, "state_changed_speed": 0, "state_changed_flow": 0,
          "congested_nonblackout_rows": int(cong.sum()), "changed_outside_congested_nonblackout": 0,
          "queue_rows": 0, "queue_changed": 0, "odme_rows": 0, "odme_changed": 0,
          "state_value_mismatch_vs_csv": 0, "state_text_not_6g": 0, "changed_by_split_panel": {},
          "max_abs_speed_change": 0.0, "max_abs_flow_change": 0.0}
    with zipfile.ZipFile(V60_ZIP) as za, zipfile.ZipFile(new) as zb:
        assert za.testzip() is None and zb.testzip() is None
        with za.open(za.namelist()[0]) as fa, zb.open(zb.namelist()[0]) as fb:
            ha, hb = fa.readline(), fb.readline()
            assert ha == hb
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
                        st["changed_outside_congested_nonblackout"] += int(not cong[sid])
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
    ap_ = json.loads((OUT / "apply_summary.json").read_text())
    exp = {f"{r['split']}|{r['panel']}": r["changed_vs_V60"] for r in ap_["changes"]}
    st["changed_matches_apply"] = all(st["changed_by_split_panel"].get(k, 0) == v for k, v in exp.items())
    st["valid"] = bool(st["rows"] == 6980503 and st["queue_changed"] == 0 and st["odme_changed"] == 0 and
                       st["state_value_mismatch_vs_csv"] == 0 and st["state_text_not_6g"] == 0 and
                       st["changed_outside_congested_nonblackout"] == 0 and st["state_changed"] > 0 and
                       st["changed_matches_apply"])
    st["v60_sha256"], st["v61_cg_sha256"] = M.sha256(V60_ZIP), M.sha256(new)
    write_new(path, json.dumps(st, indent=2))
    print(json.dumps(st, indent=2), flush=True)
    assert st["valid"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("crossfit", "apply", "package", "audit", "compare"):
        sub.add_parser(name)
    args = ap.parse_args()
    {"crossfit": cmd_crossfit, "apply": cmd_apply, "package": cmd_package, "audit": cmd_audit,
     "compare": cmd_compare}[args.cmd](args)


if __name__ == "__main__":
    main()
