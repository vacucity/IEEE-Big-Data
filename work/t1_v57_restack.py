"""V57 Task 1 (lane S): month-specific re-stacking of the Task 1 components on top of V56_T1 (state rows only).

What changes compared with V56_T1 (t1_v56_refine.py, config D6L5|sc100_tod4), on exactly the same cells (predicted
free flow by the production state, > 1.15 v_cut, outside the Task 2 blackouts; congested and blackout cells keep the
V56 value):
  1) restack  each corridor-month gets its own small second-level model, fitted on that month's hide-20% cells
              (t1_v52_monthbias.py collect, reused, not rerun) on the free cells, target = link-demeaned production
              error e = pv - ov.  Inputs are the production components relative to the production output
              (vgbm - pv, v_w48 - pv, v_w48lwr - pv), the cell's offset from its link mean prediction x = pv - m_l
              and pv / free speed.  Speed: pv' = pv - f_v(.).  Flow (optional, grid): same with per-lane flow
              components, q' = q - lanes * f_q(.).  Models: LightGBM (grid below) or a linear ridge (reference).
  2) V56      the V56 correction (per link x 7 speed bands, flow scale + 4 h blocks) is REFIT on the restacked hide
              values (pv', pq') of the same month, with the free mask of the original pv, and applied to (pv', pq').
At the real targets the same components are available: GBM = work/sub/state_<split>.csv, U-Nets =
cache/t1_unet/pred_<panel>_<split>{_w48,_w48lwr}_vp.npz, pv = V51 state rows (v51_april.zip; production chain =
V40-half).  'apply' checks that the V56 rule refit here on the hide cells reproduces the V56 state rows exactly.
Each month uses only its own visible observations (Task 1 offline; V24 / V52 / V56 permission).  No Task 2 input is
read; queue and ODME rows are not touched (the full package takes the V54 queues, as V56).

Subcommands (project root, anaconda python, OMP_NUM_THREADS=4):
  crossfit   pre-registered grid + choice + nested month check + lambda curve + per-regime RMSE (odd/even days)
  direct     neighbour-proxy transfer check of the chosen config: real targets vs hidden cells, same estimator
  apply      fit on all days of each month; writes work/sub/state_{split}_v57rs.csv from the V56 state rows
             (v56_t1_on54.zip) in which only free non-blackout target rows change; S_FD on the real frame
  package    t1_v52_monthbias package code (imported, names patched) -> work/submissions/v57rs_state.zip
  audit      t1_v52_monthbias audit code (imported, names patched): v57rs_state.zip vs v51_april.zip
  compare    line-by-line v57_rs.zip (build_queue_candidate.py with the V54 queues) vs v56_t1_on54.zip
Outputs: work/results/t1_v57_restack/.  Nothing existing is overwritten (FileExistsError).
See work/V57_RESTACK_README.md.
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
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
import t1_v52_monthbias as M  # noqa: E402
import t1_v56_refine as R  # noqa: E402
from evaluate import CLEAN, KEY_BASE, PANELS, REGIMES, STEPS, links, slot_of, t1_targets  # noqa: E402

WORK = Path(__file__).resolve().parent
OUT = WORK / "results" / "t1_v57_restack"
SPLITS = M.SPLITS
V51_ZIP = WORK / "submissions" / "v51_april.zip"
V56_ZIP = WORK / "submissions" / "v56_t1_on54.zip"
NAME = "v57rs_state"
FULL_NAME = "v57_rs"
W_STATE, W_PHYS = 0.35, 0.15
CSV = {s: WORK / "sub" / f"state_{s}_v57rs.csv" for s in SPLITS}
THREADS = 4
CFG56 = {"speed": {"kind": "dev", "edges": [-3.0, -1.5, -0.5, 0.5, 1.5, 3.0], "K2": 5},
         "flow": {"kind": "scale_then", "Kq": 100, "then": "tod", "B": 4, "K2q": 200}}
COMP_V = ("vgbm", "v_w48", "v_w48lwr")
COMP_Q = ("qgbm", "q_w48", "q_w48lwr")
VF = ("dg", "du", "dl", "x", "rel")
QF = ("qg", "qu", "ql", "xq", "relq")
GBM_BASE = dict(objective="regression", learning_rate=0.05, num_leaves=15, min_data_in_leaf=1000, lambda_l2=10.0,
                num_threads=THREADS, verbose=-1, seed=1, deterministic=True, force_col_wise=True)


def write_new(path: Path, text: str) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.write_text(text, encoding="utf-8")


# ---------------------------------------------------------------------------
# V56 correction with an explicit free mask (== t1_v56_refine.fit / predict when free = pv > 1.15 v_cut)
# ---------------------------------------------------------------------------
def fit56(pos, pv, pq_, ov, oq, t, free, fd, cfg=CFG56):
    n = fd["n"]
    lanes = fd["lanes"][pos]
    ev, eq, pql = pv - ov, (pq_ - oq) / lanes, pq_ / lanes
    cnt = np.bincount(pos[free], minlength=n).astype(float)
    par = {"cnt": cnt,
           "bv": np.bincount(pos[free], weights=ev[free], minlength=n) / np.maximum(cnt + 25, 1e-9),
           "bq": np.bincount(pos[free], weights=eq[free], minlength=n) / np.maximum(cnt + 200, 1e-9),
           "mp": np.bincount(pos[free], weights=pv[free], minlength=n) / np.maximum(cnt, 1)}
    sp, fl = cfg["speed"], cfg["flow"]
    ed = np.asarray(sp["edges"], float)
    ng = len(ed) + 1
    g = np.searchsorted(ed, pv - par["mp"][pos])
    idx = pos * ng + g
    c2 = np.bincount(idx[free], minlength=n * ng).astype(float)
    s2 = np.bincount(idx[free], weights=(ev - par["bv"][pos])[free], minlength=n * ng)
    par["dev"] = (s2 / np.maximum(c2 + sp["K2"], 1e-9)).reshape(n, ng)
    spq = np.bincount(pos[free], weights=pql[free], minlength=n)
    sq = np.bincount(pos[free], weights=eq[free], minlength=n)
    mq = spq / np.maximum(cnt, 1)
    par["r"] = sq / np.maximum(spq + fl["Kq"] * mq, 1e-9)
    gq, ngq = R.flow_group(fl, t, pql)
    idx = pos * ngq + gq
    c2 = np.bincount(idx[free], minlength=n * ngq).astype(float)
    s2 = np.bincount(idx[free], weights=(eq - par["r"][pos] * pql)[free], minlength=n * ngq)
    par["qdev"] = (s2 / np.maximum(c2 + fl["K2q"], 1e-9)).reshape(n, ngq)
    return par


def pred56(par, pos, pv, pq_, t, free, fd, cfg=CFG56):
    lanes = fd["lanes"][pos]
    pql = pq_ / lanes
    ed = np.asarray(cfg["speed"]["edges"], float)
    g = np.searchsorted(ed, pv - par["mp"][pos])
    dv = np.where(par["cnt"][pos] > 0, par["bv"][pos] + par["dev"][pos, g], 0.0)
    gq, _ = R.flow_group(cfg["flow"], t, pql)
    dq = par["r"][pos] * pql + par["qdev"][pos, gq]
    v = np.where(free, np.clip(pv - dv, M.SPEED_MIN, M.SPEED_MAX), pv)
    q = np.where(free, np.maximum(pq_ - dq * lanes, 0.0), pq_)
    return v, q


# ---------------------------------------------------------------------------
# restack: features, fit on hidden cells, predict anywhere
# ---------------------------------------------------------------------------
def link_means(pos, pv, pq_, lanes, m, n):
    cnt = np.bincount(pos[m], minlength=n)
    mp = np.bincount(pos[m], weights=pv[m], minlength=n) / np.maximum(cnt, 1)
    mq = np.bincount(pos[m], weights=(pq_ / lanes)[m], minlength=n) / np.maximum(cnt, 1)
    return mp, mq


def features(pv, pq_, cv, cq, pos, fd, mp, mq):
    """cv / cq: the three speed / flow components (GBM, w48, w48lwr) at the same cells."""
    lanes = fd["lanes"][pos]
    pql = pq_ / lanes
    return {"dg": cv[0] - pv, "du": cv[1] - pv, "dl": cv[2] - pv, "x": pv - mp[pos], "rel": pv / fd["ffs"][pos],
            "qg": (cq[0] - pq_) / lanes, "qu": (cq[1] - pq_) / lanes, "ql": (cq[2] - pq_) / lanes,
            "xq": pql - mq[pos], "relq": pql / (fd["kcl"][pos] * fd["ffs"][pos])}


def demean(pos, m, a, n):
    cnt = np.bincount(pos[m], minlength=n).astype(float)
    if a.ndim == 1:
        return a[m] - (np.bincount(pos[m], weights=a[m], minlength=n) / np.maximum(cnt, 1))[pos[m]]
    return np.column_stack([demean(pos, m, a[:, j], n) for j in range(a.shape[1])])


def fit_channel(kind: dict, X, y):
    if kind["model"] == "lin":
        XtX = X.T @ X
        lam = kind["rho"] * np.trace(XtX) / X.shape[1]
        return ("lin", np.linalg.solve(XtX + lam * np.eye(X.shape[1]), X.T @ y))
    import lightgbm as lgb
    prm = {**GBM_BASE, "num_leaves": kind["leaves"], "min_data_in_leaf": kind["mdl"]}
    return ("gbm", lgb.train(prm, lgb.Dataset(X, y), num_boost_round=kind["rounds"]))


def pred_channel(model, X):
    kind, m = model
    return X @ m if kind == "lin" else m.predict(X, num_threads=THREADS)


class Restack:
    """Fitted on hidden cells (rows `tr` = training free cells of one corridor-month)."""

    def __init__(self, cfg, hide: dict, tr: np.ndarray, fd: dict):
        self.cfg, self.fd = cfg, fd
        n, pos = fd["n"], hide["pos"]
        self.mp, self.mq = link_means(pos, hide["pv"], hide["pq"], hide["lanes"], tr, n)
        F = features(hide["pv"], hide["pq"], hide["cv"], hide["cq"], pos, fd, self.mp, self.mq)
        self.models = {}
        for ch, names, e in (("v", VF, hide["pv"] - hide["ov"]), ("q", QF, (hide["pq"] - hide["oq"]) / hide["lanes"])):
            kind = cfg.get(ch)
            if kind is None:
                continue
            names = names[:3] if kind["model"] == "lin" else names
            X = np.column_stack([F[f] for f in names])
            Xt = demean(pos, tr, X, n) if kind["model"] == "lin" else X[tr]
            self.models[ch] = (names, fit_channel(kind, Xt, demean(pos, tr, e, n)))

    def apply(self, pv, pq_, cv, cq, pos, free, lam=1.0):
        F = features(pv, pq_, cv, cq, pos, self.fd, self.mp, self.mq)
        v, q = pv.copy(), pq_.copy()
        if "v" in self.models:
            names, mdl = self.models["v"]
            dv = pred_channel(mdl, np.column_stack([F[f] for f in names]))
            v = np.where(free, np.clip(pv - lam * dv, M.SPEED_MIN, M.SPEED_MAX), pv)
        if "q" in self.models:
            names, mdl = self.models["q"]
            dq = pred_channel(mdl, np.column_stack([F[f] for f in names]))
            q = np.where(free, np.maximum(pq_ - lam * dq * self.fd["lanes"][pos], 0.0), pq_)
        return v, q


def cfg_grid():
    """Pre-registered grid: speed model x flow model."""
    sp = {"lin": {"model": "lin", "rho": 0.03}}
    for lv, md in ((7, 1000), (15, 300), (15, 1000), (15, 3000)):
        sp[f"g{lv}m{md}"] = {"model": "gbm", "leaves": lv, "mdl": md, "rounds": 100}
    fl = {"none": None, "gq": {"model": "gbm", "leaves": 15, "mdl": 5000, "rounds": 50}}
    cfgs, cplx = {}, {}
    for sn, s in sp.items():
        for fn_, f in fl.items():
            nm = f"{sn}|{fn_}"
            cfgs[nm] = {"v": s, "q": f}
            cplx[nm] = (0 if f is None else 1, 0 if s["model"] == "lin" else 1, s.get("leaves", 0), -s.get("mdl", 0))
    return cfgs, cplx


# ---------------------------------------------------------------------------
# hide data + cross-fit
# ---------------------------------------------------------------------------
def load_hide(p, s):
    C, T, Dd = M.load_hide(p, s)
    C = C.reset_index(drop=True)
    fd = M.static_fd(p)
    pos = C.pos.to_numpy(np.int64)
    pv, pq_ = C.pv.to_numpy(float), C.pq.to_numpy(float)
    return dict(C=C, T=T, Dd=Dd, fd=fd, pos=pos, pv=pv, pq=pq_, ov=C.ov.to_numpy(float), oq=C.oq.to_numpy(float),
                t=C.t.to_numpy(np.int64), par=C.day.to_numpy() % 2, lanes=fd["lanes"][pos],
                free=pv > M.FREE_RATIO * fd["vc"][pos],
                cv=[C[c].to_numpy(float) for c in COMP_V], cq=[C[c].to_numpy(float) for c in COMP_Q],
                s0=M.state_score(C, pv, pq_, fd), a0=M.t3_score(C, T, Dd, pv, pq_, fd), sc=M.t3_scales(Dd))


def crossfit_one(d, cfg, lams=(1.0,)):
    """{lam: (v, q)} -- odd/even-day cross-fit; cfg None = V56 alone (refit on the same parity)."""
    out = {lam: (d["pv"].copy(), d["pq"].copy()) for lam in lams}
    for a in (0, 1):
        tr_all = d["par"] == a
        te = ~tr_all
        rs = None if cfg is None else Restack(cfg, d, tr_all & d["free"], d["fd"])
        for lam in lams:
            if rs is None:
                pv1, pq1 = d["pv"], d["pq"]
            else:
                pv1, pq1 = rs.apply(d["pv"], d["pq"], d["cv"], d["cq"], d["pos"], d["free"], lam)
            P = fit56(d["pos"][tr_all], pv1[tr_all], pq1[tr_all], d["ov"][tr_all], d["oq"][tr_all], d["t"][tr_all],
                      d["free"][tr_all], d["fd"])
            vv, qq = pred56(P, d["pos"][te], pv1[te], pq1[te], d["t"][te], d["free"][te], d["fd"])
            out[lam][0][te], out[lam][1][te] = vv, qq
    return out


def score(d, v, q):
    s1 = M.state_score(d["C"], v, q, d["fd"])
    a1 = M.t3_score(d["C"], d["T"], d["Dd"], v, q, d["fd"])
    t3 = M.t3_delta(d["a0"], a1, d["sc"])
    return {"dS": s1["S"] - d["s0"]["S"], "dphys_est": t3["dphys_est"], "dphys_raw": t3["dphys"],
            "dFD_est": t3["dFD_est"], "dE_est": t3["dE_est"],
            "dTotal_est": W_STATE * (s1["S"] - d["s0"]["S"]) + W_PHYS * t3["dphys_est"],
            **{k: s1[k] for k in s1 if k.endswith("_rv") or k.endswith("_rq")}}


def summarise(G: pd.DataFrame, ref: str = "V56") -> pd.DataFrame:
    base = G[G.cfg == ref].set_index(["split", "panel"])
    out = []
    for (cfg, s), g in G.groupby(["cfg", "split"]):
        g = g.set_index("panel")
        b = base.loc[s].loc[g.index]
        rec = {"cfg": cfg, "split": s}
        for k in ("dS", "dphys_est", "dphys_raw", "dFD_est", "dE_est", "dTotal_est"):
            dd = g[k] - b[k]
            rec[f"d{k}_vsV56_family"] = M.family_mean(dd.to_dict())
            if k in ("dS", "dTotal_est"):
                rec[f"npos_d{k}"] = int((dd > 0).sum())
                rec[f"min_d{k}"] = float(dd.min())
        out.append(rec)
    A = pd.DataFrame(out).pivot_table(index="cfg", columns="split")
    A.columns = [f"{a}_{'march' if b == 'validation' else 'april'}" for a, b in A.columns]
    A["ddTotal_est_vsV56_both"] = (A.ddTotal_est_vsV56_family_march + A.ddTotal_est_vsV56_family_april) / 2
    return A.sort_values("ddTotal_est_vsV56_both", ascending=False)


def cmd_crossfit(args) -> None:
    """Pre-registered rule (fixed before this grid was run; exploration in the lane-S scratch folder):
    maximise ddTot_both = mean over March/April of the family-mean (0.35 dS_state + 0.15 dS_phys real-frame
    estimate) relative to V56 (V56 refit after the restack, both cross-fitted on the same day parity); eligible
    configs have >= 9/10 corridors with ddTot > 0 in each month and family-mean ddS_state > 0 in each month;
    configs within 3e-6 of the max -> simplest (no flow restack before flow restack; linear before GBM; fewer
    leaves; larger min_data_in_leaf).  Gate checked later in 'apply': S_FD on the real submission frame must not
    fall by more than 1e-4 (family mean) in either month versus V56."""
    OUT.mkdir(parents=True, exist_ok=True)
    for f in ("crossfit_grid.csv", "crossfit_summary.csv", "crossfit_choice.json", "crossfit_lambda.csv"):
        if (OUT / f).exists():
            raise FileExistsError(OUT / f)
    t0 = time.time()
    cfgs, cplx = cfg_grid()
    rows = []
    for p in PANELS:
        for s in SPLITS:
            d = load_hide(p, s)
            v, q = crossfit_one(d, None)[1.0]
            # V56 through fit56/pred56 must equal t1_v56_refine.crossfit_one exactly
            v2, q2 = R.crossfit_one(R.load_cm(p, s), CFG56)
            assert np.array_equal(v, v2) and np.array_equal(q, q2), (p, s)
            rows.append({"panel": p, "split": s, "cfg": "V56", **score(d, v, q)})
            for nm, cfg in cfgs.items():
                v, q = crossfit_one(d, cfg)[1.0]
                rows.append({"panel": p, "split": s, "cfg": nm, **score(d, v, q)})
            print("crossfit", p, s, f"{time.time() - t0:.0f}s", flush=True)
            del d
            gc.collect()
    G = pd.DataFrame(rows)
    A = summarise(G)
    A["eligible"] = ((A.npos_ddTotal_est_march >= 9) & (A.npos_ddTotal_est_april >= 9) &
                     (A.ddS_vsV56_family_march > 0) & (A.ddS_vsV56_family_april > 0))
    A.loc["V56", "eligible"] = False
    best = float(A[A.eligible].ddTotal_est_vsV56_both.max())
    near = A[A.eligible & (A.ddTotal_est_vsV56_both >= best - 3e-6)].index.tolist()
    choice = sorted(near, key=lambda k: cplx[k])[0]
    nested = {}
    for sel, rep in (("march", "april"), ("april", "march")):
        E = A[(A[f"npos_ddTotal_est_{sel}"] >= 9) & (A[f"ddS_vsV56_family_{sel}"] > 0) & (A.index != "V56")]
        pick = E[f"ddTotal_est_vsV56_family_{sel}"].idxmax()
        nested[f"choose_on_{sel}"] = {"pick": pick,
                                      f"ddTotal_{rep}_of_pick": float(A.loc[pick, f"ddTotal_est_vsV56_family_{rep}"]),
                                      f"ddTotal_{rep}_of_choice": float(A.loc[choice, f"ddTotal_est_vsV56_family_{rep}"])}
    G.to_csv(OUT / "crossfit_grid.csv", index=False)
    A.to_csv(OUT / "crossfit_summary.csv")
    # lambda curve of the chosen config (restack increment scaled by lam before the V56 refit; reported only)
    lams = (0.5, 0.75, 1.25, 1.5)
    lrows = []
    for p in PANELS:
        for s in SPLITS:
            d = load_hide(p, s)
            res = crossfit_one(d, cfgs[choice], lams)
            for lam in lams:
                lrows.append({"panel": p, "split": s, "cfg": f"lam{lam}", **score(d, *res[lam])})
            del d, res
            gc.collect()
    L_ = pd.DataFrame(lrows)
    L_.to_csv(OUT / "crossfit_lambda.csv", index=False)
    LA = summarise(pd.concat([L_, G[G.cfg.isin(["V56", choice])]]))
    ch = G[G.cfg == choice].set_index(["split", "panel"])
    b56 = G[G.cfg == "V56"].set_index(["split", "panel"])
    per_cm = {s: {p: {"dS_vsV56": float(ch.loc[(s, p), "dS"] - b56.loc[(s, p), "dS"]),
                      "dphys_est_vsV56": float(ch.loc[(s, p), "dphys_est"] - b56.loc[(s, p), "dphys_est"]),
                      "dFD_est_vsV56": float(ch.loc[(s, p), "dFD_est"] - b56.loc[(s, p), "dFD_est"]),
                      "dE_est_vsV56": float(ch.loc[(s, p), "dE_est"] - b56.loc[(s, p), "dE_est"]),
                      "dTotal_est_vsV56": float(ch.loc[(s, p), "dTotal_est"] - b56.loc[(s, p), "dTotal_est"])}
                  for p in PANELS} for s in SPLITS}
    reg = {}
    for s in SPLITS:
        for r in (1, 2, 3):
            for k in ("rv", "rq"):
                col = f"R{r}_{k}"
                dd = (ch.loc[s][col] - b56.loc[s][col]).dropna()
                reg[f"{s}|R{r}|{k}"] = {"V56": float(b56.loc[s][col].dropna().mean()), "delta": float(dd.mean()),
                                        "corridors_improved": int((dd < 0).sum()), "corridors": int(len(dd))}
    key_cols = ["ddTotal_est_vsV56_both", "ddTotal_est_vsV56_family_march", "ddTotal_est_vsV56_family_april",
                "ddS_vsV56_family_march", "ddS_vsV56_family_april", "ddphys_est_vsV56_family_march",
                "ddphys_est_vsV56_family_april", "ddFD_est_vsV56_family_march", "ddFD_est_vsV56_family_april",
                "ddE_est_vsV56_family_march", "ddE_est_vsV56_family_april",
                "npos_ddTotal_est_march", "npos_ddTotal_est_april", "npos_ddS_march", "npos_ddS_april",
                "min_ddTotal_est_march", "min_ddTotal_est_april"]
    out = {"rule": cmd_crossfit.__doc__.strip(), "chosen_name": choice, "chosen": cfgs[choice],
           "max_eligible": best, "within_3e-6": near,
           "dS_family": {s: float(A.loc[choice, f"ddS_vsV56_family_{'march' if s == 'validation' else 'april'}"])
                         for s in SPLITS},
           "summary_choice": {k: float(A.loc[choice, k]) for k in key_cols},
           "summary_all": A[key_cols + ["eligible"]].reset_index().to_dict("records"),
           "per_corridor_month": per_cm, "per_regime_rmse": reg, "nested_month_check": nested,
           "lambda_curve": LA[key_cols].reset_index().to_dict("records"),
           "inputs_sha256": {f"{p}_{s}": M.sha256(M.HIDE / f"{p}_{s}_cells.parquet") for p in PANELS for s in SPLITS},
           "seconds": round(time.time() - t0)}
    write_new(OUT / "crossfit_choice.json", json.dumps(out, indent=2))
    pd.set_option("display.width", 320)
    print(A[key_cols + ["eligible"]].to_string(float_format=lambda x: f"{x:.6f}"))
    print(LA[key_cols].to_string(float_format=lambda x: f"{x:.6f}"))
    print(json.dumps({k: out[k] for k in ("chosen_name", "max_eligible", "within_3e-6", "summary_choice",
                                          "nested_month_check", "per_regime_rmse")}, indent=2), flush=True)


# ---------------------------------------------------------------------------
# real targets: production components, V51 / V56 state rows
# ---------------------------------------------------------------------------
def zip_state(path: Path):
    import pyarrow as pa
    import pyarrow.compute as pcc
    import pyarrow.csv as pc
    with zipfile.ZipFile(path) as z:
        assert len(z.namelist()) == 1
        with z.open(z.namelist()[0]) as f:
            tb = pc.read_csv(f, convert_options=pc.ConvertOptions(
                include_columns=["submission_id", "task", "speed_kmh", "flow_vph"],
                column_types={"submission_id": pa.int64(), "speed_kmh": pa.float64(), "flow_vph": pa.float64()}))
    tb = tb.filter(pcc.equal(tb["task"], "state"))
    return tb["submission_id"].to_numpy(), tb["speed_kmh"].to_numpy(), tb["flow_vph"].to_numpy()


def gbm_csv(split: str):
    g = pd.read_csv(WORK / "sub" / f"state_{split}.csv", usecols=["panel", "timestamp", "link_id", "speed_kmh", "flow_vph"],
                    dtype={"panel": "category", "timestamp": "category", "link_id": "category"})
    g["slot"] = slot_of(pd.Series(g.timestamp.cat.categories))[g.timestamp.cat.codes.to_numpy()]
    return g.drop(columns="timestamp")


def target_arrays(p, split, gf, tday_needed=False):
    """Production components at the real targets in t1_targets order (+ checks)."""
    from t1_blend import load_preds
    import t1_unet as U
    tg = t1_targets(p, split)
    g = gf[gf.panel == p]
    pos_of = dict(zip(links(p).link_id, links(p).pos))
    gk = g.slot.to_numpy() * KEY_BASE + g.link_id.astype(str).map(pos_of).to_numpy(np.int64)
    assert np.array_equal(gk, tg["key"]), (p, split, "GBM rows")
    out = {"vgbm": g.speed_kmh.to_numpy(float), "qgbm": g.flow_vph.to_numpy(float)}
    for tag in ("_w48", "_w48lwr"):
        pp = load_preds(U.DATA / f"pred_{p}_{split}{tag}_vp.npz")
        out[f"v{tag}"] = np.concatenate([pp[d][0] for d in sorted(pp)]).astype(float)
        out[f"q{tag}"] = np.concatenate([pp[d][1] for d in sorted(pp)]).astype(float)
        assert len(out[f"v{tag}"]) == len(tg["key"]), (p, split, tag)
    if tday_needed:
        pf = pq.ParquetFile(CLEAN / p / f"{split}.parquet")
        out["day"] = np.concatenate([np.full(int(pf.read_row_group(i, columns=["is_t1_target"]).column(0).to_numpy().sum()), i)
                                     for i in range(pf.num_row_groups)])
    return tg, out


def g6(x: np.ndarray) -> np.ndarray:
    return np.array([float("%.6g" % v) for v in x])


def load_choice():
    return json.loads((OUT / "crossfit_choice.json").read_text())


# ---------------------------------------------------------------------------
# direct: neighbour-proxy transfer check of the chosen config
# ---------------------------------------------------------------------------
def cmd_direct(args) -> None:
    """Models fit on hide cells of parity a (restack + V56 refit, and V56 alone), evaluated on parity (1-a) days
    at the real targets T (production components, V51 pv) and at the hidden cells H.  Cells: predicted free, not
    blackout, both temporal neighbours visible+eligible in that run; proxy truth y = mean of the two neighbour
    observations.  Reported: mean change of the proxy squared error (new - V56) on T and H (same estimator), the
    true change on H, and the RMS size of the change on T and H."""
    import t1_assemble as A
    path = OUT / "direct_proxy.csv"
    if path.exists():
        raise FileExistsError(path)
    cfg = load_choice()["chosen"]
    K = M.read_key_state()
    sid, v51, q51 = zip_state(V51_ZIP)
    assert np.array_equal(sid, K.submission_id.to_numpy())
    rows = []
    t0 = time.time()
    for split in SPLITS:
        gf = gbm_csv(split)
        Ks = K[K.split == split]
        for p, idx in Ks.groupby("panel", observed=True).indices.items():
            fd = M.static_fd(p)
            n = fd["n"]
            tg, z = target_arrays(p, split, gf, tday_needed=True)
            ii = Ks.index.to_numpy()[idx]
            j = np.searchsorted(tg["key"], K.key.to_numpy()[ii])
            o = np.empty(len(ii), np.int64)
            o[j] = ii
            tv, tq = v51[o], q51[o]
            tpos = (tg["key"] % KEY_BASE).astype(np.int64)
            tod = ((tg["key"] // KEY_BASE) % STEPS).astype(np.int64)
            gap = A.target_blackout(p, split).astype(bool)
            tfree = (tv > M.FREE_RATIO * fd["vc"][tpos]) & ~gap
            cvT, cqT = [z[c] for c in ("vgbm", "v_w48", "v_w48lwr")], [z[c] for c in ("qgbm", "q_w48", "q_w48lwr")]
            d = load_hide(p, split)
            outT = {k: tv.copy() for k in ("base", "new")}
            outH = {k: d["pv"].copy() for k in ("base", "new")}
            for a in (0, 1):
                tr_all = d["par"] == a
                rs = Restack(cfg, d, tr_all & d["free"], fd)
                pvH, pqH = rs.apply(d["pv"], d["pq"], d["cv"], d["cq"], d["pos"], d["free"])
                pvT, pqT = rs.apply(tv, tq, cvT, cqT, tpos, tfree)
                P1 = fit56(d["pos"][tr_all], pvH[tr_all], pqH[tr_all], d["ov"][tr_all], d["oq"][tr_all], d["t"][tr_all],
                           d["free"][tr_all], fd)
                P0 = fit56(d["pos"][tr_all], d["pv"][tr_all], d["pq"][tr_all], d["ov"][tr_all], d["oq"][tr_all],
                           d["t"][tr_all], d["free"][tr_all], fd)
                te = ~tr_all
                outH["new"][te] = pred56(P1, d["pos"][te], pvH[te], pqH[te], d["t"][te], d["free"][te], fd)[0]
                outH["base"][te] = pred56(P0, d["pos"][te], d["pv"][te], d["pq"][te], d["t"][te], d["free"][te], fd)[0]
                tt = (z["day"] % 2) != a
                outT["new"][tt] = pred56(P1, tpos[tt], pvT[tt], pqT[tt], tod[tt], tfree[tt], fd)[0]
                outT["base"][tt] = pred56(P0, tpos[tt], tv[tt], tq[tt], tod[tt], tfree[tt], fd)[0]
            pf = pq.ParquetFile(CLEAN / p / f"{split}.parquet")
            hid = {dd: g.index.to_numpy() for dd, g in d["C"].groupby("day")}
            ptr = 0
            acc = {"T": [], "H": []}
            for dd in range(pf.num_row_groups):
                t = pf.read_row_group(dd, columns=["is_t1_target", "is_score_eligible", "is_t2_blackout", "speed_obs",
                                                   "flow_obs"])
                g_ = lambda c: t.column(c).to_numpy().reshape(STEPS, n)
                tgt, blk, el = g_("is_t1_target"), g_("is_t2_blackout"), g_("is_score_eligible")
                ov, oq = g_("speed_obs").astype(float), g_("flow_obs").astype(float)
                vis = np.isfinite(ov) & np.isfinite(oq) & el
                k = int(tgt.sum())
                PT = {c: np.full((STEPS, n), np.nan) for c in ("pv", "base", "new")}
                PT["pv"][tgt], PT["base"][tgt], PT["new"][tgt] = (tv[ptr:ptr + k], outT["base"][ptr:ptr + k],
                                                                  outT["new"][ptr:ptr + k])
                ptr += k
                Hm = np.zeros((STEPS, n), bool)
                PH = {c: np.full((STEPS, n), np.nan) for c in ("pv", "base", "new", "ov")}
                if dd in hid:
                    ix = hid[dd]
                    tt_, pp_ = d["t"][ix], d["pos"][ix]
                    Hm[tt_, pp_] = True
                    PH["pv"][tt_, pp_], PH["base"][tt_, pp_] = d["pv"][ix], outH["base"][ix]
                    PH["new"][tt_, pp_], PH["ov"][tt_, pp_] = outH["new"][ix], d["ov"][ix]
                for nm, sel, P, visr in (("T", tgt & ~blk, PT, vis), ("H", Hm, PH, vis & ~Hm)):
                    ok = np.zeros((STEPS, n), bool)
                    ok[1:-1] = sel[1:-1] & visr[:-2] & visr[2:]
                    ok &= P["pv"] > M.FREE_RATIO * fd["vc"][None, :]
                    ti, li = np.nonzero(ok)
                    df = {"y": 0.5 * (ov[ti - 1, li] + ov[ti + 1, li]), "base": P["base"][ti, li], "new": P["new"][ti, li]}
                    if nm == "H":
                        df["ov"] = P["ov"][ti, li]
                    acc[nm].append(pd.DataFrame(df))
            assert ptr == len(tv)
            rec = {"split": split, "panel": p}
            for nm in ("T", "H"):
                A_ = pd.concat(acc[nm], ignore_index=True)
                rec[f"{nm}_cells"] = len(A_)
                rec[f"{nm}_proxy_dsse"] = float((((A_.new - A_.y) ** 2) - ((A_.base - A_.y) ** 2)).mean())
                rec[f"{nm}_rms_change"] = float(np.sqrt(((A_.new - A_.base) ** 2).mean()))
                if nm == "H":
                    rec["H_true_dsse"] = float((((A_.new - A_.ov) ** 2) - ((A_.base - A_.ov) ** 2)).mean())
            rec["T_all_free_rms_change"] = float(np.sqrt(np.mean((outT["new"] - outT["base"])[tfree] ** 2)))
            rec["H_all_free_rms_change"] = float(np.sqrt(np.mean((outH["new"] - outH["base"])[d["free"]] ** 2)))
            rows.append(rec)
            print({k: (round(v, 5) if isinstance(v, float) else v) for k, v in rec.items()}, f"{time.time() - t0:.0f}s",
                  flush=True)
            del d, z
            gc.collect()
        del gf
    D = pd.DataFrame(rows)
    D["T_minus_H_proxy"] = D.T_proxy_dsse - D.H_proxy_dsse
    D.to_csv(path, index=False)
    pd.set_option("display.width", 250)
    print(D.round(5).to_string())
    print(D.groupby("split").mean(numeric_only=True).round(5).T.to_string())


# ---------------------------------------------------------------------------
# apply
# ---------------------------------------------------------------------------
def cmd_apply(args) -> None:
    import t1_assemble as A
    choice = load_choice()
    cfg = choice["chosen"]
    for pth in CSV.values():
        if pth.exists():
            raise FileExistsError(pth)
    for f in ("apply_changes.csv", "apply_fd_frame.csv", "apply_summary.json"):
        if (OUT / f).exists():
            raise FileExistsError(OUT / f)
    t0 = time.time()
    K = M.read_key_state()
    sid, v51, q51 = zip_state(V51_ZIP)
    assert np.array_equal(sid, K.submission_id.to_numpy())
    sid, v56, q56 = zip_state(V56_ZIP)
    assert np.array_equal(sid, K.submission_id.to_numpy())
    del sid
    print("key + V51 + V56 state rows", len(K), f"{time.time() - t0:.0f}s", flush=True)
    rows, fd_rows = [], []
    for split in SPLITS:
        gf = gbm_csv(split)
        ks = (K.split == split).to_numpy()
        S_ = K[ks].reset_index(drop=True)
        V51v, V51q, V56v, V56q = v51[ks], q51[ks], v56[ks], q56[ks]
        v_all, q_all = V56v.copy(), V56q.copy()
        for p, idx in S_.groupby("panel", observed=True).indices.items():
            fd = M.static_fd(p)
            tg, z = target_arrays(p, split, gf)
            kk = S_.key.to_numpy()[idx]
            j = np.searchsorted(tg["key"], kk)
            assert len(idx) == len(tg["key"]) and np.array_equal(tg["key"][j], kk), p
            reg = S_.mask_regime.iloc[idx].astype(str).map({r: i for i, r in enumerate(REGIMES)}).to_numpy()
            assert np.array_equal(reg, tg["regime"][j]), p
            gap = A.target_blackout(p, split)[j].astype(bool)
            pos = (kk % KEY_BASE).astype(np.int64)
            tod = ((kk // KEY_BASE) % STEPS).astype(np.int64)
            cv = [z[c][j] for c in ("vgbm", "v_w48", "v_w48lwr")]
            cq = [z[c][j] for c in ("qgbm", "q_w48", "q_w48lwr")]
            a51, b51 = V51v[idx], V51q[idx]
            a56, b56 = V56v[idx], V56q[idx]
            free = (a51 > M.FREE_RATIO * fd["vc"][pos]) & ~gap
            # production lineage: the blend of the cached components reproduces V51 on the never-stacked corridor
            # and inside the blackouts
            bv = np.clip(0.1 * cv[0] + 0.3 * cv[1] + 0.6 * cv[2], M.SPEED_MIN, M.SPEED_MAX)
            bvg = np.clip(0.5 * (cv[1] + cv[2]), M.SPEED_MIN, M.SPEED_MAX)
            lineage_gap = float(np.max(np.abs(a51[gap] - bvg[gap]))) if gap.any() else 0.0
            assert lineage_gap < 2e-3, (split, p, lineage_gap)
            if M.ALPHA[p] == 0:
                assert np.max(np.abs(a51[~gap] - bv[~gap])) < 2e-3, (split, p)
            d = load_hide(p, split)
            allh = np.ones(len(d["pv"]), bool)
            # (1) V56 lineage: the V56 rule refit here on the hide cells reproduces the V56 state rows exactly
            P56 = fit56(d["pos"], d["pv"], d["pq"], d["ov"], d["oq"], d["t"], d["free"], fd)
            a, b = pred56(P56, pos, a51, b51, tod, free, fd)
            r56v, r56q = np.where(free, g6(a), a51), np.where(free, g6(b), b51)
            n_lineage = int(((r56v != a56) | (r56q != b56)).sum())
            assert n_lineage == 0, (split, p, n_lineage)
            # (2) restack (all days of the month) + V56 refit on the restacked hide values
            rs = Restack(cfg, d, d["free"] & allh, fd)
            pvH, pqH = rs.apply(d["pv"], d["pq"], d["cv"], d["cq"], d["pos"], d["free"])
            P57 = fit56(d["pos"], pvH, pqH, d["ov"], d["oq"], d["t"], d["free"], fd)
            pvT, pqT = rs.apply(a51, b51, cv, cq, pos, free)
            a, b = pred56(P57, pos, pvT, pqT, tod, free, fd)
            v1, q1 = np.where(free, g6(a), a56), np.where(free, g6(b), b56)
            assert np.isfinite(v1).all() and np.isfinite(q1).all() and (v1 > 0).all() and (q1 >= 0).all()
            assert np.array_equal(v1[~free], a56[~free]) and np.array_equal(q1[~free], b56[~free])
            assert np.array_equal(v1[gap], a56[gap]) and np.array_equal(q1[gap], b56[gap])
            v_all[idx], q_all[idx] = v1, q1
            ch = (v1 != a56) | (q1 != b56)
            fr = M.fd_frame(p, split, tg, j, a56, b56, v1, q1, fd)
            for r_ in fr:
                fd_rows.append({"split": split, "panel": p, "regime": r_["regime"], "FD_V56": r_["FD0"],
                                "FD_V57": r_["FD1"], "low_share_V57": r_["low_share1"], "targets": r_["targets"]})
            dvH = pvH - d["pv"]
            rows.append({"split": split, "panel": p, "targets": len(idx), "blackout_kept": int(gap.sum()),
                         "cells_rule_applies": int(free.sum()), "changed_vs_V56": int(ch.sum()),
                         "speed_changed_vs_V56": int((v1 != a56).sum()), "flow_changed_vs_V56": int((q1 != b56).sum()),
                         "lineage_mismatch_V56": n_lineage, "lineage_gap_max_abs": lineage_gap,
                         "mean_speed_change_vs_V56": float(np.mean((v1 - a56)[ch])) if ch.any() else 0.0,
                         "rms_speed_change_vs_V56": float(np.sqrt(np.mean((v1 - a56)[ch] ** 2))) if ch.any() else 0.0,
                         "max_abs_speed_change_vs_V56": float(np.max(np.abs(v1 - a56))),
                         "n_abs_speed_change_gt3": int((np.abs(v1 - a56) > 3).sum()),
                         "rms_flow_change_vs_V56": float(np.sqrt(np.mean((q1 - b56)[ch] ** 2))) if ch.any() else 0.0,
                         "max_abs_flow_change_vs_V56": float(np.max(np.abs(q1 - b56))),
                         "restack_rms_dv_targets": float(np.sqrt(np.mean((pvT - a51)[free] ** 2))),
                         "restack_rms_dv_hidden": float(np.sqrt(np.mean(dvH[d["free"]] ** 2))),
                         "low_flow_share_after": float((q1 < 50).mean()),
                         "speed_min_after": float(v1.min()), "speed_max_after": float(v1.max())})
            print(split, p, rows[-1]["changed_vs_V56"], "changed vs V56", f"{time.time() - t0:.0f}s", flush=True)
            del d, z, rs
            gc.collect()
        out = S_[["panel", "timestamp", "station_id", "link_id", "mask_regime"]].copy()
        out["speed_kmh"], out["flow_vph"] = v_all, q_all
        out.to_csv(CSV[split], index=False, float_format="%.6g")
        print("wrote", CSV[split], len(out), f"{time.time() - t0:.0f}s", flush=True)
        del S_, out, gf
        gc.collect()
    Rr = pd.DataFrame(rows)
    Rr.to_csv(OUT / "apply_changes.csv", index=False)
    F = pd.DataFrame(fd_rows)
    F.to_csv(OUT / "apply_fd_frame.csv", index=False)
    fdp = F.groupby(["split", "panel"])[["FD_V56", "FD_V57"]].mean()
    summ = {"config": cfg, "config_name": choice["chosen_name"], "changes": rows,
            "S_FD_real_frame_family_mean": {
                s: {k: M.family_mean(fdp.loc[s][k].to_dict()) for k in ("FD_V56", "FD_V57")} for s in SPLITS},
            "S_FD_real_frame_per_corridor": {s: (fdp.loc[s].FD_V57 - fdp.loc[s].FD_V56).to_dict() for s in SPLITS},
            "outputs_sha256": {s: M.sha256(CSV[s]) for s in SPLITS},
            "v51_zip_sha256": M.sha256(V51_ZIP), "v56_zip_sha256": M.sha256(V56_ZIP),
            "seconds": round(time.time() - t0)}
    gate = True
    for s in SPLITS:
        x = summ["S_FD_real_frame_family_mean"][s]
        x["delta_V57_vs_V56"] = x["FD_V57"] - x["FD_V56"]
        gate &= x["delta_V57_vs_V56"] > -1e-4
    summ["gate_S_FD_real_frame_ok"] = bool(gate)
    write_new(OUT / "apply_summary.json", json.dumps(summ, indent=2))
    pd.set_option("display.width", 320)
    print(Rr.round(4).to_string(), flush=True)
    print(json.dumps({k: summ[k] for k in ("S_FD_real_frame_family_mean", "gate_S_FD_real_frame_ok")}, indent=2), flush=True)


# ---------------------------------------------------------------------------
# package / audit: the t1_v52_monthbias code, imported, with the names patched
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
    # the imported manifest carries the V52 note / input names: write the correct supplement next to it
    man = json.loads((WORK / "results" / f"{NAME}_manifest.json").read_text(encoding="utf-8"))
    sup = {"name": NAME, "zip_sha256": man["sha256"], "changed_state_rows_vs_V51": man["changed_state_rows"],
           "note": "state = V56 state rows (v56_t1_on54.zip) with the free non-blackout target rows replaced by the V57 "
                   "month-specific restack + refit V56 correction (t1_v57_restack.py); queue/ODME rows = V51 "
                   "(v51_april.zip) byte-identical; the full candidate takes the V54 queues via build_queue_candidate.py",
           "state_inputs": [{"path": str(CSV[s]), "sha256": M.sha256(CSV[s])} for s in SPLITS],
           "config": load_choice()["chosen"], "config_name": load_choice()["chosen_name"]}
    write_new(OUT / "v57rs_manifest_supplement.json", json.dumps(sup, indent=2, ensure_ascii=False))


def cmd_audit(args) -> None:
    _patch()
    M.cmd_audit(args)


def cmd_compare(args) -> None:
    """v57_rs.zip vs v56_t1_on54.zip: queue/ODME lines byte-identical, state lines differ exactly where the CSVs
    differ from V56, every differing state value equals the CSV value, text in %.6g, no Task 2 blackout target row and
    no congested row changed (checked through the 'changed' set of the CSVs vs V56 being a subset of the rule cells)."""
    new = WORK / "submissions" / f"{FULL_NAME}.zip"
    path = OUT / "compare_v56.json"
    if path.exists():
        raise FileExistsError(path)
    sp, fl, K = _new_state_arrays()
    split_arr = np.empty(len(sp), object)
    panel_arr = np.empty(len(sp), object)
    split_arr[K.submission_id.to_numpy()] = K.split.astype(str).to_numpy()
    panel_arr[K.submission_id.to_numpy()] = K.panel.astype(str).to_numpy()
    del K
    st = {"rows": 0, "state_rows": 0, "state_changed": 0, "state_changed_speed": 0, "state_changed_flow": 0,
          "queue_rows": 0, "queue_changed": 0, "odme_rows": 0, "odme_changed": 0,
          "state_value_mismatch_vs_csv": 0, "state_text_not_6g": 0, "changed_by_split_panel": {},
          "max_abs_speed_change": 0.0, "max_abs_flow_change": 0.0}
    with zipfile.ZipFile(V56_ZIP) as za, zipfile.ZipFile(new) as zb:
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
    exp = {f"{r['split']}|{r['panel']}": r["changed_vs_V56"] for r in ap_["changes"]}
    st["changed_matches_apply"] = all(st["changed_by_split_panel"].get(k, 0) == v for k, v in exp.items())
    st["valid"] = bool(st["rows"] == 6980503 and st["queue_changed"] == 0 and st["odme_changed"] == 0 and
                       st["state_value_mismatch_vs_csv"] == 0 and st["state_text_not_6g"] == 0 and
                       st["state_changed"] > 0 and st["changed_matches_apply"])
    st["v56_sha256"], st["v57_rs_sha256"] = M.sha256(V56_ZIP), M.sha256(new)
    write_new(path, json.dumps(st, indent=2))
    print(json.dumps(st, indent=2), flush=True)
    assert st["valid"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("crossfit", "direct", "apply", "package", "audit", "compare"):
        sub.add_parser(name)
    args = ap.parse_args()
    {"crossfit": cmd_crossfit, "direct": cmd_direct, "apply": cmd_apply, "package": cmd_package,
     "audit": cmd_audit, "compare": cmd_compare}[args.cmd](args)


if __name__ == "__main__":
    main()
