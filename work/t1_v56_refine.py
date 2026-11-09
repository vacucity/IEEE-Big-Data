"""V56 Task 1: refinement of the V52_t1bias per-month per-link correction (state rows of V53/V54/V55).

What changes compared with V52_t1bias (free_vq_K25_Kq200), still only on cells the production state predicts as
free flow (> 1.15 v_cut), Task 2 blackout target cells and congested cells untouched:
  speed  per-link offset b_l  ->  per-link x speed-level offset: the cell's predicted speed relative to the link
         mean free prediction, x = pv - m_l, is cut into 7 bands (edges -3, -1.5, -0.5, 0.5, 1.5, 3 km/h); the
         band offset is shrunk to b_l:  b_lg = b_l + sum_lg (e - b_l) / (n_lg + 5).  Reason: within free flow the
         production prediction is over-dispersed (error slope on x ~ +0.2; GBM 0.47, U-Nets 0.27-0.38, blend
         0.29, stacked 0.21), so high predictions are too high and low ones too low.
  flow   per-link per-lane offset  ->  per-link scale r_l = sum e_q / (sum pq_lane + 100 * mean pq_lane), then
         per-link 4-hour-block offsets of the remaining error shrunk with K2q = 200.
Chosen by the pre-registered rule below on the same odd/even-day cross-fit of the hide-20% cells collected by
t1_v52_monthbias.py collect (reused, not rerun).  Each month uses only its own visible observations (Task 1 is
offline; same permission as V24 / V52).  No Task 2 input is read; queue and ODME rows stay as in V55.

Subcommands (project root, anaconda python, OMP_NUM_THREADS=4):
  crossfit   pre-registered grid (16 speed x 7 flow configs) + choice + nested month check + lambda curve
  direct     neighbour-proxy check that the over-dispersion also holds at the REAL targets (V51 ZIP values)
  apply      fit on all days of each month, apply to the state rows (base: V55 = v55_nb.zip state rows, whose
             values are checked to equal the V52 rule applied to V51); writes work/sub/state_{split}_v56t1.csv
  package    t1_v52_monthbias package code (imported, names patched) -> work/submissions/v56t1_state.zip
  audit      t1_v52_monthbias audit code (imported, names patched): v56t1_state.zip vs v51_april.zip
  compare    line-by-line v56_t1.zip (built by build_queue_candidate.py) vs v55_nb.zip
Outputs: work/results/t1_v56_refine/.  Nothing existing is overwritten (FileExistsError).
See work/V56_T1_README.md.
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
from evaluate import CLEAN, FAMILY, KEY_BASE, PANELS, REGIMES, STEPS, links, t1_targets  # noqa: E402

WORK = Path(__file__).resolve().parent
OUT = WORK / "results" / "t1_v56_refine"
SPLITS = M.SPLITS
V51_ZIP = WORK / "submissions" / "v51_april.zip"
# state source of 'apply'.  Originally v55_nb.zip; that zip was deleted on 2026-10-20 (V55 not used).  Its state
# rows are byte-identical to v52_t1bias.zip (and v53/v54): 6,735,795 state lines, sha256 of the joined lines
# 6739e5514b220e16..., so pointing here reproduces the shipped V56 state exactly.
V55_ZIP = WORK / "submissions" / "v52_t1bias.zip"
NAME = "v56t1_state"
FULL_NAME = "v56_t1"
FREE_RATIO = M.FREE_RATIO
W_STATE, W_PHYS = 0.35, 0.15
CSV = {s: WORK / "sub" / f"state_{s}_v56t1.csv" for s in SPLITS}

V52_CFG = {"speed": {"kind": "offset"}, "flow": {"kind": "offset"}}
EDGES = {"D4": (-1.5, 0.0, 1.5), "D5": (-2.0, -1.0, 0.0, 1.0, 2.0), "D6": (-3.0, -1.5, -0.5, 0.5, 1.5, 3.0)}
LEVELS = (300.0, 600.0, 900.0, 1200.0, 1500.0)


def write_new(path: Path, text: str) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.write_text(text, encoding="utf-8")


# ---------------------------------------------------------------------------
# the correction: fit on hidden cells, predict anywhere
# ---------------------------------------------------------------------------
def fit(pos, pv, pq, ov, oq, t, fd, cfg):
    """Parameters from hidden cells (arrays of one corridor-month, any subset of days)."""
    n = fd["n"]
    lanes = fd["lanes"][pos]
    free = pv > FREE_RATIO * fd["vc"][pos]
    ev, eq, pql = pv - ov, (pq - oq) / lanes, pq / lanes
    cnt = np.bincount(pos[free], minlength=n).astype(float)
    par = {"cnt": cnt,
           "bv": np.bincount(pos[free], weights=ev[free], minlength=n) / np.maximum(cnt + 25, 1e-9),
           "bq": np.bincount(pos[free], weights=eq[free], minlength=n) / np.maximum(cnt + 200, 1e-9),
           "mp": np.bincount(pos[free], weights=pv[free], minlength=n) / np.maximum(cnt, 1)}
    sp, fl = cfg["speed"], cfg["flow"]
    if sp["kind"] == "dev":
        ed = np.asarray(sp["edges"], float)
        ng = len(ed) + 1
        g = np.searchsorted(ed, pv - par["mp"][pos])
        base = par["bv"][pos]
        cg = np.zeros(ng)
        if sp.get("corr"):
            cg = np.bincount(g[free], weights=(ev - base)[free], minlength=ng) / np.maximum(
                np.bincount(g[free], minlength=ng), 1)
            base = base + cg[g]
        idx = pos * ng + g
        c2 = np.bincount(idx[free], minlength=n * ng).astype(float)
        s2 = np.bincount(idx[free], weights=(ev - base)[free], minlength=n * ng)
        par["cg"] = cg
        par["dev"] = (s2 / np.maximum(c2 + sp["K2"], 1e-9)).reshape(n, ng)
    if fl["kind"] in ("scale", "scale_then"):
        spq = np.bincount(pos[free], weights=pql[free], minlength=n)
        sq = np.bincount(pos[free], weights=eq[free], minlength=n)
        mq = spq / np.maximum(cnt, 1)
        par["r"] = sq / np.maximum(spq + fl["Kq"] * mq, 1e-9)
        if fl["kind"] == "scale_then":
            g, ng = flow_group(fl, t, pql)
            base = par["r"][pos] * pql
            idx = pos * ng + g
            c2 = np.bincount(idx[free], minlength=n * ng).astype(float)
            s2 = np.bincount(idx[free], weights=(eq - base)[free], minlength=n * ng)
            par["qdev"] = (s2 / np.maximum(c2 + fl["K2q"], 1e-9)).reshape(n, ng)
    return par


def flow_group(fl, t, pql):
    if fl["then"] == "tod":
        ng = 24 // fl["B"]
        return np.clip(t // (12 * fl["B"]), 0, ng - 1).astype(np.int64), ng
    ed = np.asarray(LEVELS, float)
    return np.searchsorted(ed, pql), len(ed) + 1


def predict(par, pos, pv, pq, t, fd, cfg):
    """Corrected (v, q) and the mask of changed cells."""
    lanes = fd["lanes"][pos]
    pql = pq / lanes
    free = pv > FREE_RATIO * fd["vc"][pos]
    sp, fl = cfg["speed"], cfg["flow"]
    dv = par["bv"][pos].copy()
    if sp["kind"] == "dev":
        ed = np.asarray(sp["edges"], float)
        g = np.searchsorted(ed, pv - par["mp"][pos])
        dv = dv + par["cg"][g] + par["dev"][pos, g]
        dv = np.where(par["cnt"][pos] > 0, dv, 0.0)        # no hidden free cell on the link: no correction
    if fl["kind"] == "offset":
        dq = par["bq"][pos]
    else:
        dq = par["r"][pos] * pql
        if fl["kind"] == "scale_then":
            g, _ = flow_group(fl, t, pql)
            dq = dq + par["qdev"][pos, g]
    v = np.where(free, np.clip(pv - dv, M.SPEED_MIN, M.SPEED_MAX), pv)
    q = np.where(free, np.maximum(pq - dq * lanes, 0.0), pq)
    return v, q, free & ((dv != 0) | (dq != 0))


def cfg_grid():
    sp = {"s52": {"kind": "offset"}}
    for e, ed in EDGES.items():
        for K2 in (5, 10, 15):
            sp[f"{e}L{K2}"] = {"kind": "dev", "edges": list(ed), "K2": K2}
        for K2 in (25, 50):
            sp[f"{e}C{K2}"] = {"kind": "dev", "edges": list(ed), "K2": K2, "corr": True}
    fl = {"f52": {"kind": "offset"}, "sc50": {"kind": "scale", "Kq": 50}, "sc100": {"kind": "scale", "Kq": 100},
          "sc200": {"kind": "scale", "Kq": 200},
          "sc100_tod4": {"kind": "scale_then", "Kq": 100, "then": "tod", "B": 4, "K2q": 200},
          "sc100_tod6": {"kind": "scale_then", "Kq": 100, "then": "tod", "B": 6, "K2q": 200},
          "sc100_lvl800": {"kind": "scale_then", "Kq": 100, "then": "lvl", "K2q": 800}}
    cfgs, cplx = {}, {}
    for sn, s in sp.items():
        for fn_, f in fl.items():
            nm = "V52" if (sn, fn_) == ("s52", "f52") else f"{sn}|{fn_}"
            cfgs[nm] = {"speed": s, "flow": f}
            cplx[nm] = (1 if s.get("corr") else 0, {"offset": 0, "scale": 1, "scale_then": 2}[f["kind"]],
                        0 if s["kind"] == "offset" else len(s["edges"]) + 1)
    return cfgs, cplx


# ---------------------------------------------------------------------------
# crossfit
# ---------------------------------------------------------------------------
def load_cm(p, s):
    C, T, Dd = M.load_hide(p, s)
    C = C[["day", "regime", "t", "pos", "ov", "oq", "pv", "pq"]].reset_index(drop=True)
    fd = M.static_fd(p)
    pv, pq_ = C.pv.to_numpy(float), C.pq.to_numpy(float)
    return dict(C=C, T=T, Dd=Dd, fd=fd, s0=M.state_score(C, pv, pq_, fd), a0=M.t3_score(C, T, Dd, pv, pq_, fd),
                sc=M.t3_scales(Dd), arr=(C.pos.to_numpy(np.int64), pv, pq_, C.ov.to_numpy(float),
                                         C.oq.to_numpy(float), C.t.to_numpy(np.int64)))


def crossfit_one(d, cfg, lam=1.0):
    """Odd/even-day cross-fit; lam scales the increment over the V52 correction (reported only)."""
    pos, pv, pq_, ov, oq, t = d["arr"]
    par_ = d["C"].day.to_numpy() % 2
    v, q = pv.copy(), pq_.copy()
    for a in (0, 1):
        tr, te = par_ == a, par_ != a
        P = fit(pos[tr], pv[tr], pq_[tr], ov[tr], oq[tr], t[tr], d["fd"], cfg)
        vv, qq, _ = predict(P, pos[te], pv[te], pq_[te], t[te], d["fd"], cfg)
        if lam != 1.0:
            P0 = fit(pos[tr], pv[tr], pq_[tr], ov[tr], oq[tr], t[tr], d["fd"], V52_CFG)
            v0, q0, _ = predict(P0, pos[te], pv[te], pq_[te], t[te], d["fd"], V52_CFG)
            vv, qq = v0 + lam * (vv - v0), np.maximum(q0 + lam * (qq - q0), 0.0)
        v[te], q[te] = vv, qq
    return v, q


def score(d, v, q):
    C, fd = d["C"], d["fd"]
    s1 = M.state_score(C, v, q, fd)
    a1 = M.t3_score(C, d["T"], d["Dd"], v, q, fd)
    t3 = M.t3_delta(d["a0"], a1, d["sc"])
    return {"dS": s1["S"] - d["s0"]["S"], "dphys_est": t3["dphys_est"], "dphys_raw": t3["dphys"],
            "dFD_est": t3["dFD_est"], "dE_est": t3["dE_est"],
            "dTotal_est": W_STATE * (s1["S"] - d["s0"]["S"]) + W_PHYS * t3["dphys_est"],
            **{k: s1[k] for k in s1 if k.endswith("_rv") or k.endswith("_rq")}}


def summarise(R: pd.DataFrame) -> pd.DataFrame:
    base = R[R.cfg == "V52"].set_index(["split", "panel"])
    out = []
    for (cfg, s), g in R.groupby(["cfg", "split"]):
        g = g.set_index("panel")
        b = base.loc[s].loc[g.index]
        rec = {"cfg": cfg, "split": s}
        for k in ("dS", "dphys_est", "dphys_raw", "dFD_est", "dE_est", "dTotal_est"):
            rec[f"{k}_vsV51_family"] = M.family_mean(g[k].to_dict())
            dd = g[k] - b[k]
            rec[f"d{k}_vsV52_family"] = M.family_mean(dd.to_dict())
            if k in ("dS", "dTotal_est"):
                rec[f"npos_d{k}"] = int((dd > 0).sum())
                rec[f"min_d{k}"] = float(dd.min())
        out.append(rec)
    A = pd.DataFrame(out).pivot_table(index="cfg", columns="split")
    A.columns = [f"{a}_{'march' if b == 'validation' else 'april'}" for a, b in A.columns]
    A["ddTotal_est_vsV52_both"] = (A.ddTotal_est_vsV52_family_march + A.ddTotal_est_vsV52_family_april) / 2
    return A.sort_values("ddTotal_est_vsV52_both", ascending=False)


def cmd_crossfit(args) -> None:
    """Pre-registered rule (fixed before this grid was run):
    maximise ddTot_both = mean over March/April of the family-mean (0.35 dS_state + 0.15 dS_phys real-frame
    estimate) relative to V52; eligible configs have >= 9/10 corridors with ddTot > 0 in each month and family-mean
    ddS_state > 0 in each month; configs within 2e-6 of the max -> fewest parameters (per-link 'L' before
    corridor 'C' bands; offset < scale < scale+tod/level for flow; fewer bands)."""
    OUT.mkdir(parents=True, exist_ok=True)
    for f in ("crossfit_grid.csv", "crossfit_summary.csv", "crossfit_choice.json", "crossfit_lambda.csv"):
        if (OUT / f).exists():
            raise FileExistsError(OUT / f)
    t0 = time.time()
    cfgs, cplx = cfg_grid()
    rows, data = [], {}
    for p in PANELS:
        for s in SPLITS:
            d = load_cm(p, s)
            data[p, s] = d
            # the V52 config through fit/predict must equal t1_v52_monthbias.crossfit_pred exactly
            v52v, v52q, _ = M.crossfit_pred(d["C"], d["fd"], {"sel": "free", "ch": "vq", "K": 25, "Kq": 200})
            for nm, cfg in cfgs.items():
                v, q = crossfit_one(d, cfg)
                if nm == "V52":
                    assert np.array_equal(v, v52v) and np.array_equal(q, v52q), (p, s)
                rows.append({"panel": p, "split": s, "cfg": nm, **score(d, v, q)})
        print("crossfit", p, f"{time.time() - t0:.0f}s", flush=True)
    G = pd.DataFrame(rows)
    A = summarise(G)
    A["eligible"] = ((A.npos_ddTotal_est_march >= 9) & (A.npos_ddTotal_est_april >= 9) &
                     (A.ddS_vsV52_family_march > 0) & (A.ddS_vsV52_family_april > 0))
    best = float(A[A.eligible].ddTotal_est_vsV52_both.max())
    near = A[A.eligible & (A.ddTotal_est_vsV52_both >= best - 2e-6)].index.tolist()
    choice = sorted(near, key=lambda k: cplx[k])[0]
    nested = {}
    for sel, rep in (("march", "april"), ("april", "march")):
        E = A[(A[f"npos_ddTotal_est_{sel}"] >= 9) & (A[f"ddS_vsV52_family_{sel}"] > 0)]
        pick = E[f"ddTotal_est_vsV52_family_{sel}"].idxmax()
        nested[f"choose_on_{sel}"] = {"pick": pick,
                                      f"ddTotal_{rep}_of_pick": float(A.loc[pick, f"ddTotal_est_vsV52_family_{rep}"]),
                                      f"ddTotal_{rep}_of_choice": float(A.loc[choice, f"ddTotal_est_vsV52_family_{rep}"])}
    lrows = []
    for (p, s), d in data.items():
        for lam in (0.5, 0.75, 1.25, 1.5):
            v, q = crossfit_one(d, cfgs[choice], lam)
            lrows.append({"panel": p, "split": s, "cfg": f"lam{lam}", **score(d, v, q)})
    L_ = pd.DataFrame(lrows)
    LA = summarise(pd.concat([L_, G[G.cfg.isin(["V52", choice])]]))
    G.to_csv(OUT / "crossfit_grid.csv", index=False)
    A.to_csv(OUT / "crossfit_summary.csv")
    L_.to_csv(OUT / "crossfit_lambda.csv", index=False)
    ch = G[G.cfg == choice].set_index(["split", "panel"])
    b52 = G[G.cfg == "V52"].set_index(["split", "panel"])
    per_cm = {s: {p: {"dS_vsV52": float(ch.loc[(s, p), "dS"] - b52.loc[(s, p), "dS"]),
                      "dphys_est_vsV52": float(ch.loc[(s, p), "dphys_est"] - b52.loc[(s, p), "dphys_est"]),
                      "dTotal_est_vsV52": float(ch.loc[(s, p), "dTotal_est"] - b52.loc[(s, p), "dTotal_est"]),
                      "dS_vsV51": float(ch.loc[(s, p), "dS"]), "dTotal_est_vsV51": float(ch.loc[(s, p), "dTotal_est"])}
                  for p in PANELS} for s in SPLITS}
    reg = {}
    for s in SPLITS:
        for r in (1, 2, 3):
            for k in ("rv", "rq"):
                col = f"R{r}_{k}"
                dd = (ch.loc[s][col] - b52.loc[s][col]).dropna()
                reg[f"{s}|R{r}|{k}"] = {"V52": float(b52.loc[s][col].dropna().mean()), "delta": float(dd.mean()),
                                        "corridors_improved": int((dd < 0).sum()), "corridors": int(len(dd))}
    key_cols = ["ddTotal_est_vsV52_both", "ddTotal_est_vsV52_family_march", "ddTotal_est_vsV52_family_april",
                "ddS_vsV52_family_march", "ddS_vsV52_family_april", "ddphys_est_vsV52_family_march",
                "ddphys_est_vsV52_family_april", "ddphys_raw_vsV52_family_march", "ddphys_raw_vsV52_family_april",
                "npos_ddTotal_est_march", "npos_ddTotal_est_april", "npos_ddS_march", "npos_ddS_april",
                "min_ddTotal_est_march", "min_ddTotal_est_april"]
    out = {"rule": cmd_crossfit.__doc__.strip(), "chosen_name": choice, "chosen": cfgs[choice],
           "max_eligible": best, "within_2e-6": near,
           "dS_family": {s: float(A.loc[choice, f"dS_vsV51_family_{'march' if s == 'validation' else 'april'}"])
                         for s in SPLITS},
           "summary_choice": {k: float(A.loc[choice, k]) for k in key_cols},
           "summary_V52_vsV51": {k: float(A.loc["V52", k]) for k in A.columns if "vsV51" in k},
           "per_corridor_month": per_cm, "per_regime_rmse": reg, "nested_month_check": nested,
           "lambda_curve": LA[key_cols].reset_index().to_dict("records"),
           "inputs_sha256": {f"{p}_{s}": M.sha256(M.HIDE / f"{p}_{s}_cells.parquet") for p in PANELS for s in SPLITS},
           "seconds": round(time.time() - t0)}
    write_new(OUT / "crossfit_choice.json", json.dumps(out, indent=2))
    pd.set_option("display.width", 300)
    print(A[key_cols + ["eligible"]].head(30).to_string(float_format=lambda x: f"{x:.6f}"))
    print(LA[key_cols].to_string(float_format=lambda x: f"{x:.6f}"))
    print(json.dumps({k: out[k] for k in ("chosen_name", "max_eligible", "within_2e-6", "summary_choice",
                                          "nested_month_check")}, indent=2), flush=True)


# ---------------------------------------------------------------------------
# direct check at the real targets (neighbour proxy)
# ---------------------------------------------------------------------------
def target_state(zip_path: Path):
    """{(split, panel): (speed, flow)} of a ZIP's state rows in t1_targets order."""
    K = M.read_key_state()
    Z = M.read_zip_floats(zip_path)
    Z = Z[Z.task == "state"]
    assert np.array_equal(Z.submission_id.to_numpy(), K.submission_id.to_numpy())
    sp, fl = Z.speed_kmh.to_numpy(), Z.flow_vph.to_numpy()
    del Z
    out = {}
    for (split, p), idx in K.groupby(["split", "panel"], observed=True).indices.items():
        tg = t1_targets(p, split)
        kk = K.key.to_numpy()[idx]
        j = np.searchsorted(tg["key"], kk)
        assert len(idx) == len(tg["key"]) and np.array_equal(tg["key"][j], kk), (split, p)
        v, q = np.empty(len(idx)), np.empty(len(idx))
        v[j], q[j] = sp[idx], fl[idx]
        out[split, p] = (v, q)
    return out


def cmd_direct(args) -> None:
    """Same estimator on the real targets (V51 ZIP = production chain) and on the hidden cells (hide run):
    cells whose two temporal neighbours are visible+eligible in that run, predicted free, not blackout;
    proxy truth y = mean of the two neighbour observations; r' = (pv - y) minus its link mean; x = pv - m_l.
    Slope of r' on x and band means of r'.  On the hidden cells the slope with the true observation is also
    reported (the proxy is confounded by neighbour noise in the same way for both sets)."""
    OUT.mkdir(parents=True, exist_ok=True)
    for f in ("direct_cal.csv", "direct_cal_bands.csv"):
        if (OUT / f).exists():
            raise FileExistsError(OUT / f)
    st = target_state(V51_ZIP)
    ed = np.asarray(EDGES["D6"])
    rows, bands = [], []
    for p in PANELS:
        fd = M.static_fd(p)
        n = fd["n"]
        for s in SPLITS:
            C, T, Dd = M.load_hide(p, s)
            pvt = st[s, p][0]
            pf = pq.ParquetFile(CLEAN / p / f"{s}.parquet")
            hid = {d: g for d, g in C.groupby("day")}
            ptr = 0
            acc = {"T": [], "H": []}
            for d in range(pf.num_row_groups):
                t = pf.read_row_group(d, columns=["is_t1_target", "is_score_eligible", "is_t2_blackout",
                                                  "speed_obs", "flow_obs"])
                g_ = lambda c: t.column(c).to_numpy().reshape(STEPS, n)
                tgt, blk, el = g_("is_t1_target"), g_("is_t2_blackout"), g_("is_score_eligible")
                ov, oq = g_("speed_obs").astype(float), g_("flow_obs").astype(float)
                vis = np.isfinite(ov) & np.isfinite(oq) & el
                k = int(tgt.sum())
                P = np.full((STEPS, n), np.nan)
                P[tgt] = pvt[ptr:ptr + k]
                ptr += k
                H = np.zeros((STEPS, n), bool)
                PH = np.full((STEPS, n), np.nan)
                if d in hid:
                    tt, pp_ = hid[d].t.to_numpy(int), hid[d].pos.to_numpy(int)
                    H[tt, pp_] = True
                    PH[tt, pp_] = hid[d].pv.to_numpy(float)
                for nm, sel, pred, visr in (("T", tgt & ~blk, P, vis), ("H", H, PH, vis & ~H)):
                    ok = np.zeros((STEPS, n), bool)
                    ok[1:-1] = sel[1:-1] & visr[:-2] & visr[2:]
                    ok &= pred > FREE_RATIO * fd["vc"][None, :]
                    ti, li = np.nonzero(ok)
                    acc[nm].append(pd.DataFrame({"pos": li, "pv": pred[ti, li],
                                                 "y": 0.5 * (ov[ti - 1, li] + ov[ti + 1, li]),
                                                 "ov": ov[ti, li] if nm == "H" else np.nan}))
            assert ptr == len(pvt)
            rec = {"panel": p, "split": s}
            for nm in ("T", "H"):
                A_ = pd.concat(acc[nm], ignore_index=True)
                pos = A_.pos.to_numpy()
                cnt = np.bincount(pos, minlength=n)
                ml = np.bincount(pos, weights=A_.pv, minlength=n) / np.maximum(cnt, 1)
                e = (A_.pv - A_.y).to_numpy()
                x = A_.pv.to_numpy() - ml[pos]
                r = e - (np.bincount(pos, weights=e, minlength=n) / np.maximum(cnt, 1))[pos]
                rec[f"{nm}_cells"] = len(A_)
                rec[f"{nm}_slope_proxy"] = float((x * r).sum() / (x * x).sum())
                rt = None
                if nm == "H":
                    et = (A_.pv - A_.ov).to_numpy()
                    rt = et - (np.bincount(pos, weights=et, minlength=n) / np.maximum(cnt, 1))[pos]
                    rec["H_slope_true"] = float((x * rt).sum() / (x * x).sum())
                g = np.searchsorted(ed, x)
                for b in range(len(ed) + 1):
                    m = g == b
                    bands.append({"panel": p, "split": s, "set": nm, "band": b, "cells": int(m.sum()),
                                  "mean_r_proxy": float(r[m].mean()) if m.any() else np.nan,
                                  "mean_r_true": float(rt[m].mean()) if (rt is not None and m.any()) else np.nan})
            rec["T_minus_H_slope_proxy"] = rec["T_slope_proxy"] - rec["H_slope_proxy"]
            rows.append(rec)
            print(rec, flush=True)
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "direct_cal.csv", index=False)
    pd.DataFrame(bands).to_csv(OUT / "direct_cal_bands.csv", index=False)
    pd.set_option("display.width", 250)
    print(R.round(3).to_string())
    print(R.groupby("split").mean(numeric_only=True).round(3).T.to_string())
    print("corridor-months with T_slope_proxy > H_slope_proxy:", int((R.T_minus_H_slope_proxy > 0).sum()), "/", len(R))


# ---------------------------------------------------------------------------
# apply
# ---------------------------------------------------------------------------
def g6(x: np.ndarray) -> np.ndarray:
    return np.array([float("%.6g" % v) for v in x])


def target_tod(p: str, split: str) -> np.ndarray:
    """In-day 5-minute index (the hide cells' t) of the Task 1 targets, in t1_targets order, from the parquet layout."""
    pf = pq.ParquetFile(CLEAN / p / f"{split}.parquet")
    n = len(links(p))
    out = []
    for i in range(pf.num_row_groups):
        m = pf.read_row_group(i, columns=["is_t1_target"]).column(0).to_numpy().reshape(STEPS, n)
        out.append(np.nonzero(m)[0])
    return np.concatenate(out)


def cmd_apply(args) -> None:
    import t1_assemble as A
    choice = json.loads((OUT / "crossfit_choice.json").read_text())
    cfg = choice["chosen"]
    for pth in CSV.values():
        if pth.exists():
            raise FileExistsError(pth)
    for f in ("apply_changes.csv", "apply_fd_frame.csv", "apply_summary.json", "apply_params.csv"):
        if (OUT / f).exists():
            raise FileExistsError(OUT / f)
    t0 = time.time()
    K = M.read_key_state()
    Z = M.read_zip_floats(V51_ZIP)
    Z = Z[Z.task == "state"]
    assert np.array_equal(Z.submission_id.to_numpy(), K.submission_id.to_numpy())
    K["v51"], K["q51"] = Z.speed_kmh.to_numpy(), Z.flow_vph.to_numpy()
    Z = M.read_zip_floats(V55_ZIP)
    Z = Z[Z.task == "state"]
    assert np.array_equal(Z.submission_id.to_numpy(), K.submission_id.to_numpy())
    K["v55"], K["q55"] = Z.speed_kmh.to_numpy(), Z.flow_vph.to_numpy()
    del Z
    print("key + V51 + V55 state rows", len(K), f"{time.time() - t0:.0f}s", flush=True)
    rows, fd_rows, prm_rows = [], [], []
    for split in SPLITS:
        S_ = K[K.split == split].reset_index(drop=True)
        v_all, q_all = S_.v55.to_numpy().copy(), S_.q55.to_numpy().copy()
        for p, idx in S_.groupby("panel", observed=True).indices.items():
            fd = M.static_fd(p)
            C, T, Dd = M.load_hide(p, split)
            hp = (C.pos.to_numpy(np.int64), C.pv.to_numpy(float), C.pq.to_numpy(float), C.ov.to_numpy(float),
                  C.oq.to_numpy(float), C.t.to_numpy(np.int64))
            P52 = fit(*hp, fd, V52_CFG)
            P56 = fit(*hp, fd, cfg)
            tg = t1_targets(p, split)
            kk = S_.key.to_numpy()[idx]
            j = np.searchsorted(tg["key"], kk)
            assert len(idx) == len(tg["key"]) and np.array_equal(tg["key"][j], kk), p
            reg = S_.mask_regime.iloc[idx].astype(str).map({r: i for i, r in enumerate(REGIMES)}).to_numpy()
            assert np.array_equal(reg, tg["regime"][j]), p
            gap = A.target_blackout(p, split)[j].astype(bool)
            pos = (kk % KEY_BASE).astype(np.int64)
            tod = ((kk // KEY_BASE) % STEPS).astype(np.int64)
            assert np.array_equal(target_tod(p, split)[j], tod), (split, p, "time-of-day index")
            v51, q51 = S_.v51.to_numpy()[idx], S_.q51.to_numpy()[idx]
            v55, q55 = S_.v55.to_numpy()[idx], S_.q55.to_numpy()[idx]
            # (1) lineage check: V55 state rows == V52 rule (t1_v52_monthbias apply) on the V51 rows
            a, b, c52 = predict(P52, pos, v51, q51, tod, fd, V52_CFG)
            c52 &= ~gap
            r52v, r52q = np.where(c52, g6(a), v51), np.where(c52, g6(b), q51)
            n_lineage_mismatch = int(((r52v != v55) | (r52q != q55)).sum())
            assert n_lineage_mismatch == 0, (split, p, n_lineage_mismatch)
            # (2) the V56 rule on the V51 rows (cells it changes); everything else keeps the V55 value
            a, b, c56 = predict(P56, pos, v51, q51, tod, fd, cfg)
            c56 &= ~gap
            v1, q1 = np.where(c56, g6(a), v55), np.where(c56, g6(b), q55)
            assert np.isfinite(v1).all() and np.isfinite(q1).all() and (v1 > 0).all() and (q1 >= 0).all()
            assert np.array_equal(v1[gap], v55[gap]) and np.array_equal(q1[gap], q55[gap])
            assert np.array_equal(c56, c52), "V56 must touch exactly the V52 cell set"
            v_all[idx], q_all[idx] = v1, q1
            ch = (v1 != v55) | (q1 != q55)
            fr0 = M.fd_frame(p, split, tg, j, v51, q51, v55, q55, fd)
            fr1 = M.fd_frame(p, split, tg, j, v55, q55, v1, q1, fd)
            for r0, r1 in zip(fr0, fr1):
                assert r0["regime"] == r1["regime"]
                fd_rows.append({"split": split, "panel": p, "regime": r0["regime"], "FD_V51": r0["FD0"],
                                "FD_V55": r0["FD1"], "FD_V55_recomputed": r1["FD0"], "FD_V56": r1["FD1"],
                                "low_share_V56": r1["low_share1"], "targets": r1["targets"]})
            Lk = links(p)
            if cfg["speed"]["kind"] == "dev":
                for l in range(fd["n"]):
                    if P56["cnt"][l] > 0:
                        prm_rows.append({"split": split, "panel": p, "pos": l, "link_id": Lk.link_id.iloc[l],
                                         "n_free_hidden": int(P56["cnt"][l]), "b_link_V52": float(P56["bv"][l]),
                                         "m_link": float(P56["mp"][l]),
                                         **{f"band{g}_offset": float(P56["bv"][l] + P56["cg"][g] + P56["dev"][l, g])
                                            for g in range(P56["dev"].shape[1])},
                                         "flow_scale": float(P56.get("r", np.zeros(fd["n"]))[l])})
            rows.append({"split": split, "panel": p, "targets": len(idx), "blackout_kept": int(gap.sum()),
                         "cells_rule_applies": int(c56.sum()), "changed_vs_V55": int(ch.sum()),
                         "changed_vs_V51": int(((v1 != v51) | (q1 != q51)).sum()),
                         "mean_speed_change_vs_V55": float(np.mean((v1 - v55)[ch])) if ch.any() else 0.0,
                         "mean_abs_speed_change_vs_V55": float(np.mean(np.abs(v1 - v55)[ch])) if ch.any() else 0.0,
                         "max_abs_speed_change_vs_V55": float(np.max(np.abs(v1 - v55))),
                         "mean_abs_flow_change_vs_V55": float(np.mean(np.abs(q1 - q55)[ch])) if ch.any() else 0.0,
                         "max_abs_flow_change_vs_V55": float(np.max(np.abs(q1 - q55))),
                         "max_abs_speed_change_vs_V51": float(np.max(np.abs(v1 - v51))),
                         "max_abs_flow_change_vs_V51": float(np.max(np.abs(q1 - q51))),
                         "lineage_mismatch_V55_vs_V52rule": n_lineage_mismatch,
                         "low_flow_share_after": float((q1 < 50).mean()),
                         "speed_min_after": float(v1.min()), "speed_max_after": float(v1.max())})
            print(split, p, rows[-1]["changed_vs_V55"], "changed vs V55", f"{time.time() - t0:.0f}s", flush=True)
        out = S_[["panel", "timestamp", "station_id", "link_id", "mask_regime"]].copy()
        out["speed_kmh"], out["flow_vph"] = v_all, q_all
        out.to_csv(CSV[split], index=False, float_format="%.6g")
        print("wrote", CSV[split], len(out), f"{time.time() - t0:.0f}s", flush=True)
        del S_, out
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "apply_changes.csv", index=False)
    F = pd.DataFrame(fd_rows)
    F.to_csv(OUT / "apply_fd_frame.csv", index=False)
    pd.DataFrame(prm_rows).to_csv(OUT / "apply_params.csv", index=False)
    fdp = F.groupby(["split", "panel"])[["FD_V51", "FD_V55", "FD_V56"]].mean()
    summ = {"config": cfg, "config_name": choice["chosen_name"], "changes": rows,
            "S_FD_real_frame_family_mean": {
                s: {k: M.family_mean(fdp.loc[s][k].to_dict()) for k in ("FD_V51", "FD_V55", "FD_V56")} for s in SPLITS},
            "outputs_sha256": {s: M.sha256(CSV[s]) for s in SPLITS},
            "v51_zip_sha256": M.sha256(V51_ZIP), "v55_zip_sha256": M.sha256(V55_ZIP),
            "seconds": round(time.time() - t0)}
    for s in SPLITS:
        x = summ["S_FD_real_frame_family_mean"][s]
        x["delta_V56_vs_V55"] = x["FD_V56"] - x["FD_V55"]
        x["delta_V56_vs_V51"] = x["FD_V56"] - x["FD_V51"]
    write_new(OUT / "apply_summary.json", json.dumps(summ, indent=2))
    pd.set_option("display.width", 300)
    print(R.round(4).to_string(), flush=True)
    print(json.dumps(summ["S_FD_real_frame_family_mean"], indent=2), flush=True)


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


def cmd_audit(args) -> None:
    _patch()
    M.cmd_audit(args)


def cmd_compare(args) -> None:
    """(Historical audit; v56_t1.zip and v55_nb.zip were deleted on 2026-10-20, the shipped V56 is
    v56_t1_on54.zip = V54 queues on v56t1_state.zip.)
    v56_t1.zip vs v55_nb.zip: queue/ODME lines byte-identical, state lines differ exactly where the CSVs
    differ from V55, every differing state value equals the CSV value."""
    new = WORK / "submissions" / f"{FULL_NAME}.zip"
    path = OUT / "compare_v55.json"
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
    with zipfile.ZipFile(V55_ZIP) as za, zipfile.ZipFile(new) as zb:
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
    st["valid"] = bool(st["rows"] == 6980503 and st["queue_changed"] == 0 and st["odme_changed"] == 0 and
                       st["state_value_mismatch_vs_csv"] == 0 and st["state_text_not_6g"] == 0 and
                       st["state_changed"] > 0)
    st["v55_sha256"], st["v56_t1_sha256"] = M.sha256(V55_ZIP), M.sha256(new)
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
