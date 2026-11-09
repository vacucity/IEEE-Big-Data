"""V68 probe: nonlinear second stage on top of the V66 common-mode correction (free non-blackout speed).

V66 (public +0.00050) is an OLS on a few cross-sectional means of visible-cell deviations.  Here a LightGBM fitted on
the V66 hide-frame residual (ov - v66) with the cached V66 features (D1 t-2..t+2, Dw t-1..t+1, 18 band means, cnt),
the local context of t1_v59_local (db, da, n3, nsp), the regime, the production speed level (v64 / ffs, v64 / v_cut)
and the corridor id; pooled over corridors per month, odd/even-day cross-fit, shrink lambda in {0.5, 1}.
Prints dS, dE_est, dTotal40 per month vs the V66 hide frame.  Research probe.
"""
from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "6")

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import t1_v52_monthbias as M  # noqa: E402
import t1_v59_local as LOC  # noqa: E402
import t1_v66_cmode as CM  # noqa: E402
from evaluate import PANELS  # noqa: E402

OUT = Path(__file__).resolve().parent / "results" / "t1_v68_cmgbm"
PRM = dict(objective="regression", learning_rate=0.03, num_leaves=31, min_data_in_leaf=400, lambda_l2=10.0,
           feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, num_threads=6, verbose=-1, seed=68,
           deterministic=True, force_col_wise=True)
ROUNDS = 600


def feats(p, d, Lh, F, v64, pi):
    fd = d["fd"]
    pos = d["pos"]
    X = np.column_stack([F["D1"], F["Dw"], F["band"], F["cnt"], Lh["db"], Lh["da"], Lh["n3"], Lh["nsp"], d["reg"],
                         v64 / fd["ffs"][pos], v64 / fd["vc"][pos], np.full(len(pos), pi)])
    return X.astype(np.float32)


def main():
    import lightgbm as lgb
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    z64 = np.load(CM.Z64)
    gr = np.load(CM.OUT / "grams.npz")
    rows = []
    for mi, s in enumerate(CM.SPLITS):
        B = {a: CM.solve(gr[f"G_{s}_{a}_0"] + gr[f"G_{s}_{a}_1"]) for a in (0, 1)}
        off = 0
        per = {}
        for pi, p in enumerate(PANELS):
            d, Lh = LOC.load(p, s)
            v64, q64, off = CM.hide_v64(p, s, mi, d, off, z64)
            F = CM.hide_feats(p, s, d)
            m = d["free"] & (F["cnt"] >= CM.MIN_CNT)
            v66 = v64.copy()
            for a in (0, 1):
                te = np.flatnonzero((d["par"] != a) & m)
                v66[te] = np.clip(v64[te] + CM.cm_pred(F, Lh, te, *B[a]), 5.0, 120.75)
            per[p] = dict(d=d, v66=v66, q64=q64, m=m, X=feats(p, d, Lh, F, v64, pi))
        print(s, "loaded", f"{time.time() - t0:.0f}s", flush=True)
        preds = {p: np.zeros(len(x["v66"])) for p, x in per.items()}
        for a in (0, 1):
            Xs = np.concatenate([x["X"][(x["d"]["par"] == a) & x["m"]] for x in per.values()])
            ys = np.concatenate([(x["d"]["ov"] - x["v66"])[(x["d"]["par"] == a) & x["m"]] for x in per.values()])
            ok = np.isfinite(ys)
            mdl = lgb.train(PRM, lgb.Dataset(Xs[ok], ys[ok], categorical_feature=[Xs.shape[1] - 1]), ROUNDS)
            for p, x in per.items():
                k = (x["d"]["par"] != a) & x["m"]
                preds[p][k] = mdl.predict(x["X"][k], num_threads=6)
            print(s, "parity", a, "fitted", f"{time.time() - t0:.0f}s", flush=True)
        for lam in (0.5, 1.0):
            for p, x in per.items():
                d = x["d"]
                v1 = np.where(x["m"], np.clip(x["v66"] + lam * preds[p], 5.0, 120.75), x["v66"])
                s0 = M.state_score(d["C"], x["v66"], x["q64"], d["fd"])
                s1 = M.state_score(d["C"], v1, x["q64"], d["fd"])
                a0 = M.t3_score(d["C"], d["T"], d["Dd"], x["v66"], x["q64"], d["fd"])
                a1 = M.t3_score(d["C"], d["T"], d["Dd"], v1, x["q64"], d["fd"])
                t3 = M.t3_delta(a0, a1, d["sc"])
                dS = s1["S"] - s0["S"]
                rows.append(dict(split=s, lam=lam, panel=p, dS=dS, dE_est=t3["dE_est"],
                                 dTotal40=0.35 * dS + 0.15 * (t3["dFD_est"] - 0.8 * t3["dE_est"]) / 3))
            g = pd.DataFrame([r for r in rows if r["split"] == s and r["lam"] == lam]).set_index("panel")
            fam = {k: M.family_mean(g[k].to_dict()) * 1e4 for k in ("dS", "dE_est", "dTotal40")}
            print(f"{s:10s} lam {lam} dS {fam['dS']:+.2f}e-4 dE_est {fam['dE_est']:+.2f}e-4 dTotal40 "
                  f"{fam['dTotal40']:+.2f}e-4 corridors dS>0 {(g.dS > 0).sum()}/10", flush=True)
        del per
    pd.DataFrame(rows).to_csv(OUT / "probe.csv", index=False)


if __name__ == "__main__":
    main()
