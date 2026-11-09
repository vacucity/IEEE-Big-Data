"""V64 probe (lane C, round 2): the V61_cg congested-cell density model fitted on BOTH months' hide cells.

V61_cg (t1_v61_congest F600a75s) fits one model per month on that month's congested hide cells only; its estimate
(+0.0006, ~90% LWR proxy) realised in full on the public LB.  Task 1 is offline, so each month may also learn from
the other month's hide cells (V57 showed cross-month models work for the restack).  Variants, all with the V61 odd/
even-day cross-fit (each cell predicted by a model that never saw its own day):
  month     V61 as built (must reproduce +5.95e-4 / +7.09e-4)
  pooled    both months in one model, month flag as a feature
  pooled31  pooled, 31 leaves, 1200 rounds (density) / 600 (split)
Scored per month with the V61 frame (reference V60 hide frame).

  python work/t1_v64_pooled.py
"""
from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "4")

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import t1_v52_monthbias as M  # noqa: E402
import t1_v61_congest as C  # noqa: E402
from evaluate import PANELS  # noqa: E402

OUT = Path(__file__).resolve().parent / "results" / "t1_v64_pooled"
CFG = C.CFGS["F600a75s"]


def with_month(P, m):
    X = P["X"]
    return np.column_stack([X[:, :-1], np.full(len(X), m, np.float32), X[:, -1:]])


def cross(Xs, ys, ws, pars, prm, rounds):
    """Cross-fit on the union of months by day parity; returns per-month predictions."""
    X, y, w, par = (np.concatenate(a) for a in (Xs, ys, ws, pars))
    pred = np.zeros(len(y))
    for a in (0, 1):
        tr = par == a
        m = C.train(X[tr], y[tr], w[tr], prm, rounds)
        pred[~tr] = m.predict(X[~tr], num_threads=C.THREADS)
    out, off = [], 0
    for yy in ys:
        out.append(pred[off:off + len(yy)])
        off += len(yy)
    return out


def score_month(data, rh, sh, label, s, rows):
    off, recs = 0, []
    for p in PANELS:
        d = data[p]
        n = int(d["cong"].sum())
        v, q = C.frame_values(d, rh[off:off + n], sh[off:off + n], CFG)
        off += n
        rec = C.score(d, v, q)
        recs.append({"split": s, "cfg": label, "panel": p, **{k: rec[k] for k in ("dS", "dE_est", "dTotal_est")}})
    g = pd.DataFrame(recs).set_index("panel")
    print(s, label, {k: f"{M.family_mean(g[k].to_dict()) * 1e4:+.2f}e-4" for k in ("dS", "dE_est", "dTotal_est")},
          "corridors dTotal>0:", int((g.dTotal_est > 0).sum()), flush=True)
    rows.extend(recs)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    data = {s: C.load_month(s) for s in C.SPLITS}
    P = {s: C.pooled(data[s], CFG["feats"]) for s in C.SPLITS}
    print("loaded", {s: len(P[s]["r"]) for s in C.SPLITS}, f"{time.time() - t0:.0f}s", flush=True)
    rows = []
    for s in C.SPLITS:                                       # V61 as built
        rh = C.crossfit_density(P[s], CFG["rounds"])
        sh = C.crossfit_split(P[s], CFG["split"]["rounds"])
        score_month(data[s], rh, sh, "month", s, rows)
    Xs = [with_month(P[s], i) for i, s in enumerate(C.SPLITS)]
    pars = [P[s]["par"] for s in C.SPLITS]
    for label, prm, prm_s, rd, rs in (("pooled", C.PRM, C.PRM_S, CFG["rounds"], CFG["split"]["rounds"]),
                                      ("pooled31", dict(C.PRM, num_leaves=31), dict(C.PRM_S, num_leaves=31), 1200, 600)):
        rh = cross(Xs, [P[s]["r"] for s in C.SPLITS], [P[s]["wd"] for s in C.SPLITS], pars, prm, rd)
        sh = cross(Xs, [P[s]["sstar"] for s in C.SPLITS], [P[s]["ws"] for s in C.SPLITS], pars, prm_s, rs)
        for i, s in enumerate(C.SPLITS):
            score_month(data[s], rh[i], sh[i], label, s, rows)
        print(label, f"{time.time() - t0:.0f}s", flush=True)
    pd.DataFrame(rows).to_csv(OUT / "probe.csv", index=False)


if __name__ == "__main__":
    main()
