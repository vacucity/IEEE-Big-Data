#!/usr/bin/env python
"""Gap-0 CatBoost ongoing member: the V41 CatBoost recipe fitted on t2_ml2_g0 rows (labels = the six rows right
after the history), applied to the history ending at the origin row T.  Cached for the pseudo windows in
work/cache/t2_slotT/g0cat_<panel>_<split>.npz under key "catg0".

  python work/t2_cat_g0.py fit
  python work/t2_cat_g0.py cache
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "6"

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import ROOT, T2_PANELS  # noqa: E402
from task2_ml2 import DROP  # noqa: E402

MODEL = ROOT / "work" / "models" / "t2_cat_ongoing_g0.cbm"


def fit():
    from catboost import CatBoostClassifier
    import t2_ml2_g0 as G
    d = pd.concat([pd.read_parquet(G.DATA / f"{p}.parquet") for p in T2_PANELS], ignore_index=True)
    feats = [c for c in d.columns if c not in DROP]
    model = CatBoostClassifier(iterations=400, depth=7, learning_rate=.05, l2_leaf_reg=12, loss_function="Logloss",
                               random_seed=20261017, thread_count=6, verbose=100, allow_writing_files=False)
    model.fit(d[feats].fillna(-999).astype(np.float32), d.y, sample_weight=d.w.astype(np.float32))
    model.save_model(str(MODEL))
    print("saved", MODEL, len(d), flush=True)


def g0_features(ws):
    """Per window: (features dict, near-link index) on history [T-55, T], time T+1, prior at T+6."""
    from task2_ml import RADIUS_KM, ctx_of
    from task2_ml2 import features2
    from task2_rules import DEFAULTS as RULES, onset_scores
    import t2_grids
    g = t2_grids.load(ws[0].panel, ws[0].split)
    ctx = ctx_of(ws[0].panel)
    pno = T2_PANELS.index(ws[0].panel)
    out = []
    for w in ws:
        s = slice(w.T - 11, w.T + 1)
        V, Q, O = (g[k][w.day, s].astype(float) for k in ("V", "Q", "O"))
        prior = onset_scores(ctx.ev, ctx.n, w.T + 6, w.weekday, None, RULES["sigma_h"], RULES["weekend"])
        f = features2(ctx, V, Q, O, w.T + 1, int(w.weekday >= 5), prior, pno)
        with np.errstate(invalid="ignore"):
            hq = (V <= ctx.vcut).any(0)
        near = np.zeros(ctx.n, bool)
        for xq in ctx.x[hq]:
            near |= np.abs(ctx.x - xq) <= RADIUS_KM
        out.append((f, np.flatnonzero(near)))
    return ctx.n, out


def cat_g0_probs(ws, model=None):
    from catboost import CatBoostClassifier
    from task2_ml import to_frame
    if model is None:
        model = CatBoostClassifier()
        model.load_model(str(MODEL))
    feats = model.feature_names_
    n, fe = g0_features(ws)
    out = np.zeros((len(ws), 6, n), np.float32)
    frames, pos = [], []
    for i, (f, idx) in enumerate(fe):
        if len(idx):
            frames.append(pd.concat([to_frame(f, idx, h + 1)[feats] for h in range(6)], ignore_index=True))
            pos.append((i, idx))
    if frames:
        p = model.predict_proba(pd.concat(frames, ignore_index=True).fillna(-999).astype(np.float32),
                                thread_count=6)[:, 1]
        off = 0
        for i, idx in pos:
            out[i] = 0
            out[i][:, idx] = p[off:off + 6 * len(idx)].reshape(6, len(idx))
            off += 6 * len(idx)
    return out


def cache():
    import t2_pseudo as P
    from t2_slotT import CACHE
    from catboost import CatBoostClassifier
    model = CatBoostClassifier()
    model.load_model(str(MODEL))
    for split in ("validation", "private"):
        for panel in T2_PANELS:
            path = CACHE / f"g0cat_{panel}_{split}.npz"
            old = dict(np.load(path)) if path.exists() else {}
            if "catg0" in old:
                continue
            ws = [w for w in P.windows(panel, split) if w.cond == "ongoing"]
            old["catg0"] = cat_g0_probs(ws, model)
            np.savez_compressed(path, **old)
            print(split, panel, flush=True)


if __name__ == "__main__":
    np.seterr(invalid="ignore", divide="ignore")
    {"fit": fit, "cache": cache}[sys.argv[1]]()
