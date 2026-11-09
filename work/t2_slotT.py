#!/usr/bin/env python
"""Task 2 ongoing with the observation row at the forecast origin T (organizer, discussion 745700, 2026-10-15:
"timestamps at or before T, including the observations at T ... the row at T itself is released data and may
be used").  Every model so far reads the history [T-60, T-5] and skips the row at T.

Zero-retraining test: run the existing ongoing models on the history shifted by one step, [T-55, T].  Those
models forecast steps 2..7 after the last history row, i.e. T+10..T+35, so the hybrid forecast takes T+5 from
the unshifted models and T+10..T+30 from the shifted ones.

Usage (from the "IEEE Big data" folder):
  python work/t2_slotT.py cache [--splits validation private]
  python work/t2_slotT.py eval
"""
from __future__ import annotations

import argparse
import dataclasses
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import t2_cnn as C  # noqa: E402
import t2_ongoing_v27 as V27  # noqa: E402
import t2_pseudo as P  # noqa: E402
from evaluate import ROOT, T2_PANELS  # noqa: E402

CACHE = ROOT / "work" / "cache" / "t2_slotT"
KEYS = ("v8", "ml2", "cnn0", "cnn1", "cnn2", "rnn0", "rnn1")


def cache_path(panel, split):
    return CACHE / f"{panel}_{split}.npz"


def cmd_cache(args):
    CACHE.mkdir(parents=True, exist_ok=True)
    for split in args.splits:
        for panel in T2_PANELS:
            path = cache_path(panel, split)
            if path.exists():
                continue
            t0 = time.time()
            ws = [w for w in P.windows(panel, split) if w.cond == "ongoing"]
            base = dict(np.load(V27.cache_path(panel, split)))
            assert list(base["T"]) == [w.T for w in ws] and list(base["day"]) == [w.day for w in ws]
            shifted = [dataclasses.replace(w, T=w.T + 1) for w in ws]     # history [T-55, T]
            D = C.PanelData(panel, split, unmasked=False)
            neural = V27.neural_probs(D, shifted)
            gb = V27.gbm_probs(shifted)
            np.savez_compressed(path, **neural, v8=gb[0], ml2=gb[1])
            print(split, panel, len(ws), f"{time.time() - t0:.0f}s", flush=True)


def hybrid(base: dict, sh: dict) -> dict:
    """T+5 from the unshifted models, T+10..T+30 from the shifted ones (their steps 1..5)."""
    z = dict(base)
    for k in KEYS:
        z[k] = np.concatenate([base[k][:, :1], sh[k][:, :5]], 1)
    return z


def cmd_eval(args):
    cfg = dict(weights=[1, 1, 1, 5], theta=.425)
    rows = []
    for split in ("validation", "private"):
        for panel in T2_PANELS:
            if not cache_path(panel, split).exists():
                continue
            base = dict(np.load(V27.cache_path(panel, split)))
            sh = dict(np.load(cache_path(panel, split)))
            hy = hybrid(base, sh)
            for name, z in (("base", base), ("hybrid", hy)):
                for wts in ([1, 1, 1, 5], [1, 1, 2, 4], [0, 0, 0, 1], [0, 0, 1, 0], [1, 1, 0, 0]):
                    for th in (.4, .425, .45, .5):
                        pred = V27.decision(V27.probs(z, wts), dict(theta=th), z)
                        iou = V27.scores(z, pred)
                        rows.append(dict(split=split, panel=panel, model=name, w=str(wts), th=th, iou=iou.mean()))
                # per-horizon IoU-like hit rate with the v27 recipe
                pred = V27.decision(V27.probs(z, cfg["weights"]), cfg, z)
                y = z["y"] & z["mask"]
                p = pred & z["mask"]
                for h in range(6):
                    u = (y[:, h] | p[:, h]).sum()
                    rows.append(dict(split=split, panel=panel, model=name, w="h", th=h + 1,
                                     iou=float((y[:, h] & p[:, h]).sum() / max(u, 1))))
    df = pd.DataFrame(rows)
    out = df.groupby(["split", "model", "w", "th"]).iou.mean().unstack([0, 1]).round(4)
    print(out.to_string())
    print(df[(df.w == "[1, 1, 1, 5]") & (df.th == .425)].pivot_table(index="panel", columns=["split", "model"],
                                                                        values="iou").round(4).to_string())


def g0_probs(D, ws, specs):
    """Gap-0 models (trained on history [t-12, t) -> t..t+5): input = history ending at the origin row T,
    output = T+5..T+30 directly.  specs: [(name, kind, tag)], kind in {"cnn", "rnn"}."""
    import t2_rnn as R
    models = [(nm, C.load_model(tag, 96) if kind == "cnn" else R.load_model(tag), kind) for nm, kind, tag in specs]
    out = {nm: np.zeros((len(ws), 6, D.n), np.float32) for nm, _, _ in models}
    for start in range(0, len(ws), 32):
        xs = [C.make_x(D.V[w.day, w.T - 11:w.T + 1], D.Q[w.day, w.T - 11:w.T + 1], D.O[w.day, w.T - 11:w.T + 1],
                       D.static_ch, D.st["ffs"], D.st["cap"], D.vcut, w.T + 1, w.weekday)
              for w in ws[start:start + 32]]
        for nm, model, kind in models:
            pr = C.predict_probs(model, xs) if kind == "cnn" else R.predict_probs(model, xs)
            out[nm][start:start + len(xs), :, D.perm] = pr
    return out


def ml2_g0_probs(ws, model_file=None):
    """Gap-0 ml2 GBM (t2_ml2_g0.py): history [T-55, T] (rows w.T-11..w.T), time feature T+1, prior at T+6."""
    import lightgbm as lgb
    from task2_ml import RADIUS_KM, ctx_of, to_frame
    from task2_ml2 import features2
    from task2_rules import DEFAULTS as RULES, onset_scores
    b = lgb.Booster(model_file=str(model_file or ROOT / "work" / "models" / "t2_ml2_ongoing_g0.txt"))
    g = __import__("t2_grids").load(ws[0].panel, ws[0].split)
    ctx = ctx_of(ws[0].panel)
    pno = T2_PANELS.index(ws[0].panel)
    out = np.zeros((len(ws), 6, ctx.n), np.float32)
    frames, positions = [], []
    for wi, w in enumerate(ws):
        s = slice(w.T - 11, w.T + 1)
        V, Q, O = (g[k][w.day, s].astype(float) for k in ("V", "Q", "O"))
        prior = onset_scores(ctx.ev, ctx.n, w.T + 6, w.weekday, None, RULES["sigma_h"], RULES["weekend"])
        f = features2(ctx, V, Q, O, w.T + 1, int(w.weekday >= 5), prior, pno)
        with np.errstate(invalid="ignore"):
            hq = (V <= ctx.vcut).any(0)
        near = np.zeros(ctx.n, bool)
        for xq in ctx.x[hq]:
            near |= np.abs(ctx.x - xq) <= RADIUS_KM
        idx = np.flatnonzero(near)
        for h in range(6):
            positions.append((wi, h, idx))
            frames.append(to_frame(f, idx, h + 1)[b.feature_name()])
    pp = b.predict(pd.concat(frames, ignore_index=True), num_threads=4)
    off = 0
    for wi, h, idx in positions:
        out[wi, h, idx] = pp[off:off + len(idx)]
        off += len(idx)
    return out


def cmd_cache_g0(args):
    specs = [tuple(s.split(":")) for s in args.specs]          # name:kind:tag
    for split in ("validation", "private"):
        for panel in T2_PANELS:
            path = CACHE / f"g0_{panel}_{split}.npz"
            old = dict(np.load(path)) if path.exists() else {}
            todo = [s for s in specs if s[0] not in old]
            if not todo:
                continue
            ws = [w for w in P.windows(panel, split) if w.cond == "ongoing"]
            nn = [s for s in todo if s[1] in ("cnn", "rnn")]
            if nn:
                D = C.PanelData(panel, split, unmasked=False)
                old.update(g0_probs(D, ws, nn))
            for s in todo:
                if s[1] == "gbm":
                    old[s[0]] = ml2_g0_probs(ws)
            np.savez_compressed(path, **old)
            print(split, panel, [s[0] for s in todo], flush=True)


CAT_ALPHA = {"D7_I10_E": .3, "D7_I210_W": .1, "D7_I405_S": .2}     # V41/V42/V44/V45 CatBoost blends


def v44_decision(z, panel, split, cat_alpha=CAT_ALPHA, weights=(1, 1, 1, 5)):
    """V44/V45 ongoing chain on a pseudo cache: [v8, ml2, CNN3, RNN2] blend, train-only co-queue group
    smoothing (Jaccard .95, alpha .25), CatBoost blend on three corridors, window-wise expected-IoU decision."""
    from t2_group_v27 import groups, smooth
    from t2_v33_adaptive_iou import adaptive_iou
    g = groups(panel, .95)
    pr = smooth(V27.probs(z, list(weights)), g, .25)
    a = cat_alpha.get(panel, 0)
    if a:
        cat = np.load(ROOT / "work" / "results" / "t2_v41_ongoing_cat" / f"{panel}_{split}.npz")["cat"]
        pr = (1 - a) * pr + a * smooth(cat, g, .25)
    return np.stack([adaptive_iou(x) for x in pr])


def cmd_eval44(args):
    rows = []
    for split in ("validation", "private"):
        for panel in T2_PANELS:
            if not cache_path(panel, split).exists():
                continue
            base = dict(np.load(V27.cache_path(panel, split)))
            hy = hybrid(base, dict(np.load(cache_path(panel, split))))
            for name, z, ca in (("v44", base, CAT_ALPHA), ("v44_nocat", base, {}),
                                ("hyb", hy, CAT_ALPHA), ("hyb_nocat", hy, {})):
                iou = V27.scores(z, v44_decision(z, panel, split, ca))
                rows.append(dict(split=split, panel=panel, model=name, iou=iou.mean()))
    df = pd.DataFrame(rows)
    t = df.pivot_table(index="panel", columns=["split", "model"], values="iou")
    print(t.round(4).to_string())
    print(t.mean().round(4).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("cache")
    s.add_argument("--splits", nargs="+", default=["validation", "private"])
    sub.add_parser("eval")
    sub.add_parser("eval44")
    s = sub.add_parser("cache_g0")
    s.add_argument("--specs", nargs="+", required=True)
    args = ap.parse_args()
    {"cache": cmd_cache, "eval": cmd_eval, "eval44": cmd_eval44, "cache_g0": cmd_cache_g0}[args.cmd](args)
