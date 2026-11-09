#!/usr/bin/env python
"""Task 2 experiments on pseudo windows (see t2_pseudo.py): baselines and candidate decision rules."""
from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import ROOT, T2_PANELS  # noqa: E402
from t2_pseudo import HIST, HOR, Win, evaluate, history, summary, vcut  # noqa: E402
from task2_ml import RADIUS_KM, ctx_of, to_frame  # noqa: E402
from task2_ml2 import features2  # noqa: E402
from task2_rules import DEFAULTS as RULES, onset_scores  # noqa: E402

MODELS = ROOT / "work" / "models"


@lru_cache(maxsize=None)
def booster(name: str) -> lgb.Booster:
    return lgb.Booster(model_file=str(MODELS / name))


def ml2_feats(w: Win):
    ctx = ctx_of(w.panel)
    V, Q, O = (a.astype(float) for a in history(w))
    prior = onset_scores(ctx.ev, ctx.n, w.T + HOR, w.weekday, None, RULES["sigma_h"], RULES["weekend"])
    return ctx, V, features2(ctx, V, Q, O, w.T, int(w.weekday >= 5), prior, T2_PANELS.index(w.panel))


_cache: dict = {}


def ml2_prob(w: Win) -> np.ndarray:
    key = (w.panel, w.split, w.day, w.T)
    if key in _cache:
        return _cache[key]
    ctx, V, f = ml2_feats(w)
    prob = np.zeros((HOR, ctx.n))
    if w.cond == "onset":
        b = booster("t2_ml2_onset_v2.txt")
        prob[HOR - 1] = b.predict(to_frame(f, np.arange(ctx.n), None)[b.feature_name()])
    else:
        b = booster("t2_ml2_ongoing_v2.txt")
        with np.errstate(invalid="ignore"):
            hq = (V <= ctx.vcut).any(0)
        near = np.zeros(ctx.n, bool)
        for xq in ctx.x[hq]:
            near |= np.abs(ctx.x - xq) <= RADIUS_KM
        idx = np.flatnonzero(near)
        for s in range(HOR if len(idx) else 0):
            prob[s, idx] = b.predict(to_frame(f, idx, s + 1)[b.feature_name()])
    _cache[key] = prob
    return prob


def persistence(w: Win) -> np.ndarray:
    V = history(w)[0]
    n = V.shape[1]
    if w.cond == "onset":
        return np.zeros((HOR, n))
    with np.errstate(invalid="ignore"):
        q = V <= vcut(w.panel)
    seen = np.isfinite(V)
    last = np.zeros(n, bool)
    for t in range(HIST - 3, HIST):           # last visible state within the final 15 minutes
        last = np.where(seen[t], q[t], last)
    return np.repeat(last[None], HOR, 0)


if __name__ == "__main__":
    th = float(sys.argv[1]) if len(sys.argv) > 1 else 0.4
    print("persistence (ongoing)")
    print(summary(evaluate(persistence, conds=("ongoing",))).to_string())
    for t in (0.3, 0.4, 0.5):
        print(f"ml2 theta {t}")
        print(summary(evaluate(lambda w: ml2_prob(w) >= t)).to_string())


# ---------------------------------------------------------------------------
# onset decision rules on top of per-link probabilities
# ---------------------------------------------------------------------------
def components(p: np.ndarray, x: np.ndarray, lo: float, gap_km: float = 1.0) -> list[np.ndarray]:
    """Groups of links with p >= lo that sit within gap_km of each other along travel."""
    idx = np.flatnonzero(p >= lo)
    if not len(idx):
        return []
    idx = idx[np.argsort(x[idx])]
    out, cur = [], [idx[0]]
    for a, b in zip(idx[:-1], idx[1:]):
        if x[b] - x[a] > gap_km:
            out.append(np.array(cur))
            cur = []
        cur.append(b)
    out.append(np.array(cur))
    return out


def onset_block(p: np.ndarray, panel: str, lo: float = 0.1, keep: float = 0.5, second: float = 1.1) -> np.ndarray:
    """Predict the single most likely block: its links with p >= keep * block max.
    A second block is added only when its max reaches `second` x the first (never with the default)."""
    x = ctx_of(panel).x
    comps = components(p, x, lo)
    out = np.zeros_like(p, bool)
    if not comps:
        out[np.argmax(p)] = True
        return out
    best = sorted(comps, key=lambda c: -p[c].max())
    for k, c in enumerate(best[:2]):
        if k == 1 and p[c].max() < second * p[best[0]].max():
            break
        out[c[p[c] >= keep * p[c].max()]] = True
    return out


def onset_rule(rule):
    def f(w: Win) -> np.ndarray:
        pr = ml2_prob(w)
        out = np.zeros_like(pr, bool)
        out[HOR - 1] = rule(pr[HOR - 1], w.panel)
        return out
    return f


def run_onset_rules():
    rules = {
        "theta0.4": lambda p, panel: p >= 0.4,
        "theta0.4_nonempty": lambda p, panel: (p >= 0.4) if (p >= 0.4).any() else (p >= 0.5 * p.max()),
        "block_keep0.5": lambda p, panel: onset_block(p, panel, keep=0.5),
        "block_keep0.3": lambda p, panel: onset_block(p, panel, keep=0.3),
        "block_keep0.7": lambda p, panel: onset_block(p, panel, keep=0.7),
    }
    for name, r in rules.items():
        df = evaluate(onset_rule(r), conds=("onset",))
        s = summary(df)
        early = summary(df[df.day < 12])
        print(f"{name:22s} all: {s.onset.to_dict()}  early days: {early.onset.to_dict()}")


def oracle_block(w: Win) -> np.ndarray:
    """Upper bound: the candidate component (p >= 0.02) that scores best against the truth."""
    from t2_pseudo import iou
    pr = ml2_prob(w)[HOR - 1]
    best, out = -1.0, np.zeros((HOR, len(pr)), bool)
    for c in components(pr, ctx_of(w.panel).x, 0.02):
        for keep in (0.0, 0.3, 0.5):
            cand = np.zeros((HOR, len(pr)), bool)
            cand[HOR - 1, c[pr[c] >= keep * pr[c].max()]] = True
            s = iou(cand, w)
            if s > best:
                best, out = s, cand
    return out
