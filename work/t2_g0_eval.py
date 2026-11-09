#!/usr/bin/env python
"""Compare slot-T ongoing variants on the pseudo windows with the V45 decision chain.

z variants are built from the V27 cache (unshifted), the shifted cache (t2_slotT hybrid) and the gap-0 caches
(work/cache/t2_slotT/g0_*.npz, g0cat_*.npz).  Selection uses validation only (rule: April choices may use
March, never April); private-visible is printed for information.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import t2_ongoing_v27 as V27  # noqa: E402
import t2_slotT as S  # noqa: E402
from evaluate import T2_PANELS  # noqa: E402
from t2_group_v27 import groups, smooth  # noqa: E402
from t2_v33_adaptive_iou import adaptive_iou  # noqa: E402


def load_all(split, panel):
    base = dict(np.load(V27.cache_path(panel, split)))
    hy = S.hybrid(base, dict(np.load(S.cache_path(panel, split))))
    g0 = {}
    for f in (S.CACHE / f"g0_{panel}_{split}.npz", S.CACHE / f"g0cat_{panel}_{split}.npz"):
        if f.exists():
            g0.update(dict(np.load(f)))
    cat_old = np.load(V27.ROOT / "work" / "results" / "t2_v41_ongoing_cat" / f"{panel}_{split}.npz")["cat"]
    return base, hy, g0, cat_old


def decide(z, parts, weights, panel, cat=None, cat_alpha=0.0, theta=None):
    pr = sum(w * p for w, p in zip(weights, parts)) / sum(weights)
    g = groups(panel, .95)
    pr = smooth(pr, g, .25)
    if cat is not None and cat_alpha:
        pr = (1 - cat_alpha) * pr + cat_alpha * smooth(cat, g, .25)
    if theta is not None:
        return pr >= theta
    return np.stack([adaptive_iou(x) for x in pr])


def run(variants):
    rows = []
    for split in ("validation", "private"):
        for panel in T2_PANELS:
            base, hy, g0, cat_old = load_all(split, panel)
            for name, fn in variants.items():
                out = fn(base, hy, g0, cat_old, panel)
                if out is None:
                    continue
                rows.append(dict(split=split, panel=panel, model=name, iou=V27.scores(base, out).mean()))
    df = pd.DataFrame(rows)
    t = df.pivot_table(index="model", columns="split", values="iou")
    return df, t
