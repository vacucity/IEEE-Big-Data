#!/usr/bin/env python
"""Onset block choice by distance to capacity: score_b = beta * (log q_b(T) - log c_b) + gamma * trend_b, with
c_b the median flow at T over the train onsets of block b and one beta shared by all blocks (fewer free
parameters than t2_onset's per-block slopes, so rare blocks such as D12_I5_S's 89 are not left to noise)."""
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent))
import t2_onset as O  # noqa: E402
from evaluate import T2_PANELS, links  # noqa: E402
from t2_pseudo import HOR, evaluate, history, summary, windows  # noqa: E402


@lru_cache(maxsize=None)
def train_rows(panel):
    X, T = [], []
    for w in windows(panel, "train"):
        if w.cond != "onset":
            continue
        t = O.target(panel, w)
        if t is None:
            continue
        X.append(O.features(panel, history(w)[1].astype(float), w.T + HOR, w.date[:7]))
        T.append(t)
    return np.stack(X), np.stack(T)


@lru_cache(maxsize=None)
def fit(panel, l2=0.1):
    X, T = train_rows(panel)
    nb = X.shape[1]
    # c_b: median log flow at T over the windows where b is (one of) the onset block(s)
    c = np.array([np.median(X[T[:, b] > 0, b, 0]) for b in range(nb)])
    D = X[:, :, 0] - c[None]
    G = X[:, :, 1]

    def nll(th):
        s = th[0] * D + th[1] * G + np.concatenate([[0.0], th[2:]])[None]
        s = s - s.max(1, keepdims=True)
        lse = np.log(np.exp(s).sum(1))
        return -(T * (s - lse[:, None])).sum() + l2 * (th[2:] ** 2).sum() * len(X) / 100
    th = minimize(nll, np.r_[10.0, 0.0, np.zeros(nb - 1)], method="L-BFGS-B").x
    return c, th


def predict(w):
    bl = O.blocks(w.panel)
    out = np.zeros((HOR, len(links(w.panel))), bool)
    if len(bl) == 1:
        out[HOR - 1, bl[0][1]] = True
        return out
    c, th = fit(w.panel)
    X = O.features(w.panel, history(w)[1].astype(float), w.T + HOR)
    s = th[0] * (X[:, 0] - c) + th[1] * X[:, 1] + np.r_[0.0, th[2:]]
    out[HOR - 1, bl[int(np.argmax(s))][1]] = True
    return out


if __name__ == "__main__":
    multi = [p for p in T2_PANELS if len(O.blocks(p)) > 1]
    for p in multi:
        c, th = fit(p)
        print(p, "c", np.round(np.exp(c)), "theta", np.round(th, 2))
    df = evaluate(predict, panels=multi, conds=("onset",))
    print("capacity-distance:", summary(df).onset.to_dict())
    print(df.groupby(["split", "panel"]).iou.mean().unstack(0).round(3).T.to_string())
    df = evaluate(lambda w: O.predict(w, O.FLOW_USE), panels=multi, conds=("onset",))
    print("current flow logit:", summary(df).onset.to_dict())
    print(df.groupby(["split", "panel"]).iou.mean().unstack(0).round(3).T.to_string())


CAP_PANELS = ("D12_I5_N", "D12_I5_S", "D7_I405_S")


def cmd_apply(base: str, out: str, split: str, panels=CAP_PANELS) -> None:
    """Replace the onset windows of `panels` in a queue file with the capacity-distance block."""
    import pandas as pd
    from evaluate import RELEASE, slot_of
    q = pd.read_csv(base)
    changed = 0
    for p in panels:
        bl = O.blocks(p)
        c, th = fit(p)
        pos = dict(zip(links(p).link_id, range(len(links(p)))))
        win = pd.read_csv(RELEASE / "task2" / p / split / "window_index.csv").set_index("window_id")
        for wid, (Q, tod) in O.official_history_Q(p, split).items():
            X = O.features(p, Q, tod + HOR)
            s = th[0] * (X[:, 0] - c) + th[1] * X[:, 1] + np.r_[0.0, th[2:]]
            lk = set(bl[int(np.argmax(s))][1].tolist())
            T = int(slot_of(pd.Series([win.loc[wid, "forecast_origin"]]))[0])
            m = (q.window_id == wid).to_numpy()
            rows = q[m]
            new = ((slot_of(rows.timestamp) == T + HOR) & rows.link_id.map(pos).isin(lk).to_numpy()).astype(int)
            changed += int((rows.queue_pred.to_numpy() != new).sum())
            q.loc[m, "queue_pred"] = new
    q[["window_id", "timestamp", "link_id", "queue_pred"]].to_csv(out, index=False)
    print(f"wrote {out}: {changed} onset cells changed", flush=True)
