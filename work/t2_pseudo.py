#!/usr/bin/env python
"""Pseudo Task 2 windows from the released masked layer, for offline evaluation under month shift.

The leaderboard scores 40 onset and 40 ongoing windows per split, which leaves about +-0.06 of noise on
each condition. Validation and private were generated with their own seeds (other bottlenecks activate
first, capacities differ), so train-only cross-validation cannot see that shift. Here every split's visible
masked layer is cut into many windows shaped like the official ones:

  onset    no visible queue in [T-60, T] or at T+5..T+25, a visible queue at T+30
  ongoing  a visible queue in the last 10 history minutes and somewhere in the horizon
  both     the 18 steps [T-60, T+30] stay clear of every official horizon/buffer blackout

Features use the masked layer only (also on train, so all three splits look alike). Labels are the visible
queue states on visible cells (train: the unmasked layer on eligible cells). IoU is computed per window on
the scored cells and averaged per panel and condition, then over panels.

Usage (from the "IEEE Big data" folder):
  python work/t2_pseudo.py stats
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import STEPS, T2_PANELS, links  # noqa: E402
from t2_grids import load  # noqa: E402

HIST, HOR = 12, 6
ONGOING_STRIDE = 6
MAX_PERSIST_IOU = 0.9          # synthetic_release_v1.json: task2.maximum_persistence_iou


@dataclass
class Win:
    panel: str
    split: str
    day: int
    T: int                 # slot index of the forecast origin within the day
    cond: str              # "onset" / "ongoing"
    date: str
    weekday: int


@lru_cache(maxsize=None)
def vcut(panel: str) -> np.ndarray:
    return links(panel).v_cut_kmh.to_numpy(np.float32)


def label_grid(g: dict, split: str) -> np.ndarray:
    return g["Vt"] if split == "train" else g["V"]


@lru_cache(maxsize=None)
def windows(panel: str, split: str) -> tuple[Win, ...]:
    g = load(panel, split)
    vc = vcut(panel)
    Vf = g["V"]                                      # feature layer (masked)
    Vl = label_grid(g, split)
    out = []
    for d in range(Vf.shape[0]):
        with np.errstate(invalid="ignore"):
            Qf = Vf[d] <= vc                         # visible queue cells (feature layer)
            Ql = Vl[d] <= vc
        seen_l = np.isfinite(Vl[d])
        qf, ql = Qf.any(1), Ql.any(1)
        blk = g["B"][d].any(1)
        last_go = -10 ** 9
        for T in range(HIST, STEPS - HOR):
            if blk[T - HIST:T + HOR + 1].any():
                continue
            H = Qf[T - HIST:T]
            # the official rule: ongoing when the history holds >= 2 queued observations, >= 2 on one link
            established = H.sum() >= 2 and H.sum(0).max() >= 2
            if not established:
                if not qf[T] and not ql[T + 1:T + HOR].any() and ql[T + HOR]:
                    out.append(Win(panel, split, d, T, "onset", str(g["dates"][d]), int(g["weekday"][d])))
            elif ql[T + 1:T + HOR + 1].any() and T - last_go >= ONGOING_STRIDE:
                # the selector drops windows that plain persistence (last history step) already solves
                y = Ql[T + 1:T + HOR + 1] & seen_l[T + 1:T + HOR + 1]
                p = np.repeat((Qf[T - 1])[None], HOR, 0) & seen_l[T + 1:T + HOR + 1]
                if (p & y).sum() / max((p | y).sum(), 1) <= MAX_PERSIST_IOU:
                    out.append(Win(panel, split, d, T, "ongoing", str(g["dates"][d]), int(g["weekday"][d])))
                    last_go = T
    return tuple(out)


def history(w: Win) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    g = load(w.panel, w.split)
    s = slice(w.T - HIST, w.T)
    return g["V"][w.day, s], g["Q"][w.day, s], g["O"][w.day, s]


def truth(w: Win) -> tuple[np.ndarray, np.ndarray]:
    """(6, n) queue truth and (6, n) scored-cell mask for the horizon T+1..T+6."""
    g = load(w.panel, w.split)
    V = label_grid(g, w.split)[w.day, w.T + 1:w.T + HOR + 1]
    with np.errstate(invalid="ignore"):
        return (V <= vcut(w.panel)), np.isfinite(V)


def iou(pred: np.ndarray, w: Win) -> float:
    y, m = truth(w)
    p = pred.astype(bool) & m
    y = y & m
    u = (p | y).sum()
    return 1.0 if u == 0 else float((p & y).sum() / u)


def evaluate(predict, splits=("validation", "private"), panels=T2_PANELS, conds=("onset", "ongoing"),
             max_day: int | None = None, verbose: bool = False) -> pd.DataFrame:
    """predict(w) -> (6, n) 0/1 array. Returns per-window IoU rows."""
    rows = []
    for split in splits:
        for p in panels:
            for w in windows(p, split):
                if w.cond not in conds or (max_day is not None and w.day >= max_day):
                    continue
                rows.append((split, p, w.cond, w.day, w.T, iou(predict(w), w)))
    df = pd.DataFrame(rows, columns=["split", "panel", "cond", "day", "T", "iou"])
    if verbose:
        print(summary(df).to_string())
    return df


def summary(df: pd.DataFrame) -> pd.DataFrame:
    """Panel-averaged IoU per split and condition (each panel weighs the same, as on the leaderboard)."""
    pm = df.groupby(["split", "cond", "panel"]).iou.mean().reset_index()
    out = pm.groupby(["split", "cond"]).iou.mean().unstack()
    out["n_windows"] = df.groupby("split").size()
    return out.round(4)


def cmd_stats() -> None:
    rows = []
    for s in ("train", "validation", "private"):
        for p in T2_PANELS:
            ws = windows(p, s)
            rows.append((s, p, sum(w.cond == "onset" for w in ws), sum(w.cond == "ongoing" for w in ws)))
    print(pd.DataFrame(rows, columns=["split", "panel", "onset", "ongoing"]).to_string())


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "stats":
        cmd_stats()
