#!/usr/bin/env python
"""Gap-0 ml2 ongoing GBM: task2_ml2's features and window sampling, but the labels are the next six rows
right after the history (t..t+5 for a history ending at t-1), so an official window is forecast from the
history ending at the origin row T (discussion 745700: the row at T may be used) for T+5..T+30.

Usage (from the "IEEE Big data" folder):
  python work/t2_ml2_g0.py build     # rows -> work/cache/t2_ml2_g0/<panel>.parquet
  python work/t2_ml2_g0.py fit       # 2-fold early stopping, full refit -> work/models/t2_ml2_ongoing_g0.txt
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(key, "6")

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
import task2_ml2 as M  # noqa: E402
from evaluate import CACHE, CLEAN, ROOT, STEPS, T2_PANELS  # noqa: E402
from task2_ml import RADIUS_KM, ctx_of  # noqa: E402
from task2_rules import DEFAULTS as RULES, HIST_STEPS, HORIZON, onset_scores  # noqa: E402

DATA = CACHE / "t2_ml2_g0"
MODEL = ROOT / "work" / "models" / "t2_ml2_ongoing_g0.txt"


def build_panel(panel: str) -> pd.DataFrame:
    ctx = ctx_of(panel)
    n, pno, ev = ctx.n, T2_PANELS.index(panel), ctx.ev
    pf = pq.ParquetFile(CLEAN / panel / "train.parquet")
    parts = []
    for i in range(pf.num_row_groups):
        tb = pf.read_row_group(i, columns=["ts", "weekday", "speed_true", "flow_true", "occ_true", "is_score_eligible"])
        ts0 = pd.Timestamp(tb.column("ts")[0].as_py())
        date, weekday, fold = ts0.strftime("%Y-%m-%d"), int(tb.column("weekday")[0].as_py()), ts0.month % 2
        E = tb.column("is_score_eligible").to_numpy().reshape(STEPS, n)
        g = lambda c: np.where(E, tb.column(c).to_numpy().astype(float).reshape(STEPS, n), np.nan)
        V, Q, O = g("speed_true"), g("flow_true"), g("occ_true")
        with np.errstate(invalid="ignore"):
            Qd = E & (V <= ctx.vcut)
        sl = lambda T: slice(T - HIST_STEPS, T)
        cand = [T for T in range(HIST_STEPS, STEPS - HORIZON)
                if Qd[sl(T)].sum() >= 2 and Qd[sl(T)].sum(0).max() >= 2 and Qd[T:T + HORIZON].any()
                and np.isfinite(V[sl(T)]).mean() >= 0.7]
        rng = np.random.default_rng(M.SEED + 1000 * pno + i)
        chosen = []
        for T in rng.permutation(cand):
            if all(abs(T - c) >= M.MIN_SPACING for c in chosen):
                chosen.append(int(T))
            if len(chosen) >= M.WINDOWS_PER_DAY:
                break
        for T in chosen:
            # the origin is the last history row T-1; its horizon T..T+5 is "+5 .. +30 min"
            prior = onset_scores(ev, n, T - 1 + HORIZON, weekday, date, RULES["sigma_h"], RULES["weekend"])
            f = M.features2(ctx, V[sl(T)], Q[sl(T)], O[sl(T)], T, int(weekday >= 5), prior, pno)
            with np.errstate(invalid="ignore"):
                hq = (V[sl(T)] <= ctx.vcut).any(0)
            near = np.zeros(n, bool)
            for xq in ctx.x[hq]:
                near |= np.abs(ctx.x - xq) <= RADIUS_KM
            win_rows = sum(int((near & E[T + s]).sum()) for s in range(HORIZON))
            for s in range(HORIZON):
                rows = np.flatnonzero(near & E[T + s])
                df = M.frame(f, rows, s + 1, Qd[T + s, rows], fold, date, "ongoing")
                df["w"] = np.float32(1000.0 / max(win_rows, 1))
                parts.append(df)
    out = pd.concat(parts, ignore_index=True)
    out["panel"] = panel
    return out


def cmd_build():
    DATA.mkdir(parents=True, exist_ok=True)
    for p in T2_PANELS:
        if (DATA / f"{p}.parquet").exists():
            continue
        t0 = time.time()
        d = build_panel(p)
        d.to_parquet(DATA / f"{p}.parquet", index=False)
        print(p, len(d), f"{time.time() - t0:.0f}s", flush=True)


def cmd_fit():
    d = pd.concat([pd.read_parquet(DATA / f"{p}.parquet") for p in T2_PANELS], ignore_index=True)
    d = d.astype({"date": "category", "kind": "category", "panel": "category"})
    _, i0, _ = M.fit("ongoing", d[d.fold != 0], d[d.fold == 0])
    _, i1, _ = M.fit("ongoing", d[d.fold != 1], d[d.fold == 1])
    _, _, b = M.fit("ongoing", d, None, int(1.1 * (i0 + i1) / 2))
    b.save_model(str(MODEL))
    print("rounds", i0, i1, "->", MODEL, flush=True)


if __name__ == "__main__":
    {"build": cmd_build, "fit": cmd_fit}[sys.argv[1]]()
