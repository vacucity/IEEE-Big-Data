#!/usr/bin/env python
"""Task 1 first model: fill the blanked cells from what the same day shows around them.

Every day of every corridor is a (288 x links) grid in which about half of the eligible cells are
visible. The methods use only the released masked layer of the same split (allowed: Task 1 is
offline), including degraded cells (pct_observed 1-74) whose values are still published.

  temporal  per link, linear interpolation in time between the nearest visible readings
  spatial   per time step, linear interpolation along milepost between visible neighbours
  blend     inverse-gap weighting of the two (a 5-minute step counts like `km_per_step` km of road)
  tv_qb     speed from `temporal`, flow from `blend` with km_per_step = 0.3   <- default, best on T1 and T3

Usage (from the "IEEE Big data" folder):
  python work/task1_interp.py score --panels D12_I5_N            # all methods, train, local scores
  python work/task1_interp.py score --methods tv_qb               # default method, all ten corridors
  python work/task1_interp.py predict --splits validation private # write work/sub/state_<split>.csv
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import (CLEAN, KEY_BASE, PANELS, REGIMES, ROOT, STEPS, aggregate, links, log_result,  # noqa: E402
                      score_t1_panel, t1_targets, t3_proxy_panel)

OUT = ROOT / "work" / "sub"
METHODS = ("temporal", "spatial", "blend", "tv_qb")
KM_PER_STEP_BLEND, KM_PER_STEP_FLOW = 1.0, 0.3


def interp_axis(G: np.ndarray, x: np.ndarray, axis: int) -> tuple[np.ndarray, np.ndarray]:
    """Linear interpolation of NaN cells along one axis; also returns the gap to the nearest reading."""
    G = np.moveaxis(G, axis, 0)
    out = np.full_like(G, np.nan)
    gap = np.full(G.shape, np.inf)
    for j in range(G.shape[1]):
        col = G[:, j]
        ok = np.isfinite(col)
        if not ok.any():
            continue
        xs = x[ok]
        out[:, j] = np.interp(x, xs, col[ok])
        i = np.searchsorted(xs, x)
        left, right = xs[np.clip(i - 1, 0, len(xs) - 1)], xs[np.clip(i, 0, len(xs) - 1)]
        gap[:, j] = np.minimum(np.abs(x - left), np.abs(x - right))
    return np.moveaxis(out, 0, axis), np.moveaxis(gap, 0, axis)


def blend(a: np.ndarray, ga: np.ndarray, b: np.ndarray, gb: np.ndarray, km_per_step: float) -> np.ndarray:
    """Inverse-gap weighting of a temporal estimate (gap in steps) and a spatial one (gap in km)."""
    wa = np.where(np.isfinite(a), 1.0 / (ga * km_per_step + 1e-3), 0.0)
    wb = np.where(np.isfinite(b), 1.0 / (gb + 1e-3), 0.0)
    den = wa + wb
    with np.errstate(invalid="ignore"):
        return np.where(den > 0, (np.nan_to_num(a) * wa + np.nan_to_num(b) * wb) / np.maximum(den, 1e-12), np.nan)


def predict_day(V: np.ndarray, Q: np.ndarray, mp: np.ndarray, methods: tuple[str, ...]) -> dict[str, tuple]:
    """V, Q: (288, n) visible speed / flow, NaN where hidden. Returns full grids per method."""
    t = np.arange(STEPS, dtype=float)
    vt, gt = interp_axis(V, t, 0)
    qt, _ = interp_axis(Q, t, 0)
    vs, gs = interp_axis(V, mp, 1)
    qs, _ = interp_axis(Q, mp, 1)
    out = {}
    if "temporal" in methods:
        out["temporal"] = (vt, qt)
    if "spatial" in methods:
        out["spatial"] = (vs, qs)
    if "blend" in methods:
        out["blend"] = (blend(vt, gt, vs, gs, KM_PER_STEP_BLEND), blend(qt, gt, qs, gs, KM_PER_STEP_BLEND))
    if "tv_qb" in methods:
        # A link with no visible reading all day has no temporal estimate: fall back to space.
        out["tv_qb"] = (np.where(np.isfinite(vt), vt, vs), blend(qt, gt, qs, gs, KM_PER_STEP_FLOW))
    return out


def predict_panel(panel: str, split: str, methods: tuple[str, ...]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Predictions at the Task 1 targets, aligned with evaluate.t1_targets(panel, split)."""
    n = len(links(panel))
    mp = links(panel).milepost.to_numpy(float)
    preds = {m: ([], []) for m in methods}
    pf = pq.ParquetFile(CLEAN / panel / f"{split}.parquet")
    for i in range(pf.num_row_groups):
        tb = pf.read_row_group(i, columns=["is_t1_target", "speed_obs", "flow_obs"])
        tgt = tb.column("is_t1_target").to_numpy()
        if not tgt.any():
            continue
        V = tb.column("speed_obs").to_numpy().astype(float).reshape(STEPS, n)
        Q = tb.column("flow_obs").to_numpy().astype(float).reshape(STEPS, n)
        for m, (v, q) in predict_day(V, Q, mp, methods).items():
            preds[m][0].append(v.ravel()[tgt])
            preds[m][1].append(q.ravel()[tgt])
    return {m: (np.concatenate(v), np.concatenate(q)) for m, (v, q) in preds.items()}


def state_frame(panel: str, split: str, speed: np.ndarray, flow: np.ndarray) -> pd.DataFrame:
    """Official per-task Task 1 rows for one panel, from target-aligned predictions."""
    tg = t1_targets(panel, split)
    L = links(panel)
    pos = tg["key"] % KEY_BASE
    ts = pd.to_datetime(tg["key"] // KEY_BASE * 300, unit="s", utc=True).strftime("%Y-%m-%dT%H:%M:%SZ")
    return pd.DataFrame({"panel": panel, "timestamp": ts, "station_id": L.station_id.to_numpy()[pos],
                         "link_id": L.link_id.to_numpy()[pos], "mask_regime": np.array(REGIMES)[tg["regime"]],
                         "speed_kmh": speed, "flow_vph": flow})


def cmd_score(args) -> None:
    rows = {m: [] for m in args.methods}
    for p in args.panels:
        t0 = time.time()
        tg = t1_targets(p, "train")
        preds = predict_panel(p, "train", tuple(args.methods))
        for m, (v, q) in preds.items():
            s = score_t1_panel(p, tg, v, q)
            ph = t3_proxy_panel(p, tg, v, q)
            rows[m].append({"panel": p, "S_state": s["S_state"], "S_physics_proxy": ph["S_physics_proxy"],
                            "rmse_speed_R3": s["R3_rmse_speed"], "rmse_flow_lane_R3": s["R3_rmse_flow_lane"]})
        print(f"  [{p}] {time.time() - t0:.0f}s  " + "  ".join(
            f"{m}: {rows[m][-1]['S_state']:.4f}/{rows[m][-1]['S_physics_proxy']:.4f}" for m in args.methods), flush=True)
    for m, r in rows.items():
        df = pd.DataFrame(r)
        _, s_state = aggregate(dict(zip(df.panel, df.S_state)))
        _, s_phys = aggregate(dict(zip(df.panel, df.S_physics_proxy)))
        print(f"\n{m}: S_state={s_state:.4f}  S_physics_proxy={s_phys:.4f}  over {len(df)} panels")
        print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        log_result("t1", "train", f"task1_interp:{m}", f"{m} ({len(df)} panels)", s_state, dict(zip(df.panel, df.S_state)))
        log_result("t3_proxy", "train", f"task1_interp:{m}", f"{m} ({len(df)} panels)", s_phys,
                   dict(zip(df.panel, df.S_physics_proxy)))


def cmd_predict(args) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    method = args.methods[0]
    for split in args.splits:
        path = OUT / f"state_{split}.csv"
        for i, p in enumerate(args.panels):
            v, q = predict_panel(p, split, (method,))[method]
            state_frame(p, split, v, q).to_csv(path, mode="w" if i == 0 else "a", header=i == 0, index=False)
            print(f"  [{split}] {p}: {len(v):,} rows, missing {int((~np.isfinite(v) | ~np.isfinite(q)).sum())}", flush=True)
        print(f"wrote {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score")
    s.add_argument("--panels", nargs="+", default=PANELS)
    s.add_argument("--methods", nargs="+", default=list(METHODS), choices=METHODS)
    s = sub.add_parser("predict")
    s.add_argument("--panels", nargs="+", default=PANELS)
    s.add_argument("--methods", nargs=1, default=["tv_qb"], choices=METHODS)
    s.add_argument("--splits", nargs="+", default=["validation", "private"])
    args = ap.parse_args()
    {"score": cmd_score, "predict": cmd_predict}[args.cmd](args)


if __name__ == "__main__":
    main()
