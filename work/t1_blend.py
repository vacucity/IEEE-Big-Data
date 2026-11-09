#!/usr/bin/env python
"""Out-of-fold comparison and blend of the Task 1 GBM (task1_gbm, C_w111) and the U-Net (t1_unet), scored on
the held-out days of one fold (both models never saw those months).

Usage (from the "IEEE Big data" folder):
  python work/t1_blend.py gbm_oof --fold 0 --panels D7_I10_E ...    # cache GBM predictions for those days
  python work/t1_blend.py eval --fold 0 --panels ... [--tag ""]       # S_state and T3 proxy: GBM, U-Net, blends
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import (CACHE, CLEAN, EMPTY_FLOOR_VPH, EMPTY_SHARE, FLOW_NORM, FLOW_W, LWR_RESIDUAL_FLOOR,  # noqa: E402
                      PANELS, REGIMES, SPEED_NORM, SPEED_W, STEPS, links)

UNET = CACHE / "t1_unet"


def fold_days(panel: str, fold: int) -> list[int]:
    m = json.loads((UNET / f"{panel}_train_meta.json").read_text())
    return [i for i, mo in enumerate(m["month"]) if mo % 2 == fold]


def cmd_gbm_oof(args) -> None:
    import lightgbm as lgb
    import task1_gbm as G
    G.PARAMS["num_threads"] = args.threads
    models = {n: lgb.Booster(model_file=str(G.MODELS / f"t1_gbm_fold{args.fold}_{n}.txt")) for n in ("v", "q", "k")}
    for p in args.panels:
        path = UNET / f"gbm_oof_{p}_fold{args.fold}.npz"
        if path.exists():
            continue
        days = set(fold_days(p, args.fold))
        ctx = G.Panel(p)
        pf = pq.ParquetFile(CLEAN / p / "train.parquet")
        out = {}
        for i in sorted(days):
            tb = pf.read_row_group(i, columns=G.DAY_COLS)
            d = G.day_features(ctx, tb, False)
            v, q = G.predict_cells(models, d)
            kk = d["base_q"] / np.maximum(d["base_v"], 1.0) + d["lanes"] * models["k"].predict(
                G.add_k_base(d["X"]), num_threads=args.threads)
            v, q = G.combine(v, q, kk, G.COMBINE_W)
            out[f"v{i}"], out[f"q{i}"] = np.clip(v, G.SPEED_MIN, G.SPEED_MAX), q
        np.savez(path, **out)
        print(f"  {p}: {len(days)} days", flush=True)


def load_preds(path: Path) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    z = np.load(path)
    days = sorted({int(k[1:]) for k in z.files})
    return {d: (z[f"v{d}"], z[f"q{d}"]) for d in days}


def score(panel: str, preds: dict[int, tuple[np.ndarray, np.ndarray]]) -> dict:
    """S_state and the T3 proxy (as evaluate.py) restricted to the days in `preds`."""
    L = links(panel)
    n = len(L)
    lanes = L.lanes.to_numpy(float)
    ffs, cap = L.free_speed_kmh.to_numpy(float), L.capacity_vph.to_numpy(float)
    length = L.length_km.to_numpy(float)
    kcrit = cap / np.maximum(ffs, 1.0)
    kjam = np.maximum(L.k_jam.to_numpy(float), 1.05 * kcrit)
    kcl, kjl, cpl = kcrit / lanes, kjam / lanes, cap / lanes
    wl = cpl / np.maximum(kjl - kcl, 1e-9)
    pf = pq.ParquetFile(CLEAN / panel / "train.parquet")
    acc = {r: dict.fromkeys(("sv", "sq", "nt", "fdn", "fdd", "sub", "low", "ln", "ld"), 0.0) for r in range(3)}
    cols = ["mask_regime", "is_score_eligible", "is_t1_target", "speed_true", "flow_true", "on_ramp_n",
            "on_ramp_n_valid", "off_ramp_n", "off_ramp_n_valid"]
    for i, (pv, pq_) in preds.items():
        t = pf.read_row_group(i, columns=cols)
        a = acc[REGIMES.index(t.column("mask_regime")[0].as_py())]
        tgt = t.column("is_t1_target").to_numpy().reshape(STEPS, n)
        vt = t.column("speed_true").to_numpy().astype(float).reshape(STEPS, n)
        qt = t.column("flow_true").to_numpy().astype(float).reshape(STEPS, n)
        elig = t.column("is_score_eligible").to_numpy().reshape(STEPS, n)
        v, q = vt.copy(), qt.copy()
        v[tgt], q[tgt] = pv, pq_
        m = tgt & np.isfinite(vt) & np.isfinite(qt)
        a["sv"] += float(((v - vt) ** 2)[m].sum())
        a["sq"] += float((((q - qt) / lanes) ** 2)[m].sum())
        a["nt"] += int(m.sum())
        use = elig & np.isfinite(v) & np.isfinite(q)
        with np.errstate(invalid="ignore", divide="ignore"):
            ql = q / lanes
            kl = q / np.maximum(v, 1.0) / lanes
            qfd = np.where(kl <= kcl, ffs * kl, wl * np.maximum(kjl - kl, 0.0))
        a["fdn"] += float(np.abs(ql - qfd)[use].sum())
        a["fdd"] += float(np.abs(ql)[use].sum())
        sub = tgt & use
        a["sub"] += int(sub.sum())
        a["low"] += int((q[sub] < EMPTY_FLOOR_VPH).sum())
        on_ok = (t.column("on_ramp_n").to_numpy() == 0) | (t.column("on_ramp_n_valid").to_numpy() > 0)
        off_ok = (t.column("off_ramp_n").to_numpy() == 0) | (t.column("off_ramp_n_valid").to_numpy() > 0)
        ok = use[:-1] & use[1:] & (on_ok & off_ok).reshape(STEPS, n)[:-1]
        with np.errstate(invalid="ignore"):
            Ns, Nt = q / np.maximum(v, 1.0) * length, qt / np.maximum(vt, 1.0) * length
        ds, dt = np.diff(Ns, axis=0), np.diff(Nt, axis=0)
        a["ln"] += float(np.abs(ds - dt)[ok].sum())
        a["ld"] += float(np.abs(dt)[ok].sum())
    out, ss, ph = {"panel": panel}, [], []
    for r in range(3):
        a = acc[r]
        if a["nt"] == 0:
            continue
        s = SPEED_W * max(0, 1 - np.sqrt(a["sv"] / a["nt"]) / SPEED_NORM) + FLOW_W * max(0, 1 - np.sqrt(a["sq"] / a["nt"]) / FLOW_NORM)
        fd = max(0.0, 1 - a["fdn"] / max(a["fdd"], 1e-9))
        if a["sub"] and a["low"] / a["sub"] > EMPTY_SHARE:
            fd = 0.0
        lwr = max(0.0, 1 - min(1.0, a["ln"] / max(a["ld"], 1e-9) + LWR_RESIDUAL_FLOOR))
        ss.append(s)
        ph.append((fd + 2 * lwr) / 3)
        out[f"{REGIMES[r]}_fd"], out[f"{REGIMES[r]}_lwr"] = fd, lwr
        out[f"{REGIMES[r]}_rmse_v"] = np.sqrt(a["sv"] / a["nt"])
        out[f"{REGIMES[r]}_rmse_q"] = np.sqrt(a["sq"] / a["nt"])
    out["S_state"], out["phys"] = float(np.mean(ss)), float(np.mean(ph))
    return out


def cmd_eval(args) -> None:
    rows = []
    for p in args.panels:
        g = load_preds(UNET / f"gbm_oof_{p}_fold{args.fold}.npz")
        u = load_preds(UNET / f"oof_{p}_fold{args.fold}{args.tag}.npz")
        for wv, wq in [(w, w) for w in args.weights] + [(a, b) for a in args.weights for b in args.weights if a != b]:
            if args.diag_only and wv != wq:
                continue
            bl = {d: (wv * u[d][0] + (1 - wv) * g[d][0], wq * u[d][1] + (1 - wq) * g[d][1]) for d in g}
            r = score(p, bl)
            rows.append({**r, "wv": wv, "wq": wq})
            print(f"  {p} wv={wv} wq={wq}: S_state {r['S_state']:.4f} phys {r['phys']:.4f}", flush=True)
    df = pd.DataFrame(rows)
    res = df.groupby(["wv", "wq"])[["S_state", "phys"]].mean()
    res["obj"] = 0.35 * res.S_state + 0.15 * res.phys
    print(res.sort_values("obj", ascending=False).to_string(float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("gbm_oof")
    s.add_argument("--fold", type=int, default=0)
    s.add_argument("--threads", type=int, default=6)
    s.add_argument("--panels", nargs="+", default=PANELS)
    s = sub.add_parser("eval")
    s.add_argument("--fold", type=int, default=0)
    s.add_argument("--panels", nargs="+", default=PANELS)
    s.add_argument("--tag", default="")
    s.add_argument("--weights", nargs="+", type=float, default=[0.0, 0.5, 0.7, 1.0])
    s.add_argument("--diag-only", action="store_true")
    args = ap.parse_args()
    {"gbm_oof": cmd_gbm_oof, "eval": cmd_eval}[args.cmd](args)
