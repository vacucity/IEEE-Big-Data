#!/usr/bin/env python
"""V52 helper: production ongoing members on arbitrary windows, CPU only (no CUDA), cached per panel/split.

Windows: the replica-selector ongoing windows (work/cache/t2_v52_selrule/windows.csv, train months + March)
or the official ongoing windows ("official" mode, validation/private, no labels; used only to check that this
CPU re-implementation reproduces the submitted V48/V51 decisions).

Members (names as in t2_grid_april.py / t2_v50_april.py):
  rnn2     V27 BiLSTM seeds 0/1 hybrid (T+5 from history [T-60,T-5], T+10..T+30 from [T-55,T])
  cnng0_k  gap-0 CNN seeds 0/1/2          rnng0_k  gap-0 BiLSTM seeds 0/1
  ml2g0    gap-0 LightGBM                 catg0    gap-0 CatBoost
  rnnft_k  gap-0 BiLSTM seeds 0..3 fine-tuned on March, cnnft_k CNN seeds 1/2 fine-tuned on March
           (train windows: the *_ftall models; March windows: cross-fitted *_fta for days >= 15 and *_ftb for
           days < 15, so a March window is never scored by a model tuned on its own day; official April windows:
           *_ftall as in V51)
Inputs: train windows use the eligible-cell truth layer (Vt/Qt/Ot), which is exactly what the official train
window_history shows (no Task 1 holes); March windows use the released masked layer (t2_grids V/Q/O).
Labels: train Vt <= v_cut on eligible cells; March visible V <= v_cut on visible cells.

  python work/t2_v52_prodcache.py replica [--panels ...] [--splits train validation]
  python work/t2_v52_prodcache.py official [--splits validation private]
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_k] = "4"

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import t2_cnn as C  # noqa: E402
import t2_grids  # noqa: E402
import t2_pseudo as P  # noqa: E402
import t2_rnn as R  # noqa: E402
from evaluate import RELEASE, ROOT, STEPS, T2_PANELS, links, slot_of  # noqa: E402

CACHE = ROOT / "work" / "cache" / "t2_v52_selrule"
MODELS = ROOT / "work" / "models"

_orig_load = t2_grids.load


def _load_override(panel, split):
    """Train windows read the eligible-cell truth layer (= official train window_history)."""
    g = _orig_load(panel, split)
    if split == "train":
        return {**g, "V": g["Vt"], "Q": g["Qt"], "O": g["Ot"]}
    return g


t2_grids.load = _load_override

NETS_BASE = [("rnn2a", "rnn", ""), ("rnn2b", "rnn", "_s1")]                    # V27 gap-1 BiLSTMs
NETS_G0 = [("cnng0_0", "cnn", "_g0"), ("cnng0_1", "cnn", "_g0_s1"), ("cnng0_2", "cnn", "_g0_s2"),
           ("rnng0_0", "rnn", "_g0"), ("rnng0_1", "rnn", "_g0_s1")]
NETS_FT = [("rnnft_0", "rnn", "_g0_ft"), ("rnnft_1", "rnn", "_g0_s1_ft"), ("rnnft_2", "rnn", "_g0_s2_ft"),
           ("rnnft_3", "rnn", "_g0_s3_ft"), ("cnnft_1", "cnn", "_g0_s1_ft"), ("cnnft_2", "cnn", "_g0_s2_ft")]


def load_cpu(kind, tag):
    import torch
    m = C.build_model(96) if kind == "cnn" else R.build_model()
    m.load_state_dict(torch.load(MODELS / f"t2_{kind}{tag}.pt", map_location="cpu", weights_only=True))
    m.eval()
    return m


def predict_cpu(kind, m, xs):
    import torch
    (X,) = C.batch_tensors(xs) if kind == "cnn" else R.to_batch(xs)
    with torch.no_grad():
        out = torch.sigmoid(m(X).float()).numpy()
    return out[:, :, :xs[0].shape[-1]]


def nets_on(D, ws, nets, gap):
    """nets: [(name, kind, tag)], gap 0: history rows T-11..T (tod T+1); gap 1: rows T-12..T-1 (tod T)."""
    out = {nm: np.zeros((len(ws), 6, D.n), np.float32) for nm, _, _ in nets}
    models = [(nm, kind, load_cpu(kind, tag)) for nm, kind, tag in nets]
    for s in range(0, len(ws), 32):
        xs = []
        for w in ws[s:s + 32]:
            h = slice(w.T - 11, w.T + 1) if gap == 0 else slice(w.T - 12, w.T)
            xs.append(C.make_x(D.V[w.day, h], D.Q[w.day, h], D.O[w.day, h], D.static_ch, D.st["ffs"],
                               D.st["cap"], D.vcut, w.T + 1 - gap, w.weekday))
        for nm, kind, m in models:
            out[nm][s:s + len(xs), :, D.perm] = predict_cpu(kind, m, xs)
    return out


def members(panel, split, ws, ft_suffix):
    """All production members for windows ws (list of P.Win). ft_suffix: per-window 'all' / 'a' / 'b'."""
    import torch
    torch.set_num_threads(4)
    from t2_cat_g0 import cat_g0_probs
    from t2_slotT import ml2_g0_probs
    D = C.PanelData(panel, split, unmasked=(split == "train"))
    out = {}
    b1 = nets_on(D, ws, NETS_BASE, gap=1)
    b0 = nets_on(D, ws, NETS_BASE, gap=0)
    for nm in ("rnn2a", "rnn2b"):
        out[nm] = np.concatenate([b1[nm][:, :1], b0[nm][:, :5]], 1)       # t2_slotT.hybrid
    out.update(nets_on(D, ws, NETS_G0, gap=0))
    ft_suffix = np.asarray(ft_suffix)
    for nm, kind, tag in NETS_FT:
        out[nm] = np.zeros((len(ws), 6, D.n), np.float32)
    for suf in sorted(set(ft_suffix)):
        idx = np.flatnonzero(ft_suffix == suf)
        sub = [ws[i] for i in idx]
        r = nets_on(D, sub, [(nm, k, tag + suf) for nm, k, tag in NETS_FT], gap=0)
        for nm in r:
            out[nm][idx] = r[nm]
    out["ml2g0"] = ml2_g0_probs(ws)
    out["catg0"] = cat_g0_probs(ws)
    return out


def labels(panel, split, ws):
    g = t2_grids.load(panel, split)                       # train: Vt via the override
    vc = links(panel).v_cut_kmh.to_numpy(np.float32)
    V = np.stack([g["V"][w.day, w.T + 1:w.T + 7] for w in ws])
    with np.errstate(invalid="ignore"):
        return V <= vc, np.isfinite(V)


def cmd_replica(args):
    win = pd.read_csv(CACHE / "windows.csv")
    win = win[win.cond == "ongoing"].drop_duplicates(["panel", "split", "day", "T"])
    win = win[(win["T"] >= 13) & (win["T"] <= 281)]             # within-day history/horizon, like every official window
    for split in args.splits:
        assert split in ("train", "validation")
        for panel in args.panels:
            path = CACHE / f"prod_{panel}_{split}.npz"
            if path.exists():
                print("exists", path.name, flush=True)
                continue
            t0 = time.time()
            w = win[(win.panel == panel) & (win.split == split)].sort_values(["day", "T"])
            g = t2_grids.load(panel, split)
            ws = [P.Win(panel, split, int(r.day), int(r.T), "ongoing", str(g["dates"][r.day]), int(g["weekday"][r.day]))
                  for r in w.itertuples()]
            for r, x in zip(w.itertuples(), ws):
                assert str(r.date) == x.date
            suf = ["all"] * len(ws) if split == "train" else ["a" if x.day >= 15 else "b" for x in ws]
            m = members(panel, split, ws, suf)
            y, mask = labels(panel, split, ws)
            np.savez_compressed(path, day=w.day.to_numpy(), T=w["T"].to_numpy(), y=y, mask=mask,
                                **{k: v.astype(np.float16) for k, v in m.items()})
            print(split, panel, len(ws), f"{time.time() - t0:.0f}s", flush=True)


def official_ws(panel, split):
    idx = pd.read_csv(RELEASE / "task2" / panel / split / "window_index.csv")
    idx = idx[idx.condition == "queue_ongoing"]
    g = t2_grids.load(panel, split)
    dates = [str(d) for d in g["dates"]]
    ws = []
    for r in idx.itertuples(index=False):
        T = int(slot_of(pd.Series([r.forecast_origin]))[0]) % STEPS
        d = dates.index(str(r.date))
        ws.append(P.Win(panel, split, d, T, "ongoing", str(r.date), int(g["weekday"][d])))
    return idx.window_id.tolist(), ws


def cmd_official(args):
    for split in args.splits:
        for panel in args.panels:
            path = CACHE / f"official_{panel}_{split}.npz"
            if path.exists():
                print("exists", path.name, flush=True)
                continue
            ids, ws = official_ws(panel, split)
            m = members(panel, split, ws, ["all"] * len(ws))
            np.savez_compressed(path, ids=np.array(ids), day=[w.day for w in ws], T=[w.T for w in ws],
                                **{k: v.astype(np.float32) for k, v in m.items()})
            print("official", split, panel, flush=True)


if __name__ == "__main__":
    np.seterr(invalid="ignore", divide="ignore")
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("replica")
    s.add_argument("--panels", nargs="+", default=T2_PANELS)
    s.add_argument("--splits", nargs="+", default=["validation", "train"])
    s = sub.add_parser("official")
    s.add_argument("--panels", nargs="+", default=T2_PANELS)
    s.add_argument("--splits", nargs="+", default=["validation", "private"])
    a = ap.parse_args()
    {"replica": cmd_replica, "official": cmd_official}[a.cmd](a)
