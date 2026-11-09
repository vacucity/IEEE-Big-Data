"""V52 Task 1: per-month, per-link speed (and optionally flow) bias correction on top of the V51 state.

Why: the free-flow mean observed speed of a link differs between March (validation) and the train months by
~1.16 km/h RMS, April (private) vs train by ~1.19 and March vs April by ~1.17, while train months agree
within 0.03 km/h.  The production U-Nets (_vp) were self-adapted on March+April *pooled* visible cells, so
they can only learn the average of the two month offsets; each month keeps its own residual per-link bias.

Everything here uses only the visible observations of the month being corrected (Task 1 is offline, the
same permission the V24 month adaptation relies on).  No Task 2 file, window, prediction or label is read;
the queue and ODME rows of the package stay byte-identical to v51_april.zip.

Steps (run from the "IEEE Big data" folder, anaconda python):
  selftest --panel P --split S --day d   rerun the production chain on CPU without extra hiding and compare it
                                          with the V40-half state rows of that day (fidelity of the chain)
  collect  --split S [--panels ...]      every day of the month: hide an extra 20% of visible eligible cells,
                                          rerun the full production chain (GBM + 2 folds x {w48_vp, w48lwr_vp}
                                          U-Nets + frozen V31 residual stack at the panel's production alpha,
                                          blackout cells as in production) and store prediction vs observation
                                          on the hidden cells, plus Task 3 bookkeeping (FD terms, transitions)
  crossfit                               per-link bias estimated on even-index days is applied to odd-index days
                                          and vice versa (each month on its own); grid over shrinkage K
                                          (w = n/(n+K)), separate flow shrinkage Kq, cell selection (free /
                                          all / free+congested classes) and channels (v, q, vq); S_state proxy
                                          per corridor-month (R1/R2/R3 mean, 0.54/0.46), S_FD and LWR proxy
                                          before/after on the hide frame plus a per-regime real-frame scaling
                                          (estimate); choice = max family-mean 0.35*dS_state + 0.15*dS_phys(est)
                                          averaged over the two months; nested month check; lambda curve
  direct                                 per-link bias at the REAL targets (production prediction at deep-free
                                          targets minus same-day deep-free visible observations) versus the
                                          hide-protocol bias: checks that the hidden cells represent the targets
  apply                                  bias from all days of each month (chosen config) is subtracted from the
                                          state rows inside v51_april.zip of that month; Task 2 blackout target
                                          cells keep their V51 values; writes work/sub/state_{split}_v52bias.csv
                                          and S_FD before/after on the real submission frame
  package                                patch only state rows of v51_april.zip -> work/submissions/v52_t1bias.zip,
                                          validate with make_submission.validate, write the manifest
  audit                                  independent line-by-line comparison of v52 against v51

Outputs go to work/results/t1_v52_monthbias/; nothing existing is overwritten (FileExistsError instead).

Reproduce (about 2 h CPU for collect, the rest < 10 min; OMP_NUM_THREADS=4):
  python work/t1_v52_monthbias.py collect --split validation --threads 2      (and --split private)
  python work/t1_v52_monthbias.py crossfit
  python work/t1_v52_monthbias.py direct
  python work/t1_v52_monthbias.py apply
  python work/t1_v52_monthbias.py package
  python work/t1_v52_monthbias.py audit
See work/V52_T1BIAS_README.md.
"""
from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "3")

import argparse
import hashlib
import json
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import (CLEAN, FAMILY, KEY_BASE, PANELS, REGIMES, RELEASE, STEPS, links, slot_of,  # noqa: E402
                      t1_targets)

WORK = Path(__file__).resolve().parent
OUT = WORK / "results" / "t1_v52_monthbias"
HIDE = OUT / "hide"
BASE_ZIP = WORK / "submissions" / "v51_april.zip"
NAME = "v52_t1bias"
SEED = 52052
HIDE_RATE = 0.20
FREE_RATIO = 1.15             # "predicted free flow": production speed > 1.15 * v_cut (no truth needed)
SPLITS = ("validation", "private")
MONTH_OF = {"validation": "2031-03 (March)", "private": "2031-04 (April)"}
# Production alpha of the frozen residual stack per panel (V31 four at 1, V36 five at 1, V40 halves two,
# D7_I210_E never corrected).  The V40 package keeps Task 2 blackout cells at the uncorrected blend.
ALPHA = {"D12_I405_S": 1.0, "D7_I210_W": 1.0, "D12_I5_S": 1.0, "D7_I10_E": 1.0, "D7_I405_S": 1.0,
         "D12_I5_N": 1.0, "D12_I405_N": 1.0, "D7_I10_W": 0.5, "D7_I405_N": 0.5, "D7_I210_E": 0.0}
RAMP_COLS = ["on_ramp_n", "on_ramp_n_valid", "off_ramp_n", "off_ramp_n_valid"]
SPEED_MIN, SPEED_MAX = 5.0, 120.75
LWR_FLOOR = 0.045


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_new(path: Path, text: str) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.write_text(text, encoding="utf-8")


def static_fd(panel: str) -> dict:
    """Per-link (clean pos order) constants of the Task 3 frame, as evaluate.t3_proxy_panel."""
    L = links(panel)
    lanes = L.lanes.to_numpy(float)
    ffs = L.free_speed_kmh.to_numpy(float)
    cap = L.capacity_vph.to_numpy(float)
    kcrit = cap / np.maximum(ffs, 1.0)
    kjam = np.maximum(L.k_jam.to_numpy(float), 1.05 * kcrit)
    kcl, kjl, cpl = kcrit / lanes, kjam / lanes, cap / lanes
    return {"lanes": lanes, "ffs": ffs, "kcl": kcl, "kjl": kjl, "wl": cpl / np.maximum(kjl - kcl, 1e-9),
            "length": L.length_km.to_numpy(float), "vc": L.v_cut_kmh.to_numpy(float), "n": len(L)}


def fd_terms(v, q, pos, fd):
    lanes = fd["lanes"][pos]
    with np.errstate(invalid="ignore", divide="ignore"):
        ql = q / lanes
        kl = q / np.maximum(v, 1.0) / lanes
        qfd = np.where(kl <= fd["kcl"][pos], fd["ffs"][pos] * kl, fd["wl"][pos] * np.maximum(fd["kjl"][pos] - kl, 0.0))
    return np.abs(ql - qfd), np.abs(ql)


# ---------------------------------------------------------------------------
# production chain on CPU
# ---------------------------------------------------------------------------
class Chain:
    def __init__(self, threads: int = 3):
        import torch
        import lightgbm as lgb
        import t1_unet as U
        import task1_gbm as G
        import t1_v30_bridge as B
        import t1_v30_stack as S
        torch.set_num_threads(threads)
        self.torch, self.U, self.G, self.B, self.S = torch, U, G, B, S
        self.nets = {}
        for tag in B.TAGS:
            for fold in (0, 1):
                m = U.build_model(48)
                m.load_state_dict(torch.load(U.MODELS / f"t1_unet_fold{fold}{tag}_vp.pt", map_location="cpu"))
                m.eval()
                self.nets[tag, fold] = m
        self.gm = {t: lgb.Booster(model_file=str(G.MODELS / f"t1_gbm_{t}.txt")) for t in ("v", "q", "k")}
        self.sm = {t: lgb.Booster(model_file=str(S.OUT / f"model_{t}.txt")) for t in ("v", "q", "k")}
        self.fingerprint = {
            **{f"unet_fold{f}{t}_vp": sha256(U.MODELS / f"t1_unet_fold{f}{t}_vp.pt") for t in B.TAGS for f in (0, 1)},
            **{f"gbm_{t}": sha256(G.MODELS / f"t1_gbm_{t}.txt") for t in ("v", "q", "k")},
            **{f"stack_{t}": sha256(S.OUT / f"model_{t}.txt") for t in ("v", "q", "k")}}

    def unet(self, model, d, hide_travel):
        x, b = self.U.make_input(d, hide_travel)
        with self.torch.no_grad():
            out = model(self.U.to_batch([x])).float().numpy()[0]
        n = x.shape[-1]
        v = (b[0] + out[0, :, :n]) * d.st["ffs"]
        q = (b[1] + out[1, :, :n]) * d.st["cap"]
        return np.clip(v, SPEED_MIN, SPEED_MAX), np.maximum(q, 0.0)

    def run(self, ctx, tb, ud, hide, alpha, st, gbm_pair=None):
        """Production (v, q) at every target cell of table tb (targets = original targets | hide)."""
        B, S = self.B, self.S
        mt = B.masked_table(tb, hide) if hide is not None and hide.any() else tb
        target = mt.column("is_t1_target").to_numpy().reshape(STEPS, ctx.n)
        pairs = [gbm_pair if gbm_pair is not None else B.gbm(self.gm, ctx, mt)]
        comps = {"gbm": pairs[0]}
        h_tr = None if hide is None else hide[:, st["perm"]]
        for tag in B.TAGS:
            ps = [self.unet(self.nets[tag, f], ud, h_tr) for f in (0, 1)]
            pairs.append(tuple(np.mean([p[j] for p in ps], axis=0)[:, ctx.inv][target] for j in (0, 1)))
            comps[tag] = pairs[-1]
        x, bv, bq, gap, arr = S.features(ctx, mt, pairs)
        lane = np.broadcast_to(ctx.lanes, target.shape)[target]
        if alpha == 0:
            v, q = bv.copy(), bq.copy()
        else:
            v, q = B.corrected(x, bv, bq, gap, lane, self.sm, alpha)
        return v, q, bv, bq, gap, target, comps


def cmd_selftest(args) -> None:
    import t1_unet as U
    import task1_gbm as G
    ch = Chain(args.threads)
    p, split, day = args.panel, args.split, args.day
    ctx, st = G.Panel(p), U.static(p)
    src, prof = U.open_split(p, split), np.load(U.DATA / f"{p}_profiles.npy")
    pf = pq.ParquetFile(CLEAN / p / f"{split}.parquet")
    tb = pf.read_row_group(day, columns=list(dict.fromkeys(ch.B.COLS + RAMP_COLS)), use_threads=False)
    ud = U.Day(p, split, day, src, st, prof)
    t0 = time.time()
    v, q, bv, bq, gap, target, comps = ch.run(ctx, tb, ud, None, ALPHA[p], st)
    sec = time.time() - t0
    tg = t1_targets(p, split)
    counts = [int(pf.read_row_group(i, columns=["is_t1_target"]).column(0).to_numpy().sum()) for i in range(day + 1)]
    lo, hi = sum(counts[:-1]), sum(counts)
    keys = tg["key"][lo:hi]
    ref = pd.read_csv(WORK / "sub" / f"state_{split}_v40half.csv", usecols=["panel", "timestamp", "link_id", "speed_kmh", "flow_vph"])
    ref = ref[ref.panel == p]
    pos = dict(zip(links(p).link_id, links(p).pos))
    rk = slot_of(ref.timestamp) * KEY_BASE + ref.link_id.map(pos).to_numpy(np.int64)
    j = np.searchsorted(rk, keys)
    assert np.array_equal(rk[j], keys)
    rv, rq = ref.speed_kmh.to_numpy()[j], ref.flow_vph.to_numpy()[j]
    gp = pd.read_csv(WORK / "sub" / f"state_{split}.csv", usecols=["panel", "speed_kmh", "flow_vph"])
    gp = gp[gp.panel == p][["speed_kmh", "flow_vph"]].to_numpy()[lo:hi]
    res = {"panel": p, "split": split, "day": day, "targets": int(hi - lo), "chain_sec": round(sec, 1),
           "gbm_max_abs_speed_vs_state_csv": float(np.max(np.abs(comps["gbm"][0] - gp[:, 0]))),
           "gbm_max_abs_flow_vs_state_csv": float(np.max(np.abs(comps["gbm"][1] - gp[:, 1]))),
           "speed_rms_diff_vs_v40half": float(np.sqrt(np.mean((v - rv) ** 2))),
           "speed_mean_diff_vs_v40half": float(np.mean(v - rv)),
           "speed_max_abs_diff_vs_v40half": float(np.max(np.abs(v - rv))),
           "flow_rms_diff_vs_v40half": float(np.sqrt(np.mean((q - rq) ** 2))),
           "flow_mean_diff_vs_v40half": float(np.mean(q - rq))}
    # same with the cached production U-Net predictions (bf16 GPU) instead of the CPU fp32 rerun
    from t1_blend import load_preds
    pp = [load_preds(U.DATA / f"pred_{p}_{split}{t}_vp.npz")[day] for t in ch.B.TAGS]
    for t, cp in zip(ch.B.TAGS, pp):
        res[f"unet{t}_cpu_vs_cached_speed_rms"] = float(np.sqrt(np.mean((comps[t][0] - cp[0]) ** 2)))
        res[f"unet{t}_cpu_vs_cached_speed_mean"] = float(np.mean(comps[t][0] - cp[0]))
    print(json.dumps(res, indent=2), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"selftest_{p}_{split}_{day}.json"
    if not path.exists():
        path.write_text(json.dumps(res, indent=2))


# ---------------------------------------------------------------------------
# collect: hide-20%-of-visible protocol with the production chain
# ---------------------------------------------------------------------------
def cmd_collect(args) -> None:
    import t1_unet as U
    import task1_gbm as G
    HIDE.mkdir(parents=True, exist_ok=True)
    ch = Chain(args.threads)
    prot = HIDE / "protocol.json"
    if not prot.exists():
        prot.write_text(json.dumps({
            "seed": SEED, "hide_rate": HIDE_RATE, "hidden_cells": "visible (speed and flow finite) and eligible",
            "chain": "B.gbm on the masked table + mean of 2 folds per tag of *_vp U-Nets on CPU fp32 + S.features "
                     "blend + B.corrected at the production alpha; Task 2 blackout cells as in production",
            "alpha": ALPHA, "models_sha256": ch.fingerprint,
            "rule": "only visible observations of the month itself are read (T1 offline); no Task 2 input"}, indent=2))
    cols = list(dict.fromkeys(ch.B.COLS + RAMP_COLS))
    for p in args.panels:
        cpath = HIDE / f"{p}_{args.split}_cells.parquet"
        if cpath.exists():
            print("exists", cpath.name, flush=True)
            continue
        t0 = time.time()
        ctx, st, fd = G.Panel(p), U.static(p), static_fd(p)
        src, prof = U.open_split(p, args.split), np.load(U.DATA / f"{p}_profiles.npy")
        pf = pq.ParquetFile(CLEAN / p / f"{args.split}.parquet")
        n = ctx.n
        cells, trans, days = [], [], []
        for day in range(pf.num_row_groups):
            tb = pf.read_row_group(day, columns=cols, use_threads=False)
            arr = lambda c: tb.column(c).to_numpy().reshape(STEPS, n)
            ov, oq = arr("speed_obs").astype(float), arr("flow_obs").astype(float)
            elig = arr("is_score_eligible")
            vis = np.isfinite(ov) & np.isfinite(oq)
            rng = np.random.default_rng(SEED + 1000 * PANELS.index(p) + (0 if args.split == "validation" else 500) + day)
            hide = (rng.random(vis.shape) < HIDE_RATE) & vis & elig
            ud = U.Day(p, args.split, day, src, st, prof)
            v, q, bv, bq, gap, target, comps = ch.run(ctx, tb, ud, hide, ALPHA[p], st)
            use_h = hide[target]
            tt, pp_ = np.nonzero(hide)
            nh = len(tt)
            rec = {"day": np.full(nh, day, np.int16), "regime": np.full(nh, ud.regime, np.int8),
                   "t": tt.astype(np.int16), "pos": pp_.astype(np.int16),
                   "ov": ov[hide].astype(np.float32), "oq": oq[hide].astype(np.float32),
                   "pv": v[use_h].astype(np.float32), "pq": q[use_h].astype(np.float32),
                   "bv": bv[use_h].astype(np.float32), "bq": bq[use_h].astype(np.float32)}
            for nm, (cv, cq) in comps.items():
                rec[f"v{nm}"] = cv[use_h].astype(np.float32)
                rec[f"q{nm}"] = cq[use_h].astype(np.float32)
            cells.append(pd.DataFrame(rec))
            # Task 3 frame of this hide run: obs on visible cells, chain predictions on the original targets.
            Fv, Fq = ov.copy(), oq.copy()
            orig_t = target & ~hide
            Fv[orig_t], Fq[orig_t] = v[~use_h], q[~use_h]
            use = elig & np.isfinite(Fv) & np.isfinite(Fq)
            rest = use & ~hide
            pr = np.nonzero(rest)
            fn, fdd = fd_terms(Fv[rest], Fq[rest], pr[1], fd)
            ramp_ok = (((arr("on_ramp_n") == 0) | (arr("on_ramp_n_valid") > 0)) &
                       ((arr("off_ramp_n") == 0) | (arr("off_ramp_n_valid") > 0)))
            ve = vis & elig
            ok = ve[:-1] & ve[1:] & ramp_ok[:-1]
            with np.errstate(invalid="ignore", divide="ignore"):
                N = oq / np.maximum(ov, 1.0) * fd["length"][None, :]
            dN = N[1:] - N[:-1]
            hid = np.full((STEPS, n), -1, np.int64)
            hid[hide] = np.arange(nh)
            inv = ok & (hide[:-1] | hide[1:])
            ti, li = np.nonzero(inv)
            trans.append(pd.DataFrame({"day": np.full(len(ti), day, np.int16), "t": ti.astype(np.int16),
                                       "pos": li.astype(np.int16), "h0": hid[ti, li].astype(np.int32),
                                       "h1": hid[ti + 1, li].astype(np.int32),
                                       "N0": N[ti, li].astype(np.float64), "N1": N[ti + 1, li].astype(np.float64)}))
            days.append({"day": day, "regime": int(ud.regime), "n_hidden": nh, "n_target": int(orig_t.sum()),
                         "n_target_free": int((v[~use_h] > FREE_RATIO * fd["vc"][np.nonzero(orig_t)[1]]).sum()),
                         "fd_num_rest": float(fn.sum()), "fd_den_rest": float(fdd.sum()),
                         "lwr_den": float(np.abs(dN)[ok].sum()), "n_trans": int(ok.sum()),
                         "n_trans_hidden": int(inv.sum()),
                         "n_trans_target": int((use[:-1] & use[1:] & ramp_ok[:-1] & (orig_t[:-1] | orig_t[1:])).sum())})
            print(f"  {p} {args.split} day {day}: {nh} hidden, {time.time() - t0:.0f}s", flush=True)
        C = pd.concat(cells, ignore_index=True)
        # hidden-cell index of the transitions is per day: make it global
        offs = np.concatenate([[0], np.cumsum([d["n_hidden"] for d in days])[:-1]])
        T = []
        for d, tr in zip(days, trans):
            o = offs[d["day"]]
            tr["h0"] = np.where(tr.h0 >= 0, tr.h0 + o, -1)
            tr["h1"] = np.where(tr.h1 >= 0, tr.h1 + o, -1)
            T.append(tr)
        pd.concat(T, ignore_index=True).to_parquet(HIDE / f"{p}_{args.split}_trans.parquet", index=False)
        pd.DataFrame(days).to_csv(HIDE / f"{p}_{args.split}_days.csv", index=False)
        C.to_parquet(cpath, index=False)
        print(p, args.split, "done", len(C), "cells", f"{time.time() - t0:.0f}s", flush=True)


# ---------------------------------------------------------------------------
# bias estimation / application (shared by crossfit and apply)
# ---------------------------------------------------------------------------
def cell_class(pv, vc_cell, sel):
    """0 = corrected class A, 1 = class B (only for 'fc'), -1 = never corrected."""
    free = pv > FREE_RATIO * vc_cell
    if sel == "free":
        return np.where(free, 0, -1)
    if sel == "all":
        return np.zeros(len(pv), np.int64)
    if sel == "fc":
        return np.where(free, 0, 1)
    raise ValueError(sel)


def estimate(C: pd.DataFrame, fd: dict, cfg: dict) -> dict:
    """Per (class, link) shrunken mean error of the production prediction on the hidden cells in C."""
    n = fd["n"]
    pos = C.pos.to_numpy(np.int64)
    cl = cell_class(C.pv.to_numpy(float), fd["vc"][pos], cfg["sel"])
    ev = C.pv.to_numpy(float) - C.ov.to_numpy(float)
    eq = (C.pq.to_numpy(float) - C.oq.to_numpy(float)) / fd["lanes"][pos]
    out = {"bv": np.zeros((2, n)), "bq": np.zeros((2, n)), "n": np.zeros((2, n))}
    K = cfg["K"]
    for c in (0, 1):
        m = cl == c
        cnt = np.bincount(pos[m], minlength=n).astype(float)
        sv = np.bincount(pos[m], weights=ev[m], minlength=n)
        sq = np.bincount(pos[m], weights=eq[m], minlength=n)
        out["n"][c] = cnt
        out["bv"][c] = sv / np.maximum(cnt + K, 1e-9)
        out["bq"][c] = sq / np.maximum(cnt + cfg.get("Kq", K), 1e-9)
    lam = cfg.get("lam", 1.0)
    out["bv"] *= lam
    out["bq"] *= lam
    if "v" not in cfg["ch"]:
        out["bv"][:] = 0.0
    if "q" not in cfg["ch"]:
        out["bq"][:] = 0.0
    return out


def correct(pv, pq, pos, fd, b, sel):
    """Corrected (v, q) and the mask of cells that are changed."""
    cl = cell_class(pv, fd["vc"][pos], sel)
    m = cl >= 0
    c = np.where(m, cl, 0)
    dv = np.where(m, b["bv"][c, pos], 0.0)
    dq = np.where(m, b["bq"][c, pos] * fd["lanes"][pos], 0.0)
    v = np.where(m, np.clip(pv - dv, SPEED_MIN, SPEED_MAX), pv)
    q = np.where(m, np.maximum(pq - dq, 0.0), pq)
    return v, q, m & ((dv != 0) | (dq != 0))


def state_score(C, v, q, fd):
    """S_state proxy of one corridor-month: mean over the regimes present of 0.54 S_speed + 0.46 S_flow."""
    lanes = fd["lanes"][C.pos.to_numpy(np.int64)]
    reg = C.regime.to_numpy()
    ev2 = (v - C.ov.to_numpy(float)) ** 2
    eq2 = ((q - C.oq.to_numpy(float)) / lanes) ** 2
    out = {}
    s = []
    for r in range(3):
        m = reg == r
        if not m.any():
            continue
        rv, rq = np.sqrt(ev2[m].mean()), np.sqrt(eq2[m].mean())
        sr = 0.54 * max(0.0, 1 - rv / 25) + 0.46 * max(0.0, 1 - rq / 600)
        out[f"R{r + 1}_rv"], out[f"R{r + 1}_rq"], out[f"R{r + 1}_S"] = rv, rq, sr
        s.append(sr)
    out["S"] = float(np.mean(s))
    return out


def t3_score(C, T, Dd, v, q, fd):
    """S_FD, LWR proxy and S_physics proxy of the hide-run frame with the hidden cells set to (v, q)."""
    pos = C.pos.to_numpy(np.int64)
    fn, fdd = fd_terms(v, q, pos, fd)
    with np.errstate(invalid="ignore", divide="ignore"):
        Nc = q / np.maximum(v, 1.0) * fd["length"][pos]
    h0, h1 = T.h0.to_numpy(), T.h1.to_numpy()
    N0 = np.where(h0 >= 0, Nc[np.maximum(h0, 0)], T.N0.to_numpy())
    N1 = np.where(h1 >= 0, Nc[np.maximum(h1, 0)], T.N1.to_numpy())
    res = np.abs((N1 - N0) - (T.N1.to_numpy() - T.N0.to_numpy()))
    reg_day = dict(zip(Dd.day, Dd.regime))
    creg = C.regime.to_numpy()
    treg = T.day.map(reg_day).to_numpy()
    out, phys = {}, []
    for r in range(3):
        dm = Dd.regime == r
        if not dm.any():
            continue
        cm = creg == r
        s_fd = 1 - (Dd.fd_num_rest[dm].sum() + fn[cm].sum()) / max(Dd.fd_den_rest[dm].sum() + fdd[cm].sum(), 1e-9)
        e = res[treg == r].sum() / max(Dd.lwr_den[dm].sum(), 1e-9)
        s_lwr = max(0.0, 1 - min(1.0, e + LWR_FLOOR))
        out[f"R{r + 1}_FD"], out[f"R{r + 1}_E"] = s_fd, e
        phys.append((s_fd + 2 * s_lwr) / 3)
        out[f"R{r + 1}_fd_num_hidden"] = float(fn[cm].sum())
        out[f"R{r + 1}_lwr_num"] = float(res[treg == r].sum())
    out["FD"] = float(np.mean([out[k] for k in out if k.endswith("_FD")]))
    out["E"] = float(np.mean([out[k] for k in out if k.endswith("_E")]))
    out["phys"] = float(np.mean(phys))
    return out


def load_hide(p, split):
    C = pd.read_parquet(HIDE / f"{p}_{split}_cells.parquet")
    T = pd.read_parquet(HIDE / f"{p}_{split}_trans.parquet")
    Dd = pd.read_csv(HIDE / f"{p}_{split}_days.csv")
    return C, T, Dd


def crossfit_pred(C, fd, cfg, key="pv", qkey="pq"):
    """Every hidden cell corrected with the bias estimated on the other day-index parity."""
    par = (C.day.to_numpy() % 2)
    v = C[key].to_numpy(float).copy()
    q = C[qkey].to_numpy(float).copy()
    pos = C.pos.to_numpy(np.int64)
    changed = np.zeros(len(C), bool)
    for a in (0, 1):
        A = C[par == a]
        if key != "pv":
            A = A.assign(pv=A[key], pq=A[qkey])
        b = estimate(A, fd, cfg)
        m = par != a
        vv, qq, ch = correct(v[m], q[m], pos[m], fd, b, cfg["sel"])
        v[m], q[m], changed[m] = vv, qq, ch
    return v, q, changed


def cfg_name(c):
    s = f"{c['sel']}_{c['ch']}_K{c['K']}"
    if c.get("lam", 1.0) != 1.0:
        s += f"_lam{c['lam']}"
    if "Kq" in c:
        s += f"_Kq{c['Kq']}"
    return s


def family_mean(d: dict) -> float:
    fam = {}
    for p, x in d.items():
        fam.setdefault(FAMILY[p], []).append(x)
    return float(np.mean([np.mean(v) for v in fam.values()]))


def t3_scales(Dd: pd.DataFrame) -> dict:
    """Per-regime factors mapping a Task 3 proxy change measured on the hide-run frame (only the hidden cells
    corrected) to the real frame (all targets corrected).  These are ESTIMATES, not measurements:
      S_FD : the change is linear in the number of corrected cells, same frame size -> n_target / n_hidden
      LWR  : E = sum|res| / sum|dN| over five-minute transitions; only transitions touching a corrected cell
             change.  Hide frame: n_trans_hidden of the n_trans visible-visible transitions are touched;
             real frame: n_trans_target of (n_trans + n_trans_target) transitions -> factor below."""
    out = {}
    for r in range(3):
        d = Dd[Dd.regime == r]
        if len(d) == 0:
            continue
        sfd = d.n_target.sum() / max(d.n_hidden.sum(), 1)
        nt, nh, ntt = d.n_trans.sum(), d.n_trans_hidden.sum(), d.n_trans_target.sum()
        out[r] = (float(sfd), float(ntt * nt / max((nt + ntt) * nh, 1)))
    return out


def t3_delta(a0: dict, a1: dict, sc: dict) -> dict:
    """Raw (hide frame) and scaled (real-frame estimate) Task 3 proxy changes, mean over the regimes present."""
    acc = {k: [] for k in ("dFD", "dE", "dphys", "dFD_est", "dE_est", "dphys_est")}
    for r, (sfd, se) in sc.items():
        f = a1[f"R{r + 1}_FD"] - a0[f"R{r + 1}_FD"]
        e = a1[f"R{r + 1}_E"] - a0[f"R{r + 1}_E"]
        acc["dFD"].append(f)
        acc["dE"].append(e)
        acc["dphys"].append((f - 2 * e) / 3)          # S_LWR = 1 - (E + floor), far from its clip
        acc["dFD_est"].append(f * sfd)
        acc["dE_est"].append(e * se)
        acc["dphys_est"].append((f * sfd - 2 * e * se) / 3)
    return {k: float(np.mean(v)) for k, v in acc.items()}


def main_grid():
    cfgs = []
    for sel in ("free", "all", "fc"):
        for ch in ("v", "vq", "q"):
            for K in (0, 10, 25, 50, 100, 200, 400, 800, 1600):
                cfgs.append({"sel": sel, "ch": ch, "K": K})
    # flow bias is noisier than speed bias: separate (stronger) flow shrinkage Kq
    for sel in ("free", "all", "fc"):
        for K in (10, 25, 50, 100):
            for Kq in (100, 200, 800, 3200):
                cfgs.append({"sel": sel, "ch": "vq", "K": K, "Kq": Kq})
    return cfgs


def lam_grid(base: dict):
    return [{**base, "lam": l} for l in (0.5, 0.75, 0.9, 1.1, 1.25, 1.5)]


W_STATE, W_PHYS = 0.35, 0.15


def eval_cfg(C, T, Dd, fd, cfg, s0, a0, sc):
    v, q, chg = crossfit_pred(C, fd, cfg)
    s1 = state_score(C, v, q, fd)
    a1 = t3_score(C, T, Dd, v, q, fd)
    t3 = t3_delta(a0, a1, sc)
    row = {"cfg": cfg_name(cfg), "S0": s0["S"], "S1": s1["S"], "dS": s1["S"] - s0["S"],
           "changed_share": float(chg.mean()), **t3,
           "dTotal_est": W_STATE * (s1["S"] - s0["S"]) + W_PHYS * t3["dphys_est"],
           "dTotal_raw": W_STATE * (s1["S"] - s0["S"]) + W_PHYS * t3["dphys"],
           "FD0": a0["FD"], "FD1": a1["FD"], "E0": a0["E"], "E1": a1["E"]}
    for r in range(3):
        if f"R{r + 1}_S" in s0:
            for k in ("rv", "rq", "S"):
                row[f"R{r + 1}_{k}_0"], row[f"R{r + 1}_{k}_1"] = s0[f"R{r + 1}_{k}"], s1[f"R{r + 1}_{k}"]
    return row


def summarise(G_: pd.DataFrame) -> pd.DataFrame:
    agg = []
    for (cfg, split), g in G_.groupby(["cfg", "split"]):
        rec = {"cfg": cfg, "split": split, "n_pos": int((g.dS > 0).sum()), "dS_min": g.dS.min()}
        for k in ("dS", "dFD", "dE", "dphys", "dFD_est", "dE_est", "dphys_est", "dTotal_est", "dTotal_raw"):
            rec[f"{k}_family"] = family_mean(dict(zip(g.panel, g[k])))
        agg.append(rec)
    A = pd.DataFrame(agg).pivot_table(index="cfg", columns="split",
                                      values=[c for c in agg[0] if c not in ("cfg", "split")])
    A.columns = [f"{a}_{b}" for a, b in A.columns]
    for k in ("dS", "dphys_est", "dTotal_est", "dTotal_raw"):
        A[f"{k}_family_both"] = (A[f"{k}_family_validation"] + A[f"{k}_family_private"]) / 2
    return A.sort_values("dTotal_est_family_both", ascending=False)


def cmd_crossfit(args) -> None:
    """Odd/even-day cross-fit of every config; S_state proxy and Task 3 proxy per corridor-month."""
    OUT.mkdir(parents=True, exist_ok=True)
    for f in ("crossfit_grid.csv", "crossfit_summary.csv", "crossfit_choice.json", "crossfit_lambda.csv"):
        if (OUT / f).exists():
            raise FileExistsError(OUT / f)
    t0 = time.time()
    rows, data = [], {}
    grid_ = main_grid()
    for p in PANELS:
        fd = static_fd(p)
        for split in SPLITS:
            C, T, Dd = load_hide(p, split)
            C = C[["day", "regime", "t", "pos", "ov", "oq", "pv", "pq"]]
            pv, pq_ = C.pv.to_numpy(float), C.pq.to_numpy(float)
            s0, a0, sc = state_score(C, pv, pq_, fd), t3_score(C, T, Dd, pv, pq_, fd), t3_scales(Dd)
            data[p, split] = (C, T, Dd, fd, s0, a0, sc)
            for cfg in grid_:
                rows.append({"panel": p, "split": split, **eval_cfg(C, T, Dd, fd, cfg, s0, a0, sc)})
        print("crossfit", p, f"{time.time() - t0:.0f}s", flush=True)
    G_ = pd.DataFrame(rows)
    A = summarise(G_)
    pd.set_option("display.width", 250)
    show = ["dS_family_validation", "dS_family_private", "n_pos_validation", "n_pos_private", "dS_min_validation",
            "dS_min_private", "dphys_est_family_both", "dTotal_est_family_both", "dTotal_raw_family_both"]
    print(A[show].head(25).round(6).to_string(), flush=True)
    # predeclared rule: maximise the family-mean cross-fitted total-score estimate 0.35*dS_state + 0.15*dS_phys(est),
    # averaged over March and April, over the main grid.  Ties within 1e-6 go to the simpler config
    # (speed only before speed+flow, larger K).
    best = A.index[0]
    cfg = next(c for c in grid_ if cfg_name(c) == best)
    best_state_only = A.sort_values("dS_family_both", ascending=False).index[0]
    # best K per (selection, channels): does adjusting flows help, does correcting congested cells help?
    A2 = A.copy()
    A2["sel"] = [c.split("_")[0] for c in A2.index]
    A2["ch"] = [c.split("_")[1] + ("+Kq" if "_Kq" in c else "") for c in A2.index]
    per_family = A2.sort_values("dTotal_est_family_both", ascending=False).groupby(["sel", "ch"]).head(1)
    print(per_family[["sel", "ch"] + show].round(6).to_string(), flush=True)
    # nested month check: choose on one month only, report the other
    nested = {}
    for sel_split, rep_split in (("validation", "private"), ("private", "validation")):
        pick = A[f"dTotal_est_family_{sel_split}"].idxmax()
        nested[f"choose_on_{sel_split}"] = {
            "cfg": pick, f"dTotal_est_family_{rep_split}": float(A.loc[pick, f"dTotal_est_family_{rep_split}"]),
            f"dTotal_est_family_{rep_split}_of_global_choice": float(A.loc[best, f"dTotal_est_family_{rep_split}"]),
            f"dS_family_{rep_split}": float(A.loc[pick, f"dS_family_{rep_split}"])}
    # lambda curve around the chosen config (reported only, not used for the choice)
    lrows = []
    for (p, split), (C, T, Dd, fd, s0, a0, sc) in data.items():
        for c in lam_grid(cfg):
            lrows.append({"panel": p, "split": split, **eval_cfg(C, T, Dd, fd, c, s0, a0, sc)})
    L_ = pd.DataFrame(lrows)
    G_.to_csv(OUT / "crossfit_grid.csv", index=False)
    A.to_csv(OUT / "crossfit_summary.csv")
    L_.to_csv(OUT / "crossfit_lambda.csv", index=False)
    LA = summarise(pd.concat([L_, G_[G_.cfg == best]]))
    print(LA[show].round(6).to_string(), flush=True)
    chosen = G_[G_.cfg == best].set_index(["split", "panel"])
    per_cm = {s: {p: {k: float(chosen.loc[(s, p), k]) for k in
                      ("S0", "S1", "dS", "dFD", "dE", "dphys", "dphys_est", "dTotal_est", "changed_share")}
                  for p in PANELS} for s in SPLITS}
    reg_rows = {}
    for s in SPLITS:
        for r in ("R1", "R2", "R3"):
            g = chosen.loc[s]
            if f"{r}_S_0" not in g:
                continue
            g = g[g[f"{r}_S_0"].notna()]
            reg_rows[f"{s}|{r}"] = {"corridors": int(len(g)), "dS_mean": float((g[f"{r}_S_1"] - g[f"{r}_S_0"]).mean()),
                                    "speed_rmse_0": float(g[f"{r}_rv_0"].mean()), "speed_rmse_1": float(g[f"{r}_rv_1"].mean()),
                                    "flow_rmse_lane_0": float(g[f"{r}_rq_0"].mean()), "flow_rmse_lane_1": float(g[f"{r}_rq_1"].mean())}
    choice = {"rule": "maximise the family-mean cross-fitted 0.35*dS_state + 0.15*dS_physics(real-frame estimate), "
                      "averaged over March and April, over the main grid (sel x channels x K [x Kq]); "
                      "lambda curve reported only",
              "chosen": cfg, "chosen_name": best, "best_by_state_only": best_state_only,
              "dS_family": {s: float(A.loc[best, f"dS_family_{s}"]) for s in SPLITS},
              "dS_min_corridor": {s: float(A.loc[best, f"dS_min_{s}"]) for s in SPLITS},
              "n_positive_corridors": {s: int(A.loc[best, f"n_pos_{s}"]) for s in SPLITS},
              "dphys_family_raw_hide_frame": {s: float(A.loc[best, f"dphys_family_{s}"]) for s in SPLITS},
              "dphys_family_real_frame_est": {s: float(A.loc[best, f"dphys_est_family_{s}"]) for s in SPLITS},
              "dFD_family_real_frame_est": {s: float(A.loc[best, f"dFD_est_family_{s}"]) for s in SPLITS},
              "dE_family_real_frame_est": {s: float(A.loc[best, f"dE_est_family_{s}"]) for s in SPLITS},
              "dTotal_family_est": {s: float(A.loc[best, f"dTotal_est_family_{s}"]) for s in SPLITS},
              "per_corridor_month": per_cm, "per_regime": reg_rows,
              "best_per_selection_and_channels": per_family[["sel", "ch"] + show].reset_index().to_dict("records"),
              "nested_month_check": nested,
              "lambda_curve": LA[show].reset_index().to_dict("records"),
              "inputs_sha256": {f"{p}_{s}": sha256(HIDE / f"{p}_{s}_cells.parquet") for p in PANELS for s in SPLITS},
              "seconds": round(time.time() - t0)}
    (OUT / "crossfit_choice.json").write_text(json.dumps(choice, indent=2))
    print(json.dumps({k: choice[k] for k in ("chosen_name", "best_by_state_only", "dS_family", "dS_min_corridor",
                                             "n_positive_corridors", "dphys_family_real_frame_est",
                                             "dTotal_family_est", "nested_month_check", "per_regime")}, indent=2),
          flush=True)


# ---------------------------------------------------------------------------
# apply to the V51 state rows
# ---------------------------------------------------------------------------
def read_key_state():
    """State rows of submission_key.csv (categoricals, ~125 MB) with the integer cell key."""
    import pyarrow.csv as pc
    import pyarrow.compute as pcc
    tb = pc.read_csv(RELEASE / "submission_key.csv", convert_options=pc.ConvertOptions(
        include_columns=["submission_id", "task", "split", "panel", "timestamp", "station_id", "link_id", "mask_regime"],
        column_types={"timestamp": pa.dictionary(pa.int32(), pa.string())}, auto_dict_encode=True,
        auto_dict_max_cardinality=10 ** 7, strings_can_be_null=False))
    K = tb.filter(pcc.equal(tb["task"].cast(pa.string()), "state")).to_pandas().drop(columns="task")
    del tb
    slots = slot_of(pd.Series(K.timestamp.cat.categories))
    slot = slots[K.timestamp.cat.codes.to_numpy()]
    key = np.full(len(K), -1, np.int64)
    for p, idx in K.groupby("panel", observed=True).indices.items():
        lp = K.link_id.iloc[idx]
        m = pd.Series(lp.cat.categories).map(dict(zip(links(p).link_id, links(p).pos)))
        pos = m.to_numpy(float)[lp.cat.codes.to_numpy()]
        assert np.isfinite(pos).all(), p
        key[idx] = slot[idx] * KEY_BASE + pos.astype(np.int64)
    K["key"] = key
    assert K.submission_id.is_monotonic_increasing
    return K


def read_zip_floats(path: Path) -> pd.DataFrame:
    """All rows of a submission ZIP with exact (round-trip) floats."""
    with zipfile.ZipFile(path) as z:
        assert len(z.namelist()) == 1
        with z.open(z.namelist()[0]) as f:
            return pd.read_csv(f, float_precision="round_trip",
                               dtype={"submission_id": np.int64, "task": "category", "speed_kmh": np.float64,
                                      "flow_vph": np.float64, "queue_pred": np.int64, "path_flow": np.float64})


def cmd_apply(args) -> None:
    import t1_assemble as A
    choice = json.loads((OUT / "crossfit_choice.json").read_text())
    cfg = choice["chosen"]
    paths = {s: WORK / "sub" / f"state_{s}_v52bias.csv" for s in SPLITS}
    for pth in paths.values():
        if pth.exists():
            raise FileExistsError(pth)
    t0 = time.time()
    K = read_key_state()
    Z = read_zip_floats(BASE_ZIP)
    Z = Z[Z.task == "state"]
    assert np.array_equal(Z.submission_id.to_numpy(), K.submission_id.to_numpy())
    K["speed"], K["flow"] = Z.speed_kmh.to_numpy(), Z.flow_vph.to_numpy()
    del Z
    print("key + zip state rows", len(K), f"{time.time() - t0:.0f}s", flush=True)
    rows, bias_rows, fd_rows = [], [], []
    for split in SPLITS:
        S_ = K[K.split == split].reset_index(drop=True)
        v_all, q_all = S_.speed.to_numpy().copy(), S_.flow.to_numpy().copy()
        for p, idx in S_.groupby("panel", observed=True).indices.items():
            fd = static_fd(p)
            C, T, Dd = load_hide(p, split)
            b = estimate(C, fd, cfg)                       # all days of this month, this panel
            tg = t1_targets(p, split)
            assert len(idx) == len(tg["key"]), p
            blk = A.target_blackout(p, split)
            kk = S_.key.to_numpy()[idx]
            j = np.searchsorted(tg["key"], kk)
            assert np.array_equal(tg["key"][j], kk), p
            reg = S_.mask_regime.iloc[idx].astype(str).map({r: i for i, r in enumerate(REGIMES)}).to_numpy()
            assert np.array_equal(reg, tg["regime"][j]), p
            pos = (kk % KEY_BASE).astype(np.int64)
            v0, q0 = S_.speed.to_numpy()[idx], S_.flow.to_numpy()[idx]
            v1, q1, chg = correct(v0, q0, pos, fd, b, cfg["sel"])
            gap = blk[j].astype(bool)
            v1[gap], q1[gap] = v0[gap], q0[gap]
            chg &= ~gap
            # values go through the submission's 6-significant-digit text format
            v1 = np.where(chg, np.array([float("%.6g" % x) for x in v1]), v0)
            q1 = np.where(chg, np.array([float("%.6g" % x) for x in q1]), q0)
            assert np.isfinite(v1).all() and np.isfinite(q1).all() and (v1 > 0).all() and (q1 >= 0).all()
            v_all[idx], q_all[idx] = v1, q1
            L = links(p)
            for c in (0, 1):
                for l in range(fd["n"]):
                    if b["n"][c, l] > 0:
                        bias_rows.append({"split": split, "panel": p, "class": c, "pos": l, "link_id": L.link_id.iloc[l],
                                          "n": int(b["n"][c, l]), "bias_speed_kmh": float(b["bv"][c, l]),
                                          "bias_flow_vph_lane": float(b["bq"][c, l])})
            fdr = fd_frame(p, split, tg, j, v0, q0, v1, q1, fd)
            fd_rows += [{**r, "split": split, "panel": p} for r in fdr]
            ch_ = chg & ((v1 != v0) | (q1 != q0))
            rows.append({"split": split, "panel": p, "targets": len(idx), "blackout_kept": int(gap.sum()),
                         "corrected": int(ch_.sum()), "corrected_share": float(ch_.mean()),
                         "mean_speed_change": float(np.mean((v1 - v0)[ch_])) if ch_.any() else 0.0,
                         "mean_abs_speed_change": float(np.mean(np.abs(v1 - v0)[ch_])) if ch_.any() else 0.0,
                         "max_abs_speed_change": float(np.max(np.abs(v1 - v0))),
                         "mean_abs_flow_change": float(np.mean(np.abs(q1 - q0)[ch_])) if ch_.any() else 0.0,
                         "max_abs_flow_change": float(np.max(np.abs(q1 - q0))),
                         "bias_rms_speed_free_links": float(np.sqrt(np.average(b["bv"][0] ** 2, weights=np.maximum(b["n"][0], 1e-9)))),
                         "low_flow_share_before": float((q0 < 50).mean()), "low_flow_share_after": float((q1 < 50).mean()),
                         "speed_min_after": float(v1.min()), "speed_max_after": float(v1.max())})
            print(split, p, rows[-1]["corrected"], "corrected", f"{time.time() - t0:.0f}s", flush=True)
        out = S_[["panel", "timestamp", "station_id", "link_id", "mask_regime"]].copy()
        out["speed_kmh"], out["flow_vph"] = v_all, q_all
        out.to_csv(paths[split], index=False, float_format="%.6g")
        print("wrote", paths[split], len(out), f"{time.time() - t0:.0f}s", flush=True)
        del S_, out
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "apply_changes.csv", index=False)
    pd.DataFrame(bias_rows).to_csv(OUT / "apply_bias_table.csv", index=False)
    F = pd.DataFrame(fd_rows)
    F.to_csv(OUT / "apply_fd_frame.csv", index=False)
    fdp = F.groupby(["split", "panel"])[["FD0", "FD1"]].mean()
    fdp["dFD"] = fdp.FD1 - fdp.FD0
    summ = {"config": cfg, "changes": rows,
            "S_FD_real_frame_family_mean": {s: {"before": family_mean(fdp.loc[s].FD0.to_dict()),
                                                "after": family_mean(fdp.loc[s].FD1.to_dict()),
                                                "delta": family_mean(fdp.loc[s].dFD.to_dict())} for s in SPLITS},
            "outputs_sha256": {s: sha256(paths[s]) for s in SPLITS}, "base_zip_sha256": sha256(BASE_ZIP)}
    (OUT / "apply_summary.json").write_text(json.dumps(summ, indent=2))
    print(R.round(4).to_string(), flush=True)
    print(json.dumps(summ["S_FD_real_frame_family_mean"], indent=2), flush=True)


def fd_frame(p, split, tg, j, v0, q0, v1, q1, fd):
    """S_FD per regime on the real frame: visible observations + submitted target values (before/after)."""
    n = fd["n"]
    pf = pq.ParquetFile(CLEAN / p / f"{split}.parquet")
    # submitted values in t1_targets order
    tv0 = np.empty(len(tg["key"])); tq0 = np.empty(len(tg["key"])); tv1 = np.empty(len(tg["key"])); tq1 = np.empty(len(tg["key"]))
    tv0[j], tq0[j], tv1[j], tq1[j] = v0, q0, v1, q1
    acc = {r: [0.0, 0.0, 0.0, 0.0, 0, 0, 0] for r in range(3)}
    ptr = 0
    for i in range(pf.num_row_groups):
        t = pf.read_row_group(i, columns=["mask_regime", "is_score_eligible", "is_t1_target", "speed_obs", "flow_obs"])
        r = REGIMES.index(t.column("mask_regime")[0].as_py())
        tgt = t.column("is_t1_target").to_numpy()
        k = int(tgt.sum())
        elig = t.column("is_score_eligible").to_numpy()
        pos = np.tile(np.arange(n), STEPS)
        for which, (a, bq_) in enumerate(((tv0, tq0), (tv1, tq1))):
            v = t.column("speed_obs").to_numpy().astype(float)
            q = t.column("flow_obs").to_numpy().astype(float)
            v[tgt], q[tgt] = a[ptr:ptr + k], bq_[ptr:ptr + k]
            use = elig & np.isfinite(v) & np.isfinite(q)
            fn, fdd = fd_terms(v[use], q[use], pos[use], fd)
            acc[r][2 * which] += float(fn.sum())
            acc[r][2 * which + 1] += float(fdd.sum())
            if which == 0:
                acc[r][4] += int((tgt & use).sum())
                acc[r][5] += int(((q < 50) & tgt & use).sum())
            else:
                acc[r][6] += int(((q < 50) & tgt & use).sum())
        ptr += k
    out = []
    for r, a in acc.items():
        if a[1] == 0:
            continue
        out.append({"regime": REGIMES[r], "FD0": 1 - a[0] / a[1], "FD1": 1 - a[2] / a[3], "targets": a[4],
                    "low_share0": a[5] / max(a[4], 1), "low_share1": a[6] / max(a[4], 1)})
    return out


# ---------------------------------------------------------------------------
# package + audit
# ---------------------------------------------------------------------------
def new_state_arrays():
    """speed / flow of the new state CSVs indexed by submission_id (checked against the key order)."""
    K = read_key_state()
    sp = np.full(int(K.submission_id.max()) + 1, np.nan)
    fl = sp.copy()
    for s in SPLITS:
        d = pd.read_csv(WORK / "sub" / f"state_{s}_v52bias.csv", float_precision="round_trip",
                        dtype={"panel": "category", "timestamp": "category", "station_id": "category",
                               "link_id": "category", "mask_regime": "category"})
        ks = K[K.split == s]
        assert len(d) == len(ks)
        for c in ("panel", "timestamp", "station_id", "link_id", "mask_regime"):
            assert np.array_equal(d[c].astype(str).to_numpy(), ks[c].astype(str).to_numpy()), (s, c)
        ids = ks.submission_id.to_numpy()
        sp[ids], fl[ids] = d.speed_kmh.to_numpy(), d.flow_vph.to_numpy()
        del d
    return sp, fl, K


def cmd_package(args) -> None:
    from make_submission import validate
    out = WORK / "submissions" / f"{NAME}.zip"
    if out.exists():
        raise FileExistsError(out)
    man_path = WORK / "results" / f"{NAME}_manifest.json"
    if man_path.exists():
        raise FileExistsError(man_path)
    t0 = time.time()
    sp, fl, K = new_state_arrays()
    del K
    print("new state arrays", f"{time.time() - t0:.0f}s", flush=True)
    h_src, h_out = hashlib.sha256(), hashlib.sha256()
    total = changed = state_rows = 0
    with zipfile.ZipFile(BASE_ZIP) as zin:
        with zin.open(zin.namelist()[0]) as src, zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zout:
            with zout.open(f"{NAME}.csv", "w", force_zip64=True) as dst:
                header = src.readline()
                assert header.strip() == b"submission_id,task,speed_kmh,flow_vph,queue_pred,path_flow"
                dst.write(header)
                for line in src:
                    total += 1
                    if line.split(b",", 2)[1] == b"state":
                        body = line.rstrip(b"\r\n")
                        parts = body.split(b",")
                        assert len(parts) == 6, total
                        sid = int(parts[0])
                        state_rows += 1
                        a, b = sp[sid], fl[sid]
                        if a != float(parts[2]) or b != float(parts[3]):
                            changed += 1
                            parts[2], parts[3] = (b"%.6g" % a), (b"%.6g" % b)
                            line = b",".join(parts) + line[len(body):]
                    else:
                        h_src.update(line)
                        h_out.update(line)
                    dst.write(line)
    assert state_rows == int(np.isfinite(sp).sum()), "state rows of the base ZIP and the CSVs differ"
    with zipfile.ZipFile(out) as z:
        assert z.testzip() is None
    checks = validate(out)
    print("checks:", checks, flush=True)
    if not checks["ok"]:
        raise SystemExit(f"validation failed: {checks}")
    choice = json.loads((OUT / "crossfit_choice.json").read_text())
    manifest = {"name": NAME, "baseline": str(BASE_ZIP), "baseline_sha256": sha256(BASE_ZIP),
                "note": "V51 queue/ODME rows byte-identical; Task 1 state = V51 state minus a per-month (March for "
                        "validation rows, April for private rows) per-link bias of the production chain, estimated "
                        "from each month's own visible observations (hide-20% protocol); shrinkage/selection chosen "
                        "by odd/even-day cross-fitting; Task 2 blackout target cells unchanged",
                "config": choice["chosen"], "crossfit_dS_state_family": choice["dS_family"],
                "state_inputs": [{"path": str(WORK / "sub" / f"state_{s}_v52bias.csv"),
                                  "sha256": sha256(WORK / "sub" / f"state_{s}_v52bias.csv")} for s in SPLITS],
                "output": str(out), "sha256": sha256(out), "rows": total, "state_rows": state_rows,
                "changed_state_rows": changed, "queue_odme_rows_identical": h_src.hexdigest() == h_out.hexdigest(),
                "nonstate_rows_sha256": h_out.hexdigest(), "checks": checks, "public_lb": None,
                "seconds": round(time.time() - t0)}
    man_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUT / f"{NAME}_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)


def cmd_audit(args) -> None:
    """Independent of package(): reread both ZIPs, the key and the CSVs; compare every row."""
    new = WORK / "submissions" / f"{NAME}.zip"
    path = OUT / "audit.json"
    if path.exists():
        raise FileExistsError(path)
    sp, fl, K = new_state_arrays()
    split_arr = np.empty(len(sp), object)
    panel_arr = np.empty(len(sp), object)
    split_arr[K.submission_id.to_numpy()] = K.split.astype(str).to_numpy()
    panel_arr[K.submission_id.to_numpy()] = K.panel.astype(str).to_numpy()
    del K
    stats = {"rows": 0, "state_rows": 0, "state_changed": 0, "queue_changed": 0, "odme_changed": 0,
             "nonstate_bytes_differ": 0, "state_value_mismatch_vs_csv": 0, "state_text_not_6g": 0,
             "max_abs_speed_change": 0.0, "max_abs_flow_change": 0.0, "changed_by_split_panel": {}}
    with zipfile.ZipFile(BASE_ZIP) as za, zipfile.ZipFile(new) as zb:
        assert za.testzip() is None and zb.testzip() is None
        with za.open(za.namelist()[0]) as fa, zb.open(zb.namelist()[0]) as fb, \
                open(RELEASE / "submission_key.csv", "rb") as fk:
            fa.readline(); fb.readline(); fk.readline()
            for la, lb, lk in zip(fa, fb, fk):
                stats["rows"] += 1
                pa_, pb = la.rstrip(b"\r\n").split(b","), lb.rstrip(b"\r\n").split(b",")
                kk = lk.split(b",", 2)
                assert pa_[0] == pb[0] == kk[0] and pa_[1] == pb[1] == kk[1], stats["rows"]
                if pa_[1] == b"state":
                    sid = int(pa_[0])
                    stats["state_rows"] += 1
                    nv, nq = float(pb[2]), float(pb[3])
                    if nv != sp[sid] or nq != fl[sid]:
                        stats["state_value_mismatch_vs_csv"] += 1
                    if pa_[2:] != pb[2:]:
                        if (b"%.6g" % nv) != pb[2] or (b"%.6g" % nq) != pb[3]:
                            stats["state_text_not_6g"] += 1
                        stats["state_changed"] += 1
                        k = f"{split_arr[sid]}|{panel_arr[sid]}"
                        stats["changed_by_split_panel"][k] = stats["changed_by_split_panel"].get(k, 0) + 1
                        stats["max_abs_speed_change"] = max(stats["max_abs_speed_change"], abs(nv - float(pa_[2])))
                        stats["max_abs_flow_change"] = max(stats["max_abs_flow_change"], abs(nq - float(pa_[3])))
                elif la != lb:
                    stats["nonstate_bytes_differ"] += 1
                    stats["queue_changed" if pa_[1] == b"queue" else "odme_changed"] += 1
    stats["valid"] = bool(stats["rows"] == 6980503 and stats["nonstate_bytes_differ"] == 0 and
                          stats["state_value_mismatch_vs_csv"] == 0 and stats["state_text_not_6g"] == 0 and
                          stats["state_changed"] > 0)
    stats["base_sha256"], stats["new_sha256"] = sha256(BASE_ZIP), sha256(new)
    path.write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2), flush=True)
    assert stats["valid"]


def direct_panel(p, split, pv_t, thr_ratio=0.85, minc=5):
    """Does the hide-protocol bias represent the bias at the real targets?  Targets are hash-random eligible
    cells, so within a (link, day) the truth at targets and the observations at visible cells share one time
    distribution:  b_direct(l) = mean_day[ mean(prod pred at deep-free targets) - mean(obs at deep-free
    visible cells) ] estimates the production bias at the real targets with no extra hiding.  The same
    estimator on the hidden cells (pred at hidden vs obs at the other visible cells) is checked against the
    measured hidden-cell bias b_hide.  Deep free: value > 0.85 * free speed; Task 2 blackout targets excluded."""
    fd = static_fd(p)
    n, ffs = fd["n"], fd["ffs"]
    C, T, Dd = load_hide(p, split)
    pf = pq.ParquetFile(CLEAN / p / f"{split}.parquet")
    D = pf.num_row_groups
    acc = {k: np.zeros((D, n)) for k in ("st", "nt", "sv", "nv", "sh", "nh", "so", "no")}
    hid = {d: g for d, g in C.groupby("day")}
    ptr = 0
    thr = thr_ratio * ffs[None, :]
    for d in range(D):
        t = pf.read_row_group(d, columns=["is_t1_target", "is_score_eligible", "is_t2_blackout", "speed_obs", "flow_obs"])
        g_ = lambda c: t.column(c).to_numpy().reshape(STEPS, n)
        tgt, blk, el = g_("is_t1_target"), g_("is_t2_blackout"), g_("is_score_eligible")
        ov, oq = g_("speed_obs").astype(float), g_("flow_obs").astype(float)
        k = int(tgt.sum())
        P = np.full((STEPS, n), np.nan)
        P[tgt] = pv_t[ptr:ptr + k]
        ptr += k
        mt = tgt & ~blk & (P > thr)
        acc["st"][d], acc["nt"][d] = np.where(mt, P, 0).sum(0), mt.sum(0)
        vis = np.isfinite(ov) & np.isfinite(oq) & el
        mv = vis & (ov > thr)
        acc["sv"][d], acc["nv"][d] = np.where(mv, ov, 0).sum(0), mv.sum(0)
        H = np.zeros((STEPS, n), bool)
        if d in hid:
            tt, pp_ = hid[d].t.to_numpy(int), hid[d].pos.to_numpy(int)
            H[tt, pp_] = True
            ph = np.full((STEPS, n), np.nan)
            ph[tt, pp_] = hid[d].pv.to_numpy(float)
            mh = H & (ph > thr)
            acc["sh"][d], acc["nh"][d] = np.where(mh, ph, 0).sum(0), mh.sum(0)
        mo = vis & ~H & (ov > thr)
        acc["so"][d], acc["no"][d] = np.where(mo, ov, 0).sum(0), mo.sum(0)
    assert ptr == len(pv_t)
    with np.errstate(invalid="ignore", divide="ignore"):
        dt_ = np.where((acc["nt"] >= minc) & (acc["nv"] >= minc), acc["st"] / acc["nt"] - acc["sv"] / acc["nv"], np.nan)
        dh_ = np.where((acc["nh"] >= minc) & (acc["no"] >= minc), acc["sh"] / acc["nh"] - acc["so"] / acc["no"], np.nan)
        b_t, b_h = np.nanmean(dt_, 0), np.nanmean(dh_, 0)
    sel = (C.pv > thr_ratio * ffs[C.pos]).to_numpy()
    pos = C.pos.to_numpy()[sel]
    cnt = np.bincount(pos, minlength=n)
    b_hide = np.bincount(pos, weights=(C.pv - C.ov).to_numpy()[sel], minlength=n) / np.maximum(cnt, 1)
    ok = np.isfinite(b_t) & np.isfinite(b_h) & (cnt >= 50)
    sl = lambda y, x: float((x[ok] * y[ok]).sum() / (x[ok] ** 2).sum())
    rms = lambda a: float(np.sqrt((a[ok] ** 2).mean()))
    # the clean comparison is direct-at-targets vs direct-at-hidden (same estimator, same selection asymmetry):
    # if the *_vp nets had memorised the visible (= hidden) cells during their month adaptation, b_h would be
    # shrunk relative to b_t and the slope b_t on b_h would be > 1.
    return {"panel": p, "split": split, "links": int(ok.sum()), "rms_b_hide": rms(b_hide),
            "rms_direct_hidden": rms(b_h), "rms_direct_targets": rms(b_t),
            "corr_directhidden_vs_hide": float(np.corrcoef(b_h[ok], b_hide[ok])[0, 1]),
            "slope_directhidden_on_hide": sl(b_h, b_hide),
            "corr_directtargets_vs_hide": float(np.corrcoef(b_t[ok], b_hide[ok])[0, 1]),
            "slope_directtargets_on_hide": sl(b_t, b_hide),
            "corr_directtargets_vs_directhidden": float(np.corrcoef(b_t[ok], b_h[ok])[0, 1]),
            "slope_directtargets_on_directhidden": sl(b_t, b_h),
            "rms_residual_targets_after_hide_bias": float(np.sqrt(((b_t - b_hide)[ok] ** 2).mean())),
            "rms_residual_directtargets_minus_directhidden": float(np.sqrt(((b_t - b_h)[ok] ** 2).mean())),
            "mean_b_hide": float(b_hide[ok].mean()), "mean_direct_targets": float(b_t[ok].mean()),
            "mean_direct_hidden": float(b_h[ok].mean())}


def zip_state_in_target_order():
    """{(split, panel): (speed, flow)} of the V51 ZIP state rows, reordered to t1_targets order."""
    K = read_key_state()
    Z = read_zip_floats(BASE_ZIP)
    Z = Z[Z.task == "state"]
    assert np.array_equal(Z.submission_id.to_numpy(), K.submission_id.to_numpy())
    sp, fl = Z.speed_kmh.to_numpy(), Z.flow_vph.to_numpy()
    del Z
    out = {}
    for (split, p), idx in K.groupby(["split", "panel"], observed=True).indices.items():
        tg = t1_targets(p, split)
        kk = K.key.to_numpy()[idx]
        j = np.searchsorted(tg["key"], kk)
        assert len(idx) == len(tg["key"]) and np.array_equal(tg["key"][j], kk), (split, p)
        v, q = np.empty(len(idx)), np.empty(len(idx))
        v[j], q[j] = sp[idx], fl[idx]
        out[split, p] = (v, q)
    return out


def cmd_direct(args) -> None:
    path = OUT / "direct_check.csv"
    if path.exists():
        raise FileExistsError(path)
    rows = []
    state = zip_state_in_target_order()          # the exact rows that 'apply' corrects (V51 = V40-half state)
    for split in SPLITS:
        for p in PANELS:
            rows.append(direct_panel(p, split, state[split, p][0]))
            print(rows[-1], flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(path, index=False)
    pd.set_option("display.width", 250)
    print(d.round(3).to_string())
    print(d.groupby("split").mean(numeric_only=True).round(3).T.to_string())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("selftest")
    s.add_argument("--panel", default="D12_I405_S")
    s.add_argument("--split", default="validation", choices=SPLITS)
    s.add_argument("--day", type=int, default=0)
    s.add_argument("--threads", type=int, default=3)
    s = sub.add_parser("collect")
    s.add_argument("--split", required=True, choices=SPLITS)
    s.add_argument("--panels", nargs="+", default=PANELS)
    s.add_argument("--threads", type=int, default=3)
    for name in ("crossfit", "direct", "apply", "package", "audit"):
        sub.add_parser(name)
    args = ap.parse_args()
    {"selftest": cmd_selftest, "collect": cmd_collect, "crossfit": cmd_crossfit, "direct": cmd_direct,
     "apply": cmd_apply, "package": cmd_package, "audit": cmd_audit}[args.cmd](args)


if __name__ == "__main__":
    main()
