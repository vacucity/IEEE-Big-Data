"""V59 Task 1 (lane K): LOCAL neighbourhood / missingness features on top of V58 (state rows only).

V58 = v58_rs_blk.zip = V56 queue/ODME + V57_rs state on the free non-blackout target rows + V57_blk state on the
Task 2 blackout target rows.  V59 changes ONLY the rows V57_rs owns: targets predicted free by the production
state (V51 pv > 1.15 v_cut) and outside the official Task 2 blackouts.  Congested and blackout rows keep the V58
value byte for byte.

Model (per corridor-month, fitted on that month's hide-20% cells of t1_v52_monthbias.py collect, free cells):
  the V57 restack (t1_v57_restack.py: LightGBM on component deviations dg/du/dl, x, rel; target = link-demeaned
  production error) gets extra LOCAL inputs, then the V56 correction is refit on the restacked values (as V57):
    same link   db, da   5-min steps to the nearest visible cell before / after (capped at 13)
                dvb, dva nearest visible speed before / after minus the prediction; dvi = distance-weighted
                         interpolation of the two minus the prediction; dvb2/dva2 the second-nearest ones
                n3       visible cells of the same link in t-3..t+3
    adjacent    dvu, dvd up/downstream link speed at t minus the prediction; dvu1/dvd1 mean of t-1, t+1;
                dvu2/dvd2 two links away; nsp visible up/down cells in t-1..t+1
    flow        the same with per-lane flows (dqb, dqa, dqi, dqb2, dqa2, dqu, dqd, dqu1, dqd1, dqu2, dqd2)
  Visibility = speed_obs and flow_obs finite (the production chain input).  At the hide cells it is the hide-run
  visibility (real visible cells minus the reproduced hide-20% mask, seed of t1_v52_monthbias), i.e. every
  feature is computed exactly as it would be at a real target of a frame with real mask + extra hide mask; at the
  real targets it is the real visibility.  Same day only.
At the real targets the restack inputs are the cached production components (as t1_v57_restack apply) and pv =
V51 state rows; the V57 pipeline (no local features) must reproduce the V58 state rows exactly (lineage check).
Each month uses only its own visible observations (Task 1 offline).  No Task 2 input or output is read except the
public blackout flag is_t2_blackout (to keep those rows); queue and ODME rows are not touched.

Subcommands (project root, anaconda python, OMP_NUM_THREADS=4):
  features   local quantities at hide cells (H) and real targets (T) -> work/cache/t1_v59_local/
  crossfit   pre-registered grid + choice + nested month check + lambda curve + per-regime RMSE + transfer
             metrics (hide cells reweighted to the real targets' neighbourhood strata; far / near subsets)
  direct     disjoint-feature neighbour-proxy check T vs H + strata-matched local-deviation distributions T vs H
  apply      fit on all days; writes work/sub/state_{split}_v59loc.csv from the V58 state rows; real-frame S_FD
  package    t1_v52_monthbias package code (imported, names patched) -> work/submissions/v59loc_state.zip
  audit      t1_v52_monthbias audit code (imported, names patched): v59loc_state.zip vs v51_april.zip
  compare    line-by-line v59_loc.zip (build_queue_candidate.py with the V54 queues) vs v58_rs_blk.zip
Outputs: work/results/t1_v59_local/.  Nothing existing is overwritten (FileExistsError).  See work/V59_LOCAL_README.md.
"""
from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "4")

import argparse
import gc
import json
import sys
import time
import warnings
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
import t1_v52_monthbias as M  # noqa: E402
import t1_v57_restack as V  # noqa: E402
from evaluate import CLEAN, KEY_BASE, PANELS, REGIMES, STEPS, t1_targets  # noqa: E402

WORK = Path(__file__).resolve().parent
OUT = WORK / "results" / "t1_v59_local"
CACHE = WORK / "cache" / "t1_v59_local"
SPLITS = M.SPLITS
V51_ZIP = WORK / "submissions" / "v51_april.zip"
V58_ZIP = WORK / "submissions" / "v58_rs_blk.zip"
NAME = "v59loc_state"
FULL_NAME = "v59_loc"
CSV = {s: WORK / "sub" / f"state_{s}_v59loc.csv" for s in SPLITS}
THREADS = 4
GBM_BASE = dict(V.GBM_BASE, num_threads=THREADS)
W_STATE, W_PHYS = 0.35, 0.15
NONE = 99
CAP = 12

LV = ("db", "da", "dvb", "dva", "dvi", "dvu", "dvd", "dvu1", "dvd1", "n3", "nsp")
LQ = ("db", "da", "dqb", "dqa", "dqi", "dqu", "dqd", "n3", "nsp")
XV = LV + ("dvb2", "dva2", "dvu2", "dvd2")
XQ = LQ + ("dqb2", "dqa2", "dqu1", "dqd1", "dqu2", "dqd2")
MV = XV + ("reg", "nblk", "db2", "da2")
MQ = XQ + ("reg", "nblk", "db2", "da2")
V57CFG = {"v": {"leaves": 15, "mdl": 300, "rounds": 100}, "q": {"leaves": 15, "mdl": 5000, "rounds": 50}}
B_ = V57CFG
CFGS = {
    "Lvq": {"v": {**B_["v"], "local": LV}, "q": {**B_["q"], "local": LQ}},
    "Lvq_x": {"v": {**B_["v"], "local": XV}, "q": {**B_["q"], "local": XQ}},
    "Lvq_xm": {"v": {**B_["v"], "local": MV}, "q": {**B_["q"], "local": MQ}},
    "Lvq_x_r200": {"v": {**B_["v"], "local": XV, "rounds": 200}, "q": {**B_["q"], "local": XQ}},
    "Lvq_x_q3000": {"v": {**B_["v"], "local": XV}, "q": {**B_["q"], "local": XQ, "mdl": 3000, "rounds": 100}},
    "Lvq_xbig": {"v": {**B_["v"], "local": XV, "leaves": 31, "rounds": 200},
                 "q": {**B_["q"], "local": XQ, "mdl": 3000, "rounds": 100}},
    "S2_x": {"two_stage": True, "v": {**B_["v"], "local": XV}, "q": {**B_["q"], "local": XQ}},
}
CPLX = ["Lvq", "Lvq_x", "Lvq_xm", "Lvq_x_r200", "Lvq_x_q3000", "Lvq_xbig", "S2_x"]
DPROXY = {"Tonly": ({**B_["v"], "local": ("db", "da", "dvb", "dva", "dvi", "n3", "dvb2", "dva2")},
                    {**B_["q"], "local": ("db", "da", "dqb", "dqa", "dqi", "n3", "dqb2", "dqa2")}, "spatial"),
          "Sonly": ({**B_["v"], "local": ("dvu", "dvd", "dvu1", "dvd1", "nsp", "dvu2", "dvd2")},
                    {**B_["q"], "local": ("dqu", "dqd", "dqu1", "dqd1", "nsp", "dqu2", "dqd2")}, "temporal")}


def write_new(path: Path, text: str) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.write_text(text, encoding="utf-8")


# ---------------------------------------------------------------------------
# raw local quantities
# ---------------------------------------------------------------------------
def direction(p: str) -> int:
    return 1 if p.endswith("_N") or p.endswith("_E") else -1   # downstream = pos + direction


def local_at(V_, OV, OQ, lanes, blk, ti, li, dirn):
    """Raw local quantities at cells (ti, li) given the visibility V_ (STEPS x n) of the frame."""
    S, n = V_.shape
    ar = np.arange(S)[:, None]
    last_incl = np.maximum.accumulate(np.where(V_, ar, -1), axis=0)
    next_incl = np.minimum.accumulate(np.where(V_, ar, S + 1000)[::-1], axis=0)[::-1]
    lb = np.full((S, n), -1)
    lb[1:] = last_incl[:-1]
    na = np.full((S, n), S + 1000)
    na[:-1] = next_incl[1:]
    b, a = lb[ti, li], na[ti, li]
    hb, ha = b >= 0, a < S
    OQl = OQ / lanes[None, :]
    f32 = np.float32
    out = {"db": np.where(hb, ti - b, NONE).astype(f32), "da": np.where(ha, a - ti, NONE).astype(f32),
           "vb": np.where(hb, OV[np.maximum(b, 0), li], np.nan).astype(f32),
           "qb": np.where(hb, OQl[np.maximum(b, 0), li], np.nan).astype(f32),
           "va": np.where(ha, OV[np.minimum(a, S - 1), li], np.nan).astype(f32),
           "qa": np.where(ha, OQl[np.minimum(a, S - 1), li], np.nan).astype(f32)}
    b2 = np.where(hb, lb[np.maximum(b, 0), li], -1)
    a2 = np.where(ha, na[np.minimum(a, S - 1), li], S + 1000)
    hb2, ha2 = b2 >= 0, a2 < S
    out.update({"db2": np.where(hb2, ti - b2, NONE).astype(f32), "da2": np.where(ha2, a2 - ti, NONE).astype(f32),
                "vb2": np.where(hb2, OV[np.maximum(b2, 0), li], np.nan).astype(f32),
                "qb2": np.where(hb2, OQl[np.maximum(b2, 0), li], np.nan).astype(f32),
                "va2": np.where(ha2, OV[np.minimum(a2, S - 1), li], np.nan).astype(f32),
                "qa2": np.where(ha2, OQl[np.minimum(a2, S - 1), li], np.nan).astype(f32)})
    OVm, OQm = np.where(V_, OV, np.nan), np.where(V_, OQl, np.nan)
    pad = lambda X: np.pad(X, ((1, 1), (2, 2)), constant_values=np.nan)  # noqa: E731
    PV, PQ = pad(OVm), pad(OQm)
    Vp = np.pad(V_, ((1, 1), (2, 2)), constant_values=False)
    up, dn = li - dirn + 2, li + dirn + 2
    up2, dn2 = li - 2 * dirn + 2, li + 2 * dirn + 2
    out["vu"], out["vd"] = PV[ti + 1, up].astype(f32), PV[ti + 1, dn].astype(f32)
    out["qu"], out["qd"] = PQ[ti + 1, up].astype(f32), PQ[ti + 1, dn].astype(f32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        out["vu1"] = np.nanmean(np.stack([PV[ti, up], PV[ti + 2, up]]), axis=0).astype(f32)
        out["vd1"] = np.nanmean(np.stack([PV[ti, dn], PV[ti + 2, dn]]), axis=0).astype(f32)
        out["qu1"] = np.nanmean(np.stack([PQ[ti, up], PQ[ti + 2, up]]), axis=0).astype(f32)
        out["qd1"] = np.nanmean(np.stack([PQ[ti, dn], PQ[ti + 2, dn]]), axis=0).astype(f32)
    out["vu2"], out["vd2"] = PV[ti + 1, up2].astype(f32), PV[ti + 1, dn2].astype(f32)
    out["qu2"], out["qd2"] = PQ[ti + 1, up2].astype(f32), PQ[ti + 1, dn2].astype(f32)
    nsp = np.zeros(len(ti), np.int16)
    for dt in (0, 1, 2):
        nsp += Vp[ti + dt, up] + Vp[ti + dt, dn]
    out["nsp"] = nsp.astype(f32)
    V3 = np.pad(V_, ((3, 3), (0, 0)), constant_values=False)
    B3 = np.pad(blk, ((3, 3), (0, 0)), constant_values=False)
    n3 = np.zeros(len(ti), np.int16)
    nb = np.zeros(len(ti), np.int16)
    for dt in (-3, -2, -1, 1, 2, 3):
        n3 += V3[ti + 3 + dt, li]
        nb += B3[ti + 3 + dt, li]
    out["n3"], out["nblk"] = n3.astype(f32), nb.astype(f32)
    return out


def cmd_features(args) -> None:
    """H: hide-run visibility at the hide cells (mask reproduced and checked against the hide parquet);
    T: real visibility at the real targets (t1_targets order) + V51 state values, regime and blackout flag."""
    import t1_assemble as A
    CACHE.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    K = M.read_key_state()
    sid, v51, q51 = V.zip_state(V51_ZIP)
    assert np.array_equal(sid, K.submission_id.to_numpy())
    for s in SPLITS:
        Ks = K[K.split == s]
        for p, idx in Ks.groupby("panel", observed=True).indices.items():
            pH, pT = CACHE / f"{s}_{p}_H.npz", CACHE / f"{s}_{p}_T.npz"
            if pH.exists() or pT.exists():
                raise FileExistsError(pH)
            C = pd.read_parquet(M.HIDE / f"{p}_{s}_cells.parquet", columns=["day", "t", "pos"])
            fd = M.static_fd(p)
            n, lanes, dirn = fd["n"], fd["lanes"], direction(p)
            pf = pq.ParquetFile(CLEAN / p / f"{s}.parquet")
            starts = np.searchsorted(C.day.to_numpy(), np.arange(pf.num_row_groups + 1))
            accH, accT, dayT, ttT, llT = [], [], [], [], []
            for day in range(pf.num_row_groups):
                tb = pf.read_row_group(day, columns=["is_score_eligible", "is_t1_target", "is_t2_blackout",
                                                     "speed_obs", "flow_obs"])
                g = lambda c: tb.column(c).to_numpy().reshape(STEPS, n)  # noqa: E731
                el, tgt, blk = g("is_score_eligible"), g("is_t1_target"), g("is_t2_blackout")
                OV, OQ = g("speed_obs").astype(float), g("flow_obs").astype(float)
                vis = np.isfinite(OV) & np.isfinite(OQ)
                rng = np.random.default_rng(M.SEED + 1000 * PANELS.index(p) + (0 if s == "validation" else 500) + day)
                hide = (rng.random(vis.shape) < M.HIDE_RATE) & vis & el
                th, lh = np.nonzero(hide)
                c = C.iloc[starts[day]:starts[day + 1]]
                assert np.array_equal(c.t.to_numpy(), th) and np.array_equal(c.pos.to_numpy(), lh), (p, s, day)
                assert not (tgt & vis).any()
                accH.append(local_at(vis & ~hide, OV, OQ, lanes, blk, th, lh, dirn))
                tt, lt = np.nonzero(tgt)
                accT.append(local_at(vis, OV, OQ, lanes, blk, tt, lt, dirn))
                dayT.append(np.full(len(tt), day, np.int16))
                ttT.append(tt)
                llT.append(lt)
            H = {k: np.concatenate([a_[k] for a_ in accH]) for k in accH[0]}
            T = {k: np.concatenate([a_[k] for a_ in accT]) for k in accT[0]}
            tg = t1_targets(p, s)
            ii = Ks.index.to_numpy()[idx]
            j = np.searchsorted(tg["key"], K.key.to_numpy()[ii])
            assert len(ii) == len(tg["key"]) and np.array_equal(tg["key"][j], K.key.to_numpy()[ii])
            o = np.empty(len(ii), np.int64)
            o[j] = ii
            T.update({"key": tg["key"], "regime": tg["regime"].astype(np.int8), "day": np.concatenate(dayT),
                      "gap": A.target_blackout(p, s).astype(bool), "v51": v51[o], "q51": q51[o],
                      "sid": K.submission_id.to_numpy()[o]})
            assert len(T["db"]) == len(tg["key"]) and len(H["db"]) == len(C)
            # t1_targets order = day-major, then (t, pos) row-major = the np.nonzero order used above
            assert np.array_equal((tg["key"] // KEY_BASE) % STEPS, np.concatenate(ttT))
            assert np.array_equal(tg["key"] % KEY_BASE, np.concatenate(llT))
            np.savez(pH, **H)
            np.savez(pT, **T)
            print("features", s, p, "H", len(C), "T", len(tg["key"]), f"{time.time() - t0:.0f}s", flush=True)
            del C, H, T, accH, accT
            gc.collect()


def load_local(p, s, which):
    z = np.load(CACHE / f"{s}_{p}_{which}.npz")
    return {k: (z[k].astype(float) if z[k].dtype == np.float32 else z[k]) for k in z.files}


# ---------------------------------------------------------------------------
# features relative to the current prediction (identical code at hide cells and targets)
# ---------------------------------------------------------------------------
def local_feats(L, pv, pq_, lanes, reg, names):
    pql = pq_ / lanes
    db, da = np.minimum(L["db"], CAP + 1), np.minimum(L["da"], CAP + 1)
    okb, oka = L["db"] <= CAP, L["da"] <= CAP
    vb, va = np.where(okb, L["vb"], np.nan), np.where(oka, L["va"], np.nan)
    qb, qa = np.where(okb, L["qb"], np.nan), np.where(oka, L["qa"], np.nan)
    both = okb & oka
    w = np.where(both, da / np.maximum(db + da, 1), np.nan)
    vi = np.where(both, w * vb + (1 - w) * va, np.nan)
    qi = np.where(both, w * qb + (1 - w) * qa, np.nan)
    ok2b, ok2a = L["db2"] <= CAP, L["da2"] <= CAP
    F = {"db": db, "da": da, "dvb": vb - pv, "dva": va - pv, "dvi": vi - pv,
         "dvu": L["vu"] - pv, "dvd": L["vd"] - pv, "dvu1": L["vu1"] - pv, "dvd1": L["vd1"] - pv,
         "n3": L["n3"], "nsp": L["nsp"], "nblk": L["nblk"], "reg": np.asarray(reg, float),
         "dqb": qb - pql, "dqa": qa - pql, "dqi": qi - pql, "dqu": L["qu"] - pql, "dqd": L["qd"] - pql,
         "dvb2": np.where(ok2b, L["vb2"], np.nan) - pv, "dva2": np.where(ok2a, L["va2"], np.nan) - pv,
         "dqb2": np.where(ok2b, L["qb2"], np.nan) - pql, "dqa2": np.where(ok2a, L["qa2"], np.nan) - pql,
         "dqu1": L["qu1"] - pql, "dqd1": L["qd1"] - pql, "dvu2": L["vu2"] - pv, "dvd2": L["vd2"] - pv,
         "dqu2": L["qu2"] - pql, "dqd2": L["qd2"] - pql,
         "db2": np.minimum(L["db2"], CAP + 1), "da2": np.minimum(L["da2"], CAP + 1)}
    return {k: F[k] for k in names}


class LRestack:
    """V57 restack (component deviations + x + rel) plus local features, per channel.  With no 'local' entry it is
    exactly t1_v57_restack.Restack with the chosen V57 config (checked: crossfit reproduces the V57 grid rows)."""

    def __init__(self, cfg, d, L, tr, fd):
        import lightgbm as lgb
        self.cfg, self.fd = cfg, fd
        n, pos = fd["n"], d["pos"]
        self.mp, self.mq = V.link_means(pos, d["pv"], d["pq"], d["lanes"], tr, n)
        X = self.design(d["pv"], d["pq"], d["cv"], d["cq"], pos, L, d["reg"])
        self.models = {}
        for ch, e in (("v", d["pv"] - d["ov"]), ("q", (d["pq"] - d["oq"]) / d["lanes"])):
            kind = cfg.get(ch)
            if kind is None:
                continue
            prm = {**GBM_BASE, "num_leaves": kind["leaves"], "min_data_in_leaf": kind["mdl"]}
            self.models[ch] = lgb.train(prm, lgb.Dataset(X[ch][tr], V.demean(pos, tr, e, n)),
                                        num_boost_round=kind["rounds"])

    def design(self, pv, pq_, cv, cq, pos, L, reg):
        F = V.features(pv, pq_, cv, cq, pos, self.fd, self.mp, self.mq)
        lanes = self.fd["lanes"][pos]
        out = {}
        for ch, base in (("v", V.VF), ("q", V.QF)):
            kind = self.cfg.get(ch)
            if kind is None:
                continue
            cols = [F[f] for f in base]
            loc = kind.get("local", ())
            if loc:
                LF = local_feats(L, pv, pq_, lanes, reg, loc)
                cols += [LF[f] for f in loc]
            out[ch] = np.column_stack(cols)
        return out

    def apply(self, pv, pq_, cv, cq, pos, free, L, reg, lam=1.0):
        X = self.design(pv, pq_, cv, cq, pos, L, reg)
        v, q = pv.copy(), pq_.copy()
        if "v" in self.models:
            dv = self.models["v"].predict(X["v"], num_threads=THREADS)
            v = np.where(free, np.clip(pv - lam * dv, M.SPEED_MIN, M.SPEED_MAX), pv)
        if "q" in self.models:
            dq = self.models["q"].predict(X["q"], num_threads=THREADS)
            q = np.where(free, np.maximum(pq_ - lam * dq * self.fd["lanes"][pos], 0.0), pq_)
        return v, q


class Stage2:
    """Two-stage variant: second LightGBM on the residual of the V57 output (local features measured against the
    V57 output, plus the V57 base features evaluated at the V57 output)."""

    def __init__(self, cfg, d, L, v57, q57, tr):
        import lightgbm as lgb
        self.cfg, self.fd = cfg, d["fd"]
        n, pos = self.fd["n"], d["pos"]
        self.mp, self.mq = V.link_means(pos, v57, q57, d["lanes"], tr, n)
        X = self.design(L, v57, q57, pos, d["cv"], d["cq"], d["reg"])
        self.models = {}
        for ch, e in (("v", v57 - d["ov"]), ("q", (q57 - d["oq"]) / d["lanes"])):
            kind = cfg[ch]
            prm = {**GBM_BASE, "num_leaves": kind["leaves"], "min_data_in_leaf": kind["mdl"]}
            self.models[ch] = lgb.train(prm, lgb.Dataset(X[ch][tr], V.demean(pos, tr, e, n)),
                                        num_boost_round=kind["rounds"])

    def design(self, L, v57, q57, pos, cv, cq, reg):
        F = V.features(v57, q57, cv, cq, pos, self.fd, self.mp, self.mq)
        lanes = self.fd["lanes"][pos]
        out = {}
        for ch, base in (("v", V.VF), ("q", V.QF)):
            loc = self.cfg[ch]["local"]
            LF = local_feats(L, v57, q57, lanes, reg, loc)
            out[ch] = np.column_stack([F[f] for f in base] + [LF[f] for f in loc])
        return out

    def apply(self, L, v57, q57, pos, cv, cq, reg, free, lam=1.0):
        X = self.design(L, v57, q57, pos, cv, cq, reg)
        dv = self.models["v"].predict(X["v"], num_threads=THREADS)
        dq = self.models["q"].predict(X["q"], num_threads=THREADS)
        v = np.where(free, np.clip(v57 - lam * dv, M.SPEED_MIN, M.SPEED_MAX), v57)
        q = np.where(free, np.maximum(q57 - lam * dq * self.fd["lanes"][pos], 0.0), q57)
        return v, q


# ---------------------------------------------------------------------------
# hide data + cross-fit
# ---------------------------------------------------------------------------
def load(p, s):
    d = V.load_hide(p, s)
    d["reg"] = d["C"].regime.to_numpy().astype(np.int64)
    d["day"] = d["C"].day.to_numpy().astype(np.int64)
    return d, load_local(p, s, "H")


def crossfit(d, L, cfg, lams=(1.0,)):
    """{lam: (v, q)}: odd/even-day cross-fit; restack(+local) on training-parity free cells, V56 refit."""
    if cfg.get("two_stage"):
        return crossfit_s2(d, L, cfg, lams)
    out = {lam: (d["pv"].copy(), d["pq"].copy()) for lam in lams}
    for a in (0, 1):
        tr = d["par"] == a
        te = ~tr
        rs = LRestack(cfg, d, L, tr & d["free"], d["fd"])
        for lam in lams:
            pv1, pq1 = rs.apply(d["pv"], d["pq"], d["cv"], d["cq"], d["pos"], d["free"], L, d["reg"], lam)
            P = V.fit56(d["pos"][tr], pv1[tr], pq1[tr], d["ov"][tr], d["oq"][tr], d["t"][tr], d["free"][tr], d["fd"])
            vv, qq = V.pred56(P, d["pos"][te], pv1[te], pq1[te], d["t"][te], d["free"][te], d["fd"])
            out[lam][0][te], out[lam][1][te] = vv, qq
    return out


def crossfit_s2(d, L, cfg, lams=(1.0,)):
    out = {lam: (d["pv"].copy(), d["pq"].copy()) for lam in lams}
    fd = d["fd"]
    for a in (0, 1):
        tr = d["par"] == a
        te = ~tr
        rs = LRestack(V57CFG, d, L, tr & d["free"], fd)
        pv1, pq1 = rs.apply(d["pv"], d["pq"], d["cv"], d["cq"], d["pos"], d["free"], L, d["reg"])
        P = V.fit56(d["pos"][tr], pv1[tr], pq1[tr], d["ov"][tr], d["oq"][tr], d["t"][tr], d["free"][tr], fd)
        v57, q57 = V.pred56(P, d["pos"], pv1, pq1, d["t"], d["free"], fd)
        s2 = Stage2(cfg, d, L, v57, q57, tr & d["free"])
        for lam in lams:
            vv, qq = s2.apply(L, v57, q57, d["pos"], d["cv"], d["cq"], d["reg"], d["free"], lam)
            out[lam][0][te], out[lam][1][te] = vv[te], qq[te]
    return out


def strata(L, free, reg):
    db = np.minimum(L["db"], 4).astype(int) - 1
    da = np.minimum(L["da"], 4).astype(int) - 1
    ns = np.digitize(L["nsp"], [2, 4])
    return (((reg * 2 + free.astype(int)) * 4 + db) * 4 + da) * 3 + ns


def target_weights(d, L, LT, fd):
    """Hide-cell weights = share of the real non-blackout targets in the cell's stratum / share of the hide cells,
    within the regime (free flag of the targets from the V51 state, as the apply rule)."""
    ng = ~LT["gap"]
    pos = (LT["key"] % KEY_BASE).astype(np.int64)
    fT = LT["v51"] > M.FREE_RATIO * fd["vc"][pos]
    sT = strata({k: LT[k][ng] for k in ("db", "da", "nsp")}, fT[ng], LT["regime"][ng].astype(int))
    sH = strata(L, d["free"], d["reg"])
    nS = 3 * 2 * 16 * 3
    cT = np.bincount(sT, minlength=nS).astype(float)
    cH = np.bincount(sH, minlength=nS).astype(float)
    regS = np.arange(nS) // (2 * 16 * 3)
    shT = cT / np.maximum(np.bincount(regS, weights=cT, minlength=3)[regS], 1)
    shH = cH / np.maximum(np.bincount(regS, weights=cH, minlength=3)[regS], 1)
    wS = np.where(cH > 0, shT / np.maximum(shH, 1e-12), 0.0)
    unc = float(np.mean([shT[(regS == r) & (cH == 0)].sum() for r in range(3)]))
    return wS[sH], unc


def s_state_w(v, q, ov, oq, lanes, reg, w):
    out = []
    for r in range(3):
        m = reg == r
        if not m.any() or w[m].sum() <= 0:
            continue
        ww = w[m]
        rv = np.sqrt(np.sum(ww * (v[m] - ov[m]) ** 2) / ww.sum())
        rq = np.sqrt(np.sum(ww * ((q[m] - oq[m]) / lanes[m]) ** 2) / ww.sum())
        out.append(0.54 * max(0, 1 - rv / 25) + 0.46 * max(0, 1 - rq / 600))
    return float(np.mean(out))


def summarise(G: pd.DataFrame, ref: str = "V57") -> pd.DataFrame:
    b = G[G.cfg == ref].set_index(["split", "panel"])
    keys = ["dS", "dphys_est", "dFD_est", "dE_est", "dTotal_est", "S_rw", "S_far", "S_near"]
    out = []
    for nm in G.cfg.unique():
        if nm == ref:
            continue
        g = G[G.cfg == nm].set_index(["split", "panel"])
        rec = {"cfg": nm}
        for s in SPLITS:
            dd = g.loc[s][keys] - b.loc[s].loc[g.loc[s].index][keys]
            for k in keys:
                rec[f"{k}_{s[:3]}"] = M.family_mean(dd[k].to_dict())
            rec[f"npos_{s[:3]}"] = int((dd.dTotal_est > 0).sum())
            rec[f"nposS_{s[:3]}"] = int((dd.dS > 0).sum())
            rec[f"nposRW_{s[:3]}"] = int((dd.S_rw > 0).sum())
            rec[f"min_dTotal_{s[:3]}"] = float(dd.dTotal_est.min())
            rec[f"lower_bound_{s[:3]}"] = W_STATE * rec[f"dS_{s[:3]}"] + W_PHYS * rec[f"dFD_est_{s[:3]}"] / 3
        rec["dTot_both"] = (rec["dTotal_est_val"] + rec["dTotal_est_pri"]) / 2
        out.append(rec)
    return pd.DataFrame(out).set_index("cfg")


def cmd_crossfit(args) -> None:
    """Pre-registered rule (fixed before the final grid was run; exploration in the lane-K scratch folders):
    reference V57 (= V58 on the rows lane K may change), cross-fitted on odd/even days like every config.
    Maximise ddTot_both = mean over March/April of the family-mean (0.35 dS_state + 0.15 dS_phys real-frame
    estimate) relative to V57.  Eligible: >= 9/10 corridors with ddTot > 0 in each month, family-mean dS_state > 0
    in each month, and family-mean dS_state_rw > 0 in each month (hide cells reweighted to the real non-blackout
    targets' shares of regime x free x min(db,4) x min(da,4) x nsp-bucket).  Configs within 3e-6 of the max -> the
    simplest (order CPLX).  Gate (checked in apply): real-frame S_FD (family mean) must not fall by more than 1e-4
    versus V58 in either month."""
    OUT.mkdir(parents=True, exist_ok=True)
    for f in ("crossfit_grid.csv", "crossfit_summary.csv", "crossfit_choice.json", "crossfit_lambda.csv"):
        if (OUT / f).exists():
            raise FileExistsError(OUT / f)
    t0 = time.time()
    ref57 = pd.read_csv(V.OUT / "crossfit_grid.csv")
    ref57 = ref57[ref57.cfg == "g15m300|gq"].set_index(["split", "panel"])
    rows, transfer = [], []
    for p in PANELS:
        for s in SPLITS:
            d, L = load(p, s)
            fd = d["fd"]
            LT = load_local(p, s, "T")
            w, unc = target_weights(d, L, LT, fd)
            far = ((np.minimum(L["db"], L["da"]) >= 2) | (L["nsp"] <= 1)).astype(float)
            sargs = (d["ov"], d["oq"], d["lanes"], d["reg"])
            pT = (LT["key"] % KEY_BASE).astype(np.int64)
            fT = (LT["v51"] > M.FREE_RATIO * fd["vc"][pT]) & ~LT["gap"]
            transfer.append({"split": s, "panel": p, "uncovered_target_share": unc, "w_max": float(w.max()),
                             "w_ess_frac": float(w.sum() ** 2 / (w ** 2).sum() / len(w)),
                             "far_share_H_free": float(far[d["free"]].mean()),
                             "far_share_T_free": float(((np.minimum(LT["db"], LT["da"]) >= 2) | (LT["nsp"] <= 1))[fT].mean())})
            for nm, cfg in [("V57", V57CFG)] + list(CFGS.items()):
                v, q = crossfit(d, L, cfg)[1.0]
                rec = {"panel": p, "split": s, "cfg": nm, **V.score(d, v, q),
                       "S_rw": s_state_w(v, q, *sargs, w), "S_far": s_state_w(v, q, *sargs, far),
                       "S_near": s_state_w(v, q, *sargs, 1 - far)}
                if nm == "V57":
                    assert abs(rec["dTotal_est"] - ref57.loc[(s, p), "dTotal_est"]) < 1e-12, (p, s)
                rows.append(rec)
            print("crossfit", p, s, f"{time.time() - t0:.0f}s", flush=True)
            del d, L, LT
            gc.collect()
    G = pd.DataFrame(rows)
    A = summarise(G)
    A["eligible"] = ((A.npos_val >= 9) & (A.npos_pri >= 9) & (A.dS_val > 0) & (A.dS_pri > 0) &
                     (A.S_rw_val > 0) & (A.S_rw_pri > 0))
    best = float(A[A.eligible].dTot_both.max()) if A.eligible.any() else None
    choice = None
    near = []
    if best is not None:
        near = [c for c in CPLX if c in A.index and A.loc[c, "eligible"] and A.loc[c, "dTot_both"] >= best - 3e-6]
        choice = near[0]
    nested = {}
    for sel, rep in (("val", "pri"), ("pri", "val")):
        E = A[(A[f"npos_{sel}"] >= 9) & (A[f"dS_{sel}"] > 0) & (A[f"S_rw_{sel}"] > 0)]
        if len(E):
            pk = E[f"dTotal_est_{sel}"].idxmax()
            nested[f"choose_on_{sel}"] = {"pick": pk, f"dTotal_{rep}_of_pick": float(A.loc[pk, f"dTotal_est_{rep}"]),
                                          f"dTotal_{rep}_of_choice": float(A.loc[choice, f"dTotal_est_{rep}"]) if choice else None}
    G.to_csv(OUT / "crossfit_grid.csv", index=False)
    A.to_csv(OUT / "crossfit_summary.csv")
    pd.DataFrame(transfer).to_csv(OUT / "crossfit_transfer_weights.csv", index=False)
    out = {"rule": cmd_crossfit.__doc__.strip(), "chosen_name": choice, "chosen": CFGS.get(choice), "max_eligible": best,
           "dS_family": ({s: float(A.loc[choice, f"dS_{s[:3]}"]) for s in SPLITS} if choice else None),
           "within_3e-6": near, "nested_month_check": nested, "summary": A.reset_index().to_dict("records")}
    if choice is not None:
        lams = (0.5, 0.75, 1.25, 1.5)
        lrows = []
        for p in PANELS:
            for s in SPLITS:
                d, L = load(p, s)
                res = crossfit(d, L, CFGS[choice], lams)
                for lam in lams:
                    lrows.append({"panel": p, "split": s, "cfg": f"lam{lam}", **V.score(d, *res[lam]),
                                  "S_rw": np.nan, "S_far": np.nan, "S_near": np.nan})
                del d, L, res
                gc.collect()
        Lm = pd.DataFrame(lrows)
        Lm.to_csv(OUT / "crossfit_lambda.csv", index=False)
        LA = summarise(pd.concat([Lm, G[G.cfg.isin(["V57", choice])]]))
        out["lambda_curve"] = {k: {"dTot_both": float(LA.loc[k, "dTot_both"]), "dTotal_val": float(LA.loc[k, "dTotal_est_val"]),
                                   "dTotal_pri": float(LA.loc[k, "dTotal_est_pri"])} for k in LA.index}
        ch = G[G.cfg == choice].set_index(["split", "panel"])
        b = G[G.cfg == "V57"].set_index(["split", "panel"])
        out["per_corridor_month"] = {s: {p: {k: float(ch.loc[(s, p), k] - b.loc[(s, p), k])
                                             for k in ("dS", "dphys_est", "dFD_est", "dE_est", "dTotal_est", "S_rw")}
                                         for p in PANELS} for s in SPLITS}
        reg = {}
        for s in SPLITS:
            for r in (1, 2, 3):
                for k in ("rv", "rq"):
                    col = f"R{r}_{k}"
                    dd = (ch.loc[s][col] - b.loc[s][col]).dropna()
                    reg[f"{s}|R{r}|{k}"] = {"V57": float(b.loc[s][col].dropna().mean()), "delta": float(dd.mean()),
                                            "corridors_improved": int((dd < 0).sum()), "corridors": int(len(dd))}
        out["per_regime_rmse"] = reg
    out["seconds"] = round(time.time() - t0)
    write_new(OUT / "crossfit_choice.json", json.dumps(out, indent=2))
    pd.set_option("display.width", 320)
    print((A * 1).round(7).to_string())
    print(json.dumps({k: out.get(k) for k in ("chosen_name", "max_eligible", "within_3e-6", "nested_month_check",
                                              "lambda_curve", "per_regime_rmse")}, indent=2), flush=True)


def load_choice():
    return json.loads((OUT / "crossfit_choice.json").read_text())


def chosen_cfg():
    c = load_choice()
    if c["chosen_name"] is None:
        raise SystemExit("no eligible config: not built")
    return c["chosen_name"], c["chosen"]


# ---------------------------------------------------------------------------
# real targets
# ---------------------------------------------------------------------------
def target_inputs(p, split, gf):
    """Production components at the real targets in t1_targets order + the cached local quantities."""
    tg, z = V.target_arrays(p, split, gf)
    LT = load_local(p, split, "T")
    assert np.array_equal(LT["key"], tg["key"])
    z.update({"key": tg["key"], "pos": (tg["key"] % KEY_BASE).astype(np.int64),
              "tod": ((tg["key"] // KEY_BASE) % STEPS).astype(np.int64)})
    return tg, z, LT


def fit_month(cfg, d, L):
    """All days of the month: restack(+local) and V56 refit on the restacked hide values."""
    fd = d["fd"]
    if cfg.get("two_stage"):
        rs = LRestack(V57CFG, d, L, d["free"], fd)
        pvH, pqH = rs.apply(d["pv"], d["pq"], d["cv"], d["cq"], d["pos"], d["free"], L, d["reg"])
        P = V.fit56(d["pos"], pvH, pqH, d["ov"], d["oq"], d["t"], d["free"], fd)
        v57, q57 = V.pred56(P, d["pos"], pvH, pqH, d["t"], d["free"], fd)
        return rs, P, Stage2(cfg, d, L, v57, q57, d["free"])
    rs = LRestack(cfg, d, L, d["free"], fd)
    pvH, pqH = rs.apply(d["pv"], d["pq"], d["cv"], d["cq"], d["pos"], d["free"], L, d["reg"])
    P = V.fit56(d["pos"], pvH, pqH, d["ov"], d["oq"], d["t"], d["free"], fd)
    return rs, P, None


def predict_month(fitted, d, L, pv, pq_, cv, cq, pos, tod, free, LX, reg):
    rs, P, s2 = fitted
    pv1, pq1 = rs.apply(pv, pq_, cv, cq, pos, free, LX, reg)
    v, q = V.pred56(P, pos, pv1, pq1, tod, free, d["fd"])
    if s2 is not None:
        v, q = s2.apply(LX, v, q, pos, cv, cq, reg, free)
    return v, q


# ---------------------------------------------------------------------------
# direct: disjoint-feature neighbour proxy + local-deviation distributions, T vs H
# ---------------------------------------------------------------------------
def _proxy(Lx, kind):
    if kind == "temporal":
        ok = (Lx["db"] == 1) & (Lx["da"] == 1)
        return ok, 0.5 * (Lx["vb"] + Lx["va"]), 0.5 * (Lx["qb"] + Lx["qa"])
    ok = np.isfinite(Lx["vu"]) & np.isfinite(Lx["vd"]) & np.isfinite(Lx["qu"]) & np.isfinite(Lx["qd"])
    return ok, 0.5 * (Lx["vu"] + Lx["vd"]), 0.5 * (Lx["qu"] + Lx["qd"])


def cmd_direct(args) -> None:
    """(1) Tonly = V57 + temporal local features only, judged with a spatial proxy (mean of the adjacent links at
    t); Sonly = V57 + spatial local features only, judged with a temporal proxy (mean of t-1 and t+1 of the same
    link).  Fitted on parity-a hide cells (+ V56 refit), evaluated on parity 1-a days at the real targets T and the
    hide cells H with the same estimator; cells predicted free, not blackout, proxy available.  (2) RMS of the
    local deviations (neighbour obs - prediction) and component deviations at T vs H (H reweighted to the T
    regime x db x da strata): a month-adapted U-Net that had memorised the visible (= hide) cells would sit
    differently relative to its neighbours at H than at T."""
    path = OUT / "direct_proxy.csv"
    if path.exists():
        raise FileExistsError(path)
    rows, drows = [], []
    t0 = time.time()
    for s in SPLITS:
        gf = V.gbm_csv(s)
        for p in PANELS:
            d, L = load(p, s)
            fd = d["fd"]
            tg, z, LT = target_inputs(p, s, gf)
            pos, tod = z["pos"], z["tod"]
            freeT = (LT["v51"] > M.FREE_RATIO * fd["vc"][pos]) & ~LT["gap"]
            regT = LT["regime"].astype(np.int64)
            cvT = [z["vgbm"], z["v_w48"], z["v_w48lwr"]]
            cqT = [z["qgbm"], z["q_w48"], z["q_w48lwr"]]
            lanesT = fd["lanes"][pos]
            for nm, (cv_, cq_, kind) in DPROXY.items():
                res = {}
                for which, cfg in (("base", V57CFG), ("new", {"v": cv_, "q": cq_})):
                    vH, qH, vT, qT = d["pv"].copy(), d["pq"].copy(), LT["v51"].copy(), LT["q51"].copy()
                    for a in (0, 1):
                        tr = d["par"] == a
                        rs = LRestack(cfg, d, L, tr & d["free"], fd)
                        pvH, pqH = rs.apply(d["pv"], d["pq"], d["cv"], d["cq"], d["pos"], d["free"], L, d["reg"])
                        P = V.fit56(d["pos"][tr], pvH[tr], pqH[tr], d["ov"][tr], d["oq"][tr], d["t"][tr],
                                    d["free"][tr], fd)
                        a_vH, a_qH = V.pred56(P, d["pos"], pvH, pqH, d["t"], d["free"], fd)
                        pvT, pqT = rs.apply(LT["v51"], LT["q51"], cvT, cqT, pos, freeT, LT, regT)
                        a_vT, a_qT = V.pred56(P, pos, pvT, pqT, tod, freeT, fd)
                        te, tt = ~tr, (LT["day"] % 2) != a
                        vH[te], qH[te], vT[tt], qT[tt] = a_vH[te], a_qH[te], a_vT[tt], a_qT[tt]
                    res[which] = (vH, qH, vT, qT)
                okH, yvH, yqH = _proxy(L, kind)
                okT, yvT, yqT = _proxy(LT, kind)
                mH, mT = okH & d["free"], okT & freeT
                (b_vH, b_qH, b_vT, b_qT), (n_vH, n_qH, n_vT, n_qT) = res["base"], res["new"]
                lH = d["lanes"]
                rows.append({"split": s, "panel": p, "cfg": nm, "proxy": kind, "nH": int(mH.sum()), "nT": int(mT.sum()),
                             "H_proxy_dsse_v": float(np.mean((n_vH - yvH)[mH] ** 2 - (b_vH - yvH)[mH] ** 2)),
                             "T_proxy_dsse_v": float(np.mean((n_vT - yvT)[mT] ** 2 - (b_vT - yvT)[mT] ** 2)),
                             "H_true_dsse_v": float(np.mean((n_vH - d["ov"])[mH] ** 2 - (b_vH - d["ov"])[mH] ** 2)),
                             "H_proxy_dsse_q": float(np.mean(((n_qH / lH - yqH) ** 2 - (b_qH / lH - yqH) ** 2)[mH])),
                             "T_proxy_dsse_q": float(np.mean(((n_qT / lanesT - yqT) ** 2 - (b_qT / lanesT - yqT) ** 2)[mT])),
                             "H_true_dsse_q": float(np.mean((((n_qH - d["oq"]) / lH) ** 2 - ((b_qH - d["oq"]) / lH) ** 2)[mH])),
                             "H_rms_dv": float(np.sqrt(np.mean((n_vH - b_vH)[mH] ** 2))),
                             "T_rms_dv": float(np.sqrt(np.mean((n_vT - b_vT)[mT] ** 2))),
                             "H_rms_dq": float(np.sqrt(np.mean(((n_qH - b_qH) / lH)[mH] ** 2))),
                             "T_rms_dq": float(np.sqrt(np.mean(((n_qT - b_qT) / lanesT)[mT] ** 2)))})
            # (2) deviation distributions
            names = ("dvb", "dva", "dvi", "dvu", "dvd", "dqi", "dqb")
            FH = local_feats(L, d["pv"], d["pq"], d["lanes"], d["reg"], names)
            FH.update({"dg": d["cv"][0] - d["pv"], "du": d["cv"][1] - d["pv"], "dl": d["cv"][2] - d["pv"]})
            FT = local_feats(LT, LT["v51"], LT["q51"], lanesT, regT, names)
            FT.update({"dg": cvT[0] - LT["v51"], "du": cvT[1] - LT["v51"], "dl": cvT[2] - LT["v51"]})
            st = lambda X, r: (r * 4 + np.minimum(X["db"], 4).astype(int) - 1) * 4 + np.minimum(X["da"], 4).astype(int) - 1  # noqa: E731
            sH, sT = st(L, d["reg"]), st(LT, regT)
            cH = np.bincount(sH[d["free"]], minlength=48).astype(float)
            cT = np.bincount(sT[freeT], minlength=48).astype(float)
            w = np.where(cH > 0, (cT / cT.sum()) / np.maximum(cH / cH.sum(), 1e-12), 0)[sH]
            rec = {"split": s, "panel": p}
            for k in FH:
                xh, xt, wh = FH[k][d["free"]], FT[k][freeT], w[d["free"]]
                okh, okt = np.isfinite(xh), np.isfinite(xt)
                rec[f"{k}_rms_H"] = float(np.sqrt(np.sum(wh[okh] * xh[okh] ** 2) / wh[okh].sum()))
                rec[f"{k}_rms_T"] = float(np.sqrt(np.mean(xt[okt] ** 2)))
            drows.append(rec)
            print("direct", s, p, f"{time.time() - t0:.0f}s", flush=True)
            del d, L, z, LT
            gc.collect()
        del gf
    D = pd.DataFrame(rows)
    D.to_csv(path, index=False)
    E = pd.DataFrame(drows)
    E.to_csv(OUT / "direct_deviation_rms.csv", index=False)
    pd.set_option("display.width", 250)
    num = [c for c in D.columns if c not in ("split", "panel", "cfg", "proxy")]
    print(D.groupby(["cfg", "split"])[num].mean().T.round(4).to_string())
    print(E.groupby("split").mean(numeric_only=True).T.round(3).to_string(), flush=True)


# ---------------------------------------------------------------------------
# apply
# ---------------------------------------------------------------------------
def cmd_apply(args) -> None:
    cname, cfg = chosen_cfg()
    for pth in CSV.values():
        if pth.exists():
            raise FileExistsError(pth)
    for f in ("apply_changes.csv", "apply_fd_frame.csv", "apply_summary.json", "apply_buckets.csv", "blackout_sids.npy"):
        if (OUT / f).exists():
            raise FileExistsError(OUT / f)
    t0 = time.time()
    K = M.read_key_state()
    sid, v51, q51 = V.zip_state(V51_ZIP)
    assert np.array_equal(sid, K.submission_id.to_numpy())
    sid, v58, q58 = V.zip_state(V58_ZIP)
    assert np.array_equal(sid, K.submission_id.to_numpy())
    del sid
    print("key + V51 + V58 state rows", len(K), f"{time.time() - t0:.0f}s", flush=True)
    rows, fd_rows, buck, blk_sids = [], [], [], []
    for split in SPLITS:
        gf = V.gbm_csv(split)
        ks = (K.split == split).to_numpy()
        S_ = K[ks].reset_index(drop=True)
        V51v, V51q, V58v, V58q = v51[ks], q51[ks], v58[ks], q58[ks]
        v_all, q_all = V58v.copy(), V58q.copy()
        for p, idx in S_.groupby("panel", observed=True).indices.items():
            fd = M.static_fd(p)
            tg, z, LT = target_inputs(p, split, gf)
            kk = S_.key.to_numpy()[idx]
            j = np.searchsorted(tg["key"], kk)
            assert len(idx) == len(tg["key"]) and np.array_equal(tg["key"][j], kk), p
            reg = S_.mask_regime.iloc[idx].astype(str).map({r: i for i, r in enumerate(REGIMES)}).to_numpy()
            assert np.array_equal(reg, tg["regime"][j]), p
            # everything below in t1_targets order (index jj), mapped back with j at the end
            a51, b51 = np.empty(len(j)), np.empty(len(j))
            a58, b58 = np.empty(len(j)), np.empty(len(j))
            a51[j], b51[j], a58[j], b58[j] = V51v[idx], V51q[idx], V58v[idx], V58q[idx]
            assert np.array_equal(a51, LT["v51"]) and np.array_equal(b51, LT["q51"])
            gap = LT["gap"]
            pos, tod = z["pos"], z["tod"]
            regT = LT["regime"].astype(np.int64)
            free = (a51 > M.FREE_RATIO * fd["vc"][pos]) & ~gap
            cv = [z["vgbm"], z["v_w48"], z["v_w48lwr"]]
            cq = [z["qgbm"], z["q_w48"], z["q_w48lwr"]]
            d, L = load(p, split)
            # (1) lineage: the V57 pipeline (no local features) reproduces the V58 rows on the free non-blackout cells
            f57 = fit_month(V57CFG, d, L)
            a, b = predict_month(f57, d, L, a51, b51, cv, cq, pos, tod, free, LT, regT)
            n_lineage = int(((V.g6(a)[free] != a58[free]) | (V.g6(b)[free] != b58[free])).sum())
            assert n_lineage == 0, (split, p, n_lineage)
            # (2) chosen config, all days of the month
            fit = fit_month(cfg, d, L)
            a, b = predict_month(fit, d, L, a51, b51, cv, cq, pos, tod, free, LT, regT)
            v1, q1 = np.where(free, V.g6(a), a58), np.where(free, V.g6(b), b58)
            assert np.isfinite(v1).all() and np.isfinite(q1).all() and (v1 > 0).all() and (q1 >= 0).all()
            assert np.array_equal(v1[~free], a58[~free]) and np.array_equal(q1[~free], b58[~free])
            assert np.array_equal(v1[gap], a58[gap]) and np.array_equal(q1[gap], b58[gap])
            # hide cells: change of the same fit (in-sample) for the T-vs-H size comparison
            vH0, qH0 = predict_month(f57, d, L, d["pv"], d["pq"], d["cv"], d["cq"], d["pos"], d["t"], d["free"], L, d["reg"])
            vH1, qH1 = predict_month(fit, d, L, d["pv"], d["pq"], d["cv"], d["cq"], d["pos"], d["t"], d["free"], L, d["reg"])
            v_all[idx], q_all[idx] = v1[j], q1[j]
            blk_sids.append(LT["sid"][gap])
            ch = (v1 != a58) | (q1 != b58)
            fr = M.fd_frame(p, split, tg, np.arange(len(j)), a58, b58, v1, q1, fd)
            for r_ in fr:
                fd_rows.append({"split": split, "panel": p, "regime": r_["regime"], "FD_V58": r_["FD0"],
                                "FD_V59": r_["FD1"], "low_share_V59": r_["low_share1"], "targets": r_["targets"]})
            dv, dq = (v1 - a58)[free], ((q1 - b58) / fd["lanes"][pos])[free]
            dvh, dqh = (vH1 - vH0)[d["free"]], ((qH1 - qH0) / d["lanes"])[d["free"]]
            rows.append({"split": split, "panel": p, "targets": len(idx), "blackout_kept": int(gap.sum()),
                         "cells_rule_applies": int(free.sum()), "changed_vs_V58": int(ch.sum()),
                         "speed_changed_vs_V58": int((v1 != a58).sum()), "flow_changed_vs_V58": int((q1 != b58).sum()),
                         "lineage_mismatch_V58": n_lineage,
                         "T_mean_dv": float(dv.mean()), "H_mean_dv": float(dvh.mean()),
                         "T_rms_dv": float(np.sqrt(np.mean(dv ** 2))), "H_rms_dv": float(np.sqrt(np.mean(dvh ** 2))),
                         "T_mean_dq_lane": float(dq.mean()), "H_mean_dq_lane": float(dqh.mean()),
                         "T_rms_dq_lane": float(np.sqrt(np.mean(dq ** 2))), "H_rms_dq_lane": float(np.sqrt(np.mean(dqh ** 2))),
                         "max_abs_speed_change_vs_V58": float(np.max(np.abs(v1 - a58))),
                         "n_abs_speed_change_gt3": int((np.abs(v1 - a58) > 3).sum()),
                         "H_n_abs_speed_change_gt3": int((np.abs(dvh) > 3).sum()), "H_free": int(d["free"].sum()),
                         "max_abs_flow_change_vs_V58": float(np.max(np.abs(q1 - b58))),
                         "low_flow_share_after": float((q1 < 50).mean()),
                         "speed_min_after": float(v1.min()), "speed_max_after": float(v1.max())})
            dmT = np.minimum(np.minimum(LT["db"], LT["da"]), 4)[free]
            dxT = np.minimum(np.maximum(LT["db"], LT["da"]), 4)[free]
            dmH = np.minimum(np.minimum(L["db"], L["da"]), 4)[d["free"]]
            dxH = np.minimum(np.maximum(L["db"], L["da"]), 4)[d["free"]]
            for lo in range(1, 5):
                for hi in range(lo, 5):
                    mT, mH = (dmT == lo) & (dxT == hi), (dmH == lo) & (dxH == hi)
                    buck.append({"split": split, "panel": p, "dmin": lo, "dmax": hi, "nT": int(mT.sum()), "nH": int(mH.sum()),
                                 "T_ss_dv": float(np.sum(dv[mT] ** 2)), "H_ss_dv": float(np.sum(dvh[mH] ** 2)),
                                 "T_ss_dq": float(np.sum(dq[mT] ** 2)), "H_ss_dq": float(np.sum(dqh[mH] ** 2))})
            print(split, p, rows[-1]["changed_vs_V58"], "changed vs V58", f"{time.time() - t0:.0f}s", flush=True)
            del d, L, z, LT, fit, f57
            gc.collect()
        out = S_[["panel", "timestamp", "station_id", "link_id", "mask_regime"]].copy()
        out["speed_kmh"], out["flow_vph"] = v_all, q_all
        out.to_csv(CSV[split], index=False, float_format="%.6g")
        print("wrote", CSV[split], len(out), f"{time.time() - t0:.0f}s", flush=True)
        del S_, out, gf
        gc.collect()
    np.save(OUT / "blackout_sids.npy", np.sort(np.concatenate(blk_sids)))
    Rr = pd.DataFrame(rows)
    Rr.to_csv(OUT / "apply_changes.csv", index=False)
    Bk = pd.DataFrame(buck)
    Bk.to_csv(OUT / "apply_buckets.csv", index=False)
    F = pd.DataFrame(fd_rows)
    F.to_csv(OUT / "apply_fd_frame.csv", index=False)
    fdp = F.groupby(["split", "panel"])[["FD_V58", "FD_V59"]].mean()
    gb = Bk.groupby(["dmin", "dmax"])[["nT", "nH", "T_ss_dv", "H_ss_dv", "T_ss_dq", "H_ss_dq"]].sum()
    summ = {"config": cfg, "config_name": cname, "changes": rows,
            "S_FD_real_frame_family_mean": {
                s: {k: M.family_mean(fdp.loc[s][k].to_dict()) for k in ("FD_V58", "FD_V59")} for s in SPLITS},
            "S_FD_real_frame_per_corridor": {s: (fdp.loc[s].FD_V59 - fdp.loc[s].FD_V58).to_dict() for s in SPLITS},
            "change_rms_by_distance_bucket": {f"{a}-{b}": {"nT": int(r.nT), "nH": int(r.nH),
                                                           "T_rms_dv": float(np.sqrt(r.T_ss_dv / max(r.nT, 1))),
                                                           "H_rms_dv": float(np.sqrt(r.H_ss_dv / max(r.nH, 1))),
                                                           "T_rms_dq": float(np.sqrt(r.T_ss_dq / max(r.nT, 1))),
                                                           "H_rms_dq": float(np.sqrt(r.H_ss_dq / max(r.nH, 1)))}
                                              for (a, b), r in gb.iterrows()},
            "outputs_sha256": {s: M.sha256(CSV[s]) for s in SPLITS},
            "v51_zip_sha256": M.sha256(V51_ZIP), "v58_zip_sha256": M.sha256(V58_ZIP),
            "seconds": round(time.time() - t0)}
    gate = True
    for s in SPLITS:
        x = summ["S_FD_real_frame_family_mean"][s]
        x["delta_V59_vs_V58"] = x["FD_V59"] - x["FD_V58"]
        gate &= x["delta_V59_vs_V58"] > -1e-4
    summ["gate_S_FD_real_frame_ok"] = bool(gate)
    write_new(OUT / "apply_summary.json", json.dumps(summ, indent=2))
    pd.set_option("display.width", 320)
    print(Rr.round(4).to_string(), flush=True)
    print(json.dumps({k: summ[k] for k in ("S_FD_real_frame_family_mean", "gate_S_FD_real_frame_ok",
                                           "change_rms_by_distance_bucket")}, indent=2), flush=True)
    assert gate, "real-frame S_FD gate failed"


# ---------------------------------------------------------------------------
# package / audit / compare
# ---------------------------------------------------------------------------
def _patch():
    V.NAME = NAME
    V.OUT = OUT
    V.CSV = CSV
    M.NAME = NAME
    M.OUT = OUT
    M.BASE_ZIP = V51_ZIP
    M.new_state_arrays = V._new_state_arrays


def cmd_package(args) -> None:
    cname, cfg = chosen_cfg()
    _patch()
    M.cmd_package(args)
    man = json.loads((WORK / "results" / f"{NAME}_manifest.json").read_text(encoding="utf-8"))
    sup = {"name": NAME, "zip_sha256": man["sha256"], "changed_state_rows_vs_V51": man["changed_state_rows"],
           "note": "state = V58 state rows (v58_rs_blk.zip) with the free non-blackout target rows replaced by the V59 "
                   "restack with local neighbourhood / missingness features + refit V56 correction (t1_v59_local.py); "
                   "Task 2 blackout target rows = V57_blk (unchanged from V58); queue/ODME rows = V51 (v51_april.zip) "
                   "byte-identical; the full candidate takes the V54 queues via build_queue_candidate.py",
           "state_inputs": [{"path": str(CSV[s]), "sha256": M.sha256(CSV[s])} for s in SPLITS],
           "config": cfg, "config_name": cname}
    write_new(OUT / "v59loc_manifest_supplement.json", json.dumps(sup, indent=2, ensure_ascii=False))


def cmd_audit(args) -> None:
    _patch()
    M.cmd_audit(args)


def cmd_compare(args) -> None:
    """v59_loc.zip vs v58_rs_blk.zip: queue/ODME lines byte-identical; state lines differ exactly where the CSVs
    differ from V58 (per split-panel counts = apply); every differing value equals the CSV value, %.6g text; no
    Task 2 blackout target row changed (blackout_sids.npy from apply)."""
    new = WORK / "submissions" / f"{FULL_NAME}.zip"
    path = OUT / "compare_v58.json"
    if path.exists():
        raise FileExistsError(path)
    _patch()
    sp, fl, K = V._new_state_arrays()
    split_arr = np.empty(len(sp), object)
    panel_arr = np.empty(len(sp), object)
    split_arr[K.submission_id.to_numpy()] = K.split.astype(str).to_numpy()
    panel_arr[K.submission_id.to_numpy()] = K.panel.astype(str).to_numpy()
    del K
    blk = np.zeros(len(sp), bool)
    blk[np.load(OUT / "blackout_sids.npy")] = True
    st = {"rows": 0, "state_rows": 0, "state_changed": 0, "state_changed_speed": 0, "state_changed_flow": 0,
          "blackout_state_rows": int(blk.sum()), "blackout_state_changed": 0,
          "queue_rows": 0, "queue_changed": 0, "odme_rows": 0, "odme_changed": 0,
          "state_value_mismatch_vs_csv": 0, "state_text_not_6g": 0, "changed_by_split_panel": {},
          "max_abs_speed_change": 0.0, "max_abs_flow_change": 0.0}
    with zipfile.ZipFile(V58_ZIP) as za, zipfile.ZipFile(new) as zb:
        assert za.testzip() is None and zb.testzip() is None
        with za.open(za.namelist()[0]) as fa, zb.open(zb.namelist()[0]) as fb:
            ha, hb = fa.readline(), fb.readline()
            assert ha == hb
            for la, lb in zip(fa, fb):
                st["rows"] += 1
                pa_, pb = la.rstrip(b"\r\n").split(b","), lb.rstrip(b"\r\n").split(b",")
                assert pa_[0] == pb[0] and pa_[1] == pb[1], st["rows"]
                if pa_[1] == b"state":
                    sid = int(pa_[0])
                    st["state_rows"] += 1
                    nv, nq = float(pb[2]), float(pb[3])
                    if nv != sp[sid] or nq != fl[sid]:
                        st["state_value_mismatch_vs_csv"] += 1
                    if la != lb:
                        if (b"%.6g" % nv) != pb[2] or (b"%.6g" % nq) != pb[3] or pa_[4:] != pb[4:]:
                            st["state_text_not_6g"] += 1
                        st["state_changed"] += 1
                        st["blackout_state_changed"] += int(blk[sid])
                        st["state_changed_speed"] += int(pa_[2] != pb[2])
                        st["state_changed_flow"] += int(pa_[3] != pb[3])
                        k = f"{split_arr[sid]}|{panel_arr[sid]}"
                        st["changed_by_split_panel"][k] = st["changed_by_split_panel"].get(k, 0) + 1
                        st["max_abs_speed_change"] = max(st["max_abs_speed_change"], abs(nv - float(pa_[2])))
                        st["max_abs_flow_change"] = max(st["max_abs_flow_change"], abs(nq - float(pa_[3])))
                elif pa_[1] == b"queue":
                    st["queue_rows"] += 1
                    st["queue_changed"] += int(la != lb)
                else:
                    st["odme_rows"] += 1
                    st["odme_changed"] += int(la != lb)
            assert fa.read() == b"" and fb.read() == b""
    st["changed_by_split"] = {s: sum(v for k, v in st["changed_by_split_panel"].items() if k.startswith(s))
                              for s in SPLITS}
    ap_ = json.loads((OUT / "apply_summary.json").read_text())
    exp = {f"{r['split']}|{r['panel']}": r["changed_vs_V58"] for r in ap_["changes"]}
    st["changed_matches_apply"] = all(st["changed_by_split_panel"].get(k, 0) == v for k, v in exp.items())
    st["valid"] = bool(st["rows"] == 6980503 and st["queue_changed"] == 0 and st["odme_changed"] == 0 and
                       st["state_value_mismatch_vs_csv"] == 0 and st["state_text_not_6g"] == 0 and
                       st["blackout_state_changed"] == 0 and st["state_changed"] > 0 and st["changed_matches_apply"])
    st["v58_sha256"], st["v59_loc_sha256"] = M.sha256(V58_ZIP), M.sha256(new)
    write_new(path, json.dumps(st, indent=2))
    print(json.dumps(st, indent=2), flush=True)
    assert st["valid"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("features", "crossfit", "direct", "apply", "package", "audit", "compare"):
        sub.add_parser(name)
    args = ap.parse_args()
    {"features": cmd_features, "crossfit": cmd_crossfit, "direct": cmd_direct, "apply": cmd_apply,
     "package": cmd_package, "audit": cmd_audit, "compare": cmd_compare}[args.cmd](args)


if __name__ == "__main__":
    main()
