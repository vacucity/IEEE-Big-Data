"""V66 Task 1 (lane CM): corridor-wide per-timestep SPEED common-mode correction on top of V64 (+ zero-risk cap clip).

Finding (research round wf13/t3t4, re-checked here): the 5-minute speed noise of a corridor has a component shared
by all its links at the same timestamp (~32% of the noise variance, no decay with distance, none across corridors,
none in flow).  V64 leaves it in its free-flow speed errors.  It can be estimated at a target cell from the OTHER
links' visible observations of the same month (Task 1 is offline) and added to the V64 speed.

Features (identical code at the hide-20% cells and at the real targets, `frame_feats`):
  visible  = is_score_eligible & speed_obs, flow_obs finite & kr < 0.8 (kr = per-lane density / k_crit)
             (hide frame: minus ALL hide cells of the day; the hidden / target cell itself is never visible)
  D1(t)    = mean over the visible links with t-1, t, t+1 visible of v_t - (v_{t-1} + v_{t+1}) / 2   (NaN if < 5 links)
  Dw(t)    = mean over the visible links with >= 2 visible in t-3..t+3 \\ t of v_t - mean(visible v in t-3..t+3 \\ t)
             (NaN if < 5 links)
  cnt(t)   = number of links contributing to D1(t)
  x1       = [D1(t-2..t+2), Dw(t-1..t+1)] x {1, db==1, da==1} + const                                  (25 columns)
             (db / da = steps to the nearest visible cell of the same link before / after, t1_v59_local cache)
  x2       = speed band means of Pw = v - mean(visible v in t-3..t+3 \\ t) over the visible links at link distance
             1-2 / 3-5 / 6-10 below / above the cell, at t-1, t, t+1, + const                        (19 columns)
             (= wf13/t3t4/local_mode.py speed band, without its flow band)

PRE-REGISTERED DECISION RULE (fixed before `check` was run; `register` writes it with a timestamp to
results/t1_v66_cmode/rule.json, which `check` requires and copies into check.json):
  Row set R (the only rows V66 may change): V64 Task 1 state rows in the V59_loc row set -- targets predicted free by
    V51 (pv > 1.15 v_cut) and outside the official Task 2 blackouts -- with cnt(t) >= 5 at the real frame.
    Only speed_kmh changes; flow_vph and every other row (congested, blackout, queue, ODME) stay byte-identical.
  Candidates (only these two, lambda = 1, no other variants, no tuning):
    C1 'wide'       v' = clip(v64 + x1'b1, 5, 120.75)
    C2 'wide+vband' v' = clip(v64 + x1'b1 + x2'b2, 5, 120.75), b2 = OLS of (ov - v64 - x1'b1) on x2 with the
                    stage-1 b1 of the same training cells (in-sample stage 1, then stage 2; same in every fold)
  Fit: per month, OLS pooled over the ten corridors on that month's hide cells that are free (V51-chain hide
    prediction > 1.15 v_cut) and have cnt >= 5; target ov - v64 (v64 = hide-frame V64: cross-fitted V59_loc on
    free cells, pooled V64 model on congested cells).  Evaluation = odd/even-day cross-fit; apply = all days.
  Gates (each month separately, family means over the corridors, hide frame, parity cross-fit, relative to V64):
    G1 dS_rw > 0          (S_state reweighted to the real free non-blackout target strata, t1_v59_local)
    G2 >= 9/10 corridors with dS > 0
    G3 dTotal40 = 0.35 dS + 0.15 (dFD_est - 2 * 0.4 * dE_est) / 3 > 0     (LWR proxy credited at 40%)
    G4 (apply, real frame) S_FD change of the final values >= -0.0005 (family mean of the per-corridor regime mean)
  Choice: C2 if C2 passes G1-G3 AND in both months dS(C2) - dS(C1) >= 5e-5 AND dE_est(C2) <= dE_est(C1);
    otherwise C1 if C1 passes G1-G3; otherwise nothing is built.  If the chosen candidate fails G4 in apply: if it
    is C2 and C1 passed G1-G3, C1 is applied instead (G4 re-checked); otherwise nothing is built.
  Never: public-LB feedback, Task 2 data (only the public blackout flag to exclude those rows), other-month data.
Add-on (documented separately; not part of the decision; applied after the CM step to the same row set R):
  cap   v <= 101.2 on the links whose train speed maximum is exactly 101.2 with > 50% of the train cells at it
        (expected: 11 links, D7_I10_E 8 / D7_I10_W 3; checked against the March / April visible maxima)
  kjam  v >= q / k_jam (scorer's k_jam = max(k_jam, 1.05 k_crit)) where q / v > k_jam.  The seven V64 rows with
        q / v > k_jam are congested / blackout rows OUTSIDE R and therefore stay unchanged (same-row-set rule).

Subcommands (project root, anaconda python):
  register  write results/t1_v66_cmode/rule.json (write-once, timestamp, sha256 of this file)
  check     hide-frame cross-fit of C1 / C2 (+ half-month, other-month, day-shift placebo, cap add-on for info);
            gates and choice -> check.json, check_corridors.csv, grams.npz, cap_links.json
  apply     full-month betas; real-frame features; V66 speeds on R; S_FD gate; stream v64_pool.zip ->
            work/submissions/v66_cm.zip replacing only the speed field; validate; manifest
  audit     independent line-by-line comparison v64_pool.zip vs v66_cm.zip -> audit_v2.json
Outputs: work/results/t1_v66_cmode/, work/results/v66_cm_manifest.json, work/submissions/v66_cm.zip.
Nothing existing is overwritten (FileExistsError).  See work/V66_CMODE_README.md.
"""
from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "4")

import gzip
import hashlib
import io
import json
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
import t1_v52_monthbias as M  # noqa: E402
import t1_v57_restack as V  # noqa: E402
import t1_v59_local as LOC  # noqa: E402
import t1_v61_congest as Cg  # noqa: E402
from evaluate import CLEAN, KEY_BASE, PANELS, STEPS, links  # noqa: E402

WORK = Path(__file__).resolve().parent
SUB = WORK / "submissions"
OUT = WORK / "results" / "t1_v66_cmode"
CACHE = OUT / "cache"
BASE, NAME = "v64_pool", "v66_cm"
SPLITS = ("validation", "private")
CFG64 = Cg.CFGS["F600a75s"]
LAM64 = 0.9                                     # V64 = t1_v64_apply.CANDS["pooled31_l09"]
Z64 = WORK / "results" / "t1_v65_lwrcong" / "cache" / "v64cf.npz"
V59CF = WORK / "results" / "t1_v61_congest" / "cache"
PAD = 6
KR_MAX = 0.8
MIN_CNT = 5
BANDS = ((1, 2), (3, 5), (6, 10))
N1, N2 = 25, 19
W_STATE, W_PHYS, LWR_REAL = 0.35, 0.15, 0.4
CAP_VALUE = 101.2
CANDS = ("C1", "C2")
MODES = ("parity", "half_month", "other_month")

RULE = {
    "row_set": "V64 T1 rows with V51 pv > 1.15 v_cut, outside the official T2 blackouts (= V59_loc row set) and "
               "real-frame cnt >= 5; only speed_kmh changes",
    "candidates": {"C1": "wide: v' = clip(v64 + x1'b1, 5, 120.75), x1 = [D1(t-2..t+2), Dw(t-1..t+1)] x {1, db==1, "
                         "da==1} + const",
                   "C2": "wide+vband: v' = clip(v64 + x1'b1 + x2'b2, 5, 120.75), x2 = 18 speed band means + const, "
                         "b2 on the in-sample stage-1 residual"},
    "lambda": 1.0,
    "fit": "per month, OLS pooled over the 10 corridors, hide cells free & cnt >= 5, target ov - v64; evaluation "
           "odd/even-day cross-fit; apply all days",
    "gates_each_month": {"G1": "family dS_rw > 0", "G2": ">= 9/10 corridors dS > 0",
                         "G3": "family dTotal40 = 0.35 dS + 0.15 (dFD_est - 0.8 dE_est)/3 > 0",
                         "G4": "apply: real-frame S_FD change (family mean) >= -0.0005"},
    "choice": "C2 if C2 passes G1-G3 and in both months dS(C2)-dS(C1) >= 5e-5 and dE_est(C2) <= dE_est(C1); else C1 "
              "if C1 passes G1-G3; else not built.  G4 failure: C2 -> C1 if C1 passed G1-G3 (re-check G4); else "
              "not built",
    "addon": "after CM on the same row set: v <= 101.2 on the train-cap-101.2 links; v >= q/k_jam where q/v > k_jam "
             "(the 7 V64 rows with q/v > k_jam are outside R and stay unchanged)",
    "never": "public LB, Task 2 data other than the public blackout flag, other-month data",
}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for blk in iter(lambda: f.read(1 << 22), b""):
            h.update(blk)
    return h.hexdigest()


def write_new(path: Path, text: str) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.write_text(text, encoding="utf-8")


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S %z")


# ---------------------------------------------------------------------------
# features (identical code at hide cells and real targets)
# ---------------------------------------------------------------------------
def frame_feats(p: str, s: str, day, tt, pos, fd: dict, hide: bool) -> dict:
    """Common-mode features at the cells (day, tt, pos).  hide=True: the cells themselves are removed from the
    visibility of their day (hide frame); hide=False: real frame (the cells are real targets, never visible)."""
    n = fd["n"]
    day, tt, pos = (np.asarray(a, np.int64) for a in (day, tt, pos))
    pf = pq.ParquetFile(CLEAN / p / f"{s}.parquet")
    nd = pf.num_row_groups
    D1 = np.full((nd, STEPS + 2 * PAD), np.nan)
    Dw = np.full((nd, STEPS + 2 * PAD), np.nan)
    K = np.zeros((nd, STEPS + 2 * PAD))
    band = np.zeros((len(day), len(BANDS) * 2 * 3))
    for dd in range(nd):
        tb = pf.read_row_group(dd, columns=["is_score_eligible", "speed_obs", "flow_obs"])
        g = lambda c: tb.column(c).to_numpy(zero_copy_only=False).reshape(STEPS, n)  # noqa: E731
        v, q, el = g("speed_obs").astype(float), g("flow_obs").astype(float), g("is_score_eligible")
        ok = el & np.isfinite(v) & np.isfinite(q)
        m = np.flatnonzero(day == dd)
        hid = np.zeros((STEPS, n), bool)
        if hide:
            hid[tt[m], pos[m]] = True
        with np.errstate(invalid="ignore", divide="ignore"):
            kr = q / fd["lanes"][None] / np.maximum(v, 1.0) / fd["kcl"][None]
        okv = ok & ~hid & (kr < KR_MAX)
        assert not okv[tt[m], pos[m]].any(), ("feature cell visible", p, s, dd)
        Yz = np.where(okv, v, 0.0)
        # D1: second difference, cross-sectional mean
        a = Yz[1:-1] - 0.5 * (Yz[:-2] + Yz[2:])
        mm = okv[1:-1] & okv[:-2] & okv[2:]
        cnt = mm.sum(1)
        D1[dd, PAD + 1:PAD + STEPS - 1] = np.where(cnt >= MIN_CNT, np.where(mm, a, 0).sum(1) / np.maximum(cnt, 1), np.nan)
        K[dd, PAD + 1:PAD + STEPS - 1] = cnt
        # Pw: deviation from the visible t-3..t+3 \ t mean of the same link
        S = np.zeros((STEPS, n))
        Nn = np.zeros((STEPS, n))
        for L in (-3, -2, -1, 1, 2, 3):
            sh = np.zeros((STEPS, n))
            ok2 = np.zeros((STEPS, n), bool)
            if L > 0:
                sh[:-L], ok2[:-L] = Yz[L:], okv[L:]
            else:
                sh[-L:], ok2[-L:] = Yz[:L], okv[:L]
            S += np.where(ok2, sh, 0)
            Nn += ok2
        mw = okv & (Nn >= 2)
        Pw = np.where(mw, Yz - S / np.maximum(Nn, 1), 0.0)
        cw = mw.sum(1)
        Dw[dd, PAD:PAD + STEPS] = np.where(cw >= MIN_CNT, Pw.sum(1) / np.maximum(cw, 1), np.nan)
        # speed band means at link distance 1-2 / 3-5 / 6-10 below / above, t-1, t, t+1
        if len(m):
            cs = np.concatenate([np.zeros((STEPS, 1)), np.cumsum(Pw, 1)], 1)
            cc = np.concatenate([np.zeros((STEPS, 1)), np.cumsum(mw, 1)], 1)
            ti, li = tt[m], pos[m]
            cols = []
            for (lo_, hi_) in BANDS:
                for side in (-1, 1):
                    if side > 0:
                        lo, hi = np.clip(li + lo_, 0, n), np.clip(li + hi_ + 1, 0, n)
                    else:
                        lo, hi = np.clip(li - hi_, 0, n), np.clip(li - lo_ + 1, 0, n)
                    for dt in (-1, 0, 1):
                        t2 = np.clip(ti + dt, 0, STEPS - 1)
                        ssum = cs[t2, hi] - cs[t2, lo]
                        c_ = cc[t2, hi] - cc[t2, lo]
                        cols.append(np.where(c_ > 0, ssum / np.maximum(c_, 1), 0.0))
            band[m] = np.column_stack(cols)
    gx = lambda A, dy, L: np.nan_to_num(A[dy, tt + PAD + L], nan=0.0)  # noqa: E731
    sh_day = (day + 1) % nd                       # placebo: same t, next day of the month
    return {"D1": np.column_stack([gx(D1, day, L) for L in range(-2, 3)]),
            "Dw": np.column_stack([gx(Dw, day, L) for L in (-1, 0, 1)]),
            "cnt": K[day, tt + PAD], "band": band,
            "D1_pl": np.column_stack([gx(D1, sh_day, L) for L in range(-2, 3)]),
            "Dw_pl": np.column_stack([gx(Dw, sh_day, L) for L in (-1, 0, 1)])}


def design1(F: dict, Lc: dict, placebo: bool = False) -> np.ndarray:
    X = np.column_stack([F["D1_pl"], F["Dw_pl"]] if placebo else [F["D1"], F["Dw"]])
    pb = (Lc["db"] == 1).astype(float)[:, None]
    na = (Lc["da"] == 1).astype(float)[:, None]
    return np.column_stack([X, X * pb, X * na, np.ones(len(X))])


def design2(F: dict) -> np.ndarray:
    return np.column_stack([F["band"].astype(float), np.ones(len(F["band"]))])


CHUNK = 200_000


def _sub(F: dict, Lc: dict, k: np.ndarray) -> tuple[dict, dict]:
    return {key: F[key][k] for key in ("D1", "Dw", "band", "D1_pl", "Dw_pl")}, {"db": Lc["db"][k], "da": Lc["da"][k]}


def cm_pred(F: dict, Lc: dict, idx: np.ndarray, b1, b2=None, placebo: bool = False) -> np.ndarray:
    """x1'b1 (+ x2'b2) at the cells idx, in chunks (memory)."""
    out = np.zeros(len(idx))
    for st in range(0, len(idx), CHUNK):
        Fk, Lk = _sub(F, Lc, idx[st:st + CHUNK])
        y = design1(Fk, Lk, placebo) @ b1
        if b2 is not None:
            y = y + design2(Fk) @ b2
        out[st:st + CHUNK] = y
    return out


def hide_feats(p: str, s: str, d: dict) -> dict:
    f = CACHE / f"H_{s}_{p}.npz"
    if f.exists():
        z = np.load(f)
        return {k: z[k] for k in z.files}
    CACHE.mkdir(parents=True, exist_ok=True)
    C = d["C"]
    F = frame_feats(p, s, C.day.to_numpy(), C.t.to_numpy(), C.pos.to_numpy(), d["fd"], hide=True)
    np.savez(f, **F)
    return F


# ---------------------------------------------------------------------------
# hide frame
# ---------------------------------------------------------------------------
def hide_v64(p: str, s: str, mi: int, d: dict, off: int, z64) -> tuple[np.ndarray, np.ndarray, int]:
    """Hide-frame V64: cross-fitted V59_loc (free cells) + pooled V64 congested model (congested cells)."""
    zb = np.load(V59CF / f"v59cf_{s}_{p}.npz")
    v64, q64 = zb["v"].astype(float), zb["q"].astype(float)
    cong = ~d["free"]
    nc = int(cong.sum())
    rh, sh = z64[f"rh{mi}"][off:off + nc], z64[f"sh{mi}"][off:off + nc]
    v64[cong], q64[cong] = Cg.corrected(d["pv"][cong], d["pq"][cong], rh, sh, CFG64, LAM64)
    return v64, q64, off + nc


def solve(G: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """b1 (stage 1) and b2 (stage 2 on the in-sample stage-1 residual) from the Gram of Z = [x1, x2, e]."""
    ie = N1 + N2
    A1, c1 = G[:N1, :N1], G[:N1, ie]
    b1 = np.linalg.lstsq(A1, c1, rcond=None)[0]
    A2 = G[N1:ie, N1:ie]
    c2 = G[N1:ie, ie] - G[N1:ie, :N1] @ b1
    b2 = np.linalg.lstsq(A2, c2, rcond=None)[0]
    return b1, b2


def solve_pl(G: np.ndarray) -> np.ndarray:
    return np.linalg.lstsq(G[:N1, :N1], G[:N1, N1], rcond=None)[0]


def cap_links_check() -> dict:
    """Links whose train speed maximum is exactly CAP_VALUE with > 50% of the eligible train cells at it, and the
    March / April visible maxima on them."""
    out = {}
    for p in PANELS:
        if not p.startswith("D7_I10"):
            continue
        tr = pq.read_table(CLEAN / p / "train.parquet", columns=["link_id", "speed_obs", "is_score_eligible"]).to_pandas()
        tr = tr[tr.is_score_eligible & tr.speed_obs.notna()]
        g = tr.groupby("link_id", observed=True).speed_obs
        cap = g.max()
        frac = (tr.speed_obs.to_numpy() >= cap.reindex(tr.link_id).to_numpy() - 1e-3)
        frac = pd.Series(frac, index=tr.link_id.to_numpy()).groupby(level=0).mean()
        sel = cap.index[(np.abs(cap.to_numpy() - CAP_VALUE) < 1e-6) & (frac.reindex(cap.index).to_numpy() > 0.5)]
        del tr
        mx = {}
        for s in SPLITS:
            va = pq.read_table(CLEAN / p / f"{s}.parquet", columns=["link_id", "speed_obs", "is_score_eligible"]).to_pandas()
            va = va[va.is_score_eligible & va.speed_obs.notna() & va.link_id.isin(list(sel))]
            mx[s] = {str(k): float(x) for k, x in va.groupby("link_id", observed=True).speed_obs.max().items()}
            del va
        out[p] = {"links": sorted(str(x) for x in sel), "train_frac_at_cap": {str(k): float(frac[k]) for k in sel},
                  "month_max": mx}
    return out


def cmd_register() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rec = {"registered_at": now(), "script": str(Path(__file__).resolve()), "script_sha256": sha256(Path(__file__)),
           "rule": RULE, "note": "written before any check run of this script; the rule is also in the docstring"}
    write_new(OUT / "rule.json", json.dumps(rec, indent=2))
    print(json.dumps(rec, indent=2))


def score_rec(d, v1, q64, s64, a64, w, base_rw):
    s1 = M.state_score(d["C"], v1, q64, d["fd"])
    a1 = M.t3_score(d["C"], d["T"], d["Dd"], v1, q64, d["fd"])
    t3 = M.t3_delta(a64, a1, d["sc"])
    dS = s1["S"] - s64["S"]
    srw = LOC.s_state_w(v1, q64, d["ov"], d["oq"], d["lanes"], d["reg"], w) - base_rw
    rec = {"dS": dS, "dS_rw": srw, "dFD_est": t3["dFD_est"], "dE_est": t3["dE_est"], "dphys_est": t3["dphys_est"],
           "dTotal_est": W_STATE * dS + W_PHYS * t3["dphys_est"],
           "dTotal40": W_STATE * dS + W_PHYS * (t3["dFD_est"] - 2 * LWR_REAL * t3["dE_est"]) / 3}
    for r in range(3):
        if f"R{r + 1}_rv" in s1:
            rec[f"R{r + 1}_drv"] = s1[f"R{r + 1}_rv"] - s64[f"R{r + 1}_rv"]
    return rec


def cmd_check() -> None:
    rule = json.loads((OUT / "rule.json").read_text())
    for f in ("check.json", "check_corridors.csv", "grams.npz"):
        if (OUT / f).exists():
            raise FileExistsError(OUT / f)
    t0 = time.time()
    caps = cap_links_check()
    write_new(OUT / "cap_links.json", json.dumps(caps, indent=2))
    print("cap links", {p: c["links"] for p, c in caps.items()}, f"{time.time() - t0:.0f}s", flush=True)
    z64 = np.load(Z64)
    # ---- pass A: Gram matrices per (month, parity, half) ----
    G = {(s, a, h): np.zeros((N1 + N2 + 1, N1 + N2 + 1)) for s in SPLITS for a in (0, 1) for h in (0, 1)}
    Gp = {(s, a): np.zeros((N1 + 1, N1 + 1)) for s in SPLITS for a in (0, 1)}
    cov = []
    for mi, s in enumerate(SPLITS):
        off = 0
        for p in PANELS:
            d, Lh = LOC.load(p, s)
            v64, q64, off = hide_v64(p, s, mi, d, off, z64)
            F = hide_feats(p, s, d)
            m = d["free"] & (F["cnt"] >= MIN_CNT)
            idx = np.flatnonzero(m)
            e = d["ov"] - v64
            for st in range(0, len(idx), CHUNK):
                k_ = idx[st:st + CHUNK]
                Fk, Lk = _sub(F, Lh, k_)
                Z = np.column_stack([design1(Fk, Lk), design2(Fk), e[k_]])
                Zp = np.column_stack([design1(Fk, Lk, placebo=True), e[k_]])
                par, half = d["par"][k_], (d["day"][k_] >= 15).astype(int)
                for a in (0, 1):
                    for h in (0, 1):
                        k = (par == a) & (half == h)
                        G[(s, a, h)] += Z[k].T @ Z[k]
                    Gp[(s, a)] += Zp[par == a].T @ Zp[par == a]
                del Z, Zp
            cov.append({"split": s, "panel": p, "hide_free": int(d["free"].sum()), "hide_m": int(m.sum()),
                        "hide_cov": float(m.sum() / max(d["free"].sum(), 1))})
            print("A", s, p, cov[-1], f"{time.time() - t0:.0f}s", flush=True)
            del d, Lh, F
        assert off == len(z64[f"rh{mi}"]), ("congested offset", s, off)
    np.savez(OUT / "grams.npz", **{f"G_{s}_{a}_{h}": G[(s, a, h)] for s in SPLITS for a in (0, 1) for h in (0, 1)},
             **{f"Gp_{s}_{a}": Gp[(s, a)] for s in SPLITS for a in (0, 1)})
    B = {}
    for s in SPLITS:
        for a in (0, 1):
            B[(s, "par", a)] = solve(G[(s, a, 0)] + G[(s, a, 1)])
            B[(s, "half", a)] = solve(G[(s, 0, a)] + G[(s, 1, a)])
            B[(s, "pl", a)] = solve_pl(Gp[(s, a)])
        B[(s, "all")] = solve(sum(G[(s, a, h)] for a in (0, 1) for h in (0, 1)))
    # ---- pass B: scores ----
    rows = []
    capset = {p: set(c["links"]) for p, c in caps.items()}
    for mi, s in enumerate(SPLITS):
        o = [x for x in SPLITS if x != s][0]
        off = 0
        for p in PANELS:
            d, Lh = LOC.load(p, s)
            v64, q64, off = hide_v64(p, s, mi, d, off, z64)
            F = hide_feats(p, s, d)
            m = d["free"] & (F["cnt"] >= MIN_CNT)
            LT = LOC.load_local(p, s, "T")
            w, unc = LOC.target_weights(d, Lh, LT, d["fd"])
            del LT
            s64 = M.state_score(d["C"], v64, q64, d["fd"])
            a64 = M.t3_score(d["C"], d["T"], d["Dd"], v64, q64, d["fd"])
            base_rw = LOC.s_state_w(v64, q64, d["ov"], d["oq"], d["lanes"], d["reg"], w)
            lid = links(p).set_index("pos").link_id.reindex(d["pos"]).to_numpy()
            oncap = np.isin(lid, list(capset.get(p, set()))) & m
            half = d["day"] >= 15
            for mode in MODES:
                for cand in CANDS:
                    pred = np.zeros(len(v64))
                    if mode == "other_month":
                        b1, b2 = B[(o, "all")]
                        te = np.flatnonzero(m)
                        pred[te] = cm_pred(F, Lh, te, b1, b2 if cand == "C2" else None)
                    else:
                        for a in (0, 1):
                            te = np.flatnonzero(((d["par"] != a) if mode == "parity" else (half != bool(a))) & m)
                            b1, b2 = B[(s, "par" if mode == "parity" else "half", a)]
                            pred[te] = cm_pred(F, Lh, te, b1, b2 if cand == "C2" else None)
                    v1 = np.clip(v64 + pred, M.SPEED_MIN, M.SPEED_MAX)
                    rec = score_rec(d, v1, q64, s64, a64, w, base_rw)
                    rows.append({"split": s, "panel": p, "cand": cand, "mode": mode, **rec,
                                 "n_changed": int(m.sum()), "rms_dv": float(np.sqrt(np.mean(pred[m] ** 2)))})
                    if mode == "parity":
                        vc = np.where(oncap, np.minimum(v1, CAP_VALUE), v1)
                        rec = score_rec(d, vc, q64, s64, a64, w, base_rw)
                        rows.append({"split": s, "panel": p, "cand": cand + "+cap", "mode": mode, **rec,
                                     "n_changed": int(m.sum()), "n_cap_clipped": int((vc < v1).sum())})
            pred = np.zeros(len(v64))
            for a in (0, 1):
                te = np.flatnonzero((d["par"] != a) & m)
                pred[te] = cm_pred(F, Lh, te, B[(s, "pl", a)], placebo=True)
            rec = score_rec(d, np.clip(v64 + pred, M.SPEED_MIN, M.SPEED_MAX), q64, s64, a64, w, base_rw)
            rows.append({"split": s, "panel": p, "cand": "C1_placebo_dayshift", "mode": "parity", **rec,
                         "n_changed": int(m.sum())})
            print("B", s, p, {r_["cand"] + "/" + r_["mode"][:3]: round(r_["dS"] * 1e4, 3) for r_ in rows[-9:]},
                  f"{time.time() - t0:.0f}s", flush=True)
            del d, Lh, F
        assert off == len(z64[f"rh{mi}"]), ("congested offset", s, off)
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "check_corridors.csv", index=False)
    # ---- gates and choice (pre-registered) ----
    fam = lambda g, k: M.family_mean(dict(zip(g.panel, g[k])))  # noqa: E731
    summ = {}
    for cand in sorted(R.cand.unique()):
        for mode in MODES:
            for s in SPLITS:
                g = R[(R.cand == cand) & (R["mode"] == mode) & (R.split == s)]
                if g.empty:
                    continue
                summ.setdefault(cand, {}).setdefault(mode, {})[s] = {
                    **{k: fam(g, k) for k in ("dS", "dS_rw", "dFD_est", "dE_est", "dphys_est", "dTotal_est", "dTotal40")},
                    "npos_dS": int((g.dS > 0).sum()), "min_dS": float(g.dS.min())}
    gates = {}
    for cand in CANDS:
        gm = {}
        for s in SPLITS:
            x = summ[cand]["parity"][s]
            gm[s] = {"G1_dS_rw>0": x["dS_rw"] > 0, "G2_npos>=9": x["npos_dS"] >= 9, "G3_dTotal40>0": x["dTotal40"] > 0}
        gates[cand] = {"per_month": gm, "passed_G1_G3": all(all(v.values()) for v in gm.values())}
    c1, c2 = summ["C1"]["parity"], summ["C2"]["parity"]
    margin = {s: {"dS_gain": c2[s]["dS"] - c1[s]["dS"], "dE_C2_minus_C1": c2[s]["dE_est"] - c1[s]["dE_est"]}
              for s in SPLITS}
    c2_better = all(margin[s]["dS_gain"] >= 5e-5 and margin[s]["dE_C2_minus_C1"] <= 0 for s in SPLITS)
    if gates["C2"]["passed_G1_G3"] and c2_better:
        chosen = "C2"
    elif gates["C1"]["passed_G1_G3"]:
        chosen = "C1"
    else:
        chosen = None
    # reproduction of the research numbers (wf13/t3t4/cm_wide_robust.csv parity dS, C1 = 'wide')
    repro = None
    ref = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    if ref is not None and ref.exists():
        rr = pd.read_csv(ref)
        rr = rr[rr["mode"] == "parity"].set_index(["split", "panel"]).dS
        mine = R[(R.cand == "C1") & (R["mode"] == "parity")].set_index(["split", "panel"]).dS
        repro = {"reference": str(ref), "max_abs_diff_dS": float((mine - rr.reindex(mine.index)).abs().max())}
    out = {"rule": rule, "checked_at": now(), "script_sha256": sha256(Path(__file__)), "summary": summ,
           "gates": gates, "C2_vs_C1": margin, "C2_margin_rule_met": bool(c2_better), "chosen": chosen,
           "betas_all": {s: {"b1": B[(s, "all")][0].tolist(), "b2": B[(s, "all")][1].tolist()} for s in SPLITS},
           "hide_coverage": cov, "reproduction_vs_research": repro, "seconds": round(time.time() - t0)}
    write_new(OUT / "check.json", json.dumps(out, indent=2, default=float))
    pd.set_option("display.width", 250)
    for cand in summ:
        for mode in summ[cand]:
            for s in summ[cand][mode]:
                x = summ[cand][mode][s]
                print(f"{cand:22s} {mode:11s} {s:10s} " + " ".join(f"{k} {v * 1e4:+.3f}e-4" for k, v in x.items()
                                                                       if k not in ("npos_dS",)) + f" npos {x['npos_dS']}")
    print(json.dumps({"gates": gates, "C2_vs_C1": margin, "chosen": chosen, "repro": repro}, indent=2, default=float))


# ---------------------------------------------------------------------------
# real frame
# ---------------------------------------------------------------------------
def betas_from_grams(s: str) -> tuple[np.ndarray, np.ndarray]:
    z = np.load(OUT / "grams.npz")
    return solve(sum(z[f"G_{s}_{a}_{h}"] for a in (0, 1) for h in (0, 1)))


def fd_gate(FDf: pd.DataFrame, cand: str, which: str) -> dict:
    r = {}
    for s in SPLITS:
        g = FDf[(FDf.split == s) & (FDf.cand == cand) & (FDf.which == which)]
        g = g.groupby("panel")[["FD_V64", "FD_new"]].mean()
        r[s] = M.family_mean((g.FD_new - g.FD_V64).to_dict()) if len(g) else None
    return r


def cmd_apply() -> None:
    chk = json.loads((OUT / "check.json").read_text())
    if chk["chosen"] is None:
        raise SystemExit("check: no candidate passed G1-G3 -> V66 not built")
    out_zip = SUB / f"{NAME}.zip"
    man_path = WORK / "results" / f"{NAME}_manifest.json"
    for f in (out_zip, man_path, OUT / "apply_summary.json", OUT / "apply_corridors.csv", OUT / "apply_fd_frame.csv",
              OUT / "patch_v66_cm.csv.gz", OUT / "rowset_R_sids.npy"):
        if f.exists():
            raise FileExistsError(f)
    caps = json.loads((OUT / "cap_links.json").read_text())
    capset = {p: set(c["links"]) for p, c in caps.items()}
    t0 = time.time()
    K = M.read_key_state()
    sid, v64, q64 = V.zip_state(SUB / f"{BASE}.zip")
    assert np.array_equal(sid, K.submission_id.to_numpy())
    sid60, v60, q60 = V.zip_state(SUB / "v60_loc_blk.zip")
    assert np.array_equal(sid60, sid)
    keys = K.key.to_numpy()
    del sid60, K
    print("V64 + V60 state rows", len(sid), f"{time.time() - t0:.0f}s", flush=True)
    B = {s: betas_from_grams(s) for s in SPLITS}
    for s in SPLITS:
        assert np.allclose(B[s][0], chk["betas_all"][s]["b1"]) and np.allclose(B[s][1], chk["betas_all"][s]["b2"])
    vals = {c: [] for c in CANDS}          # cand -> [(sids of R, final speeds on R)]
    rows, fd_rows, R_sids = [], [], []
    for split in SPLITS:
        b1, b2 = B[split]
        for p in PANELS:
            fd = M.static_fd(p)
            LT = LOC.load_local(p, split, "T")
            ii = np.searchsorted(sid, LT["sid"])
            assert np.array_equal(sid[ii], LT["sid"]) and np.array_equal(keys[ii], LT["key"]), p
            a64, b64 = v64[ii], q64[ii]
            pos = (LT["key"] % KEY_BASE).astype(np.int64)
            tt = ((LT["key"] // KEY_BASE) % STEPS).astype(np.int64)
            free = (LT["v51"] > M.FREE_RATIO * fd["vc"][pos]) & ~LT["gap"]
            # lineage: V64 keeps the V59_loc values (= V60) on the V59_loc row set
            assert np.array_equal(a64[free], v60[ii][free]) and np.array_equal(b64[free], q60[ii][free]), ("lineage", p)
            F = frame_feats(p, split, LT["day"], tt, pos, fd, hide=False)
            Rm = free & (F["cnt"] >= MIN_CNT)
            iR = np.flatnonzero(Rm)
            lid = links(p).set_index("pos").link_id.reindex(pos[iR]).to_numpy()
            oncap = np.isin(lid, list(capset.get(p, set())))
            kj = (fd["kjl"] * fd["lanes"])[pos[iR]]
            qR, v0R = b64[iR], a64[iR]
            rec = {"split": split, "panel": p, "targets": len(LT["key"]), "blackout_rows": int(LT["gap"].sum()),
                   "v59loc_rows": int(free.sum()), "R_rows": int(len(iR)), "real_cov": float(len(iR) / max(free.sum(), 1)),
                   "real_cnt_median_free": float(np.median(F["cnt"][free])) if free.any() else np.nan,
                   "cap_link_rows_R": int(oncap.sum())}
            for cand in CANDS:
                cm = cm_pred(F, LT, iR, b1, b2 if cand == "C2" else None)
                vcm = np.clip(v0R + cm, M.SPEED_MIN, M.SPEED_MAX)
                vcap = np.where(oncap, np.minimum(vcm, CAP_VALUE), vcm)
                with np.errstate(divide="ignore", invalid="ignore"):
                    jam = qR / vcap > kj
                vf = V.g6(np.where(jam, qR / kj, vcap))
                vcm6 = V.g6(vcm)
                assert np.isfinite(vf).all() and (vf > 0).all()
                dvc = vcm - v0R
                rec.update({f"{cand}_mean_dv": float(dvc.mean()), f"{cand}_rms_dv": float(np.sqrt(np.mean(dvc ** 2))),
                            f"{cand}_max_abs_dv": float(np.abs(dvc).max()) if len(dvc) else 0.0,
                            f"{cand}_n_abs_dv_gt3": int((np.abs(dvc) > 3).sum()),
                            f"{cand}_changed_rows": int((vf != v0R).sum()),
                            f"{cand}_cap_clipped": int((oncap & (vcm6 > CAP_VALUE)).sum()),
                            f"{cand}_v64_over_cap_in_R": int((oncap & (v0R > CAP_VALUE)).sum()),
                            f"{cand}_kjam_fixed": int(jam.sum())})
                vals[cand].append((LT["sid"][iR], vf))
                todo = [("final", vf)] + ([("cm_only", vcm6)] if cand == chk["chosen"] else [])
                for which, vR in todo:
                    v1 = a64.copy()
                    v1[iR] = vR
                    for r_ in M.fd_frame(p, split, {"key": LT["key"]}, np.arange(len(a64)), a64, b64, v1, b64, fd):
                        fd_rows.append({"split": split, "panel": p, "cand": cand, "which": which,
                                        "regime": r_["regime"], "FD_V64": r_["FD0"], "FD_new": r_["FD1"]})
            # T vs H: size of the correction at the hide cells (in-sample full-month betas)
            zH = np.load(CACHE / f"H_{split}_{p}.npz")
            FH = {k: zH[k] for k in zH.files}
            CH = pd.read_parquet(M.HIDE / f"{p}_{split}_cells.parquet", columns=["pos", "pv"])
            LhH = LOC.load_local(p, split, "H")
            mH = (CH.pv.to_numpy() > M.FREE_RATIO * fd["vc"][CH.pos.to_numpy()]) & (FH["cnt"] >= MIN_CNT)
            freeH = CH.pv.to_numpy() > M.FREE_RATIO * fd["vc"][CH.pos.to_numpy()]
            iH = np.flatnonzero(mH)
            cH1 = cm_pred(FH, LhH, iH, b1)
            cH2 = cm_pred(FH, LhH, iH, b1, b2)
            rec.update({"H_rms_C1": float(np.sqrt(np.mean(cH1 ** 2))), "H_rms_C2": float(np.sqrt(np.mean(cH2 ** 2))),
                        "H_mean_C1": float(cH1.mean()), "H_mean_C2": float(cH2.mean()),
                        "H_cov": float(mH.sum() / max(freeH.sum(), 1))})
            del FH, zH, CH, LhH
            R_sids.append(LT["sid"][iR])
            rows.append(rec)
            print(split, p, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items()
                             if k not in ("split", "panel")}, f"{time.time() - t0:.0f}s", flush=True)
            del LT, F
    A = pd.DataFrame(rows)
    A.to_csv(OUT / "apply_corridors.csv", index=False)
    FDf = pd.DataFrame(fd_rows)
    FDf.to_csv(OUT / "apply_fd_frame.csv", index=False)
    g4 = {c: {"final": fd_gate(FDf, c, "final"), "cm_only": fd_gate(FDf, c, "cm_only")} for c in CANDS}
    passes = {c: all(x is not None and x >= -5e-4 for x in g4[c]["final"].values()) for c in CANDS}
    used = chk["chosen"]
    if not passes[used]:
        if used == "C2" and chk["gates"]["C1"]["passed_G1_G3"] and passes["C1"]:
            used = "C1"
        else:
            write_new(OUT / "apply_summary.json", json.dumps({"built": False, "reason": "G4 failed", "G4": g4,
                                                              "corridors": rows}, indent=2, default=float))
            raise SystemExit(f"G4 failed: {g4}")
    R_all = np.sort(np.concatenate(R_sids))
    np.save(OUT / "rowset_R_sids.npy", R_all)
    # patch: sorted sids whose %.6g speed changes
    ps = np.concatenate([x[0] for x in vals[used]])
    pv_new = np.concatenate([x[1] for x in vals[used]])
    del vals
    o = np.argsort(ps)
    ps, pv_new = ps[o], pv_new[o]
    jj = np.searchsorted(sid, ps)
    pv_old = v64[jj]
    ch = pv_new != pv_old
    ps, pv_new, pv_old = ps[ch], pv_new[ch], pv_old[ch]
    assert np.all(np.diff(ps) > 0)
    print("patch rows", len(ps), f"{time.time() - t0:.0f}s", flush=True)
    seen, k, prev = 0, 0, 0
    src = zipfile.ZipFile(SUB / f"{BASE}.zip")
    with src.open(src.namelist()[0]) as fin, zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zout, \
            zout.open(f"{NAME}.csv", "w", force_zip64=True) as dst:
        for i, raw in enumerate(io.BufferedReader(fin, 1 << 22)):
            if i == 0:
                assert raw.rstrip(b"\r\n") == b"submission_id,task,speed_kmh,flow_vph,queue_pred,path_flow"
                dst.write(raw)
                continue
            sid_ = int(raw.split(b",", 1)[0])
            assert sid_ > prev
            prev = sid_
            if k >= len(ps) or sid_ != ps[k]:
                dst.write(raw)
                continue
            body = raw.rstrip(b"\r\n")
            parts = body.split(b",")
            assert parts[1] == b"state" and len(parts) == 6 and float(parts[2]) == pv_old[k], (i, parts)
            parts[2] = b"%.6g" % pv_new[k]
            dst.write(b",".join(parts) + raw[len(body):])
            k += 1
            seen += 1
    assert seen == len(ps), (seen, len(ps))
    with zipfile.ZipFile(out_zip) as z:
        assert z.testzip() is None
    with gzip.open(OUT / "patch_v66_cm.csv.gz", "wt", encoding="utf-8", newline="") as f:
        pd.DataFrame({"submission_id": ps, "speed_v64": pv_old, "speed_v66": pv_new}).to_csv(
            f, index=False, float_format="%.6g")
    print("zip written", f"{time.time() - t0:.0f}s", flush=True)
    from make_submission import validate
    checks = validate(out_zip)
    if not checks["ok"]:
        raise ValueError(checks)
    summ = {"built": True, "candidate_used": used, "candidate_chosen_by_check": chk["chosen"], "G4": g4,
            "G4_passes": passes, "changed_rows": int(len(ps)), "R_rows": int(len(R_all)),
            "changed_by_split": {s: int(A[A.split == s][f"{used}_changed_rows"].sum()) for s in SPLITS},
            "cap_clipped_rows": int(A[f"{used}_cap_clipped"].sum()),
            "kjam_fixed_rows": int(A[f"{used}_kjam_fixed"].sum()),
            "corridors": rows, "checks": checks, "seconds": round(time.time() - t0)}
    write_new(OUT / "apply_summary.json", json.dumps(summ, indent=2, default=float))
    man = {"name": NAME, "base": BASE, "base_sha256": sha256(SUB / f"{BASE}.zip"), "output": str(out_zip),
           "output_sha256": sha256(out_zip), "script": "work/t1_v66_cmode.py", "script_sha256": sha256(Path(__file__)),
           "candidate": used, "changed_state_rows": int(len(ps)), "changed_columns": ["speed_kmh"],
           "row_set": RULE["row_set"], "rowset_R_rows": int(len(R_all)), "G4_real_frame_S_FD": g4[used],
           "patch": str(OUT / "patch_v66_cm.csv.gz"), "patch_sha256": sha256(OUT / "patch_v66_cm.csv.gz"),
           "checks": checks, "public_lb": None,
           "note": "V64 lines streamed; only the speed field of the changed state rows replaced (%.6g, line endings "
                   "kept); queue / ODME / flow bytes identical (see results/t1_v66_cmode/audit.json)"}
    write_new(man_path, json.dumps(man, indent=2, default=float))
    print(json.dumps({k_: summ[k_] for k_ in ("candidate_used", "G4", "changed_rows", "R_rows", "changed_by_split",
                                              "cap_clipped_rows", "kjam_fixed_rows", "checks")}, indent=2, default=float))


def zip_lines(path: Path):
    z = zipfile.ZipFile(path)
    assert len(z.namelist()) == 1
    with z.open(z.namelist()[0]) as f:
        for raw in io.BufferedReader(f, 1 << 22):
            yield raw


def cmd_audit() -> None:
    """Independent of apply's bookkeeping: every line of v66_cm.zip vs v64_pool.zip, against the saved row set R and
    the patch file; also the categories of the changed rows from the t1_v59_local target cache."""
    # audit.json (first run, 2026-10-24 12:26) failed ONLY on a wrong criterion -- "every line ends with CRLF" -- while
    # v64_pool.zip itself has 725 queue lines ending with LF; the criterion is now "same line endings as V64"
    AUD = OUT / "audit_v2.json"
    if AUD.exists():
        raise FileExistsError(AUD)
    R_all = np.load(OUT / "rowset_R_sids.npy")
    pt = pd.read_csv(OUT / "patch_v66_cm.csv.gz")
    patch = pt.submission_id.to_numpy()
    assert np.all(np.diff(patch) > 0) and np.isin(patch, R_all).all()
    st = {"rows": 0, "identical": 0, "changed": 0, "outside_R_changed": 0, "nonstate_changed": 0,
          "other_field_changed": 0, "line_ending_changed": 0, "changed_not_in_patch": 0, "patch_value_mismatch": 0,
          "by_task": {}, "crlf_lines": 0, "crlf_lines_v64": 0}
    dv = np.zeros(len(patch))
    k = 0
    for i, (l0, l1) in enumerate(zip(zip_lines(SUB / f"{BASE}.zip"), zip_lines(SUB / f"{NAME}.zip"))):
        if i == 0:
            assert l0 == l1
            continue
        st["rows"] += 1
        task = l0.split(b",", 2)[1].decode()
        st["by_task"][task] = st["by_task"].get(task, 0) + 1
        st["crlf_lines"] += l1.endswith(b"\r\n")
        st["crlf_lines_v64"] += l0.endswith(b"\r\n")
        if l0 == l1:
            st["identical"] += 1
            continue
        st["changed"] += 1
        if task != "state":
            st["nonstate_changed"] += 1
            continue
        b0, b1 = l0.rstrip(b"\r\n"), l1.rstrip(b"\r\n")
        if l0[len(b0):] != l1[len(b1):]:
            st["line_ending_changed"] += 1
        p0, p1 = b0.split(b","), b1.split(b",")
        if [x for j, x in enumerate(p0) if j != 2] != [x for j, x in enumerate(p1) if j != 2]:
            st["other_field_changed"] += 1
        sid_ = int(p0[0])
        j = np.searchsorted(R_all, sid_)
        if j >= len(R_all) or R_all[j] != sid_:
            st["outside_R_changed"] += 1
        while k < len(patch) and patch[k] < sid_:
            k += 1
        if k >= len(patch) or patch[k] != sid_:
            st["changed_not_in_patch"] += 1
            continue
        if float(p1[2]) != pt.speed_v66.iat[k] or float(p0[2]) != pt.speed_v64.iat[k]:
            st["patch_value_mismatch"] += 1
        dv[k] = float(p1[2]) - float(p0[2])
    st["patch_rows"] = int(len(patch))
    st["max_abs_dv"] = float(np.abs(dv).max()) if len(dv) else 0.0
    st["mean_dv"] = float(dv.mean()) if len(dv) else 0.0
    st["rms_dv"] = float(np.sqrt(np.mean(dv ** 2))) if len(dv) else 0.0
    # categories of the changed rows (t1_v59_local target cache: V51 free flag, blackout flag)
    cat = {"free_nonblackout": 0, "congested": 0, "blackout": 0}
    for s in SPLITS:
        for p in PANELS:
            LT = LOC.load_local(p, s, "T")
            fd = M.static_fd(p)
            pos = (LT["key"] % KEY_BASE).astype(np.int64)
            hit = np.isin(LT["sid"], patch)
            fr = LT["v51"] > M.FREE_RATIO * fd["vc"][pos]
            cat["free_nonblackout"] += int((hit & fr & ~LT["gap"]).sum())
            cat["congested"] += int((hit & ~fr & ~LT["gap"]).sum())
            cat["blackout"] += int((hit & LT["gap"]).sum())
            del LT
    st["changed_row_categories"] = cat
    st["ok"] = (st["nonstate_changed"] == 0 and st["other_field_changed"] == 0 and st["outside_R_changed"] == 0
                and st["line_ending_changed"] == 0 and st["changed_not_in_patch"] == 0
                and st["patch_value_mismatch"] == 0 and st["changed"] == len(patch) and st["crlf_lines"] == st["crlf_lines_v64"]
                and cat["congested"] == 0 and cat["blackout"] == 0 and cat["free_nonblackout"] == len(patch))
    st["v64_sha256"], st["v66_sha256"] = sha256(SUB / f"{BASE}.zip"), sha256(SUB / f"{NAME}.zip")
    write_new(AUD, json.dumps(st, indent=2))
    print(json.dumps(st, indent=2))
    assert st["ok"], st


if __name__ == "__main__":
    {"register": cmd_register, "check": cmd_check, "apply": cmd_apply, "audit": cmd_audit}[sys.argv[1]]()
