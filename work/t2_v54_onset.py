#!/usr/bin/env python
"""V54 lane B: onset block choice measured on official-like replica onset windows.

Evaluation set
  * replica onset windows of work/cache/t2_v52_selrule/windows.csv (official-like greedy selector, shared 360-min
    spacing, first qualifying origin; built by `python work/t2_v52_replica.py build`): the 9 train months (labels =
    speed_true <= v_cut on eligible cells, the official train-window proxy) and March (labels = visible masked
    layer), one row per unique (corridor, split, origin).  Spacing-bound onset windows (T = previous origin + 360
    min) are dropped: none of the 120 official onset windows (train / March / April) is spacing-bound, and the
    replica's spacing-bound onsets often queue before T+30.  `first5` marks the first five qualifying onsets of a
    month (the closest analogue of the official windows).
  * multi-block corridors only (D7_I210_W, D7_I405_S, D12_I5_N, D12_I5_S); the four others have one block.
  * April (private) replica / pseudo windows are never built or read here.

Models (choosers of the T+30 link set; the onset step itself is organizer-confirmed)
  * block models are fitted on the train pseudo onset windows of t2_pseudo (as production) and scored on the
    replica windows leave-one-train-month-out (fit without month m, score month m); March replica windows are
    scored by the all-train fit, using only their own history before T (causal).
  * ml2 (the D7_I210_W production model, LightGBM over all corridors) is refitted per held-out train month with
    the production parameters and 810 rounds; its time-of-day prior feature is recomputed without that month.

Pre-registered decision (fixed before the cv was run; train months only)
  Per corridor, a candidate replaces production iff on the train replica windows (LOMO)
    (i) pooled mean IoU gain > 0, (ii) day-clustered bootstrap 95 % lower bound > 0, and
    (iii) months better >= 5 of 9 and months better > months worse.
  If several pass, the one with the largest pooled train gain is used.  March replica windows are reported as
  out-of-sample confirmation only and are never used for a choice that affects March (public) rows.  For April
  rows (allowed to use all of March) the adopted rule is applied only if its March replica gain is >= 0.
  DEVIATION (after the first cv run; see eligible() and work/V54_ONSET_README.md): the literal rule picks ml2 on
  D7_I405_S / D12_I5_N / D12_I5_S (train time-of-day habit; March replica -0.13 / -0.19 / -0.38), so on the
  flow-chooser corridors the habit families (ml2, train phase prior) are not eligible.  decisions_literal.json keeps
  the literal outcome, decisions.json (used by build) the restricted one.

Usage (from the "IEEE Big data" folder; CPU only; OMP_NUM_THREADS=4)
  python work/t2_v54_onset.py cases      # replica / pseudo / official onset cases -> work/cache/t2_v54_onset/ (~1 min)
  python work/t2_v54_onset.py ml2        # LOMO ml2 refits (~4 min) -> work/cache/t2_v54_onset/ml2_probs.pkl
  python work/t2_v54_onset.py cv         # tables + decisions -> work/results/t2_v54_onset/ (~30 s)
  python work/t2_v54_onset.py selector   # record-only: does the selector add onset information? (~30 s)
  python work/t2_v54_onset.py build      # queue_*_v54_onset.csv from the V53 queues (refuses to overwrite; --dry-run)
  python work/t2_v54_onset.py asof-check <empty scratch dir>   # per-window truncation test of the official choice
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent))
import t2_onset as O                 # noqa: E402
import t2_onset_cap as CAP           # noqa: E402
import t2_v52_d12s_t15 as D12        # noqa: E402  (feats(); read-only use)
from evaluate import RELEASE, ROOT, T2_PANELS, links, slot_of  # noqa: E402
from t2_grids import load            # noqa: E402
from t2_pseudo import HIST, HOR, Win, truth, windows  # noqa: E402

warnings.simplefilter("ignore")
WORK = ROOT / "work"
CACHE = WORK / "cache" / "t2_v54_onset"
OUT = WORK / "results" / "t2_v54_onset"
REPLICA = WORK / "cache" / "t2_v52_selrule" / "windows.csv"
PANELS = ("D7_I210_W", "D7_I405_S", "D12_I5_N", "D12_I5_S")
BASE_Q = {"validation": WORK / "sub" / "queue_validation_v52_selrule.csv",
          "private": WORK / "sub" / "queue_private_v52_selrule.csv"}
NEW_Q = {"validation": WORK / "sub" / "queue_validation_v54_onset.csv",
         "private": WORK / "sub" / "queue_private_v54_onset.csv"}
ML2_PROD = WORK / "models" / "t2_ml2_onset_v2.txt"
ML2_ROUNDS = 810                     # trees in the production onset booster (int(1.1 * mean early-stop rounds))
THREADS = 4
N_BOOT = 2000


# ---------------------------------------------------------------------------
# features
# ---------------------------------------------------------------------------
def _lf(Q: np.ndarray, members: np.ndarray, steps: slice) -> float:
    v = O.block_flows(Q, members, steps)
    return float(np.log(max(v, 100.0))) if np.isfinite(v) else np.nan


def feats13(panel: str, Q13: np.ndarray) -> dict:
    """Q13 = (13, n) flows T-60..T (row 12 = the origin row T, NaN when unknown).
    X7 = production-identical (lq, t60_30) + trends t30 t20 t15 t10 t5 from rows T-60..T-5 (D12.feats);
    F  = per-slot block log flow (nb, 13); I30 = log flow T-30..T-5 minus log flow T-60..T-35."""
    bl = O.blocks(panel)
    X7 = D12.feats(panel, Q13[:HIST])
    F = np.array([[_lf(Q13, m, slice(s, s + 1)) for s in range(HIST + 1)] for m, _ in bl])
    i30 = np.array([_lf(Q13, m, slice(6, 12)) - _lf(Q13, m, slice(0, 6)) for m, _ in bl])
    return dict(X7=X7, F=F, i30=np.nan_to_num(i30))


def canon_sets(panel: str) -> list[np.ndarray]:
    n = len(links(panel))
    out = []
    for _, c in O.blocks(panel):
        s = np.zeros(n, bool)
        s[c] = True
        out.append(s)
    return out


def iou_set(pred: np.ndarray, y6: np.ndarray, m6: np.ndarray) -> float:
    """IoU over the 6 x n horizon of a prediction that is `pred` at T+30 and empty before."""
    p = np.zeros_like(y6)
    p[HOR - 1] = pred
    p &= m6
    y = y6 & m6
    u = (p | y).sum()
    return 1.0 if u == 0 else float((p & y).sum() / u)


# ---------------------------------------------------------------------------
# cases
# ---------------------------------------------------------------------------
def selector_meta(wi: pd.DataFrame) -> pd.DataFrame:
    """Per official window, from window_index origins <= T only: gap to the previous origin (either condition),
    spacing-bound flag and the onset quota position."""
    w = wi.copy()
    w["o"] = pd.to_datetime(w.forecast_origin)
    w = w.sort_values("o")
    w["gap"] = (w.o.diff().dt.total_seconds() / 60).fillna(-1).astype(int)
    w["spacing"] = w.gap == 360
    w["k"] = w.groupby("condition").cumcount() + 1
    return w.set_index("window_id")


def build_cases(panel: str) -> dict:
    path = CACHE / f"cases_{panel}.pkl"
    if path.exists():
        return pickle.load(open(path, "rb"))
    t0 = time.time()
    cs = canon_sets(panel)
    nb = len(cs)
    pairs = [(a, b) for a in range(nb) for b in range(a + 1, nb)]
    # (1) fit rows: train pseudo onset windows (production training set of t2_onset_cap)
    g = load(panel, "train")
    fit = []
    for w in windows(panel, "train"):
        if w.cond != "onset":
            continue
        t = O.target(panel, w)
        if t is None:
            continue
        f = feats13(panel, g["Q"][w.day, w.T - HIST:w.T + 1].astype(float))
        fit.append(dict(month=w.date[:7], date=w.date, day=w.day, T=w.T, t=t, **f))
    # (2) replica onset windows (train + March), non-spacing, unique origins
    rep = pd.read_csv(REPLICA)
    rep = rep[(rep.panel == panel) & (rep.cond == "onset")].sort_values(["split", "month", "t", "start"])
    rep = rep.drop_duplicates(["split", "month", "t"])           # t restarts every month
    evr = []
    for split in ("train", "validation"):
        g = load(panel, split)
        for x in rep[rep.split == split].itertuples():
            if bool(x.spacing):
                continue
            assert str(g["dates"][x.day]) == x.date, (panel, split, x.day, x.date)
            w = Win(panel, split, int(x.day), int(x.T), "onset", x.date, int(g["weekday"][x.day]))
            y6, m6 = truth(w)
            f = feats13(panel, g["Q"][x.day, x.T - HIST:x.T + 1].astype(float))
            I = np.array([iou_set(c, y6, m6) for c in cs])
            IU = np.array([iou_set(cs[a] | cs[b], y6, m6) for a, b in pairs])
            tb = np.array([bool((y6[HOR - 1] & m6[HOR - 1] & c).any()) for c in cs])
            with np.errstate(invalid="ignore"):
                hq = int((g["V"][x.day, x.T - HIST:x.T + 1] <= links(panel).v_cut_kmh.to_numpy()).sum())
            evr.append(dict(split=split, month=x.date[:7], date=x.date, day=int(x.day), T=int(x.T),
                            weekday=int(g["weekday"][x.day]), first5=bool(x.start == 0), k=int(x.k), gap=int(x.gap),
                            first_step=int(np.flatnonzero((y6 & m6).any(1))[0]) + 1 if (y6 & m6).any() else -1,
                            hist_q=hq, y6=y6, m6=m6, I=I, IU=IU, tb=tb, **f))
    # (3) official onset windows (March, April): published history + the masked-layer origin row T
    off = []
    for split in ("validation", "private"):
        wi = pd.read_csv(RELEASE / "task2" / panel / split / "window_index.csv")
        meta = selector_meta(wi)
        g = load(panel, split)
        dmap = {str(d): i for i, d in enumerate(g["dates"])}
        for wid, (Q, tod) in O.official_history_Q(panel, split).items():
            r = meta.loc[wid]
            T = int(tod)                                   # slot of day of the origin
            day = dmap[str(r.date)]
            Qg = g["Q"][day, T - HIST:T + 1].astype(float)
            both = np.isfinite(Qg[:HIST]) & np.isfinite(Q)
            assert np.allclose(Qg[:HIST][both], Q[both], rtol=1e-4, atol=0.5), wid   # history == masked layer
            Q13 = np.vstack([Q, Qg[HIST:]])
            off.append(dict(split=split, wid=wid, date=str(r.date), origin=r.forecast_origin, T=T, tod=tod,
                            gap=int(r.gap), spacing=bool(r.spacing), k=int(r.k), **feats13(panel, Q13)))
    res = dict(panel=panel, pairs=pairs, fit=fit, rows=evr, official=off)
    CACHE.mkdir(parents=True, exist_ok=True)
    pickle.dump(res, open(path, "wb"))
    print(f"  {panel}: fit {len(fit)}, replica {len(evr)} (train {sum(r['split'] == 'train' for r in evr)}), "
          f"official {len(off)}  {time.time() - t0:.0f}s", flush=True)
    return res


def cmd_cases(_args) -> None:
    for p in PANELS:
        build_cases(p)


# ---------------------------------------------------------------------------
# ml2 (D7_I210_W production chooser) refitted leave-one-train-month-out
# ---------------------------------------------------------------------------
def _ml2_rows():
    from task2_ml import ctx_of
    from task2_rules import onset_events
    parts = []
    for p in T2_PANELS:
        d = pd.read_parquet(WORK / "cache" / "t2_ml2" / f"{p}.parquet", filters=[("kind", "==", "onset")])
        d = d.drop(columns=["step"], errors="ignore")
        ctx = ctx_of(p)
        n = ctx.n
        d["pos"] = ctx.order[np.rint(d.pos_rel.to_numpy(float) * (n - 1)).astype(int)]
        d["panel"] = p
        parts.append(d)
    return parts, {p: onset_events(p) for p in T2_PANELS}


def _prior_column(d: pd.DataFrame, ev: pd.DataFrame, n: int, skip_month: str | None) -> np.ndarray:
    """Recompute the ml2 'prior' feature of training rows (leave-one-day-out as in task2_ml2.build_panel), with
    the events of skip_month also removed."""
    from task2_rules import DEFAULTS as RULES, onset_scores
    e = ev if skip_month is None else ev[ev.date.str[:7] != skip_month]
    out = np.zeros(len(d), np.float32)
    for (date, tod, wk), idx in d.groupby(["date", "tod_T", "weekend"], observed=True).indices.items():
        pr = onset_scores(e, n, int(tod) + HOR, 5 if wk >= 0.5 else 0, str(date), RULES["sigma_h"], RULES["weekend"])
        out[idx] = pr[d.pos.to_numpy()[idx]]
    return out


def _ml2_window_prob(booster, panel: str, split: str, day: int, T: int, ev: pd.DataFrame) -> np.ndarray:
    from task2_ml import ctx_of, to_frame
    from task2_ml2 import features2
    from task2_rules import DEFAULTS as RULES, onset_scores
    ctx = ctx_of(panel)
    g = load(panel, split)
    s = slice(T - HIST, T)
    V, Q, Oc = (g[k][day, s].astype(float) for k in ("V", "Q", "O"))
    wd = int(g["weekday"][day])
    prior = onset_scores(ev, ctx.n, T + HOR, wd, None, RULES["sigma_h"], RULES["weekend"])
    f = features2(ctx, V, Q, Oc, T, int(wd >= 5), prior, T2_PANELS.index(panel))
    return booster.predict(to_frame(f, np.arange(ctx.n), None)[booster.feature_name()])


def cmd_ml2(_args) -> None:
    import lightgbm as lgb
    from task2_ml import ctx_of
    from task2_ml2 import PARAMS
    path = CACHE / "ml2_probs.pkl"
    if path.exists():
        print("exists:", path)
        return
    prod = lgb.Booster(model_file=str(ML2_PROD))
    feats = prod.feature_name()
    parts, evs = _ml2_rows()
    cases = {p: build_cases(p) for p in PANELS}
    # sanity: recomputed prior (no month skipped) == stored prior
    for d, p in zip(parts, T2_PANELS):
        pr = _prior_column(d, evs[p], ctx_of(p).n, None)
        assert np.allclose(pr, d.prior.to_numpy(np.float32), atol=1e-5), p
    months = sorted({r["month"] for r in cases[PANELS[0]]["rows"] if r["split"] == "train"})
    probs = {"lomo": {}, "prod": {}}
    params = {**PARAMS["onset"], "num_threads": THREADS}
    for m in months:
        t0 = time.time()
        tr = []
        for d, p in zip(parts, T2_PANELS):
            x = d[d.date.astype(str).str[:7] != m].copy()
            x["prior"] = _prior_column(x, evs[p], ctx_of(p).n, m)
            tr.append(x)
        tr = pd.concat(tr, ignore_index=True)
        b = lgb.train(params, lgb.Dataset(tr[feats], tr.y, weight=tr.w), ML2_ROUNDS)
        for p in PANELS:
            ev = evs[p][evs[p].date.str[:7] != m]
            for r in cases[p]["rows"]:
                if r["split"] == "train" and r["month"] == m:
                    probs["lomo"][(p, "train", r["day"], r["T"])] = _ml2_window_prob(b, p, "train", r["day"], r["T"], ev)
        print(f"  ml2 fold {m}: {len(tr):,} rows, {time.time() - t0:.0f}s", flush=True)
    for p in PANELS:
        for r in cases[p]["rows"]:
            key = (p, r["split"], r["day"], r["T"])
            probs["prod"][key] = _ml2_window_prob(prod, p, r["split"], r["day"], r["T"], evs[p])
            if r["split"] == "validation":
                probs["lomo"][key] = probs["prod"][key]          # March: the production (all-train) model
    pickle.dump(probs, open(path, "wb"))
    print("wrote", path)


def official_ml2(panel: str, split: str) -> dict[str, np.ndarray]:
    """Production ml2 onset probabilities at T+30 for the official windows (the V48 cache)."""
    P = pickle.load(open(WORK / "sub" / "probs" / f"t2ml2_probs_{split}_{panel}.pkl", "rb"))
    return {w["window_id"]: w["prob"][HOR - 1] for w in P if w["condition"] == "queue_onset"}


# ---------------------------------------------------------------------------
# block-choice models
# ---------------------------------------------------------------------------
X7COL = {k: i for i, k in enumerate(["lq", "t60_30", "t30", "t20", "t15", "t10", "t5"])}


def stack(rows: list[dict]) -> dict:
    A = {k: np.stack([r[k] for r in rows]) for k in ("X7", "F", "i30")}
    A["phase"] = np.array([r["T"] % 6 for r in rows])
    return A


def zfeat(A: dict, name: str, LP: np.ndarray | None = None) -> np.ndarray:
    """(N, nb) per-block feature."""
    F = A["F"]
    d = lambda a, b: np.nan_to_num(F[..., a] - F[..., b])
    if name in X7COL:
        return A["X7"][..., X7COL[name]]
    if name == "i30":
        return A["i30"]
    if name == "J5":
        return d(5, 4)
    if name == "J6":
        return d(6, 5)
    if name == "J11":
        return d(11, 10)
    if name == "JT":                                   # jump into the origin row T (allowed, discussion 745700)
        return d(12, 11)
    if name == "ph":                                   # log P_train(T mod 6 | block), from the fit rows
        return LP[:, A["phase"]].T
    raise KeyError(name)


def phase_prior(t: np.ndarray, phase: np.ndarray) -> np.ndarray:
    nb = t.shape[1]
    n = np.zeros((nb, 6))
    for f in range(6):
        n[:, f] = t[phase == f].sum(0)
    return np.log((n + 1.0) / (n.sum(1, keepdims=True) + 6.0))


def fit_cap(A: dict, t: np.ndarray, cols: list[str], l2: float = 0.1) -> dict:
    """t2_onset_cap form: s_b = th0 * (lq_b - c_b) + sum_k th_k * z_kb + a_b (a_0 = 0); cols=['t60_30'] on the
    production fit rows reproduces t2_onset_cap.fit exactly."""
    LP = phase_prior(t, A["phase"]) if "ph" in cols else None
    lq = A["X7"][..., 0]
    nb = lq.shape[1]
    c = np.array([np.median(lq[t[:, b] > 0, b]) if (t[:, b] > 0).any() else np.median(lq[:, b]) for b in range(nb)])
    D = lq - c[None]
    G = np.stack([zfeat(A, k, LP) for k in cols], -1) if cols else np.zeros(lq.shape + (0,))
    k = len(cols)

    def nll(th):
        s = th[0] * D + G @ th[1:1 + k] + np.concatenate([[0.0], th[1 + k:]])[None]
        s = s - s.max(1, keepdims=True)
        lse = np.log(np.exp(s).sum(1))
        return -(t * (s - lse[:, None])).sum() + l2 * (th[1 + k:] ** 2).sum() * len(t) / 100
    th = minimize(nll, np.r_[10.0, np.zeros(k), np.zeros(nb - 1)], method="L-BFGS-B").x
    return dict(kind="cap", c=c, th=th, cols=list(cols), LP=LP)


def prob_cap(m: dict, A: dict) -> np.ndarray:
    lq = A["X7"][..., 0]
    k = len(m["cols"])
    G = np.stack([zfeat(A, c, m["LP"]) for c in m["cols"]], -1) if k else np.zeros(lq.shape + (0,))
    s = m["th"][0] * (lq - m["c"]) + G @ m["th"][1:1 + k] + np.r_[0.0, m["th"][1 + k:]]
    s = s - s.max(-1, keepdims=True)
    e = np.exp(s)
    return e / e.sum(-1, keepdims=True)


def _x3(A: dict) -> np.ndarray:
    return np.concatenate([A["X7"][..., :2], np.zeros(A["X7"].shape[:2] + (1,))], -1)


def fit_flow(panel: str, A: dict, t: np.ndarray):
    return O.Model(panel, O.FLOW_USE).fit(list(zip(_x3(A), t)))


def prob_flow(m, A: dict) -> np.ndarray:
    return np.stack([m.predict_block(x) for x in _x3(A)])


CAP_CANDS = {"cap:t60_30": ["t60_30"], "cap:t15": ["t15"], "cap:t20": ["t20"], "cap:t15+t60_30": ["t15", "t60_30"],
             "cap:i30": ["i30"], "cap:lq": [], "cap:t60_30+J": ["t60_30", "J5", "J6", "J11"],
             "cap:t15+J": ["t15", "J5", "J6", "J11"], "cap:t60_30+ph": ["t60_30", "ph"], "cap:t15+ph": ["t15", "ph"],
             "cap:t60_30+J+ph": ["t60_30", "J5", "J6", "J11", "ph"], "cap:t60_30+JT": ["t60_30", "J6", "JT"],
             "cap:t15+JT": ["t15", "J6", "JT"]}
EIOU_OF = ("cap:t60_30", "cap:t15", "cap:t60_30+ph", "cap:t60_30+J", "flow")
ML2_TH = (0.3, 0.4, 0.5, 0.6)
PROD = {"D7_I210_W": "ml2:0.4", "D7_I405_S": "cap:t60_30", "D12_I5_N": "cap:t60_30", "D12_I5_S": "cap:t60_30"}


def is_habit(cand: str) -> bool:
    """Families whose block choice rests on the train time-of-day / phase habit: ml2 (tod_T + train onset prior
    features) and the train phase prior 'ph'."""
    return cand.startswith("ml2") or "ph" in cand.split(":")[-1].split("+")


def eligible(panel: str, cand: str) -> bool:
    """DEVIATION from the pre-registration (stated in work/V54_ONSET_README.md): on the corridors where
    production chooses by flow/capacity (t2_onset_cap.CAP_PANELS) the habit families are not eligible.  Reason
    documented long before this run (t2_onset.py docstring, V18/V24): March/April were generated with a new seed and
    other blocks break first there, so a train habit does not transfer, and leave-one-TRAIN-month-out cannot see that
    shift (all train months share one scenario).  The literal rule picks ml2 on all three of these corridors; their
    March replica gains are -0.13 / -0.19 / -0.38 IoU per window (decisions_literal.json).  On D7_I210_W ml2 is the
    production family and stays eligible."""
    return panel not in CAP.CAP_PANELS or not is_habit(cand)


def eiou_choice(P: np.ndarray, cs: list[np.ndarray], pairs) -> tuple[np.ndarray, list]:
    """Expected-IoU decision over single blocks and pairwise unions, truth approximated by the canonical set of
    the breaking block: E[IoU(S)] = sum_b P_b |S & C_b| / |S | C_b|.  Options 0..nb-1 = single blocks, then the
    pairs."""
    opts = [cs[b] for b in range(len(cs))] + [cs[a] | cs[b] for a, b in pairs]
    M = np.array([[(o & c).sum() / (o | c).sum() for c in cs] for o in opts])      # options x blocks
    return (P @ M.T).argmax(1), opts


def ml2_set(p: np.ndarray, th: float) -> np.ndarray:
    """task2_ml.threshold for onset windows: links with p >= th, else the single most likely link."""
    s = p >= th
    if not s.any():
        s[int(np.argmax(p))] = True
    return s


def ml2_block(p: np.ndarray, cs: list[np.ndarray]) -> int:
    return int(np.argmax([p[c].mean() for c in cs]))


def evaluate_panel(panel: str, ml2: dict) -> dict:
    """Per candidate: IoU of every replica row (train rows LOMO, March rows all-train fit) + official choices."""
    res = build_cases(panel)
    cs = canon_sets(panel)
    pairs = res["pairs"]
    nb = len(cs)
    fit, rows, off = res["fit"], res["rows"], res["official"]
    Af, Ar, Ao = stack(fit), stack(rows), stack(off)
    tf = np.stack([r["t"] for r in fit])
    fm = np.array([r["month"] for r in fit])
    rm = np.array([r["month"] for r in rows])
    rs = np.array([r["split"] for r in rows])
    I = np.stack([r["I"] for r in rows])
    IU = np.stack([r["IU"] for r in rows])
    months = sorted(set(rm[rs == "train"]))
    sub = lambda A, m: {k: v[m] for k, v in A.items()}
    out = {}
    if panel in CAP.CAP_PANELS:                        # production equivalence of the all-train cap fit
        c0, th0 = CAP.fit(panel)
        mf = fit_cap(Af, tf, ["t60_30"])
        assert np.allclose(mf["c"], c0) and np.allclose(mf["th"], th0, atol=1e-4), (panel, mf["th"], th0)

    def run_prob(fitter, prober):
        P = np.zeros((len(rows), nb))
        for m in months:
            mod = fitter(sub(Af, fm != m), tf[fm != m])
            sel = (rs == "train") & (rm == m)
            P[sel] = prober(mod, sub(Ar, sel))
        full = fitter(Af, tf)
        sel = rs == "validation"
        P[sel] = prober(full, sub(Ar, sel))
        return P, prober(full, Ao)

    probs = {name: run_prob(lambda A, t, c=cols: fit_cap(A, t, c), prob_cap) for name, cols in CAP_CANDS.items()}
    probs["flow"] = run_prob(lambda A, t: fit_flow(panel, A, t), prob_flow)
    ar = np.arange(len(rows))
    for name, (P, Po) in probs.items():
        out[name] = dict(iou=I[ar, P.argmax(1)], P=P, Po=Po, off=[cs[b] for b in Po.argmax(1)])
    IIa = np.concatenate([I, IU], 1)
    for name in EIOU_OF:
        P, Po = probs[name]
        ch, opts = eiou_choice(P, cs, pairs)
        cho, _ = eiou_choice(Po, cs, pairs)
        out["eiou:" + name] = dict(iou=IIa[ar, ch], off=[opts[c] for c in cho])
    # ml2: train rows = LOMO refits, March rows = production model, official = production cache (V48)
    pm = np.stack([ml2["lomo"][(panel, r["split"], r["day"], r["T"])] for r in rows])
    po = {}
    for split in ("validation", "private"):
        po.update(official_ml2(panel, split))
    pmo = [po[o["wid"]] for o in off]
    for th in ML2_TH:
        out[f"ml2:{th}"] = dict(iou=np.array([iou_set(ml2_set(p, th), r["y6"], r["m6"]) for p, r in zip(pm, rows)]),
                                off=[ml2_set(p, th) for p in pmo])
    chb = np.array([ml2_block(p, cs) for p in pm])
    out["ml2:block"] = dict(iou=I[ar, chb], off=[cs[ml2_block(p, cs)] for p in pmo])
    pin = np.stack([ml2["prod"][(panel, r["split"], r["day"], r["T"])] for r in rows])
    out["ml2:0.4(in-sample)"] = dict(iou=np.array([iou_set(ml2_set(p, 0.4), r["y6"], r["m6"]) for p, r in zip(pin, rows)]),
                                     off=None, record_only=True)
    out["oracle:block"] = dict(iou=I.max(1), off=None, record_only=True)
    out["oracle:block+union"] = dict(iou=IIa.max(1), off=None, record_only=True)
    meta = pd.DataFrame({k: [r[k] for r in rows] for k in ("split", "month", "date", "day", "T", "first5", "k", "gap",
                                                           "first_step", "hist_q", "weekday")})
    meta["phase"] = meta["T"] % 6
    meta["true_blocks"] = ["".join(str(b) for b in np.flatnonzero(r["tb"])) or "-" for r in rows]
    return dict(panel=panel, out=out, meta=meta, official=off, cs=cs, pairs=pairs, probs=probs, ml2=pm)


def boot_ci(d: np.ndarray, keys: np.ndarray, rng) -> tuple[float, float]:
    uk, inv = np.unique(keys, return_inverse=True)
    s = np.bincount(inv, d, len(uk))
    c = np.bincount(inv, None, len(uk))
    pick = rng.integers(0, len(uk), (N_BOOT, len(uk)))
    bs = s[pick].sum(1) / c[pick].sum(1)
    return float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))


def passes(r: dict) -> bool:
    """Pre-registered train criterion."""
    return bool(r["train_gain"] > 0 and r["train_lo"] > 0 and r["months_better"] >= 5
                and r["months_better"] > r["months_worse"])


def cmd_cv(_args) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ml2 = pickle.load(open(CACHE / "ml2_probs.pkl", "rb"))
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    pd.set_option("display.max_rows", 400)
    summ, monthly, wrows, offrows, decisions, literal = [], [], [], [], {}, {}
    rng = np.random.default_rng(54)
    for panel in PANELS:
        E = evaluate_panel(panel, ml2)
        out, meta = E["out"], E["meta"]
        tr = (meta.split == "train").to_numpy()
        va = ~tr
        f5 = meta.first5.to_numpy()
        base = out[PROD[panel]]["iou"]
        prod_off = out[PROD[panel]]["off"]
        for name, o in out.items():
            d = o["iou"] - base
            rec = dict(panel=panel, cand=name, prod=PROD[panel], record_only=bool(o.get("record_only", False)))
            for s, m in (("train", tr), ("train_first5", tr & f5), ("march", va), ("march_first5", va & f5)):
                lo, hi = boot_ci(d[m], meta.date[m].to_numpy(), rng)
                rec.update({f"{s}_n": int(m.sum()), f"{s}_iou": float(o["iou"][m].mean()),
                            f"{s}_gain": float(d[m].mean()), f"{s}_lo": lo, f"{s}_hi": hi})
            pmn = meta[tr].assign(d=d[tr]).groupby("month").d.mean()
            rec.update(months_better=int((pmn > 1e-12).sum()), months_worse=int((pmn < -1e-12).sum()),
                       win_better=int((d[tr] > 1e-12).sum()), win_worse=int((d[tr] < -1e-12).sum()),
                       march_win_better=int((d[va] > 1e-12).sum()), march_win_worse=int((d[va] < -1e-12).sum()))
            if o.get("off") is not None:
                ch = [E["official"][i]["split"] for i in range(len(prod_off)) if not np.array_equal(o["off"][i], prod_off[i])]
                rec["official_changed_val"] = ch.count("validation")
                rec["official_changed_priv"] = ch.count("private")
            rec["passes"] = passes(rec) and not rec["record_only"] and name != PROD[panel]
            summ.append(rec)
            for mo in sorted(meta.month.unique()):
                m = (meta.month == mo).to_numpy()
                monthly.append(dict(panel=panel, cand=name, month=mo, n=int(m.sum()), iou=float(o["iou"][m].mean()),
                                    gain=float(d[m].mean())))
            wrows.append(pd.DataFrame(dict(panel=panel, cand=name, idx=np.arange(len(meta)), iou=o["iou"])))
        S = pd.DataFrame([r for r in summ if r["panel"] == panel])
        for store, filt in ((literal, lambda c: True), (decisions, lambda c: eligible(panel, c))):
            ok = S[S.passes & S.cand.map(filt)]
            if len(ok):
                best = ok.sort_values("train_gain", ascending=False).iloc[0]
                store[panel] = dict(rule=best.cand, train_gain=best.train_gain, train_lo=best.train_lo,
                                    months_better=int(best.months_better), months_worse=int(best.months_worse),
                                    march_gain=best.march_gain, apply_april=bool(best.march_gain >= 0),
                                    passing=list(ok.cand))
            else:
                store[panel] = dict(rule=PROD[panel], passing=[], apply_april=True)
        for i, o in enumerate(E["official"]):
            rec = dict(panel=panel, split=o["split"], wid=o["wid"], origin=o["origin"][5:16], phase=o["T"] % 6,
                       gap=o["gap"], k=o["k"])
            for name, oo in out.items():
                if oo.get("off") is None:
                    continue
                rec[name] = " ".join(map(str, np.flatnonzero(oo["off"][i])))
                if "Po" in oo:
                    rec[name + "_p"] = " ".join(f"{v:.3f}" for v in oo["Po"][i])
            offrows.append(rec)
        meta.assign(panel=panel).to_csv(OUT / f"replica_meta_{panel}.csv", index=False)
        pickle.dump(E, open(CACHE / f"eval_{panel}.pkl", "wb"))
    S = pd.DataFrame(summ)
    S.to_csv(OUT / "cv_summary.csv", index=False)
    pd.DataFrame(monthly).to_csv(OUT / "cv_months.csv", index=False)
    pd.concat(wrows).to_csv(OUT / "cv_windows.csv", index=False)
    pd.DataFrame(offrows).to_csv(OUT / "official_windows.csv", index=False)
    (OUT / "decisions.json").write_text(json.dumps(decisions, indent=2, default=float), encoding="utf-8")
    (OUT / "decisions_literal.json").write_text(json.dumps(literal, indent=2, default=float), encoding="utf-8")
    show = ["panel", "cand", "train_n", "train_iou", "train_gain", "train_lo", "train_hi", "months_better",
            "months_worse", "win_better", "win_worse", "train_first5_gain", "march_n", "march_iou", "march_gain",
            "march_lo", "march_hi", "official_changed_val", "official_changed_priv", "passes"]
    print(S[show].round(3).to_string(index=False))
    print("\nliteral pre-registered decisions (NOT used):", json.dumps(literal, indent=1, default=float))
    print("\ndecisions (habit families not eligible on the flow-chooser corridors):",
          json.dumps(decisions, indent=1, default=float))


# ---------------------------------------------------------------------------
# official windows: build and per-window truncation check
# ---------------------------------------------------------------------------
def choose(panel: str, rule: str, off: list[dict], ml2_off: dict[str, np.ndarray]) -> list[tuple]:
    """(T+30 link set, block probabilities or None) per official row under `rule`, every model fitted on the
    train pseudo onset windows only (ml2: the production all-train booster)."""
    res = build_cases(panel)
    cs = canon_sets(panel)
    Af, tf = stack(res["fit"]), np.stack([r["t"] for r in res["fit"]])
    Ao = stack(off)
    if rule.startswith("ml2:"):
        pm = [ml2_off[o["wid"]] for o in off]
        if rule == "ml2:block":
            return [(cs[ml2_block(p, cs)], None) for p in pm]
        return [(ml2_set(p, float(rule[4:])), None) for p in pm]
    base = rule[5:] if rule.startswith("eiou:") else rule
    if base == "flow":
        P = prob_flow(fit_flow(panel, Af, tf), Ao)
    else:
        P = prob_cap(fit_cap(Af, tf, CAP_CANDS[base]), Ao)
    if rule.startswith("eiou:"):
        ch, opts = eiou_choice(P, cs, res["pairs"])
        return [(opts[c], p) for c, p in zip(ch, P)]
    return [(cs[int(p.argmax())], p) for p in P]


def all_official_ml2(panel: str) -> dict[str, np.ndarray]:
    out = {}
    for split in ("validation", "private"):
        out.update(official_ml2(panel, split))
    return out


def cmd_build(args) -> None:
    dec = json.loads((OUT / "decisions.json").read_text(encoding="utf-8"))
    dry = "--dry-run" in args.rest
    for p in NEW_Q.values():
        if p.exists() and not dry:
            raise FileExistsError(f"refusing to overwrite {p}")
    log = []
    for split in ("validation", "private"):
        q = pd.read_csv(BASE_Q[split], dtype={"link_id": str})
        changed = 0
        for panel in PANELS:
            rule = dec[panel]["rule"]
            if split == "private" and not dec[panel]["apply_april"]:
                rule = PROD[panel]
            res = build_cases(panel)
            ml2o = all_official_ml2(panel)
            prod = choose(panel, PROD[panel], res["official"], ml2o)
            new = choose(panel, rule, res["official"], ml2o)
            L = links(panel)
            pos = dict(zip(L.link_id.astype(str), range(len(L))))
            for i, o in enumerate(res["official"]):
                if o["split"] != split:
                    continue
                m = (q.window_id == o["wid"]).to_numpy()
                rows = q[m]
                T0 = int(slot_of(pd.Series([o["origin"]]))[0])
                sl = slot_of(rows.timestamp)
                lp = rows.link_id.map(pos).to_numpy()
                old = rows.queue_pred.to_numpy()
                expect = ((sl == T0 + HOR) & prod[i][0][lp]).astype(int)
                assert (old == expect).all(), f"V53 row of {o['wid']} differs from the production chooser"
                nv = ((sl == T0 + HOR) & new[i][0][lp]).astype(int)
                n_ch = int((old != nv).sum())
                if n_ch:
                    q.loc[m, "queue_pred"] = nv
                    changed += n_ch
                fmt = lambda p: " ".join(f"{v:.3f}" for v in p) if p is not None else ""
                log.append(dict(split=split, panel=panel, window_id=o["wid"], origin=o["origin"], phase=o["T"] % 6,
                                rule=rule, old_links=" ".join(map(str, np.flatnonzero(prod[i][0]))),
                                new_links=" ".join(map(str, np.flatnonzero(new[i][0]))), cells_changed=n_ch,
                                p_prod=fmt(prod[i][1]), p_new=fmt(new[i][1]),
                                ml2_T30=" ".join(f"{k}:{ml2o[o['wid']][k]:.3f}" for k in
                                                 np.argsort(ml2o[o["wid"]])[::-1][:6]) if panel == "D7_I210_W" else ""))
        if not dry:
            q.to_csv(NEW_Q[split], index=False)
        print(f"{'(dry run) ' if dry else ''}{NEW_Q[split].name}: {len(q):,} rows, {changed} cells changed vs "
              f"{BASE_Q[split].name}", flush=True)
    LG = pd.DataFrame(log)
    if not dry:
        LG.to_csv(OUT / "change_log_all_windows.csv", index=False)
        LG[LG.cells_changed > 0].to_csv(OUT / "change_log.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 60)
    print(LG[LG.cells_changed > 0].to_string(index=False))


def _truncated_official(panel: str, cut_slots: int, rel: Path) -> tuple[list[dict], dict[str, np.ndarray]]:
    """Official rows rebuilt from data with every row stamped > T - 5*cut_slots min deleted: window_history from the
    truncated release copy `rel`, the origin row from the clean parquet filtered by timestamp, and ml2 recomputed
    with task2_ml2.window_probs2 on the truncated history."""
    import evaluate
    import pyarrow.parquet as pq
    import task2_ml2
    import lightgbm as lgb
    from evaluate import CLEAN
    real = evaluate.RELEASE
    evaluate.RELEASE, task2_ml2.RELEASE = rel, rel
    try:
        b = lgb.Booster(model_file=str(ML2_PROD))
        fn = lambda X: b.predict(X[b.feature_name()])
        ml2o = {}
        off = []
        n = len(links(panel))
        for split in ("validation", "private"):
            for w in task2_ml2.window_probs2(panel, split, {"onset": fn, "ongoing": lambda X: np.zeros(len(X))}, False):
                if w["condition"] == "queue_onset":
                    ml2o[w["window_id"]] = w["prob"][HOR - 1]
            wi = pd.read_csv(rel / "task2" / panel / split / "window_index.csv")
            meta = selector_meta(wi)
            pf = pq.ParquetFile(CLEAN / panel / f"{split}.parquet")
            dates = [str(pf.read_row_group(i, columns=["date"]).column("date")[0].as_py()) for i in range(pf.num_row_groups)]
            for wid, (Q, tod) in O.official_history_Q(panel, split).items():
                r = meta.loc[wid]
                T = int(tod)
                t = pf.read_row_group(dates.index(str(r.date)), columns=["ts", "flow_clean"]).to_pandas()
                ts = pd.to_datetime(t.ts, utc=True)
                keep = ts <= pd.Timestamp(r.forecast_origin) - pd.Timedelta(minutes=5 * cut_slots)
                fl = np.where(keep, t.flow_clean.to_numpy(float), np.nan).reshape(-1, n)
                Q13 = np.vstack([Q, fl[T][None]])
                off.append(dict(split=split, wid=wid, date=str(r.date), origin=r.forecast_origin, T=T, tod=tod,
                                gap=int(r.gap), spacing=bool(r.spacing), k=int(r.k), **feats13(panel, Q13)))
        return off, ml2o
    finally:
        evaluate.RELEASE, task2_ml2.RELEASE = real, real


def cmd_selector(_args) -> None:
    """Record-only: does the selector add information to the onset block choice?
      (a) spacing-bound share of the replica onset windows vs the official ones (0 of 120);
      (b) second-stage LOMO: P'_b ∝ P_b exp(beta[z, b]) with z = gap bin to the previous window (either
          condition) / onset quota position k / first-window-of-month flag, beta fitted on replica train rows of the
          other months (L2 = 1), March rows by the all-train fit;
      (c) online prior from the aftermath (visible queue at T_j+95..T_j+105, i.e. right after the blackout) of the
          EARLIER onset windows j of the same selector run: P'_b ∝ P_b ((n_b + 1) / (N + nb))^lam."""
    from scipy.optimize import minimize as _min
    rep = pd.read_csv(REPLICA)
    rows = []
    on = rep[rep.cond == "onset"].sort_values("start").drop_duplicates(["panel", "split", "month", "t"])
    for (sp, p), g in on.groupby(["split", "panel"]):
        rows.append(dict(check="spacing_share", panel=p, split=sp, n=len(g), value=float(g.spacing.mean())))
    for split in ("train", "validation", "private"):
        for p in T2_PANELS:
            m = selector_meta(pd.read_csv(RELEASE / "task2" / p / split / "window_index.csv"))
            m = m[m.condition == "queue_onset"]
            rows.append(dict(check="official_spacing_share", panel=p, split=split, n=len(m), value=float(m.spacing.mean())))
    base_of = {"D7_I210_W": "ml2:block", "D7_I405_S": "cap:t60_30", "D12_I5_N": "cap:t60_30", "D12_I5_S": "cap:t60_30"}
    rng = np.random.default_rng(7)
    for p in PANELS:
        E = pickle.load(open(CACHE / f"eval_{p}.pkl", "rb"))
        meta, out, cs = E["meta"], E["out"], E["cs"]
        res = build_cases(p)
        I = np.stack([r["I"] for r in res["rows"]])
        tgt = np.stack([r["tb"] for r in res["rows"]]).astype(float)
        nb = I.shape[1]
        base = base_of[p]
        if "P" in out[base]:
            P = out[base]["P"]
        else:
            S = np.stack([[m[c].mean() for c in cs] for m in E["ml2"]]) + 1e-3
            P = S / S.sum(1, keepdims=True)
        ar = np.arange(len(I))
        b0 = I[ar, P.argmax(1)]
        tr = (meta.split == "train").to_numpy()
        gap = meta.gap.to_numpy()
        gmin = np.where(gap < 0, -1, 5 * gap)                       # replica gap is in 5-min slots
        feats = {"gap_bin": np.select([gmin < 0, gmin < 720, gmin < 1440], [0, 1, 2], 3),
                 "quota_k": meta.k.to_numpy() - 1, "first_of_run": (gap < 0).astype(int)}

        def record(check, b1):
            d = b1 - b0
            lo, hi = boot_ci(d[tr], meta.date[tr].to_numpy(), rng)
            pm = pd.Series(d[tr]).groupby(meta.month[tr].to_numpy()).mean()
            rows.append(dict(check=check, panel=p, split="train/march", n=int(tr.sum()), base=base,
                             train_gain=float(d[tr].mean()), train_lo=lo, train_hi=hi,
                             months_better=int((pm > 1e-12).sum()), months_worse=int((pm < -1e-12).sum()),
                             march_gain=float(d[~tr].mean())))
        for name, z in feats.items():
            nz = int(z.max()) + 1

            def fit(sel):
                Z, L, T = np.eye(nz)[z[sel]], np.log(P[sel] + 1e-9), tgt[sel]
                ok = T.sum(1) > 0
                Z, L, T = Z[ok], L[ok], T[ok] / T[ok].sum(1, keepdims=True)

                def nll(th):
                    s = L + Z @ th.reshape(nz, nb)
                    s = s - s.max(1, keepdims=True)
                    return -(T * (s - np.log(np.exp(s).sum(1))[:, None])).sum() + (th ** 2).sum()
                return _min(nll, np.zeros(nz * nb), method="L-BFGS-B").x.reshape(nz, nb)
            Pn = np.zeros_like(P)
            for mo in sorted(meta.month[tr].unique()):
                sel = tr & (meta.month == mo).to_numpy()
                Pn[sel] = np.log(P[sel] + 1e-9) + fit(tr & ~sel)[z[sel]]
            Pn[~tr] = np.log(P[~tr] + 1e-9) + fit(tr)[z[~tr]]
            record(f"stage2_{name}", I[ar, Pn.argmax(1)])
        # (c) aftermath of earlier onset windows of the same run
        vc = links(p).v_cut_kmh.to_numpy()
        R = rep[(rep.panel == p) & (rep.cond == "onset")]
        first = R.sort_values("start").drop_duplicates(["split", "month", "t"])
        lay = {s: load(p, s) for s in ("train", "validation")}
        cnt = np.zeros((len(meta), nb))
        for i, r in enumerate(meta.itertuples()):
            f = first[(first.split == r.split) & (first.date == r.date) & (first["T"] == r.T)].iloc[0]
            g = lay[r.split]
            VL = g["Vt"] if r.split == "train" else g["V"]
            for e in R[(R.split == r.split) & (R.month == f.month) & (R.start == f.start) & (R.t < f.t)].itertuples():
                if e.T + 21 < 288:
                    with np.errstate(invalid="ignore"):
                        A = (VL[e.day, e.T + 19:e.T + 22] <= vc).any(0)
                    cnt[i] += [A[m].any() for m, _ in O.blocks(p)]
        for lam in (0.5, 1.0, 2.0):
            Pn = P * ((cnt + 1) / (cnt.sum(1, keepdims=True) + nb)) ** lam
            record(f"aftermath_prior_lam{lam}", I[ar, Pn.argmax(1)])
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "selector_check.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 200)
    print(df.round(4).to_string(index=False))


def cmd_asof(args) -> None:
    """Organizer-style check: delete every row stamped after each window's T (the T row itself is allowed), rebuild
    the inputs, rerun the chooser; then a control that also deletes the T row (must move something when a rule
    uses the T row) and one that deletes T-5 as well."""
    import shutil
    if not args.rest:
        raise SystemExit("usage: asof-check <empty scratch dir>")
    sp = Path(args.rest[0])
    if sp.exists() and any(sp.iterdir()):
        raise FileExistsError(f"{sp} is not empty")
    dec = json.loads((OUT / "decisions.json").read_text(encoding="utf-8"))
    rules = {p: sorted({dec[p]["rule"], PROD[p]}) for p in PANELS}
    ref = {}
    for p in PANELS:
        res = build_cases(p)
        ml2o = all_official_ml2(p)
        for rule in rules[p]:
            for o, (s, pr) in zip(res["official"], choose(p, rule, res["official"], ml2o)):
                ref[(rule, o["wid"])] = (s, pr, ml2o[o["wid"]])
    rows = []
    for tag, cut in (("delete_rows_after_T", 0), ("control_delete_T_row", 1), ("control_delete_T-5_row", 2)):
        rel = sp / tag
        n_drop = 0
        for p in PANELS:
            for split in ("validation", "private"):
                src = RELEASE / "task2" / p / split
                dst = rel / "task2" / p / split
                dst.mkdir(parents=True, exist_ok=True)
                shutil.copy(src / "window_index.csv", dst / "window_index.csv")
                win = pd.read_csv(src / "window_index.csv").set_index("window_id")
                h = pd.read_parquet(src / "window_history.parquet")
                org = pd.to_datetime(h.window_id.map(win.forecast_origin), utc=True) - pd.Timedelta(minutes=5 * cut)
                keep = (pd.to_datetime(h.timestamp, utc=True) <= org).to_numpy()    # history itself ends at T-5
                n_drop += int((~keep).sum())
                h[keep].to_parquet(dst / "window_history.parquet", index=False)
        for p in PANELS:
            off, ml2o = _truncated_official(p, cut, rel)
            for rule in rules[p]:
                flips, dp, dml2 = 0, 0.0, 0.0
                for o, (s, pr) in zip(off, choose(p, rule, off, ml2o)):
                    s0, p0, m0 = ref[(rule, o["wid"])]
                    flips += int(not np.array_equal(s, s0))
                    if pr is not None:
                        dp = max(dp, float(np.abs(pr - p0).max()))
                    dml2 = max(dml2, float(np.abs(ml2o[o["wid"]] - m0).max()))
                rows.append(dict(test=tag, history_rows_dropped=n_drop, panel=p, rule=rule, windows=len(off),
                                 choice_flips=flips, max_abs_dp=dp, max_abs_d_ml2=dml2))
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "asof_check.csv", index=False)
    pd.set_option("display.width", 250)
    print(df.to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["cases", "ml2", "cv", "selector", "build", "asof-check"])
    ap.add_argument("rest", nargs="*")
    a, extra = ap.parse_known_args()
    a.rest = list(a.rest) + extra
    {"cases": cmd_cases, "ml2": cmd_ml2, "cv": cmd_cv, "selector": cmd_selector, "build": cmd_build,
     "asof-check": cmd_asof}[a.cmd](a)
