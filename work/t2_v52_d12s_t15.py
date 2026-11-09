#!/usr/bin/env python
"""V52 lever L2: a 15-minute demand trend in the capacity-distance onset model (t2_onset_cap).

Production (V48 = V51 on the public split) chooses the onset block of D12_I5_N / D12_I5_S / D7_I405_S with
    score_b = th0 * (lq_b - c_b) + th1 * trend60_b + a_b,
lq_b = log median block flow T-15..T-5, trend60_b = lq_b - log median block flow T-60..T-35, c_b = median train
lq at the onsets of block b (t2_onset_cap.fit).  Here trend60 is replaced (or complemented) by shorter trends,
e.g. trend15_b = log flow T-15..T-5 - log flow T-30..T-20 (same block_flows link range as production).

Selection protocol (train only; the public LB and April are never used to choose)
  * every model is fitted on train pseudo onset windows only (t2_pseudo.windows, masked-layer features);
  * train held-out = leave-one-train-month-out: each train month is scored by a model fitted on the other months;
  * per corridor, the trend-15 rule replaces production iff its pooled train-LOMO IoU gain is > 0 AND the 95 %
    day-clustered bootstrap interval of that gain lies above 0 (pre-registered from round 1; t10/t20/t60+t15 and
    the l2 grid are robustness checks, they are not candidates);
  * March pseudo onset windows (and the June official train windows) are reported as confirmation only.  A March
    window is scored by the all-train model on its own 60-minute history before T, so it is causal; March data is
    not used to fit or select anything for the March (public) windows.
  * April (private) official windows only receive the same train-fitted rule; April pseudo windows are never
    built here.

Usage (from the "IEEE Big data" folder; CPU only, ~2-4 min):
  python work/t2_v52_d12s_t15.py cv       # tables -> work/results/t2_v52_d12s/
  python work/t2_v52_d12s_t15.py build    # queue_validation_v52_d12s.csv / queue_private_v52_d12s.csv + change log
  python work/t2_v52_d12s_t15.py audit    # record-only diagnostics (June windows, calibration, April refit)
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent))
import t2_onset as O          # noqa: E402
import t2_onset_cap as CAP    # noqa: E402
from evaluate import RELEASE, ROOT, links, slot_of  # noqa: E402
from t2_grids import load     # noqa: E402
from t2_pseudo import HIST, HOR, Win, history, iou, windows  # noqa: E402

warnings.simplefilter("ignore")
WORK = ROOT / "work"
OUT = WORK / "results" / "t2_v52_d12s"
PANELS = CAP.CAP_PANELS                      # ("D12_I5_N", "D12_I5_S", "D7_I405_S")
BASE_Q = {"validation": WORK / "sub" / "queue_validation_v51_april.csv",
          "private": WORK / "sub" / "queue_private_v51_april.csv"}
NEW_Q = {"validation": WORK / "sub" / "queue_validation_v52_d12s.csv",
         "private": WORK / "sub" / "queue_private_v52_d12s.csv"}
# trends as (recent, earlier) slices of the 12-slot history (index 0 = T-60, index 11 = T-5)
TR = {"t30": (slice(9, 12), slice(3, 6)), "t20": (slice(8, 12), slice(4, 8)), "t15": (slice(9, 12), slice(6, 9)),
      "t10": (slice(10, 12), slice(8, 10)), "t5": (slice(11, 12), slice(10, 11))}
FEAT = ["lq", "t60_30", *TR]                  # X column order; lq and t60_30 are exactly O.features cols 0/1
COL = {k: i for i, k in enumerate(FEAT)}
CANDIDATE, PROD = "t15", "t60_30"
N_BOOT = 2000


# ---------------------------------------------------------------------------
# features and cases
# ---------------------------------------------------------------------------
def feats(panel: str, Q: np.ndarray) -> np.ndarray:
    """(n_blocks, len(FEAT)) from a (12, n) flow history (production block_flows link ranges)."""
    X0 = O.features(panel, Q, HOR)[:, :2]                 # production lq, trend60 (tod prior unused)
    def L(members, steps):
        v = O.block_flows(Q, members, steps)
        return np.log(max(v, 100.0)) if np.isfinite(v) else np.nan
    extra = []
    for members, _ in O.blocks(panel):
        r = []
        for a, b in TR.values():
            x, y = L(members, a), L(members, b)
            r.append(x - y if np.isfinite(x) and np.isfinite(y) else 0.0)
        extra.append(r)
    return np.concatenate([X0, np.array(extra, float)], 1)


def block_preds(panel: str) -> list[np.ndarray]:
    n = len(links(panel))
    out = []
    for _, canon in O.blocks(panel):
        p = np.zeros((HOR, n), bool)
        p[HOR - 1, canon] = True
        out.append(p)
    return out


def cases(panel: str) -> dict:
    """Train + March pseudo onset windows and official onset windows (train June / March / April)."""
    path = OUT / f"cases_{panel}.pkl"
    if path.exists():
        return pickle.load(open(path, "rb"))
    bp = block_preds(panel)
    rows = []
    for split in ("train", "validation"):                 # never the private pseudo windows
        for w in windows(panel, split):
            if w.cond != "onset":
                continue
            rows.append(dict(split=split, month=w.date[:7], date=w.date, day=w.day, T=w.T,
                             X=feats(panel, history(w)[1].astype(float)), t=O.target(panel, w),
                             I=np.array([iou(p, w) for p in bp])))
    off = []
    for split in ("train", "validation", "private"):
        win = pd.read_csv(RELEASE / "task2" / panel / split / "window_index.csv").set_index("window_id")
        hq = O.official_history_Q(panel, split)
        if split == "train":
            g = load(panel, "train")
            dmap = {str(d): i for i, d in enumerate(g["dates"])}
        for wid, (Q, tod) in hq.items():
            r = dict(split=split, wid=wid, date=win.loc[wid, "date"], origin=win.loc[wid, "forecast_origin"],
                     tod=tod, X=feats(panel, Q), I=None)
            if split == "train":                          # proxy labels: speed_true <= v_cut on eligible cells
                w = Win(panel, "train", dmap[r["date"]], tod, "onset", r["date"], 0)
                r["I"] = np.array([iou(p, w) for p in bp])
            off.append(r)
    res = dict(rows=rows, official=off)
    OUT.mkdir(parents=True, exist_ok=True)
    pickle.dump(res, open(path, "wb"))
    return res


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------
def fit_cap(X: np.ndarray, T: np.ndarray, tcols: list[int], l2: float = 0.1) -> dict:
    """t2_onset_cap.fit with the trend column(s) `tcols` (tcols=[COL['t60_30']] on all train == production)."""
    nb, k = X.shape[1], len(tcols)
    c = np.array([np.median(X[T[:, b] > 0, b, 0]) if (T[:, b] > 0).any() else np.median(X[:, b, 0])
                  for b in range(nb)])
    D = X[:, :, 0] - c[None]
    G = X[:, :, tcols]

    def nll(th):
        s = th[0] * D + G @ th[1:1 + k] + np.concatenate([[0.0], th[1 + k:]])[None]
        s = s - s.max(1, keepdims=True)
        lse = np.log(np.exp(s).sum(1))
        return -(T * (s - lse[:, None])).sum() + l2 * (th[1 + k:] ** 2).sum() * len(X) / 100
    th = minimize(nll, np.r_[10.0, np.zeros(k), np.zeros(nb - 1)], method="L-BFGS-B").x
    return dict(form="cap", c=c, th=th, tcols=list(tcols))


def fit_std(X: np.ndarray, T: np.ndarray, tcols: list[int], l2: float = 0.1) -> dict:
    """Round-1 parameterisation (standardised free slopes on lq and trends + block intercepts; no c_b)."""
    cols = [0, *tcols]
    nb = X.shape[1]
    Z = X[:, :, cols]
    sd = Z.reshape(-1, len(cols)).std(0) + 1e-6
    Z = Z / sd

    def nll(th):
        s = Z @ th[:len(cols)] + np.r_[0.0, th[len(cols):]][None]
        s = s - s.max(1, keepdims=True)
        lse = np.log(np.exp(s).sum(1))
        return -(T * (s - lse[:, None])).sum() + l2 * (th[len(cols):] ** 2).sum() * len(X) / 100
    th = minimize(nll, np.r_[np.zeros(len(cols)), np.zeros(nb - 1)], method="L-BFGS-B").x
    return dict(form="std", sd=sd, th=th, cols=cols)


def scores(m: dict, X: np.ndarray) -> np.ndarray:
    if m["form"] == "cap":
        th, k = m["th"], len(m["tcols"])
        return th[0] * (X[..., 0] - m["c"]) + X[..., m["tcols"]] @ th[1:1 + k] + np.r_[0.0, th[1 + k:]]
    Z = X[..., m["cols"]] / m["sd"]
    return Z @ m["th"][:len(m["cols"])] + np.r_[0.0, m["th"][len(m["cols"]):]]


def softmax(s: np.ndarray) -> np.ndarray:
    s = s - s.max(-1, keepdims=True)
    e = np.exp(s)
    return e / e.sum(-1, keepdims=True)


FITS = {"cap": fit_cap, "std": fit_std}


def arrays(res: dict):
    R = res["rows"]
    nb = len(R[0]["I"])
    X = np.stack([r["X"] for r in R])
    I = np.stack([r["I"] for r in R])
    ok = np.array([r["t"] is not None for r in R])
    T = np.stack([r["t"] if r["t"] is not None else np.zeros(nb) for r in R])
    meta = pd.DataFrame({k: [r[k] for r in R] for k in ("split", "month", "date", "day", "T")})
    return X, T, I, ok, meta


def run(res: dict, tcols: list[int], form: str = "cap", l2: float = 0.1):
    """Train LOMO + March (all-train fit) probabilities, the all-train model, and June-out model."""
    X, T, I, ok, meta = arrays(res)
    tr = (meta.split == "train").to_numpy()
    P = np.zeros(I.shape)
    folds = {}
    for m in sorted(meta.month[tr].unique()):
        sel = tr & (meta.month == m).to_numpy()
        f = FITS[form](X[tr & ok & ~sel], T[tr & ok & ~sel], tcols, l2)
        folds[m] = f
        P[sel] = softmax(scores(f, X[sel]))
    full = FITS[form](X[tr & ok], T[tr & ok], tcols, l2)
    va = (meta.split == "validation").to_numpy()
    P[va] = softmax(scores(full, X[va]))
    return P, full, folds


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------
def boot(d: np.ndarray, keys: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    uk, inv = np.unique(keys, return_inverse=True)
    s = np.bincount(inv, d, len(uk))
    c = np.bincount(inv, None, len(uk))
    bs = []
    for _ in range(N_BOOT):
        pick = rng.integers(0, len(uk), len(uk))
        bs.append(s[pick].sum() / c[pick].sum())
    return float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))


VARIANTS = {                       # name -> trend columns (all share lq and the c_b capacity term)
    "prod(t60_30)": ["t60_30"], "t15": ["t15"], "t10": ["t10"], "t20": ["t20"], "t30": ["t30"], "t5": ["t5"],
    "t15+t60_30": ["t15", "t60_30"], "t10+t60_30": ["t10", "t60_30"], "t20+t60_30": ["t20", "t60_30"],
    "lq only": [],
}


def cmd_cv() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    summ, months, dis, offrows, l2rows = [], [], [], [], []
    decisions = {}
    for panel in PANELS:
        res = cases(panel)
        X, T, I, ok, meta = arrays(res)
        nb = I.shape[1]
        tr = (meta.split == "train").to_numpy()
        va = ~tr
        # production equivalence: all-train prod-form fit == t2_onset_cap.fit
        c0, th0 = CAP.fit(panel)
        P_prod, full_prod, folds_prod = run(res, [COL["t60_30"]])
        assert np.allclose(full_prod["c"], c0) and np.allclose(full_prod["th"], th0, atol=1e-4), (panel, full_prod, th0)
        base = {}
        for form in ("cap", "std"):
            base[form] = I[np.arange(len(I)), run(res, [COL["t60_30"]], form)[0].argmax(1)]
        rng = np.random.default_rng(0)
        store = {}
        for form in ("cap", "std"):
            for name, tc in VARIANTS.items():
                if form == "std" and name not in ("prod(t60_30)", "t15", "t10", "t20", "t15+t60_30"):
                    continue
                P, full, folds = run(res, [COL[c] for c in tc], form)
                iu = I[np.arange(len(I)), P.argmax(1)]
                d = iu - base[form]
                store[(form, name)] = (P, full, folds, iu)
                rec = dict(panel=panel, form=form, variant=name)
                for s, m in (("train", tr), ("march", va)):
                    lo, hi = boot(d[m], meta.date[m].to_numpy(), rng)
                    rec.update({f"{s}_n": int(m.sum()), f"{s}_days": int(meta.date[m].nunique()),
                                f"{s}_iou": iu[m].mean(), f"{s}_gain": d[m].mean(), f"{s}_lo": lo, f"{s}_hi": hi,
                                f"{s}_early_gain": d[m & (meta.day < 10).to_numpy()].mean()})
                pm = meta[tr].assign(iou=iu[tr], d=d[tr]).groupby("month").agg(iou=("iou", "mean"), d=("d", "mean"))
                rec.update(months_better=int((pm.d > 1e-12).sum()), months_worse=int((pm.d < -1e-12).sum()),
                           month_mean_gain=float(pm.d.mean()))
                if form == "cap":
                    rec["coef"] = " ".join(f"{v:.2f}" for v in full["th"][:1 + len(tc)])
                summ.append(rec)
                if form == "cap" and name in ("prod(t60_30)", "t15", "t10", "t20", "t15+t60_30"):
                    for mo in sorted(meta.month.unique()):
                        m = (meta.month == mo).to_numpy()
                        months.append(dict(panel=panel, variant=name, month=mo, split=meta.split[m].iloc[0],
                                           n=int(m.sum()), iou=iu[m].mean(), gain=d[m].mean()))
        # l2 robustness of the candidate (cap form)
        for l2 in (0.01, 0.1, 1.0):
            for name in ("prod(t60_30)", "t15"):
                P = run(res, [COL[c] for c in VARIANTS[name]], "cap", l2)[0]
                iu = I[np.arange(len(I)), P.argmax(1)]
                l2rows.append(dict(panel=panel, l2=l2, variant=name, train=iu[tr].mean(), march=iu[va].mean()))
        # disagreement statistics prod vs candidate (cap form)
        Pp, Pc = store[("cap", "prod(t60_30)")][0], store[("cap", CANDIDATE)][0]
        a, b = Pp.argmax(1), Pc.argmax(1)
        ia, ib = I[np.arange(len(I)), a], I[np.arange(len(I)), b]
        for s, m in (("train", tr), ("march", va)):
            k = m & (a != b)
            dis.append(dict(panel=panel, split=s, n=int(m.sum()), disagree=int(k.sum()),
                            cand_better=int((ib > ia)[k].sum()), prod_better=int((ia > ib)[k].sum()),
                            mean_dIoU_on_disagree=float((ib - ia)[k].mean()) if k.any() else 0.0))
        # pre-registered train criterion
        r = next(x for x in summ if x["panel"] == panel and x["form"] == "cap" and x["variant"] == CANDIDATE)
        adopt = bool(r["train_gain"] > 0 and r["train_lo"] > 0)
        decisions[panel] = dict(rule=CANDIDATE if adopt else "prod", train_gain=r["train_gain"], train_lo=r["train_lo"],
                                train_hi=r["train_hi"], months_better=r["months_better"], months_worse=r["months_worse"],
                                march_gain=r["march_gain"], march_lo=r["march_lo"], march_hi=r["march_hi"])
        # official windows: June (fold model leaving June out), March / April (all-train model)
        for o in res["official"]:
            rec = dict(panel=panel, split=o["split"], wid=o["wid"], date=o["date"], origin=o["origin"][11:16])
            for name in ("prod(t60_30)", CANDIDATE, "t10", "t20", "t15+t60_30"):
                _, full, folds, _ = store[("cap", name)]
                f = folds["2030-06"] if o["split"] == "train" else full
                p = softmax(scores(f, o["X"][None]))[0]
                rec[f"{name}_arg"] = int(p.argmax())
                rec[f"{name}_p"] = " ".join(f"{v:.3f}" for v in p)
                if o["I"] is not None:
                    rec[f"{name}_iou"] = float(o["I"][p.argmax()])
            if o["I"] is not None:
                rec["oracle_iou_per_block"] = " ".join(f"{v:.2f}" for v in o["I"])
            rec["lq_flow"] = " ".join(f"{np.exp(v):.0f}" for v in o["X"][:, 0])
            rec["t60_30"] = " ".join(f"{v:+.3f}" for v in o["X"][:, COL["t60_30"]])
            rec["t15"] = " ".join(f"{v:+.3f}" for v in o["X"][:, COL["t15"]])
            offrows.append(rec)
    S = pd.DataFrame(summ)
    S.to_csv(OUT / "cv_summary.csv", index=False)
    M = pd.DataFrame(months)
    M.to_csv(OUT / "cv_months.csv", index=False)
    pd.DataFrame(dis).to_csv(OUT / "disagreements.csv", index=False)
    OFF = pd.DataFrame(offrows)
    OFF.to_csv(OUT / "official_windows.csv", index=False)
    pd.DataFrame(l2rows).to_csv(OUT / "l2_robustness.csv", index=False)
    (OUT / "decisions.json").write_text(json.dumps(decisions, indent=2), encoding="utf-8")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    show = ["panel", "form", "variant", "train_iou", "train_gain", "train_lo", "train_hi", "months_better",
            "months_worse", "march_iou", "march_gain", "march_lo", "march_hi", "train_early_gain", "march_early_gain"]
    print(S[show].round(3).to_string(index=False))
    print("\nmonth-by-month (cap form):")
    print(M.pivot_table(index=["panel", "month"], columns="variant", values="iou").round(3).to_string())
    print("\n", pd.DataFrame(dis).round(3).to_string(index=False))
    print("\nl2:", pd.DataFrame(l2rows).pivot_table(index=["panel", "l2"], columns="variant").round(3).to_string())
    print("\ndecisions:", json.dumps(decisions, indent=1))
    cols = ["panel", "split", "wid", "origin", "prod(t60_30)_arg", "prod(t60_30)_p", "t15_arg", "t15_p", "t10_arg",
            "t20_arg", "t15+t60_30_arg"]
    print(OFF[cols].to_string(index=False))
    J = OFF[OFF.split == "train"]
    print("\nJune official train windows (fold model leaving June out; proxy labels speed_true <= v_cut):")
    print(J.groupby("panel")[["prod(t60_30)_iou", "t15_iou", "t10_iou", "t20_iou", "t15+t60_30_iou"]].mean().round(3))


# ---------------------------------------------------------------------------
# queue files
# ---------------------------------------------------------------------------
def cmd_build() -> None:
    dec = json.loads((OUT / "decisions.json").read_text(encoding="utf-8"))
    for p in NEW_Q.values():
        if p.exists():
            raise FileExistsError(f"refusing to overwrite {p}")
    log = []
    for split in ("validation", "private"):
        q = pd.read_csv(BASE_Q[split], dtype={"link_id": str})
        changed = 0
        for panel in PANELS:
            res = cases(panel)
            X, T, I, ok, meta = arrays(res)
            tr = (meta.split == "train").to_numpy() & ok
            prod = fit_cap(X[tr], T[tr], [COL["t60_30"]])
            cand = fit_cap(X[tr], T[tr], [COL[CANDIDATE]])
            use = cand if dec[panel]["rule"] == CANDIDATE else prod
            bl = O.blocks(panel)
            L = links(panel)
            pos = dict(zip(L.link_id.astype(str), range(len(L))))
            for o in res["official"]:
                if o["split"] != split:
                    continue
                pp, pc, pu = (softmax(scores(m, o["X"][None]))[0] for m in (prod, cand, use))
                m = (q.window_id == o["wid"]).to_numpy()
                rows = q[m]
                T0 = int(slot_of(pd.Series([o["origin"]]))[0])
                sl = slot_of(rows.timestamp)
                lp = rows.link_id.map(pos).to_numpy()
                old_on = set(lp[rows.queue_pred.to_numpy() == 1].tolist())
                assert (sl[rows.queue_pred.to_numpy() == 1] == T0 + HOR).all(), o["wid"]
                old_block = [k for k, (_, cn) in enumerate(bl) if set(cn.tolist()) == old_on]
                # V51 must equal the production argmax on these corridors (else we would not patch blindly)
                assert old_block == [int(pp.argmax())], (o["wid"], old_block, pp)
                new_b = int(pu.argmax())
                new = ((sl == T0 + HOR) & np.isin(lp, bl[new_b][1])).astype(int)
                n_ch = int((rows.queue_pred.to_numpy() != new).sum())
                if n_ch:
                    q.loc[m, "queue_pred"] = new
                    changed += n_ch
                log.append(dict(split=split, panel=panel, window_id=o["wid"], date=o["date"], origin=o["origin"],
                                rule=dec[panel]["rule"], old_block=old_block[0],
                                old_links=" ".join(map(str, bl[old_block[0]][1])), new_block=new_b,
                                new_links=" ".join(map(str, bl[new_b][1])), cells_changed=n_ch,
                                p_prod=" ".join(f"{v:.3f}" for v in pp), p_t15=" ".join(f"{v:.3f}" for v in pc)))
        q.to_csv(NEW_Q[split], index=False)
        print(f"wrote {NEW_Q[split]}: {len(q):,} rows, {changed} cells changed vs {BASE_Q[split].name}", flush=True)
    LG = pd.DataFrame(log)
    LG.to_csv(OUT / "change_log_all_windows.csv", index=False)
    LG[LG.cells_changed > 0].to_csv(OUT / "change_log.csv", index=False)
    pd.set_option("display.width", 250)
    print(LG.to_string(index=False))


def cmd_audit() -> None:
    """Record-only diagnostics (nothing here feeds a decision): June official per-window IoU, coefficients,
    calibration of the t15 choice on train-LOMO disagreements, an April-only train+March refit (allowed for
    April by the rules), and which block breaks first in train vs March pseudo windows."""
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    OFF = pd.read_csv(OUT / "official_windows.csv")
    print(OFF[OFF.split == "train"][["panel", "wid", "origin", "prod(t60_30)_arg", "prod(t60_30)_iou", "t15_arg",
                                     "t15_iou", "oracle_iou_per_block"]].to_string(index=False))
    for panel in PANELS:
        res = cases(panel)
        X, T, I, ok, meta = arrays(res)
        tr = (meta.split == "train").to_numpy()
        va = ~tr
        Pp, fp, _ = run(res, [COL["t60_30"]])
        Pc, fc, _ = run(res, [COL["t15"]])
        print(f"\n== {panel}  c_b {np.round(np.exp(fp['c']))}  theta prod {np.round(fp['th'], 2)}  theta t15 {np.round(fc['th'], 2)}")
        a, b = Pp.argmax(1), Pc.argmax(1)
        ar = np.arange(len(I))
        ia, ib = I[ar, a], I[ar, b]
        k = tr & (a != b)
        margin = Pc[ar, b] - Pc[ar, a]
        for lo, hi in [(0, .25), (.25, .5), (.5, 1.01)]:
            s = k & (margin >= lo) & (margin < hi)
            if s.any():
                print(f"   train-LOMO disagreements, t15 margin [{lo},{hi}): n={s.sum()} t15 better {(ib > ia)[s].sum()} "
                      f"prod better {(ia > ib)[s].sum()} mean dIoU {(ib - ia)[s].mean():+.3f}")
        pmax = Pc.max(1)
        for lo, hi in [(0, .5), (.5, .7), (.7, .9), (.9, 1.01)]:
            s = tr & (pmax >= lo) & (pmax < hi)
            if s.any():
                print(f"   t15 calibration (train LOMO) p in [{lo},{hi}): n={s.sum()} mean p {pmax[s].mean():.2f} IoU {ib[s].mean():.2f}")
        trv = ok & (tr | va)
        for name, tc in (("prod", [COL["t60_30"]]), ("t15", [COL["t15"]])):
            f = fit_cap(X[trv], T[trv], tc)
            f0 = fc if name == "t15" else fp
            pr = [o for o in res["official"] if o["split"] == "private"]
            print(f"   April official argmax {name}: train+March fit {[int(np.argmax(scores(f, o['X']))) for o in pr]}"
                  f"  train-only fit {[int(np.argmax(scores(f0, o['X']))) for o in pr]}")
        best = I.argmax(1)
        for s, m in (("train", tr), ("march", va)):
            print(f"   {s}: first-breaking block counts {np.bincount(best[m & (I.max(1) > 0)], minlength=I.shape[1])}"
                  f" (none {int((m & (I.max(1) == 0)).sum())}); prod picks {np.bincount(a[m], minlength=I.shape[1])}"
                  f" t15 picks {np.bincount(b[m], minlength=I.shape[1])}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["cv", "build", "audit"])
    a = ap.parse_args()
    {"cv": cmd_cv, "build": cmd_build, "audit": cmd_audit}[a.cmd]()
