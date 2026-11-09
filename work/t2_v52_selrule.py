#!/usr/bin/env python
"""V52: selector-conditioned post-processing of the production ongoing predictions (Task 2).

Why.  The official ongoing windows were cut by the documented greedy selector (config/synthetic_release_v1.json
task2 policy; organizer replies 745700 / 749278 allow conditioning a forecast on it): one forward pass per
(panel, split), >= 360 min to the previous window of either condition, history coverage >= 0.7, an established
visible queue history, a queue in the horizon and latent persistence IoU <= 0.9.  The production ongoing models
were trained on generic windows and do not know this.  At the origin T, from the window_index origins strictly
before T and masked-layer rows <= T only, every official ongoing window falls into one class
(t2_v52_replica.py replicates the selector on train months: 39/40 onset and 32/40 ongoing June-2030 matches):

  spacing   T = previous origin (either condition) + 360 min: T-5 was blocked by the spacing rule
  constr    not spacing-bound and the T-5 history (rows T-65..T-10) was already established with coverage
            >= 0.7, so T-5 was a valid ongoing candidate that the selector rejected for latent persistence
            IoU > 0.9: the latent queue at T..T+25 is a near copy of the queue at T-10, and the change that
            lets T through (IoU <= 0.9) lands at T+25/T+30
  newborn   the history only became established at T (a young, growing queue)

Truth structure of ~1300 constrained replica windows (train months, t2_v52_selrule_eval.py): relative to the h1
queue set, h2..h4 lose/gain only 0.4-4 % of the cells, h5 5-7 %, h6 17-29 %.

RULE "hyb" (chosen on the 9 train months, confirmed out-of-sample on March replica pseudo windows; see
work/V52_SELRULE_README.md), constrained windows only:
  * D7_I210_E from 17:00: h1..h5 := the h1 link set, h6 empty (the evening 5-link block dissolves)
  * every other constrained window: Bayesian conditioning of the production chain probability on the
    selector event (condition(), train-only co-queue group coupling, K_MC = 20000, MIN_ACC = .02), the same
    expected-IoU decision on h2..h6, h1 kept
  spacing / newborn / onset windows are never changed.  Replica evidence (Delta IoU x1000 per constrained
  window): train V48 +34.1, V51 +33.6 (9/9 months); March V48 +16.2, V51 +14.8.

Causality.  The class uses (a) window_index origins strictly before T (either condition) and (b) masked-layer
rows T-65..T-10 of the same day, both stamped <= T; the rule only rewrites the base (V48 / V51) binary
prediction of the same window.  `check` recomputes every class from inputs where every row after T and every
origin after T are deleted and asserts identical classes.

Usage (from the "IEEE Big data" folder):
  python work/t2_v52_selrule.py classify            # class of every official ongoing window
  python work/t2_v52_selrule.py check               # truncation (as-of-T) check of the classification
  python work/t2_v52_selrule.py build               # writes work/sub/queue_{validation,private}_v52_selrule.csv
"""
from __future__ import annotations

import argparse
import json
import sys
import zlib
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import CLEAN, RELEASE, ROOT, STEPS, T2_PANELS, links, slot_of  # noqa: E402

WORK = ROOT / "work"
OUT = WORK / "results" / "t2_v52_selrule"
NAME = "v52_selrule"
BASES = {"validation": "v48_g0", "private": "v51_april"}      # V51 validation queue == V48 byte for byte
HIST = 12
SPACING_MIN = 360
MIN_COV = 0.7


# ----------------------------------------------------------------------------------------------------------
# classification at T (causal: window_index origins < T, masked-layer rows <= T)
# ----------------------------------------------------------------------------------------------------------
@lru_cache(maxsize=64)
def _day_rows(panel, split, date):
    """(288, n) visible speed (masked layer) and is_score_eligible for one date, pos order."""
    pf = pq.ParquetFile(CLEAN / panel / f"{split}.parquet")
    n = len(links(panel))
    for i in range(pf.num_row_groups):
        md = pf.read_row_group(i, columns=["date"]).column("date")[0].as_py()
        if str(md) == date:
            t = pf.read_row_group(i, columns=["speed_clean", "is_score_eligible"])
            V = t.column("speed_clean").to_numpy(zero_copy_only=False).astype(np.float32).reshape(STEPS, n)
            E = t.column("is_score_eligible").to_numpy(zero_copy_only=False).astype(bool).reshape(STEPS, n)
            return V, E
    raise KeyError(date)


def window_class(prev_origins: pd.Series, Tt: pd.Timestamp, V_upto: np.ndarray, E_upto: np.ndarray,
                 vc: np.ndarray) -> dict:
    """Class of an ongoing window with origin Tt (slot T = len(V_upto) - 1).

    prev_origins: window_index origins (either condition) strictly before Tt.  V_upto / E_upto: the day's
    masked-layer speed / eligibility rows 0..T (nothing after T is passed in)."""
    T = len(V_upto) - 1
    assert T >= HIST + 1, "history crosses midnight"
    gap = (Tt - prev_origins.max()).total_seconds() / 60 if len(prev_origins) else np.nan
    H5 = V_upto[T - HIST - 1:T - 1]                           # rows T-65 .. T-10
    with np.errstate(invalid="ignore"):
        q5 = np.isfinite(H5) & (H5 <= vc)
        qT = np.isfinite(V_upto[T]) & (V_upto[T] <= vc)
    est5 = bool(q5.sum() >= 2 and q5.sum(0).max() >= 2)
    cov5 = float(E_upto[T - HIST - 1:T - 1].mean())
    spacing = bool(np.isfinite(gap) and abs(gap - SPACING_MIN) < 1e-6)
    cls = "spacing" if spacing else ("constr" if (est5 and cov5 >= MIN_COV) else "newborn")
    return dict(T=T, hour=Tt.hour, gap_min=gap, spacing=spacing, est_T5=est5, cov_T5=round(cov5, 4),
                n_vis_T=int(qT.sum()), cls=cls)


def official_classes(panel: str, split: str, truncate: bool = False) -> pd.DataFrame:
    """truncate=True: the as-of-T check -- rows after T are overwritten with garbage and origins after T are
    dropped before calling window_class (identical output expected)."""
    wi = pd.read_csv(RELEASE / "task2" / panel / split / "window_index.csv")
    wi["Tt"] = pd.to_datetime(wi.forecast_origin)
    wi = wi.sort_values("Tt").reset_index(drop=True)
    vc = links(panel).v_cut_kmh.to_numpy(np.float32)
    rng = np.random.default_rng(0)
    rows = []
    for _, r in wi.iterrows():
        if r.condition != "queue_ongoing":
            continue
        T = int(slot_of(pd.Series([r.forecast_origin]))[0]) % STEPS
        V, E = _day_rows(panel, split, str(r.date))
        prev = wi.Tt[wi.Tt < r.Tt]
        if truncate:
            V = V.copy()
            E = E.copy()
            V[T + 1:] = rng.uniform(0, 130, V[T + 1:].shape)
            E[T + 1:] = rng.random(E[T + 1:].shape) < 0.5
            prev = wi.Tt[wi.Tt <= r.Tt]
            prev = prev[prev < r.Tt]
        c = window_class(prev, r.Tt, V[:T + 1], E[:T + 1], vc)
        rows.append(dict(window_id=r.window_id, panel=panel, split=split, date=str(r.date), **c))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------------------------
# post-processing rules on one window's binary prediction P (6, n) (rows h1..h6 = T+5..T+30)
# ----------------------------------------------------------------------------------------------------------
def rule_pin(P, upto):
    """h1..h_upto := the h1 link set (the selector's T-5 rejection pins the latent queue through T+25)."""
    Q = P.copy()
    Q[:upto] = P[0]
    return Q


def coupling_ids(panel: str, p: np.ndarray, kind: str) -> np.ndarray:
    """Link -> coupling id for the comonotone sampler (used by t2_v52_selrule_eval.py).  'group': train-only
    co-queue groups (Jaccard >= .95, t2_group_v27), other links alone; 'seg': contiguous runs (travel order) of
    links the model may queue (max_h p >= 0.2) share one id; 'corr': the whole corridor shares one id."""
    n = p.shape[1]
    if kind == "corr":
        return np.zeros(n, int)
    ids = np.arange(n)
    if kind == "group":
        from t2_group_v27 import groups
        for k, g in enumerate(groups(panel, .95)):
            ids[np.asarray(g)] = n + k
        return ids
    if kind == "seg":
        from t1_unet import static
        perm = static(panel)["perm"]
        on = p.max(0)[perm] >= 0.2
        cur = -1
        for j, i in enumerate(perm):
            if on[j]:
                if j == 0 or not on[j - 1]:
                    cur = n + j
                ids[i] = cur
        return ids
    raise ValueError(kind)


def condition(p: np.ndarray, cls: str, ids: np.ndarray, K: int = 2000, seed: int = 0, min_acc: float = 0.02,
              target: float = 0.9):
    """Bayesian conditioning of the production forecast on the selector event of the window's class.

    USED BY V52: decide("cond" / "hyb") calls it on every constrained window outside the dissolve rule with
    cls="constr", group coupling, K=K_MC (20000), seed=crc32(window_id), min_acc=MIN_ACC (.02) (frozen).  The
    defaults K=2000 / seed=0 and the cls="spacing" branch are only used by the evaluation runs in
    t2_v52_selrule_eval.py; the spacing branch is NOT used for V52 (spacing windows are never changed).

    Generative model: Y[h, i] = 1[U_c(i) < p[h, i]] with one uniform per coupling id (comonotone in time and
    within a coupling unit): every per-cell marginal equals the production probability, each unit switches at
    most once in the horizon, units are independent.  Selector event (anchor = state at T+5, taken as the
    latent state at T-10..T, which constrained windows imply):
      constr   T-5 rejected: IoU(A x6, [A, A, Y2..Y5]) > 0.9     and   T accepted: IoU(A x6, [A, Y2..Y6]) <= 0.9
      spacing  T accepted only
    Returns (conditional marginals q, acceptance rate); q = p when the event has < min_acc prior mass."""
    if cls not in ("constr", "spacing"):
        return p, 1.0
    rng = np.random.default_rng(seed)
    uid, inv = np.unique(ids, return_inverse=True)
    U = rng.random((K, len(uid)), dtype=np.float32)[:, inv]                 # (K, n)
    Y = U[:, None, :] < p[None].astype(np.float32)                         # (K, 6, n)
    A = Y[:, 0]
    a = A.sum(1).astype(np.float64)
    inter = (Y & A[:, None]).sum(2).astype(np.float64)
    union = (Y | A[:, None]).sum(2).astype(np.float64)
    i2, u2 = a + inter[:, 1:6].sum(1), a + union[:, 1:6].sum(1)
    ok = (u2 > 0) & (i2 <= target * u2)
    if cls == "constr":
        i1, u1 = 2 * a + inter[:, 1:5].sum(1), 2 * a + union[:, 1:5].sum(1)
        ok &= (u1 > 0) & (i1 > target * u1)
    acc = float(ok.mean())
    if acc < min_acc:
        return p, acc
    return Y[ok].mean(0).astype(np.float32), acc


def rule_dissolve(P):
    """D7_I210_E evening dissolve rule (round-1 R1, re-selected on train months): h1..h5 = the h1 link set,
    h6 empty."""
    Q = P.copy()
    Q[:5] = P[0]
    Q[5] = False
    return Q


# production chains whose probabilities the conditioning needs (t2_v52_selrule_eval.CHAINS)
CHAIN = {"validation": "v48", "private": "v51"}
K_MC = 20000            # Monte-Carlo samples of the comonotone sampler
MIN_ACC = 0.02          # skip the conditioning when the selector event has < 2 % prior mass
DISSOLVE = ("D7_I210_E", 17)      # corridor, first hour of the dissolve rule

# the rule used for V52, chosen on the train months (t2_v52_selrule_eval.py run2/run3: "hyb_k1")
RULE = "hyb"
RULE_NAMES = ("base", "pin4", "cond", "dissolve", "hyb")


def seed_of(window_id: str) -> int:
    return zlib.crc32(window_id.encode("ascii"))


def decide(rule: str, P: np.ndarray, p: np.ndarray | None, P0: np.ndarray | None, ctx: dict):
    """New binary prediction of one ongoing window.

    P   submitted (base) binary prediction (6, n), h1..h6 = T+5..T+30
    p   the production chain probability recomputed on CPU (t2_v52_prodcache.py official) and P0 =
        adaptive_iou(p) (== P up to <= 1 cell per corridor); only the conditioning needs them
    rules  pin4      constrained: h1..h4 := h1 set (round-1 idea; rejected by the evaluation)
           cond      constrained: Bayesian conditioning of p on the selector event (condition(), group
                     coupling, K_MC samples, MIN_ACC floor), then the same expected-IoU decision on h2..h6
                     (h1 kept: the event's anchors L(T-10), L(T-5), L(T) are approximated by Y1); applied as the
                     flips it makes relative to P0, so every other cell stays the submitted one
           dissolve  constrained D7_I210_E windows from DISSOLVE[1] h: rule_dissolve
           hyb       dissolve where it applies, cond on every other constrained window
    Returns (new P, tag, acceptance rate)."""
    P = P.astype(bool)
    if rule == "base" or ctx["cls"] != "constr":
        return P, "", np.nan
    if rule == "pin4":
        return rule_pin(P, 4), "pin4", np.nan
    dis = ctx["panel"] == DISSOLVE[0] and int(ctx["hour"]) >= DISSOLVE[1]
    if rule in ("dissolve", "hyb") and dis:
        return rule_dissolve(P), "dissolve", np.nan
    if rule in ("cond", "hyb"):
        ids = coupling_ids(ctx["panel"], p, "group")
        q, acc = condition(p, "constr", ids, K=K_MC, seed=seed_of(ctx["window_id"]), min_acc=MIN_ACC)
        if q is p:
            return P, "cond_skip", acc
        from t2_v33_adaptive_iou import adaptive_iou
        N = P ^ (adaptive_iou(q) ^ P0)
        N[0] = P[0]                    # keep the base h1 row (the event's anchor is approximated by Y1)
        return N, "cond", acc
    return P, "", np.nan


def official_probs(panel: str, split: str):
    """window_id -> production chain probability (6, n) for the official ongoing windows."""
    from t2_v52_selrule_eval import chain_prob, pools
    f = WORK / "cache" / "t2_v52_selrule" / f"official_{panel}_{split}.npz"
    if not f.exists():
        raise SystemExit(f"missing {f}: run  python work/t2_v52_prodcache.py official")
    z = dict(np.load(f))
    pr = chain_prob(pools(z), panel, CHAIN[split])
    return {str(w): pr[i] for i, w in enumerate(z["ids"])}


# ----------------------------------------------------------------------------------------------------------
def cmd_classify(args):
    df = pd.concat([official_classes(p, s) for s in ("validation", "private") for p in T2_PANELS],
                   ignore_index=True)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "official_classes.csv", index=False)
    print(df.groupby(["split", "cls"]).size().unstack())
    print(df.to_string())


def cmd_check(args):
    a = pd.concat([official_classes(p, s) for s in ("validation", "private") for p in T2_PANELS],
                  ignore_index=True)
    b = pd.concat([official_classes(p, s, truncate=True) for s in ("validation", "private") for p in T2_PANELS],
                  ignore_index=True)
    same = a.equals(b)
    saved = pd.read_csv(OUT / "official_classes.csv") if (OUT / "official_classes.csv").exists() else None
    print("classes identical after deleting rows / origins after T:", same)
    if saved is not None:
        print("classes identical to official_classes.csv:", bool((saved.cls.values == a.cls.values).all()))
    if not same:
        raise SystemExit("truncation check FAILED")


def cmd_build(args):
    rule = args.rule or RULE
    cls = pd.read_csv(OUT / "official_classes.csv") if (OUT / "official_classes.csv").exists() else None
    if cls is None:
        raise SystemExit("run classify first")
    cls = cls.set_index("window_id")
    log = []
    for split, base in BASES.items():
        dst = WORK / "sub" / f"queue_{split}_{NAME}.csv"
        if dst.exists() and not args.dry_run:
            raise FileExistsError(dst)
        q = pd.read_csv(WORK / "sub" / f"queue_{split}_{base}.csv", dtype={"link_id": str})
        q0 = q.copy()
        q["slot"] = slot_of(q.timestamp)
        for panel in T2_PANELS:
            L = links(panel)
            pos = dict(zip(L.link_id.astype(str), range(len(L))))
            wi = pd.read_csv(RELEASE / "task2" / panel / split / "window_index.csv")
            probs = official_probs(panel, split) if rule in ("cond", "hyb") else {}
            for r in wi[wi.condition == "queue_ongoing"].itertuples(index=False):
                m = q.window_id.eq(r.window_id).to_numpy()
                row = q.loc[m]
                T = int(slot_of(pd.Series([r.forecast_origin]))[0])
                hor = (row.slot.to_numpy() - T - 1).astype(int)
                ix = row.link_id.map(pos).to_numpy().astype(int)
                assert hor.min() == 0 and hor.max() == 5 and len(row) == 6 * len(set(ix))
                P = np.zeros((6, len(L)), bool)
                P[hor, ix] = row.queue_pred.to_numpy(int) == 1
                c = cls.loc[r.window_id].to_dict()
                c["panel"], c["window_id"] = panel, r.window_id
                p = probs.get(r.window_id)
                P0 = None
                if p is not None:
                    from t2_v33_adaptive_iou import adaptive_iou
                    P0 = adaptive_iou(p)
                    ndiff = int((P0 != P).sum())
                    assert ndiff <= 2, (r.window_id, ndiff)          # CPU recompute == submitted (<= 1 cell seen)
                N, tag, acc = decide(rule, P, p, P0, c)
                new = N[hor, ix].astype(int)
                old = row.queue_pred.to_numpy(int)
                q.loc[m, "queue_pred"] = new
                log.append(dict(split=split, window_id=r.window_id, panel=panel, cls=c["cls"], hour=c["hour"],
                                gap_min=c["gap_min"], est_T5=c["est_T5"], cov_T5=c["cov_T5"],
                                rule=tag, acc=acc, recompute_diff=ndiff if p is not None else -1,
                                n_pred_base="-".join(str(int(x)) for x in P.sum(1)),
                                n_pred_new="-".join(str(int(x)) for x in N.sum(1)), changed=int((old != new).sum()),
                                added=int(((old == 0) & (new == 1)).sum()),
                                removed=int(((old == 1) & (new == 0)).sum())))
        q = q.drop(columns="slot")
        assert q.drop(columns="queue_pred").equals(q0.drop(columns="queue_pred"))
        ong = q.window_id.isin(cls.index[cls.split == split])
        assert (q.queue_pred[~ong] == q0.queue_pred[~ong]).all(), "only ongoing rows may change"
        print(split, "->", dst.name, "changed cells", int((q.queue_pred != q0.queue_pred).sum()), flush=True)
        if not args.dry_run:
            q.to_csv(dst, index=False)
    lg = pd.DataFrame(log)
    if not args.dry_run:
        lg.to_csv(OUT / f"{NAME}_changes.csv", index=False)
        (OUT / f"{NAME}_rule.json").write_text(json.dumps(dict(rule=rule, bases=BASES, chain=CHAIN, K_MC=K_MC,
                                                                MIN_ACC=MIN_ACC, DISSOLVE=DISSOLVE), indent=2))
    print(lg[lg.changed > 0].to_string())
    print(lg.groupby("split")[["changed", "added", "removed"]].sum())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("classify")
    sub.add_parser("check")
    s = sub.add_parser("build")
    s.add_argument("--rule", default=None, choices=RULE_NAMES)
    s.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    {"classify": cmd_classify, "check": cmd_check, "build": cmd_build}[a.cmd](a)
