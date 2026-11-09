#!/usr/bin/env python
"""V52: evaluate selector-conditioned post-processing rules against the production ongoing chains.

Windows: replica-selector ongoing windows (t2_v52_replica.py build) on the 9 train months (production members
were trained there: in-sample, favours the model) and on the March visible masked layer (out-of-sample for
the V48 chain; the V51 chain uses March-fine-tuned members cross-fitted by half month).  April is never used.
Members: t2_v52_prodcache.py (CPU re-implementation, reproduces the submitted V48/V51 decisions on the official
windows to <= 1 cell per panel).  Labels: train eligible truth, March visible speed <= v_cut.

  python work/t2_v52_selrule_eval.py run      # per-window IoU table for every rule and both chains
  python work/t2_v52_selrule_eval.py report   # summary tables (class, month, corridor, horizon)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import t2_grids  # noqa: E402
from evaluate import ROOT, T2_PANELS, links  # noqa: E402
from t2_group_v27 import groups, smooth  # noqa: E402
from t2_v33_adaptive_iou import adaptive_iou  # noqa: E402
import t2_v52_selrule as S  # noqa: E402

CACHE = ROOT / "work" / "cache" / "t2_v52_selrule"
OUT = ROOT / "work" / "results" / "t2_v52_selrule"


def pools(z):
    return dict(ml2g0=z["ml2g0"], catg0=z["catg0"], rnn2=(z["rnn2a"] + z["rnn2b"]) / 2,
                cnng0=np.mean([z[f"cnng0_{k}"] for k in range(3)], 0), rnng0=(z["rnng0_0"] + z["rnng0_1"]) / 2,
                rnnft=(z["rnnft_0"] + z["rnnft_1"]) / 2, rnnftx=(z["rnnft_2"] + z["rnnft_3"]) / 2,
                cnnftx=(z["cnnft_1"] + z["cnnft_2"]) / 2)


CHAINS = {"v48": ([("ml2g0", 2), ("cnng0", 1), ("rnng0", 5), ("rnn2", 2)], .25),
          "v51": ([("rnnft", 4), ("rnnftx", 2), ("cnnftx", 4)], 0.0)}


def chain_prob(pool, panel, chain):
    """The blended, group-smoothed probability that t2_g0_eval.decide feeds to adaptive_iou."""
    spec, a = CHAINS[chain]
    pr = sum(w * pool[n].astype(np.float32) for n, w in spec) / sum(w for _, w in spec)
    g = groups(panel, .95)
    pr = smooth(pr, g, .25)
    if a:
        pr = (1 - a) * pr + a * smooth(pool["catg0"].astype(np.float32), g, .25)
    return pr


def iou(P, y, m):
    p, t = P & m, y & m
    u = (p | t).sum()
    return 1.0 if u == 0 else float((p & t).sum() / u)


def pin(P, k, A=None):
    Q = P.copy()
    Q[:k] = P[0] if A is None else A
    return Q


def tilt(p, A, hs, target=0.9, floor=0.02):
    """I-projection of the factorized model onto the selector's persistence constraint (linearised):
    raise the change odds (anchor cells leaving, other cells joining) of frames hs by a common factor e^lam
    until E[|A x6 & Y|] <= target * E[|A x6 | Y|].  Cells with p < floor are not tilted (no diffuse growth)."""
    a = A.astype(bool)

    def moments(q):
        return (q[:, a].sum(), (a.sum() * q.shape[0]) + q[:, ~a].sum())

    I, U = moments(p)
    if U == 0 or I <= target * U:
        return p
    lg = np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1))
    free = p >= floor
    sign = np.where(a[None, :], -1.0, 1.0)

    def tilted(lam):
        q = p.copy()
        for h in hs:
            m = free[h]
            q[h, m] = 1 / (1 + np.exp(-(lg[h, m] + sign[0, m] * lam)))
        return q

    lo, hi = 0.0, 12.0
    q = tilted(hi)
    I, U = moments(q)
    if I > target * U:
        return q
    for _ in range(30):
        mid = (lo + hi) / 2
        I, U = moments(tilted(mid))
        if I > target * U:
            lo = mid
        else:
            hi = mid
    return tilted(hi)


def rules_for(P, p, c):
    """All candidate rules for one window. c: context (cls, panel, hour, vis_T, qvis_T)."""
    cls = c["cls"]
    out = {"base": P}
    con = cls == "constr"
    # --- constrained windows: pin the h1 set ---
    for k in (2, 3, 4, 5, 6):
        out[f"pin{k}"] = pin(P, k) if con else P
    # pin to the latent estimate at T: visible origin-row state where visible, model h1 elsewhere
    A = np.where(c["vis_T"], c["qvis_T"], P[0])
    out["pinv4"] = pin(P, 4, A) if con else P
    # probability-level pinning, then the same expected-IoU decision
    if con:
        q = p.copy()
        q[1:4] = p[0]
        out["ppin4"] = adaptive_iou(q)
        q = p.copy()
        q[1:5] = p[0]
        out["ppin5"] = adaptive_iou(q)
    else:
        out["ppin4"] = out["ppin5"] = P
    # pin4 + time-warp: h5 := model h3/h4 (the model's change compressed into h5/h6)
    if con:
        Q = pin(P, 4)
        Q[4] = P[3]
        out["pin4w"] = Q
    else:
        out["pin4w"] = P
    # pin5 + drop h6 (round-1 R1 generalised) and R1 itself
    if con:
        Q = pin(P, 5)
        Q[5] = False
        out["pin5d6"] = Q
    else:
        out["pin5d6"] = P
    out["R1"] = out["pin5d6"] if (con and c["panel"] == "D7_I210_E" and c["hour"] >= 17) else P
    # R2: constrained, small h1 set and the base clears the queue somewhere in h2..h5 -> carry h1 through h6
    n = P.sum(1)
    r2 = con and 0 < n[0] <= 6 and (n[1:5] == 0).any()
    out["R2"] = pin(P, 6) if r2 else P
    out["pin4R2"] = pin(P, 6) if r2 else out["pin4"]
    # --- selector-consistent tilts ---
    if con:
        q = p.copy()
        q[1:4] = p[0]
        out["tilt56"] = adaptive_iou(tilt(q, P[0], (4, 5)))
        out["tilt56np"] = adaptive_iou(tilt(p, P[0], (4, 5)))
    else:
        out["tilt56"] = out["tilt56np"] = P
    A5 = np.where(c["vis_T5"], c["qvis_T5"], P[0])
    out["tiltall"] = adaptive_iou(tilt(p, A5, range(6)))
    out["tiltnc"] = out["tiltall"] if not con else P
    # --- Bayesian conditioning on the selector event (comonotone sampler, t2_v52_selrule.condition) ---
    seed = c["seed"]
    for kind in ("group", "seg", "corr"):
        ids = S.coupling_ids(c["panel"], p, kind)
        for scope in ("c", "cs"):
            cl = cls if (scope == "cs" or cls == "constr") else "none"
            q, acc = S.condition(p, cl, ids, seed=seed)
            out[f"cond_{kind}_{scope}"] = adaptive_iou(q) if q is not p else P
            if kind == "seg" and scope == "c":
                out["cond_seg_c_mix"] = adaptive_iou(0.5 * p + 0.5 * q) if q is not p else P
                c["acc_seg"] = acc
    # --- pinning applied to other classes (controls) ---
    out["all_pin4"] = pin(P, 4)
    out["sp_pin4"] = pin(P, 4) if cls == "spacing" else P
    out["nb_pin4"] = pin(P, 4) if cls == "newborn" else P
    # --- newborn: monotone growth (cells never leave) / probability boost ---
    if cls == "newborn":
        out["nb_mono"] = np.logical_or.accumulate(P, axis=0)
        out["nb_boost"] = adaptive_iou(np.clip(p * 1.25, 0, 1))
        out["nb_damp"] = adaptive_iou(p * 0.8)
    else:
        out["nb_mono"] = out["nb_boost"] = out["nb_damp"] = P
    return out


def cmd_run(args):
    win = pd.read_csv(CACHE / "windows.csv")
    win = win[win.cond == "ongoing"].drop_duplicates(["panel", "split", "day", "T", "cls"])
    win = win[~win.blk_T5.astype(bool)]          # March: T-5 skipped for a blackout, no official analogue
    win = win[(win["T"] >= 13) & (win["T"] <= 281)]   # history/horizon within the day (all official windows)
    rows, hrows = [], []
    for split in ("validation", "train"):
        for panel in T2_PANELS:
            f = CACHE / f"prod_{panel}_{split}.npz"
            if not f.exists():
                print("missing", f.name, flush=True)
                continue
            z = dict(np.load(f))
            key = {(int(d), int(t)): i for i, (d, t) in enumerate(zip(z["day"], z["T"]))}
            pool = pools(z)
            probs = {ch: chain_prob(pool, panel, ch) for ch in CHAINS}
            g = t2_grids.load(panel, split)
            Vvis = g["Vt"] if split == "train" else g["V"]
            vc = links(panel).v_cut_kmh.to_numpy(np.float32)
            w = win[(win.panel == panel) & (win.split == split)]
            for r in w.itertuples(index=False):
                i = key[(int(r.day), int(r.T))]
                y, m = z["y"][i].astype(bool), z["mask"][i].astype(bool)
                vT, v5 = Vvis[r.day, r.T], Vvis[r.day, r.T - 1]
                with np.errstate(invalid="ignore"):
                    c = dict(cls=r.cls, panel=panel, hour=int(r.T) // 12, vis_T=np.isfinite(vT), qvis_T=vT <= vc,
                             vis_T5=np.isfinite(v5), qvis_T5=v5 <= vc, seed=int(r.day) * 288 + int(r.T))
                for ch in CHAINS:
                    p = probs[ch][i]
                    P = adaptive_iou(p)
                    rs = rules_for(P, p, c)
                    base = rs["base"]
                    rec = dict(split=split, panel=panel, month=r.month, day=int(r.day), T=int(r.T), cls=r.cls,
                               hour=c["hour"], chain=ch, piou=r.piou, k=int(r.k), n_true=int((y & m).sum()),
                               n_pred=int(base.sum()), acc_seg=c.get("acc_seg", np.nan))
                    for nm, Q in rs.items():
                        rec[nm] = iou(Q, y, m)
                        rec["ch_" + nm] = int((Q != base).sum())
                    rows.append(rec)
                    for nm in ("base", "pin4", "pin5", "ppin4"):
                        Q = rs[nm] & m
                        t = y & m
                        for h in range(6):
                            hrows.append((split, panel, r.cls, ch, nm, h + 1, int((Q[h] & t[h]).sum()),
                                          int((Q[h] | t[h]).sum())))
            print(split, panel, len(w), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "eval_windows.csv", index=False)
    pd.DataFrame(hrows, columns=["split", "panel", "cls", "chain", "rule", "h", "inter", "union"]).to_csv(
        OUT / "eval_horizon.csv", index=False)


def r1_mask(P, hour, panel, cls, h0=17, pin=True):
    """Train-selected dissolve rule (round-1 R1): constrained D7_I210_E windows from hour h0: h1..h5 = the h1
    link set (pin=True) or the model's own h1..h5 (pin=False), h6 empty."""
    if cls != "constr" or panel != "D7_I210_E" or hour < h0:
        return P
    Q = P.copy()
    if pin:
        Q[:5] = P[0]
    Q[5] = False
    return Q


def cmd_run2(args):
    """Second pass, reduced rule set: the Bayesian selector conditioning with group coupling at K = 2000 (as in
    run) and K = 20000 with three acceptance floors, the D7_I210_E dissolve rule at several hour thresholds,
    and the hybrid (dissolve rule where it applies, conditioning on the other constrained windows)."""
    win = pd.read_csv(CACHE / "windows.csv")
    win = win[win.cond == "ongoing"].drop_duplicates(["panel", "split", "day", "T", "cls"])
    win = win[~win.blk_T5.astype(bool)]
    win = win[(win["T"] >= 13) & (win["T"] <= 281)]
    rows, hrows = [], []
    for split in ("validation", "train"):
        for panel in T2_PANELS:
            z = dict(np.load(CACHE / f"prod_{panel}_{split}.npz"))
            key = {(int(d), int(t)): i for i, (d, t) in enumerate(zip(z["day"], z["T"]))}
            pool = pools(z)
            probs = {ch: chain_prob(pool, panel, ch) for ch in CHAINS}
            w = win[(win.panel == panel) & (win.split == split)]
            for r in w.itertuples(index=False):
                i = key[(int(r.day), int(r.T))]
                y, m = z["y"][i].astype(bool), z["mask"][i].astype(bool)
                hour = int(r.T) // 12
                for ch in CHAINS:
                    p = probs[ch][i]
                    P = adaptive_iou(p)
                    rs = {"base": P}
                    if r.cls == "constr":
                        ids = S.coupling_ids(panel, p, "group")
                        q, acc = S.condition(p, "constr", ids, K=2000, seed=int(r.day) * 288 + int(r.T))
                        rs["cg2k"] = adaptive_iou(q) if q is not p else P
                        q, acc = S.condition(p, "constr", ids, K=args.K, seed=int(r.day) * 288 + int(r.T) + 7,
                                             min_acc=0.0)
                        Qc = adaptive_iou(q) if acc > 0 else P
                        for nm, a in (("cg", 0.02), ("cg_a005", 0.005), ("cg_a001", 0.001)):
                            rs[nm] = Qc if acc >= a else P
                    else:
                        acc = np.nan
                        rs["cg2k"] = rs["cg"] = rs["cg_a005"] = rs["cg_a001"] = P
                    for h0 in (15, 16, 17, 18, 19):
                        rs[f"R1h{h0}"] = r1_mask(P, hour, panel, r.cls, h0)
                    rs["R1d6"] = r1_mask(P, hour, panel, r.cls, 17, pin=False)
                    r1 = r.cls == "constr" and panel == "D7_I210_E" and hour >= 17
                    rs["hyb"] = rs["R1h17"] if r1 else rs["cg"]
                    rs["hyb_a005"] = rs["R1h17"] if r1 else rs["cg_a005"]
                    rec = dict(split=split, panel=panel, month=r.month, day=int(r.day), T=int(r.T), cls=r.cls,
                               hour=hour, chain=ch, k=int(r.k), acc=acc, n_true=int((y & m).sum()),
                               n_pred=int(P.sum()))
                    for nm, Q in rs.items():
                        rec[nm] = iou(Q, y, m)
                        rec["ch_" + nm] = int((Q != P).sum())
                    rows.append(rec)
                    for nm in ("base", "cg", "hyb"):
                        Q = rs[nm] & m
                        t = y & m
                        for h in range(6):
                            hrows.append((split, panel, r.cls, ch, nm, h + 1, int((Q[h] & t[h]).sum()),
                                          int((Q[h] | t[h]).sum())))
            print(split, panel, len(w), flush=True)
    pd.DataFrame(rows).to_csv(OUT / "eval2_windows.csv", index=False)
    pd.DataFrame(hrows, columns=["split", "panel", "cls", "chain", "rule", "h", "inter", "union"]).to_csv(
        OUT / "eval2_horizon.csv", index=False)


def cmd_run3(args):
    """Third pass: the conditioning with the base h1 row kept (the h1 change in run2 comes only from the
    anchor approximation A = Y1), same samples as run2's cg (K, seed)."""
    win = pd.read_csv(CACHE / "windows.csv")
    win = win[win.cond == "ongoing"].drop_duplicates(["panel", "split", "day", "T", "cls"])
    win = win[~win.blk_T5.astype(bool)]
    win = win[(win["T"] >= 13) & (win["T"] <= 281) & (win.cls == "constr")]
    rows = []
    for split in ("validation", "train"):
        for panel in T2_PANELS:
            z = dict(np.load(CACHE / f"prod_{panel}_{split}.npz"))
            key = {(int(d), int(t)): i for i, (d, t) in enumerate(zip(z["day"], z["T"]))}
            pool = pools(z)
            probs = {ch: chain_prob(pool, panel, ch) for ch in CHAINS}
            w = win[(win.panel == panel) & (win.split == split)]
            for r in w.itertuples(index=False):
                i = key[(int(r.day), int(r.T))]
                y, m = z["y"][i].astype(bool), z["mask"][i].astype(bool)
                hour = int(r.T) // 12
                for ch in CHAINS:
                    p = probs[ch][i]
                    P = adaptive_iou(p)
                    ids = S.coupling_ids(panel, p, "group")
                    q, acc = S.condition(p, "constr", ids, K=args.K, seed=int(r.day) * 288 + int(r.T) + 7, min_acc=0.0)
                    Qc = adaptive_iou(q) if acc >= 0.02 else P
                    Qk = Qc.copy()
                    Qk[0] = P[0]
                    r1 = panel == "D7_I210_E" and hour >= 17
                    rs = {"base": P, "cg": Qc, "cg_k1": Qk}
                    rs["hyb"] = r1_mask(P, hour, panel, "constr", 17) if r1 else Qc
                    rs["hyb_k1"] = rs["hyb"] if r1 else Qk
                    rec = dict(split=split, panel=panel, month=r.month, day=int(r.day), T=int(r.T), cls=r.cls,
                               hour=hour, chain=ch, acc=acc)
                    for nm, Q in rs.items():
                        rec[nm] = iou(Q, y, m)
                        rec["ch_" + nm] = int((Q != P).sum())
                    rows.append(rec)
            print(split, panel, len(w), flush=True)
    pd.DataFrame(rows).to_csv(OUT / "eval3_windows.csv", index=False)


RULE_COLS = ["pin2", "pin3", "pin4", "pin5", "pin6", "pinv4", "ppin4", "ppin5", "pin4w", "pin5d6", "R1", "R2",
             "pin4R2", "tilt56", "tilt56np", "tiltall", "tiltnc", "cond_group_c", "cond_group_cs", "cond_seg_c",
             "cond_seg_cs", "cond_corr_c", "cond_corr_cs", "cond_seg_c_mix", "all_pin4", "sp_pin4", "nb_pin4", "nb_mono", "nb_boost", "nb_damp"]


def cmd_report(args):
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    df = pd.read_csv(OUT / "eval_windows.csv")
    df["src"] = np.where(df.split == "train", "train", "march")
    for nm in RULE_COLS:
        df["d_" + nm] = df[nm] - df.base
    print("windows per source/class/chain:")
    print(df.groupby(["src", "chain", "cls"]).size().unstack())
    print("\nmean base IoU per class:")
    print(df.groupby(["src", "chain", "cls"]).base.mean().unstack().round(4))
    print("\n== mean IoU delta vs base over ALL ongoing windows (x1000), by source/chain ==")
    t = df.groupby(["src", "chain"])[["d_" + n for n in RULE_COLS]].mean() * 1000
    print(t.round(2).T.to_string())
    for cls in ("constr", "newborn", "spacing"):
        sub = df[df.cls == cls]
        print(f"\n== class {cls}: mean delta (x1000) / windows better / worse / mean changed cells ==")
        rows = []
        for (src, ch), g in sub.groupby(["src", "chain"]):
            for nm in RULE_COLS:
                d = g["d_" + nm]
                if (g["ch_" + nm] > 0).sum() == 0:
                    continue
                rows.append(dict(src=src, chain=ch, rule=nm, n=len(g), d1000=1000 * d.mean(),
                                 se1000=1000 * d.std() / np.sqrt(len(g)), better=int((d > 1e-9).sum()),
                                 worse=int((d < -1e-9).sum()), cells=g["ch_" + nm].mean()))
        print(pd.DataFrame(rows).round(2).to_string(index=False))
    key = args.rules
    for nm in key:
        print(f"\n== {nm}: constr windows, mean delta x1000 by month (both chains) ==")
        sub = df[df.cls == "constr"]
        t = sub.pivot_table(index="month", columns="chain", values="d_" + nm, aggfunc="mean") * 1000
        t["n"] = sub[sub.chain == "v48"].groupby("month").size()
        t["better_v48"] = sub[sub.chain == "v48"].groupby("month")["d_" + nm].apply(lambda s: int((s > 1e-9).sum()))
        t["worse_v48"] = sub[sub.chain == "v48"].groupby("month")["d_" + nm].apply(lambda s: int((s < -1e-9).sum()))
        print(t.round(2).to_string())
        print(f"-- {nm}: constr windows by corridor (train | march) --")
        t = sub.pivot_table(index="panel", columns=["src", "chain"], values="d_" + nm, aggfunc="mean") * 1000
        n = sub[sub.chain == "v48"].groupby(["panel", "src"]).size().unstack()
        print(pd.concat([t.round(2), n.add_prefix("n_")], axis=1).to_string())
        print(f"-- {nm}: constr windows by hour band (v48 chain) --")
        s48 = sub[sub.chain == "v48"].copy()
        s48["band"] = pd.cut(s48.hour, [0, 10, 13, 16, 18, 24], right=False)
        print((s48.pivot_table(index="band", columns="src", values="d_" + nm, aggfunc="mean", observed=True) * 1000)
              .round(2).to_string())
    h = pd.read_csv(OUT / "eval_horizon.csv")
    h["src"] = np.where(h.split == "train", "train", "march")
    print("\n== per-horizon IoU (sum inter / sum union), constr windows ==")
    t = h[h.cls == "constr"].groupby(["src", "chain", "rule", "h"])[["inter", "union"]].sum()
    t = (t.inter / t.union).unstack("h").round(4)
    print(t.to_string())


def load_eval(name="eval_windows.csv"):
    df = pd.read_csv(OUT / name)
    df["src"] = np.where(df.split == "train", "train", "march")
    win = pd.read_csv(CACHE / "windows.csv")
    first = win[(win.cond == "ongoing") & (win.start == 0)][["panel", "split", "day", "T"]].drop_duplicates()
    first["first"] = True
    df = df.merge(first, on=["panel", "split", "day", "T"], how="left")
    df["first"] = df["first"].fillna(False).astype(bool)
    return df


def cmd_summary(args):
    """Compact evidence tables: every rule over ALL ongoing windows (each rule only touches its class),
    months / corridors positive, the official-analog subset (first 5 windows of each month, start day 0),
    and the per-class gain scaled to the official class counts."""
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 60)
    pd.set_option("display.max_rows", 500)
    df = load_eval(args.file)
    rules = args.rules or [c for c in df.columns if c in RULE_COLS or c in RUN2_COLS]
    off = pd.read_csv(OUT / "official_classes.csv").groupby(["split", "cls"]).size()
    rows = []
    for (src, ch), g in df.groupby(["src", "chain"]):
        for nm in rules:
            d = g[nm] - g.base
            bym, byp = d.groupby(g.month).mean(), d.groupby(g.panel).mean()
            ccls = d.groupby(g.cls).mean()
            # expected official gain: per-class mean delta x official class count / 40 windows x 0.15
            exp = {sp: 0.15 * sum(ccls.get(c, 0.0) * off.get((sp, c), 0) for c in ("constr", "newborn", "spacing")) / 40
                   for sp in ("validation", "private")}
            rows.append(dict(src=src, chain=ch, rule=nm, n=len(g), d1000=1000 * d.mean(),
                             se1000=1000 * d.std() / np.sqrt(len(g)), win=int((d > 1e-9).sum()),
                             lose=int((d < -1e-9).sum()),
                             months=f"{int((bym > 1e-9).sum())}+/{int((bym < -1e-9).sum())}-",
                             panels=f"{int((byp > 1e-9).sum())}+/{int((byp < -1e-9).sum())}-",
                             first1000=1000 * d[g["first"]].mean(),
                             constr1000=1000 * ccls.get("constr", np.nan), newborn1000=1000 * ccls.get("newborn", np.nan),
                             spacing1000=1000 * ccls.get("spacing", np.nan), cells=int(g["ch_" + nm].sum()),
                             exp_val=exp["validation"], exp_priv=exp["private"]))
    t = pd.DataFrame(rows)
    t.to_csv(OUT / args.file.replace("windows", "summary"), index=False)
    print("== all ongoing windows: delta IoU x1000 vs base (train months in-sample for the models; March out-of-sample) ==")
    print(t.round(4).sort_values(["src", "chain", "d1000"], ascending=[True, True, False]).to_string(index=False))
    for nm in args.detail:
        print(f"\n== {nm}: by month (mean delta x1000 over all ongoing windows of the month; n_chg = windows changed) ==")
        rows = []
        for (m, ch), g in df.groupby(["month", "chain"]):
            d = g[nm] - g.base
            rows.append(dict(month=m, chain=ch, n=len(g), n_chg=int((g["ch_" + nm] > 0).sum()), base=g.base.mean(),
                             d1000=1000 * d.mean(), better=int((d > 1e-9).sum()), worse=int((d < -1e-9).sum())))
        m = pd.DataFrame(rows).pivot(index="month", columns="chain")
        print(m.round(4).to_string())
        print(f"-- {nm}: by corridor --")
        rows = []
        for (src, p, ch), g in df.groupby(["src", "panel", "chain"]):
            d = g[nm] - g.base
            rows.append(dict(src=src, panel=p, chain=ch, n=len(g), n_chg=int((g["ch_" + nm] > 0).sum()),
                             d1000=1000 * d.mean(), better=int((d > 1e-9).sum()), worse=int((d < -1e-9).sum())))
        print(pd.DataFrame(rows).pivot(index="panel", columns=["src", "chain"], values=["n_chg", "d1000", "better", "worse"])
              .round(2).to_string())
    h = pd.read_csv(OUT / args.file.replace("windows", "horizon"))
    h["src"] = np.where(h.split == "train", "train", "march")
    print("\n== per-horizon IoU (sum inter / sum union), constrained windows ==")
    t = h[h.cls == "constr"].groupby(["src", "chain", "rule", "h"])[["inter", "union"]].sum()
    print((t.inter / t.union).unstack("h").round(4).to_string())
    if args.by_panel:
        print("\n== per-horizon IoU, constrained windows, by corridor (v48 chain) ==")
        t = h[(h.cls == "constr") & (h.chain == "v48")].groupby(["panel", "src", "rule", "h"])[["inter", "union"]].sum()
        print((t.inter / t.union).unstack("h").round(4).to_string())


RUN2_COLS = ["cg2k", "cg", "cg_a005", "cg_a001", "R1h15", "R1h16", "R1h17", "R1h18", "R1h19", "R1d6", "hyb", "hyb_a005"]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("run")
    s = sub.add_parser("run2")
    s.add_argument("--K", type=int, default=20000)
    s = sub.add_parser("run3")
    s.add_argument("--K", type=int, default=20000)
    s = sub.add_parser("report")
    s.add_argument("--rules", nargs="*", default=["pin4", "pin5", "ppin4"])
    s = sub.add_parser("summary")
    s.add_argument("--rules", nargs="*", default=None)
    s.add_argument("--detail", nargs="*", default=["pin4"])
    s.add_argument("--file", default="eval_windows.csv")
    s.add_argument("--by-panel", action="store_true")
    a = ap.parse_args()
    {"run": cmd_run, "run2": cmd_run2, "run3": cmd_run3, "report": cmd_report, "summary": cmd_summary}[a.cmd](a)
