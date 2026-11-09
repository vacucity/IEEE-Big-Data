"""V59 lane: gap-specialist fine-tune of the production Task-1 U-Nets for the official Task-2 blackouts (T, T+90].

Only the Task-1 target rows inside official Task-2 blackouts change (against V58 = work/submissions/v58_rs_blk.zip);
every other state row, every queue row and every ODME row stays byte-identical.

Why (work/V57_BLACKOUT_README.md): inside an official blackout the production writes the mean of the two U-Net tags
(w48_vp, w48lwr_vp; V57_blk adds a LightGBM residual).  The U-Nets were trained with corridor gaps at RANDOM times;
official windows sit at queue events chosen by the selector (onset: no established queue in T-60..T-5, a queue enters
the horizon, mostly at T+30; ongoing: an established queue at T whose shape changes in T+5..T+30), the history rows
T-60..T-5 are fully visible (no Task-1 holes) and the window's condition is public (window_index.csv).

Model: the production U-Net (width 48) with 3 extra input channels inside each official(-like) gap row -
gap flag, k/18 (position in the gap) and condition (+1 onset / -1 ongoing) - initialised from the production weights
(zero weights for the new channels, so the start is exactly the production net), then fine-tuned on official-like
gaps of the TRAIN months with their truth: replica selector origins (work/cache/t2_v52_selrule/windows.csv, train),
every cell of the corridor blanked in rows T+1..T+18, history rows T-12..T-1 un-hidden to the truth with probability
P_FULL (official layout) or left with their Task-1 holes (ruler layout), loss = production score-unit loss on the gap
cells (weight 1) + the other Task-1 targets of the day (weight W_OTHER), + 0.05 x Task-3 conservation term for the
lwr tag (as its production recipe).  Fold k starts from t1_unet_fold{k}{tag}_vp.pt and sees only train months with
month % 2 != k (its own training months), so months with month % 2 == k are held out for fold k.

Steps (project root, anaconda python):
  runs                      training/eval runs on the train months (cache/train_runs.csv)
  occ                       occ_true cache of the train months (history un-hiding needs the occupancy channel)
  train --tag T --fold k    fine-tune -> work/models/t1_unet_fold{k}{T}_v59bs.pt
  heldout                   train held-out months, official layout: production fold-k net vs specialist fold-k net
  wide --split S            specialist outputs on the V57 ruler windows (results/t1_v57_blkruler, same runs and
                            cells as sim_wide) -> results/t1_v59_blkspec/wide/
  crossfit                  V57 residual GBM on top of the specialist fill, day-parity cross-fit, pre-registered
                            comparison with V57_blk on the ruler (docstring of cmd_crossfit)
  official                  specialist outputs around the 160 official windows
  fit / apply / package / audit / compare      (only if crossfit says the candidate wins)

Rules: Task 1 is offline.  Inputs: train-month truth (training only), the visible observations of March/April, the
public window_index.csv (origin + condition), frozen production models.  No hidden label of March/April, no Task-2
prediction, no public-LB feedback.  Nothing existing is overwritten.
"""
from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "4")

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
import t1_unet as U  # noqa: E402
from evaluate import CLEAN, PANELS, STEPS, T2_PANELS, links  # noqa: E402

WORK = Path(__file__).resolve().parent
OUT = WORK / "results" / "t1_v59_blkspec"
CACHE = OUT / "cache"
WIDE = OUT / "wide"
OFFICIAL = OUT / "official"
MODELS = WORK / "models"
RULER = WORK / "results" / "t1_v57_blkruler"
V57 = WORK / "results" / "t1_v57_blackout"
SPLITS = ("validation", "private")
TAGS = ("_w48", "_w48lwr")
OUT_SUFFIX = "_v59bs"
GAP, HIST = 18, 12
RUN_SEP = 48
EXTRA_C = 3
N_IN2 = U.N_IN + EXTRA_C
WIDTH = 48
P_FULL = 0.5
W_OTHER = 0.2
SEED = 59


def write_new(path: Path, text: str) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.write_text(text, encoding="utf-8")


# ---------------------------------------------------------------------------
# model with extra channels
# ---------------------------------------------------------------------------
def build_model2(width: int = WIDTH):
    old = U.N_IN
    try:
        U.N_IN = N_IN2
        return U.build_model(width)
    finally:
        U.N_IN = old


def init_from_production(tag: str, fold: int, device: str):
    import torch
    sd = torch.load(U.MODELS / f"t1_unet_fold{fold}{tag}_vp.pt", map_location="cpu")
    m = build_model2()
    new = m.state_dict()
    for k, v in sd.items():
        if new[k].shape == v.shape:
            new[k] = v
        else:                                    # first conv / 1x1 skip of the first block: zero extra channels
            assert k in ("e1.c1.weight", "e1.skip.weight"), k
            w = torch.zeros(new[k].shape, dtype=v.dtype)
            w[:, :v.shape[1]] = v
            new[k] = w
    m.load_state_dict(new)
    return m.to(device)


def load_spec(tag: str, fold: int, device: str = "cuda"):
    import torch
    m = build_model2()
    m.load_state_dict(torch.load(MODELS / f"t1_unet_fold{fold}{tag}{OUT_SUFFIX}.pt", map_location="cpu"))
    m.eval()
    return m.to(device)


def load_prod(tag: str, fold: int, device: str = "cuda"):
    import torch
    m = U.build_model(WIDTH)
    m.load_state_dict(torch.load(U.MODELS / f"t1_unet_fold{fold}{tag}_vp.pt", map_location="cpu"))
    m.eval()
    return m.to(device)


# ---------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------
def extra_channels(n: int, gaps) -> np.ndarray:
    """gaps: iterable of (T, cond) -> (3, 288, n): gap flag, k/18, condition (+1 onset, -1 ongoing) in rows T+1..T+18."""
    x = np.zeros((EXTRA_C, STEPS, n), np.float32)
    for T, cond in gaps:
        r = np.arange(T + 1, min(T + 1 + GAP, STEPS))
        x[0, r] = 1.0
        x[1, r] = ((r - T) / GAP)[:, None]
        x[2, r] = 1.0 if cond == "onset" else -1.0
    return x


def make_input2(d, gaps, hide=None, unhide=None, occ=None):
    """Production input of day d (travel order) with `hide` blanked, optional history un-hiding (train only: cells in
    `unhide` get their truth, occupancy from `occ`) and the 3 gap channels."""
    if unhide is not None and unhide.any():
        o = d.obs
        o[..., 0] = np.where(unhide, d.truth[..., 0], o[..., 0])
        o[..., 1] = np.where(unhide, d.truth[..., 1], o[..., 1])
        o[..., 2] = np.where(unhide, np.nan_to_num(occ), o[..., 2])
        o[..., 3] = np.where(unhide, 1.0, o[..., 3])
    x, b = U.make_input(d, hide)
    return np.concatenate([x, extra_channels(x.shape[-1], gaps)]), b


def predict_np(model, x, b, st, fp32=True):
    """(v km/h, q vph) in travel order, as t1_unet.predict_day / t1_v52_monthbias.Chain.unet."""
    import torch
    dev = next(model.parameters()).device
    with torch.no_grad():
        X = U.to_batch([x]).to(dev)
        if fp32:
            out = model(X).float().cpu().numpy()[0]
        else:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(X).float().cpu().numpy()[0]
    n = x.shape[-1]
    v = (b[0] + out[0, :, :n]) * st["ffs"]
    q = (b[1] + out[1, :, :n]) * st["cap"]
    return np.clip(v, 5.0, 120.75), np.maximum(q, 0.0)


# ---------------------------------------------------------------------------
# training runs (train months) and the occupancy cache
# ---------------------------------------------------------------------------
def pack(rows):
    """[(T, cond), ...] of one day -> runs whose gaps are >= RUN_SEP slots apart (t1_v57_blkruler.pack_runs)."""
    open_runs = []
    for T, c in sorted(rows):
        for run in open_runs:
            if all(abs(T - x[0]) >= RUN_SEP for x in run):
                run.append((T, c))
                break
        else:
            open_runs.append([(T, c)])
    return open_runs


def cmd_runs(args) -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / "train_runs.csv"
    if path.exists():
        raise FileExistsError(path)
    W = pd.read_csv(WORK / "cache" / "t2_v52_selrule" / "windows.csv")
    W = W[W.split == "train"].drop_duplicates(["panel", "month", "t", "cond"])
    W = W[(W["T"] >= HIST) & (W["T"] <= STEPS - GAP - 1)]
    rows = []
    for p, g in W.groupby("panel"):
        meta = json.loads((U.DATA / f"{p}_train_meta.json").read_text())
        di = {d: i for i, d in enumerate(meta["dates"])}
        for date, gd in g.groupby("date"):
            i = di[date]
            for ir, run in enumerate(pack(list(zip(gd["T"].astype(int), gd.cond)))):
                rows.append(dict(panel=p, day=i, date=date, month=meta["month"][i], regime=meta["regime"][i], sub=ir,
                                 gaps=";".join(f"{T}:{c}" for T, c in run), n_gaps=len(run)))
    R = pd.DataFrame(rows)
    R.to_csv(path, index=False)
    print(R.groupby("panel").agg(runs=("day", "size"), gaps=("n_gaps", "sum")), flush=True)
    print("total runs", len(R), "gaps", int(R.n_gaps.sum()), flush=True)


def parse_gaps(s: str):
    return [(int(a), b) for a, b in (x.split(":") for x in s.split(";"))]


def cmd_occ(args) -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    for p in T2_PANELS:
        path = CACHE / f"occ_{p}_train.npy"
        if path.exists():
            print("exists", path.name)
            continue
        perm = U.static(p)["perm"]
        n = len(perm)
        pf = pq.ParquetFile(CLEAN / p / "train.parquet")
        a = np.lib.format.open_memmap(str(path) + ".tmp.npy", "w+", np.float16, (pf.num_row_groups, STEPS, n))
        for i in range(pf.num_row_groups):
            t = pf.read_row_group(i, columns=["occ_true"])
            a[i] = t.column(0).to_numpy(zero_copy_only=False).astype(np.float32).reshape(STEPS, n)[:, perm]
        a.flush()
        del a
        os.replace(str(path) + ".tmp.npy", path)
        print("occ", p, flush=True)


def train_sample(d, gaps, rng, occ_day, p_full):
    """Input, baseline, truth, weights of one training run (train month, truth layer)."""
    n = d.obs.shape[1]
    hide = np.zeros((STEPS, n), bool)
    unhide = np.zeros((STEPS, n), bool)
    tr = np.moveaxis(d.truth, -1, 0)
    ok = np.isfinite(tr).all(0)
    vis0 = d.obs[..., 3] > 0.5
    gaprow = np.zeros(STEPS, bool)
    for T, _ in gaps:
        hide[T + 1:T + 1 + GAP] = True
        gaprow[T + 1:T + 1 + GAP] = True
        if rng.random() < p_full:
            unhide[T - HIST:T] = ok[T - HIST:T] & ~vis0[T - HIST:T]
    x, b = make_input2(d, gaps, hide, unhide, occ_day)
    w = np.where(gaprow[:, None] & ok, 1.0, 0.0) + np.where(~gaprow[:, None] & d.tgt & ok & ~unhide, W_OTHER, 0.0)
    return x, b, np.nan_to_num(tr), w.astype(np.float32), ok.astype(np.float32)


def cmd_train(args) -> None:
    import torch
    torch.manual_seed(SEED + 10 * args.fold + (0 if args.tag == "_w48" else 1))
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    rng = np.random.default_rng(SEED + 10 * args.fold + (0 if args.tag == "_w48" else 1))
    out = MODELS / f"t1_unet_fold{args.fold}{args.tag}{OUT_SUFFIX}{args.suffix}.pt"
    if out.exists():
        raise FileExistsError(out)
    R = pd.read_csv(CACHE / "train_runs.csv")
    R = R[R.month % 2 != args.fold].reset_index(drop=True)
    srcs = {p: U.open_split(p, "train") for p in T2_PANELS}
    sts = {p: U.static(p) for p in T2_PANELS}
    profs = {p: np.load(U.DATA / f"{p}_profiles.npy") for p in T2_PANELS}
    occs = {p: np.load(CACHE / f"occ_{p}_train.npy", mmap_mode="r") for p in T2_PANELS}
    dev = "cuda"
    model = init_from_production(args.tag, args.fold, dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    batches_per_epoch = sum(math.ceil(len(g) / args.batch) for _, g in R.groupby("panel"))
    total = batches_per_epoch * args.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=total, pct_start=0.1,
                                                anneal_strategy="cos", div_factor=10, final_div_factor=10)
    lwr = 0.05 if args.tag == "_w48lwr" else 0.0
    print(f"fold {args.fold} {args.tag}: {len(R)} runs, {int(R.n_gaps.sum())} gaps, {batches_per_epoch} steps/epoch, "
          f"lr {args.lr}, epochs {args.epochs}, lwr {lwr}", flush=True)
    log = []
    for ep in range(args.epochs):
        t0, tot, cnt = time.time(), 0.0, 0
        batches = []
        for p, g in R.groupby("panel"):
            idx = rng.permutation(g.index.to_numpy())
            batches += [(p, idx[k:k + args.batch]) for k in range(0, len(idx), args.batch)]
        model.train()
        for bi in rng.permutation(len(batches)):
            p, idx = batches[bi]
            st = sts[p]
            xs, bs, ts, ws, es = [], [], [], [], []
            for j in idx:
                r = R.loc[j]
                d = U.Day(p, "train", int(r.day), srcs[p], st, profs[p])
                x, b, tr, w, ok = train_sample(d, parse_gaps(r.gaps), rng, np.asarray(occs[p][int(r.day)], np.float32),
                                               args.p_full)
                xs.append(x); bs.append(b); ts.append(tr); ws.append(w[None]); es.append(ok[None])
            X, Bb, TR = U.to_batch(xs).to(dev), U.to_batch(bs).to(dev), U.to_batch(ts).to(dev)
            S, EL = U.to_batch(ws)[:, 0].to(dev), U.to_batch(es)[:, 0].to(dev)
            n, Wd = len(st["ffs"]), X.shape[-1]
            stt = np.zeros((2, Wd), np.float32)
            stt[0, :n], stt[1, :n] = st["ffs"] / 25, st["cap"] / st["lanes"] / 600
            ST = torch.from_numpy(stt)[None, :, None, :].expand(len(idx), 2, 1, Wd).to(dev)
            phy = np.ones((3, Wd), np.float32)
            phy[0, :n], phy[1, :n], phy[2, :n] = st["ffs"], st["cap"], st["length"]
            PH = torch.from_numpy(phy)[None, :, None, :].expand(len(idx), 3, 1, Wd).to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                o = model(X)
            loss = U.loss_terms(o.float(), Bb, TR, S, ST)
            if lwr > 0:
                loss = loss + lwr * U.lwr_term(o.float(), Bb, TR, (S >= 0.99).float(), EL, PH)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += float(loss)
            cnt += 1
        log.append({"epoch": ep + 1, "loss": tot / cnt, "sec": round(time.time() - t0)})
        print(f"  epoch {ep + 1}/{args.epochs}: loss {tot / cnt:.5f} ({time.time() - t0:.0f}s)", flush=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out)
    print("saved", out, flush=True)


# ---------------------------------------------------------------------------
# held-out train months, official layout
# ---------------------------------------------------------------------------
def cmd_heldout(args) -> None:
    """Fold k nets on the train months with month % 2 == k (held out for fold k of both the production and the
    specialist): every replica gap blanked, history rows fully visible (official layout) or with their holes.
    Metrics on the eligible gap cells against the truth: speed RMSE, flow RMSE per lane, by panel and condition."""
    import torch
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"heldout{args.suffix}.csv"
    if path.exists():
        raise FileExistsError(path)
    R = pd.read_csv(CACHE / "train_runs.csv")
    rng = np.random.default_rng(5)
    rows = []
    for fold in (0, 1):
        nets = {}
        for tag in TAGS:
            nets["prod", tag] = load_prod(tag, fold)
            nets["spec", tag] = load_spec(tag, fold) if not args.suffix else _load_spec_sfx(tag, fold, args.suffix)
        Rf = R[R.month % 2 == fold]
        if args.max_runs:
            Rf = Rf.groupby("panel", group_keys=False).apply(lambda g: g.iloc[::max(1, len(g) // args.max_runs)])
        for p, g in Rf.groupby("panel"):
            src, st, prof = U.open_split(p, "train"), U.static(p), np.load(U.DATA / f"{p}_profiles.npy")
            occ = np.load(CACHE / f"occ_{p}_train.npy", mmap_mode="r")
            lanes = st["lanes"]
            for layout in ("official", "holes"):
                acc = {}
                for _, r in g.iterrows():
                    gaps = parse_gaps(r.gaps)
                    d = U.Day(p, "train", int(r.day), src, st, prof)
                    n = d.obs.shape[1]
                    tr = np.moveaxis(d.truth, -1, 0)
                    ok = np.isfinite(tr).all(0)
                    vis0 = d.obs[..., 3] > 0.5
                    hide = np.zeros((STEPS, n), bool)
                    unhide = np.zeros((STEPS, n), bool)
                    for T, _ in gaps:
                        hide[T + 1:T + 1 + GAP] = True
                        if layout == "official":
                            unhide[T - HIST:T] = ok[T - HIST:T] & ~vis0[T - HIST:T]
                    x2, b = make_input2(d, gaps, hide, unhide, np.asarray(occ[int(r.day)], np.float32))
                    x1 = x2[:U.N_IN]
                    out = {}
                    for (kind, tag), m in nets.items():
                        out[kind, tag] = predict_np(m, x1 if kind == "prod" else x2, b, st)
                    vt, qt = tr[0] * st["ffs"], tr[1] * st["cap"]
                    for T, c in gaps:
                        rr = slice(T + 1, T + 1 + GAP)
                        m_ = ok[rr]
                        for kind in ("prod", "spec"):
                            v = .5 * (out[kind, "_w48"][0] + out[kind, "_w48lwr"][0])[rr]
                            q = .5 * (out[kind, "_w48"][1] + out[kind, "_w48lwr"][1])[rr]
                            a = acc.setdefault((c, kind), [0, 0.0, 0.0])
                            a[0] += int(m_.sum())
                            a[1] += float(((v - vt[rr]) ** 2)[m_].sum())
                            a[2] += float((((q - qt[rr]) / lanes) ** 2)[m_].sum())
                for (c, kind), a in acc.items():
                    rows.append(dict(fold=fold, panel=p, layout=layout, cond=c, kind=kind, cells=a[0],
                                     rmse_v=math.sqrt(a[1] / a[0]), rmse_q_lane=math.sqrt(a[2] / a[0]),
                                     sse_v=a[1], sse_q=a[2]))
                print(fold, p, layout, {k: round(math.sqrt(a[1] / a[0]), 3) for k, a in acc.items()}, flush=True)
        del nets
        torch.cuda.empty_cache()
    D = pd.DataFrame(rows)
    D.to_csv(path, index=False)
    s = D.groupby(["layout", "cond", "kind"])[["cells", "sse_v", "sse_q"]].sum()
    s["rmse_v"], s["rmse_q_lane"] = np.sqrt(s.sse_v / s.cells), np.sqrt(s.sse_q / s.cells)
    print(s[["cells", "rmse_v", "rmse_q_lane"]].round(3).to_string(), flush=True)
    piv = D.pivot_table(index=["layout", "panel"], columns="kind", values=["sse_v", "sse_q"], aggfunc="sum")
    piv["mse_v_red"] = 1 - piv["sse_v"]["spec"] / piv["sse_v"]["prod"]
    piv["mse_q_red"] = 1 - piv["sse_q"]["spec"] / piv["sse_q"]["prod"]
    print(piv[["mse_v_red", "mse_q_red"]].round(3).to_string(), flush=True)


def _load_spec_sfx(tag, fold, sfx):
    import torch
    m = build_model2()
    m.load_state_dict(torch.load(MODELS / f"t1_unet_fold{fold}{tag}{OUT_SUFFIX}{sfx}.pt", map_location="cpu"))
    m.eval()
    return m.to("cuda")


# ---------------------------------------------------------------------------
# specialist outputs on the V57 ruler windows and on the official windows
# ---------------------------------------------------------------------------
def official_gaps():
    OW = pd.read_csv(V57 / "official_windows.csv")
    return {(r.panel, r.split, int(r.day)): [] for r in OW.itertuples()}, OW


def day_official(OW, p, sp, day):
    o = OW[(OW.panel == p) & (OW.split == sp) & (OW.day == day)]
    return [(int(T), c) for T, c in zip(o["T"], o.cond)]


def nets_spec(sfx=""):
    return {(tag, f): (_load_spec_sfx(tag, f, sfx) if sfx else load_spec(tag, f)) for tag in TAGS for f in (0, 1)}


def grid_outputs(nets, d, gaps, hide, st, inv):
    x, b = make_input2(d, gaps, hide)
    grid = {}
    for tag in TAGS:
        ps = [predict_np(nets[tag, f], x, b, st) for f in (0, 1)]
        for j, c in enumerate("vq"):
            grid[f"{c}{tag}"] = np.mean([z[j] for z in ps], axis=0)[:, inv]
            for f in (0, 1):
                grid[f"{c}{tag}f{f}"] = ps[f][j][:, inv]
    grid["pv"] = .5 * (grid["v_w48"] + grid["v_w48lwr"])
    grid["pq"] = .5 * (grid["q_w48"] + grid["q_w48lwr"])
    return grid


def cmd_wide(args) -> None:
    """Same runs and cells as t1_v57_blkruler.py wide (rows T-6..T+24 of every cell around each simulated origin);
    U-Net columns from the specialist (simulated gaps and the day's official blackouts flagged with their condition)."""
    import t1_v57_blkruler as RL
    _, OW = official_gaps()
    nets = nets_spec(args.suffix)
    out_dir = WIDE if not args.suffix else OUT / f"wide{args.suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    W = pd.read_csv(RULER / f"origins_{args.split}.csv")
    W = W[~W.blk_wide]
    for p in args.panels:
        path = out_dir / f"{p}_{args.split}_win.parquet"
        if path.exists():
            print("exists", path.name, flush=True)
            continue
        t0 = time.time()
        st = U.static(p)
        inv = np.argsort(st["perm"])
        src, prof = U.open_split(p, args.split), np.load(U.DATA / f"{p}_profiles.npy")
        n = len(st["perm"])
        recs = []
        for ir, (day, run) in enumerate(RL.pack_runs(W[W.panel == p])):
            d = U.Day(p, args.split, day, src, st, prof)
            vis = d.obs[..., 3] > 0.5
            rows = np.zeros(STEPS, bool)
            for r in run:
                rows[r["T"] + 1:r["T"] + 1 + GAP] = True
            hide = vis & rows[:, None]
            gaps = [(int(r["T"]), r["cond"]) for r in run] + day_official(OW, p, args.split, day)
            grid = grid_outputs(nets, d, gaps, hide, st, inv)
            for r in run:
                T = r["T"]
                ks = np.arange(-6, GAP + 7)
                tt = T + ks
                okr = (tt >= 0) & (tt < STEPS)
                ks, tt = ks[okr], tt[okr]
                ti, li = np.repeat(tt, n), np.tile(np.arange(n), len(tt))
                rec = {"day": np.full(len(ti), day, np.int16), "run": np.full(len(ti), ir, np.int32),
                       "T": np.full(len(ti), T, np.int16), "k": np.repeat(ks, n).astype(np.int8),
                       "pos": li.astype(np.int16)}
                for nm, a in grid.items():
                    rec[nm] = a[ti, li].astype(np.float32)
                recs.append(pd.DataFrame(rec))
        pd.concat(recs, ignore_index=True).to_parquet(path, index=False)
        print(p, args.split, "done", f"{time.time() - t0:.0f}s", flush=True)


def cmd_official(args) -> None:
    """Specialist outputs (both tags and folds, GPU fp32) on every official blackout day, rows T-6..T+24 of every
    cell around each official window, keys as t1_v57_blackout official/ (merged onto it in the apply step)."""
    _, OW = official_gaps()
    nets = nets_spec(args.suffix)
    out_dir = OFFICIAL if not args.suffix else OUT / f"official{args.suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    for p in T2_PANELS:
        for sp in SPLITS:
            path = out_dir / f"{p}_{sp}_win.parquet"
            if path.exists():
                print("exists", path.name, flush=True)
                continue
            st = U.static(p)
            inv = np.argsort(st["perm"])
            src, prof = U.open_split(p, sp), np.load(U.DATA / f"{p}_profiles.npy")
            n = len(st["perm"])
            ow = OW[(OW.panel == p) & (OW.split == sp)]
            recs = []
            for day, g in ow.groupby("day"):
                d = U.Day(p, sp, int(day), src, st, prof)
                gaps = [(int(T), c) for T, c in zip(g["T"], g.cond)]
                assert not (d.obs[..., 3] > 0.5)[np.concatenate([np.arange(T + 1, T + 1 + GAP) for T, _ in gaps])].any()
                grid = grid_outputs(nets, d, gaps, None, st, inv)
                for T, c in gaps:
                    ks = np.arange(-6, GAP + 7)
                    tt = T + ks
                    okr = (tt >= 0) & (tt < STEPS)
                    ks, tt = ks[okr], tt[okr]
                    ti, li = np.repeat(tt, n), np.tile(np.arange(n), len(tt))
                    rec = {"day": np.full(len(ti), day, np.int16), "T": np.full(len(ti), T, np.int16),
                           "k": np.repeat(ks, n).astype(np.int8), "pos": li.astype(np.int16)}
                    for nm, a in grid.items():
                        rec[nm] = a[ti, li].astype(np.float32)
                    recs.append(pd.DataFrame(rec))
            pd.concat(recs, ignore_index=True).to_parquet(path, index=False)
            print(p, sp, "official done", flush=True)


# ---------------------------------------------------------------------------
# ruler features and the pre-registered comparison with V57_blk
# ---------------------------------------------------------------------------
STATIC_COLS = ["day", "run", "regime", "T", "k", "t", "pos", "cond", "elig", "vis", "tgt", "hidden", "ramp_ok", "ov", "oq"]
EXTRA_FEATS = ["pv0_r", "pq0_l", "dsp_v_r", "dsp_q_l"]


def spec_long(p, sp, sfx=""):
    """sim_wide rows of (p, sp) with the U-Net columns replaced by the specialist's (same runs, same row order)."""
    base = pd.read_parquet(RULER / "sim_wide" / f"{p}_{sp}_win.parquet", columns=STATIC_COLS)
    sw = pd.read_parquet((WIDE if not sfx else OUT / f"wide{sfx}") / f"{p}_{sp}_win.parquet")
    assert len(base) == len(sw)
    for c in ("day", "run", "T", "k", "pos"):
        assert np.array_equal(base[c].to_numpy(), sw[c].to_numpy()), (p, sp, c)
    for c in sw.columns:
        if c not in base:
            base[c] = sw[c].to_numpy()
    return base


def spec_features(p, sp, sfx=""):
    import t1_v57_blackout as V
    fdir = OUT / f"features{sfx}"
    path = fdir / f"{p}_{sp}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    A, meta = V.win_tensors(p, spec_long(p, sp, sfx))
    F = V.gap_features(p, A, meta)
    fdir.mkdir(parents=True, exist_ok=True)
    F.to_parquet(path, index=False)
    return F


def slim_tensors(p, sp):
    """The parts of t1_v57_blackout.sim_tensors that window_sums reads (ov shape, vis, elig, hidden, ramp_ok)."""
    import t1_v57_blackout as V
    df = pd.read_parquet(RULER / "sim_wide" / f"{p}_{sp}_win.parquet",
                         columns=["day", "T", "k", "pos", "ov", "elig", "vis", "hidden", "ramp_ok"])
    n = len(links(p))
    df = df[(df.k >= V.K0) & (df.k <= V.K1)]
    w = df.day.astype(np.int64).to_numpy() * 1000 + df["T"].astype(np.int64).to_numpy()
    uw, wi = np.unique(w, return_inverse=True)
    ki = df.k.to_numpy(np.int64) - V.K0
    li = df.pos.to_numpy(np.int64)
    A = {"ov": np.full((len(uw), V.NK, n), np.nan, np.float32)}
    A["ov"][wi, ki, li] = df.ov.to_numpy(np.float32)
    for c in ("elig", "vis", "hidden", "ramp_ok"):
        a = np.zeros((len(uw), V.NK, n), bool)
        a[wi, ki, li] = df[c].to_numpy(bool)
        A[c] = a
    return A


def load_ruler(sfx=""):
    """Production features (V57 cache, row order of t1_v57_blackout.load_sim_all) + the specialist's, aligned."""
    import t1_v57_blackout as V
    P, S, TEN = [], [], {}
    for p in T2_PANELS:
        for sp in SPLITS:
            F0 = V.sim_features(p, sp)
            F1 = spec_features(p, sp, sfx)
            for c in ("w", "k", "pos", "ov", "oq"):
                assert np.array_equal(F0[c].to_numpy(), F1[c].to_numpy(), equal_nan=True), (p, sp, c)
            for F in (F0, F1):
                F["panel"], F["split"] = p, sp
            P.append(F0)
            S.append(F1)
            TEN[p, sp] = slim_tensors(p, sp)
    A0 = pd.concat(P, ignore_index=True)
    A1 = pd.concat(S, ignore_index=True)
    for A in (A0, A1):
        A["par"] = (A.day % 2).astype(np.int8)
        A["half"] = (A.day >= 15).astype(np.int8)
        A["sid"] = (A.split == "private").astype(np.int8)
    ffs, lanes = A0.ffs.to_numpy(np.float32), A0.lanes.to_numpy(np.float32)
    A1["pv0_r"] = A0.pv.to_numpy(np.float32) / ffs
    A1["pq0_l"] = A0.pq.to_numpy(np.float32) / lanes
    A1["dsp_v_r"] = (A1.pv.to_numpy(np.float32) - A0.pv.to_numpy(np.float32)) / ffs
    A1["dsp_q_l"] = (A1.pq.to_numpy(np.float32) - A0.pq.to_numpy(np.float32)) / lanes
    return A0, A1, TEN


def eval_fill(ALL0, TEN, v, q, FR, OW, base_ws):
    import t1_v57_blackout as V
    rows = []
    for (p, sp), A in TEN.items():
        m = ((ALL0.panel == p) & (ALL0.split == sp)).to_numpy()
        ws1 = V.window_sums(p, A, ALL0[m], v[m], q[m])
        tot, _ = V.deltas(V.real_frame(p, sp, base_ws[p, sp], FR, OW), V.real_frame(p, sp, ws1, FR, OW))
        rows.append({"panel": p, "split": sp, **tot})
    return pd.DataFrame(rows)


def gbm_cross(ALLx, feats, base_v, base_q, split_col, params, rounds):
    import lightgbm as lgb
    X = ALLx[feats].to_numpy(np.float32)
    lanes = ALLx.lanes.to_numpy(float)
    yv = ALLx.ov.to_numpy(float) - base_v
    yq = (ALLx.oq.to_numpy(float) - base_q) / lanes
    g = ALLx[split_col].to_numpy()
    pv_, pq_ = np.full(len(ALLx), np.nan), np.full(len(ALLx), np.nan)
    for a in (0, 1):
        fit, app = g == a, g != a
        mv = lgb.train(params, lgb.Dataset(X[fit], yv[fit], feature_name=feats, free_raw_data=True), rounds)
        pv_[app] = mv.predict(X[app], num_threads=4)
        mq = lgb.train(params, lgb.Dataset(X[fit], yq[fit], feature_name=feats, free_raw_data=True), rounds)
        pq_[app] = mq.predict(X[app], num_threads=4)
    del X
    return pv_, pq_


def mk_fill(base_v, base_q, lanes, rv, rq, lv, lq):
    import t1_v57_blackout as V
    return (np.clip(base_v + lv * rv, V.SPEED_MIN, V.SPEED_MAX), np.maximum(base_q + lq * rq * lanes, 0.0))


def versus(R: pd.DataFrame, Rb: pd.DataFrame) -> dict:
    """Candidate R vs V57_blk Rb, both dTotal vs the production fill per corridor-month."""
    import t1_v57_blackout as V
    m = R.merge(Rb, on=["panel", "split"], suffixes=("", "_b"))
    out = {}
    for sp in SPLITS:
        g = m[m.split == sp]
        out[f"dTotal_{sp}"] = V.family_mean(dict(zip(g.panel, g.dTotal)))
        out[f"dS_{sp}"] = V.family_mean(dict(zip(g.panel, g.dS)))
        out[f"dphys_{sp}"] = V.family_mean(dict(zip(g.panel, g.dphys)))
        out[f"gain_vs_v57_{sp}"] = out[f"dTotal_{sp}"] - V.family_mean(dict(zip(g.panel, g.dTotal_b)))
        out[f"n_beat_v57_{sp}"] = int((g.dTotal > g.dTotal_b).sum())
        out[f"n_pos_{sp}"] = int((g.dTotal > 0).sum())
    out["gain_vs_v57_both"] = (out["gain_vs_v57_validation"] + out["gain_vs_v57_private"]) / 2
    out["dTotal_both"] = (out["dTotal_validation"] + out["dTotal_private"]) / 2
    out["wins"] = bool(all(out[f"gain_vs_v57_{s}"] > 0 and out[f"n_beat_v57_{s}"] >= 7 for s in SPLITS))
    return out


def cmd_crossfit(args) -> None:
    """PRE-REGISTERED (written before the first ruler run of any specialist; see README):
      ruler      V57 simulated blackouts on the March / April visible layers (results/t1_v57_blkruler, 977 windows),
                 real-frame emulation of t1_v57_blackout (official blackout target counts, condition mix, mask rates),
                 dTotal = 0.35 dS_state + 0.15 dS_phys(est) against the production fill, family mean (D12_I405_* = 0)
      reference  V57_blk = production + (0.75, 0.5) x the V57 residual GBM, its day-parity cross-fit predictions
                 (results/t1_v57_blackout/crossfit_pred_parity.npz)
      candidates fixed set:
        S0   specialist fill alone (mean of 2 tags x 2 folds; the specialist is fitted on train months only, so it is
             out-of-sample on March / April)
        S1   S0 + V57-type residual GBM (t1_v57_blackout FEATS / GBM_PARAMS / 400 rounds, features computed from the
             specialist outputs), day-parity cross-fit on both months and all corridors
        S2   as S1 with 4 extra features: production fill (pv0_r, pq0_l) and specialist - production (dsp_v_r, dsp_q_l)
        for S1 / S2 the lambda pair is chosen by the V57 rule on their own grid (LAM_V x LAM_Q; eligible: family
        dTotal > 0 and dS > 0 in both months and dTotal > 0 in >= 7/8 corridors; smallest lambda_v, then lambda_q,
        within 95% of the best mean-over-months dTotal)
      win        a candidate wins iff in BOTH months its family dTotal > V57_blk's and it beats V57_blk in >= 7/8
                 corridors; among the winners the one with the largest mean-over-months dTotal is chosen
    Reported only: block-halves cross-fit (days < 15 vs >= 15) and nested month fits of the chosen GBM candidate,
    bootstrap of 5 + 5 windows per corridor-month against V57_blk, MSE by condition."""
    import t1_v57_blackout as V
    OUT.mkdir(parents=True, exist_ok=True)
    tag = args.suffix
    files = [OUT / f"crossfit{tag}_{x}" for x in ("summary.csv", "corridors.csv", "choice.json")]
    for f in files:
        if f.exists():
            raise FileExistsError(f)
    t0 = time.time()
    FR, OW = V.load_frame()
    ALL0, ALL1, TEN = load_ruler(args.suffix)
    lanes = ALL0.lanes.to_numpy(float)
    print("ruler cells", len(ALL0), f"{time.time() - t0:.0f}s", flush=True)
    base_ws = {}
    for (p, sp), A in TEN.items():
        m = ((ALL0.panel == p) & (ALL0.split == sp)).to_numpy()
        base_ws[p, sp] = V.window_sums(p, A, ALL0[m], ALL0.pv.to_numpy(float)[m], ALL0.pq.to_numpy(float)[m])
    z = np.load(V57 / "crossfit_pred_parity.npz")
    v57 = mk_fill(ALL0.pv.to_numpy(float), ALL0.pq.to_numpy(float), lanes, z["pv"], z["pq"], 0.75, 0.5)
    Rb = eval_fill(ALL0, TEN, *v57, FR, OW, base_ws)
    Rb["cfg"] = "V57_blk"
    print("V57_blk", {k: round(v, 6) for k, v in versus(Rb, Rb).items() if "dTotal" in k or "dS" in k}, flush=True)
    sv, sq = ALL1.pv.to_numpy(float), ALL1.pq.to_numpy(float)
    rows, cm = [], [Rb]
    R0 = eval_fill(ALL0, TEN, sv, sq, FR, OW, base_ws)
    R0["cfg"] = "S0"
    cm.append(R0)
    rows.append({"cand": "S0", "cfg": "S0", "lam_v": 0.0, "lam_q": 0.0, **versus(R0, Rb)})
    print("S0", rows[-1], f"{time.time() - t0:.0f}s", flush=True)
    preds = {}
    for cand, feats in (("S1", V.FEATS), ("S2", V.FEATS + EXTRA_FEATS)):
        rv, rq = gbm_cross(ALL1, feats, sv, sq, "par", V.GBM_PARAMS, V.GBM_ROUNDS)
        preds[cand] = (rv, rq)
        print(cand, "cross-fit", f"{time.time() - t0:.0f}s", flush=True)
        for lv in V.LAM_V:
            for lq in V.LAM_Q:
                R = eval_fill(ALL0, TEN, *mk_fill(sv, sq, lanes, rv, rq, lv, lq), FR, OW, base_ws)
                R["cfg"] = f"{cand}_v{lv}_q{lq}"
                cm.append(R)
                rows.append({"cand": cand, "cfg": R.cfg.iloc[0], "lam_v": lv, "lam_q": lq, **versus(R, Rb),
                             "elig_v57rule": bool(all(
                                 V.family_mean(dict(zip(R[R.split == s].panel, R[R.split == s].dTotal))) > 0 and
                                 V.family_mean(dict(zip(R[R.split == s].panel, R[R.split == s].dS))) > 0 and
                                 int((R[R.split == s].dTotal > 0).sum()) >= 7 for s in SPLITS))})
    G_ = pd.DataFrame(rows)
    picks = {"S0": "S0"}
    for cand in ("S1", "S2"):
        el = G_[(G_.cand == cand) & G_.elig_v57rule]
        if el.empty:
            continue
        best = el.dTotal_both.max()
        near = el[el.dTotal_both >= 0.95 * best].sort_values(["lam_v", "lam_q"])
        picks[cand] = near.cfg.iloc[0]
    P = G_.set_index("cfg").loc[list(picks.values())]
    winners = P[P.wins]
    chosen = None if winners.empty else winners.dTotal_both.idxmax()
    print(G_.round(6).to_string(), flush=True)
    print("picks", picks, "chosen", chosen, flush=True)
    rob = {}
    D = pd.DataFrame({"split": ALL0.split, "cond": ALL0.condname, "panel": ALL0.panel,
                      "e_prod": (ALL0.pv - ALL0.ov) ** 2, "e_v57": (v57[0] - ALL0.ov) ** 2,
                      "f_prod": ((ALL0.pq - ALL0.oq) / ALL0.lanes) ** 2, "f_v57": ((v57[1] - ALL0.oq) / ALL0.lanes) ** 2})
    if chosen is not None:
        cand = P.loc[chosen, "cand"]
        lv, lq = float(P.loc[chosen, "lam_v"]), float(P.loc[chosen, "lam_q"])
        if cand == "S0":
            vc, qc = sv, sq
        else:
            vc, qc = mk_fill(sv, sq, lanes, *preds[cand], lv, lq)
            feats = V.FEATS + (EXTRA_FEATS if cand == "S2" else [])
            hv, hq = gbm_cross(ALL1, feats, sv, sq, "half", V.GBM_PARAMS, V.GBM_ROUNDS)
            zb = V.cross_predict(ALL0.assign(), "half")
            v57h = mk_fill(ALL0.pv.to_numpy(float), ALL0.pq.to_numpy(float), lanes, *zb, 0.75, 0.5)
            rob["block_halves"] = versus(eval_fill(ALL0, TEN, *mk_fill(sv, sq, lanes, hv, hq, lv, lq), FR, OW, base_ws),
                                         eval_fill(ALL0, TEN, *v57h, FR, OW, base_ws))
            print("block halves", rob["block_halves"], f"{time.time() - t0:.0f}s", flush=True)
        D["e_c"], D["f_c"] = (vc - ALL0.ov) ** 2, ((qc - ALL0.oq) / ALL0.lanes) ** 2
        ws_c, ws_b = {}, {}
        for (p, sp), A in TEN.items():
            m = ((ALL0.panel == p) & (ALL0.split == sp)).to_numpy()
            ws_c[p, sp] = V.window_sums(p, A, ALL0[m], vc[m], qc[m])
            ws_b[p, sp] = V.window_sums(p, A, ALL0[m], v57[0][m], v57[1][m])
        rob["bootstrap_vs_v57"] = V.bootstrap(ALL0, TEN, ws_b, ws_c, FR, OW)
        np.savez_compressed(OUT / f"crossfit{tag}_chosen_fill.npz", v=vc, q=qc)
    by = D.groupby(["split", "cond"]).agg(**{c: (c, "mean") for c in D.columns if c[:2] in ("e_", "f_")})
    G_.to_csv(files[0], index=False)
    pd.concat(cm, ignore_index=True).to_csv(files[1], index=False)
    write_new(files[2], json.dumps({"rule": cmd_crossfit.__doc__, "picks": picks, "chosen": chosen,
                                    "chosen_row": None if chosen is None else P.loc[chosen].to_dict(),
                                    "dS_family": None if chosen is None else
                                    {s: float(P.loc[chosen, f"dS_{s}"]) for s in SPLITS},
                                    "candidates": P.reset_index().to_dict("records"), "robustness": rob,
                                    "mse_by_split_cond": by.reset_index().to_dict("records"),
                                    "seconds": round(time.time() - t0)}, indent=2, default=float))
    pd.set_option("display.width", 250)
    print(P.round(6).to_string(), flush=True)
    print(by.round(3).to_string(), flush=True)
    print(json.dumps(rob, indent=2, default=float), flush=True)


# ---------------------------------------------------------------------------
# phase B: month self-supervision at the simulated origins (day-parity cross-fitted)
# ---------------------------------------------------------------------------
SS_SUFFIX = "_v59ss"
SS_RAND = 0.05


def ruler_runs(splits=SPLITS):
    """The V57 ruler runs (t1_v57_blkruler.pack_runs on the usable origins) as rows (panel, split, day, gaps)."""
    import t1_v57_blkruler as RL
    rows = []
    for sp in splits:
        W = pd.read_csv(RULER / f"origins_{sp}.csv")
        W = W[~W.blk_wide]
        for p in T2_PANELS:
            for ir, (day, run) in enumerate(RL.pack_runs(W[W.panel == p])):
                rows.append(dict(panel=p, split=sp, day=int(day), run=ir,
                                 gaps=";".join(f"{int(r['T'])}:{r['cond']}" for r in run)))
    return pd.DataFrame(rows)


def ss_sample(d, gaps, off_gaps, rng):
    """March/April run: visible cells of the simulated gap rows hidden and supervised by their observation (weight 1),
    SS_RAND of the other visible cells hidden with weight W_OTHER; official blackouts of the day flagged."""
    n = d.obs.shape[1]
    vis = d.obs[..., 3] > 0.5
    rows = np.zeros(STEPS, bool)
    for T, _ in gaps:
        rows[T + 1:T + 1 + GAP] = True
    hide_g = vis & rows[:, None]
    hide_r = vis & ~rows[:, None] & (rng.random((STEPS, n)) < SS_RAND)
    x, b = make_input2(d, list(gaps) + list(off_gaps), hide_g | hide_r)
    lab = np.stack([np.where(vis, d.obs[..., 0], 0), np.where(vis, d.obs[..., 1], 0)])
    w = np.where(hide_g, 1.0, 0.0) + np.where(hide_r, W_OTHER, 0.0)
    return x, b, lab, w.astype(np.float32), vis.astype(np.float32)


def cmd_train_ss(args) -> None:
    """Start from the specialist fold net; per epoch every simulated-origin run of the March/April days with
    day % 2 == parity (all days if parity < 0) once, plus as many randomly drawn train-month runs of the fold's own
    months (truth labels, recipe of cmd_train); lr OneCycle, lwr 0.05 on the train batches of the lwr tag."""
    import torch
    sfx = f"{SS_SUFFIX}_p{args.parity}" if args.parity >= 0 else f"{SS_SUFFIX}_all"
    out = MODELS / f"t1_unet_fold{args.fold}{args.tag}{sfx}.pt"
    if out.exists():
        raise FileExistsError(out)
    seed = SEED + 1000 + 10 * args.fold + (0 if args.tag == "_w48" else 1) + 100 * (args.parity + 1)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    rng = np.random.default_rng(seed)
    RR = ruler_runs()
    if args.parity >= 0:
        RR = RR[RR.day % 2 == args.parity]
    RR = RR.reset_index(drop=True)
    RT = pd.read_csv(CACHE / "train_runs.csv")
    RT = RT[RT.month % 2 != args.fold].reset_index(drop=True)
    _, OW = official_gaps()
    srcs = {(p, s): U.open_split(p, s) for p in T2_PANELS for s in ("train",) + SPLITS}
    sts = {p: U.static(p) for p in T2_PANELS}
    profs = {p: np.load(U.DATA / f"{p}_profiles.npy") for p in T2_PANELS}
    occs = {p: np.load(CACHE / f"occ_{p}_train.npy", mmap_mode="r") for p in T2_PANELS}
    dev = "cuda"
    model = load_spec(args.tag, args.fold, dev)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    lwr = 0.05 if args.tag == "_w48lwr" else 0.0

    def make_batches():
        items = [("ss", j) for j in range(len(RR))]
        items += [("tr", int(j)) for j in rng.choice(len(RT), size=len(RR), replace=len(RR) > len(RT))]
        groups = {}
        for kind, j in items:
            p = RR.panel[j] if kind == "ss" else RT.panel[j]
            groups.setdefault((kind, p), []).append(j)
        out_ = []
        for (kind, p), js in groups.items():
            js = list(rng.permutation(js))
            out_ += [(kind, p, js[k:k + args.batch]) for k in range(0, len(js), args.batch)]
        return [out_[i] for i in rng.permutation(len(out_))]

    n_steps = len(make_batches()) * args.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=n_steps + 5, pct_start=0.1,
                                                anneal_strategy="cos", div_factor=10, final_div_factor=10)
    print(f"ss fold {args.fold} {args.tag} parity {args.parity}: {len(RR)} ruler runs + as many train runs, "
          f"~{n_steps // args.epochs} steps/epoch", flush=True)
    for ep in range(args.epochs):
        t0, tot, cnt = time.time(), {"ss": 0.0, "tr": 0.0}, {"ss": 0, "tr": 0}
        for kind, p, js in make_batches():
            st = sts[p]
            xs, bs, ts, ws, es = [], [], [], [], []
            for j in js:
                if kind == "ss":
                    r = RR.loc[j]
                    d = U.Day(p, r.split, int(r.day), srcs[p, r.split], st, profs[p])
                    x, b, tr, w, ok = ss_sample(d, parse_gaps(r.gaps), day_official(OW, p, r.split, int(r.day)), rng)
                else:
                    r = RT.loc[j]
                    d = U.Day(p, "train", int(r.day), srcs[p, "train"], st, profs[p])
                    x, b, tr, w, ok = train_sample(d, parse_gaps(r.gaps), rng,
                                                   np.asarray(occs[p][int(r.day)], np.float32), P_FULL)
                xs.append(x); bs.append(b); ts.append(tr); ws.append(w[None]); es.append(ok[None])
            X, Bb, TR = U.to_batch(xs).to(dev), U.to_batch(bs).to(dev), U.to_batch(ts).to(dev)
            S, EL = U.to_batch(ws)[:, 0].to(dev), U.to_batch(es)[:, 0].to(dev)
            n, Wd = len(st["ffs"]), X.shape[-1]
            stt = np.zeros((2, Wd), np.float32)
            stt[0, :n], stt[1, :n] = st["ffs"] / 25, st["cap"] / st["lanes"] / 600
            ST = torch.from_numpy(stt)[None, :, None, :].expand(len(js), 2, 1, Wd).to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                o = model(X)
            loss = U.loss_terms(o.float(), Bb, TR, S, ST)
            if lwr > 0:
                phy = np.ones((3, Wd), np.float32)
                phy[0, :n], phy[1, :n], phy[2, :n] = st["ffs"], st["cap"], st["length"]
                PH = torch.from_numpy(phy)[None, :, None, :].expand(len(js), 3, 1, Wd).to(dev)
                loss = loss + lwr * U.lwr_term(o.float(), Bb, TR, (S >= 0.99).float(), EL, PH)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot[kind] += float(loss)
            cnt[kind] += 1
        print(f"  epoch {ep + 1}/{args.epochs}: loss ss {tot['ss'] / max(cnt['ss'], 1):.5f} "
              f"train {tot['tr'] / max(cnt['tr'], 1):.5f} ({time.time() - t0:.0f}s)", flush=True)
    torch.save(model.state_dict(), out)
    print("saved", out, flush=True)


def cmd_wide_ss(args) -> None:
    """Ruler outputs of the month-self-supervised nets: a run on a day with parity b uses the nets fitted on the
    days with parity 1 - b (cross-fit); same rows as cmd_wide -> results/t1_v59_blkspec/wide_ss/."""
    import torch
    _, OW = official_gaps()
    nets = {}
    for par in (0, 1):
        for tag in TAGS:
            for f in (0, 1):
                m = build_model2()
                m.load_state_dict(torch.load(MODELS / f"t1_unet_fold{f}{tag}{SS_SUFFIX}_p{par}.pt", map_location="cpu"))
                m.eval()
                nets[par, tag, f] = m.to("cuda")
    out_dir = OUT / "wide_ss"
    out_dir.mkdir(parents=True, exist_ok=True)
    RR = ruler_runs((args.split,))
    for p in T2_PANELS:
        path = out_dir / f"{p}_{args.split}_win.parquet"
        if path.exists():
            print("exists", path.name, flush=True)
            continue
        st = U.static(p)
        inv = np.argsort(st["perm"])
        src, prof = U.open_split(p, args.split), np.load(U.DATA / f"{p}_profiles.npy")
        n = len(st["perm"])
        recs = []
        for _, r in RR[RR.panel == p].iterrows():
            day = int(r.day)
            d = U.Day(p, args.split, day, src, st, prof)
            vis = d.obs[..., 3] > 0.5
            gaps0 = parse_gaps(r.gaps)
            rows = np.zeros(STEPS, bool)
            for T, _ in gaps0:
                rows[T + 1:T + 1 + GAP] = True
            par = 1 - day % 2
            sub = {(tag, f): nets[par, tag, f] for tag in TAGS for f in (0, 1)}
            grid = grid_outputs(sub, d, gaps0 + day_official(OW, p, args.split, day), vis & rows[:, None], st, inv)
            for T, _ in gaps0:
                ks = np.arange(-6, GAP + 7)
                tt = T + ks
                okr = (tt >= 0) & (tt < STEPS)
                ks, tt = ks[okr], tt[okr]
                ti, li = np.repeat(tt, n), np.tile(np.arange(n), len(tt))
                rec = {"day": np.full(len(ti), day, np.int16), "run": np.full(len(ti), int(r.run), np.int32),
                       "T": np.full(len(ti), T, np.int16), "k": np.repeat(ks, n).astype(np.int8),
                       "pos": li.astype(np.int16)}
                for nm, a in grid.items():
                    rec[nm] = a[ti, li].astype(np.float32)
                recs.append(pd.DataFrame(rec))
        pd.concat(recs, ignore_index=True).to_parquet(path, index=False)
        print(p, args.split, "wide_ss done", flush=True)


def cmd_crossfit_ss(args) -> None:
    """PRE-REGISTERED for phase B (written before any phase-B ruler number was seen):
      reference  the phase-A choice (crossfit_choice.json, S2_v0.625_q0.5: specialist + residual GBM), its
                 cross-fitted ruler fill (crossfit_chosen_fill.npz); V57_blk reported too
      candidates S3  month-self-supervised specialist alone (nets fitted on the other day parity)
                 S4  S3 + residual GBM with the S2 feature set (FEATS + production fill + S3-minus-production),
                     day-parity cross-fit, lambda pair by the V57 rule on its own grid
      win        family dTotal > the reference in BOTH months and better than the reference in >= 7/8 corridors
                 per month; among winners the larger mean-over-months dTotal"""
    import t1_v57_blackout as V
    files = [OUT / f"crossfit_ss_{x}" for x in ("summary.csv", "corridors.csv", "choice.json")]
    for f in files:
        if f.exists():
            raise FileExistsError(f)
    t0 = time.time()
    FR, OW = V.load_frame()
    ALL0, ALL3, TEN = load_ruler("_ss")
    lanes = ALL0.lanes.to_numpy(float)
    base_ws = {}
    for (p, sp), A in TEN.items():
        m = ((ALL0.panel == p) & (ALL0.split == sp)).to_numpy()
        base_ws[p, sp] = V.window_sums(p, A, ALL0[m], ALL0.pv.to_numpy(float)[m], ALL0.pq.to_numpy(float)[m])
    z = np.load(V57 / "crossfit_pred_parity.npz")
    Rv57 = eval_fill(ALL0, TEN, *mk_fill(ALL0.pv.to_numpy(float), ALL0.pq.to_numpy(float), lanes, z["pv"], z["pq"],
                                          0.75, 0.5), FR, OW, base_ws)
    za = np.load(OUT / "crossfit_chosen_fill.npz")
    Rref = eval_fill(ALL0, TEN, za["v"], za["q"], FR, OW, base_ws)
    print("reference (phase A)", versus(Rref, Rv57), flush=True)
    sv, sq = ALL3.pv.to_numpy(float), ALL3.pq.to_numpy(float)
    rows, cm = [], []
    R3 = eval_fill(ALL0, TEN, sv, sq, FR, OW, base_ws)
    R3["cfg"] = "S3"
    cm.append(R3)
    rows.append({"cand": "S3", "cfg": "S3", "lam_v": 0.0, "lam_q": 0.0, **versus(R3, Rref),
                 "gain_vs_v57_true_both": versus(R3, Rv57)["gain_vs_v57_both"]})
    print("S3", rows[-1], flush=True)
    feats = V.FEATS + EXTRA_FEATS
    rv, rq = gbm_cross(ALL3, feats, sv, sq, "par", V.GBM_PARAMS, V.GBM_ROUNDS)
    print("S4 cross-fit", f"{time.time() - t0:.0f}s", flush=True)
    for lv in V.LAM_V:
        for lq in V.LAM_Q:
            R = eval_fill(ALL0, TEN, *mk_fill(sv, sq, lanes, rv, rq, lv, lq), FR, OW, base_ws)
            R["cfg"] = f"S4_v{lv}_q{lq}"
            cm.append(R)
            rows.append({"cand": "S4", "cfg": R.cfg.iloc[0], "lam_v": lv, "lam_q": lq, **versus(R, Rref),
                         "gain_vs_v57_true_both": versus(R, Rv57)["gain_vs_v57_both"],
                         "elig_v57rule": bool(all(
                             V.family_mean(dict(zip(R[R.split == s].panel, R[R.split == s].dTotal))) > 0 and
                             V.family_mean(dict(zip(R[R.split == s].panel, R[R.split == s].dS))) > 0 and
                             int((R[R.split == s].dTotal > 0).sum()) >= 7 for s in SPLITS))})
    G_ = pd.DataFrame(rows)
    picks = {"S3": "S3"}
    el = G_[(G_.cand == "S4") & (G_.elig_v57rule == True)]  # noqa: E712
    if not el.empty:
        best = el.dTotal_both.max()
        picks["S4"] = el[el.dTotal_both >= 0.95 * best].sort_values(["lam_v", "lam_q"]).cfg.iloc[0]
    P = G_.set_index("cfg").loc[list(picks.values())]
    winners = P[P.wins]
    chosen = None if winners.empty else winners.dTotal_both.idxmax()
    pd.set_option("display.width", 250)
    print(G_.round(6).to_string(), flush=True)
    print("picks", picks, "chosen", chosen, flush=True)
    rob = {}
    if chosen is not None:
        lv, lq = float(P.loc[chosen, "lam_v"]), float(P.loc[chosen, "lam_q"])
        if P.loc[chosen, "cand"] == "S3":
            vc, qc = sv, sq
        else:
            vc, qc = mk_fill(sv, sq, lanes, rv, rq, lv, lq)
            hv, hq = gbm_cross(ALL3, feats, sv, sq, "half", V.GBM_PARAMS, V.GBM_ROUNDS)
            rob["block_halves_vs_phaseA_parity_fill"] = versus(
                eval_fill(ALL0, TEN, *mk_fill(sv, sq, lanes, hv, hq, lv, lq), FR, OW, base_ws), Rref)
        ws_c, ws_r = {}, {}
        for (p, sp), A in TEN.items():
            m = ((ALL0.panel == p) & (ALL0.split == sp)).to_numpy()
            ws_c[p, sp] = V.window_sums(p, A, ALL0[m], vc[m], qc[m])
            ws_r[p, sp] = V.window_sums(p, A, ALL0[m], za["v"][m], za["q"][m])
        rob["bootstrap_vs_phaseA"] = V.bootstrap(ALL0, TEN, ws_r, ws_c, FR, OW)
        rob["vs_v57"] = versus(eval_fill(ALL0, TEN, vc, qc, FR, OW, base_ws), Rv57)
        np.savez_compressed(OUT / "crossfit_ss_chosen_fill.npz", v=vc, q=qc)
        D = pd.DataFrame({"split": ALL0.split, "cond": ALL0.condname, "e_prod": (ALL0.pv - ALL0.ov) ** 2,
                          "e_ref": (za["v"] - ALL0.ov) ** 2, "e_c": (vc - ALL0.ov) ** 2,
                          "f_prod": ((ALL0.pq - ALL0.oq) / ALL0.lanes) ** 2,
                          "f_ref": ((za["q"] - ALL0.oq) / ALL0.lanes) ** 2, "f_c": ((qc - ALL0.oq) / ALL0.lanes) ** 2})
        rob["mse_by_split_cond"] = D.groupby(["split", "cond"]).mean(numeric_only=True).reset_index().to_dict("records")
    G_.to_csv(files[0], index=False)
    pd.concat(cm, ignore_index=True).to_csv(files[1], index=False)
    write_new(files[2], json.dumps({"rule": cmd_crossfit_ss.__doc__, "picks": picks, "chosen": chosen,
                                    "chosen_row": None if chosen is None else P.loc[chosen].to_dict(),
                                    "dS_family": None if chosen is None else
                                    {s: float(P.loc[chosen, f"dS_{s}"]) for s in SPLITS},
                                    "reference_vs_v57": versus(Rref, Rv57),
                                    "candidates": P.reset_index().to_dict("records"), "robustness": rob,
                                    "seconds": round(time.time() - t0)}, indent=2, default=float))
    print(P.round(6).to_string(), flush=True)
    print(json.dumps(rob, indent=2, default=float), flush=True)


# ---------------------------------------------------------------------------
# final GBM, official windows, apply, package, audit, compare (only after crossfit chose a winner)
# ---------------------------------------------------------------------------
V58_ZIP = WORK / "submissions" / "v58_rs_blk.zip"
V51_ZIP = WORK / "submissions" / "v51_april.zip"
NAME = "v59blk_state"
FULL_NAME = "v59_blk"
CSV = {s: WORK / "sub" / f"state_{s}_v59blk.csv" for s in SPLITS}
GBM_FILES = {c: MODELS / f"t1_blk_v59_{c}.txt" for c in "vq"}


def load_choice():
    ch = json.loads((OUT / "crossfit_choice.json").read_text())
    if ch["chosen"] is None:
        raise SystemExit("crossfit chose no candidate: nothing to apply")
    row = ch["chosen_row"]
    return ch["chosen"], row["cand"], float(row["lam_v"]), float(row["lam_q"])


def cand_feats(cand):
    import t1_v57_blackout as V
    return V.FEATS + (EXTRA_FEATS if cand == "S2" else [])


def cmd_fit(args) -> None:
    import lightgbm as lgb
    import t1_v57_blackout as V
    name, cand, lv, lq = load_choice()
    if cand == "S0":
        print("S0 needs no GBM")
        return
    for pth in GBM_FILES.values():
        if pth.exists():
            raise FileExistsError(pth)
    _, ALL1, _ = load_ruler()
    feats = cand_feats(cand)
    X = ALL1[feats].to_numpy(np.float32)
    lanes = ALL1.lanes.to_numpy(float)
    ys = {"v": ALL1.ov.to_numpy(float) - ALL1.pv.to_numpy(float),
          "q": (ALL1.oq.to_numpy(float) - ALL1.pq.to_numpy(float)) / lanes}
    for c in "vq":
        lgb.train(V.GBM_PARAMS, lgb.Dataset(X, ys[c], feature_name=feats), V.GBM_ROUNDS).save_model(str(GBM_FILES[c]))
        print("saved", GBM_FILES[c], flush=True)


def official_long(p, sp):
    """V57 official rows (static columns + production CPU U-Net) and the same rows with the specialist columns."""
    D0 = pd.read_parquet(V57 / "official" / f"{p}_{sp}_win.parquet")
    S_ = pd.read_parquet(OFFICIAL / f"{p}_{sp}_win.parquet")
    assert len(D0) == len(S_)
    for c in ("day", "T", "k", "pos"):
        assert np.array_equal(D0[c].to_numpy(), S_[c].to_numpy()), (p, sp, c)
    D1 = D0.copy()
    for c in S_.columns:
        if c not in ("day", "T", "k", "pos"):
            D1[c] = S_[c].to_numpy()
    return D0, D1


def official_fill(cand, lv, lq):
    import lightgbm as lgb
    import t1_v57_blackout as V
    from evaluate import KEY_BASE
    boosters = None if cand == "S0" else {c: lgb.Booster(model_file=str(GBM_FILES[c])) for c in "vq"}
    out = {}
    for p in T2_PANELS:
        for sp in SPLITS:
            D0, D1 = official_long(p, sp)
            A0, m0 = V.win_tensors(p, D0)
            A1, m1 = V.win_tensors(p, D1)
            F0, F = V.gap_features(p, A0, m0), V.gap_features(p, A1, m1)
            for c in ("w", "k", "pos"):
                assert np.array_equal(F0[c].to_numpy(), F[c].to_numpy())
            ffs, lanes = F.ffs.to_numpy(np.float32), F.lanes.to_numpy(np.float32)
            F["pv0_r"] = F0.pv.to_numpy(np.float32) / ffs
            F["pq0_l"] = F0.pq.to_numpy(np.float32) / lanes
            F["dsp_v_r"] = (F.pv.to_numpy(np.float32) - F0.pv.to_numpy(np.float32)) / ffs
            F["dsp_q_l"] = (F.pq.to_numpy(np.float32) - F0.pq.to_numpy(np.float32)) / lanes
            if boosters is None:
                rv = rq = np.zeros(len(F))
            else:
                X = F[cand_feats(cand)].to_numpy(np.float32)
                rv, rq = boosters["v"].predict(X, num_threads=4), boosters["q"].predict(X, num_threads=4)
            v, q = mk_fill(F.pv.to_numpy(float), F.pq.to_numpy(float), lanes.astype(float), rv, rq, lv, lq)
            pf = pq.ParquetFile(CLEAN / p / f"{sp}.parquet")
            slot0 = {}
            for d in F.day.unique():
                ts = pf.read_row_group(int(d), columns=["ts"]).column(0).to_numpy()
                slot0[d] = int(ts.astype("datetime64[m]").astype(np.int64)[0] // 5)
            t_abs = F.day.map(slot0).to_numpy(np.int64) + F["T"].to_numpy(np.int64) + F.k.to_numpy(np.int64)
            F["key"] = t_abs * KEY_BASE + F.pos.to_numpy(np.int64)
            F["v_new"], F["q_new"], F["pv_prod_cpu"], F["pq_prod_cpu"] = v, q, F0.pv.to_numpy(), F0.pq.to_numpy()
            out[sp, p] = F
    return out


def g6(x: np.ndarray) -> np.ndarray:
    return np.array([float("%.6g" % v) for v in x])


def cmd_apply(args) -> None:
    import t1_assemble as TA
    import t1_v52_monthbias as M
    import zipfile
    from evaluate import t1_targets
    name, cand, lv, lq = load_choice()
    for pth in CSV.values():
        if pth.exists():
            raise FileExistsError(pth)
    for f in ("apply_changes.csv", "apply_summary.json", "apply_cells.parquet"):
        if (OUT / f).exists():
            raise FileExistsError(OUT / f)
    t0 = time.time()
    fills = official_fill(cand, lv, lq)
    K = M.read_key_state()
    with zipfile.ZipFile(V58_ZIP) as z, z.open(z.namelist()[0]) as f:
        Z = pd.read_csv(f, float_precision="round_trip", usecols=["submission_id", "task", "speed_kmh", "flow_vph"],
                        dtype={"submission_id": np.int64, "task": "category", "speed_kmh": str, "flow_vph": str})
    Z = Z[Z.task == "state"]
    assert np.array_equal(Z.submission_id.to_numpy(), K.submission_id.to_numpy())
    sv_txt, sq_txt = Z.speed_kmh.to_numpy(), Z.flow_vph.to_numpy()
    K["v58"] = pd.to_numeric(Z.speed_kmh, errors="raise").to_numpy(float)
    K["q58"] = pd.to_numeric(Z.flow_vph, errors="raise").to_numpy(float)
    n_not6g = int(sum(("%.6g" % a) != b for a, b in zip(K.v58.to_numpy(), sv_txt))) + \
        int(sum(("%.6g" % a) != b for a, b in zip(K.q58.to_numpy(), sq_txt)))
    del Z, sv_txt, sq_txt
    assert n_not6g == 0, f"{n_not6g} V58 state values are not %.6g text"
    print("key + V58 state rows", len(K), f"{time.time() - t0:.0f}s", flush=True)
    rows, cells = [], []
    for split in SPLITS:
        S_ = K[K.split == split].reset_index(drop=True)
        v_all, q_all = S_.v58.to_numpy().copy(), S_.q58.to_numpy().copy()
        for p, idx in S_.groupby("panel", observed=True).indices.items():
            tg = t1_targets(p, split)
            kk = S_.key.to_numpy()[idx]
            j = np.searchsorted(tg["key"], kk)
            assert len(idx) == len(tg["key"]) and np.array_equal(tg["key"][j], kk), p
            gap = TA.target_blackout(p, split)[j].astype(bool)
            if p not in T2_PANELS:
                assert not gap.any()
                continue
            F = fills[split, p]
            assert len(F) == int(gap.sum()), (split, p, len(F), int(gap.sum()))
            loc = pd.Series(np.arange(len(idx)), index=kk)
            r = loc.reindex(F.key.to_numpy()).to_numpy()
            assert np.isfinite(r).all() and gap[r.astype(np.int64)].all()
            ii = idx[r.astype(np.int64)]
            v0, q0 = v_all[ii], q_all[ii]
            v1, q1 = g6(F.v_new.to_numpy()), g6(F.q_new.to_numpy())
            v_all[ii], q_all[ii] = v1, q1
            ch = (v1 != v0) | (q1 != q0)
            cells.append(pd.DataFrame({"split": split, "panel": p, "key": F.key, "window_id": F.window_id, "k": F.k,
                                       "pos": F.pos, "cond": F.condname, "v58": v0, "q58": q0, "v59": v1, "q59": q1,
                                       "pv_prod_cpu": F.pv_prod_cpu, "pq_prod_cpu": F.pq_prod_cpu,
                                       "pv_spec": F.pv, "pq_spec": F.pq}))
            rows.append({"split": split, "panel": p, "targets": len(idx), "blackout_targets": int(gap.sum()),
                         "changed": int(ch.sum()), "mean_dv": float(np.mean(v1 - v0)),
                         "mean_abs_dv": float(np.mean(np.abs(v1 - v0))), "max_abs_dv": float(np.max(np.abs(v1 - v0))),
                         "mean_dq": float(np.mean(q1 - q0)), "mean_abs_dq": float(np.mean(np.abs(q1 - q0))),
                         "max_abs_dq": float(np.max(np.abs(q1 - q0))),
                         "mean_abs_spec_minus_prod_v": float(np.mean(np.abs(F.pv - F.pv_prod_cpu)))})
            print(split, p, rows[-1], f"{time.time() - t0:.0f}s", flush=True)
        out = S_[["panel", "timestamp", "station_id", "link_id", "mask_regime"]].copy()
        out["speed_kmh"], out["flow_vph"] = v_all, q_all
        out.to_csv(CSV[split], index=False, float_format="%.6g")
        print("wrote", CSV[split], len(out), f"{time.time() - t0:.0f}s", flush=True)
        del S_, out
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "apply_changes.csv", index=False)
    C = pd.concat(cells, ignore_index=True)
    C.to_parquet(OUT / "apply_cells.parquet", index=False)
    by = C.assign(dv=C.v59 - C.v58, dq=C.q59 - C.q58).groupby(["split", "cond", "k"])[["dv", "dq"]].mean()
    write_new(OUT / "apply_summary.json", json.dumps({
        "chosen": name, "cand": cand, "lam_v": lv, "lam_q": lq, "changes": rows,
        "mean_change_vs_v58_by_split_cond_k": by.reset_index().to_dict("records"),
        "outputs_sha256": {s: M.sha256(CSV[s]) for s in SPLITS}, "v58_zip_sha256": M.sha256(V58_ZIP),
        "models_sha256": {f"{t}_f{f}": M.sha256(MODELS / f"t1_unet_fold{f}{t}{OUT_SUFFIX}.pt") for t in TAGS
                          for f in (0, 1)},
        "gbm_sha256": {c: M.sha256(p_) for c, p_ in GBM_FILES.items() if p_.exists()},
        "seconds": round(time.time() - t0)}, indent=2, default=float))
    pd.set_option("display.width", 300)
    print(R.round(4).to_string(), flush=True)


def _new_state_arrays():
    import t1_v52_monthbias as M
    K = M.read_key_state()
    sp = np.full(int(K.submission_id.max()) + 1, np.nan)
    fl = sp.copy()
    for s in SPLITS:
        d = pd.read_csv(CSV[s], float_precision="round_trip",
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


def _patch():
    import t1_v52_monthbias as M
    M.NAME = NAME
    M.OUT = OUT
    M.BASE_ZIP = V51_ZIP
    M.new_state_arrays = _new_state_arrays
    return M


def cmd_package(args) -> None:
    M = _patch()
    M.cmd_package(args)
    write_new(OUT / "v59blk_state_manifest_supplement.json", json.dumps({
        "note": "the manifest written by the imported t1_v52_monthbias package code carries the V52 note and "
                "state_inputs; the actual state inputs of v59blk_state.zip are listed here",
        "state_inputs": [{"path": str(CSV[s]), "sha256": M.sha256(CSV[s])} for s in SPLITS],
        "content": "all state rows = V58 (v58_rs_blk.zip) except the Task-1 targets inside official Task-2 "
                   "blackouts (V59 gap-specialist U-Net fill); queue/ODME rows = v51_april.zip (replaced by V54 "
                   "queues in build_queue_candidate)",
        "output_sha256": M.sha256(WORK / "submissions" / f"{NAME}.zip")}, indent=2))


def cmd_audit(args) -> None:
    M = _patch()
    M.cmd_audit(args)


def cmd_compare(args) -> None:
    """v59_blk.zip vs v58_rs_blk.zip, line by line: queue/ODME lines byte-identical, state lines differ only at
    Task-1 targets inside official Task-2 blackouts, every state value equals the CSV value, %.6g text."""
    import zipfile
    import t1_assemble as TA
    import t1_v52_monthbias as M
    from evaluate import t1_targets
    new = WORK / "submissions" / f"{FULL_NAME}.zip"
    path = OUT / "compare_v58.json"
    if path.exists():
        raise FileExistsError(path)
    sp, fl, K = _new_state_arrays()
    blk_id = np.zeros(len(sp), bool)
    for (split, p), idx in K.groupby(["split", "panel"], observed=True).indices.items():
        tg = t1_targets(p, split)
        kk = K.key.to_numpy()[idx]
        j = np.searchsorted(tg["key"], kk)
        blk_id[K.submission_id.to_numpy()[idx]] = TA.target_blackout(p, split)[j].astype(bool)
    split_arr = np.empty(len(sp), object)
    panel_arr = np.empty(len(sp), object)
    split_arr[K.submission_id.to_numpy()] = K.split.astype(str).to_numpy()
    panel_arr[K.submission_id.to_numpy()] = K.panel.astype(str).to_numpy()
    del K
    st = {"rows": 0, "state_rows": 0, "state_changed": 0, "state_changed_outside_blackout": 0,
          "blackout_state_rows": int(blk_id.sum()), "state_changed_speed": 0, "state_changed_flow": 0,
          "queue_rows": 0, "queue_changed": 0, "odme_rows": 0, "odme_changed": 0, "header_identical": False,
          "state_value_mismatch_vs_csv": 0, "state_text_not_6g": 0, "changed_by_split_panel": {},
          "max_abs_speed_change": 0.0, "max_abs_flow_change": 0.0}
    with zipfile.ZipFile(V58_ZIP) as za, zipfile.ZipFile(new) as zb:
        assert za.testzip() is None and zb.testzip() is None
        with za.open(za.namelist()[0]) as fa, zb.open(zb.namelist()[0]) as fb:
            st["header_identical"] = fa.readline() == fb.readline()
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
                        st["state_changed_outside_blackout"] += int(not blk_id[sid])
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
    st["valid"] = bool(st["header_identical"] and st["rows"] == 6980503 and st["queue_changed"] == 0 and
                       st["odme_changed"] == 0 and st["state_value_mismatch_vs_csv"] == 0 and
                       st["state_text_not_6g"] == 0 and st["state_changed_outside_blackout"] == 0 and
                       0 < st["state_changed"] <= st["blackout_state_rows"])
    st["v58_sha256"], st["v59_blk_sha256"] = M.sha256(V58_ZIP), M.sha256(new)
    write_new(path, json.dumps(st, indent=2))
    print(json.dumps(st, indent=2), flush=True)
    assert st["valid"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("runs")
    sub.add_parser("occ")
    s = sub.add_parser("train")
    s.add_argument("--tag", required=True, choices=TAGS)
    s.add_argument("--fold", type=int, required=True, choices=[0, 1])
    s.add_argument("--epochs", type=int, default=8)
    s.add_argument("--batch", type=int, default=4)
    s.add_argument("--lr", type=float, default=2e-4)
    s.add_argument("--p-full", type=float, default=P_FULL)
    s.add_argument("--suffix", default="")
    s = sub.add_parser("heldout")
    s.add_argument("--suffix", default="")
    s.add_argument("--max-runs", type=int, default=0)
    s = sub.add_parser("wide")
    s.add_argument("--split", required=True, choices=SPLITS)
    s.add_argument("--panels", nargs="+", default=T2_PANELS)
    s.add_argument("--suffix", default="")
    s = sub.add_parser("official")
    s.add_argument("--suffix", default="")
    s = sub.add_parser("crossfit")
    s.add_argument("--suffix", default="")
    for name in ("fit", "apply", "package", "audit", "compare", "crossfit_ss"):
        sub.add_parser(name)
    s = sub.add_parser("train_ss")
    s.add_argument("--tag", required=True, choices=TAGS)
    s.add_argument("--fold", type=int, required=True, choices=[0, 1])
    s.add_argument("--parity", type=int, required=True, choices=[-1, 0, 1])
    s.add_argument("--epochs", type=int, default=6)
    s.add_argument("--batch", type=int, default=4)
    s.add_argument("--lr", type=float, default=1e-4)
    s = sub.add_parser("wide_ss")
    s.add_argument("--split", required=True, choices=SPLITS)
    args = ap.parse_args()
    {"runs": cmd_runs, "occ": cmd_occ, "train": cmd_train, "heldout": cmd_heldout, "wide": cmd_wide,
     "official": cmd_official, "crossfit": cmd_crossfit, "fit": cmd_fit, "apply": cmd_apply,
     "package": cmd_package, "audit": cmd_audit, "compare": cmd_compare, "train_ss": cmd_train_ss,
     "wide_ss": cmd_wide_ss, "crossfit_ss": cmd_crossfit_ss}[args.cmd](args)


if __name__ == "__main__":
    main()
