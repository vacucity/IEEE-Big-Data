#!/usr/bin/env python
"""Task 1 (and through it Task 3) with a 2-D U-Net over whole days: time (288 steps) x links in travel order.

Why a second model next to the LightGBM corrections
  * 68% of the speed error the GBM leaves sits on the 1% of cells where a queue starts or ends between two
    visible readings; deciding them needs the queue's shape in space and time, not the nearest readings.
  * In validation/private every scored Task 2 window blanks the whole corridor for 90 minutes, and Task 1 still
    asks for the cells inside (2% of the targets). Train never shows such a gap, so the GBM's features are out
    of range there: simulated on train its speed RMSE inside the gap is 11-14 km/h against 1.9 elsewhere.
A U-Net sees the whole day (Task 1 is offline, both sides of a cell are allowed) and is trained with such gaps
inserted at random, supervised on every eligible cell it cannot see (train ships the unmasked layer).

Usage (from the "IEEE Big data" folder):
  python work/t1_unet.py build                    # memmaps under work/cache/t1_unet
  python work/t1_unet.py train --fold 0           # fold k trains on months with month % 2 != k; -1 = all
  python work/t1_unet.py oof --fold 0             # out-of-fold predictions of fold k -> cache
  python work/t1_unet.py predict                  # fold -1 model -> validation/private predictions -> cache
"""
from __future__ import annotations

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
from evaluate import CACHE, CLEAN, PANELS, REGIMES, ROOT, STEPS, links  # noqa: E402

DATA = CACHE / "t1_unet"
MODELS = ROOT / "work" / "models"
SPLITS = ("train", "validation", "private")
OBS_C = 8          # v/ffs, q/cap, occ, visible, on-ramp/cap, on-ramp ok, off-ramp/cap, off-ramp ok
GAP = 18
SEED = 2026


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def travel_perm(panel: str) -> np.ndarray:
    """pos -> travel order (index increases along the direction of traffic)."""
    report = json.loads((CLEAN / "cleaning_report.json").read_text(encoding="utf-8"))
    share = report[panel]["network"]["upstream_has_lower_milepost_share"]
    n = len(links(panel))
    return np.arange(n) if share is None or share >= 0.5 else np.arange(n)[::-1].copy()


def static(panel: str) -> dict:
    L = links(panel)
    perm = travel_perm(panel)
    return {"perm": perm, "ffs": L.free_speed_kmh.to_numpy(np.float32)[perm],
            "cap": L.capacity_vph.to_numpy(np.float32)[perm], "lanes": L.lanes.to_numpy(np.float32)[perm],
            "length": L.length_km.to_numpy(np.float32)[perm], "n_on": L.on_ramp_n.to_numpy(np.float32)[perm],
            "n_off": L.off_ramp_n.to_numpy(np.float32)[perm]}


def build(panel: str, split: str) -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    out = DATA / f"{panel}_{split}"
    if (out.parent / f"{out.name}_meta.json").exists():
        return
    st = static(panel)
    perm, n = st["perm"], len(st["perm"])
    pf = pq.ParquetFile(CLEAN / panel / f"{split}.parquet")
    D = pf.num_row_groups
    obs = np.lib.format.open_memmap(f"{out}_obs.npy", "w+", np.float16, (D, STEPS, n, OBS_C))
    flags = np.lib.format.open_memmap(f"{out}_flags.npy", "w+", np.uint8, (D, STEPS, n))   # bit0 tgt, bit1 blk
    pct = np.lib.format.open_memmap(f"{out}_pct.npy", "w+", np.uint8, (D, STEPS, n))
    if split == "train":
        truth = np.lib.format.open_memmap(f"{out}_truth.npy", "w+", np.float16, (D, STEPS, n, 2))
    meta = {"dates": [], "weekday": [], "regime": [], "month": []}
    cols = ["date", "weekday", "mask_regime", "ts", "pct_observed", "is_t1_target", "is_t2_blackout",
            "speed_obs", "flow_obs", "occ_obs", "on_ramp_flow_partial", "on_ramp_n_valid", "off_ramp_flow_partial",
            "off_ramp_n_valid"] + (["speed_true", "flow_true", "is_score_eligible"] if split == "train" else [])
    for i in range(D):
        t = pf.read_row_group(i, columns=cols)
        g = lambda c: t.column(c).to_numpy().astype(np.float32).reshape(STEPS, n)[:, perm]
        v, q, o = g("speed_obs"), g("flow_obs"), g("occ_obs")
        vis = np.isfinite(v) & np.isfinite(q)
        ron, roff = g("on_ramp_flow_partial"), g("off_ramp_flow_partial")
        ok_on, ok_off = g("on_ramp_n_valid") > 0, g("off_ramp_n_valid") > 0
        x = np.stack([np.where(vis, v / st["ffs"], 0), np.where(vis, q / st["cap"], 0),
                      np.where(vis & np.isfinite(o), o, 0), vis,
                      np.where(ok_on & np.isfinite(ron), ron / st["cap"], 0), ok_on,
                      np.where(ok_off & np.isfinite(roff), roff / st["cap"], 0), ok_off], -1)
        obs[i] = x.astype(np.float16)
        tgt = t.column("is_t1_target").to_numpy().reshape(STEPS, n)[:, perm]
        blk = t.column("is_t2_blackout").to_numpy().reshape(STEPS, n)[:, perm]
        flags[i] = tgt.astype(np.uint8) | (blk.astype(np.uint8) << 1)
        pct[i] = np.clip(g("pct_observed"), 0, 100).astype(np.uint8)
        if split == "train":
            e = t.column("is_score_eligible").to_numpy().reshape(STEPS, n)[:, perm]
            vt, qt = g("speed_true"), g("flow_true")
            okt = e & np.isfinite(vt) & np.isfinite(qt)
            truth[i] = np.stack([np.where(okt, vt / st["ffs"], np.nan), np.where(okt, qt / st["cap"], np.nan)],
                                -1).astype(np.float16)
        meta["dates"].append(t.column("date")[0].as_py())
        meta["weekday"].append(int(t.column("weekday")[0].as_py()))
        meta["regime"].append(REGIMES.index(t.column("mask_regime")[0].as_py()))
        meta["month"].append(pd.Timestamp(t.column("ts")[0].as_py()).month)
    obs.flush(); flags.flush(); pct.flush()
    if split == "train":
        truth.flush()
    (out.parent / f"{out.name}_meta.json").write_text(json.dumps(meta))


def profiles(panel: str) -> np.ndarray:
    """(2, 288, n, 2) usual v/ffs and q/cap per weekday/weekend, travel order (task1_gbm's profiles)."""
    from task1_gbm import Panel
    ctx = Panel(panel)
    st = static(panel)
    pv, pq_ = ctx.prof_v[:, :, st["perm"]], ctx.prof_q[:, :, st["perm"]]
    return np.stack([pv / st["ffs"], pq_ / st["cap"]], -1).astype(np.float32)


def open_split(panel: str, split: str) -> dict:
    base = DATA / f"{panel}_{split}"
    d = {"obs": np.load(f"{base}_obs.npy", mmap_mode="r"), "flags": np.load(f"{base}_flags.npy", mmap_mode="r"),
         "pct": np.load(f"{base}_pct.npy", mmap_mode="r"),
         "meta": json.loads((DATA / f"{panel}_{split}_meta.json").read_text())}
    if split == "train":
        d["truth"] = np.load(f"{base}_truth.npy", mmap_mode="r")
    return d


def cmd_build(args) -> None:
    for p in args.panels:
        for s in SPLITS:
            t0 = time.time()
            build(p, s)
            print(f"  {p} {s}: {time.time() - t0:.0f}s", flush=True)
        np.save(DATA / f"{p}_profiles.npy", profiles(p))


# ---------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------
def interp_time(G: np.ndarray, vis: np.ndarray) -> np.ndarray:
    """Per link linear interpolation in time between visible cells; one-sided beyond the ends; NaN if none."""
    T, n = G.shape
    idx = np.arange(T)[:, None]
    prev = np.maximum.accumulate(np.where(vis, idx, -1), axis=0)
    nxt = np.minimum.accumulate(np.where(vis, idx, T)[::-1], axis=0)[::-1]
    hp, hn = prev >= 0, nxt < T
    vp = np.take_along_axis(G, prev.clip(0), 0)
    vn = np.take_along_axis(G, nxt.clip(0, T - 1), 0)
    w = np.where(hp & hn, (idx - prev) / np.maximum(nxt - prev, 1), 0.0)
    return np.where(hp & hn, vp + (vn - vp) * w, np.where(hp, vp, np.where(hn, vn, np.nan)))


class Day:
    """Everything one (panel, split, day) contributes; arrays in travel order."""

    def __init__(self, panel: str, split: str, i: int, src: dict, st: dict, prof: np.ndarray):
        self.panel, self.split, self.i = panel, split, i
        self.obs = np.asarray(src["obs"][i], np.float32)                  # (288, n, 8)
        fl = np.asarray(src["flags"][i])
        self.tgt, self.blk = (fl & 1) > 0, (fl & 2) > 0
        self.pct = np.asarray(src["pct"][i], np.float32) / 100.0
        self.truth = np.asarray(src["truth"][i], np.float32) if "truth" in src else None
        m = src["meta"]
        self.weekday, self.regime, self.month = m["weekday"][i], m["regime"][i], m["month"][i]
        self.st, self.prof = st, prof[int(self.weekday >= 5)]


def make_input(day: Day, hide: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(C, 288, n) input and (2, 288, n) baseline (v/ffs, q/cap). `hide` blanks extra visible cells."""
    o = day.obs.copy()
    vis = o[..., 3] > 0.5
    if hide is not None:
        vis = vis & ~hide
        o[..., :4] *= vis[..., None]
    st = day.st
    n = vis.shape[1]
    base = []
    for c in (0, 1):
        b = interp_time(o[..., c], vis)
        base.append(np.where(np.isfinite(b), b, day.prof[..., c]))
    base = np.stack(base)                                                    # (2, 288, n)
    tod = np.arange(STEPS) / STEPS * 2 * np.pi
    bc = lambda a: np.broadcast_to(np.asarray(a, np.float32)[None, :], (STEPS, n))
    bt = lambda a: np.broadcast_to(np.asarray(a, np.float32)[:, None], (STEPS, n))
    ch = [o[..., k] for k in range(OBS_C)] + [day.pct, day.prof[..., 0], day.prof[..., 1], base[0], base[1],
          bc(st["ffs"] / 120), bc(st["lanes"] / 6), bc(st["cap"] / st["lanes"] / 2500),
          bc(np.log(st["length"] + 0.01) / 3), bc(st["n_on"]), bc(st["n_off"]), bt(np.sin(tod)), bt(np.cos(tod)),
          np.full((STEPS, n), float(day.weekday >= 5), np.float32), np.full((STEPS, n), day.regime / 2, np.float32)]
    return np.stack(ch).astype(np.float32), base.astype(np.float32)


N_IN = OBS_C + 1 + 2 + 2 + 6 + 2 + 2


def augment_hide(day: Day, rng: np.random.Generator) -> np.ndarray:
    """Training-time blanking: Task 2 style 90-minute corridor gaps and a few extra random cells."""
    hide = np.zeros(day.tgt.shape, bool)
    if rng.random() < 0.6:
        for _ in range(rng.integers(1, 3)):
            s0 = int(rng.integers(12, STEPS - GAP - 1))
            hide[s0:s0 + GAP] = True
    hide |= rng.random(hide.shape) < 0.08
    return hide


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------
def build_model(width: int = 32):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class Block(nn.Module):
        def __init__(self, cin, cout):
            super().__init__()
            self.c1 = nn.Conv2d(cin, cout, 3, padding=1)
            self.n1 = nn.GroupNorm(8, cout)
            self.c2 = nn.Conv2d(cout, cout, 3, padding=1)
            self.n2 = nn.GroupNorm(8, cout)
            self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

        def forward(self, x):
            h = F.gelu(self.n1(self.c1(x)))
            h = self.n2(self.c2(h))
            return F.gelu(h + self.skip(x))

    class UNet(nn.Module):
        def __init__(self, cin, w):
            super().__init__()
            self.e1, self.e2, self.e3 = Block(cin, w), Block(w, 2 * w), Block(2 * w, 4 * w)
            self.e4, self.m = Block(4 * w, 8 * w), Block(8 * w, 8 * w)
            self.d3, self.d2, self.d1 = Block(12 * w, 4 * w), Block(6 * w, 2 * w), Block(3 * w, w)
            self.head = nn.Conv2d(w, 2, 1)

        def forward(self, x):
            e1 = self.e1(x)
            e2 = self.e2(F.max_pool2d(e1, 2))
            e3 = self.e3(F.max_pool2d(e2, 2))
            e4 = self.m(self.e4(F.max_pool2d(e3, 2)))
            def up(a, ref):
                b, c, h, w = a.shape
                return a[:, :, :, None, :, None].expand(b, c, h, 2, w, 2).reshape(b, c, 2 * h, 2 * w)
            d3 = self.d3(torch.cat([up(e4, e3), e3], 1))
            d2 = self.d2(torch.cat([up(d3, e2), e2], 1))
            d1 = self.d1(torch.cat([up(d2, e1), e1], 1))
            return self.head(d1)

    return UNet(N_IN, width)


def pad8(n: int) -> int:
    return int(math.ceil(n / 8) * 8)


def to_batch(xs: list[np.ndarray]):
    import torch
    n = xs[0].shape[-1]
    X = np.zeros((len(xs), xs[0].shape[0], STEPS, pad8(n)), np.float32)
    for k, x in enumerate(xs):
        X[k, :, :, :n] = x
    return torch.from_numpy(X)


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------
def day_lists(fold: int, split: str = "train"):
    out = []
    for p in PANELS:
        m = json.loads((DATA / f"{p}_{split}_meta.json").read_text())
        for i, month in enumerate(m["month"]):
            if fold < 0 or month % 2 != fold:
                out.append((p, i))
    return out


def loss_terms(pred, base, truth, sup, st_t):
    """Weighted squared error in score units (km/h / 25, vph per lane / 600) over the supervised cells."""
    v = base[:, 0] + pred[:, 0]
    q = base[:, 1] + pred[:, 1]
    ev = (v - truth[:, 0]) * st_t[:, 0]
    eq = (q - truth[:, 1]) * st_t[:, 1]
    l = 0.54 * ev ** 2 + 0.46 * eq ** 2
    return (l * sup).sum() / sup.sum().clamp(min=1.0)


def lwr_term(pred, base, truth, sup, elig, phys):
    """E_LWR of the batch: |dN_sub - dN_true| over 5-minute transitions on a link, N = q/v * L, where the
    hidden (supervised) cells carry the prediction and every other eligible cell its truth; normalised by
    sum |dN_true| over all eligible transitions, as Task 3 does."""
    import torch
    ffs, cap, length = phys[:, 0], phys[:, 1], phys[:, 2]           # (B, 1, W)
    v = (base[:, 0] + pred[:, 0]) * ffs
    q = torch.relu(base[:, 1] + pred[:, 1]) * cap
    vt, qt = truth[:, 0] * ffs, truth[:, 1] * cap
    n_sub = q / v.clamp(min=5.0) * length
    n_true = qt / vt.clamp(min=1.0) * length
    n_mix = torch.where(sup > 0.5, n_sub, n_true)
    ok = (elig[:, 1:] * elig[:, :-1])
    d_true = n_true[:, 1:] - n_true[:, :-1]
    res = ((n_mix[:, 1:] - n_mix[:, :-1]) - d_true).abs() * ok
    return res.sum() / ((d_true.abs() * ok).sum() + 1e-6)


def cmd_train(args) -> None:
    import torch
    torch.manual_seed(SEED + 100 * args.seed)
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    rng = np.random.default_rng(SEED + args.fold + 100 * args.seed)
    days = day_lists(args.fold)
    srcs = {p: open_split(p, "train") for p in PANELS}
    sts = {p: static(p) for p in PANELS}
    profs = {p: np.load(DATA / f"{p}_profiles.npy") for p in PANELS}
    dev = "cuda"
    model = build_model(args.width).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    by_panel = {}
    for p, i in days:
        by_panel.setdefault(p, []).append(i)
    steps_per_epoch = sum(math.ceil(len(v) / args.batch) for v in by_panel.values())
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps_per_epoch * args.epochs,
                                                pct_start=0.1)
    print(f"fold {args.fold}: {len(days)} days, {steps_per_epoch} steps/epoch, "
          f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params", flush=True)
    for ep in range(args.epochs):
        t0, tot_l, cnt = time.time(), 0.0, 0
        batches = []
        for p, idx in by_panel.items():
            idx = rng.permutation(idx)
            batches += [(p, idx[k:k + args.batch]) for k in range(0, len(idx), args.batch)]
        model.train()
        for bi in rng.permutation(len(batches)):
            p, idx = batches[bi]
            st = sts[p]
            xs, bs, ts, ss, es = [], [], [], [], []
            for i in idx:
                d = Day(p, "train", int(i), srcs[p], st, profs[p])
                hide = augment_hide(d, rng)
                x, b = make_input(d, hide)
                tr = np.moveaxis(d.truth, -1, 0)
                sup = (d.tgt | (hide & (d.obs[..., 3] > 0.5))) & np.isfinite(tr).all(0)
                xs.append(x); bs.append(b); ts.append(np.nan_to_num(tr)); ss.append(sup[None].astype(np.float32))
                es.append(np.isfinite(tr).all(0)[None].astype(np.float32))
            X, Bb, TR = to_batch(xs).to(dev), to_batch(bs).to(dev), to_batch(ts).to(dev)
            S = to_batch(ss)[:, 0].to(dev)
            EL = to_batch(es)[:, 0].to(dev)
            n, W = len(st["ffs"]), X.shape[-1]
            stt = np.zeros((2, W), np.float32)
            stt[0, :n] = st["ffs"] / 25
            stt[1, :n] = st["cap"] / st["lanes"] / 600
            ST = torch.from_numpy(stt)[None, :, None, :].expand(len(idx), 2, 1, W).to(dev)
            phy = np.ones((3, W), np.float32)
            phy[0, :n], phy[1, :n], phy[2, :n] = st["ffs"], st["cap"], st["length"]
            PH = torch.from_numpy(phy)[None, :, None, :].expand(len(idx), 3, 1, W).to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(X)
            loss = loss_terms(out.float(), Bb, TR, S, ST)
            if args.lwr > 0:
                loss = loss + args.lwr * lwr_term(out.float(), Bb, TR, S, EL, PH)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot_l += float(loss)
            cnt += 1
        print(f"  epoch {ep + 1}/{args.epochs}: loss {tot_l / cnt:.5f} ({time.time() - t0:.0f}s)", flush=True)
        MODELS.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), MODELS / f"t1_unet_fold{args.fold}{args.tag}.pt")


# ---------------------------------------------------------------------------
# prediction
# ---------------------------------------------------------------------------
def predict_day(model, d: Day, hide: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(288, n) speed km/h and flow vph, travel order."""
    import torch
    x, b = make_input(d, hide)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(to_batch([x]).cuda()).float().cpu().numpy()[0]
    n = x.shape[-1]
    v = (b[0] + out[0, :, :n]) * d.st["ffs"]
    q = (b[1] + out[1, :, :n]) * d.st["cap"]
    return np.clip(v, 5.0, 120.75), np.maximum(q, 0.0)


def load_model(fold: int, width: int, tag: str = ""):
    import torch
    m = build_model(width).cuda()
    m.load_state_dict(torch.load(MODELS / f"t1_unet_fold{fold}{tag}.pt", map_location="cuda"))
    m.eval()
    return m


def target_order_preds(panel: str, split: str, model, days=None, hide_fn=None) -> dict:
    """day -> (speed, flow) at the Task 1 targets in clean_data order (raster over (tod, pos))."""
    src, st, prof = open_split(panel, split), static(panel), np.load(DATA / f"{panel}_profiles.npy")
    inv = np.argsort(st["perm"])
    out = {}
    for i in (days if days is not None else range(len(src["meta"]["month"]))):
        d = Day(panel, split, i, src, st, prof)
        v, q = predict_day(model, d, hide_fn(d) if hide_fn else None)
        tg = d.tgt[:, inv]
        out[i] = (v[:, inv][tg], q[:, inv][tg])
    return out


def cmd_oof(args) -> None:
    """Out-of-fold predictions of fold k for its held-out months, saved per panel."""
    model = load_model(args.fold, args.width, args.tag)
    for p in args.panels:
        m = json.loads((DATA / f"{p}_train_meta.json").read_text())
        days = [i for i, mo in enumerate(m["month"]) if mo % 2 == args.fold]
        pr = target_order_preds(p, "train", model, days)
        np.savez(DATA / f"oof_{p}_fold{args.fold}{args.tag}.npz", **{f"v{i}": v for i, (v, q) in pr.items()},
                 **{f"q{i}": q for i, (v, q) in pr.items()})
        print(f"  {p}: {len(days)} days", flush=True)


def cmd_predict(args) -> None:
    """Validation/private predictions; --fold -2 averages the two fold models instead of one full model."""
    models = [load_model(k, args.width, args.tag) for k in ((0, 1) if args.fold == -2 else (args.fold,))]
    for split in ("validation", "private"):
        for p in args.panels:
            prs = [target_order_preds(p, split, m) for m in models]
            pr = {i: (np.mean([x[i][0] for x in prs], 0), np.mean([x[i][1] for x in prs], 0)) for i in prs[0]}
            np.savez(DATA / f"pred_{p}_{split}{args.tag}.npz", **{f"v{i}": v for i, (v, q) in pr.items()},
                     **{f"q{i}": q for i, (v, q) in pr.items()})
            print(f"  {p} {split}", flush=True)


def cmd_adapt(args) -> None:
    """Self-supervised fine-tuning on the target months' own visible cells (Task 1 is offline, so every
    observation of the month may be used). Random visible cells and Task 2 style gaps are hidden and
    predicted from the rest; the Task 1 targets themselves stay unseen. --target heldout adapts fold k's
    model to its held-out train months (to measure the effect against the out-of-fold truth); --target
    valpriv adapts it to validation + private for the submission."""
    import torch
    torch.manual_seed(SEED + 7)
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    rng = np.random.default_rng(SEED + 7 + args.fold)
    if args.target == "heldout":
        days = [(p, "train", i) for p in PANELS
                for i, mo in enumerate(json.loads((DATA / f"{p}_train_meta.json").read_text())["month"])
                if mo % 2 == args.fold]
    else:
        days = [(p, s, i) for s in ("validation", "private") for p in PANELS
                for i in range(len(json.loads((DATA / f"{p}_{s}_meta.json").read_text())["month"]))
                if args.parity < 0 or i % 2 == args.parity]
    srcs = {(p, s): open_split(p, s) for p, s, _ in days}
    sts = {p: static(p) for p in PANELS}
    profs = {p: np.load(DATA / f"{p}_profiles.npy") for p in PANELS}
    model = load_model(args.fold, args.width, args.tag)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    groups = {}
    for p, s, i in days:
        groups.setdefault((p, s), []).append(i)
    for ep in range(args.epochs):
        t0, tot, cnt = time.time(), 0.0, 0
        batches = []
        for key, idx in groups.items():
            idx = rng.permutation(idx)
            batches += [(key, idx[k:k + args.batch]) for k in range(0, len(idx), args.batch)]
        for bi in rng.permutation(len(batches)):
            (p, s), idx = batches[bi]
            st = sts[p]
            xs, bs, ts, ss, es = [], [], [], [], []
            for i in idx:
                d = Day(p, s, int(i), srcs[(p, s)], st, profs[p])
                vis = d.obs[..., 3] > 0.5
                hide = augment_hide(d, rng) & vis
                x, b = make_input(d, hide)
                lab = np.stack([np.where(vis, d.obs[..., 0], 0), np.where(vis, d.obs[..., 1], 0)])
                xs.append(x); bs.append(b); ts.append(lab)
                ss.append(hide[None].astype(np.float32)); es.append(vis[None].astype(np.float32))
            X, Bb, TR = to_batch(xs).cuda(), to_batch(bs).cuda(), to_batch(ts).cuda()
            S, EL = to_batch(ss)[:, 0].cuda(), to_batch(es)[:, 0].cuda()
            n, W = len(st["ffs"]), X.shape[-1]
            stt = np.zeros((2, W), np.float32)
            stt[0, :n], stt[1, :n] = st["ffs"] / 25, st["cap"] / st["lanes"] / 600
            ST = torch.from_numpy(stt)[None, :, None, :].expand(len(idx), 2, 1, W).cuda()
            phy = np.ones((3, W), np.float32)
            phy[0, :n], phy[1, :n], phy[2, :n] = st["ffs"], st["cap"], st["length"]
            PH = torch.from_numpy(phy)[None, :, None, :].expand(len(idx), 3, 1, W).cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(X)
            loss = loss_terms(out.float(), Bb, TR, S, ST)
            if args.lwr > 0:
                loss = loss + args.lwr * lwr_term(out.float(), Bb, TR, S, EL, PH)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss)
            cnt += 1
        print(f"  adapt epoch {ep + 1}/{args.epochs}: loss {tot / cnt:.5f} ({time.time() - t0:.0f}s)", flush=True)
    torch.save(model.state_dict(), MODELS / f"t1_unet_fold{args.fold}{args.out_tag}.pt")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("build")
    s.add_argument("--panels", nargs="+", default=PANELS)
    for name in ("train", "oof", "predict"):
        s = sub.add_parser(name)
        s.add_argument("--fold", type=int, default=-1)
        s.add_argument("--width", type=int, default=32)
        s.add_argument("--tag", default="")
        s.add_argument("--panels", nargs="+", default=PANELS)
        if name == "train":
            s.add_argument("--epochs", type=int, default=25)
            s.add_argument("--batch", type=int, default=4)
            s.add_argument("--lr", type=float, default=2e-3)
            s.add_argument("--lwr", type=float, default=0.0, help="weight of the Task 3 conservation residual")
            s.add_argument("--seed", type=int, default=0)
    s = sub.add_parser("adapt")
    s.add_argument("--fold", type=int, required=True)
    s.add_argument("--width", type=int, default=32)
    s.add_argument("--tag", default="")
    s.add_argument("--out-tag", required=True)
    s.add_argument("--target", choices=["heldout", "valpriv"], required=True)
    s.add_argument("--epochs", type=int, default=3)
    s.add_argument("--batch", type=int, default=4)
    s.add_argument("--lr", type=float, default=2e-4)
    s.add_argument("--lwr", type=float, default=0.0)
    s.add_argument("--parity", type=int, default=-1, help="valpriv: adapt on day index parity only (check)")
    args = ap.parse_args()
    {"build": cmd_build, "train": cmd_train, "oof": cmd_oof, "predict": cmd_predict, "adapt": cmd_adapt}[args.cmd](args)
