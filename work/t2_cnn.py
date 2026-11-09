#!/usr/bin/env python
"""Task 2 queue_ongoing with a convolutional network over the whole corridor.

The per-link GBM (task2_ml2) sees each link with a handful of neighbour features; a 1-D U-Net along the
corridor sees the full 60-minute history of every link at once (queue extents, tails, neighbouring blocks).
Input per link: 12 history steps x (speed/ffs, flow/cap, occupancy, visible, queued) + static link
features + time of day; output: queue logits for T+5..T+30. Loss: masked BCE plus (1 - soft IoU) per window,
the leaderboard's per-window score. History cells are randomly hidden in training so the model copes with
the sparser masked layer of the pseudo windows as well as the raw official history.

Information contract: only the history [T-60, T) and train.

Usage (from the "IEEE Big data" folder):
  python work/t2_cnn.py train [--epochs 12]
  python work/t2_cnn.py pseudo                    # validation/private pseudo windows vs ml2
  python work/t2_cnn.py predict --base work/sub/queue_{split}_v18.csv --out work/sub/queue_{split}_v21.csv
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import RELEASE, ROOT, STEPS, T2_PANELS, links, slot_of  # noqa: E402
from t1_unet import static  # noqa: E402
from t2_grids import load as load_grids  # noqa: E402

HIST, HOR = 12, 6
MODELS = ROOT / "work" / "models"
SEED = 2026
N_IN = 5 * HIST + 6 + 3


class PanelData:
    def __init__(self, panel: str, split: str = "train", unmasked: bool = True):
        st = static(panel)
        self.panel, self.perm, self.st = panel, st["perm"], st
        L = links(panel)
        self.vcut = L.v_cut_kmh.to_numpy(np.float32)[self.perm]
        g = load_grids(panel, split)
        keys = ("Vt", "Qt", "Ot") if unmasked else ("V", "Q", "O")
        self.V, self.Q, self.O = (g[k][:, :, self.perm].astype(np.float16) for k in keys)
        self.weekday = g["weekday"]
        self.dates = [str(d) for d in g["dates"]]
        load_grids.cache_clear()
        n = len(self.perm)
        self.static_ch = np.stack([st["ffs"] / 120, st["lanes"] / 6, st["cap"] / st["lanes"] / 2500,
                                   np.log(st["length"] + 0.01) / 3, st["n_on"], st["n_off"]]).astype(np.float32)
        self.n = n

    def windows(self, gap: int = 1) -> list[tuple[int, int]]:
        """(day, T) with an established queue in the history and a queue in the horizon (official rule).
        gap=1: horizon T+1..T+6 (history ends at T-1, the row at T unused); gap=0: horizon T..T+5."""
        out = []
        for d in range(self.V.shape[0]):
            with np.errstate(invalid="ignore"):
                Qd = self.V[d].astype(np.float32) <= self.vcut
            for T in range(HIST, STEPS - HOR, 2):
                H = Qd[T - HIST:T]
                if H.sum() >= 2 and H.sum(0).max() >= 2 and Qd[T + gap:T + gap + HOR].any():
                    out.append((d, T))
        return out


def make_x(V, Q, O, st_ch, ffs, cap, vcut, tod: int, weekday: int, hide=None) -> np.ndarray:
    """(N_IN, n) input from (12, n) history arrays (NaN = not visible)."""
    V, Q, O = (np.asarray(a, np.float32) for a in (V, Q, O))
    vis = np.isfinite(V) & np.isfinite(Q)
    if hide is not None:
        vis &= ~hide
    with np.errstate(invalid="ignore"):
        qd = vis & (V <= vcut)
    ch = [np.where(vis, V / ffs, 0), np.where(vis, Q / cap, 0), np.where(vis & np.isfinite(O), O, 0), vis, qd]
    x = np.concatenate([np.asarray(c, np.float32) for c in ch], 0)                  # (60, n)
    n = V.shape[1]
    ang = tod / STEPS * 2 * np.pi
    ctx = np.array([np.sin(ang), np.cos(ang), float(weekday >= 5)], np.float32)[:, None].repeat(n, 1)
    return np.concatenate([x, st_ch, ctx], 0)


def build_model(width: int = 96):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class Block(nn.Module):
        def __init__(self, cin, cout):
            super().__init__()
            self.c1, self.n1 = nn.Conv1d(cin, cout, 5, padding=2), nn.GroupNorm(8, cout)
            self.c2, self.n2 = nn.Conv1d(cout, cout, 5, padding=2), nn.GroupNorm(8, cout)
            self.skip = nn.Conv1d(cin, cout, 1) if cin != cout else nn.Identity()

        def forward(self, x):
            h = F.gelu(self.n1(self.c1(x)))
            return F.gelu(self.n2(self.c2(h)) + self.skip(x))

    class Net(nn.Module):
        def __init__(self, w):
            super().__init__()
            self.e1, self.e2, self.e3, self.e4 = Block(N_IN, w), Block(w, 2 * w), Block(2 * w, 4 * w), Block(4 * w, 4 * w)
            self.d3, self.d2, self.d1 = Block(8 * w, 2 * w), Block(4 * w, w), Block(2 * w, w)
            self.head = nn.Conv1d(w, HOR, 1)

        def forward(self, x):
            e1 = self.e1(x)
            e2 = self.e2(F.max_pool1d(e1, 2))
            e3 = self.e3(F.max_pool1d(e2, 2))
            e4 = self.e4(F.max_pool1d(e3, 2))
            up = lambda a: a.repeat_interleave(2, dim=-1)
            d3 = self.d3(torch.cat([up(e4), e3], 1))
            d2 = self.d2(torch.cat([up(d3), e2], 1))
            d1 = self.d1(torch.cat([up(d2), e1], 1))
            return self.head(d1)

    return Net(width)


def pad8(n: int) -> int:
    return int(math.ceil(n / 8) * 8)


def batch_tensors(xs, ys=None, ms=None):
    import torch
    n = xs[0].shape[-1]
    W = pad8(n)
    X = np.zeros((len(xs), xs[0].shape[0], W), np.float32)
    for k, x in enumerate(xs):
        X[k, :, :n] = x
    out = [torch.from_numpy(X)]
    if ys is not None:
        Y = np.zeros((len(ys), HOR, W), np.float32)
        M = np.zeros((len(ys), HOR, W), np.float32)
        for k, (y, m) in enumerate(zip(ys, ms)):
            Y[k, :, :n], M[k, :, :n] = y, m
        out += [torch.from_numpy(Y), torch.from_numpy(M)]
    return out


def cmd_train(args) -> None:
    import torch
    import torch.nn.functional as F
    torch.manual_seed(SEED + args.seed)
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    rng = np.random.default_rng(SEED + args.seed)
    data = {p: PanelData(p) for p in T2_PANELS}
    wins = {p: d.windows(args.gap) for p, d in data.items()}
    print({p: len(w) for p, w in wins.items()}, flush=True)
    model = build_model(args.width).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps_per_epoch = args.per_epoch // args.batch
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps_per_epoch * args.epochs,
                                                pct_start=0.1)
    for ep in range(args.epochs):
        t0, tot = time.time(), 0.0
        model.train()
        for it in range(steps_per_epoch):
            p = T2_PANELS[it % len(T2_PANELS)]
            D, W = data[p], wins[p]
            xs, ys, ms = [], [], []
            for j in rng.integers(0, len(W), args.batch):
                d, T = W[j]
                h = slice(T - HIST, T)
                hide = rng.random((HIST, D.n)) < rng.uniform(0, 0.5)
                xs.append(make_x(D.V[d, h], D.Q[d, h], D.O[d, h], D.static_ch, D.st["ffs"], D.st["cap"], D.vcut,
                                 T, int(D.weekday[d]), hide))
                fut = D.V[d, T + args.gap:T + args.gap + HOR].astype(np.float32)
                m = np.isfinite(fut)
                with np.errstate(invalid="ignore"):
                    ys.append((m & (fut <= D.vcut)).astype(np.float32))
                ms.append(m.astype(np.float32))
            X, Y, M = (t.cuda() for t in batch_tensors(xs, ys, ms))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logit = model(X)
            logit = logit.float()
            bce = (F.binary_cross_entropy_with_logits(logit, Y, reduction="none") * M).sum() / M.sum().clamp(min=1)
            pr = torch.sigmoid(logit) * M
            inter = (pr * Y).sum((1, 2))
            union = (pr + Y - pr * Y).sum((1, 2))
            iou = ((inter + 1) / (union + 1)).mean()
            loss = bce + args.iou_w * (1 - iou)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += float(loss)
        print(f"  epoch {ep + 1}/{args.epochs}: loss {tot / steps_per_epoch:.4f} ({time.time() - t0:.0f}s)", flush=True)
        MODELS.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), MODELS / f"t2_cnn{args.tag}.pt")


def load_model(tag: str, width: int):
    import torch
    m = build_model(width).cuda()
    m.load_state_dict(torch.load(MODELS / f"t2_cnn{tag}.pt", map_location="cuda"))
    m.eval()
    return m


def predict_probs(model, xs) -> np.ndarray:
    import torch
    (X,) = batch_tensors(xs)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        out = torch.sigmoid(model(X.cuda()).float()).cpu().numpy()
    return out[:, :, :xs[0].shape[-1]]


def cmd_pseudo(args) -> None:
    from t2_pseudo import evaluate, summary
    import t2_exp
    model = load_model(args.tag, args.width)
    cache = {}

    def probs(w):
        key = (w.panel, w.split, w.day, w.T)
        if key not in cache:
            if (w.panel, w.split) not in cache:
                cache[(w.panel, w.split)] = PanelData(w.panel, w.split, unmasked=False)
            D = cache[(w.panel, w.split)]
            h = slice(w.T - HIST, w.T)
            x = make_x(D.V[w.day, h], D.Q[w.day, h], D.O[w.day, h], D.static_ch, D.st["ffs"], D.st["cap"], D.vcut,
                       w.T, w.weekday)
            pr = predict_probs(model, [x])[0]
            out = np.zeros_like(pr)
            out[:, D.perm] = pr                                        # travel order -> pos order
            cache[key] = out
        return cache[key]
    for th in (0.4, 0.5):
        df = evaluate(lambda w: probs(w) >= th, conds=("ongoing",))
        print(f"cnn theta {th}:", summary(df).ongoing.to_dict(), flush=True)
    for a in (0.5,):
        df = evaluate(lambda w: (a * probs(w) + (1 - a) * t2_exp.ml2_prob(w)) >= 0.4, conds=("ongoing",))
        print(f"cnn/ml2 mean, theta 0.4:", summary(df).ongoing.to_dict(), flush=True)
        print(df.groupby(["split", "panel"]).iou.mean().unstack(0).round(3).T.to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("train", "pseudo"):
        s = sub.add_parser(name)
        s.add_argument("--tag", default="")
        s.add_argument("--width", type=int, default=96)
        if name == "train":
            s.add_argument("--epochs", type=int, default=12)
            s.add_argument("--batch", type=int, default=32)
            s.add_argument("--per-epoch", type=int, default=32000)
            s.add_argument("--lr", type=float, default=2e-3)
            s.add_argument("--iou-w", type=float, default=0.5)
            s.add_argument("--seed", type=int, default=0)
            s.add_argument("--gap", type=int, default=1, help="1: forecast T+1..T+6 from rows <T (old); 0: T..T+5")
    args = ap.parse_args()
    {"train": cmd_train, "pseudo": cmd_pseudo}[args.cmd](args)
