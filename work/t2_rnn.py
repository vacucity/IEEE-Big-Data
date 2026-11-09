#!/usr/bin/env python
"""Task 2 queue_ongoing with recurrent networks: a GRU reads each link's 60-minute history over time, then a
bidirectional LSTM runs along the corridor (upstream <-> downstream), the direction queues propagate in, so a
link's forecast can draw on a bottleneck far downstream. Same inputs, loss and information contract as
t2_cnn.py (history [T-60, T) only); a different inductive bias for the ensemble.

Usage (from the "IEEE Big data" folder):
  python work/t2_rnn.py train [--epochs 16 --seed 0 --tag ""]
  python work/t2_rnn.py pseudo [--tags ""]
"""
from __future__ import annotations

import argparse
import sys
import time
from functools import lru_cache
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import t2_cnn as C  # noqa: E402
from evaluate import ROOT, T2_PANELS  # noqa: E402

HIST, HOR = C.HIST, C.HOR
N_TIME, N_STATIC = 5, 6 + 3                 # per-step channels of make_x; static + context channels
MODELS = ROOT / "work" / "models"


def build_model(hid_t: int = 64, hid_s: int = 128):
    import torch
    import torch.nn as nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = nn.Linear(N_TIME + N_STATIC, hid_t)
            self.gru = nn.GRU(hid_t, hid_t, batch_first=True)
            self.mix = nn.Sequential(nn.Linear(hid_t + N_STATIC + N_TIME * 2, hid_s), nn.GELU())
            self.lstm = nn.LSTM(hid_s, hid_s, num_layers=2, batch_first=True, bidirectional=True, dropout=0.1)
            self.head = nn.Sequential(nn.Linear(2 * hid_s + hid_s, hid_s), nn.GELU(), nn.Linear(hid_s, HOR))

        def forward(self, x):                                  # x: (B, C, n) as built by t2_cnn.make_x
            B, _, n = x.shape
            seq = x[:, :N_TIME * HIST].reshape(B, N_TIME, HIST, n).permute(0, 3, 2, 1)      # (B, n, 12, 5)
            stat = x[:, N_TIME * HIST:].permute(0, 2, 1)                                     # (B, n, 9)
            s = torch.cat([seq, stat[:, :, None, :].expand(B, n, HIST, N_STATIC)], -1)
            h = torch.relu(self.inp(s.reshape(B * n, HIST, -1)))
            _, hT = self.gru(h)                                                              # (1, B*n, hid_t)
            e = hT[0].reshape(B, n, -1)
            last = seq[:, :, -1, :]                                                          # last step, raw
            mean = seq.mean(2)
            z = self.mix(torch.cat([e, stat, last, mean], -1))                               # (B, n, hid_s)
            o, _ = self.lstm(z)
            return self.head(torch.cat([o, z], -1)).permute(0, 2, 1)                         # (B, 6, n)

    return Net()


def to_batch(xs, ys=None, ms=None):
    import torch
    out = [torch.from_numpy(np.stack(xs).astype(np.float32))]
    if ys is not None:
        out += [torch.from_numpy(np.stack(ys).astype(np.float32)), torch.from_numpy(np.stack(ms).astype(np.float32))]
    return out


def cmd_train(args) -> None:
    import torch
    import torch.nn.functional as F
    torch.manual_seed(C.SEED + args.seed)
    # cuDNN's deterministic LSTM path is several times slower here; the RNN is an ensemble member only
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = False, True
    rng = np.random.default_rng(C.SEED + args.seed)
    data = {p: C.PanelData(p) for p in T2_PANELS}
    wins = {p: d.windows(args.gap) for p, d in data.items()}
    model = build_model().cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = args.per_epoch // args.batch
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps * args.epochs, pct_start=0.1)
    print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params", flush=True)
    for ep in range(args.epochs):
        t0, tot = time.time(), 0.0
        model.train()
        for it in range(steps):
            p = T2_PANELS[it % len(T2_PANELS)]
            D, W = data[p], wins[p]
            xs, ys, ms = [], [], []
            for j in rng.integers(0, len(W), args.batch):
                d, T = W[j]
                h = slice(T - HIST, T)
                hide = rng.random((HIST, D.n)) < rng.uniform(0, 0.5)
                xs.append(C.make_x(D.V[d, h], D.Q[d, h], D.O[d, h], D.static_ch, D.st["ffs"], D.st["cap"], D.vcut,
                                   T, int(D.weekday[d]), hide))
                fut = D.V[d, T + args.gap:T + args.gap + HOR].astype(np.float32)
                m = np.isfinite(fut)
                with np.errstate(invalid="ignore"):
                    ys.append((m & (fut <= D.vcut)).astype(np.float32))
                ms.append(m.astype(np.float32))
            X, Y, M = (t.cuda() for t in to_batch(xs, ys, ms))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logit = model(X)
            logit = logit.float()
            bce = (F.binary_cross_entropy_with_logits(logit, Y, reduction="none") * M).sum() / M.sum().clamp(min=1)
            pr = torch.sigmoid(logit) * M
            iou = (((pr * Y).sum((1, 2)) + 1) / ((pr + Y - pr * Y).sum((1, 2)) + 1)).mean()
            loss = bce + args.iou_w * (1 - iou)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += float(loss)
        print(f"  epoch {ep + 1}/{args.epochs}: loss {tot / steps:.4f} ({time.time() - t0:.0f}s)", flush=True)
        MODELS.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), MODELS / f"t2_rnn{args.tag}.pt")


@lru_cache(maxsize=8)
def load_model(tag: str = ""):
    import torch
    m = build_model().cuda()
    m.load_state_dict(torch.load(MODELS / f"t2_rnn{tag}.pt", map_location="cuda", weights_only=True))
    m.eval()
    return m


def predict_probs(model, xs) -> np.ndarray:
    import torch
    (X,) = to_batch(xs)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        return torch.sigmoid(model(X.cuda()).float()).cpu().numpy()


@lru_cache(maxsize=16)
def panel_data(panel: str, split: str):
    return C.PanelData(panel, split, unmasked=False)


_c: dict = {}


def rnn_prob(w, tags=("",)) -> np.ndarray:
    key = (tuple(tags), w.panel, w.split, w.day, w.T)
    if key not in _c:
        D = panel_data(w.panel, w.split)
        h = slice(w.T - HIST, w.T)
        x = C.make_x(D.V[w.day, h], D.Q[w.day, h], D.O[w.day, h], D.static_ch, D.st["ffs"], D.st["cap"], D.vcut,
                     w.T, w.weekday)
        pr = np.mean([predict_probs(load_model(t), [x])[0] for t in tags], 0)
        out = np.zeros_like(pr)
        out[:, D.perm] = pr
        _c[key] = out
    return _c[key]


def cmd_pseudo(args) -> None:
    from t2_pseudo import evaluate, summary
    import t2_ens as E
    import t2_exp
    tags = tuple(args.tags)
    cnn = ("", "_s1", "_s2")
    df = evaluate(lambda w: rnn_prob(w, tags) >= 0.45, conds=("ongoing",))
    print("rnn alone 0.45      :", summary(df).ongoing.to_dict(), flush=True)
    base = lambda w: (E.v8_prob(w) + t2_exp.ml2_prob(w) + E.cnn_prob(w, cnn)) / 3
    df = evaluate(lambda w: base(w) >= 0.45, conds=("ongoing",))
    print("v23 ensemble        :", summary(df).ongoing.to_dict(), flush=True)
    for name, f in {
        "v8+ml2+cnn+rnn /4": lambda w: (E.v8_prob(w) + t2_exp.ml2_prob(w) + E.cnn_prob(w, cnn) + rnn_prob(w, tags)) / 4,
        "cnn+rnn /2": lambda w: (E.cnn_prob(w, cnn) + rnn_prob(w, tags)) / 2,
        "gbms/4+cnn/4+rnn/4": lambda w: (0.5 * E.v8_prob(w) + 0.5 * t2_exp.ml2_prob(w) + E.cnn_prob(w, cnn)
                                         + rnn_prob(w, tags)) / 3,
    }.items():
        for th in (0.45, 0.5):
            df = evaluate(lambda w: f(w) >= th, conds=("ongoing",))
            print(f"{name:20s} {th}:", summary(df).ongoing.to_dict(), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("train")
    s.add_argument("--tag", default="")
    s.add_argument("--epochs", type=int, default=16)
    s.add_argument("--batch", type=int, default=32)
    s.add_argument("--per-epoch", type=int, default=32000)
    s.add_argument("--lr", type=float, default=2e-3)
    s.add_argument("--iou-w", type=float, default=0.5)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--gap", type=int, default=1, help="1: forecast T+1..T+6 from rows <T (old); 0: T..T+5")
    s = sub.add_parser("pseudo")
    s.add_argument("--tags", nargs="+", default=[""])
    args = ap.parse_args()
    {"train": cmd_train, "pseudo": cmd_pseudo}[args.cmd](args)


def official_rnn(panel: str, split: str, tags=("",)) -> dict[str, np.ndarray]:
    """window_id -> (6, n) RNN probabilities (pos order) for the official ongoing windows."""
    import pandas as pd
    from evaluate import RELEASE, links, slot_of
    from t1_unet import static
    st = static(panel)
    perm = st["perm"]
    L = links(panel)
    n = len(L)
    vcut = L.v_cut_kmh.to_numpy(np.float32)[perm]
    static_ch = np.stack([st["ffs"] / 120, st["lanes"] / 6, st["cap"] / st["lanes"] / 2500,
                          np.log(st["length"] + 0.01) / 3, st["n_on"], st["n_off"]]).astype(np.float32)
    base = RELEASE / "task2" / panel / split
    win = pd.read_csv(base / "window_index.csv")
    hist = pd.read_parquet(base / "window_history.parquet")
    hist["pos"] = hist.link_id.map(dict(zip(L.link_id, range(n))))
    hist["slot"] = slot_of(hist.timestamp)
    out = {}
    for w in win.itertuples(index=False):
        if w.condition != "queue_ongoing":
            continue
        T = int(slot_of(pd.Series([w.forecast_origin]))[0])
        origin = pd.Timestamp(w.forecast_origin)
        h = hist[hist.window_id == w.window_id]
        ok = (h.is_score_eligible.astype(bool) & h.speed_kmh.notna()).to_numpy()
        si = (h.slot - (T - HIST)).to_numpy()
        m = ok & (si >= 0) & (si < HIST)
        g = {}
        for c in ("speed_kmh", "flow_vph", "occupancy"):
            a = np.full((HIST, n), np.nan, np.float32)
            a[si[m], h.pos.to_numpy()[m]] = h[c].to_numpy()[m]
            g[c] = a[:, perm]
        x = C.make_x(g["speed_kmh"], g["flow_vph"], g["occupancy"], static_ch, st["ffs"], st["cap"], vcut,
                     origin.hour * 12 + origin.minute // 5, origin.weekday())
        pr = np.mean([predict_probs(load_model(t), [x])[0] for t in tags], 0)
        o = np.zeros_like(pr)
        o[:, perm] = pr
        out[w.window_id] = o
    return out
