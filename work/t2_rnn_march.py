#!/usr/bin/env python
"""Gap-0 ongoing BiLSTM fine-tuned on March (validation) visible windows, for the APRIL windows only.

Organizer ruling (748789/749601): April windows may use train + the whole validation month to fit and select;
April itself may not be used.  The train-fitted BiLSTM never saw the new generation seed nor the noisy masked
layer it is applied to; here it continues training on a mix of train windows and March masked-layer windows
(labels: visible speed <= v_cut on visible cells, as in the pseudo windows).

Selection protocol (March only): fine-tune on March days < 15 (fold a) and score the March pseudo windows with
day >= 15, and the reverse (fold b).  April pseudo windows are never used to choose.

  python work/t2_rnn_march.py finetune --base _g0 --fold a --tag _g0_fta
  python work/t2_rnn_march.py eval --base _g0 --tags _g0_fta:a _g0_ftb:b
  python work/t2_rnn_march.py finetune --base _g0 --fold all --tag _g0_ftall     # for April
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import t2_cnn as C  # noqa: E402
import t2_pseudo as P  # noqa: E402
import t2_rnn as R  # noqa: E402
from evaluate import T2_PANELS  # noqa: E402

HIST, HOR = C.HIST, C.HOR
SPLIT_DAY = 15


def fold_days(D, fold):
    """Training days of a fold.  "lt<N>": March days with index < N only (causal cutoff for March windows on
    day index >= N; index 0 = 1 March)."""
    days = np.arange(D.V.shape[0])
    if fold.startswith("lt"):
        return days < int(fold[2:])
    return {"a": days < SPLIT_DAY, "b": days >= SPLIT_DAY, "all": np.ones(len(days), bool)}[fold]


def test_days(day, fold, span=3):
    """Pseudo windows scored for a model tuned on `fold`: the other half month, or for "lt<N>" the next `span`
    days N..N+span-1 (official March windows sit in the first days of the month)."""
    if fold.startswith("lt"):
        n = int(fold[2:])
        return (day >= n) & (day < n + span)
    return {"a": day >= SPLIT_DAY, "b": day < SPLIT_DAY, "none": np.ones(len(day), bool)}[fold]


def cmd_finetune(args) -> None:
    import torch
    import torch.nn.functional as F
    torch.manual_seed(C.SEED + 100 + args.seed)
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = False, True
    rng = np.random.default_rng(C.SEED + 100 + args.seed)
    tr = {p: C.PanelData(p) for p in T2_PANELS}
    trw = {p: d.windows(0) for p, d in tr.items()}
    mar, marw, lab = {}, {}, {}
    for p in T2_PANELS:
        if args.source == "march":
            D = C.PanelData(p, "validation", unmasked=False)
            ok = fold_days(D, args.fold)
            mar[p], marw[p], lab[p] = D, [(d, T) for d, T in D.windows(0) if ok[d]], D
        else:
            # train masked layer as input (what official windows look like), TRUE queue labels from the train panel
            D = C.PanelData(p, "train", unmasked=False)
            mar[p], marw[p], lab[p] = D, D.windows(0), tr[p]
    print(args.source, "windows", {p: len(w) for p, w in marw.items()}, flush=True)
    if args.kind == "cnn":
        torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    model = (R.build_model() if args.kind == "rnn" else C.build_model(96)).cuda()
    model.load_state_dict(torch.load(R.MODELS / f"t2_{args.kind}{args.base}.pt", map_location="cuda", weights_only=True))
    batcher = R.to_batch if args.kind == "rnn" else C.batch_tensors
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = args.windows // args.batch
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.1)
    n_m = int(round(args.batch * args.mix))
    t0, tot = time.time(), 0.0
    model.train()
    for it in range(steps):
        p = T2_PANELS[it % len(T2_PANELS)]
        xs, ys, ms = [], [], []
        for src, D, W, k, L in (("tr", tr[p], trw[p], args.batch - n_m, tr[p]),
                                ("m", mar[p], marw[p], n_m, lab[p])):
            if k == 0 or not W:
                continue
            for j in rng.integers(0, len(W), k):
                d, T = W[j]
                h = slice(T - HIST, T)
                hmax = 0.5 if src == "tr" else args.march_hide
                hide = rng.random((HIST, D.n)) < rng.uniform(0, hmax)
                xs.append(C.make_x(D.V[d, h], D.Q[d, h], D.O[d, h], D.static_ch, D.st["ffs"], D.st["cap"], D.vcut,
                                   T, int(D.weekday[d]), hide))
                fut = L.V[d, T:T + HOR].astype(np.float32)
                m = np.isfinite(fut)
                with np.errstate(invalid="ignore"):
                    ys.append((m & (fut <= D.vcut)).astype(np.float32))
                ms.append(m.astype(np.float32))
        X, Y, M = (t.cuda() for t in batcher(xs, ys, ms))
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
        if (it + 1) % 100 == 0:
            print(f"  step {it + 1}/{steps}: loss {tot / 100:.4f} ({time.time() - t0:.0f}s)", flush=True)
            tot = 0.0
    import torch as _t
    _t.save(model.state_dict(), R.MODELS / f"t2_{args.kind}{args.tag}.pt")
    print("saved", R.MODELS / f"t2_{args.kind}{args.tag}.pt", flush=True)


def probs_on(D, ws, tag, kind="rnn"):
    """(N, 6, n) gap-0 probabilities on pseudo windows (history ending at the origin row T, as t2_slotT.g0_probs)."""
    model = R.load_model(tag) if kind == "rnn" else C.load_model(tag, 96)
    pred = R.predict_probs if kind == "rnn" else C.predict_probs
    out = np.zeros((len(ws), HOR, D.n), np.float32)
    for s in range(0, len(ws), 64):
        xs = [C.make_x(D.V[w.day, w.T - 11:w.T + 1], D.Q[w.day, w.T - 11:w.T + 1], D.O[w.day, w.T - 11:w.T + 1],
                       D.static_ch, D.st["ffs"], D.st["cap"], D.vcut, w.T + 1, w.weekday) for w in ws[s:s + 64]]
        out[s:s + len(xs), :, D.perm] = pred(model, xs)
    return out


def cmd_eval(args) -> None:
    import pandas as pd
    import t2_g0_eval as E
    import t2_ongoing_v27 as V27
    import json
    cfg = json.loads(Path(args.config).read_text()) if args.config else None
    rows = []
    for panel in T2_PANELS:
        D = C.PanelData(panel, "validation", unmasked=False)
        ws = [w for w in P.windows(panel, "validation") if w.cond == "ongoing"]
        base, hy, g0, _ = E.load_all("validation", panel)
        assert list(base["day"]) == [w.day for w in ws]
        day = np.array([w.day for w in ws])
        pb = probs_on(D, ws, args.base)
        for spec in args.tags:
            tag, fold = spec.split(":")
            test = test_days(day, fold)
            pf = probs_on(D, ws, tag)
            for name, pr in (("base", pb), ("ft", pf)):
                for th in (.4, .45, .5):
                    s = V27.scores(base, pr >= th)[test]
                    rows.append(dict(panel=panel, fold=fold, model=name, kind=f"alone{th}", iou=s.mean(), n=len(s)))
                if cfg is not None:
                    # the production blend with this BiLSTM in place of the gap-0 seed whose tag is args.base
                    rk = [k for k in g0 if k.startswith("rnng0_")]
                    key = {"_g0": "rnng0_0"}.get(args.base, None)
                    gg = dict(g0)
                    if key is not None and name == "ft":
                        gg[key] = pr
                    cg = np.mean([gg[k] for k in g0 if k.startswith("cnng0_")], 0)
                    rg = np.mean([gg[k] for k in rk], 0)
                    w = dict(cfg["members"])
                    out = E.decide(hy, [gg["ml2g0"], cg, rg, (hy["rnn0"] + hy["rnn1"]) / 2],
                                   [w["ml2g0"], w["cnng0"], w["rnng0"], w["rnn2"]], panel, gg["catg0"],
                                   cfg["cat_alpha_default"])
                    s = V27.scores(base, out)[test]
                    rows.append(dict(panel=panel, fold=fold, model=name, kind="blend", iou=s.mean(), n=len(s)))
        print(panel, flush=True)
    df = pd.DataFrame(rows)
    t = df.pivot_table(index=["kind", "fold"], columns="model", values="iou", aggfunc="mean")
    t["delta"] = t["ft"] - t["base"]
    print(t.round(4).to_string())
    pt = df.pivot_table(index=["panel"], columns=["kind", "model"], values="iou", aggfunc="mean")
    print(pt.round(3).to_string())


def cmd_cache(args) -> None:
    """Cross-fitted March probabilities for blend selection, stored next to the gap-0 caches:
    work/cache/t2_slotT/ft_<panel>_validation.npz[name] = fold-b model on days < 15, fold-a model on days >= 15.
    With --all-tag also ft_<panel>_private.npz[name] from the model fine-tuned on all of March (record only)."""
    import t2_slotT as S
    for panel in T2_PANELS:
        for split in ("validation", "private") if args.all_tag else ("validation",):
            path = S.CACHE / f"ft_{panel}_{split}.npz"
            old = dict(np.load(path)) if path.exists() else {}
            D = C.PanelData(panel, split, unmasked=False)
            ws = [w for w in P.windows(panel, split) if w.cond == "ongoing"]
            if split == "validation":
                day = np.array([w.day for w in ws])
                pa, pb = probs_on(D, ws, args.tag_a, args.kind), probs_on(D, ws, args.tag_b, args.kind)
                old[args.name] = np.where((day >= SPLIT_DAY)[:, None, None], pa, pb)
            else:
                old[args.name] = probs_on(D, ws, args.all_tag, args.kind)
            tmp = path.with_name(path.stem + ".tmp.npz")   # write then rename: a killed run cannot corrupt the file
            np.savez_compressed(tmp, **old)
            tmp.replace(path)
            print(panel, split, args.name, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("finetune")
    s.add_argument("--base", default="_g0")
    s.add_argument("--fold", default="all", help="a | b | all | lt<N> (March day index < N)")
    s.add_argument("--source", choices=("march", "trainmask"), default="march")
    s.add_argument("--kind", choices=("rnn", "cnn"), default="rnn")
    s.add_argument("--tag", required=True)
    s.add_argument("--windows", type=int, default=24000)
    s.add_argument("--batch", type=int, default=32)
    s.add_argument("--mix", type=float, default=0.5, help="share of March windows in each batch")
    s.add_argument("--march-hide", type=float, default=0.2)
    s.add_argument("--lr", type=float, default=3e-4)
    s.add_argument("--iou-w", type=float, default=0.5)
    s.add_argument("--seed", type=int, default=0)
    s = sub.add_parser("eval")
    s.add_argument("--base", default="_g0")
    s.add_argument("--tags", nargs="+", required=True, help="tag:fold (fold the model was fine-tuned on)")
    s.add_argument("--config", default="work/results/t2_v48/config.json")
    s = sub.add_parser("cache")
    s.add_argument("--name", required=True)
    s.add_argument("--tag-a", required=True)
    s.add_argument("--tag-b", required=True)
    s.add_argument("--all-tag", default=None)
    s.add_argument("--kind", choices=("rnn", "cnn"), default="rnn")
    args = ap.parse_args()
    {"finetune": cmd_finetune, "eval": cmd_eval, "cache": cmd_cache}[args.cmd](args)
