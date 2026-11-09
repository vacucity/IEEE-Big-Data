"""V61 lane B3: second round on the Task-1 target cells INSIDE official Task-2 blackouts (the 128,465 state rows that
V60 = work/submissions/v60_loc_blk.zip took from V59_blk).  Only those rows may change against V60; every other state
row, every queue row and every ODME row stays byte-identical.

PRE-REGISTERED (this docstring is frozen into results/t1_v61_blk3/prereg.json by `prereg` BEFORE any number of this
lane is computed; `crossfit` refuses to run if the docstring changed afterwards):
  ruler      the V57 ruler exactly as V59 used it: the 977 simulated official-like blackouts on the March / April visible
             layers (results/t1_v57_blkruler), the real-frame emulation of t1_v57_blackout (window_sums / real_frame /
             deltas: official blackout target counts, condition mix, mask rates), dTotal = 0.35 dS_state +
             0.15 dS_phys(est) against the production fill, family mean over the ten corridors (D12_I405_* = 0)
  reference  S2 = V59_blk = phase-A specialist + residual GBM, lambda (0.625, 0.5); its day-parity cross-fitted ruler
             fill results/t1_v59_blkspec/crossfit_chosen_fill.npz (re-evaluated here; must reproduce the S2 rows of
             crossfit_corridors.csv)
  GBMs       every residual GBM: t1_v57_blackout GBM_PARAMS / GBM_ROUNDS (63 leaves, lr .05, 400 rounds, deterministic),
             day-parity cross-fit pooling both months and all eight T2 corridors (fit parity a, predict parity 1 - a);
             S2 feature set = t1_v57_blackout.FEATS (computed from the candidate's own U-Net columns) + pv0_r, pq0_l
             (production fill) + dsp_v_r, dsp_q_l (candidate U-Net mean - production); fill = base + lambda * residual
  lambda     for E2..E5 the V57 rule on LAM_V x LAM_Q: eligible = family dTotal > 0 and dS > 0 in both months and
             dTotal > 0 in >= 7/8 corridors per month; pick the smallest lambda_v, then lambda_q, among the eligible
             configurations within 95% of the best eligible mean-over-months dTotal; no eligible configuration ->
             the candidate drops out
  candidates fixed set (nothing else is evaluated for selection):
    E1  0.5 x S2 fill + 0.5 x S4 fill (speed and flow).  S4 = S4_v0.5_q0.0 of V59 phase B (the V57-rule pick in
        crossfit_ss_choice.json: month-self-supervised nets of the other day parity + residual GBM with the S2 feature
        set, lambda (0.5, 0.0)); its cross-fitted fill is recomputed here and must reproduce the S4_v0.5_q0.0 rows of
        crossfit_ss_corridors.csv.  No free parameter.
    E2  base = phase-A specialist mean (as S2); residual GBM with the S2 feature set + SS_FEATS = dss_v_r =
        (pv_ss - pv_spec) / ffs, dss_q_l = (pq_ss - pq_spec) / lanes (pv_ss / pq_ss = S3, the self-supervised nets of
        the other day parity); lambda by the V57 rule.
    E3  8-net specialist: the 4 phase-A nets + 4 new nets "s1" trained with the phase-A recipe (train_runs.csv of
        t1_v59_blkspec, fold months, 8 epochs, lr 2e-4 OneCycle pct .1 div 10/10, AdamW wd 1e-4, batch 4, P_FULL .5,
        W_OTHER .2, lwr .05 for the lwr tag, bf16) with new seeds (61 + 10 fold + tag index).  Fix of the V59 OneCycle
        total-steps bug: every epoch's batch list is drawn up front from its own rng, total_steps = their exact count.
        Per-net columns averaged over the seeds (fold columns = mean over seeds of that fold, tag = mean of folds,
        pv / pq = mean of tags); S2 feature set from the ensemble; lambda by the V57 rule.
    E4  as E3, but the 4 new nets "ext" (seeds 62 + 10 fold + tag index) are trained on an enlarged train-gap sample:
        the replica selector (t2_v52_replica.Timeline rules on the train truth layer, cap 5, onset rule not_est) run
        as greedy forward passes from slot s*288 + o for every start day s and o in OFFSETS = 0, 48, 96, 144, 192, 240
        (the original windows.csv used o = 0 only); unique (panel, day, T, cond) origins with 12 <= T <= 269; all
        original gaps kept, new gaps drawn uniformly at random (seed 61) up to EXT_FACTOR = 2 x the original count;
        packed per day into runs (gaps >= 48 slots apart) as t1_v59_blkspec.pack; same 8 epochs.
    E5  12-net specialist: phase-A + s1 + ext nets; rest as E3.
  win rule   (same as V59 phase B, reference S2): a candidate wins iff in BOTH months its family dTotal > S2's AND its
             corridor dTotal > S2's in >= 7 of the 8 T2 corridors in each month
  choice     winners in decreasing mean-over-months family dTotal; the first one whose block-halves cross-fit
             (GBMs fitted on days < 15 vs >= 15; S2 refitted the same way; E1 = mean of the block-halves S2 and S4
             fills) gain over S2 is > 0 in both months is chosen.  No winner (or none passes) -> nothing is built.
             The rule is not relaxed after seeing numbers.
  report     (not used for the choice) bootstrap of 5 onset + 5 ongoing windows per corridor-month vs S2 (300 draws),
             MSE by condition, per corridor-month dS_state (measured, simulation protocol) and dS_phys / dTotal
             (estimated), train held-out RMSE of the new nets (official layout)
  packaging  official windows: the chosen candidate's nets on the official blackout days (other-parity ss nets as in
             V59 phase B), GBM(s) refitted on all ruler rows of both months; start from the state rows of V60
             (v60_loc_blk.zip), replace ONLY the Task-1 targets inside official Task-2 blackouts (%.6g); state-only zip
             v61b3_state.zip on the v51_april.zip base (t1_v52_monthbias package/audit code, names patched); then
             build_queue_candidate.py --name v61_b3 with the V54 onset queues; line-by-line comparison with V60.

Steps (project root, anaconda python; nothing existing is overwritten - FileExistsError):
  prereg                          freeze this docstring
  runs_ext                        E4 enlarged train-gap sample -> results/t1_v61_blk3/cache/train_runs_ext.csv
  train --set s1|ext --tag T --fold k        -> work/models/t1_unet_fold{k}{T}_v61b3_{set}.pt
  wide --set s1|ext --split S     ruler outputs of the new nets (rows/cells of t1_v59_blkspec wide)
  official --set s1|ext           official-window outputs of the new nets
  crossfit                        the pre-registered comparison -> crossfit_{summary,corridors}.csv, crossfit_choice.json
  heldout                         report only: train held-out months, official layout
  fit / apply / package / audit / compare     only if crossfit chose a candidate

Rules: Task 1 is offline.  Inputs: train-month truth (training only), the visible observations of March / April
(ruler labels, GBM fitting, the V59 self-supervised nets), the public window_index.csv (origin + condition), frozen
production / V59 models.  No hidden label of March / April, no Task-2 prediction, no public-LB feedback.
"""
from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "4")

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import t1_unet as U  # noqa: E402
import t1_v59_blkspec as B  # noqa: E402
from evaluate import CLEAN, STEPS, T2_PANELS  # noqa: E402

WORK = Path(__file__).resolve().parent
OUT = WORK / "results" / "t1_v61_blk3"
CACHE = OUT / "cache"
MODELS = WORK / "models"
SPLITS = B.SPLITS
TAGS = B.TAGS
GAP, HIST = B.GAP, B.HIST
NET_SFX = "_v61b3"
SETS = {"s1": dict(seed=61, runs="phase_a"), "ext": dict(seed=62, runs="ext")}
OFFSETS = (0, 48, 96, 144, 192, 240)
EXT_FACTOR = 2.0
EXT_SEED = 61
SS_FEATS = ["dss_v_r", "dss_q_l"]
REF_CFG = "S2_v0.625_q0.5"
REF_LAM = (0.625, 0.5)
S4_CFG = "S4_v0.5_q0.0"
S4_LAM = (0.5, 0.0)
CANDS = {"E1": None, "E2": None, "E3": ("A", "s1"), "E4": ("A", "ext"), "E5": ("A", "s1", "ext")}
NETCOLS = [f"{c}{t}f{f}" for t in TAGS for c in "vq" for f in (0, 1)]
V60_ZIP = WORK / "submissions" / "v60_loc_blk.zip"
V51_ZIP = WORK / "submissions" / "v51_april.zip"
NAME = "v61b3_state"
FULL_NAME = "v61_b3"
CSV = {s: WORK / "sub" / f"state_{s}_v61b3.csv" for s in SPLITS}


def write_new(path: Path, text: str) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.write_text(text, encoding="utf-8")


def sha_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def check_prereg() -> dict:
    pr = json.loads((OUT / "prereg.json").read_text(encoding="utf-8"))
    assert pr["docstring_sha256"] == sha_text(__doc__), "the pre-registered docstring was changed"
    return pr


def cmd_prereg(args) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    write_new(OUT / "prereg.json", json.dumps({
        "written": time.strftime("%Y-%m-%d %H:%M:%S"), "docstring": __doc__, "docstring_sha256": sha_text(__doc__),
        "constants": {"OFFSETS": OFFSETS, "EXT_FACTOR": EXT_FACTOR, "EXT_SEED": EXT_SEED, "SETS": SETS,
                      "SS_FEATS": SS_FEATS, "REF_CFG": REF_CFG, "REF_LAM": REF_LAM, "S4_CFG": S4_CFG,
                      "S4_LAM": S4_LAM, "CANDS": {k: v for k, v in CANDS.items()}},
        "note": "written before any number of lane B3 was computed"}, indent=2))
    print("prereg written", OUT / "prereg.json", sha_text(__doc__))


# ---------------------------------------------------------------------------
# E4: enlarged train-gap sample
# ---------------------------------------------------------------------------
def select_from(tl, t0: int, cap: int = 5):
    """t2_v52_replica.Timeline.select (onset rule not_est) started at slot t0 instead of a day's first slot."""
    import t2_v52_replica as RP
    cnt = {"onset": 0, "ongoing": 0}
    last, out = None, []
    for t in range(max(RP.HIST, t0), tl.n - RP.HOR):
        if cnt["onset"] >= cap and cnt["ongoing"] >= cap:
            break
        if last is not None and t - last < RP.SPACING:
            continue
        if tl.blk[t] or tl.cov[t] < RP.MIN_COV:
            continue
        if tl.est[t]:
            if cnt["ongoing"] >= cap or not tl.QL[t + 1:t + RP.HOR + 1].any():
                continue
            if tl.piou(t) > RP.MAX_PIOU:
                continue
            cond = "ongoing"
        else:
            if cnt["onset"] >= cap or not (tl.QV[t + 1:t + RP.HOR + 1] & tl.E[t + 1:t + RP.HOR + 1]).any():
                continue
            cond = "onset"
        cnt[cond] += 1
        out.append((t, cond))
        last = t
    return out


def cmd_runs_ext(args) -> None:
    import t2_v52_replica as RP
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / "train_runs_ext.csv"
    if path.exists():
        raise FileExistsError(path)
    t0 = time.time()
    R0 = pd.read_csv(B.CACHE / "train_runs.csv")
    orig = {(r.panel, int(r.day), T, c) for r in R0.itertuples() for T, c in B.parse_gaps(r.gaps)}
    new, off0, stats = set(), set(), []
    metas = {}
    for p in T2_PANELS:
        lay = RP.layers(p, "train")
        meta = json.loads((U.DATA / f"{p}_train_meta.json").read_text())
        metas[p] = meta
        di = {d: i for i, d in enumerate(meta["dates"])}
        for m in sorted({str(d)[:7] for d in lay["dates"]}):
            tl = RP.Timeline(p, "train", m, lay)
            for s in range(0, len(tl.dates) - 2):
                for o in OFFSETS:
                    for t, c in select_from(tl, s * STEPS + o):
                        T = t % STEPS
                        if not (HIST <= T <= STEPS - GAP - 1):
                            continue
                        key = (p, di[tl.dates[t // STEPS]], T, c)
                        if o == 0:
                            off0.add(key)
                        if key not in orig:
                            new.add(key)
        stats.append(dict(panel=p, orig=sum(k[0] == p for k in orig), new=sum(k[0] == p for k in new)))
        print(p, stats[-1], f"{time.time() - t0:.0f}s", flush=True)
    assert off0 == orig, ("offset-0 passes do not reproduce train_runs.csv", len(off0), len(orig),
                          len(off0 - orig), len(orig - off0))
    new = sorted(new)
    rng = np.random.default_rng(EXT_SEED)
    n_add = min(len(new), int(round((EXT_FACTOR - 1) * len(orig))))
    pick = sorted(rng.choice(len(new), size=n_add, replace=False).tolist())
    keys = sorted(orig) + [new[i] for i in pick]
    K = pd.DataFrame(keys, columns=["panel", "day", "T", "cond"])
    K["is_new"] = [k not in orig for k in keys]
    rows = []
    for (p, day), g in K.groupby(["panel", "day"]):
        meta = metas[p]
        for ir, run in enumerate(B.pack(list(zip(g["T"].astype(int), g.cond)))):
            rows.append(dict(panel=p, day=int(day), date=meta["dates"][day], month=meta["month"][day],
                             regime=meta["regime"][day], sub=ir, gaps=";".join(f"{T}:{c}" for T, c in run),
                             n_gaps=len(run)))
    R = pd.DataFrame(rows)
    R.to_csv(path, index=False)
    K.to_csv(CACHE / "train_gaps_ext.csv", index=False)
    S = pd.DataFrame(stats)
    S["kept_new"] = S.panel.map(K[K.is_new].groupby("panel").size()).fillna(0).astype(int)
    S["runs_orig"] = S.panel.map(R0.groupby("panel").size())
    S["runs_ext"] = S.panel.map(R.groupby("panel").size())
    write_new(CACHE / "train_runs_ext_stats.json", json.dumps({
        "orig_gaps": len(orig), "new_unique_gaps": len(new), "added": n_add, "ext_gaps": len(K),
        "orig_runs": len(R0), "ext_runs": len(R), "cond_orig": K[~K.is_new].cond.value_counts().to_dict(),
        "cond_added": K[K.is_new].cond.value_counts().to_dict(), "per_panel": S.to_dict("records"),
        "seconds": round(time.time() - t0)}, indent=2))
    print(S.to_string(), flush=True)
    print("orig gaps", len(orig), "new unique", len(new), "added", n_add, "runs", len(R0), "->", len(R), flush=True)


# ---------------------------------------------------------------------------
# training of the new specialist nets (phase-A recipe, OneCycle total-steps fix)
# ---------------------------------------------------------------------------
def net_path(set_: str, tag: str, fold: int) -> Path:
    return MODELS / f"t1_unet_fold{fold}{tag}{NET_SFX}_{set_}.pt"


def set_runs(set_: str) -> pd.DataFrame:
    if SETS[set_]["runs"] == "phase_a":
        return pd.read_csv(B.CACHE / "train_runs.csv")
    return pd.read_csv(CACHE / "train_runs_ext.csv")


def cmd_train(args) -> None:
    import torch
    set_, tag, fold = args.set, args.tag, args.fold
    out = net_path(set_, tag, fold)
    if out.exists():
        raise FileExistsError(out)
    seed = SETS[set_]["seed"] + 10 * fold + TAGS.index(tag)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    rng_b = np.random.default_rng(seed)               # batch lists
    rng_s = np.random.default_rng(seed + 5000)        # sample draws (history un-hiding)
    R = set_runs(set_)
    R = R[R.month % 2 != fold].reset_index(drop=True)
    epoch_batches = []
    for _ in range(args.epochs):
        batches = []
        for p, g in R.groupby("panel"):
            idx = rng_b.permutation(g.index.to_numpy())
            batches += [(p, idx[k:k + args.batch]) for k in range(0, len(idx), args.batch)]
        epoch_batches.append([batches[i] for i in rng_b.permutation(len(batches))])
    total = sum(len(b) for b in epoch_batches)
    srcs = {p: U.open_split(p, "train") for p in T2_PANELS}
    sts = {p: U.static(p) for p in T2_PANELS}
    profs = {p: np.load(U.DATA / f"{p}_profiles.npy") for p in T2_PANELS}
    occs = {p: np.load(B.CACHE / f"occ_{p}_train.npy", mmap_mode="r") for p in T2_PANELS}
    dev = "cuda"
    model = B.init_from_production(tag, fold, dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=total, pct_start=0.1,
                                                anneal_strategy="cos", div_factor=10, final_div_factor=10)
    lwr = 0.05 if tag == "_w48lwr" else 0.0
    print(f"set {set_} fold {fold} {tag} seed {seed}: {len(R)} runs, {int(R.n_gaps.sum())} gaps, total steps {total}, "
          f"lr {args.lr}, epochs {args.epochs}, lwr {lwr}", flush=True)
    steps = 0
    log = []
    for ep, batches in enumerate(epoch_batches):
        t0, tot, cnt = time.time(), 0.0, 0
        model.train()
        for p, idx in batches:
            st = sts[p]
            xs, bs, ts, ws, es = [], [], [], [], []
            for j in idx:
                r = R.loc[j]
                d = U.Day(p, "train", int(r.day), srcs[p], st, profs[p])
                x, b, tr, w, ok = B.train_sample(d, B.parse_gaps(r.gaps), rng_s,
                                                 np.asarray(occs[p][int(r.day)], np.float32), args.p_full)
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
            steps += 1
            tot += float(loss)
            cnt += 1
        log.append({"epoch": ep + 1, "loss": tot / cnt, "sec": round(time.time() - t0)})
        print(f"  epoch {ep + 1}/{args.epochs}: loss {tot / cnt:.5f} ({time.time() - t0:.0f}s)", flush=True)
    assert steps == total, (steps, total)
    MODELS.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out)
    print("saved", out, "steps", steps, flush=True)


def load_net(set_: str, tag: str, fold: int, device: str = "cuda"):
    import torch
    m = B.build_model2()
    m.load_state_dict(torch.load(net_path(set_, tag, fold), map_location="cpu"))
    m.eval()
    return m.to(device)


# ---------------------------------------------------------------------------
# outputs of the new nets: ruler windows and official windows (same rows as t1_v59_blkspec wide / official)
# ---------------------------------------------------------------------------
def _records(grid, day, T, n, run=None):
    ks = np.arange(-6, GAP + 7)
    tt = T + ks
    okr = (tt >= 0) & (tt < STEPS)
    ks, tt = ks[okr], tt[okr]
    ti, li = np.repeat(tt, n), np.tile(np.arange(n), len(tt))
    rec = {"day": np.full(len(ti), day, np.int16)}
    if run is not None:
        rec["run"] = np.full(len(ti), run, np.int32)
    rec.update({"T": np.full(len(ti), T, np.int16), "k": np.repeat(ks, n).astype(np.int8), "pos": li.astype(np.int16)})
    for nm, a in grid.items():
        rec[nm] = a[ti, li].astype(np.float32)
    return pd.DataFrame(rec)


def cmd_wide(args) -> None:
    import t1_v57_blkruler as RL
    _, OW = B.official_gaps()
    nets = {(tag, f): load_net(args.set, tag, f) for tag in TAGS for f in (0, 1)}
    out_dir = OUT / f"wide_{args.set}"
    out_dir.mkdir(parents=True, exist_ok=True)
    W = pd.read_csv(B.RULER / f"origins_{args.split}.csv")
    W = W[~W.blk_wide]
    for p in T2_PANELS:
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
            gaps = [(int(r["T"]), r["cond"]) for r in run] + B.day_official(OW, p, args.split, day)
            grid = B.grid_outputs(nets, d, gaps, vis & rows[:, None], st, inv)
            for r in run:
                recs.append(_records(grid, day, int(r["T"]), n, ir))
        pd.concat(recs, ignore_index=True).to_parquet(path, index=False)
        print(p, args.split, "done", f"{time.time() - t0:.0f}s", flush=True)


def cmd_official(args) -> None:
    _, OW = B.official_gaps()
    nets = {(tag, f): load_net(args.set, tag, f) for tag in TAGS for f in (0, 1)}
    out_dir = OUT / f"official_{args.set}"
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
                grid = B.grid_outputs(nets, d, gaps, None, st, inv)
                for T, _ in gaps:
                    recs.append(_records(grid, day, T, n))
            pd.concat(recs, ignore_index=True).to_parquet(path, index=False)
            print(p, sp, "official done", flush=True)


def set_dir(set_: str, kind: str) -> Path:
    """kind 'wide' (ruler) or 'official'; set 'A' = phase-A specialist (t1_v59_blkspec), else a new set."""
    if set_ == "A":
        return B.WIDE if kind == "wide" else B.OFFICIAL
    return OUT / f"{kind}_{set_}"


def ens_columns(dfs) -> dict:
    """Equal-weight ensemble over sets (each set = 2 tags x 2 folds): fold columns = mean over sets, tag = mean of
    folds, pv / pq = mean of tags (the production / phase-A convention)."""
    out = {c: np.mean([d[c].to_numpy(np.float64) for d in dfs], axis=0) for c in NETCOLS}
    for t in TAGS:
        for c in "vq":
            out[f"{c}{t}"] = 0.5 * (out[f"{c}{t}f0"] + out[f"{c}{t}f1"])
    out["pv"] = 0.5 * (out["v_w48"] + out["v_w48lwr"])
    out["pq"] = 0.5 * (out["q_w48"] + out["q_w48lwr"])
    return {k: v.astype(np.float32) for k, v in out.items()}


def ens_features(p: str, sp: str, sets, name: str) -> pd.DataFrame:
    import t1_v57_blackout as V
    fdir = OUT / f"features_{name}"
    path = fdir / f"{p}_{sp}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    base = pd.read_parquet(B.RULER / "sim_wide" / f"{p}_{sp}_win.parquet", columns=B.STATIC_COLS)
    dfs = []
    for s in sets:
        d = pd.read_parquet(set_dir(s, "wide") / f"{p}_{sp}_win.parquet", columns=["day", "run", "T", "k", "pos"] + NETCOLS)
        assert len(d) == len(base)
        for c in ("day", "run", "T", "k", "pos"):
            assert np.array_equal(base[c].to_numpy(), d[c].to_numpy()), (p, sp, s, c)
        dfs.append(d)
    for c, a in ens_columns(dfs).items():
        base[c] = a
    del dfs
    A, meta = V.win_tensors(p, base)
    F = V.gap_features(p, A, meta)
    fdir.mkdir(parents=True, exist_ok=True)
    F.to_parquet(path, index=False)
    return F


# ---------------------------------------------------------------------------
# ruler: production frame, candidate feature matrices, evaluation
# ---------------------------------------------------------------------------
KEEP0 = ["w", "wi", "k", "pos", "day", "regime", "condname", "ov", "oq", "pv", "pq", "ffs", "lanes"]


def load_prod():
    import t1_v57_blackout as V
    P, TEN = [], {}
    for p in T2_PANELS:
        for sp in SPLITS:
            F0 = V.sim_features(p, sp)[KEEP0].copy()
            F0["panel"], F0["split"] = p, sp
            P.append(F0)
            TEN[p, sp] = B.slim_tensors(p, sp)
    A0 = pd.concat(P, ignore_index=True)
    A0["panel"] = A0.panel.astype("category")
    A0["split"] = A0.split.astype("category")
    A0["par"] = (A0.day % 2).astype(np.int8)
    A0["half"] = (A0.day >= 15).astype(np.int8)
    return A0, TEN


def cand_frame(kind: str, A0: pd.DataFrame, sets=None, name=None) -> pd.DataFrame:
    """Feature frame of a U-Net fill aligned with A0: kind 'A' (phase-A specialist), 'ss' (V59 phase-B S3 nets of
    the other parity), 'ens' (ensemble of `sets`).  Adds the 4 S2 extra features relative to production."""
    import t1_v57_blackout as V
    parts = []
    for p in T2_PANELS:
        for sp in SPLITS:
            if kind == "A":
                pth = B.OUT / "features" / f"{p}_{sp}.parquet"
                assert pth.exists(), pth
                F = pd.read_parquet(pth)
            elif kind == "ss":
                pth = B.OUT / "features_ss" / f"{p}_{sp}.parquet"
                assert pth.exists(), pth
                F = pd.read_parquet(pth)
            else:
                F = ens_features(p, sp, sets, name)
            parts.append(F[list(dict.fromkeys(V.FEATS + ["w", "k", "pos", "ov", "oq", "pv", "pq"]))].copy())
    A1 = pd.concat(parts, ignore_index=True)
    for c in ("w", "k", "pos", "ov", "oq"):
        assert np.array_equal(A0[c].to_numpy(), A1[c].to_numpy(), equal_nan=True), (kind, c)
    ffs, lanes = A0.ffs.to_numpy(np.float32), A0.lanes.to_numpy(np.float32)
    A1["pv0_r"] = A0.pv.to_numpy(np.float32) / ffs
    A1["pq0_l"] = A0.pq.to_numpy(np.float32) / lanes
    A1["dsp_v_r"] = (A1.pv.to_numpy(np.float32) - A0.pv.to_numpy(np.float32)) / ffs
    A1["dsp_q_l"] = (A1.pq.to_numpy(np.float32) - A0.pq.to_numpy(np.float32)) / lanes
    for c in A1.columns:
        if A1[c].dtype == np.float64:
            A1[c] = A1[c].astype(np.float32)
    return A1


def gbm_cross(X, feats, A0, base_v, base_q, split_col):
    import lightgbm as lgb
    import t1_v57_blackout as V
    lanes = A0.lanes.to_numpy(float)
    yv = A0.ov.to_numpy(float) - base_v
    yq = (A0.oq.to_numpy(float) - base_q) / lanes
    g = A0[split_col].to_numpy()
    pv_, pq_ = np.full(len(A0), np.nan), np.full(len(A0), np.nan)
    for a in (0, 1):
        fit, app = g == a, g != a
        mv = lgb.train(V.GBM_PARAMS, lgb.Dataset(X[fit], yv[fit], feature_name=feats, free_raw_data=True), V.GBM_ROUNDS)
        pv_[app] = mv.predict(X[app], num_threads=4)
        mq = lgb.train(V.GBM_PARAMS, lgb.Dataset(X[fit], yq[fit], feature_name=feats, free_raw_data=True), V.GBM_ROUNDS)
        pq_[app] = mq.predict(X[app], num_threads=4)
    return pv_, pq_


def mk_fill(bv, bq, lanes, rv, rq, lv, lq):
    import t1_v57_blackout as V
    return (np.clip(bv + lv * rv, V.SPEED_MIN, V.SPEED_MAX), np.maximum(bq + lq * rq * lanes, 0.0))


class Ruler:
    def __init__(self):
        import t1_v57_blackout as V
        self.V = V
        self.FR, self.OW = V.load_frame()
        self.A0, self.TEN = load_prod()
        self.lanes = self.A0.lanes.to_numpy(float)
        self.masks = {(p, sp): ((self.A0.panel == p) & (self.A0.split == sp)).to_numpy() for (p, sp) in self.TEN}
        self.base_ws = {k: V.window_sums(k[0], A, self.A0[self.masks[k]], self.A0.pv.to_numpy(float)[self.masks[k]],
                                         self.A0.pq.to_numpy(float)[self.masks[k]]) for k, A in self.TEN.items()}
        self.sub = {k: self.A0[m] for k, m in self.masks.items()}

    def ws(self, v, q):
        return {k: self.V.window_sums(k[0], A, self.sub[k], v[self.masks[k]], q[self.masks[k]])
                for k, A in self.TEN.items()}

    def eval(self, v, q, cfg=""):
        V = self.V
        rows = []
        for (p, sp), w1 in self.ws(v, q).items():
            tot, per = V.deltas(V.real_frame(p, sp, self.base_ws[p, sp], self.FR, self.OW),
                                V.real_frame(p, sp, w1, self.FR, self.OW))
            r1 = V.real_frame(p, sp, w1, self.FR, self.OW)
            rows.append({"cfg": cfg, "panel": p, "split": sp, **tot,
                         "rmse_v_blk": float(np.sqrt(w1.sv.sum() / w1.n.sum())),
                         "rmse_q_blk": float(np.sqrt(w1.sq.sum() / w1.n.sum()))})
        return pd.DataFrame(rows)


def fam(R: pd.DataFrame, sp: str, col: str) -> float:
    import t1_v57_blackout as V
    g = R[R.split == sp]
    return V.family_mean(dict(zip(g.panel, g[col])))


def versus(R: pd.DataFrame, Rr: pd.DataFrame) -> dict:
    """Candidate R vs reference Rr (both dTotal vs the production fill per corridor-month)."""
    m = R.merge(Rr, on=["panel", "split"], suffixes=("", "_r"))
    out = {}
    for sp in SPLITS:
        g = m[m.split == sp]
        out[f"dTotal_{sp}"] = fam(m, sp, "dTotal")
        out[f"dS_{sp}"] = fam(m, sp, "dS")
        out[f"dphys_{sp}"] = fam(m, sp, "dphys")
        out[f"gain_vs_ref_{sp}"] = out[f"dTotal_{sp}"] - fam(m, sp, "dTotal_r")
        out[f"gainS_vs_ref_{sp}"] = out[f"dS_{sp}"] - fam(m, sp, "dS_r")
        out[f"n_beat_ref_{sp}"] = int((g.dTotal > g.dTotal_r).sum())
        out[f"n_pos_{sp}"] = int((g.dTotal > 0).sum())
    out["dTotal_both"] = (out["dTotal_validation"] + out["dTotal_private"]) / 2
    out["gain_vs_ref_both"] = (out["gain_vs_ref_validation"] + out["gain_vs_ref_private"]) / 2
    out["wins"] = bool(all(out[f"gain_vs_ref_{s}"] > 0 and out[f"n_beat_ref_{s}"] >= 7 for s in SPLITS))
    return out


def elig_v57(R: pd.DataFrame) -> bool:
    return bool(all(fam(R, s, "dTotal") > 0 and fam(R, s, "dS") > 0 and int((R[R.split == s].dTotal > 0).sum()) >= 7
                    for s in SPLITS))


def lam_grid(ru: Ruler, cand, bv, bq, rv, rq, Rref, rows, cm):
    import t1_v57_blackout as V
    for lv in V.LAM_V:
        for lq in V.LAM_Q:
            cfg = f"{cand}_v{lv}_q{lq}"
            R = ru.eval(*mk_fill(bv, bq, ru.lanes, rv, rq, lv, lq), cfg)
            cm.append(R)
            rows.append({"cand": cand, "cfg": cfg, "lam_v": lv, "lam_q": lq, **versus(R, Rref), "elig_v57rule": elig_v57(R)})
    G = pd.DataFrame([r for r in rows if r["cand"] == cand])
    el = G[G.elig_v57rule]
    if el.empty:
        return None
    best = el.dTotal_both.max()
    return el[el.dTotal_both >= 0.95 * best].sort_values(["lam_v", "lam_q"]).cfg.iloc[0]


def reproduce(R: pd.DataFrame, path: Path, cfg: str, tol: float = 1e-9) -> float:
    C = pd.read_csv(path)
    C = C[C.cfg == cfg]
    m = R.merge(C, on=["panel", "split"], suffixes=("", "_old"))
    assert len(m) == len(R) == 16, (cfg, len(m))
    d = float(np.max(np.abs(m.dTotal - m.dTotal_old)))
    assert d < tol, (cfg, d)
    return d


def cmd_crossfit(args) -> None:
    import t1_v57_blackout as V
    pr = check_prereg()
    files = [OUT / f"crossfit{args.suffix}_{x}" for x in ("summary.csv", "corridors.csv", "choice.json")]
    for f in files:
        if f.exists():
            raise FileExistsError(f)
    t0 = time.time()
    ru = Ruler()
    A0, lanes = ru.A0, ru.lanes
    print("ruler cells", len(A0), f"{time.time() - t0:.0f}s", flush=True)
    feats_s2 = V.FEATS + B.EXTRA_FEATS
    # reference S2 (V59_blk) and its reproduction
    za = np.load(B.OUT / "crossfit_chosen_fill.npz")
    ref_v, ref_q = za["v"], za["q"]
    Rref = ru.eval(ref_v, ref_q, "S2")
    repro = {"S2": reproduce(Rref, B.OUT / "crossfit_corridors.csv", REF_CFG)}
    cm, rows = [Rref], []
    rows.append({"cand": "S2", "cfg": "S2", "lam_v": REF_LAM[0], "lam_q": REF_LAM[1], **versus(Rref, Rref),
                 "elig_v57rule": elig_v57(Rref)})
    print("S2 reproduced", repro, f"{time.time() - t0:.0f}s", flush=True)
    fills, picks, preds = {}, {}, {}
    # phase-A and ss frames
    A1 = cand_frame("A", A0)
    A3 = cand_frame("ss", A0)
    s1v, s1q = A1.pv.to_numpy(float), A1.pq.to_numpy(float)
    s3v, s3q = A3.pv.to_numpy(float), A3.pq.to_numpy(float)
    # S4 recomputed (component of E1)
    X3 = A3[feats_s2].to_numpy(np.float32)
    r4v, r4q = gbm_cross(X3, feats_s2, A0, s3v, s3q, "par")
    del X3
    s4 = mk_fill(s3v, s3q, lanes, r4v, r4q, *S4_LAM)
    R4 = ru.eval(*s4, "S4")
    repro["S4"] = reproduce(R4, B.OUT / "crossfit_ss_corridors.csv", S4_CFG)
    cm.append(R4)
    rows.append({"cand": "S4", "cfg": "S4", "lam_v": S4_LAM[0], "lam_q": S4_LAM[1], **versus(R4, Rref),
                 "elig_v57rule": elig_v57(R4)})
    print("S4 reproduced", repro, f"{time.time() - t0:.0f}s", flush=True)
    # E1
    e1 = (0.5 * (ref_v + s4[0]), 0.5 * (ref_q + s4[1]))
    R = ru.eval(*e1, "E1")
    cm.append(R)
    rows.append({"cand": "E1", "cfg": "E1", "lam_v": np.nan, "lam_q": np.nan, **versus(R, Rref), "elig_v57rule": elig_v57(R)})
    picks["E1"], fills["E1"] = "E1", e1
    print("E1", {k: v for k, v in rows[-1].items() if k.startswith(("gain", "n_beat", "wins"))}, flush=True)
    # E2
    A1["dss_v_r"] = ((s3v - s1v) / A0.ffs.to_numpy(float)).astype(np.float32)
    A1["dss_q_l"] = ((s3q - s1q) / lanes).astype(np.float32)
    f2 = feats_s2 + SS_FEATS
    X = A1[f2].to_numpy(np.float32)
    rv, rq = gbm_cross(X, f2, A0, s1v, s1q, "par")
    del X
    preds["E2"] = (s1v, s1q, rv, rq, f2)
    picks["E2"] = lam_grid(ru, "E2", s1v, s1q, rv, rq, Rref, rows, cm)
    print("E2 pick", picks["E2"], f"{time.time() - t0:.0f}s", flush=True)
    del A3
    # E3..E5
    for cand in ("E3", "E4", "E5"):
        Ac = cand_frame("ens", A0, CANDS[cand], cand.lower())
        bv, bq = Ac.pv.to_numpy(float), Ac.pq.to_numpy(float)
        X = Ac[feats_s2].to_numpy(np.float32)
        del Ac
        rv, rq = gbm_cross(X, feats_s2, A0, bv, bq, "par")
        del X
        preds[cand] = (bv, bq, rv, rq, feats_s2)
        R = ru.eval(bv, bq, f"{cand}_unet")
        cm.append(R)
        rows.append({"cand": f"{cand}_unet", "cfg": f"{cand}_unet", "lam_v": 0.0, "lam_q": 0.0, **versus(R, Rref),
                     "elig_v57rule": elig_v57(R)})
        picks[cand] = lam_grid(ru, cand, bv, bq, rv, rq, Rref, rows, cm)
        print(cand, "pick", picks[cand], f"{time.time() - t0:.0f}s", flush=True)
    for cand in ("E2", "E3", "E4", "E5"):
        if picks[cand] is not None:
            bv, bq, rv, rq, _ = preds[cand]
            g = [r for r in rows if r["cfg"] == picks[cand]][0]
            fills[cand] = mk_fill(bv, bq, lanes, rv, rq, g["lam_v"], g["lam_q"])
    G = pd.DataFrame(rows)
    picks = {k: v for k, v in picks.items() if v is not None}
    P = G.set_index("cfg").loc[list(picks.values())].copy()
    P["cand"] = list(picks.keys())
    winners = P[P.wins].sort_values("dTotal_both", ascending=False)
    pd.set_option("display.width", 250)
    print(G.round(7).to_string(), flush=True)
    print("picks", picks, "winners", list(winners.cand), flush=True)
    # gate: block-halves cross-fit vs S2 refitted the same way
    rob, chosen = {}, None
    if len(winners):
        X = A1[feats_s2].to_numpy(np.float32)
        hv, hq = gbm_cross(X, feats_s2, A0, s1v, s1q, "half")
        del X
        ref_h = mk_fill(s1v, s1q, lanes, hv, hq, *REF_LAM)
        Rref_h = ru.eval(*ref_h, "S2_half")
        for cfg, r in winners.iterrows():
            cand = r.cand
            if cand == "E1":
                A3 = cand_frame("ss", A0)
                X = A3[feats_s2].to_numpy(np.float32)
                del A3
                h4v, h4q = gbm_cross(X, feats_s2, A0, s3v, s3q, "half")
                del X
                s4h = mk_fill(s3v, s3q, lanes, h4v, h4q, *S4_LAM)
                fh = (0.5 * (ref_h[0] + s4h[0]), 0.5 * (ref_h[1] + s4h[1]))
            else:
                bv, bq, _, _, ff = preds[cand]
                if cand == "E2":
                    X = A1[ff].to_numpy(np.float32)
                else:
                    Ac = cand_frame("ens", A0, CANDS[cand], cand.lower())
                    X = Ac[ff].to_numpy(np.float32)
                    del Ac
                hv2, hq2 = gbm_cross(X, ff, A0, bv, bq, "half")
                del X
                fh = mk_fill(bv, bq, lanes, hv2, hq2, r.lam_v, r.lam_q)
            Rh = ru.eval(*fh, f"{cand}_half")
            cm += [Rh]
            rob[f"block_halves_{cand}"] = versus(Rh, Rref_h)
            ok = all(rob[f"block_halves_{cand}"][f"gain_vs_ref_{s}"] > 0 for s in SPLITS)
            print("gate", cand, ok, rob[f"block_halves_{cand}"], f"{time.time() - t0:.0f}s", flush=True)
            if ok:
                chosen = cfg
                break
        cm.append(Rref_h)
    if chosen is not None:
        cand = P.loc[chosen, "cand"]
        vc, qc = fills[cand]
        wr, wc = ru.ws(ref_v, ref_q), ru.ws(vc, qc)
        rob["bootstrap_vs_S2"] = ru.V.bootstrap(A0, ru.TEN, wr, wc, ru.FR, ru.OW)
        D = pd.DataFrame({"split": A0.split.astype(str), "cond": A0.condname,
                          "e_ref": (ref_v - A0.ov) ** 2, "e_c": (vc - A0.ov) ** 2,
                          "f_ref": ((ref_q - A0.oq) / A0.lanes) ** 2, "f_c": ((qc - A0.oq) / A0.lanes) ** 2})
        rob["mse_by_split_cond"] = D.groupby(["split", "cond"]).mean(numeric_only=True).reset_index().to_dict("records")
        np.savez_compressed(OUT / f"crossfit{args.suffix}_chosen_fill.npz", v=vc, q=qc)
    G.to_csv(files[0], index=False)
    pd.concat(cm, ignore_index=True).to_csv(files[1], index=False)
    write_new(files[2], json.dumps({
        "rule": __doc__, "prereg_written": pr["written"], "reproduced_max_abs_dTotal": repro, "picks": picks,
        "winners": list(winners.cand), "chosen": chosen,
        "chosen_cand": None if chosen is None else P.loc[chosen, "cand"],
        "chosen_row": None if chosen is None else P.loc[chosen].to_dict(),
        "dS_family": None if chosen is None else {s: float(P.loc[chosen, f"dS_{s}"]) for s in SPLITS},
        "candidates": P.reset_index().to_dict("records"), "robustness": rob,
        "seconds": round(time.time() - t0)}, indent=2, default=float))
    print(P.round(7).to_string(), flush=True)
    print(json.dumps(rob, indent=2, default=float), flush=True)
    print("chosen", chosen, f"{time.time() - t0:.0f}s", flush=True)


# ---------------------------------------------------------------------------
# report only: train held-out months, official layout
# ---------------------------------------------------------------------------
def cmd_heldout(args) -> None:
    import torch
    path = OUT / "heldout.csv"
    if path.exists():
        raise FileExistsError(path)
    R = pd.read_csv(B.CACHE / "train_runs.csv")
    rows = []
    for fold in (0, 1):
        nets = {}
        for tag in TAGS:
            nets["prod", tag] = B.load_prod(tag, fold)
            nets["A", tag] = B.load_spec(tag, fold)
            for s in SETS:
                if net_path(s, tag, fold).exists():
                    nets[s, tag] = load_net(s, tag, fold)
        kinds = sorted({k for k, _ in nets})
        combos = {k: [k] for k in kinds}
        for nm, parts in (("A+s1", ["A", "s1"]), ("A+ext", ["A", "ext"]), ("A+s1+ext", ["A", "s1", "ext"])):
            if all(x in kinds for x in parts):
                combos[nm] = parts
        Rf = R[R.month % 2 == fold]
        Rf = Rf.groupby("panel", group_keys=False).apply(lambda g: g.iloc[::max(1, len(g) // args.max_runs)])
        for p, g in Rf.groupby("panel"):
            src, st, prof = U.open_split(p, "train"), U.static(p), np.load(U.DATA / f"{p}_profiles.npy")
            occ = np.load(B.CACHE / f"occ_{p}_train.npy", mmap_mode="r")
            lanes = st["lanes"]
            acc = {}
            for _, r in g.iterrows():
                gaps = B.parse_gaps(r.gaps)
                d = U.Day(p, "train", int(r.day), src, st, prof)
                n = d.obs.shape[1]
                tr = np.moveaxis(d.truth, -1, 0)
                ok = np.isfinite(tr).all(0)
                vis0 = d.obs[..., 3] > 0.5
                hide = np.zeros((STEPS, n), bool)
                unhide = np.zeros((STEPS, n), bool)
                for T, _ in gaps:
                    hide[T + 1:T + 1 + GAP] = True
                    unhide[T - HIST:T] = ok[T - HIST:T] & ~vis0[T - HIST:T]
                x2, b = B.make_input2(d, gaps, hide, unhide, np.asarray(occ[int(r.day)], np.float32))
                x1 = x2[:U.N_IN]
                out = {k: B.predict_np(m, x1 if k[0] == "prod" else x2, b, st) for k, m in nets.items()}
                vt, qt = tr[0] * st["ffs"], tr[1] * st["cap"]
                for T, c in gaps:
                    rr = slice(T + 1, T + 1 + GAP)
                    m_ = ok[rr]
                    for nm, parts in combos.items():
                        v = np.mean([out[k, t][0] for k in parts for t in TAGS], axis=0)[rr]
                        q = np.mean([out[k, t][1] for k in parts for t in TAGS], axis=0)[rr]
                        a = acc.setdefault((c, nm), [0, 0.0, 0.0])
                        a[0] += int(m_.sum())
                        a[1] += float(((v - vt[rr]) ** 2)[m_].sum())
                        a[2] += float((((q - qt[rr]) / lanes) ** 2)[m_].sum())
            for (c, nm), a in acc.items():
                rows.append(dict(fold=fold, panel=p, cond=c, kind=nm, cells=a[0], sse_v=a[1], sse_q=a[2],
                                 rmse_v=math.sqrt(a[1] / a[0]), rmse_q_lane=math.sqrt(a[2] / a[0])))
            print(fold, p, {k: round(math.sqrt(a[1] / a[0]), 3) for k, a in acc.items() if k[0] == "onset"}, flush=True)
        del nets
        torch.cuda.empty_cache()
    D = pd.DataFrame(rows)
    D.to_csv(path, index=False)
    s = D.groupby(["kind"])[["cells", "sse_v", "sse_q"]].sum()
    s["rmse_v"], s["rmse_q_lane"] = np.sqrt(s.sse_v / s.cells), np.sqrt(s.sse_q / s.cells)
    print(s[["cells", "rmse_v", "rmse_q_lane"]].round(3).to_string(), flush=True)
    s = D.groupby(["panel", "kind"])[["cells", "sse_v", "sse_q"]].sum()
    s["rmse_v"], s["rmse_q_lane"] = np.sqrt(s.sse_v / s.cells), np.sqrt(s.sse_q / s.cells)
    print(s[["rmse_v", "rmse_q_lane"]].unstack().round(3).to_string(), flush=True)


# ---------------------------------------------------------------------------
# final GBMs, official windows, apply, package, audit, compare (only after crossfit chose a winner)
# ---------------------------------------------------------------------------
def load_choice():
    ch = json.loads((OUT / "crossfit_choice.json").read_text())
    if ch["chosen"] is None:
        raise SystemExit("crossfit chose no candidate: nothing to build")
    row = ch["chosen_row"]
    return ch["chosen"], ch["chosen_cand"], float(row["lam_v"]), float(row["lam_q"])


def gbm_files(name: str) -> dict:
    return {c: MODELS / f"t1_blk_v61b3_{name}_{c}.txt" for c in "vq"}


def cmd_fit(args) -> None:
    """Residual GBM(s) of the chosen candidate refitted on all ruler rows of both months (E1: the S4 GBM; the S2 GBM
    of V59 is reused as is)."""
    import lightgbm as lgb
    import t1_v57_blackout as V
    _, cand, _, _ = load_choice()
    A0, _ = load_prod()
    feats_s2 = V.FEATS + B.EXTRA_FEATS
    if cand == "E1":
        name, A1, ff = "s4", cand_frame("ss", A0), feats_s2
    elif cand == "E2":
        A1 = cand_frame("A", A0)
        A3 = cand_frame("ss", A0)
        A1["dss_v_r"] = ((A3.pv.to_numpy(float) - A1.pv.to_numpy(float)) / A0.ffs.to_numpy(float)).astype(np.float32)
        A1["dss_q_l"] = ((A3.pq.to_numpy(float) - A1.pq.to_numpy(float)) / A0.lanes.to_numpy(float)).astype(np.float32)
        del A3
        name, ff = "e2", feats_s2 + SS_FEATS
    else:
        name, A1, ff = cand.lower(), cand_frame("ens", A0, CANDS[cand], cand.lower()), feats_s2
    files = gbm_files(name)
    for pth in files.values():
        if pth.exists():
            raise FileExistsError(pth)
    X = A1[ff].to_numpy(np.float32)
    lanes = A0.lanes.to_numpy(float)
    ys = {"v": A0.ov.to_numpy(float) - A1.pv.to_numpy(float), "q": (A0.oq.to_numpy(float) - A1.pq.to_numpy(float)) / lanes}
    for c in "vq":
        lgb.train(V.GBM_PARAMS, lgb.Dataset(X, ys[c], feature_name=ff), V.GBM_ROUNDS).save_model(str(files[c]))
        print("saved", files[c], flush=True)


def official_frames(p: str, sp: str, sets):
    """V57 official rows (static columns + production CPU U-Net) -> gap features of production (F0) and of the
    ensemble of `sets` (sets of the 'official' kind; 'A' = phase-A, 'ss' = V59 phase-B other-parity nets)."""
    import t1_v57_blackout as V
    D0 = pd.read_parquet(V.OUT / "official" / f"{p}_{sp}_win.parquet")
    dfs = []
    for s in sets:
        d = pd.read_parquet((B.OUT / "official_ss" if s == "ss" else set_dir(s, "official")) / f"{p}_{sp}_win.parquet")
        assert len(d) == len(D0)
        for c in ("day", "T", "k", "pos"):
            assert np.array_equal(D0[c].to_numpy(), d[c].to_numpy()), (p, sp, s, c)
        dfs.append(d)
    D1 = D0.copy()
    if len(dfs) == 1:
        for c in dfs[0].columns:
            if c not in ("day", "T", "k", "pos"):
                D1[c] = dfs[0][c].to_numpy()
    else:
        for c, a in ens_columns(dfs).items():
            D1[c] = a
    A0_, m0 = V.win_tensors(p, D0)
    A1_, m1 = V.win_tensors(p, D1)
    F0, F = V.gap_features(p, A0_, m0), V.gap_features(p, A1_, m1)
    for c in ("w", "k", "pos"):
        assert np.array_equal(F0[c].to_numpy(), F[c].to_numpy())
    ffs, lanes = F.ffs.to_numpy(np.float32), F.lanes.to_numpy(np.float32)
    F["pv0_r"] = F0.pv.to_numpy(np.float32) / ffs
    F["pq0_l"] = F0.pq.to_numpy(np.float32) / lanes
    F["dsp_v_r"] = (F.pv.to_numpy(np.float32) - F0.pv.to_numpy(np.float32)) / ffs
    F["dsp_q_l"] = (F.pq.to_numpy(np.float32) - F0.pq.to_numpy(np.float32)) / lanes
    return F0, F


def official_fill_cand(cand: str, lv: float, lq: float) -> dict:
    """(split, panel) -> frame with key, window/cell ids and v_new / q_new of the chosen candidate, plus the S2
    (= V59_blk) official fill recomputed with the V59 code (v_s2 / q_s2) for the V60 consistency check."""
    import lightgbm as lgb
    import t1_v57_blackout as V
    feats_s2 = V.FEATS + B.EXTRA_FEATS
    s2 = B.official_fill("S2", *REF_LAM)
    out = {}
    if cand == "E1":
        bo = {c: lgb.Booster(model_file=str(f)) for c, f in gbm_files("s4").items()}
    elif cand == "E2":
        bo = {c: lgb.Booster(model_file=str(f)) for c, f in gbm_files("e2").items()}
    else:
        bo = {c: lgb.Booster(model_file=str(f)) for c, f in gbm_files(cand.lower()).items()}
    for p in T2_PANELS:
        for sp in SPLITS:
            F2 = s2[sp, p]
            if cand == "E1":
                _, F = official_frames(p, sp, ["ss"])
                X = F[feats_s2].to_numpy(np.float32)
                v4, q4 = mk_fill(F.pv.to_numpy(float), F.pq.to_numpy(float), F.lanes.to_numpy(float),
                                 bo["v"].predict(X, num_threads=4), bo["q"].predict(X, num_threads=4), *S4_LAM)
                v, q = 0.5 * (F2.v_new.to_numpy() + v4), 0.5 * (F2.q_new.to_numpy() + q4)
                spec_v = F.pv.to_numpy()
            elif cand == "E2":
                _, F = official_frames(p, sp, ["A"])
                _, F3 = official_frames(p, sp, ["ss"])
                F["dss_v_r"] = (F3.pv.to_numpy(float) - F.pv.to_numpy(float)) / F.ffs.to_numpy(float)
                F["dss_q_l"] = (F3.pq.to_numpy(float) - F.pq.to_numpy(float)) / F.lanes.to_numpy(float)
                X = F[feats_s2 + SS_FEATS].to_numpy(np.float32)
                v, q = mk_fill(F.pv.to_numpy(float), F.pq.to_numpy(float), F.lanes.to_numpy(float),
                               bo["v"].predict(X, num_threads=4), bo["q"].predict(X, num_threads=4), lv, lq)
                spec_v = F.pv.to_numpy()
            else:
                _, F = official_frames(p, sp, list(CANDS[cand]))
                X = F[feats_s2].to_numpy(np.float32)
                v, q = mk_fill(F.pv.to_numpy(float), F.pq.to_numpy(float), F.lanes.to_numpy(float),
                               bo["v"].predict(X, num_threads=4), bo["q"].predict(X, num_threads=4), lv, lq)
                spec_v = F.pv.to_numpy()
            for c in ("w", "k", "pos"):
                assert np.array_equal(F[c].to_numpy(), F2[c].to_numpy()), (p, sp, c)
            G = F2[["key", "window_id", "k", "pos", "condname", "v_new", "q_new", "pv_prod_cpu", "pq_prod_cpu"]].copy()
            G = G.rename(columns={"v_new": "v_s2", "q_new": "q_s2"})
            G["v_new"], G["q_new"], G["unet_v"] = v, q, spec_v
            out[sp, p] = G
    return out


def g6(x: np.ndarray) -> np.ndarray:
    return np.array([float("%.6g" % v) for v in x])


def cmd_apply(args) -> None:
    import zipfile
    import t1_assemble as TA
    import t1_v52_monthbias as M
    from evaluate import t1_targets
    name, cand, lv, lq = load_choice()
    for pth in CSV.values():
        if pth.exists():
            raise FileExistsError(pth)
    for f in ("apply_changes.csv", "apply_summary.json", "apply_cells.parquet"):
        if (OUT / f).exists():
            raise FileExistsError(OUT / f)
    t0 = time.time()
    fills = official_fill_cand(cand, lv, lq)
    K = M.read_key_state()
    with zipfile.ZipFile(V60_ZIP) as z, z.open(z.namelist()[0]) as f:
        Z = pd.read_csv(f, float_precision="round_trip", usecols=["submission_id", "task", "speed_kmh", "flow_vph"],
                        dtype={"submission_id": np.int64, "task": "category", "speed_kmh": str, "flow_vph": str})
    Z = Z[Z.task == "state"]
    assert np.array_equal(Z.submission_id.to_numpy(), K.submission_id.to_numpy())
    sv_txt, sq_txt = Z.speed_kmh.to_numpy(), Z.flow_vph.to_numpy()
    K["v60"] = pd.to_numeric(Z.speed_kmh, errors="raise").to_numpy(float)
    K["q60"] = pd.to_numeric(Z.flow_vph, errors="raise").to_numpy(float)
    n_not6g = int(sum(("%.6g" % a) != b for a, b in zip(K.v60.to_numpy(), sv_txt))) + \
        int(sum(("%.6g" % a) != b for a, b in zip(K.q60.to_numpy(), sq_txt)))
    del Z, sv_txt, sq_txt
    assert n_not6g == 0, f"{n_not6g} V60 state values are not %.6g text"
    print("key + V60 state rows", len(K), f"{time.time() - t0:.0f}s", flush=True)
    rows, cells = [], []
    for split in SPLITS:
        S_ = K[K.split == split].reset_index(drop=True)
        v_all, q_all = S_.v60.to_numpy().copy(), S_.q60.to_numpy().copy()
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
            n_s2_mismatch = int(((g6(F.v_s2.to_numpy()) != v0) | (g6(F.q_s2.to_numpy()) != q0)).sum())
            v_all[ii], q_all[ii] = v1, q1
            ch = (v1 != v0) | (q1 != q0)
            cells.append(pd.DataFrame({"split": split, "panel": p, "key": F.key, "window_id": F.window_id, "k": F.k,
                                       "pos": F.pos, "cond": F.condname, "v60": v0, "q60": q0, "v61": v1, "q61": q1,
                                       "pv_prod_cpu": F.pv_prod_cpu, "pq_prod_cpu": F.pq_prod_cpu, "unet_v": F.unet_v}))
            rows.append({"split": split, "panel": p, "targets": len(idx), "blackout_targets": int(gap.sum()),
                         "changed": int(ch.sum()), "v60_vs_recomputed_S2_mismatch": n_s2_mismatch,
                         "mean_dv": float(np.mean(v1 - v0)), "mean_abs_dv": float(np.mean(np.abs(v1 - v0))),
                         "max_abs_dv": float(np.max(np.abs(v1 - v0))), "mean_dq": float(np.mean(q1 - q0)),
                         "mean_abs_dq": float(np.mean(np.abs(q1 - q0))), "max_abs_dq": float(np.max(np.abs(q1 - q0)))})
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
    by = C.assign(dv=C.v61 - C.v60, dq=C.q61 - C.q60).groupby(["split", "cond", "k"])[["dv", "dq"]].mean()
    nets = {}
    for s in ("s1", "ext"):
        for t in TAGS:
            for f in (0, 1):
                if net_path(s, t, f).exists():
                    nets[f"{s}{t}_f{f}"] = M.sha256(net_path(s, t, f))
    write_new(OUT / "apply_summary.json", json.dumps({
        "chosen": name, "cand": cand, "lam_v": lv, "lam_q": lq, "changes": rows,
        "mean_change_vs_v60_by_split_cond_k": by.reset_index().to_dict("records"),
        "outputs_sha256": {s: M.sha256(CSV[s]) for s in SPLITS}, "v60_zip_sha256": M.sha256(V60_ZIP),
        "new_nets_sha256": nets,
        "gbm_sha256": {f.name: M.sha256(f) for nm in ("s4", "e2", "e3", "e4", "e5") for f in gbm_files(nm).values()
                       if f.exists()},
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
    write_new(OUT / "v61b3_state_manifest_supplement.json", json.dumps({
        "note": "the manifest written by the imported t1_v52_monthbias package code carries the V52 note and "
                "state_inputs; the actual state inputs of v61b3_state.zip are listed here",
        "state_inputs": [{"path": str(CSV[s]), "sha256": M.sha256(CSV[s])} for s in SPLITS],
        "content": "all state rows = V60 (v60_loc_blk.zip) except the Task-1 targets inside official Task-2 "
                   "blackouts (V61 lane B3 fill); queue/ODME rows = v51_april.zip (replaced by V54 queues in "
                   "build_queue_candidate)",
        "output_sha256": M.sha256(WORK / "submissions" / f"{NAME}.zip")}, indent=2))


def cmd_audit(args) -> None:
    M = _patch()
    M.cmd_audit(args)


def cmd_compare(args) -> None:
    """v61_b3.zip vs v60_loc_blk.zip, line by line: queue/ODME lines byte-identical, state lines differ only at
    Task-1 targets inside official Task-2 blackouts, every state value equals the CSV value, %.6g text."""
    import zipfile
    import t1_assemble as TA
    import t1_v52_monthbias as M
    from evaluate import t1_targets
    new = WORK / "submissions" / f"{FULL_NAME}.zip"
    path = OUT / "compare_v60.json"
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
    with zipfile.ZipFile(V60_ZIP) as za, zipfile.ZipFile(new) as zb:
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
    st["v60_sha256"], st["v61_b3_sha256"] = M.sha256(V60_ZIP), M.sha256(new)
    write_new(path, json.dumps(st, indent=2))
    print(json.dumps(st, indent=2), flush=True)
    assert st["valid"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for nm in ("prereg", "runs_ext", "fit", "apply", "package", "audit", "compare"):
        sub.add_parser(nm)
    s = sub.add_parser("crossfit")
    s.add_argument("--suffix", default="")
    s = sub.add_parser("train")
    s.add_argument("--set", required=True, choices=list(SETS))
    s.add_argument("--tag", required=True, choices=TAGS)
    s.add_argument("--fold", type=int, required=True, choices=[0, 1])
    s.add_argument("--epochs", type=int, default=8)
    s.add_argument("--batch", type=int, default=4)
    s.add_argument("--lr", type=float, default=2e-4)
    s.add_argument("--p-full", type=float, default=B.P_FULL)
    s = sub.add_parser("wide")
    s.add_argument("--set", required=True, choices=list(SETS))
    s.add_argument("--split", required=True, choices=SPLITS)
    s = sub.add_parser("official")
    s.add_argument("--set", required=True, choices=list(SETS))
    s = sub.add_parser("heldout")
    s.add_argument("--max-runs", type=int, default=120)
    args = ap.parse_args()
    {"prereg": cmd_prereg, "runs_ext": cmd_runs_ext, "train": cmd_train, "wide": cmd_wide, "official": cmd_official,
     "crossfit": cmd_crossfit, "heldout": cmd_heldout, "fit": cmd_fit, "apply": cmd_apply, "package": cmd_package,
     "audit": cmd_audit, "compare": cmd_compare}[args.cmd](args)


if __name__ == "__main__":
    main()
