"""V57 lane B ruler: the production Task-1 chain inside simulated official-like Task-2 blackouts.

Why: in validation/private every official Task-2 window blanks the whole corridor for (T, T+90]; the Task-1 targets
inside (~2% on the T2 corridors, 0 on D12_I405_*) cannot be measured on val/private, and the hide-20% protocol of
t1_v52_monthbias never produces such gaps.  The only earlier number "measured on the official train blackouts"
(work/results/t1_v29_blackout, 1.74 km/h) is NOT a blackout measurement: in train the blackout cells were not
blanked (65% of them are visible in clean_data/*/train.parquet), so the U-Net saw the corridor around those targets.

What this script does (Task 1 is offline; only the visible observations of the month itself are read; nothing of
Task 2 is read except the official blackout flags that are part of the T1 release):
  origins --split S   official-like origins on the visible layer of month S: the official selector's per-origin rules
                      (history coverage >= 0.7 over T-60..T-5; established history -> ongoing, needs a latent queue in
                      T+5..T+30 and latent persistence IoU <= 0.9; else onset, needs a visible eligible queued cell in
                      T+5..T+30; spacing >= 360 min; at most 5 per condition) run as a greedy forward pass from every
                      start day, union of the unique origins (same construction as work/cache/t2_v52_selrule/windows.csv
                      for March).  Origins whose rows T-60..T+120 touch an official blackout are dropped (the simulated
                      gap needs observations inside and visible edges).  Used ONLY to place simulated T1 blackouts; no
                      Task-2 row, rule or selection depends on them.
  sim --split S       for every day: pack the origins into runs (gaps of one run >= 48 slots apart), hide every visible
                      cell of the corridor in rows T+1..T+18 and flag them is_t2_blackout, rerun the production chain
                      exactly as t1_v52_monthbias.Chain.run (GBM + 2 folds x {w48_vp, w48lwr_vp} + frozen V31 stack at
                      the production alpha; blackout targets get the unstacked U-Net mean, as in production) and store
                      prediction / components / observation of every hidden cell.

Outputs: work/results/t1_v57_blkruler/ (origins_{split}.csv, sim/{panel}_{split}_gap.parquet, runs csv).
Nothing existing is overwritten.

  python work/t1_v57_blkruler.py origins --split validation      (and private)
  python work/t1_v57_blkruler.py sim --split validation --threads 4 [--panels ...]
  python work/t1_v57_blkruler.py sim --split validation --threads 4 --fast      (U-Net only -> sim_fast/)
  python work/t1_v57_blkruler.py wide --split validation --threads 4             (rows T-6..T+24, all cells -> sim_wide/)
  python work/t1_v57_blkruler.py report        (after t1_v57_blackout.py frame) -> ruler_summary.csv, ruler_loss.csv
"""
from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "4")

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import CLEAN, STEPS, T2_PANELS, links  # noqa: E402

WORK = Path(__file__).resolve().parent
OUT = WORK / "results" / "t1_v57_blkruler"
SIM = OUT / "sim"
SIM_FAST = OUT / "sim_fast"
SIM_WIDE = OUT / "sim_wide"
GAP = 18
HIST, HOR, SPACING, MIN_COV, MAX_PIOU, CAP = 12, 6, 72, 0.7, 0.9, 5
MARGIN_BEFORE, MARGIN_AFTER = 12, 24      # rows T-12..T+18+24 must be free of official blackouts
RUN_SEP = 48                              # two simulated gaps of one chain run are >= 48 slots apart


def _iou(a, b):
    u = (a | b).sum()
    return 1.0 if u == 0 else float((a & b).sum() / u)


def gapfill(S: np.ndarray, limit: int = 6) -> np.ndarray:
    df = pd.DataFrame(S)
    f = df.ffill(limit=limit).to_numpy()
    b = df.bfill(limit=limit).to_numpy()
    return np.where(np.isfinite(S), S, np.where(np.isfinite(f) & np.isfinite(b), 0.5 * (f + b),
                                                np.where(np.isfinite(f), f, b)))


def month_layers(panel: str, split: str):
    n = len(links(panel))
    pf = pq.ParquetFile(CLEAN / panel / f"{split}.parquet")
    D = pf.num_row_groups
    S = np.full((D, STEPS, n), np.nan, np.float32)
    VIS = np.full((D, STEPS, n), np.nan, np.float32)
    E = np.zeros((D, STEPS, n), bool)
    B = np.zeros((D, STEPS, n), bool)
    dates, regime = [], []
    for i in range(D):
        t = pf.read_row_group(i, columns=["date", "mask_regime", "is_score_eligible", "is_t2_blackout",
                                          "speed_clean", "speed_obs"])
        dates.append(str(t.column("date")[0].as_py()))
        regime.append(str(t.column("mask_regime")[0].as_py()))
        g = lambda c: t.column(c).to_numpy(zero_copy_only=False).astype(np.float32).reshape(STEPS, n)
        E[i] = t.column("is_score_eligible").to_numpy(zero_copy_only=False).astype(bool).reshape(STEPS, n)
        B[i] = t.column("is_t2_blackout").to_numpy(zero_copy_only=False).astype(bool).reshape(STEPS, n)
        S[i] = g("speed_obs")
        VIS[i] = g("speed_clean")
    return S, VIS, E, B, dates, regime


class Timeline:
    """Replica of the selector rules on one (panel, month) visible layer (t2_v52_replica.Timeline logic)."""

    def __init__(self, panel, split):
        S, VIS, E, B, self.dates, self.regime = month_layers(panel, split)
        vc = links(panel).v_cut_kmh.to_numpy(np.float32)
        n = len(vc)
        D = S.shape[0]
        S, VIS, self.E = S.reshape(-1, n), VIS.reshape(-1, n), E.reshape(-1, n)
        self.B = B.reshape(-1, n).any(1)
        with np.errstate(invalid="ignore"):
            self.QL = gapfill(S) <= vc
            self.QV = VIS <= vc
        self.n = len(self.QL)
        cs = np.concatenate([[0], np.cumsum(self.E.mean(1))])
        self.cov = np.full(self.n, np.nan)
        self.cov[HIST:] = (cs[HIST:self.n] - cs[:self.n - HIST]) / HIST
        qc = np.concatenate([np.zeros((1, n)), np.cumsum(self.QV, 0)])
        hs = np.zeros(self.QV.shape, np.int32)
        hs[HIST:] = (qc[HIST:] - qc[:-HIST])[:self.n - HIST]
        self.hsum, self.hmax = hs.sum(1), hs.max(1)
        self.est = (self.hsum >= 2) & (self.hmax >= 2)
        bc = np.concatenate([[0], np.cumsum(self.B)])
        self.blk = np.ones(self.n, bool)                 # rows t-12..t+6 touch a blackout (selector rule)
        lo, hi = np.arange(self.n) - HIST, np.arange(self.n) + HOR + 1
        ok = (lo >= 0) & (hi <= self.n)
        self.blk[ok] = (bc[hi[ok]] - bc[lo[ok]]) > 0
        self.blk_wide = np.ones(self.n, bool)            # rows t-12..t+42 touch a blackout (ruler rule)
        lo, hi = np.arange(self.n) - MARGIN_BEFORE, np.arange(self.n) + GAP + MARGIN_AFTER + 1
        ok = (lo >= 0) & (hi <= self.n)
        self.blk_wide[ok] = (bc[hi[ok]] - bc[lo[ok]]) > 0
        self.D = D

    def piou(self, t):
        return _iou(np.repeat(self.QL[t - 1][None], HOR, 0), self.QL[t + 1:t + HOR + 1])

    def select(self, start_day):
        cnt = {"onset": 0, "ongoing": 0}
        last, out = None, []
        for t in range(max(HIST, start_day * STEPS), self.n - HOR):
            if cnt["onset"] >= CAP and cnt["ongoing"] >= CAP:
                break
            if last is not None and t - last < SPACING:
                continue
            if self.blk[t] or self.cov[t] < MIN_COV:
                continue
            if self.est[t]:
                if cnt["ongoing"] >= CAP or not self.QL[t + 1:t + HOR + 1].any():
                    continue
                p = self.piou(t)
                if p > MAX_PIOU:
                    continue
                cond = "ongoing"
            else:
                if cnt["onset"] >= CAP or not (self.QV[t + 1:t + HOR + 1] & self.E[t + 1:t + HOR + 1]).any():
                    continue
                cond, p = "onset", 0.0
            cnt[cond] += 1
            out.append(dict(start=start_day, cond=cond, t=t, day=t // STEPS, T=t % STEPS, piou=round(p, 4),
                            spacing=bool(last is not None and t - last == SPACING)))
            last = t
        return out


def cmd_origins(args):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"origins_{args.split}.csv"
    if path.exists():
        raise FileExistsError(path)
    rows = []
    for p in T2_PANELS:
        tl = Timeline(p, args.split)
        allw = []
        for s in range(0, tl.D - 2):
            allw += tl.select(s)
        w = pd.DataFrame(allw)
        w = w.sort_values("start").drop_duplicates(["t", "cond"]).sort_values("t")
        w["blk_wide"] = tl.blk_wide[w.t.to_numpy()]
        w["date"] = [tl.dates[d] for d in w.day]
        w["regime"] = [tl.regime[d] for d in w.day]
        w.insert(0, "panel", p)
        w.insert(1, "split", args.split)
        rows.append(w)
        print(p, args.split, "unique origins", len(w), "usable", int((~w.blk_wide).sum()), flush=True)
    W = pd.concat(rows, ignore_index=True)
    W.to_csv(path, index=False)
    print(W[~W.blk_wide].groupby(["panel", "cond"]).size().unstack())


def pack_runs(O: pd.DataFrame) -> list[tuple[int, list]]:
    """Group the usable origins of a panel into chain runs: one day per run, gaps >= RUN_SEP slots apart."""
    runs = []
    for day, g in O.groupby("day"):
        open_runs: list[list] = []
        for _, r in g.sort_values("T").iterrows():
            for run in open_runs:
                if all(abs(r["T"] - x["T"]) >= RUN_SEP for x in run):
                    run.append(r.to_dict())
                    break
            else:
                open_runs.append([r.to_dict()])
        runs += [(int(day), run) for run in open_runs]
    return runs


def unet_only(ch, ctx, tb, ud, hide, st):
    """Fast path: inside a Task-2 blackout the production output is exactly the unstacked mean of the two U-Net
    tags (t1_v30_stack.features: bv[gap] = .5*(unet + lwr); t1_v30_bridge.corrected: v[gap] = bv[gap]); GBM and the
    residual stack never reach these cells.  Same arithmetic as Chain.run for those cells; per-fold outputs kept.
    Checked bit-identical against the full chain on D7_I10_W March (fastcheck)."""
    mt = ch.B.masked_table(tb, hide)
    n = ctx.n
    target = mt.column("is_t1_target").to_numpy().reshape(STEPS, n)
    gap = mt.column("is_t2_blackout").to_numpy().reshape(STEPS, n)[target]
    h_tr = hide[:, st["perm"]]
    comps, pairs = {}, []
    for tag in ch.B.TAGS:
        ps = [ch.unet(ch.nets[tag, f], ud, h_tr) for f in (0, 1)]
        pairs.append(tuple(np.mean([x[j] for x in ps], axis=0)[:, ctx.inv][target] for j in (0, 1)))
        comps[tag] = pairs[-1]
        for f in (0, 1):
            comps[f"{tag}f{f}"] = tuple(ps[f][j][:, ctx.inv][target] for j in (0, 1))
    bv = np.full(int(target.sum()), np.nan)
    bq = np.full(int(target.sum()), np.nan)
    bv[gap] = .5 * (pairs[0][0][gap] + pairs[1][0][gap])
    bq[gap] = .5 * (pairs[0][1][gap] + pairs[1][1][gap])
    return bv.copy(), bq.copy(), bv, bq, gap, target, comps


def cmd_sim(args):
    import t1_unet as U
    import task1_gbm as G
    import t1_v52_monthbias as M
    SIM.mkdir(parents=True, exist_ok=True)
    W = pd.read_csv(OUT / f"origins_{args.split}.csv")
    W = W[~W.blk_wide]
    ch = M.Chain(args.threads)
    prot = OUT / "sim_protocol.json"
    if not prot.exists():
        prot.write_text(json.dumps({
            "gap_rows": "T+1..T+18 of every origin, all visible cells hidden and flagged is_t2_blackout",
            "chain": "t1_v52_monthbias.Chain.run (CPU fp32), production alpha per panel "
                     "(t1_v52_monthbias.ALPHA); blackout targets get the unstacked U-Net mean as in production",
            "origins": "origins_{split}.csv, rows with blk_wide False",
            "run_separation_slots": RUN_SEP, "models_sha256": ch.fingerprint}, indent=2))
    cols = list(dict.fromkeys(ch.B.COLS + M.RAMP_COLS))
    sim_dir = SIM_FAST if args.fast else SIM
    sim_dir.mkdir(parents=True, exist_ok=True)
    for p in args.panels:
        path = sim_dir / f"{p}_{args.split}_gap.parquet"
        if path.exists():
            print("exists", path.name, flush=True)
            continue
        t0 = time.time()
        ctx, st = G.Panel(p), U.static(p)
        src, prof = U.open_split(p, args.split), np.load(U.DATA / f"{p}_profiles.npy")
        pf = pq.ParquetFile(CLEAN / p / f"{args.split}.parquet")
        n = ctx.n
        runs = pack_runs(W[W.panel == p])
        recs, rinfo = [], []
        for ir, (day, run) in enumerate(runs):
            tb = pf.read_row_group(day, columns=cols, use_threads=False)
            arr = lambda c: tb.column(c).to_numpy().reshape(STEPS, n)
            ov, oq = arr("speed_obs").astype(float), arr("flow_obs").astype(float)
            elig = arr("is_score_eligible")
            vis = np.isfinite(ov) & np.isfinite(oq)
            rows = np.zeros(STEPS, bool)
            origin_of = np.full(STEPS, -1, np.int32)
            for r in run:
                rows[r["T"] + 1:r["T"] + 1 + GAP] = True
                origin_of[r["T"] + 1:r["T"] + 1 + GAP] = r["T"]
            hide = vis & rows[:, None]
            blk = arr("is_t2_blackout") | rows[:, None]
            tb2 = ch.B.replace(tb, "is_t2_blackout", blk)
            ud = U.Day(p, args.split, day, src, st, prof)
            if args.fast:
                v, q, bv, bq, gap, target, comps = unet_only(ch, ctx, tb2, ud, hide, st)
            else:
                v, q, bv, bq, gap, target, comps = ch.run(ctx, tb2, ud, hide, M.ALPHA[p], st)
            use_h = hide[target]
            assert gap[use_h].all()
            tt, pp_ = np.nonzero(hide)
            cond = {r["T"]: r["cond"] for r in run}
            Ts = origin_of[tt]
            rec = {"day": np.full(len(tt), day, np.int16), "run": np.full(len(tt), ir, np.int32),
                   "regime": np.full(len(tt), ud.regime, np.int8), "T": Ts.astype(np.int16),
                   "k": (tt - Ts).astype(np.int8), "t": tt.astype(np.int16), "pos": pp_.astype(np.int16),
                   "cond": np.array([cond[x] for x in Ts]), "elig": elig[hide],
                   "ov": ov[hide].astype(np.float32), "oq": oq[hide].astype(np.float32),
                   "pv": v[use_h].astype(np.float32), "pq": q[use_h].astype(np.float32),
                   "bv": bv[use_h].astype(np.float32), "bq": bq[use_h].astype(np.float32)}
            for nm, (cv, cq) in comps.items():
                rec[f"v{nm}"] = cv[use_h].astype(np.float32)
                rec[f"q{nm}"] = cq[use_h].astype(np.float32)
            recs.append(pd.DataFrame(rec))
            rinfo.append({"run": ir, "day": day, "origins": ";".join(f"{r['T']}:{r['cond']}" for r in run),
                          "n_hidden": int(hide.sum()), "sec": round(time.time() - t0, 1)})
            print(f"  {p} {args.split} run {ir + 1}/{len(runs)} day {day} gaps {len(run)}: {int(hide.sum())} hidden, "
                  f"{time.time() - t0:.0f}s", flush=True)
        pd.DataFrame(rinfo).to_csv(sim_dir / f"{p}_{args.split}_runs.csv", index=False)
        pd.concat(recs, ignore_index=True).to_parquet(path, index=False)
        print(p, args.split, "done", len(runs), "runs", f"{time.time() - t0:.0f}s", flush=True)


def cmd_wide(args):
    """Same simulated gaps as sim --fast, but stores rows T-6..T+24 of EVERY cell of the corridor around each origin:
    observation (NaN if not visible), flags (visible, eligible, original T1 target, hidden, ramp_ok) and the U-Net
    outputs (per tag and fold, mean = production fill inside the gap).  Outside the gap the stored U-Net mean is NOT
    the production value (which adds GBM and the stack); it is used only as edge information."""
    import t1_unet as U
    import task1_gbm as G
    import t1_v52_monthbias as M
    W = pd.read_csv(OUT / f"origins_{args.split}.csv")
    W = W[~W.blk_wide]
    ch = M.Chain(args.threads)
    cols = list(dict.fromkeys(ch.B.COLS + M.RAMP_COLS))
    SIM_WIDE.mkdir(parents=True, exist_ok=True)
    K0, K1 = -6, GAP + 6
    for p in args.panels:
        path = SIM_WIDE / f"{p}_{args.split}_win.parquet"
        if path.exists():
            print("exists", path.name, flush=True)
            continue
        t0 = time.time()
        ctx, st = G.Panel(p), U.static(p)
        src, prof = U.open_split(p, args.split), np.load(U.DATA / f"{p}_profiles.npy")
        pf = pq.ParquetFile(CLEAN / p / f"{args.split}.parquet")
        n = ctx.n
        runs = pack_runs(W[W.panel == p])
        recs = []
        for ir, (day, run) in enumerate(runs):
            tb = pf.read_row_group(day, columns=cols, use_threads=False)
            arr = lambda c: tb.column(c).to_numpy().reshape(STEPS, n)
            ov, oq = arr("speed_obs").astype(float), arr("flow_obs").astype(float)
            elig = arr("is_score_eligible")
            tgt = arr("is_t1_target")
            vis = np.isfinite(ov) & np.isfinite(oq)
            ramp_ok = (((arr("on_ramp_n") == 0) | (arr("on_ramp_n_valid") > 0)) &
                       ((arr("off_ramp_n") == 0) | (arr("off_ramp_n_valid") > 0)))
            rows = np.zeros(STEPS, bool)
            for r in run:
                rows[r["T"] + 1:r["T"] + 1 + GAP] = True
            hide = vis & rows[:, None]
            ud = U.Day(p, args.split, day, src, st, prof)
            h_tr = hide[:, st["perm"]]
            grid = {}
            for tag in ch.B.TAGS:
                ps = [ch.unet(ch.nets[tag, f], ud, h_tr) for f in (0, 1)]
                for j, c in enumerate("vq"):
                    grid[f"{c}{tag}"] = np.mean([x[j] for x in ps], axis=0)[:, ctx.inv]
                    for f in (0, 1):
                        grid[f"{c}{tag}f{f}"] = ps[f][j][:, ctx.inv]
            pv = .5 * (grid["v_w48"] + grid["v_w48lwr"])
            pq_ = .5 * (grid["q_w48"] + grid["q_w48lwr"])
            for r in run:
                T = r["T"]
                ks = np.arange(K0, K1 + 1)
                tt = T + ks
                okr = (tt >= 0) & (tt < STEPS)
                ks, tt = ks[okr], tt[okr]
                kk = np.repeat(ks, n)
                ti = np.repeat(tt, n)
                li = np.tile(np.arange(n), len(tt))
                rec = {"day": np.full(len(ti), day, np.int16), "run": np.full(len(ti), ir, np.int32),
                       "regime": np.full(len(ti), ud.regime, np.int8), "T": np.full(len(ti), T, np.int16),
                       "k": kk.astype(np.int8), "t": ti.astype(np.int16), "pos": li.astype(np.int16),
                       "cond": np.full(len(ti), r["cond"]), "elig": elig[ti, li], "vis": vis[ti, li],
                       "tgt": tgt[ti, li], "hidden": hide[ti, li], "ramp_ok": ramp_ok[ti, li],
                       "ov": ov[ti, li].astype(np.float32), "oq": oq[ti, li].astype(np.float32),
                       "pv": pv[ti, li].astype(np.float32), "pq": pq_[ti, li].astype(np.float32)}
                for nm, a in grid.items():
                    rec[nm] = a[ti, li].astype(np.float32)
                recs.append(pd.DataFrame(rec))
            print(f"  {p} {args.split} run {ir + 1}/{len(runs)} day {day} gaps {len(run)} {time.time() - t0:.0f}s", flush=True)
        pd.concat(recs, ignore_index=True).to_parquet(path, index=False)
        print(p, args.split, "done", len(runs), "runs", f"{time.time() - t0:.0f}s", flush=True)


def cmd_report(args):
    """Ruler numbers from sim_wide/: in-gap error of the production fill per corridor-month and condition, outside
    error (hide-20% cells of t1_v52_monthbias), and the real-frame loss of the blackouts (official target counts,
    condition mix and mask rates; Task-3 emulation of t1_v57_blackout.window_sums) against (i) a perfect in-gap fill
    and (ii) an in-gap error at the outside level (S_state only).  Needs t1_v57_blackout.py frame."""
    import t1_v57_blackout as V
    for f in ("ruler_summary.csv", "ruler_loss.csv"):
        if (OUT / f).exists():
            raise FileExistsError(OUT / f)
    FR, OW = V.load_frame()
    rows, lrows = [], []
    for p in T2_PANELS:
        lanes = links(p).lanes.to_numpy(float)
        vc = links(p).v_cut_kmh.to_numpy(float)
        for sp in ("validation", "private"):
            A, meta = V.sim_tensors(p, sp)
            F = V.sim_features(p, sp)
            ln = lanes[F.pos.to_numpy()]
            om = V.outside_mse(p, sp)
            for c in ("all", "onset", "ongoing"):
                m = np.ones(len(F), bool) if c == "all" else (F.condname == c).to_numpy()
                g = F[m]
                ev, eq = g.pv - g.ov, (g.pq - g.oq) / ln[m]
                qd = (g.ov.to_numpy() <= vc[g.pos.to_numpy()])
                rows.append(dict(panel=p, split=sp, cond=c, windows=int(g.w.nunique()), cells=int(len(g)),
                                 rmse_v_gap=float(np.sqrt((ev ** 2).mean())), rmse_q_lane_gap=float(np.sqrt((eq ** 2).mean())),
                                 bias_v_gap=float(ev.mean()), bias_q_lane_gap=float(eq.mean()),
                                 rmse_v_out=float(np.sqrt(np.mean([om[r][0] for r in om]))),
                                 rmse_q_lane_out=float(np.sqrt(np.mean([om[r][1] for r in om]))),
                                 queued_share=float(qd.mean()),
                                 sse_v_share_queued=float((ev.to_numpy() ** 2)[qd].sum() / (ev.to_numpy() ** 2).sum())))
            ws0 = V.window_sums(p, A, F, F.pv.to_numpy(float), F.pq.to_numpy(float))
            wsp = V.window_sums(p, A, F, F.ov.to_numpy(float), F.oq.to_numpy(float))
            r0 = V.real_frame(p, sp, ws0, FR, OW)
            tot, _ = V.deltas(r0, V.real_frame(p, sp, wsp, FR, OW))
            # (ii) S_state with the in-gap MSE at the outside level
            fr = FR[(FR.panel == p) & (FR.split == sp)].set_index("regime")
            same = []
            for reg in fr.index:
                f = fr.loc[reg]
                mv, mq = om.get(reg, om[min(om)])
                same.append(0.54 * (1 - np.sqrt(mv) / 25) + 0.46 * (1 - np.sqrt(mq) / 600) - r0[reg]["S"])
            lrows.append(dict(panel=p, split=sp, blackout_targets=int(fr.n_tgt_blk.sum()),
                              blackout_share=float(fr.n_tgt_blk.sum() / fr.n_tgt.sum()),
                              **{f"loss_vs_perfect_{k}": v for k, v in tot.items()},
                              loss_vs_outside_level_dS=float(np.mean(same)),
                              E_lwr_from_gap=float(np.mean([r0[r]["E"] for r in r0]))))
            print(p, sp, lrows[-1], flush=True)
    R = pd.DataFrame(rows)
    Lo = pd.DataFrame(lrows)
    R.to_csv(OUT / "ruler_summary.csv", index=False)
    Lo.to_csv(OUT / "ruler_loss.csv", index=False)
    pd.set_option("display.width", 250)
    print(R.round(3).to_string(index=False))
    for sp in ("validation", "private"):
        g = Lo[Lo.split == sp]
        print(sp, {k: round(V.family_mean(dict(zip(g.panel, g[k]))), 6) for k in
                   ("loss_vs_perfect_dS", "loss_vs_perfect_dphys", "loss_vs_perfect_dLWR", "loss_vs_perfect_dTotal",
                    "loss_vs_outside_level_dS")})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("origins")
    s.add_argument("--split", required=True, choices=["validation", "private"])
    s = sub.add_parser("sim")
    s.add_argument("--split", required=True, choices=["validation", "private"])
    s.add_argument("--threads", type=int, default=4)
    s.add_argument("--panels", nargs="+", default=T2_PANELS)
    s.add_argument("--fast", action="store_true", help="U-Net only (identical output on blackout cells)")
    s = sub.add_parser("wide")
    s.add_argument("--split", required=True, choices=["validation", "private"])
    s.add_argument("--threads", type=int, default=4)
    s.add_argument("--panels", nargs="+", default=T2_PANELS)
    sub.add_parser("report")
    a = ap.parse_args()
    {"origins": cmd_origins, "sim": cmd_sim, "wide": cmd_wide, "report": cmd_report}[a.cmd](a)
