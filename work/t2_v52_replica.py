#!/usr/bin/env python
"""V52 helper: replica of the official Task 2 window selector, run on train months (truth layer) and on the
March (validation) visible masked layer.  Never run on April (private): the module refuses that split.

Selector (synthetic_release_v1.json task2 policy + round-1 reverse engineering, kaggle-intel replica4):
  * one greedy forward pass over a (panel, month) timeline, 5-min slots, history rows t-12..t-1 (T-60..T-5),
    horizon rows t+1..t+6 (T+5..T+30)
  * spacing >= 360 min (72 slots) to the last selected window of EITHER condition
  * history coverage = mean(is_score_eligible) over the 12 history rows >= 0.7
  * established history (>= 2 visible queued obs, >= 2 on one link) -> ongoing, else onset
  * ongoing needs a latent queue in the horizon and latent persistence IoU(L(T-5) x 6, L(T+5..T+30)) <= 0.9
  * onset needs a visible (eligible) queued cell in the horizon
  * at most CAP windows per condition (official: 5); later qualifying origins of a full condition are skipped
    without blocking the other condition
Latent queue = per-link gap-filled speed (<= 6 slots forward/backward, midpoint when both) <= v_cut.
  train: speed_true on every link (clean_data), validation: speed_obs on every observed link.
Visible layer: train = speed_true on eligible cells (what the official train window_history shows),
  validation = the released masked layer (t2_grids V).  Validation candidates whose rows t-12..t+6 touch a
  Task 2 blackout are skipped (no labels there).

March only: a candidate whose rows touch a blackout is skipped, so T-5 of a window right after a blackout was
not rejected by the selector rules (blk_T5 = True); such windows have no official analogue and are dropped
from the rule evaluation (t2_v52_selrule_eval.py).

Window classes at the origin T, computable at test time from window_index origins <= T and data <= T:
  spacing   T - previous origin (either condition) == 360 min
  constr    not spacing-bound and the history of T-5 (rows T-65..T-10) was already established: T-5 was a
            valid ongoing candidate, so it was rejected for latent persistence > 0.9
  newborn   otherwise (the queue history only became established at T)

  python work/t2_v52_replica.py check          # replica vs the official June-2030 train windows
  python work/t2_v52_replica.py build          # replica ongoing windows -> work/cache/t2_v52_selrule/windows.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import CLEAN, RELEASE, ROOT, STEPS, T2_PANELS, links, slot_of  # noqa: E402

CACHE = ROOT / "work" / "cache" / "t2_v52_selrule"
SPACING = 72
HIST, HOR = 12, 6
MIN_COV, MAX_PIOU = 0.7, 0.9


def _iou(a, b):
    u = (a | b).sum()
    return 1.0 if u == 0 else float((a & b).sum() / u)


def layers(panel: str, split: str) -> dict:
    """(days, 288, n) arrays in pos order: S latent-source speed, VIS visible speed (NaN = not visible),
    ELIG is_score_eligible, B blackout; plus dates.  Cached."""
    if split == "private":
        raise ValueError("April (private) pseudo windows are never built (organizer ruling 748789)")
    path = CACHE / f"layers_{panel}_{split}.npz"
    if path.exists():
        z = np.load(path, allow_pickle=False)
        return {k: z[k] for k in z.files}
    n = len(links(panel))
    pf = pq.ParquetFile(CLEAN / panel / f"{split}.parquet")
    lat_col = "speed_true" if split == "train" else "speed_obs"
    cols = ["date", "is_score_eligible", "is_t2_blackout", "speed_clean", lat_col]
    D = pf.num_row_groups
    out = dict(S=np.full((D, STEPS, n), np.nan, np.float32), VIS=np.full((D, STEPS, n), np.nan, np.float32),
               ELIG=np.zeros((D, STEPS, n), bool), B=np.zeros((D, STEPS, n), bool))
    dates = []
    for i in range(D):
        t = pf.read_row_group(i, columns=cols)
        dates.append(str(t.column("date")[0].as_py()))
        g = lambda c: t.column(c).to_numpy(zero_copy_only=False).astype(np.float32).reshape(STEPS, n)
        E = t.column("is_score_eligible").to_numpy(zero_copy_only=False).astype(bool).reshape(STEPS, n)
        out["ELIG"][i] = E
        out["B"][i] = t.column("is_t2_blackout").to_numpy(zero_copy_only=False).astype(bool).reshape(STEPS, n)
        lat = g(lat_col)
        out["S"][i] = lat
        out["VIS"][i] = np.where(E, lat, np.nan) if split == "train" else g("speed_clean")
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, dates=np.array(dates), **out)
    return {**out, "dates": np.array(dates)}


def gapfill(S: np.ndarray, limit: int = 6) -> np.ndarray:
    """(t, n) speed, NaN gaps filled per link from the nearest observations (<= limit slots)."""
    df = pd.DataFrame(S)
    f = df.ffill(limit=limit).to_numpy()
    b = df.bfill(limit=limit).to_numpy()
    return np.where(np.isfinite(S), S, np.where(np.isfinite(f) & np.isfinite(b), 0.5 * (f + b),
                                                np.where(np.isfinite(f), f, b)))


class Timeline:
    """One (panel, split, month) timeline of consecutive days."""

    def __init__(self, panel: str, split: str, month: str, lay: dict | None = None):
        lay = lay if lay is not None else layers(panel, split)
        days = [i for i, d in enumerate(lay["dates"]) if d.startswith(month)]
        assert days and days == list(range(days[0], days[-1] + 1))
        self.panel, self.split, self.month, self.day0 = panel, split, month, days[0]
        self.dates = [str(lay["dates"][i]) for i in days]
        sl = slice(days[0], days[-1] + 1)
        vc = links(panel).v_cut_kmh.to_numpy(np.float32)
        S = lay["S"][sl].reshape(-1, len(vc))
        VIS = lay["VIS"][sl].reshape(-1, len(vc))
        self.E = lay["ELIG"][sl].reshape(-1, len(vc))
        # train is complete (its official-window blackouts are not blanked); validation blackouts have no data
        self.B = lay["B"][sl].reshape(-1, len(vc)).any(1) if split != "train" else np.zeros(len(S), bool)
        with np.errstate(invalid="ignore"):
            self.QL = gapfill(S) <= vc                       # latent queue proxy
            self.QV = VIS <= vc                              # visible queued obs (NaN -> False)
        self.n = len(self.QL)
        cs = np.concatenate([[0], np.cumsum(self.E.mean(1))])
        self.cov = np.full(self.n, np.nan)
        self.cov[HIST:] = (cs[HIST:self.n] - cs[:self.n - HIST]) / HIST    # cov[t] = mean E[t-12:t]
        qc = np.concatenate([np.zeros((1, self.QV.shape[1])), np.cumsum(self.QV, 0)])
        hs = np.zeros(self.QV.shape, np.int32)
        hs[HIST:] = (qc[HIST:] - qc[:-HIST])[:self.n - HIST]  # hs[t] = per-link queued count in t-12..t-1
        self.hsum, self.hmax = hs.sum(1), hs.max(1)
        self.est = (self.hsum >= 2) & (self.hmax >= 2)
        bc = np.concatenate([[0], np.cumsum(self.B)])
        self.blk = np.zeros(self.n, bool)                    # rows t-12..t+6 touch a blackout
        lo, hi = np.arange(self.n) - HIST, np.arange(self.n) + HOR + 1
        ok = (lo >= 0) & (hi <= self.n)
        self.blk[ok] = (bc[hi[ok]] - bc[lo[ok]]) > 0
        self.blk[~ok] = True

    def piou(self, t: int) -> float:
        return _iou(np.repeat(self.QL[t - 1][None], HOR, 0), self.QL[t + 1:t + HOR + 1])

    def select(self, start_day: int = 0, cap: int = 5, onset_rule: str = "not_est"):
        """Greedy forward pass from the first slot of day index start_day; returns window dicts."""
        cnt = {"onset": 0, "ongoing": 0}
        last, out = None, []
        for t in range(max(HIST, start_day * STEPS), self.n - HOR):
            if cnt["onset"] >= cap and cnt["ongoing"] >= cap:
                break
            if last is not None and t - last < SPACING:
                continue
            if self.blk[t] or self.cov[t] < MIN_COV:
                continue
            if self.est[t]:
                if cnt["ongoing"] >= cap or not self.QL[t + 1:t + HOR + 1].any():
                    continue
                p = self.piou(t)
                if p > MAX_PIOU:
                    continue
                cond = "ongoing"
            else:
                if onset_rule == "clean" and self.hsum[t] > 0:
                    continue
                if cnt["onset"] >= cap or not (self.QV[t + 1:t + HOR + 1] & self.E[t + 1:t + HOR + 1]).any():
                    continue
                cond, p = "onset", 0.0
            cnt[cond] += 1
            sb = last is not None and t - last == SPACING
            out.append(dict(panel=self.panel, split=self.split, month=self.month, start=start_day,
                            cond=cond, k=cnt[cond], t=t, day=self.day0 + t // STEPS, T=t % STEPS,
                            date=self.dates[t // STEPS], piou=round(p, 4), gap=(t - last) if last is not None else -1,
                            spacing=bool(sb), est_T5=bool(self.est[t - 1]), cov_T5=float(self.cov[t - 1]),
                            blk_T5=bool(self.blk[t - 1]), piou_T5=round(self.piou(t - 1), 4)))
            last = t
        return out


def classify(spacing: bool, est_T5: bool, cov_T5: float = 1.0) -> str:
    if spacing:
        return "spacing"
    return "constr" if (est_T5 and cov_T5 >= MIN_COV) else "newborn"


def train_months(panel: str) -> list[str]:
    lay = layers(panel, "train")
    return sorted({d[:7] for d in lay["dates"]})


def cmd_check(args):
    rows = []
    for panel in T2_PANELS:
        tl = Timeline(panel, "train", "2030-06")
        wi = pd.read_csv(RELEASE / "task2" / panel / "train" / "window_index.csv")
        wi["t"] = [tl.dates.index(d) * STEPS + int(s) % STEPS
                   for d, s in zip(wi.date.astype(str), slot_of(wi.forecast_origin))]
        for rule in ("not_est", "clean"):
            sel = pd.DataFrame(tl.select(0, 5, rule))
            for cond, oc in (("onset", "queue_onset"), ("ongoing", "queue_ongoing")):
                off = set(wi.t[wi.condition == oc])
                rep = set(sel.t[sel.cond == cond]) if len(sel) else set()
                rows.append(dict(panel=panel, rule=rule, cond=cond, match=len(off & rep), n_off=len(off)))
    df = pd.DataFrame(rows)
    print(df.pivot_table(index="panel", columns=["rule", "cond"], values="match", aggfunc="sum").to_string())
    print(df.groupby(["rule", "cond"])[["match", "n_off"]].sum())


def cmd_build(args):
    rows = []
    for panel in T2_PANELS:
        for split in args.splits:
            lay = layers(panel, split)
            months = sorted({d[:7] for d in lay["dates"]})
            for m in months:
                tl = Timeline(panel, split, m, lay)
                ndays = len(tl.dates)
                for s in range(0, ndays - 2, args.start_every):
                    for w in tl.select(s, args.cap, args.onset_rule):
                        rows.append(w)
            print(panel, split, len(rows), flush=True)
    df = pd.DataFrame(rows)
    df["cls"] = [classify(a, b, c) if k == "ongoing" else "" for a, b, c, k in
                 zip(df.spacing, df.est_T5, df.cov_T5, df.cond)]
    CACHE.mkdir(parents=True, exist_ok=True)
    out = CACHE / args.out
    df.to_csv(out, index=False)
    on = df[df.cond == "ongoing"]
    print(out, len(df), "ongoing", len(on), "unique ongoing (panel,split,t)",
          len(on.drop_duplicates(["panel", "split", "t"])))
    print(on.drop_duplicates(["panel", "split", "t", "cls"]).groupby(["split", "cls"]).size().unstack())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check")
    s = sub.add_parser("build")
    s.add_argument("--splits", nargs="+", default=["train", "validation"])
    s.add_argument("--cap", type=int, default=5)
    s.add_argument("--start-every", type=int, default=1)
    s.add_argument("--onset-rule", default="not_est")
    s.add_argument("--out", default="windows.csv")
    a = ap.parse_args()
    if "private" in getattr(a, "splits", []):
        raise SystemExit("private split not allowed")
    {"check": cmd_check, "build": cmd_build}[a.cmd](a)
