#!/usr/bin/env python
"""Task 2 rule model: a recurrent-bottleneck prior for queue_onset, persistence for queue_ongoing.

Why these two rules
  queue_onset    The window selector keeps the earliest origin whose horizon holds a queue, so the
                 queue appears only at T+30 (organizer-confirmed, allowed). Speeds before T do not
                 say where: the links about to queue still run at ~1.65x the queue cutoff. But onsets
                 recur at the same bottlenecks, so we learn P(link queued at onset | time of day) from
                 every onset event in the 273 train days and mark the likely links at T+30 only.
  queue_ongoing  A visible queue is carried forward: the last observed queue state at each link,
                 repeated over the six horizon steps.

Information contract: only train (allowed in full) and the window history [T-60, T) are used.

Usage (from the "IEEE Big data" folder):
  python work/task2_rules.py tune                 # leave-one-day-out search on the train windows
  python work/task2_rules.py predict --split all  # write work/sub/queue_<split>.csv
"""
from __future__ import annotations

import argparse
import itertools
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import CACHE, CLEAN, RELEASE, ROOT, STEPS, T2_PANELS, aggregate, links, score_t2, slot_of  # noqa: E402

OUT = ROOT / "work" / "sub"
HIST_STEPS, HORIZON = 12, 6
# Chosen by `tune` (leave-one-day-out on the 80 train windows); see the README notes in the reply.
DEFAULTS = {"sigma_h": 1.0, "theta": 0.3, "weekend": 0.2, "carry": 2}
# Ongoing-window variants. "gapfill1km" bridges gaps of up to 1 km inside a visible queue and holds it over the
# horizon: on denoised (3x3 median) train labels it beats plain persistence 0.728 vs 0.702, on the raw noisy
# labels it is 0.006 worse; the leaderboard labels are noise-free, so it is tested there.
VARIANTS = {"persist": DEFAULTS,
            # Onset prior that also learns from the split's own onset events before T: validation and private were
            # generated from new seeds and some corridors queue at new places there (e.g. D7_I210_W, D12_I5_S).
            "monthaware": {**DEFAULTS, "month_aware": True, "n0": 10},
            "gapfill1km": {**DEFAULTS, "ongoing": "kinematic", "gap_km": 1.0, "trend": 6, "damp": 0.0,
                           "search_km": 3.0, "vmax": 2.0}}


# ---------------------------------------------------------------------------
# onset events from the full train split
# ---------------------------------------------------------------------------
def onset_events(panel: str) -> pd.DataFrame:
    """One row per (event, link): a queue appears at tod t after 25 queue-free minutes, and the hour
    before that holds no established queue - the same situation the onset selector picks."""
    path = CACHE / f"t2_onset_events_{panel}.parquet"
    src = CLEAN / panel / "train.parquet"
    if path.exists() and path.stat().st_mtime >= src.stat().st_mtime:
        return pd.read_parquet(path)
    L = links(panel)
    n = len(L)
    vcut = L.v_cut_kmh.to_numpy()
    rows, event = [], 0
    pf = pq.ParquetFile(src)
    for i in range(pf.num_row_groups):
        t = pf.read_row_group(i, columns=["ts", "speed_true", "is_score_eligible"])
        ts0 = pd.Timestamp(t.column("ts")[0].as_py())
        with np.errstate(invalid="ignore"):
            Q = (t.column("is_score_eligible").to_numpy()
                 & (t.column("speed_true").to_numpy() <= np.tile(vcut, STEPS))).reshape(STEPS, n)
        any_t = Q.any(axis=1)
        for k in range(HIST_STEPS + HORIZON, STEPS):
            if not any_t[k] or any_t[k - HORIZON + 1:k].any():
                continue
            H = Q[k - HORIZON - HIST_STEPS:k - HORIZON]           # history [T-60, T) with T = k-6
            if H.sum() >= 2 and H.sum(axis=0).max() >= 2:          # established queue -> ongoing
                continue
            for p in np.flatnonzero(Q[k]):
                rows.append((event, ts0.strftime("%Y-%m-%d"), ts0.weekday(), k, int(p)))
            event += 1
    ev = pd.DataFrame(rows, columns=["event", "date", "weekday", "tod", "pos"])
    CACHE.mkdir(parents=True, exist_ok=True)
    ev.to_parquet(path, index=False)
    return ev


def split_onset_events(panel: str, split: str) -> pd.DataFrame:
    """Onset events seen in a split's own released masked layer (visible eligible cells only). In validation
    and private the Task 2 horizons and buffers are blank, so a scored window never sees its own answer."""
    path = CACHE / f"t2_onset_events_{panel}_{split}_visible.parquet"
    src = CLEAN / panel / f"{split}.parquet"
    if path.exists() and path.stat().st_mtime >= src.stat().st_mtime:
        return pd.read_parquet(path)
    L = links(panel)
    n, vcut = len(L), L.v_cut_kmh.to_numpy()
    rows, event = [], 0
    pf = pq.ParquetFile(src)
    for i in range(pf.num_row_groups):
        t = pf.read_row_group(i, columns=["ts", "speed_clean"])
        ts0 = pd.Timestamp(t.column("ts")[0].as_py())
        with np.errstate(invalid="ignore"):
            v = t.column("speed_clean").to_numpy().astype(float).reshape(STEPS, n)
            Q = np.isfinite(v) & (v <= vcut)
        any_t = Q.any(axis=1)
        for k in range(HIST_STEPS + HORIZON, STEPS):
            if not any_t[k] or any_t[k - HORIZON + 1:k].any():
                continue
            H = Q[k - HORIZON - HIST_STEPS:k - HORIZON]
            if H.sum() >= 2 and H.sum(axis=0).max() >= 2:
                continue
            for p in np.flatnonzero(Q[k]):
                rows.append((event, ts0.strftime("%Y-%m-%d"), ts0.weekday(), k, int(p)))
            event += 1
    ev = pd.DataFrame(rows, columns=["event", "date", "weekday", "tod", "pos"])
    CACHE.mkdir(parents=True, exist_ok=True)
    ev.to_parquet(path, index=False)
    return ev


def month_aware_scores(panel: str, split: str, ev_train: pd.DataFrame, n_links: int, tod_target: int, weekday: int,
                       date: str, tod_origin: int, params: dict) -> np.ndarray:
    """Blend of the train onset prior and the split's own onset events strictly before the origin T.
    Weight of the split part = m / (m + n0), m = number of the split's events before T."""
    if split == "train":       # local check: the same month of train stands in for "this split"
        ev_s = ev_train[(ev_train.date.str[:7] == date[:7])]
        ev_t = ev_train[ev_train.date != date]
    else:
        ev_s, ev_t = split_onset_events(panel, split), ev_train
    ev_s = ev_s[(ev_s.date < date) | ((ev_s.date == date) & (ev_s.tod <= tod_origin))]
    s_train = onset_scores(ev_t, n_links, tod_target, weekday, None, params["sigma_h"], params["weekend"])
    m = ev_s.event.nunique()
    if m == 0:
        return s_train
    s_split = onset_scores(ev_s, n_links, tod_target, weekday, None, params["sigma_h"], params["weekend"])
    w = m / (m + params["n0"])
    return w * s_split + (1 - w) * s_train


def onset_scores(ev: pd.DataFrame, n_links: int, tod: int, weekday: int, date: str | None,
                 sigma_h: float, weekend: float) -> np.ndarray:
    """P(link queued at onset) weighted by time-of-day distance and weekday/weekend match."""
    if date is not None:                       # leave-one-day-out when scoring train windows
        ev = ev[ev.date != date]
    e = ev.drop_duplicates("event")[["event", "tod", "weekday"]]
    w = np.exp(-0.5 * ((e.tod.to_numpy() - tod) / (sigma_h * 12)) ** 2) if np.isfinite(sigma_h) else np.ones(len(e))
    same = (e.weekday.to_numpy() >= 5) == (weekday >= 5)
    w = w * np.where(same, 1.0, weekend)
    if w.sum() <= 0:
        return np.zeros(n_links)
    wmap = pd.Series(w, index=e.event.to_numpy())
    score = np.zeros(n_links)
    np.add.at(score, ev.pos.to_numpy(), ev.event.map(wmap).to_numpy())
    return score / w.sum()


# ---------------------------------------------------------------------------
# predictions in the official per-task format
# ---------------------------------------------------------------------------
@lru_cache(maxsize=None)
def travel_axis(panel: str) -> np.ndarray:
    """Position of every link (pos order) in km along the direction of travel (increasing downstream)."""
    import json
    rep = json.loads((CLEAN / "cleaning_report.json").read_text(encoding="utf-8"))
    share = rep[panel]["network"]["upstream_has_lower_milepost_share"]
    d = 1 if share is None or share >= 0.5 else -1
    return links(panel).milepost.to_numpy(float) * d


def clusters(x: np.ndarray, queued: np.ndarray, max_gap_km: float) -> list[tuple[float, float]]:
    """Queued stretches along the travel axis as (tail, head) in km; gaps up to max_gap_km are bridged."""
    xs = np.sort(x[queued])
    if len(xs) == 0:
        return []
    out, tail, head = [], xs[0], xs[0]
    for v in xs[1:]:
        if v - head <= max_gap_km:
            head = v
        else:
            out.append((tail, head))
            tail = head = v
    out.append((tail, head))
    return out


def ongoing_kinematic(h: pd.DataFrame, x: np.ndarray, vcut: np.ndarray, params: dict) -> np.ndarray:
    """Queue tails extrapolated at the speed they moved over the last `trend` steps; heads held at the
    bottleneck; every link between tail and head marked. Returns a (HORIZON, n) 0/1 array."""
    n = len(x)
    pred = np.zeros((HORIZON, n), dtype=np.int8)
    steps = sorted(h.slot.unique())
    if not steps:
        return pred
    # Queue state per history step, carrying a link's last valid reading forward up to `carry` steps.
    Q = np.full((len(steps), n), np.nan)
    for i, s in enumerate(steps):
        g = h[h.slot == s]
        Q[i, g.pos.to_numpy()] = (g.speed_kmh.to_numpy() <= vcut[g.pos.to_numpy()]).astype(float)
    for i in range(1, len(steps)):
        for k in range(1, params["carry"] + 1):
            if i - k < 0:
                break
            fill = np.isnan(Q[i]) & ~np.isnan(Q[i - k])
            Q[i, fill] = Q[i - k, fill]
    last = np.nan_to_num(Q[-1]).astype(bool)
    for tail, head in clusters(x, last, params["gap_km"]):
        # Tail trajectory of this stretch over the trend window.
        ts, tt = [], []
        for i in range(max(0, len(steps) - params["trend"]), len(steps)):
            qi = np.nan_to_num(Q[i]).astype(bool) & (x >= tail - params["search_km"]) & (x <= head)
            if qi.any():
                ts.append(i)
                tt.append(x[qi].min())
        v = np.polyfit(ts, tt, 1)[0] if len(ts) >= 3 else 0.0           # km per step (negative = growing)
        v = float(np.clip(v * params["damp"], -params["vmax"], params["vmax"]))
        for s in range(HORIZON):
            tail_s = tail + v * (s + 1)
            if tail_s <= head:
                pred[s, (x >= tail_s) & (x <= head)] = 1
    return pred


@lru_cache(maxsize=None)
def load_split(panel: str, split: str):
    L = links(panel)
    pos_of = dict(zip(L.link_id, L.pos))
    base = RELEASE / "task2" / panel / split
    win = pd.read_csv(base / "window_index.csv")
    tmpl = pd.read_csv(base / "sample_submission_queue.csv", usecols=["window_id", "timestamp", "link_id"])
    tmpl["slot"], tmpl["pos"] = slot_of(tmpl.timestamp), tmpl.link_id.map(pos_of).astype(np.int64)
    hist = pd.read_parquet(base / "window_history.parquet")
    hist["slot"], hist["pos"] = slot_of(hist.timestamp), hist.link_id.map(pos_of).astype(np.int64)
    return L, win, tmpl, hist, onset_events(panel)


def predict_panel(panel: str, split: str, params: dict, loo: bool = False) -> pd.DataFrame:
    L, win, tmpl, hist, ev = load_split(panel, split)
    n = len(L)
    vcut = L.v_cut_kmh.to_numpy()
    out = []
    for w in win.itertuples(index=False):
        T = int(slot_of(pd.Series([w.forecast_origin]))[0])
        cells = tmpl[tmpl.window_id == w.window_id].copy()
        pred = np.zeros((HORIZON, n), dtype=np.int8)               # rows: T+5 .. T+30
        if w.condition == "queue_onset":
            origin = pd.Timestamp(w.forecast_origin)
            tod = origin.hour * 12 + origin.minute // 5 + HORIZON
            if params.get("month_aware"):
                s = month_aware_scores(panel, split, ev, n, tod, origin.weekday(), w.date, tod - HORIZON, params)
            else:
                s = onset_scores(ev, n, tod, origin.weekday(), w.date if loo else None,
                                 params["sigma_h"], params["weekend"])
            chosen = np.flatnonzero(s >= params["theta"])
            if len(chosen) == 0 and s.max() > 0:                   # a queue is guaranteed: mark the best
                chosen = [int(np.argmax(s))]
            pred[HORIZON - 1, chosen] = 1
        else:
            h = hist[(hist.window_id == w.window_id) & hist.slot.lt(T)]
            h = h[h.is_score_eligible.astype(bool) & h.speed_kmh.notna()]
            if params.get("ongoing", "persist") == "kinematic":
                pred = ongoing_kinematic(h, travel_axis(panel), vcut, params)
            else:
                h = h[h.slot >= h.slot.max() - params["carry"]]     # last observation within `carry` steps
                last = h.sort_values("slot").groupby("pos").tail(1)
                queued = last.pos[last.speed_kmh.to_numpy() <= vcut[last.pos.to_numpy()]].to_numpy()
                pred[:, queued] = 1
        step = (cells.slot.to_numpy() - T) // 1 - 1
        ok = (step >= 0) & (step < HORIZON)
        cells["queue_pred"] = 0
        cells.loc[ok, "queue_pred"] = pred[step[ok], cells.pos.to_numpy()[ok]]
        out.append(cells)
    return pd.concat(out, ignore_index=True)


def score_train(params: dict) -> tuple[float, dict]:
    pred = pd.concat([predict_panel(p, "train", params, loo=True) for p in T2_PANELS])
    detail, panel_scores = score_t2(pred[["window_id", "slot", "pos", "queue_pred"]], T2_PANELS)
    cond = detail.groupby("condition").IoU.mean().to_dict()
    return aggregate(panel_scores)[1], cond


def cmd_tune(_args) -> None:
    for p in T2_PANELS:
        ev = onset_events(p)
        print(f"  {p}: {ev.event.nunique()} onset events in train")
    grid = {"sigma_h": [0.5, 1.0, 2.0, 4.0, np.inf], "theta": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "weekend": [1.0, 0.2, 0.0], "carry": [DEFAULTS["carry"]]}
    rows = []
    for values in itertools.product(*grid.values()):
        params = dict(zip(grid, values))
        s, cond = score_train(params)
        rows.append({**params, "S_queue": s, **cond})
    res = pd.DataFrame(rows).sort_values("queue_onset", ascending=False)
    print("\nOnset rule, leave-one-day-out on train windows (top 15):")
    print(res.head(15).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    rows = []
    for carry in (0, 1, 2, 3):
        s, cond = score_train({**DEFAULTS, "carry": carry})
        rows.append({"carry": carry, "S_queue": s, **cond})
    print("\nOngoing rule (carry = how many earlier steps may stand in for a missing last reading):")
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.4f}"))


def cmd_predict(args) -> None:
    splits = ["train", "validation", "private"] if args.split == "all" else [args.split]
    params = VARIANTS[args.variant]
    suffix = "" if args.variant == "persist" else f"_{args.variant}"
    OUT.mkdir(parents=True, exist_ok=True)
    for split in splits:
        pred = pd.concat([predict_panel(p, split, params, loo=(split == "train")) for p in T2_PANELS])
        path = OUT / f"queue_{split}{suffix}.csv"
        pred[["window_id", "timestamp", "link_id", "queue_pred"]].to_csv(path, index=False)
        print(f"wrote {path} ({len(pred):,} rows, {int(pred.queue_pred.sum()):,} queued cells)")
    if "train" in splits:
        s, cond = score_train(params)
        print(f"train proxy S_queue = {s:.4f}  by condition {cond}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("tune")
    p = sub.add_parser("predict")
    p.add_argument("--split", default="all", choices=["train", "validation", "private", "all"])
    p.add_argument("--variant", default="persist", choices=list(VARIANTS))
    args = ap.parse_args()
    {"tune": cmd_tune, "predict": cmd_predict}[args.cmd](args)


if __name__ == "__main__":
    main()
