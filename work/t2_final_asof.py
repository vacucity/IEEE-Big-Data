#!/usr/bin/env python
"""Final package: per-window truncation ("as-of T") check of the FINAL Task 2 queue rows of submission A.

The organizers rerun the code with every data row stamped after each Task 2 window's origin T deleted and check that
the predictions do not change.  For each of the 160 official windows W (origin T on day d of its month) this script
builds a truncated view of every data source, regenerates W's queue rows from that view only, and compares them cell
by cell with the final rows: March = sub/queue_validation_v54_onset.csv, April = sub/queue_private_v72_april.csv
(both first checked against the queue rows of submissions/v84_observation_flow.zip = A; B = A + the 4 hedge cells of
results/t2_v66_hedge/rule.json).

Truncated view of W (in memory; the official months are read only through it while W is regenerated)
  window_index.csv    every corridor, both months: origins after T removed (W's own row kept, future_coverage blanked
                      in W's month); a March window sees no April rows at all, an April window all of March
  window_history      rows of the remaining windows stamped <= T
  t2_grids V/Q/O/B    masked layer (clean speed / flow / occupancy, blackout flag), rebuilt here from clean_data and
                      asserted equal to cache/t2_grids: days after d removed, rows of day d after T blanked (NaN / 0);
                      April = zero days for a March window
  class day rows      speed_clean / is_score_eligible of the V52 class: rows after T blanked, later days raise
Regenerated inside the view (no official cache, probs pickle or class file is read)
  ongoing  V27 members (CNN x3 / BiLSTM x2 on window_history [T-60, T-5], the t2_ongoing_v27.official_cache inputs;
           ml2 = task2_ml2.window_probs2 with the shipped booster; v8 := NaN, see below), V46 shifted members
           (t2_ongoing_v27.neural_probs / gbm_probs on [T-55, T]), gap-0 members (t2_slotT.g0_probs: V48 nets, April
           also the six V72 March fine-tunes), ml2g0, catg0 -> V48 / V72 blend (t2_v71_april.decide,
           t2_final_ongoing.chain_p) -> V52 class (t2_v52_selrule.official_classes) and, for March, the CPU V48 chain
           (t2_v52_prodcache.members) -> t2_v52_selrule.decide('hyb')
  onset    V54 corridors: the official case row (t2_v54_onset.build_cases code: window_history + grid row T) +
           t2_v54_onset.choose with decisions.json; D7_I10_W t2_onset.official_onset; D7_I405_N t2_v30_onset rf5;
           D7_I10_E / D7_I210_E ml2 onset probability >= 0.5 (argmax fallback); ml2 = task2_ml2.window_probs2
Train-only inputs are read as usual: train grids, onset events, the V30 train rows, group pairs, models, and the train
fit rows of the V54 choosers (recomputed here from train and asserted equal to the cases_*.pkl rows).

Guard: while W is regenerated, reading any file whose path below the project root contains 'validation', 'private' or
'official_' raises (the release files are served by the view), and a March window may not load a model tuned on
March (*_ft*.pt, *_hw*.pt, *_m{a,b,all}.txt).  Every file read is logged (asof_files_read.txt).

GPU batch shape: the bf16 nets give batch-size dependent floats (up to 3e-2 between a batch of 5 and of 1), but a row's
output depends only on its own input and the batch shape (zero / copy padding of the other rows is bitwise neutral,
GPU and CPU).  Production ran the 5 ongoing windows of a corridor-month in one batch, so W runs in a batch of 5 copies
of itself at its production position: bitwise the production floats, and no other window's data.  --natural also runs
the batch the production code forms on a truncated release (only the windows with origin <= T present) and reports
the cells that move from that float noise alone (information, not part of PASS).
v8 (the task2_ml.py LightGBM key of the V27 cache, refitted inside task2_ml.py predict, no shipped booster) is NaN in
the view: it reaches no decision (agent A's zeroing test); a use would propagate the NaN into the blend.

Passes per window: trunc (rows <= T kept, the organizers' rule) -> PASS iff 0 differing cells for all 160 windows;
control (row T deleted too; the gap-0 members and V54's JT read it) and, onset windows only, control_T5 (rows >= T-5
deleted as well; every onset rule reads T-5) must move cells, otherwise the check would be vacuous.

  python work/t2_final_asof.py [--out-dir work/results/t2_asof] [--windows ID ...] [--natural] [--no-control]
                               [--no-zip]
Outputs (refuses to overwrite): <out>/asof.log, asof_windows.csv, asof_files_read.txt.  GPU; reads nothing it writes.
"""
from __future__ import annotations

import os

for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(key, "4")

import argparse  # noqa: E402
import builtins  # noqa: E402
import dataclasses  # noqa: E402
import json  # noqa: E402
import pickle  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
import zipfile  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

warnings.simplefilter("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import t2_grids  # noqa: E402

ORIG_LOAD = t2_grids.load            # t2_v52_prodcache replaces t2_grids.load on import; production never imports it
import t2_cnn as C  # noqa: E402
import t2_ens  # noqa: E402
import t2_final_onset as F1  # noqa: E402
import t2_final_ongoing as F2  # noqa: E402
import t2_onset as O  # noqa: E402
import t2_pseudo as PS  # noqa: E402
import t2_rnn as R  # noqa: E402
import t2_slotT as S  # noqa: E402
import t2_v30_onset as V30  # noqa: E402
import t2_v50_april as A  # noqa: E402
import t2_v52_d12s_t15 as D12  # noqa: E402
import t2_v52_selrule as S52  # noqa: E402
import t2_v54_onset as V54  # noqa: E402
import t2_v71_april as V71  # noqa: E402
import task2_ml  # noqa: E402
import task2_ml2  # noqa: E402
import t2_ongoing_v27 as V27  # noqa: E402
import t2_v52_prodcache as PC  # noqa: E402

t2_grids.load = ORIG_LOAD
from evaluate import CACHE, CLEAN, RELEASE, ROOT, STEPS, T2_PANELS, links, slot_of  # noqa: E402
from t2_cat_g0 import cat_g0_probs  # noqa: E402
from t2_v33_adaptive_iou import adaptive_iou  # noqa: E402
from t2_v52_selrule_eval import chain_prob, pools  # noqa: E402

WORK = ROOT / "work"
MODELS = WORK / "models"
ZIP_A = WORK / "submissions" / "v84_observation_flow.zip"
SPLITS = ("validation", "private")              # March 2031 before April 2031 (asserted)
HOR = 6
BATCH = 5                                       # ongoing windows per corridor-month = production batch (asserted)
FORBIDDEN = re.compile(r"(validation|private|official_)", re.I)
MARCH_TUNED = re.compile(r"_(ft|hw\d+)(a|b|all)?\.pt$|_m(a|b|all)\.txt$")
ML2_BOOSTERS = {k: MODELS / f"t2_ml2_{k}_probs.txt" for k in ("onset", "ongoing")}   # the sub/probs pkl producers
V27_NETS = (("cnn0", "cnn", ""), ("cnn1", "cnn", "_s1"), ("cnn2", "cnn", "_s2"), ("rnn0", "rnn", ""),
            ("rnn1", "rnn", "_s1"))
LOG = None


def log(msg: str = "") -> None:
    print(msg, flush=True)
    if LOG is not None:
        LOG.write(msg + "\n")
        LOG.flush()


# ----------------------------------------------------------------------------------------------------------
# guard
# ----------------------------------------------------------------------------------------------------------
class Guard:
    def __init__(self):
        self.armed = False
        self.march = False
        self.read: set[str] = set()
        self.root = str(ROOT.resolve()).lower()

    def check(self, path) -> None:
        if not self.armed:
            return
        s = str(Path(path).resolve())
        self.read.add(s)
        rel = s[len(self.root):] if s.lower().startswith(self.root) else ""
        if FORBIDDEN.search(rel):
            raise RuntimeError(f"as-of guard: forbidden read during regeneration: {s}")
        if self.march and MARCH_TUNED.search(rel):
            raise RuntimeError(f"as-of guard: a March window loads a March-tuned model: {s}")


GUARD = Guard()


def _path_arg(a, k):
    p = a[0] if a else next((k[x] for x in ("filepath_or_buffer", "path", "file", "source", "fname", "f",
                                            "model_file") if x in k), None)
    return p if isinstance(p, (str, os.PathLike)) else None


def _wrap(owner, name, serve=None):
    f = getattr(owner, name)

    def wrapped(*a, **k):
        p = _path_arg(a, k)
        if p is not None and GUARD.armed:
            if serve is not None:
                hit = serve(p, a, k)
                if hit is not None:
                    return hit
            GUARD.check(p)
        return f(*a, **k)
    setattr(owner, name, wrapped)


def install_guard(view: "View") -> None:
    import catboost
    import lightgbm
    import torch
    _wrap(pd, "read_csv", view.serve)
    _wrap(pd, "read_parquet", view.serve)
    for owner, name in ((pq, "ParquetFile"), (pq, "read_table"), (np, "load"), (torch, "load"), (builtins, "open")):
        _wrap(owner, name)
    for name in ("open", "read_text", "read_bytes"):
        orig = getattr(Path, name)

        def method(self_path, *a, _orig=orig, **k):
            GUARD.check(self_path)
            return _orig(self_path, *a, **k)
        setattr(Path, name, method)
    booster = lightgbm.Booster

    def lgb_booster(*a, **k):
        if k.get("model_file") is not None:
            GUARD.check(k["model_file"])
        return booster(*a, **k)
    lightgbm.Booster = lgb_booster
    cb_load = catboost.CatBoost.load_model

    def cb_load_model(self_model, fname=None, *a, **k):
        if fname is not None:
            GUARD.check(fname)
        return cb_load(self_model, fname, *a, **k)
    catboost.CatBoost.load_model = cb_load_model


def cached_loader(fn, path_of):
    """Model weights are fixed artifacts: load once, but run the guard on every request."""
    memo = {}

    def load(*a):
        GUARD.check(path_of(*a))
        if a not in memo:
            memo[a] = fn(*a)
        return memo[a]
    load.cache_clear = lambda: None
    return load


# ----------------------------------------------------------------------------------------------------------
# full sources (read before the guard is armed) and the truncated view
# ----------------------------------------------------------------------------------------------------------
def grid_from_clean(panel: str, split: str) -> tuple[dict, np.ndarray]:
    """t2_grids.build for an official split, in memory (+ is_score_eligible E for the V52 class)."""
    n = len(links(panel))
    pf = pq.ParquetFile(CLEAN / panel / f"{split}.parquet")
    cols = ["date", "weekday", "is_score_eligible", "is_t2_blackout", "speed_clean", "flow_clean", "occ_clean"]
    D = pf.num_row_groups
    g = {k: np.full((D, STEPS, n), np.nan, np.float32) for k in ("V", "Q", "O")}
    g["B"] = np.zeros((D, STEPS, n), np.int8)
    E = np.zeros((D, STEPS, n), bool)
    dates, wd = [], []
    for i in range(D):
        t = pf.read_row_group(i, columns=cols)
        assert t.num_rows == STEPS * n
        dates.append(t.column("date")[0].as_py())
        wd.append(int(t.column("weekday")[0].as_py()))
        E[i] = t.column("is_score_eligible").to_numpy(zero_copy_only=False).astype(bool).reshape(STEPS, n)
        col = lambda c: t.column(c).to_numpy().astype(np.float32).reshape(STEPS, n)
        g["V"][i], g["Q"][i], g["O"][i] = col("speed_clean"), col("flow_clean"), col("occ_clean")
        g["B"][i] = t.column("is_t2_blackout").to_numpy().reshape(STEPS, n)
    g["dates"], g["weekday"] = np.array(dates), np.array(wd)
    with np.load(t2_grids.GRIDS / f"{panel}_{split}.npz") as z:      # the cache the production chain read
        for k in ("V", "Q", "O", "B"):
            assert np.array_equal(z[k], g[k], equal_nan=k != "B"), (panel, split, k, "cache/t2_grids differs")
        assert [str(x) for x in z["dates"]] == [str(x) for x in g["dates"]]
    return g, E


class View:
    """Every official source of both months, truncated at one window's origin on request."""

    def __init__(self):
        self.wi, self.hist, self.grid_full, self.E = {}, {}, {}, {}
        for p in T2_PANELS:
            for s in SPLITS:
                base = RELEASE / "task2" / p / s
                wi = pd.read_csv(base / "window_index.csv")
                h = pd.read_parquet(base / "window_history.parquet")
                self.wi[(p, s)] = (wi, pd.to_datetime(wi.forecast_origin, utc=True).dt.tz_localize(None).to_numpy())
                self.hist[(p, s)] = (h, pd.to_datetime(h.timestamp, utc=True).dt.tz_localize(None).to_numpy())
                self.grid_full[(p, s)], self.E[(p, s)] = grid_from_clean(p, s)
        last = {s: max(str(x) for p in T2_PANELS for x in self.grid_full[(p, s)]["dates"]) for s in SPLITS}
        first = {s: min(str(x) for p in T2_PANELS for x in self.grid_full[(p, s)]["dates"]) for s in SPLITS}
        assert last["validation"] < first["private"], (first, last)
        self.cur = None
        self.root = RELEASE.resolve()

    # ---- truncation point
    def set(self, split: str, date: str, T: int, Tt: np.datetime64, cut_from: int, panel: str) -> dict:
        """Truncate at the origin Tt (slot T of `date`): rows of that date from slot cut_from on are deleted (T + 1:
        the rule; T: control; T - 1: control_T5), so are all later days and the later month, in every corridor."""
        hist_cut = Tt - np.timedelta64(5 * (T + 1 - cut_from), "m")      # last history timestamp kept
        self.cur = dict(split=split, date=date, T=T, Tt=Tt, cut=cut_from, hist_cut=hist_cut)
        self.memo = {}
        stats = dict(origins_deleted=0, history_rows_deleted=0, days_deleted=0, visible_cells_deleted=0)
        for (p, s), (wi, tt) in self.wi.items():
            keep = self._keep_index(s, tt)
            stats["origins_deleted"] += int((~keep).sum())
            h, ts = self.hist[(p, s)]
            stats["history_rows_deleted"] += int((~self._keep_hist(s, h, ts, set(wi.window_id[keep]))).sum())
            g, o = self.grid_full[(p, s)], self._order(s)
            nd = 0 if o > 0 else len(g["dates"]) if o < 0 else self._day(p, s) + 1
            vis = np.isfinite(g["V"][nd:]).sum() + (np.isfinite(g["V"][nd - 1, cut_from:]).sum() if o == 0 else 0)
            stats["visible_cells_deleted"] += int(vis)                   # masked-layer cells, all corridors
            stats["days_deleted"] += (len(g["dates"]) - nd) if p == panel else 0
        return stats

    def _day(self, p: str, s: str) -> int:
        return [str(x) for x in self.grid_full[(p, s)]["dates"]].index(self.cur["date"])

    def _order(self, s: str) -> int:
        return SPLITS.index(s) - SPLITS.index(self.cur["split"])

    def _keep_index(self, s, tt):
        o = self._order(s)
        return np.ones(len(tt), bool) if o < 0 else np.zeros(len(tt), bool) if o > 0 else tt <= self.cur["Tt"]

    def _keep_hist(self, s, h, ts, ids):
        o = self._order(s)
        if o != 0:
            return np.full(len(h), o < 0)
        return (ts <= self.cur["hist_cut"]) & h.window_id.isin(ids).to_numpy()

    # ---- providers
    def window_index(self, p: str, s: str) -> pd.DataFrame:
        wi, tt = self.wi[(p, s)]
        out = wi[self._keep_index(s, tt)].copy()
        if self._order(s) == 0:
            out["future_coverage"] = np.nan
        return out

    def window_history(self, p: str, s: str) -> pd.DataFrame:
        wi, tt = self.wi[(p, s)]
        h, ts = self.hist[(p, s)]
        return h[self._keep_hist(s, h, ts, set(wi.window_id[self._keep_index(s, tt)]))].copy()

    def grid(self, p: str, s: str) -> dict:
        if s == "train":
            return ORIG_LOAD(p, s)
        if (p, s) in self.memo:
            return self.memo[(p, s)]
        g, o = self.grid_full[(p, s)], self._order(s)
        if o < 0:
            out = g
        else:
            nd = 0 if o > 0 else self._day(p, s) + 1
            out = {k: g[k][:nd].copy() for k in ("V", "Q", "O", "B", "dates", "weekday")}
            if o == 0:
                c = self.cur["cut"]
                for k in ("V", "Q", "O"):
                    out[k][nd - 1, c:] = np.nan
                out["B"][nd - 1, c:] = 0
        self.memo[(p, s)] = out
        return out

    def day_rows(self, p: str, s: str, date: str):
        """t2_v52_selrule._day_rows in the view."""
        g, o = self.grid_full[(p, s)], self._order(s)
        if o > 0:
            raise KeyError(f"{date}: month after the origin (deleted)")
        di = [str(x) for x in g["dates"]].index(date)
        V, E = g["V"][di].copy(), self.E[(p, s)][di].copy()
        if o == 0:
            if di > self._day(p, s):
                raise KeyError(f"{date}: day after the origin (deleted)")
            if di == self._day(p, s):
                V[self.cur["cut"]:] = np.nan
                E[self.cur["cut"]:] = False
        return V, E

    def serve(self, path, a, k):
        """pd.read_csv / read_parquet of a release window_index / window_history file -> the view's frame."""
        try:
            rel = Path(path).resolve().relative_to(self.root).parts
        except ValueError:
            return None
        if len(rel) != 4 or rel[0] != "task2" or rel[2] not in SPLITS:
            return None
        assert len(a) <= 1 and not k, f"unexpected read arguments for {path}: {a[1:]} {k}"
        if rel[3] == "window_index.csv":
            return self.window_index(rel[1], rel[2])
        if rel[3] == "window_history.parquet":
            return self.window_history(rel[1], rel[2])
        return None


VIEW: View | None = None


def grid_provider(panel, split):
    return VIEW.grid(panel, split)


grid_provider.cache_clear = lambda: None


def forbidden(name):
    def f(*a, **k):
        raise RuntimeError(f"as-of check: {name} must not be called (it reads a full-month cache)")
    return f


# ----------------------------------------------------------------------------------------------------------
# train-only inputs of the V54 choosers
# ----------------------------------------------------------------------------------------------------------
def v54_fit_rows(panel: str) -> list[dict]:
    """t2_v54_onset.build_cases part (1): the train pseudo onset windows the choosers are fitted on."""
    g = ORIG_LOAD(panel, "train")
    fit = []
    for w in PS.windows(panel, "train"):
        if w.cond != "onset":
            continue
        t = O.target(panel, w)
        if t is None:
            continue
        f = V54.feats13(panel, g["Q"][w.day, w.T - PS.HIST:w.T + 1].astype(float))
        fit.append(dict(month=w.date[:7], date=w.date, day=w.day, T=w.T, t=t, **f))
    return fit


def same_rows(a: list[dict], b: list[dict], keys) -> bool:
    return len(a) == len(b) and all(all(np.array_equal(np.asarray(x[k]), np.asarray(y[k]), equal_nan=True)
                                        for k in keys) for x, y in zip(a, b))


# ----------------------------------------------------------------------------------------------------------
# regeneration of one window inside the view
# ----------------------------------------------------------------------------------------------------------
def ml2_prob(panel: str, split: str, wid: str, kind: str) -> np.ndarray:
    """(6, n) ml2 probability of window wid: task2_ml2.window_probs2 with the shipped booster (sub/probs producer)."""
    import lightgbm as lgb
    b = lgb.Booster(model_file=str(ML2_BOOSTERS[kind]))
    zero = lambda X: np.zeros(len(X))
    fns = {"onset": zero, "ongoing": zero, kind: lambda X: b.predict(X[b.feature_name()])}
    for w in task2_ml2.window_probs2(panel, split, fns, by_fold=False):
        if w["window_id"] == wid:
            return w["prob"]
    raise KeyError(wid)


def v54_official_row(panel: str, split: str, wid: str) -> dict:
    """t2_v54_onset.build_cases part (3) for one official window, from the view."""
    wi = pd.read_csv(RELEASE / "task2" / panel / split / "window_index.csv")
    meta = V54.selector_meta(wi)
    g = V54.load(panel, split)
    dmap = {str(d): i for i, d in enumerate(g["dates"])}
    Q, tod = O.official_history_Q(panel, split)[wid]
    r = meta.loc[wid]
    T = int(tod)
    day = dmap[str(r.date)]
    Qg = g["Q"][day, T - PS.HIST:T + 1].astype(float)
    both = np.isfinite(Qg[:PS.HIST]) & np.isfinite(Q)
    assert np.allclose(Qg[:PS.HIST][both], Q[both], rtol=1e-4, atol=0.5), wid
    Q13 = np.vstack([Q, Qg[PS.HIST:]])
    return dict(split=split, wid=wid, date=str(r.date), origin=r.forecast_origin, T=T, tod=tod, gap=int(r.gap),
                spacing=bool(r.spacing), k=int(r.k), **V54.feats13(panel, Q13))


def onset_rows(panel: str, split: str, wid: str, ctx: dict) -> tuple[np.ndarray, str, dict]:
    """(6, n) onset prediction (the T+30 link set), rule name, diagnostics."""
    n = len(links(panel))
    diag = {}
    if panel in V54.PANELS:
        dec = ctx["dec"][panel]
        rule = dec["rule"] if (split == "validation" or dec["apply_april"]) else V54.PROD[panel]
        row = v54_official_row(panel, split, wid)
        diag["v54_row"] = row
        ml2o = {}
        if rule.startswith("ml2"):
            ml2o[wid] = diag["ml2"] = ml2_prob(panel, split, wid, "onset")[HOR - 1]
        ((s, _),) = V54.choose(panel, rule, [row], ml2o)
    elif panel == "D7_I10_W":
        rule = "flow logit (t2_onset)"
        _, lk = O.official_onset(panel, split)[wid]
        s = np.zeros(n, bool)
        s[lk] = True
    elif panel == "D7_I405_N":
        rule = "rf5 (t2_v30_onset)"
        rec = {r[0]: r for r in V30.official(panel, split)}[wid]
        s = ctx["rf5"].predict({"F": rec[2][None]})[0]
    else:
        rule = "ml2 >= 0.5"
        p = diag["ml2"] = ml2_prob(panel, split, wid, "onset")[HOR - 1]
        s = p >= F1.ML2_TH
        if not s.any():
            s[int(np.argmax(p))] = True
    N = np.zeros((HOR, n), bool)
    N[HOR - 1] = np.asarray(s, bool)
    return N, rule, diag


def v27_members(panel: str, split: str, wids: list[str]) -> dict:
    """The network members of t2_ongoing_v27.official_cache (same inputs from window_history [T-60, T-5], same nets,
    one batch), for the windows wids of the view."""
    st = C.static(panel)
    perm = st["perm"]
    L = links(panel)
    n = len(L)
    pos = dict(zip(L.link_id, range(n)))
    vcut = L.v_cut_kmh.to_numpy(np.float32)[perm]
    static_ch = np.stack([st["ffs"] / 120, st["lanes"] / 6, st["cap"] / st["lanes"] / 2500,
                          np.log(st["length"] + .01) / 3, st["n_on"], st["n_off"]]).astype(np.float32)
    base = RELEASE / "task2" / panel / split
    win = pd.read_csv(base / "window_index.csv").set_index("window_id")
    hist = pd.read_parquet(base / "window_history.parquet")
    hist["pos"] = hist.link_id.map(pos)
    hist["slot"] = slot_of(hist.timestamp)
    xs = []
    for wid in wids:
        w = win.loc[wid]
        origin = pd.Timestamp(w.forecast_origin)
        T = int(slot_of(pd.Series([w.forecast_origin]))[0])
        h = hist[hist.window_id == wid]
        si = (h.slot - (T - 12)).to_numpy()
        mask = (h.is_score_eligible.astype(bool) & h.speed_kmh.notna()).to_numpy() & (si >= 0) & (si < 12)
        assert (pd.to_datetime(h.loc[mask, "timestamp"]) < origin).all()
        g = []
        for col in ("speed_kmh", "flow_vph", "occupancy"):
            a = np.full((12, n), np.nan, np.float32)
            a[si[mask], h.pos.to_numpy()[mask]] = h[col].to_numpy()[mask]
            g.append(a[:, perm])
        xs.append(C.make_x(*g, static_ch, st["ffs"], st["cap"], vcut, T, origin.weekday()))
    out = {}
    for name, kind, tag in V27_NETS:
        m = C.load_model(tag, 96) if kind == "cnn" else R.load_model(tag)
        pr = C.predict_probs(m, xs) if kind == "cnn" else R.predict_probs(m, xs)
        a = np.zeros((len(xs), 6, n), np.float32)
        a[:, :, perm] = pr
        out[name] = a
    return out


def view_ongoing(panel: str, split: str) -> list[str]:
    wi = pd.read_csv(RELEASE / "task2" / panel / split / "window_index.csv")
    return wi.window_id[wi.condition == "queue_ongoing"].tolist()


def ongoing_rows(panel: str, split: str, wid: str, mode: str) -> tuple[np.ndarray, str, dict]:
    """(6, n) final ongoing prediction of window wid from the view.  mode 'shape': a batch of BATCH copies of the
    window at its production position; 'natural': the view's ongoing windows (origin <= T) in one batch."""
    D = C.PanelData(panel, split, unmasked=False)
    wi = pd.read_csv(RELEASE / "task2" / panel / split / "window_index.csv").set_index("window_id")
    ids = view_ongoing(panel, split)
    i0 = ids.index(wid)

    def win(x):
        r = wi.loc[x]
        d = D.dates.index(str(r.date))
        return PS.Win(panel, split, d, int(slot_of(pd.Series([r.forecast_origin]))[0]) % STEPS, "ongoing",
                      str(r.date), int(D.weekday[d]))
    if mode == "shape":
        wids, i = [wid] * BATCH, i0
    else:
        wids, i = ids, i0
    ws = [win(x) for x in wids]
    one = lambda dd: {k: v[i:i + 1] for k, v in dd.items()}
    n = D.n
    z = v27_members(panel, split, wids)
    z["v8"] = np.full((len(ws), 6, n), np.nan, np.float32)            # pass-through key, no shipped producer
    z["ml2"] = np.repeat(ml2_prob(panel, split, wid, "ongoing")[None], len(ws), 0)
    shifted = [dataclasses.replace(w, T=w.T + 1) for w in ws]          # history [T-55, T]
    sh = V27.neural_probs(D, shifted)
    sh["v8"], sh["ml2"] = V27.gbm_probs(shifted)
    specs = A.V48_SPECS + (V71.V72_SPECS if split == "private" else [])
    g0 = S.g0_probs(D, ws, specs)
    g0["ml2g0"] = S.ml2_g0_probs(ws)
    g0["catg0"] = cat_g0_probs(ws)
    z1, sh1, g01 = one(z), one(sh), one(g0)
    hy, pool = A.pool_of(z1, sh1, g01)
    for fam in ("rnnft48", "rnnftx48", "cnnftx48", "rnnft96", "cnnftx96"):
        ks = [k for k in sorted(g01) if k.startswith(fam + "_")]
        if ks:
            pool[fam] = np.mean([g01[k] for k in ks], 0)
    pool = {k: v for k, v in pool.items() if v is not None}
    w = F2.WEIGHTS[split]
    P = V71.decide(hy, pool, panel, w)[0]
    p = F2.chain_p(pool, panel, w)[0]
    assert np.array_equal(adaptive_iou(p), P), (wid, "chain_p / decide mismatch")
    cls = S52.official_classes(panel, split).set_index("window_id")
    c = cls.loc[wid].to_dict()
    c["panel"], c["window_id"] = panel, wid
    cpu = None
    if split == "validation":           # V52 conditioned March on the CPU V48 chain (t2_v52_prodcache official)
        ft, PC.NETS_FT = PC.NETS_FT, []                # March-tuned members: not part of the V48 chain, not loaded
        try:
            cpu = PC.members(panel, split, ws, ["all"] * len(ws))
        finally:
            PC.NETS_FT = ft
        for nm, _, _ in ft:
            cpu[nm] = np.full_like(cpu["rnn2a"], np.nan)
        pc = chain_prob(pools(one(cpu)), panel, S52.CHAIN[split])[0]
    else:
        pc = p
    N, tag, acc = S52.decide(S52.RULE, P, pc, adaptive_iou(pc), c)
    diag = dict(z=z1, sh=sh1, g0=g01, cls=c, cpu=one(cpu) if cpu is not None else None,
                rows_identical=bool(mode != "shape" or all(np.array_equal(v[0], v[j]) for v in
                                                          (z["rnn0"], g0["rnng0_0"]) for j in range(BATCH))))
    return N.astype(bool), f"{c['cls']}/{tag or '-'}", diag


# ----------------------------------------------------------------------------------------------------------
def expected(split: str) -> dict[str, np.ndarray]:
    """window_id -> (6, n) final queue rows of submission A."""
    q = pd.read_csv(F1.FINAL_Q[split], dtype={"link_id": str})
    q["slot"] = slot_of(q.timestamp)
    out = {}
    for panel in T2_PANELS:
        wi = pd.read_csv(RELEASE / "task2" / panel / split / "window_index.csv")
        T0 = dict(zip(wi.window_id, slot_of(wi.forecast_origin)))
        pos = dict(zip(links(panel).link_id.astype(str), range(len(links(panel)))))
        n = len(pos)
        for wid, r in q[q.window_id.isin(T0)].groupby("window_id"):
            E = np.full((HOR, n), -1, np.int64)
            E[(r.slot.to_numpy() - T0[wid] - 1).astype(int), r.link_id.map(pos).to_numpy()] = r.queue_pred.to_numpy()
            assert (E >= 0).all(), wid
            out[wid] = E.astype(bool)
        assert set(out) >= set(wi.window_id), (panel, split)
    return out


def anchor_zip() -> None:
    """The two final queue files == the queue rows of submission A (via kaggle_public/submission_key.csv)."""
    key = pd.read_csv(RELEASE / "submission_key.csv", usecols=["submission_id", "task", "window_id", "timestamp",
                                                               "link_id"], dtype=str)
    key = key[key.task == "queue"].drop(columns="task")
    with zipfile.ZipFile(ZIP_A) as z:
        sub = pd.read_csv(z.open(z.namelist()[0]), usecols=["submission_id", "task", "queue_pred"], dtype=str)
    sub = sub[sub.task == "queue"].drop(columns="task")
    m = key.merge(sub, on="submission_id", how="inner", validate="one_to_one")
    assert len(m) == len(key) == len(sub), (len(m), len(key), len(sub))
    total = 0
    for split, path in F1.FINAL_Q.items():
        q = pd.read_csv(path, dtype=str)
        mm = q.merge(m, on=["window_id", "timestamp", "link_id"], how="left", validate="one_to_one")
        assert mm.queue_pred_y.notna().all(), path
        d = int((mm.queue_pred_x.astype(int) != mm.queue_pred_y.astype(float).astype(int)).sum())
        total += len(q)
        log(f"anchor: {path.name} vs {ZIP_A.name}: {len(q):,} queue rows, {d} differ")
        assert d == 0, f"{path} differs from the queue rows of {ZIP_A}"
    assert total == len(sub), (total, len(sub))
    log(f"anchor: the two files cover all {len(sub):,} queue rows of A")


def main() -> None:
    global LOG, VIEW
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=str(WORK / "results" / "t2_asof"))
    ap.add_argument("--windows", nargs="+")
    ap.add_argument("--natural", action="store_true", help="also the natural-batch pass (float-noise information)")
    ap.add_argument("--no-control", action="store_true")
    ap.add_argument("--no-zip", action="store_true", help="skip the check of the two files against A's zip")
    a = ap.parse_args()
    out = Path(a.out_dir)
    files = [out / "asof.log", out / "asof_windows.csv", out / "asof_files_read.txt"]
    for f in files:
        if f.exists():
            raise FileExistsError(f"refusing to overwrite {f}")
    out.mkdir(parents=True, exist_ok=True)
    LOG = files[0].open("w", encoding="utf-8")
    t_all = time.time()
    log(f"t2_final_asof  {time.strftime('%Y-%m-%d %H:%M:%S')}  natural={a.natural} control={not a.no_control}")

    # ---- setup (guard not armed): anchors, full sources, train-only inputs, freshness of the train caches
    if not a.no_zip:
        t0 = time.time()
        anchor_zip()
        log(f"anchor {time.time() - t0:.0f}s")
    exp = {s: expected(s) for s in SPLITS}
    for p in T2_PANELS:
        src = CLEAN / p / "train.parquet"
        gp = t2_grids.GRIDS / f"{p}_train.npz"
        ev = CACHE / f"t2_onset_events_{p}.parquet"
        assert gp.exists() and gp.stat().st_mtime >= src.stat().st_mtime, f"stale {gp}"
        assert ev.exists() and ev.stat().st_mtime >= src.stat().st_mtime, f"stale {ev}"
    assert (V30.CACHE / "D7_I405_N_train.npz").exists()
    t0 = time.time()
    VIEW = View()
    for p in T2_PANELS:
        for s in SPLITS:
            wi, tt = VIEW.wi[(p, s)]
            go = (wi.condition == "queue_ongoing").to_numpy()
            # production batch = the ongoing rows in file order; they are in origin order, so a window's position
            # is the number of earlier ongoing origins, known at T
            assert (np.diff(tt[go].astype(np.int64)) > 0).all(), (p, s, "ongoing windows not in origin order")
            assert int(go.sum()) == BATCH, (p, s)
    log(f"full sources read, grids rebuilt from clean_data == cache/t2_grids  {time.time() - t0:.0f}s")
    t0 = time.time()
    fit = {p: v54_fit_rows(p) for p in V54.PANELS}
    for p in V54.PANELS:
        with open(V54.CACHE / f"cases_{p}.pkl", "rb") as f:
            cached = pickle.load(f)
        assert same_rows(fit[p], cached["fit"], ("T", "day", "t", "X7", "F", "i30")), (p, "V54 fit rows differ")
        fit[p] = dict(panel=p, pairs=cached["pairs"], fit=fit[p])
        assert fit[p]["pairs"] == [(x, y) for x in range(len(V54.canon_sets(p))) for y in range(x + 1,
                                                                                               len(V54.canon_sets(p)))]
    rf5 = V30.Joint("rf5").fit(V30.rows("D7_I405_N", "train"))
    ctx = dict(dec=json.loads((V54.OUT / "decisions.json").read_text(encoding="utf-8")), rf5=rf5)
    ref_cases = {}
    for p in V54.PANELS:
        with open(V54.CACHE / f"cases_{p}.pkl", "rb") as f:
            ref_cases[p] = {o["wid"]: o for o in pickle.load(f)["official"]}
    ref_classes = pd.read_csv(S52.OUT / "official_classes.csv").set_index("window_id")
    log(f"train-only inputs: V54 fit rows recomputed == cases_*.pkl, rf5 fitted  {time.time() - t0:.0f}s")

    # ---- providers and guard
    t2_grids.load = C.load_grids = PS.load = V54.load = D12.load = grid_provider
    S52._day_rows = VIEW.day_rows
    V54.build_cases = lambda panel: fit[panel]
    for mod, name in ((V54, "official_ml2"), (V54, "all_official_ml2"), (S52, "official_probs"),
                      (V27, "official_cache"), (task2_ml, "cached_inputs")):
        setattr(mod, name, forbidden(f"{mod.__name__}.{name}"))
    C.load_model = cached_loader(C.load_model, lambda tag, width=96: MODELS / f"t2_cnn{tag}.pt")
    R.load_model = cached_loader(R.load_model, lambda tag="": MODELS / f"t2_rnn{tag}.pt")
    PC.load_cpu = cached_loader(PC.load_cpu, lambda kind, tag: MODELS / f"t2_{kind}{tag}.pt")
    install_guard(VIEW)

    todo = []
    for split in SPLITS:
        for panel in T2_PANELS:
            wi, tt = VIEW.wi[(panel, split)]
            for r, t in zip(wi.itertuples(index=False), tt):
                if a.windows is None or r.window_id in a.windows:
                    todo.append((split, panel, r, t))
    if a.windows:
        missing = set(a.windows) - {r.window_id for _, _, r, _ in todo}
        if missing:
            raise SystemExit(f"not official windows: {sorted(missing)}")
    log(f"{len(todo)} windows")
    rows = []
    for split, panel, r, Tt in todo:
        t0 = time.time()
        wid, cond = r.window_id, ("onset" if r.condition == "queue_onset" else "ongoing")
        T = int(slot_of(pd.Series([r.forecast_origin]))[0]) % STEPS
        E = exp[split][wid]
        passes = [("trunc", T + 1, "shape")]
        if not a.no_control:
            passes.append(("control", T, "shape"))
            if cond == "onset":
                passes.append(("control_T5", T - 1, "shape"))
        if a.natural and cond == "ongoing":
            passes.append(("natural", T + 1, "natural"))
        rec = dict(window_id=wid, split=split, panel=panel, cond=cond, date=str(r.date), origin=r.forecast_origin,
                   cells=int(E.size), queued=int(E.sum()))
        for name, cut, mode in passes:
            stats = VIEW.set(split, str(r.date), T, Tt, cut, panel)
            for fn in (PS.windows, R.panel_data, t2_ens.panel_data):
                fn.cache_clear()
            GUARD.armed, GUARD.march = True, split == "validation"
            try:
                if cond == "onset":
                    N, rule, diag = onset_rows(panel, split, wid, ctx)
                else:
                    N, rule, diag = ongoing_rows(panel, split, wid, mode)
            finally:
                GUARD.armed = GUARD.march = False
            rec[f"{name}_diff"] = int((N != E).sum())
            if name == "trunc":
                rec.update(rule=rule, **stats)
                if cond == "onset":
                    rec["pred_links"] = " ".join(map(str, np.flatnonzero(N[HOR - 1])))
                    if "v54_row" in diag:            # regenerated case row vs the production cases cache
                        o = ref_cases[panel][wid]
                        rec["v54_row_eq_cache"] = all(np.array_equal(diag["v54_row"][k], o[k], equal_nan=True)
                                                      for k in ("X7", "F", "i30", "T", "gap", "k"))
                    if "ml2" in diag:
                        with open(WORK / "sub" / "probs" / f"t2ml2_probs_{split}_{panel}.pkl", "rb") as f:
                            pk = {x["window_id"]: x["prob"] for x in pickle.load(f)}
                        rec["ml2_eq_cache"] = bool(np.array_equal(diag["ml2"], pk[wid][HOR - 1]))
                else:
                    rec["rows_identical_in_batch"] = diag["rows_identical"]
                    c = ref_classes.loc[wid]
                    rec["cls_eq_cache"] = bool(c.cls == diag["cls"]["cls"] and bool(c.spacing) == diag["cls"]["spacing"]
                                               and abs(float(c.cov_T5) - diag["cls"]["cov_T5"]) < 1e-9)
                    zc = dict(np.load(V27.CACHE / f"official_{panel}_{split}.npz"))
                    j = [str(x) for x in zc["ids"]].index(wid)
                    rec["v27_eq_cache"] = all(np.array_equal(diag["z"][k][0], zc[k][j]) for k, _, _ in V27_NETS)
                    sc = np.load(WORK / "results" / "t2_v46_slotT" / f"{panel}_{split}.npz")
                    j = [str(x) for x in sc["ids"]].index(wid)
                    rec["v46_eq_cache"] = all(np.array_equal(diag["sh"][k][0], sc[f"sh_{k}"][j]) for k in S.KEYS)
                    if diag["cpu"] is not None:
                        cc = np.load(WORK / "cache" / "t2_v52_selrule" / f"official_{panel}_{split}.npz")
                        j = [str(x) for x in cc["ids"]].index(wid)
                        rec["cpu_eq_cache"] = all(np.array_equal(diag["cpu"][k][0], cc[k][j])
                                                  for k in diag["cpu"] if not k.startswith(("rnnft", "cnnft")))
        rec["seconds"] = round(time.time() - t0, 1)
        rows.append(rec)
        log(f"{wid:30s} {cond:7s} {rec['rule']:26s} trunc_diff={rec['trunc_diff']}"
            + "".join(f" {k}={rec[k]}" for k in ("control_diff", "control_T5_diff", "natural_diff") if k in rec)
            + "".join(f" {k}={rec[k]}" for k in ("v54_row_eq_cache", "ml2_eq_cache", "v27_eq_cache", "v46_eq_cache",
                                                 "cpu_eq_cache", "cls_eq_cache") if k in rec)
            + f" days_del={rec['days_deleted']} origins_del={rec['origins_deleted']}"
              f" hist_del={rec['history_rows_deleted']} {rec['seconds']}s")

    df = pd.DataFrame(rows)
    df.to_csv(files[1], index=False)
    files[2].write_text("\n".join(sorted(GUARD.read)) + "\n", encoding="utf-8")
    log("")
    pd.set_option("display.width", 250)
    agg = {"windows": ("window_id", "size"), "cells": ("cells", "sum"), "trunc_diff": ("trunc_diff", "sum")}
    for k in ("control_diff", "control_T5_diff", "natural_diff"):
        if k in df:
            agg[k] = (k, "sum")
            agg[k.replace("_diff", "_windows_moved")] = (k, lambda x: int((x > 0).sum()))
    log(df.groupby(["split", "cond"]).agg(**agg).to_string())
    eq_cols = [k for k in ("v54_row_eq_cache", "ml2_eq_cache", "v27_eq_cache", "v46_eq_cache", "cpu_eq_cache",
                           "cls_eq_cache", "rows_identical_in_batch") if k in df]
    for k in eq_cols:
        v = df[k].dropna()
        log(f"{k}: {int(v.astype(bool).sum())}/{len(v)} windows")
    ok = bool((df.trunc_diff == 0).all())
    ctrl = int(df.get("control_diff", pd.Series([0])).sum()) + int(df.get("control_T5_diff", pd.Series([0])).sum())
    log(f"\nTOTAL cells {int(df.cells.sum()):,} in {len(df)} windows; trunc pass differing cells: "
        f"{int(df.trunc_diff.sum())}")
    if not a.no_control:
        log(f"controls (row T / rows >= T-5 deleted): {ctrl} cells moved -> the check is "
            f"{'not vacuous' if ctrl > 0 else 'VACUOUS'}")
        ok &= ctrl > 0
    log(f"files read while regenerating: {len(GUARD.read)} (list in {files[2].name})")
    log(f"runtime {time.time() - t_all:.0f}s")
    log("PASS" if ok else "FAIL")
    LOG.close()
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    np.seterr(invalid="ignore", divide="ignore")
    main()
