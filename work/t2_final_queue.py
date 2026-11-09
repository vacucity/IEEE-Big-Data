"""Final package: writes the five Task 2 queue files the rest of the chain reads, rebuilt in one pass from the member
models (t2_final_onset.onset_sets, t2_final_ongoing.ongoing_preds) instead of the historical patch chain
(v14 -> v18 -> v24 -> v30 onset, V48 / V51 ongoing, V52 conditioning, V54 onset), and compares them with the originals.

  queue_validation_v54_onset.csv   March: V54 onset rules + V48 ongoing blend + V52 'hyb' conditioning
  queue_private_v54_onset.csv      April: V54 onset rules + V51 ongoing blend + V52 'hyb' conditioning
  queue_private_v52_selrule.csv    April before V54: the same ongoing rows, onset rows of the production choosers
                                   (V54.PROD) on the four V54 corridors (2 D7_I210_W cells differ)
  queue_validation_v51_april.csv   March before V52: V48 ongoing blend WITHOUT the V52 conditioning (= the V48 rows,
                                   t2_v50_april.py copied queue_validation_v48_g0.csv), pre-V54 onset rows
  queue_private_v51_april.csv      April before V52: V51 ongoing blend without the conditioning, pre-V54 onset rows
                                   (the onset rows of both v51 files are those of the V30 chain = V54.PROD, as in
                                   v52_selrule; V52_d12s changed 2 D12_I5_S March windows, V52_selrule restored them)
t2_v71_april.py --variant v72 reads the two April v52/v54 files and writes the final April rows
(queue_private_v72_april.csv); the build_queue_candidate.py steps of the T1 stages read the two v54_onset files;
make_submission.py (run_all stage v51_zip) reads the two v51_april files.

Row layout (as the originals): header window_id,timestamp,link_id,queue_pred, CRLF line ends; all onset windows of
all corridors first, then all ongoing windows; corridors in evaluate.T2_PANELS order; inside a corridor the rows of
kaggle_public/task2/<panel>/<split>/sample_submission_queue.csv of that condition, in file order.

  python work/t2_final_queue.py write [--out-dir work/sub]     # GPU; refuses to overwrite any of the five files
  python work/t2_final_queue.py compare --out-dir DIR [--ref-dir work/sub]   # cells and bytes vs the originals

Two inputs had no inference-only producer; these subcommands rebuild them from the shipped models (existing files are
kept, never overwritten, as t2_v52_prodcache.py does; any --out-dir for a test run):
  python work/t2_final_queue.py ml2-probs [--out-dir work/sub/probs]
      t2ml2_probs_<split>_<panel>.pkl: ml2 probabilities of every official window (boosters t2_ml2_{onset,ongoing}
      _probs.txt, task2_ml2.window_probs2); task2_ml2.py predict --tag _probs wrote them while refitting the boosters
  python work/t2_final_queue.py slotT [--out-dir work/results/t2_v46_slotT]
      <panel>_<split>.npz sh_* members: the V27 members on the shifted history [T-55, T] (GPU), the same calls as
      t2_v46_slotT.py, which also needs the V45 queue files as its base and writes queue_*_v46_slotT.csv.  Only the
      ids / T / sh_* keys are rebuilt: the original npz also held 'old' / 'new' (the V45 / V46 decisions), read only
      by the off-chain t2_v47_g0.py --check-dir, never by this package

Caches `write` reads, in the order a clean copy builds them (run_all.py stage t2_queue): t2_grids.py;
t2_group_v27.py learn; ml2-probs; task2_ml.py build + predict --tag _probs (sub/probs/t2ml_lgbm_probs_*, only a
pass-through key of the V27 cache); slotT (first builds cache/t2_ongoing_v27/official_*.npz); t2_v52_prodcache.py
official; t2_v52_selrule.py classify (+ check); t2_v52_replica.py build; t2_v54_onset.py cases.  Frozen inputs:
results/t2_v54_onset/decisions.json, results/t2_v51/config.json and results/t2_v51/<panel>_private.npz (the V51
decision the April rebuild is asserted against).

Truncation check of the final rows (all 160 windows, every row after T deleted): work/t2_final_asof.py.
"""
from __future__ import annotations

import argparse
import filecmp
import hashlib
import os
import sys
import time
from pathlib import Path

for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(key, "4")

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import RELEASE, ROOT, T2_PANELS, links, slot_of  # noqa: E402

WORK = ROOT / "work"
SUB = WORK / "sub"
# name -> (split, onset rows, ongoing rows): onset "v54" = onset_sets, "pre_v54" = onset_sets_pre_v54; ongoing "hyb" =
# ongoing_preds (V52-conditioned), "pre_hyb" = its decision before the conditioning
FILES = {"queue_validation_v54_onset.csv": ("validation", "v54", "hyb"),
         "queue_private_v54_onset.csv": ("private", "v54", "hyb"),
         "queue_private_v52_selrule.csv": ("private", "pre_v54", "hyb"),
         "queue_validation_v51_april.csv": ("validation", "pre_v54", "pre_hyb"),
         "queue_private_v51_april.csv": ("private", "pre_v54", "pre_hyb")}
COLS = ["window_id", "timestamp", "link_id"]
HOR = 6
EOL = "\r\n"            # the originals were written by pandas on Windows


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def template(split: str) -> pd.DataFrame:
    """Key rows of one split in the layout of the original files, with panel, condition and origin slot T."""
    parts = {"queue_onset": [], "queue_ongoing": []}
    for panel in T2_PANELS:
        base = RELEASE / "task2" / panel / split
        s = pd.read_csv(base / "sample_submission_queue.csv", dtype=str)[COLS]
        wi = pd.read_csv(base / "window_index.csv")
        s["panel"] = panel
        s["cond"] = s.window_id.map(dict(zip(wi.window_id, wi.condition)))
        s["T"] = s.window_id.map(dict(zip(wi.window_id, slot_of(wi.forecast_origin))))
        assert s.cond.isin(list(parts)).all() and s["T"].notna().all(), (panel, split)
        for c in parts:
            parts[c].append(s[s.cond == c])
    return pd.concat(parts["queue_onset"] + parts["queue_ongoing"], ignore_index=True)


def fill(t: pd.DataFrame, onset: dict, ongoing: dict) -> pd.DataFrame:
    """queue_pred of template t: onset[panel][wid] = (n,) link set at T+30 (0 elsewhere), ongoing[panel][wid] =
    (6, n) prediction for T+5..T+30 (pos order)."""
    pred = np.full(len(t), -1, np.int64)
    hor = slot_of(t.timestamp) - t["T"].to_numpy(np.int64) - 1
    assert ((hor >= 0) & (hor < HOR)).all()
    lpos = np.full(len(t), -1, np.int64)
    for panel, rows in t.groupby("panel", sort=False).indices.items():
        pos = dict(zip(links(panel).link_id.astype(str), range(len(links(panel)))))
        lpos[rows] = t.link_id.iloc[rows].map(pos).to_numpy()
    assert (lpos >= 0).all()
    for (panel, cond, wid), idx in t.groupby(["panel", "cond", "window_id"], sort=False).indices.items():
        src = onset[panel] if cond == "queue_onset" else ongoing[panel]
        P, h, ix = src[wid], hor[idx], lpos[idx]
        pred[idx] = ((h == HOR - 1) & P[ix]) if cond == "queue_onset" else P[h, ix]
    for panel in T2_PANELS:
        for cond, src in (("queue_onset", onset[panel]), ("queue_ongoing", ongoing[panel])):
            assert set(t.window_id[(t.panel == panel) & (t.cond == cond)]) == set(src), (panel, cond)
    assert (pred >= 0).all()
    return t[COLS].assign(queue_pred=pred)


def build() -> dict[str, pd.DataFrame]:
    """name -> rebuilt frame of the five files."""
    import t2_final_onset as F1
    import t2_final_ongoing as F2
    import t2_v54_onset as V54
    t0 = time.time()
    onset = {"v54": {}, "pre_v54": {}}
    for split in ("validation", "private"):
        for panel in T2_PANELS:
            onset["v54"][(split, panel)] = F1.onset_sets(panel, split)
            # onset_sets_pre_v54 == onset_sets off the four V54 corridors: reuse instead of refitting
            onset["pre_v54"][(split, panel)] = (F1.onset_sets_pre_v54(panel, split) if panel in V54.PANELS
                                                else onset["v54"][(split, panel)])
    print(f"onset sets {time.time() - t0:.0f}s", flush=True)
    ongoing = {"hyb": {}, "pre_hyb": {}}
    for split, variant in (("validation", None), ("private", "v51")):
        for panel in T2_PANELS:
            t1 = time.time()
            pre = {}
            ongoing["hyb"][(split, panel)] = F2.ongoing_preds(panel, split, variant=variant, pre=pre)
            ongoing["pre_hyb"][(split, panel)] = pre
            print(f"ongoing {split} {panel} {time.time() - t1:.0f}s", flush=True)
    out = {}
    for name, (split, on_kind, go_kind) in FILES.items():
        on = {p: onset[on_kind][(split, p)] for p in T2_PANELS}
        go = {p: ongoing[go_kind][(split, p)] for p in T2_PANELS}
        out[name] = fill(template(split), on, go)
    print(f"built in {time.time() - t0:.0f}s", flush=True)
    return out


def cmd_write(args) -> None:
    out_dir = Path(args.out_dir)
    for name in FILES:
        if (out_dir / name).exists():
            raise FileExistsError(f"refusing to overwrite {out_dir / name}")
    frames = build()
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in frames.items():
        path = out_dir / name
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")
        df.to_csv(path, index=False, lineterminator=EOL)
        print(f"wrote {path}  rows {len(df):,}  queued {int(df.queue_pred.sum()):,}  sha256 {sha256(path)}",
              flush=True)


def cmd_compare(args) -> None:
    import repro_check as RC
    out_dir, ref_dir = Path(args.out_dir), Path(args.ref_dir)
    bad = 0
    structural = 0                       # missing file or different row keys: fatal even with TFB_TOLERANT=1
    for name in FILES:
        a, b = out_dir / name, ref_dir / name
        if not a.exists():
            print(f"{name}: MISSING in {out_dir}")
            bad += 1
            structural += 1
            continue
        x = pd.read_csv(a, dtype=str)
        y = pd.read_csv(b, dtype=str)
        assert list(x.columns) == list(y.columns) == COLS + ["queue_pred"], name
        same_keys = len(x) == len(y) and x[COLS].equals(y[COLS])
        cells = int((x.queue_pred != y.queue_pred).sum()) if same_keys else -1
        same_bytes = filecmp.cmp(a, b, shallow=False)
        print(f"{name}: rows {len(x):,} vs {len(y):,}  keys identical {same_keys}  mismatching cells {cells}  "
              f"bytes identical {same_bytes}\n    {sha256(a)}  {a}\n    {sha256(b)}  {b}")
        if cells > 0:
            d = x[x.queue_pred != y.queue_pred]
            for wid, n in d.groupby("window_id").size().items():
                print(f"    MISMATCH {wid}: {n} cells")
        bad += int(not same_keys or cells != 0 or not same_bytes)
        structural += int(not same_keys)
    print(f"ALL {len(FILES)} FILES BYTE-IDENTICAL" if bad == 0 else f"{bad} file(s) differ")
    if bad and (structural or not RC.tolerant()):
        raise SystemExit(1)
    if bad:
        print("TOLERATED (TFB_TOLERANT=1): queue_pred values differ from the reference copies (cells listed above); "
              "row keys are identical", flush=True)


def cmd_ml2_probs(args) -> None:
    import pickle
    import lightgbm as lgb
    import task2_ml2
    out_dir = Path(args.out_dir)
    paths = {(s, p): out_dir / f"t2ml2_probs_{s}_{p}.pkl" for s in ("validation", "private") for p in T2_PANELS}
    fns = {}
    for kind in ("onset", "ongoing"):
        b = lgb.Booster(model_file=str(WORK / "models" / f"t2_ml2_{kind}_probs.txt"))
        fns[kind] = lambda X, b=b: b.predict(X[b.feature_name()])
    out_dir.mkdir(parents=True, exist_ok=True)
    for (split, panel), path in paths.items():
        if path.exists():
            print(f"exists {path} (kept)", flush=True)
            continue
        t0 = time.time()
        probs = task2_ml2.window_probs2(panel, split, fns, by_fold=False)
        with path.open("wb") as f:
            pickle.dump(probs, f)
        print(f"wrote {path}  windows {len(probs)}  {time.time() - t0:.0f}s", flush=True)


def cmd_slotT(args) -> None:
    import dataclasses
    import t2_cnn as C
    import t2_ongoing_v27 as V27
    from evaluate import STEPS
    from t2_v46_slotT import official_windows
    out_dir = Path(args.out_dir)
    paths = {(s, p): out_dir / f"{p}_{s}.npz" for s in ("validation", "private") for p in T2_PANELS}
    out_dir.mkdir(parents=True, exist_ok=True)
    for (split, panel), path in paths.items():
        if path.exists():
            print(f"exists {path} (kept)", flush=True)
            continue
        t0 = time.time()
        z = V27.official_cache(panel, split)
        D = C.PanelData(panel, split, unmasked=False)
        ws = official_windows(panel, split, D)
        assert [w.T for w in ws] == list(z["T"] % STEPS), (panel, split)
        shifted = [dataclasses.replace(w, T=w.T + 1) for w in ws]          # history [T-55, T]
        sh = V27.neural_probs(D, shifted)
        sh["v8"], sh["ml2"] = V27.gbm_probs(shifted)
        np.savez_compressed(path, ids=z["ids"], T=z["T"], **{f"sh_{k}": v for k, v in sh.items()})
        print(f"wrote {path}  windows {len(ws)}  {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("write")
    s.add_argument("--out-dir", default=str(SUB))
    s = sub.add_parser("compare")
    s.add_argument("--out-dir", required=True)
    s.add_argument("--ref-dir", default=str(SUB))
    s = sub.add_parser("ml2-probs")
    s.add_argument("--out-dir", default=str(SUB / "probs"))
    s = sub.add_parser("slotT")
    s.add_argument("--out-dir", default=str(WORK / "results" / "t2_v46_slotT"))
    a = ap.parse_args()
    {"write": cmd_write, "compare": cmd_compare, "ml2-probs": cmd_ml2_probs, "slotT": cmd_slotT}[a.cmd](a)
