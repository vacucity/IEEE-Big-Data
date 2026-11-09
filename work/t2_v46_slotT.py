#!/usr/bin/env python
"""V46: V45 with Task 2 ongoing forecasts that also read the observation row at the forecast origin T.

Organizer, discussion 745700 (2026-10-15): "timestamps at or before T, including the observations at T ... the
window history covers T-60 to T-5, and the row at T itself is released data and may be used."  The row at T is
in mainline_states_masked (Task 1 mask applies; the Task 2 blackout starts at T+5).

Zero retraining: every existing ongoing member (v8 GBM, ml2 GBM, CNN x3, BiLSTM x2) is run on the history
shifted by one row, [T-55, T]; their steps 1..5 give T+10..T+30, and T+5 keeps the unshifted forecast
(t2_slotT.hybrid).  The rest of the V45 chain is unchanged: [1,1,1,5] blend, train-only co-queue group
smoothing (.95/.25), CatBoost blends (D7_I10_E .3, D7_I210_W .1, D7_I405_S .2; CatBoost stays on the
unshifted history), window-wise expected-IoU decision.  Onset, state and ODME rows are V45's.

Pseudo windows (visible validation month), V45 chain: 0.8823 -> 0.8958, all eight corridors up.

Usage (from the "IEEE Big data" folder):
  python work/t2_v46_slotT.py
"""
from __future__ import annotations

import dataclasses
import json
import os
import sys
from pathlib import Path

for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(key, "4")

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import t2_cnn as C  # noqa: E402
import t2_ongoing_v27 as V27  # noqa: E402
from evaluate import RELEASE, ROOT, STEPS, T2_PANELS, links, slot_of  # noqa: E402
from t2_group_v27 import groups, smooth  # noqa: E402
from t2_pseudo import Win  # noqa: E402
from t2_slotT import CAT_ALPHA, hybrid  # noqa: E402
from t2_v33_adaptive_iou import adaptive_iou  # noqa: E402
from t2_v41_ongoing_cat_official import cat_official  # noqa: E402

WORK = ROOT / "work"
OUT = WORK / "results" / "t2_v46_slotT"
BASE = "v45_405s_ongoing"
NAME = "v46_slotT"


def official_windows(panel, split, D):
    idx = pd.read_csv(RELEASE / "task2" / panel / split / "window_index.csv")
    ws = []
    for r in idx[idx.condition == "queue_ongoing"].itertuples(index=False):
        T = int(slot_of(pd.Series([r.forecast_origin]))[0]) % STEPS
        d = D.dates.index(str(r.date))
        assert np.isfinite(D.V[d, T].astype(np.float32)).any(), (r.window_id, "no visible cell at T")
        ws.append(Win(panel, split, d, T, "ongoing", str(r.date), int(D.weekday[d])))
    return ws


def chain(z, panel, cat):
    g = groups(panel, .95)
    pr = smooth(V27.probs(z, [1, 1, 1, 5]), g, .25)
    a = CAT_ALPHA.get(panel, 0)
    if a:
        pr = (1 - a) * pr + a * smooth(cat, g, .25)
    return np.stack([adaptive_iou(x) for x in pr])


def main():
    from catboost import CatBoostClassifier
    OUT.mkdir(parents=True, exist_ok=True)
    model = CatBoostClassifier()
    model.load_model(str(WORK / "models" / "t2_v41_ongoing_cat.cbm"))
    records = []
    for split in ("validation", "private"):
        dst = WORK / "sub" / f"queue_{split}_{NAME}.csv"
        if dst.exists():
            raise FileExistsError(dst)
        q = pd.read_csv(WORK / "sub" / f"queue_{split}_{BASE}.csv", dtype={"link_id": str})
        q["slot"] = slot_of(q.timestamp)
        for panel in T2_PANELS:
            z = V27.official_cache(panel, split)
            D = C.PanelData(panel, split, unmasked=False)
            ws = official_windows(panel, split, D)
            assert [w.T for w in ws] == list(z["T"] % STEPS)
            # same inputs as the official cache (grids carry the window histories)
            gb0 = V27.gbm_probs(ws)
            assert np.abs(gb0[0] - z["v8"]).max() < 1e-3 and np.abs(gb0[1] - z["ml2"]).max() < 1e-3, panel
            shifted = [dataclasses.replace(w, T=w.T + 1) for w in ws]
            sh = V27.neural_probs(D, shifted)
            sh["v8"], sh["ml2"] = V27.gbm_probs(shifted)
            hy = hybrid(z, sh)
            cat = cat_official(panel, split, model, model.feature_names_, z) if CAT_ALPHA.get(panel) else None
            old_pred, new_pred = chain(z, panel, cat), chain(hy, panel, cat)
            np.savez_compressed(OUT / f"{panel}_{split}.npz", ids=z["ids"], T=z["T"],
                                **{f"sh_{k}": v for k, v in sh.items()}, old=old_pred, new=new_pred)
            pos = dict(zip(links(panel).link_id.astype(str), range(len(links(panel)))))
            for i, wid in enumerate(z["ids"]):
                mask = q.window_id.eq(wid).to_numpy()
                row = q.loc[mask]
                hor = (row.slot.to_numpy() - z["T"][i] - 1).astype(int)
                assert len(row) and ((hor >= 0) & (hor < 6)).all()
                ix = row.link_id.map(pos).to_numpy().astype(int)
                old = row.queue_pred.to_numpy(int)
                assert np.array_equal(old_pred[i, hor, ix].astype(int), old), (split, wid, "V45 not reproduced")
                new = new_pred[i, hor, ix].astype(int)
                records.append(dict(split=split, panel=panel, window_id=wid, changed=int((old != new).sum()),
                                    added=int(((old == 0) & (new == 1)).sum()),
                                    removed=int(((old == 1) & (new == 0)).sum()), n_old=int(old.sum()),
                                    n_new=int(new.sum())))
                q.loc[mask, "queue_pred"] = new
            print(split, panel, "windows", len(ws), "changed cells",
                  sum(r["changed"] for r in records if r["split"] == split and r["panel"] == panel), flush=True)
        q.drop(columns="slot").to_csv(dst, index=False)
    ch = pd.DataFrame(records)
    ch.to_csv(OUT / "official_changes.csv", index=False)
    print(ch.groupby("split")[["changed", "added", "removed", "n_old", "n_new"]].sum())
    (OUT / "manifest.json").write_text(json.dumps({
        "base": BASE, "name": NAME, "rule": "discussion 745700: row at T may be used",
        "members": "v8, ml2, CNN x3, BiLSTM x2 on history [T-55, T] for T+10..T+30; T+5 unshifted",
        "chain": "[1,1,1,5], groups .95/.25, CatBoost " + json.dumps(CAT_ALPHA) + ", adaptive IoU",
    }, indent=2))


if __name__ == "__main__":
    main()
