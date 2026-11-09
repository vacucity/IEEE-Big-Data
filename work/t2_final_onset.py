"""Final package: every official onset window (March + April, 8 corridors) rebuilt in one pass, cell-checked
against the submitted queue files.

The onset rows of the final submission came out of five patches (v14 ml2 -> v18 t2_onset.py -> v24
t2_onset_cap.py -> v30 t2_v30_onset.py -> v54 t2_v54_onset.py).  The rule each corridor ends up with:
  D12_I5_S, D12_I5_N, D7_I405_S, D7_I210_W   t2_v54_onset decisions.json (cap t15+J, cap t15+JT, flow logit,
                                             ml2 block-argmax), via t2_v54_onset.choose
  D7_I10_W                                   flow logit over the train blocks (t2_onset.official_onset)
  D7_I405_N                                  set-level random forest rf5 (t2_v30_onset, seed 3030)
  D7_I10_E, D7_I210_E                        ml2 onset probability at T+30 >= 0.5, else the argmax link
                                             (task2_ml.threshold, theta_onset 0.5)
Every model is fitted on train only and reads only the published history (+ the origin row for V54's JT).
Onset predictions sit at T+30 only (organizer-confirmed onset step); T+5..T+25 are 0.

  python work/t2_final_onset.py check      # rebuild and compare with the final queue files (no file written)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import t2_onset as O  # noqa: E402
import t2_v54_onset as V54  # noqa: E402
from evaluate import RELEASE, ROOT, T2_PANELS, links, slot_of  # noqa: E402

WORK = ROOT / "work"
HOR = 6
FINAL_Q = {"validation": WORK / "sub" / "queue_validation_v54_onset.csv",
           "private": WORK / "sub" / "queue_private_v72_april.csv"}
ML2_TH = 0.5


def onset_sets(panel: str, split: str) -> dict[str, np.ndarray]:
    """window_id -> boolean (n,) link set predicted at T+30."""
    n = len(links(panel))
    if panel in V54.PANELS:
        dec = json.loads((V54.OUT / "decisions.json").read_text(encoding="utf-8"))[panel]
        rule = dec["rule"] if (split == "validation" or dec["apply_april"]) else V54.PROD[panel]
        res = V54.build_cases(panel)
        ch = V54.choose(panel, rule, res["official"], V54.all_official_ml2(panel))
        return {o["wid"]: s for o, (s, _) in zip(res["official"], ch) if o["split"] == split}
    if panel == "D7_I10_W":
        out = {}
        for wid, (_, lk) in O.official_onset(panel, split).items():
            s = np.zeros(n, bool)
            s[lk] = True
            out[wid] = s
        return out
    if panel == "D7_I405_N":
        import t2_v30_onset as V30
        model = V30.Joint("rf5").fit(V30.rows(panel, "train"))
        recs = V30.official(panel, split)
        pred = model.predict({"F": np.stack([r[2] for r in recs])})
        return {wid: np.asarray(p, bool) for (wid, _, _), p in zip(recs, pred)}
    out = {}
    for wid, p in V54.official_ml2(panel, split).items():
        s = p >= ML2_TH
        if not s.any():
            s[int(np.argmax(p))] = True
        out[wid] = s
    return out


def onset_sets_pre_v54(panel: str, split: str) -> dict[str, np.ndarray]:
    """onset_sets before V54: the production chooser V54.PROD on the four V54 corridors (the onset rows of
    queue_*_v52_selrule.csv, which t2_v54_onset.py build patched); every other corridor as onset_sets."""
    if panel not in V54.PANELS:
        return onset_sets(panel, split)
    res = V54.build_cases(panel)
    ch = V54.choose(panel, V54.PROD[panel], res["official"], V54.all_official_ml2(panel))
    return {o["wid"]: s for o, (s, _) in zip(res["official"], ch) if o["split"] == split}


def cmd_check() -> None:
    bad = 0
    for split, path in FINAL_Q.items():
        q = pd.read_csv(path, dtype={"link_id": str})
        q["slot"] = slot_of(q.timestamp)
        for panel in T2_PANELS:
            wi = pd.read_csv(RELEASE / "task2" / panel / split / "window_index.csv")
            wi = wi[wi.condition == "queue_onset"]
            sets = onset_sets(panel, split)
            assert set(sets) == set(wi.window_id), (panel, split)
            pos = dict(zip(links(panel).link_id.astype(str), range(len(links(panel)))))
            for w in wi.itertuples(index=False):
                T = int(slot_of(pd.Series([w.forecast_origin]))[0])
                r = q[q.window_id == w.window_id]
                exp = ((r.slot.to_numpy() == T + HOR) & sets[w.window_id][r.link_id.map(pos).to_numpy()]).astype(int)
                d = int((exp != r.queue_pred.to_numpy()).sum())
                bad += d
                print(f"{split:10s} {w.window_id:32s} links {' '.join(map(str, np.flatnonzero(sets[w.window_id])))}"
                      f"{'' if d == 0 else f'   MISMATCH {d} cells'}", flush=True)
    print("TOTAL mismatching cells:", bad)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "check":
        cmd_check()
