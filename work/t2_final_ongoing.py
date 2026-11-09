"""Final package: every official ongoing window (March + April, 8 corridors) rebuilt in one pass from the member
models, cell-checked against the submitted queue files.

  March  V48 blend (ml2g0 2, cnng0 1, rnng0 5, rnn2 2, CatBoost 0.25; group smoothing; V33 adaptive IoU)
  April  V72 blend (March fine-tunes rnnft96 4 [96k windows], rnnftx48 2 [48k], cnnftx96 4 [96k]; same smoothing /
         decision)
  both   then the V52 'hyb' selector conditioning of the constrained windows (t2_v52_selrule.decide, classes from
         t2_v52_selrule classify), h1 kept
Member probabilities are recomputed from the model files (t2_slotT.g0_probs, ml2_g0_probs, cat_g0_probs; the V46
shifted hybrid members from results/t2_v46_slotT).
ongoing_preds(panel, "private", variant="v51") gives the pre-V72 April rows (V51 blend + hyb) that
t2_final_queue.py writes into queue_private_v54_onset.csv / v52_selrule.csv; its pre= argument returns the decisions
before the V52 conditioning (March V48 / April V51 rows of queue_*_v51_april.csv).

  python work/t2_final_ongoing.py check      # no file written
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(key, "4")

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import repro_check as RC  # noqa: E402
import t2_cnn as C  # noqa: E402
import t2_ongoing_v27 as V27  # noqa: E402
import t2_slotT as S  # noqa: E402
import t2_v50_april as A  # noqa: E402
import t2_v52_selrule as S52  # noqa: E402
import t2_v71_april as V71  # noqa: E402
from evaluate import ROOT, STEPS, T2_PANELS, links, slot_of  # noqa: E402
from t2_cat_g0 import cat_g0_probs  # noqa: E402
from t2_group_v27 import groups, smooth  # noqa: E402
from t2_v33_adaptive_iou import adaptive_iou  # noqa: E402
from t2_v46_slotT import official_windows  # noqa: E402

WORK = ROOT / "work"
FINAL_Q = {"validation": WORK / "sub" / "queue_validation_v54_onset.csv",
           "private": WORK / "sub" / "queue_private_v72_april.csv"}
WEIGHTS = {"validation": V71.W51["v48"], "private": V71.W72}


def chain_p(pool, panel, w):
    names = [n for n in V71.ORDER if w.get(n)]
    pr = sum(w[n] * pool[n] for n in names) / sum(w[n] for n in names)
    g = groups(panel, .95)
    pr = smooth(pr, g, .25)
    if w["cat"]:
        pr = (1 - w["cat"]) * pr + w["cat"] * smooth(pool["catg0"], g, .25)
    return pr


V51_SPECS = [sp for sp in A.FT_SPECS if V71.W51["weights"].get(sp[0].rsplit("_", 1)[0])]    # weighted V51 members


def ongoing_preds(panel: str, split: str, variant: str | None = None,
                  pre: dict | None = None) -> dict[str, np.ndarray]:
    """window_id -> final (6, n) binary prediction (pos order).

    variant="v51" (April only): the V51 blend instead of V72, asserted against results/t2_v51 and conditioned on the
    CPU V51 probabilities like March -- the ongoing rows of queue_private_v52_selrule.csv / v54_onset.csv.
    pre: if a dict is given, it receives window_id -> the blend decision BEFORE the V52 'hyb' conditioning (March V48,
    April V51 with variant="v51": the ongoing rows of queue_{validation,private}_v51_april.csv)."""
    v51 = variant == "v51"
    assert variant in (None, "v51") and (split == "private" or not v51), (split, variant)
    w = V71.W51["weights"] if v51 else WEIGHTS[split]
    z = V27.official_cache(panel, split)
    D = C.PanelData(panel, split, unmasked=False)
    ws = official_windows(panel, split, D)
    assert [x.T for x in ws] == list(z["T"] % STEPS)
    v46 = np.load(WORK / "results" / "t2_v46_slotT" / f"{panel}_{split}.npz")
    sh = {k[3:]: v46[k] for k in v46.files if k.startswith("sh_")}
    specs = A.V48_SPECS + (V51_SPECS if v51 else V71.V72_SPECS if split == "private" else [])
    g0 = S.g0_probs(D, ws, specs)
    g0["ml2g0"] = S.ml2_g0_probs(ws)
    g0["catg0"] = cat_g0_probs(ws)
    hy, pool = A.pool_of(z, sh, g0)
    for fam in ("rnnft48", "rnnftx48", "cnnftx48", "rnnft96", "cnnftx96"):
        ks = [k for k in sorted(g0) if k.startswith(fam + "_")]
        if ks:
            pool[fam] = np.mean([g0[k] for k in ks], 0)
    pool = {k: v for k, v in pool.items() if v is not None}
    P = V71.decide(hy, pool, panel, w)
    p = chain_p(pool, panel, w)
    assert np.array_equal(np.stack([adaptive_iou(x) for x in p]), P), (panel, split)
    if v51:
        rec = np.load(WORK / "results" / "t2_v51" / f"{panel}_private.npz")["new"]
        assert rec.shape == P.shape, (panel, rec.shape, P.shape)
        RC.soft(np.array_equal(P, rec), f"{panel} April V51 ongoing decision differs from the recorded "
                f"results/t2_v51 in {int((P != rec).sum())} cells (with TFB_TOLERANT=1 the recomputed decision is used)")
    cls = pd.read_csv(S52.OUT / "official_classes.csv").set_index("window_id")
    # March: V52 was built with the CPU re-implementation of the V48 chain (t2_v52_prodcache.py official); its
    # conditioning flips are applied relative to that P0, exactly as t2_v52_selrule.build did.  April (V71/V72) used the
    # probabilities computed here.  April V51 (variant "v51") was conditioned by t2_v52_selrule.build like March, on the
    # CPU V51 chain.
    cpu = S52.official_probs(panel, split) if (split == "validation" or v51) else {}
    out = {}
    for i, wid in enumerate(z["ids"]):
        wid = str(wid)
        if pre is not None:
            pre[wid] = P[i].copy()
        c = cls.loc[wid].to_dict()
        c["panel"], c["window_id"] = panel, wid
        pc = cpu.get(wid, p[i])
        out[wid], _, _ = S52.decide(S52.RULE, P[i], pc, adaptive_iou(pc), c)
    return out


def cmd_check() -> None:
    bad = 0
    for split, path in FINAL_Q.items():
        q = pd.read_csv(path, dtype={"link_id": str})
        q["slot"] = slot_of(q.timestamp)
        for panel in T2_PANELS:
            preds = ongoing_preds(panel, split)
            win = pd.read_csv(V27.RELEASE / "task2" / panel / split / "window_index.csv") if hasattr(V27, "RELEASE") \
                else pd.read_csv(ROOT / "kaggle_public" / "task2" / panel / split / "window_index.csv")
            T0 = {r.window_id: int(slot_of(pd.Series([r.forecast_origin]))[0]) for r in win.itertuples(index=False)}
            pos = dict(zip(links(panel).link_id.astype(str), range(len(links(panel)))))
            nb = 0
            for wid, N in preds.items():
                r = q[q.window_id == wid]
                hor = (r.slot.to_numpy() - T0[wid] - 1).astype(int)
                exp = N[hor, r.link_id.map(pos).to_numpy()].astype(int)
                d = int((exp != r.queue_pred.to_numpy()).sum())
                nb += d
                if d:
                    print(f"  MISMATCH {split} {wid}: {d} cells", flush=True)
            bad += nb
            print(f"{split:10s} {panel:10s} windows {len(preds)} mismatching cells {nb}", flush=True)
    print("TOTAL mismatching cells:", bad)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "check":
        cmd_check()
