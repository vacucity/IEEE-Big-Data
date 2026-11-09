#!/usr/bin/env python
"""V50: April (private) ongoing windows from members fine-tuned on March; March (validation) rows = the base.

Organizer ruling (748789/749601): an April window may use train + the whole validation month to fit and to
select; the competition contract says private is "a different month of the same scenario" as validation.
The March windows themselves keep the base build (a model tuned on March may not score March's earlier days).

Members on the April official windows: the V48 pool (hybrid rnn2, ml2g0, catg0, cnng0 x3, rnng0 x2) plus
rnnft (gap-0 BiLSTM seeds 0/1 fine-tuned on all of March), cnnft (gap-0 CNN seed 0 fine-tuned on March) and
ml2ft (gap-0 ml2 LightGBM refitted on train + March rows, t2_ml2_march.py),
blended with the weights t2_grid_april.py picked on cross-fitted March pseudo windows.

  python work/t2_v50_april.py --config work/results/t2_v50/config.json --base v48_g0 --name v50_april
  python work/t2_v50_april.py --config work/results/t2_v51/config.json --base v48_g0 --name v51_april --out-dir t2_v51
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(key, "4")

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import t2_cnn as C  # noqa: E402
import t2_ongoing_v27 as V27  # noqa: E402
import t2_slotT as S  # noqa: E402
from evaluate import ROOT, STEPS, T2_PANELS, links, slot_of  # noqa: E402
from t2_cat_g0 import cat_g0_probs  # noqa: E402
from t2_g0_eval import decide as decide_chain  # noqa: E402
from t2_v46_slotT import official_windows  # noqa: E402

WORK = ROOT / "work"
V48_SPECS = [("cnng0_0", "cnn", "_g0"), ("cnng0_1", "cnn", "_g0_s1"), ("cnng0_2", "cnn", "_g0_s2"),
             ("rnng0_0", "rnn", "_g0"), ("rnng0_1", "rnn", "_g0_s1")]
FT_SPECS = [("rnnft_0", "rnn", "_g0_ftall"), ("rnnft_1", "rnn", "_g0_s1_ftall"), ("cnnft_0", "cnn", "_g0_ftall"),
            # V51 families (t2_grid_april.py FAMILIES rnnftx/cnnftx); loaded only when weighted
            ("rnnftx_2", "rnn", "_g0_s2_ftall"), ("rnnftx_3", "rnn", "_g0_s3_ftall"),
            ("cnnftx_1", "cnn", "_g0_s1_ftall"), ("cnnftx_2", "cnn", "_g0_s2_ftall")]
NAMES = ("ml2g0", "ml2ft", "cnng0", "rnng0", "rnn2", "rnnft", "cnnft", "rnnftx", "cnnftx")
ML2FT_MODEL = WORK / "models" / "t2_ml2_ongoing_g0_mall.txt"      # t2_ml2_march.py fit --fold all


def pool_of(z, sh, g0):
    hy = S.hybrid(z, sh)
    def fam(pre):
        ks = [k for k in sorted(g0) if k.startswith(pre + "_")]
        return np.mean([g0[k] for k in ks], 0) if ks else None
    return hy, dict(ml2g0=g0["ml2g0"], catg0=g0["catg0"], rnn2=(hy["rnn0"] + hy["rnn1"]) / 2,
                    cnng0=fam("cnng0"), rnng0=fam("rnng0"), rnnft=fam("rnnft"), cnnft=fam("cnnft"),
                    rnnftx=fam("rnnftx"), cnnftx=fam("cnnftx"), ml2ft=g0.get("ml2ft"))


def decide(hy, pool, panel, w):
    names = [n for n in NAMES if w.get(n)]
    return decide_chain(hy, [pool[n] for n in names], [w[n] for n in names], panel, pool["catg0"], w["cat"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--base", default="v48_g0")
    ap.add_argument("--name", default="v50_april")
    ap.add_argument("--check-dir", default="t2_v48")
    ap.add_argument("--out-dir", default="t2_v50")
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    out = WORK / "results" / args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    src_val = WORK / "sub" / f"queue_validation_{args.base}.csv"
    dst_val = WORK / "sub" / f"queue_validation_{args.name}.csv"
    dst = WORK / "sub" / f"queue_private_{args.name}.csv"
    for p in (dst_val, dst):
        if p.exists():
            raise FileExistsError(p)
    shutil.copyfile(src_val, dst_val)                       # March windows: unchanged
    q = pd.read_csv(WORK / "sub" / f"queue_private_{args.base}.csv", dtype={"link_id": str})
    q["slot"] = slot_of(q.timestamp)
    records = []
    for panel in T2_PANELS:
        z = V27.official_cache(panel, "private")
        D = C.PanelData(panel, "private", unmasked=False)
        ws = official_windows(panel, "private", D)
        assert [w.T for w in ws] == list(z["T"] % STEPS)
        v46 = np.load(WORK / "results" / "t2_v46_slotT" / f"{panel}_private.npz")
        sh = {k[3:]: v46[k] for k in v46.files if k.startswith("sh_")}
        prev = np.load(WORK / "results" / args.check_dir / f"{panel}_private.npz")["new"]
        specs = [sp for sp in FT_SPECS if sp[0].startswith(("rnnft_", "cnnft_"))
                 or cfg["weights"].get(sp[0].rsplit("_", 1)[0])]
        g0 = S.g0_probs(D, ws, V48_SPECS + specs)
        g0["ml2g0"] = S.ml2_g0_probs(ws)
        g0["catg0"] = cat_g0_probs(ws)
        if cfg["weights"].get("ml2ft"):
            g0["ml2ft"] = S.ml2_g0_probs(ws, model_file=ML2FT_MODEL)
        hy, pool = pool_of(z, sh, g0)
        # the base rows must come back exactly from the V48 recipe before anything is replaced
        assert np.array_equal(decide(hy, pool, panel, cfg["v48"]), prev), (panel, "V48 not reproduced")
        new_pred = decide(hy, pool, panel, cfg["weights"])
        np.savez_compressed(out / f"{panel}_private.npz", ids=z["ids"], new=new_pred, **g0)
        pos = dict(zip(links(panel).link_id.astype(str), range(len(links(panel)))))
        for i, wid in enumerate(z["ids"]):
            mask = q.window_id.eq(wid).to_numpy()
            row = q.loc[mask]
            hor = (row.slot.to_numpy() - z["T"][i] - 1).astype(int)
            ix = row.link_id.map(pos).to_numpy().astype(int)
            old = row.queue_pred.to_numpy(int)
            assert np.array_equal(prev[i, hor, ix].astype(int), old), (wid, "base file differs from check-dir")
            new = new_pred[i, hor, ix].astype(int)
            records.append(dict(split="private", panel=panel, window_id=wid, changed=int((old != new).sum()),
                                added=int(((old == 0) & (new == 1)).sum()),
                                removed=int(((old == 1) & (new == 0)).sum())))
            q.loc[mask, "queue_pred"] = new
        print(panel, sum(r["changed"] for r in records if r["panel"] == panel), flush=True)
    q.drop(columns="slot").to_csv(dst, index=False)
    ch = pd.DataFrame(records)
    ch.to_csv(out / f"{args.name}_changes.csv", index=False)
    print(ch[["changed", "added", "removed"]].sum().to_dict(), "windows touched:", int((ch.changed > 0).sum()))


if __name__ == "__main__":
    main()
