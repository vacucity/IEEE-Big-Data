"""V71: April (private) ongoing windows from the 48k-window March fine-tunes; everything else = V69.

Members (t2_rnn_march.py --windows 48000, results/ft70.status): rnnft48 = BiLSTM s0/s1 (_g0_hw48all,
_g0_s1_hw48all), rnnftx48 = BiLSTM s2/s3, cnnftx48 = CNN s1/s2, blended with the V51 weights (4 / 2 / 4; no weight
search).  Evidence (t2_v71_grid48.py, March pseudo ongoing, cross-fitted by half month): V51 0.9224 (early 0.9330,
late 0.9144) -> 0.9254 (0.9353, 0.9166); each single family swap also improves both halves.  The pre-set rule (gain
>= 0.001 and the nested half-month check of the best grid pick both ways) failed in one direction (late-picked
weights 4/4/2 score early 0.9322 < 0.9330); the user chose on 2026-10-25 to adopt this pre-specified swap anyway
(disclose in the report).  April is never used to choose.

Chain for the April ongoing rows: V71 blend -> V33 adaptive IoU -> V52 'hyb' selector conditioning (constrained
windows; same rule, K, floor and seeds as t2_v52_selrule.py) ; onset rows = V54 ; March rows = V69 untouched.
Checks: the V48 and V51 recipes reproduce their saved official decisions from the same probabilities, and V51 + hyb is
compared with the submitted V52 rows.

  python work/t2_v71_april.py                 # -> sub/queue_private_v71_april.csv, submissions/v71_april.zip
"""
from __future__ import annotations

import io
import json
import os
import sys
import zipfile
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
from evaluate import RELEASE, ROOT, STEPS, T2_PANELS, links, slot_of  # noqa: E402
from t2_cat_g0 import cat_g0_probs  # noqa: E402
from t2_g0_eval import decide as decide_chain  # noqa: E402
from t2_group_v27 import groups, smooth  # noqa: E402
from t2_v33_adaptive_iou import adaptive_iou  # noqa: E402
from t2_v46_slotT import official_windows  # noqa: E402

WORK = ROOT / "work"
OUT = WORK / "results" / "t2_v71"
SUB = WORK / "submissions"
BASE_ZIP, NAME = "v69_cmgbm", "v71_april"
NEW_SPECS = [("rnnft48_0", "rnn", "_g0_hw48all"), ("rnnft48_1", "rnn", "_g0_s1_hw48all"),
             ("rnnftx48_2", "rnn", "_g0_s2_hw48all"), ("rnnftx48_3", "rnn", "_g0_s3_hw48all"),
             ("cnnftx48_1", "cnn", "_g0_s1_hw48all"), ("cnnftx48_2", "cnn", "_g0_s2_hw48all")]
W51 = json.loads((WORK / "results" / "t2_v51" / "config.json").read_text())
W71 = {**dict.fromkeys(W51["weights"], 0), "rnnft48": 4, "rnnftx48": 2, "cnnftx48": 4, "cat": 0}


ORDER = (*A.NAMES, "rnnft48", "rnnftx48", "cnnftx48", "rnnft96", "cnnftx96")   # summation order (float-exact)
# V72 candidate (t2_v72_grid96.py): BiLSTM s0/s1 and CNN s1/s2 fine-tuned on 96k windows, BiLSTM s2/s3 stay 48k
V72_SPECS = [("rnnft96_0", "rnn", "_g0_hw96all"), ("rnnft96_1", "rnn", "_g0_s1_hw96all"),
             ("rnnftx48_2", "rnn", "_g0_s2_hw48all"), ("rnnftx48_3", "rnn", "_g0_s3_hw48all"),
             ("cnnftx96_1", "cnn", "_g0_s1_hw96all"), ("cnnftx96_2", "cnn", "_g0_s2_hw96all")]
W72 = {**dict.fromkeys(W51["weights"], 0), "rnnft96": 4, "rnnftx48": 2, "cnnftx96": 4, "cat": 0}
VARIANTS = {"v71": ("v71_april", NEW_SPECS, W71, OUT), "v72": ("v72_april", V72_SPECS, W72, WORK / "results" / "t2_v72")}


def decide(hy, pool, panel, w):
    names = [n for n in ORDER if w.get(n)]
    return decide_chain(hy, [pool[n] for n in names], [w[n] for n in names], panel, pool["catg0"], w["cat"])


def chain_p(pool, panel, w):
    """The blended, group-smoothed probability decide() feeds to adaptive_iou (cat weight 0 for V51/V71)."""
    assert not w["cat"]
    names = [n for n in ORDER if w.get(n)]
    pr = sum(w[n] * pool[n].astype(np.float32) for n in names) / sum(w[n] for n in names)
    return smooth(pr, groups(panel, .95), .25)


def main():
    NAME, SPECS, WTS, OUT = VARIANTS[sys.argv[sys.argv.index('--variant') + 1] if '--variant' in sys.argv else 'v71']
    OUT.mkdir(parents=True, exist_ok=True)
    dst = WORK / "sub" / f"queue_private_{NAME}.csv"
    out_zip = SUB / f"{NAME}.zip"
    for p in (dst, out_zip):
        if p.exists():
            raise FileExistsError(p)
    cls = pd.read_csv(S52.OUT / "official_classes.csv").set_index("window_id")
    q = pd.read_csv(WORK / "sub" / "queue_private_v54_onset.csv", dtype={"link_id": str})
    q52 = pd.read_csv(WORK / "sub" / "queue_private_v52_selrule.csv", dtype={"link_id": str})
    assert q.drop(columns="queue_pred").equals(q52.drop(columns="queue_pred"))
    q0 = q.copy()
    q["slot"] = slot_of(q.timestamp)
    log = []
    for panel in T2_PANELS:
        z = V27.official_cache(panel, "private")
        D = C.PanelData(panel, "private", unmasked=False)
        ws = official_windows(panel, "private", D)
        assert [w.T for w in ws] == list(z["T"] % STEPS)
        v46 = np.load(WORK / "results" / "t2_v46_slotT" / f"{panel}_private.npz")
        sh = {k[3:]: v46[k] for k in v46.files if k.startswith("sh_")}
        g0 = S.g0_probs(D, ws, A.V48_SPECS + A.FT_SPECS + SPECS)
        g0["ml2g0"] = S.ml2_g0_probs(ws)
        g0["catg0"] = cat_g0_probs(ws)
        hy, pool = A.pool_of(z, sh, g0)
        for fam in ("rnnft48", "rnnftx48", "cnnftx48", "rnnft96", "cnnftx96"):
            ks = [k for k in sorted(g0) if k.startswith(fam + "_")]
            if ks:
                pool[fam] = np.mean([g0[k] for k in ks], 0)
        pool = {k: v for k, v in pool.items() if v is not None}
        P48 = decide(hy, pool, panel, W51["v48"])
        P51 = decide(hy, pool, panel, W51["weights"])
        for tag, Pt in (("v48", P48), ("v51", P51)):     # recorded decisions (stop on a difference unless tolerant)
            rec = np.load(WORK / "results" / f"t2_{tag}" / f"{panel}_private.npz")["new"]
            assert rec.shape == Pt.shape, (panel, tag, rec.shape, Pt.shape)
            RC.soft(np.array_equal(Pt, rec), f"{panel} April {tag.upper()} ongoing decision differs from the recorded "
                    f"results/t2_{tag} in {int((Pt != rec).sum())} cells (with TFB_TOLERANT=1 the recomputed "
                    "decision is used)")
        P71 = decide(hy, pool, panel, WTS)
        p51, p71 = chain_p(pool, panel, W51["weights"]), chain_p(pool, panel, WTS)
        assert np.array_equal(np.stack([adaptive_iou(x) for x in p71]), P71)
        np.savez_compressed(OUT / f"{panel}_private.npz", ids=z["ids"], p71=p71, P71=P71, P51=P51)
        pos = dict(zip(links(panel).link_id.astype(str), range(len(links(panel)))))
        for i, wid in enumerate(z["ids"]):
            wid = str(wid)
            m = q.window_id.eq(wid).to_numpy()
            row = q.loc[m]
            hor = (row.slot.to_numpy() - z["T"][i] - 1).astype(int)
            ix = row.link_id.map(pos).to_numpy().astype(int)
            c = cls.loc[wid].to_dict()
            c["panel"], c["window_id"] = panel, wid
            N51, _, _ = S52.decide(S52.RULE, P51[i], p51[i], P51[i], c)
            N71, tag, acc = S52.decide(S52.RULE, P71[i], p71[i], P71[i], c)
            old = row.queue_pred.to_numpy(int)
            new = N71[hor, ix].astype(int)
            log.append(dict(panel=panel, window_id=wid, cls=c["cls"], rule=tag, acc=acc,
                            v52_repro_diff=int((N51[hor, ix].astype(int) != old).sum()),
                            changed=int((old != new).sum()), added=int(((old == 0) & (new == 1)).sum()),
                            removed=int(((old == 1) & (new == 0)).sum())))
            q.loc[m, "queue_pred"] = new
        print(panel, {k: sum(r[k] for r in log if r["panel"] == panel) for k in ("changed", "v52_repro_diff")},
              flush=True)
    q = q.drop(columns="slot")
    assert q.drop(columns="queue_pred").equals(q0.drop(columns="queue_pred"))
    wi = pd.concat([pd.read_csv(RELEASE / "task2" / p / "private" / "window_index.csv") for p in T2_PANELS])
    onset = q.window_id.isin(wi.window_id[wi.condition == "queue_onset"])
    assert (q.queue_pred[onset] == q0.queue_pred[onset]).all(), "onset rows must stay V54"
    q.to_csv(dst, index=False)
    lg = pd.DataFrame(log)
    lg.to_csv(OUT / f"{NAME}_changes.csv", index=False)
    print(lg[lg.changed > 0].to_string())
    print(lg[["changed", "added", "removed", "v52_repro_diff"]].sum().to_dict(),
          "windows touched:", int((lg.changed > 0).sum()))
    # patch the V69 zip: only the changed private queue cells
    key = pd.read_csv(ROOT / "kaggle_public" / "submission_key.csv", usecols=["submission_id", "task", "window_id",
                                                                             "timestamp", "link_id"], dtype=str)
    key = key[key.task == "queue"]
    ch = q[q.queue_pred.to_numpy() != q0.queue_pred.to_numpy()]
    mk = ch.merge(key, on=["window_id", "timestamp", "link_id"], how="left", validate="one_to_one")
    assert mk.submission_id.notna().all()
    patch = {int(s): (str(v), str(1 - v)) for s, v in zip(mk.submission_id, mk.queue_pred.astype(int))}
    src = zipfile.ZipFile(SUB / f"{BASE_ZIP}.zip")
    seen = 0
    with src.open(src.namelist()[0]) as fin, zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zout, \
            zout.open(f"{NAME}.csv", "w", force_zip64=True) as dstf:
        for i, raw in enumerate(io.BufferedReader(fin, 1 << 22)):
            if i == 0:
                dstf.write(raw)
                continue
            t = patch.get(int(raw.split(b",", 1)[0]))
            if t is None:
                dstf.write(raw)
                continue
            body = raw.rstrip(b"\r\n")
            parts = body.split(b",")
            assert parts[1] == b"queue" and int(float(parts[4])) == int(t[1]), (i, parts, t)
            parts[4] = t[0].encode()
            dstf.write(b",".join(parts) + raw[len(body):])
            seen += 1
    assert seen == len(patch), (seen, len(patch))
    from make_submission import validate
    checks = validate(out_zip)
    if not checks["ok"]:
        raise ValueError(checks)
    (WORK / "results" / f"{NAME}_manifest.json").write_text(json.dumps(dict(
        name=NAME, base=BASE_ZIP, changed_queue_cells=len(patch), weights=WTS,
        queue_private=str(dst), checks=checks), indent=2, default=str), encoding="utf-8")
    print("changed cells", len(patch), checks)


if __name__ == "__main__":
    main()
