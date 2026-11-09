"""Build V41 queue: a small CatBoost ensemble only on confirmed ongoing panels."""
from __future__ import annotations
import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "4"
from pathlib import Path
import json
import hashlib
from catboost import CatBoostClassifier
import numpy as np
import pandas as pd

from evaluate import RELEASE, T2_PANELS, links, slot_of
from task2_ml import ctx_of, to_frame, RADIUS_KM
from task2_ml2 import features2
from task2_rules import DEFAULTS, onset_scores
from t2_ongoing_v27 import official_cache, probs as base_probs
from t2_group_v27 import groups, smooth
from t2_v33_adaptive_iou import adaptive_iou

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "t2_v41_ongoing_cat"
MODEL = ROOT / "models" / "t2_v41_ongoing_cat.cbm"
PANELS = ("D7_I10_E", "D7_I210_W")
ALPHA = .1


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(4 << 20), b""):
            h.update(b)
    return h.hexdigest()


def cat_official(panel, split, model, feats, z):
    ctx = ctx_of(panel)
    folder = RELEASE / "task2" / panel / split
    win = pd.read_csv(folder / "window_index.csv")
    hist = pd.read_parquet(folder / "window_history.parquet")
    link_ids = links(panel).link_id.astype(str).to_numpy()
    hist["pos"] = hist.link_id.map(dict(zip(link_ids, range(ctx.n))))
    hist["slot"] = slot_of(hist.timestamp)
    rows = win[win.condition == "queue_ongoing"]
    assert list(rows.window_id) == list(z["ids"])
    out = np.zeros_like(z["ml2"], dtype=np.float32)
    for i, w in enumerate(rows.itertuples(index=False)):
        origin = pd.Timestamp(w.forecast_origin)
        T = int(slot_of(pd.Series([w.forecast_origin]))[0])
        assert T == z["T"][i]
        tod = origin.hour * 12 + origin.minute // 5
        h = hist[hist.window_id == w.window_id]
        si = (h.slot - (T - 12)).to_numpy()
        mask = h.is_score_eligible.astype(bool).to_numpy() & h.speed_kmh.notna().to_numpy()
        mask &= (si >= 0) & (si < 12)
        assert (pd.to_datetime(h.loc[mask, "timestamp"], utc=True) < origin).all()
        grids = []
        for col in ("speed_kmh", "flow_vph", "occupancy"):
            a = np.full((12, ctx.n), np.nan, np.float32)
            a[si[mask], h.pos.to_numpy()[mask]] = h[col].to_numpy()[mask]
            grids.append(a)
        prior = onset_scores(ctx.ev, ctx.n, tod + 6, origin.weekday(), None,
                             DEFAULTS["sigma_h"], DEFAULTS["weekend"])
        fs = features2(ctx, *grids, tod, int(origin.weekday() >= 5), prior,
                       T2_PANELS.index(panel))
        with np.errstate(invalid="ignore"):
            hq = (grids[0] <= ctx.vcut).any(0)
        near = np.zeros(ctx.n, bool)
        for xq in ctx.x[hq]:
            near |= np.abs(ctx.x - xq) <= RADIUS_KM
        idx = np.flatnonzero(near)
        if len(idx):
            frame = pd.concat([to_frame(fs, idx, h + 1)[feats] for h in range(6)],
                              ignore_index=True).fillna(-999).astype(np.float32)
            p = model.predict_proba(frame, thread_count=4)[:, 1]
            out[i][:, idx] = p.reshape(6, len(idx))
    return out


def main():
    check = pd.read_csv(OUT / "panel_scores.csv")
    for panel in PANELS:
        for split in ("validation", "private"):
            q = check[(check.panel == panel) & (check.split == split)]
            score = q.set_index("alpha").iou
            assert score.loc[ALPHA] > score.loc[0.], (panel, split)
    model = CatBoostClassifier()
    model.load_model(str(MODEL))
    feats = model.feature_names_
    records = []
    for split in ("validation", "private"):
        src = ROOT / "sub" / f"queue_{split}_v33_adaptive_iou.csv"
        dest = ROOT / "sub" / f"queue_{split}_v41_ongoing_cat.csv"
        if dest.exists():
            raise FileExistsError(dest)
        q = pd.read_csv(src, dtype={"link_id": str})
        q["slot"] = slot_of(q.timestamp)
        for panel in PANELS:
            z = official_cache(panel, split)
            cat = cat_official(panel, split, model, feats, z)
            np.savez_compressed(OUT / f"official_{panel}_{split}.npz",
                                ids=z["ids"], T=z["T"], cat=cat)
            g = groups(panel, .95)
            baseline = smooth(base_probs(z, [1, 1, 1, 5]), g, .25)
            candidate = (1 - ALPHA) * baseline + ALPHA * smooth(cat, g, .25)
            old = np.stack([adaptive_iou(x) for x in baseline])
            new = np.stack([adaptive_iou(x) for x in candidate])
            pos = dict(zip(links(panel).link_id.astype(str), range(ctx_of(panel).n)))
            for i, wid in enumerate(z["ids"]):
                mask = q.window_id.eq(wid).to_numpy()
                row = q.loc[mask]
                assert len(row) > 0
                hor = (row.slot.to_numpy() - z["T"][i] - 1).astype(int)
                ix = row.link_id.map(pos).to_numpy()
                assert np.isfinite(ix).all() and ((hor >= 0) & (hor < 6)).all()
                ix = ix.astype(int)
                previous = row.queue_pred.to_numpy(int)
                assert np.array_equal(old[i, hor, ix].astype(int), previous), (panel, split, wid)
                updated = new[i, hor, ix].astype(int)
                changed = int(np.sum(previous != updated))
                records.append(dict(split=split, panel=panel, window_id=wid,
                                    rows=len(row), changed=changed,
                                    added=int(np.sum((previous == 0) & (updated == 1))),
                                    removed=int(np.sum((previous == 1) & (updated == 0)))))
                q.loc[mask, "queue_pred"] = updated
            print("built", split, panel, flush=True)
        q.drop(columns="slot").to_csv(dest, index=False)
        assert len(q) == len(pd.read_csv(src))
        print("wrote", dest, flush=True)
    changes = pd.DataFrame(records)
    changes.to_csv(OUT / "official_changes.csv", index=False)
    manifest = dict(base="V40", panels=PANELS, alpha=ALPHA,
                    queue_changes=changes.groupby("split")[["changed", "added", "removed"]].sum().to_dict(),
                    model_sha256=sha(MODEL),
                    queue_sha256={s: sha(ROOT / "sub" / f"queue_{s}_v41_ongoing_cat.csv")
                                  for s in ("validation", "private")})
    (OUT / "official_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(changes.groupby(["split", "panel"])[["changed", "added", "removed"]].sum().to_string())


if __name__ == "__main__":
    np.seterr(invalid="ignore", divide="ignore")
    main()
