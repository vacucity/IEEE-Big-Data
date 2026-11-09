#!/usr/bin/env python
"""Batched, cached ongoing-window experiments, with a validation-first selection.

All features use [T-60,T).  The private month is a confirmation set, never a
training source.  Cached labels are visible pseudo-window labels, not official
hidden competition labels.  Run ``cache``, ``select``, then ``confirm``.
"""
from __future__ import annotations

import os
for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "4"
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
torch.set_num_threads(4)
import t2_cnn as C
import t2_rnn as R
import t2_ens as E
import t2_pseudo as P
import t2_exp as EX
from evaluate import ROOT, T2_PANELS, links
from task2_ml import ctx_of, hist_features, to_frame, RADIUS_KM
from task2_ml2 import features2
from task2_rules import DEFAULTS as RULES, onset_scores

OUT = ROOT / "work" / "results" / "t2_ongoing_v27"
CACHE = ROOT / "work" / "cache" / "t2_ongoing_v27"


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024*1024),b""):
            digest.update(block)
    return digest.hexdigest()


def cache_path(panel, split):
    return CACHE / f"{panel}_{split}.npz"


def gbm_probs(ws):
    """Batch horizons into one prediction call, explicitly cap CPU threads."""
    ctx = ctx_of(ws[0].panel)
    bs = [E.v8_booster(), EX.booster("t2_ml2_ongoing_v2.txt")]
    out = [np.zeros((len(ws), 6, ctx.n), np.float32) for _ in bs]
    for start in range(0, len(ws), 24):
        frames = [[], []]
        positions = []
        for wi, w in enumerate(ws[start:start + 24], start):
            V, Q, O = (a.astype(float) for a in P.history(w))
            prior = onset_scores(ctx.ev, ctx.n, w.T + 6, w.weekday, None,
                                 RULES["sigma_h"], RULES["weekend"])
            fs = [hist_features(ctx, V, Q, O, w.T, int(w.weekday >= 5), prior),
                  features2(ctx, V, Q, O, w.T, int(w.weekday >= 5), prior, T2_PANELS.index(w.panel))]
            with np.errstate(invalid="ignore"):
                hq = (V <= ctx.vcut).any(0)
            near = np.zeros(ctx.n, bool)
            for xq in ctx.x[hq]:
                near |= np.abs(ctx.x - xq) <= RADIUS_KM
            idx = np.flatnonzero(near)
            for h in range(6):
                positions.append((wi, h, idx))
                for k, b in enumerate(bs):
                    frames[k].append(to_frame(fs[k], idx, h + 1)[b.feature_name()])
        for k, b in enumerate(bs):
            pp = b.predict(pd.concat(frames[k], ignore_index=True), num_threads=4)
            offset = 0
            for wi, h, idx in positions:
                out[k][wi, h, idx] = pp[offset:offset + len(idx)]
                offset += len(idx)
    return out


def neural_probs(D, ws):
    models = [("cnn0", C.load_model("", 96)), ("cnn1", C.load_model("_s1", 96)),
              ("cnn2", C.load_model("_s2", 96)),
              ("rnn0", R.load_model("")), ("rnn1", R.load_model("_s1"))]
    out = {name: np.zeros((len(ws), 6, D.n), np.float32) for name, _ in models}
    for start in range(0, len(ws), 32):
        xs = [C.make_x(D.V[w.day, w.T-12:w.T], D.Q[w.day, w.T-12:w.T],
                       D.O[w.day, w.T-12:w.T], D.static_ch, D.st["ffs"], D.st["cap"],
                       D.vcut, w.T, w.weekday) for w in ws[start:start+32]]
        for name, model in models:
            if name.startswith("cnn"):
                pr = C.predict_probs(model, xs)
            else:
                pr = R.predict_probs(model, xs)
            out[name][start:start+len(xs), :, D.perm] = pr
    return out


def cmd_cache(args):
    CACHE.mkdir(parents=True, exist_ok=True)
    for split in args.splits:
        for panel in T2_PANELS:
            path = cache_path(panel, split)
            if path.exists():
                print("exists", path.name, flush=True)
                continue
            start = time.time()
            ws = [w for w in P.windows(panel, split) if w.cond == "ongoing"]
            y, mask = zip(*(P.truth(w) for w in ws))
            D = C.PanelData(panel, split, unmasked=False)
            neural = neural_probs(D, ws)
            gb = gbm_probs(ws)
            np.savez_compressed(path, **neural, v8=gb[0], ml2=gb[1], y=np.stack(y), mask=np.stack(mask),
                                day=[w.day for w in ws], T=[w.T for w in ws], date=[w.date for w in ws],
                                perm=D.perm)
            print(split, panel, len(ws), f"{time.time()-start:.1f}s", flush=True)


def cmd_verify(args):
    audit = []
    for panel in T2_PANELS:
        w = next(w for w in P.windows(panel,"validation") if w.cond=="ongoing")
        z = dict(np.load(cache_path(panel,"validation")))
        reference = {"v8":E.v8_prob(w), "ml2":EX.ml2_prob(w),
                     "cnn":E.cnn_prob(w,("","_s1","_s2")), "rnn":R.rnn_prob(w,("","_s1"))}
        batched = dict(v8=z["v8"][0],ml2=z["ml2"][0],
                       cnn=(z["cnn0"][0]+z["cnn1"][0]+z["cnn2"][0])/3,
                       rnn=(z["rnn0"][0]+z["rnn1"][0])/2)
        diffs = {k:float(np.max(np.abs(reference[k]-batched[k]))) for k in reference}
        assert diffs["v8"]<1e-6 and diffs["ml2"]<1e-6, diffs
        assert diffs["cnn"]<.01 and diffs["rnn"]<.01, diffs
        audit.append(dict(panel=panel,**diffs))
        print(audit[-1],flush=True)
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/"batch_parity.json").write_text(json.dumps(audit,indent=2))


def load(split):
    return {p: dict(np.load(cache_path(p, split))) for p in T2_PANELS}


def probs(z, weights=(1, 1, 2, 4)):
    parts = [z["v8"], z["ml2"], (z["cnn0"] + z["cnn1"] + z["cnn2"])/3,
             (z["rnn0"] + z["rnn1"])/2]
    return sum(w*p for w, p in zip(weights, parts))/sum(weights)


def decision(pr, config, z):
    smooth = config.get("smooth", 0)
    if smooth:
        along = pr[..., z["perm"]]
        along = (1-smooth)*along + smooth*(np.concatenate([along[..., :1], along[..., :-1]], -1)
                  + np.concatenate([along[..., 1:], along[..., -1:]], -1))/2
        out = np.zeros_like(pr)
        out[..., z["perm"]] = along
        pr = out
    temporal = config.get("temporal", 0)
    if temporal:
        pr = (1-temporal)*pr + temporal*(np.concatenate([pr[:, :1], pr[:, :-1]], 1)
                   + np.concatenate([pr[:, 1:], pr[:, -1:]], 1))/2
    theta = config.get("theta", .45)
    return pr >= theta


def scores(z, pred):
    y = z["y"] & z["mask"]
    pred = pred & z["mask"]
    union = (y | pred).sum((1, 2))
    return np.divide((y & pred).sum((1, 2)), union, out=np.ones(len(y)), where=union>0)


def evaluate_cache(data, config):
    rows = []
    for panel, z in data.items():
        pr = probs(z, config["weights"])
        ious = scores(z, decision(pr, config, z))
        rows.extend(dict(panel=panel, day=int(d), T=int(t), date=str(date), iou=float(iou))
                    for d, t, date, iou in zip(z["day"], z["T"], z["date"], ious))
    return pd.DataFrame(rows)


BASE = dict(weights=[1, 1, 2, 4], theta=.45, smooth=0, temporal=0)


def cmd_select(args):
    OUT.mkdir(parents=True, exist_ok=True)
    data = load("validation")
    configs = [BASE]
    for weights in ([1,1,2,4], [1,1,2,6], [1,1,2,8], [1,1,1,5], [1,1,0,6],
                    [1,1,4,6], [1,0,2,5], [0,1,2,5], [0,0,1,3]):
        for theta in (.35, .375, .4, .425, .45, .475, .5, .525, .55):
            configs.append(dict(weights=weights, theta=theta, smooth=0, temporal=0))
    # Small predefined regularisers; preserve comparable thresholds/weights.
    for smooth in (.1, .2, .3):
        for theta in (.4, .45, .5):
            configs.append(dict(weights=[1,1,2,4], theta=theta, smooth=smooth, temporal=0))
    for temporal in (.2, .4, .6):
        for theta in (.4, .45, .5):
            configs.append(dict(weights=[1,1,2,4], theta=theta, smooth=0, temporal=temporal))
    rows = []
    for i, config in enumerate(configs):
        df = evaluate_cache(data, config)
        pm = df.groupby("panel").iou.mean()
        halves = [float(df.loc[(df.day<15)==first].groupby("panel").iou.mean().mean()) for first in (True, False)]
        rows.append(dict(id=i, score=float(pm.mean()), early=halves[0], late=halves[1],
                         **{p:float(pm[p]) for p in T2_PANELS}, config=json.dumps(config)))
    ranked = pd.DataFrame(rows).sort_values("score", ascending=False)
    ranked.to_csv(OUT / "selection_validation.csv", index=False)
    winner = json.loads(ranked.iloc[0].config)
    (OUT / "selected.json").write_text(json.dumps(winner, indent=2))
    print(ranked[["id","score","early","late","config"]].head(15).to_string(index=False), flush=True)
    print("BASE", rows[0], flush=True)


def cmd_confirm(args):
    chosen = json.loads((OUT / "selected.json").read_text())
    rows, windows = [], []
    for split in ("validation", "private"):
        data = load(split)
        for name, config in (("v26", BASE), ("v27", chosen)):
            df = evaluate_cache(data, config)
            df["split"], df["model"] = split, name
            windows.append(df)
            for panel, group in df.groupby("panel"):
                rows.append(dict(split=split, model=name, panel=panel, iou=group.iou.mean(), n=len(group)))
    pd.concat(windows).to_csv(OUT / "confirmed_windows.csv", index=False)
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "confirmed_panels.csv", index=False)
    (OUT/"confirmed_manifest.json").write_text(json.dumps(dict(config=chosen,
          panels_sha256=file_sha256(OUT/"confirmed_panels.csv")),indent=2))
    print(result.pivot(index=["split","panel"], columns="model", values="iou").to_string())
    print(result.groupby(["split","model"]).iou.mean().to_string(), flush=True)


def official_cache(panel, split):
    """Reuse only strictly preceding, score-eligible official history."""
    import pickle
    from evaluate import RELEASE, slot_of
    path = CACHE / f"official_{panel}_{split}.npz"
    if path.exists():
        return dict(np.load(path))
    st = C.static(panel)
    perm = st["perm"]
    L = links(panel)
    n = len(L)
    pos = dict(zip(L.link_id, range(n)))
    vcut = L.v_cut_kmh.to_numpy(np.float32)[perm]
    static_ch = np.stack([st["ffs"] / 120, st["lanes"] / 6, st["cap"] / st["lanes"] / 2500,
                         np.log(st["length"] + .01) / 3, st["n_on"], st["n_off"]]).astype(np.float32)
    base = RELEASE / "task2" / panel / split
    win = pd.read_csv(base / "window_index.csv")
    hist = pd.read_parquet(base / "window_history.parquet")
    hist["pos"] = hist.link_id.map(pos)
    hist["slot"] = slot_of(hist.timestamp)
    xs, ids, origins = [], [], []
    for w in win.itertuples(index=False):
        if w.condition != "queue_ongoing":
            continue
        origin = pd.Timestamp(w.forecast_origin)
        T = int(slot_of(pd.Series([w.forecast_origin]))[0])
        h = hist[hist.window_id == w.window_id]
        si = (h.slot-(T-12)).to_numpy()
        mask = (h.is_score_eligible.astype(bool) & h.speed_kmh.notna()).to_numpy() & (si>=0) & (si<12)
        assert (pd.to_datetime(h.loc[mask,"timestamp"]) < origin).all()
        g = []
        for col in ("speed_kmh","flow_vph","occupancy"):
            a = np.full((12,n),np.nan,np.float32)
            a[si[mask],h.pos.to_numpy()[mask]] = h[col].to_numpy()[mask]
            g.append(a[:,perm])
        xs.append(C.make_x(*g,static_ch,st["ffs"],st["cap"],vcut,T,origin.weekday()))
        ids.append(w.window_id)
        origins.append(T)
    models = [("cnn0",C.load_model("",96)),("cnn1",C.load_model("_s1",96)),
              ("cnn2",C.load_model("_s2",96)),("rnn0",R.load_model("")),("rnn1",R.load_model("_s1"))]
    out = dict(ids=np.array(ids), T=np.array(origins), perm=perm)
    for name,model in models:
        pr = C.predict_probs(model,xs) if name.startswith("cnn") else R.predict_probs(model,xs)
        a = np.zeros((len(xs),6,n),np.float32)
        a[:,:,perm] = pr
        out[name] = a
    for key,prefix in (("v8","t2ml_lgbm"),("ml2","t2ml2")):
        with open(ROOT/"work"/"sub"/"probs"/f"{prefix}_probs_{split}_{panel}.pkl","rb") as f:
            records = {w["window_id"]:w for w in pickle.load(f)}
        out[key] = np.stack([records[wid]["prob"] for wid in ids])
    # Save inputs so an independently trained candidate reuses the same contract.
    out["x"] = np.stack(xs)
    np.savez_compressed(path,**out)
    return out


def cmd_official(args):
    from evaluate import slot_of
    if args.kind == "lovasz":
        import t2_ongoing_v27_train as FT
        config = FT.checked_selection()
        candidate = FT.load_model(config["model_sha256"])
        reports = pd.read_csv(OUT/"lovasz_confirmed_panels.csv")
        manifest_path, report_path = OUT/"lovasz_confirmed_manifest.json", OUT/"lovasz_confirmed_panels.csv"
        candidate_name = "v27_lovasz"
        reference_name = "v27_rules"
    else:
        config = json.loads((OUT/"selected.json").read_text())
        reports = pd.read_csv(OUT/"confirmed_panels.csv")
        manifest_path, report_path = OUT/"confirmed_manifest.json", OUT/"confirmed_panels.csv"
        candidate_name = "v27"
        reference_name = "v26"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("config") != config or manifest.get("panels_sha256") != file_sha256(report_path):
        raise ValueError("Confirmation report does not match selection/checkpoint; rerun confirm")
    summary = reports.groupby(["split","model"]).iou.mean().unstack()
    if not (summary[candidate_name] > summary[reference_name]).all():
        raise ValueError("Candidate did not improve both months; refusing official CSV")
    audit = []
    for split in ("validation","private"):
        q = pd.read_csv(args.base.format(split=split))
        q["slot"] = slot_of(q.timestamp)
        changed, baseline_disagreements = 0, 0
        for panel in T2_PANELS:
            z = official_cache(panel,split)
            if args.kind == "lovasz":
                a = np.zeros_like(z["rnn0"])
                a[:,:,z["perm"]] = R.predict_probs(candidate,list(z["x"]))
                pr = (1-config["alpha"])*probs(z,config["base_config"]["weights"])+config["alpha"]*a
                pred = pr>=config["theta"]
            else:
                pr = probs(z,config["weights"])
                pred = decision(pr,config,z)
            baseline = probs(z)>=.45
            np.savez_compressed(CACHE/f"candidate_{args.kind}_{panel}_{split}.npz",ids=z["ids"],prob=pr,pred=pred,T=z["T"])
            pos = dict(zip(links(panel).link_id,range(len(links(panel)))))
            for i,wid in enumerate(z["ids"]):
                mask = (q.window_id==wid).to_numpy()
                rows = q[mask]
                horizon = (rows.slot-z["T"][i]-1).to_numpy()
                assert ((horizon>=0)&(horizon<6)).all()
                idx = rows.link_id.map(pos).to_numpy()
                old = rows.queue_pred.to_numpy()
                baseline_disagreements += int((baseline[i,horizon,idx]!=old).sum())
                new = pred[i,horizon,idx].astype(int)
                changed += int((new!=old).sum())
                q.loc[mask,"queue_pred"] = new
        out = args.out.format(split=split)
        q[["window_id","timestamp","link_id","queue_pred"]].to_csv(out,index=False)
        audit.append(dict(split=split,output=out,changed=changed,baseline_disagreements=baseline_disagreements))
        print(audit[-1],flush=True)
    recipe = dict(kind=args.kind,config=config,history="[T-60,T)",validation_selection=True,
                  official_hidden_labels_used=False,metrics=summary.to_dict(),audit=audit)
    (OUT/f"official_{args.kind}_recipe.json").write_text(json.dumps(recipe,indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("cache")
    p.add_argument("--splits", nargs="+", default=["validation", "private"])
    sub.add_parser("verify")
    sub.add_parser("select")
    sub.add_parser("confirm")
    p = sub.add_parser("official")
    p.add_argument("--kind",choices=["rules","lovasz"],default="rules")
    p.add_argument("--base",default="work/sub/queue_{split}_v26.csv")
    p.add_argument("--out",default="work/sub/queue_{split}_v27_ongoing.csv")
    args = parser.parse_args()
    globals()["cmd_" + args.cmd](args)
