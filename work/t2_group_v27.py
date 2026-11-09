"""Train-only queue co-occurrence smoothing, validated before private confirmation."""
from __future__ import annotations
import os
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "2"
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from evaluate import ROOT, T2_PANELS, links
from t1_unet import static

OUT = ROOT / "work" / "results" / "t2_group_v27"
CACHE = ROOT / "work" / "cache" / "t2_ongoing_v27"


def learn():
    OUT.mkdir(parents=True, exist_ok=True)
    for panel in T2_PANELS:
        L = links(panel)
        with np.load(ROOT / "work" / "cache" / "t2_grids" / f"{panel}_train.npz") as z:
            V = z["Vt"].reshape(-1, len(L))
        Y = V <= L.v_cut_kmh.to_numpy()[None]
        M = np.isfinite(V)
        perm = static(panel)["perm"]
        rows = []
        # Local pairs only: all group membership is learned from train months.
        for delta in range(1, 5):
            for a, b in zip(perm[:-delta], perm[delta:]):
                valid = M[:, a] & M[:, b]
                a_y, b_y = Y[:, a] & valid, Y[:, b] & valid
                inter, union = int((a_y & b_y).sum()), int((a_y | b_y).sum())
                if inter >= 100:
                    rows.append(dict(a=int(a), b=int(b), inter=inter, union=union, jaccard=inter/union))
        pd.DataFrame(rows).to_csv(OUT / f"{panel}_pairs.csv", index=False)
        print(panel, "pairs", len(rows), flush=True)


def groups(panel, threshold):
    df = pd.read_csv(OUT / f"{panel}_pairs.csv")
    # Complete-link grouping prevents chains of only weakly related endpoints.
    n = len(links(panel))
    similarity = np.eye(n)
    for row in df.itertuples(index=False):
        similarity[row.a, row.b] = similarity[row.b, row.a] = row.jaccard
    result = [[int(i)] for i in static(panel)["perm"]]
    changed = True
    while changed:
        changed = False
        for k in range(len(result)-1):
            if np.min(similarity[np.ix_(result[k], result[k+1])]) >= threshold:
                result[k:k+2] = [result[k] + result[k+1]]
                changed = True
                break
    return [g for g in result if len(g) > 1]


def smooth(prob, group_list, alpha):
    result = prob.copy()
    for g in group_list:
        result[..., g] = (1-alpha)*prob[..., g] + alpha*prob[..., g].mean(-1, keepdims=True)
    return result


def load(split):
    data = {}
    for p in T2_PANELS:
        with np.load(CACHE / f"{p}_{split}.npz") as f:
            z = dict(f)
        pr = (z["v8"]+z["ml2"]+2*(z["cnn0"]+z["cnn1"]+z["cnn2"])/3+2*(z["rnn0"]+z["rnn1"]))/8
        data[p] = {"pr": pr, "y": z["y"], "mask": z["mask"], "day": z["day"], "T": z["T"]}
    return data


def evaluate(data, cfg):
    rows = []
    for p, z in data.items():
        gs = groups(p, cfg["similarity"]) if cfg["alpha"] else []
        pred = smooth(z["pr"], gs, cfg["alpha"]) >= cfg["theta"]
        pred &= z["mask"]
        y = z["y"] & z["mask"]
        inter, union = (pred & y).sum((1,2)), (pred | y).sum((1,2))
        scores = np.divide(inter, union, out=np.ones(len(union)), where=union>0)
        rows.extend(dict(panel=p, day=int(d), T=int(t), iou=float(s)) for d,t,s in zip(z["day"],z["T"],scores))
    return pd.DataFrame(rows)


BASE = dict(similarity=1, alpha=0, theta=.45)


def select():
    data = load("validation")
    configs = [BASE] + [dict(similarity=s, alpha=a, theta=t) for s in (.8,.9,.95)
                       for a in (.25,.5,.75,1) for t in (.4,.45,.5)]
    rows = []
    for cfg in configs:
        df = evaluate(data, cfg)
        rows.append(dict(score=df.groupby("panel").iou.mean().mean(),
                         early=df[df.day<15].groupby("panel").iou.mean().mean(),
                         late=df[df.day>=15].groupby("panel").iou.mean().mean(), **cfg))
    rank = pd.DataFrame(rows).sort_values("score", ascending=False)
    rank.to_csv(OUT / "validation_selection.csv", index=False)
    chosen = {k: float(rank.iloc[0][k]) for k in BASE}
    (OUT / "selected.json").write_text(json.dumps(chosen, indent=2), encoding="utf-8")
    print(rank.head(12).to_string(index=False), flush=True)


def confirm():
    cfg = json.loads((OUT / "selected.json").read_text())
    frames = []
    for split in ("validation", "private"):
        data = load(split)
        for name, c in (("v26", BASE), ("group", cfg)):
            df = evaluate(data, c).assign(split=split, model=name)
            frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(OUT / "confirmed_windows.csv", index=False)
    print(out.groupby(["split","model","panel"]).iou.mean().groupby(["split","model"]).mean().to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", choices=["learn", "select", "confirm"])
    globals()[ap.parse_args().cmd]()
