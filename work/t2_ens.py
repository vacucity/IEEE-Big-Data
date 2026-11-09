#!/usr/bin/env python
"""Ongoing-window ensembles on the pseudo windows: v8 GBM (task2_ml), ml2 GBM (task2_ml2) and the CNN (t2_cnn)."""
from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

import lightgbm as lgb
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import ROOT, T2_PANELS  # noqa: E402
from t2_pseudo import HIST, HOR, evaluate, history, summary  # noqa: E402
import t2_cnn as C  # noqa: E402
import t2_exp  # noqa: E402
from task2_ml import LGB_PARAMS, RADIUS_KM, ctx_of, fit, hist_features, load_rows, to_frame  # noqa: E402
from task2_rules import DEFAULTS as RULES, onset_scores  # noqa: E402

V8 = ROOT / "work" / "models" / "t2_v8_ongoing.txt"


@lru_cache(maxsize=None)
def v8_booster() -> lgb.Booster:
    if not V8.exists():
        d = load_rows(T2_PANELS, "ongoing")
        _, i0 = fit("lgbm", d[d.fold != 0], d[d.fold == 0], params=LGB_PARAMS)
        _, i1 = fit("lgbm", d[d.fold != 1], d[d.fold == 1], params=LGB_PARAMS)
        feats = [c for c in d.columns if c not in ("y", "fold", "date", "kind", "panel")]
        b = lgb.train(LGB_PARAMS, lgb.Dataset(d[feats], d.y), int(1.1 * (i0 + i1) / 2))
        b.save_model(str(V8))
        print(f"  v8 ongoing rounds {i0}/{i1}", flush=True)
    return lgb.Booster(model_file=str(V8))


_c: dict = {}


def v8_prob(w) -> np.ndarray:
    key = ("v8", w.panel, w.split, w.day, w.T)
    if key not in _c:
        ctx = ctx_of(w.panel)
        V, Q, O = (a.astype(float) for a in history(w))
        prior = onset_scores(ctx.ev, ctx.n, w.T + HOR, w.weekday, None, RULES["sigma_h"], RULES["weekend"])
        f = hist_features(ctx, V, Q, O, w.T, int(w.weekday >= 5), prior)
        with np.errstate(invalid="ignore"):
            hq = (V <= ctx.vcut).any(0)
        near = np.zeros(ctx.n, bool)
        for xq in ctx.x[hq]:
            near |= np.abs(ctx.x - xq) <= RADIUS_KM
        idx = np.flatnonzero(near)
        b = v8_booster()
        prob = np.zeros((HOR, ctx.n))
        for s in range(HOR if len(idx) else 0):
            prob[s, idx] = b.predict(to_frame(f, idx, s + 1)[b.feature_name()])
        _c[key] = prob
    return _c[key]


CNN_TAGS = ("",)


@lru_cache(maxsize=8)
def cnn_model(tag: str = ""):
    return C.load_model(tag, 96)


@lru_cache(maxsize=16)
def panel_data(panel: str, split: str):
    return C.PanelData(panel, split, unmasked=False)


def cnn_prob(w, tags=None) -> np.ndarray:
    """Mean CNN probability over the seeds in `tags` (default CNN_TAGS), pos order."""
    tags = tuple(tags or CNN_TAGS)
    key = ("cnn", tags, w.panel, w.split, w.day, w.T)
    if key not in _c:
        D = panel_data(w.panel, w.split)
        h = slice(w.T - HIST, w.T)
        x = C.make_x(D.V[w.day, h], D.Q[w.day, h], D.O[w.day, h], D.static_ch, D.st["ffs"], D.st["cap"], D.vcut,
                     w.T, w.weekday)
        pr = np.mean([C.predict_probs(cnn_model(t), [x])[0] for t in tags], 0)
        out = np.zeros_like(pr)
        out[:, D.perm] = pr
        _c[key] = out
    return _c[key]


if __name__ == "__main__":
    v8_booster()
    combos = {"v8": (1, 0, 0), "ml2": (0, 1, 0), "cnn": (0, 0, 1), "v8+cnn": (0.5, 0, 0.5),
              "ml2+cnn": (0, 0.5, 0.5), "v8+ml2+cnn": (1 / 3, 1 / 3, 1 / 3), "v8/4+ml2/4+cnn/2": (0.25, 0.25, 0.5)}
    for name, (a, b, c) in combos.items():
        f = lambda w: (a * (v8_prob(w) if a else 0) + b * (t2_exp.ml2_prob(w) if b else 0)
                       + c * (cnn_prob(w) if c else 0)) >= 0.4
        df = evaluate(f, conds=("ongoing",))
        print(f"{name:18s}", summary(df).ongoing.to_dict(), flush=True)


# ---------------------------------------------------------------------------
# official windows
# ---------------------------------------------------------------------------
def official_cnn(panel: str, split: str, tags=None) -> dict[str, np.ndarray]:
    """window_id -> (6, n) CNN probabilities (pos order) for the official ongoing windows."""
    import pandas as pd
    from evaluate import RELEASE, links, slot_of
    st = C.static(panel)
    perm = st["perm"]
    L = links(panel)
    n = len(L)
    vcut = L.v_cut_kmh.to_numpy(np.float32)[perm]
    static_ch = np.stack([st["ffs"] / 120, st["lanes"] / 6, st["cap"] / st["lanes"] / 2500,
                          np.log(st["length"] + 0.01) / 3, st["n_on"], st["n_off"]]).astype(np.float32)
    base = RELEASE / "task2" / panel / split
    win = pd.read_csv(base / "window_index.csv")
    hist = pd.read_parquet(base / "window_history.parquet")
    hist["pos"] = hist.link_id.map(dict(zip(L.link_id, range(n))))
    hist["slot"] = slot_of(hist.timestamp)
    out = {}
    for w in win.itertuples(index=False):
        if w.condition != "queue_ongoing":
            continue
        T = int(slot_of(pd.Series([w.forecast_origin]))[0])
        origin = pd.Timestamp(w.forecast_origin)
        h = hist[hist.window_id == w.window_id]
        ok = (h.is_score_eligible.astype(bool) & h.speed_kmh.notna()).to_numpy()
        si = (h.slot - (T - HIST)).to_numpy()
        m = ok & (si >= 0) & (si < HIST)
        g = {}
        for c in ("speed_kmh", "flow_vph", "occupancy"):
            a = np.full((HIST, n), np.nan, np.float32)
            a[si[m], h.pos.to_numpy()[m]] = h[c].to_numpy()[m]
            g[c] = a[:, perm]
        x = C.make_x(g["speed_kmh"], g["flow_vph"], g["occupancy"], static_ch, st["ffs"], st["cap"], vcut,
                     origin.hour * 12 + origin.minute // 5, origin.weekday())
        pr = np.mean([C.predict_probs(cnn_model(t), [x])[0] for t in (tags or CNN_TAGS)], 0)
        o = np.zeros_like(pr)
        o[:, perm] = pr
        out[w.window_id] = o
    return out


def cmd_official(base: str, out: str, theta: float = 0.45, tags=None, rnn_tags=None, weights=None) -> None:
    """Ongoing windows of a base queue file -> mean of v8, ml2 and CNN probabilities >= theta."""
    import pickle
    import pandas as pd
    from evaluate import links, slot_of
    probs_dir = ROOT / "work" / "sub" / "probs"
    for split in ("validation", "private"):
        q = pd.read_csv(base.format(split=split))
        q["slot"] = slot_of(q.timestamp)
        changed = 0
        for p in T2_PANELS:
            pos = dict(zip(links(p).link_id, range(len(links(p)))))
            v8 = {w["window_id"]: w for w in pickle.load(open(probs_dir / f"t2ml_lgbm_probs_{split}_{p}.pkl", "rb"))}
            m2 = {w["window_id"]: w for w in pickle.load(open(probs_dir / f"t2ml2_probs_{split}_{p}.pkl", "rb"))}
            cn = official_cnn(p, split, tags)
            rn = None
            if rnn_tags:
                import t2_rnn
                rn = t2_rnn.official_rnn(p, split, tuple(rnn_tags))
            for wid, pc in cn.items():
                parts = [v8[wid]["prob"], m2[wid]["prob"], pc] + ([rn[wid]] if rn is not None else [])
                wts = weights or [1.0] * len(parts)
                pr = sum(a * b for a, b in zip(wts, parts)) / sum(wts)
                T = v8[wid]["T"]
                m = (q.window_id == wid).to_numpy()
                r = q[m]
                s = (r.slot - T - 1).to_numpy()
                assert (s >= 0).all() and (s < HOR).all()
                new = (pr[s, r.link_id.map(pos).to_numpy()] >= theta).astype(int)
                changed += int((new != r.queue_pred.to_numpy()).sum())
                q.loc[m, "queue_pred"] = new
        q[["window_id", "timestamp", "link_id", "queue_pred"]].to_csv(out.format(split=split), index=False)
        print(f"wrote {out.format(split=split)}: {changed} ongoing cells changed", flush=True)


def iou_opt(prob: np.ndarray, floor: float = 0.05) -> np.ndarray:
    """Per-window decision that maximises the ratio of expectations E|S and Y| / E|S or Y| for calibrated
    cell probabilities: take the k most likely cells, k chosen per window (an adaptive threshold)."""
    p = prob.ravel()
    order = np.argsort(-p)
    ps = p[order]
    cs = np.cumsum(ps)
    tot = ps.sum()
    k = np.arange(1, len(ps) + 1)
    ratio = cs / (k + tot - cs)
    best = int(np.argmax(ratio)) + 1 if tot > 0 else 0
    out = np.zeros(p.shape, bool)
    out[order[:best]] = True
    out &= p >= floor
    return out.reshape(prob.shape)
