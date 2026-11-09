#!/usr/bin/env python
"""Task 2 queue_onset as a choice between a corridor's recurrent bottleneck blocks.

What the data shows
  * An onset is a whole bottleneck block switching to congestion in one five-minute step (e.g. D7_I10_E links
    4-10 drop from ~113 to ~37 km/h together), with no speed precursor in the hour before.
  * It happens when the 30-minute demand step pushes the local flow over that bottleneck's capacity, so the
    block that breaks first is the one whose flow at T sits closest to its capacity.
  * Validation and private were drawn with their own seeds: the corridor-wide demand shape differs (D12_I5_S
    in validation carries ~500 vph less at links 43-45 than at 10-15, against ~150 in train), so a different
    block breaks first. A model keyed on time of day keeps choosing the train habit; flows at T carry the
    month's own answer.

Model (per panel): conditional logit over the blocks,
    score_b = a_b + beta * log q_b(T) + gamma * [log q_b(T) - log q_b(T-45..T-30)] + delta * log prior_b(tod)
fitted on the train onset windows built by t2_pseudo (masked-layer features, as in validation/private).
The block intercepts a_b absorb fixed capacity differences. The forecast is the canonical link set of the
best block at T+30, nothing at T+5..T+25 (the onset step is organizer-confirmed).

Information contract: features come only from the 60-minute history before T; train supplies the blocks,
their canonical links and the model coefficients.
"""
from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import RELEASE, T2_PANELS, links  # noqa: E402
from t2_pseudo import HIST, HOR, Win, evaluate, history, summary, truth, windows  # noqa: E402
from task2_rules import onset_events  # noqa: E402

MIN_SHARE, GAP, CANON = 0.03, 2, 0.3
SIGMA_TOD = 12.0          # 1 hour kernel for the time-of-day prior
L2 = 0.01


@lru_cache(maxsize=None)
def blocks(panel: str) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    """(member links, canonical links) per block, from the train onset events.

    Frequent onset links are joined when they sit within GAP positions and break together (co-occur in at
    least 30% of the rarer link's events), so D7_I210_W's 3-4 and 6-7 stay apart while 29-31,33 stay one."""
    ev = onset_events(panel)
    n_ev = ev.event.nunique()
    freq = ev.groupby("pos").event.nunique()
    keep = np.sort(freq[freq >= max(3, MIN_SHARE * n_ev)].index.to_numpy())
    sets = {p: set(ev.event[ev.pos == p]) for p in keep}
    parent = {p: p for p in keep}

    def root(p):
        while parent[p] != p:
            p = parent[p]
        return p
    for i, a in enumerate(keep):
        for b in keep[i + 1:]:
            if b - a > GAP + 1:
                break
            if len(sets[a] & sets[b]) >= 0.3 * min(len(sets[a]), len(sets[b])):
                parent[root(b)] = root(a)
    comps: dict = {}
    for p in keep:
        comps.setdefault(root(p), []).append(p)
    out = []
    for g in sorted(comps.values()):
        members = np.array(sorted(g))
        evs = set().union(*(sets[p] for p in g))
        f = ev[ev.event.isin(evs) & ev.pos.isin(members)].groupby("pos").event.nunique() / len(evs)
        canon = np.sort(f[f >= CANON].index.to_numpy())
        out.append((members, canon if len(canon) else members))
    return tuple(out)


@lru_cache(maxsize=None)
def tod_prior(panel: str, skip_month: str | None = None) -> np.ndarray:
    """(n_blocks, 288) share of train onset events at each tod that involve each block (kernel smoothed).
    skip_month leaves that month out, so a train window never sees its own event."""
    ev = onset_events(panel)
    if skip_month is not None:
        ev = ev[ev.date.str[:7] != skip_month]
    bl = blocks(panel)
    e = ev.drop_duplicates("event")[["event", "tod"]].set_index("event").tod
    grid = np.arange(288)
    K = np.exp(-0.5 * ((grid[None, :] - e.to_numpy()[:, None]) / SIGMA_TOD) ** 2)      # events x tod
    tot = K.sum(0) + 1e-9
    out = []
    for members, _ in bl:
        hit = ev[ev.pos.isin(members)].event.unique()
        out.append((K[np.isin(e.index.to_numpy(), hit)].sum(0) + 0.5) / (tot + 0.5 * len(bl)))
    return np.array(out)


def block_flows(Q: np.ndarray, members: np.ndarray, steps: slice) -> float:
    lo, hi = max(members.min() - 2, 0), min(members.max() + 3, Q.shape[1])
    x = Q[steps, lo:hi]
    return float(np.nanmedian(x)) if np.isfinite(x).any() else np.nan


def features(panel: str, Q: np.ndarray, tod_target: int, skip_month: str | None = None) -> np.ndarray:
    """(n_blocks, 3): log flow at T, its change against T-60..T-30, log tod prior."""
    bl = blocks(panel)
    pr = tod_prior(panel, skip_month)[:, min(tod_target, 287)]
    rows = []
    for k, (members, _) in enumerate(bl):
        q0 = block_flows(Q, members, slice(HIST - 3, HIST))
        if not np.isfinite(q0):
            q0 = block_flows(Q, members, slice(HIST - 6, HIST))
        q1 = block_flows(Q, members, slice(0, 6))
        rows.append([np.log(max(q0, 100.0)) if np.isfinite(q0) else np.nan,
                     np.log(max(q0, 100.0)) - np.log(max(q1, 100.0)) if np.isfinite(q0) and np.isfinite(q1) else 0.0,
                     np.log(pr[k])])
    X = np.array(rows, float)
    # a block with no visible flow gets the mean of the others (no evidence either way)
    if np.isnan(X[:, 0]).any():
        X[np.isnan(X[:, 0]), 0] = np.nanmean(X[:, 0]) if np.isfinite(X[:, 0]).any() else np.log(3000.0)
    return X


def target(panel: str, w: Win) -> np.ndarray | None:
    y, m = truth(w)
    yl = y[HOR - 1] & m[HOR - 1]
    t = np.array([float(yl[members].any()) for members, _ in blocks(panel)])
    return t / t.sum() if t.sum() else None


class Model:
    """Conditional logit over blocks. use = (relative log flow, flow trend, log tod prior, own absolute flow)."""

    def __init__(self, panel: str, use=(True, True, True, True)):
        self.panel, self.use = panel, np.array(use, bool)
        self.nb = len(blocks(panel))
        self.theta = np.zeros(2 * self.nb + 3)

    def prep(self, X: np.ndarray, fit: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """Relative part: each feature centred across the blocks of a window and scaled by its train spread.
        Absolute part: each block's own log flow, standardised over the panel."""
        Xc = X - X.mean(-2, keepdims=True)
        if fit:
            self.sd = Xc.reshape(-1, X.shape[-1]).std(0) + 1e-6
            self.mu_a, self.sd_a = X[..., 0].mean(), X[..., 0].std() + 1e-6
        return Xc / self.sd, (X[..., 0] - self.mu_a) / self.sd_a

    def _scores(self, th, Xc, Xa):
        nb = self.nb
        a, beta, coef = th[:nb], th[nb:2 * nb] * self.use[3], th[2 * nb:] * self.use[:3]
        return a + Xc @ coef + beta * Xa

    def fit(self, data: list[tuple[np.ndarray, np.ndarray]]) -> "Model":
        if self.nb == 1:
            return self
        Xc, Xa = self.prep(np.stack([d[0] for d in data]), fit=True)   # (N, nb, 3), (N, nb)
        Ts = np.stack([d[1] for d in data])                              # (N, nb)
        nb = self.nb

        def nll(th):
            s = self._scores(th, Xc, Xa)
            s = s - s.max(1, keepdims=True)
            lse = np.log(np.exp(s).sum(1))
            ll = (Ts * (s - lse[:, None])).sum()
            g_s = Ts - np.exp(s - lse[:, None])
            ga = g_s.sum(0)
            gb = (g_s * Xa).sum(0) * self.use[3]
            gc = np.einsum("nb,nbk->k", g_s, Xc) * self.use[:3]
            pen = np.concatenate([np.full(nb, 0.01), np.full(nb, L2), np.full(3, L2)])
            return -ll + 0.5 * (pen * th ** 2).sum(), -np.concatenate([ga, gb, gc]) + pen * th

        self.theta = minimize(nll, np.zeros(2 * nb + 3), jac=True, method="L-BFGS-B").x
        return self

    def predict_block(self, X: np.ndarray) -> np.ndarray:
        if self.nb == 1:
            return np.ones(1)
        Xc, Xa = self.prep(X[None])
        s = self._scores(self.theta, Xc, Xa)[0]
        p = np.exp(s - s.max())
        return p / p.sum()


def train_data(panel: str) -> list[tuple[np.ndarray, np.ndarray]]:
    out = []
    for w in windows(panel, "train"):
        if w.cond != "onset":
            continue
        t = target(panel, w)
        if t is None:
            continue
        out.append((features(panel, history(w)[1].astype(float), w.T + HOR, w.date[:7]), t))
    return out


@lru_cache(maxsize=None)
def fitted(panel: str, use=(True, True, True, True)) -> Model:
    return Model(panel, use).fit(train_data(panel))


def predict(w: Win, use=(True, True, True, True)) -> np.ndarray:
    n = len(links(w.panel))
    m = fitted(w.panel, use)
    p = m.predict_block(features(w.panel, history(w)[1].astype(float), w.T + HOR))
    out = np.zeros((HOR, n), bool)
    out[HOR - 1, blocks(w.panel)[int(np.argmax(p))][1]] = True
    return out


# ---------------------------------------------------------------------------
# official windows
# ---------------------------------------------------------------------------
FLOW_USE = (True, True, False, True)          # relative flow, trend, own absolute flow; no time-of-day prior
# Corridors with competing bottlenecks where the flow choice beats the per-link model on both the validation
# and the private pseudo windows; single-block corridors give the same answer either way, and D7_I210_W keeps
# the per-link model (its 6-7 block breaks at ~05:30 on low flow, which only the time of day separates).
FLOW_PANELS = ("D12_I5_N", "D12_I5_S", "D7_I405_S", "D7_I10_W")


def official_history_Q(panel: str, split: str) -> dict[str, tuple[np.ndarray, int]]:
    """window_id -> ((12, n) eligible flow grid of the published history, forecast-origin tod)."""
    from evaluate import RELEASE, slot_of
    n = len(links(panel))
    base = RELEASE / "task2" / panel / split
    win = pd.read_csv(base / "window_index.csv")
    hist = pd.read_parquet(base / "window_history.parquet")
    hist["pos"] = hist.link_id.map(dict(zip(links(panel).link_id, range(n))))
    hist["slot"] = slot_of(hist.timestamp)
    out = {}
    for w in win.itertuples(index=False):
        if w.condition != "queue_onset":
            continue
        T = int(slot_of(pd.Series([w.forecast_origin]))[0])
        h = hist[hist.window_id == w.window_id]
        ok = (h.is_score_eligible.astype(bool) & h.flow_vph.notna()).to_numpy()
        si = (h.slot - (T - HIST)).to_numpy()
        m = ok & (si >= 0) & (si < HIST)
        Q = np.full((HIST, n), np.nan)
        Q[si[m], h.pos.to_numpy()[m]] = h.flow_vph.to_numpy()[m]
        origin = pd.Timestamp(w.forecast_origin)
        out[w.window_id] = (Q, origin.hour * 12 + origin.minute // 5)
    return out


def official_onset(panel: str, split: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """window_id -> (block probabilities, links predicted at T+30) from the flow model."""
    m = fitted(panel, FLOW_USE)
    out = {}
    for wid, (Q, tod) in official_history_Q(panel, split).items():
        p = m.predict_block(features(panel, Q, tod + HOR))
        out[wid] = (p, blocks(panel)[int(np.argmax(p))][1])
    return out


def cmd_apply(base: str, out: str, split: str, panels=FLOW_PANELS) -> None:
    """Replace the onset windows of `panels` in a queue file with the flow model's block."""
    from evaluate import slot_of
    q = pd.read_csv(base)
    q["panel"] = q.window_id.str.extract(r"^Q2_(.+)_(?:validation|private|train)_\d+$")[0]
    changed = 0
    for p in panels:
        pos = dict(zip(links(p).link_id, range(len(links(p)))))
        win = pd.read_csv(RELEASE / "task2" / p / split / "window_index.csv").set_index("window_id")
        for wid, (_, lk) in official_onset(p, split).items():
            T = int(slot_of(pd.Series([win.loc[wid, "forecast_origin"]]))[0])
            m = (q.window_id == wid).to_numpy()
            rows = q[m]
            new = (slot_of(rows.timestamp) == T + HOR) & rows.link_id.map(pos).isin(set(lk.tolist())).to_numpy()
            changed += int((rows.queue_pred.to_numpy() != new.astype(int)).sum())
            q.loc[m, "queue_pred"] = new.astype(int)
    q[["window_id", "timestamp", "link_id", "queue_pred"]].to_csv(out, index=False)
    print(f"wrote {out}: {len(q):,} rows, {changed} cells changed in {len(panels)} panels")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "apply":
        # python work/t2_onset.py apply <split> <base.csv> <out.csv>
        cmd_apply(sys.argv[3], sys.argv[4], sys.argv[2])
        sys.exit(0)
    for p in T2_PANELS:
        print(p, [(int(m.min()), int(m.max()), list(c)) for m, c in blocks(p)])
    variants = {"rel+trend+prior+abs": (True, True, True, True), "rel+trend+abs": (True, True, False, True),
                "rel+trend": (True, True, False, False), "rel+trend+prior": (True, True, True, False),
                "prior+abs": (False, False, True, True)}
    for name, use in variants.items():
        df = evaluate(lambda w: predict(w, use), splits=("train", "validation", "private"), conds=("onset",))
        print(f"{name:22s}", summary(df).onset.to_dict(), " early:", summary(df[df.day < 12]).onset.to_dict())
        print(df.groupby(["split", "panel"]).iou.mean().unstack(0).round(3).T.to_string())
