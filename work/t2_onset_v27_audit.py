"""Train-month-held-out onset decisions, using only the released pre-origin history.

This experiment is deliberately separate from the established V26 implementation.
Visible validation/private windows are noisy proxies, never hidden competition labels.
"""
from __future__ import annotations

import os
for _k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_k] = '2'
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import softmax
import t2_onset as O
from evaluate import T2_PANELS, links
from t2_pseudo import windows, history, truth, HOR

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / 'cache' / 't2_onset_v27'
OUT = ROOT / 'results'

def load_rows(panel, split):
    path = CACHE / f'{panel}_{split}.npz'
    if path.exists():
        with np.load(path) as z:
            return dict(z)
    ws = [w for w in windows(panel, split) if w.cond == 'onset']
    rows = {'X': [], 'Y': [], 'M': [], 'T': [], 'date': [], 'tod': []}
    for w in ws:
        y, m = truth(w)
        yy, mm = y[-1], m[-1]
        t = np.array([float(yy[mem].any()) for mem, _ in O.blocks(panel)])
        rows['X'].append(O.features(panel, history(w)[1].astype(float), w.T + HOR,
                                     w.date[:7] if split == 'train' else None))
        rows['Y'].append(yy)
        rows['M'].append(mm)
        rows['T'].append(t / max(t.sum(), 1))
        rows['date'].append(w.date)
        rows['tod'].append(w.T + HOR)
    rows = {k: np.array(v) for k, v in rows.items()}
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **rows)
    return rows

def take(data, ix):
    return {k: v[ix] for k, v in data.items()}

class Cap:
    def __init__(self, l2=.1, trend=True, temperature=1):
        self.l2, self.trend, self.temperature = l2, trend, temperature
    def fit(self, d):
        X, T = d['X'], d['T']
        keep = T.sum(1) > 0
        X, T = X[keep], T[keep]
        nb = X.shape[1]
        self.c = np.array([np.median(X[T[:, b] > 0, b, 0]) if (T[:, b] > 0).any()
                           else np.median(X[:, b, 0]) for b in range(nb)])
        D, G = X[:, :, 0] - self.c[None], X[:, :, 1] * self.trend
        def loss(th):
            s = th[0] * D + th[1] * G + np.r_[0., th[2:]][None]
            s -= s.max(1, keepdims=True)
            lse = np.log(np.exp(s).sum(1))
            p = np.exp(s - lse[:, None])
            ds = p - T
            reg = self.l2 * len(X) / 100
            grad = np.r_[(ds * D).sum(), (ds * G).sum(), ds[:, 1:].sum(0)]
            grad[2:] += 2 * reg * th[2:]
            return -(T * (s - lse[:, None])).sum() + reg * (th[2:] ** 2).sum(), grad
        self.th = minimize(loss, np.r_[10., 0., np.zeros(nb-1)], jac=True, method='L-BFGS-B').x
        return self
    def predict(self, X):
        s = self.th[0] * (X[:, :, 0] - self.c) + self.th[1] * X[:, :, 1] * self.trend + np.r_[0., self.th[2:]]
        return softmax(s / self.temperature, axis=1)

def shapes(panel, d=None, threshold=.3):
    out = []
    for b, (members, canonical) in enumerate(O.blocks(panel)):
        mask = np.zeros(len(links(panel)), bool)
        if d is None:
            mask[canonical] = True
        else:
            sel = d['T'][:, b] > 0
            freq = np.sum(d['Y'][sel] & d['M'][sel], axis=0) / np.maximum(np.sum(d['M'][sel], axis=0), 1)
            mask[members] = freq[members] >= threshold
            if not mask.any():
                mask[canonical] = True
        out.append(mask)
    return np.array(out)

def score(pred, d):
    pp = pred & d['M']
    yy = d['Y'] & d['M']
    return (pp & yy).sum(1) / np.maximum((pp | yy).sum(1), 1)

def audit():
    allrows = []
    for panel in T2_PANELS:
        ds = {s: load_rows(panel, s) for s in ('train', 'validation', 'private')}
        tr = ds['train']
        months = np.char.substr(tr['date'], 0, 7) if hasattr(np.char, 'substr') else np.array([v[:7] for v in tr['date']])
        fixed = shapes(panel)
        print(panel, 'n', {k: len(v['X']) for k, v in ds.items()}, 'blocks', [np.flatnonzero(s).tolist() for s in fixed], flush=True)
        variants = [('cap_base', .1, True, None)]
        variants += [(f'cap_l2_{l2:g}', l2, True, None) for l2 in (0., 1., 10., 100.)]
        variants += [('cap_no_trend', .1, False, None)]
        variants += [(f'shape_{th:g}', .1, True, th) for th in (.15, .3, .5, .7, .85)]
        for hold in [*np.unique(months), 'all']:
            train = tr if hold == 'all' else take(tr, months != hold)
            evals = {s: ds[s] for s in ('validation', 'private')} if hold == 'all' else {f'train_{hold}': take(tr, months == hold)}
            for name, reg, trend, th in variants:
                model = Cap(reg, trend).fit(train)
                shape = fixed if th is None else shapes(panel, train, th)
                for split, d in evals.items():
                    prob = model.predict(d['X'])
                    pred = shape[prob.argmax(1)]
                    scores = score(pred, d)
                    for date, tod, value in zip(d['date'], d['tod'], scores):
                        allrows.append((panel, split, name, date, tod, value))
        pd.DataFrame(allrows, columns=['panel', 'split', 'variant', 'date', 'tod', 'iou']).to_csv(OUT/'t2_onset_v27_audit_windows.csv', index=False)
    df = pd.DataFrame(allrows, columns=['panel', 'split', 'variant', 'date', 'tod', 'iou'])
    df['group'] = df.split.where(~df.split.str.startswith('train'), 'train_oof')
    p = df.groupby(['group', 'panel', 'variant']).iou.mean().unstack('variant')
    p.to_csv(OUT/'t2_onset_v27_audit_summary.csv')
    print(p.round(4).to_string(), flush=True)

if __name__ == '__main__':
    audit()
