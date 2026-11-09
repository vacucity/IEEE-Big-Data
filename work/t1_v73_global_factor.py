"""Residual matrix-factor estimates for Task 1, evaluated against V69 (V72 state).

Only released visible Task-1 observations enter factor estimates; every artificial
holdout is removed before estimating ANY factor. Task 1 is offline. Queue models
are not read. The speed correction is evaluated on parity folds and by fitting on
the other month. Fixed ridge design; no hyperparameter search.
"""
from __future__ import annotations
import os
for _k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_k] = '4'
import argparse
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.ndimage import gaussian_filter1d
import t1_v52_monthbias as M
import t1_v59_local as LOC
import t1_v66_cmode as CM
import t1_v68_cmgbm as G
from evaluate import CLEAN, PANELS, STEPS

OUT = Path(__file__).resolve().parent / 'results/t1_v73_global_factor'
SPLITS = ('validation', 'private')
SIGMAS = (1., 2., 4., 8., 16.)


def baseline(split):
    path = OUT / f'baseline_{split}.npz'
    if path.exists():
        return dict(np.load(path))
    import lightgbm as lgb
    z64 = np.load(CM.Z64)
    gr = np.load(CM.OUT / 'grams.npz')
    B = {a: CM.solve(gr[f'G_{split}_{a}_0'] + gr[f'G_{split}_{a}_1']) for a in (0, 1)}
    off, per = 0, {}
    for pi, p in enumerate(PANELS):
        d, lh = LOC.load(p, split)
        v64, q64, off = CM.hide_v64(p, split, SPLITS.index(split), d, off, z64)
        F = CM.hide_feats(p, split, d)
        m = d['free'] & (F['cnt'] >= CM.MIN_CNT)
        v66 = v64.copy()
        for a in (0, 1):
            k = np.flatnonzero((d['par'] != a) & m)
            v66[k] = np.clip(v64[k] + CM.cm_pred(F, lh, k, *B[a]), M.SPEED_MIN, M.SPEED_MAX)
        per[p] = dict(d=d, v66=v66, q=q64, m=m, X=G.feats(p, d, lh, F, v64, pi))
    out = {}
    for a in (0, 1):
        xx = np.concatenate([x['X'][(x['d']['par'] == a) & x['m']] for x in per.values()])
        yy = np.concatenate([(x['d']['ov'] - x['v66'])[(x['d']['par'] == a) & x['m']] for x in per.values()])
        ok = np.isfinite(yy)
        mdl = lgb.train(G.PRM, lgb.Dataset(xx[ok], yy[ok], categorical_feature=[xx.shape[1]-1]), G.ROUNDS)
        for p, x in per.items():
            out.setdefault(p+'_v', x['v66'].copy())
            k = (x['d']['par'] != a) & x['m']
            out[p+'_v'][k] += mdl.predict(x['X'][k], num_threads=4)
        print('baseline', split, a, flush=True)
    caps = json.loads((CM.OUT / 'cap_links.json').read_text())
    from evaluate import links
    for p, x in per.items():
        v = np.clip(out[p+'_v'], M.SPEED_MIN, M.SPEED_MAX)
        lid = links(p).link_id.to_numpy()[x['d']['pos']]
        hit = np.isin(lid, caps.get(p, {}).get('links', [])) & x['m']
        v[hit] = np.minimum(v[hit], CM.CAP_VALUE)
        out[p+'_v'], out[p+'_q'] = v, x['q']
    np.savez_compressed(path, **out)
    return out


def factors(p, split, day, tt, pos, fd, hide):
    """Monthly and daily additive factors plus temporal residual smoothers."""
    path = OUT / f'{"H" if hide else "T"}_{split}_{p}.npz'
    if path.exists():
        z = dict(np.load(path))
        assert np.array_equal(z['key'], (day*STEPS+tt)*1000+pos)
        return z
    pf = pq.ParquetFile(CLEAN / p / f'{split}.parquet')
    vv, mm = [], []
    for dd in range(pf.num_row_groups):
        tb = pf.read_row_group(dd, columns=['is_score_eligible', 'speed_obs', 'flow_obs'])
        get = lambda c: tb.column(c).to_numpy().reshape(STEPS, fd['n'])
        v, q = get('speed_obs').astype(float), get('flow_obs').astype(float)
        with np.errstate(invalid='ignore', divide='ignore'):
            ok = get('is_score_eligible') & np.isfinite(v) & np.isfinite(q) & (q/fd['lanes']/np.maximum(v,1)/fd['kcl'] < .8)
        ii = np.flatnonzero(day == dd)
        if hide:
            ok[tt[ii], pos[ii]] = False
        assert not ok[tt[ii], pos[ii]].any(), ('target visible', split, p, dd)
        vv.append(np.where(ok, v, 0))
        mm.append(ok)
    v, m = np.stack(vv), np.stack(mm)
    cnt = m.sum(2)
    nc = m.sum((0,1))
    aa = np.where(nc > 0, v.sum((0,1))/np.maximum(nc,1), fd['ffs'])
    for _ in range(8):
        bb = np.sum(np.where(m, v-aa, 0), 2)/(cnt+2)
        aa = np.sum(np.where(m, v-bb[:,:,None], 0), (0,1))/np.maximum(nc,1)
    month_pred = aa[None,None,:]+bb[:,:,None]
    nd = m.sum(1)
    ad = np.zeros_like(nd, dtype=float)
    for _ in range(6):
        bd = np.sum(np.where(m, v-aa-ad[:,None,:], 0),2)/(cnt+2)
        ad = np.sum(np.where(m,v-aa-bd[:,:,None],0),1)/(nd+24)
    daily_pred = aa[None,None,:]+ad[:,None,:]+bd[:,:,None]
    vals = [month_pred[day,tt,pos], daily_pred[day,tt,pos]]
    rr = np.where(m,v-daily_pred,0)
    for sig in SIGMAS:
        num = gaussian_filter1d(rr,sig,axis=1,mode='constant')
        den = gaussian_filter1d(m.astype(float),sig,axis=1,mode='constant')
        pr = daily_pred + num/np.maximum(den,.02)
        vals.append(pr[day,tt,pos])
    z = dict(pred=np.column_stack(vals).astype(np.float32), cnt=cnt[day,tt].astype(np.float32),
             key=(day*STEPS+tt)*1000+pos)
    np.savez_compressed(path,**z)
    return z


def design(f, v0, local):
    delta = np.clip(f['pred']-v0[:,None],-12,12)
    mods = [np.ones(len(v0)), (local['db']==1), (local['da']==1), 1/(1+f['cnt']/10)]
    return np.column_stack([delta*x[:,None] for x in mods]+[np.ones(len(v0))])


def gram(x, y):
    return x.T@x, x.T@y


def solve(g, b):
    return np.linalg.solve(g+np.diag(np.maximum(np.diag(g),1)*.002),b)


def probe():
    OUT.mkdir(parents=True,exist_ok=True)
    per, grams = {}, {}
    t0=time.time()
    for split in SPLITS:
        zz=baseline(split)
        for p in PANELS:
            d,lh=LOC.load(p,split)
            c=d['C']
            f=factors(p,split,c.day.to_numpy(),c.t.to_numpy(),c.pos.to_numpy(),d['fd'],True)
            v0,q0=zz[p+'_v'],zz[p+'_q']
            x=design(f,v0,lh)
            y=d['ov']-v0
            m=d['free'] & np.isfinite(y)
            per[split,p]=dict(d=d,x=x,v=v0,q=q0,m=m,y=y)
            for a in (0,1):
                k=m & (d['par']==a)
                gg,bb=gram(x[k],y[k])
                if (split,a) not in grams:
                    grams[split,a]=(gg,bb)
                else:
                    g1,b1=grams[split,a]
                    grams[split,a]=(g1+gg,b1+bb)
            print('features',split,p,f'{time.time()-t0:.0f}s',flush=True)
    rows=[]
    for split in SPLITS:
        g0,b0=grams[split,0];g1,b1=grams[split,1]
        np.save(OUT/f'beta_{split}.npy',solve(g0+g1,b0+b1))
        for mode in ('parity','other_month'):
            other=SPLITS[1-SPLITS.index(split)]
            for p in PANELS:
                x=per[split,p];d=x['d'];delta=np.zeros(len(x['v']))
                if mode=='parity':
                    for a in (0,1):
                        k=x['m'] & (d['par']!=a)
                        delta[k]=x['x'][k]@solve(*grams[split,a])
                else:
                    ga,ba=grams[other,0];gb,bb=grams[other,1]
                    delta[x['m']]=x['x'][x['m']]@solve(ga+gb,ba+bb)
                vv=np.clip(x['v']+delta,M.SPEED_MIN,M.SPEED_MAX)
                s0=M.state_score(d['C'],x['v'],x['q'],d['fd'])
                s1=M.state_score(d['C'],vv,x['q'],d['fd'])
                a0=M.t3_score(d['C'],d['T'],d['Dd'],x['v'],x['q'],d['fd'])
                a1=M.t3_score(d['C'],d['T'],d['Dd'],vv,x['q'],d['fd'])
                dt=M.t3_delta(a0,a1,d['sc']);ds=s1['S']-s0['S']
                row=dict(split=split,mode=mode,panel=p,dS=ds,dFD=dt['dFD_est'],dE=dt['dE_est'],
                         dTotal40=.35*ds+.05*(dt['dFD_est']-.8*dt['dE_est']),
                         rms=float(np.sqrt(np.mean(delta[x['m']]**2))))
                rows.append(row)
            dd=pd.DataFrame([r for r in rows if r['split']==split and r['mode']==mode])
            means={k:M.family_mean(dd.set_index('panel')[k].to_dict()) for k in ('dS','dE','dTotal40')}
            print(split,mode,means,'positive',int((dd.dS>0).sum()),flush=True)
    pd.DataFrame(rows).to_csv(OUT/'probe.csv',index=False)


if __name__=='__main__':
    probe()
