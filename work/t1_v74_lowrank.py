"""Visible-only low-rank corridor factors, incremental to V73.

Task 1 is offline. Hide all evaluation cells before fitting matrix factors.
Fixed ranks 2/4/8 are regression inputs, not a leaderboard search. Regressions
use disjoint days, half months and the opposite month. Never used by Task 2.
"""
from __future__ import annotations
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[k]='2'
import argparse
import json
import time
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.ndimage import gaussian_filter1d
from scipy.linalg import eigh
import t1_v73_global_factor as F
import t1_v73_factor_verify as VFY
from evaluate import CLEAN, PANELS, STEPS

OUT=F.OUT.parent/'t1_v74_lowrank'
RANKS=(2,4,8)


def lowrank(values,mask):
    shape=values.shape; n=shape[-1]
    X=values.reshape(-1,n).astype(np.float32); m=mask.reshape(-1,n)
    count=m.sum(0)
    mu=(X*m).sum(0)/np.maximum(count,1)
    scale=np.maximum(np.sqrt(((X-mu)**2*m).sum(0)/np.maximum(count,1)),1.)
    obs=np.where(m,(X-mu)/scale,0).astype(np.float32)
    fill=obs.copy()
    r=min(8,n)
    for _ in range(12):
        _,B=eigh(fill.T@fill,subset_by_index=(n-r,n-1),check_finite=False)
        B=B[:,::-1].copy()
        scores=fill@B
        pred=scores@B.T
        fill=np.where(m,obs,pred)
    preds=[]
    for rank in RANKS:
        rr=min(rank,r)
        pred=(scores[:,:rr]@B[:,:rr].T)*scale+mu
        preds.append(pred.reshape(shape))
    # Smooth factor scores in each day's time dimension, never across midnight.
    st=scores.reshape(*shape[:2],r)
    for sigma in (1.,4.,12.):
        sm=gaussian_filter1d(st,sigma,axis=1,mode='nearest').reshape(-1,r)
        preds.append((sm@B.T*scale+mu).reshape(shape))
    return preds


def features(panel,split,day,tt,pos,fd,hide):
    path=OUT/f'{"H" if hide else "T"}_{split}_{panel}.npz'
    key=(day*STEPS+tt)*1000+pos
    if path.exists():
        z=dict(np.load(path));assert np.array_equal(key,z['key']);return z
    pf=pq.ParquetFile(CLEAN/panel/f'{split}.parquet')
    vv=[];qq=[];mm=[]
    for dd in range(pf.num_row_groups):
        t=pf.read_row_group(dd,columns=['speed_obs','flow_obs','is_score_eligible'])
        a=lambda c:t.column(c).to_numpy().reshape(STEPS,fd['n'])
        v=a('speed_obs').astype(np.float32);q=a('flow_obs').astype(np.float32)/fd['lanes']
        with np.errstate(invalid='ignore',divide='ignore'):
            m=a('is_score_eligible')&np.isfinite(v)&np.isfinite(q)&(q/np.maximum(v,1)/fd['kcl']<.8)
        ii=np.flatnonzero(day==dd)
        if hide:m[tt[ii],pos[ii]]=False
        assert not m[tt[ii],pos[ii]].any()
        vv.append(np.where(m,v,0));qq.append(np.where(m,q,0));mm.append(m)
    v,q,m=np.stack(vv),np.stack(qq),np.stack(mm)
    z=dict(key=key,cnt=m.sum(2)[day,tt].astype(np.float32))
    for name,x in [('v',v),('q',q)]:
        pred=lowrank(x,m)
        z[name]=np.column_stack([p[day,tt,pos] for p in pred]).astype(np.float32)
    np.savez_compressed(path,**z)
    return z


def design(feat,base,lh,kind):
    lim=12 if kind=='v' else 200
    delta=np.clip(feat[kind]-base[:,None],-lim,lim)
    mods=[np.ones(len(base)),(lh['db']==1),(lh['da']==1),1/(1+feat['cnt']/10)]
    return np.column_stack([delta*x[:,None] for x in mods]+[np.ones(len(base))])


def load(panel,split):
    d,lh=F.LOC.load(panel,split);c=d['C']
    z=np.load(F.OUT/f'check_global_clip1_{split}_{panel}.npz')
    feat=features(panel,split,c.day.to_numpy(),c.t.to_numpy(),c.pos.to_numpy(),d['fd'],True)
    return d,lh,z,feat


def run():
    OUT.mkdir(parents=True,exist_ok=True);t0=time.time()
    grams={};beta={};rows=[]
    for split in F.SPLITS:
        for panel in PANELS:
            d,lh,z,f=load(panel,split)
            lanes=d['fd']['lanes'][d['pos']]
            for kind in ('v','q'):
                base=z['v1'] if kind=='v' else z['q']/lanes
                y=(d['ov'] if kind=='v' else d['oq']/lanes)-base
                x=design(f,base,lh,kind);m=d['free']&np.isfinite(y)
                for mode in ('parity','half_month','all'):
                    fold=d['par'] if mode=='parity' else (d['day']>=15).astype(int)
                    for a in (0,1) if mode!='all' else (0,):
                        k=m&((fold==a) if mode!='all' else True)
                        gg,bb=F.gram(x[k],y[k]);key=split,kind,mode,a
                        if key not in grams:grams[key]=(gg,bb)
                        else:grams[key]=(grams[key][0]+gg,grams[key][1]+bb)
            print('factors',split,panel,f'{time.time()-t0:.0f}s',flush=True)
    for k,(g,b) in grams.items():beta[k]=F.solve(g,b)
    np.savez(OUT/'coefficients.npz',**{'__'.join(map(str,k)):b for k,b in beta.items()})
    for split in F.SPLITS:
        for panel in PANELS:
            d,lh,z,f=load(panel,split);lanes=d['fd']['lanes'][d['pos']]
            m=d['free'];v0=z['v1'];q0=z['q'];xs={k:design(f,v0 if k=='v' else q0/lanes,lh,k) for k in ('v','q')}
            lt=F.LOC.load_local(panel,split,'T');w,unc=F.LOC.target_weights(d,lh,lt,d['fd'])
            st=(d['ov'],d['oq'],lanes,d['reg'],w)
            s0=F.LOC.s_state_w(v0,q0,*st);a0=F.M.t3_score(d['C'],d['T'],d['Dd'],v0,q0,d['fd'])
            for mode in ('parity','half_month','other_month'):
                delta={k:np.zeros(len(m)) for k in ('v','q')}
                for kind in ('v','q'):
                    if mode=='other_month':
                        other=F.SPLITS[1-F.SPLITS.index(split)]
                        delta[kind][m]=xs[kind][m]@beta[other,kind,'all',0]
                    else:
                        fold=d['par'] if mode=='parity' else (d['day']>=15).astype(int)
                        for a in (0,1):
                            k=m&(fold!=a);delta[kind][k]=xs[kind][k]@beta[split,kind,mode,a]
                # Fixed limits prevent large corrections of ambiguous/rare masks.
                v1=VFY.clamp(panel,d['pos'],m,v0+np.clip(delta['v'],-1,1),q0)
                q1=np.where(m,np.maximum(q0+lanes*np.clip(delta['q'],-30,30),0),q0)
                for cand,v,q in [('speed',v1,q0),('flow',v0,q1),('both',v1,q1)]:
                    ds=F.LOC.s_state_w(v,q,*st)-s0
                    a1=F.M.t3_score(d['C'],d['T'],d['Dd'],v,q,d['fd'])
                    dt=F.M.t3_delta(a0,a1,d['sc'])
                    rows.append(dict(split=split,panel=panel,mode=mode,cand=cand,dS_rw=ds,dE=dt['dE_est'],
                                     dFD=dt['dFD_est'],dTotal40=.35*ds+.05*(dt['dFD_est']-.8*dt['dE_est'])))
            print('verified',split,panel,f'{time.time()-t0:.0f}s',flush=True)
    r=pd.DataFrame(rows);r.to_csv(OUT/'checks.csv',index=False)
    sm=[]
    for (s,mode,cand),g in r.groupby(['split','mode','cand']):
        sm.append(dict(split=s,mode=mode,cand=cand,**{k:F.M.family_mean(g.set_index('panel')[k].to_dict()) for k in ('dS_rw','dE','dFD','dTotal40')}))
    sm=pd.DataFrame(sm);sm.to_csv(OUT/'summary.csv',index=False)
    print(sm.to_string(index=False),flush=True)
    passed=[c for c,g in sm.groupby('cand') if (g.dS_rw>0).all() and (g.dTotal40>0).all()]
    # A beneficial speed correction must not conceal a failing flow component.
    if 'both' in passed and not {'speed','flow'}.issubset(passed):
        passed.remove('both')
    chosen=max(passed,key=lambda c:sm[sm.cand==c].dTotal40.min()) if passed else None
    (OUT/'decision.json').write_text(json.dumps(dict(chosen=chosen,gate='positive weighted state and conservative total in all 6 month/fold checks; joint requires each component to pass separately',public_lb=None),indent=2))
    print('chosen',chosen,flush=True)


if __name__=='__main__':run()
