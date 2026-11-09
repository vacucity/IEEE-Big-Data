"""Evaluate the learned V82 signal beyond its old clipping boundary.

No new fitting or leaderboard-based cell selection: both stage predictions
are frozen for each existing held-out fold. Freeze the bound on parity,
then require the same monthwise gates on half-month and opposite-month.
"""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='2'
import argparse,json,gc
import numpy as np
import pandas as pd
import t1_v82_cmode_interactions as I
from t1_v82_cmode_interactions import V,S,B,N,C,A,F,ROOT,PANELS

OUT=ROOT/'work/results/t1_v83_congest_refine'
CAPS=(1.25,1.5)
PROTOCOL=dict(base='V82 confirmed 0.90333',model='Frozen V81 common-mode + V82 state interaction ridge; evaluate total clipping boundary only',
    caps=CAPS,selection='Select on parity; both months state and conservative total >0, LWR delta <0, >=7 positive-total corridors; maximize minimum total.',
    confirmation='Freeze cap before half-month/opposite-month checks, same gates in both months.',
    scope='Congested non-blackout speed only',caveat=N.PROTOCOL['caveat'],data=V.PROTOCOL['data'])


def transform(v,q,cm,residual,cap):
    delta=np.clip(cm,-1,1)+np.clip(residual,-.5,.5)
    return np.clip(v+np.clip(delta,-cap,cap),C.M.SPEED_MIN,C.M.SPEED_MAX),q.copy()


def score(mode,caps):
    xo,_,z=N.dataset();old=N.heads(xo,z,mode,'old')
    cm=np.load(S.OUT/f'pred_{mode}.npy');residual=np.load(I.OUT/f'pred_{mode}.npy');rows=[]
    for si,s in enumerate(F.SPLITS):
        for pi,p in enumerate(PANELS):
            d=N.load_one(p,s);m=d['cong'];ix=z['owner']==si*10+pi
            va,q0=N.incumbent_free(s,mode,p,pi,d)
            va[m],q0[m]=C.corrected(d['pv'][m],d['pq'][m],old['r'][ix],old['sstar'][ix],A.CFG,N.LAM)
            v0=va.copy();v0[m],_=transform(va[m],q0[m],cm[ix],residual[ix],1.)
            ref,_=S.transform(va[m],q0[m],V.delta(cm[ix],residual[ix],'interaction',1.))
            assert np.array_equal(ref,v0[m])
            rw,_=C.target_weights(d,C.LOC.load_local(p,s,'T'),d['fd']);args=(d['ov'],d['oq'],d['lanes'],d['reg'],rw)
            st=C.LOC.s_state_w(v0,q0,*args);a0=C.M.t3_score(d['C'],d['T'],d['Dd'],v0,q0,d['fd'])
            for cap in caps:
                v1=va.copy();v1[m],_=transform(va[m],q0[m],cm[ix],residual[ix],cap)
                ds=C.LOC.s_state_w(v1,q0,*args)-st
                dt=C.M.t3_delta(a0,C.M.t3_score(d['C'],d['T'],d['Dd'],v1,q0,d['fd']),d['sc'])
                rows.append(dict(split=s,panel=p,mode=mode,cap=cap,dS=ds,dE=dt['dE_est'],dFD=dt['dFD_est'],
                                 dTotal=.35*ds+.05*(dt['dFD_est']-.8*dt['dE_est']),changed=int((v1!=v0).sum())))
            del d;gc.collect()
        print('scored',mode,s,flush=True)
    return pd.DataFrame(rows)


def check(mode):
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'protocol.json').write_text(json.dumps(PROTOCOL,indent=2))
    caps=CAPS if mode=='parity' else (json.loads((OUT/'selection.json').read_text())['cap'],)
    assert None not in caps
    df=score(mode,caps);df.to_csv(OUT/f'checks_{mode}.csv',index=False);rows=[]
    for (s,cap),g in df.groupby(['split','cap']):
        rows.append(dict(split=s,mode=mode,cap=cap,positive_corridors=int((g.dTotal>0).sum()),
                         **{k:C.M.family_mean(g.set_index('panel')[k].to_dict()) for k in ('dS','dE','dFD','dTotal')}))
    sm=pd.DataFrame(rows);sm.to_csv(OUT/f'summary_{mode}.csv',index=False)
    passes=lambda g:bool((g.dS>0).all() and (g.dE<0).all() and (g.dTotal>0).all() and (g.positive_corridors>=7).all())
    if mode=='parity':
        eligible=[c for c,g in sm.groupby('cap') if passes(g)]
        cap=max(eligible,key=lambda c:sm[sm.cap==c].dTotal.min()) if eligible else None
        result=dict(cap=cap,eligible=eligible);path='selection.json'
    else:result=dict(cap=caps[0],passed=passes(sm));path=f'decision_{mode}.json'
    (OUT/path).write_text(json.dumps(result,indent=2));print(sm.to_string(index=False),flush=True);print(result,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['parity','half_month','other_month']);a=ap.parse_args();check(a.mode)
