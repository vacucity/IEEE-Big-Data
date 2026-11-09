"""State-dependent common-mode residual without independent state offsets.

The increment is identically zero if the observed common-mode signal is zero.
Ridge interactions vary signal coefficients with incumbent congestion level.
"""
import argparse,json
import numpy as np
import pandas as pd
import t1_v82_congest_residual as V
from t1_v82_congest_residual import S,B,N,C,A,F,ROOT,PANELS

OUT=ROOT/'work/results/t1_v82_cmode_interactions'
ALPHAS=(.5,1.)
RIDGE=.01
PROTOCOL=dict(base='V81 confirmed 0.90329',model='90-feature ridge residual: common-mode terms, spatial bands, and common-mode x congestion interactions; no intercept',
    alpha_grid=ALPHAS,ridge=RIDGE,extra_cap=.5,total_speed_cap=1.,
    invariant='Zero observed common-mode and band signals imply exactly zero new correction',
    selection=V.PROTOCOL['selection'],confirmation=V.PROTOCOL['confirmation'],targets=V.PROTOCOL['targets'],
    scope=V.PROTOCOL['scope'],data=V.PROTOCOL['data'],caveat=V.PROTOCOL['caveat'])

def features(xo,cmx):
    wide=cmx[:,:24];v=np.clip(xo[:,0],0,1.5)-.75;k=np.clip(xo[:,2],0,3)-1.
    x=np.column_stack([wide,cmx[:,25:43],wide*v[:,None],wide*k[:,None]])
    assert x.shape[1]==90 and np.isfinite(x).all();return x

def fit(x,y,w):
    w=w/w.mean();g=x.T@(w[:,None]*x);b=x.T@(w*y)
    return np.linalg.solve(g+np.eye(len(g))*(RIDGE*np.trace(g)/len(g)),b)

def predictions(mode):
    path=OUT/f'pred_{mode}.npy'
    if path.exists():return np.load(path)
    xo,xn,z=N.dataset();d=B.data();x=features(xo,d['X']);pred=np.zeros(len(xo));audit=[]
    for a in (0,1):
        tr=np.flatnonzero(z[mode]==a);te=np.flatnonzero(z[mode]!=a);beta0=np.load(S.OUT/f'beta_{mode}_{a}.npy')
        cm=d['X'][tr,:25]@beta0;v,q=V.teacher(xo,d,tr,mode,a);v81,_=S.transform(v,q,cm)
        beta=fit(x[tr],np.clip(d['ov'][tr]-v81,-5,5),V.weights(d,z,tr,mode,a));np.save(OUT/f'beta_{mode}_{a}.npy',beta);pred[te]=x[te]@beta
        err=float(np.max(np.abs(d['X'][te,:25]@beta0-np.load(S.OUT/f'pred_{mode}.npy')[te])));assert err<1e-12
        audit.append(dict(fold=a,v81_replay_max_error=err));print('interaction fit',mode,a,flush=True)
    np.save(path,pred);(OUT/f'baseline_replay_{mode}.json').write_text(json.dumps(audit,indent=2));return pred

def check(mode):
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'protocol.json').write_text(json.dumps(PROTOCOL,indent=2))
    if mode=='parity':alphas=ALPHAS
    else:
        a=json.loads((OUT/'selection.json').read_text())['alpha'];assert a is not None;alphas=(a,)
    df=pd.DataFrame(V.score(mode,{'interaction':predictions(mode)},alphas));df.to_csv(OUT/f'checks_{mode}.csv',index=False);sm=[]
    for (s,alpha),g in df.groupby(['split','alpha']):
        sm.append(dict(split=s,mode=mode,alpha=alpha,positive_corridors=int((g.dTotal>0).sum()),
            **{k:C.M.family_mean(g.set_index('panel')[k].to_dict()) for k in ('dS','dE','dFD','dTotal')}))
    sm=pd.DataFrame(sm);sm.to_csv(OUT/f'summary_{mode}.csv',index=False)
    passes=lambda g:bool((g.dS>0).all() and (g.dE<0).all() and (g.dTotal>0).all() and (g.positive_corridors>=7).all())
    if mode=='parity':
        eligible=[a for a,g in sm.groupby('alpha') if passes(g)];alpha=max(eligible,key=lambda a:sm[sm.alpha==a].dTotal.min()) if eligible else None
        result=dict(alpha=alpha,eligible=eligible);path='selection.json'
    else:result=dict(alpha=alphas[0],passed=passes(sm));path=f'decision_{mode}.json'
    (OUT/path).write_text(json.dumps(result,indent=2));print(sm.to_string(index=False),flush=True);print(result,flush=True)

def full():
    choice=json.loads((OUT/'selection.json').read_text());assert choice['alpha'] is not None
    for mode in ('half_month','other_month'):assert json.loads((OUT/f'decision_{mode}.json').read_text())['passed']
    path=OUT/'beta_full.npy'
    if path.exists():return np.load(path)
    xo,xn,z=N.dataset();d=B.data();exact=N.old_inputs_exact();idx=np.arange(len(xo));v,q=V.teacher(exact,d,idx);beta0=S.full();cm=d['X'][:,:25]@beta0
    v81,_=S.transform(v,q,cm);beta=fit(features(exact,d['X']),np.clip(d['ov']-v81,-5,5),V.weights(d,z,idx));np.save(path,beta);return beta

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('cmd',choices=['parity','half_month','other_month','full']);a=ap.parse_args()
    full() if a.cmd=='full' else check(a.cmd)
