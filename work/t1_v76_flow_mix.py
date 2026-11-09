"""Test a bounded nonlinear flow residual separately from the speed candidate.

Same visible-only context, all six day/month folds, fixed 15 vph/lane bound.
Reject early if either month's parity result fails; no queue inputs or changes.
"""
from __future__ import annotations
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[k]='2'
import gc,json,time
import numpy as np
import pandas as pd
import lightgbm as lgb
import t1_v76_context_mix as G
import t1_v73_global_factor as F
import t1_v73_factor_verify as VFY
from evaluate import PANELS

OUT=G.OUT.parent/'t1_v76_flow_mix'
PRM=dict(G.B.PRM,num_threads=2,max_bin=127,seed=76176)
ROUNDS=350
LIMIT=15.


def labels(own):
    path=OUT/'flow_labels.npy'
    if path.exists():return np.load(path)
    y=np.empty(len(own),np.float32)
    for si,s in enumerate(F.SPLITS):
        z=F.baseline(s)
        for pi,p in enumerate(PANELS):
            d,lh=F.LOC.load(p,s);m=d['free'];lane=d['fd']['lanes'][d['pos']]
            y[own==si*10+pi]=((d['oq']-z[p+'_q'])/lane/100)[m]
            del d,lh;gc.collect()
    np.save(path,y);return y


def score(own,pred,mode):
    rows=[];ref=np.load(G.OUT/f'baseline_{mode}.npy')
    for si,s in enumerate(F.SPLITS):
        z=F.baseline(s)
        for pi,p in enumerate(PANELS):
            d,lh=F.LOC.load(p,s);m=d['free'];idx=own==si*10+pi;fd=d['fd'];lane=fd['lanes'][d['pos']]
            rw,_=F.LOC.target_weights(d,lh,F.LOC.load_local(p,s,'T'),fd)
            dv=np.zeros(len(m));dv[m]=np.clip(ref[idx],-1,1)
            v=VFY.clamp(p,d['pos'],m,z[p+'_v']+dv,z[p+'_q']);q0=z[p+'_q']
            dq=np.zeros(len(m));dq[m]=np.clip(pred[idx]*100,-LIMIT,LIMIT)*lane[m]
            q1=np.where(m,np.maximum(0,np.minimum(q0+dq,v*lane*fd['kjl'][d['pos']])),q0)
            st=(d['ov'],d['oq'],lane,d['reg'],rw)
            ds=F.LOC.s_state_w(v,q1,*st)-F.LOC.s_state_w(v,q0,*st)
            a0=F.M.t3_score(d['C'],d['T'],d['Dd'],v,q0,fd);a1=F.M.t3_score(d['C'],d['T'],d['Dd'],v,q1,fd)
            dt=F.M.t3_delta(a0,a1,d['sc'])
            rows.append(dict(split=s,panel=p,mode=mode,dS_rw=ds,dE=dt['dE_est'],dFD=dt['dFD_est'],
                dTotal40=.35*ds+.05*(dt['dFD_est']-.8*dt['dE_est'])))
            del d,lh;gc.collect()
    return rows


def main():
    OUT.mkdir(parents=True,exist_ok=True);t0=time.time()
    meta=np.load(G.OUT/'feature_rows.npz');own=meta['owner'];y=labels(own)
    X=np.load(G.OUT/'X_context.npy',mmap_mode='r');rows=[];summaries=[];accepted=True
    for mode in ('parity','half_month','other_month'):
        path=OUT/f'pred_{mode}.npy';fold=meta[mode]
        if path.exists():pred=np.load(path)
        else:
            pred=np.zeros(len(y))
            for a in (0,1):
                tr=fold==a;mp=OUT/f'model_{mode}_{a}.txt'
                if mp.exists():mdl=lgb.Booster(model_file=str(mp))
                else:
                    mdl=lgb.train(PRM,lgb.Dataset(X[tr],y[tr],categorical_feature=[X.shape[1]-1]),num_boost_round=ROUNDS)
                    mdl.save_model(str(mp))
                pred[~tr]=mdl.predict(X[~tr],num_threads=2)
                del mdl;gc.collect()
                print('fit',mode,a,round(time.time()-t0),flush=True)
            np.save(path,pred)
        rr=pd.DataFrame(score(own,pred,mode));rows.extend(rr.to_dict('records'))
        for s,g in rr.groupby('split'):
            rec=dict(mode=mode,split=s,positive_corridors=int((g.dTotal40>0).sum()),
                **{k:F.M.family_mean(g.set_index('panel')[k].to_dict()) for k in ('dS_rw','dE','dFD','dTotal40')})
            summaries.append(rec);print(rec,flush=True)
            accepted &= rec['dS_rw']>0 and rec['dTotal40']>0
        pd.DataFrame(rows).to_csv(OUT/'checks.csv',index=False)
        pd.DataFrame(summaries).to_csv(OUT/'summary.csv',index=False)
        if not accepted:break
    if accepted:
        mdl=lgb.train(PRM,lgb.Dataset(X,y,categorical_feature=[X.shape[1]-1]),num_boost_round=ROUNDS)
        mdl.save_model(str(OUT/'model.txt'))
    (OUT/'decision.json').write_text(json.dumps(dict(accepted=bool(accepted),limit=LIMIT,
        completed_groups=len(summaries),gate='positive reweighted state and conservative total in all six checks; stop early on failure'),indent=2))
    print('accepted',bool(accepted),round(time.time()-t0),flush=True)


if __name__=='__main__':main()
