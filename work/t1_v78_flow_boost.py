"""Test additional flow boosting against the confirmed V77 baseline.

Keep the 77 visible-only features and the 60 vph/lane bound. Add 350 trees
to each saved 650-tree V77 fold model, verifying that its original prefix
exactly reproduces the saved baseline. Require six positive held-out checks.
"""
from __future__ import annotations
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[k]='4'
import gc,json,time
import numpy as np
import pandas as pd
import lightgbm as lgb
import t1_v77_flow_factors as P
from t1_v77_flow_factors import F,G,VFY,PANELS

OUT=P.OUT.parent/'t1_v78_flow_boost'
EXTRA_ROUNDS=350
LIMIT=60.
features=P.features


def score(own,pred,mode):
    rows=[];vp=np.load(G.OUT/f'context_{mode}.npy');ref=np.load(P.OUT/f'pred_{mode}.npy')
    for si,s in enumerate(F.SPLITS):
        z=F.baseline(s)
        for pi,p in enumerate(PANELS):
            d,lh=F.LOC.load(p,s);m=d['free'];idx=own==si*10+pi;fd=d['fd'];lane=fd['lanes'][d['pos']]
            rw,_=F.LOC.target_weights(d,lh,F.LOC.load_local(p,s,'T'),fd)
            dv=np.zeros(len(m));dv[m]=np.clip(vp[idx],-2,2)
            v=VFY.clamp(p,d['pos'],m,z[p+'_v']+dv,z[p+'_q']);cap=v*lane*fd['kjl'][d['pos']]
            qs=[]
            for pr in (ref,pred):
                dq=np.zeros(len(m));dq[m]=np.clip(pr[idx]*100,-LIMIT,LIMIT)*lane[m]
                qs.append(np.where(m,np.maximum(0,np.minimum(z[p+'_q']+dq,cap)),z[p+'_q']))
            q0,q1=qs;st=(d['ov'],d['oq'],lane,d['reg'],rw)
            ds=F.LOC.s_state_w(v,q1,*st)-F.LOC.s_state_w(v,q0,*st)
            a0=F.M.t3_score(d['C'],d['T'],d['Dd'],v,q0,fd);a1=F.M.t3_score(d['C'],d['T'],d['Dd'],v,q1,fd)
            dt=F.M.t3_delta(a0,a1,d['sc'])
            rows.append(dict(split=s,panel=p,mode=mode,dS_rw=ds,dE=dt['dE_est'],dFD=dt['dFD_est'],
                dTotal40=.35*ds+.05*(dt['dFD_est']-.8*dt['dE_est'])))
            del d,lh;gc.collect()
    return rows


def main():
    OUT.mkdir(parents=True,exist_ok=True);t0=time.time();meta=np.load(G.OUT/'feature_rows.npz');own=meta['owner']
    X=np.load(P.OUT/'X.npy',mmap_mode='r');y=np.load(P.Q.OUT/'flow_labels.npy');rows=[];summary=[];prefix=[];accepted=True
    for mode in ('parity','half_month','other_month'):
        fold=meta[mode];pp=OUT/f'pred_{mode}.npy';ref=np.load(P.OUT/f'pred_{mode}.npy')
        if pp.exists():
            pred=np.load(pp);prefix.extend(json.loads((OUT/f'prefix_{mode}.json').read_text()))
        else:
            pred=np.zeros(len(y));pm=[]
            for a in (0,1):
                tr=fold==a;mp=OUT/f'model_{mode}_{a}.txt'
                if mp.exists():mdl=lgb.Booster(model_file=str(mp))
                else:
                    mdl=lgb.train(P.PRM,lgb.Dataset(X[tr],y[tr],categorical_feature=[X.shape[1]-1]),
                        num_boost_round=EXTRA_ROUNDS,init_model=str(P.OUT/f'model_{mode}_{a}.txt'))
                    assert mdl.num_trees()==P.ROUNDS+EXTRA_ROUNDS
                    mdl.save_model(str(mp))
                check=mdl.predict(X[~tr],num_iteration=P.ROUNDS,num_threads=4)
                err=float(np.max(np.abs(check-ref[~tr])));assert err<1e-10,(mode,a,err)
                pm.append(dict(mode=mode,fold=a,max_baseline_difference=err))
                pred[~tr]=mdl.predict(X[~tr],num_threads=4)
                del mdl,check;gc.collect();print('fit',mode,a,'prefix',err,'seconds',round(time.time()-t0),flush=True)
            (OUT/f'prefix_{mode}.json').write_text(json.dumps(pm,indent=2));prefix.extend(pm);np.save(pp,pred)
        rr=pd.DataFrame(score(own,pred,mode));rows.extend(rr.to_dict('records'))
        for s,g in rr.groupby('split'):
            rec=dict(limit=LIMIT,mode=mode,split=s,positive_corridors=int((g.dTotal40>0).sum()),
                **{k:F.M.family_mean(g.set_index('panel')[k].to_dict()) for k in ('dS_rw','dE','dFD','dTotal40')})
            summary.append(rec);print(rec,flush=True)
            accepted &= rec['dS_rw']>0 and rec['dTotal40']>0
        pd.DataFrame(rows).to_csv(OUT/'checks.csv',index=False);pd.DataFrame(summary).to_csv(OUT/'summary.csv',index=False)
        if not accepted:break
    if accepted:
        assert len(summary)==6
        mdl=lgb.train(P.PRM,lgb.Dataset(X,y,categorical_feature=[X.shape[1]-1]),
            num_boost_round=EXTRA_ROUNDS,init_model=str(P.OUT/'model.txt'))
        assert mdl.num_trees()==P.ROUNDS+EXTRA_ROUNDS
        mdl.save_model(str(OUT/'model.txt'))
    (OUT/'prefix_audit.json').write_text(json.dumps(prefix,indent=2))
    (OUT/'feature_alignment.json').write_text((P.OUT/'feature_alignment.json').read_text())
    (OUT/'decision.json').write_text(json.dumps(dict(limit=LIMIT if accepted else None,trees=P.ROUNDS+EXTRA_ROUNDS,
        gate='exact V77 model-prefix reproduction and six positive weighted state and conservative total checks'),indent=2))
    print('accepted',bool(accepted),'seconds',round(time.time()-t0),flush=True)


if __name__=='__main__':main()
