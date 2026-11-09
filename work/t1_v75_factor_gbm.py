"""Nonlinear multiscale factor mixer replacing V73+V74 speed corrections.

Predict the V69 residual from visible-only additive and low-rank factors.
Fixed small trees and correction bound; days/months are held out for validation.
"""
from __future__ import annotations
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[k]='4'
import json,time
import numpy as np
import pandas as pd
import lightgbm as lgb
import t1_v73_global_factor as F
import t1_v74_lowrank as L
import t1_v73_factor_verify as VFY
import t1_v75_density_objective as D
from evaluate import PANELS

OUT=F.OUT.parent/'t1_v75_factor_gbm'
PRM=dict(objective='regression',learning_rate=.025,num_leaves=31,min_data_in_leaf=1500,
         lambda_l2=30.,num_threads=4,verbose=-1,seed=75075,deterministic=True,force_col_wise=True)
ROUNDS=300
LIMIT=1.


def features(p,s,d,lh,v,q,pi,hide):
    c=d['C'] if hide else None
    day=c.day.to_numpy() if hide else lh['day'];tt=c.t.to_numpy() if hide else (lh['key']//1000)%288
    pos=c.pos.to_numpy() if hide else lh['key']%1000
    fd=d['fd'];f=F.factors(p,s,day,tt,pos,fd,hide);l=L.features(p,s,day,tt,pos,fd,hide)
    return np.column_stack([np.clip(f['pred']-v[:,None],-12,12),np.clip(l['v']-v[:,None],-12,12),
        np.clip((l['q']-q[:,None]/fd['lanes'][pos,None])/100,-3,3),v/fd['ffs'][pos],v/fd['vc'][pos],
        q/fd['lanes'][pos]/1000,f['cnt'],lh['db'],lh['da'],lh['nsp'],lh['n3'],
        d['reg'] if hide else lh['regime'],fd['lanes'][pos],np.full(len(v),pi)]).astype(np.float32)


def run():
    OUT.mkdir(parents=True,exist_ok=True);t0=time.time();data={};xx=[];yy=[];owners=[];pars=[];halfs=[];months=[]
    for si,s in enumerate(F.SPLITS):
        base=F.baseline(s)
        for pi,p in enumerate(PANELS):
            d,lh=F.LOC.load(p,s);old,q=D.reference(p,s,d,lh);v=base[p+'_v'];m=d['free']
            x=features(p,s,d,lh,v,q,pi,True);rw,unc=F.LOC.target_weights(d,lh,F.LOC.load_local(p,s,'T'),d['fd'])
            xx.append(x[m]);yy.append((d['ov']-v)[m]);owners.append(np.full(m.sum(),si*10+pi));pars.append(d['par'][m]);halfs.append((d['day'][m]>=15).astype(int));months.append(np.full(m.sum(),si))
            data[s,p]=dict(d=d,old=old,q=q,v=v,m=m,rw=rw)
        print('loaded',s,f'{time.time()-t0:.0f}s',flush=True)
    X=np.concatenate(xx);y=np.concatenate(yy);own=np.concatenate(owners);par=np.concatenate(pars);half=np.concatenate(halfs);month=np.concatenate(months)
    del xx,yy
    rows=[]
    for mode,fold in [('parity',par),('half_month',half),('other_month',month)]:
        pred=np.zeros(len(X))
        for a in (0,1):
            tr=fold==a
            mdl=lgb.train(PRM,lgb.Dataset(X[tr],y[tr],categorical_feature=[X.shape[1]-1]),num_boost_round=ROUNDS)
            pred[~tr]=mdl.predict(X[~tr],num_threads=4)
            print('fit',mode,a,f'{time.time()-t0:.0f}s',flush=True)
        for si,s in enumerate(F.SPLITS):
            for pi,p in enumerate(PANELS):
                a=data[s,p];d=a['d'];m=a['m'];delta=np.zeros(len(m));delta[m]=np.clip(pred[own==si*10+pi],-LIMIT,LIMIT)
                new=VFY.clamp(p,d['pos'],m,a['v']+delta,a['q'])
                st=(d['ov'],d['oq'],d['fd']['lanes'][d['pos']],d['reg'],a['rw'])
                ds=F.LOC.s_state_w(new,a['q'],*st)-F.LOC.s_state_w(a['old'],a['q'],*st)
                a0=F.M.t3_score(d['C'],d['T'],d['Dd'],a['old'],a['q'],d['fd']);a1=F.M.t3_score(d['C'],d['T'],d['Dd'],new,a['q'],d['fd'])
                dt=F.M.t3_delta(a0,a1,d['sc'])
                rows.append(dict(split=s,panel=p,mode=mode,dS_rw=ds,dE=dt['dE_est'],dFD=dt['dFD_est'],dTotal40=.35*ds+.05*(dt['dFD_est']-.8*dt['dE_est'])))
        g=pd.DataFrame(rows);g=g[g['mode']==mode]
        print('result',mode,{s:{k:F.M.family_mean(g[g.split==s].set_index('panel')[k].to_dict()) for k in ('dS_rw','dTotal40')} for s in F.SPLITS},flush=True)
    r=pd.DataFrame(rows);r.to_csv(OUT/'checks.csv',index=False);sm=[]
    for (s,mode),g in r.groupby(['split','mode']):
        sm.append(dict(split=s,mode=mode,positive_corridors=int((g.dTotal40>0).sum()),**{k:F.M.family_mean(g.set_index('panel')[k].to_dict()) for k in ('dS_rw','dE','dFD','dTotal40')}))
    sm=pd.DataFrame(sm);sm.to_csv(OUT/'summary.csv',index=False);print(sm.to_string(index=False),flush=True)
    accepted=bool((sm.dS_rw>0).all() and (sm.dTotal40>0).all())
    if accepted:
        mdl=lgb.train(PRM,lgb.Dataset(X,y,categorical_feature=[X.shape[1]-1]),num_boost_round=ROUNDS)
        mdl.save_model(str(OUT/'model.txt'))
    (OUT/'decision.json').write_text(json.dumps(dict(accepted=accepted,limit=LIMIT,gate='positive weighted state and conservative total in all six month/fold checks'),indent=2))
    print('accepted',accepted,flush=True)


if __name__=='__main__':run()
