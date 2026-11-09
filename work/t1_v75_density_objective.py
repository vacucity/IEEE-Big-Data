"""Incremental congested-state refinement with a joint state/density objective.

The V64 model predicts a density median and a separate state split. Here fit
a bounded residual in log density, differentiating the squared speed/flow
errors and a smooth absolute accumulation error together. Features only use
released visible observations after deleting all artificial holdouts.
Task 1 is offline; both months may be used. No queue predictions are read.
"""
from __future__ import annotations
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[k]='2'
import json,time
import numpy as np
import pandas as pd
import lightgbm as lgb
import t1_v61_congest as C
import t1_v73_global_factor as F
import t1_v74_lowrank as L
import t1_v73_factor_verify as VFY
from evaluate import PANELS

OUT=F.OUT.parent/'t1_v75_density'
ALPHA=.75
LIMIT=.08
PRM=dict(learning_rate=.035,num_leaves=15,min_data_in_leaf=300,lambda_l2=10.,
         num_threads=2,verbose=-1,seed=75075,deterministic=True,force_col_wise=True)
ROUNDS=220


def derivatives(r,z):
    v0,q0,N0,ov,oq,No,a,b,c=z.T
    v=v0*np.exp(-ALPHA*r);q=q0*np.exp((1-ALPHA)*r);N=N0*np.exp(r)
    dv=-ALPHA*v;dq=(1-ALPHA)*q;err=N-No
    tau=np.maximum(.05*N0,.5);den=np.sqrt(err*err+tau*tau)
    grad=2*a*(v-ov)*dv+2*b*(q-oq)*dq+c*err*N/den
    hess=2*a*(dv*dv+(v-ov)*ALPHA**2*v)+2*b*(dq*dq+(q-oq)*(1-ALPHA)**2*q)
    hess+=c*(tau*tau*N*N/den**3+err*N/den)
    gn=2*a*dv*dv+2*b*dq*dq+c*tau*tau*N*N/den**3
    return grad,np.maximum(hess,.05*gn+1e-14)


def loss(r,z):
    v0,q0,N0,ov,oq,No,a,b,c=z.T
    v=v0*np.exp(-ALPHA*r);q=q0*np.exp((1-ALPHA)*r);N=N0*np.exp(r)
    tau=np.maximum(.05*N0,.5)
    return a*(v-ov)**2+b*(q-oq)**2+c*(np.sqrt((N-No)**2+tau*tau)-tau)


def train(X,Z):
    h0=derivatives(np.zeros(len(Z)),Z)[1];scale=max(float(h0.mean()),1e-14)
    def objective(pred,data):
        g,h=derivatives(pred,Z);return g/scale,h/scale
    return lgb.train(dict(PRM,objective=objective),lgb.Dataset(X,label=np.zeros(len(X)),
                       categorical_feature=[X.shape[1]-1]),num_boost_round=ROUNDS)


def feats(d,lh,v,q,pi):
    dd={**d,'pv':v,'pq':q}
    x=C.cong_feats(dd,lh,C.CFGS['F600a75s']['feats'])
    # Original state vs corrected density estimates indicate V64's adjustment.
    x=np.column_stack([x,np.log(np.maximum(q/v,1e-3)/np.maximum(d['pq']/d['pv'],1e-3)),
                       v-d['pv'],q/d['lanes']-d['pq']/d['lanes'],np.full(len(v),pi)])
    return x


def reference(p,s,d,lh):
    z=np.load(F.OUT/f'check_global_clip1_{s}_{p}.npz')
    # V74 parity crossfit for an accurate full-frame comparator; congested
    # cells have been unchanged since V64, irrespective of these free cells.
    feat=L.features(p,s,d['C'].day.to_numpy(),d['C'].t.to_numpy(),d['C'].pos.to_numpy(),d['fd'],True)
    cf=np.load(L.OUT/'coefficients.npz');x=L.design(feat,z['v1'],lh,'v');dv=np.zeros(len(x))
    for a in (0,1):
        k=d['free']&(d['par']!=a);dv[k]=x[k]@cf[f'{s}__v__parity__{a}']
    v=VFY.clamp(p,d['pos'],d['free'],z['v1']+np.clip(dv,-1,1),z['q'])
    return v,z['q']


def run():
    OUT.mkdir(parents=True,exist_ok=True);t0=time.time();data={};xs=[];zs=[];owners=[];par=[];half=[];month=[]
    for si,s in enumerate(F.SPLITS):
        for pi,p in enumerate(PANELS):
            d,lh=F.LOC.load(p,s);v,q=reference(p,s,d,lh);m=~d['free'];d['cong']=m;d['L']=lh
            a,b,c=C.cell_weights(d,v,q);N=C.dens(v,q,d['fd'],d['pos']);No=C.dens(d['ov'],d['oq'],d['fd'],d['pos'])
            # Fixed density coefficient uses the full previously validated
            # congested-cell LWR transfer, not the free-flow 40% proxy factor.
            Z=np.column_stack([v,q/d['lanes'],N,d['ov'],d['oq']/d['lanes'],No,a,b,c])
            X=feats(d,lh,v,q,pi)
            xs.append(X[m]);zs.append(Z[m]);owners.append(np.full(m.sum(),si*10+pi));par.append(d['par'][m]);half.append((d['day'][m]>=15).astype(int));month.append(np.full(m.sum(),si))
            w,unc=C.target_weights(d,F.LOC.load_local(p,s,'T'),d['fd'])
            data[s,p]=dict(d=d,v=v,q=q,m=m,w=w)
        print('loaded',s,f'{time.time()-t0:.0f}s',flush=True)
    X=np.concatenate(xs);Z=np.concatenate(zs);own=np.concatenate(owners);par=np.concatenate(par);half=np.concatenate(half);month=np.concatenate(month)
    # Numerical gradient check of the actual optimized loss.
    k=np.arange(min(len(Z),100));r=np.full(len(k),.012);eps=1e-6
    numeric=(loss(r+eps,Z[k])-loss(r-eps,Z[k]))/(2*eps)
    analytic=derivatives(r,Z[k])[0]
    assert np.allclose(numeric,analytic,rtol=1e-5,atol=1e-9)
    rows=[];preds={}
    for mode,fold in [('parity',par),('half_month',half),('other_month',month)]:
        pred=np.zeros(len(X))
        for a in (0,1):
            tr=fold==a;mdl=train(X[tr],Z[tr]);pred[~tr]=mdl.predict(X[~tr],num_threads=2)
        preds[mode]=pred
        for si,s in enumerate(F.SPLITS):
            for pi,p in enumerate(PANELS):
                x=data[s,p];d=x['d'];m=x['m'];delta=np.clip(pred[own==si*10+pi],-LIMIT,LIMIT)
                v,q=x['v'].copy(),x['q'].copy();v[m]=np.clip(v[m]*np.exp(-ALPHA*delta),F.M.SPEED_MIN,F.M.SPEED_MAX);q[m]*=np.exp((1-ALPHA)*delta)
                st=(d['ov'],d['oq'],d['lanes'],d['reg'],x['w'])
                ds=F.LOC.s_state_w(v,q,*st)-F.LOC.s_state_w(x['v'],x['q'],*st)
                a0=F.M.t3_score(d['C'],d['T'],d['Dd'],x['v'],x['q'],d['fd']);a1=F.M.t3_score(d['C'],d['T'],d['Dd'],v,q,d['fd']);dt=F.M.t3_delta(a0,a1,d['sc'])
                rows.append(dict(mode=mode,split=s,panel=p,dS_rw=ds,dE=dt['dE_est'],dFD=dt['dFD_est'],
                                 dTotal=.35*ds+.05*(dt['dFD_est']-2*dt['dE_est']),
                                 dTotal40=.35*ds+.05*(dt['dFD_est']-.8*dt['dE_est']),rms_log=float(np.sqrt(np.mean(delta**2)))))
        print('crossfit',mode,f'{time.time()-t0:.0f}s',flush=True)
    r=pd.DataFrame(rows);r.to_csv(OUT/'checks.csv',index=False)
    summary=[]
    for (s,mode),g in r.groupby(['split','mode']):
        summary.append(dict(split=s,mode=mode,positive_corridors=int((g.dTotal>0).sum()),
                            **{k:F.M.family_mean(g.set_index('panel')[k].to_dict()) for k in ('dS_rw','dE','dFD','dTotal','dTotal40')}))
    sm=pd.DataFrame(summary);sm.to_csv(OUT/'summary.csv',index=False);print(sm.to_string(index=False),flush=True)
    accepted=bool((sm.dS_rw>=0).all() and (sm.dTotal40>0).all() and (sm.positive_corridors>=7).all())
    if accepted:
        mdl=train(X,Z);mdl.save_model(str(OUT/'model.txt'))
    np.savez_compressed(OUT/'oof.npz',owner=own,**preds)
    (OUT/'decision.json').write_text(json.dumps(dict(accepted=accepted,limit=LIMIT,alpha=ALPHA,
        gate='nonnegative weighted state, positive conservative total, at least 7/10 positive corridors in every month/fold'),indent=2))
    print('accepted',accepted,flush=True)


if __name__=='__main__':run()
