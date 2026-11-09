"""Extend V75's factor mixer with visible local context; compare OOF to V75.

Two candidates: same V75 features with 900 rounds, and enriched local context
with 700 rounds/min_leaf 500. Bounds 1/2 km/h are checked rather than assumed.
Every candidate is evaluated on parity, half-month and opposite-month folds.
Inference uses only Task-1 observations; no queue rows are touched.
"""
from __future__ import annotations
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[k]='4'
import argparse,json,time,gc
import numpy as np
import pandas as pd
import lightgbm as lgb
import t1_v75_factor_gbm as B
import t1_v73_global_factor as F
import t1_v73_factor_verify as VFY
from evaluate import PANELS

OUT=F.OUT.parent/'t1_v76_context_mix'
SPECS={'deep':(B.PRM,900),'context':(dict(B.PRM,min_data_in_leaf=500,seed=76076),700)}
LIMITS=(1.,2.)


def features(p,s,d,lh,v,q,pi,hide):
    base=B.features(p,s,d,lh,v,q,pi,hide)
    pos=d['pos'] if hide else (lh['key']%1000).astype(int)
    t=d['C'].t.to_numpy() if hide else (lh['key']//1000)%288
    fd=d['fd'];lane=fd['lanes'][pos];qpl=q/lane
    vnames=('vb','va','vb2','va2','vu','vd','vu1','vd1','vu2','vd2')
    qnames=('qb','qa','qb2','qa2','qu','qd','qu1','qd1','qu2','qd2')
    vs=np.column_stack([np.clip(lh[k]-v,-25,25) for k in vnames])
    qs=np.column_stack([np.clip((lh[k]-qpl)/100,-8,8) for k in qnames])
    db,da=lh['db'],lh['da'];den=np.maximum(db+da,1)
    interp=(da*lh['vb']+db*lh['va'])/den-v
    slope=(lh['va']-lh['vb'])/den
    return np.column_stack([base[:,:-1],vs,qs,lh['db2'],lh['da2'],lh['nblk'],
        np.clip(interp,-25,25),np.clip(slope,-10,10),pos/max(fd['n']-1,1),
        fd['ffs'][pos]/100,fd['vc'][pos]/100,np.sin(t*2*np.pi/288),np.cos(t*2*np.pi/288),base[:,-1]]).astype(np.float32)


def load():
    meta=OUT/'feature_rows.npz'
    if meta.exists():
        z=np.load(meta)
        return None,{k:np.load(OUT/f'X_{k}.npy',mmap_mode='r') for k in SPECS},z['y'],z['owner'],{k:z[k] for k in ('parity','half_month','other_month')}
    total=sum(int(np.load(F.OUT/f'check_global_clip1_{s}_{p}.npz')['mask'].sum()) for s in F.SPLITS for p in PANELS)
    XS={};y=np.empty(total,np.float32);own=np.empty(total,np.int8);par=np.empty(total,np.int8);half=np.empty(total,np.int8);month=np.empty(total,np.int8);off=0
    for si,s in enumerate(F.SPLITS):
        z=F.baseline(s)
        for pi,p in enumerate(PANELS):
            d,lh=F.LOC.load(p,s);v,q=z[p+'_v'],z[p+'_q'];m=d['free']
            X=B.features(p,s,d,lh,v,q,pi,True);E=features(p,s,d,lh,v,q,pi,True)
            if not XS:
                XS={k:np.lib.format.open_memmap(OUT/f'X_{k}.npy',mode='w+',dtype=np.float32,shape=(total,x.shape[1])) for k,x in [('deep',X),('context',E)]}
            end=off+int(m.sum());XS['deep'][off:end]=X[m];XS['context'][off:end]=E[m]
            y[off:end]=(d['ov']-v)[m];own[off:end]=si*10+pi;par[off:end]=d['par'][m];half[off:end]=(d['day'][m]>=15).astype(int);month[off:end]=si
            off=end
            del d,lh,X,E;gc.collect()
        print('loaded',s,flush=True)
    assert off==total
    for x in XS.values():x.flush()
    np.savez(meta,y=y,owner=own,parity=par,half_month=half,other_month=month)
    return None,XS,y,own,{'parity':par,'half_month':half,'other_month':month}


def score(data,own,pred,ref,mode,kind,limit):
    rows=[]
    for si,s in enumerate(F.SPLITS):
        z=F.baseline(s)
        for pi,p in enumerate(PANELS):
            d,lh=F.LOC.load(p,s);rw,unc=F.LOC.target_weights(d,lh,F.LOC.load_local(p,s,'T'),d['fd'])
            a=dict(v=z[p+'_v'],q=z[p+'_q'],rw=rw);m=d['free'];k=own==si*10+pi
            delta=np.zeros(len(m));delta[m]=np.clip(pred[k],-limit,limit)
            dr=np.zeros(len(m));dr[m]=np.clip(ref[k],-B.LIMIT,B.LIMIT)
            v0=VFY.clamp(p,d['pos'],m,a['v']+dr,a['q']);v1=VFY.clamp(p,d['pos'],m,a['v']+delta,a['q'])
            st=(d['ov'],d['oq'],d['fd']['lanes'][d['pos']],d['reg'],a['rw'])
            ds=F.LOC.s_state_w(v1,a['q'],*st)-F.LOC.s_state_w(v0,a['q'],*st)
            a0=F.M.t3_score(d['C'],d['T'],d['Dd'],v0,a['q'],d['fd']);a1=F.M.t3_score(d['C'],d['T'],d['Dd'],v1,a['q'],d['fd']);dt=F.M.t3_delta(a0,a1,d['sc'])
            rows.append(dict(cand=kind+'_clip'+str(int(limit)),kind=kind,limit=limit,split=s,panel=p,mode=mode,
                             dS_rw=ds,dE=dt['dE_est'],dFD=dt['dFD_est'],dTotal40=.35*ds+.05*(dt['dFD_est']-.8*dt['dE_est'])))
    return rows


def run(args):
    OUT.mkdir(parents=True,exist_ok=True);t0=time.time();data,XS,y,own,folds=load();rows=[]
    for mode in args.modes:
        fold=folds[mode];refpath=OUT/f'baseline_{mode}.npy'
        ref=np.load(refpath) if refpath.exists() else np.zeros(len(y))
        for kind in ('deep','context'):
            pp=OUT/f'{kind}_{mode}.npy';X=XS[kind];prm,rounds=SPECS[kind]
            if pp.exists():pred=np.load(pp)
            else:
                pred=np.zeros(len(y))
                for a in (0,1):
                    tr=fold==a;mdl=lgb.train(prm,lgb.Dataset(X[tr],y[tr],categorical_feature=[X.shape[1]-1]),num_boost_round=rounds)
                    pred[~tr]=mdl.predict(X[~tr],num_threads=4)
                    if kind=='deep' and not refpath.exists():ref[~tr]=mdl.predict(X[~tr],num_iteration=B.ROUNDS,num_threads=4)
                    mdl.save_model(str(OUT/f'{kind}_{mode}_{a}.txt'))
                    del mdl;gc.collect()
                    print('fit',mode,kind,a,f'{time.time()-t0:.0f}s',flush=True)
                np.save(pp,pred)
            if kind=='deep' and not refpath.exists():np.save(refpath,ref)
            assert ref.shape==pred.shape
            for limit in LIMITS:
                rows.extend(score(data,own,pred,ref,mode,kind,limit))
            print('scored',mode,kind,f'{time.time()-t0:.0f}s',flush=True)
        pd.DataFrame(rows).to_csv(OUT/'checks.csv',index=False)
    r=pd.DataFrame(rows);sm=[]
    for (cand,s,mode),g in r.groupby(['cand','split','mode']):
        sm.append(dict(cand=cand,split=s,mode=mode,positive_corridors=int((g.dTotal40>0).sum()),**{k:F.M.family_mean(g.set_index('panel')[k].to_dict()) for k in ('dS_rw','dE','dFD','dTotal40')}))
    sm=pd.DataFrame(sm);sm.to_csv(OUT/'summary.csv',index=False);print(sm.to_string(index=False),flush=True)
    eligible=[c for c,g in sm.groupby('cand') if len(g)==6 and (g.dS_rw>0).all() and (g.dTotal40>0).all()]
    chosen=max(eligible,key=lambda c:sm[sm.cand==c].dTotal40.min()) if eligible else None
    if chosen:
        kind,lim=chosen.split('_clip');X=XS[kind];prm,rounds=SPECS[kind]
        mdl=lgb.train(prm,lgb.Dataset(X,y,categorical_feature=[X.shape[1]-1]),num_boost_round=rounds);mdl.save_model(str(OUT/'model.txt'))
    (OUT/'decision.json').write_text(json.dumps(dict(chosen=chosen,gate='six positive reweighted state and conservative total checks; maximize worst month/fold'),indent=2))
    print('chosen',chosen,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--modes',nargs='+',default=['parity','half_month','other_month']);run(ap.parse_args())
