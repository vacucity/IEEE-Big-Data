"""Extend V81's confirmed congested-speed common-mode correction.

Compare spatial-band ridge replacement and a bounded nonlinear residual.
Every residual target comes from a teacher fitted on the same training fold.
Select on parity, freeze, then confirm half-month and opposite-month splits.
"""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='2'
import argparse,json,gc,time
import numpy as np
import pandas as pd
import lightgbm as lgb
import t1_v81_congest_speed as S
from t1_v81_congest_speed import B,N,C,A,F,ROOT,PANELS

OUT=ROOT/'work/results/t1_v82_congest_residual'
ALPHAS=(.5,1.)
KINDS=('band','gbm')
ROUNDS=240
PRM=dict(objective='regression',num_leaves=15,min_data_in_leaf=1000,learning_rate=.025,
    lambda_l2=50.,max_bin=63,num_threads=2,verbosity=-1,seed=82082,deterministic=True,force_col_wise=True)
EXTRA_CAP=.5
PROTOCOL=dict(base='V81 confirmed 0.90329',candidates=KINDS,alpha_grid=ALPHAS,
    nonlinear_parameters=PRM,rounds=ROUNDS,extra_residual_cap=EXTRA_CAP,total_speed_cap=1.,
    selection='Parity: both months positive reweighted state and conservative total, negative LWR delta, >=7 positive-total corridors; maximize minimum total.',
    confirmation='Freeze kind and alpha before half-month and opposite-month; same gate in every month.',
    data='Only visible Task1 observations, all artificial hidden cells excluded from feature extraction.',
    targets='Same-training-fold old teachers and V81 coefficients; never use reverse-OOF training residuals.',
    scope='Congested non-blackout speed only; flow/queue/ODME unchanged',caveat=N.PROTOCOL['caveat'])

def weights(d,z,idx,mode=None,a=None):
    counts=d['counts_parity'].sum(0) if mode is None else d['counts_'+mode][a]
    return 1/np.maximum(counts[z['owner'][idx],d['reg'][idx].astype(int)],1)

def teacher(xo,d,idx,mode=None,a=None):
    rp=A.OUT/'density_l1.txt' if mode is None else N.OUT/f'model_old_{mode}_{a}_r.txt'
    sp=A.OUT/'split_l2.txt' if mode is None else N.OUT/f'model_old_{mode}_{a}_sstar.txt'
    md=lgb.Booster(model_file=str(rp));ms=lgb.Booster(model_file=str(sp))
    return C.corrected(d['pv'][idx],d['pq'][idx],md.predict(xo[idx],num_threads=2),ms.predict(xo[idx],num_threads=2),A.CFG,N.LAM)

def nonlinear_features(xn,xcm,pv,pq,lanes,cm):
    vv=np.clip(pv+np.clip(cm,-1,1),C.M.SPEED_MIN,C.M.SPEED_MAX)
    context=np.column_stack([np.clip(cm,-1,1),vv/100,pq/lanes/600,pq/lanes/np.maximum(vv,1)/50])
    return np.column_stack([xn[:,:-1],xcm,context,xn[:,-1]]).astype(np.float32)

def delta(cm,new,kind,alpha):
    if kind=='band':return (1-alpha)*cm+alpha*new
    return np.clip(cm,-1,1)+alpha*np.clip(new,-EXTRA_CAP,EXTRA_CAP)

def predictions(mode,kinds):
    path=OUT/f'pred_{mode}.npz'
    if path.exists():return dict(np.load(path))
    xo,xn,z=N.dataset();d=B.data();old=N.heads(xo,z,mode,'old');ref=np.load(S.OUT/f'pred_{mode}.npy')
    pred={k:np.zeros(len(xo)) for k in kinds};audit=[]
    for a in (0,1):
        tr=np.flatnonzero(z[mode]==a);te=np.flatnonzero(z[mode]!=a)
        beta0=np.load(S.OUT/f'beta_{mode}_{a}.npy');cmtr=d['X'][tr,:25]@beta0
        err=float(np.max(np.abs(d['X'][te,:25]@beta0-ref[te])));assert err<1e-12
        v,q=teacher(xo,d,tr,mode,a);w=weights(d,z,tr,mode,a)
        vt,qt=C.corrected(d['pv'][te],d['pq'][te],old['r'][te],old['sstar'][te],A.CFG,N.LAM)
        if 'band' in kinds:
            beta=B.ridge(d['X'][tr],np.clip(d['ov'][tr]-v,-10,10),w)
            pred['band'][te]=d['X'][te]@beta;np.save(OUT/f'beta_band_{mode}_{a}.npy',beta)
        if 'gbm' in kinds:
            X=nonlinear_features(xn[tr],d['X'][tr],v,q,d['lanes'][tr],cmtr)
            v81,_=S.transform(v,q,cmtr);y=np.clip(d['ov'][tr]-v81,-5,5)
            mp=OUT/f'model_gbm_{mode}_{a}.txt'
            if mp.exists():model=lgb.Booster(model_file=str(mp))
            else:
                model=lgb.train(PRM,lgb.Dataset(X,y,weight=w/w.mean(),categorical_feature=[X.shape[1]-1]),num_boost_round=ROUNDS);model.save_model(str(mp))
            XT=nonlinear_features(xn[te],d['X'][te],vt,qt,d['lanes'][te],ref[te]);pred['gbm'][te]=model.predict(XT,num_threads=2)
            del X,XT;gc.collect()
        audit.append(dict(fold=a,v81_oof_replay_max_error=err,train_rows=len(tr),evaluation_rows=len(te)))
        print('trained',mode,a,kinds,flush=True)
    np.savez_compressed(path,**pred);(OUT/f'baseline_replay_{mode}.json').write_text(json.dumps(audit,indent=2));return pred

def score(mode,preds,alphas):
    xo,xn,z=N.dataset();old=N.heads(xo,z,mode,'old');cm=np.load(S.OUT/f'pred_{mode}.npy');rows=[]
    for si,s in enumerate(F.SPLITS):
        for pi,p in enumerate(PANELS):
            d=N.load_one(p,s);m=d['cong'];ix=z['owner']==si*10+pi
            va,q0=N.incumbent_free(s,mode,p,pi,d);va[m],q0[m]=C.corrected(d['pv'][m],d['pq'][m],old['r'][ix],old['sstar'][ix],A.CFG,N.LAM)
            v0=va.copy();v0[m],_=S.transform(va[m],q0[m],cm[ix])
            rw,_=C.target_weights(d,C.LOC.load_local(p,s,'T'),d['fd']);args=(d['ov'],d['oq'],d['lanes'],d['reg'],rw)
            st=C.LOC.s_state_w(v0,q0,*args);a0=C.M.t3_score(d['C'],d['T'],d['Dd'],v0,q0,d['fd'])
            for kind,pred in preds.items():
                for alpha in alphas:
                    v1=va.copy();v1[m],_=S.transform(va[m],q0[m],delta(cm[ix],pred[ix],kind,alpha))
                    ds=C.LOC.s_state_w(v1,q0,*args)-st;dt=C.M.t3_delta(a0,C.M.t3_score(d['C'],d['T'],d['Dd'],v1,q0,d['fd']),d['sc'])
                    rows.append(dict(split=s,panel=p,mode=mode,kind=kind,alpha=alpha,dS=ds,dE=dt['dE_est'],dFD=dt['dFD_est'],dTotal=.35*ds+.05*(dt['dFD_est']-.8*dt['dE_est'])))
            del d;gc.collect()
        print('scored',mode,s,flush=True)
    return rows

def check(mode):
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'protocol.json').write_text(json.dumps(PROTOCOL,indent=2))
    if mode=='parity':kinds=KINDS;alphas=ALPHAS
    else:
        choice=json.loads((OUT/'selection.json').read_text());assert choice['kind'] is not None
        kinds=(choice['kind'],);alphas=(choice['alpha'],)
    preds=predictions(mode,kinds);df=pd.DataFrame(score(mode,preds,alphas));df.to_csv(OUT/f'checks_{mode}.csv',index=False);sm=[]
    for (s,kind,alpha),g in df.groupby(['split','kind','alpha']):
        sm.append(dict(split=s,mode=mode,kind=kind,alpha=alpha,positive_corridors=int((g.dTotal>0).sum()),
            **{k:C.M.family_mean(g.set_index('panel')[k].to_dict()) for k in ('dS','dE','dFD','dTotal')}))
    sm=pd.DataFrame(sm);sm.to_csv(OUT/f'summary_{mode}.csv',index=False)
    passes=lambda g:bool((g.dS>0).all() and (g.dE<0).all() and (g.dTotal>0).all() and (g.positive_corridors>=7).all())
    if mode=='parity':
        eligible=[(k,a) for (k,a),g in sm.groupby(['kind','alpha']) if passes(g)]
        kind,alpha=max(eligible,key=lambda ka:sm[(sm.kind==ka[0])&(sm.alpha==ka[1])].dTotal.min()) if eligible else (None,None)
        result=dict(kind=kind,alpha=alpha,eligible=eligible);path='selection.json'
    else:result=dict(kind=kinds[0],alpha=alphas[0],passed=passes(sm));path=f'decision_{mode}.json'
    (OUT/path).write_text(json.dumps(result,indent=2));print(sm.to_string(index=False),flush=True);print(result,flush=True)

def full():
    choice=json.loads((OUT/'selection.json').read_text());assert choice['kind'] is not None
    for mode in ('half_month','other_month'):assert json.loads((OUT/f'decision_{mode}.json').read_text())['passed']
    xo,xn,z=N.dataset();d=B.data();xo=N.old_inputs_exact();idx=np.arange(len(xo));v,q=teacher(xo,d,idx);w=weights(d,z,idx)
    beta0=S.full();cm=d['X'][:,:25]@beta0
    replay=B.ridge(d['X'][:,:25],np.clip(d['ov']-v,-10,10),w);err=float(np.max(np.abs(replay-beta0)));assert err<1e-12
    (OUT/'full_v81_replay.json').write_text(json.dumps(dict(beta_max_error=err),indent=2))
    if choice['kind']=='band':
        path=OUT/'beta_band_full.npy'
        if not path.exists():np.save(path,B.ridge(d['X'],np.clip(d['ov']-v,-10,10),w))
    else:
        path=OUT/'model_gbm_full.txt'
        if not path.exists():
            X=nonlinear_features(xn,d['X'],v,q,d['lanes'],cm);v81,_=S.transform(v,q,cm);y=np.clip(d['ov']-v81,-5,5)
            lgb.train(PRM,lgb.Dataset(X,y,weight=w/w.mean(),categorical_feature=[X.shape[1]-1]),num_boost_round=ROUNDS).save_model(str(path))
    return choice

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('cmd',choices=['parity','half_month','other_month','full']);a=ap.parse_args()
    full() if a.cmd=='full' else check(a.cmd)
