"""Observation-only residual: do not expose the in-sample teacher prediction as a feature."""
import argparse,json,gc
import numpy as np
import pandas as pd
import lightgbm as lgb
import t1_v84_flow_context as K
from t1_v84_flow_context import T,F,G,P,N,C,ROOT,PRM,ROUNDS,ALPHAS

OUT=ROOT/'work/results/t1_v84_context_continuation'
DATA_OUT=K.OUT
extra=K.extra
EXTRA_CAP=K.EXTRA_CAP
PROTOCOL=dict(K.PROTOCOL,model='Observation-only 180-tree residual; remove teacher prediction input and teacher-relative feature construction',
              reason='Avoid shortcut calibration of optimistic in-sample teacher values; residual target remains same-training-fold.',features=120)


def design(base,extra,idx,raw=None):
    return np.column_stack([base[idx,:-1],extra[idx],base[idx,-1]]).astype(np.float32)


def predictions(mode):
    path=OUT/f'pred_{mode}.npy'
    if path.exists():return np.load(path)
    meta,x,e=K.load();ref=np.load(T.OUT/f'raw_{mode}.npy');y=np.load(P.Q.OUT/'flow_labels.npy');pred=np.zeros(len(y));audit=[]
    for a in (0,1):
        tr=np.flatnonzero(meta[mode]==a);te=np.flatnonzero(meta[mode]!=a);mp=OUT/f'model_{mode}_{a}.txt'
        teacher=lgb.Booster(model_file=str(T.OUT/f'raw_{mode}_{a}.txt'))
        jj=te[np.linspace(0,len(te)-1,4096,dtype=int)];err=float(np.max(np.abs(T.predict(teacher,x,jj)-ref[jj])));assert err<1e-10
        if mp.exists():model=lgb.Booster(model_file=str(mp))
        else:
            rp=DATA_OUT/f'teacher_train_{mode}_{a}.npy'
            if rp.exists():raw=np.load(rp)
            else:raw=T.predict(teacher,x,tr);np.save(rp,raw)
            xx=design(x,e,tr);model=lgb.train(PRM,lgb.Dataset(xx,y[tr]-raw,categorical_feature=[xx.shape[1]-1]),num_boost_round=ROUNDS,callbacks=[T.training_progress])
            model.save_model(str(mp));del xx,raw;gc.collect()
        for st in range(0,len(te),65536):
            ii=te[st:st+65536];pred[ii]=model.predict(design(x,e,ii),num_threads=4)
        audit.append(dict(fold=a,teacher_oof_replay_error=err));del teacher,model;gc.collect();print('observation residual fit',mode,a,flush=True)
    np.save(path,pred);(OUT/f'teacher_replay_{mode}.json').write_text(json.dumps(audit,indent=2));return pred


def check(mode):
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'protocol.json').write_text(json.dumps(PROTOCOL,indent=2))
    alphas=ALPHAS if mode=='parity' else (json.loads((OUT/'selection.json').read_text())['alpha'],);assert None not in alphas
    df=K.score(mode,predictions(mode),alphas);df.to_csv(OUT/f'checks_{mode}.csv',index=False);rows=[]
    for (s,alpha),g in df.groupby(['split','alpha']):
        rows.append(dict(split=s,mode=mode,alpha=alpha,positive_corridors=int((g.dTotal>0).sum()),**{k:F.M.family_mean(g.set_index('panel')[k].to_dict()) for k in ('dS','dE','dFD','dTotal')}))
    sm=pd.DataFrame(rows);sm.to_csv(OUT/f'summary_{mode}.csv',index=False)
    passes=lambda g:bool((g.dS>0).all() and (g.dE<0).all() and (g.dTotal>0).all() and (g.positive_corridors>=7).all())
    if mode=='parity':
        eligible=[a for a,g in sm.groupby('alpha') if passes(g)];alpha=max(eligible,key=lambda a:sm[sm.alpha==a].dTotal.min()) if eligible else None
        result=dict(alpha=alpha,eligible=eligible);path='selection.json'
    else:result=dict(alpha=alphas[0],passed=passes(sm));path=f'decision_{mode}.json'
    (OUT/path).write_text(json.dumps(result,indent=2));print(sm.to_string(index=False),flush=True);print(result,flush=True)


def full():
    alpha=json.loads((OUT/'selection.json').read_text())['alpha'];assert alpha is not None
    for mode in ('half_month','other_month'):
        d=json.loads((OUT/f'decision_{mode}.json').read_text());assert d['passed'] and d['alpha']==alpha
    mp=OUT/'model_full.txt'
    if mp.exists():return alpha
    meta,x,e=K.load();idx=np.arange(len(meta['owner']));rp=DATA_OUT/'teacher_train_full.npy'
    if rp.exists():raw=np.load(rp)
    else:raw=T.predict(lgb.Booster(model_file=str(T.OUT/'model.txt')),x,idx);np.save(rp,raw)
    xx=design(x,e,idx);y=np.load(P.Q.OUT/'flow_labels.npy')-raw
    lgb.train(PRM,lgb.Dataset(xx,y,categorical_feature=[xx.shape[1]-1]),num_boost_round=ROUNDS,callbacks=[T.training_progress]).save_model(str(mp));return alpha


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['parity','half_month','other_month','full']);a=ap.parse_args()
    full() if a.mode=='full' else check(a.mode)
