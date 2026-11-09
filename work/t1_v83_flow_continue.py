"""Continue the incumbent flow learner; evaluate against complete V82 states."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='4'
import argparse,json,gc,time
import numpy as np
import pandas as pd
import lightgbm as lgb
import t1_v79_flow_detail as Q
from t1_v79_flow_detail import F,G,P,PANELS
import t1_v82_cmode_interactions as I
from t1_v82_cmode_interactions import S,B,N,C,A,ROOT
import t1_v83_congest_refine as R

OUT=ROOT/'work/results/t1_v83_flow_continue'
ORIGINAL=1300
ROUNDS=300
ALPHAS=(.5,1.)
LIMIT=60.
PRM=dict(Q.PRM,seed=83083)
features=P.features
predict=Q.predict
PROTOCOL=dict(base='V82 confirmed 0.90333',model='300 additional 63-leaf flow trees after frozen 1300-tree incumbent; original 77 visible-only features',
    original_trees=ORIGINAL,extra_trees=ROUNDS,parameters=PRM,alphas=ALPHAS,flow_limit_per_lane=LIMIT,
    selection='Parity: state and conservative total >0, LWR delta <0, >=7 positive-total corridors in both months; maximize minimum total.',
    confirmation='Freeze alpha before half-month and opposite-month; identical gates in both months.',
    baseline='V76 free speed, V79 free flow, V81/V82 congestion corrections reconstructed in every comparison',
    scope='Free-flow non-blackout flow only; speeds, queues, ODME unchanged',data=I.PROTOCOL['data'],caveat=N.PROTOCOL['caveat'])


def training_progress(env):
    done=env.iteration-env.begin_iteration+1
    if done%100==0:print('new trees',done,'/',env.end_iteration-env.begin_iteration,flush=True)


def prefix_check(model,teacher_path,X,idx,ref):
    old=lgb.Booster(model_file=str(teacher_path));a=old.dump_model();b=model.dump_model(num_iteration=ORIGINAL)
    for k in ('feature_names','objective','tree_info','average_output','pandas_categorical'):assert a.get(k)==b.get(k),k
    jj=idx[np.linspace(0,len(idx)-1,min(8192,len(idx)),dtype=int)]
    e=float(np.max(np.abs(predict(model,X,jj,ORIGINAL)-ref[jj])));assert e<1e-10
    e2=float(np.max(np.abs(predict(model,X,jj)-(ref[jj]+predict(model,X,jj,start=ORIGINAL)))));assert e2<1e-10
    return dict(original_trees_identical=True,rows=len(jj),baseline_max_error=e,reconstruction_max_error=e2)


def predictions(mode):
    path=OUT/f'raw_{mode}.npy'
    if path.exists():return np.load(path)
    meta=np.load(G.OUT/'feature_rows.npz');X=np.load(P.OUT/'X.npy',mmap_mode='r');y=np.load(P.Q.OUT/'flow_labels.npy')
    ref=np.load(Q.OUT/f'pred_{mode}.npy');pred=np.zeros(len(y));audit=[];start=time.time()
    for a in (0,1):
        tr=np.flatnonzero(meta[mode]==a);te=np.flatnonzero(meta[mode]!=a);mp=OUT/f'raw_{mode}_{a}.txt';old=Q.OUT/f'model_{mode}_{a}.txt'
        if mp.exists():model=lgb.Booster(model_file=str(mp))
        else:
            model=lgb.train(PRM,lgb.Dataset(X[tr],y[tr],categorical_feature=[X.shape[1]-1]),num_boost_round=ROUNDS,init_model=str(old),callbacks=[training_progress])
            assert model.num_trees()==ORIGINAL+ROUNDS;model.save_model(str(mp))
        audit.append(dict(mode=mode,fold=a,**prefix_check(model,old,X,te,ref)))
        pred[te]=ref[te]+predict(model,X,te,start=ORIGINAL)
        del model;gc.collect();print('flow fit',mode,a,'seconds',round(time.time()-start),flush=True)
    np.save(path,pred);(OUT/f'prefix_{mode}.json').write_text(json.dumps(audit,indent=2));return pred


def score(mode,raw,alphas):
    meta=np.load(G.OUT/'feature_rows.npz');own=meta['owner'];ref=np.load(Q.OUT/f'pred_{mode}.npy')
    xo,_,z=N.dataset();old=N.heads(xo,z,mode,'old');cm=np.load(S.OUT/f'pred_{mode}.npy');r82=np.load(I.OUT/f'pred_{mode}.npy');rows=[]
    for si,s in enumerate(F.SPLITS):
        anchor=F.baseline(s)
        for pi,p in enumerate(PANELS):
            d=N.load_one(p,s);m=d['free'];ix=own==si*10+pi;cg=d['cong'];ci=z['owner']==si*10+pi;fd=d['fd'];lane=d['lanes']
            v0,q0=N.incumbent_free(s,mode,p,pi,d)
            va,qa=C.corrected(d['pv'][cg],d['pq'][cg],old['r'][ci],old['sstar'][ci],A.CFG,N.LAM)
            v0[cg],q0[cg]=R.transform(va,qa,cm[ci],r82[ci],1.)
            cap=v0[m]*lane[m]*fd['kjl'][d['pos'][m]];qa=anchor[p+'_q'][m]
            reconstructed=np.maximum(0,np.minimum(qa+np.clip(ref[ix]*100,-LIMIT,LIMIT)*lane[m],cap))
            assert np.array_equal(reconstructed,q0[m])
            rw,_=F.LOC.target_weights(d,d['L'],F.LOC.load_local(p,s,'T'),fd);args=(d['ov'],d['oq'],lane,d['reg'],rw)
            st=F.LOC.s_state_w(v0,q0,*args);a0=F.M.t3_score(d['C'],d['T'],d['Dd'],v0,q0,fd)
            for alpha in alphas:
                pred=ref[ix]+alpha*(raw[ix]-ref[ix]);q1=q0.copy()
                q1[m]=np.maximum(0,np.minimum(qa+np.clip(pred*100,-LIMIT,LIMIT)*lane[m],cap))
                ds=F.LOC.s_state_w(v0,q1,*args)-st;dt=F.M.t3_delta(a0,F.M.t3_score(d['C'],d['T'],d['Dd'],v0,q1,fd),d['sc'])
                rows.append(dict(split=s,panel=p,mode=mode,alpha=alpha,dS=ds,dE=dt['dE_est'],dFD=dt['dFD_est'],
                                 dTotal=.35*ds+.05*(dt['dFD_est']-.8*dt['dE_est'])))
            del d;gc.collect()
        print('flow scored',mode,s,flush=True)
    return pd.DataFrame(rows)


def check(mode):
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'protocol.json').write_text(json.dumps(PROTOCOL,indent=2))
    alphas=ALPHAS if mode=='parity' else (json.loads((OUT/'selection.json').read_text())['alpha'],);assert None not in alphas
    df=score(mode,predictions(mode),alphas);df.to_csv(OUT/f'checks_{mode}.csv',index=False);rows=[]
    for (s,alpha),g in df.groupby(['split','alpha']):
        rows.append(dict(split=s,mode=mode,alpha=alpha,positive_corridors=int((g.dTotal>0).sum()),
                         **{k:F.M.family_mean(g.set_index('panel')[k].to_dict()) for k in ('dS','dE','dFD','dTotal')}))
    sm=pd.DataFrame(rows);sm.to_csv(OUT/f'summary_{mode}.csv',index=False)
    passes=lambda g:bool((g.dS>0).all() and (g.dE<0).all() and (g.dTotal>0).all() and (g.positive_corridors>=7).all())
    if mode=='parity':
        eligible=[a for a,g in sm.groupby('alpha') if passes(g)]
        alpha=max(eligible,key=lambda a:sm[sm.alpha==a].dTotal.min()) if eligible else None
        result=dict(alpha=alpha,eligible=eligible);path='selection.json'
    else:result=dict(alpha=alphas[0],passed=passes(sm));path=f'decision_{mode}.json'
    (OUT/path).write_text(json.dumps(result,indent=2));print(sm.to_string(index=False),flush=True);print(result,flush=True)


def full():
    choice=json.loads((OUT/'selection.json').read_text());alpha=choice['alpha'];assert alpha is not None
    for mode in ('half_month','other_month'):
        d=json.loads((OUT/f'decision_{mode}.json').read_text());assert d['passed'] and d['alpha']==alpha
    dest=OUT/'model.txt'
    if dest.exists():return choice
    X=np.load(P.OUT/'X.npy',mmap_mode='r');y=np.load(P.Q.OUT/'flow_labels.npy');mp=OUT/'raw_full.txt'
    if mp.exists():model=lgb.Booster(model_file=str(mp))
    else:
        model=lgb.train(PRM,lgb.Dataset(X,y,categorical_feature=[X.shape[1]-1]),num_boost_round=ROUNDS,init_model=str(Q.OUT/'model.txt'),callbacks=[training_progress])
        assert model.num_trees()==ORIGINAL+ROUNDS;model.save_model(str(mp))
    ix=np.linspace(0,len(y)-1,8192,dtype=int);teacher=lgb.Booster(model_file=str(Q.OUT/'model.txt'));ref=np.zeros(len(y));ref[ix]=predict(teacher,X,ix)
    prefix=prefix_check(model,Q.OUT/'model.txt',X,ix,ref);base=predict(model,X,ix,ORIGINAL);raw=predict(model,X,ix)
    leaves=[t['num_leaves'] for t in model.dump_model()['tree_info']]
    for t in range(ORIGINAL,model.num_trees()):
        for j in range(leaves[t]):model.set_leaf_output(t,j,model.get_leaf_output(t,j)*alpha)
    err=float(np.max(np.abs(predict(model,X,ix)-(base+alpha*(raw-base)))));assert err<1e-10
    model.save_model(str(dest));(OUT/'full_audit.json').write_text(json.dumps(dict(**prefix,scaling_max_error=err),indent=2));return choice


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['parity','half_month','other_month','full']);a=ap.parse_args()
    full() if a.mode=='full' else check(a.mode)
