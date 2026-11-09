"""Fit more detailed flow residual trees after the confirmed V78 ensemble.

Fixed 63 leaves, min leaf 750, 300 new trees. Compare half and full new-tree
updates using six blocked checks. Prefix structure and sampled OOF predictions
must match V78 exactly; inference uses the original 77 visible-only features.
"""
from __future__ import annotations
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[k]='4'
import gc,json,time
import numpy as np
import pandas as pd
import lightgbm as lgb
import t1_v78_flow_step as A
from t1_v78_flow_step import F,G,P,PANELS
from t1_v77_flow_factors import VFY

OUT=P.OUT.parent/'t1_v79_flow_detail'
ORIGINAL=1000
ROUNDS=300
LIMIT=60.
ALPHAS=(.5,1.)
PRM=dict(P.PRM,num_leaves=63,min_data_in_leaf=750,seed=79079)
features=P.features


def predict(model,X,idx,iterations=None,start=0):
    out=np.empty(len(idx))
    for i in range(0,len(idx),65536):
        out[i:i+65536]=model.predict(X[idx[i:i+65536]],num_iteration=iterations,start_iteration=start,num_threads=4)
    return out


def prefix_check(model,path,X,idx,ref):
    teacher=lgb.Booster(model_file=str(path));a=teacher.dump_model();b=model.dump_model(num_iteration=ORIGINAL)
    for k in ('feature_names','objective','tree_info','average_output','pandas_categorical'):assert a.get(k)==b.get(k),k
    jj=idx[np.linspace(0,len(idx)-1,min(8192,len(idx)),dtype=int)]
    err=float(np.max(np.abs(predict(model,X,jj,ORIGINAL)-ref[jj])));assert err<1e-10
    full=predict(model,X,jj);increment=predict(model,X,jj,start=ORIGINAL)
    reconstruction=float(np.max(np.abs(full-(ref[jj]+increment))));assert reconstruction<1e-10
    return dict(original_tree_structures_identical=True,sampled_rows=len(jj),max_baseline_difference=err,
        max_increment_reconstruction_error=reconstruction)


def score(own,raw,mode,alphas=ALPHAS):
    rows=[];vp=np.load(G.OUT/f'context_{mode}.npy');ref=np.load(A.OUT/f'pred_{mode}.npy')
    for si,s in enumerate(F.SPLITS):
        z=F.baseline(s)
        for pi,p in enumerate(PANELS):
            d,lh=F.LOC.load(p,s);m=d['free'];idx=own==si*10+pi;fd=d['fd'];lane=fd['lanes'][d['pos']]
            rw,_=F.LOC.target_weights(d,lh,F.LOC.load_local(p,s,'T'),fd)
            dv=np.zeros(len(m));dv[m]=np.clip(vp[idx],-2,2)
            v=VFY.clamp(p,d['pos'],m,z[p+'_v']+dv,z[p+'_q']);cap=v*lane*fd['kjl'][d['pos']]
            dq=np.zeros(len(m));dq[m]=np.clip(ref[idx]*100,-LIMIT,LIMIT)*lane[m]
            q0=np.where(m,np.maximum(0,np.minimum(z[p+'_q']+dq,cap)),z[p+'_q'])
            st=(d['ov'],d['oq'],lane,d['reg'],rw);s0=F.LOC.s_state_w(v,q0,*st)
            a0=F.M.t3_score(d['C'],d['T'],d['Dd'],v,q0,fd)
            for alpha in alphas:
                pr=ref[idx]+alpha*(raw[idx]-ref[idx]);dq[m]=np.clip(pr*100,-LIMIT,LIMIT)*lane[m]
                q1=np.where(m,np.maximum(0,np.minimum(z[p+'_q']+dq,cap)),z[p+'_q'])
                ds=F.LOC.s_state_w(v,q1,*st)-s0;a1=F.M.t3_score(d['C'],d['T'],d['Dd'],v,q1,fd);dt=F.M.t3_delta(a0,a1,d['sc'])
                rows.append(dict(alpha=alpha,mode=mode,split=s,panel=p,dS_rw=ds,dE=dt['dE_est'],dFD=dt['dFD_est'],
                    dTotal40=.35*ds+.05*(dt['dFD_est']-.8*dt['dE_est'])))
            del d,lh;gc.collect()
    return rows


def compile_model(src,dest,X,idx,alpha):
    model=lgb.Booster(model_file=str(src));base=predict(model,X,idx,ORIGINAL);raw=predict(model,X,idx)
    counts=[t['num_leaves'] for t in model.dump_model()['tree_info']]
    for i in range(ORIGINAL,model.num_trees()):
        for j in range(counts[i]):model.set_leaf_output(i,j,model.get_leaf_output(i,j)*alpha)
    final=predict(model,X,idx);err=float(np.max(np.abs(final-(base+alpha*(raw-base)))))
    assert err<1e-10 and np.array_equal(base,predict(model,X,idx,ORIGINAL))
    model.save_model(str(dest));return dict(model=dest.name,sample_rows=len(idx),max_arithmetic_error=err,original_prefix_identical=True)


def main():
    OUT.mkdir(parents=True,exist_ok=True);t0=time.time();meta=np.load(G.OUT/'feature_rows.npz');own=meta['owner'];y=np.load(P.Q.OUT/'flow_labels.npy')
    X=np.load(P.OUT/'X.npy',mmap_mode='r');rows=[];summary=[];prefix=[];allowed=list(ALPHAS)
    for mode in ('parity','half_month','other_month'):
        fold=meta[mode];ref=np.load(A.OUT/f'pred_{mode}.npy');pp=OUT/f'raw_{mode}.npy'
        if pp.exists():
            raw=np.load(pp);prefix.extend(json.loads((OUT/f'prefix_{mode}.json').read_text()))
        else:
            raw=np.zeros(len(y));audit=[]
            for a in (0,1):
                tr=np.flatnonzero(fold==a);te=np.flatnonzero(fold!=a);mp=OUT/f'raw_{mode}_{a}.txt';teacher=A.OUT/f'model_{mode}_{a}.txt'
                if mp.exists():model=lgb.Booster(model_file=str(mp))
                else:
                    model=lgb.train(PRM,lgb.Dataset(X[tr],y[tr],categorical_feature=[X.shape[1]-1]),
                        num_boost_round=ROUNDS,init_model=str(teacher))
                    assert model.num_trees()==ORIGINAL+ROUNDS;model.save_model(str(mp))
                check=prefix_check(model,teacher,X,te,ref);audit.append(dict(mode=mode,fold=a,**check))
                # The prefix is structurally identical and has cached OOF output.
                # Evaluate only new trees, after checking equality to full inference.
                raw[te]=ref[te]+predict(model,X,te,start=ORIGINAL)
                del model;gc.collect();print('detail fit',mode,a,check,'seconds',round(time.time()-t0),flush=True)
            (OUT/f'prefix_{mode}.json').write_text(json.dumps(audit,indent=2));prefix.extend(audit);np.save(pp,raw)
        rr=pd.DataFrame(score(own,raw,mode));rows.extend(rr.to_dict('records'))
        for (alpha,s),g in rr.groupby(['alpha','split']):
            rec=dict(alpha=alpha,mode=mode,split=s,positive_corridors=int((g.dTotal40>0).sum()),
                **{k:F.M.family_mean(g.set_index('panel')[k].to_dict()) for k in ('dS_rw','dE','dFD','dTotal40')})
            summary.append(rec);print(rec,flush=True)
            if (rec['dS_rw']<=0 or rec['dTotal40']<=0) and alpha in allowed:allowed.remove(alpha)
        pd.DataFrame(rows).to_csv(OUT/'checks.csv',index=False);pd.DataFrame(summary).to_csv(OUT/'summary.csv',index=False)
        if not allowed:break
    sm=pd.DataFrame(summary);alpha=max(allowed,key=lambda a:sm[sm.alpha==a].dTotal40.min()) if allowed else None
    scaling=[]
    if alpha is not None:
        assert len(sm[sm.alpha==alpha])==6
        mp=OUT/'raw_model.txt'
        if not mp.exists():
            model=lgb.train(PRM,lgb.Dataset(X,y,categorical_feature=[X.shape[1]-1]),num_boost_round=ROUNDS,
                init_model=str(A.OUT/'model.txt'))
            assert model.num_trees()==ORIGINAL+ROUNDS;model.save_model(str(mp));del model;gc.collect()
        for mode in ('parity','half_month','other_month'):
            ref=np.load(A.OUT/f'pred_{mode}.npy');raw=np.load(OUT/f'raw_{mode}.npy');np.save(OUT/f'pred_{mode}.npy',ref+alpha*(raw-ref))
            for a in (0,1):
                ii=np.flatnonzero(meta[mode]!=a);idx=ii[np.linspace(0,len(ii)-1,4096,dtype=int)]
                scaling.append(compile_model(OUT/f'raw_{mode}_{a}.txt',OUT/f'model_{mode}_{a}.txt',X,idx,alpha))
        scaling.append(compile_model(mp,OUT/'model.txt',X,np.linspace(0,len(y)-1,8192,dtype=int),alpha))
    (OUT/'feature_alignment.json').write_text((A.OUT/'feature_alignment.json').read_text())
    (OUT/'prefix_audit.json').write_text(json.dumps(prefix,indent=2));(OUT/'scaling_audit.json').write_text(json.dumps(scaling,indent=2))
    (OUT/'decision.json').write_text(json.dumps(dict(alpha=alpha,limit=LIMIT if alpha is not None else None,original_trees=ORIGINAL,total_trees=ORIGINAL+ROUNDS,
        gate='six positive weighted state and conservative total checks against V78; maximize worst gain'),indent=2))
    print('chosen alpha',alpha,'seconds',round(time.time()-t0),flush=True)


if __name__=='__main__':main()
