"""Validate a larger step for only the new 350 trees, relative to V77.

The alpha=2 candidate was added after observing alpha=1 OOF residuals. It must
improve both weighted state and conservative total over alpha=1 in ALL six
groups. Compile the accepted scale into the last 350 trees' leaf outputs and
check inference against the arithmetic combination, keeping the V77 prefix.
"""
from __future__ import annotations
import gc,json,shutil,time
import numpy as np
import pandas as pd
import lightgbm as lgb
import t1_v78_flow_boost as R
from t1_v78_flow_boost import P,F,G,PANELS

OUT=P.OUT.parent/'t1_v78_flow_step'
features=P.features
ALPHA=2.


def compile_model(source,dest,X,idx,alpha):
    model=lgb.Booster(model_file=str(source));xx=X[idx]
    base=model.predict(xx,num_iteration=P.ROUNDS,num_threads=4)
    old=model.predict(xx,num_threads=4)
    trees=model.dump_model()['tree_info']
    counts=[t['num_leaves'] for t in trees];del trees;gc.collect()
    for i in range(P.ROUNDS,model.num_trees()):
        for j in range(counts[i]):model.set_leaf_output(i,j,model.get_leaf_output(i,j)*alpha)
    new=model.predict(xx,num_threads=4);error=float(np.max(np.abs(new-(base+alpha*(old-base)))))
    assert error<1e-10,(str(source),error)
    assert np.array_equal(base,model.predict(xx,num_iteration=P.ROUNDS,num_threads=4))
    model.save_model(str(dest));return dict(model=source.name,sample_rows=len(idx),max_arithmetic_error=error,original_prefix_predictions_identical=True)


def main():
    OUT.mkdir(parents=True,exist_ok=True);t0=time.time()
    assert json.loads((R.OUT/'decision.json').read_text())['limit']==R.LIMIT
    meta=np.load(G.OUT/'feature_rows.npz');own=meta['owner'];rows=[];summaries=[]
    for mode in ('parity','half_month','other_month'):
        old=np.load(P.OUT/f'pred_{mode}.npy');raw=np.load(R.OUT/f'pred_{mode}.npy');pred=old+ALPHA*(raw-old)
        rr=pd.DataFrame(R.score(own,pred,mode));rows.extend(rr.to_dict('records'))
        for s,g in rr.groupby('split'):
            rec=dict(limit=R.LIMIT,mode=mode,split=s,positive_corridors=int((g.dTotal40>0).sum()),
                **{k:F.M.family_mean(g.set_index('panel')[k].to_dict()) for k in ('dS_rw','dE','dFD','dTotal40')})
            summaries.append(rec);print(rec,flush=True)
    sm=pd.DataFrame(summaries);sm.to_csv(OUT/'trial_summary_alpha2.csv',index=False)
    pd.DataFrame(rows).to_csv(OUT/'trial_checks_alpha2.csv',index=False)
    base=pd.read_csv(R.OUT/'summary.csv').set_index(['mode','split']);cur=sm.set_index(['mode','split']).reindex(base.index)
    gain=cur[['dS_rw','dTotal40']]-base[['dS_rw','dTotal40']]
    gain.to_csv(OUT/'increment_over_alpha1.csv')
    alpha=ALPHA if (gain.dS_rw>0).all() and (gain.dTotal40>0).all() else 1.
    if alpha==ALPHA:
        sm.to_csv(OUT/'summary.csv',index=False);pd.DataFrame(rows).to_csv(OUT/'checks.csv',index=False)
    else:
        shutil.copyfile(R.OUT/'summary.csv',OUT/'summary.csv');shutil.copyfile(R.OUT/'checks.csv',OUT/'checks.csv')
    X=np.load(P.OUT/'X.npy',mmap_mode='r');audit=[]
    for mode in ('parity','half_month','other_month'):
        old=np.load(P.OUT/f'pred_{mode}.npy');raw=np.load(R.OUT/f'pred_{mode}.npy')
        np.save(OUT/f'pred_{mode}.npy',old+alpha*(raw-old))
        for a in (0,1):
            ii=np.flatnonzero(meta[mode]!=a);idx=ii[np.linspace(0,len(ii)-1,min(4096,len(ii)),dtype=int)]
            audit.append(compile_model(R.OUT/f'model_{mode}_{a}.txt',OUT/f'model_{mode}_{a}.txt',X,idx,alpha))
    idx=np.linspace(0,len(own)-1,8192,dtype=int)
    audit.append(compile_model(R.OUT/'model.txt',OUT/'model.txt',X,idx,alpha))
    (OUT/'scaling_audit.json').write_text(json.dumps(audit,indent=2))
    for name in ('prefix_audit.json','feature_alignment.json'):
        shutil.copyfile(R.OUT/name,OUT/name)
    (OUT/'decision.json').write_text(json.dumps(dict(limit=R.LIMIT,alpha=alpha,original_trees=P.ROUNDS,total_trees=P.ROUNDS+R.EXTRA_ROUNDS,
        gate='alpha2 must improve state and conservative total over alpha1 in all six groups; otherwise retain alpha1'),indent=2))
    print('chosen alpha',alpha,'seconds',round(time.time()-t0),flush=True)


if __name__=='__main__':main()
