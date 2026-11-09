"""V81: multiscale visible-state context for congested density reconstruction.

Task 1 is offline. All artificial hidden cells are removed before any smoothing;
queue forecasts are untouched. Compare equal-capacity old/new specialists with
parity selection, then frozen half-month and opposite-month confirmation.
"""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='2'
import argparse,json,time,gc
from functools import lru_cache
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import lightgbm as lgb
from scipy.ndimage import gaussian_filter
import t1_v61_congest as C
import t1_v64_apply as A
import t1_v73_global_factor as F
import t1_v73_factor_verify as VFY
import t1_v76_context_mix as SPEED
import t1_v79_flow_detail as FLOW
from evaluate import ROOT,CLEAN,PANELS,STEPS

OUT=ROOT/'work/results/t1_v81_density_context'
KERNELS=((1,0),(2,0),(4,0),(8,0),(1,1),(2,1),(4,1),(1,3),(2,3),(4,3))
ALPHAS=(.5,1.)
OLD='pooled31_l09'
PRD,PRS,RD,RS,LAM=A.CANDS[OLD]
PRD=dict(PRD,num_threads=2);PRS=dict(PRS,num_threads=2)
PROTOCOL=dict(base='V79 0.90321; original congested specialist is V64 pooled31_l09',
    model='Same 31-leaf density L1 and iso-density L2 heads with multiscale speed/flow/density/occupancy context',
    rounds=[RD,RS],kernels=KERNELS,alphas=ALPHAS,
    scope='Task1 congested, non-blackout rows only; queue/ODME and all other state bytes fixed',
    selection='Parity folds: both months positive reweighted state and conservative total, negative LWR delta, >=7/10 positive-total corridors. Maximize worst total gain.',
    confirmation='Frozen alpha must meet the same criteria in half-month and opposite-month folds for both months.',
    inference_gate='Real-frame FD and minimum confirmed total must remain positive after replacing proxy FD.',
    leakage='All artificially hidden observations removed before feature extraction; no hidden Kaggle labels. Task1 offline visible March and April observations allowed.',
    caveat='Inherited V79 base estimators are existing cross-fitted predictors; only specialist heads are refitted for each new blocked fold. LWR is a proxy, not hidden leaderboard truth.')

def load_one(p,s):
    d,L=C.LOC.load(p,s);d['L']=L;d['cong']=~d['free']
    d['bv'],d['bq']=C.v59_baseline(p,s,d,L)
    a,b,c=C.cell_weights(d,d['bv'],d['bq'])
    Np=C.dens(d['pv'],d['pq'],d['fd'],d['pos']);No=C.dens(d['ov'],d['oq'],d['fd'],d['pos'])
    d['r']=No/np.maximum(Np,1e-6)-1;d['wd']=c*Np
    pql=d['pq']/d['lanes'];oql=d['oq']/d['lanes'];h=a*d['pv']**2+b*pql**2
    d['sstar']=np.clip(-(a*d['pv']*(d['pv']-d['ov'])+b*pql*(pql-oql))/h,-.5,.5);d['ws']=h
    return d

def frame_features(v,q,o,eligible,hidden,fd,tt,pos):
    """Query cells must be invisible. Arithmetic never sees a hidden value."""
    mask=eligible&np.isfinite(v)&np.isfinite(q)&~hidden
    assert not mask[tt,pos].any()
    v=np.where(mask,v,0);q=np.where(mask,q,0);o=np.where(mask&np.isfinite(o),o,0)
    ch=np.stack([v/fd['ffs'],q/fd['lanes']/np.maximum(fd['kcl']*fd['ffs'],1),
        q/fd['lanes']/np.maximum(v,1)/fd['kcl'],o],-1)
    ch=np.where(mask[...,None],ch,0);m=mask.astype(float);out=[]
    for st,sp in KERNELS:
        den=gaussian_filter(m,(st,sp),mode='constant')
        means=gaussian_filter(ch,(st,sp,0),mode='constant')/np.maximum(den[...,None],1e-8)
        mom=gaussian_filter(ch[...,[0,2]]**2,(st,sp,0),mode='constant')/np.maximum(den[...,None],1e-8)
        sd=np.sqrt(np.maximum(mom-means[...,[0,2]]**2,0))
        z=np.column_stack([means[tt,pos],sd[tt,pos],den[tt,pos]])
        z[den[tt,pos]<.01,:6]=np.nan;out.append(z)
    return np.column_stack(out).astype(np.float32)

def extra(p,s,d,hide=True):
    kind='H' if hide else 'T';path=OUT/f'extra_{kind}_{s}_{p}.npz'
    m=d['cong'];pos=d['pos'][m]
    if hide:
        day=d['day'][m];tt=d['C'].t.to_numpy()[m]
    else:
        day=d['key'][m]//1000//288;tt=d['key'][m]//1000%288
    key=(day*288+tt)*1000+pos
    if path.exists():
        z=np.load(path);assert np.array_equal(key,z['key']);return z['X']
    pf=pq.ParquetFile(CLEAN/p/f'{s}.parquet');parts=[];days=[];audits=[]
    for dd in range(pf.num_row_groups):
        ii=np.flatnonzero(day==dd)
        if not len(ii):continue
        tb=pf.read_row_group(dd,columns=['is_score_eligible','speed_obs','flow_obs','occ_obs','weekday'])
        get=lambda c:tb.column(c).to_numpy().reshape(288,d['fd']['n'])
        v=get('speed_obs').astype(float);q=get('flow_obs').astype(float);o=get('occ_obs').astype(float)
        h=np.zeros(v.shape,bool)
        if hide:
            allidx=np.flatnonzero(d['day']==dd);h[d['C'].t.to_numpy()[allidx],d['pos'][allidx]]=True
        z=frame_features(v,q,o,get('is_score_eligible'),h,d['fd'],tt[ii],pos[ii])
        # One full day per panel verifies replacement of every hidden channel.
        if not parts:
            vp=v.copy();qp=q.copy();op=o.copy();vp[h]=1234567;qp[h]=-1234567;op[h]=7654321
            zz=frame_features(vp,qp,op,get('is_score_eligible'),h,d['fd'],tt[ii],pos[ii])
            assert np.array_equal(z,zz,equal_nan=True)
        wd=int(tb.column('weekday')[0].as_py())
        ctx=np.column_stack([np.sin(tt[ii]*2*np.pi/288),np.cos(tt[ii]*2*np.pi/288),np.full(len(ii),wd>=5),pos[ii]/max(d['fd']['n']-1,1)])
        parts.append(np.column_stack([z,ctx]));days.extend(ii.tolist())
    assert np.array_equal(days,np.arange(len(key)))
    X=np.concatenate(parts).astype(np.float32);np.savez_compressed(path,X=X,key=key)
    return X

def dataset():
    path=OUT/'rows.npz'
    if path.exists():
        z=dict(np.load(path));return np.load(OUT/'X_old.npy',mmap_mode='r'),np.load(OUT/'X_new.npy',mmap_mode='r'),z
    old=[];new=[];acc={k:[] for k in ('r','wd','sstar','ws','parity','half_month','other_month','owner')}
    for si,s in enumerate(F.SPLITS):
        for pi,p in enumerate(PANELS):
            d=load_one(p,s);m=d['cong'];ex=extra(p,s,d)
            base=np.column_stack([C.cong_feats(d,d['L'],A.CFG['feats'])[m],np.full(m.sum(),si),np.full(m.sum(),pi)])
            old.append(base.astype(np.float32));new.append(np.column_stack([base[:,:-1],ex,base[:,-1]]).astype(np.float32))
            for k in ('r','wd','sstar','ws'):acc[k].append(d[k][m])
            for k,val in [('parity',d['par'][m]),('half_month',(d['day'][m]>=15).astype(int)),('other_month',np.full(m.sum(),si)),('owner',np.full(m.sum(),si*10+pi))]:acc[k].append(val)
            print('features',s,p,int(m.sum()),new[-1].shape[1],flush=True);del d;gc.collect()
    z={k:np.concatenate(a) for k,a in acc.items()}
    for k in ('wd','ws'):
        for si in (0,1):m=z['other_month']==si;z[k][m]/=z[k][m].mean()
    xo=np.concatenate(old);xn=np.concatenate(new);np.save(OUT/'X_old.npy',xo);np.save(OUT/'X_new.npy',xn);np.savez_compressed(path,**z)
    return xo,xn,z

def old_inputs_exact():
    """Production teacher was fitted on float64 inputs; retain that precision."""
    path=OUT/'X_old_exact.npy'
    if path.exists():return np.load(path,mmap_mode='r')
    parts=[]
    for si,s in enumerate(F.SPLITS):
        for pi,p in enumerate(PANELS):
            d=load_one(p,s);m=d['cong']
            parts.append(np.column_stack([C.cong_feats(d,d['L'],A.CFG['feats'])[m],np.full(m.sum(),si),np.full(m.sum(),pi)]))
            del d;gc.collect()
    x=np.concatenate(parts);assert x.shape==(115148,32);np.save(path,x);return x

def heads(X,z,mode,kind):
    path=OUT/f'pred_{kind}_{mode}.npz'
    if kind=='old' and mode=='parity':
        # The exact incumbent OOF cache is preferable to numerical re-training.
        ref=np.load(ROOT/'work/results/t1_v65_lwrcong/cache/v64cf.npz')
        out={'r':np.r_[ref['rh0'],ref['rh1']], 'sstar':np.r_[ref['sh0'],ref['sh1']]}
        assert len(out['r'])==len(X) and (z['other_month'][:len(ref['rh0'])]==0).all()
        if path.exists():
            rebuilt=np.load(path)
            audit={k:float(np.max(np.abs(rebuilt[k]-out[k]))) for k in out}
            (OUT/'parity_incumbent_rebuild_audit.json').write_text(json.dumps(audit,indent=2))
        return out
    if path.exists():return dict(np.load(path))
    out={}
    for target,weight,prm,rounds in [('r','wd',PRD,RD),('sstar','ws',PRS,RS)]:
        pred=np.zeros(len(X));fold=z[mode]
        for a in (0,1):
            tr=fold==a;mp=OUT/f'model_{kind}_{mode}_{a}_{target}.txt'
            if mp.exists():model=lgb.Booster(model_file=str(mp))
            else:model=C.train(X[tr],z[target][tr],z[weight][tr],prm,rounds);model.save_model(str(mp))
            pred[~tr]=model.predict(X[~tr],num_threads=2)
            print('fit',kind,mode,target,a,flush=True)
        out[target]=pred
    np.savez_compressed(path,**out);return out

@lru_cache(maxsize=2)
def free_inputs(s,mode):
    return (F.baseline(s),np.load(SPEED.OUT/'feature_rows.npz')['owner'],
        np.load(SPEED.OUT/f'context_{mode}.npy'),np.load(FLOW.OUT/f'pred_{mode}.npy'))

def incumbent_free(s,mode,p,pi,d):
    z,own,allv,allq=free_inputs(s,mode);v=z[p+'_v'].copy();q=z[p+'_q'].copy();m=d['free']
    ix=own==F.SPLITS.index(s)*10+pi
    vp=allv[ix];qp=allq[ix]*100
    dv=np.zeros(len(m));dv[m]=np.clip(vp,-2,2);v=VFY.clamp(p,d['pos'],m,v+dv,q)
    lane=d['fd']['lanes'][d['pos']];dq=np.zeros(len(m));dq[m]=np.clip(qp,-60,60)*lane[m]
    q=np.where(m,np.maximum(0,np.minimum(q+dq,v*lane*d['fd']['kjl'][d['pos']])),q)
    return v,q

def score(z,old,new,mode,alphas):
    rows=[]
    for si,s in enumerate(F.SPLITS):
        for pi,p in enumerate(PANELS):
            d=load_one(p,s);m=d['cong'];ix=z['owner']==si*10+pi
            v0,q0=incumbent_free(s,mode,p,pi,d)
            v0[m],q0[m]=C.corrected(d['pv'][m],d['pq'][m],old['r'][ix],old['sstar'][ix],A.CFG,LAM)
            rw,_=C.target_weights(d,C.LOC.load_local(p,s,'T'),d['fd'])
            args=(d['ov'],d['oq'],d['lanes'],d['reg'],rw)
            s0=C.LOC.s_state_w(v0,q0,*args);a0=C.M.t3_score(d['C'],d['T'],d['Dd'],v0,q0,d['fd'])
            for alpha in alphas:
                r=old['r'][ix]+alpha*(new['r'][ix]-old['r'][ix]);sh=old['sstar'][ix]+alpha*(new['sstar'][ix]-old['sstar'][ix])
                v1,q1=v0.copy(),q0.copy();v1[m],q1[m]=C.corrected(d['pv'][m],d['pq'][m],r,sh,A.CFG,LAM)
                ds=C.LOC.s_state_w(v1,q1,*args)-s0
                dt=C.M.t3_delta(a0,C.M.t3_score(d['C'],d['T'],d['Dd'],v1,q1,d['fd']),d['sc'])
                rows.append(dict(split=s,panel=p,mode=mode,alpha=alpha,dS=ds,dE=dt['dE_est'],dFD=dt['dFD_est'],
                    dTotal40=.35*ds+.05*(dt['dFD_est']-.8*dt['dE_est'])))
            del d;gc.collect()
        print('scored',mode,s,flush=True)
    return rows

def check(mode):
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'protocol.json').write_text(json.dumps(PROTOCOL,indent=2))
    if mode=='parity':alphas=ALPHAS
    else:
        decision=json.loads((OUT/'selection.json').read_text());assert decision['alpha'] is not None;alphas=(decision['alpha'],)
    xo,xn,z=dataset();old=heads(xo,z,mode,'old');new=heads(xn,z,mode,'new')
    rows=score(z,old,new,mode,alphas);df=pd.DataFrame(rows);df.to_csv(OUT/f'checks_{mode}.csv',index=False);sm=[]
    for (s,alpha),g in df.groupby(['split','alpha']):
        sm.append(dict(split=s,mode=mode,alpha=alpha,positive_corridors=int((g.dTotal40>0).sum()),
            **{k:C.M.family_mean(g.set_index('panel')[k].to_dict()) for k in ('dS','dE','dFD','dTotal40')}))
    sm=pd.DataFrame(sm);sm.to_csv(OUT/f'summary_{mode}.csv',index=False)
    passes=lambda g:bool((g.dS>0).all() and (g.dE<0).all() and (g.dTotal40>0).all() and (g.positive_corridors>=7).all())
    if mode=='parity':
        eligible=[a for a,g in sm.groupby('alpha') if passes(g)]
        alpha=max(eligible,key=lambda a:sm[sm.alpha==a].dTotal40.min()) if eligible else None
        result=dict(alpha=alpha,eligible=eligible);(OUT/'selection.json').write_text(json.dumps(result,indent=2))
    else:
        result=dict(alpha=alphas[0],passed=passes(sm));(OUT/f'decision_{mode}.json').write_text(json.dumps(result,indent=2))
    print(sm.to_string(index=False),flush=True);print(result,flush=True)

def full():
    for mode in ('half_month','other_month'):assert json.loads((OUT/f'decision_{mode}.json').read_text())['passed']
    xo,xn,z=dataset()
    for target,weight,prm,rounds in [('r','wd',PRD,RD),('sstar','ws',PRS,RS)]:
        path=OUT/f'model_full_{target}.txt'
        if not path.exists():C.train(xn,z[target],z[weight],prm,rounds).save_model(str(path))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('cmd',choices=['parity','half_month','other_month','full']);a=ap.parse_args()
    full() if a.cmd=='full' else check(a.cmd)
