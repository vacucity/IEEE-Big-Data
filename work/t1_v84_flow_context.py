"""V84: occupancy and multiscale local context after the V83 flow teacher."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='4'
import argparse,json,gc,time
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import lightgbm as lgb
from scipy.ndimage import gaussian_filter
import t1_v83_flow_continue as T
from t1_v83_flow_continue import Q,F,G,P,PANELS,N,C,A,I,S,ROOT,R
from evaluate import CLEAN

OUT=ROOT/'work/results/t1_v84_flow_context'
KERNELS=((1,0),(2,0),(4,0),(8,0),(2,1),(4,2),(1,3),(2,5))
ALPHAS=(.5,1.)
EXTRA_CAP=.05 # normalized flow label; 5 vehicles/hour/lane
ROUNDS=180
PRM=dict(objective='regression',learning_rate=.025,num_leaves=31,min_data_in_leaf=2000,
         lambda_l2=50.,max_bin=63,num_threads=4,verbosity=-1,seed=84084,deterministic=True,force_col_wise=True)
PROTOCOL=dict(base='V83 confirmed 0.90334',model='180-tree residual with visible occupancy and local multiscale state context after frozen V83 flow teacher',
    parameters=PRM,rounds=ROUNDS,kernels=KERNELS,alpha_grid=ALPHAS,extra_cap_vph_lane=5.,total_cap_vph_lane=60.,
    selection=T.PROTOCOL['selection'],confirmation=T.PROTOCOL['confirmation'],scope=T.PROTOCOL['scope'],
    data='All artificial hide cells removed from speed, flow and occupancy before smoothing. Task1 offline visible observations only.',
    teacher='Residual targets use V83 teacher fitted on the same training fold, never reverse-OOF teacher.',caveat=N.PROTOCOL['caveat'])


def frame(v,q,o,el,h,fd,tt,pos):
    ok=el&np.isfinite(v)&np.isfinite(q)&np.isfinite(o)&~h
    assert not ok[tt,pos].any()
    vv=np.where(ok,v,0);qq=np.where(ok,q/fd['lanes'],0);oo=np.where(ok,o,0)
    ch=np.stack([qq/100,vv/100,qq/np.maximum(vv,1)/50,oo],-1)
    ch=np.where(ok[...,None],ch,0);mask=ok.astype(float);parts=[]
    for st,sp in KERNELS:
        den=gaussian_filter(mask,(st,sp),mode='constant')
        means=gaussian_filter(ch,(st,sp,0),mode='constant')/np.maximum(den[...,None],1e-8)
        a=np.column_stack([means[tt,pos],den[tt,pos]])
        a[den[tt,pos]<.01,:4]=np.nan;parts.append(a)
    return np.column_stack(parts).astype(np.float32)


def extra(p,s,d,lt,v,q,hide):
    fd=d['fd'];m=d['free'] if hide else ((lt['v51']>F.M.FREE_RATIO*fd['vc'][lt['key']%1000])&~lt['gap'])
    day=d['day'][m].astype(np.int64) if hide else lt['day'][m].astype(np.int64)
    tt=d['C'].t.to_numpy(dtype=np.int64)[m] if hide else (lt['key'][m]//1000%288).astype(np.int64)
    pos=d['pos'][m].astype(np.int64) if hide else (lt['key'][m]%1000).astype(np.int64)
    key=(day*288+tt)*1000+pos;path=OUT/f'extra_{"H" if hide else "T"}_{s}_{p}.npz'
    if path.exists():
        z=np.load(path);assert np.array_equal(key,z['key']);return z['X']
    pf=pq.ParquetFile(CLEAN/p/f'{s}.parquet');x=np.empty((len(day),43),np.float32);written=np.zeros(len(day),bool);audit=None
    for dd in range(pf.num_row_groups):
        ii=np.flatnonzero(day==dd)
        if not len(ii):continue
        tb=pf.read_row_group(dd,columns=['is_score_eligible','speed_obs','flow_obs','occ_obs','weekday'])
        get=lambda c:tb.column(c).to_numpy().reshape(288,fd['n'])
        vv=get('speed_obs').astype(float);qq=get('flow_obs').astype(float);oo=get('occ_obs').astype(float);h=np.zeros(vv.shape,bool)
        if hide:
            jj=np.flatnonzero(d['day']==dd);h[d['C'].t.to_numpy(dtype=np.int64)[jj],d['pos'][jj]]=True
        xx=frame(vv,qq,oo,get('is_score_eligible'),h,fd,tt[ii],pos[ii])
        if hide and audit is None:
            vp=vv.copy();qp=qq.copy();op=oo.copy();vp[h]=1234567.;qp[h]=-7654321.;op[h]=2345678.
            poisoned=frame(vp,qp,op,get('is_score_eligible'),h,fd,tt[ii],pos[ii]);assert np.array_equal(xx,poisoned,equal_nan=True)
            audit=dict(panel=p,split=s,day=int(dd),hidden_cells=int(h.sum()),queried_rows=len(ii),all_channels_poison_invariant=True)
        wd=int(tb.column('weekday')[0].as_py());context=np.column_stack([np.full(len(ii),np.sin(2*np.pi*wd/7)),np.full(len(ii),np.cos(2*np.pi*wd/7)),np.full(len(ii),wd>=5)])
        x[ii]=np.column_stack([xx,context]);written[ii]=True
    assert written.all()
    # Express local flow/speed means relative to the same immutable anchor as the old features.
    x[:,np.arange(0,40,5)]-=q[m,None]/fd['lanes'][pos,None]/100
    x[:,np.arange(1,40,5)]-=v[m,None]/100
    np.savez_compressed(path,key=key,X=x)
    if audit:(OUT/f'poison_{s}_{p}.json').write_text(json.dumps(audit,indent=2))
    return x


def load():
    OUT.mkdir(parents=True,exist_ok=True);meta=np.load(G.OUT/'feature_rows.npz');dest=OUT/'extra.npy'
    if (OUT/'features_ready.json').exists():return meta,np.load(P.OUT/'X.npy',mmap_mode='r'),np.load(dest,mmap_mode='r')
    own=meta['owner'];ex=np.lib.format.open_memmap(dest,mode='w+',dtype=np.float32,shape=(len(own),43));off=0
    for si,s in enumerate(F.SPLITS):
        anchor=F.baseline(s)
        for pi,p in enumerate(PANELS):
            d,lt=F.LOC.load(p,s);xx=extra(p,s,d,lt,anchor[p+'_v'],anchor[p+'_q'],True);end=off+len(xx)
            assert np.all(own[off:end]==si*10+pi) and (own==si*10+pi).sum()==len(xx)
            ex[off:end]=xx;off=end;print('context features',s,p,len(xx),flush=True);del d,lt,xx;gc.collect()
    assert off==len(own);ex.flush();(OUT/'features_ready.json').write_text(json.dumps(dict(rows=len(own),extra_columns=43),indent=2))
    return meta,np.load(P.OUT/'X.npy',mmap_mode='r'),np.load(dest,mmap_mode='r')


def design(base,extra,idx,raw):
    a=np.array(extra[idx],copy=True);a[:,np.arange(0,40,5)]-=raw[:,None]
    return np.column_stack([base[idx,:-1],a,raw,base[idx,-1]]).astype(np.float32)


def predictions(mode):
    path=OUT/f'pred_{mode}.npy'
    if path.exists():return np.load(path)
    meta,x,e=load();ref=np.load(T.OUT/f'raw_{mode}.npy');y=np.load(P.Q.OUT/'flow_labels.npy');pred=np.zeros(len(y));audits=[]
    assert json.loads((T.OUT/'selection.json').read_text())['alpha']==1.
    for a in (0,1):
        tr=np.flatnonzero(meta[mode]==a);te=np.flatnonzero(meta[mode]!=a);mp=OUT/f'model_{mode}_{a}.txt'
        teacher=lgb.Booster(model_file=str(T.OUT/f'raw_{mode}_{a}.txt'))
        jj=te[np.linspace(0,len(te)-1,4096,dtype=int)];error=float(np.max(np.abs(T.predict(teacher,x,jj)-ref[jj])));assert error<1e-10
        if mp.exists():model=lgb.Booster(model_file=str(mp))
        else:
            rp=OUT/f'teacher_train_{mode}_{a}.npy'
            if rp.exists():raw=np.load(rp)
            else:raw=T.predict(teacher,x,tr);np.save(rp,raw)
            xx=design(x,e,tr,raw);model=lgb.train(PRM,lgb.Dataset(xx,y[tr]-raw,categorical_feature=[xx.shape[1]-1]),num_boost_round=ROUNDS,callbacks=[T.training_progress])
            model.save_model(str(mp));del xx,raw;gc.collect()
        for start in range(0,len(te),65536):
            ii=te[start:start+65536];pred[ii]=model.predict(design(x,e,ii,ref[ii]),num_threads=4)
        audits.append(dict(fold=a,teacher_oof_replay_error=error));del model,teacher;gc.collect();print('context fit',mode,a,flush=True)
    np.save(path,pred);(OUT/f'teacher_replay_{mode}.json').write_text(json.dumps(audits,indent=2));return pred


def score(mode,pred,alphas):
    meta=np.load(G.OUT/'feature_rows.npz');own=meta['owner'];ref=np.load(T.OUT/f'raw_{mode}.npy')
    xo,_,z=N.dataset();old=N.heads(xo,z,mode,'old');cm=np.load(S.OUT/f'pred_{mode}.npy');r82=np.load(I.OUT/f'pred_{mode}.npy');rows=[]
    for si,s in enumerate(F.SPLITS):
        anchor=F.baseline(s)
        for pi,p in enumerate(PANELS):
            d=N.load_one(p,s);m=d['free'];ix=own==si*10+pi;cg=d['cong'];ci=z['owner']==si*10+pi;fd=d['fd'];lane=d['lanes']
            v0,q0=N.incumbent_free(s,mode,p,pi,d);va,qa=C.corrected(d['pv'][cg],d['pq'][cg],old['r'][ci],old['sstar'][ci],A.CFG,N.LAM)
            v0[cg],q0[cg]=R.transform(va,qa,cm[ci],r82[ci],1.)
            cap=v0[m]*lane[m]*fd['kjl'][d['pos'][m]];qa=anchor[p+'_q'][m]
            q0[m]=np.maximum(0,np.minimum(qa+np.clip(ref[ix]*100,-60,60)*lane[m],cap))
            rw,_=F.LOC.target_weights(d,d['L'],F.LOC.load_local(p,s,'T'),fd);args=(d['ov'],d['oq'],lane,d['reg'],rw)
            st=F.LOC.s_state_w(v0,q0,*args);a0=F.M.t3_score(d['C'],d['T'],d['Dd'],v0,q0,fd)
            for alpha in alphas:
                raw=ref[ix]+alpha*np.clip(pred[ix],-EXTRA_CAP,EXTRA_CAP);q1=q0.copy()
                q1[m]=np.maximum(0,np.minimum(qa+np.clip(raw*100,-60,60)*lane[m],cap))
                ds=F.LOC.s_state_w(v0,q1,*args)-st;dt=F.M.t3_delta(a0,F.M.t3_score(d['C'],d['T'],d['Dd'],v0,q1,fd),d['sc'])
                rows.append(dict(split=s,panel=p,mode=mode,alpha=alpha,dS=ds,dE=dt['dE_est'],dFD=dt['dFD_est'],dTotal=.35*ds+.05*(dt['dFD_est']-.8*dt['dE_est'])))
            del d;gc.collect()
        print('context scored',mode,s,flush=True)
    return pd.DataFrame(rows)


def check(mode):
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'protocol.json').write_text(json.dumps(PROTOCOL,indent=2))
    alphas=ALPHAS if mode=='parity' else (json.loads((OUT/'selection.json').read_text())['alpha'],);assert None not in alphas
    df=score(mode,predictions(mode),alphas);df.to_csv(OUT/f'checks_{mode}.csv',index=False);rows=[]
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
    meta,x,e=load();idx=np.arange(len(meta['owner']));rp=OUT/'teacher_train_full.npy'
    if rp.exists():raw=np.load(rp)
    else:raw=T.predict(lgb.Booster(model_file=str(T.OUT/'model.txt')),x,idx);np.save(rp,raw)
    xx=design(x,e,idx,raw);y=np.load(P.Q.OUT/'flow_labels.npy')-raw
    lgb.train(PRM,lgb.Dataset(xx,y,categorical_feature=[xx.shape[1]-1]),num_boost_round=ROUNDS,callbacks=[T.training_progress]).save_model(str(mp));return alpha


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('cmd',choices=['features','parity','half_month','other_month','full']);a=ap.parse_args()
    load() if a.cmd=='features' else full() if a.cmd=='full' else check(a.cmd)
