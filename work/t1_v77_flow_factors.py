"""Extend the successful V76 flow mixer with visible-only corridor flow factors.

Raw and log-flow monthly/daily factors and same-day residual smoothers.
All artificial holdouts are removed before factor construction. Evaluate against
V76, retaining its speed, on disjoint days, half months, and opposite months.
"""
from __future__ import annotations
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[k]='4'
import gc,json,time
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import lightgbm as lgb
from scipy.ndimage import gaussian_filter1d
import t1_v76_flow_mix as Q
from t1_v76_flow_mix import F,G,VFY,PANELS
from evaluate import CLEAN,STEPS

OUT=G.OUT.parent/'t1_v77_flow_factors'
PRM=dict(Q.PRM,num_threads=4,seed=77077)
ROUNDS=650
LIMITS=(30.,60.)


def additive(v,m):
    cnt=m.sum(2);nc=m.sum((0,1));aa=v.sum((0,1))/np.maximum(nc,1)
    for _ in range(8):
        bb=np.sum(np.where(m,v-aa,0),2)/(cnt+2)
        aa=np.sum(np.where(m,v-bb[:,:,None],0),(0,1))/np.maximum(nc,1)
    month=aa[None,None,:]+bb[:,:,None]
    nd=m.sum(1);ad=np.zeros_like(nd,dtype=np.float32)
    for _ in range(6):
        bd=np.sum(np.where(m,v-aa-ad[:,None,:],0),2)/(cnt+2)
        ad=np.sum(np.where(m,v-aa-bd[:,:,None],0),1)/(nd+24)
    daily=aa[None,None,:]+ad[:,None,:]+bd[:,:,None]
    yield month;yield daily
    rr=np.where(m,v-daily,0)
    for sigma in (1.,2.,4.,8.,16.):
        num=gaussian_filter1d(rr,sigma,axis=1,mode='constant')
        den=gaussian_filter1d(m.astype(np.float32),sigma,axis=1,mode='constant')
        yield daily+num/np.maximum(den,.02)


def factors(p,s,day,t,pos,fd,hide):
    day=np.asarray(day,np.int64);t=np.asarray(t,np.int64);pos=np.asarray(pos,np.int64)
    key=(day*STEPS+t)*1000+pos;path=OUT/f'{"H" if hide else "T"}_{s}_{p}.npz'
    if path.exists():
        z=dict(np.load(path));assert np.array_equal(z['key'],key);return z['pred']
    pf=pq.ParquetFile(CLEAN/p/f'{s}.parquet');qq=[];mm=[]
    for dd in range(pf.num_row_groups):
        tb=pf.read_row_group(dd,columns=['is_score_eligible','speed_obs','flow_obs'])
        get=lambda c:tb.column(c).to_numpy().reshape(STEPS,fd['n'])
        v=get('speed_obs').astype(np.float32);q=get('flow_obs').astype(np.float32)/fd['lanes']
        with np.errstate(invalid='ignore',divide='ignore'):
            m=get('is_score_eligible')&np.isfinite(v)&np.isfinite(q)&(q/np.maximum(v,1)/fd['kcl']<.8)
        ii=np.flatnonzero(day==dd)
        if hide:m[t[ii],pos[ii]]=False
        assert not m[t[ii],pos[ii]].any(),('visible target',p,s,dd)
        qq.append(np.where(m,q,0).astype(np.float32));mm.append(m)
    q=np.stack(qq);m=np.stack(mm);vals=[]
    for log in (False,True):
        a=np.log1p(np.maximum(q,0)) if log else q
        for pr in additive(a,m):
            z=pr[day,t,pos]
            vals.append(np.expm1(np.clip(z,0,np.log1p(5000))) if log else z)
    pred=np.column_stack(vals).astype(np.float32)
    assert np.isfinite(pred).all()
    temp=path.with_name(path.stem+'.pending.npz')
    np.savez_compressed(temp,pred=pred,key=key);temp.replace(path);return pred


def extra(p,s,d,lh,v,q,hide):
    c=d['C'] if hide else None;fd=d['fd']
    day=c.day.to_numpy() if hide else lh['day'];t=c.t.to_numpy() if hide else (lh['key']//1000)%288
    pos=c.pos.to_numpy() if hide else lh['key']%1000
    f=factors(p,s,day,t,pos,fd,hide);qpl=q/fd['lanes'][pos]
    den=np.maximum(lh['db']+lh['da'],1)
    ip=(lh['da']*lh['qb']+lh['db']*lh['qa'])/den-qpl
    slope=(lh['qa']-lh['qb'])/den
    return np.column_stack([np.clip((f-qpl[:,None])/100,-10,10),np.clip(ip/100,-10,10),
        np.clip(slope/100,-5,5),qpl/np.maximum(fd['kcl'][pos]*fd['vc'][pos],1)]).astype(np.float32)


def features(p,s,d,lh,v,q,pi,hide):
    x=G.features(p,s,d,lh,v,q,pi,hide);e=extra(p,s,d,lh,v,q,hide)
    return np.column_stack([x[:,:-1],e,x[:,-1]]).astype(np.float32)


def load():
    meta=np.load(G.OUT/'feature_rows.npz');own=meta['owner'];path=OUT/'X.npy'
    if (OUT/'features_ready.json').exists():return meta,np.load(path,mmap_mode='r')
    base=np.load(G.OUT/'X_context.npy',mmap_mode='r');X=None;off=0
    for si,s in enumerate(F.SPLITS):
        z=F.baseline(s)
        for pi,p in enumerate(PANELS):
            d,lh=F.LOC.load(p,s);mask=d['free'];idx=own==si*10+pi;end=off+int(mask.sum())
            assert idx.sum()==mask.sum() and np.all(own[off:end]==si*10+pi)
            e=extra(p,s,d,lh,z[p+'_v'],z[p+'_q'],True)[mask]
            if X is None:X=np.lib.format.open_memmap(path,mode='w+',dtype=np.float32,shape=(len(own),base.shape[1]+e.shape[1]))
            X[off:end,:base.shape[1]-1]=base[off:end,:-1]
            X[off:end,base.shape[1]-1:-1]=e;X[off:end,-1]=base[off:end,-1];off=end
            print('features',s,p,flush=True);del d,lh,e;gc.collect()
    X.flush();assert off==len(own)
    (OUT/'features_ready.json').write_text(json.dumps(dict(rows=len(own),columns=X.shape[1])))
    return meta,X


def score(own,pred,mode):
    rows=[];vp=np.load(G.OUT/f'context_{mode}.npy');qp=np.load(Q.OUT/f'pred_{mode}.npy')*100
    for si,s in enumerate(F.SPLITS):
        z=F.baseline(s)
        for pi,p in enumerate(PANELS):
            d,lh=F.LOC.load(p,s);m=d['free'];idx=own==si*10+pi;fd=d['fd'];lane=fd['lanes'][d['pos']]
            rw,_=F.LOC.target_weights(d,lh,F.LOC.load_local(p,s,'T'),fd)
            dv=np.zeros(len(m));dv[m]=np.clip(vp[idx],-2,2)
            v=VFY.clamp(p,d['pos'],m,z[p+'_v']+dv,z[p+'_q']);qbase=z[p+'_q'];dq=np.zeros(len(m));dq[m]=np.clip(qp[idx],-30,30)*lane[m]
            cap=v*lane*fd['kjl'][d['pos']]
            q0=np.where(m,np.maximum(0,np.minimum(qbase+dq,cap)),qbase)
            st=(d['ov'],d['oq'],lane,d['reg'],rw);s0=F.LOC.s_state_w(v,q0,*st)
            a0=F.M.t3_score(d['C'],d['T'],d['Dd'],v,q0,fd)
            for limit in LIMITS:
                dq[m]=np.clip(pred[idx]*100,-limit,limit)*lane[m]
                q1=np.where(m,np.maximum(0,np.minimum(qbase+dq,cap)),qbase)
                ds=F.LOC.s_state_w(v,q1,*st)-s0
                a1=F.M.t3_score(d['C'],d['T'],d['Dd'],v,q1,fd);dt=F.M.t3_delta(a0,a1,d['sc'])
                rows.append(dict(limit=limit,split=s,panel=p,mode=mode,dS_rw=ds,dE=dt['dE_est'],dFD=dt['dFD_est'],
                    dTotal40=.35*ds+.05*(dt['dFD_est']-.8*dt['dE_est'])))
            del d,lh;gc.collect()
    return rows


def main():
    OUT.mkdir(parents=True,exist_ok=True);t0=time.time();meta,X=load();own=meta['owner'];y=np.load(Q.OUT/'flow_labels.npy')
    rows=[];summary=[];allowed=list(LIMITS)
    for mode in ('parity','half_month','other_month'):
        fold=meta[mode];pp=OUT/f'pred_{mode}.npy'
        if pp.exists():pred=np.load(pp)
        else:
            pred=np.zeros(len(y))
            for a in (0,1):
                tr=fold==a;mp=OUT/f'model_{mode}_{a}.txt'
                if mp.exists():mdl=lgb.Booster(model_file=str(mp))
                else:
                    mdl=lgb.train(PRM,lgb.Dataset(X[tr],y[tr],categorical_feature=[X.shape[1]-1]),num_boost_round=ROUNDS)
                    mdl.save_model(str(mp))
                pred[~tr]=mdl.predict(X[~tr],num_threads=4)
                del mdl;gc.collect();print('fit',mode,a,round(time.time()-t0),flush=True)
            np.save(pp,pred)
        rr=pd.DataFrame(score(own,pred,mode));rows.extend(rr.to_dict('records'))
        for (limit,s),g in rr.groupby(['limit','split']):
            rec=dict(limit=limit,mode=mode,split=s,positive_corridors=int((g.dTotal40>0).sum()),
                **{k:F.M.family_mean(g.set_index('panel')[k].to_dict()) for k in ('dS_rw','dE','dFD','dTotal40')})
            summary.append(rec);print(rec,flush=True)
            if (rec['dS_rw']<=0 or rec['dTotal40']<=0) and limit in allowed:allowed.remove(limit)
        pd.DataFrame(rows).to_csv(OUT/'checks.csv',index=False);pd.DataFrame(summary).to_csv(OUT/'summary.csv',index=False)
        if not allowed:break
    sm=pd.DataFrame(summary)
    chosen=max(allowed,key=lambda l:sm[sm.limit==l].dTotal40.min()) if allowed else None
    if chosen:
        assert len(sm[sm.limit==chosen])==6
        mdl=lgb.train(PRM,lgb.Dataset(X,y,categorical_feature=[X.shape[1]-1]),num_boost_round=ROUNDS)
        mdl.save_model(str(OUT/'model.txt'))
    (OUT/'decision.json').write_text(json.dumps(dict(limit=chosen,gate='six positive weighted state and conservative total checks against V76; maximize worst gain'),indent=2))
    print('chosen',chosen,'seconds',round(time.time()-t0),flush=True)


if __name__=='__main__':main()
