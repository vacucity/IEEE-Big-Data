"""Frozen-model residual calibration on strictly separate OOF months.

Base models exclude every even month. Fit the small calibrator on 6/8/10,
select strength on December, confirm on February. No Kaggle split is read.
This writes research artifacts only; CPU is capped at two threads.
"""
from __future__ import annotations
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[k]='2'
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
import pyarrow as pa
import pyarrow.parquet as pq
import task1_gbm as G
from evaluate import CLEAN, links, REGIMES
from t1_blend import UNET,load_preds

pa.set_cpu_count(2); pa.set_io_thread_count(2)
OUT=Path(__file__).resolve().parent/'results/t1_v30_stack'
PANELS=('D12_I405_S','D7_I210_W','D12_I5_S','D7_I10_E')
FIT_MONTHS=(6,8,10)
SEED=30531
PARAMS=dict(objective='regression',learning_rate=.04,num_leaves=31,
    min_data_in_leaf=400,lambda_l2=10,feature_fraction=.9,
    bagging_fraction=.8,bagging_freq=1,num_threads=2,verbose=-1,seed=SEED,
    deterministic=True,force_col_wise=True)
ROUNDS=250
COLS=list(dict.fromkeys(G.DAY_COLS+['occ_obs','speed_true','flow_true','is_t2_blackout',
    'is_score_eligible','on_ramp_n','on_ramp_n_valid','off_ramp_n','off_ramp_n_valid']))

def shift(a,dt,dx):
    out=np.full_like(a,np.nan)
    t0,t1=max(0,dt),min(len(a),len(a)+dt)
    x0,x1=max(0,dx),min(a.shape[1],a.shape[1]+dx)
    out[t0:t1,x0:x1]=a[t0-dt:t1-dt,x0-dx:x1-dx]
    return out

def features(ctx,tb,preds):
    d=G.day_features(ctx,tb,False)
    x=d['X'].copy()
    n=ctx.n
    arr=lambda c:tb.column(c).to_numpy().reshape(288,n)
    target=arr('is_t1_target')
    gap=arr('is_t2_blackout')[target]
    pairs=preds
    bv=.1*pairs[0][0]+.3*pairs[1][0]+.6*pairs[2][0]
    bq=.2*pairs[0][1]+.1*pairs[1][1]+.7*pairs[2][1]
    bv[gap]=.5*(pairs[1][0][gap]+pairs[2][0][gap])
    bq[gap]=.5*(pairs[1][1][gap]+pairs[2][1][gap])
    lane=d['lanes']
    bk=bq/np.maximum(bv,1)/lane
    x['ensemble_v']=bv; x['ensemble_q_l']=bq/lane; x['ensemble_k_l']=bk
    for tag,(pv,pq_) in zip(('gbm','unet','lwr'),pairs):
        x[tag+'_v_res']=pv-bv
        x[tag+'_q_res']=(pq_-bq)/lane
        x[tag+'_k_res']=pq_/np.maximum(pv,1)/lane-bk
    occ=arr('occ_obs').astype(float)
    a,b,c,e,_,_=G.nearest_along(occ,np.arange(288),0)
    x['occ_prev']=a[target];x['occ_next']=b[target]
    x['occ_interp']=G.lin(a,b,c,e)[target]
    fullv=arr('speed_obs').astype(float);fullq=arr('flow_obs').astype(float)
    fullv[target]=bv;fullq[target]=bq
    fullk=fullq/np.maximum(fullv,1)/ctx.lanes
    for field,a in [('v',fullv),('q_l',fullq/ctx.lanes),('k_l',fullk)]:
        af=a[:,ctx.perm]
        for dt,dx in ((1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)):
            x[f'{field}_t{dt}_x{dx}']=shift(af,dt,dx)[:,ctx.inv][target]
    x=x.replace([np.inf,-np.inf],np.nan).astype(np.float32)
    return x,bv,bq,gap,arr

def iterate(panel,months):
    ctx=G.Panel(panel)
    meta=json.loads((UNET/f'{panel}_train_meta.json').read_text())
    pp=[load_preds(UNET/f'{prefix}_{panel}_fold0{tag}.npz') for prefix,tag in
        [('gbm_oof',''),('oof','_w48'),('oof','_w48lwr')]]
    pf=pq.ParquetFile(CLEAN/panel/'train.parquet')
    for day in sorted(pp[0]):
        mo=meta['month'][day]
        if mo not in months:continue
        tb=pf.read_row_group(day,columns=COLS,use_threads=False)
        x,bv,bq,gap,arr=features(ctx,tb,[p[day] for p in pp])
        yield day,mo,x,bv,bq,gap,arr,ctx

def stats(pv,pq_,arr,ctx):
    target=arr('is_t1_target')
    tv,tq=arr('speed_true').astype(float),arr('flow_true').astype(float)
    v,q=tv.copy(),tq.copy();v[target]=pv;q[target]=pq_
    use=arr('is_score_eligible')&np.isfinite(tv)&np.isfinite(tq)
    m=target&use
    L=links(ctx.name);ffs=L.free_speed_kmh.to_numpy();cap=L.capacity_vph.to_numpy()
    length=L.length_km.to_numpy();lane=L.lanes.to_numpy()
    kc=cap/ffs;kj=np.maximum(L.k_jam.to_numpy(),1.05*kc)
    k=q/np.maximum(v,1)
    qfd=np.minimum(ffs*k,cap/(kj-kc)*np.maximum(kj-k,0))
    dt=np.diff(tq/np.maximum(tv,1)*length,axis=0)
    dp=np.diff(k*length,axis=0)
    ok=use[:-1]&use[1:]&((arr('on_ramp_n')==0)|(arr('on_ramp_n_valid')>0))[:-1]&((arr('off_ramp_n')==0)|(arr('off_ramp_n_valid')>0))[:-1]
    return dict(n=int(m.sum()),sv=float(np.square(v-tv)[m].sum()),
        sq=float(np.square((q-tq)/lane)[m].sum()),
        fdn=float(np.abs((q-qfd)/lane)[use].sum()),fdd=float(np.abs(q/lane)[use].sum()),
        ln=float(np.abs(dp-dt)[ok].sum()),ld=float(np.abs(dt)[ok].sum()))

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    start=time.time()
    protocol=dict(seed=SEED,panels=PANELS,fit_months=FIT_MONTHS,selection_month=12,
        confirmation_month=2,base_training_months=[7,9,11,1],base_fold=0,
        sample_rate=.12,rounds=ROUNDS,params=PARAMS,
        caveat='T3 uses observed truth dN proxy with oracle context, not official hidden conservation RHS; baseline is non-adapted fold0, not full V28.')
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
    parts=[]
    trainpath=OUT/'train.parquet'
    if not trainpath.exists():
        for ip,panel in enumerate(PANELS):
            for day,mo,x,bv,bq,gap,arr,ctx in iterate(panel,FIT_MONTHS):
                tg=arr('is_t1_target')
                tv,tq=arr('speed_true')[tg],arr('flow_true')[tg]
                lane=np.broadcast_to(ctx.lanes,tg.shape)[tg]
                keep=(np.random.default_rng(SEED+1000*ip+day).random(len(x))<.12)&~gap&np.isfinite(tv)&np.isfinite(tq)
                part=x.loc[keep].copy()
                part['yv']=(tv-bv)[keep];part['yq']=((tq-bq)/lane)[keep]
                part['yk']=(tq/np.maximum(tv,1)/lane-bq/np.maximum(bv,1)/lane)[keep]
                parts.append(part)
            print('features',panel,'rows',sum(len(z) for z in parts),'sec',round(time.time()-start),flush=True)
        train=pd.concat(parts,ignore_index=True);train.to_parquet(trainpath,index=False);del parts
    else:train=pd.read_parquet(trainpath)
    names=[c for c in train if c not in ('yv','yq','yk')]
    models={}
    for target in ('v','q','k'):
        path=OUT/f'model_{target}.txt'
        if path.exists():models[target]=lgb.Booster(model_file=str(path))
        else:
            params=PARAMS|({'objective':'regression_l1'} if target=='k' else {})
            models[target]=lgb.train(params,lgb.Dataset(train[names],train['y'+target]),num_boost_round=ROUNDS)
            models[target].save_model(str(path))
        print('fitted',target,'sec',round(time.time()-start),flush=True)
    del train
    rows=[]
    for panel in PANELS:
        for day,mo,x,bv,bq,gap,arr,ctx in iterate(panel,(12,2)):
            dv,dq,dk=(models[t].predict(x,num_threads=2) for t in ('v','q','k'))
            dv[gap]=0;dq[gap]=0;dk[gap]=0
            tg=arr('is_t1_target');lane=np.broadcast_to(ctx.lanes,tg.shape)[tg]
            for variant in ('baseline','vq_0.25','vq_0.5','vq_1','vqk_0.25','vqk_0.5','vqk_1','k_0.25','k_0.5','k_1'):
                if variant=='baseline':v,q=bv,bq
                else:
                    kind,ss=variant.split('_');a=float(ss)
                    v=bv+a*dv if kind!='k' else bv.copy()
                    q=bq+a*dq*lane if kind!='k' else bq.copy()
                    v=np.clip(v,5,120.75);q=np.maximum(q,0)
                    if kind in ('k','vqk'):
                        kk=np.maximum(bq/np.maximum(bv,1)+a*dk*lane,1e-3)
                        # Closest v/q point in Task1 score units at the calibrated density.
                        wv,wq=.54/25**2,.46/(600*lane)**2
                        v=np.clip((wv*v+wq*kk*q)/(wv+wq*kk**2),5,120.75)
                        q=kk*v
                rows.append(dict(panel=panel,day=day,month=mo,regime=REGIMES.index(arr('mask_regime')[0,0]),
                    variant=variant,**stats(v,q,arr,ctx)))
        print('evaluated',panel,'sec',round(time.time()-start),flush=True)
    raw=pd.DataFrame(rows);raw.to_csv(OUT/'days.csv',index=False)
    sums=raw.groupby(['month','panel','regime','variant'])[['n','sv','sq','fdn','fdd','ln','ld']].sum()
    sums['rmse_v']=np.sqrt(sums.sv/sums.n);sums['rmse_q_l']=np.sqrt(sums.sq/sums.n)
    sums['state']=1-.54*np.minimum(sums.rmse_v/25,1)-.46*np.minimum(sums.rmse_q_l/600,1)
    sums['fd']=1-np.minimum(sums.fdn/sums.fdd,1);sums['lwr_proxy']=1-np.minimum(sums.ln/sums.ld+.045,1)
    sums['physics_proxy']=(sums.fd+2*sums.lwr_proxy)/3
    sums['mixed_proxy']=.35*sums.state+.15*sums.physics_proxy
    sums.to_csv(OUT/'panel_regimes.csv')
    metrics=['rmse_v','rmse_q_l','state','physics_proxy','mixed_proxy']
    panels=sums.groupby(['month','panel','variant'])[metrics].mean().reset_index()
    panels.to_csv(OUT/'panels.csv',index=False)
    summary=panels.groupby(['month','variant'])[metrics].mean().reset_index()
    for k in ('state','physics_proxy','mixed_proxy'):
        baseline=summary[summary.variant=='baseline'].set_index('month')[k]
        summary['delta_'+k]=summary[k]-summary.month.map(baseline)
    summary.to_csv(OUT/'summary.csv',index=False)
    print(summary.to_string(index=False),flush=True)

if __name__=='__main__':main()
