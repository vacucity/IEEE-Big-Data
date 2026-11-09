"""Frozen residual calibration on held-out visible cells after honest month adaptation.

Every U-Net is freshly adapted only on even-index days. Odd-index days are
evaluated with an additional 20% of visible measurements hidden before every
base model and residual feature. Calibrators are frozen from train-only OOF.
"""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='2'
import argparse
import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import lightgbm as lgb
import t1_unet as U
import task1_gbm as G
import t1_v30_stack as S
from evaluate import PANELS

OUT=U.ROOT/'work/results/t1_v30_bridge'
SEED=30541
TAGS=('_w48','_w48lwr')
COLS=list(dict.fromkeys(G.DAY_COLS+['occ_obs','is_t2_blackout','is_score_eligible']))
pa.set_cpu_count(2);pa.set_io_thread_count(2)
G.PARAMS['num_threads']=2

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def adapt():
    import torch
    torch.set_num_threads(2)
    OUT.mkdir(parents=True,exist_ok=True)
    protocol=dict(seed=SEED,adaptation_day_index_parity=0,evaluation_day_index_parity=1,
        adaptation_epochs=6,adaptation_lr=.0002,extra_hide_rate=.2,
        base='full GBM plus two-fold w48 and two-fold w48lwr; each U-Net freshly month-adapted on even days only',
        selection='fixed train-OOF selected vqk_1; vqk_0.5 predeclared conservative alternative; validation then private confirmation',
        stack_models={k:sha(S.OUT/f'model_{k}.txt') for k in ('v','q','k')},
        caveat='Artificial hidden visible cells proxy; no actual official T1 target labels are read. Matched four-model architecture, but half-day adaptation differs from full production adaptation.')
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
    for tag in TAGS:
        for fold in (0,1):
            out_tag=tag+'_v30even'
            path=U.MODELS/f't1_unet_fold{fold}{out_tag}.pt'
            if path.exists():continue
            args=SimpleNamespace(fold=fold,width=48,tag=tag,out_tag=out_tag,
                target='valpriv',parity=0,epochs=6,batch=4,lr=.0002,lwr=.05 if 'lwr' in tag else 0.)
            U.cmd_adapt(args)
            torch.cuda.empty_cache()
            print('adapt complete',fold,tag,flush=True)

def replace(tb,name,a):
    idx=tb.schema.get_field_index(name)
    return tb.set_column(idx,name,pa.array(a.ravel(),type=tb.schema.field(name).type))

def masked_table(tb,hide):
    n=hide.shape[1]
    for c in ('speed_obs','flow_obs','occ_obs'):
        a=tb.column(c).to_numpy().reshape(288,n).copy()
        a[hide]=np.nan;tb=replace(tb,c,a)
    target=tb.column('is_t1_target').to_numpy().reshape(288,n)|hide
    return replace(tb,'is_t1_target',target)

def gbm(models,ctx,tb):
    d=G.day_features(ctx,tb,False)
    v,q=G.predict_cells(models,d)
    kk=d['base_q']/np.maximum(d['base_v'],1)+d['lanes']*models['k'].predict(G.add_k_base(d['X']),num_threads=2)
    v,q=G.combine(v,q,kk,G.COMBINE_W)
    return np.clip(v,G.SPEED_MIN,G.SPEED_MAX),q

def corrected(x,bv,bq,gap,lane,models,alpha):
    dv,dq,dk=(models[t].predict(x,num_threads=2) for t in ('v','q','k'))
    dv[gap]=0;dq[gap]=0;dk[gap]=0
    v=np.clip(bv+alpha*dv,5,120.75);q=np.maximum(bq+alpha*dq*lane,0)
    kk=np.maximum(bq/np.maximum(bv,1)+alpha*dk*lane,1e-3)
    wv,wq=.54/25**2,.46/(600*lane)**2
    v=np.clip((wv*v+wq*kk*q)/(wv+wq*kk**2),5,120.75)
    q=kk*v
    # Exact identity on real Task 2 blackout targets: no extrapolation of stack.
    v[gap]=bv[gap];q[gap]=bq[gap]
    return v,q

def evaluate(split, max_days=None, panels=None):
    import torch
    torch.set_num_threads(2)
    OUT.mkdir(parents=True,exist_ok=True)
    models={(tag,fold):U.load_model(fold,48,tag+'_v30even') for tag in TAGS for fold in (0,1)}
    gm={t:lgb.Booster(model_file=str(G.MODELS/f't1_gbm_{t}.txt')) for t in ('v','q','k')}
    sm={t:lgb.Booster(model_file=str(S.OUT/f'model_{t}.txt')) for t in ('v','q','k')}
    rows=[];start=time.time()
    panel_list=panels or PANELS
    for ip,panel in enumerate(panel_list):
        ctx=G.Panel(panel);src=U.open_split(panel,split);st=U.static(panel)
        prof=np.load(U.DATA/f'{panel}_profiles.npy');pf=pq.ParquetFile(U.CLEAN/panel/f'{split}.parquet')
        odd_days=list(range(1,pf.num_row_groups,2))
        if max_days is not None and len(odd_days)>max_days:
            picks=np.linspace(0,len(odd_days)-1,max_days).round().astype(int)
            odd_days=[odd_days[i] for i in picks]
        for day in odd_days:
            ud=U.Day(panel,split,day,src,st,prof)
            tb=pf.read_row_group(day,columns=COLS,use_threads=False)
            vis=np.isfinite(tb.column('speed_obs').to_numpy()).reshape(288,ctx.n)
            vis&=np.isfinite(tb.column('flow_obs').to_numpy()).reshape(288,ctx.n)
            hide=(np.random.default_rng(SEED+1000*ip+day).random(vis.shape)<.2)&vis
            mt=masked_table(tb,hide)
            target=mt.column('is_t1_target').to_numpy().reshape(288,ctx.n)
            pairs=[gbm(gm,ctx,mt)]
            for tag in TAGS:
                ps=[U.predict_day(models[tag,fold],ud,hide[:,st['perm']]) for fold in (0,1)]
                pairs.append(tuple(np.mean([p[j] for p in ps],axis=0)[:,ctx.inv][target] for j in (0,1)))
            x,bv,bq,gap,arr=S.features(ctx,mt,pairs)
            lane=np.broadcast_to(ctx.lanes,target.shape)[target]
            use=hide[target]
            tv=tb.column('speed_obs').to_numpy().reshape(288,ctx.n)[hide]
            tq=tb.column('flow_obs').to_numpy().reshape(288,ctx.n)[hide]
            for alpha in (0.,.5,1.):
                v,q=(bv,bq) if alpha==0 else corrected(x,bv,bq,gap,lane,sm,alpha)
                v,q=v[use],q[use];ll=lane[use]
                rows.append(dict(split=split,panel=panel,day=day,regime=ud.regime,alpha=alpha,
                    n=len(tv),sv=float(np.square(v-tv).sum()),sq=float(np.square((q-tq)/ll).sum()),
                    abs_k=float(np.abs(q/np.maximum(v,1)/ll-tq/np.maximum(tv,1)/ll).sum())))
        print(split,panel,'complete',round(time.time()-start),'sec',flush=True)
    raw=pd.DataFrame(rows);raw.to_csv(OUT/f'{split}_days.csv',index=False)
    pr=raw.groupby(['split','panel','regime','alpha'])[['n','sv','sq','abs_k']].sum()
    pr['rv']=np.sqrt(pr.sv/pr.n);pr['rq']=np.sqrt(pr.sq/pr.n);pr['mae_k']=pr.abs_k/pr.n
    pr['state']=1-.54*pr.rv/25-.46*pr.rq/600
    pr.to_csv(OUT/f'{split}_panel_regimes.csv')
    panels=pr.groupby(['split','panel','alpha'])[['rv','rq','mae_k','state']].mean().reset_index()
    panels.to_csv(OUT/f'{split}_panels.csv',index=False)
    summary=panels.groupby(['split','alpha'])[['rv','rq','mae_k','state']].mean().reset_index()
    summary['delta_state']=summary.state-float(summary.loc[summary.alpha==0,'state'].iloc[0])
    summary.to_csv(OUT/f'{split}_summary.csv',index=False)
    print(summary.to_string(index=False),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('cmd',choices=['adapt','validation','private'])
    ap.add_argument('--max-days',type=int,default=None)
    ap.add_argument('--panels',nargs='+',default=None)
    args=ap.parse_args()
    adapt() if args.cmd=='adapt' else evaluate(args.cmd,args.max_days,args.panels)
