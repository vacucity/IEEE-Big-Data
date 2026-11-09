"""V40: reduce V36 frozen state-residual strength on two proxy-confirmed corridors."""
from __future__ import annotations
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[k]='2'
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import lightgbm as lgb
import task1_gbm as G
import t1_v30_bridge as B
import t1_v30_stack as S
import t1_unet as U
from evaluate import t1_targets,links,slot_of,KEY_BASE
from t1_blend import load_preds

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results'/'t1_v40_half'
PANELS=('D7_I10_W','D7_I405_N')

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    protocol=dict(base='V36',panels=PANELS,old_alpha=1.,new_alpha=.5,
        evidence='t1_v36_transfer visible-cell state proxy, validation and private; no official hidden labels',
        caution='Expected gain is very small; Task 3 effect is unknown; preserve V36 as best confirmed until Kaggle result.')
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
    src=ROOT/'results'/'t1_v36_transfer'
    previous=json.loads((src/'protocol.json').read_text())
    models={t:lgb.Booster(model_file=str(S.OUT/f'model_{t}.txt')) for t in ('v','q','k')}
    assert all(B.sha(S.OUT/f'model_{t}.txt')==previous['frozen_models'][t] for t in models)
    rows=[];start=time.time()
    for split in ('validation','private'):
        base_path=ROOT/'sub'/f'state_{split}_v36transfer.csv'
        gbm_path=ROOT/'sub'/f'state_{split}.csv'
        gf=pd.read_csv(gbm_path,usecols=['panel','timestamp','link_id','speed_kmh','flow_vph'])
        bf=pd.read_csv(base_path,usecols=['panel','timestamp','link_id','speed_kmh','flow_vph'])
        corrections={}
        for panel in PANELS:
            ctx=G.Panel(panel);pf=pq.ParquetFile(U.CLEAN/panel/f'{split}.parquet')
            pp=[load_preds(U.DATA/f'pred_{panel}_{split}{tag}_vp.npz') for tag in B.TAGS]
            gg=gf[gf.panel==panel];bb=bf[bf.panel==panel]
            pos=dict(zip(links(panel).link_id,links(panel).pos))
            key=lambda f:slot_of(f.timestamp)*KEY_BASE+f.link_id.map(pos).to_numpy(np.int64)
            keys=t1_targets(panel,split)['key']
            assert np.array_equal(key(gg),keys) and np.array_equal(key(bb),keys)
            gp=gg[['speed_kmh','flow_vph']].to_numpy()
            orig=bb[['speed_kmh','flow_vph']].to_numpy()
            vs=[];qs=[];ix=0;max_err=0.;blackout=0
            for day in range(pf.num_row_groups):
                tb=pf.read_row_group(day,columns=B.COLS,use_threads=False)
                count=int(tb.column('is_t1_target').to_numpy().sum())
                pairs=[tuple(gp[ix:ix+count,j] for j in (0,1))]+[p[day] for p in pp]
                x,bv,bq,gap,arr=S.features(ctx,tb,pairs)
                target=arr('is_t1_target');lane=np.broadcast_to(ctx.lanes,target.shape)[target]
                fv,fq=B.corrected(x,bv,bq,gap,lane,models,1.)
                hv,hq=B.corrected(x,bv,bq,gap,lane,models,.5)
                old=orig[ix:ix+count]
                err=float(np.max(np.abs(np.column_stack([fv,fq])-old)))
                max_err=max(max_err,err)
                assert err<.002,(split,panel,day,err)
                v=old[:,0]+(hv-fv);q=old[:,1]+(hq-fq)
                v[gap]=old[gap,0];q[gap]=old[gap,1]
                assert np.array_equal(v[gap],old[gap,0]) and np.array_equal(q[gap],old[gap,1])
                assert np.isfinite(v).all() and np.isfinite(q).all() and (v>0).all() and (q>=0).all()
                vs.append(v);qs.append(q);blackout+=int(gap.sum());ix+=count
            assert ix==len(orig)
            v=np.concatenate(vs);q=np.concatenate(qs)
            corrections[panel]=(keys,v,q,pos)
            rows.append(dict(split=split,panel=panel,targets=ix,blackout_unchanged=blackout,
                reconstruct_max_abs=max_err,changed_speed=int(np.count_nonzero(v!=orig[:,0])),
                changed_flow=int(np.count_nonzero(q!=orig[:,1])),
                mean_abs_speed_change=float(np.mean(np.abs(v-orig[:,0]))),
                mean_abs_flow_change=float(np.mean(np.abs(q-orig[:,1])))))
            print(split,panel,'done',ix,'sec',round(time.time()-start),flush=True)
        del gf,bf
        path=ROOT/'sub'/f'state_{split}_v40half.csv'
        if path.exists():raise FileExistsError(path)
        first=True
        for c in pd.read_csv(base_path,chunksize=500000):
            for panel,idx in c.groupby('panel').indices.items():
                if panel not in corrections:continue
                keys,v,q,pos=corrections[panel]
                kk=slot_of(c.timestamp.iloc[idx])*KEY_BASE+c.link_id.iloc[idx].map(pos).to_numpy(np.int64)
                jj=np.searchsorted(keys,kk)
                assert np.array_equal(keys[jj],kk)
                c.loc[c.index[idx],'speed_kmh']=v[jj]
                c.loc[c.index[idx],'flow_vph']=q[jj]
            c.to_csv(path,index=False,mode='w' if first else 'a',header=first)
            first=False
        print('wrote',path,flush=True)
    changes=pd.DataFrame(rows)
    changes.to_csv(OUT/'official_changes.csv',index=False)
    (OUT/'official_manifest.json').write_text(json.dumps(dict(protocol=protocol,
        changes=rows,output_sha256={s:B.sha(ROOT/'sub'/f'state_{s}_v40half.csv') for s in ('validation','private')}),indent=2))

if __name__=='__main__':main()
