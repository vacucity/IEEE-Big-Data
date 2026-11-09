"""Apply the frozen V31 calibrator to newly confirmed corridors of V33 state.

Reads visible official inputs and cached production predictions only. The
predeclared transfer selection plus independent private confirmation is required.
Existing V31-corrected corridors and T2/ODME files remain unchanged.
"""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='2'
import hashlib
import json
import time
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import lightgbm as lgb
import t1_v30_bridge as B
import t1_v30_stack as S
import t1_unet as U
import task1_gbm as G
from evaluate import PANELS,t1_targets,links,slot_of,KEY_BASE
from t1_blend import load_preds

OUT=U.ROOT/'work/results/t1_v36_transfer'

def main():
    selection=json.loads((OUT/'validation_selection.json').read_text())
    confirmation=json.loads((OUT/'confirmation.json').read_text())
    protocol=json.loads((OUT/'protocol.json').read_text())
    assert selection['selected']==confirmation['validation_selected']
    alpha=1.0;chosen=confirmation['private_confirmed']
    assert chosen and set(chosen).isdisjoint(S.PANELS)
    assert set(chosen).issubset(set(selection['selected']))
    models={t:lgb.Booster(model_file=str(S.OUT/f'model_{t}.txt')) for t in ('v','q','k')}
    assert all(B.sha(S.OUT/f'model_{t}.txt')==protocol['frozen_models'][t] for t in models)
    rows=[];start=time.time()
    for split in ('validation','private'):
        gbm_path=U.ROOT/f'work/sub/state_{split}.csv'
        base_path=U.ROOT/f'work/sub/state_{split}_v30stack.csv'
        # Loading only scalar prediction columns is modest relative to the full CSV.
        gf=pd.read_csv(gbm_path,usecols=['panel','speed_kmh','flow_vph','timestamp','link_id'])
        bf=pd.read_csv(base_path,usecols=['panel','speed_kmh','flow_vph','timestamp','link_id'])
        corrections={}
        for panel in chosen:
            ctx=G.Panel(panel);pf=pq.ParquetFile(U.CLEAN/panel/f'{split}.parquet')
            pp=[load_preds(U.DATA/f'pred_{panel}_{split}{tag}_vp.npz') for tag in B.TAGS]
            gg=gf[gf.panel==panel];bb=bf[bf.panel==panel]
            pos=dict(zip(links(panel).link_id,links(panel).pos))
            key=lambda f:slot_of(f.timestamp)*KEY_BASE+f.link_id.map(pos).to_numpy(np.int64)
            keys=t1_targets(panel,split)['key']
            assert np.array_equal(key(gg),keys) and np.array_equal(key(bb),keys),panel
            gp=gg[['speed_kmh','flow_vph']].to_numpy();bp=bb[['speed_kmh','flow_vph']].to_numpy()
            vs=[];qs=[];ix=0;max_reconstruct=0.;blackout=0
            for day in range(pf.num_row_groups):
                tb=pf.read_row_group(day,columns=B.COLS,use_threads=False)
                count=int(tb.column('is_t1_target').to_numpy().sum())
                pairs=[tuple(gp[ix:ix+count,j] for j in (0,1))]+[p[day] for p in pp]
                x,bv,bq,gap,arr=S.features(ctx,tb,pairs)
                reconstructed=np.column_stack([bv,bq]);orig=bp[ix:ix+count]
                err=float(np.max(np.abs(reconstructed-orig)))
                max_reconstruct=max(max_reconstruct,err)
                # Cached U-Net outputs are float32; re-forming their weighted
                # sum can differ from the saved CSV by one float32 flow ULP.
                assert err<0.002,(panel,day,err)
                target=arr('is_t1_target');lane=np.broadcast_to(ctx.lanes,target.shape)[target]
                v,q=B.corrected(x,bv,bq,gap,lane,models,alpha)
                # Preserve the exact existing submission baseline, including
                # its last round-off bit, while applying the learned delta.
                v+=orig[:,0]-bv;q+=orig[:,1]-bq
                v[gap]=orig[gap,0];q[gap]=orig[gap,1]
                assert np.array_equal(v[gap],orig[gap,0]) and np.array_equal(q[gap],orig[gap,1])
                assert np.isfinite(v).all() and np.isfinite(q).all() and (q>=0).all()
                vs.append(v);qs.append(q);blackout+=int(gap.sum());ix+=count
            v=np.concatenate(vs);q=np.concatenate(qs)
            assert ix==len(bp)
            corrections[panel]=(keys,v,q,pos)
            rows.append(dict(split=split,panel=panel,targets=ix,blackout_unchanged=blackout,
                production_reconstruction_max_abs=max_reconstruct,
                mean_abs_speed_change=float(np.mean(np.abs(v-bp[:,0]))),
                mean_abs_flow_change=float(np.mean(np.abs(q-bp[:,1]))),
                max_abs_speed_change=float(np.max(np.abs(v-bp[:,0]))),
                max_abs_flow_change=float(np.max(np.abs(q-bp[:,1])))))
            print(split,panel,'calibrated',ix,'sec',round(time.time()-start),flush=True)
        del gf,bf
        path=U.ROOT/f'work/sub/state_{split}_v36transfer.csv'
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
    pd.DataFrame(rows).to_csv(OUT/'official_changes.csv',index=False)
    manifest=dict(selection=selection,confirmation=confirmation,protocol=protocol,
        output_files={s:dict(path=str(U.ROOT/f'work/sub/state_{s}_v36transfer.csv'),
            sha256=B.sha(U.ROOT/f'work/sub/state_{s}_v36transfer.csv')) for s in ('validation','private')},
        code_sha256={p.name:B.sha(p) for p in (S.Path(__file__),S.Path(B.__file__),S.Path(S.__file__))})
    (OUT/'official_manifest.json').write_text(json.dumps(manifest,indent=2))

if __name__=='__main__':main()
