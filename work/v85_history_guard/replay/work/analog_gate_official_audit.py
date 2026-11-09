"""Audit official application of the historical-evidence gate, no scoring/submissions."""
from pathlib import Path
import os,sys,json,zlib,time,gc
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='3'
import numpy as np,pandas as pd,pyarrow.parquet as pq
ROOT=Path(__file__).resolve().parents[1];W=ROOT/'work';OUT=Path(os.environ['V85_HANDOFF_AUDIT'])
sys.path[:0]=[str(W),str(W/'v85_project/work')]
import test_historical_analogs as H
import v85_run_decisions as E
import v85_decoders as D
import t2_v52_selrule as S
from evaluate import T2_PANELS,links
q=E.submitted();rows=[];cellrows=[]
for panel in T2_PANELS:
 tick=time.time();L=links(panel);vc=L.v_cut_kmh.to_numpy(np.float32);n=len(L)
 bins=[a for a in np.array_split(np.arange(n),min(16,n)) if len(a)]
 tf,tq,tvalid,dates=H.load_train(panel,vc,bins,return_dates=True);assert max(dates)<'2031-03-01'
 for split in ['validation','private']:
  p,z=E.probability(panel,split,'official');ctx=S.official_classes(panel,split).set_index('window_id',verify_integrity=True)
  pf=pq.ParquetFile(H.SRC/panel/(split+'.parquet'))
  dmap={str(pf.read_row_group(i,columns=['date']).column('date')[0].as_py()):i for i in range(pf.num_row_groups)}
  for j,wid0 in enumerate(z['ids']):
   wid=str(wid0);c=ctx.loc[wid];T=int(c['T']);cl=str(c['cls']);pp=p[j]
   p0=D.ratio(pp);replay=p0.copy();q0=pp;acc=None;cpu_add=np.zeros_like(replay);filtered=np.zeros_like(replay)
   counts=np.zeros_like(pp,dtype=np.int16);analog=np.zeros_like(pp);history_days=0
   dissolve=cl=='constr' and panel=='D7_I210_E' and T//12>=17
   if cl=='constr':
    ids=S.coupling_ids(panel,pp,'group');Y,acc=D.samples(pp,ids,zlib.crc32(wid.encode('ascii')),constrained=True)
    if dissolve:replay=S.rule_dissolve(p0)
    elif acc>=.02:q0=Y.mean(0).astype(np.float32);replay=D.ratio(q0);replay[0]=p0[0]
   official=E.official_base(q,panel,split,wid,pp.shape)
   if cl=='constr' and not dissolve and acc>=.02:
    t=pf.read_row_group(dmap[str(c['date'])],columns=['date','pos','speed_clean','is_score_eligible'])
    assert t.num_rows==288*n and np.array_equal(t['pos'].to_numpy(),np.tile(np.arange(n),288))
    raw=t['speed_clean'].to_numpy().reshape(288,n).astype(np.float32);e=t['is_score_eligible'].to_numpy().reshape(288,n).astype(bool)
    v=np.full((1,288,n),np.nan,np.float32);m=np.zeros((1,288,n),bool);v[0,:T+1]=raw[:T+1];m[0,:T+1]=e[:T+1]
    query=H.block_features(H.compact(v,m,vc,bins))[0,T-11]
    times=np.arange(max(11,T-12),min(281,T+12)+1);candidates=tf[:,times-11,:].reshape(-1,tf.shape[-1])
    distances=np.mean((candidates-query)**2,axis=1);ix=np.argpartition(distances,H.K-1)[:H.K];di,ti=ix//len(times),times[ix%len(times)]
    history_days=len(set(dates[x] for x in di));h=np.arange(1,7)[None,:];obs=tvalid[di[:,None],ti[:,None]+h];yy=tq[di[:,None],ti[:,None]+h]
    counts=obs.sum(0);analog=np.divide((yy&obs).sum(0),counts,out=q0.copy(),where=counts>0).astype(np.float32)
    candidate=replay^D.ratio(.9*q0+.1*analog)^D.ratio(q0);candidate[0]=replay[0]
    cpu_add=(~replay)&candidate;filtered=cpu_add&(counts>=20)&((analog-q0)>=.3)
   effective_all=cpu_add&(~official);effective_gate=filtered&(~official)
   assert not (effective_gate & ~effective_all).any()
   rows.append(dict(panel=panel,split=split,window_id=wid,date=str(c['date']),T=T,cls=cl,acceptance=acc,history_days=history_days,
                    replay_mismatch=int((official!=replay).sum()),ungated_suggested=int(cpu_add.sum()),gated_suggested=int(filtered.sum()),
                    ungated_official_changes=int(effective_all.sum()),gated_official_changes=int(effective_gate.sum())))
   for h,pos in np.argwhere(effective_all):
    cellrows.append(dict(panel=panel,split=split,window_id=wid,date=str(c['date']),T=T,horizon_minutes=int((h+1)*5),
      link_id=str(L.link_id.iloc[pos]),position=int(pos),before=0,after=1,gated=bool(effective_gate[h,pos]),
      valid_history=int(counts[h,pos]),history_probability=float(analog[h,pos]),cpu_probability=float(q0[h,pos]),
      probability_gap=float(analog[h,pos]-q0[h,pos]),baseline_replay_mismatch=bool(official[h,pos]!=replay[h,pos])))
  del p,z
 print(panel,'done',round(time.time()-tick,1),'s',flush=True)
 del tf,tq,tvalid;gc.collect()
d=pd.DataFrame(rows);c=pd.DataFrame(cellrows,columns=['panel','split','window_id','date','T','horizon_minutes','link_id','position','before','after','gated','valid_history','history_probability','cpu_probability','probability_gap','baseline_replay_mismatch'])
assert len(d)==80 and d.window_id.is_unique and (d.groupby('split').size()==40).all()
assert len(c)==int(d.ungated_official_changes.sum()) and int(c.gated.sum())==int(d.gated_official_changes.sum())
d.to_csv(OUT/'official_gate_windows_audit.csv',index=False);c.to_csv(OUT/'official_gate_cells_audit.csv',index=False)
groups=d.groupby('split')[['ungated_official_changes','gated_official_changes']].sum()
result={'scope':'8 panels x 5 ongoing windows x 2 splits; not onset; no official labels/scores; no file submission','by_split':groups.to_dict('index'),
 'windows_with_ungated':d[d.ungated_official_changes>0].groupby('split').size().to_dict(),
 'windows_with_gate':d[d.gated_official_changes>0].groupby('split').size().to_dict(),
 'modified_windows_with_cpu_replay_mismatch':int(((d.gated_official_changes>0)&(d.replay_mismatch>0)).sum())}
(OUT/'official_gate_summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2),flush=True)
