"""Verify audit counts, criteria and original submission values without generating a candidate."""
from pathlib import Path
import json,zipfile
import numpy as np,pandas as pd
R=Path(__file__).resolve().parents[1];O=R/'outputs/analog-evidence-gates'
d=pd.read_csv(O/'official_gate_windows_audit.csv');c=pd.read_csv(O/'official_gate_cells_audit.csv')
assert len(d)==80 and d.window_id.is_unique and d.groupby('split').size().eq(40).all()
assert not c.duplicated(['window_id','horizon_minutes','link_id']).any()
assert c.before.eq(0).all() and c.after.eq(1).all() and (~c.baseline_replay_mismatch).all()
assert c.gated.eq((c.valid_history>=20)&(c.probability_gap>=.3)).all()
assert (c.valid_history.between(0,30)&c.history_probability.between(0,1)&c.cpu_probability.between(0,1)).all()
assert np.allclose(c.probability_gap,c.history_probability-c.cpu_probability,atol=1e-7)
counts=d.groupby('split')[['ungated_official_changes','gated_official_changes']].sum()
for split,g in c.groupby('split'):
 assert len(g)==counts.loc[split,'ungated_official_changes'] and int(g.gated.sum())==counts.loc[split,'gated_official_changes']
prior=pd.read_csv(R/'outputs/analog-gain-trace/正式窗口改动明细_仅审计.csv')
current=c[c.panel=='D7_I210_W'];keys=['split','window_id','horizon_minutes','link_id']
assert set(map(tuple,prior[keys].to_numpy()))==set(map(tuple,current[keys].to_numpy()))
q=pd.read_csv(W:=R/'work/v85_project/work/results/v85_experiment/v84_queue.csv')
q['timestamp']=pd.to_datetime(q.timestamp,utc=True)
c['timestamp']=pd.to_datetime(c.date,utc=True)+pd.to_timedelta(c['T']*5+c.horizon_minutes,unit='m')
merged=c.merge(q,on=['split','window_id','link_id','timestamp'],how='left',validate='one_to_one')
assert merged.submission_id.notna().all() and merged.queue_pred.eq(merged.before).all()
ids=set(merged.submission_id.astype(int));found=[]
with zipfile.ZipFile(r'D:\Users\mixri\Desktop\v84_observation_flow.zip') as z:
 names=[f for f in z.namelist() if not f.endswith('/')];assert len(names)==1
 with z.open(names[0]) as stream:
  for chunk in pd.read_csv(stream,usecols=['submission_id','queue_pred'],chunksize=300000):
   hit=chunk[chunk.submission_id.isin(ids)]
   if len(hit):found.append(hit)
found=pd.concat(found,ignore_index=True)
assert len(found)==len(ids)==len(c) and found.submission_id.is_unique and found.queue_pred.eq(0).all()
status={'window_count_and_thresholds':'pass','target_panel_prior_audit_match':'pass','all_12_cached_keys_match_original_v84_zip':'pass','scope':'read-only mapping and math checks, does not test official prediction quality'}
(O/'official_gate_verification.json').write_text(json.dumps(status,indent=2),encoding='utf-8');print(json.dumps(status,indent=2))
