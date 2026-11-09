"""Produce two user-requested experimental ZIPs from V84, preserving unmodified CSV bytes."""
from pathlib import Path
import hashlib,json,zipfile
from contextlib import ExitStack
import numpy as np,pandas as pd
R=Path(__file__).resolve().parents[1];OUT=R/'outputs/two-submission-trials';OUT.mkdir(parents=True,exist_ok=True)
original=Path(r'D:\Users\mixri\Desktop\v84_observation_flow.zip')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
assert sha(original)=='a429599ade6ee8663b28f850f86acc0819ba426d2b2f9cc0861b2831edc36090'
queue=pd.read_csv(R/'work/v85_project/work/results/v85_experiment/v84_queue.csv')
queue['timestamp']=pd.to_datetime(queue.timestamp,utc=True)
assert queue.submission_id.is_unique and len(queue)==174000
spec=[('V85_history_guard','analog-evidence-gates/official_gate_cells_audit.csv',True,{'validation':8,'private':2}),
      ('V86_spacetime_guard','neighbor-gate-study/official_neighbor_cells_audit.csv',False,{'validation':62,'private':32})]
allmaps={};mapped={}
for name,path,only_gated,expected in spec:
 c=pd.read_csv(R/'outputs'/path)
 if only_gated:c=c[c.gated].copy()
 assert not c.duplicated(['split','window_id','horizon_minutes','link_id']).any()
 assert c.groupby('split').size().to_dict()==expected
 assert c.before.isin([0,1]).all() and c.after.isin([0,1]).all() and (c.before!=c.after).all()
 if not only_gated:assert c.before.eq(c.cpu_base).all() and c.after.eq(c.cpu_candidate).all()
 c['timestamp']=pd.to_datetime(c.date,utc=True)+pd.to_timedelta(c['T']*5+c.horizon_minutes,unit='m')
 x=c.merge(queue,on=['split','panel','window_id','link_id','timestamp'],how='left',validate='one_to_one')
 assert x.submission_id.notna().all() and x.queue_pred.eq(x.before).all() and x.submission_id.is_unique
 x['submission_id']=x.submission_id.astype(int);x['candidate']=name
 mapped[name]=x;allmaps[name]={int(r.submission_id):(int(r.before),int(r.after)) for r in x.itertuples()}
 x[['candidate','submission_id','split','panel','window_id','timestamp','link_id','before','after']].to_csv(OUT/(name+'_changes.csv'),index=False)
assert not set(allmaps[spec[0][0]]).intersection(allmaps[spec[1][0]])
targets=set().union(*[set(x) for x in allmaps.values()]);seen=set()
with zipfile.ZipFile(original) as src,ExitStack() as stack:
 files=[f for f in src.infolist() if not f.is_dir()];assert len(files)==1
 inp=stack.enter_context(src.open(files[0]))
 outs={}
 for name,_,_,_ in spec:
  z=stack.enter_context(zipfile.ZipFile(OUT/(name+'.zip'),'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True))
  outs[name]=stack.enter_context(z.open('submission.csv','w'))
 header=inp.readline();assert header.rstrip(b'\r\n')==b'submission_id,task,speed_kmh,flow_vph,queue_pred,path_flow'
 for f in outs.values():f.write(header)
 buffers={k:bytearray() for k in outs};row=0
 for line in inp:
  row+=1
  if row in targets:
   endings=line[len(line.rstrip(b'\r\n')):];parts=line.rstrip(b'\r\n').split(b',')
   assert len(parts)==6 and int(parts[0])==row and parts[1]==b'queue'
   for name,f in outs.items():
    if row in allmaps[name]:
     before,after=allmaps[name][row];assert parts[4]==str(before).encode()
     a=parts.copy();a[4]=str(after).encode();buffers[name].extend(b','.join(a)+endings)
    else:buffers[name].extend(line)
   seen.add(row)
  else:
   for buf in buffers.values():buf.extend(line)
  if len(next(iter(buffers.values())))>=1024*1024:
   for name,f in outs.items():f.write(buffers[name]);buffers[name].clear()
 for name,f in outs.items():
  if buffers[name]:f.write(buffers[name])
 assert row==6980503 and seen==targets
summary={'original_sha256':sha(original),'rows':row,'candidates':{}}
for name,_,_,expected in spec:
 p=OUT/(name+'.zip')
 summary['candidates'][name]={'sha256':sha(p),'zip_bytes':p.stat().st_size,'changes_by_split':expected,'changes':len(allmaps[name])}
(OUT/'build_manifest.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print(json.dumps(summary,indent=2),flush=True)
