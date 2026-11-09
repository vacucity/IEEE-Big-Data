"""Independently stream-verify candidate ZIPs against original at byte level."""
from pathlib import Path
import hashlib,json,zipfile
from contextlib import ExitStack
import pandas as pd
R=Path(__file__).resolve().parents[1];O=R/'outputs/two-submission-trials';original=Path(r'D:\Users\mixri\Desktop\v84_observation_flow.zip')
names=['V85_history_guard','V86_spacetime_guard']
manifest=json.loads((O/'build_manifest.json').read_text())
targets={}
for name in names:
 d=pd.read_csv(O/(name+'_changes.csv'));assert d.submission_id.is_unique
 targets[name]={int(r.submission_id):(int(r.before),int(r.after)) for r in d.itertuples()}
 with zipfile.ZipFile(O/(name+'.zip')) as z:assert z.namelist()==['submission.csv']
for name in names:
 h=hashlib.sha256()
 with (O/(name+'.zip')).open('rb') as f:
  for x in iter(lambda:f.read(4*1024*1024),b''):h.update(x)
 assert h.hexdigest()==manifest['candidates'][name]['sha256']
found={name:set() for name in names}
with ExitStack() as stack:
 src=stack.enter_context(zipfile.ZipFile(original));inp=stack.enter_context(src.open(src.namelist()[0]))
 streams={}
 for name in names:
  z=stack.enter_context(zipfile.ZipFile(O/(name+'.zip')));streams[name]=stack.enter_context(z.open('submission.csv'))
 head=inp.readline();assert all(f.readline()==head for f in streams.values())
 row=0
 for line in inp:
  row+=1
  for name,f in streams.items():
   actual=f.readline()
   if row in targets[name]:
    before,after=targets[name][row];end=line[len(line.rstrip(b'\r\n')):];parts=line.rstrip(b'\r\n').split(b',')
    assert len(parts)==6 and int(parts[0])==row and parts[1]==b'queue' and parts[4]==str(before).encode()
    parts[4]=str(after).encode();assert actual==b','.join(parts)+end
    found[name].add(row)
   else:assert actual==line,(name,row)
 assert row==6980503
 for name,f in streams.items():assert f.read(1)==b'' and found[name]==set(targets[name])
summary={'rows_per_file':row,'format_and_header':'matches original byte for byte','unchanged_rows':'byte identical to V84','all_changes':'exactly match verified submission IDs and queue fields','archive_crc':'passed while reading entire files','candidate_count':len(names),'changes':{name:len(found[name]) for name in names},'limitation':'does not evaluate quality or Kaggle score'}
(O/'verification.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print(json.dumps(summary,indent=2),flush=True)
