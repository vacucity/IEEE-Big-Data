from pathlib import Path
import sys,json,zlib,warnings,zipfile,time,os
warnings.filterwarnings('ignore',category=RuntimeWarning)
import numpy as np,pandas as pd
r=Path(__file__).resolve().parent/'v85_project';sys.path.insert(0,str(r/'work'));sys.path.insert(0,str(r.parent))
from evaluate import RELEASE,T2_PANELS,links
from t2_group_v27 import groups,smooth
import t2_v52_selrule as S
import v85_decoders as D
out=r/'work/results'/os.environ.get('TFB_EXPERIMENT','v85_experiment')

def probability(panel,split,mode):
 z=dict(np.load(out/f'nets_{panel}_{split}_{mode}.npz',allow_pickle=False))
 if split=='private':
  p=(4*(z['rnnft96_0']+z['rnnft96_1'])/2+2*(z['rnnftx48_2']+z['rnnftx48_3'])/2+4*(z['cnnftx96_1']+z['cnnftx96_2'])/2)/10
 else:
  gb=dict(np.load(out/f'gbm_{panel}_{split}_{mode}.npz'))
  a=np.concatenate([z['rnn2a_g1'][:,:1],z['rnn2a_g0'][:,:5]],1);b=np.concatenate([z['rnn2b_g1'][:,:1],z['rnn2b_g0'][:,:5]],1)
  p=(2*gb['ml2g0']+(z['cnng0_0']+z['cnng0_1']+z['cnng0_2'])/3+5*(z['rnng0_0']+z['rnng0_1'])/2+2*(a+b)/2)/10
 gs=groups(panel,.95);p=smooth(p,gs,.25)
 if split=='validation':p=.75*p+.25*smooth(gb['catg0'],gs,.25)
 return p.astype(np.float32),z

def submitted():
 path=Path(os.environ['V85_HANDOFF_QUEUE'])
 if path.exists():return pd.read_csv(path)
 chunks=[]
 with zipfile.ZipFile(r'D:\Users\mixri\Desktop\v84_observation_flow.zip') as zz,zz.open(zz.namelist()[0]) as f:
  for key,val in zip(pd.read_csv(RELEASE/'submission_key.csv',chunksize=300000,low_memory=False),pd.read_csv(f,chunksize=300000)):
   m=key.task=='queue'
   if m.any():
    a=key.loc[m,['submission_id','split','panel','window_id','timestamp','link_id']].copy();a['queue_pred']=val.loc[m,'queue_pred'].to_numpy();chunks.append(a)
 q=pd.concat(chunks);q.to_csv(path,index=False);return q

def official_base(q,panel,split,wid,shape):
 L=links(panel);pos=dict(zip(L.link_id.astype(str),L.pos));rows=q[(q.window_id==wid)];times=sorted(rows.timestamp.unique());hor={t:i for i,t in enumerate(times)};a=np.zeros(shape,bool);a[rows.timestamp.map(hor),rows.link_id.map(pos)]=rows.queue_pred.astype(bool);return a

def main(mode):
 records=[];patches=[];q=submitted() if mode=='official' else None
 for split in (('validation','private') if mode=='official' else ('validation',)):
  for panel in T2_PANELS:
   t=time.time();pp,z=probability(panel,split,mode)
   if mode=='proxy':ctx=pd.read_csv(out/f'windows_{panel}_{split}.csv').set_index('window_id')
   else:ctx=S.official_classes(panel,split).set_index('window_id')
   for i,wid in enumerate(z['ids']):
    wid=str(wid);c=ctx.loc[wid];cl=str(c['cls']);T=int(c['T']);p=pp[i];p0=D.ratio(p);ids=S.coupling_ids(panel,p,'group');seed=zlib.crc32(wid.encode('ascii'));con=cl=='constr';dissolve=con and panel=='D7_I210_E' and T//12>=17
    Y,acc=D.samples(p,ids,seed,constrained=con)
    replay=p0.copy()
    if dissolve:replay=S.rule_dissolve(p0)
    elif con and acc>=.02:
     replay=D.ratio(Y.mean(0).astype(np.float32));replay[0]=p0[0]
    base=official_base(q,panel,split,wid,p.shape) if mode=='official' else replay
    if dissolve or (con and acc<.02):v85=base.copy();v86=base.copy();audit={'reason':'retain_existing_rule','accepted_samples':len(Y)}
    else:v85,v86,audit=D.decode(base,Y,seed^8586)
    row=dict(split=split,panel=panel,window_id=wid,day=int(z['day'][i]),T=T,cls=cl,acceptance=acc,replay_mismatch=int((base!=replay).sum()),**audit)
    if mode=='proxy':
     row.update(base=D.iou(base,z['y'][i],z['mask'][i]),v85=D.iou(v85,z['y'][i],z['mask'][i]),v86=D.iou(v86,z['y'][i],z['mask'][i]))
    else:
     sub=q[q.window_id==wid];L=links(panel);pos=dict(zip(L.link_id.astype(str),L.pos));times=sorted(sub.timestamp.unique());hor={t:j for j,t in enumerate(times)}
     for name,pred in [('v85',v85),('v86',v86)]:
      values=pred[sub.timestamp.map(hor),sub.link_id.map(pos)].astype(int);old=sub.queue_pred.to_numpy();m=values!=old
      patches.extend(dict(version=name,submission_id=int(sid),old=int(o),new=int(n)) for sid,o,n in zip(sub.submission_id.to_numpy()[m],old[m],values[m]))
    records.append(row)
   pd.DataFrame(records).to_csv(out/f'{mode}_decisions.csv',index=False);print(split,panel,mode,'windows',len(z['ids']),'seconds',round(time.time()-t),'mismatch',sum(a['replay_mismatch'] for a in records if a['panel']==panel and a['split']==split),flush=True)
 if patches:pd.DataFrame(patches).to_csv(out/'official_patches.csv',index=False)
 if mode=='proxy':
  df=pd.DataFrame(records);summary={}
  for part,m in [('early',df.day<15),('late',df.day>=15),('all',np.ones(len(df),bool))]:
   ss=df[m];av=ss.groupby('panel')[['base','v85','v86']].mean().mean();summary[part]={'windows':len(ss),**av.to_dict(),'delta_v85':float(av.v85-av.base),'delta_v86':float(av.v86-av.base)}
  (out/'proxy_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main(sys.argv[1])
