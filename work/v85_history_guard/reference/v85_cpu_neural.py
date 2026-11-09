from pathlib import Path
import sys,os,argparse,json,gc,time
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='4'
import numpy as np,pandas as pd,torch
root=Path(__file__).resolve().parent/'v85_project';sys.path.insert(0,str(root/'work'))
import t2_cnn as C,t2_rnn as R,t2_pseudo as P,t2_grids as G,t2_v52_replica as Rep
from evaluate import RELEASE,links,T2_PANELS
OUT=root/'work/results'/os.environ.get('TFB_EXPERIMENT','v85_experiment');OUT.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(4)

def windows(panel,split,mode):
 if mode=='official':
  ix=pd.read_csv(RELEASE/'task2'/panel/split/'window_index.csv'); ix=ix[ix.condition=='queue_ongoing'].copy(); g=G.load(panel,split)
  ds=list(g['dates'].astype(str));rs=[]
  for x in ix.itertuples():
   ts=pd.Timestamp(x.forecast_origin);d=ds.index(str(x.date));rs.append(dict(day=d,T=ts.hour*12+ts.minute//5,date=str(x.date),weekday=int(g['weekday'][d]),window_id=x.window_id,cls=''))
  return pd.DataFrame(rs)
 path=OUT/f'windows_{panel}_{split}.csv'
 if path.exists():return pd.read_csv(path)
 tl=Rep.Timeline(panel,split,'2031-03');rs=[]
 for start in range(0,29,int(os.environ.get('TFB_START_EVERY','3'))):rs.extend(tl.select(start,int(os.environ.get('TFB_CAP','5')),'not_est'))
 df=pd.DataFrame(rs);df=df[(df.cond=='ongoing')&~df.blk_T5&(df['T']>=13)&(df['T']<=281)].drop_duplicates(['day','T']).copy()
 df['cls']=[Rep.classify(a,b,c) for a,b,c in zip(df.spacing,df.est_T5,df.cov_T5)]
 df['weekday']=[pd.Timestamp(x).weekday() for x in df.date];df['window_id']=[f'proxy_{panel}_{d}_{t}' for d,t in zip(df.day,df['T'])]
 if os.environ.get('TFB_EXCLUDE_EXPLORATION')=='1':
  old=pd.read_csv(root/'work/results/v85_experiment'/f'windows_{panel}_{split}.csv')
  df=df[~df.window_id.isin(old.window_id)].copy()
 df.to_csv(path,index=False);return df

def predict(panel,split,mode):
 path=OUT/f'nets_{panel}_{split}_{mode}.npz'
 if path.exists():print('exists',path.name,flush=True);return
 t=time.time();df=windows(panel,split,mode);ws=[P.Win(panel,split,int(r.day),int(r.T),'ongoing',str(r.date),int(r.weekday)) for r in df.itertuples()]
 D=C.PanelData(panel,split,unmasked=False);result={}
 specs=([('rnnft96_0','rnn','_g0_hw96all',0),('rnnft96_1','rnn','_g0_s1_hw96all',0),('rnnftx48_2','rnn','_g0_s2_hw48all',0),('rnnftx48_3','rnn','_g0_s3_hw48all',0),('cnnftx96_1','cnn','_g0_s1_hw96all',0),('cnnftx96_2','cnn','_g0_s2_hw96all',0)] if split=='private' else
 [('rnn2a_g1','rnn','',1),('rnn2b_g1','rnn','_s1',1),('rnn2a_g0','rnn','',0),('rnn2b_g0','rnn','_s1',0),('cnng0_0','cnn','_g0',0),('cnng0_1','cnn','_g0_s1',0),('cnng0_2','cnn','_g0_s2',0),('rnng0_0','rnn','_g0',0),('rnng0_1','rnn','_g0_s1',0)])
 for name,kind,tag,gap in specs:
  model=C.build_model(96) if kind=='cnn' else R.build_model(); model.load_state_dict(torch.load(root/'work/models'/f't2_{kind}{tag}.pt',map_location='cpu',weights_only=True));model.eval();out=np.zeros((len(ws),6,D.n),np.float32)
  for start in range(0,len(ws),16):
   sub=ws[start:start+16];xs=[]
   for w in sub:
    sl=slice(w.T-11,w.T+1) if gap==0 else slice(w.T-12,w.T)
    xs.append(C.make_x(D.V[w.day,sl],D.Q[w.day,sl],D.O[w.day,sl],D.static_ch,D.st['ffs'],D.st['cap'],D.vcut,w.T+1-gap,w.weekday))
   (X,)=C.batch_tensors(xs) if kind=='cnn' else R.to_batch(xs)
   with torch.inference_mode():pr=torch.sigmoid(model(X).float()).numpy()[:,:,:D.n]
   out[start:start+len(xs),:,D.perm]=pr
  result[name]=out;del model;gc.collect();print(panel,split,mode,name,len(ws),'seconds',round(time.time()-t),flush=True)
 result['day']=df.day.to_numpy();result['T']=df['T'].to_numpy();result['ids']=df.window_id.to_numpy(dtype='U100');result['cls']=df.cls.to_numpy(dtype='U30')
 if mode=='proxy':
  g=G.load(panel,split);vc=links(panel).v_cut_kmh.to_numpy(np.float32);vs=np.stack([g['V'][w.day,w.T+1:w.T+7] for w in ws]) if ws else np.empty((0,6,D.n));result['y']=vs<=vc;result['mask']=np.isfinite(vs)
 np.savez_compressed(path,**result);print('SAVED',path.name,round(time.time()-t),flush=True);G.load.cache_clear()

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--split',choices=['validation','private'],required=True);ap.add_argument('--mode',choices=['official','proxy'],default='official');ap.add_argument('--panels',nargs='*',default=T2_PANELS);a=ap.parse_args()
 for p in a.panels:predict(p,a.split,a.mode)
