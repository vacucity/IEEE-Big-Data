from pathlib import Path
import sys,time,os
import numpy as np,pandas as pd
r=Path(__file__).resolve().parent/'v85_project';sys.path.insert(0,str(r/'work'));sys.path.insert(0,str(r.parent))
import v85_cpu_neural as N
from evaluate import T2_PANELS
from t2_slotT import ml2_g0_probs
from t2_cat_g0 import cat_g0_probs
import t2_pseudo as P,t2_grids as G
out=N.OUT
for mode in (('proxy',) if os.environ.get('TFB_PROXY_ONLY')=='1' else ('official','proxy')):
 for panel in T2_PANELS:
  path=out/f'gbm_{panel}_validation_{mode}.npz'
  if path.exists():continue
  t=time.time();df=N.windows(panel,'validation',mode);ws=[P.Win(panel,'validation',int(x.day),int(x.T),'ongoing',str(x.date),int(x.weekday)) for x in df.itertuples()]
  if ws:ml=ml2_g0_probs(ws);cat=cat_g0_probs(ws)
  else:ml=cat=np.empty((0,6,len(N.links(panel))))
  np.savez_compressed(path,ml2g0=ml,catg0=cat);print(panel,mode,len(ws),round(time.time()-t),'sec',flush=True);G.load.cache_clear()
