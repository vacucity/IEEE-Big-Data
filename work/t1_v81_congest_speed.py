"""Direct common-mode SPEED residual for congested cells, flow kept fixed."""
import argparse,json,gc
import numpy as np
import pandas as pd
import lightgbm as lgb
import t1_v81_congest_cmode as B
from t1_v81_congest_cmode import N,C,A,F,ROOT,PANELS

OUT=ROOT/'work/results/t1_v81_congest_speed'
CAP=1.
PROTOCOL=dict(base='V79 .90321',features='25 wide common-mode features; fixed before this experiment',
    model='Metric-normalized ridge residual speed after V79; 1 km/h bound; flow unchanged',
    ridge=B.RIDGE,teacher='same training fold for fitting residual targets; exact incumbent original parity predictions at evaluation',
    gate='Each month positive weighted state and conservative total, negative LWR proxy delta, >=7 positive-total corridors; parity then frozen half-month/opposite-month checks',
    application='Real-frame FD substituted into every confirmed total; each must stay positive',
    leakage=N.PROTOCOL['leakage'],caveat=N.PROTOCOL['caveat'],scope='Congested non-blackout Task1 speeds only')

def transform(v,q,delta):return np.clip(v+np.clip(delta,-CAP,CAP),C.M.SPEED_MIN,C.M.SPEED_MAX),q.copy()

def fit_beta(xo,z,d,mode,a):
    tr=z[mode]==a;md=lgb.Booster(model_file=str(N.OUT/f'model_old_{mode}_{a}_r.txt'));ms=lgb.Booster(model_file=str(N.OUT/f'model_old_{mode}_{a}_sstar.txt'))
    v,q=C.corrected(d['pv'][tr],d['pq'][tr],md.predict(xo[tr],num_threads=2),ms.predict(xo[tr],num_threads=2),A.CFG,N.LAM)
    y=np.clip(d['ov'][tr]-v,-10,10);counts=d['counts_'+mode][a];w=1/np.maximum(counts[z['owner'][tr],d['reg'][tr].astype(int)],1)
    return B.ridge(d['X'][tr,:25],y,w)

def predictions(xo,z,d,mode):
    path=OUT/f'pred_{mode}.npy'
    if path.exists():return np.load(path)
    pred=np.zeros(len(xo))
    for a in (0,1):
        beta=fit_beta(xo,z,d,mode,a);te=z[mode]!=a;pred[te]=d['X'][te,:25]@beta;np.save(OUT/f'beta_{mode}_{a}.npy',beta)
    np.save(path,pred);return pred

def score(z,old,pred,mode):
    rows=[]
    for si,s in enumerate(F.SPLITS):
        for pi,p in enumerate(PANELS):
            d=N.load_one(p,s);m=d['cong'];ix=z['owner']==si*10+pi
            v0,q0=N.incumbent_free(s,mode,p,pi,d);v0[m],q0[m]=C.corrected(d['pv'][m],d['pq'][m],old['r'][ix],old['sstar'][ix],A.CFG,N.LAM)
            rw,_=C.target_weights(d,C.LOC.load_local(p,s,'T'),d['fd']);args=(d['ov'],d['oq'],d['lanes'],d['reg'],rw)
            s0=C.LOC.s_state_w(v0,q0,*args);a0=C.M.t3_score(d['C'],d['T'],d['Dd'],v0,q0,d['fd'])
            v1=v0.copy();v1[m],_=transform(v0[m],q0[m],pred[ix]);ds=C.LOC.s_state_w(v1,q0,*args)-s0
            dt=C.M.t3_delta(a0,C.M.t3_score(d['C'],d['T'],d['Dd'],v1,q0,d['fd']),d['sc'])
            rows.append(dict(split=s,panel=p,mode=mode,dS=ds,dE=dt['dE_est'],dFD=dt['dFD_est'],dTotal=.35*ds+.05*(dt['dFD_est']-.8*dt['dE_est'])))
            del d;gc.collect()
        print('speed scored',mode,s,flush=True)
    return rows

def check(mode):
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'protocol.json').write_text(json.dumps(PROTOCOL,indent=2))
    if mode!='parity':assert json.loads((OUT/'decision_parity.json').read_text())['passed']
    xo,xn,z=N.dataset();old=N.heads(xo,z,mode,'old');d=B.data();pred=predictions(xo,z,d,mode)
    df=pd.DataFrame(score(z,old,pred,mode));df.to_csv(OUT/f'checks_{mode}.csv',index=False);sm=[]
    for s,g in df.groupby('split'):
        sm.append(dict(split=s,mode=mode,positive_corridors=int((g.dTotal>0).sum()),
            **{k:C.M.family_mean(g.set_index('panel')[k].to_dict()) for k in ('dS','dE','dFD','dTotal')}))
    sm=pd.DataFrame(sm);sm.to_csv(OUT/f'summary_{mode}.csv',index=False)
    passed=bool((sm.dS>0).all() and (sm.dE<0).all() and (sm.dTotal>0).all() and (sm.positive_corridors>=7).all())
    (OUT/f'decision_{mode}.json').write_text(json.dumps(dict(passed=passed),indent=2));print(sm.to_string(index=False),flush=True);print('passed',passed,flush=True)

def full():
    for mode in ('parity','half_month','other_month'):assert json.loads((OUT/f'decision_{mode}.json').read_text())['passed']
    path=OUT/'beta_full.npy'
    if path.exists():return np.load(path)
    xo,xn,z=N.dataset();d=B.data();xo=N.old_inputs_exact();md=lgb.Booster(model_file=str(A.OUT/'density_l1.txt'));ms=lgb.Booster(model_file=str(A.OUT/'split_l2.txt'))
    v,q=C.corrected(d['pv'],d['pq'],md.predict(xo,num_threads=2),ms.predict(xo,num_threads=2),A.CFG,N.LAM)
    counts=d['counts_parity'].sum(0);w=1/np.maximum(counts[z['owner'],d['reg'].astype(int)],1)
    beta=B.ridge(d['X'][:,:25],np.clip(d['ov']-v,-10,10),w);np.save(path,beta);return beta

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('cmd',choices=['parity','half_month','other_month','full']);a=ap.parse_args()
    full() if a.cmd=='full' else check(a.cmd)
