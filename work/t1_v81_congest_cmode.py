"""Direct residual common-mode regression along the incumbent iso-density line.

Training targets use a teacher fitted on the SAME training fold. Held-out
teachers never supply training residuals. All features exclude hidden cells.
"""
import argparse,json,gc
import numpy as np
import pandas as pd
import lightgbm as lgb
import t1_v81_density_context as N
from t1_v81_density_context import C,A,F,ROOT,PANELS
import t1_v66_cmode as CM

OUT=ROOT/'work/results/t1_v81_congest_cmode'
KINDS=('wide','band')
CAP=1.
RIDGE=.01
PROTOCOL=dict(base='V79 .90321',model='Direct metric-weighted common-mode residual ridge along fixed q/v',
    candidates=KINDS,speed_cap=CAP,ridge_trace_fraction=RIDGE,
    train='Each fold target uses an old teacher fitted only to that same training fold; exact original V64 parity OOF for evaluation.',
    selection='Both parity months positive reweighted state and .35 state + .05 FD; >=7 positive-total corridors; maximize worst total.',
    confirmation='Freeze feature kind, require same gate for both months in half-month and opposite-month folds.',
    scope='Task1 congested non-blackout state only; preserve density except final numeric rounding',
    inputs='25 wide common-mode features or 43 wide+spatial-band features, all hidden observations removed',
    caveat=N.PROTOCOL['caveat'],leakage=N.PROTOCOL['leakage'])

def design(f,lh):
    wide=CM.design1(f,lh)
    return np.column_stack([wide,CM.design2(f)[:,:-1]]).astype(float)

def data():
    path=OUT/'data.npz'
    if path.exists():return dict(np.load(path))
    acc={k:[] for k in ('X','pv','pq','ov','oq','lanes','reg')};counts={mode:np.zeros((2,20,3),int) for mode in ('parity','half_month','other_month')}
    for si,s in enumerate(F.SPLITS):
        for pi,p in enumerate(PANELS):
            d=N.load_one(p,s);m=d['cong'];f=CM.hide_feats(p,s,d);acc['X'].append(design(f,d['L'])[m])
            for k in ('pv','pq','ov','oq','lanes','reg'):acc[k].append(d[k][m])
            for mode,fold in [('parity',d['par']),('half_month',(d['day']>=15).astype(int)),('other_month',np.full(len(m),si))]:
                for a in (0,1):counts[mode][a,si*10+pi]=np.bincount(d['reg'][fold==a],minlength=3)
            print('cm data',s,p,int(m.sum()),flush=True);del d;gc.collect()
    z={k:np.concatenate(v) for k,v in acc.items()};z.update({'counts_'+k:v for k,v in counts.items()})
    assert len(z['X'])==115148 and np.isfinite(z['X']).all();np.savez_compressed(path,**z);return z

def ridge(X,y,w):
    w=w/w.mean();G=X.T@(w[:,None]*X);b=X.T@(w*y)
    penalty=RIDGE*np.trace(G)/len(G);reg=np.eye(len(G))*penalty;reg[24,24]=penalty*.01
    return np.linalg.solve(G+reg,b)

def targets(d,z,idx,v,q,counts):
    # Fixed normalization is independent of held-out labels.
    a=.54/25**2;b=.46/600**2
    lane=d['lanes'][idx];ql=q/lane;k=ql/v
    h=a+b*k*k;dv=(a*(d['ov'][idx]-v)+b*k*(d['oq'][idx]/lane-ql))/h
    w=h/np.maximum(counts[z['owner'][idx],d['reg'][idx].astype(int)],1)
    return np.clip(dv,-10,10),w

def transform(v,q,delta):
    vv=np.clip(v+np.clip(delta,-CAP,CAP),C.M.SPEED_MIN,C.M.SPEED_MAX);qq=q*(vv/v)
    assert np.allclose(qq/vv,q/v,rtol=2e-15,atol=1e-12)
    return vv,qq

def predictions(xo,z,d,mode,kinds):
    path=OUT/f'pred_{mode}.npz'
    if path.exists():return dict(np.load(path))
    pred={k:np.zeros(len(xo)) for k in kinds}
    for a in (0,1):
        tr=z[mode]==a;te=~tr
        # These old models have never fitted te labels. No reverse-OOF target.
        md=lgb.Booster(model_file=str(N.OUT/f'model_old_{mode}_{a}_r.txt'))
        ms=lgb.Booster(model_file=str(N.OUT/f'model_old_{mode}_{a}_sstar.txt'))
        v,q=C.corrected(d['pv'][tr],d['pq'][tr],md.predict(xo[tr],num_threads=2),ms.predict(xo[tr],num_threads=2),A.CFG,N.LAM)
        y,w=targets(d,z,np.flatnonzero(tr),v,q,d['counts_'+mode][a]);betas={}
        for kind in kinds:
            nf=25 if kind=='wide' else 43;X=d['X'][:,:nf];beta=ridge(X[tr],y,w);betas[kind]=beta;pred[kind][te]=X[te]@beta
        np.savez(OUT/f'beta_{mode}_{a}.npz',**betas);print('cm fit',mode,a,flush=True)
    np.savez_compressed(path,**pred);return pred

def score(z,old,pred,mode):
    rows=[]
    for si,s in enumerate(F.SPLITS):
        for pi,p in enumerate(PANELS):
            d=N.load_one(p,s);m=d['cong'];ix=z['owner']==si*10+pi
            v0,q0=N.incumbent_free(s,mode,p,pi,d);v0[m],q0[m]=C.corrected(d['pv'][m],d['pq'][m],old['r'][ix],old['sstar'][ix],A.CFG,N.LAM)
            rw,_=C.target_weights(d,C.LOC.load_local(p,s,'T'),d['fd']);args=(d['ov'],d['oq'],d['lanes'],d['reg'],rw)
            s0=C.LOC.s_state_w(v0,q0,*args);a0=C.M.t3_score(d['C'],d['T'],d['Dd'],v0,q0,d['fd'])
            for kind,delta in pred.items():
                v1,q1=v0.copy(),q0.copy();v1[m],q1[m]=transform(v0[m],q0[m],delta[ix])
                ds=C.LOC.s_state_w(v1,q1,*args)-s0;dt=C.M.t3_delta(a0,C.M.t3_score(d['C'],d['T'],d['Dd'],v1,q1,d['fd']),d['sc'])
                assert abs(dt['dE_est'])<1e-10
                rows.append(dict(split=s,panel=p,mode=mode,kind=kind,dS=ds,dE=dt['dE_est'],dFD=dt['dFD_est'],dTotal=.35*ds+.05*dt['dFD_est']))
            del d;gc.collect()
        print('cm scored',mode,s,flush=True)
    return rows

def check(mode):
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'protocol.json').write_text(json.dumps(PROTOCOL,indent=2))
    if mode=='parity':kinds=KINDS
    else:
        kind=json.loads((OUT/'selection.json').read_text())['kind'];assert kind is not None;kinds=(kind,)
    xo,xn,z=N.dataset();old=N.heads(xo,z,mode,'old');d=data();pred=predictions(xo,z,d,mode,kinds)
    df=pd.DataFrame(score(z,old,pred,mode));df.to_csv(OUT/f'checks_{mode}.csv',index=False);sm=[]
    for (s,kind),g in df.groupby(['split','kind']):
        sm.append(dict(split=s,mode=mode,kind=kind,positive_corridors=int((g.dTotal>0).sum()),
            **{k:C.M.family_mean(g.set_index('panel')[k].to_dict()) for k in ('dS','dE','dFD','dTotal')}))
    sm=pd.DataFrame(sm);sm.to_csv(OUT/f'summary_{mode}.csv',index=False)
    passes=lambda g:bool((g.dS>0).all() and (g.dTotal>0).all() and (g.positive_corridors>=7).all())
    if mode=='parity':
        eligible=[k for k,g in sm.groupby('kind') if passes(g)];kind=max(eligible,key=lambda k:sm[sm.kind==k].dTotal.min()) if eligible else None
        decision=dict(kind=kind,eligible=eligible);dst='selection.json'
    else:decision=dict(kind=kinds[0],passed=passes(sm));dst=f'decision_{mode}.json'
    (OUT/dst).write_text(json.dumps(decision,indent=2));print(sm.to_string(index=False),flush=True);print(decision,flush=True)

def full():
    for mode in ('half_month','other_month'):assert json.loads((OUT/f'decision_{mode}.json').read_text())['passed']
    kind=json.loads((OUT/'selection.json').read_text())['kind'];nf=25 if kind=='wide' else 43
    path=OUT/'beta_full.npy'
    if path.exists():return np.load(path)
    xo,xn,z=N.dataset();d=data();xo=N.old_inputs_exact()
    # Full teacher is the exact existing production V64 model.
    md=lgb.Booster(model_file=str(A.OUT/'density_l1.txt'));ms=lgb.Booster(model_file=str(A.OUT/'split_l2.txt'))
    v,q=C.corrected(d['pv'],d['pq'],md.predict(xo,num_threads=2),ms.predict(xo,num_threads=2),A.CFG,N.LAM)
    y,w=targets(d,z,np.arange(len(xo)),v,q,d['counts_parity'].sum(0));beta=ridge(d['X'][:,:nf],y,w)
    np.save(path,beta);return beta

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('cmd',choices=['parity','half_month','other_month','full']);a=ap.parse_args()
    full() if a.cmd=='full' else check(a.cmd)
