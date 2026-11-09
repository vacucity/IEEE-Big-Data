"""Independently check flow bounds 15/30 and their combination with V76 speed.

The 30 vph/lane alternative was added after the initial parity flow result.
Require six positive flow-only AND joint checks, then maximize worst joint gain.
"""
from __future__ import annotations
import json,gc,sys
import numpy as np
import pandas as pd
import t1_v76_flow_mix as Q
from t1_v76_flow_mix import F,G,VFY,PANELS

OUT=G.OUT.parent/'t1_v76_joint'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    precompute='--precompute' in sys.argv
    if not precompute:assert json.loads((Q.OUT/'decision.json').read_text())['accepted']
    choice=json.loads((G.OUT/'decision.json').read_text())['chosen']
    kind,vl=choice.split('_clip');vl=float(vl)
    own=np.load(G.OUT/'feature_rows.npz')['owner'];rows=[]
    for mode in ('parity','half_month','other_month'):
        if precompute and not (Q.OUT/f'pred_{mode}.npy').exists():continue
        cache=OUT/f'checks_{mode}.csv'
        if cache.exists():
            cached=pd.read_csv(cache)
            assert len(cached)==80 and not cached.duplicated(['cand','limit','split','panel','mode']).any()
            rows.extend(cached.to_dict('records'));continue
        start=len(rows)
        vr=np.load(G.OUT/f'baseline_{mode}.npy');vp=np.load(G.OUT/f'{kind}_{mode}.npy')
        qp=np.load(Q.OUT/f'pred_{mode}.npy')*100
        for si,s in enumerate(F.SPLITS):
            z=F.baseline(s)
            for pi,p in enumerate(PANELS):
                d,lh=F.LOC.load(p,s);m=d['free'];idx=own==si*10+pi;fd=d['fd'];lane=fd['lanes'][d['pos']]
                rw,_=F.LOC.target_weights(d,lh,F.LOC.load_local(p,s,'T'),fd)
                v0d=np.zeros(len(m));v1d=v0d.copy();v0d[m]=np.clip(vr[idx],-1,1);v1d[m]=np.clip(vp[idx],-vl,vl)
                q0=z[p+'_q'];v0=VFY.clamp(p,d['pos'],m,z[p+'_v']+v0d,q0);v1=VFY.clamp(p,d['pos'],m,z[p+'_v']+v1d,q0)
                st=(d['ov'],d['oq'],lane,d['reg'],rw);s0=F.LOC.s_state_w(v0,q0,*st)
                a0=F.M.t3_score(d['C'],d['T'],d['Dd'],v0,q0,fd)
                for limit in (15.,30.):
                    dq=np.zeros(len(m));dq[m]=np.clip(qp[idx],-limit,limit)*lane[m]
                    for cand,v in [('flow',v0),('joint',v1)]:
                        q1=np.where(m,np.maximum(0,np.minimum(q0+dq,v*lane*fd['kjl'][d['pos']])),q0)
                        ds=F.LOC.s_state_w(v,q1,*st)-s0
                        a1=F.M.t3_score(d['C'],d['T'],d['Dd'],v,q1,fd);dt=F.M.t3_delta(a0,a1,d['sc'])
                        rows.append(dict(cand=cand,limit=limit,split=s,panel=p,mode=mode,dS_rw=ds,dE=dt['dE_est'],dFD=dt['dFD_est'],
                            dTotal40=.35*ds+.05*(dt['dFD_est']-.8*dt['dE_est'])))
                del d,lh;gc.collect()
        temp=cache.with_suffix('.tmp');pd.DataFrame(rows[start:]).to_csv(temp,index=False);temp.replace(cache)
        print('joint checked',mode,flush=True)
    if precompute:
        print('Precomputed available folds; no selection yet',flush=True);return
    r=pd.DataFrame(rows);assert len(r)==240 and not r.duplicated(['cand','limit','split','panel','mode']).any()
    r.to_csv(OUT/'checks.csv',index=False);sm=[]
    for (cand,limit,s,mode),g in r.groupby(['cand','limit','split','mode']):
        sm.append(dict(cand=cand,limit=limit,split=s,mode=mode,positive_corridors=int((g.dTotal40>0).sum()),
            **{k:F.M.family_mean(g.set_index('panel')[k].to_dict()) for k in ('dS_rw','dE','dFD','dTotal40')}))
    sm=pd.DataFrame(sm);sm.to_csv(OUT/'summary.csv',index=False)
    ref=pd.read_csv(Q.OUT/'summary.csv').set_index(['split','mode'])
    check=sm[(sm.cand=='flow')&(sm.limit==Q.LIMIT)].set_index(['split','mode']).reindex(ref.index)
    cols=['dS_rw','dE','dFD','dTotal40']
    assert np.allclose(check[cols],ref[cols],rtol=0,atol=1e-11)
    allowed=[]
    for limit,g in sm.groupby('limit'):
        if len(g)==12 and (g.dS_rw>0).all() and (g.dTotal40>0).all():allowed.append(limit)
    chosen=max(allowed,key=lambda lim:sm[(sm.limit==lim)&(sm.cand=='joint')].dTotal40.min()) if allowed else None
    (OUT/'decision.json').write_text(json.dumps(dict(flow_limit=chosen,speed_choice=choice,
        gate='six positive state/total checks for flow alone and joint; maximize worst joint gain; bounds 15 and 30'),indent=2))
    print(sm.to_string(index=False),flush=True);print('chosen',chosen,flush=True)


if __name__=='__main__':main()
