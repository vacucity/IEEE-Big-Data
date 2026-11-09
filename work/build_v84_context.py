"""Build a validated occupancy/context flow residual on V83."""
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
import t1_v84_flow_context as K
from t1_v84_flow_context import T,F,PANELS,C,ROOT,OUT
import build_v81_density as PACK
import build_v76 as AN
import repro_check as RC

BASE=ROOT/'work/submissions/v83_flow_continue.zip'
NAME='v84_flow_context'
ANCHOR=AN.ANCHOR
FEATURE_OUT=K.OUT


def prepare():
    alpha=K.full();model=lgb.Booster(model_file=str(OUT/'model_full.txt'))
    teacher=lgb.Booster(model_file=str(T.OUT/'model.txt'))
    meta=dict(base_sha256=PACK.digest(BASE),model_sha256=PACK.digest(OUT/'model_full.txt'),
              teacher_sha256=PACK.digest(T.OUT/'model.txt'),anchor_sha256=PACK.digest(ANCHOR),alpha=alpha)
    RC.base_sha(BASE,'ddab5075d8dde15e7faed95b97e3a6fe8ba89c1591ee42636017f37ac0bc1e26','build_v84_context base (V83 v83_flow_continue)',got=meta['base_sha256'])
    poison=[json.loads(p.read_text()) for p in FEATURE_OUT.glob('poison_*.json')]
    assert len(poison)==20 and all(p['all_channels_poison_invariant'] for p in poison)
    path=OUT/'patch.npz'
    if path.exists():
        assert json.loads((OUT/'patch_meta.json').read_text())==meta;return dict(np.load(path))
    sid,v0,q0=C.V.zip_state(BASE);sa,va,qa=C.V.zip_state(ANCHOR);assert np.array_equal(sid,sa)
    key=F.M.read_key_state();assert np.array_equal(sid,key.submission_id.to_numpy());keys=key.key.to_numpy();del key
    q1=q0.copy();eligible=np.zeros(len(sid),bool);rows=[];fds=[]
    for s in F.SPLITS:
        for pi,p in enumerate(PANELS):
            fd=F.M.static_fd(p);lt=F.LOC.load_local(p,s,'T');ii=np.searchsorted(sid,lt['sid'])
            assert np.array_equal(sid[ii],lt['sid']) and np.array_equal(keys[ii],lt['key'])
            pos=(lt['key']%1000).astype(int);lane=fd['lanes'][pos]
            m=(lt['v51']>F.M.FREE_RATIO*fd['vc'][pos])&~lt['gap'];eligible[ii]=m;jj=np.flatnonzero(m)
            x=T.features(p,s,{'fd':fd},lt,va[ii],qa[ii],pi,False)
            raw=T.predict(teacher,x,jj,iterations=1300)+T.predict(teacher,x,jj,start=1300)
            ex=K.extra(p,s,{'fd':fd},lt,va[ii],qa[ii],False);assert len(ex)==len(jj)
            cap=v0[ii][m]*lane[m]*fd['kjl'][pos[m]]
            reference=C.V.g6(np.maximum(0,np.minimum(qa[ii][m]+np.clip(raw*100,-60,60)*lane[m],cap)))
            assert np.array_equal(reference,q0[ii][m]),('V83 flow replay',s,p)
            residual=np.empty(len(jj));xm=x[jj]
            for start in range(0,len(jj),65536):
                ix=np.arange(start,min(start+65536,len(jj)))
                residual[ix]=model.predict(K.design(xm,ex,ix,raw[ix]),num_threads=4)
            new=q0[ii].copy();pr=raw+alpha*np.clip(residual,-K.EXTRA_CAP,K.EXTRA_CAP)
            new[m]=C.V.g6(np.maximum(0,np.minimum(qa[ii][m]+np.clip(pr*100,-60,60)*lane[m],cap)))
            assert np.isfinite(new).all() and (new>=0).all();q1[ii]=new
            maxchange=float(np.max(np.abs(new[m]-q0[ii][m])/lane[m]));assert maxchange<=alpha*5+.002
            rows.append(dict(split=s,panel=p,eligible=int(m.sum()),changed=int((new!=q0[ii]).sum()),v83_flow_replay_exact=True,
                             max_increment_per_lane=maxchange,rms_increment_per_lane=float(np.sqrt(np.mean(((new[m]-q0[ii][m])/lane[m])**2)))))
            for rr in F.M.fd_frame(p,s,{'key':lt['key']},np.arange(len(ii)),v0[ii],q0[ii],v0[ii],new,fd):fds.append(dict(split=s,panel=p,**rr))
            print('context apply',rows[-1],flush=True)
    changed=q1!=q0;assert eligible[changed].all() and changed.any()
    pd.DataFrame(rows).to_csv(OUT/'apply_changes.csv',index=False);ff=pd.DataFrame(fds);ff.to_csv(OUT/'apply_fd.csv',index=False)
    sm=pd.concat([pd.read_csv(OUT/f'summary_{mode}.csv') for mode in ('parity','half_month','other_month')]);sm=sm[sm.alpha==alpha];assert len(sm)==6;checks={}
    for s in F.SPLITS:
        g=ff[ff.split==s].groupby('panel')[['FD0','FD1']].mean();dfd=F.M.family_mean((g.FD1-g.FD0).to_dict());ss=sm[sm.split==s]
        gain=float((.35*ss.dS+.05*(dfd-.8*ss.dE)).min());assert gain>0,(s,dfd,gain)
        checks[s]=dict(real_frame_dFD=dfd,min_conservative_proxy_gain=gain)
    (OUT/'apply_checks.json').write_text(json.dumps(checks,indent=2))
    z=dict(sid=sid[changed],oldv=v0[changed],newv=v0[changed],oldq=q0[changed],newq=q1[changed])
    np.savez_compressed(path,**z);(OUT/'patch_meta.json').write_text(json.dumps(meta,indent=2));return z


def build():
    PACK.N=K;PACK.OUT=OUT;PACK.BASE=BASE;PACK.BASE_PUBLIC_LB=.90334;PACK.NAME=NAME
    PACK.DEST=BASE.parent/(NAME+'.zip');PACK.PENDING=BASE.parent/(NAME+'.pending.zip');PACK.prepare=prepare;PACK.main()


if __name__=='__main__':build()
