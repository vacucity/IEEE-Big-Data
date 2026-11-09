"""Build the accepted V83 flow continuation on immutable V82."""
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
import t1_v83_flow_continue as K
from t1_v83_flow_continue import Q,F,G,P,PANELS,N,C,ROOT,OUT
import build_v81_density as PACK
import build_v76 as AN
import repro_check as RC

BASE=ROOT/'work/submissions/v82_congest_interactions.zip'
NAME='v83_flow_continue'
ANCHOR=AN.ANCHOR


def prepare():
    choice=K.full();model=lgb.Booster(model_file=str(OUT/'model.txt'))
    meta=dict(base_sha256=PACK.digest(BASE),model_sha256=PACK.digest(OUT/'model.txt'),
              anchor_sha256=PACK.digest(ANCHOR),teacher_sha256=PACK.digest(Q.OUT/'model.txt'),alpha=choice['alpha'])
    RC.base_sha(BASE,'a4b4676d74607080bb2fb06cada210412220473b323d0b1ef8ce1561b50b8f34','build_v83_flow base (V82 v82_congest_interactions)',got=meta['base_sha256'])
    old=lgb.Booster(model_file=str(Q.OUT/'model.txt'));a=old.dump_model();b=model.dump_model(num_iteration=K.ORIGINAL)
    for k in ('feature_names','objective','tree_info','average_output','pandas_categorical'):assert a.get(k)==b.get(k),k
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
            m=(lt['v51']>F.M.FREE_RATIO*fd['vc'][pos])&~lt['gap'];eligible[ii]=m
            x=K.features(p,s,{'fd':fd},lt,va[ii],qa[ii],pi,False);jj=np.flatnonzero(m)
            ref=K.predict(model,x,jj,iterations=K.ORIGINAL)
            inc=K.predict(model,x,jj,start=K.ORIGINAL)
            cap=v0[ii][m]*lane[m]*fd['kjl'][pos[m]]
            reconstruct=C.V.g6(np.maximum(0,np.minimum(qa[ii][m]+np.clip(ref*100,-K.LIMIT,K.LIMIT)*lane[m],cap)))
            assert np.array_equal(reconstruct,q0[ii][m]),('V82 free flow replay',s,p,float(np.max(np.abs(reconstruct-q0[ii][m]))))
            new=q0[ii].copy();new[m]=C.V.g6(np.maximum(0,np.minimum(qa[ii][m]+np.clip((ref+inc)*100,-K.LIMIT,K.LIMIT)*lane[m],cap)))
            assert np.isfinite(new).all() and (new>=0).all();q1[ii]=new
            rows.append(dict(split=s,panel=p,eligible=int(m.sum()),changed=int((new!=q0[ii]).sum()),v82_flow_replay_exact=True,
                             rms_increment_per_lane=float(np.sqrt(np.mean(((new[m]-q0[ii][m])/lane[m])**2)))))
            for rr in F.M.fd_frame(p,s,{'key':lt['key']},np.arange(len(ii)),v0[ii],q0[ii],v0[ii],new,fd):fds.append(dict(split=s,panel=p,**rr))
            print('flow apply',rows[-1],flush=True)
    changed=q1!=q0;assert eligible[changed].all() and changed.any()
    pd.DataFrame(rows).to_csv(OUT/'apply_changes.csv',index=False);ff=pd.DataFrame(fds);ff.to_csv(OUT/'apply_fd.csv',index=False)
    sm=pd.concat([pd.read_csv(OUT/f'summary_{mode}.csv') for mode in ('parity','half_month','other_month')]);sm=sm[sm.alpha==choice['alpha']]
    assert len(sm)==6;checks={}
    for s in F.SPLITS:
        g=ff[ff.split==s].groupby('panel')[['FD0','FD1']].mean();dfd=F.M.family_mean((g.FD1-g.FD0).to_dict());ss=sm[sm.split==s]
        gain=float((.35*ss.dS+.05*(dfd-.8*ss.dE)).min());assert gain>0,(s,dfd,gain)
        checks[s]=dict(real_frame_dFD=dfd,min_conservative_proxy_gain=gain)
    (OUT/'apply_checks.json').write_text(json.dumps(checks,indent=2))
    z=dict(sid=sid[changed],oldv=v0[changed],newv=v0[changed],oldq=q0[changed],newq=q1[changed])
    np.savez_compressed(path,**z);(OUT/'patch_meta.json').write_text(json.dumps(meta,indent=2));return z


if __name__=='__main__':
    PACK.N=K;PACK.OUT=OUT;PACK.BASE=BASE;PACK.BASE_PUBLIC_LB=.90333;PACK.NAME=NAME
    PACK.DEST=BASE.parent/(NAME+'.zip');PACK.PENDING=BASE.parent/(NAME+'.pending.zip');PACK.prepare=prepare;PACK.main()
