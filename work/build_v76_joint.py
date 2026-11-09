"""Package jointly validated V76 speed and flow residuals on V75."""
from __future__ import annotations
import io,json,time,zipfile
from itertools import zip_longest
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
import build_v76 as S
import t1_v76_joint_check as J
from t1_v76_joint_check import Q,F,G,PANELS

OUT=J.OUT
BASE=S.BASE
NAME='v76_context_flow'
DEST=S.WORK/'submissions'/f'{NAME}.zip'


def prepare():
    choice=json.loads((OUT/'decision.json').read_text());limit=choice['flow_limit'];assert limit is not None
    assert json.loads((Q.OUT/'decision.json').read_text())['accepted']
    patch=OUT/'patch.npz'
    if patch.exists():return dict(np.load(patch))
    sm=pd.read_csv(OUT/'summary.csv');sm=sm[(sm.cand=='joint')&(sm.limit==limit)]
    assert len(sm)==6 and (sm.dS_rw>0).all() and (sm.dTotal40>0).all()
    speed,cand=S.prepare();assert cand==choice['speed_choice']
    sid,v0,q0=S.V.zip_state(BASE);si,va,qa=S.V.zip_state(S.ANCHOR)
    assert np.array_equal(sid,si) and np.array_equal(q0,qa)
    key=F.M.read_key_state();assert np.array_equal(sid,key.submission_id.to_numpy());keys=key.key.to_numpy();del key
    v1=v0.copy();ii=np.searchsorted(sid,speed['sid']);assert np.array_equal(sid[ii],speed['sid']) and np.array_equal(v0[ii],speed['old'])
    v1[ii]=speed['new'];q1=q0.copy();eligible=np.zeros(len(sid),bool);rows=[];fds=[]
    model=lgb.Booster(model_file=str(Q.OUT/'model.txt'))
    for s in F.SPLITS:
        for pi,p in enumerate(PANELS):
            fd=F.M.static_fd(p);lt=F.LOC.load_local(p,s,'T');ii=np.searchsorted(sid,lt['sid'])
            assert np.array_equal(sid[ii],lt['sid']) and np.array_equal(keys[ii],lt['key'])
            pos=(lt['key']%1000).astype(int);lane=fd['lanes'][pos]
            mask=(lt['v51']>F.M.FREE_RATIO*fd['vc'][pos])&~lt['gap'];eligible[ii]=mask
            x=G.features(p,s,{'fd':fd},lt,va[ii],qa[ii],pi,False)
            dq=np.clip(model.predict(x,num_threads=4)*100,-limit,limit)*lane
            new=np.minimum(np.maximum(q0[ii]+dq,0),v1[ii]*lane*fd['kjl'][pos])
            new=np.where(mask,S.V.g6(new),q0[ii]);q1[ii]=new
            assert np.isfinite(new).all() and (new>=0).all()
            rows.append(dict(split=s,panel=p,flow_changes=int((new!=q0[ii]).sum()),rms_dq_per_lane=float(np.sqrt(np.mean(((new-q0[ii])/lane)[mask]**2)))))
            for r in F.M.fd_frame(p,s,{'key':lt['key']},np.arange(len(ii)),v0[ii],q0[ii],v1[ii],new,fd):fds.append(dict(split=s,panel=p,**r))
            print('apply joint',rows[-1],flush=True)
    changed=(v1!=v0)|(q1!=q0);assert eligible[changed].all()
    pd.DataFrame(rows).to_csv(OUT/'apply.csv',index=False);ff=pd.DataFrame(fds);ff.to_csv(OUT/'apply_fd.csv',index=False);checks={}
    for s in F.SPLITS:
        d=ff[ff.split==s].groupby('panel')[['FD0','FD1']].mean();dfd=F.M.family_mean((d.FD1-d.FD0).to_dict());ss=sm[sm.split==s]
        gain=float((.35*ss.dS_rw+.05*(dfd-.8*ss.dE)).min());assert gain>0,(s,gain)
        checks[s]=dict(real_frame_dFD=dfd,min_proxy_total_gain=gain)
    (OUT/'apply_checks.json').write_text(json.dumps(checks,indent=2))
    z=dict(sid=sid[changed],old_v=v0[changed],new_v=v1[changed],old_q=q0[changed],new_q=q1[changed])
    np.savez_compressed(patch,**z);return z


def audit(z,ix):
    n=ns=nq=nr=0
    with zipfile.ZipFile(BASE) as za,zipfile.ZipFile(DEST) as zb:
        assert zb.testzip() is None
        with za.open(za.namelist()[0]) as fa,zb.open(zb.namelist()[0]) as fb:
            for a,b in zip_longest(io.BufferedReader(fa,1<<22),io.BufferedReader(fb,1<<22)):
                assert a is not None and b is not None
                if n==0:assert a==b
                elif a!=b:
                    x=a.rstrip(b'\r\n').split(b',');y=b.rstrip(b'\r\n').split(b',');sid=int(x[0]);j=ix[sid] if sid<len(ix) else -1
                    assert j>=0 and len(x)==len(y) and x[1]==y[1]==b'state'
                    cols=[i for i,(u,v) in enumerate(zip(x,y)) if u!=v];assert set(cols).issubset({2,3})
                    assert float(x[2])==z['old_v'][j] and float(y[2])==z['new_v'][j]
                    assert float(x[3])==z['old_q'][j] and float(y[3])==z['new_q'][j]
                    ns+=2 in cols;nq+=3 in cols;nr+=1
                n+=1
    assert n-1==6980503 and nr==len(z['sid'])
    assert ns==int((z['old_v']!=z['new_v']).sum()) and nq==int((z['old_q']!=z['new_q']).sum())
    return dict(rows=n-1,changed_rows=nr,speed_changes=ns,flow_changes=nq,all_other_bytes_identical=True,zip_crc_ok=True)


def main():
    if DEST.exists():raise FileExistsError(DEST)
    t0=time.time();z=prepare();ix=np.full(int(z['sid'].max())+1,-1,np.int32);ix[z['sid']]=np.arange(len(z['sid']),dtype=np.int32);seen=0
    with zipfile.ZipFile(BASE) as zin,zipfile.ZipFile(DEST,'w',zipfile.ZIP_DEFLATED) as zout:
        with zin.open(zin.namelist()[0]) as fin,zout.open(NAME+'.csv','w',force_zip64=True) as fout:
            for i,raw in enumerate(io.BufferedReader(fin,1<<22)):
                if i==0:fout.write(raw);continue
                sid=int(raw.split(b',',1)[0]);j=ix[sid] if sid<len(ix) else -1
                if j<0:fout.write(raw);continue
                body=raw.rstrip(b'\r\n');parts=body.split(b',')
                assert parts[1]==b'state' and float(parts[2])==z['old_v'][j] and float(parts[3])==z['old_q'][j]
                if z['new_v'][j]!=z['old_v'][j]:parts[2]=('%.6g'%z['new_v'][j]).encode()
                if z['new_q'][j]!=z['old_q'][j]:parts[3]=('%.6g'%z['new_q'][j]).encode()
                fout.write(b','.join(parts)+raw[len(body):]);seen+=1
    assert seen==len(z['sid']);print('built; validating',seen,flush=True)
    from make_submission import validate
    checks=validate(DEST);assert checks['ok'],checks
    independent=audit(z,ix)
    manifest=dict(name=NAME,base=BASE.name,base_public_lb=.90277,public_lb=None,choice=json.loads((OUT/'decision.json').read_text()),
        sha256=S.B.sha256(DEST),base_sha256=S.B.sha256(BASE),anchor=S.ANCHOR.name,anchor_sha256=S.B.sha256(S.ANCHOR),
        checks=checks,audit=independent,proxy_checks=json.loads((OUT/'apply_checks.json').read_text()),
        speed_model_sha256=S.B.sha256(G.OUT/'model.txt'),flow_model_sha256=S.B.sha256(Q.OUT/'model.txt'),
        sources={p.name:S.B.sha256(p) for p in [Path(__file__),Path(J.__file__),Path(Q.__file__),Path(G.__file__),Path(S.__file__)]},seconds=round(time.time()-t0))
    (S.WORK/'results'/f'{NAME}_manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest,indent=2),flush=True)


if __name__=='__main__':main()
