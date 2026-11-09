"""Apply a validated V78 flow replacement while preserving every other V77 field."""
from __future__ import annotations
import io,json,time,zipfile
from itertools import zip_longest
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
import t1_v78_flow_step as N
import build_v76 as S
import repro_check as RC
from t1_v78_flow_step import F,G,PANELS

WORK=Path(__file__).resolve().parent
OUT=N.OUT
BASE=WORK/'submissions/v77_flow_factors.zip'
ANCHOR=S.ANCHOR
NAME='v78_flow_step'
DEST=WORK/'submissions'/f'{NAME}.zip'


def check_prefix(model):
    old=lgb.Booster(model_file=str(N.P.OUT/'model.txt'))
    a=old.dump_model();b=model.dump_model(num_iteration=N.P.ROUNDS)
    for k in ('feature_names','objective','tree_info','average_output','pandas_categorical'):
        assert a.get(k)==b.get(k),('changed V77 prefix',k)
    (OUT/'full_prefix_audit.json').write_text(json.dumps(dict(original_trees=N.P.ROUNDS,original_tree_structures_identical=True),indent=2))


def prepare():
    choice=json.loads((OUT/'decision.json').read_text());limit=choice['limit'];assert limit is not None
    meta=dict(limit=limit,model_sha256=S.B.sha256(OUT/'model.txt'),base_sha256=S.B.sha256(BASE))
    RC.base_sha(BASE,'b68f1d20666ea6db52cf7e017df495ce23a97add96fab3065eb75ad3dd97029f','build_v78_step base (V77 v77_flow_factors)',got=meta['base_sha256'])
    path=OUT/'patch.npz'
    if path.exists():
        assert json.loads((OUT/'patch_meta.json').read_text())==meta
        return dict(np.load(path))
    sm=pd.read_csv(OUT/'summary.csv');sm=sm[sm.limit==limit]
    assert len(sm)==6 and (sm.dS_rw>0).all() and (sm.dTotal40>0).all()
    sid,v0,q0=S.V.zip_state(BASE);si,va,qa=S.V.zip_state(ANCHOR);assert np.array_equal(sid,si)
    key=F.M.read_key_state();assert np.array_equal(sid,key.submission_id.to_numpy());keys=key.key.to_numpy();del key
    model=lgb.Booster(model_file=str(OUT/'model.txt'));check_prefix(model);q1=q0.copy();eligible=np.zeros(len(sid),bool);rows=[];fds=[]
    for s in F.SPLITS:
        for pi,p in enumerate(PANELS):
            fd=F.M.static_fd(p);lt=F.LOC.load_local(p,s,'T');ii=np.searchsorted(sid,lt['sid'])
            assert np.array_equal(sid[ii],lt['sid']) and np.array_equal(keys[ii],lt['key'])
            pos=(lt['key']%1000).astype(int);lane=fd['lanes'][pos]
            mask=(lt['v51']>F.M.FREE_RATIO*fd['vc'][pos])&~lt['gap'];eligible[ii]=mask
            x=N.features(p,s,{'fd':fd},lt,va[ii],qa[ii],pi,False)
            dq=np.clip(model.predict(x,num_threads=4)*100,-limit,limit)*lane
            new=np.maximum(0,np.minimum(qa[ii]+dq,v0[ii]*lane*fd['kjl'][pos]))
            new=np.where(mask,S.V.g6(new),q0[ii]);q1[ii]=new
            assert np.isfinite(new).all() and (new>=0).all()
            rows.append(dict(split=s,panel=p,changed=int((new!=q0[ii]).sum()),rms_dq_per_lane=float(np.sqrt(np.mean(((new-q0[ii])/lane)[mask]**2)))))
            for r in F.M.fd_frame(p,s,{'key':lt['key']},np.arange(len(ii)),v0[ii],q0[ii],v0[ii],new,fd):fds.append(dict(split=s,panel=p,**r))
            print('apply',rows[-1],flush=True)
    change=q1!=q0;assert eligible[change].all()
    pd.DataFrame(rows).to_csv(OUT/'apply.csv',index=False);ff=pd.DataFrame(fds);ff.to_csv(OUT/'apply_fd.csv',index=False);checks={}
    for s in F.SPLITS:
        d=ff[ff.split==s].groupby('panel')[['FD0','FD1']].mean();dfd=F.M.family_mean((d.FD1-d.FD0).to_dict());ss=sm[sm.split==s]
        gain=float((.35*ss.dS_rw+.05*(dfd-.8*ss.dE)).min());assert gain>0,(s,gain)
        checks[s]=dict(real_frame_dFD=dfd,min_proxy_total_gain=gain)
    (OUT/'apply_checks.json').write_text(json.dumps(checks,indent=2))
    z=dict(sid=sid[change],old=q0[change],new=q1[change]);np.savez_compressed(path,**z)
    (OUT/'patch_meta.json').write_text(json.dumps(meta,indent=2));return z


def audit(z,ix):
    n=nq=0
    with zipfile.ZipFile(BASE) as za,zipfile.ZipFile(DEST) as zb:
        assert zb.testzip() is None
        with za.open(za.namelist()[0]) as fa,zb.open(zb.namelist()[0]) as fb:
            for a,b in zip_longest(io.BufferedReader(fa,1<<22),io.BufferedReader(fb,1<<22)):
                assert a is not None and b is not None
                if n==0:assert a==b
                elif a!=b:
                    x=a.rstrip(b'\r\n').split(b',');y=b.rstrip(b'\r\n').split(b',');sid=int(x[0]);j=ix[sid] if sid<len(ix) else -1
                    assert j>=0 and len(x)==len(y) and x[1]==b'state'
                    assert [i for i,(u,v) in enumerate(zip(x,y)) if u!=v]==[3]
                    assert float(x[3])==z['old'][j] and float(y[3])==z['new'][j];nq+=1
                n+=1
    assert n-1==6980503 and nq==len(z['sid'])
    return dict(rows=n-1,flow_changes=nq,all_other_bytes_identical=True,zip_crc_ok=True)


def main():
    if DEST.exists():raise FileExistsError(DEST)
    t0=time.time();z=prepare();ix=np.full(int(z['sid'].max())+1,-1,np.int32);ix[z['sid']]=np.arange(len(z['sid']),dtype=np.int32);seen=0
    with zipfile.ZipFile(BASE) as zin,zipfile.ZipFile(DEST,'w',zipfile.ZIP_DEFLATED) as zout:
        with zin.open(zin.namelist()[0]) as fin,zout.open(NAME+'.csv','w',force_zip64=True) as fout:
            for i,raw in enumerate(io.BufferedReader(fin,1<<22)):
                if i==0:fout.write(raw);continue
                sid=int(raw.split(b',',1)[0]);j=ix[sid] if sid<len(ix) else -1
                if j<0:fout.write(raw);continue
                body=raw.rstrip(b'\r\n');parts=body.split(b',');assert parts[1]==b'state' and float(parts[3])==z['old'][j]
                parts[3]=('%.6g'%z['new'][j]).encode();fout.write(b','.join(parts)+raw[len(body):]);seen+=1
    assert seen==len(z['sid']);print('built; validating',seen,flush=True)
    from make_submission import validate
    checks=validate(DEST);assert checks['ok'],checks
    independent=audit(z,ix)
    manifest=dict(name=NAME,base=BASE.name,base_public_lb=.90316,public_lb=None,choice=json.loads((OUT/'decision.json').read_text()),
        sha256=S.B.sha256(DEST),base_sha256=S.B.sha256(BASE),anchor=ANCHOR.name,anchor_sha256=S.B.sha256(ANCHOR),
        checks=checks,audit=independent,proxy_checks=json.loads((OUT/'apply_checks.json').read_text()),model_sha256=S.B.sha256(OUT/'model.txt'),
        feature_alignment=json.loads((OUT/'feature_alignment.json').read_text()),
        step_scaling_audit=json.loads((OUT/'scaling_audit.json').read_text()),fold_prefix_audit=json.loads((OUT/'prefix_audit.json').read_text()),full_prefix_audit=json.loads((OUT/'full_prefix_audit.json').read_text()),
        sources={p.name:S.B.sha256(p) for p in [Path(__file__),Path(N.__file__),Path(N.R.__file__),Path(N.P.__file__),Path(G.__file__)]},seconds=round(time.time()-t0))
    (WORK/'results'/f'{NAME}_manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest,indent=2),flush=True)


if __name__=='__main__':main()
