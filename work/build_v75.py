"""Build V75 factor GBM on V74, preserving every non-speed field byte for byte.

The new speed mixer replaces the V73/V74 corrections: it predicts a bounded
V69 residual, not another delta on V74. The V72 artifact contains those same
V69 state values. The comparison and all untouched values use V74.
"""
from __future__ import annotations
import io,json,time,zipfile,hashlib
from itertools import zip_longest
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
import t1_v75_factor_gbm as G
import t1_v73_global_factor as F
import t1_v73_factor_verify as VFY
import t1_v57_restack as V
from evaluate import PANELS,KEY_BASE,STEPS

WORK=Path(__file__).resolve().parent
OUT=G.OUT
BASE=WORK/'submissions/v74_lowrank.zip'
ANCHOR=WORK/'submissions/v72_april.zip'
NAME='v75_factor_gbm'
DEST=WORK/'submissions'/f'{NAME}.zip'


def sha256(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<22),b''):h.update(b)
    return h.hexdigest()


def prepare():
    cache=OUT/'patch.npz'
    if cache.exists():return dict(np.load(cache))
    decision=json.loads((OUT/'decision.json').read_text());assert decision['accepted']
    model=lgb.Booster(model_file=str(OUT/'model.txt'))
    sm=pd.read_csv(OUT/'summary.csv');assert (sm.dS_rw>0).all() and (sm.dTotal40>0).all()
    sid,v0,q0=V.zip_state(BASE);si,va,qa=V.zip_state(ANCHOR)
    assert np.array_equal(sid,si) and np.array_equal(q0,qa)
    K=F.M.read_key_state();assert np.array_equal(sid,K.submission_id.to_numpy())
    keys=K.key.to_numpy();del K
    v1=v0.copy();eligible=np.zeros(len(v0),bool);rows=[];fds=[]
    for s in F.SPLITS:
        for pi,p in enumerate(PANELS):
            fd=F.M.static_fd(p);lt=F.LOC.load_local(p,s,'T')
            ii=np.searchsorted(sid,lt['sid']);assert np.array_equal(sid[ii],lt['sid']) and np.array_equal(keys[ii],lt['key'])
            pos=(lt['key']%KEY_BASE).astype(int)
            mask=(lt['v51']>F.M.FREE_RATIO*fd['vc'][pos])&~lt['gap']
            x=G.features(p,s,{'fd':fd},lt,va[ii],qa[ii],pi,False)
            delta=np.clip(model.predict(x,num_threads=4),-G.LIMIT,G.LIMIT)
            new=VFY.clamp(p,pos,mask,np.where(mask,va[ii]+delta,v0[ii]),q0[ii])
            new=np.where(mask,V.g6(new),v0[ii])
            assert np.isfinite(new).all() and (new>0).all()
            v1[ii]=new;eligible[ii]=mask;dv=(new-v0[ii])[mask]
            rows.append(dict(split=s,panel=p,changed=int((new!=v0[ii]).sum()),rms_dv=float(np.sqrt(np.mean(dv**2))),
                             max_abs_dv=float(np.max(np.abs(dv))),mean_dv=float(dv.mean())))
            for r in F.M.fd_frame(p,s,{'key':lt['key']},np.arange(len(ii)),v0[ii],q0[ii],new,q0[ii],fd):
                fds.append(dict(split=s,panel=p,**r))
            print('applied',rows[-1],flush=True)
    change=v1!=v0;assert eligible[change].all()
    pd.DataFrame(rows).to_csv(OUT/'apply.csv',index=False)
    ff=pd.DataFrame(fds);ff.to_csv(OUT/'apply_fd.csv',index=False);checks={}
    for s in F.SPLITS:
        g=ff[ff.split==s].groupby('panel')[['FD0','FD1']].mean()
        dfd=F.M.family_mean((g.FD1-g.FD0).to_dict());ss=sm[sm.split==s]
        total=float((.35*ss.dS_rw+.05*(dfd-.8*ss.dE)).min())
        assert total>0,(s,total)
        checks[s]=dict(real_frame_dFD=dfd,min_proxy_total_gain=total)
    (OUT/'apply_checks.json').write_text(json.dumps(checks,indent=2))
    z=dict(sid=sid[change],old=v0[change],new=v1[change])
    np.savez_compressed(cache,**z);return z


def audit(z):
    ns=0;n=0
    ix=np.full(int(z['sid'].max())+1,-1,np.int32);ix[z['sid']]=np.arange(len(z['sid']),dtype=np.int32)
    with zipfile.ZipFile(BASE) as za,zipfile.ZipFile(DEST) as zb:
        assert zb.testzip() is None
        with za.open(za.namelist()[0]) as fa,zb.open(zb.namelist()[0]) as fb:
            for a,b in zip_longest(io.BufferedReader(fa,1<<22),io.BufferedReader(fb,1<<22)):
                assert a is not None and b is not None
                if n==0:assert a==b
                elif a!=b:
                    x=a.rstrip(b'\r\n').split(b',');y=b.rstrip(b'\r\n').split(b',')
                    sid=int(x[0]);j=ix[sid] if sid<len(ix) else -1
                    assert j>=0 and len(x)==len(y) and x[1]==b'state'
                    assert [i for i,(u,v) in enumerate(zip(x,y)) if u!=v]==[2]
                    assert float(x[2])==z['old'][j] and float(y[2])==z['new'][j]
                    ns+=1
                n+=1
    assert n-1==6980503 and ns==len(z['sid'])
    return dict(rows=n-1,speed_changes=ns,all_other_bytes_identical=True,zip_crc_ok=True)


def main():
    if DEST.exists():raise FileExistsError(DEST)
    t0=time.time();z=prepare();ix=np.full(int(z['sid'].max())+1,-1,np.int32)
    ix[z['sid']]=np.arange(len(z['sid']),dtype=np.int32);seen=0
    with zipfile.ZipFile(BASE) as zin,zipfile.ZipFile(DEST,'w',zipfile.ZIP_DEFLATED) as zout:
        with zin.open(zin.namelist()[0]) as fin,zout.open(NAME+'.csv','w',force_zip64=True) as fout:
            for line_no,raw in enumerate(io.BufferedReader(fin,1<<22)):
                if line_no==0:fout.write(raw);continue
                sid=int(raw.split(b',',1)[0]);j=ix[sid] if sid<len(ix) else -1
                if j<0:fout.write(raw);continue
                body=raw.rstrip(b'\r\n');parts=body.split(b',')
                assert parts[1]==b'state' and float(parts[2])==z['old'][j]
                parts[2]=('%.6g'%z['new'][j]).encode()
                fout.write(b','.join(parts)+raw[len(body):]);seen+=1
    assert seen==len(z['sid']);print('built; validating',seen,flush=True)
    from make_submission import validate
    checks=validate(DEST);assert checks['ok'],checks
    independent=audit(z)
    manifest=dict(name=NAME,base=BASE.name,base_public_lb=.90261,public_lb=None,
                  base_sha256=sha256(BASE),sha256=sha256(DEST),anchor=ANCHOR.name,anchor_sha256=sha256(ANCHOR),
                  checks=checks,audit=independent,proxy_checks=json.loads((OUT/'apply_checks.json').read_text()),
                  model_sha256=sha256(OUT/'model.txt'),sources={p.name:sha256(p) for p in [Path(__file__),Path(G.__file__)]},
                  seconds=round(time.time()-t0))
    (WORK/'results'/f'{NAME}_manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(manifest,indent=2),flush=True)


if __name__=='__main__':main()
