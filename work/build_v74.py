"""V74: incremental low-rank speed correction on confirmed V73 (0.90259).

Optional queue patch is loaded only from a separately verified decision file.
All other values are preserved. Full template + independent byte-diff audit.
"""
from __future__ import annotations
import io
import json
import time
import zipfile
import numpy as np
import pandas as pd
import t1_v74_lowrank as L
import t1_v73_global_factor as F
import t1_v73_factor_verify as VFY
import t1_v57_restack as V
from evaluate import PANELS, KEY_BASE, STEPS

WORK=F.OUT.parents[1]
BASE=WORK/'submissions/v73_factor_guard.zip'
NAME='v74_lowrank'
DEST=WORK/'submissions'/f'{NAME}.zip'


def prepare_speed():
    cache=L.OUT/'patch.npz'
    if cache.exists():
        z=dict(np.load(cache));return z
    summary=pd.read_csv(L.OUT/'summary.csv')
    # Flow-only failed one month/fold check; exclude flow even though a joint
    # candidate could hide that failure under the speed improvement.
    ss=summary[summary.cand=='speed']
    assert (ss.dS_rw>0).all() and (ss.dTotal40>0).all()
    cf=np.load(L.OUT/'coefficients.npz')
    sid,v0,q0=V.zip_state(BASE)
    K=F.M.read_key_state()
    assert np.array_equal(K.submission_id.to_numpy(),sid)
    keys=K.key.to_numpy();del K
    v1=v0.copy();eligible=np.zeros(len(v0),bool);rows=[];fds=[]
    for split in F.SPLITS:
        for p in PANELS:
            fd=F.M.static_fd(p);lt=F.LOC.load_local(p,split,'T')
            ii=np.searchsorted(sid,lt['sid'])
            assert np.array_equal(sid[ii],lt['sid']) and np.array_equal(keys[ii],lt['key'])
            pos=(lt['key']%KEY_BASE).astype(int);tt=((lt['key']//KEY_BASE)%STEPS).astype(int)
            free=(lt['v51']>F.M.FREE_RATIO*fd['vc'][pos])&~lt['gap']
            feat=L.features(p,split,lt['day'],tt,pos,fd,False)
            xx=L.design(feat,v0[ii],lt,'v')
            dv=np.clip(xx@cf[f'{split}__v__all__0'],-1.,1.)
            new=VFY.clamp(p,pos,free,np.where(free,v0[ii]+dv,v0[ii]),q0[ii])
            new=np.where(free,V.g6(new),v0[ii])
            assert np.isfinite(new).all() and (new>0).all()
            v1[ii]=new;eligible[ii]=free
            dd=(new-v0[ii])[free]
            rows.append(dict(split=split,panel=p,changed=int((new!=v0[ii]).sum()),
                             rms_dv=float(np.sqrt(np.mean(dd**2))),max_abs_dv=float(np.max(np.abs(dd)))))
            for r in F.M.fd_frame(p,split,{'key':lt['key']},np.arange(len(ii)),v0[ii],q0[ii],new,q0[ii],fd):
                fds.append(dict(split=split,panel=p,**r))
            print('apply',rows[-1],flush=True)
    change=v1!=v0
    assert np.all(eligible[change])
    assert np.max(np.abs(v1[change]-v0[change]))<=1.001
    pd.DataFrame(rows).to_csv(L.OUT/'apply.csv',index=False)
    fdf=pd.DataFrame(fds);fdf.to_csv(L.OUT/'apply_fd.csv',index=False)
    checks={}
    for split in F.SPLITS:
        g=fdf[fdf.split==split].groupby('panel')[['FD0','FD1']].mean()
        dfd=F.M.family_mean((g.FD1-g.FD0).to_dict())
        a=ss[ss.split==split]
        lower=float((.35*a.dS_rw+.05*(dfd-.8*a.dE)).min())
        assert lower>0,(split,lower)
        checks[split]=dict(real_frame_dFD=dfd,min_proxy_total_gain=lower)
    (L.OUT/'apply_checks.json').write_text(json.dumps(checks,indent=2))
    z=dict(sid=sid[change],old=v0[change],new=v1[change])
    np.savez_compressed(cache,**z)
    return z


def audit(speed,queue):
    from itertools import zip_longest
    n=0;ns=0;nq=0
    speed_idx={int(s):i for i,s in enumerate(speed['sid'])}
    with zipfile.ZipFile(BASE) as za,zipfile.ZipFile(DEST) as zb:
        assert zb.testzip() is None
        with za.open(za.namelist()[0]) as aa,zb.open(zb.namelist()[0]) as bb:
            for a,b in zip_longest(io.BufferedReader(aa,1<<22),io.BufferedReader(bb,1<<22)):
                assert a is not None and b is not None
                if n==0:assert a==b
                else:
                    if a!=b:
                        x=a.rstrip(b'\r\n').split(b',');y=b.rstrip(b'\r\n').split(b',');sid=int(x[0])
                        assert x[0]==y[0] and len(x)==len(y)
                        diff=[i for i,(u,v) in enumerate(zip(x,y)) if u!=v]
                        if sid in queue:
                            assert diff==[4] and x[1]==b'queue' and y[4]==str(queue[sid]['new']).encode()
                            nq+=1
                        else:
                            j=speed_idx[sid]
                            assert diff==[2] and x[1]==b'state'
                            assert float(x[2])==speed['old'][j] and float(y[2])==speed['new'][j]
                            ns+=1
                n+=1
    assert n-1==6980503 and ns==len(speed['sid']) and nq==len(queue),(n,ns,nq)
    return dict(rows=n-1,speed_changes=ns,queue_changes=nq,all_other_bytes_identical=True,crc_ok=True)


def main():
    import sys
    speed=prepare_speed()
    if '--prepare-only' in sys.argv:return
    if DEST.exists():raise FileExistsError(DEST)
    queue={}
    qp=WORK/'results/t2_v74/queue_patch.json'
    if qp.exists():
        js=json.loads(qp.read_text());assert js['approved_by_offline_checks']
        queue={int(k):v for k,v in js['patch'].items()}
    index=np.full(int(speed['sid'].max())+1,-1,np.int32)
    index[speed['sid']]=np.arange(len(speed['sid']),dtype=np.int32)
    ns=0;nq=0;t0=time.time()
    with zipfile.ZipFile(BASE) as zin,zipfile.ZipFile(DEST,'w',zipfile.ZIP_DEFLATED) as zout:
        with zin.open(zin.namelist()[0]) as fin,zout.open(NAME+'.csv','w',force_zip64=True) as fout:
            for i,raw in enumerate(io.BufferedReader(fin,1<<22)):
                if i==0:fout.write(raw);continue
                sid=int(raw.split(b',',1)[0]);j=index[sid] if sid<len(index) else -1
                if j<0 and sid not in queue:fout.write(raw);continue
                body=raw.rstrip(b'\r\n');parts=body.split(b',')
                if sid in queue:
                    assert parts[1]==b'queue' and int(parts[4])==queue[sid]['old']
                    parts[4]=str(queue[sid]['new']).encode();nq+=1
                else:
                    assert parts[1]==b'state' and float(parts[2])==speed['old'][j]
                    parts[2]=('%.6g'%speed['new'][j]).encode();ns+=1
                fout.write(b','.join(parts)+raw[len(body):])
    assert ns==len(speed['sid']) and nq==len(queue)
    print('built; full validation',ns,nq,flush=True)
    from make_submission import validate
    checks=validate(DEST);assert checks['ok'],checks
    independent=audit(speed,queue)
    manifest=dict(name=NAME,base=BASE.name,base_public_lb=.90259,public_lb=None,
                  sha256=F.CM.sha256(DEST),base_sha256=F.CM.sha256(BASE),
                  checks=checks,audit=independent,proxy_checks=json.loads((L.OUT/'apply_checks.json').read_text()),
                  queue_rule=json.loads(qp.read_text())['rule'] if qp.exists() else None,
                  scripts={p.name:F.CM.sha256(p) for p in [__import__('pathlib').Path(__file__),__import__('pathlib').Path(L.__file__)]},
                  seconds=round(time.time()-t0))
    (WORK/'results'/f'{NAME}_manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(manifest,indent=2),flush=True)


if __name__=='__main__':main()
