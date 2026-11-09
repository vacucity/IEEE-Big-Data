"""Apply validated factor residual correction to V72 state rows and audit the ZIP.

Usage: python work/t1_v73_factor_apply.py
Task 1 offline only. V72 queue/ODME, congested states and T2 blackout states remain
byte-identical. Base never overwritten. Output is submissions/v73_factor.zip.
"""
from __future__ import annotations
import io
import json
import sys
import time
import zipfile
import numpy as np
import pandas as pd
import t1_v73_global_factor as F
import t1_v73_factor_verify as VFY
import t1_v57_restack as V
from evaluate import PANELS, STEPS, KEY_BASE

WORK=F.OUT.parents[1]
BASE=WORK/'submissions/v72_april.zip'
GUARD='--guard' in sys.argv
TAG='_guard' if GUARD else ''
NAME='v73_factor'+TAG
DEST=WORK/'submissions'/f'{NAME}.zip'


def audit(expected):
    from itertools import zip_longest
    n=0;chg=[]
    with zipfile.ZipFile(BASE) as za,zipfile.ZipFile(DEST) as zb:
        assert zb.testzip() is None
        with za.open(za.namelist()[0]) as fa,zb.open(zb.namelist()[0]) as fb:
            for a,b in zip_longest(pd.read_csv(fa,chunksize=400000),pd.read_csv(fb,chunksize=400000)):
                assert a is not None and b is not None and len(a)==len(b)
                for col in a.columns:
                    if col!='speed_kmh':
                        assert np.array_equal(a[col].to_numpy(),b[col].to_numpy()),col
                diff=a.speed_kmh.to_numpy()!=b.speed_kmh.to_numpy()
                assert (a.task[diff]=='state').all()
                chg.extend(a.submission_id.to_numpy()[diff].tolist());n+=len(a)
    assert n==6980503,n
    assert np.array_equal(np.array(chg),expected),('change IDs',len(chg),len(expected))
    return dict(rows=n,changed_speed_rows=len(chg),non_speed_fields_identical=True,zip_crc_ok=True)


def main():
    if DEST.exists():raise FileExistsError(DEST)
    t0=time.time()
    decision=json.loads((F.OUT/f'decision{TAG}.json').read_text())
    cand=decision['chosen'];assert cand is not None
    cf=np.load(F.OUT/f'coefficients{TAG}.npz')
    K=F.M.read_key_state()
    sid,v0,q0=V.zip_state(BASE)
    assert np.array_equal(sid,K.submission_id.to_numpy())
    keys=K.key.to_numpy();del K
    v1=v0.copy();eligible=np.zeros(len(v0),bool);rows=[];fds=[]
    for split in F.SPLITS:
        for p in PANELS:
            fd=F.M.static_fd(p);lt=F.LOC.load_local(p,split,'T')
            ii=np.searchsorted(sid,lt['sid'])
            assert np.array_equal(sid[ii],lt['sid']) and np.array_equal(keys[ii],lt['key'])
            pos=(lt['key']%KEY_BASE).astype(int);tt=((lt['key']//KEY_BASE)%STEPS).astype(int)
            free=(lt['v51']>F.M.FREE_RATIO*fd['vc'][pos])&~lt['gap']
            feat=F.factors(p,split,lt['day'],tt,pos,fd,False)
            xx=F.design(feat,v0[ii],lt)
            dv=xx@cf[cand+'__'+split+'__'+p]
            if GUARD:dv=np.clip(dv,-1.,1.)
            new=np.where(free,v0[ii]+dv,v0[ii])
            new=VFY.clamp(p,pos,free,new,q0[ii])
            # Six significant digits match the existing artifact convention.
            new=np.where(free,V.g6(new),v0[ii])
            assert np.isfinite(new).all() and (new>0).all()
            v1[ii]=new;eligible[ii]=free
            dd=(new-v0[ii])[free]
            rows.append(dict(split=split,panel=p,eligible=int(free.sum()),changed=int(np.sum(new!=v0[ii])),
                             mean_dv=float(dd.mean()),rms_dv=float(np.sqrt(np.mean(dd**2))),
                             p999_abs_dv=float(np.quantile(np.abs(dd),.999)),max_abs_dv=float(np.max(np.abs(dd)))))
            for r in F.M.fd_frame(p,split,{'key':lt['key']},np.arange(len(ii)),v0[ii],q0[ii],new,q0[ii],fd):
                fds.append(dict(split=split,panel=p,**r))
            print(rows[-1],f'{time.time()-t0:.0f}s',flush=True)
    change=v1!=v0
    assert np.all(eligible[change])
    chids=sid[change]
    pd.DataFrame(rows).to_csv(F.OUT/f'apply_corridors{TAG}.csv',index=False)
    fdf=pd.DataFrame(fds);fdf.to_csv(F.OUT/f'apply_fd_frame{TAG}.csv',index=False)
    sm=pd.read_csv(F.OUT/f'summary{TAG}.csv');sm=sm[sm.cand==cand]
    check={}
    for split in F.SPLITS:
        g=fdf[fdf.split==split].groupby('panel')[['FD0','FD1']].mean()
        dfd=F.M.family_mean((g.FD1-g.FD0).to_dict())
        ss=sm[sm.split==split]
        conservative=float((.35*ss.dS_rw+.05*(dfd-.8*ss.dE)).min())
        assert conservative>0,(split,conservative)
        check[split]=dict(real_frame_dFD=dfd,min_validation_estimated_total_gain=conservative)
    np.savez_compressed(F.OUT/f'patch{TAG}.npz',sid=chids,old=v0[change],new=v1[change])
    idx=np.full(int(sid.max())+1,-1,dtype=np.int32)
    idx[sid]=np.arange(len(sid),dtype=np.int32)
    seen=0
    with zipfile.ZipFile(BASE) as zin,zipfile.ZipFile(DEST,'w',zipfile.ZIP_DEFLATED) as zout:
        with zin.open(zin.namelist()[0]) as fin,zout.open(NAME+'.csv','w',force_zip64=True) as fout:
            for line_no,raw in enumerate(io.BufferedReader(fin,1<<22)):
                if line_no==0:fout.write(raw);continue
                submission_id=int(raw.split(b',',1)[0])
                j=idx[submission_id] if submission_id<len(idx) else -1
                if j<0 or not change[j]:fout.write(raw);continue
                body=raw.rstrip(b'\r\n');parts=body.split(b',')
                assert parts[1]==b'state' and float(parts[2])==v0[j]
                parts[2]=('%.6g'%v1[j]).encode()
                fout.write(b','.join(parts)+raw[len(body):]);seen+=1
    assert seen==len(chids)
    print('ZIP built',len(chids),'changed; validating',flush=True)
    from make_submission import validate
    checks=validate(DEST);assert checks['ok'],checks
    independent=audit(chids)
    manifest=dict(name=NAME,base='v72_april',base_public_lb=.90228,
                  base_sha256=F.CM.sha256(BASE),sha256=F.CM.sha256(DEST),
                  chosen=cand,factor_residual_limit_kmh=1.0 if GUARD else None,
                  checks=checks,audit=independent,proxy_checks=check,
                  changed_by_split=pd.DataFrame(rows).groupby('split').changed.sum().to_dict(),
                  coefficient_sha256=F.CM.sha256(F.OUT/f'coefficients{TAG}.npz'),
                  source_scripts={p.name:F.CM.sha256(p) for p in [__import__('pathlib').Path(F.__file__),
                                       __import__('pathlib').Path(VFY.__file__),__import__('pathlib').Path(__file__)]},
                  public_lb=None,seconds=round(time.time()-t0))
    (WORK/'results'/f'{NAME}_manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(manifest,indent=2),flush=True)


if __name__=='__main__':main()
