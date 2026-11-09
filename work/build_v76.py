"""Package an independently validated V76 context mixer on the V75 artifact."""
from __future__ import annotations
import io,json,time,zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
import t1_v76_context_mix as G
import t1_v75_factor_gbm as OLD
import t1_v73_global_factor as F
import t1_v73_factor_verify as VFY
import t1_v57_restack as V
import build_v75 as B
from evaluate import PANELS,KEY_BASE

WORK=Path(__file__).resolve().parent
OUT=G.OUT
BASE=WORK/'submissions/v75_factor_gbm.zip'
ANCHOR=WORK/'submissions/v72_april.zip'
NAME='v76_context_mix'
DEST=WORK/'submissions'/f'{NAME}.zip'


def prepare():
    decision=json.loads((OUT/'decision.json').read_text());cand=decision['chosen'];assert cand
    kind,lim=cand.split('_clip');limit=float(lim)
    patch=OUT/'patch.npz'
    if patch.exists():return dict(np.load(patch)),cand
    sm=pd.read_csv(OUT/'summary.csv');sm=sm[sm.cand==cand]
    assert len(sm)==6 and (sm.dS_rw>0).all() and (sm.dTotal40>0).all()
    model=lgb.Booster(model_file=str(OUT/'model.txt'))
    sid,v0,q0=V.zip_state(BASE);si,va,qa=V.zip_state(ANCHOR)
    assert np.array_equal(sid,si) and np.array_equal(q0,qa)
    key=F.M.read_key_state();assert np.array_equal(sid,key.submission_id.to_numpy())
    keys=key.key.to_numpy();del key
    v1=v0.copy();eligible=np.zeros(len(v0),bool);rows=[];fds=[]
    for s in F.SPLITS:
        for pi,p in enumerate(PANELS):
            fd=F.M.static_fd(p);lt=F.LOC.load_local(p,s,'T');ii=np.searchsorted(sid,lt['sid'])
            assert np.array_equal(sid[ii],lt['sid']) and np.array_equal(keys[ii],lt['key'])
            pos=(lt['key']%KEY_BASE).astype(int);mask=(lt['v51']>F.M.FREE_RATIO*fd['vc'][pos])&~lt['gap']
            fn=G.features if kind=='context' else OLD.features
            x=fn(p,s,{'fd':fd},lt,va[ii],qa[ii],pi,False)
            dv=np.clip(model.predict(x,num_threads=4),-limit,limit)
            new=VFY.clamp(p,pos,mask,np.where(mask,va[ii]+dv,v0[ii]),q0[ii])
            new=np.where(mask,V.g6(new),v0[ii]);assert np.isfinite(new).all() and (new>0).all()
            v1[ii]=new;eligible[ii]=mask;dd=(new-v0[ii])[mask]
            rows.append(dict(split=s,panel=p,changed=int((new!=v0[ii]).sum()),rms_dv=float(np.sqrt(np.mean(dd**2))),max_abs_dv=float(np.max(np.abs(dd)))))
            for r in F.M.fd_frame(p,s,{'key':lt['key']},np.arange(len(ii)),v0[ii],q0[ii],new,q0[ii],fd):fds.append(dict(split=s,panel=p,**r))
            print('apply',rows[-1],flush=True)
    change=v1!=v0;assert eligible[change].all()
    pd.DataFrame(rows).to_csv(OUT/'apply.csv',index=False);ff=pd.DataFrame(fds);ff.to_csv(OUT/'apply_fd.csv',index=False)
    checks={}
    for s in F.SPLITS:
        g=ff[ff.split==s].groupby('panel')[['FD0','FD1']].mean();dfd=F.M.family_mean((g.FD1-g.FD0).to_dict());ss=sm[sm.split==s]
        gain=float((.35*ss.dS_rw+.05*(dfd-.8*ss.dE)).min());assert gain>0,(s,gain)
        checks[s]=dict(real_frame_dFD=dfd,min_proxy_total_gain=gain)
    (OUT/'apply_checks.json').write_text(json.dumps(checks,indent=2))
    z=dict(sid=sid[change],old=v0[change],new=v1[change]);np.savez_compressed(patch,**z);return z,cand


def main():
    if DEST.exists():raise FileExistsError(DEST)
    t0=time.time();z,cand=prepare();ix=np.full(int(z['sid'].max())+1,-1,np.int32);ix[z['sid']]=np.arange(len(z['sid']),dtype=np.int32);seen=0
    with zipfile.ZipFile(BASE) as zin,zipfile.ZipFile(DEST,'w',zipfile.ZIP_DEFLATED) as zout:
        with zin.open(zin.namelist()[0]) as fin,zout.open(NAME+'.csv','w',force_zip64=True) as fout:
            for i,raw in enumerate(io.BufferedReader(fin,1<<22)):
                if i==0:fout.write(raw);continue
                sid=int(raw.split(b',',1)[0]);j=ix[sid] if sid<len(ix) else -1
                if j<0:fout.write(raw);continue
                body=raw.rstrip(b'\r\n');parts=body.split(b',');assert parts[1]==b'state' and float(parts[2])==z['old'][j]
                parts[2]=('%.6g'%z['new'][j]).encode();fout.write(b','.join(parts)+raw[len(body):]);seen+=1
    assert seen==len(z['sid']);print('built; validating',seen,flush=True)
    from make_submission import validate
    checks=validate(DEST);assert checks['ok'],checks
    B.BASE,B.DEST=BASE,DEST;independent=B.audit(z)
    manifest=dict(name=NAME,base=BASE.name,base_public_lb=.90277,public_lb=None,chosen=cand,
        base_sha256=B.sha256(BASE),sha256=B.sha256(DEST),anchor=ANCHOR.name,anchor_sha256=B.sha256(ANCHOR),
        checks=checks,audit=independent,proxy_checks=json.loads((OUT/'apply_checks.json').read_text()),model_sha256=B.sha256(OUT/'model.txt'),
        sources={p.name:B.sha256(p) for p in [Path(__file__),Path(G.__file__),Path(B.__file__)]},seconds=round(time.time()-t0))
    (WORK/'results'/f'{NAME}_manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest,indent=2),flush=True)


if __name__=='__main__':main()
