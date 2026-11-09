"""Build an accepted congested-density V81 on the byte-preserved V79 ZIP."""
import io,json,time,zipfile,hashlib
from itertools import zip_longest
import numpy as np
import pandas as pd
import lightgbm as lgb
import t1_v81_density_context as N
from t1_v81_density_context import C,A,F,ROOT,PANELS,OUT
from make_submission import validate
import repro_check as RC

BASE=ROOT/'work/submissions/v79_flow_detail.zip'
BASE_PUBLIC_LB=.90321
NAME='v81_density_context'
DEST=BASE.parent/f'{NAME}.zip'
PENDING=BASE.parent/f'{NAME}.pending.zip'

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for x in iter(lambda:f.read(1<<23),b''):h.update(x)
    return h.hexdigest()

def prepare():
    choice=json.loads((OUT/'selection.json').read_text());assert choice['alpha'] is not None
    for mode in ('half_month','other_month'):
        d=json.loads((OUT/f'decision_{mode}.json').read_text());assert d['passed'] and d['alpha']==choice['alpha']
    N.full();alpha=choice['alpha']
    meta=dict(base_sha256=digest(BASE),alpha=alpha,
        new_density_sha256=digest(OUT/'model_full_r.txt'),new_split_sha256=digest(OUT/'model_full_sstar.txt'))
    RC.base_sha(BASE,'83df8bf65742ce3e0830a94357d6ceb735ea3e7c94f254d30ec4919cb8113ce3','build_v81_density base (V79 v79_flow_detail)',got=meta['base_sha256'])
    path=OUT/'patch.npz'
    if path.exists():
        assert json.loads((OUT/'patch_meta.json').read_text())==meta
        return dict(np.load(path))
    models={k:lgb.Booster(model_file=str(v)) for k,v in {'oldr':A.OUT/'density_l1.txt','olds':A.OUT/'split_l2.txt',
        'newr':OUT/'model_full_r.txt','news':OUT/'model_full_sstar.txt'}.items()}
    sid,v0,q0=C.V.zip_state(BASE);patch=[];rows=[];fds=[]
    for si,s in enumerate(F.SPLITS):
        gf=C.V.gbm_csv(s)
        for pi,p in enumerate(PANELS):
            tg,d,lt=C.target_inputs(p,s,gf);d['key']=tg['key'];m=d['cong']
            ii=np.searchsorted(sid,lt['sid']);assert np.array_equal(sid[ii],lt['sid']) and np.array_equal(tg['key'],lt['key'])
            assert not (m & lt['gap']).any()
            ex=N.extra(p,s,d,hide=False)
            old=np.column_stack([C.cong_feats(d,lt,A.CFG['feats'])[m],np.full(m.sum(),si),np.full(m.sum(),pi)])
            new=np.column_stack([old[:,:-1],ex,old[:,-1]]).astype(np.float32)
            r0=models['oldr'].predict(old,num_threads=2);s0=models['olds'].predict(old,num_threads=2)
            bv,bq=C.corrected(d['pv'][m],d['pq'][m],r0,s0,A.CFG,N.LAM)
            assert np.array_equal(C.V.g6(bv),v0[ii][m]) and np.array_equal(C.V.g6(bq),q0[ii][m]),('V79 congested baseline mismatch',s,p)
            r1=r0+alpha*(models['newr'].predict(new,num_threads=2)-r0)
            s1=s0+alpha*(models['news'].predict(new,num_threads=2)-s0)
            nv,nq=C.corrected(d['pv'][m],d['pq'][m],r1,s1,A.CFG,N.LAM)
            v1,q1=v0[ii].copy(),q0[ii].copy();v1[m],q1[m]=C.V.g6(nv),C.V.g6(nq)
            changed=(v1!=v0[ii])|(q1!=q0[ii]);assert not (changed&~m).any()
            patch.append(dict(sid=sid[ii][changed],oldv=v0[ii][changed],oldq=q0[ii][changed],newv=v1[changed],newq=q1[changed]))
            for r in C.M.fd_frame(p,s,tg,np.arange(len(ii)),v0[ii],q0[ii],v1,q1,d['fd']):fds.append(dict(split=s,panel=p,**r))
            rows.append(dict(split=s,panel=p,eligible=int(m.sum()),changed=int(changed.sum()),baseline_exact=True,
                speed_rms=float(np.sqrt(np.mean((v1[m]-v0[ii][m])**2))),flow_rms=float(np.sqrt(np.mean((q1[m]-q0[ii][m])**2)))))
            print('apply',rows[-1],flush=True)
    ff=pd.DataFrame(fds);ff.to_csv(OUT/'apply_fd.csv',index=False);pd.DataFrame(rows).to_csv(OUT/'apply_changes.csv',index=False)
    sm=pd.concat([pd.read_csv(OUT/f'summary_{mode}.csv') for mode in ('parity','half_month','other_month')]);sm=sm[sm.alpha==alpha]
    checks={}
    for s in F.SPLITS:
        d=ff[ff.split==s].groupby('panel')[['FD0','FD1']].mean();dfd=C.M.family_mean((d.FD1-d.FD0).to_dict());g=sm[sm.split==s]
        gain=float((.35*g.dS+.05*(dfd-.8*g.dE)).min());assert dfd>0 and gain>0,(s,dfd,gain)
        checks[s]=dict(real_frame_dFD=dfd,min_conservative_proxy_gain=gain)
    (OUT/'apply_checks.json').write_text(json.dumps(checks,indent=2))
    z={k:np.concatenate([x[k] for x in patch]) for k in patch[0]};ix=np.argsort(z['sid']);z={k:a[ix] for k,a in z.items()}
    assert len(z['sid'])>0 and len(np.unique(z['sid']))==len(z['sid'])
    np.savez_compressed(path,**z);(OUT/'patch_meta.json').write_text(json.dumps(meta,indent=2));return z

def main():
    if DEST.exists() or PENDING.exists():raise FileExistsError(DEST if DEST.exists() else PENDING)
    start=time.time();z=prepare();ix=np.full(int(z['sid'].max())+1,-1,np.int32);ix[z['sid']]=np.arange(len(z['sid']),dtype=np.int32);seen=0
    with zipfile.ZipFile(BASE) as zin,zipfile.ZipFile(PENDING,'w',zipfile.ZIP_DEFLATED) as zout:
        with zin.open(zin.namelist()[0]) as fin,zout.open(NAME+'.csv','w',force_zip64=True) as fout:
            for i,raw in enumerate(io.BufferedReader(fin,1<<22)):
                if i==0:fout.write(raw);continue
                sid=int(raw.split(b',',1)[0]);j=ix[sid] if sid<len(ix) else -1
                if j<0:fout.write(raw);continue
                body=raw.rstrip(b'\r\n');p=body.split(b',');assert p[1]==b'state' and float(p[2])==z['oldv'][j] and float(p[3])==z['oldq'][j]
                p[2]=('%.6g'%z['newv'][j]).encode();p[3]=('%.6g'%z['newq'][j]).encode()
                fout.write(b','.join(p)+raw[len(body):]);seen+=1
    assert seen==len(z['sid']);print('ZIP written; running complete audit',seen,flush=True)
    checks=validate(PENDING);assert checks['ok'],checks
    changed=nv=nq=count=0
    with zipfile.ZipFile(BASE) as za,zipfile.ZipFile(PENDING) as zb:
        assert zb.testzip() is None
        with za.open(za.namelist()[0]) as fa,zb.open(zb.namelist()[0]) as fb:
            for a,b in zip_longest(io.BufferedReader(fa,1<<22),io.BufferedReader(fb,1<<22)):
                assert a is not None and b is not None
                if count==0:assert a==b
                elif a!=b:
                    aa=a.rstrip(b'\r\n').split(b',');bb=b.rstrip(b'\r\n').split(b',');sid=int(aa[0]);j=ix[sid] if sid<len(ix) else -1
                    assert j>=0 and aa[:2]==bb[:2] and aa[4:]==bb[4:] and aa[1]==b'state'
                    assert float(aa[2])==z['oldv'][j] and float(aa[3])==z['oldq'][j]
                    assert float(bb[2])==z['newv'][j] and float(bb[3])==z['newq'][j]
                    changed+=1;nv+=aa[2]!=bb[2];nq+=aa[3]!=bb[3]
                count+=1
    assert count-1==6980503 and changed==seen
    sha=digest(PENDING);PENDING.replace(DEST)
    manifest=dict(name=NAME,base=BASE.name,baseline_public_lb=BASE_PUBLIC_LB,public_lb=None,sha256=sha,bytes=DEST.stat().st_size,
        changed_state_rows=changed,speed_changes=nv,flow_changes=nq,queue_odme_and_other_state_bytes_identical=True,
        checks=checks,rows=count-1,zip_crc=True,choice=json.loads((OUT/'selection.json').read_text()),
        proxy_checks=json.loads((OUT/'apply_checks.json').read_text()),protocol=N.PROTOCOL,
        model_and_base=json.loads((OUT/'patch_meta.json').read_text()),seconds=round(time.time()-start,1))
    (ROOT/'work/results'/f'{NAME}_manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest,indent=2),flush=True)

if __name__=='__main__':main()
