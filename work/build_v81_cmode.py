"""Build the accepted direct congested common-mode candidate on V79."""
import argparse,json
import numpy as np
import pandas as pd
import lightgbm as lgb
import build_v81_density as PACK
import repro_check as RC
import t1_v81_congest_cmode as CM
import t1_v81_congest_speed as SPEED
from t1_v81_congest_cmode import N,C,A,F,ROOT,PANELS

def prepare():
    module=SELECTED;out=module.OUT
    for mode in ('parity','half_month','other_month'):
        if module is CM and mode=='parity':assert json.loads((out/'selection.json').read_text())['kind'] is not None
        else:assert json.loads((out/f'decision_{mode}.json').read_text())['passed']
    beta=module.full();nf=len(beta)
    choice=dict(variant='speed' if module is SPEED else 'iso',kind='wide' if nf==25 else 'band')
    (out/'build_choice.json').write_text(json.dumps(choice,indent=2))
    # The shared writer expects a selection record for its manifest.
    if module is SPEED:(out/'selection.json').write_text(json.dumps(choice,indent=2))
    meta=dict(base_sha256=PACK.digest(PACK.BASE),model_sha256=PACK.digest(out/'beta_full.npy'),**choice)
    RC.base_sha(PACK.BASE,'83df8bf65742ce3e0830a94357d6ceb735ea3e7c94f254d30ec4919cb8113ce3','build_v81_cmode base (V79 v79_flow_detail)',got=meta['base_sha256'])
    path=out/'patch.npz'
    if path.exists():
        assert json.loads((out/'patch_meta.json').read_text())==meta
        return dict(np.load(path))
    oldr=lgb.Booster(model_file=str(A.OUT/'density_l1.txt'));olds=lgb.Booster(model_file=str(A.OUT/'split_l2.txt'))
    sid,v0,q0=C.V.zip_state(PACK.BASE);patch=[];rows=[];fds=[]
    for si,s in enumerate(F.SPLITS):
        gf=C.V.gbm_csv(s)
        for pi,p in enumerate(PANELS):
            tg,d,lt=C.target_inputs(p,s,gf);m=d['cong'];ii=np.searchsorted(sid,lt['sid'])
            assert np.array_equal(sid[ii],lt['sid']) and np.array_equal(tg['key'],lt['key']) and not (m&lt['gap']).any()
            xo=np.column_stack([C.cong_feats(d,lt,A.CFG['feats'])[m],np.full(m.sum(),si),np.full(m.sum(),pi)])
            r=oldr.predict(xo,num_threads=2);sh=olds.predict(xo,num_threads=2);bv,bq=C.corrected(d['pv'][m],d['pq'][m],r,sh,A.CFG,N.LAM)
            assert np.array_equal(C.V.g6(bv),v0[ii][m]) and np.array_equal(C.V.g6(bq),q0[ii][m]),('V79 mismatch',s,p)
            featpath=out/f'T_{s}_{p}.npz';keys=tg['key'][m]
            if featpath.exists():
                zz=np.load(featpath);assert np.array_equal(zz['key'],keys);X=zz['X']
            else:
                cf=CM.CM.frame_feats(p,s,lt['day'][m],keys//1000%288,d['pos'][m],d['fd'],hide=False)
                X=CM.design(cf,{'db':lt['db'][m],'da':lt['da'][m]})[:,:nf]
                np.savez_compressed(featpath,key=keys,X=X)
            vv,qq=module.transform(v0[ii][m],q0[ii][m],X@beta)
            v1,q1=v0[ii].copy(),q0[ii].copy();v1[m],q1[m]=C.V.g6(vv),C.V.g6(qq)
            if module is SPEED:assert np.array_equal(q1,q0[ii])
            change=(v1!=v0[ii])|(q1!=q0[ii]);assert not (change&~m).any()
            patch.append(dict(sid=sid[ii][change],oldv=v0[ii][change],oldq=q0[ii][change],newv=v1[change],newq=q1[change]))
            for rr in C.M.fd_frame(p,s,tg,np.arange(len(ii)),v0[ii],q0[ii],v1,q1,d['fd']):fds.append(dict(split=s,panel=p,**rr))
            rows.append(dict(split=s,panel=p,eligible=int(m.sum()),changed=int(change.sum()),baseline_exact=True,
                max_speed_change=float(np.max(np.abs(v1[m]-v0[ii][m]))),speed_rms=float(np.sqrt(np.mean((v1[m]-v0[ii][m])**2)))))
            print('cm apply',rows[-1],flush=True)
    pd.DataFrame(rows).to_csv(out/'apply_changes.csv',index=False);ff=pd.DataFrame(fds);ff.to_csv(out/'apply_fd.csv',index=False)
    sm=pd.concat([pd.read_csv(out/f'summary_{mode}.csv') for mode in ('parity','half_month','other_month')])
    if module is CM:sm=sm[sm.kind==choice['kind']]
    checks={}
    for s in F.SPLITS:
        g=ff[ff.split==s].groupby('panel')[['FD0','FD1']].mean();dfd=C.M.family_mean((g.FD1-g.FD0).to_dict());ss=sm[sm.split==s]
        gain=float((.35*ss.dS+.05*(dfd-.8*ss.dE)).min());assert gain>0,(s,dfd,gain)
        checks[s]=dict(real_frame_dFD=dfd,min_conservative_proxy_gain=gain)
    (out/'apply_checks.json').write_text(json.dumps(checks,indent=2))
    z={k:np.concatenate([a[k] for a in patch]) for k in patch[0]};order=np.argsort(z['sid']);z={k:a[order] for k,a in z.items()}
    assert len(z['sid'])>0 and len(np.unique(z['sid']))==len(z['sid'])
    np.savez_compressed(path,**z);(out/'patch_meta.json').write_text(json.dumps(meta,indent=2));return z

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--variant',choices=['speed','iso'],required=True);a=ap.parse_args()
    SELECTED=SPEED if a.variant=='speed' else CM
    PACK.N=SELECTED;PACK.OUT=SELECTED.OUT;PACK.NAME='v81_congest_'+a.variant;PACK.DEST=PACK.BASE.parent/(PACK.NAME+'.zip');PACK.PENDING=PACK.BASE.parent/(PACK.NAME+'.pending.zip')
    PACK.prepare=prepare;PACK.main()
