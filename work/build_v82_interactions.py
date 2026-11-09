"""Build the frozen V82 common-mode interaction residual on audited V81."""
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
import t1_v82_cmode_interactions as I
from t1_v82_cmode_interactions import V,S,B,N,C,A,F,ROOT,PANELS,OUT
import build_v81_density as PACK
import repro_check as RC

BASE=ROOT/'work/submissions/v81_congest_speed.zip'
NAME='v82_congest_interactions'


def prepare():
    choice=json.loads((OUT/'selection.json').read_text());alpha=choice['alpha'];assert alpha is not None
    for mode in ('half_month','other_month'):
        decision=json.loads((OUT/f'decision_{mode}.json').read_text())
        assert decision['passed'] and decision['alpha']==alpha
    poison=json.loads((B.OUT/'hidden_value_poison_audit.json').read_text())
    assert poison['all_common_features_exact'] and poison['all_hidden_speed_and_flow_replaced']
    beta=I.full();beta0=S.full()
    meta=dict(base_sha256=PACK.digest(BASE),model_sha256=PACK.digest(OUT/'beta_full.npy'),
              incumbent_beta_sha256=PACK.digest(S.OUT/'beta_full.npy'),alpha=alpha,kind='interaction')
    RC.base_sha(BASE,'785c1d3751017d559a9fe8739bd5c7e2cc728d21fe865fea245b9f6581fe0245','build_v82_interactions base (V81 v81_congest_speed)',got=meta['base_sha256'])
    path=OUT/'patch.npz'
    if path.exists():
        assert json.loads((OUT/'patch_meta.json').read_text())==meta
        return dict(np.load(path))
    # The existing production teacher, before the V81 common-mode correction.
    md=lgb.Booster(model_file=str(A.OUT/'density_l1.txt'))
    ms=lgb.Booster(model_file=str(A.OUT/'split_l2.txt'))
    sid,v0,q0=C.V.zip_state(BASE);patch=[];rows=[];fds=[]
    for si,s in enumerate(F.SPLITS):
        gf=C.V.gbm_csv(s)
        for pi,p in enumerate(PANELS):
            tg,d,lt=C.target_inputs(p,s,gf);m=d['cong'];ii=np.searchsorted(sid,lt['sid'])
            assert np.array_equal(sid[ii],lt['sid']) and np.array_equal(tg['key'],lt['key'])
            assert not (m & lt['gap']).any()
            xo=np.column_stack([C.cong_feats(d,lt,A.CFG['feats'])[m],np.full(m.sum(),si),np.full(m.sum(),pi)])
            va,qa=C.corrected(d['pv'][m],d['pq'][m],md.predict(xo,num_threads=2),ms.predict(xo,num_threads=2),A.CFG,N.LAM)
            va,qa=C.V.g6(va),C.V.g6(qa);assert np.array_equal(qa,q0[ii][m])
            cp=OUT/f'cm_T_{s}_{p}.npz';key=lt['key'][m]
            if cp.exists():
                zz=np.load(cp);assert np.array_equal(zz['key'],key);cmx=zz['X']
            else:
                cf=B.CM.frame_feats(p,s,lt['day'][m],key//1000%288,d['pos'][m],d['fd'],hide=False)
                cmx=B.design(cf,{'db':lt['db'][m],'da':lt['da'][m]})
                np.savez_compressed(cp,key=key,X=cmx)
            original=np.load(S.OUT/f'T_{s}_{p}.npz')
            assert np.array_equal(original['key'],key) and np.array_equal(original['X'],cmx[:,:25])
            cm=cmx[:,:25]@beta0;vb,_=S.transform(va,qa,cm)
            assert np.array_equal(C.V.g6(vb),v0[ii][m]),('V81 reproduction',s,p)
            residual=I.features(xo,cmx)@beta
            vn,_=S.transform(va,qa,V.delta(cm,residual,'interaction',alpha))
            v1=v0[ii].copy();v1[m]=C.V.g6(vn);change=v1!=v0[ii]
            assert not (change & ~m).any()
            maximum=float(np.max(np.abs(v1[m]-v0[ii][m])))
            assert maximum<=alpha*V.EXTRA_CAP+2e-4
            patch.append(dict(sid=sid[ii][change],oldv=v0[ii][change],oldq=q0[ii][change],newv=v1[change],newq=q0[ii][change]))
            for rr in C.M.fd_frame(p,s,tg,np.arange(len(ii)),v0[ii],q0[ii],v1,q0[ii],d['fd']):
                fds.append(dict(split=s,panel=p,**rr))
            rows.append(dict(split=s,panel=p,eligible=int(m.sum()),changed=int(change.sum()),
                             v81_baseline_exact=True,common_feature_cache_exact=True,max_increment=maximum))
            print('apply',rows[-1],flush=True)
    pd.DataFrame(rows).to_csv(OUT/'apply_changes.csv',index=False)
    ff=pd.DataFrame(fds);ff.to_csv(OUT/'apply_fd.csv',index=False)
    sm=pd.concat([pd.read_csv(OUT/f'summary_{mode}.csv') for mode in ('parity','half_month','other_month')])
    sm=sm[sm.alpha==alpha];assert len(sm)==6;checks={}
    for s in F.SPLITS:
        g=ff[ff.split==s].groupby('panel')[['FD0','FD1']].mean()
        dfd=C.M.family_mean((g.FD1-g.FD0).to_dict());ss=sm[sm.split==s]
        gain=float((.35*ss.dS+.05*(dfd-.8*ss.dE)).min());assert gain>0,(s,dfd,gain)
        checks[s]=dict(real_frame_dFD=dfd,min_conservative_proxy_gain=gain)
    (OUT/'apply_checks.json').write_text(json.dumps(checks,indent=2))
    z={k:np.concatenate([a[k] for a in patch]) for k in patch[0]};order=np.argsort(z['sid'])
    z={k:a[order] for k,a in z.items()};assert len(z['sid'])>0 and len(np.unique(z['sid']))==len(z['sid'])
    assert np.array_equal(z['oldq'],z['newq'])
    np.savez_compressed(path,**z);(OUT/'patch_meta.json').write_text(json.dumps(meta,indent=2))
    return z


def build():
    PACK.N=I;PACK.OUT=OUT;PACK.BASE=BASE;PACK.BASE_PUBLIC_LB=.90329;PACK.NAME=NAME
    PACK.DEST=BASE.parent/(NAME+'.zip');PACK.PENDING=BASE.parent/(NAME+'.pending.zip')
    PACK.prepare=prepare;PACK.main()


if __name__=='__main__':build()
