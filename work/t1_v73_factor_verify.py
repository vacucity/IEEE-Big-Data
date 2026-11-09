"""Check the global-factor correction across day folds, months and target-mask strata.

The panel-shrink candidate was added AFTER the initial global probe. It shrinks
each panel's regression toward the pooled fit with a fixed 0.1 diagonal penalty.
No grid search. Both candidates undergo identical parity, half-month and
other-month checks. Final coefficients fit the available same-month holdouts.
"""
from __future__ import annotations
import json
import sys
import numpy as np
import pandas as pd
import t1_v73_global_factor as F
from evaluate import PANELS, links

OUT=F.OUT
GUARD='--guard' in sys.argv
TAG='_guard' if GUARD else ''
CANDS=('global_clip1',) if GUARD else ('global','panel_shrink')


def beta(gs, pool, cand):
    g,b=gs; gg,bb=pool
    b0=F.solve(gg,bb)
    if cand.startswith('global'):
        return b0
    return b0+np.linalg.solve(g+np.diag(np.maximum(np.diag(g),1)*.1),b-g@b0)


def clamp(p, pos, mask, v, q):
    fd=F.M.static_fd(p)
    c=json.loads((F.CM.OUT/'cap_links.json').read_text()).get(p,{}).get('links',[])
    oncap=np.isin(links(p).link_id.to_numpy()[pos],c)&mask
    v=np.clip(v,F.M.SPEED_MIN,F.M.SPEED_MAX)
    v[oncap]=np.minimum(v[oncap],F.CM.CAP_VALUE)
    kj=(fd['kjl']*fd['lanes'])[pos]
    return np.where(mask,np.maximum(v,q/kj),v)


def run():
    data={}; grams={}; pool={}; rows=[]; dailies=[]
    for s in F.SPLITS:
        z=F.baseline(s)
        for p in PANELS:
            d,lh=F.LOC.load(p,s);c=d['C']
            f=F.factors(p,s,c.day.to_numpy(),c.t.to_numpy(),c.pos.to_numpy(),d['fd'],True)
            v,q=z[p+'_v'],z[p+'_q']; x=F.design(f,v,lh);y=d['ov']-v
            m=d['free']&np.isfinite(y)
            lt=F.LOC.load_local(p,s,'T')
            w,unc=F.LOC.target_weights(d,lh,lt,d['fd'])
            data[s,p]=dict(d=d,x=x,y=y,m=m,v=v,q=q,w=w,unc=unc)
            for mode in ('parity','half_month','all'):
                fold=d['par'] if mode=='parity' else (d['day']>=15).astype(int)
                for a in (0,1) if mode!='all' else (0,):
                    k=m & ((fold==a) if mode!='all' else True)
                    g,b=F.gram(x[k],y[k]);grams[s,p,mode,a]=(g,b)
                    key=s,mode,a
                    if key not in pool: pool[key]=(g.copy(),b.copy())
                    else: pool[key]=(pool[key][0]+g,pool[key][1]+b)
        print('loaded',s,flush=True)
    coeff={}
    for cand in CANDS:
        for s in F.SPLITS:
            for p in PANELS:
                coeff[cand+'__'+s+'__'+p]=beta(grams[s,p,'all',0],pool[s,'all',0],cand)
            for mode in ('parity','half_month','other_month'):
                for p in PANELS:
                    a=data[s,p];d=a['d'];delta=np.zeros(len(a['v']))
                    if mode=='other_month':
                        other=F.SPLITS[1-F.SPLITS.index(s)]
                        b=beta(grams[other,p,'all',0],pool[other,'all',0],cand)
                        delta[a['m']]=a['x'][a['m']]@b
                    else:
                        fold=d['par'] if mode=='parity' else (d['day']>=15).astype(int)
                        for k in (0,1):
                            ix=a['m']&(fold!=k)
                            b=beta(grams[s,p,mode,k],pool[s,mode,k],cand)
                            delta[ix]=a['x'][ix]@b
                    if cand=='global_clip1':
                        delta=np.clip(delta,-1.,1.)
                    vv=clamp(p,d['pos'],a['m'],a['v']+delta,a['q'])
                    s0=F.M.state_score(d['C'],a['v'],a['q'],d['fd'])
                    s1=F.M.state_score(d['C'],vv,a['q'],d['fd'])
                    st=(d['ov'],d['oq'],d['fd']['lanes'][d['pos']],d['reg'],a['w'])
                    rw=F.LOC.s_state_w(vv,a['q'],*st)-F.LOC.s_state_w(a['v'],a['q'],*st)
                    a0=F.M.t3_score(d['C'],d['T'],d['Dd'],a['v'],a['q'],d['fd'])
                    a1=F.M.t3_score(d['C'],d['T'],d['Dd'],vv,a['q'],d['fd'])
                    dt=F.M.t3_delta(a0,a1,d['sc']);ds=s1['S']-s0['S']
                    rows.append(dict(cand=cand,split=s,mode=mode,panel=p,dS=ds,dS_rw=rw,
                                     dE=dt['dE_est'],dFD=dt['dFD_est'],dTotal40=.35*rw+.05*(dt['dFD_est']-.8*dt['dE_est']),
                                     rms=float(np.sqrt(np.mean(delta[a['m']]**2))),uncovered=a['unc']))
                    if mode=='parity':
                        np.savez_compressed(OUT/f'check_{cand}_{s}_{p}.npz',v0=a['v'],v1=vv,q=a['q'],mask=a['m'])
                        for day in np.unique(d['day']):
                            k=d['day']==day
                            dailies.append(dict(cand=cand,split=s,panel=p,day=int(day),reg=int(d['reg'][k][0]),
                                                n=float(a['w'][k].sum()),e0=float(np.sum(a['w'][k]*(a['v'][k]-d['ov'][k])**2)),
                                                e1=float(np.sum(a['w'][k]*(vv[k]-d['ov'][k])**2))))
                rr=pd.DataFrame(rows);rr=rr[(rr.cand==cand)&(rr.split==s)&(rr['mode']==mode)]
                print(cand,s,mode,{k:F.M.family_mean(rr.set_index('panel')[k].to_dict()) for k in ('dS','dS_rw','dTotal40')},
                      'positive',int((rr.dS>0).sum()),flush=True)
    r=pd.DataFrame(rows);r.to_csv(OUT/f'robust{TAG}.csv',index=False)
    dd=pd.DataFrame(dailies);dd.to_csv(OUT/f'daily{TAG}.csv',index=False)
    np.savez(OUT/f'coefficients{TAG}.npz',**coeff)
    summary=[]
    for (cand,s,mode),g in r.groupby(['cand','split','mode']):
        summary.append(dict(cand=cand,split=s,mode=mode,**{k:F.M.family_mean(g.set_index('panel')[k].to_dict()) for k in ('dS','dS_rw','dE','dFD','dTotal40')}))
    sm=pd.DataFrame(summary);sm.to_csv(OUT/f'summary{TAG}.csv',index=False)
    # Descriptive date-cluster uncertainty, with all corridors sharing each date draw.
    rng=np.random.default_rng(73073);cis={}
    for s in F.SPLITS:
        nd=int(dd[dd.split==s].day.max())+1
        draws=rng.multinomial(nd,np.ones(nd)/nd,size=1000)
        for cand in CANDS:
            vals={}
            for p in PANELS:
                a=dd[(dd.split==s)&(dd.cand==cand)&(dd.panel==p)];ps=[]
                for reg in sorted(a.reg.unique()):
                    b=a[a.reg==reg];w=draws[:,b.day.to_numpy()]
                    n=w@b.n.to_numpy();ee0=w@b.e0.to_numpy();ee1=w@b.e1.to_numpy()
                    ps.append(.54/25*(np.sqrt(ee0/np.maximum(n,1))-np.sqrt(ee1/np.maximum(n,1))))
                vals[p]=np.mean(ps,axis=0)
            from evaluate import FAMILY
            fam=[np.mean([v for p,v in vals.items() if FAMILY[p]==f],axis=0) for f in set(FAMILY.values())]
            ci=np.quantile(np.mean(fam,axis=0),[.025,.5,.975]).tolist()
            cis[cand+'__'+s]=ci
    eligible=[]
    for cand in CANDS:
        g=sm[sm.cand==cand]
        if (g.dS_rw>0).all() and (g.dTotal40>0).all() and all(cis[cand+'__'+s][0]>0 for s in F.SPLITS):
            eligible.append(cand)
    chosen=max(eligible,key=lambda c:float(sm[sm.cand==c].dTotal40.min())) if eligible else None
    if GUARD and chosen is not None:
        old=pd.read_csv(OUT/'summary.csv')
        old=old[old.cand=='global'].set_index(['split','mode'])
        cur=sm.set_index(['split','mode'])
        if not (cur.dTotal40>old.dTotal40.reindex(cur.index)).all():
            chosen=None
    decision=dict(chosen=chosen,bootstrap_weighted_dS=cis,selection='maximise worst month/mode dTotal40 among positive reweighted checks and positive date-bootstrap lower bounds',
                  caveat='visible artificial holdouts; not official hidden scores; panel_shrink added after initial exploratory probe; clip1 added after diagnosing the harmful unbounded tails',
                  additional_guard_gate='if --guard, all six month/mode total-gain estimates must exceed the unbounded global candidate')
    (OUT/f'decision{TAG}.json').write_text(json.dumps(decision,indent=2))
    print(json.dumps(decision,indent=2),flush=True)


if __name__=='__main__':run()
