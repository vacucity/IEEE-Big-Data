"""Month-held-out structured onset prediction, strict pre-origin features.

Joint reward regressors predict the expected masked IoU of each train-derived
complete queue set. No canonical single-block restriction is imposed. Private
confirmation is a separate command after validation selection is written.
"""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='2'
import argparse, json, pickle, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from t2_onset_v27_audit import load_rows, take, score
from t2_pseudo import windows, history, HOR
from evaluate import links, T2_PANELS, RELEASE, slot_of
from task2_ml import ctx_of, to_frame
from task2_ml2 import features2
from task2_rules import onset_scores, DEFAULTS
import lightgbm as lgb

ROOT=Path(__file__).resolve().parent
CACHE=ROOT/'cache'/'t2_v30_onset'
OUT=ROOT/'results'/'t2_v30_onset'
PANELS=['D7_I10_E','D7_I210_E','D7_I210_W','D7_I405_N']
VARIANTS=['rf5','rf15','et5','et15']

def feature(panel,v,q,o,tod,weekday):
    """Only the supplied 12 historical observations; no fitted target priors."""
    L=links(panel)
    feats=[]
    for g in (q/L.capacity_vph.to_numpy(),v/L.free_speed_kmh.to_numpy(),o):
        vals=[np.nanmedian(g[s],axis=0) for s in (slice(-3,None),slice(-6,-3),slice(0,6))]
        vals.extend([vals[0]-vals[1],vals[0]-vals[2]])
        for val in vals:
            feats.append(val)
    local=np.stack(feats,axis=1)
    globalf=np.r_[np.nanquantile(local,[.25,.5,.75],axis=0).ravel(),
                   np.sin(tod*2*np.pi/288),np.cos(tod*2*np.pi/288),weekday>=5]
    return np.r_[local.ravel(),globalf].astype(np.float32)

def rows(panel,split):
    path=CACHE/f'{panel}_{split}.npz'
    if path.exists():
        with np.load(path) as z: return dict(z)
    d=load_rows(panel,split)
    ws=[w for w in windows(panel,split) if w.cond=='onset']
    assert len(ws)==len(d['Y'])
    d['F']=np.stack([feature(panel,*history(w),w.T,w.weekday) for w in ws])
    if split!='train':
        ctx=ctx_of(panel)
        model=lgb.Booster(model_file=str(ROOT/'models'/'t2_ml2_onset_v2.txt'))
        frames=[]
        for w in ws:
            v,q,o=history(w)
            prior=onset_scores(ctx.ev,ctx.n,w.T+HOR,w.weekday,None,DEFAULTS['sigma_h'],DEFAULTS['weekend'])
            f=features2(ctx,v,q,o,w.T,int(w.weekday>=5),prior,T2_PANELS.index(panel))
            frames.append(to_frame(f,np.arange(ctx.n),None)[model.feature_name()])
        d['baseprob']=model.predict(pd.concat(frames,ignore_index=True),num_threads=2).reshape(len(ws),ctx.n)
    CACHE.mkdir(exist_ok=True,parents=True)
    np.savez_compressed(path,**d)
    return d

def reward(candidates,d):
    p=candidates[None]&d['M'][:,None]
    y=d['Y'][:,None]&d['M'][:,None]
    return (p&y).sum(2)/np.maximum((p|y).sum(2),1)

class Joint:
    def __init__(self,variant): self.variant=variant
    def fit(self,d):
        n=d['Y'].shape[1]
        # The inventory, relevant positions and imputation all use this fold only.
        freq=(d['Y']&d['M']).sum(0)/np.maximum(d['M'].sum(0),1)
        active=np.flatnonzero(freq>=.01)
        near=np.unique(np.clip((active[:,None]+np.arange(-2,3)).ravel(),0,n-1))
        self.cols=np.r_[(near[:,None]*15+np.arange(15)).ravel(),np.arange(n*15,d['F'].shape[1])]
        self.imp=SimpleImputer(strategy='median')
        X=self.imp.fit_transform(d['F'][:,self.cols])
        self.candidates=np.unique(d['Y'],axis=0)
        self.candidates=self.candidates[self.candidates.any(1)]
        y=reward(self.candidates,d)
        cls=RandomForestRegressor if self.variant.startswith('rf') else ExtraTreesRegressor
        self.model=cls(n_estimators=180,min_samples_leaf=int(self.variant[2:]),max_features=.6,
                       random_state=3030,n_jobs=2).fit(X,y)
        self.marginal=cls(n_estimators=180,min_samples_leaf=int(self.variant[2:]),max_features=.6,
                         random_state=3030,n_jobs=2).fit(X,d['Y'].astype(float))
        self.constant=self.candidates[y.mean(0).argmax()]
        return self
    def predict(self,d):
        X=self.imp.transform(d['F'][:,self.cols])
        return self.candidates[self.model.predict(X).argmax(1)]
    def margin(self,d): return self.marginal.predict(self.imp.transform(d['F'][:,self.cols]))>=.5

def run_select():
    OUT.mkdir(exist_ok=True,parents=True)
    protocol={'panels':PANELS,'variants':VARIANTS,'seed':3030,'trees':180,'max_features':.6,
      'features':'12 steps strictly before T; per-link flow/capacity,speed/free speed,occupancy recent medians and trends; time and global quartiles',
      'targets':'train full labels at T+30; score masks honor eligibility',
      'selection':'joint variant must improve OOF over its matched marginal regressor, and validation over incumbent ML2 threshold .5. Choose best validation gain per panel. Then fixed private confirmation.',
      'limitation':'Validation/private pseudo labels are visible noisy speed-derived labels, not official latent truth; validation month reused in earlier research.'}
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
    allrows=[]; selected={}
    for panel in PANELS:
        tr=rows(panel,'train'); va=rows(panel,'validation')
        months=np.array([d[:7] for d in tr['date']])
        print(panel,'train',len(tr['Y']),'val',len(va['Y']),flush=True)
        va_base=score(va['baseprob']>=.5,va)
        for date,tod,value in zip(va['date'],va['tod'],va_base): allrows.append((panel,'validation','base',date,int(tod),float(value)))
        for variant in VARIANTS:
            for hold in [*np.unique(months),'all']:
                train=tr if hold=='all' else take(tr,months!=hold)
                d=va if hold=='all' else take(tr,months==hold)
                model=Joint(variant).fit(train)
                split='validation' if hold=='all' else 'oof'
                for kind,pred in [('joint',model.predict(d)),('marginal',model.margin(d)),('constant',np.tile(model.constant,(len(d['Y']),1)))]:
                    for date,tod,value in zip(d['date'],d['tod'],score(pred,d)):
                        allrows.append((panel,split,variant+'_'+kind,date,int(tod),float(value)))
            print(panel,variant,'finished',flush=True)
        df=pd.DataFrame(allrows,columns=['panel','split','variant','date','tod','iou'])
        df.to_csv(OUT/'windows.csv',index=False)
        stats=df[df.panel==panel].groupby(['split','variant']).iou.mean()
        choices=[]
        for variant in VARIANTS:
            joint=stats['validation',variant+'_joint']; marginal=stats['oof',variant+'_marginal']; oof=stats['oof',variant+'_joint']
            if joint>va_base.mean()+.001 and oof>marginal:
                choices.append((joint,variant))
        selected[panel]=max(choices)[1] if choices else None
        print(stats.to_string(), 'SELECTED',selected[panel],flush=True)
    df.groupby(['split','panel','variant']).iou.mean().to_csv(OUT/'summary.csv')
    (OUT/'selected.json').write_text(json.dumps(selected,indent=2))

def official(panel,split):
    n=len(links(panel)); path=RELEASE/'task2'/panel/split
    index=pd.read_csv(path/'window_index.csv'); hist=pd.read_parquet(path/'window_history.parquet')
    hist['pos']=hist.link_id.map(dict(zip(links(panel).link_id,range(n))))
    hist['slot']=slot_of(hist.timestamp)
    records=[]
    for w in index.itertuples(index=False):
        if w.condition!='queue_onset': continue
        origin=pd.Timestamp(w.forecast_origin); T=int(slot_of(pd.Series([w.forecast_origin]))[0])
        h=hist[hist.window_id==w.window_id]; si=(h.slot-(T-12)).to_numpy()
        valid=h.is_score_eligible.astype(bool).to_numpy()&(si>=0)&(si<12)
        grids=[]
        for col in ('speed_kmh','flow_vph','occupancy'):
            a=np.full((12,n),np.nan); a[si[valid],h.pos.to_numpy()[valid]]=h[col].to_numpy()[valid]; grids.append(a)
        # All selected source timestamps are strictly before origin.
        assert (pd.to_datetime(h.loc[valid,'timestamp'])<origin).all()
        records.append((w.window_id,T,feature(panel,*grids,origin.hour*12+origin.minute//5,origin.weekday())))
    return records

def confirm():
    selected=json.loads((OUT/'selected.json').read_text()); output=[]; models={}
    for panel,variant in selected.items():
        if variant is None: continue
        model=Joint(variant).fit(rows(panel,'train')); d=rows(panel,'private')
        for name,pred in [('base',d['baseprob']>=.5),(variant,model.predict(d))]:
            for date,tod,v in zip(d['date'],d['tod'],score(pred,d)):output.append((panel,name,date,int(tod),v))
        models[panel]=model
    df=pd.DataFrame(output,columns=['panel','variant','date','tod','iou'])
    df.to_csv(OUT/'private_windows.csv',index=False)
    print(df.groupby(['panel','variant']).iou.mean().to_string(),flush=True)
    # This exports research alternatives; recommendation still requires confirmation.
    changes=[]
    for split in ('validation','private'):
        base=pd.read_csv(ROOT/'sub'/f'queue_{split}_sota28_group.csv')
        out=base.copy()
        for panel,model in models.items():
            records=official(panel,split)
            d={'F':np.stack([r[2] for r in records])}
            pos=dict(zip(links(panel).link_id,range(len(links(panel)))))
            for (wid,T,_),pred in zip(records,model.predict(d)):
                mask=out.window_id==wid; r=out.loc[mask]
                yy=(slot_of(r.timestamp)==T+HOR)&r.link_id.map(pos).isin(np.flatnonzero(pred)).to_numpy()
                count=int((yy!=r.queue_pred.to_numpy()).sum())
                out.loc[mask,'queue_pred']=yy.astype(int)
                changes.append({'split':split,'panel':panel,'window_id':wid,'changed_cells':count,'positions':np.flatnonzero(pred).tolist()})
        out.to_csv(ROOT/'sub'/f'queue_{split}_v30_onset.csv',index=False)
    (OUT/'official_changes.json').write_text(json.dumps(changes,indent=2))
    print(json.dumps(changes,indent=2),flush=True)

if __name__=='__main__':
    warnings.filterwarnings('ignore',category=RuntimeWarning)
    parser=argparse.ArgumentParser(); parser.add_argument('command',choices=['select','confirm']); args=parser.parse_args()
    {'select':run_select,'confirm':confirm}[args.command]()
