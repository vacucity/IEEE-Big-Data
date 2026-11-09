"""Exploratory historical-analog queue test; no competition submissions are written.

Train historical library: June 2030 to February 2031. Query: existing March
proxy windows. Only data at or before the forecast origin enters retrieval.
Historical future observations supply noisy proxy labels. March has already been
used for experimentation; this is not an independent final confirmation.
"""
from pathlib import Path
import os, sys, json, zlib, gc, time
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '3'
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'v85_project' / 'work'))
sys.path.insert(0, str(ROOT))
from evaluate import T2_PANELS, links
import t2_v52_selrule as S
import v85_decoders as D
import v85_run_decisions as R

CONFIRM = os.environ.get('ANALOG_CONFIRM') == '1'
AUDIT = os.environ.get('ANALOG_CELL_AUDIT') == '1'
OUT = ROOT.parent / 'outputs' / (('historical-analog-confirmation-cell-audit' if CONFIRM else 'historical-analog-cell-audit') if AUDIT else ('historical-analog-confirmation' if CONFIRM else 'historical-analog-test'))
if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
SRC = Path(os.environ['V85_HANDOFF_CLEAN'])
RESULTS = ROOT / 'v85_project' / 'work' / 'results' / ('v85_confirmation2' if CONFIRM else 'v85_experiment')
ALPHAS = (0.1,) if CONFIRM or AUDIT else (0.1, 0.3, 0.5, 1.0)
K = 30

def compact(speed, eligible, vc, bins):
    """Per time and spatial bin: normalized speed, queue share, coverage."""
    ok = np.isfinite(speed) & eligible
    ratio = np.divide(speed, vc, out=np.zeros_like(speed), where=ok)
    ratio = np.clip(ratio, 0, 2)
    chunks = []
    for indices in bins:
        valid = ok[..., indices]
        count = valid.sum(-1)
        denom = np.maximum(count, 1)
        mean = (np.where(valid, ratio[..., indices], 0).sum(-1) / denom)
        queue = (valid & (speed[..., indices] <= vc[indices])).sum(-1) / denom
        coverage = count / len(indices)
        # Coverage is explicitly part of the metric; missing bins use neutral values.
        chunks.extend((np.where(count, mean, 1.), np.where(count, queue, 0.), coverage))
    return np.stack(chunks, axis=-1).astype(np.float32)

def block_features(compact_data):
    """Three successive four-step history summaries, ending at T inclusive."""
    cs = np.concatenate([np.zeros_like(compact_data[:, :1]), np.cumsum(compact_data, axis=1)], axis=1)
    ts = np.arange(11, 282)
    return np.concatenate([(cs[:, ts - off + 1] - cs[:, ts - off - 3]) / 4 for off in (8, 4, 0)], axis=-1)

def iou(pred, y, mask):
    return D.iou(pred, y, mask)

def load_train(panel, vc, bins, return_dates=False, return_speed=False):
    pf = pq.ParquetFile(SRC / panel / 'train.parquet')
    n = len(vc)
    speeds = np.empty((pf.num_row_groups, 288, n), np.float32)
    eligible = np.empty_like(speeds, bool)
    dates = []
    for i in range(pf.num_row_groups):
        t = pf.read_row_group(i, columns=['date', 'speed_true', 'is_score_eligible'])
        dates.append(str(t.column('date')[0].as_py()))
        speeds[i] = t.column('speed_true').to_numpy(zero_copy_only=False).astype(np.float32).reshape(288, n)
        eligible[i] = t.column('is_score_eligible').to_numpy(zero_copy_only=False).astype(bool).reshape(288, n)
    assert max(dates) < '2031-03-01'
    features = block_features(compact(speeds, eligible, vc, bins))
    valid = np.isfinite(speeds) & eligible
    queued = valid & (speeds <= vc)
    if return_speed:
        return features, queued, valid, speeds, dates
    if return_dates:
        return features, queued, valid, dates
    return features, queued, valid

def main():
    rows = []
    cells = []
    original = pd.read_csv(RESULTS / 'proxy_decisions.csv').set_index('window_id')
    old_analog = pd.read_csv(ROOT.parent / 'outputs' / ('historical-analog-confirmation' if CONFIRM else 'historical-analog-test') / 'window_results.csv').set_index('window_id') if AUDIT else None
    for panel in T2_PANELS:
        if not (original.panel == panel).any():
            continue
        start = time.time()
        meta = links(panel)
        vc = meta.v_cut_kmh.to_numpy(np.float32)
        bins = [x for x in np.array_split(np.arange(len(vc)), min(16, len(vc))) if len(x)]
        train_f, train_q, train_valid = load_train(panel, vc, bins)
        layer = np.load(ROOT / 'v85_project' / 'work' / 'cache' / 't2_v52_selrule' / f'layers_{panel}_validation.npz')
        val_speed, val_e = layer['VIS'], layer['ELIG']
        val_f = block_features(compact(val_speed, val_e, vc, bins))
        probs, z = R.probability(panel, 'validation', 'proxy')
        for i, wid0 in enumerate(z['ids']):
            wid = str(wid0); day = int(z['day'][i]); T = int(z['T'][i]); cls = str(z['cls'][i])
            p = probs[i]; q0 = p
            ids = S.coupling_ids(panel, p, 'group')
            Y, acceptance = D.samples(p, ids, zlib.crc32(wid.encode('ascii')), constrained=(cls == 'constr'))
            p0 = D.ratio(p)
            dissolve = cls == 'constr' and panel == 'D7_I210_E' and T // 12 >= 17
            if dissolve:
                base = S.rule_dissolve(p0)
            elif cls == 'constr' and acceptance >= .02:
                q0 = Y.mean(0).astype(np.float32)
                base = D.ratio(q0); base[0] = p0[0]
            else:
                base = p0
            y, mask = z['y'][i], z['mask'][i]
            score_base = iou(base, y, mask)
            assert abs(score_base - float(original.loc[wid, 'base'])) < 1e-10, wid
            base_intersection = int((base & y & mask).sum())
            base_union = int(((base | y) & mask).sum())
            # Candidate origin lies within +/- 60 minutes of query's clock time.
            times = np.arange(max(11, T - 12), min(281, T + 12) + 1)
            candidates = train_f[:, times - 11, :].reshape(-1, train_f.shape[-1])
            query = val_f[day, T - 11]
            dist = np.mean((candidates - query) ** 2, axis=1)
            ind = np.argpartition(dist, K-1)[:K]
            di, ti = ind // len(times), times[ind % len(times)]
            step = np.arange(1, 7)[None, :]
            obs = train_valid[di[:, None], ti[:, None] + step]
            pastq = train_q[di[:, None], ti[:, None] + step]
            counts = obs.sum(0)
            analog = np.divide((pastq & obs).sum(0), counts, out=q0.copy(), where=counts > 0).astype(np.float32)
            rec = dict(panel=panel, window_id=wid, day=day, T=T, cls=cls, base=score_base,
                       base_intersection=base_intersection, base_union=base_union,
                       nearest_distance=float(np.min(dist[ind])), mean_distance=float(np.mean(dist[ind])),
                       mean_analog_observations=float(counts.mean()), analog_missing_cells=int((counts == 0).sum()))
            for alpha in ALPHAS:
                tag = f'a{int(alpha*100):03d}'
                if dissolve or (cls == 'constr' and acceptance < .02):
                    pred = base.copy()
                else:
                    pred = base ^ D.ratio((1-alpha)*q0+alpha*analog) ^ D.ratio(q0)
                    pred[0] = base[0]
                rec[tag] = iou(pred, y, mask)
                rec[tag+'_flips'] = int((pred != base).sum())
                if AUDIT:
                    assert abs(rec[tag] - float(old_analog.loc[wid,tag])) < 1e-10, wid
                    for h, pos in np.argwhere(pred != base):
                        eligible = bool(mask[h, pos])
                        cells.append(dict(panel=panel, window_id=wid, date=str(layer['dates'][day]), origin_slot=T,
                            forecast_time=str(pd.Timestamp(str(layer['dates'][day])) + pd.Timedelta(minutes=5*(T+h+1))),
                            horizon_minutes=int(5*(h+1)), link_id=str(meta.link_id.iloc[pos]), position=int(pos),
                            before=int(base[h,pos]), after=int(pred[h,pos]), scored_in_proxy=eligible,
                            proxy_label=int(y[h,pos]) if eligible else None,
                            effect=('corrected' if pred[h,pos] == y[h,pos] else 'introduced_error') if eligible else 'unobserved',
                            base_probability=float(q0[h,pos]), analog_probability=float(analog[h,pos]),
                            historical_observation_count=int(counts[h,pos]),
                            window_base=score_base, window_after=rec[tag]))
            rows.append(rec)
        print(panel, 'windows',len(z['ids']), 'seconds',round(time.time()-start,1),flush=True)
        del train_f, train_q, train_valid, val_f, layer, probs, z
        gc.collect()
    df = pd.DataFrame(rows)
    assert len(df) == len(original) == (11 if CONFIRM else 251)
    df.to_csv(OUT / 'window_results.csv', index=False)
    if AUDIT:
        pd.DataFrame(cells).to_csv(OUT / 'changed_cells.csv', index=False)
    summary = {'scope': ('11 distinct additional March proxy windows, same month; NOT a fully independent test' if CONFIRM else 'March 2031 proxy windows, explored before; NOT official score or untouched test'),
               'history': 'June 2030-February 2031, same panel; historical observed speed threshold is noisy proxy',
               'method': '30 nearest compressed histories, origin only and earlier; +/-60min; fixed predeclared alphas; h1 unchanged',
               'windows':len(df), 'scores':{}}
    for alpha in ALPHAS:
        col=f'a{int(alpha*100):03d}'; delta=df[col]-df.base
        item={'changed_windows':int((df[col+'_flips']>0).sum()), 'positive_score_windows':int((delta>1e-12).sum()),
              'negative_score_windows':int((delta< -1e-12).sum())}
        for name,sel in [('early',df.day<15),('late',df.day>=15),('all',np.ones(len(df),bool))]:
            sub=df[sel].copy();sub['delta']=delta[sel];item[name+'_panel_macro_delta']=float(sub.groupby('panel').delta.mean().mean()) if len(sub) else None
        summary['scores'][col]=item
    (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)

if __name__ == '__main__': main()
