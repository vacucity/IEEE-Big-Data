"""Recompute common-mode features with ALL held-out speed/flow values poisoned."""
import json
import numpy as np
import pyarrow as pa
import t1_v81_congest_cmode as B

def main():
    p,s='D7_I10_E','validation';d=B.N.load_one(p,s);cm=B.CM
    ref=cm.hide_feats(p,s,d);original=cm.pq.ParquetFile
    clean=cm.frame_feats(p,s,d['day'],d['C'].t.to_numpy(),d['pos'],d['fd'],hide=True)
    comparison={k:dict(dtype_cached=str(ref[k].dtype),dtype_recomputed=str(clean[k].dtype),
        max_abs=float(np.max(np.abs(ref[k]-clean[k]))),different=int((ref[k]!=clean[k]).sum())) for k in ref}
    print('cache comparison',comparison,flush=True)
    class PoisonReader:
        def __init__(self,*args,**kwargs):self.file=original(*args,**kwargs);self.num_row_groups=self.file.num_row_groups
        def read_row_group(self,i,columns):
            tb=self.file.read_row_group(i,columns=columns);rows=np.flatnonzero(d['day']==i)
            inds=d['C'].t.to_numpy(dtype=np.int64)[rows]*d['fd']['n']+d['pos'][rows]
            assert (inds>=0).all() and (inds<tb.num_rows).all()
            for name,value in [('speed_obs',1234567.),('flow_obs',-7654321.)]:
                if name not in tb.column_names:continue
                a=tb.column(name).to_numpy().copy();a[inds]=value;j=tb.column_names.index(name)
                tb=tb.set_column(j,name,pa.array(a,type=tb.schema.field(name).type))
            return tb
    cm.pq.ParquetFile=PoisonReader
    try:new=cm.frame_feats(p,s,d['day'],d['C'].t.to_numpy(),d['pos'],d['fd'],hide=True)
    finally:cm.pq.ParquetFile=original
    assert set(new)==set(ref)
    for k in ref:assert np.array_equal(clean[k],new[k],equal_nan=True),('poison changed features',k)
    for k in ref:assert np.allclose(ref[k],clean[k],rtol=0,atol=1e-10,equal_nan=True),('cached feature mismatch',k)
    x0=B.design(clean,d['L']);x1=B.design(new,d['L']);assert np.array_equal(x0,x1)
    result=dict(panel=p,split=s,hidden_cells=len(d['pv']),congested_cells=int(d['cong'].sum()),
        all_hidden_speed_and_flow_replaced=True,all_common_features_exact=True,no_source_files_modified=True,cache_comparison=comparison)
    (B.OUT/'hidden_value_poison_audit.json').write_text(json.dumps(result,indent=2));print(result,flush=True)

if __name__=='__main__':main()
