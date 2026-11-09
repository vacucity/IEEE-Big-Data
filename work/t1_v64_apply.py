"""V64 (lane C round 2): the V61_cg congested non-blackout density model, refitted on BOTH months' hide cells.

Probe (work/t1_v64_pooled.py, V61 frame, odd/even-day cross-fit): V61 as built +5.95e-4 / +7.09e-4 (March / April,
estimated dTotal vs V60); pooled over both months with a month flag, 31 leaves, 1200 / 600 rounds ("pooled31")
+7.42e-4 / +8.71e-4.  V61_cg's LWR-dominated estimate realised in full on the public LB (+0.00058), unlike the
blackout-cell variant (V63, -0.00002).  Expected gain of V64 over V62 ~ +0.00015.

  check   pooled31 cross-fit + the V61 safety rule: family-mean dE_est < 0, dS_state >= 0 and the reweighted
          (real congested non-blackout target neighbourhoods) dS_state >= 0 in EACH month, >= 8/10 corridors with
          dTotal > 0.  Writes results/t1_v64/check.json; apply refuses to run unless it passed.
  apply   fit pooled31 on all hide cells of both months; recompute the congested non-blackout targets (the 269,474
          rows V61_cg changed; V51 production as the base, exactly as V61); patch V62 (v62_cg_b3.zip) -> v64_pool.zip;
          real-frame S_FD gate (family mean must not fall by more than 1e-4 vs V62 in either month); validate.
"""
from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "4")

import hashlib
import io
import json
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import t1_v52_monthbias as M  # noqa: E402
import t1_v57_restack as V  # noqa: E402
import t1_v59_local as LOC  # noqa: E402
import t1_v61_congest as C  # noqa: E402
import t1_v64_pooled as PO  # noqa: E402
from evaluate import PANELS  # noqa: E402

WORK = Path(__file__).resolve().parent
SUB = WORK / "submissions"
OUT = WORK / "results" / "t1_v64"
BASE, NAME = "v62_cg_b3", "v64_pool"
CFG = C.CFGS["F600a75s"]
CANDS = {   # name: (density params, split params, density rounds, split rounds, density-step lambda)
    "pooled31": (dict(C.PRM, num_leaves=31), dict(C.PRM_S, num_leaves=31), 1200, 600, 1.0),
    "pooled15": (C.PRM, C.PRM_S, 600, 300, 1.0),
    "pooled31_l09": (dict(C.PRM, num_leaves=31), dict(C.PRM_S, num_leaves=31), 1200, 600, 0.9),
    "pooled15_l09": (C.PRM, C.PRM_S, 600, 300, 0.9),
}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for blk in iter(lambda: f.read(1 << 22), b""):
            h.update(blk)
    return h.hexdigest()


def load():
    data = {s: C.load_month(s) for s in C.SPLITS}
    P = {s: C.pooled(data[s], CFG["feats"]) for s in C.SPLITS}
    Xs = [PO.with_month(P[s], i) for i, s in enumerate(C.SPLITS)]
    return data, P, Xs


def cmd_check():
    """Pre-declared: among CANDS, eligible = in EACH month family dE_est < 0, dS >= 0, reweighted dS >= 0 and >= 8/10
    corridors dTotal > 0 (the V61 rule); choose the eligible one with the largest mean-over-months dTotal_est."""
    OUT.mkdir(parents=True, exist_ok=True)
    data, P, Xs = load()
    pars = [P[s]["par"] for s in C.SPLITS]
    allres = {}
    for name, (prm_d, prm_s, r_d, r_s, lam) in CANDS.items():
        rh = PO.cross(Xs, [P[s]["r"] for s in C.SPLITS], [P[s]["wd"] for s in C.SPLITS], pars, prm_d, r_d)
        sh = PO.cross(Xs, [P[s]["sstar"] for s in C.SPLITS], [P[s]["ws"] for s in C.SPLITS], pars, prm_s, r_s)
        allres[name] = check_one(data, rh, sh, lam, name)
    elig = {k: v for k, v in allres.items() if v["passed"]}
    chosen = max(elig, key=lambda k: elig[k]["dTot_both"]) if elig else None
    (OUT / "check.json").write_text(json.dumps({"candidates": allres, "chosen": chosen, "passed": chosen is not None},
                                               indent=2))
    print("chosen", chosen, flush=True)


def check_one(data, rh, sh, lam, name):
    res, ok = {}, True
    for i, s in enumerate(C.SPLITS):
        off, recs = 0, []
        for p in PANELS:
            d = data[s][p]
            n = int(d["cong"].sum())
            v, q = C.frame_values(d, rh[i][off:off + n], sh[i][off:off + n], CFG, lam)
            off += n
            wrw, _ = C.target_weights(d, LOC.load_local(p, s, "T"), d["fd"])
            rec = C.score(d, v, q, wrw)
            recs.append({"panel": p, **{k: rec[k] for k in ("dS", "dE_est", "dTotal_est", "S_rw")}})
        g = pd.DataFrame(recs).set_index("panel")
        fam = {k: M.family_mean(g[k].to_dict()) for k in ("dS", "dE_est", "dTotal_est", "S_rw")}
        fam["npos"] = int((g.dTotal_est > 0).sum())
        res[s] = fam
        ok &= fam["dE_est"] < 0 and fam["dS"] >= 0 and fam["S_rw"] >= 0 and fam["npos"] >= 8
        print(name, s, {k: (f"{v * 1e4:+.2f}e-4" if k != "npos" else v) for k, v in fam.items()}, flush=True)
    res["passed"] = bool(ok)
    res["dTot_both"] = (res[C.SPLITS[0]]["dTotal_est"] + res[C.SPLITS[1]]["dTotal_est"]) / 2
    return res


def cmd_apply():
    chk = json.loads((OUT / "check.json").read_text())
    assert chk["passed"], "check did not pass"
    PRM_D, PRM_S, R_D, R_S, LAM = CANDS[chk["chosen"]]
    out_zip = SUB / f"{NAME}.zip"
    if out_zip.exists():
        raise FileExistsError(out_zip)
    t0 = time.time()
    data, P, Xs = load()
    X = np.concatenate(Xs)
    md = C.train(X, np.concatenate([P[s]["r"] for s in C.SPLITS]),
                 np.concatenate([P[s]["wd"] for s in C.SPLITS]), PRM_D, R_D)
    ms = C.train(X, np.concatenate([P[s]["sstar"] for s in C.SPLITS]),
                 np.concatenate([P[s]["ws"] for s in C.SPLITS]), PRM_S, R_S)
    md.save_model(str(OUT / "density_l1.txt"))
    ms.save_model(str(OUT / "split_l2.txt"))
    print("models fitted", f"{time.time() - t0:.0f}s", flush=True)
    K = M.read_key_state()
    # v51_april.zip is gone; the V51 production values are cached in the local-feature data (LT["v51"/"q51"]) and
    # V60 keeps exactly those values on every congested non-blackout target (checked below, as V61 did)
    sid, v60, q60 = V.zip_state(C.V60_ZIP)
    assert np.array_equal(sid, K.submission_id.to_numpy())
    sid62, v62, q62 = V.zip_state(SUB / f"{BASE}.zip")
    assert np.array_equal(sid62, K.submission_id.to_numpy())
    patch, rows, fd_rows = {}, [], []
    for mi, split in enumerate(C.SPLITS):
        gf = V.gbm_csv(split)
        ks = (K.split == split).to_numpy()
        S_ = K[ks].reset_index(drop=True)
        sidS, V60v, V60q, V62v, V62q = S_.submission_id.to_numpy(), v60[ks], q60[ks], v62[ks], q62[ks]
        gidx = S_.groupby("panel", observed=True).indices
        for i, p in enumerate(PANELS):
            idx = gidx[p]
            tg, dT, LT = C.target_inputs(p, split, gf)
            kk = S_.key.to_numpy()[idx]
            j = np.searchsorted(tg["key"], kk)
            assert len(idx) == len(tg["key"]) and np.array_equal(tg["key"][j], kk), p
            a60, b60, a62, b62, sids = (np.empty(len(j)) for _ in range(5))
            a60[j], b60[j], a62[j], b62[j], sids[j] = V60v[idx], V60q[idx], V62v[idx], V62q[idx], sidS[idx]
            a51, b51 = LT["v51"], LT["q51"]
            c = dT["cong"]
            assert np.array_equal(a60[c], a51[c]) and np.array_equal(b60[c], b51[c]), ("V60 lineage", split, p)
            assert not (c & LT["gap"]).any()
            Xc = np.column_stack([C.cong_feats(dT, LT, CFG["feats"]), np.full(len(c), mi, np.float32),
                                  np.full(len(c), i)])[c]
            rT = md.predict(Xc, num_threads=C.THREADS)
            sT = ms.predict(Xc, num_threads=C.THREADS)
            vn, qn = C.corrected(a51[c], b51[c], rT, sT, CFG, LAM)
            v1, q1 = a62.copy(), b62.copy()
            v1[c], q1[c] = V.g6(vn), V.g6(qn)
            assert np.isfinite(v1).all() and np.isfinite(q1).all() and (v1 > 0).all() and (q1 >= 0).all()
            ch = (v1 != a62) | (q1 != b62)
            assert not (ch & ~c).any()
            for k_ in np.flatnonzero(ch):
                patch[int(sids[k_])] = ("%.6g" % v1[k_], "%.6g" % q1[k_], "%.6g" % a62[k_], "%.6g" % b62[k_])
            fd = M.static_fd(p)
            for r_ in M.fd_frame(p, split, tg, np.arange(len(j)), a62, b62, v1, q1, fd):
                fd_rows.append({"split": split, "panel": p, "regime": r_["regime"], "FD_V62": r_["FD0"],
                                "FD_V64": r_["FD1"]})
            rows.append({"split": split, "panel": p, "congested_nonblackout": int(c.sum()), "changed": int(ch.sum()),
                         "mean_dv": float((v1 - a62)[c].mean()), "rms_dv": float(np.sqrt(((v1 - a62)[c] ** 2).mean()))})
            print(rows[-1], f"{time.time() - t0:.0f}s", flush=True)
            del tg, dT, LT
    FDf = pd.DataFrame(fd_rows)
    gate = {}
    for s in C.SPLITS:
        g = FDf[FDf.split == s].groupby("panel")[["FD_V62", "FD_V64"]].mean()
        gate[s] = M.family_mean((g.FD_V64 - g.FD_V62).to_dict())
        assert gate[s] > -1e-4, ("S_FD gate", s, gate[s])
    print("S_FD gate (family mean change)", gate, flush=True)
    src = zipfile.ZipFile(SUB / f"{BASE}.zip")
    seen = 0
    with src.open(src.namelist()[0]) as fin, zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zout, \
            zout.open(f"{NAME}.csv", "w", force_zip64=True) as dst:
        for i, raw in enumerate(io.BufferedReader(fin, 1 << 22)):
            if i == 0:
                dst.write(raw)
                continue
            t = patch.get(int(raw.split(b",", 1)[0]))
            if t is None:
                dst.write(raw)
                continue
            body = raw.rstrip(b"\r\n")
            parts = body.split(b",")
            assert parts[1] == b"state" and parts[2].decode() == t[2] and parts[3].decode() == t[3], (i, parts[:4], t)
            parts[2], parts[3] = t[0].encode(), t[1].encode()
            dst.write(b",".join(parts) + raw[len(body):])
            seen += 1
    assert seen == len(patch), (seen, len(patch))
    from make_submission import validate
    checks = validate(out_zip)
    if not checks["ok"]:
        raise ValueError(checks)
    pd.DataFrame(rows).to_csv(OUT / "apply_changes.csv", index=False)
    FDf.to_csv(OUT / "apply_fd_frame.csv", index=False)
    (WORK / "results" / f"{NAME}_manifest.json").write_text(json.dumps({
        "name": NAME, "base": BASE, "base_sha256": sha256(SUB / f"{BASE}.zip"), "output_sha256": sha256(out_zip),
        "changed_state_rows": len(patch), "check": chk, "fd_gate": gate, "checks": checks,
        "model": {"chosen": chk["chosen"], "pooled": "both months, month flag"}},
        indent=2, default=float), encoding="utf-8")
    print("changed rows", len(patch), checks, f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    {"check": cmd_check, "apply": cmd_apply}[sys.argv[1]]()
