"""V69: nonlinear second stage on top of the V66 common-mode correction (V66_cm public 0.90223, best).

Probe (work/t1_v68_cmgbm.py, hide frame, odd/even-day cross-fit, vs the V66 hide frame): LightGBM on the V66
residual with the V66 common-mode features (D1, Dw, 18 band means, cnt), local context (db, da, n3, nsp), regime,
speed level (v64/ffs, v64/v_cut) and corridor id; lambda 1: dS +1.07e-4 / +0.56e-4, dTotal40 +0.45e-4 / +0.25e-4
(March / April), dS > 0 in 10/10 corridors (March).  Same row set as V66 (free non-blackout targets with cnt >= 5),
speed only; then the V66 cap (101.2 on the cap links) and k_jam fix.  One model per month, fitted on all of that
month's hide cells (targets: residual vs the cross-fitted V66 hide values).  Task 1 offline.

  python work/t1_v69_cmgbm_apply.py
"""
from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "6")

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
import t1_v66_cmode as CM  # noqa: E402
import t1_v68_cmgbm as G  # noqa: E402
from evaluate import KEY_BASE, PANELS, STEPS, links  # noqa: E402

WORK = Path(__file__).resolve().parent
SUB = WORK / "submissions"
OUT = WORK / "results" / "t1_v69_cmgbm"
BASE, NAME = "v66_cm", "v69_cmgbm"
LAM = 1.0


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for blk in iter(lambda: f.read(1 << 22), b""):
            h.update(blk)
    return h.hexdigest()


def fit_month(s, mi, z64, gr):
    import lightgbm as lgb
    B = {a: CM.solve(gr[f"G_{s}_{a}_0"] + gr[f"G_{s}_{a}_1"]) for a in (0, 1)}
    off = 0
    Xs, ys = [], []
    for pi, p in enumerate(PANELS):
        d, Lh = LOC.load(p, s)
        v64, q64, off = CM.hide_v64(p, s, mi, d, off, z64)
        F = CM.hide_feats(p, s, d)
        m = d["free"] & (F["cnt"] >= CM.MIN_CNT)
        v66 = v64.copy()
        for a in (0, 1):
            te = np.flatnonzero((d["par"] != a) & m)
            v66[te] = np.clip(v64[te] + CM.cm_pred(F, Lh, te, *B[a]), 5.0, 120.75)
        X = G.feats(p, d, Lh, F, v64, pi)
        y = d["ov"] - v66
        k = m & np.isfinite(y)
        Xs.append(X[k])
        ys.append(y[k])
    X, y = np.concatenate(Xs), np.concatenate(ys)
    return lgb.train(G.PRM, lgb.Dataset(X, y, categorical_feature=[X.shape[1] - 1]), G.ROUNDS)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    out_zip = SUB / f"{NAME}.zip"
    if out_zip.exists():
        raise FileExistsError(out_zip)
    t0 = time.time()
    z64 = np.load(CM.Z64)
    gr = np.load(CM.OUT / "grams.npz")
    models = {}
    for mi, s in enumerate(CM.SPLITS):
        models[s] = fit_month(s, mi, z64, gr)
        models[s].save_model(str(OUT / f"gbm_{s}.txt"))
        print("fitted", s, f"{time.time() - t0:.0f}s", flush=True)
    caps = json.loads((CM.OUT / "cap_links.json").read_text())
    capset = {p: set(c["links"]) for p, c in caps.items()}
    K = M.read_key_state()
    sid, v66, q66 = V.zip_state(SUB / f"{BASE}.zip")
    sid64, v64, _ = V.zip_state(SUB / "v64_pool.zip")
    assert np.array_equal(sid, K.submission_id.to_numpy()) and np.array_equal(sid64, sid)
    keys = K.key.to_numpy()
    del K
    patch, rows = {}, []
    for s in CM.SPLITS:
        for pi, p in enumerate(PANELS):
            fd = M.static_fd(p)
            LT = LOC.load_local(p, s, "T")
            ii = np.searchsorted(sid, LT["sid"])
            assert np.array_equal(sid[ii], LT["sid"]) and np.array_equal(keys[ii], LT["key"]), p
            pos = (LT["key"] % KEY_BASE).astype(np.int64)
            tt = ((LT["key"] // KEY_BASE) % STEPS).astype(np.int64)
            free = (LT["v51"] > M.FREE_RATIO * fd["vc"][pos]) & ~LT["gap"]
            F = CM.frame_feats(p, s, LT["day"], tt, pos, fd, hide=False)
            iR = np.flatnonzero(free & (F["cnt"] >= CM.MIN_CNT))
            dT = {"fd": fd, "pos": pos, "reg": LT["regime"].astype(np.int64)}
            X = G.feats(p, dT, LT, F, v64[ii], pi)[iR]
            pred = models[s].predict(X, num_threads=6)
            v0, q0 = v66[ii][iR], q66[ii][iR]
            v1 = np.clip(v0 + LAM * pred, M.SPEED_MIN, M.SPEED_MAX)
            lid = links(p).set_index("pos").link_id.reindex(pos[iR]).to_numpy()
            oncap = np.isin(lid, list(capset.get(p, set())))
            v1 = np.where(oncap, np.minimum(v1, CM.CAP_VALUE), v1)
            kj = (fd["kjl"] * fd["lanes"])[pos[iR]]
            with np.errstate(divide="ignore", invalid="ignore"):
                jam = q0 / v1 > kj
            v1 = V.g6(np.where(jam, q0 / kj, v1))
            ch = v1 != v0
            sR = LT["sid"][iR]
            for k_ in np.flatnonzero(ch):
                patch[int(sR[k_])] = ("%.6g" % v1[k_], "%.6g" % v0[k_])
            dv = (v1 - v0)[ch]
            rows.append({"split": s, "panel": p, "R": len(iR), "changed": int(ch.sum()),
                         "rms_dv": float(np.sqrt(np.mean(dv ** 2))) if len(dv) else 0.0,
                         "max_abs_dv": float(np.abs(dv).max()) if len(dv) else 0.0})
            print(rows[-1], f"{time.time() - t0:.0f}s", flush=True)
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
            assert parts[1] == b"state" and parts[2].decode() == t[1], (i, parts[:4], t)
            parts[2] = t[0].encode()
            dst.write(b",".join(parts) + raw[len(body):])
            seen += 1
    assert seen == len(patch), (seen, len(patch))
    from make_submission import validate
    checks = validate(out_zip)
    if not checks["ok"]:
        raise ValueError(checks)
    pd.DataFrame(rows).to_csv(OUT / "apply_changes.csv", index=False)
    (WORK / "results" / f"{NAME}_manifest.json").write_text(json.dumps({
        "name": NAME, "base": BASE, "base_sha256": sha256(SUB / f"{BASE}.zip"), "output_sha256": sha256(out_zip),
        "changed_state_rows": len(patch), "lam": LAM, "checks": checks,
        "probe": "results/t1_v68_cmgbm/probe.csv (lambda 1: dS +1.07e-4 / +0.56e-4, dTotal40 +0.45e-4 / +0.25e-4)"},
        indent=2), encoding="utf-8")
    print("changed rows", len(patch), checks, f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
