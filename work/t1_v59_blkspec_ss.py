"""V59 EXTRA candidate (phase B, disclosed separately from the pre-registered phase-A choice S2 = v59_blk):
month-self-supervised gap specialist.

Nets: work/models/t1_unet_fold{f}{tag}_v59ss_p{b}.pt = the phase-A specialist (t1_unet_fold{f}{tag}_v59bs.pt)
fine-tuned (t1_v59_blkspec.py train_ss, lr 1e-4, 8 epochs) on the V57 ruler runs of the March/April days with
day % 2 == b (visible cells of the simulated gap rows hidden and supervised by their own observations, 5% random
other visible cells with weight 0.2) mixed 1:1 with train-month official-like gap runs (truth labels).

PRE-REGISTERED here before any phase-B ruler number was computed (wide_ss / crossfit_ss had not been run):
  ruler + rule     t1_v59_blkspec.cmd_crossfit_ss (reference = phase-A S2_v0.625_q0.5 cross-fitted fill; S3 = ss
                   nets alone, S4 = S3 + residual GBM with the S2 feature set, lambda by the V57 rule; a candidate wins
                   iff in BOTH months its family dTotal > the reference's and it beats the reference in >= 7/8
                   corridors per month)
  official use     a window on a day with parity b is filled by the nets fitted on the days with parity 1 - b
                   (exactly the ruler protocol); the parity -1 nets (_v59ss_all, fitted on all ruler days) are NOT used
                   for the candidate (the ruler cannot evaluate them), only reported as a diagnostic
  residual GBM     (S4 only) refit on all ruler rows of both months with the S4 features, lambda from crossfit_ss
  packaging        only the Task-1 targets inside official Task-2 blackouts change against V58 (v58_rs_blk.zip);
                   state-only zip v59blkss_state.zip on the v51_april.zip base (t1_v52_monthbias package/audit code),
                   then build_queue_candidate.py --name v59_blkss with the V54 queues
  recommendation   v59_blkss is recommended over v59_blk only if it wins the phase-B rule AND its block-halves
                   cross-fit and the bootstrap against the reference are positive in both months

Steps (project root, anaconda python):
  wide_ss --split S        (= t1_v59_blkspec.py wide_ss)
  crossfit_ss              (= t1_v59_blkspec.py crossfit_ss)
  official_ss              ss nets on the official windows (other-parity nets) -> results/t1_v59_blkspec/official_ss/
  fit_ss / apply_ss / package_ss / audit_ss / compare_ss      (only if crossfit_ss chose a candidate)
Nothing existing is overwritten (FileExistsError).
"""
from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "4")

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import t1_v59_blkspec as B  # noqa: E402
import t1_unet as U  # noqa: E402
from evaluate import T2_PANELS  # noqa: E402

OUT = B.OUT
PKG = OUT / "ss_pkg"
OFFICIAL_SS = OUT / "official_ss"
NAME = "v59blkss_state"
FULL_NAME = "v59_blkss"
CSV = {s: B.WORK / "sub" / f"state_{s}_v59blkss.csv" for s in B.SPLITS}
GBM_FILES = {c: B.MODELS / f"t1_blk_v59ss_{c}.txt" for c in "vq"}
SS_NETS = {(par, tag, f): B.MODELS / f"t1_unet_fold{f}{tag}{B.SS_SUFFIX}_p{par}.pt"
           for par in (0, 1) for tag in B.TAGS for f in (0, 1)}


def load_choice_ss():
    ch = json.loads((OUT / "crossfit_ss_choice.json").read_text())
    if ch["chosen"] is None:
        raise SystemExit("crossfit_ss chose no candidate: nothing to apply")
    row = ch["chosen_row"]
    return ch["chosen"], row["cand"], float(row["lam_v"]), float(row["lam_q"])


def feats_ss(cand):
    import t1_v57_blackout as V
    return V.FEATS + B.EXTRA_FEATS if cand == "S4" else []


def cmd_official_ss(args) -> None:
    import torch
    _, OW = B.official_gaps()
    nets = {}
    for k, pth in SS_NETS.items():
        m = B.build_model2()
        m.load_state_dict(torch.load(pth, map_location="cpu"))
        m.eval()
        nets[k] = m.to("cuda")
    OFFICIAL_SS.mkdir(parents=True, exist_ok=True)
    for p in T2_PANELS:
        for sp in B.SPLITS:
            path = OFFICIAL_SS / f"{p}_{sp}_win.parquet"
            if path.exists():
                print("exists", path.name, flush=True)
                continue
            st = U.static(p)
            inv = np.argsort(st["perm"])
            src, prof = U.open_split(p, sp), np.load(U.DATA / f"{p}_profiles.npy")
            n = len(st["perm"])
            ow = OW[(OW.panel == p) & (OW.split == sp)]
            recs = []
            for day, g in ow.groupby("day"):
                d = U.Day(p, sp, int(day), src, st, prof)
                gaps = [(int(T), c) for T, c in zip(g["T"], g.cond)]
                assert not (d.obs[..., 3] > 0.5)[np.concatenate([np.arange(T + 1, T + 1 + B.GAP) for T, _ in gaps])].any()
                par = 1 - int(day) % 2
                sub = {(tag, f): nets[par, tag, f] for tag in B.TAGS for f in (0, 1)}
                grid = B.grid_outputs(sub, d, gaps, None, st, inv)
                for T, c in gaps:
                    ks = np.arange(-6, B.GAP + 7)
                    tt = T + ks
                    okr = (tt >= 0) & (tt < B.STEPS)
                    ks, tt = ks[okr], tt[okr]
                    ti, li = np.repeat(tt, n), np.tile(np.arange(n), len(tt))
                    rec = {"day": np.full(len(ti), day, np.int16), "T": np.full(len(ti), T, np.int16),
                           "k": np.repeat(ks, n).astype(np.int8), "pos": li.astype(np.int16)}
                    for nm, a in grid.items():
                        rec[nm] = a[ti, li].astype(np.float32)
                    recs.append(pd.DataFrame(rec))
            pd.concat(recs, ignore_index=True).to_parquet(path, index=False)
            print(p, sp, "official_ss done", flush=True)


def cmd_fit_ss(args) -> None:
    import lightgbm as lgb
    import t1_v57_blackout as V
    name, cand, lv, lq = load_choice_ss()
    if cand == "S3":
        print("S3 needs no GBM")
        return
    for pth in GBM_FILES.values():
        if pth.exists():
            raise FileExistsError(pth)
    _, ALL3, _ = B.load_ruler("_ss")
    feats = feats_ss(cand)
    X = ALL3[feats].to_numpy(np.float32)
    lanes = ALL3.lanes.to_numpy(float)
    ys = {"v": ALL3.ov.to_numpy(float) - ALL3.pv.to_numpy(float),
          "q": (ALL3.oq.to_numpy(float) - ALL3.pq.to_numpy(float)) / lanes}
    for c in "vq":
        lgb.train(V.GBM_PARAMS, lgb.Dataset(X, ys[c], feature_name=feats), V.GBM_ROUNDS).save_model(str(GBM_FILES[c]))
        print("saved", GBM_FILES[c], flush=True)


def _patch_b():
    """Point the phase-A apply/package/compare code of t1_v59_blkspec at the phase-B files."""
    name, cand, lv, lq = load_choice_ss()
    B.OFFICIAL = OFFICIAL_SS
    B.GBM_FILES = GBM_FILES
    B.CSV = CSV
    B.NAME = NAME
    B.FULL_NAME = FULL_NAME
    B.OUT = PKG
    # phase-A code: cand "S0" = specialist alone (no GBM), "S2" = S2 feature set; S3 / S4 are their phase-B analogues
    B.load_choice = lambda: (name, {"S3": "S0", "S4": "S2"}[cand], lv, lq)
    return name, cand, lv, lq


def cmd_apply_ss(args) -> None:
    PKG.mkdir(parents=True, exist_ok=True)
    name, cand, lv, lq = _patch_b()
    B.cmd_apply(args)
    import t1_v52_monthbias as M
    write = {"note": "apply_summary.json in this folder was written by t1_v59_blkspec.cmd_apply; its 'cand' is the "
                     "phase-A alias (S0 = S3, S2 = S4) and its models_sha256 lists the phase-A starting nets; the "
                     "nets and GBM actually used are listed here",
             "chosen": name, "cand": cand, "lam_v": lv, "lam_q": lq}
    write["ss_nets_sha256"] = {f"p{a}{t}_f{f}": M.sha256(pth) for (a, t, f), pth in SS_NETS.items()}
    write["gbm_sha256"] = {c: M.sha256(p_) for c, p_ in GBM_FILES.items() if p_.exists()}
    write["outputs_sha256"] = {s: M.sha256(CSV[s]) for s in B.SPLITS}
    B.write_new(PKG / "apply_supplement.json", json.dumps(write, indent=2))


def cmd_package_ss(args) -> None:
    PKG.mkdir(parents=True, exist_ok=True)
    _patch_b()
    # t1_v52_monthbias.cmd_package reads <OUT>/crossfit_choice.json for its manifest: give it the phase-B choice
    src = OUT / "crossfit_ss_choice.json"
    if not (PKG / "crossfit_choice.json").exists():
        B.write_new(PKG / "crossfit_choice.json", src.read_text(encoding="utf-8"))
    M = B._patch()
    M.cmd_package(args)
    B.write_new(PKG / "v59blkss_state_manifest_supplement.json", json.dumps({
        "note": "the manifest written by the imported t1_v52_monthbias package code carries the V52 note and "
                "state_inputs; the actual state inputs of v59blkss_state.zip are listed here; "
                "ss_pkg/crossfit_choice.json is a verbatim copy of crossfit_ss_choice.json (read by that code)",
        "state_inputs": [{"path": str(CSV[s]), "sha256": M.sha256(CSV[s])} for s in B.SPLITS],
        "content": "all state rows = V58 (v58_rs_blk.zip) except the Task-1 targets inside official Task-2 "
                   "blackouts (V59 phase-B month-self-supervised gap-specialist fill); queue/ODME rows = v51_april.zip "
                   "(replaced by V54 queues in build_queue_candidate)",
        "output_sha256": M.sha256(B.WORK / "submissions" / f"{NAME}.zip")}, indent=2))


def cmd_audit_ss(args) -> None:
    _patch_b()
    M = B._patch()
    M.cmd_audit(args)


def cmd_compare_ss(args) -> None:
    """v59_blkss.zip vs v58_rs_blk.zip with t1_v59_blkspec.cmd_compare (-> ss_pkg/compare_v58.json), then vs v59_blk.zip:
    lines may differ only at Task-1 targets inside official blackouts (-> ss_pkg/compare_v59blk.json)."""
    import zipfile
    _patch_b()
    B.cmd_compare(args)
    path = PKG / "compare_v59blk.json"
    if path.exists():
        raise FileExistsError(path)
    st = {"rows": 0, "differ": 0, "differ_nonstate": 0}
    with zipfile.ZipFile(B.WORK / "submissions" / "v59_blk.zip") as za, \
            zipfile.ZipFile(B.WORK / "submissions" / f"{FULL_NAME}.zip") as zb:
        with za.open(za.namelist()[0]) as fa, zb.open(zb.namelist()[0]) as fb:
            assert fa.readline() == fb.readline()
            for la, lb in zip(fa, fb):
                st["rows"] += 1
                if la != lb:
                    st["differ"] += 1
                    st["differ_nonstate"] += int(la.split(b",", 2)[1] != b"state")
    write_json = json.dumps(st, indent=2)
    B.write_new(path, write_json)
    print(write_json, flush=True)
    assert st["differ_nonstate"] == 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("wide_ss")
    s.add_argument("--split", required=True, choices=B.SPLITS)
    for name in ("crossfit_ss", "official_ss", "fit_ss", "apply_ss", "package_ss", "audit_ss", "compare_ss"):
        sub.add_parser(name)
    args = ap.parse_args()
    {"wide_ss": B.cmd_wide_ss, "crossfit_ss": B.cmd_crossfit_ss, "official_ss": cmd_official_ss, "fit_ss": cmd_fit_ss,
     "apply_ss": cmd_apply_ss, "package_ss": cmd_package_ss, "audit_ss": cmd_audit_ss,
     "compare_ss": cmd_compare_ss}[args.cmd](args)


if __name__ == "__main__":
    main()
