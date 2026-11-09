#!/usr/bin/env python
"""Single entry point of the final package: rebuilds the two selected submissions stage by stage.

Final submissions (team vacucity; private leaderboard: V86 0.89962, V87 0.89956)
  V87 = work/submissions/v87_combined.zip         sha256 6bc4e383...  public 0.90368: V84 + 104 Task 2 queue_pred flips
      (the 94 of V86 and 10 of the V85 history guard)
  V86 = work/submissions/v86_spacetime_guard.zip  sha256 9e1ba052...  public 0.90359: V84 + 94 queue_pred flips of the
      spatio-temporal guard
  Both are V84 (below, public 0.90336) plus guard flips: the 10 V85 history-guard cells are recomputed (stage v85),
  the 94 V86 cells ship as a frozen change list (results/final_v86_v87/); build_final_v86_v87.py (stage final)
  applies them.  Every other byte equals V84.

Lineage of V84: clean data -> T1 production chain (GBM + _vp U-Nets + V30/V36/V40) -> V51 zip (+ T2 queues, ODME)
  -> V52 -> V56 -> {V57rs, V57blk} -> V58 -> {V59loc, V59blk} -> V60 -> {V61cg, V61b3} -> V62 -> V64 -> V66 -> V69
  -> V72 (T2 April ongoing from 96k/48k March fine-tunes) -> V73 guard -> V74 -> V75 -> V76 -> V77 -> V78 -> V79
  -> V81 -> V82 -> V83 -> V84 ; V86 = V84 + spatio-temporal guard flips ; V87 = V86 + history guard flips.  Each stage runs existing work/ scripts; every stage writes its own
  outputs (CSV / zip / result caches) and the next stage reads them.  Nothing reads a historical zip.
  Not rebuilt (off the lineage): V71, V73 without --guard, V76 context-only, V78 boost, V79 speed, V80, the failed
  V81-V84 candidate builders and every register_*/record_* script (they only patch manifests/submissions.csv).

Mode: inference from the shipped trained neural models (U-Nets, BiLSTM, CNN) and frozen pre-registered rules
  (--list-frozen).  The CPU LightGBM / ridge models of the lanes are refitted inside their stages exactly as in the
  original runs (deterministic=True, fixed seeds); their files (work/models/t1_blk_v57_*, t1_blk_v59_*,
  t1_blk_v61b3_s4_*, t1_cg_v61_*) are stage OUTPUTS and must not exist beforehand.  In general every script refuses to
  overwrite its outputs (FileExistsError), so run in a tree without work/sub, work/submissions and the stage result
  folders, and compare with a reference tree via --ref-root.  --retrain additionally runs the recorded GPU training
  commands of a stage (retrain_cmds) before it; move the corresponding shipped model files aside first.
  Builders V77-V84 compare their base zip with the development sha256 through repro_check.base_sha: a mismatch is a
  warning (the 1% tolerance does not need byte identity across platforms); set TFB_STRICT=1 to make it an error.

Hardware: a CUDA GPU with bf16 support (developed on an RTX 4070 Laptop GPU, CUDA 12.1, torch 2.5.1) for the
  U-Net / BiLSTM / CNN inference stages (t1_base, t2_queue, t1_v59blk, t1_v61b3, t2_v72, t2_check); every other
  stage is CPU-only (LightGBM 4.6.0, numpy 1.26.4; see work/ENVIRONMENT.txt, work/requirements.txt).  Disk:
  clean_data ~7 GB; the V73-V84 result caches alone ~6.5 GB.
Runtime (the minutes of --plan are the times measured in the clean-copy run of 2026-11-04 ... 11-08 on the reference
  machine, interruptions excluded): 18.2 h of stage time in total, GPU stages 3.6 h; the T1 calibration stages
  v51_zip ... t1_v69 (9.3 h, of which t1_v52 4.4 h) and the late T1 stages t1_v73 ... t1_v84 (5.0 h) dominate, the
  three T2 stages take 40 min and v85 + final + verify 6 min.  --retrain adds roughly 7 h of GPU training for the
  stages that list retrain_cmds.
Status: "verified" = re-run in a clean copy of the package (official data + the shipped code, models and frozen
  decision files only) with outputs identical to the development build: CSV / parquet / model files byte-identical and
  zips content-identical where the development tree still holds them (V51, V60 ... V84, V86, V87; the twelve zips of
  V52 ... V59 are no longer kept there and were checked only through their state CSVs and the later zips), except for
  the byte differences listed in a stage's notes (timing fields, absolute paths, the V84 hash recorded in the v85
  manifest, an optional column of the development run, float noise up to 1.2e-13).  All 36 stages ran in
  the clean-copy run of 2026-11-04 ... 11-08 (python work/run_all.py --run-all --resume --force --ref-root
  <development tree>, resumed after interruptions), which ended with verify PASSED; the earlier clean room of
  2026-10-25/26 covered data ... t1_v57rs.  Statuses "partial" / "todo" (need --force) are no longer used.
  A stage with refuse_if will not start while any of those files exists (its commands overwrite caches in place).

Usage (from the "IEEE Big data" folder, with the anaconda python):
  python work/run_all.py --plan                     # stages, commands, outputs, status, measured runtimes
  python work/run_all.py --list-frozen              # shipped artifacts each stage needs (exists / size / sha256)
  python work/run_all.py --check                    # outputs vs the sha256 recorded in results/*manifest*.json
  python work/run_all.py --check --ref-root D:/ref  # + line-by-line content comparison with a reference tree
  python work/run_all.py --run t1_v66 t1_v69        # run the named stages (in pipeline order)
  python work/run_all.py --run-all --force          # everything (--force: also stages marked todo; none is)
  python work/run_all.py --from t1_v73 --force --ref-root D:/ref   # resume; compare each stage's outputs as it ends
  python work/run_all.py --run-all --resume --force # skip the stages that already ended without error
  python work/run_all.py --run verify               # V87 and V86 vs the submitted files (recorded sha256 + content of the
                                                    # copies in work/reference/final/, or of --ref-root)
  python work/run_all.py --run-all --tolerant       # another GPU model: see "Tolerant mode" below
Tolerant mode (--tolerant, or the environment variable TFB_TOLERANT=1): the bf16 GPU members of Task 2 can flip a few
  queue cells on a GPU model other than the development one (RTX 4070 Laptop); by default the checks that compare
  them with decisions recorded on the development GPU then stop the run (t2_queue and t2_v72: the V48 / V51 April
  decisions; t2_check: the five queue files vs work/reference/t2_queue; v85: V84's queue rows and the recomputed
  V85 cells; final: the "before" value of every flipped cell).  With --tolerant they print "TOLERATED" with the
  number of differing cells and the run continues with the values recomputed on this machine (for the V85 / V86
  guard cells: the recorded ones), and verify reports the differing lines of V87 / V86 without failing.  Structural
  checks (row sets, keys, counts, as-of truncation) stay fatal.  The 1% tolerance of the award rule is far larger
  than a few cells.  To continue a run that stopped at one of these checks: remove the partial outputs the error or
  the "refuses" message names, then python work/run_all.py --run-all --resume --tolerant.
Every command is logged with its return code and seconds to work/results/run_all_log.csv; a stage that ends without
error writes work/results/run_all_done/<stage>.ok, which --resume uses to skip it.
"""
from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import io
import itertools
import json
import os
import shlex
import subprocess
import sys
import time
import zipfile
from pathlib import Path

WORK = Path(__file__).resolve().parent
ROOT = WORK.parent
PY = sys.executable
LOG = WORK / "results" / "run_all_log.csv"
DONE = WORK / "results" / "run_all_done"   # <stage>.ok written when a stage ends without error (used by --resume)
SPLITS = ("validation", "private")
FINAL = {"V87": ("work/submissions/v87_combined.zip",
                 "6bc4e383967c9dbf96d8b343879b8b729436f3118636619bb113bde635e3dff0"),
         "V86": ("work/submissions/v86_spacetime_guard.zip",
                 "9e1ba0523c88b6ad70aeccf95c41e64a3c1639558c9c4a555867c75fc7cf1867")}
V84 = ("work/submissions/v84_observation_flow.zip",     # end of the model chain, base of V86 / V87
       "a429599ade6ee8663b28f850f86acc0819ba426d2b2f9cc0861b2831edc36090")
WRITE_ONCE = ("work/sub/", "work/submissions/", "work/models/")
V54Q = "work/sub/queue_validation_v54_onset.csv work/sub/queue_private_v54_onset.csv"
T2_REF = "work/reference/t2_queue"      # shipped copies of the five original T2 queue files (t2_check compare)
FINAL_REF = "work/reference/final"      # package only: copies of the submitted V87 and V86 (default reference of verify)
# Working folders the stage scripts write into.  Some scripts create them only in a code path that the stage does not
# take (they existed in the development tree), so a clean tree needs them up front; cmd_run creates them, together with
# the folders of every stage's outs.  No script behaves differently when one of them exists.
WORK_DIRS = ["work/cache/t2_ongoing_v27", "work/cache/t2_onset_v27", "work/cache/t2_slotT", "work/cache/t2_v30_onset",
             *[f"work/results/{d}" for d in (
                 "t1_v52_monthbias/hide", "t1_v56_refine", "t1_v57_blackout/official", "t1_v57_blackout/features",
                 "t1_v57_blkruler/sim", "t1_v57_blkruler/sim_fast", "t1_v57_blkruler/sim_wide", "t1_v57_restack",
                 "t1_v59_blkspec/cache", "t1_v59_blkspec/wide", "t1_v59_blkspec/official", "t1_v59_local",
                 "t1_v61_blk3/cache", "t1_v64_pooled", "t1_v68_cmgbm", "t1_v69_cmgbm", "t1_v82_congest_residual",
                 "t1_v83_congest_refine", "t1_v84_flow_context", "t2_ongoing_v27", "t2_v30_onset",
                 "t2_v33_adaptive_iou", "t2_v41_ongoing_cat", "t2_v52_d12s", "t2_v71")]]


# ------------------------------------------------------------------ helpers for the stage table
def both(fmt: str) -> list[str]:
    return [fmt.format(s=s) for s in SPLITS]


def state(tag: str) -> list[str]:
    return both("work/sub/state_{s}_" + tag + ".csv")


def zips(*names: str) -> list[str]:
    return [f"work/submissions/{n}.zip" for n in names]


def models(*names: str) -> list[str]:
    return [f"work/models/{n}" for n in names]


def unets(suffix: str) -> list[str]:
    return [f"work/models/t1_unet_fold{f}{t}{suffix}.pt" for t in ("_w48", "_w48lwr") for f in (0, 1)]


def bqc(name: str, state_zip: str, note: str) -> str:
    """build_queue_candidate.py exactly as in the original status files (V54 onset queues on a state-only zip)."""
    return (f"python work/build_queue_candidate.py --name {name} --base work/submissions/{state_zip}.zip "
            f"--queue {V54Q} --note {shlex.quote(note)}")


def merge(base: str, a: str, b: str, name: str, note: str) -> str:
    return f"python work/merge_state_zips.py --base {base} --a {a} --b {b} --name {name} --note {shlex.quote(note)}"


PROD_CHAIN = [*unets("_vp"), *models("t1_gbm_v.txt", "t1_gbm_q.txt", "t1_gbm_k.txt"),
              *[f"work/results/t1_v30_stack/model_{t}.txt" for t in "vqk"]]
# Task 2 (t2_queue / t2_v72 / t2_check)
T2_CORRIDORS = ("D7_I10_E", "D7_I10_W", "D7_I210_E", "D7_I210_W", "D7_I405_N", "D7_I405_S", "D12_I5_N", "D12_I5_S")
T2_QUEUE_FILES = ("queue_validation_v54_onset.csv", "queue_private_v54_onset.csv", "queue_private_v52_selrule.csv",
                  "queue_validation_v51_april.csv", "queue_private_v51_april.csv")
# pre-registered / recorded decisions: V54 onset rules, V51 + V48 weights, and the saved V48 / V51 April decisions
# that t2_final_ongoing (variant v51) and t2_v71_april.py assert their rebuilds against (regenerating them would need
# the off-chain t2_v47_g0 / t2_v50_april chain and its V45-V48 base queue files)
T2_DECISIONS = ["work/results/t2_v54_onset/decisions.json", "work/results/t2_v51/config.json",
                "work/results/t2_v48/config.json",
                *[f"work/results/t2_v{v}/{p}_private.npz" for v in (48, 51) for p in T2_CORRIDORS]]
T2_MODELS = models("t2_ml2_onset_probs.txt", "t2_ml2_ongoing_probs.txt", "t2_ml2_ongoing_v2.txt", "t2_v8_ongoing.txt",
                   "t2_ml2_ongoing_g0.txt", "t2_cat_ongoing_g0.cbm",
                   "t2_cnn.pt", "t2_cnn_s1.pt", "t2_cnn_s2.pt", "t2_rnn.pt", "t2_rnn_s1.pt",
                   "t2_cnn_g0.pt", "t2_cnn_g0_s1.pt", "t2_cnn_g0_s2.pt", "t2_rnn_g0.pt", "t2_rnn_g0_s1.pt",
                   "t2_rnn_g0_ftall.pt", "t2_rnn_g0_s1_ftall.pt", "t2_rnn_g0_s2_ftall.pt", "t2_rnn_g0_s3_ftall.pt",
                   "t2_cnn_g0_ftall.pt", "t2_cnn_g0_s1_ftall.pt", "t2_cnn_g0_s2_ftall.pt")
T2_V72_MODELS = models("t2_rnn_g0_hw96all.pt", "t2_rnn_g0_s1_hw96all.pt", "t2_rnn_g0_s2_hw48all.pt",
                       "t2_rnn_g0_s3_hw48all.pt", "t2_cnn_g0_s1_hw96all.pt", "t2_cnn_g0_s2_hw96all.pt")
# task2_ml.py LightGBM probabilities: a pass-through key of the V27 official cache (zeroing test: 0 cells; NaN in the
# truncation check: 0 cells); task2_ml.py build/predict would refit them and write extra queue files, so they ship
T2_LGBM_PKLS = [f"work/sub/probs/t2ml_lgbm_probs_{s}_{p}.pkl" for s in SPLITS for p in T2_CORRIDORS]
OLD_PARITY = [f"work/results/t1_v81_density_context/model_old_parity_{a}_{t}.txt" for a in (0, 1) for t in ("r", "sstar")]

# id, desc, cmds (relative to ROOT, "python" = this interpreter), outs, gpu, status, minutes (measured), frozen
# (shipped artifacts the stage reads but does not write), optional retrain_cmds (GPU training, run before cmds with
# --retrain), refuse_if (paths whose existence blocks the stage before any command), compare ((rebuilt, shipped) pairs
# compared line by line after the commands; reported, not fatal) and notes (shown by --plan).
STAGES = [
    dict(id="data", gpu=False, status="verified", minutes=37,
         desc="clean parquet layer from kaggle_public (clean room 2026-10-25 and clean copy 2026-11-04: 40/40 parquet "
              "files byte-identical)",
         cmds=["python work/clean_data.py"],
         outs=["clean_data"], frozen=[],
         notes=["clean copy 2026-11-04: the 40 parquet files and data_dictionary.csv byte-identical (sha256); "
                "cleaning_report.{json,csv} differ only in the per-table 'seconds' timing fields"]),
    # T4 right after data: it depends only on kaggle_public, and v51_zip copies its ODME rows into the first zip, from
    # which every later zip (V52 ... V84, V86, V87) carries them byte for byte.  Running it early also fails fast.
    dict(id="t4", gpu=False, status="verified", minutes=69,
         desc="Task 4 ODME = official NNLS reference (lambda 0.05) of trafficflowbench-public-main, both scored "
              "months, rebuilt into work/baseline_rebuild and compared with the shipped work/baseline CSVs that "
              "v51_zip reads",
         cmds=[f"python trafficflowbench-public-main/src/task4/build_task4_odme_artifacts.py --release-root "
               f"kaggle_public --split {s} --output-root work/baseline_rebuild/task4_{s}" for s in SPLITS],
         outs=both("work/baseline_rebuild/task4_{s}/public_reference.csv"),
         compare=[(f"work/baseline_rebuild/task4_{s}/public_reference.csv",
                   f"work/baseline/task4_{s}/public_reference.csv") for s in SPLITS],
         refuse_if=both("work/baseline_rebuild/task4_{s}"),
         frozen=[*both("work/baseline/task4_{s}/public_reference.csv"),
                 "trafficflowbench-public-main/src/task4/build_task4_odme_artifacts.py",
                 "trafficflowbench-public-main/src/task1/baseline_task1_historical_mean.py",
                 "trafficflowbench-public-main/config/corridors.json"],
         notes=["same call as work/build_baseline_submission.sh (2026-10-05 run, ~18 + 21 min by file mtimes), but "
                "with a NEW output root: the official script overwrites its outputs in place, so it never writes into "
                "work/baseline; after the commands run_all compares the rebuilt CSVs line by line with the shipped "
                "ones (a difference is reported; the chain continues with the shipped CSVs)",
                "do not run it in the development tree (refuses if work/baseline_rebuild/task4_* exists)",
                "clean copy 2026-11-04: both rebuilt public_reference.csv content-identical to the shipped ones "
                "(35,355 lines each)"]),
    dict(id="t1_base", gpu=True, status="verified", minutes=81,
         desc="production T1 chain: GBM + _w48_vp/_w48lwr_vp U-Nets (mean of 2 folds) blended by t1_assemble, then "
              "the V30/V31 residual stack, V36 transfer and V40 half alpha -> v40half state",
         cmds=["python work/t1_unet.py build",
               "python work/task1_gbm.py predict",
               "python work/t1_unet.py predict --fold -2 --width 48 --tag _w48_vp",
               "python work/t1_unet.py predict --fold -2 --width 48 --tag _w48lwr_vp",
               *[f"python work/t1_assemble.py --split {s} --gbm work/sub/state_{s}.csv --out work/sub/state_{s}_vp.csv "
                 "--tags _w48_vp _w48lwr_vp --wv 0.3 0.6 --wq 0.1 0.7 --gap-wv 0.5 0.5 --gap-wq 0.5 0.5"
                 for s in SPLITS],
               "python work/t1_v30_official.py", "python work/t1_v36_official.py",
               "python work/t1_v40_half_official.py"],
         outs=[*both("work/sub/state_{s}.csv"), *state("vp"), *state("v30stack"), *state("v36transfer"),
               *state("v40half")],
         frozen=[*PROD_CHAIN,
                 "work/results/t1_v30_bridge/selected.json", "work/results/t1_v30_bridge/protocol.json",
                 "work/results/t1_v30_stack/protocol.json",
                 "work/results/t1_v36_transfer/validation_selection.json",
                 "work/results/t1_v36_transfer/confirmation.json", "work/results/t1_v36_transfer/protocol.json"],
         notes=["clean room 2026-10-25: 10/10 state CSVs and 210/210 cache files byte-identical (inference); the GBM "
                "and U-Net training commands are not part of this table"]),
    dict(id="t2_queue", gpu=True, status="verified", minutes=14,
         desc="Task 2 queue files rebuilt in one pass from the member models (replaces the v14->v18->v24->v30 onset / "
              "V48 / V51 / V52 / V54 patch chain): March V54 onset + V48 ongoing + V52 hyb, April V54 onset + V51 + "
              "hyb, April pre-V54 onset (v52_selrule), and the pre-V52 v51_april pair for v51_zip",
         cmds=["python work/t2_grids.py",
               "python work/t2_group_v27.py learn",
               "python work/t2_final_queue.py ml2-probs",
               "python work/t2_final_queue.py slotT",
               "python work/t2_v52_prodcache.py official",
               "python work/t2_v52_selrule.py classify",
               "python work/t2_v52_selrule.py check",
               "python work/t2_v52_replica.py build",
               "python work/t2_v54_onset.py cases",
               "python work/t2_final_queue.py write"],
         outs=[*[f"work/sub/{f}" for f in T2_QUEUE_FILES],
               "work/cache/t2_grids", "work/results/t2_group_v27", "work/sub/probs", "work/cache/t2_ongoing_v27",
               "work/results/t2_v46_slotT", "work/cache/t2_v52_selrule",
               "work/results/t2_v52_selrule/official_classes.csv", "work/cache/t2_v54_onset"],
         refuse_if=["work/sub/queue_validation_v54_onset.csv"],
         frozen=[*T2_DECISIONS, *T2_MODELS, *T2_LGBM_PKLS],
         notes=["verified 2026-11-02 in the development tree into test folders: write = the five files byte-identical "
                "to the originals (sha 7fd27659 / 69deca56 / ebbd352a / e5284810 / 690d1f9a; write 281 s), ml2-probs "
                "16/16 pkls byte-identical (16 s), slotT sh_* max diff 0.0 (44 s), prodcache official in memory max "
                "diff 0.0 (~40 s), selrule check 5.5 s, replica build byte-identical (37 s with the layers cache); "
                "t2_grids, group learn, the layers / V27 official caches and cases were not timed (cases ~1 min)",
                "clean copy 2026-11-04: the five queue files and official_classes.csv content-identical to the "
                "development ones (stage t2_check later found the five byte-identical to work/reference/t2_queue)",
                "REFUSES to start while work/sub/queue_validation_v54_onset.csv exists: t2_group_v27.py learn, "
                "t2_v52_selrule.py classify and t2_v52_replica.py build overwrite their outputs in place and write "
                "refuses to overwrite the queue files -- run it only in a clean copy",
                "task2_ml.py build / predict --tag _probs are NOT run: the 16 sub/probs/t2ml_lgbm_probs_*.pkl ship as "
                "frozen inputs (pass-through key of the V27 official cache; their values reach no cell: agent A's "
                "zeroing test, and t2_final_asof.py runs with that member set to NaN); a clean copy therefore ships "
                "work/sub/probs/ holding only these 16 files (ml2-probs adds the t2ml2 pkls next to them), so the "
                "'outputs already present' warning for work/sub/probs is expected",
                "auto-built on first use: cache/t2_ongoing_v27/official_*.npz (by slotT), "
                "cache/t2_v52_selrule/layers_*.npz (by replica build), cache/t2_onset_events_*.parquet, the V30 rf5 "
                "train rows cache/t2_v30_onset/D7_I405_N_train.npz",
                "cache/t2_v52_selrule/windows.csv (t2_v52_replica.py build) is also an input of t1_v59_blkspec.py "
                "runs (a retrain_cmd of stage t1_v59blk: simulated gaps at the train replica origins)",
                "slotT rebuilds only the ids / T / sh_* keys of results/t2_v46_slotT/*.npz (the original 'old' / "
                "'new' keys were read only by the off-chain t2_v47_g0.py --check-dir)",
                "needs xgboost (task2_ml.py imports it at module level; the whole T2 chain imports task2_ml)"]),
    dict(id="v51_zip", gpu=False, status="verified", minutes=3,
         desc="state v40half + queue v51 + ODME -> v51_april.zip (base zip of every state-only zip below)",
         cmds=["python work/make_submission.py --name v51_april --state work/sub/state_validation_v40half.csv "
               "work/sub/state_private_v40half.csv --queue work/sub/queue_validation_v51_april.csv "
               "work/sub/queue_private_v51_april.csv --odme work/baseline/task4_validation/public_reference.csv "
               "work/baseline/task4_private/public_reference.csv"],
         outs=zips("v51_april"), frozen=both("work/baseline/task4_{s}/public_reference.csv"),
         notes=["clean copy 2026-11-04: v51_april.zip content-identical, built from the queue files that t2_queue "
                "wrote (the clean room of 2026-10-25 had used hand-copied queue_*_v51_april.csv)",
                "the ODME rows come from the shipped work/baseline CSVs (stage t4 rebuilds and compares them)"]),
    dict(id="t1_v52", gpu=False, status="verified", minutes=262,
         desc="per-month per-link bias on the hide-20% frame of the production chain (V52_T1BIAS_README.md)",
         cmds=["python work/t1_v52_monthbias.py collect --split validation --threads 4",
               "python work/t1_v52_monthbias.py collect --split private --threads 6",
               "python work/t1_v52_monthbias.py crossfit", "python work/t1_v52_monthbias.py apply",
               "python work/t1_v52_monthbias.py package"],
         outs=[*state("v52bias"), *zips("v52_t1bias")], frozen=PROD_CHAIN,
         notes=["clean room: validation collect 8,299 s (BelowNormal priority next to the GPU chain); CSVs identical",
                "clean copy 2026-11-04 ... 11-06, in three attempts (two interrupted): collect keeps one finished "
                "result per corridor in results/t1_v52_monthbias/hide/ and skips it when rerun, so the resumed runs "
                "continued with the next corridor; validation collect 7,706 s (first attempt), private collect "
                "7,407 s (704 + 484 s in the first, 1,226 + 375 s in the second and 4,618 s in the third attempt, "
                "from the cache file times and the command start times), crossfit / apply / package 594 s; minutes = "
                "the sum; state CSVs content-identical; v52_t1bias.zip is no longer kept in the development tree and "
                "was not compared (V60 and every later zip content-identical)"]),
    dict(id="t1_v56", gpu=False, status="verified", minutes=13,
         desc="per-link x band refinement (V56_T1_README.md), then V54 onset queues on the state zip",
         cmds=["python work/t1_v56_refine.py crossfit", "python work/t1_v56_refine.py apply",
               "python work/t1_v56_refine.py package",
               bqc("v56_t1_on54", "v56t1_state",
                   "V56 = V54_onset queue (LB 0.89981) + V56_T1 state (t1_v56_refine.py: per-link x 7-band speed "
                   "offsets, per-link flow scale + 4h time-of-day offsets; free-flow cells only)")],
         outs=[*state("v56t1"), *zips("v56t1_state", "v56_t1_on54")], frozen=[]),
    dict(id="t1_v57rs", gpu=False, status="verified", minutes=23,
         desc="month-specific restack of the T1 components on free non-blackout cells (V57_RESTACK_README.md)",
         cmds=["python work/t1_v57_restack.py crossfit", "python work/t1_v57_restack.py apply",
               "python work/t1_v57_restack.py package", "python work/t1_v57_restack.py audit",
               bqc("v57_rs", "v57rs_state",
                   "V57_rs = V54 queues + V57 state (V56 state + month-specific GBM restack of the T1 components on "
                   "free non-blackout cells, V56 correction refit); t1_v57_restack.py"),
               "python work/t1_v57_restack.py compare"],
         outs=[*state("v57rs"), *zips("v57rs_state", "v57_rs")], frozen=[],
         notes=["clean room 2026-10-25 verified crossfit/apply/package (state CSVs identical); the clean copy of "
                "2026-11-06 ran all six commands; state CSVs content-identical, the two zips not compared (no "
                "development copy)"]),
    dict(id="t1_v57blk", gpu=False, status="verified", minutes=65,
         desc="blackout lane: simulated official-like gaps (ruler) + gap residual GBM (V57_BLACKOUT_README.md)",
         cmds=[*both("python work/t1_v57_blkruler.py origins --split {s}"),
               *both("python work/t1_v57_blkruler.py wide --split {s} --threads 4"),
               "python work/t1_v57_blackout.py frame", "python work/t1_v57_blackout.py crossfit",
               "python work/t1_v57_blackout.py official --threads 4", "python work/t1_v57_blackout.py fit",
               "python work/t1_v57_blackout.py apply", "python work/t1_v57_blackout.py package",
               "python work/t1_v57_blackout.py audit",
               bqc("v57_blk", "v57blk_state",
                   "V57_blk = V56 (v56_t1_on54) with only the Task-1 targets inside official Task-2 blackouts "
                   "refilled by the V57 gap residual GBM (lambda_v 0.75, lambda_q 0.5); queue rows = V54 onset (= V56)"),
               "python work/t1_v57_blackout.py compare"],
         outs=[*state("v57blk"), *models("t1_blk_v57_v.txt", "t1_blk_v57_q.txt"), *zips("v57blk_state", "v57_blk")],
         frozen=PROD_CHAIN,
         notes=["'wide', not 'sim --fast': t1_v57_blackout.py reads results/t1_v57_blkruler/sim_wide",
                "fit writes work/models/t1_blk_v57_{v,q}.txt (deterministic CPU LightGBM) and refuses to overwrite"]),
    dict(id="merge_v58", gpu=False, status="verified", minutes=2,
         desc="disjoint-row merge V58 = V56 + V57rs free rows + V57blk blackout rows",
         cmds=[merge("v56_t1_on54", "v57_rs", "v57_blk", "v58_rs_blk",
                     "V58 = V56 queue/ODME + V57_rs non-blackout free-flow state (month GBM restack) + V57_blk "
                     "blackout-target state (gap residual GBM, lambda_v .75 / lambda_q .5)")],
         outs=zips("v58_rs_blk"), frozen=[]),
    dict(id="t1_v59loc", gpu=False, status="verified", minutes=37,
         desc="local-neighbourhood restack of the free non-blackout rows on top of V58",
         cmds=["python work/t1_v59_local.py features", "python work/t1_v59_local.py crossfit",
               "python work/t1_v59_local.py apply", "python work/t1_v59_local.py package",
               "python work/t1_v59_local.py audit",
               bqc("v59_loc", "v59loc_state",
                   "V59 = V58 (v58_rs_blk) with the free non-blackout T1 state rows re-predicted by the V57 restack + "
                   "local neighbourhood features (t1_v59_local.py, Lvq_x_q3000) + refit V56; blackout rows = "
                   "V57_blk; V54 queues; ODME = V51"),
               "python work/t1_v59_local.py compare"],
         outs=[*state("v59loc"), *zips("v59loc_state", "v59_loc"), "work/cache/t1_v59_local"], frozen=[]),
    dict(id="t1_v59blk", gpu=True, status="verified", minutes=47,
         desc="gap-specialist U-Nets + residual GBM S2 on the blackout rows; phase-B self-supervised nets (S4 inputs "
              "of t1_v61b3)",
         cmds=[*both("python work/t1_v59_blkspec.py wide --split {s}"), "python work/t1_v59_blkspec.py crossfit",
               "python work/t1_v59_blkspec.py official", "python work/t1_v59_blkspec.py fit",
               "python work/t1_v59_blkspec.py apply", "python work/t1_v59_blkspec.py package",
               "python work/t1_v59_blkspec.py audit",
               bqc("v59_blk", "v59blk_state",
                   "V59_blk = V58 (v58_rs_blk) with only the Task-1 targets inside official Task-2 blackouts refilled "
                   "by the V59 gap-specialist U-Net (train-month official-like gap fine-tune, condition channels) + "
                   "residual GBM S2 (lambda_v 0.625, lambda_q 0.5); queue rows = V54 onset (= V58)"),
               "python work/t1_v59_blkspec.py compare",
               *both("python work/t1_v59_blkspec_ss.py wide_ss --split {s}"),
               "python work/t1_v59_blkspec_ss.py crossfit_ss", "python work/t1_v59_blkspec_ss.py official_ss"],
         retrain_cmds=["python work/t1_v59_blkspec.py runs", "python work/t1_v59_blkspec.py occ",
                       *[f"python work/t1_v59_blkspec.py train --tag {t} --fold {f} --epochs 8 --lr 2e-4"
                         for f in (0, 1) for t in ("_w48", "_w48lwr")],
                       *[f"python work/t1_v59_blkspec.py train_ss --tag {t} --fold {f} --parity {b} --epochs 8 --lr 1e-4"
                         for f in (0, 1) for b in (0, 1) for t in ("_w48", "_w48lwr")]],
         outs=[*state("v59blk"), *models("t1_blk_v59_v.txt", "t1_blk_v59_q.txt"), *zips("v59blk_state", "v59_blk"),
               "work/results/t1_v59_blkspec/crossfit_ss_choice.json", "work/results/t1_v59_blkspec/official_ss"],
         frozen=[*unets("_v59bs"), *unets("_v59ss_p0"), *unets("_v59ss_p1")],
         notes=["fit writes work/models/t1_blk_v59_{v,q}.txt and refuses to overwrite",
                "wide_ss (10-21 01:20, no status file) must precede crossfit_ss; the phase-B outputs are read by "
                "t1_v61b3 crossfit/apply",
                "retrain_cmds: 4 gap-specialist + 8 self-supervised U-Nets (~1.5 h GPU, t1_v59_bs_tr*/t1_v59_ss_f* "
                "status files); the parity -1 nets are diagnostics only and not retrained",
                "clean copy 2026-11-06: state CSVs and models identical, the two zips not compared (no development "
                "copy); crossfit_ss_choice.json differs only in its 'seconds' timing field"]),
    dict(id="merge_v60", gpu=False, status="verified", minutes=2,
         desc="disjoint-row merge V60 = V58 + V59loc free rows + V59blk blackout rows",
         cmds=[merge("v58_rs_blk", "v59_loc", "v59_blk", "v60_loc_blk",
                     "V60 = V58 queue/ODME + V59_loc non-blackout free-flow state (local-feature restack) + V59_blk "
                     "blackout-target state (gap-specialist U-Net + residual GBM S2, lambda_v .625 / lambda_q .5); "
                     "pending review of V59_blk")],
         outs=zips("v60_loc_blk"), frozen=[],
         notes=["clean copy 2026-11-06: content-identical; its logged 22,775 s include a 6.3 h Modern Standby pause "
                "(display timeout), so minutes = the ~110 s of the other two merges"]),
    dict(id="t1_v61cg", gpu=False, status="verified", minutes=18,
         desc="lane C: congested non-blackout rows from the pooled-month L1 density specialist F600a75s",
         cmds=["python work/t1_v61_congest.py crossfit", "python work/t1_v61_congest.py apply",
               "python work/t1_v61_congest.py package", "python work/t1_v61_congest.py audit",
               bqc("v61_cg", "v61cg_state",
                   "V61_cg = V60 (v60_loc_blk) + lane C: congested non-blackout T1 state rows replaced by the "
                   "pooled-month L1 density specialist F600a75s (t1_v61_congest.py); queue = V54 onset (unchanged vs "
                   "V60); ODME unchanged"),
               "python work/t1_v61_congest.py compare"],
         outs=[*state("v61cg"), *[f"work/models/t1_cg_v61_{s}_{c}.txt" for s in SPLITS for c in "ds"],
               *zips("v61cg_state", "v61_cg"), "work/results/t1_v61_congest/cache"],
         frozen=[],
         notes=["apply fits and writes work/models/t1_cg_v61_{split}_{d,s}.txt and refuses if they exist"]),
    dict(id="t1_v61b3", gpu=True, status="verified", minutes=48,
         desc="lane B3: blackout rows refilled by E1 = 0.5 S2 + 0.5 S4 (V59 nets + new s1/ext gap nets)",
         cmds=[*[f"python work/t1_v61_blk3.py wide --set {k} --split {s}" for k in ("s1", "ext") for s in SPLITS],
               "python work/t1_v61_blk3.py official --set s1", "python work/t1_v61_blk3.py official --set ext",
               "python work/t1_v61_blk3.py crossfit", "python work/t1_v61_blk3.py fit",
               "python work/t1_v61_blk3.py apply", "python work/t1_v61_blk3.py package",
               "python work/t1_v61_blk3.py audit",
               bqc("v61_b3", "v61b3_state",
                   "V61 lane B3: V60 + E1 blackout refill (0.5 S2 + 0.5 S4) on the 128,465 official Task-2 blackout "
                   "T1 target rows; V54 onset queues")],
         retrain_cmds=["python work/t1_v61_blk3.py runs_ext",
                       *[f"python work/t1_v61_blk3.py train --set {k} --tag {t} --fold {f}"
                         for k in ("s1", "ext") for f in (0, 1) for t in ("_w48", "_w48lwr")]],
         outs=[*state("v61b3"), *models("t1_blk_v61b3_s4_v.txt", "t1_blk_v61b3_s4_q.txt"),
               *zips("v61b3_state", "v61_b3")],
         frozen=["work/results/t1_v61_blk3/prereg.json", *unets("_v61b3_s1"), *unets("_v61b3_ext")],
         notes=["wide/official --set s1|ext = GPU inference with the shipped nets (steps 5-7 of the original "
                "t1_v61_b3_s1/ext training chains, 213 s); crossfit/apply read their outputs",
                "crossfit asserts that the script docstring still matches the pre-registered prereg.json",
                "fit writes work/models/t1_blk_v61b3_s4_{v,q}.txt and refuses to overwrite",
                "retrain_cmds: 8 gap U-Nets (s1 3,937 s + ext 8,162 s GPU)"]),
    dict(id="merge_v62", gpu=False, status="verified", minutes=2,
         desc="disjoint-row merge V62 = V60 + V61cg congested rows + V61b3 blackout rows",
         cmds=[merge("v60_loc_blk", "v61_cg", "v61_b3", "v62_cg_b3",
                     "V62 = V60 + V61_cg (congested non-blackout L1 density, public 0.90159) + V61_b3 (blackout E1 = "
                     "0.5 S2 + 0.5 S4); disjoint rows")],
         outs=zips("v62_cg_b3"), frozen=[]),
    dict(id="t1_v64", gpu=False, status="verified", minutes=10,
         desc="congested non-blackout rows from the pooled two-month L1 density model (pooled31_l09)",
         cmds=["python work/t1_v64_apply.py check", "python work/t1_v64_apply.py apply"],
         outs=[*zips("v64_pool"), "work/results/t1_v64/density_l1.txt", "work/results/t1_v64/split_l2.txt"],
         frozen=[]),
    dict(id="t1_v65cache", gpu=False, status="verified", minutes=2,
         desc="cross-fitted V64 density/split on the hide cells (cache v64cf.npz read by t1_v66, t1_v69 and t1_v81)",
         cmds=["python work/t1_v65_lwrcong.py structure", "python work/t1_v65_lwrcong.py baseline"],
         outs=["work/results/t1_v65_lwrcong/cache/v64cf.npz"], frozen=[],
         notes=["'baseline' wrote v64cf.npz originally (10-24 02:12, with baseline_classes.csv); it needs "
                "structure_real.csv from 'structure'. The later 'crossfit' (t1_v65_cf.status) only reused the cache "
                "and also needs the prereg files and the whole rejected candidate grid"]),
    dict(id="t1_v66", gpu=False, status="verified", minutes=16,
         desc="corridor-wide per-time speed common mode on the free non-blackout rows (V66_CMODE_README.md)",
         cmds=["python work/t1_v66_cmode.py check", "python work/t1_v66_cmode.py apply",
               "python work/t1_v66_cmode.py audit"],
         outs=[*zips("v66_cm"), "work/results/t1_v66_cmode/grams.npz", "work/results/t1_v66_cmode/cap_links.json",
               "work/results/t1_v66_cmode/cache"],
         frozen=["work/results/t1_v66_cmode/rule.json"],
         notes=["check refuses to run without the pre-registered rule.json (write-once, 2026-10-24); never run "
                "'register' (it would write a new timestamp)"]),
    dict(id="t1_v69", gpu=False, status="verified", minutes=6,
         desc="LightGBM second stage on the V66 residual (speed, free rows)",
         cmds=["python work/t1_v69_cmgbm_apply.py"],
         outs=zips("v69_cmgbm"), frozen=[]),
    dict(id="t2_v72", gpu=True, status="verified", minutes=3,
         desc="April ongoing: BiLSTM s0/s1 and CNN s1/s2 96k-window March fine-tunes + BiLSTM s2/s3 48k (weights "
              "4/2/4) + V52 hyb conditioning; patches 55 private queue cells of the V69 zip",
         cmds=["python work/t2_v71_april.py --variant v72"],
         retrain_cmds=[f"python work/t2_rnn_march.py finetune {a} --fold all --tag {t} --windows 96000" for a, t in (
             ("--base _g0", "_g0_hw96all"), ("--base _g0_s1", "_g0_s1_hw96all"),
             ("--kind cnn --base _g0_s1", "_g0_s1_hw96all"), ("--kind cnn --base _g0_s2", "_g0_s2_hw96all"))],
         outs=["work/sub/queue_private_v72_april.csv", *zips("v72_april")],
         frozen=[*T2_DECISIONS, *T2_MODELS, *T2_V72_MODELS],
         notes=["reads the t2_queue outputs (results/t2_v46_slotT, cache/t2_ongoing_v27/official_*.npz, "
                "results/t2_v52_selrule/official_classes.csv, sub/queue_private_v52_selrule.csv and "
                "v54_onset.csv); results/t2_v48 and t2_v51/<corridor>_private.npz are NOT caches but frozen recorded "
                "decisions: the V48 / V51 recipes are asserted to reproduce them before the V72 rows are written "
                "(with --tolerant a difference is reported as TOLERATED and the recomputed decision is used; the "
                "same holds for the April V51 assertion of t2_final_queue.py write in stage t2_queue)",
                "retrain_cmds = the four 96k fine-tunes (7,388 s GPU, results/ft72all.status)",
                "the selection run t2_v72_grid96.py cannot be repeated (cache/t2_slotT/ft_D7_I405_N_validation.npz "
                "is incomplete); its decision is recorded in results/t2_v72_grid96/decision.json"]),
    dict(id="t1_v73", gpu=False, status="verified", minutes=11,
         desc="V73 guard: visible-only additive factor speed correction, clipped to +-1 km/h, on free non-blackout "
              "state rows of V72",
         cmds=["python work/t1_v73_factor_verify.py", "python work/t1_v73_factor_verify.py --guard",
               "python work/t1_v73_factor_apply.py --guard"],
         outs=[*zips("v73_factor_guard"), "work/results/t1_v73_global_factor/summary.csv",
               "work/results/t1_v73_global_factor/decision_guard.json",
               "work/results/t1_v73_global_factor/coefficients_guard.npz"],
         frozen=[],
         notes=["the non-guard verify is a prerequisite: --guard reads its summary.csv (cand 'global') as an extra "
                "gate. Never run t1_v73_factor_apply.py without --guard (off-lineage v73_factor.zip)",
                "clean copy 2026-11-06: zip content-identical; summary.csv, decision_guard.json and "
                "coefficients_guard.npz differ byte-wise by float noise only (max abs 4e-14)"]),
    dict(id="t1_v74", gpu=False, status="verified", minutes=8,
         desc="low-rank speed factor correction (ridge) on V73",
         cmds=["python work/t1_v74_lowrank.py", "python work/build_v74.py"],
         outs=[*zips("v74_lowrank"), "work/results/t1_v74_lowrank/coefficients.npz"], frozen=[],
         notes=["results/t2_v74/queue_patch.json must NOT exist (the builder would patch queue rows)",
                "build_v74/build_v75/build_v76(_joint) reuse an existing patch.npz without a meta check: never copy "
                "patch.npz files into a fresh tree",
                "clean copy 2026-11-06: zip content-identical; coefficients.npz differs by float noise only (max abs "
                "1.2e-13)"]),
    dict(id="t1_v75", gpu=False, status="verified", minutes=12,
         desc="LightGBM factor speed correction (limit 1.0) on V74",
         cmds=["python work/t1_v75_factor_gbm.py", "python work/build_v75.py"],
         outs=[*zips("v75_factor_gbm"), "work/results/t1_v75_factor_gbm/model.txt"], frozen=[]),
    dict(id="t1_v76", gpu=False, status="verified", minutes=67,
         desc="context speed model (context_clip2) + flow model, joint bound check, joint build on V75",
         cmds=["python work/t1_v76_context_mix.py", "python work/t1_v76_flow_mix.py",
               "python work/t1_v76_joint_check.py", "python work/build_v76_joint.py"],
         outs=[*zips("v76_context_flow"), "work/results/t1_v76_context_mix/model.txt",
               "work/results/t1_v76_flow_mix/model.txt"], frozen=[],
         notes=["context_mix also trains the rejected 'deep' candidate: its baseline_{mode}.npy is needed",
                "do not run build_v76.py (off-lineage v76_context_mix.zip); build_v76_joint imports its prepare()",
                "flow_mix wall clock was 2h40m including a 2h16m idle gap; ~25 min of work"]),
    dict(id="t1_v77", gpu=False, status="verified", minutes=35,
         desc="flow factor model (650 rounds, limit 60) on V76",
         cmds=["python work/t1_v77_flow_factors.py", "python work/audit_v77_feature_alignment.py",
               "python work/build_v77.py"],
         outs=[*zips("v77_flow_factors"), "work/results/t1_v77_flow_factors/model.txt",
               "work/results/t1_v77_flow_factors/X.npy", "work/results/t1_v77_flow_factors/feature_alignment.json"],
         frozen=[],
         notes=["build_v77 ... build_v84 check their base zip sha256 via repro_check.base_sha (warning by default, "
                "error with TFB_STRICT=1)",
                "audit_v77_feature_alignment.py writes feature_alignment.json (direct features == cached X.npy rows "
                "on two sample blocks), which build_v77 puts in its manifest and t1_v78/t1_v79 copy; in the "
                "development tree that file came from an unrecorded ad-hoc run of the same check (a clean-copy run "
                "stopped at build_v77 without it, 2026-11-06)",
                "clean copy 2026-11-08 with the audit script: feature_alignment.json, model.txt and X.npy "
                "byte-identical, zip content-identical (the first version of the audit stopped the run once at "
                "01:45 because it counted NaN in the same cells as unequal; it now compares with equal_nan)"]),
    dict(id="t1_v78", gpu=False, status="verified", minutes=35,
         desc="flow boost (650+350 trees) and alpha-2 compiled step model on V77",
         cmds=["python work/t1_v78_flow_boost.py", "python work/t1_v78_flow_step.py", "python work/build_v78_step.py"],
         outs=[*zips("v78_flow_step"), "work/results/t1_v78_flow_step/model.txt"], frozen=[],
         notes=["not build_v78.py (alpha-1 alternative, off-lineage v78_flow_boost.zip)"]),
    dict(id="t1_v79", gpu=False, status="verified", minutes=31,
         desc="flow detail continuation (1300 trees, alpha 1) on V78",
         cmds=["python work/t1_v79_flow_detail.py", "python work/build_v79_flow.py"],
         outs=[*zips("v79_flow_detail"), "work/results/t1_v79_flow_detail/model.txt"], frozen=[]),
    dict(id="t1_v81", gpu=False, status="verified", minutes=16,
         desc="congested non-blackout speed: closed-form ridge on the V64 teacher + common-mode features, on V79",
         cmds=["python work/t1_v81_congest_speed.py parity", "python work/t1_v81_congest_speed.py half_month",
               "python work/t1_v81_congest_speed.py other_month", "python work/audit_v81_common_features.py",
               "python work/t1_v81_congest_speed.py full", "python work/build_v81_cmode.py --variant speed"],
         outs=[*zips("v81_congest_speed"), "work/results/t1_v81_congest_speed/beta_full.npy",
               "work/results/t1_v81_congest_cmode/hidden_value_poison_audit.json"],
         # FROZEN GATE INPUTS without a producer in the current code: heads() returns the v64cf cache for
         # (old, parity) and never trains these, but fit_beta (parity gate) and t1_v82_congest_residual.teacher load
         # them.  They only feed the V81/V82 parity gate decisions, not beta_full or the zip.
         frozen=OLD_PARITY,
         notes=["FROZEN GATE INPUTS: results/t1_v81_density_context/model_old_parity_{0,1}_{r,sstar}.txt (trained "
                "10-28 20:33 by an earlier revision of t1_v81_density_context.heads; no current producer)",
                "run the gate steps one after another (concurrent runs write the same old-head files)"]),
    dict(id="t1_v82", gpu=False, status="verified", minutes=8,
         desc="common-mode x congestion interaction terms (closed form, alpha 1.0) on V81",
         cmds=["python work/t1_v82_cmode_interactions.py parity", "python work/t1_v82_cmode_interactions.py half_month",
               "python work/t1_v82_cmode_interactions.py other_month", "python work/t1_v82_cmode_interactions.py full",
               "python work/build_v82_interactions.py"],
         outs=[*zips("v82_congest_interactions"), "work/results/t1_v82_cmode_interactions/beta_full.npy"],
         frozen=OLD_PARITY,
         notes=["the parity gate loads the same frozen model_old_parity_* files as t1_v81"]),
    dict(id="t1_v83", gpu=False, status="verified", minutes=36,
         desc="LightGBM continuation of the V79 flow model (1600 trees, alpha 1.0) on V82",
         cmds=["python work/t1_v83_flow_continue.py parity", "python work/t1_v83_flow_continue.py half_month",
               "python work/t1_v83_flow_continue.py other_month", "python work/t1_v83_flow_continue.py full",
               "python work/build_v83_flow.py"],
         outs=[*zips("v83_flow_continue"), "work/results/t1_v83_flow_continue/model.txt"], frozen=[]),
    dict(id="t1_v84", gpu=False, status="verified", minutes=44,
         desc="V84 (base of the final files): observation-only 180-tree flow continuation on V83 (context features, "
              "poison-audited)",
         cmds=["python work/t1_v84_flow_context.py features",
               "python work/t1_v84_context_continuation.py parity",
               "python work/t1_v84_context_continuation.py half_month",
               "python work/t1_v84_context_continuation.py other_month",
               "python work/t1_v84_context_continuation.py full", "python work/build_v84_observation.py"],
         outs=[*zips("v84_observation_flow"), "work/results/t1_v84_context_continuation/model_full.txt"], frozen=[],
         notes=["build_v84_observation.py, not build_v84_context.py; never run t1_v84_flow_context.py parity/full "
                "(failed candidate). 'features' writes the extra/poison caches (20 poison files) the builder asserts"]),
    dict(id="t2_check", gpu=True, status="verified", minutes=23,
         desc="read-only T2 checks of the model-built queue rows of V84 (160 windows): onset and ongoing rebuilt cell by cell, "
              "the per-window truncation (as-of T) check, and the t2_queue files byte-compared with shipped copies",
         cmds=["python work/t2_final_onset.py check",
               "python work/t2_final_ongoing.py check",
               "python work/t2_final_asof.py",
               f"python work/t2_final_queue.py compare --out-dir work/sub --ref-dir {T2_REF}"],
         outs=["work/results/t2_asof/asof.log", "work/results/t2_asof/asof_windows.csv",
               "work/results/t2_asof/asof_files_read.txt"],
         frozen=[*[f"{T2_REF}/{f}" for f in T2_QUEUE_FILES], *T2_DECISIONS, *T2_MODELS, *T2_V72_MODELS],
         notes=["runs after t1_v84: t2_final_asof.py first checks the two final queue files against the queue rows "
                "of V84 (work/submissions/v84_observation_flow.zip), so V84 must exist (in a clean tree the stage failed "
                "when it ran right after t2_v72, 2026-11-06); the checks read sub/queue_validation_v54_onset.csv and "
                "queue_private_v72_april.csv",
                "2026-11-02 development tree: onset check 124 s, ongoing check 56-100 s, TOTAL mismatching cells 0; "
                "t2_final_asof.py --natural: PASS, 174,000 cells of 160 windows, 0 differ after deleting every row "
                "after T; controls (row T / rows >= T-5 deleted) moved 319 cells in 59 windows; natural-batch pass "
                "0 cells; 1,357 s (~16 min without --natural); work/results/t2_asof/asof.log",
                f"{T2_REF}/ = reference copies of the five original queue files (from the development work/sub) that "
                "the package must ship; compare exits 1 on any cell or byte difference (with --tolerant only on a "
                "missing file or different row keys; value differences are listed and TOLERATED)",
                "t2_final_asof.py refuses to overwrite work/results/t2_asof/asof*; it writes nothing else",
                "clean copy 2026-11-08 (1,362 s): TOTAL mismatching cells 0, as-of pass 0 of 174,000 cells differ, the "
                "five queue files byte-identical; the three asof files differ from the development ones only in "
                "the per-window 'seconds' column, the natural_diff column of the development run (run there with "
                "--natural) and the absolute data paths in asof_files_read.txt (same 48 files)"]),
    dict(id="v85", gpu=False, status="verified", minutes=1,
         desc="V85 history guard recomputed on the rebuilt V84: CPU replay from frozen probabilities + retrieval of "
              "30 training-period analogues of each window's history up to T; 10 not-queued -> queued cells",
         cmds=["python work/build_v85_history_guard.py"],
         outs=["work/results/v85_history_guard/v84_queue.csv",
               "work/results/v85_history_guard/gate/official_gate_windows_audit.csv",
               "work/results/v85_history_guard/gate/official_gate_cells_audit.csv",
               "work/results/v85_history_guard/gate/official_gate_summary.json",
               "work/results/v85_history_guard/V85_history_guard_changes.csv",
               "work/results/v85_history_guard/manifest.json"],
         frozen=["work/v85_history_guard"],
         notes=["work/v85_history_guard/ = the team member's V85 reproduction hand-off (code, 24 frozen CPU "
                "probability caches, 8 train-period co-queue tables, original audit, proxy results); its "
                "v85_project/work/evaluate.py differs from ours only in reading the data paths from the environment",
                "asserts: V84 queue rows == the team member's; recomputed audit == original audit; selected cells == "
                "official V85 list == V85 rows of the V87 list; with --tolerant a difference (or a stop of the audit "
                "script) is reported as TOLERATED and the recorded official V85 cells are written instead "
                "(manifest.json: tolerated_fallback_to_recorded_list)",
                "2026-11-08: PASS with this script in the development tree on the original V84 (85 s; rerun after a "
                "last edit of the script, outputs byte-identical to the run before it) and in the clean copy on the "
                "rebuilt V84 (88 s): v84_queue.csv, the three audit files and the change list content-identical, "
                "manifest.json differs only in seconds and in the recorded sha256 of V84 (the rebuilt V84 has other "
                "bytes)"]),
    dict(id="final", gpu=False, status="verified", minutes=4,
         desc="final V86 / V87 = V84 + the recorded Task 2 queue guard flips (94 / 104 cells), each asserted against "
              "V84's queue_pred",
         cmds=["python work/build_final_v86_v87.py"],
         outs=[*zips("v86_spacetime_guard", "v87_combined"), "work/results/final_v86_v87_manifest.json"],
         frozen=["work/results/final_v86_v87/V86_spacetime_guard_changes.csv",
                 "work/results/final_v86_v87/V87_changes.csv"],
         notes=["the change lists (candidate, submission_id, split, panel, window_id, timestamp, link_id, before, "
                "after) are the decisions of the V86 spatio-temporal guard and the V85 history guard, run by a team "
                "member on V84; the V86 rows are frozen (its code is not part of this package), the V85 rows are recomputed by stage v85 and asserted equal (report section Final queue guards)",
                "with --tolerant a listed cell whose V84 value is not the recorded 'before' is reported as TOLERATED "
                "and still set to the recorded 'after' (manifest: tolerated_before_mismatches)",
                "2026-11-08 development tree (original V84): both rebuilt files byte-identical to the submitted ones; "
                "clean copy (rebuilt V84): both content-identical (6,980,504 lines), line ends differ on the same 521 "
                "lines as V84 (219 s)"]),
    dict(id="verify", gpu=False, status="verified", minutes=1, internal="verify",
         desc="read-only: V87 and V86 vs the recorded sha256 and line by line vs the submitted files (--ref-root, default "
              "work/reference/final/ = the copies shipped in the package)",
         cmds=[], outs=[], frozen=[],
         notes=["passes iff each final zip matches its recorded sha256 or is content-identical to the reference copy; "
                "always allowed (no --force needed)",
                "clean copy 2026-11-08: PASSED (sha256 differs, content identical for both; 22 s)"]),
]


# ------------------------------------------------------------------ hashes and comparisons
def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for blk in iter(lambda: f.read(1 << 22), b""):
            h.update(blk)
    return h.hexdigest()


def _is_sha(x) -> bool:
    return isinstance(x, str) and len(x) == 64 and all(c in "0123456789abcdef" for c in x)


def norm(p: str) -> str:
    """Manifest path -> path relative to ROOT ("work/..."), whatever machine or folder wrote it."""
    p = p.replace("\\", "/")
    root = str(ROOT).replace("\\", "/").rstrip("/") + "/"
    if p.lower().startswith(root.lower()):
        return p[len(root):]
    i = p.rfind("/work/")
    if i >= 0:
        return p[i + 1:]
    p = p.lstrip("./")
    return p if p.startswith("work/") else "work/" + p


def recorded_hashes() -> dict[str, dict[str, str]]:
    """{relative path: {sha256: source manifest}} from results/*manifest*.json and results/*/*manifest*.json.

    Collected: every {"path", "sha256"} entry at any depth (state/queue inputs, output_files, supplements);
    top-level output + sha256/output_sha256; manifests with only name + sha256/output_sha256/zip_sha256 ->
    work/submissions/<name>.zip; base/baseline + base_sha256/baseline_sha256 -> that base zip; and the two final zips.
    A path can carry several recorded values (e.g. the 10-19 and 10-23 builds of v51_april.zip)."""
    ref: dict[str, dict[str, str]] = {}

    def add(path: str, sha: str, src: str) -> None:
        d = ref.setdefault(path, {})
        if sha not in d or src.startswith(Path(path).stem + "_manifest"):   # prefer the zip's own manifest as source
            d[sha] = src

    def walk(x, src: str) -> None:
        if isinstance(x, dict):
            if isinstance(x.get("path"), str) and _is_sha(x.get("sha256")):
                add(norm(x["path"]), x["sha256"], src)
            for v in x.values():
                walk(v, src)
        elif isinstance(x, list):
            for v in x:
                walk(v, src)

    files = sorted(glob.glob(str(WORK / "results" / "*manifest*.json"))) + \
        sorted(glob.glob(str(WORK / "results" / "*" / "*manifest*.json")))
    for f in files:
        try:
            m = json.loads(Path(f).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(m, dict):
            continue
        src = Path(f).name
        walk(m, src)
        sha = next((m[k] for k in ("sha256", "output_sha256", "zip_sha256") if _is_sha(m.get(k))), None)
        if sha:
            if isinstance(m.get("output"), str):
                add(norm(m["output"]), sha, src)
            elif isinstance(m.get("name"), str) and "supplement" not in src:
                add(f"work/submissions/{m['name']}.zip", sha, src)
        for bk, hk in (("base", "base_sha256"), ("baseline", "baseline_sha256")):
            b, h = m.get(bk), m.get(hk)
            if isinstance(b, str) and _is_sha(h):
                rel = norm(b) if ("/" in b or "\\" in b) else f"work/submissions/{b.removesuffix('.zip')}.zip"
                add(rel, h, src)
    for rel, h in FINAL.values():
        add(rel, h, "run_all.FINAL")
    return ref


def _lines(path: Path):
    """Raw lines of a .csv, or of the single CSV member of a .zip."""
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as z:
            names = [n for n in z.namelist() if not n.endswith("/")]
            if len(names) != 1:
                raise ValueError(f"{path.name}: expected one member, found {len(names)}")
            with z.open(names[0]) as f:
                yield from io.BufferedReader(f, 1 << 22)
    else:
        with path.open("rb") as f:
            yield from f


def compare_content(new: Path, ref: Path) -> tuple[bool, str]:
    """Line-by-line comparison (.zip member / .csv) with an explicit line-count check; other files by sha256.
    Line endings are compared separately (Linux LF vs Windows CRLF is reported, not counted as a difference)."""
    if new.suffix.lower() not in (".zip", ".csv"):
        same = sha256(new) == sha256(ref)
        return same, "bytes identical" if same else "bytes differ"
    n_new = n_ref = ndiff = neol = 0
    first = None
    by_task: dict[str, int] = {}
    submission = None
    for a, b in itertools.zip_longest(_lines(new), _lines(ref)):
        n_new += a is not None
        n_ref += b is not None
        if a is None or b is None:
            continue
        sa, sb = a.rstrip(b"\r\n"), b.rstrip(b"\r\n")
        if submission is None:
            submission = sa.startswith(b"submission_id,task")
        if sa != sb:
            ndiff += 1
            if first is None:
                first = (n_new, sa, sb)
            if submission and n_new > 1:
                t = sa.split(b",", 2)[1].decode(errors="replace")
                by_task[t] = by_task.get(t, 0) + 1
        elif a != b:
            neol += 1
    ok = n_new == n_ref and ndiff == 0
    if ok:
        msg = f"content identical, {n_new:,} lines"
    else:
        count = (f"LINE COUNT DIFFERS: new {n_new:,} vs ref {n_ref:,} lines" if n_new != n_ref else
                 f"{n_new:,} lines (same count)")
        tasks = ", ".join(f"{k} {v:,}" for k, v in sorted(by_task.items()))
        msg = f"{count}; {ndiff:,} differing lines in the common part" + (f" ({tasks})" if tasks else "")
    if neol:
        msg += f"; line endings differ on {neol:,} lines"
    if first is not None:
        msg += (f"\n        first difference at line {first[0]:,}:\n          new: {first[1][:160].decode(errors='replace')}"
                f"\n          ref: {first[2][:160].decode(errors='replace')}")
    return ok, msg


def find_ref(ref_root: Path, rel: str) -> Path | None:
    """Reference copy of ROOT/rel: ref_root/rel, ref_root/<rel without work/> (ref_root = a work tree), or for a
    submission zip also ref_root/<file name> (a flat folder of zips)."""
    cands = [ref_root / rel]
    if rel.startswith("work/"):
        cands.append(ref_root / rel[len("work/"):])
    if rel.startswith("work/submissions/"):
        cands.append(ref_root / Path(rel).name)
    return next((c for c in cands if c.exists()), None)


def compare_outs(s: dict, ref_root: Path) -> None:
    for o in s["outs"]:
        p = ROOT / o
        if not p.exists():
            print(f"  [ref] {o}: MISSING locally")
            continue
        if p.is_dir():
            continue
        r = find_ref(ref_root, o)
        if r is None:
            print(f"  [ref] {o}: no reference copy under {ref_root}")
            continue
        ok, msg = compare_content(p, r)
        print(f"  [ref] {o}: {'OK ' if ok else 'DIFF '}{msg}", flush=True)


def verify_final(ref_root: Path | None) -> int:
    """V87 and V86 vs the recorded SHA-256 of the submitted files, and by content (line by line, line endings reported
    separately) vs a reference copy: under --ref-root if given, else the copies of the submitted files that the
    package ships in work/reference/final/.  A rebuilt zip rarely has the recorded bytes (the submitted zips end 725
    lines in LF and the rest in CRLF, a rebuild only 204), so a content match is what normally makes verify pass."""
    if ref_root is None and (ROOT / FINAL_REF).is_dir():
        ref_root = ROOT / FINAL_REF
        print(f"verify: comparing with the shipped copies of the submitted files in {FINAL_REF}/", flush=True)
    ok_all = True
    for k, (rel, want) in FINAL.items():
        p = ROOT / rel
        if not p.exists():
            print(f"verify {k}: {rel} MISSING")
            ok_all = False
            continue
        h = sha256(p)
        ok = h == want
        print(f"verify {k}: {rel} sha256 {h[:16]} {'== recorded' if ok else '!= recorded ' + want[:16]}", flush=True)
        if ref_root is not None:
            r = find_ref(ref_root, rel)
            if r is None:
                print(f"verify {k}: no reference copy under {ref_root}")
            else:
                same, msg = compare_content(p, r)
                print(f"verify {k}: vs {r}: {msg}", flush=True)
                ok = ok or same
        ok_all &= ok
    if ok_all:
        print("verify: PASSED")
        return 0
    if os.environ.get("TFB_TOLERANT", "") == "1":
        print("verify: NOT identical, TOLERATED (TFB_TOLERANT=1): the number of differing lines and the first one are "
              "shown above; score the rebuilt files to check the 1% tolerance of the award rule")
        return 0
    print("verify: NOT identical (see above; the 1% score tolerance may still hold)")
    return 1


# ------------------------------------------------------------------ CLI commands
def cmd_plan() -> None:
    tot, unknown = 0.0, []
    for s in STAGES:
        m = s.get("minutes")
        if m is None:
            unknown.append(s["id"])
        else:
            tot += m
        mins = "?" if m is None else f"~{m:g} min"
        print(f"[{s['status']:8s}] {s['id']:12s} {'GPU' if s['gpu'] else 'CPU'} {mins:>9s}  {s['desc']}")
        for c in s["cmds"]:
            print(f"        $ {c}")
        if s.get("internal"):
            print(f"        (internal: {s['internal']})")
        for c in s.get("retrain_cmds", []):
            print(f"        retrain $ {c}")
        for o in s["outs"]:
            print(f"        -> {o}")
        for new, ref in s.get("compare", []):
            print(f"        compare {new} == {ref}")
        for r in s.get("refuse_if", []):
            print(f"        refuses to start if {r} exists")
        if s.get("frozen"):
            print(f"        frozen inputs: {len(s['frozen'])} entries (--list-frozen)")
        for n in s.get("notes", []):
            print(f"        note: {n}")
    print(f"\n{len(STAGES)} stages; total ~{tot / 60:.1f} h without retraining (stage times measured in the clean-copy "
          f"run of 2026-11-04 ... 11-08 on the reference machine)"
          f"{' + unknown: ' + ', '.join(unknown) if unknown else ''}")


def expand(entry: str) -> list[Path]:
    if any(ch in entry for ch in "*?["):
        return sorted(Path(p) for p in glob.glob(str(ROOT / entry)))
    return [ROOT / entry]


def cmd_list_frozen() -> None:
    seen: dict[str, str] = {}
    missing = 0
    for s in STAGES:
        if not s.get("frozen"):
            continue
        print(s["id"])
        for e in s["frozen"]:
            paths = expand(e)
            if not paths:
                print(f"    MISSING  {e}  (pattern matched nothing)")
                missing += 1
                continue
            for p in paths:
                rel = p.relative_to(ROOT).as_posix()
                if not p.exists():
                    print(f"    MISSING  {rel}")
                    missing += 1
                    continue
                if rel not in seen:
                    seen[rel] = sha256(p) if p.is_file() else "directory"
                size = p.stat().st_size if p.is_file() else 0
                print(f"    ok  {size:>12,}  {seen[rel][:16]}  {rel}")
    print(f"\n{len(seen)} distinct frozen files present, {missing} missing")


def cmd_check(ref_root: Path | None = None) -> None:
    ref = recorded_hashes()
    for s in STAGES:
        for o in s["outs"]:
            p = ROOT / o
            if not p.exists():
                st = "MISSING"
            elif p.is_dir():
                st = f"dir ({sum(1 for x in p.rglob('*') if x.is_file())} files)"
            else:
                h = sha256(p)
                rec = ref.get(o)
                if not rec:
                    st = "no recorded hash"
                elif h in rec:
                    st = f"sha256 OK ({rec[h]})"
                else:
                    st = "sha256 DIFFERS from " + ", ".join(f"{k[:12]} ({v})" for k, v in rec.items())
            print(f"{s['id']:12s} {o:66s} {st}", flush=True)
        if ref_root is not None:
            compare_outs(s, ref_root)


def log_row(started: str, sid: str, cmd: str, rc, secs: float) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    new = not LOG.exists()
    with LOG.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["started", "stage", "cmd", "rc", "seconds"])
        w.writerow([started, sid, cmd, rc, f"{secs:.1f}"])


def run_cmd(sid: str, c: str) -> int:
    parts = shlex.split(c)
    if parts[0] == "python":
        parts[0] = PY
    started, t0 = time.strftime("%Y-%m-%d %H:%M:%S"), time.time()
    print(f"[{sid}] $ {c}", flush=True)
    try:
        rc = subprocess.call(parts, cwd=ROOT)
    except KeyboardInterrupt:
        log_row(started, sid, c, "interrupted", time.time() - t0)
        raise SystemExit(130)
    secs = time.time() - t0
    log_row(started, sid, c, rc, secs)
    print(f"[{sid}] rc={rc} {secs:.0f}s", flush=True)
    return rc


def cuda_problem() -> str | None:
    try:
        import torch
    except Exception as e:  # noqa: BLE001
        return f"torch cannot be imported ({e})"
    if not torch.cuda.is_available():
        return "torch.cuda.is_available() is False"
    if not torch.cuda.is_bf16_supported():
        return "the GPU does not support bf16"
    return None


def refused(s: dict) -> list[str]:
    return [r for r in s.get("refuse_if", []) if (ROOT / r).exists()]


def run_compare(s: dict) -> None:
    """Stage-level comparison of a rebuilt output with a shipped copy (reported, never fatal)."""
    for new, ref in s.get("compare", []):
        started, t0 = time.strftime("%Y-%m-%d %H:%M:%S"), time.time()
        if not (ROOT / new).exists() or not (ROOT / ref).exists():
            ok, msg = False, f"missing {new if not (ROOT / new).exists() else ref}"
        else:
            ok, msg = compare_content(ROOT / new, ROOT / ref)
        print(f"  [compare] {new} vs {ref}: {'OK ' if ok else 'DIFF '}{msg}", flush=True)
        log_row(started, s["id"], f"internal:compare {new} {ref}", 0 if ok else 1, time.time() - t0)


def keep_awake() -> None:
    """Windows only, for the long --run-all: ask the OS not to sleep and to keep the display on while this process
    runs (on laptops with Modern Standby a screen-off timeout pauses every process even with ES_SYSTEM_REQUIRED; a
    clean-copy run stalled 6.3 h that way), and ignore Ctrl+C in this console (inherited by the stage processes; a
    stray Ctrl+C killed two earlier runs).  All are released when the process exits; close the console window to
    stop a run.  A closed lid or an explicit Sleep / sign-out still stops the run."""
    if sys.platform == "win32":
        try:
            import ctypes
            k = ctypes.windll.kernel32
            # ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED
            k.SetThreadExecutionState(0x80000000 | 0x00000001 | 0x00000002)
            k.SetConsoleCtrlHandler(None, True)
            print("keep-awake requested (system and display); Ctrl+C is ignored (close the window to stop)", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"WARNING: could not request keep-awake / Ctrl+C protection ({e})", flush=True)


def mark_done(sid: str) -> None:
    DONE.mkdir(parents=True, exist_ok=True)
    (DONE / f"{sid}.ok").write_text(time.strftime("%Y-%m-%d %H:%M:%S") + "\n", encoding="utf-8")


def cmd_run(sel: list[dict], force: bool, retrain: bool, ref_root: Path | None, resume: bool = False) -> None:
    if resume:
        done = [s for s in sel if (DONE / f"{s['id']}.ok").exists()]
        for s in done:
            print(f"--resume: skipping {s['id']} (ended without error at "
                  f"{(DONE / (s['id'] + '.ok')).read_text(encoding='utf-8').strip()})", flush=True)
        sel = [s for s in sel if s not in done]
        if not sel:
            print("--resume: every selected stage has already ended without error", flush=True)
            return
    keep_awake()
    for d in WORK_DIRS:
        (ROOT / d).mkdir(parents=True, exist_ok=True)
    todo = [s["id"] for s in sel if s["status"] != "verified" and not s.get("internal")]
    if todo and not force:
        raise SystemExit("stages not yet fully re-run in a clean copy (status partial / todo): " + " ".join(todo) +
                         "\nadd --force to run them anyway")
    blocked = {s["id"]: refused(s) for s in sel if refused(s)}
    if blocked:
        raise SystemExit("refusing to start (these stages overwrite caches in place; use a clean copy): " +
                         "; ".join(f"{k}: {' '.join(v)} exist(s)" for k, v in blocked.items()))
    print("stages:", " ".join(s["id"] for s in sel), flush=True)
    if any(s["gpu"] for s in sel):
        prob = cuda_problem()
        if prob:
            print(f"WARNING: GPU stages selected but {prob}; they will fail", flush=True)
    if retrain and any(s.get("retrain_cmds") for s in sel):
        print("WARNING --retrain: the training scripts refuse to overwrite model files; move the shipped models of "
              "these stages aside first: " + " ".join(s["id"] for s in sel if s.get("retrain_cmds")), flush=True)
    for s in sel:
        m = s.get("minutes")
        print(f"\n===== {s['id']} ({'GPU' if s['gpu'] else 'CPU'}, ~{'?' if m is None else m} min): {s['desc']}",
              flush=True)
        (DONE / f"{s['id']}.ok").unlink(missing_ok=True)
        if s.get("internal") == "verify":
            started, t0 = time.strftime("%Y-%m-%d %H:%M:%S"), time.time()
            rc = verify_final(ref_root)
            log_row(started, s["id"], "internal:verify", rc, time.time() - t0)
            if rc:
                raise SystemExit(rc)
            mark_done(s["id"])
            continue
        if refused(s):
            raise SystemExit(f"{s['id']} refuses to start: {' '.join(refused(s))} exist(s)")
        for o in s["outs"]:            # output folders (a clean tree has none of them; no script refuses an existing one)
            p = ROOT / o
            (p.parent if p.suffix else p).mkdir(parents=True, exist_ok=True)
        exist = [o for o in s["outs"] if o.startswith(WRITE_ONCE) and (ROOT / o).exists() and (ROOT / o).is_file()]
        if exist:
            print("WARNING: outputs already present (the scripts refuse to overwrite them): " + " ".join(exist),
                  flush=True)
        for c in (s.get("retrain_cmds", []) if retrain else []) + s["cmds"]:
            rc = run_cmd(s["id"], c)
            if rc != 0:
                raise SystemExit(rc)
        run_compare(s)
        if ref_root is not None:
            compare_outs(s, ref_root)
        mark_done(s["id"])


def main() -> None:
    ids = [s["id"] for s in STAGES]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", action="store_true", help="print stages, commands, outputs, status, runtimes")
    ap.add_argument("--check", action="store_true", help="outputs vs the sha256 recorded in the manifests")
    ap.add_argument("--list-frozen", action="store_true", help="shipped artifacts each stage needs")
    ap.add_argument("--ref-root", type=Path, metavar="DIR",
                    help="reference tree (layout of the project root, of work/, or a flat folder of zips): content "
                         "comparison with --check, after each stage with --run/--run-all/--from, and in verify")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--run", nargs="+", metavar="ID", help="run these stages (in pipeline order)")
    g.add_argument("--run-all", action="store_true", help="run every stage")
    g.add_argument("--from", dest="from_id", metavar="ID", help="run from this stage to the end")
    ap.add_argument("--force", action="store_true", help="also run stages whose status is todo")
    ap.add_argument("--retrain", action="store_true", help="run each stage's retrain_cmds before its cmds")
    ap.add_argument("--resume", action="store_true",
                    help="with --run/--run-all/--from: skip stages that already ended without error "
                         "(work/results/run_all_done/<stage>.ok)")
    ap.add_argument("--tolerant", action="store_true",
                    help="set TFB_TOLERANT=1 for every stage: on another GPU model, differences from the decisions "
                         "recorded on the development GPU are reported (TOLERATED) instead of stopping the run")
    a = ap.parse_args()
    if a.tolerant:
        os.environ["TFB_TOLERANT"] = "1"
    if os.environ.get("TFB_TOLERANT", "") == "1":
        print("TFB_TOLERANT=1: recorded-decision checks report differences instead of stopping (README section 5)",
              flush=True)
    bad = [i for i in (a.run or []) + ([a.from_id] if a.from_id else []) if i not in ids]
    if bad:
        ap.error(f"unknown stage(s) {bad}; valid: {' '.join(ids)}")
    if a.ref_root is not None and not a.ref_root.is_dir():
        ap.error(f"--ref-root {a.ref_root} is not a directory")
    sel = ([s for s in STAGES if s["id"] in set(a.run)] if a.run else
           STAGES[ids.index(a.from_id):] if a.from_id else list(STAGES) if a.run_all else [])
    if not (a.plan or a.check or a.list_frozen or sel or a.ref_root):
        ap.print_help()
        return
    if a.plan:
        cmd_plan()
    if a.list_frozen:
        cmd_list_frozen()
    if a.check or (a.ref_root is not None and not sel):
        cmd_check(a.ref_root)
    if sel:
        cmd_run(sel, a.force, a.retrain, a.ref_root, a.resume)


if __name__ == "__main__":
    main()
