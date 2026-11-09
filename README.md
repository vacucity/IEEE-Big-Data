# TrafficFlowBench 2026 — team *vacucity* — reproduction package

This package rebuilds our two final Kaggle submissions for the 2026 IEEE Big Data Cup *TrafficFlowBench*
from the official data release and the trained models and frozen decision files shipped here.

| | rebuilt file (submitted as) | SHA-256 | public LB | private LB |
|---|---|---|---|---|
| **V87** | `work/submissions/v87_combined.zip` (`V87_combined.zip`) | `6bc4e383967c9dbf96d8b343879b8b729436f3118636619bb113bde635e3dff0` | 0.90368 | 0.89956 |
| **V86** | `work/submissions/v86_spacetime_guard.zip` (`V86_spacetime_guard.zip`) | `9e1ba0523c88b6ad70aeccf95c41e64a3c1639558c9c4a555867c75fc7cf1867` | 0.90359 | 0.89962 |
| V84 (base, not selected) | `work/submissions/v84_observation_flow.zip` | `a429599ade6ee8663b28f850f86acc0819ba426d2b2f9cc0861b2831edc36090` | 0.90336 | not recorded |

V84 is the end of our model chain. V86 and V87 are V84 with Task 2 `queue_pred` cells flipped by two queue guards
that a team member ran on V84 (no model is retrained): V86 flips the 94 cells of a spatio-temporal guard (62
validation / 32 private, 18 windows; 63 queued → not queued, 31 not queued → queued); V87 flips those 94 and the 10
cells of the V85 history guard (8 / 2, all not queued → queued), 104 cells in 20 windows. Each zip holds one member
`submission.csv` (6,980,503 rows); every other byte of it equals V84's, so the Task 1 and Task 4 rows, and the Task 3
score computed from Task 1, are those of V84. Scores are from the Kaggle submission page. The guards are described in
the report, section *Final queue guards*.

The method, its validation and all compliance disclosures are described in the PDF report
(`report/main.pdf`, also sent with the package e-mail).

## 1. Requirements

* Reference machine, the only one we ran on: Intel Core i7-13700H (14 cores / 20 threads), 16 GB RAM (15.6 GB
  usable, enough for every stage), NVIDIA GeForce RTX 4070 Laptop GPU 8 GB (driver 555.97), Windows 11 build 26200.
  Linux is untested (the keep-awake and Ctrl+C handling of `run_all.py` are Windows-only).
* About 45 GB free disk for the files the run regenerates (`clean_data/` and the outputs under `work/`; our clean
  copy held 38 GB of them at the end of the run: `clean_data/` 7.3 GB, new files under `work/` 30.7 GB), plus about
  12 GB for the unzipped official release (`kaggle_public/`, 11.5 GiB) and about 1.2 GB for this package, i.e. about
  58 GB in total.
* A CUDA GPU with bf16 support and a driver for CUDA 12.1 for the U-Net, CNN and BiLSTM inference stages
  (`t1_base`, `t2_queue`, `t1_v59blk`, `t1_v61b3`, `t2_v72`, `t2_check`); 8 GB was enough (peak use not measured).
  All other stages, including `v85`, `final` and `verify`, run on CPU.
* Python 3.11.5 and the pinned packages in `work/requirements.txt`
  (numpy 1.26.4, pandas 2.0.3, pyarrow 22.0.0, scipy 1.10.1, scikit-learn 1.7.2, lightgbm 4.6.0, catboost 1.2.8,
  xgboost 2.1.3, torch 2.5.1+cu121). Exact versions used are recorded in `work/ENVIRONMENT.txt` (its
  `Windows-10-10.0.26200` is how Python 3.11 reports Windows 11 build 26200). Our runs used the development Anaconda
  environment; the install commands below have not been tested in a fresh environment.

```
conda create -n tfb python=3.11.5
conda activate tfb
pip install -r work/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
```

## 2. Data

Download the official release (*TrafficFlowBench Release 1.0*, as its `README.md` is titled) from the competition's
Data tab and unzip it so that the folder `kaggle_public/` (containing `README.md`, `submission_key.csv`,
`sample_submission.csv`, `task1/`, `task2/`, `task4/`, `corridors/`, `config/`) sits next to `work/`. Our copy is
11.5 GiB unzipped; its `submission_key.csv` has 642,748,735 bytes and SHA-256
`8232c9ce6682fdd3120e6729a30e4514866ab492ef7b167ad3952e00478bd071`.

```
<package root>/
  README.md, LICENSE
  MANIFEST.json             <- size and SHA-256 of every shipped file
  report/main.pdf
  kaggle_public/            <- official release (not shipped)
  clean_data/               <- written by stage data
  work/                     <- code, trained models, frozen decision files, requirements.txt, ENVIRONMENT.txt
    reference/t2_queue/     <- copies of the five original Task 2 queue files (compared by t2_check)
    reference/final/        <- copies of the submitted V87 and V86 (compared by verify)
    results/final_v86_v87/  <- frozen V86 and V87 change lists (read by stage final)
    v85_history_guard/      <- the team member's V85 hand-off: guard code and frozen inputs (run by stage v85)
  trafficflowbench-public-main/   <- the official Task 4 builder used by stage t4 (shipped subset)
```

All paths are derived from the scripts' own location, so the package can live anywhere. `run_all.py --list-frozen`
maps the shipped models and decision files to the stages that read them.

## 3. Run

Everything goes through one entry script, `work/run_all.py`. Run it from the package root:

```
python work/run_all.py --plan              # stage table: commands, outputs, GPU need, status, measured runtime
python work/run_all.py --list-frozen       # shipped inputs of each stage (exists / size / SHA-256 prefix)
python work/run_all.py --run-all --force   # rebuild everything in order (log: work/results/run_all_log.csv)
python work/run_all.py --run-all --tolerant   # recommended on any GPU other than ours (section 5)
python work/run_all.py --run verify        # repeat only the comparison of V87 and V86 (last stage of --run-all)
python work/run_all.py --check             # SHA-256 of every output vs the manifests in work/results/ (see below)
```

* `--from <stage>` runs from a stage to the end; `--run <stage ...>` runs single stages (in pipeline order).
* `--force` runs stages marked `todo` or `partial`; since our clean-copy run all 36 stages are marked `verified`
  (section 5), so it is no longer needed, but it does no harm.
* `--check`: the package ships no result manifests, so in a fresh package `--check` compares each output with the
  hash its own builder recorded during this run, where there is one. This is a self-consistency check, not a
  comparison with the development build; only `verify` compares V87 and V86 with the submitted files. `--ref-root`
  adds a content comparison of every stage output with a reference tree (with `--check`, or after each stage of a run).
* `--resume` (with `--run-all`, `--from` or `--run`) skips stages that already ended without error (marker
  `work/results/run_all_done/<stage>.ok`). It skips whole stages only: an interrupted stage restarts from its first
  command, and many scripts refuse to overwrite files they had already written. So first delete what the interrupted
  stage wrote (its `->` outputs in `--plan`, e.g. `work/results/t2_asof/asof*` for `t2_check`, but not the shipped
  inputs listed by `--list-frozen`, e.g. `work/sub/probs/t2ml_lgbm_probs_*.pkl`) and what blocks its start
  (`refuses to start if` lines in `--plan`, e.g. `work/baseline_rebuild/task4_*` for `t4`); if the rerun stops with a
  `FileExistsError`, delete the file it names. Then rerun the same command with `--resume`.
* On Windows every run asks the OS to keep the system and the display on (with Modern Standby a screen-off timeout
  otherwise pauses the run; our clean-copy run stalled 6.3 h that way) and ignores Ctrl+C. To stop a run, close its
  console window; continue later with `--resume`. A closed lid, Sleep or signing out still pauses or stops the run.
  Keep the machine on mains power.
* `--retrain` first runs a stage's optional GPU training commands (`retrain $` lines in `--plan`, about 7 h in
  total). Only `t1_v59blk`, `t1_v61b3` and `t2_v72` have them; none is needed to reproduce V87 and V86. The U-Net
  training commands refuse to overwrite existing nets, so first move the stage's shipped nets out of `work/models/`
  (`t1_unet_fold*_v59bs.pt`, `*_v59ss_p0.pt`, `*_v59ss_p1.pt` for `t1_v59blk`; `*_v61b3_s1.pt`, `*_v61b3_ext.pt`
  for `t1_v61b3`). The `t1_v61b3` retraining reads the `train_runs.csv` written by the `t1_v59blk` retraining. The
  `t2_v72` fine-tunes start from the shipped base nets (leave them in place) and overwrite the four shipped
  `*_hw96all.pt` files without asking; keep a copy. Retraining is not guaranteed to be bit-exact (see section 6).

Our clean-copy check ran `python work/run_all.py --run-all --resume --force --ref-root <development tree>` in a copy
of the package next to the release. `--ref-root` compares each stage's outputs by content with the development tree
as the stage ends (outputs without a counterpart there print `no reference copy`); `verify` then compares V87 and
V86 with the submitted files in that tree instead of `work/reference/final/`.

Runtime, measured in that clean-copy run on the reference machine without retraining (the stage times of `--plan`,
interruptions excluded): 18.2 h of stage time in total (3.6 h of it in the six GPU stages); by group: data cleaning
37 min, Task 4 reference 69 min, Task 1 production chain 81 min, calibration stages `v51_zip` … `t1_v69` 9.3 h
(`t1_v52` alone 4.4 h), late Task 1 stages `t1_v73` … `t1_v84` 5.0 h, the three Task 2 stages (`t2_queue`, `t2_v72`,
`t2_check`) 40 min, `v85`, `final` and `verify` 6 min. The run itself went from 4 to 8 November. It was resumed with
`--resume` after interrupted consoles (closed windows or a stray Ctrl+C) and after five stops that led to changes in
`run_all.py` or to new scripts: a working folder that `t2_queue` needs (now created by `run_all.py`), `t2_check`
placed before V84 existed (now after `t1_v84`), the feature-alignment file that `build_v77.py` reads (now written by
the new `audit_v77_feature_alignment.py`, whose first version stopped the run once more because it counted NaN in the
same cells as unequal), and `final` reached before stage `v85` was added. A 6.3 h standby pause inside `merge_v60`
only delayed the run (excluded from the times above). No stage script was changed after its stage had passed; the
only code files under `work/` that changed after the run began are
`run_all.py`, `audit_v77_feature_alignment.py` and the scripts of the stages added then, `build_v85_history_guard.py`
and `build_final_v86_v87.py` (with the hand-off folder `work/v85_history_guard/`). The final files are written to
the paths in the table above.

## 4. What the stages do

`run_all.py --plan` prints the full table. In order:

1. `data` — clean the release into per-corridor parquet files (see *Preprocessing* below). `t4` — rebuild the
   official Task 4 NNLS reference with the organisers' builder into `work/baseline_rebuild/` and compare it line by
   line with the shipped `work/baseline/task4_*/public_reference.csv`, which the zips carry (a difference is
   reported, not fatal).
2. `t1_base` — Task 1 production chain: LightGBM residual models and two month-adapted whole-day U-Nets
   (shipped), blend, residual stack.
3. `t2_queue` — Task 2 queue files written in one pass from the shipped member models
   (`work/t2_final_queue.py write`); byte-identical to the originals.
4. `v51_zip` … `t1_v69` — the first zip (Task 1 state + Task 2 queues + Task 4 ODME), then the Task 1 calibration
   lanes (per-month bias, restack, blackout specialists, congested density, speed common mode). Parallel lanes patch
   the same base and are combined by the disjoint-row merges `merge_v58`, `merge_v60` and `merge_v62`; the other
   stages patch the previous zip.
5. `t2_v72` — April ongoing rows from the March-fine-tuned members (55 April queue cells of the V69 zip).
6. `t1_v73` … `t1_v84` — late Task 1 residual corrections (factor and ridge models and LightGBM boosters, each with
   its held-out checks), giving **V84**, the end of the model chain and the base of both final files.
7. `t2_check` — read-only Task 2 checks of V84: re-derives all 160 official windows from the models cell by cell,
   checks the two queue files V84 was built from (`work/sub/queue_validation_v54_onset.csv`,
   `queue_private_v72_april.csv`) against V84's queue rows, runs the truncation (as-of) check that deletes every
   row after each window's origin, and compares the five `t2_queue` files cell by cell with
   `work/reference/t2_queue/`. It needs V84, so it runs after `t1_v84`. These checks cover V84's model-built queue
   rows; V86 and V87 differ from them in the 94 / 104 guard cells (the 10 V85 cells recomputed by stage `v85`, all
   applied by stage `final`).
8. `v85` — recomputes the 10 V85 cells on the rebuilt V84 with the team member's own code
   (`work/build_v85_history_guard.py` around `work/v85_history_guard/`, section 7): on the `constr` official ongoing
   windows the guard replays the V84-era CPU ensemble from frozen probabilities with the selection-rule conditioning,
   finds the 30 training-period histories (June 2030 – February 2031) closest to the window's own history up to and
   including its origin T, and flips a cell from not queued to queued only where decoding 0.9 × replay probability +
   0.1 × analogue queue share, instead of the replay probability alone, turns it on (first step kept fixed), V84 has 0,
   at least 20 analogues observe it and their queue share exceeds the replay probability by at least 0.30 (12
   candidates, 10 kept). It asserts that V84's queue rows equal the team member's, the recomputed audit the shipped
   one (line endings ignored), and the cells both the official V85 list and the V85 rows of the V87 list.
9. `final` — `work/build_final_v86_v87.py` writes V86 = V84 + the frozen V86 change list and V87 = V84 + the V86
   list + the `v85` cells (asserted equal to the frozen V87 list), asserting that each listed cell is a V84 queue row
   with the recorded `before` value and copying all other bytes; the zips are written like the submitted ones (one
   member `submission.csv`, deflate, 1980 timestamp). The team member describes the V86 guard as correcting V84's
   queue predictions by spatio-temporal linkage and gating by neighbouring historical observations; flips whose
   CPU replay disagreed with the submitted base were not applied, and 4 ambiguous flips were blocked. Its code is
   not shipped, so we cannot show that it uses only data up to each window's origin T (report, section X,
   disclosure 6); a re-run with the rows after each origin deleted reproduces these 94 recorded cells by
   construction and does not test them.
10. `verify` — compares V87 and V86 with the recorded SHA-256 of the submitted files and by content (line by line,
   line endings reported separately) with `work/reference/final/` (or `--ref-root`); it passes if each file matches
   either way. In a rebuilt tree it passes by content: from V60 on the rebuilt zips hold the same content as ours
   but different line ends (521 lines in every full zip, 570 in the two V61 state-only zips; the original V84, V86
   and V87 end 725 lines in LF and the rest in CRLF, their rebuilds 204), and V86 and V87 inherit this from V84.

A registered private-only hedge of V84 was built but not selected; it is not part of V86 or V87.

**Preprocessing** (stage `data`, no randomness). `work/clean_data.py` turns the release into one table per corridor
and split; nothing is deleted and no missing value is filled. Steps: merge the network tables (links,
fundamental-diagram parameters, topology, ramp map) into one link table ordered by milepost; map the Task 1 target
cells and Task 2 window spans to integer keys; place each day's masked layer (on train also the unmasked truth layer)
on a complete time × link grid; aggregate valid on/off-ramp flows onto their mainline link; validate keys, grid
completeness, flags and masked vs truth agreement; add quality tiers, strict values on score-eligible cells
(`pct_observed` ≥ 75) and anomaly flags; derive time features, per-lane flow, density q/v, speed ratio and the queue
label (speed ≤ 0.6 × free-flow speed). Outputs: `clean_data/<corridor>/{train,validation,private}.parquet` (zstd, one
row group per day) and `links.parquet`, plus `clean_data/cleaning_report.{json,csv}` and `data_dictionary.csv`. The
model features built on this layer are described in the report (sections *Task 1* and *Task 2*).

## 5. Verification status and tolerance

* **Full clean-copy run (2026-11-04 … 11-08), all 36 stages:** a copy of this package next to the official release,
  without any development output, ran `--run-all` (resumed as described in section 3) and ended with `verify`
  passing: the rebuilt V87 and V86 are content-identical to the submitted files (6,980,504 lines each; SHA-256
  differs because 521 lines carry CRLF instead of LF, the same lines as in the rebuilt V84). From `t1_base` on, every
  stage compared its outputs with the development tree (`--ref-root`); stage `data` writes a folder, which
  `--ref-root` skips, so its 40 parquet files and `data_dictionary.csv` were compared by SHA-256 after the run, and
  stage `t4` compared the rebuilt Task 4 reference with the shipped copy itself. The cleaned parquet files, the Task 4
  reference, every state and queue CSV and every zip that the development tree still holds (V51, V60 … V84, V86,
  V87) are content- or byte-identical, and so are the LightGBM models refitted inside the stages and the ridge
  coefficients of `t1_v81` / `t1_v82` (bytes). The twelve intermediate zips of V52 … V59 are no longer kept in the
  development tree and were not compared directly; their state CSVs were, and V60 and every later zip, which carry
  their rows forward, are content-identical. Among the compared outputs, byte differences occur only in result files
  that record timings, paths, hashes or float noise: the `seconds` fields of `clean_data/cleaning_report.*`,
  `t1_v59blk`'s `crossfit_ss_choice.json` and the `v85` manifest (which also records the SHA-256 of the rebuilt V84);
  float differences of at most 1.2e-13 in the `t1_v73` and `t1_v74` coefficients and summaries; and the as-of log of
  `t2_check` (per-window seconds, absolute data paths, and an optional column of the development run). Diagnostic
  result files that are not stage outputs (for example the manifests of the zip builders) also differ in recorded
  zip hashes and line-end counts. `run_all.py --plan` marks every stage `verified` and lists these differences in its
  notes.
* Task 2: the queue files written from the models are byte-identical to the originals, all 160 official windows of
  V84 re-derive with 0 mismatching cells, and the truncation check changes 0 of 174,000 cells (also in the clean
  copy).
* Final files: the shipped `build_v85_history_guard.py` recomputes the guard's audit identically and re-derives its
  10 cells (with numpy 1.26.4 / pandas 2.0.3), on the original V84 in the development tree and on the rebuilt V84 in
  the clean copy; `final` writes V86 and V87 byte-identical to the submitted files from the original V84 and
  content-identical from the rebuilt one.
* Seven builders on the chain (`build_v77.py`, `build_v78_step.py`, `build_v79_flow.py`, `build_v81_cmode.py`,
  `build_v82_interactions.py`, `build_v83_flow.py`, and `build_v84_observation.py` through the `build()` of
  `build_v84_context.py`) check the SHA-256 of their input zip against the development build (`build_v81_density.py`
  holds the same check, but `build_v81_cmode.py` replaces the `prepare()` that contains it). A rebuilt zip does not
  have those bytes, even on the development machine: the original zips hold a historical mix of CRLF and LF line ends
  that the rebuild repeats only in part (the clean-copy log reports `line endings differ on 521 lines` for V60 and
  every later full zip, 570 for the two V61 state-only zips, with identical content), and
  another platform or zlib version can add further byte differences. The mismatch warning is therefore expected and
  the run continues; every value-replay assertion inside the builders still applies. `TFB_STRICT=1` makes the hash
  checks fatal, which only makes sense on the original development zips (a rebuild would stop at `build_v77`).
* **Another GPU model: use `--tolerant`.** The Task 2 BiLSTM / CNN members run in bf16, and bf16 kernels differ
  between GPU models, so a few queue cells can flip on a GPU other than ours (RTX 4070 Laptop). The award rule's 1%
  tolerance on the private score absorbs a few cells, but by default the run stops at the first check that compares
  recomputed Task 2 decisions with the ones recorded on our GPU:
  `t2_final_queue.py write` (stage `t2_queue`, April V51) and `t2_v71_april.py --variant v72` (stage `t2_v72`, V48
  and V51) against `work/results/t2_v48/`, `t2_v51/`; `t2_final_queue.py compare` (stage `t2_check`) against
  `work/reference/t2_queue/`; `v85` on V84's queue rows and the recomputed V85 cells; `final` on the `before` value
  of every flipped cell; and `verify`. With `python work/run_all.py --run-all --tolerant` (or the environment
  variable `TFB_TOLERANT=1`) each of these prints a line starting with `TOLERATED` with the number of differing cells
  and the run goes on: the Task 2 rows use the decisions recomputed on your GPU; `v85` writes the recorded V85 cells
  if its recomputation differs or stops, and `final` sets every listed guard cell to its recorded `after` value
  (the submitted one), as it already does for the V86 cells; `verify` reports the differing lines of V87 / V86 and
  passes. Structural checks (row sets, keys, counts) and the as-of truncation check stay fatal. `grep TOLERATED` in
  the log lists every tolerated difference. If a default run already stopped at one of these checks, delete the
  partial outputs of that stage (section 3) and continue with `--run-all --resume --tolerant`. We tested this mode
  on our machine (2026-11-09) by changing recorded V48 / V51 decisions and two V84 queue cells on purpose: every
  check reported the difference and the run continued; the rebuilt Task 2 queue files and V72 outputs stayed
  byte-identical (the decisions recomputed on our GPU are the submitted ones), `v85` wrote the recorded V85 cells,
  `final` set the guard cells to their recorded values, and V87 / V86 then differed from the submitted files only
  in the changed V84 cell that lies outside the guard lists, as a flip on another GPU would. The default mode
  stopped at each of these checks. On our GPU no check fires, so both modes give the same files.

## 6. Random seeds

Every random draw is seeded in code; nothing on the default path depends on the clock or on the run order.
Command-line seeds exist only for the optional training commands (`--seed`, default 0; in `t1_unet.py`, `t2_cnn.py`
and `t2_rnn.py` it is the index s of a network in its ensemble, e.g. `_g0_s1` was trained with `--seed 1`). Column
*acts* says when a seed matters: **run** = on the default path (the model is refitted inside its stage in every run),
**retrain** = only when a shipped model is retrained. corridor = position in `kaggle_public/config/corridors.json`
(Task 2: among the eight Task 2 corridors); day = day index in the split (one parquet row group per day); tag = 0 for
`_w48`, 1 for `_w48lwr`.

| component | script(s) | seed(s) | acts |
|---|---|---|---|
| Task 1 LightGBM base models | `task1_gbm.py` | LightGBM `seed=2026`; training-row sample `default_rng(2026 + 1000·corridor + day)` | retrain |
| Task 1 whole-day U-Nets | `t1_unet.py` | `torch.manual_seed(2026 + 100·s)`, `default_rng(2026 + fold + 100·s)` (batch order, gap and cell hiding); month adaptation (`_vp`): `torch.manual_seed(2026 + 7)`, `default_rng(2026 + 7 + fold)`; cuDNN deterministic | retrain |
| V30 bridge / stack | `t1_v30_bridge.py`, `t1_v30_stack.py` | stack: LightGBM `seed=30531`, training-row 12 % sample `default_rng(30531 + 1000·i + day)`; bridge: hide-20 % masks `default_rng(30541 + 1000·i + day)`, its `_v30even` nets adapted with the `_vp` adaptation seeds; i = position in the script's own corridor list (stack: D12_I405_S, D7_I210_W, D12_I5_S, D7_I10_E; bridge: its `--panels` list), not the global index | retrain (stack models and selection are frozen) |
| V52 per-month bias | `t1_v52_monthbias.py` | hide-20 % masks `default_rng(52052 + 1000·corridor + 500·[April] + day)`, reused by V59loc | run |
| V57 restack / blackout lane, V59loc | `t1_v57_restack.py`, `t1_v57_blackout.py`, `t1_v59_local.py` | LightGBM 1 (`t1_v57_restack.GBM_BASE`, also V59loc) and 57 (`t1_v57_blackout.GBM_PARAMS`, gap residual GBM) | run |
| V59 blackout specialists | `t1_v59_blkspec.py` | residual GBM S2: LightGBM 57 (`GBM_PARAMS`); nets `59 + 10·fold + tag` (self-supervised `_v59ss` nets `+ 1000 + 100·(parity + 1)`); cuDNN deterministic | run (GBM), retrain (nets) |
| V61 congested density / blackout nets | `t1_v61_congest.py`, `t1_v61_blk3.py` | congest: LightGBM 1 (also V64 and the V65 cache); blk3: residual GBMs LightGBM 57, nets `61 + 10·fold + tag` (set s1) or `62 + 10·fold + tag` (set ext), sample draws `+ 5000`, extra gaps `default_rng(61)`; cuDNN deterministic | run (GBMs), retrain (nets) |
| V65 cache, V69, V73 | `t1_v65_lwrcong.py` (`structure`, `baseline`), `t1_v68_cmgbm.py` (parameters), `t1_v69_cmgbm_apply.py`, `t1_v73_*.py` | V65 cache `v64cf.npz` (read by V66, V69, V73, V81): LightGBM 1; V69: LightGBM 68 (V73 reuses 68); V73 gate: date bootstrap `default_rng(73073)` | run |
| V75–V84 residual corrections | `t1_v75_*.py` … `t1_v84_*.py` | LightGBM 75075 (V75 and the V76 "deep" model), 76076 / 76176 (V76 context / flow), 77077 (V77, V78), 79079 (V79), 83083 (V83), 84084 (V84); V81 and V82 are closed-form ridge fits without a seed (teachers: the V64 LightGBM models, seed 1) | run |
| Task 2 LightGBM members | `task2_ml2.py`, `t2_ml2_g0.py` | LightGBM `seed=2026`; ongoing training-window sample `default_rng(2026 + 1000·corridor + day)` | retrain |
| Task 2 CatBoost member | `t2_cat_g0.py` (on the `t2_ml2_g0.py` rows) | `random_seed=20261017` | retrain |
| Task 2 CNN / BiLSTM | `t2_cnn.py`, `t2_rnn.py`, `t2_rnn_march.py` | `2026 + s`; March fine-tunes `2026 + 100 + --seed` = 2126 (the recorded commands use the default `--seed 0` and pick the base network with `--base`); CNN cuDNN deterministic, BiLSTM not | retrain |
| Task 2 onset random forest (D7_I405_N) | `t2_v30_onset.py`, fitted by `t2_final_onset.py` | scikit-learn `random_state=3030` (180 trees, `n_jobs=2`; refitted on train in `t2_queue` and `t2_check`, no shipped model) | run |
| Task 2 selection-rule conditioning (V52, V72) | `t2_v52_selrule.py` | Monte Carlo with K = 20,000 draws, `seed = crc32(window_id)` per window | run |
| V85 history guard: conditional sampling of the CPU replay | `v85_history_guard/replay/work/analog_gate_official_audit.py` (`v85_decoders.samples`) | K = 20,000 conditional samples, `default_rng(crc32(window_id))` per window | run (`v85`) |
| V86 spatio-temporal guard | not shipped (frozen change list) | not available (code not shipped) | none on the default path |

The LightGBM models that the default path fits (from V57 on, inside their stages) and the shipped V30 stack and V81
parity-gate boosters (`work/results/t1_v30_stack/model_*.txt`,
`work/results/t1_v81_density_context/model_old_parity_*.txt`) use `deterministic=True, force_col_wise=True`. The
shipped Task 1 base boosters (`work/models/t1_gbm_{v,q,k}.txt`) and Task 2 LightGBM members
(`work/models/t2_ml2_*.txt`, and `t2_v8_ongoing.txt`, fitted by `t2_ens.py` with the `task2_ml.py` parameters) were
trained with seed 2026 on 16 threads but without these flags (their model files record `deterministic: 0`); the
default path only predicts with them, and retraining them is not guaranteed to be bit-exact.
The U-Nets and CNNs were trained with deterministic cuDNN under bf16 autocast (bit-exact retraining not tested); the
BiLSTM members used non-deterministic cuDNN kernels, so retraining them is not bit-exact. The default path runs
inference from all shipped networks. Data cleaning and Task 4 (NNLS) involve no randomness.

The other fixed seeds in the code do not affect the submitted values: 0, 5, 7 and 54 drive cross-validation,
bootstrap intervals or the junk values of the truncation (as-of) checks; 57 also seeds the report-only window
bootstraps of V57 / V59 / V61b3; day·288 + T (+ 7) seeds the V52 rule-selection runs of `t2_v52_selrule_eval.py`;
65 seeds only the V65 transition-aware candidates (`t1_v65_lwrcong.py`, none chosen); 82082 belongs to the rejected
V82 GBM candidate of `t1_v82_congest_residual.py`, which `run_all.py` never trains; 2033 seeds the IoU (Lovász)
fine-tune of `t2_ongoing_v27_train.py`, rejected on validation (model not shipped). `task2_ml.py` (the v8 LightGBM
and an XGBoost comparison; `seed=2026`, ongoing-window sample `default_rng(2026 + corridor)`) ships because the chain
imports its helpers; its shipped probabilities `work/sub/probs/t2ml_lgbm_probs_*.pkl` are a pass-through key of the
V27 cache and reach no cell.

## 7. External data, pretrained models and licenses

* **Data.** Only the official TrafficFlowBench release (`kaggle_public/`: train, the masked validation and private
  months, the corridor networks and configuration). Its license: "Subject to the Competition Rules"; the network
  topology is derived in part from OpenStreetMap (ODbL, https://www.openstreetmap.org/copyright). No external
  dataset, no calendar or weather data, no hand labelling and no answers from outside the release were used.
* **Pretrained models.** None from outside the project; every network and booster was trained by us on the release
  only. The base models (Task 1 LightGBM and U-Nets, the V30 stack, the V59 / V61 gap-specialist nets, every Task 2
  member) were trained on the training layer. Because Task 1 is offline, some Task 1 models also used the visible,
  non-target cells of the validation and private months, with visible cells hidden again and reconstructed (the
  Task 1 target cells are never seen): of the shipped models the four month-adapted U-Nets (`_vp`), the eight
  self-supervised blackout nets (`_v59ss_p0`, `_v59ss_p1`) and the four V81 parity-gate heads
  (`model_old_parity_*`); and every other calibration model of the stages `t1_v52` … `t1_v84`, which is refitted
  inside its stage rather than shipped. The April Task 2 members were in addition fine-tuned on the validation month
  (March), as allowed by the organisers' ruling of 15 October 2026. The V85 guard's frozen probabilities come from
  our own Task 2 members.
* **APIs and paid tools.** None; no network access at run time.
* **Third-party code.** `trafficflowbench-public-main/` is a subset of the organisers' public repository
  (https://github.com/jacky850/trafficflowbench-public, MIT License, see its `LICENSE`), used by stage `t4` to build
  the official Task 4 NNLS reference.
* **Team-member code.** `work/v85_history_guard/` is a team member's V85 hand-off (`先读_复现交接.md`, in Chinese):
  guard code (`replay/work/`), 24 frozen CPU probability caches, 8 training-period co-queue tables, the original audit,
  the original build and cache scripts (`reference/`, not run here), the author's proxy results (`proxy/`; selection
  evidence on repeatedly used March windows, not an independent test) and a screenshot of V85's public score on the
  Kaggle submission page (`evidence/v85-public-score.png`). The author tested it with Python 3.13.9, numpy 2.3.5,
  pandas 2.3.3 and pyarrow 23.0.0; stage `v85` runs it with the versions of section 1. Its `evaluate.py` differs from
  ours only in reading the data paths from the environment. The V84 / V85 zips and `inputs/v84_queue.csv` listed in
  its `SHA256SUMS.txt` are not shipped (the other listed files match), so its own `rebuild_v85.py` and
  `replay_gate_from_cache.py` do not run as shipped; stage `v85` writes those queue rows from the rebuilt V84, checks
  their hash (line endings ignored) and runs the audit script directly. Its author released it under our MIT
  License (`LICENSE`). The V86 guard's code is not
  included (its decisions ship as the frozen change list of stage `final`).
* **Libraries.** numpy, pandas, scipy, scikit-learn, PyTorch (BSD); LightGBM (MIT); pyarrow, CatBoost, XGBoost
  (Apache-2.0); matplotlib for the report figures only (PSF-based license); all open source. The CUDA 12.1 build of
  PyTorch comes with NVIDIA's CUDA and cuDNN runtime libraries, which are freely redistributable but not open source.
* **This package.** Our code is released under the MIT License (`LICENSE`). The shipped models and the other files
  derived from the release are provided only to verify our submissions and remain subject to the Competition Rules.

## 8. Contact

Team *vacucity* (Kaggle team "vacucity"):

* Chujie Wang (Kaggle wcj030728), The University of Chicago
* Jiayu Liu (Kaggle lazywhale110), Tianjin University
* Feiyan Fu (Kaggle feiyanfu), Linkerbot Beijing Technology Co., Ltd.
* Ao Li (Kaggle mixriaoliao), Fuzhou University of International Studies and Trade

Contact: Chujie Wang, chujiewang424@gmail.com. Code: https://github.com/vacucity/IEEE-Big-Data
