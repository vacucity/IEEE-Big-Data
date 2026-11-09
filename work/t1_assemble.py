#!/usr/bin/env python
"""Blend the Task 1 GBM file with the U-Net predictions into a new per-task state file.

Cells outside the Task 2 blackouts take  sum_k w_k * U-Net_k + (1 - sum_k w_k) * GBM  (weights from
t1_blend.py / t1_blend3.py); cells inside a blackout (whole corridor blank for 90 minutes, where the GBM's
inputs are out of range) take their own weights. One weight per --tags entry (U-Net prediction files).

Usage (from the "IEEE Big data" folder):
  python work/t1_assemble.py --split validation --gbm work/sub/state_validation.csv \
      --out work/sub/state_validation_unet.csv --tags "" _lwr05 --wv 0.4 0.3 --wq 0.2 0.5 \
      --gap-wv 1 0 --gap-wq 0.5 0.5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import CACHE, CLEAN, KEY_BASE, PANELS, links, slot_of, t1_targets  # noqa: E402

UNET = CACHE / "t1_unet"


def target_blackout(panel: str, split: str) -> np.ndarray:
    """is_t2_blackout for the Task 1 targets, in t1_targets order."""
    pf = pq.ParquetFile(CLEAN / panel / f"{split}.parquet")
    out = []
    for i in range(pf.num_row_groups):
        t = pf.read_row_group(i, columns=["is_t1_target", "is_t2_blackout"])
        m = t.column("is_t1_target").to_numpy()
        out.append(t.column("is_t2_blackout").to_numpy()[m])
    return np.concatenate(out)


def unet_preds(panel: str, split: str, tag: str) -> tuple[np.ndarray, np.ndarray]:
    z = np.load(UNET / f"pred_{panel}_{split}{tag}.npz")
    days = sorted({int(k[1:]) for k in z.files})
    return np.concatenate([z[f"v{d}"] for d in days]), np.concatenate([z[f"q{d}"] for d in days])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--gbm", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tags", nargs="+", default=[""])
    ap.add_argument("--wv", type=float, nargs="+", required=True)
    ap.add_argument("--wq", type=float, nargs="+", required=True)
    ap.add_argument("--gap-wv", type=float, nargs="+", required=True)
    ap.add_argument("--gap-wq", type=float, nargs="+", required=True)
    args = ap.parse_args()
    k = len(args.tags)
    assert len(args.wv) == len(args.wq) == len(args.gap_wv) == len(args.gap_wq) == k
    ref = {}
    for p in PANELS:
        tg = t1_targets(p, args.split)
        us = [unet_preds(p, args.split, t) for t in args.tags]
        blk = target_blackout(p, args.split)
        assert all(len(u[0]) == len(tg["key"]) for u in us) and len(blk) == len(tg["key"]), p
        ref[p] = (tg["key"], us, blk, dict(zip(links(p).link_id, links(p).pos)))
    first, n_rows, n_gap = True, 0, 0
    for c in pd.read_csv(args.gbm, chunksize=1_000_000):
        sv, sq = c.speed_kmh.to_numpy(float).copy(), c.flow_vph.to_numpy(float).copy()
        for p, idx in c.groupby("panel").indices.items():
            keys, us, blk, pos = ref[p]
            k = slot_of(c.timestamp.iloc[idx]) * KEY_BASE + c.link_id.iloc[idx].map(pos).astype(np.int64).to_numpy()
            j = np.searchsorted(keys, k)
            assert np.array_equal(keys[j], k), f"{p}: rows not found among the targets"
            g = blk[j]
            v_new, q_new = np.zeros(len(j)), np.zeros(len(j))
            wv_tot, wq_tot = np.zeros(len(j)), np.zeros(len(j))
            for (uv, uq), a, b, ga, gb in zip(us, args.wv, args.wq, args.gap_wv, args.gap_wq):
                wv, wq = np.where(g, ga, a), np.where(g, gb, b)
                v_new += wv * uv[j]
                q_new += wq * uq[j]
                wv_tot += wv
                wq_tot += wq
            sv[idx] = v_new + (1 - wv_tot) * sv[idx]
            sq[idx] = q_new + (1 - wq_tot) * sq[idx]
            n_gap += int(g.sum())
        c["speed_kmh"], c["flow_vph"] = np.clip(sv, 5.0, 120.75), np.maximum(sq, 0.0)
        c.to_csv(args.out, mode="w" if first else "a", header=first, index=False)
        first = False
        n_rows += len(c)
    print(f"wrote {args.out}: {n_rows:,} rows, {n_gap:,} inside Task 2 blackouts")


if __name__ == "__main__":
    main()
