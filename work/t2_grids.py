#!/usr/bin/env python
"""Day-by-link grids for Task 2 work, cached as .npz per panel and split.

Arrays are (days, 288, n_links) float32 in pos order:
  V, Q, O     visible masked layer on eligible cells (speed_clean, flow_clean, occ_clean), NaN elsewhere
  Vt, Qt, Ot  train only: unmasked layer on eligible cells (speed_true, flow_true, occ_true)
  B           1 inside a Task 2 horizon or its buffer (is_t2_blackout)
plus dates (str) and weekday (int).

Usage (from the "IEEE Big data" folder):
  python work/t2_grids.py            # builds every T2 panel x {train, validation, private}
"""
from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import CACHE, CLEAN, STEPS, T2_PANELS, links  # noqa: E402

GRIDS = CACHE / "t2_grids"
SPLITS = ("train", "validation", "private")


def build(panel: str, split: str) -> Path:
    path = GRIDS / f"{panel}_{split}.npz"
    src = CLEAN / panel / f"{split}.parquet"
    if path.exists() and path.stat().st_mtime >= src.stat().st_mtime and (split != "train" or "Ot" in np.load(path).files):
        return path
    n = len(links(panel))
    pf = pq.ParquetFile(src)
    cols = ["date", "weekday", "is_score_eligible", "is_t2_blackout", "speed_clean", "flow_clean", "occ_clean"]
    if split == "train":
        cols += ["speed_true", "flow_true", "occ_true"]
    D = pf.num_row_groups
    out = {k: np.full((D, STEPS, n), np.nan, np.float32) for k in ("V", "Q", "O")}
    if split == "train":
        out.update({k: np.full((D, STEPS, n), np.nan, np.float32) for k in ("Vt", "Qt", "Ot")})
    out["B"] = np.zeros((D, STEPS, n), np.int8)
    dates, wd = [], []
    for i in range(D):
        t = pf.read_row_group(i, columns=cols)
        assert t.num_rows == STEPS * n
        dates.append(t.column("date")[0].as_py())
        wd.append(int(t.column("weekday")[0].as_py()))
        E = t.column("is_score_eligible").to_numpy().reshape(STEPS, n)
        g = lambda c: t.column(c).to_numpy().astype(np.float32).reshape(STEPS, n)
        out["V"][i], out["Q"][i], out["O"][i] = g("speed_clean"), g("flow_clean"), g("occ_clean")
        out["B"][i] = t.column("is_t2_blackout").to_numpy().reshape(STEPS, n)
        if split == "train":
            out["Vt"][i] = np.where(E, g("speed_true"), np.nan)
            out["Qt"][i] = np.where(E, g("flow_true"), np.nan)
            out["Ot"][i] = np.where(E, g("occ_true"), np.nan)
    GRIDS.mkdir(parents=True, exist_ok=True)
    np.savez(path, dates=np.array(dates), weekday=np.array(wd), **out)
    return path


@lru_cache(maxsize=4)
def load(panel: str, split: str) -> dict:
    z = np.load(build(panel, split))
    return {k: z[k] for k in z.files}


if __name__ == "__main__":
    for p in T2_PANELS:
        for s in SPLITS:
            print(p, s, build(p, s), flush=True)
