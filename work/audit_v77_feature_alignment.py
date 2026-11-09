"""V77 feature-cache alignment audit; build_v77.py copies its result into the V77 manifest.

For two sample blocks (validation D7_I405_S, private D12_I5_N) the features computed directly with
t1_v77_flow_factors.features must equal the cached rows of results/t1_v77_flow_factors/X.npy exactly (NaN in the
same cells counts as equal).  Writes
results/t1_v77_flow_factors/feature_alignment.json (later copied by t1_v78_flow_boost / t1_v79_flow_detail).
It is the check of build_v79_speed.feature_check applied to the V77 cache; in the development tree the file was
written on 2026-10-27 by an ad-hoc run of this check, which no script recorded (a clean-copy run stopped at
build_v77 without it, 2026-11-06).  Read-only apart from the one JSON file; refuses to overwrite it.

  python work/audit_v77_feature_alignment.py
"""
from __future__ import annotations

import json

import numpy as np

import t1_v77_flow_factors as N
from t1_v77_flow_factors import F, G, PANELS

SAMPLES = [("validation", "D7_I405_S"), ("private", "D12_I5_N")]


def main() -> None:
    out = N.OUT / "feature_alignment.json"
    if out.exists():
        raise FileExistsError(out)
    meta = np.load(G.OUT / "feature_rows.npz")
    X = np.load(N.OUT / "X.npy", mmap_mode="r")
    rows = []
    for s, p in SAMPLES:
        si, pi = F.SPLITS.index(s), PANELS.index(p)
        d, lh = F.LOC.load(p, s)
        z = F.baseline(s)
        direct = N.features(p, s, d, lh, z[p + "_v"], z[p + "_q"], pi, True)[d["free"]]
        cached = X[meta["owner"] == si * 10 + pi]
        # the context features contain NaN (missing neighbours); NaN in the same cells counts as equal
        assert np.array_equal(direct, cached, equal_nan=True), (s, p, direct.shape, cached.shape)
        rows.append(dict(split=s, panel=p, rows=len(direct), columns=direct.shape[1], exact_cache_alignment=True))
        print(rows[-1], flush=True)
    out.write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
