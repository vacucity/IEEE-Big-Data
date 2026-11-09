"""One predeclared adaptive IoU decision on the frozen V28 ongoing probabilities.

This experiment changes the decision rule, not the trained models. Select on
validation visible pseudo windows; run private confirmation only if selection
passes. These proxy labels are not the withheld leaderboard targets.
"""

from __future__ import annotations

import json
import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate import ROOT, T2_PANELS, links, slot_of
from t2_group_v27 import groups, smooth
from t2_ongoing_v27 import load, probs, official_cache


OUT = ROOT / "work" / "results" / "t2_v33_adaptive_iou"
CONFIG = {"similarity": 0.95, "alpha": 0.25, "theta": 0.425, "weights": [1, 1, 1, 5]}


def adaptive_iou(pr: np.ndarray) -> np.ndarray:
    """Maximize ratio of expected intersection to expected union for one window."""
    flat = pr.ravel()
    order = np.argsort(-flat)
    sorted_prob = flat[order]
    intersection = np.cumsum(sorted_prob)
    k = np.arange(1, len(flat) + 1)
    union = k + sorted_prob.sum() - intersection
    ratio = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)
    best = int(np.argmax(ratio)) + 1 if sorted_prob.sum() > 0 else 0
    out = np.zeros(len(flat), dtype=bool)
    out[order[:best]] = True
    return out.reshape(pr.shape)


def window_scores(pred: np.ndarray, y: np.ndarray, mask: np.ndarray) -> np.ndarray:
    p, truth = pred & mask, y & mask
    intersection = (p & truth).sum(axis=(1, 2))
    union = (p | truth).sum(axis=(1, 2))
    return np.divide(intersection, union, out=np.ones(len(union)), where=union > 0)


def evaluate(split: str) -> pd.DataFrame:
    rows = []
    for panel, z in load(split).items():
        pr = smooth(probs(z, CONFIG["weights"]), groups(panel, CONFIG["similarity"]), CONFIG["alpha"])
        baseline = pr >= CONFIG["theta"]
        proposal = np.stack([adaptive_iou(p) for p in pr])
        for name, pred in (("v32_ongoing", baseline), ("adaptive_iou", proposal)):
            scores = window_scores(pred, z["y"], z["mask"])
            for day, T, score in zip(z["day"], z["T"], scores):
                rows.append({"split": split, "panel": panel, "day": int(day), "T": int(T),
                             "rule": name, "iou": float(score)})
    return pd.DataFrame(rows)


def summary(rows: pd.DataFrame) -> dict:
    out = {}
    for rule, d in rows.groupby("rule"):
        out[rule] = {
            "overall": float(d.groupby("panel").iou.mean().mean()),
            "early": float(d[d.day < 15].groupby("panel").iou.mean().mean()),
            "late": float(d[d.day >= 15].groupby("panel").iou.mean().mean()),
            "panel": d.groupby("panel").iou.mean().to_dict(),
        }
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "protocol.json").write_text(json.dumps({
        "decision": "select k maximizing ratio of expected intersection to expected union, no tuned parameters",
        "incumbent": CONFIG,
        "selection": "validation overall +0.001 minimum, early and late nonnegative",
        "confirmation": "private overall positive; otherwise do not build a submission",
        "limitation": "visible noisy pseudo-window labels, repeatedly used validation month",
    }, indent=2) + "\n")
    val = evaluate("validation")
    val.to_csv(OUT / "validation_windows.csv", index=False)
    s = summary(val)
    base, candidate = s["v32_ongoing"], s["adaptive_iou"]
    delta = {k: candidate[k] - base[k] for k in ("overall", "early", "late")}
    accepted = bool(delta["overall"] >= 0.001 and delta["early"] >= 0 and delta["late"] >= 0)
    report = {"validation": s, "validation_delta": delta, "accepted_for_private_confirmation": accepted}
    (OUT / "selection.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    if not accepted:
        return
    private = evaluate("private")
    private.to_csv(OUT / "private_windows.csv", index=False)
    report["private"] = summary(private)
    report["recommended"] = (report["private"]["adaptive_iou"]["overall"]
                             > report["private"]["v32_ongoing"]["overall"])
    (OUT / "confirmation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"private": report["private"], "recommended": report["recommended"]}, indent=2))


def official_counts() -> None:
    """Unlabeled change scope; never use official output appearance to tune rule."""
    rows = []
    for split in ("validation", "private"):
        for panel in T2_PANELS:
            z = official_cache(panel, split)
            pr = smooth(probs(z, CONFIG["weights"]), groups(panel, CONFIG["similarity"]), CONFIG["alpha"])
            old = pr >= CONFIG["theta"]
            new = np.stack([adaptive_iou(p) for p in pr])
            rows.append({"split": split, "panel": panel, "windows": len(pr),
                         "changed_cells": int((old != new).sum()),
                         "old_positive_cells": int(old.sum()),
                         "new_positive_cells": int(new.sum())})
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "official_count_audit.csv", index=False)
    print(out.to_string(index=False))
    print(out.groupby("split")[["changed_cells", "old_positive_cells", "new_positive_cells"]].sum().to_string())


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def official_build() -> None:
    report = json.loads((OUT / "confirmation.json").read_text())
    assert report["accepted_for_private_confirmation"] and report["recommended"]
    expected = {"validation": 31, "private": 43}
    manifest = {"config": CONFIG, "files": {}, "onset_source": "V30", "state_source": "V31"}
    for split in ("validation", "private"):
        source = ROOT / "work" / "sub" / f"queue_{split}_v30_onset.csv"
        dest = ROOT / "work" / "sub" / f"queue_{split}_v33_adaptive_iou.csv"
        if dest.exists():
            raise FileExistsError(dest)
        q = pd.read_csv(source)
        q["slot"] = slot_of(q.timestamp)
        changed = 0
        for panel in T2_PANELS:
            z = official_cache(panel, split)
            pr = smooth(probs(z, CONFIG["weights"]), groups(panel, CONFIG["similarity"]), CONFIG["alpha"])
            old = pr >= CONFIG["theta"]
            new = np.stack([adaptive_iou(p) for p in pr])
            pos = dict(zip(links(panel).link_id, range(len(links(panel)))))
            for i, window_id in enumerate(z["ids"]):
                mask = (q.window_id == window_id).to_numpy()
                row = q.loc[mask]
                assert len(row) > 0, window_id
                horizon = (row.slot - z["T"][i] - 1).to_numpy()
                link_pos = row.link_id.map(pos).to_numpy()
                assert np.isfinite(link_pos).all() and ((horizon >= 0) & (horizon < 6)).all()
                old_rows = old[i, horizon, link_pos.astype(int)].astype(int)
                assert np.array_equal(old_rows, row.queue_pred.to_numpy()), window_id
                new_rows = new[i, horizon, link_pos.astype(int)].astype(int)
                changed += int((new_rows != old_rows).sum())
                q.loc[mask, "queue_pred"] = new_rows
        assert changed == expected[split], (split, changed)
        q.drop(columns="slot").to_csv(dest, index=False)
        manifest["files"][split] = {"source": str(source), "source_sha256": sha256(source),
                                    "output": str(dest), "sha256": sha256(dest),
                                    "changed_ongoing_cells": changed}
    (OUT / "official_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    commands = {"evaluate": main, "official-counts": official_counts, "official-build": official_build}
    parser.add_argument("command", choices=tuple(commands), nargs="?", default="evaluate")
    commands[parser.parse_args().command]()
