#!/usr/bin/env python
"""TrafficFlowBench data cleaning pipeline: merge -> validate -> clean -> derive -> export.

Turns the raw Kaggle release (about 9,700 files, 12 GB) into one tidy, analysis-ready
table per corridor and split. Every row is one detector link in one 5-minute interval,
and every cell carries an explicit status, so nothing has to be inferred from nulls.

Pipeline (one function per step):
  1. Network tables  links + fd_parameters + topology + ramp map -> one link table
                     ordered by milepost, with cross-table consistency checks.
  2. Task templates  Task 1 target cells and Task 2 window spans -> integer keys.
  3. Mainline merge  per day: masked layer (plus the unmasked truth layer on train)
                     placed on a complete (timestamp x link) grid.
  4. Ramp merge      per day: ramp validity rule, then on/off flows aggregated onto
                     the mainline link each ramp attaches to.
  5. Validation      duplicate keys, grid completeness, flag definitions, masked vs
                     truth agreement, and every blank cell accounted for.
  6. Cleaning        quality tiers, strict clean values (eligible cells only) and
                     anomaly flags. Nothing is deleted and no missing value is filled.
  7. Derived fields  time features, per-lane flow, density q/v, speed ratio, queue label.
  8. Export          zstd parquet per corridor/split, link table with a detector-bias
                     diagnostic, data dictionary and a step-by-step cleaning report.

Usage (from the "IEEE Big data" folder):
  python work/clean_data.py                                    # all corridors and splits
  python work/clean_data.py --panels D7_I405_N --splits train  # one corridor, train only
  python work/clean_data.py --panels D12_I405_S --max-days 3   # quick smoke test
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
DEFAULT_RELEASE = HERE.parent / "kaggle_public"
DEFAULT_OUT = HERE.parent / "clean_data"

SPLITS = ("train", "validation", "private")
STEPS = 288                    # 5-minute intervals per day
KEY_BASE = 1000                # cell key = absolute 5-min slot * KEY_BASE + link position
ELIGIBLE_PCT = 75              # official score-eligibility threshold on pct_observed
QUEUE_RATIO = 0.60             # official queue rule: speed <= 0.6 * free-flow speed
FREEFLOW_RATIO = 0.85          # cells above this speed ratio count as free flow (bias check)
SPIKE_KMH = 40.0               # jump in and out of a single cell larger than this = spike
SPEED_CEILING = 120.75         # release clips speed here (1.8-18% of cells sit exactly on it)
OCC_CEILING = 0.75             # release clips occupancy here (saturates in heavy congestion)
DENSITY_PER_OCC = 285.0        # density_occ_linear_vehpkm == 285 * occupancy, dropped as redundant

DATA_DICTIONARY = {
    "panel": "Directional corridor id",
    "split": "train / validation / private",
    "date": "Calendar date (synthetic calendar)",
    "timestamp": "Interval start, UTC text exactly as released (use for submission keys)",
    "ts": "Interval start as a UTC datetime",
    "tod": "Time-of-day index 0..287 (5-minute steps)",
    "hour": "Hour of day (UTC)",
    "weekday": "Day of week, 0 = Monday",
    "is_weekend": "Saturday or Sunday",
    "station_id": "Detector id (one detector per link)",
    "link_id": "Mainline link id",
    "pos": "Link position along the corridor, ordered by milepost then link_id",
    "mask_regime": "Task 1 masking regime of the day: R1 20%, R2 30%, R3 50% of eligible cells",
    "pct_observed": "Share of the interval the detector reported (0-100)",
    "quality_tier": "full (100) / partial (75-99) / degraded (1-74) / missing (0)",
    "is_score_eligible": "Official rule: pct_observed >= 75 (and values present in the unmasked layer)",
    "is_t1_target": "Cell is in the official Task 1 template (masked, must be reconstructed)",
    "is_t2_history": "Inside a published Task 2 window history [T-60, T)",
    "is_t2_blackout": "Inside a Task 2 horizon or its 60-minute buffer (T, T+90] - deliberately blank",
    "is_blanked": "Eligible cell whose values are null in the released masked layer",
    "speed_obs": "Speed visible in the released masked layer, km/h (includes degraded cells)",
    "flow_obs": "Flow visible in the released masked layer, veh/h, whole link",
    "occ_obs": "Occupancy visible in the released masked layer, 0-0.75",
    "speed_clean": "speed_obs on eligible cells only, else null (strict analysis value)",
    "flow_clean": "flow_obs on eligible cells only, else null",
    "occ_clean": "occ_obs on eligible cells only, else null",
    "flow_per_lane": "flow_clean / lanes (lanes from fd_parameters.csv, as the scorer uses)",
    "density_qv": "flow_clean / speed_clean, veh/km (the scorer's density definition)",
    "speed_ratio": "speed_clean / free-flow speed",
    "flow_cap_ratio": "flow_clean / capacity",
    "is_queued": "speed_clean <= 0.6 * free-flow speed (null when speed_clean is null)",
    "speed_true": "Train only: speed from the unmasked layer (answer for blanked cells)",
    "flow_true": "Train only: flow from the unmasked layer",
    "occ_true": "Train only: occupancy from the unmasked layer",
    "is_queued_true": "Train only: queue label computed from speed_true on eligible cells",
    "flag_flow_over_capacity": "Visible flow above link capacity (noise or capacity mismatch)",
    "flag_speed_at_ceiling": "Visible speed sits exactly on the 120.75 km/h clipping ceiling",
    "flag_occ_at_ceiling": "Visible occupancy sits exactly on the 0.75 clipping ceiling",
    "flag_speed_spike": "Visible speed jumps > 40 km/h into and back out of this single cell",
    "on_ramp_n": "Number of on-ramps attached to this link",
    "on_ramp_n_valid": "Attached on-ramps with a valid reading (eligible, pct>=75, finite)",
    "on_ramp_flow": "Total on-ramp flow when every attached on-ramp is valid; 0 if none attached; else null",
    "on_ramp_flow_partial": "Sum over the valid on-ramps only (lower bound; scorer's convention)",
    "off_ramp_n": "Number of off-ramps attached to this link",
    "off_ramp_n_valid": "Attached off-ramps with a valid reading",
    "off_ramp_flow": "Total off-ramp flow when every attached off-ramp is valid; 0 if none; else null",
    "off_ramp_flow_partial": "Sum over the valid off-ramps only",
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--release-root", type=Path, default=DEFAULT_RELEASE, help="the kaggle_public folder")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--panels", nargs="*", help="corridors to process (default: all ten)")
    ap.add_argument("--splits", nargs="*", choices=SPLITS, default=list(SPLITS))
    ap.add_argument("--max-days", type=int, help="process only the first N days per split (testing)")
    return ap.parse_args()


def date_of(path: Path) -> str:
    """synthetic_mainline_2030_06_01.parquet -> 2030-06-01"""
    return path.stem.split("_", 2)[-1].replace("_", "-")


def slot_of(text: pd.Series) -> np.ndarray:
    """Absolute 5-minute slot index of released timestamp text."""
    ts = pd.to_datetime(text, format="%Y-%m-%dT%H:%M:%SZ", utc=True)
    return ts.values.astype("datetime64[m]").astype(np.int64) // 5


def tod_of(text: pd.Series) -> np.ndarray:
    """Time-of-day index from 'YYYY-MM-DDTHH:MM:SSZ' without full datetime parsing."""
    return (text.str[11:13].astype(np.int16) * 12 + text.str[14:16].astype(np.int16) // 5).to_numpy()


def split_ids(value: object, known: set[str]) -> list[str]:
    return [x for x in str(value).split(";") if x and x in known] if isinstance(value, str) else []


# ---------------------------------------------------------------------------
# Step 1. Network tables -> one link table (路网静态表合并与一致性检查)
# ---------------------------------------------------------------------------
def build_link_table(panel_dir: Path, first_day_file: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    net = panel_dir / "network"
    links = pd.read_csv(net / "links.csv")
    fd = pd.read_csv(net / "fd_parameters.csv")
    topo = pd.read_csv(net / "lwr_mainline_topology.csv", dtype=str).drop_duplicates("link_id").set_index("link_id")
    amap = pd.read_csv(net / "ramp_attachment_map.csv", dtype=str)
    checks: dict = {}

    shared = ["length_km", "lanes", "free_speed_kmh", "capacity_vph"]
    m = links.merge(fd, on="link_id", how="outer", suffixes=("", "_fd"), indicator=True)
    checks["links_vs_fd_unmatched_rows"] = int((m["_merge"] != "both").sum())
    both = m[m["_merge"] == "both"]
    checks["links_vs_fd_value_mismatches"] = int(sum((~np.isclose(both[c], both[c + "_fd"])).sum() for c in shared))

    # fd_parameters.csv is authoritative: the official scorer takes lanes and FD values from it.
    st = fd.copy()
    topo_lanes = pd.to_numeric(topo["lanes"], errors="coerce")
    topo_cap = pd.to_numeric(topo["capacity_vph"], errors="coerce")
    checks["topology_lanes_differ_share"] = round(float((st.link_id.map(topo_lanes) != st.lanes).mean()), 4)
    checks["topology_capacity_differ_share"] = round(float((~np.isclose(st.link_id.map(topo_cap), st.capacity_vph)).mean()), 4)
    checks["links_missing_from_topology"] = int((~st.link_id.isin(topo.index)).sum())

    det = pd.read_parquet(first_day_file, columns=["station_id", "link_id", "milepost"]).drop_duplicates("link_id")
    checks["stations_per_link_max"] = int(det.groupby("link_id").station_id.nunique().max())
    st = st.merge(det, on="link_id", how="left")
    checks["links_without_detector"] = int(st.station_id.isna().sum())
    st = st.sort_values(["milepost", "link_id"]).reset_index(drop=True)  # same order as the Task 3 scorer
    st["pos"] = np.arange(len(st), dtype=np.int16)
    st["v_cut_kmh"] = QUEUE_RATIO * st.free_speed_kmh
    st["capacity_per_lane_vph"] = st.capacity_vph / st.lanes

    # Conservation neighbours come from incoming/outgoing_link_ids, never from from_node/to_node
    # (organizer guidance). Ids outside the detector set (CONN-*, unsensored links) are boundaries.
    known = set(st.link_id)
    up = [split_ids(topo["incoming_link_ids"].get(l), known) for l in st.link_id]
    down = [split_ids(topo["outgoing_link_ids"].get(l), known) for l in st.link_id]
    st["up_links"] = [";".join(x) for x in up]
    st["down_links"] = [";".join(x) for x in down]
    st["n_up"] = np.array([len(x) for x in up], dtype=np.int8)
    st["n_down"] = np.array([len(x) for x in down], dtype=np.int8)
    mp = st.set_index("link_id").milepost
    pairs = [(u, l) for l, ups in zip(st.link_id, up) for u in ups]
    checks["upstream_pairs"] = len(pairs)
    checks["upstream_has_lower_milepost_share"] = round(float(np.mean([mp[u] < mp[l] for u, l in pairs])), 4) if pairs else None
    checks["links_with_no_detector_neighbour"] = int(((st.n_up == 0) & (st.n_down == 0)).sum())

    # Ramps: ramp_attachment_map.csv is what the scorer uses; confirm the other two sources agree.
    syn = pd.read_csv(net / "synthetic_ramp_attachment_map.csv", dtype=str)
    checks["ramp_map_equals_synthetic_map"] = bool(amap.equals(syn))
    topo_pairs = {(r, l) for col in ("on_ramp_link_ids", "off_ramp_link_ids")
                  for l, v in topo[col].items() for r in split_ids(v, set(amap.ramp_link_id))}
    checks["ramp_map_equals_topology_ramps"] = topo_pairs == set(zip(amap.ramp_link_id, amap.nearest_mainline_link_id))
    checks["ramp_types"] = sorted(amap.ramp_type.unique().tolist())
    checks["ramps_on_unknown_link"] = int((~amap.nearest_mainline_link_id.isin(known)).sum())
    amap = amap.rename(columns={"nearest_mainline_link_id": "link_id"})
    for typ, prefix in (("OR", "on"), ("FR", "off")):
        counts = amap[amap.ramp_type == typ].groupby("link_id").size()
        st[f"{prefix}_ramp_n"] = st.link_id.map(counts).fillna(0).astype(np.int8)
    checks["n_links"], checks["n_on_ramps"], checks["n_off_ramps"] = len(st), int(st.on_ramp_n.sum()), int(st.off_ramp_n.sum())
    return st, amap, checks


# ---------------------------------------------------------------------------
# Step 2. Task templates -> integer keys (任务模板转为格子键)
# ---------------------------------------------------------------------------
def task1_target_keys(release: Path, panel: str, split: str, pos_of: dict) -> tuple[np.ndarray, int, int]:
    path = release / "task1" / panel / split / "sample_submission_state.csv"
    parts, rows, unknown = [], 0, 0
    for chunk in pd.read_csv(path, usecols=["timestamp", "link_id"], chunksize=1_000_000):
        pos = chunk.link_id.map(pos_of)
        unknown += int(pos.isna().sum())
        ok = pos.notna().to_numpy()
        parts.append(slot_of(chunk.timestamp[ok]) * KEY_BASE + pos[ok].astype(np.int64).to_numpy())
        rows += len(chunk)
    return np.unique(np.concatenate(parts)), rows, unknown


def task2_slots(release: Path, panel: str, split: str) -> tuple[np.ndarray, np.ndarray, int]:
    """Absolute slots of window histories [T-60, T) and of horizon+buffer spans (T, T+90]."""
    path = release / "task2" / panel / split / "window_index.csv"
    if not path.exists():  # D12_I405_N/S have no Task 2 windows
        return np.empty(0, np.int64), np.empty(0, np.int64), 0
    w = pd.read_csv(path)
    t = slot_of(w.forecast_origin)
    history = np.unique((t[:, None] + np.arange(-12, 0)[None, :]).ravel())
    blackout = np.unique((t[:, None] + np.arange(1, 19)[None, :]).ravel())
    return history, blackout, len(w)


# ---------------------------------------------------------------------------
# Steps 3-7 for one day (单日：合并 -> 校验 -> 清洗 -> 派生)
# ---------------------------------------------------------------------------
def scatter(n_cells: int, idx: np.ndarray, values, fill=np.nan, dtype=np.float32) -> np.ndarray:
    out = np.full(n_cells, fill, dtype=dtype)
    out[idx] = values
    return out


def process_day(panel: str, split: str, date: str, regime: str, masked_path: Path, truth_path: Path | None,
                ramp_path: Path | None, st: pd.DataFrame, amap: pd.DataFrame, t1_keys: np.ndarray,
                t2_hist: np.ndarray, t2_black: np.ndarray, stats: Counter, bias: dict) -> pd.DataFrame:
    n_links = len(st)
    n = STEPS * n_links
    pos_of = dict(zip(st.link_id, st.pos))
    day_slot0 = int(pd.Timestamp(date).value // (300 * 10**9))

    # --- Step 3. mainline merge onto the full (tod x link) grid --------------------------------
    m = pd.read_parquet(masked_path)
    stats["raw_rows"] += len(m)
    pos = m.link_id.map(pos_of)
    stats["rows_unknown_link"] += int(pos.isna().sum())
    m = m[pos.notna()]
    idx = tod_of(m.timestamp).astype(np.int64) * n_links + pos[pos.notna()].astype(np.int64).to_numpy()
    uniq, first = np.unique(idx, return_index=True)
    stats["duplicate_keys"] += len(idx) - len(uniq)
    m, idx = m.iloc[first], uniq
    stats["grid_cells_added"] += n - len(idx)  # cells absent from the file (0 in this release)

    pct = scatter(n, idx, m.pct_observed.to_numpy(), fill=0, dtype=np.int16)
    elig = scatter(n, idx, m.is_score_eligible.to_numpy().astype(bool), fill=False, dtype=bool)
    speed = scatter(n, idx, m.speed_kmh.to_numpy())
    flow = scatter(n, idx, m.flow_vph.to_numpy())
    occ = scatter(n, idx, m.occupancy.to_numpy())

    # --- Step 5a. validate the released flags against their published definitions ------------
    present = ~np.isnan(speed)
    sub = lambda a: a[idx]
    stats["flag_mismatch_is_imputed"] += int((m.is_imputed.to_numpy().astype(bool) != (sub(pct) < 100)).sum())
    stats["flag_mismatch_is_missing"] += int((m.is_missing.to_numpy().astype(bool) != (sub(pct) == 0)).sum())
    stats["flag_mismatch_eligible_vs_pct"] += int((sub(elig) != (sub(pct) >= ELIGIBLE_PCT)).sum())
    dens = m.density_occ_linear_vehpkm.to_numpy()
    ok = ~np.isnan(dens)
    stats["density_not_285x_occupancy"] += int((~np.isclose(dens[ok], DENSITY_PER_OCC * m.occupancy.to_numpy()[ok])).sum())

    # --- Step 2 applied: cell status from the templates ----------------------------------------
    lo, hi = np.searchsorted(t1_keys, [day_slot0 * KEY_BASE, (day_slot0 + STEPS) * KEY_BASE])
    day_keys = t1_keys[lo:hi]
    is_target = scatter(n, (day_keys // KEY_BASE - day_slot0) * n_links + day_keys % KEY_BASE, True, fill=False, dtype=bool)
    tod_grid = np.repeat(np.arange(STEPS), n_links)
    is_hist = np.isin(tod_grid + day_slot0, t2_hist)
    is_black = np.isin(tod_grid + day_slot0, t2_black)
    is_blanked = elig & ~present

    # --- Step 5b. every blank must be explained ------------------------------------------------
    stats["cells"] += n
    stats["t1_targets"] += int(is_target.sum())
    stats["blanked_eligible"] += int(is_blanked.sum())
    stats["target_but_visible"] += int((is_target & ~is_blanked).sum())
    stats["blanked_in_t2_blackout"] += int((is_blanked & is_black & ~is_target).sum())
    stats["blanked_unexplained"] += int((is_blanked & ~is_target & ~is_black).sum())
    stats["t2_history_cells"] += int(is_hist.sum())
    stats["t2_blackout_cells"] += int(is_black.sum())

    # --- Step 6. cleaning: tiers, strict values, anomaly flags --------------------------------
    tier = np.select([pct == 100, pct >= ELIGIBLE_PCT, pct > 0], ["full", "partial", "degraded"], "missing")
    for t in ("full", "partial", "degraded", "missing"):
        stats[f"tier_{t}"] += int((tier == t).sum())
    use = elig & present
    speed_c, flow_c, occ_c = (np.where(use, a, np.nan).astype(np.float32) for a in (speed, flow, occ))
    stats["clean_cells"] += int(use.sum())
    stats["degraded_with_values"] += int(((tier == "degraded") & present).sum())

    lanes = np.tile(st.lanes.to_numpy(np.float32), STEPS)
    ffs = np.tile(st.free_speed_kmh.to_numpy(np.float32), STEPS)
    cap = np.tile(st.capacity_vph.to_numpy(np.float32), STEPS)
    with np.errstate(invalid="ignore"):
        f_cap = present & (flow > cap)
        f_vceil = present & (speed >= SPEED_CEILING - 1e-6)
        f_oceil = present & (occ >= OCC_CEILING - 1e-6)
        S = speed.reshape(STEPS, n_links)
        d_in = np.full_like(S, np.nan); d_in[1:] = S[1:] - S[:-1]
        d_out = np.full_like(S, np.nan); d_out[:-1] = S[:-1] - S[1:]
        f_spike = ((np.abs(d_in) > SPIKE_KMH) & (np.abs(d_out) > SPIKE_KMH) & (np.sign(d_in) == np.sign(d_out))).ravel()
    for name, flag in (("flow_over_capacity", f_cap), ("speed_at_ceiling", f_vceil),
                       ("occ_at_ceiling", f_oceil), ("speed_spike", f_spike)):
        stats[f"flag_{name}"] += int(flag.sum())
    stats["visible_cells"] += int(present.sum())

    # Detector-bias diagnostic: speed minus the mean of its milepost neighbours, free flow only.
    Sc = speed_c.reshape(STEPS, n_links)
    ff = Sc > FREEFLOW_RATIO * ffs.reshape(STEPS, n_links)
    left = np.full_like(Sc, np.nan); left[:, 1:] = np.where(ff[:, :-1], Sc[:, :-1], np.nan)
    right = np.full_like(Sc, np.nan); right[:, :-1] = np.where(ff[:, 1:], Sc[:, 1:], np.nan)
    cnt = (~np.isnan(left)).astype(np.float32) + (~np.isnan(right))
    with np.errstate(invalid="ignore", divide="ignore"):
        nb = (np.nan_to_num(left) + np.nan_to_num(right)) / cnt
    diff = np.where(ff & (cnt > 0), Sc - nb, np.nan)
    good = ~np.isnan(diff)
    bias["sum"] += np.nansum(diff, axis=0)
    bias["n"] += good.sum(axis=0)

    # --- Step 4. ramps: validity rule, then aggregate onto the attached mainline link ---------
    ramp_cols = {}
    if ramp_path is None or not ramp_path.exists():  # keep the schema; every attached ramp is unavailable
        for prefix in ("on", "off"):
            n_att = np.tile(st[f"{prefix}_ramp_n"].to_numpy(), STEPS)
            ramp_cols[f"{prefix}_ramp_n"] = n_att
            ramp_cols[f"{prefix}_ramp_n_valid"] = np.zeros(n, np.int8)
            ramp_cols[f"{prefix}_ramp_flow"] = np.where(n_att == 0, 0.0, np.nan).astype(np.float32)
            ramp_cols[f"{prefix}_ramp_flow_partial"] = ramp_cols[f"{prefix}_ramp_flow"].copy()
    else:
        r = pd.read_parquet(ramp_path, columns=["timestamp", "ramp_link_id", "ramp_type", "flow_vph",
                                                "pct_observed", "is_score_eligible"])
        stats["ramp_rows"] += len(r)
        r = r.merge(amap[["ramp_link_id", "link_id"]], on="ramp_link_id", how="left")
        stats["ramp_rows_unattached"] += int(r.link_id.isna().sum())
        r = r[r.link_id.isin(pos_of)]
        valid = (r.is_score_eligible.astype(bool) & (r.pct_observed >= ELIGIBLE_PCT) & r.flow_vph.notna()).to_numpy()
        stats["ramp_rows_valid"] += int(valid.sum())
        stats["ramp_rows_degraded_with_value"] += int((~valid & r.flow_vph.notna().to_numpy()).sum())
        ridx = tod_of(r.timestamp).astype(np.int64) * n_links + r.link_id.map(pos_of).astype(np.int64).to_numpy()
        for typ, prefix in (("OR", "on"), ("FR", "off")):
            sel = (r.ramp_type == typ).to_numpy() & valid
            n_valid = np.bincount(ridx[sel], minlength=n).astype(np.int8)
            total = np.bincount(ridx[sel], weights=r.flow_vph.to_numpy()[sel], minlength=n).astype(np.float32)
            n_att = np.tile(st[f"{prefix}_ramp_n"].to_numpy(), STEPS)
            complete = n_valid == n_att
            ramp_cols[f"{prefix}_ramp_n"] = n_att
            ramp_cols[f"{prefix}_ramp_n_valid"] = n_valid
            # Unavailable ramp = missing evidence, never a measured zero.
            ramp_cols[f"{prefix}_ramp_flow"] = np.where(n_att == 0, 0.0, np.where(complete, total, np.nan)).astype(np.float32)
            ramp_cols[f"{prefix}_ramp_flow_partial"] = np.where((n_att > 0) & (n_valid == 0), np.nan, total).astype(np.float32)
            stats[f"{prefix}_ramp_cells_attached"] += int((n_att > 0).sum())
            stats[f"{prefix}_ramp_cells_complete"] += int(((n_att > 0) & complete).sum())

    # --- Step 7. derived fields and assembly --------------------------------------------------
    ts_text = scatter(n, idx, m.timestamp.to_numpy(), fill=None, dtype=object)
    if (ts_text == None).any():  # noqa: E711 - rebuild text for cells absent from the file
        miss = ts_text == None  # noqa: E711
        base = pd.Timestamp(date, tz="UTC") + pd.to_timedelta(tod_grid[miss] * 5, unit="min")
        ts_text[miss] = base.strftime("%Y-%m-%dT%H:%M:%SZ")
    ts = pd.Timestamp(date, tz="UTC") + pd.to_timedelta(tod_grid * 5, unit="min")
    weekday = pd.Timestamp(date).weekday()
    with np.errstate(invalid="ignore", divide="ignore"):
        out = pd.DataFrame({
            "panel": panel, "split": split, "date": date, "timestamp": ts_text, "ts": ts,
            "tod": tod_grid.astype(np.int16), "hour": (tod_grid // 12).astype(np.int8),
            "weekday": np.int8(weekday), "is_weekend": weekday >= 5,
            "station_id": np.tile(st.station_id.to_numpy(), STEPS), "link_id": np.tile(st.link_id.to_numpy(), STEPS),
            "pos": np.tile(st.pos.to_numpy(), STEPS), "mask_regime": regime,
            "pct_observed": pct.astype(np.int8), "quality_tier": tier, "is_score_eligible": elig,
            "is_t1_target": is_target, "is_t2_history": is_hist, "is_t2_blackout": is_black, "is_blanked": is_blanked,
            "speed_obs": speed, "flow_obs": flow, "occ_obs": occ,
            "speed_clean": speed_c, "flow_clean": flow_c, "occ_clean": occ_c,
            "flow_per_lane": (flow_c / lanes).astype(np.float32),
            "density_qv": (flow_c / speed_c).astype(np.float32),
            "speed_ratio": (speed_c / ffs).astype(np.float32),
            "flow_cap_ratio": (flow_c / cap).astype(np.float32),
            "is_queued": pd.array(np.where(np.isnan(speed_c), None, speed_c <= QUEUE_RATIO * ffs), dtype="boolean"),
        })
    if split == "train" and truth_path is None:  # keep the schema if a truth file were ever absent
        for c in ("speed_true", "flow_true", "occ_true"):
            out[c] = np.full(n, np.nan, np.float32)
        out["is_queued_true"] = pd.array([None] * n, dtype="boolean")
    if truth_path is not None:
        t = pd.read_parquet(truth_path, columns=["timestamp", "link_id", "speed_kmh", "flow_vph", "occupancy"])
        tp = t.link_id.map(pos_of)
        t = t[tp.notna()]
        tidx = tod_of(t.timestamp).astype(np.int64) * n_links + tp[tp.notna()].astype(np.int64).to_numpy()
        sp_t = scatter(n, tidx, t.speed_kmh.to_numpy())
        out["speed_true"] = sp_t
        out["flow_true"] = scatter(n, tidx, t.flow_vph.to_numpy())
        out["occ_true"] = scatter(n, tidx, t.occupancy.to_numpy())
        with np.errstate(invalid="ignore"):
            q = np.where(elig & ~np.isnan(sp_t), sp_t <= QUEUE_RATIO * ffs, None)
        out["is_queued_true"] = pd.array(q, dtype="boolean")
        # Step 5c. the masked layer must equal the truth wherever it shows a value.
        both = present & ~np.isnan(sp_t)
        stats["masked_vs_truth_mismatch"] += int((~np.isclose(speed[both], sp_t[both], atol=1e-4)).sum())
        stats["target_without_truth"] += int((is_target & np.isnan(sp_t)).sum())
        stats["queued_true_cells"] += int(np.nansum(q[elig & ~np.isnan(sp_t)].astype(float)))
        stats["queue_label_cells"] += int((elig & ~np.isnan(sp_t)).sum())
    out["flag_flow_over_capacity"] = f_cap
    out["flag_speed_at_ceiling"] = f_vceil
    out["flag_occ_at_ceiling"] = f_oceil
    out["flag_speed_spike"] = f_spike
    for name, values in ramp_cols.items():
        out[name] = values
    stats["days"] += 1
    stats[f"days_{regime}"] += 1
    return out


# ---------------------------------------------------------------------------
# Step 8. per corridor driver and export (逐走廊处理与导出)
# ---------------------------------------------------------------------------
def day_files(panel_dir: Path, split: str) -> list[tuple[str, str, Path]]:
    found = []
    for regime_dir in sorted((panel_dir / split / "mainline_states_masked").glob("mask_regime=*")):
        regime = regime_dir.name.split("=")[1]
        found += [(date_of(p), regime, p) for p in regime_dir.glob("*.parquet")]
    return sorted(found)


def process_panel(release: Path, out_root: Path, panel: str, splits: list[str], max_days: int | None) -> dict:
    panel_dir = release / "corridors" / panel
    first_day = sorted((panel_dir / "train" / "mainline_states").glob("*/*.parquet"))[0]
    st, amap, net_checks = build_link_table(panel_dir, first_day)
    pos_of = dict(zip(st.link_id, st.pos))
    bias = {"sum": np.zeros(len(st)), "n": np.zeros(len(st), dtype=np.int64)}
    report = {"network": net_checks}
    (out_root / panel).mkdir(parents=True, exist_ok=True)

    for split in splits:
        t0 = time.time()
        stats: Counter = Counter()
        t1_keys, t1_rows, t1_unknown = task1_target_keys(release, panel, split, pos_of)
        t2_hist, t2_black, n_windows = task2_slots(release, panel, split)
        stats["t1_template_rows"], stats["t1_template_unknown_link"], stats["t2_windows"] = t1_rows, t1_unknown, n_windows
        days = day_files(panel_dir, split)[:max_days] if max_days else day_files(panel_dir, split)
        truth = {date_of(p): p for p in (panel_dir / split / "mainline_states").glob("*/*.parquet")} if split == "train" else {}
        ramps = {date_of(p): p for p in (panel_dir / split / "ramp_states").glob("*/*.parquet")}
        path = out_root / panel / f"{split}.parquet"
        writer = None
        try:
            for i, (date, regime, mpath) in enumerate(days, 1):
                frame = process_day(panel, split, date, regime, mpath, truth.get(date) if split == "train" else None,
                                    ramps.get(date), st, amap, t1_keys, t2_hist, t2_black, stats, bias)
                if split == "train" and date not in truth:
                    stats["train_days_without_truth"] += 1
                if date not in ramps:
                    stats["days_without_ramp_file"] += 1
                table = pa.Table.from_pandas(frame, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(path, table.schema, compression="zstd")
                writer.write_table(table)
                if i % 30 == 0 or i == len(days):
                    print(f"  [{panel} {split}] {i}/{len(days)} days", flush=True)
        finally:
            if writer is not None:
                writer.close()
        if max_days is None:  # the template covers the full split; compare only on complete runs
            stats["t1_template_rows_not_matched"] = stats["t1_template_rows"] - stats["t1_targets"]
        stats["seconds"] = round(time.time() - t0, 1)
        report[split] = dict(stats)

    with np.errstate(invalid="ignore", divide="ignore"):
        st["freeflow_bias_kmh"] = (bias["sum"] / bias["n"]).astype(np.float32)
    st["freeflow_bias_n"] = bias["n"]
    st.to_parquet(out_root / panel / "links.parquet", index=False)
    b = st.dropna(subset=["freeflow_bias_kmh"])
    report["network"]["detectors_bias_over_3kmh"] = int((b.freeflow_bias_kmh.abs() > 3).sum())
    report["network"]["worst_bias_detectors"] = (b.reindex(b.freeflow_bias_kmh.abs().sort_values(ascending=False).index)
                                                 .head(5)[["link_id", "freeflow_bias_kmh"]].round(2).values.tolist())
    return report


def main() -> None:
    args = parse_args()
    release = args.release_root.resolve()
    if not (release / "corridors").is_dir():
        raise SystemExit(f"--release-root must be the kaggle_public folder (with corridors/ inside): {release}")
    panels_all = [p["corridor_id"] for p in
                  json.loads((release / "config" / "corridors.json").read_text(encoding="utf-8"))["panels"]]
    panels = args.panels or panels_all
    unknown = sorted(set(panels) - set(panels_all))
    if unknown:
        raise SystemExit(f"unknown panels: {unknown}")
    out_root = args.out.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    report_path = out_root / "cleaning_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
    for panel in panels:
        print(f"[{panel}]", flush=True)
        report[panel] = process_panel(release, out_root, panel, list(args.splits), args.max_days)
        report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")  # checkpoint per corridor

    rows = [{"panel": p, "section": s, "metric": k, "value": v}
            for p, sections in report.items() for s, metrics in sections.items() for k, v in metrics.items()]
    pd.DataFrame(rows).to_csv(out_root / "cleaning_report.csv", index=False)
    pd.DataFrame({"column": list(DATA_DICTIONARY), "description": list(DATA_DICTIONARY.values())}).to_csv(
        out_root / "data_dictionary.csv", index=False)

    print("\nSummary (cells in millions; checks should be 0):")
    for p in panels:
        for s in args.splits:
            r = report[p].get(s, {})
            if not r:
                continue
            print(f"  {p:11s} {s:10s} days={r.get('days', 0):3d} cells={r.get('cells', 0) / 1e6:6.2f} "
                  f"clean={r.get('clean_cells', 0) / max(r.get('cells', 1), 1):.1%} targets={r.get('t1_targets', 0):>9,} "
                  f"| dup={r.get('duplicate_keys', 0)} unexplained_blanks={r.get('blanked_unexplained', 0)} "
                  f"target_visible={r.get('target_but_visible', 0)} truth_mismatch={r.get('masked_vs_truth_mismatch', 0)} "
                  f"template_unmatched={r.get('t1_template_rows_not_matched', '-')}")
    print(f"\nWrote {out_root}")


if __name__ == "__main__":
    main()
