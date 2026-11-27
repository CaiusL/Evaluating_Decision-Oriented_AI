#!/usr/bin/env python3
"""Build 181-dim Behavior Cloning dataset from data/winner-builds-and-controls.json.

Uses the 12 train / 4 val winner-grouped split from reports/winner_data_audit.json so
no winner identity or match_id leaks across splits.

Key design choices:
  - Filters out in_pit == True frames (7,287 / 166,656) so BC supervises only on-track driving.
  - Reconstructs exact Frenet (s, n_lat, heading_error) and 24-point track preview_120 via
    TrackTable.project() and TrackTable.preview_24().
  - Normalizes steering_angle_rad / 0.312 -> [-1, 1] to match RaceEnv's steer command convention.
  - Sets obs_std = 1e6 (obs_mean = 0) on the 8 unlogged internal RaceEnv dimensions
    (vy, rpm, ers_soc_mj, fuel_kg, tyre_temp_c, tyre_wear, front_sliding, rear_sliding) so
    both BC training and closed-loop RaceEnv evaluation map them cleanly to ~0 without OOD drift.

Run with project virtualenv:
  .venv-decision/bin/python scripts/build_winner_bc_dataset.py
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from self_jev.sim.track import load_track

DEFAULT_WINNER_JSON = REPO_ROOT / "data" / "winner-builds-and-controls.json"
DEFAULT_AUDIT_JSON = REPO_ROOT / "reports" / "winner_data_audit.json"
DEFAULT_OUT_DIR = REPO_ROOT / "data" / "winner_bc"

STEER_NORM_RAD = 0.312
# Indices in 181-dim RaceEnv obs that are NOT logged in the public winner export:
# 2: vy, 8: rpm/10000, 11: ers_soc_mj, 12: fuel_kg, 15: tyre_temp_c/100,
# 16: tyre_wear, 17: front_sliding, 18: rear_sliding
UNLOGGED_OBS_INDICES = (2, 8, 11, 12, 15, 16, 17, 18)


def _wrap_angle(a: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(a), np.cos(a))


def build_entry_arrays(
    entry: dict[str, Any],
    ci: dict[str, int],
    track_cache: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    tr_name = entry["track"].lower()
    trk = track_cache[tr_name]
    rows = entry["control_trace"]
    n = len(rows)

    elapsed = np.array([float(r[ci["elapsed_s"]]) for r in rows], dtype=np.float32)
    lap = np.array([int(r[ci["lap"]]) for r in rows], dtype=np.int32)
    x_m = np.array([float(r[ci["x_m"]]) for r in rows], dtype=np.float32)
    y_m = np.array([float(r[ci["y_m"]]) for r in rows], dtype=np.float32)
    hdg = np.array([float(r[ci["heading_rad"]]) for r in rows], dtype=np.float32)
    speed = np.array([float(r[ci["speed_mps"]]) for r in rows], dtype=np.float32)
    wet = np.array([float(r[ci["track_wetness"]]) for r in rows], dtype=np.float32)
    rain = np.array([float(r[ci["rain_intensity"]]) for r in rows], dtype=np.float32)
    thr = np.array([float(r[ci["throttle"]]) for r in rows], dtype=np.float32)
    brk = np.array([float(r[ci["brake"]]) for r in rows], dtype=np.float32)
    steer_rad = np.array([float(r[ci["steering_angle_rad"]]) for r in rows], dtype=np.float32)
    gear = np.array([float(r[ci["gear"]]) for r in rows], dtype=np.float32)
    aero = np.array([float(r[ci["aero_mode"]]) for r in rows], dtype=np.float32)
    ers = np.array([int(r[ci["ers_mode"]]) for r in rows], dtype=np.int64)
    ovr = np.array([1.0 if r[ci["override_active"]] else 0.0 for r in rows], dtype=np.float32)
    aid = np.array([1.0 if r[ci["aid_active"]] else 0.0 for r in rows], dtype=np.float32)
    in_pit = np.array([bool(r[ci["in_pit"]]) for r in rows], dtype=bool)

    # Project full trajectory onto TrackTable centerline in one vectorized batch
    with torch.inference_mode():
        wx_t = torch.from_numpy(x_m)
        wy_t = torch.from_numpy(y_m)
        yaw_t = torch.from_numpy(hdg)
        s_t, n_lat_t, he_t, _ = trk.project(wx_t, wy_t, yaw_t, prev_idx=None)
        preview_120 = trk.preview_24(s_t).cpu().numpy().astype(np.float32)
        n_lat = n_lat_t.cpu().numpy().astype(np.float32)
        he = he_t.cpu().numpy().astype(np.float32)

    dt = np.diff(elapsed, prepend=elapsed[0] - (1.0 / 24.0))
    dt = np.clip(dt, 0.015, 0.10)
    dh = _wrap_angle(np.diff(hdg, prepend=hdg[0]))
    yaw_rate = np.clip(dh / dt, -3.5, 3.5).astype(np.float32)
    g_long = np.clip((np.diff(speed, prepend=speed[0]) / dt) / 9.81, -8.0, 4.0).astype(np.float32)
    g_lat = np.clip((speed * yaw_rate) / 9.81, -7.0, 7.0).astype(np.float32)

    steer_norm = np.clip(steer_rad / STEER_NORM_RAD, -1.0, 1.0).astype(np.float32)
    prev_thr = np.concatenate([[0.0], thr[:-1]]).astype(np.float32)
    prev_brk = np.concatenate([[0.0], brk[:-1]]).astype(np.float32)
    prev_steer = np.concatenate([[0.0], steer_norm[:-1]]).astype(np.float32)

    # Assemble 181-dim obs matching RaceEnv.get_obs(include_map_ext=False)
    obs = np.zeros((n, 181), dtype=np.float32)
    obs[:, 0] = speed
    obs[:, 1] = speed  # vx approx speed
    obs[:, 2] = 0.0    # vy unlogged
    obs[:, 3] = n_lat
    obs[:, 4] = he
    obs[:, 5] = yaw_rate
    obs[:, 6] = prev_steer
    obs[:, 7] = gear
    obs[:, 9] = prev_thr
    obs[:, 10] = prev_brk
    obs[:, 13] = g_lat
    obs[:, 14] = g_long
    obs[:, 19] = ((wet < 0.5) & (lap >= 2)).astype(np.float32)  # aid_avail (JSON laps start at 1)
    obs[:, 20] = (wet < 0.5).astype(np.float32)                 # ovr_avail
    obs[:, 21] = 1.0                                             # is_race
    obs[:, 22:142] = preview_120
    # rivals_36: solo-car empty pad [5.0, 0.0, 0.0, 0.0] x 9
    for r_i in range(9):
        obs[:, 142 + r_i * 4] = 5.0
    obs[:, 178] = wet
    obs[:, 179] = rain
    obs[:, 180] = (30.0 - 8.0 * wet) / 40.0

    cont = np.stack([thr, brk, steer_norm], axis=1).astype(np.float32)
    flags = np.stack([aero, ovr, aid], axis=1).astype(np.float32)

    # Filter out in_pit frames and first frame of race (lights-out standstill derivative artifact)
    keep = (~in_pit)
    keep[0] = False
    return obs[keep], cont[keep], ers[keep], flags[keep]


def main() -> None:
    ap = argparse.ArgumentParser(description="Build 181-dim winner BC dataset on 12/4 winner-grouped split.")
    ap.add_argument("--winner-json", type=Path, default=DEFAULT_WINNER_JSON)
    ap.add_argument("--audit-json", type=Path, default=DEFAULT_AUDIT_JSON)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = ap.parse_args()

    doc = json.loads(args.winner_json.read_text(encoding="utf-8"))
    audit = json.loads(args.audit_json.read_text(encoding="utf-8"))
    split = audit["recommended_shared_source_split"]
    train_keys = set(split["train_entries"])
    val_keys = set(split["val_entries"])
    ci = {c: i for i, c in enumerate(doc["control_trace_columns"])}

    track_cache = {t: load_track(t, device="cpu") for t in ("parkland", "highland", "crossover")}

    buckets: dict[str, dict[str, list[np.ndarray]]] = {
        "train": {"obs": [], "cont": [], "ers": [], "flags": []},
        "val": {"obs": [], "cont": [], "ers": [], "flags": []},
    }
    entry_counts: dict[str, int] = {}

    for key, e in sorted(doc["winners"].items()):
        part = "val" if key in val_keys else "train"
        obs, cont, ers, flags = build_entry_arrays(e, ci, track_cache)
        buckets[part]["obs"].append(obs)
        buckets[part]["cont"].append(cont)
        buckets[part]["ers"].append(ers)
        buckets[part]["flags"].append(flags)
        entry_counts[key] = int(len(obs))
        print(f"  [{part:5s}] {key:22s} -> {len(obs):6d} on-track frames")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tr_obs = np.concatenate(buckets["train"]["obs"], axis=0).astype(np.float32)
    tr_cont = np.concatenate(buckets["train"]["cont"], axis=0).astype(np.float32)
    tr_ers = np.concatenate(buckets["train"]["ers"], axis=0).astype(np.int64)
    tr_flags = np.concatenate(buckets["train"]["flags"], axis=0).astype(np.float32)

    va_obs = np.concatenate(buckets["val"]["obs"], axis=0).astype(np.float32)
    va_cont = np.concatenate(buckets["val"]["cont"], axis=0).astype(np.float32)
    va_ers = np.concatenate(buckets["val"]["ers"], axis=0).astype(np.int64)
    va_flags = np.concatenate(buckets["val"]["flags"], axis=0).astype(np.float32)

    obs_mean = tr_obs.mean(axis=0).astype(np.float32)
    obs_std = tr_obs.std(axis=0).astype(np.float32)
    obs_std = np.where(obs_std < 1e-4, 1.0, obs_std).astype(np.float32)
    # Neutralize unlogged internal dims so RaceEnv runtime values don't cause OOD drift
    for idx in UNLOGGED_OBS_INDICES:
        obs_mean[idx] = 0.0
        obs_std[idx] = 1e6

    np.savez_compressed(
        args.out_dir / "train.npz",
        obs=tr_obs, cont=tr_cont, ers=tr_ers, flags=tr_flags,
        obs_mean=obs_mean, obs_std=obs_std,
    )
    np.savez_compressed(
        args.out_dir / "val.npz",
        obs=va_obs, cont=va_cont, ers=va_ers, flags=va_flags,
    )

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_winner_json": "data/winner-builds-and-controls.json",
        "split_source": "reports/winner_data_audit.json",
        "steer_norm_rad": STEER_NORM_RAD,
        "unlogged_obs_indices_masked": list(UNLOGGED_OBS_INDICES),
        "train_entries": sorted(train_keys),
        "val_entries": sorted(val_keys),
        "train_rows": int(len(tr_obs)),
        "val_rows": int(len(va_obs)),
        "per_entry_on_track_rows": entry_counts,
    }
    (args.out_dir / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved winner BC dataset in {args.out_dir}: train={len(tr_obs)}, val={len(va_obs)}")


if __name__ == "__main__":
    main()
