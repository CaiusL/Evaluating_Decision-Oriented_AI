#!/usr/bin/env python3
"""Milestone 3 & 4: Racing-Only Dataset Generation & Lineage Split Manifest (racing_engineer_v1).

Generates matched JSONL (for Racing Qwen3.5-0.8B LoRA) and structured numeric features
(for NN-3 MLP) from complete remaining-race rollouts in RaceEnv with strict race-lineage splits:
  - train: 180 parent races (9 cells x 20 seeds: 3201..3220)
  - dev:    90 parent races (9 cells x 10 seeds: 3301..3310)
  - cal:    90 parent races (9 cells x 10 seeds: 3401..3410)
  - confirm (manifest only, never rolled out here): 180 parent races (3501..3520)
  - historical_regression (manifest only): 45 parent races (2001..2005)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "decision_eval" / "src"
for p in (str(REPO_ROOT), str(SRC_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from decision_eval.racing_contract import (  # noqa: E402
    BUILD_OPTIONS,
    OBJECTIVE_VERSION,
    PACE_OPTIONS,
    SCHEMA_VERSION,
    VERIFIED_BUILD_DESCS,
    VERIFIED_PACE_DESCS,
    action_spec_hash,
    build_live_lap_obs,
    build_pre_race_obs,
    extract_build_features,
    extract_pace_features,
    generate_scenarios_for_seeds,
    render_obs_text,
)
from decision_eval.racing_rollouts import (  # noqa: E402
    DEFAULT_DRIVER,
    evaluate_pace_counterfactuals_batched,
    init_env_for_scenarios,
    load_frozen_sim_assets,
    rank_candidate_outcomes,
    run_closed_loop_policy,
    snapshot_env,
    step_driver_batch,
)
from self_jev.sim.design import COMPOUNDS  # noqa: E402

DATA_DIR = REPO_ROOT / "data" / "racing_v1"

EXCLUDED_PRIOR_SEEDS = [101, 202, 303, 404, 505, 1001, 1002, 1003, 1004, 1005, 3101, 3102, 3103]
HISTORICAL_SEEDS = [2001, 2002, 2003, 2004, 2005]
TRAIN_SEEDS = list(range(3201, 3221))   # 20 per cell = 180 parent races
DEV_SEEDS = list(range(3301, 3311))     # 10 per cell = 90 parent races
CAL_SEEDS = list(range(3401, 3411))     # 10 per cell = 90 parent races
CONFIRM_SEEDS = list(range(3501, 3521)) # 20 per cell = 180 parent races


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def build_split_manifest() -> dict[str, Any]:
    sets = {
        "excluded_prior": set(EXCLUDED_PRIOR_SEEDS),
        "historical_regression": set(HISTORICAL_SEEDS),
        "train": set(TRAIN_SEEDS),
        "dev": set(DEV_SEEDS),
        "cal": set(CAL_SEEDS),
        "confirm": set(CONFIRM_SEEDS),
    }
    names = list(sets.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            overlap = sets[names[i]] & sets[names[j]]
            assert not overlap, f"Seed overlap between {names[i]} and {names[j]}: {overlap}"

    return {
        "schema_version": SCHEMA_VERSION,
        "objective_version": OBJECTIVE_VERSION,
        "action_spec_hash": action_spec_hash(),
        "driver_weights": str(DEFAULT_DRIVER.name),
        "driver_sha256": sha256_file(DEFAULT_DRIVER),
        "lineage_rule": "All builds, lap snapshots, and counterfactual rollouts stay strictly within their parent episode_seed split.",
        "splits": {
            "train": {"seeds": TRAIN_SEEDS, "parent_races": len(TRAIN_SEEDS) * 9},
            "dev": {"seeds": DEV_SEEDS, "parent_races": len(DEV_SEEDS) * 9},
            "cal": {"seeds": CAL_SEEDS, "parent_races": len(CAL_SEEDS) * 9},
            "confirm": {"seeds": CONFIRM_SEEDS, "parent_races": len(CONFIRM_SEEDS) * 9},
            "historical_regression": {"seeds": HISTORICAL_SEEDS, "parent_races": len(HISTORICAL_SEEDS) * 9},
        },
        "excluded_prior_seeds": EXCLUDED_PRIOR_SEEDS,
    }


def generate_split_track_worker(split_name: str, track_id: str, seeds: list[int]) -> list[dict[str, Any]]:
    torch.set_num_threads(1)
    device = torch.device("cpu")
    driver_model, obs_mean, obs_std, is_residual, cfg, builds = load_frozen_sim_assets(device=device)
    scs = [sc for sc in generate_scenarios_for_seeds(seeds) if sc["track"] == track_id]
    cont_mode = "balanced" if track_id == "parkland" else "push_max"
    driver_hash = sha256_file(DEFAULT_DRIVER)
    spec_hash = action_spec_hash()

    records: list[dict[str, Any]] = []

    # 1. Stage A — Pre-race build supervision (all 4 builds rolled out under track_rule continuation + stay_out)
    build_outcomes_by_sc: dict[str, list[dict[str, Any]]] = {sc["scenario_id"]: [] for sc in scs}
    for b_name in BUILD_OPTIONS:
        fn = lambda kind, obs, opts, descs, bn=b_name, cm=cont_mode: (
            bn if kind == "build" else (cm if kind == "pace" else "stay_out"),
            True,
            1.0,
        )
        b_rows = run_closed_loop_policy(
            scs, fn, driver_model, obs_mean, obs_std, is_residual, cfg, builds, device, pits_enabled=False
        )
        for r in b_rows:
            build_outcomes_by_sc[r["scenario_id"]].append({
                "option": b_name,
                "finished": r["finished"],
                "status": r["status"],
                "finish_time_s": r["finish_time_s"],
                "completed_laps": r["completed_laps"],
                "dist_m": r["dist_m"],
            })

    best_build_for_sc: dict[str, str] = {}
    for sc in scs:
        sid = sc["scenario_id"]
        outcomes = build_outcomes_by_sc[sid]
        ranked = rank_candidate_outcomes(outcomes, time_tol_s=0.50)
        best_build_for_sc[sid] = ranked["gold_option_id"]
        obs = build_pre_race_obs(sc, laps_total=4)
        records.append({
            "schema_version": SCHEMA_VERSION,
            "id": f"{split_name}_{sid}_build",
            "source_id": sid,
            "snapshot_id": f"{sid}_pre_race",
            "split": split_name,
            "decision_type": "build",
            "context": render_obs_text(obs, include_contract=True),
            "question": "Select the pre-race car build to maximize 4-lap completion and minimize race time.",
            "options": list(BUILD_OPTIONS),
            "option_descriptions": list(VERIFIED_BUILD_DESCS),
            "gold_index": ranked["gold_index"],
            "gold_option_id": ranked["gold_option_id"],
            "acceptable_option_ids": ranked["acceptable_option_ids"],
            "telemetry": obs,
            "label_info": {
                "status": ranked["status"],
                "objective_version": OBJECTIVE_VERSION,
                "continuation_policy_id": f"track_rule_{cont_mode}_stay_out",
                "outcomes": outcomes,
            },
            "provenance": {
                "driver_hash": driver_hash,
                "action_spec_hash": spec_hash,
                "episode_seed": sc["seed"],
                "cell_id": sc["cell_id"],
            },
        })

    # 2. Stage B — Per-lap pace supervision: capture snapshots along reference build (dry_balanced)
    #    in ONE vectorized pass, then evaluate all 4 pace counterfactuals in batched chunks!
    env = init_env_for_scenarios(track_id, scs, builds["dry_balanced"], cfg, device, laps_total=4)
    w_n = len(scs)
    captured = [set() for _ in range(w_n)]
    prev_lap_start_t = [0.0] * w_n
    last_lap_time: list[float | None] = [None] * w_n
    off_ticks = [0] * w_n
    grazes = [0] * w_n
    snap_items: list[dict[str, Any]] = []

    for step_i in range(30 * 800):
        for wi, sc in enumerate(scs):
            if int(env.status[wi].item()) != 1:
                continue
            clap = int(env.lap[wi].item())
            if clap in (0, 1, 2, 3) and clap not in captured[wi]:
                captured[wi].add(clap)
                t_now = float(env.t_race[wi].item())
                if clap > 0:
                    last_lap_time[wi] = t_now - prev_lap_start_t[wi]
                    prev_lap_start_t[wi] = t_now
                obs_p = build_live_lap_obs(
                    sc=sc,
                    chosen_build="dry_balanced",
                    current_mode="balanced" if clap == 0 else cont_mode,
                    lap_completed=clap,
                    laps_total=4,
                    elapsed_s=t_now,
                    wetness=float(env.wetness[wi].item()),
                    rain_intensity=float(env.rain_intensity[wi].item()),
                    track_temp_c=float(env.track_temp_c[wi].item()),
                    tyre_compound=COMPOUNDS[int(env.compound[wi].item())],
                    tyre_wear=float(env.tyre_wear[wi].item()),
                    tyre_temp_c=float(env.tyre_temp_c[wi].item()),
                    ers_soc_mj=float(env.ers_soc_mj[wi].item()),
                    fuel_kg=float(env.fuel_kg[wi].item()),
                    pit_stops=int(env.pit_stops[wi].item()),
                    off_track_ticks_so_far=off_ticks[wi],
                    wall_grazes_so_far=grazes[wi],
                    last_lap_time_s=last_lap_time[wi],
                    decision_type="pace",
                )
                snap_items.append({
                    "sc": sc,
                    "lap": clap,
                    "obs": obs_p,
                    "snap": snapshot_env(env, wi),
                })
        step_driver_batch(env, [cont_mode] * w_n, [False] * w_n, [0] * w_n, driver_model, obs_mean, obs_std, is_residual)
        for wi in range(w_n):
            if int(env.status[wi].item()) == 1:
                if int(env.off_track[wi].item()) > 0:
                    off_ticks[wi] += 1
                if int(env.wall_graze[wi].item()) > 0:
                    grazes[wi] += 1
        if (step_i + 1) % 32 == 0 and (env.status != 1).all():
            break

    # Evaluate pace counterfactuals in chunks of 48 snapshots (= 192 worlds per RaceEnv batch)
    chunk_size = 48
    for start_i in range(0, len(snap_items), chunk_size):
        chunk = snap_items[start_i : start_i + chunk_size]
        cf_outcomes = evaluate_pace_counterfactuals_batched(
            track_id, builds["dry_balanced"], chunk, cont_mode, driver_model, obs_mean, obs_std, is_residual, cfg, device
        )
        for it, outcomes in zip(chunk, cf_outcomes):
            sc = it["sc"]
            sid = sc["scenario_id"]
            clap = it["lap"]
            obs_p = it["obs"]
            ranked = rank_candidate_outcomes(outcomes, time_tol_s=0.50)
            records.append({
                "schema_version": SCHEMA_VERSION,
                "id": f"{split_name}_{sid}_lap{clap}_pace",
                "source_id": sid,
                "snapshot_id": f"{sid}_lap{clap}",
                "split": split_name,
                "decision_type": "pace",
                "context": render_obs_text(obs_p, include_contract=True),
                "question": "Select the driving pace & ERS mode for this lap.",
                "options": list(PACE_OPTIONS),
                "option_descriptions": list(VERIFIED_PACE_DESCS),
                "gold_index": ranked["gold_index"],
                "gold_option_id": ranked["gold_option_id"],
                "acceptable_option_ids": ranked["acceptable_option_ids"],
                "telemetry": obs_p,
                "label_info": {
                    "status": ranked["status"],
                    "objective_version": OBJECTIVE_VERSION,
                    "continuation_policy_id": f"track_rule_{cont_mode}_stay_out",
                    "outcomes": outcomes,
                },
                "provenance": {
                    "driver_hash": driver_hash,
                    "action_spec_hash": spec_hash,
                    "episode_seed": sc["seed"],
                    "cell_id": sc["cell_id"],
                },
            })
    return records


def save_nn3_features_npz(all_splits: dict[str, list[dict[str, Any]]], out_path: Path) -> None:
    arrays: dict[str, np.ndarray] = {}
    for sp, rows in all_splits.items():
        b_rows = [r for r in rows if r["decision_type"] == "build" and r["label_info"]["status"] != "all_failed"]
        p_rows = [r for r in rows if r["decision_type"] == "pace" and r["label_info"]["status"] != "all_failed"]

        arrays[f"{sp}_build_X"] = np.stack([extract_build_features(r["telemetry"]) for r in b_rows], axis=0)
        arrays[f"{sp}_build_y"] = np.array([int(r["gold_index"]) for r in b_rows], dtype=np.int64)
        b_mask = np.zeros((len(b_rows), len(BUILD_OPTIONS)), dtype=np.float32)
        for i, r in enumerate(b_rows):
            for opt in r["acceptable_option_ids"]:
                b_mask[i, BUILD_OPTIONS.index(opt)] = 1.0
        arrays[f"{sp}_build_acc_mask"] = b_mask

        arrays[f"{sp}_pace_X"] = np.stack([extract_pace_features(r["telemetry"]) for r in p_rows], axis=0)
        arrays[f"{sp}_pace_y"] = np.array([int(r["gold_index"]) for r in p_rows], dtype=np.int64)
        p_mask = np.zeros((len(p_rows), len(PACE_OPTIONS)), dtype=np.float32)
        for i, r in enumerate(p_rows):
            for opt in r["acceptable_option_ids"]:
                p_mask[i, PACE_OPTIONS.index(opt)] = 1.0
        arrays[f"{sp}_pace_acc_mask"] = p_mask

    np.savez(out_path, **arrays)


def compute_train_9cell_lookup(train_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute 9-cell lookup baseline strictly from train split build outcomes."""
    by_cell: dict[str, dict[str, dict[str, Any]]] = {}
    for r in train_rows:
        if r["decision_type"] != "build":
            continue
        cid = r["provenance"]["cell_id"]
        for o in r["label_info"]["outcomes"]:
            rec = by_cell.setdefault(cid, {}).setdefault(o["option"], {"finishes": 0, "n": 0, "times": []})
            rec["n"] += 1
            if o["finished"] and o["finish_time_s"] is not None:
                rec["finishes"] += 1
                rec["times"].append(float(o["finish_time_s"]))

    lookup_build: dict[str, str] = {}
    detail: dict[str, Any] = {}
    for cid, bmap in sorted(by_cell.items()):
        ranked = [
            {
                "build": b,
                "finishes": d["finishes"],
                "n": d["n"],
                "mean_time_s": round(float(np.mean(d["times"])), 2) if d["times"] else None,
            }
            for b, d in bmap.items()
        ]
        ranked.sort(key=lambda x: (-x["finishes"], x["mean_time_s"] if x["mean_time_s"] is not None else 1e9))
        lookup_build[cid] = ranked[0]["build"]
        detail[cid] = ranked
    return {"lookup_9cell_build": lookup_build, "by_cell": detail}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", default=None, help="<split>:<track>")
    ap.add_argument("--out-file", type=Path, default=None)
    args = ap.parse_args()

    split_seeds = {"train": TRAIN_SEEDS, "dev": DEV_SEEDS, "cal": CAL_SEEDS}
    if args.worker:
        sp, tr = args.worker.split(":", 1)
        recs = generate_split_track_worker(sp, tr, split_seeds[sp])
        payload = json.dumps(recs)
        if args.out_file:
            args.out_file.write_text(payload, encoding="utf-8")
        else:
            print(payload)
        return

    t0 = time.perf_counter()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tmp_dir = DATA_DIR / "_tmp_workers"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    manifest = build_split_manifest()
    (DATA_DIR / "split_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"[Dataset v1] Wrote split_manifest.json (zero overlap verified across all 6 seed groups).", flush=True)

    script_path = str(Path(__file__).resolve())
    jobs: list[tuple[str, list[str], Path]] = []
    for sp in ("train", "dev", "cal"):
        for tr in ("parkland", "highland", "crossover"):
            fp = tmp_dir / f"{sp}_{tr}.json"
            jobs.append((f"{sp}:{tr}", [sys.executable, script_path, "--worker", f"{sp}:{tr}", "--out-file", str(fp)], fp))

    pending = list(jobs)
    running: list[tuple[str, subprocess.Popen[str], Path]] = []
    out_map: dict[str, list[dict[str, Any]]] = {}
    while pending or running:
        while pending and len(running) < 9:
            tag, cmd, fp = pending.pop(0)
            running.append((tag, subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True), fp))
        still = []
        for tag, proc, fp in running:
            ret = proc.poll()
            if ret is None:
                still.append((tag, proc, fp))
                continue
            _, err = proc.communicate()
            if ret != 0:
                raise RuntimeError(f"Dataset worker {tag} failed ({ret}): {err}")
            out_map[tag] = json.loads(fp.read_text(encoding="utf-8"))
            print(f"  completed {tag}: {len(out_map[tag])} records ({time.perf_counter()-t0:.1f}s elapsed)", flush=True)
        running = still
        if running:
            time.sleep(0.2)

    all_splits: dict[str, list[dict[str, Any]]] = {}
    stats: dict[str, Any] = {}
    for sp in ("train", "dev", "cal"):
        rows = [r for tr in ("parkland", "highland", "crossover") for r in out_map[f"{sp}:{tr}"]]
        all_splits[sp] = rows
        jsonl_path = DATA_DIR / f"{sp}.jsonl"
        jsonl_path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
        stats[sp] = {
            "total": len(rows),
            "by_type": dict(Counter(r["decision_type"] for r in rows)),
            "by_status": dict(Counter(r["label_info"]["status"] for r in rows)),
            "build_gold": dict(Counter(r["gold_option_id"] for r in rows if r["decision_type"] == "build")),
            "pace_gold": dict(Counter(r["gold_option_id"] for r in rows if r["decision_type"] == "pace")),
            "sha256": sha256_file(jsonl_path),
        }

    save_nn3_features_npz(all_splits, DATA_DIR / "features_nn3.npz")
    lookup_9cell = compute_train_9cell_lookup(all_splits["train"])
    (DATA_DIR / "lookup_9cell_train.json").write_text(json.dumps(lookup_9cell, indent=2) + "\n", encoding="utf-8")

    summary = {
        "schema_version": SCHEMA_VERSION,
        "wall_s": round(time.perf_counter() - t0, 1),
        "stats": stats,
        "lookup_9cell_build": lookup_9cell["lookup_9cell_build"],
    }
    (DATA_DIR / "dataset_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"\n[Dataset v1 DONE in {summary['wall_s']}s] Saved train/dev/cal JSONL + features_nn3.npz + lookup_9cell_train.json")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
