#!/usr/bin/env python3
"""Milestone 1 & 2 Diagnostics (Fully batched snapshots + 2-way Qwen pool):
  1. M2QwenLoRARunner scoring contract verification (3-option pit & 4-option build/pace masking).
  2. Exact RaceEnv snapshot/restore replay invariance check.
  3. 27-scenario prompt-bridge diagnostic (9 cells x 3 fresh dev seeds [3101, 3102, 3103]).
  4. Milestone 2 counterfactual action audit (builds + batched pace + batched pit + 9-cell lookup).
"""
from __future__ import annotations

import argparse
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
SCRIPTS_DIR = REPO_ROOT / "scripts"
SRC_DIR = REPO_ROOT / "decision_eval" / "src"
for p in (str(REPO_ROOT), str(SCRIPTS_DIR), str(SRC_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import eval_race_engineer_v2 as re_eval  # noqa: E402
from decision_eval import paths  # noqa: E402
from decision_eval.racing_contract import (  # noqa: E402
    BUILD_OPTIONS,
    PACE_OPTIONS,
    PIT_OPTIONS,
    VERIFIED_BUILD_DESCS,
    VERIFIED_PACE_DESCS,
    VERIFIED_PIT_DESCS,
    action_spec_hash,
    generate_scenarios_for_seeds,
    render_obs_text,
)
from decision_eval.racing_rollouts import (  # noqa: E402
    evaluate_pace_counterfactuals_batched,
    evaluate_pit_counterfactuals_batched,
    init_env_for_scenarios,
    load_frozen_sim_assets,
    rank_candidate_outcomes,
    run_closed_loop_policy,
    snapshot_env,
    step_driver_batch,
    verify_snapshot_replay_invariance,
)
from decision_eval.runners import M2QwenLoRARunner  # noqa: E402

OUT_PATH = REPO_ROOT / "reports" / "racing_engineer_v1" / "milestone1_diagnostics.json"
BRIDGE_SEEDS = [3101, 3102, 3103]
AUDIT_SEEDS = [3201, 3202, 3203, 3204, 3205]


def paired_summary(base_rows: list[dict[str, Any]], cand_rows: list[dict[str, Any]], tol: float = 0.50) -> dict[str, Any]:
    rescued = regressed = both_fin = both_fail = wins = ties_fin = losses = 0
    deltas: list[float] = []
    for b, c in zip(base_rows, cand_rows):
        bf, cf = b["finished"], c["finished"]
        if cf and not bf:
            rescued += 1
            wins += 1
        elif bf and not cf:
            regressed += 1
            losses += 1
        elif not bf and not cf:
            both_fail += 1
        else:
            both_fin += 1
            d = float(c["finish_time_s"]) - float(b["finish_time_s"])
            deltas.append(d)
            if d < -tol:
                wins += 1
            elif d > tol:
                losses += 1
            else:
                ties_fin += 1
    ds = sorted(deltas)
    return {
        "n": len(base_rows),
        "base_finishes": sum(1 for r in base_rows if r["finished"]),
        "cand_finishes": sum(1 for r in cand_rows if r["finished"]),
        "rescued": rescued,
        "regressed": regressed,
        "common_finishers_n": both_fin,
        "mean_paired_delta_s": round(float(np.mean(ds)), 3) if ds else None,
        "median_paired_delta_s": round(float(np.median(ds)), 3) if ds else None,
        "wtl": {"wins": wins, "ties_finished": ties_fin, "ties_both_failed": both_fail, "losses": losses},
    }


def summarize_arm(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ts = [float(r["finish_time_s"]) for r in rows if r["finished"] and r["finish_time_s"] is not None]
    return {
        "n": len(rows),
        "finishes": sum(1 for r in rows if r["finished"]),
        "avg_finish_time_s": round(float(np.mean(ts)), 2) if ts else None,
        "pits_total": sum(r["pit_stops"] for r in rows),
        "delegated_calls": sum(r["delegated_calls"] for r in rows),
        "illegal_proposed": sum(r["illegal_proposed"] for r in rows),
        "build_counts": dict(Counter(r["chosen_build"] for r in rows)),
        "mode_counts": dict(Counter(m for r in rows for m in r["mode_history"])),
    }


def run_bridge_worker(arm: str, track: str) -> list[dict[str, Any]]:
    torch.set_num_threads(4 if arm.startswith("m2_") else 1)
    device = torch.device("cpu")
    driver_model, obs_mean, obs_std, is_residual, cfg, builds = load_frozen_sim_assets(device=device)
    scs = [sc for sc in generate_scenarios_for_seeds(BRIDGE_SEEDS) if sc["track"] == track]

    if arm == "driver_only":
        fn = lambda kind, obs, opts, descs: ({"build": "dry_balanced", "pace": "balanced", "pit": "stay_out"}[kind], True, 1.0)
        pits_on = False
    elif arm == "track_rule":
        fn = lambda kind, obs, opts, descs: (
            ("dry_balanced" if kind == "build" else (("balanced" if obs["track"] == "parkland" else "push_max") if kind == "pace" else "stay_out")),
            True,
            1.0,
        )
        pits_on = False
    elif arm == "m2_current_prompt":
        runner = M2QwenLoRARunner(paths.M2_QWEN_LORA_S42_DIR, device="cpu")
        pits_on = True

        def fn(kind: str, obs: dict[str, Any], opts: list[str], descs: list[str]):
            if kind == "build":
                st = (
                    f"Track: {obs['track']} | Weather forecast: regime={obs['weather_regime']}, "
                    f"wetness_start={obs['forecast_wet_start']:.2f}, wetness_end={obs['forecast_wet_end']:.2f}, "
                    f"rain_prob={obs['forecast_rain_prob']:.2f}, track_temp_c={obs['track_temp_c']:.1f}."
                )
                r = runner.score_decision(st, "Which car build should we start this 4-lap race with?", opts, re_eval.BUILD_DESCS)
            elif kind == "pace":
                st = (
                    f"Live Race Telemetry — Track: {obs['track']}, Lap {obs['laps_completed'] + 1}/{obs['laps_total']}, "
                    f"elapsed={obs['elapsed_s']:.1f}s, wetness={obs['wetness']:.2f} (forecast_end={obs['forecast_wet_end']:.2f}), "
                    f"tyre_compound={obs['tyre_compound']}, tyre_wear={obs['tyre_wear']:.2f}, ers_soc={obs['ers_soc_mj']:.2f} MJ, "
                    f"pit_stops={obs['pit_stops']}."
                )
                r = runner.score_decision(st, "Which driving pace & ERS mode should the RL driver use for this lap?", opts, re_eval.PACE_DESCS)
            else:
                st = (
                    f"Live Race Telemetry — Track: {obs['track']}, Lap {obs['laps_completed'] + 1}/{obs['laps_total']}, "
                    f"elapsed={obs['elapsed_s']:.1f}s, wetness={obs['wetness']:.2f} (forecast_end={obs['forecast_wet_end']:.2f}), "
                    f"tyre_compound={obs['tyre_compound']}, tyre_wear={obs['tyre_wear']:.2f}, ers_soc={obs['ers_soc_mj']:.2f} MJ, "
                    f"pit_stops={obs['pit_stops']}."
                )
                r = runner.score_decision(st, "Should the car stay out or pit for fresh tyres at the end of this lap?", opts, re_eval.PIT_DESCS)
            return r["choice"], bool(r["parse_ok"]), float(r["confidence"])
    elif arm == "m2_verified_runbook":
        runner = M2QwenLoRARunner(paths.M2_QWEN_LORA_S42_DIR, device="cpu")
        pits_on = True
        q_map = {
            "build": "Select the pre-race car build to maximize 4-lap completion and minimize race time.",
            "pace": "Select the driving pace & ERS mode for this lap.",
            "pit": "Select whether to stay out or pit at the end of this lap.",
        }

        def fn(kind: str, obs: dict[str, Any], opts: list[str], descs: list[str]):
            st = render_obs_text(obs, include_contract=True)
            r = runner.score_decision(st, q_map[kind], opts, descs)
            return r["choice"], bool(r["parse_ok"]), float(r["confidence"])
    else:
        raise ValueError(f"Unknown arm: {arm}")

    return run_closed_loop_policy(
        scs, fn, driver_model, obs_mean, obs_std, is_residual, cfg, builds, device, pits_enabled=pits_on
    )


def run_audit_worker(track: str) -> dict[str, Any]:
    """Fully batched audit: 4 build batches + 1 snapshot capture batch + 1 pace CF batch + 1 pit CF batch."""
    torch.set_num_threads(1)
    device = torch.device("cpu")
    driver_model, obs_mean, obs_std, is_residual, cfg, builds = load_frozen_sim_assets(device=device)
    audit_scs = [sc for sc in generate_scenarios_for_seeds(AUDIT_SEEDS) if sc["track"] == track]

    # 1. Build audit across 3 weather cells x 5 seeds = 15 scenarios per build (4 vectorized passes)
    build_rows: dict[str, list[dict[str, Any]]] = {}
    for b_name in BUILD_OPTIONS:
        fn = lambda kind, obs, opts, descs, bn=b_name: (bn if kind == "build" else ("balanced" if kind == "pace" else "stay_out"), True, 1.0)
        build_rows[b_name] = run_closed_loop_policy(
            audit_scs, fn, driver_model, obs_mean, obs_std, is_residual, cfg, builds, device, pits_enabled=False
        )

    # 2. Capture snapshots for 3 weather cells x 2 seeds (3201, 3202) in ONE batched pass
    snap_scs = [sc for sc in audit_scs if sc["seed"] in AUDIT_SEEDS[:2]]
    cont_mode = "balanced" if track == "parkland" else "push_max"
    env = init_env_for_scenarios(track, snap_scs, builds["dry_balanced"], cfg, device, laps_total=4)
    w_n = len(snap_scs)
    captured = [set() for _ in range(w_n)]
    pace_items: list[dict[str, Any]] = []
    pit_items: list[dict[str, Any]] = []

    for step_i in range(30 * 800):
        for wi, sc in enumerate(snap_scs):
            if int(env.status[wi].item()) != 1:
                continue
            clap = int(env.lap[wi].item())
            if clap in (0, 1, 2) and clap not in captured[wi]:
                captured[wi].add(clap)
                snap = snapshot_env(env, wi)
                pace_items.append({"scenario_id": sc["scenario_id"], "lap": clap, "snap": snap})
                if clap in (1, 2):
                    pit_items.append({
                        "scenario_id": sc["scenario_id"],
                        "lap": clap,
                        "tyre_wear": round(float(env.tyre_wear[wi].item()), 3),
                        "snap": snap,
                    })
        step_driver_batch(env, [cont_mode] * w_n, [False] * w_n, [0] * w_n, driver_model, obs_mean, obs_std, is_residual)
        if (step_i + 1) % 32 == 0 and (env.status != 1).all():
            break

    # 3. ONE batched RaceEnv call for ALL pace counterfactuals + ONE for ALL pit counterfactuals
    pace_outcomes_all = evaluate_pace_counterfactuals_batched(
        track, builds["dry_balanced"], pace_items, cont_mode, driver_model, obs_mean, obs_std, is_residual, cfg, device
    )
    pit_outcomes_all = evaluate_pit_counterfactuals_batched(
        track, builds["dry_balanced"], pit_items, cont_mode, driver_model, obs_mean, obs_std, is_residual, cfg, device
    )

    pace_rows: list[dict[str, Any]] = []
    for it, p_out in zip(pace_items, pace_outcomes_all):
        p_rank = rank_candidate_outcomes(p_out)
        bal_o = next(o for o in p_out if o["option"] == "balanced")
        best_o = p_out[p_rank["gold_index"]]
        pace_rows.append({
            "scenario_id": it["scenario_id"],
            "lap": it["lap"],
            "continuation_mode": cont_mode,
            "gold_option": p_rank["gold_option_id"],
            "acceptable": p_rank["acceptable_option_ids"],
            "status": p_rank["status"],
            "gain_vs_balanced_s": (
                round(float(bal_o["finish_time_s"]) - float(best_o["finish_time_s"]), 3)
                if (bal_o["finished"] and best_o["finished"])
                else None
            ),
            "rescued_vs_balanced": bool(best_o["finished"] and not bal_o["finished"]),
            "outcomes": p_out,
        })

    pit_rows: list[dict[str, Any]] = []
    for it, pit_out in zip(pit_items, pit_outcomes_all):
        pit_rank = rank_candidate_outcomes(pit_out)
        stay_o = next(o for o in pit_out if o["option"] == "stay_out")
        best_p = pit_out[pit_rank["gold_index"]]
        pit_rows.append({
            "scenario_id": it["scenario_id"],
            "lap": it["lap"],
            "tyre_wear": it["tyre_wear"],
            "gold_option": pit_rank["gold_option_id"],
            "pit_helped": bool(
                pit_rank["gold_option_id"] != "stay_out"
                and (
                    (best_p["finished"] and not stay_o["finished"])
                    or (
                        best_p["finished"]
                        and stay_o["finished"]
                        and float(best_p["finish_time_s"]) < float(stay_o["finish_time_s"]) - 0.5
                    )
                )
            ),
            "outcomes": pit_out,
        })

    return {"build_rows": build_rows, "pace_rows": pace_rows, "pit_rows": pit_rows}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bridge-worker", default=None, help="<arm>:<track>")
    ap.add_argument("--audit-worker", default=None, help="<track>")
    ap.add_argument("--out-file", type=Path, default=None, help="Write worker JSON to file to avoid pipe buffer deadlock")
    args = ap.parse_args()

    if args.bridge_worker:
        arm, tr = args.bridge_worker.split(":", 1)
        payload = json.dumps(run_bridge_worker(arm, tr))
        if args.out_file:
            args.out_file.write_text(payload, encoding="utf-8")
        else:
            print(payload)
        return
    if args.audit_worker:
        payload = json.dumps(run_audit_worker(args.audit_worker))
        if args.out_file:
            args.out_file.write_text(payload, encoding="utf-8")
        else:
            print(payload)
        return

    t0 = time.perf_counter()
    torch.set_num_threads(2)
    device = torch.device("cpu")
    driver_model, obs_mean, obs_std, is_residual, cfg, builds = load_frozen_sim_assets(device=device)

    print("[1/4] Auditing M2QwenLoRARunner scoring contract...", flush=True)
    runner = M2QwenLoRARunner(paths.M2_QWEN_LORA_S42_DIR, device="cpu")
    res4 = runner.score_decision("Track: highland, Dry", "Pick build", BUILD_OPTIONS, VERIFIED_BUILD_DESCS)
    res3 = runner.score_decision("Track: highland, Lap 2/4, tyre_wear=0.25", "Pit?", PIT_OPTIONS, VERIFIED_PIT_DESCS)
    contract_check = {
        "model_loaded": str(paths.M2_QWEN_LORA_S42_DIR.name),
        "letter_ids_verified": runner.letter_ids,
        "temperature": runner.temperature,
        "four_opt_probs_len": len(res4["probabilities"]),
        "four_opt_sum": round(sum(res4["probabilities"]), 6),
        "four_opt_latency_ms": res4["latency_ms"],
        "three_opt_probs_len": len(res3["probabilities"]),
        "three_opt_sum": round(sum(res3["probabilities"]), 6),
        "three_opt_latency_ms": res3["latency_ms"],
        "passed": (
            len(res4["probabilities"]) == 4
            and abs(sum(res4["probabilities"]) - 1.0) < 1e-4
            and len(res3["probabilities"]) == 3
            and abs(sum(res3["probabilities"]) - 1.0) < 1e-4
        ),
    }
    del runner
    print(f"  Runner check passed={contract_check['passed']} (4-opt {res4['latency_ms']}ms, 3-opt {res3['latency_ms']}ms)", flush=True)

    print("[2/4] Verifying RaceEnv snapshot/restore replay invariance...", flush=True)
    bridge_scenarios = generate_scenarios_for_seeds(BRIDGE_SEEDS)
    inv_check = verify_snapshot_replay_invariance(
        bridge_scenarios[0], driver_model, obs_mean, obs_std, is_residual, cfg, builds, device
    )
    print(f"  Snapshot invariance passed={inv_check['passed']} (diff={inv_check['abs_diff_s']}s, ticks={inv_check['tick_diff']})", flush=True)
    assert inv_check["passed"], f"Snapshot replay invariance failed: {inv_check}"

    print("[3/4 & 4/4] Phase A: RaceEnv refs + batched audits (9 concurrent), Phase B: Qwen bridge (2 concurrent)...", flush=True)
    tmp_dir = OUT_PATH.parent / "_tmp_workers"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    script_path = str(Path(__file__).resolve())
    light_jobs: list[tuple[str, list[str], Path]] = []
    for tr in ("parkland", "highland", "crossover"):
        fp = tmp_dir / f"audit_{tr}.json"
        light_jobs.append((f"audit:{tr}", [sys.executable, script_path, "--audit-worker", tr, "--out-file", str(fp)], fp))
    for arm in ("driver_only", "track_rule"):
        for tr in ("parkland", "highland", "crossover"):
            fp = tmp_dir / f"bridge_{arm}_{tr}.json"
            light_jobs.append((f"bridge:{arm}:{tr}", [sys.executable, script_path, "--bridge-worker", f"{arm}:{tr}", "--out-file", str(fp)], fp))

    heavy_jobs: list[tuple[str, list[str], Path]] = []
    for arm in ("m2_current_prompt", "m2_verified_runbook"):
        for tr in ("parkland", "highland", "crossover"):
            fp = tmp_dir / f"bridge_{arm}_{tr}.json"
            heavy_jobs.append((f"bridge:{arm}:{tr}", [sys.executable, script_path, "--bridge-worker", f"{arm}:{tr}", "--out-file", str(fp)], fp))

    out_map: dict[str, Any] = {}

    def _run_pool(job_list: list[tuple[str, list[str], Path]], max_conc: int) -> None:
        pending: list[tuple[str, list[str], Path]] = []
        for tag, cmd, fp in job_list:
            if fp.is_file() and fp.stat().st_size > 2:
                out_map[tag] = json.loads(fp.read_text(encoding="utf-8"))
                print(f"  reused cached {tag}", flush=True)
            else:
                pending.append((tag, cmd, fp))
        running: list[tuple[str, subprocess.Popen[str], Path]] = []
        while pending or running:
            while pending and len(running) < max_conc:
                tag, cmd, fp = pending.pop(0)
                running.append((tag, subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True), fp))
            still: list[tuple[str, subprocess.Popen[str], Path]] = []
            for tag, proc, fp in running:
                ret = proc.poll()
                if ret is None:
                    still.append((tag, proc, fp))
                    continue
                _, err = proc.communicate()
                if ret != 0:
                    raise RuntimeError(f"Worker {tag} failed ({ret}): {err}")
                out_map[tag] = json.loads(fp.read_text(encoding="utf-8"))
                print(f"  completed {tag} ({time.perf_counter()-t0:.1f}s elapsed)", flush=True)
            running = still
            if running:
                time.sleep(0.2)

    _run_pool(light_jobs, max_conc=9)
    _run_pool(heavy_jobs, max_conc=2)

    arms_rows: dict[str, list[dict[str, Any]]] = {}
    for arm in ("driver_only", "track_rule", "m2_current_prompt", "m2_verified_runbook"):
        rows = [r for tr in ("parkland", "highland", "crossover") for r in out_map[f"bridge:{arm}:{tr}"]]
        arms_rows[arm] = rows
        s = summarize_arm(rows)
        print(
            f"  [BRIDGE] {arm:20s} | fin={s['finishes']}/{s['n']} | avg={s['avg_finish_time_s']}s | "
            f"pits={s['pits_total']} | modes={s['mode_counts']}",
            flush=True,
        )

    build_by_cell: dict[str, dict[str, dict[str, Any]]] = {}
    pace_audit_rows: list[dict[str, Any]] = []
    pit_audit_rows: list[dict[str, Any]] = []
    for tr in ("parkland", "highland", "crossover"):
        ad = out_map[f"audit:{tr}"]
        for b_name, rows in ad["build_rows"].items():
            for r in rows:
                cid = r["cell_id"]
                rec = build_by_cell.setdefault(cid, {}).setdefault(b_name, {"finishes": 0, "n": 0, "times": []})
                rec["n"] += 1
                if r["finished"] and r["finish_time_s"] is not None:
                    rec["finishes"] += 1
                    rec["times"].append(float(r["finish_time_s"]))
        pace_audit_rows.extend(ad["pace_rows"])
        pit_audit_rows.extend(ad["pit_rows"])

    lookup_9cell_build: dict[str, str] = {}
    build_audit_summary: dict[str, Any] = {}
    for cid, bmap in build_by_cell.items():
        ranked = [
            {
                "build": b_name,
                "finishes": st["finishes"],
                "n": st["n"],
                "mean_time_s": round(float(np.mean(st["times"])), 2) if st["times"] else None,
            }
            for b_name, st in bmap.items()
        ]
        ranked.sort(key=lambda x: (-x["finishes"], x["mean_time_s"] if x["mean_time_s"] is not None else 1e9))
        lookup_9cell_build[cid] = ranked[0]["build"]
        build_audit_summary[cid] = {"best_build": ranked[0]["build"], "candidates": ranked}

    pit_helped_count = sum(1 for r in pit_audit_rows if r["pit_helped"])
    pace_gains = [r["gain_vs_balanced_s"] for r in pace_audit_rows if r["gain_vs_balanced_s"] is not None]
    pace_gold_counts = dict(Counter(r["gold_option"] for r in pace_audit_rows))
    print(
        f"  [AUDIT] builds_9cell={lookup_9cell_build}\n"
        f"  [AUDIT] pace_gold={pace_gold_counts}, mean_gain_vs_bal={round(float(np.mean(pace_gains)), 2) if pace_gains else None}s | "
        f"pit_helped={pit_helped_count}/{len(pit_audit_rows)}",
        flush=True,
    )

    report = {
        "milestone": "racing_engineer_v1_milestone1_2_diagnostics",
        "action_spec_hash": action_spec_hash(),
        "wall_s": round(time.perf_counter() - t0, 1),
        "runner_contract_check": contract_check,
        "snapshot_replay_invariance": inv_check,
        "prompt_bridge_27": {
            "seeds": BRIDGE_SEEDS,
            "arms": {k: summarize_arm(v) for k, v in arms_rows.items()},
            "paired_vs_driver_only": {
                k: paired_summary(arms_rows["driver_only"], v)
                for k, v in arms_rows.items()
                if k != "driver_only"
            },
            "paired_verified_vs_current_prompt": paired_summary(
                arms_rows["m2_current_prompt"], arms_rows["m2_verified_runbook"]
            ),
        },
        "counterfactual_action_audit": {
            "build_audit_seeds": AUDIT_SEEDS,
            "lookup_9cell_build": lookup_9cell_build,
            "build_by_cell": build_audit_summary,
            "pace_snapshots_n": len(pace_audit_rows),
            "pace_gold_counts": pace_gold_counts,
            "pace_mean_gain_vs_balanced_s": round(float(np.mean(pace_gains)), 3) if pace_gains else None,
            "pace_rescued_vs_balanced_n": sum(1 for r in pace_audit_rows if r["rescued_vs_balanced"]),
            "pit_snapshots_n": len(pit_audit_rows),
            "pit_helped_n": pit_helped_count,
            "enable_pits_in_stage_b": (pit_helped_count > 0),
            "pace_rows": pace_audit_rows,
            "pit_rows": pit_audit_rows,
        },
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Saved Milestone 1 & 2 diagnostics -> {OUT_PATH} ({report['wall_s']}s)")


if __name__ == "__main__":
    main()
