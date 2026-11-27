#!/usr/bin/env python3
"""Closed-loop 4-lap simulator evaluation of all decision models using Pace Eval v2.

Evaluates every decision model found in reports/full_benchmark/benchmark_raw.jsonl
(kev-0.8b, qwen-0.8b-ll, qwen-0.8b-json, self-jev-minilm, car-build-nn, plus oracle)
paired with the frozen jev_rl driver on Suite 3 (suite3_car_build_test.jsonl, n=34)
for full 4-lap races in self_jev.sim.env.RaceEnv.

Scores each model with the auditable v2 Competitive Completion Rate contract from
scripts/eval_pace_accuracy.py against:
  1. External AGX Winner Reference (data/agx_winner_times.json)
  2. Same-Scenario Sim Oracle (fastest legal 4-lap build in that exact scenario)
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
for p in (str(REPO_ROOT), str(SCRIPTS_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import eval_pace_accuracy as ep
from self_jev.car_build_nn.builds import ABSTAIN_LABEL, BUILD_NAMES, get_candidate_builds
from self_jev.car_build_nn.dataset import _load_frozen_driver
from self_jev.sim.env import RaceEnv
from self_jev.sim.physics import PhysicsConfig
from self_jev.sim.residual import assist_ers_mode, combine_residual_actions

DEFAULT_SUITE3 = REPO_ROOT / "data" / "benchmarks" / "suite3_car_build_test.jsonl"
DEFAULT_RAW = REPO_ROOT / "reports" / "full_benchmark" / "benchmark_raw.jsonl"
DEFAULT_DRIVER = REPO_ROOT / "models" / "jev_rl_v7.mapdata.npz"
DEFAULT_WINNERS = REPO_ROOT / "data" / "agx_winner_times.json"
DEFAULT_OUT = REPO_ROOT / "reports" / "models_pace_v2_summary.json"


def infer_scenario_weather(wet_start: float, wet_end: float) -> dict[str, Any]:
    """Map continuous (wet_start, wet_end) scenario parameters to AGX weather regime."""
    delta = wet_end - wet_start
    if delta >= 0.20:
        regime = "Wetting"
    elif delta <= -0.20:
        regime = "Drying"
    else:
        avg = 0.5 * (wet_start + wet_end)
        if avg < 0.08:
            regime = "Dry"
        elif avg < 0.48:
            regime = "Damp"
        elif avg < 0.78:
            regime = "Wet"
        else:
            regime = "Saturated"
    return {
        "regime": regime,
        "source": "heuristic",
        "mapping_version": "sim-wetness-v1",
        "evidence": {"wet_start": wet_start, "wet_end": wet_end},
        "ambiguities": ["mapped_from_continuous_wetness"],
    }


def _extract_scenario(item: dict[str, Any]) -> dict[str, float]:
    f = item["features"]
    return {
        "rain_prob": float(f[4]),
        "wet_start": float(f[5]),
        "wet_end": float(f[6]),
        "track_temp_c": float(f[7]),
        "recent_off_track_rate": float(f[8]),
        "recent_wear_rate": float(f[9]),
        "forecast_confidence": float(f[10]),
        "sensor_missing_flag": float(f[11]),
    }


@torch.inference_mode()
def simulate_one_track_build(
    track_id: str,
    bname: str,
    scenarios: list[dict[str, float]],
    driver_path: Path,
    device_str: str = "cpu",
    laps_total: int = 4,
    max_steps: int = 30 * 700,
) -> list[dict[str, Any]]:
    torch.set_num_threads(1)
    device = torch.device(device_str)
    cal_path = REPO_ROOT / "self_jev" / "sim" / "calibrated_params.json"
    cfg = PhysicsConfig.from_json(cal_path) if cal_path.is_file() else PhysicsConfig()
    builds = get_candidate_builds()
    car = builds[bname]
    driver_model, obs_mean, obs_std, is_residual = _load_frozen_driver(driver_path, device)

    worlds = len(scenarios)
    env = RaceEnv(
        track_id=track_id,
        worlds=worlds,
        cars_per_world=1,
        laps_total=laps_total,
        phase="race",
        weather_regime="dry",
        car=car,
        cfg=cfg,
        device=device,
    )
    for wi, sc in enumerate(scenarios):
        env.w_state["wet_start"][wi] = sc["wet_start"]
        env.w_state["wet_end"][wi] = sc["wet_end"]
        env.w_state["ramp_start_s"][wi] = 40.0
        env.w_state["ramp_end_s"][wi] = 360.0
        env.w_state["wetness"][wi] = sc["wet_start"]
        env.w_state["rain_intensity"][wi] = sc["wet_start"] * 0.5
        env.w_state["track_temp_c"][wi] = sc["track_temp_c"]
        env.w_state["forecast_rain_prob"][wi] = sc["rain_prob"]
    env.reset_all()

    for step_i in range(max_steps):
        raw = env.get_obs(include_map_ext=False)
        x = torch.clamp(
            torch.nan_to_num((raw - obs_mean) / obs_std, nan=0.0, posinf=10.0, neginf=-10.0),
            -10.0,
            10.0,
        )
        mu, ers_logits, flag_logits, _ = driver_model(x)
        ers = torch.argmax(ers_logits, dim=-1)
        if is_residual:
            thr, brk, st_cmd = combine_residual_actions(raw, mu)
            ers = assist_ers_mode(raw, ers, thr, brk)
        else:
            thr = torch.clamp(mu[:, 0], 0.0, 1.0)
            brk = torch.clamp(mu[:, 1], 0.0, 1.0)
            st_cmd = torch.clamp(mu[:, 2], -1.0, 1.0)

        env.step({
            "throttle": thr,
            "brake": brk,
            "steer": st_cmd,
            "ers_mode": ers,
            "aero_mode": (flag_logits[:, 0] > 0.0).float(),
            "override": (flag_logits[:, 1] > 0.0).float(),
            "aid": (flag_logits[:, 2] > 0.0).float(),
            "pit_request": torch.zeros(env.N, device=device),
            "pit_compound": torch.ones(env.N, device=device, dtype=torch.long),
        })
        if (step_i + 1) % 64 == 0 and (env.status != 1).all():
            break

    rows = []
    for wi in range(worlds):
        st0 = int(env.status[wi].item())
        lap0 = int(env.lap[wi].item())
        ft0 = float(env.finish_t[wi].item())
        dist0 = float(env.dist[wi].item())
        rows.append({
            "status": st0,
            "finished": st0 == 2,
            "completed_laps": laps_total if st0 == 2 else max(0, lap0),
            "finish_time_s": round(ft0, 3) if st0 == 2 and ft0 > 0 else None,
            "dist_m": round(dist0, 1),
        })
    return rows


def _run_workers_parallel(
    by_track: dict[str, list[dict[str, Any]]],
    driver_path: Path,
    suite3_path: Path,
    required_laps: int,
    device_str: str,
    max_workers: int = 6,
) -> dict[tuple[str, str], dict[str, Any]]:
    """Run all (track, build) simulations in parallel via subprocess (avoids SemLock limits)."""
    jobs = [(tr, bname) for tr in sorted(by_track.keys()) for bname in BUILD_NAMES]
    pending = list(jobs)
    running: list[tuple[str, str, subprocess.Popen[str]]] = []
    sim_lookup: dict[tuple[str, str], dict[str, Any]] = {}

    script_path = str(Path(__file__).resolve())
    while pending or running:
        while pending and len(running) < max_workers:
            tr, bname = pending.pop(0)
            cmd = [
                sys.executable,
                script_path,
                "--worker",
                f"{tr}:{bname}",
                "--suite3",
                str(suite3_path),
                "--driver",
                str(driver_path),
                "--laps",
                str(required_laps),
                "--device",
                device_str,
            ]
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            running.append((tr, bname, proc))

        still_running: list[tuple[str, str, subprocess.Popen[str]]] = []
        for tr, bname, proc in running:
            ret = proc.poll()
            if ret is None:
                still_running.append((tr, bname, proc))
                continue
            out, err = proc.communicate()
            if ret != 0:
                raise RuntimeError(f"Worker {tr}:{bname} failed (exit {ret}): {err}")
            rows = json.loads(out)
            t_items = by_track[tr]
            for idx, it in enumerate(t_items):
                sim_lookup[(it["id"], bname)] = rows[idx]
        running = still_running
        if running:
            time.sleep(0.15)
    return sim_lookup


def run_eval(
    suite3_path: Path = DEFAULT_SUITE3,
    raw_path: Path = DEFAULT_RAW,
    driver_path: Path = DEFAULT_DRIVER,
    winners_path: Path = DEFAULT_WINNERS,
    out_path: Path | None = DEFAULT_OUT,
    device_str: str = "cpu",
    max_workers: int = 6,
) -> dict[str, Any]:
    w_doc = json.loads(winners_path.read_text(encoding="utf-8"))
    meta = w_doc.get("metadata") or {}
    required_laps = int(meta.get("required_laps", 4))
    ref_version = str(meta.get("reference_version", "unversioned"))
    winners = {tr.lower(): dict(d) for tr, d in w_doc["agx_winner_times_seconds"].items()}

    items = [json.loads(ln) for ln in suite3_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    raw_records = [json.loads(ln) for ln in raw_path.read_text(encoding="utf-8").splitlines() if ln.strip()]

    preds_by_model: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for r in raw_records:
        if r.get("suite") == "suite3_car_build" and r.get("split") == "test" and r.get("order") == "orig":
            preds_by_model[r["model"]][r["id"]] = r

    builds = get_candidate_builds()
    by_track: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for it in items:
        by_track[it["track"]].append(it)

    t0 = time.perf_counter()
    sim_lookup = _run_workers_parallel(
        by_track=by_track,
        driver_path=driver_path,
        suite3_path=suite3_path,
        required_laps=required_laps,
        device_str=device_str,
        max_workers=max_workers,
    )
    sim_wall_s = round(time.perf_counter() - t0, 2)

    # Diagnostic summary per build across all actionable items
    build_diag: dict[str, Any] = {}
    actionable_items = [it for it in items if not it.get("is_abstain_gold", False)]
    for bname in BUILD_NAMES:
        fin = [sim_lookup[(it["id"], bname)] for it in actionable_items if sim_lookup[(it["id"], bname)]["finished"]]
        times = sorted(float(r["finish_time_s"]) for r in fin if r["finish_time_s"] is not None)
        statuses = defaultdict(int)
        for it in actionable_items:
            statuses[str(sim_lookup[(it["id"], bname)]["status"])] += 1
        build_diag[bname] = {
            "finished_4lap_n": len(fin),
            "total_actionable_n": len(actionable_items),
            "median_finish_time_s": round(times[len(times) // 2], 2) if times else None,
            "min_finish_time_s": round(times[0], 2) if times else None,
            "status_counts": dict(statuses),
        }

    # Best legal 4-lap sim finish time per scenario (same-scenario sim oracle reference)
    sim_oracle_time: dict[str, float | None] = {}
    for it in items:
        is_unseen = it.get("bucket") == "unseen_lever"
        legal_times = []
        for bname in BUILD_NAMES:
            if is_unseen and builds[bname].start_compound == "soft":
                continue
            sr = sim_lookup[(it["id"], bname)]
            if sr["finished"] and sr["finish_time_s"] is not None:
                legal_times.append(float(sr["finish_time_s"]))
        sim_oracle_time[it["id"]] = min(legal_times) if legal_times else None

    model_order = sorted(preds_by_model.keys()) + ["oracle"]
    models_report: dict[str, Any] = {}

    for m in model_order:
        ext_entries: list[tuple[dict[str, Any], dict[str, Any]]] = []
        ora_entries: list[tuple[dict[str, Any], dict[str, Any]]] = []
        abstain_gold_n = 0
        safe_hold_n = 0
        false_abstain_n = 0
        rule_violation_n = 0
        exact_gold_match_n = 0
        total_items_n = 0

        for it in items:
            iid = it["id"]
            tr = it["track"].lower()
            sc = _extract_scenario(it)
            weather_info = infer_scenario_weather(sc["wet_start"], sc["wet_end"])
            is_unseen = it.get("bucket") == "unseen_lever"
            is_abstain_gold = bool(it.get("is_abstain_gold", False))

            if m == "oracle":
                pred_label = it["gold_label"]
            else:
                rec = preds_by_model[m].get(iid)
                if not rec:
                    continue
                pred_label = rec.get("pred_label")

            total_items_n += 1
            if pred_label == it["gold_label"]:
                exact_gold_match_n += 1

            if is_abstain_gold:
                abstain_gold_n += 1
                if pred_label == ABSTAIN_LABEL:
                    safe_hold_n += 1
                continue

            # Actionable 4-lap race scenario
            if pred_label == ABSTAIN_LABEL or pred_label not in builds:
                false_abstain_n += 1
                norm = {
                    "match_id": iid,
                    "track": tr,
                    "track_display": tr.capitalize(),
                    "classified": False,
                    "classification_source": "official",
                    "legal": True,
                    "terminal_failure": True,
                    "full_distance_completed": False,
                    "finish_time_s": None,
                    "timing_compatible": False,
                    "weather": weather_info,
                    "observed_laps": 0,
                    "rank": None,
                }
            else:
                sr = sim_lookup[(iid, pred_label)]
                rule_legal = not (is_unseen and builds[pred_label].start_compound == "soft")
                if not rule_legal:
                    rule_violation_n += 1
                finished = bool(sr["finished"])
                ft = sr["finish_time_s"]
                norm = {
                    "match_id": iid,
                    "track": tr,
                    "track_display": tr.capitalize(),
                    "classified": finished and rule_legal,
                    "classification_source": "official",
                    "legal": rule_legal,
                    "terminal_failure": not finished,
                    "full_distance_completed": finished and sr["completed_laps"] == required_laps,
                    "finish_time_s": ft,
                    "timing_compatible": finished and ft is not None,
                    "weather": weather_info,
                    "observed_laps": sr["completed_laps"],
                    "rank": None,
                }

            # 1. External AGX reference
            regime = weather_info["regime"]
            ref_t = winners.get(tr, {}).get(regime) if regime else None
            if ref_t is None:
                ext_sc = ep.score_normalized_run(
                    norm, reference_applicable=False, reference_time_s=None, unavailable_reason=f"null_reference:{tr}/{regime}"
                )
            else:
                ext_sc = ep.score_normalized_run(norm, reference_applicable=True, reference_time_s=ref_t)
            ext_entries.append((norm, ext_sc))

            # 2. Same-scenario sim oracle reference
            ora_t = sim_oracle_time[iid]
            if ora_t is None:
                ora_sc = ep.score_normalized_run(
                    norm, reference_applicable=False, reference_time_s=None, unavailable_reason="no_legal_sim_finisher"
                )
            else:
                ora_sc = ep.score_normalized_run(norm, reference_applicable=True, reference_time_s=ora_t)
            ora_entries.append((norm, ora_sc))

        models_report[m] = {
            "total_items_n": total_items_n,
            "suite3_exact_accuracy": round(exact_gold_match_n / total_items_n, 4) if total_items_n else None,
            "actionable_n": len(ext_entries),
            "abstain_gold_n": abstain_gold_n,
            "safe_hold_rate": round(safe_hold_n / abstain_gold_n, 4) if abstain_gold_n else None,
            "false_abstain_n": false_abstain_n,
            "rule_violation_n": rule_violation_n,
            "external_agx_reference": ep.summarize_scored_slice(ext_entries),
            "same_scenario_sim_oracle": ep.summarize_scored_slice(ora_entries),
        }

    report = {
        "evaluator_version": ep.EVALUATOR_VERSION,
        "reference_version": ref_version,
        "required_laps": required_laps,
        "driver_weights": str(driver_path.name),
        "suite3_items": len(items),
        "sim_wall_s": sim_wall_s,
        "build_diagnostics": build_diag,
        "models": models_report,
    }
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="Closed-loop 4-lap Pace Eval v2 across all decision models.")
    ap.add_argument("--worker", default=None, help="Internal worker mode: <track>:<build>")
    ap.add_argument("--suite3", type=Path, default=DEFAULT_SUITE3)
    ap.add_argument("--driver", type=Path, default=DEFAULT_DRIVER)
    ap.add_argument("--laps", type=int, default=4)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    if args.worker:
        tr, bname = args.worker.split(":", 1)
        items = [json.loads(ln) for ln in args.suite3.read_text(encoding="utf-8").splitlines() if ln.strip()]
        t_items = [it for it in items if it["track"] == tr]
        scs = [_extract_scenario(it) for it in t_items]
        rows = simulate_one_track_build(
            track_id=tr,
            bname=bname,
            scenarios=scs,
            driver_path=args.driver,
            device_str=args.device,
            laps_total=args.laps,
        )
        print(json.dumps(rows))
        return

    rep = run_eval(suite3_path=args.suite3, driver_path=args.driver, device_str=args.device, max_workers=args.workers)
    print(f"Simulated 4-lap Suite 3 races in {rep['sim_wall_s']}s (driver={rep['driver_weights']})\n")
    print("=== BUILD 4-LAP COMPLETION DIAGNOSTICS ===")
    for b, d in rep["build_diagnostics"].items():
        print(f"  {b:20s} | fin4={d['finished_4lap_n']:2d}/{d['total_actionable_n']:2d} | med={d['median_finish_time_s']}s min={d['min_finish_time_s']}s | statuses={d['status_counts']}")
    print("\n=== EXTERNAL AGX WINNER REFERENCE ===")
    for m, d in rep["models"].items():
        ov = d["external_agx_reference"]
        r = ov["competitive_completion_rate_by_threshold"]
        print(
            f"  {m:16s} | acc={d['suite3_exact_accuracy']} | elig={ov['reference_applicable_n']:2d}/{d['actionable_n']:2d} "
            f"comp4={ov['valid_full_distance_gap_n']:2d} | @0%={r['le_0pct']} @5%={r['le_5pct']} @10%={r['le_10pct']} @20%={r['le_20pct']} "
            f"| med_gap={ov['median_gap_s_among_comparable_finishes']}s ({ov['median_gap_pct_among_comparable_finishes']}%) "
            f"| safe_hold={d['safe_hold_rate']} rule_viol={d['rule_violation_n']}"
        )
    print("\n=== SAME-SCENARIO SIM ORACLE REFERENCE ===")
    for m, d in rep["models"].items():
        ov = d["same_scenario_sim_oracle"]
        r = ov["competitive_completion_rate_by_threshold"]
        print(
            f"  {m:16s} | elig={ov['reference_applicable_n']:2d}/{d['actionable_n']:2d} "
            f"comp4={ov['valid_full_distance_gap_n']:2d} | @0%={r['le_0pct']} @1%={r['le_1pct']} @5%={r['le_5pct']} @10%={r['le_10pct']} "
            f"| med_gap={ov['median_gap_s_among_comparable_finishes']}s ({ov['median_gap_pct_among_comparable_finishes']}%)"
        )


if __name__ == "__main__":
    main()
