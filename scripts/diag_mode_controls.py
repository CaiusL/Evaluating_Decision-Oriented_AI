#!/usr/bin/env python3
"""Diagnostic control sweep for the 4-lap closed-loop Decision Model + Driver interface.

Runs two mandatory checks before any in-race model training:
  1. Neutral-adapter parity check (dry_balanced + balanced + stay_out) vs driver_only.
  2. Constant driving-mode sweep (push_max, balanced, conserve_tyres, wet_caution)
     with fixed build (dry_balanced) and no pits (stay_out) across all 9 canonical
     scenarios to measure whether the current mode modifiers create useful speed/safety
     tradeoffs on jev_rl_v7.mapdata.npz.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
for p in (str(REPO_ROOT), str(SCRIPTS_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import eval_race_engineer_v2 as re_eval
from self_jev.car_build_nn.builds import get_candidate_builds
from self_jev.car_build_nn.dataset import _load_frozen_driver
from self_jev.sim.physics import PhysicsConfig

DEFAULT_OUT = REPO_ROOT / "reports" / "mode_control_diagnostics.json"
CONTROL_MODES = ["push_max", "balanced", "conserve_tyres", "wet_caution"]


class FixedModeAdapter:
    """Deterministic adapter forcing fixed build, fixed driving mode, and stay_out."""

    def __init__(self, mode: str, build: str = "dry_balanced") -> None:
        self.mode = mode
        self.build = build

    def decide(
        self,
        kind: str,
        state: str,
        question: str,
        options: list[str],
        option_descriptions: list[str],
        telemetry: dict[str, Any],
    ) -> str:
        if kind == "build":
            return self.build
        if kind == "pace":
            return self.mode
        return "stay_out"


def run_control_sweep(
    driver_path: Path = re_eval.DEFAULT_DRIVER,
    winners_path: Path = re_eval.DEFAULT_WINNERS,
    out_path: Path = DEFAULT_OUT,
    max_workers: int = 6,
) -> dict[str, Any]:
    w_doc = json.loads(winners_path.read_text(encoding="utf-8"))
    winners = {tr.lower(): dict(d) for tr, d in w_doc["agx_winner_times_seconds"].items()}

    t0 = time.perf_counter()
    jobs = [(mode, sc["id"]) for mode in CONTROL_MODES for sc in re_eval.CANONICAL_SCENARIOS]
    script_path = str(Path(__file__).resolve())
    pending = list(jobs)
    running: list[tuple[str, str, subprocess.Popen[str]]] = []
    results_map: dict[tuple[str, str], dict[str, Any]] = {}

    while pending or running:
        while pending and len(running) < max_workers:
            mode, sid = pending.pop(0)
            cmd = [sys.executable, script_path, "--worker", f"{mode}:{sid}", "--driver", str(driver_path)]
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            running.append((mode, sid, proc))
        still: list[tuple[str, str, subprocess.Popen[str]]] = []
        for mode, sid, proc in running:
            ret = proc.poll()
            if ret is None:
                still.append((mode, sid, proc))
                continue
            out, err = proc.communicate()
            if ret != 0:
                raise RuntimeError(f"Worker {mode}:{sid} failed ({ret}): {err}")
            results_map[(mode, sid)] = json.loads(out)
        running = still
        if running:
            time.sleep(0.15)

    # Compare against saved driver_only baseline for neutral-adapter parity
    saved_re_path = REPO_ROOT / "reports" / "race_engineer_v2_summary.json"
    saved_driver_only: dict[str, float | None] = {}
    if saved_re_path.is_file():
        srep = json.loads(saved_re_path.read_text(encoding="utf-8"))
        d_runs = (srep.get("models") or {}).get("driver_only", {}).get("runs", [])
        for r in d_runs:
            saved_driver_only[r["scenario_id"]] = r.get("finish_time_s")

    balanced_by_sc = {
        sc["id"]: results_map[("balanced", sc["id"])]["finish_time_s"]
        for sc in re_eval.CANONICAL_SCENARIOS
    }

    # Neutral-adapter parity check
    parity_diffs = []
    for sc in re_eval.CANONICAL_SCENARIOS:
        sid = sc["id"]
        t_bal = balanced_by_sc.get(sid)
        t_saved = saved_driver_only.get(sid)
        if t_bal is not None and t_saved is not None:
            parity_diffs.append(round(abs(float(t_bal) - float(t_saved)), 4))
    max_parity_diff = max(parity_diffs) if parity_diffs else None

    modes_summary: dict[str, Any] = {}
    for mode in CONTROL_MODES:
        runs = [results_map[(mode, sc["id"])] for sc in re_eval.CANONICAL_SCENARIOS]
        fin_ts = [float(r["finish_time_s"]) for r in runs if r["finished"] and r["finish_time_s"] is not None]
        paired_deltas = []
        wins = ties = losses = 0
        agx_gaps = []
        status_counts: dict[str, int] = {}

        for r in runs:
            st_k = str(r["status"])
            status_counts[st_k] = status_counts.get(st_k, 0) + 1
            sid = r["scenario_id"]
            bt = balanced_by_sc.get(sid)
            rt = r["finish_time_s"]
            if r["finished"] and rt is not None:
                ref_t = winners.get(r["track"].lower(), {}).get(r["weather_regime"])
                if ref_t is not None:
                    agx_gaps.append(round(float(rt) - float(ref_t), 3))
            if r["finished"] and bt is not None and rt is not None:
                d = round(float(rt) - float(bt), 3)
                paired_deltas.append(d)
                if d < -0.5:
                    wins += 1
                elif d > 0.5:
                    losses += 1
                else:
                    ties += 1
            elif r["finished"] and bt is None:
                wins += 1
            elif (not r["finished"]) and bt is not None:
                losses += 1
            else:
                ties += 1

        modes_summary[mode] = {
            "scenarios_n": len(runs),
            "finishes_4lap_n": len(fin_ts),
            "avg_finish_time_s_finishers_only": round(sum(fin_ts) / len(fin_ts), 2) if fin_ts else None,
            "paired_n_vs_balanced": len(paired_deltas),
            "paired_mean_delta_vs_balanced_s": round(sum(paired_deltas) / len(paired_deltas), 2) if paired_deltas else None,
            "wtl_vs_balanced": {"wins": wins, "ties": ties, "losses": losses},
            "median_agx_gap_s": round(sorted(agx_gaps)[len(agx_gaps) // 2], 2) if agx_gaps else None,
            "status_counts": status_counts,
            "runs": runs,
        }

    # Per-scenario oracle across the 4 fixed modes (does ANY mode beat balanced?)
    per_scenario_best: list[dict[str, Any]] = []
    for sc in re_eval.CANONICAL_SCENARIOS:
        sid = sc["id"]
        bal_t = balanced_by_sc.get(sid)
        cands = []
        for mode in CONTROL_MODES:
            r = results_map[(mode, sid)]
            if r["finished"] and r["finish_time_s"] is not None:
                cands.append((float(r["finish_time_s"]), mode))
        best_t, best_m = min(cands) if cands else (None, None)
        gain_s = round(best_t - bal_t, 2) if (best_t is not None and bal_t is not None) else None
        per_scenario_best.append({
            "scenario_id": sid,
            "balanced_t": bal_t,
            "best_mode": best_m,
            "best_t": best_t,
            "gain_vs_balanced_s": gain_s,
        })

    report = {
        "driver_weights": str(driver_path.name),
        "fixed_build": "dry_balanced",
        "fixed_pit_policy": "stay_out",
        "scenarios_n": len(re_eval.CANONICAL_SCENARIOS),
        "wall_s": round(time.perf_counter() - t0, 2),
        "neutral_adapter_parity": {
            "compared_scenarios_n": len(parity_diffs),
            "max_abs_diff_s_vs_saved_driver_only": max_parity_diff,
            "passed_within_0_1s": (max_parity_diff is not None and max_parity_diff <= 0.1),
        },
        "modes": modes_summary,
        "per_scenario_best_mode": per_scenario_best,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="4-mode diagnostic control sweep & neutral-adapter parity check.")
    ap.add_argument("--worker", default=None, help="Internal worker: <mode>:<scenario_id>")
    ap.add_argument("--driver", type=Path, default=re_eval.DEFAULT_DRIVER)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    if args.worker:
        torch.set_num_threads(1)
        mode, sid = args.worker.split(":", 1)
        sc = next(s for s in re_eval.CANONICAL_SCENARIOS if s["id"] == sid)
        device = torch.device("cpu")
        cal_path = REPO_ROOT / "self_jev" / "sim" / "calibrated_params.json"
        cfg = PhysicsConfig.from_json(cal_path) if cal_path.is_file() else PhysicsConfig()
        builds = get_candidate_builds()
        driver_model, obs_mean, obs_std, is_residual = _load_frozen_driver(args.driver, device)
        row = re_eval.run_closed_loop_race(
            sc, FixedModeAdapter(mode), driver_model, obs_mean, obs_std, is_residual, cfg, builds, device, laps_total=4
        )
        print(json.dumps(row))
        return

    rep = run_control_sweep(driver_path=args.driver, max_workers=args.workers)
    print(f"Finished 4-mode control sweep in {rep['wall_s']}s")
    print("Neutral-adapter parity:", rep["neutral_adapter_parity"])
    print("\n=== FIXED-MODE CONTROL SWEEP (build=dry_balanced, pit=stay_out) ===")
    for mode, d in rep["modes"].items():
        wtl = d["wtl_vs_balanced"]
        print(
            f"  {mode:15s} | fin4={d['finishes_4lap_n']}/9 | avg={d['avg_finish_time_s_finishers_only']}s "
            f"| paired_d={d['paired_mean_delta_vs_balanced_s']}s (n={d['paired_n_vs_balanced']}) "
            f"| W-T-L={wtl['wins']}-{wtl['ties']}-{wtl['losses']} | statuses={d['status_counts']}"
        )
    print("\n=== PER-SCENARIO BEST MODE VS BALANCED ===")
    for r in rep["per_scenario_best_mode"]:
        print(f"  {r['scenario_id']:20s} | balanced={r['balanced_t']}s | best={r['best_mode']} ({r['best_t']}s) | gain={r['gain_vs_balanced_s']}s")


if __name__ == "__main__":
    main()
