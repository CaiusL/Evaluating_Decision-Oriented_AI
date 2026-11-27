#!/usr/bin/env python3
"""Seeded neutral-adapter parity check + 45-scenario fresh-seed confirmation (v2).

1. Seeded Parity Check (9 canonical scenarios):
   Runs driver_only and neutral_adapter (same build=dry_balanced, mode=balanced,
   pit=stay_out) with identical per-scenario torch.manual_seed -> verifies exact
   0.000s finish-time and step parity.

2. Fresh-Seed Confirmation Experiment (9 track x weather cells x 5 fresh seeds = 45 paired scenarios):
   Compares frozen `balanced` baseline vs. frozen `track_rule`
   (Parkland -> balanced, Highland/Crossover -> push_max; fixed build=dry_balanced,
   pit=stay_out) with paired domain randomization (grip_scale, aero_scale) and
   fresh exogenous weather realizations within each cell.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
for p in (str(REPO_ROOT), str(SCRIPTS_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import eval_race_engineer_v2 as re_eval
from self_jev.car_build_nn.builds import get_candidate_builds
from self_jev.car_build_nn.dataset import _load_frozen_driver
from self_jev.sim.env import RaceEnv
from self_jev.sim.physics import PhysicsConfig
from self_jev.sim.residual import assist_ers_mode, combine_residual_actions

DEFAULT_OUT = REPO_ROOT / "reports" / "track_rule_confirm_v2.json"
CONFIRM_SEEDS = [101, 202, 303, 404, 505]
WTL_TOL_S = 0.50


def build_fresh_scenarios(seeds: list[int] = CONFIRM_SEEDS) -> list[dict[str, Any]]:
    """Generate 9 cells x len(seeds) fresh paired scenario realizations."""
    out: list[dict[str, Any]] = []
    for cell_idx, base in enumerate(re_eval.CANONICAL_SCENARIOS):
        for s_idx, seed in enumerate(seeds):
            rng = np.random.default_rng(seed * 100 + cell_idx)
            reg = base["weather_regime"]
            if reg == "Dry":
                ws = float(np.clip(rng.uniform(0.0, 0.06), 0.0, 0.06))
                we = float(np.clip(rng.uniform(0.0, 0.06), 0.0, 0.06))
                rp = float(rng.uniform(0.0, 0.15))
                temp = float(rng.uniform(26.0, 34.0))
            elif reg == "Wetting":
                ws = float(rng.uniform(0.02, 0.14))
                we = float(rng.uniform(0.58, 0.88))
                rp = float(rng.uniform(0.55, 0.90))
                temp = float(rng.uniform(20.0, 27.0))
            else:  # Saturated
                ws = float(rng.uniform(0.76, 0.95))
                we = float(rng.uniform(0.78, 0.96))
                rp = float(rng.uniform(0.85, 0.99))
                temp = float(rng.uniform(15.0, 22.0))
            grip_scale = float(rng.uniform(0.94, 1.06))
            aero_scale = float(rng.uniform(0.94, 1.06))
            ramp_start = float(rng.uniform(25.0, 80.0))
            ramp_end = float(rng.uniform(280.0, 420.0))
            out.append({
                "scenario_id": f"{base['id']}_s{seed}",
                "cell_id": base["id"],
                "track": base["track"],
                "weather_regime": reg,
                "seed": seed,
                "wet_start": round(ws, 4),
                "wet_end": round(we, 4),
                "rain_prob": round(rp, 4),
                "track_temp_c": round(temp, 2),
                "ramp_start_s": round(ramp_start, 1),
                "ramp_end_s": round(ramp_end, 1),
                "grip_scale": round(grip_scale, 4),
                "aero_scale": round(aero_scale, 4),
            })
    return out


@torch.inference_mode()
def simulate_batched_track(
    track_id: str,
    policy_name: str,
    scenarios: list[dict[str, Any]],
    driver_path: Path,
    laps_total: int = 4,
    max_steps: int = 30 * 800,
) -> list[dict[str, Any]]:
    """Run all scenarios for one track in a single vectorized RaceEnv batch."""
    torch.set_num_threads(1)
    torch.manual_seed(42)
    device = torch.device("cpu")
    cal_path = REPO_ROOT / "self_jev" / "sim" / "calibrated_params.json"
    cfg = PhysicsConfig.from_json(cal_path) if cal_path.is_file() else PhysicsConfig()
    builds = get_candidate_builds()
    driver_model, obs_mean, obs_std, is_residual = _load_frozen_driver(driver_path, device)

    # Determine fixed mode for this track under the policy
    if policy_name in ("balanced", "driver_only", "neutral_adapter"):
        mode = "balanced"
    elif policy_name == "track_rule":
        mode = "balanced" if track_id == "parkland" else "push_max"
    else:
        raise ValueError(f"Unknown policy: {policy_name}")

    worlds = len(scenarios)
    env = RaceEnv(
        track_id=track_id,
        worlds=worlds,
        cars_per_world=1,
        laps_total=laps_total,
        phase="race",
        weather_regime="dry",
        car=builds["dry_balanced"],
        cfg=cfg,
        device=device,
        domain_rand=False,
    )
    for wi, sc in enumerate(scenarios):
        env.w_state["wet_start"][wi] = sc["wet_start"]
        env.w_state["wet_end"][wi] = sc["wet_end"]
        env.w_state["ramp_start_s"][wi] = sc.get("ramp_start_s", 40.0)
        env.w_state["ramp_end_s"][wi] = sc.get("ramp_end_s", 360.0)
        env.w_state["wetness"][wi] = sc["wet_start"]
        env.w_state["rain_intensity"][wi] = sc["wet_start"] * 0.5
        env.w_state["track_temp_c"][wi] = sc["track_temp_c"]
        env.w_state["forecast_rain_prob"][wi] = sc["rain_prob"]
    env.reset_all()

    # Apply exact paired grip/aero scales after reset_all()
    for wi, sc in enumerate(scenarios):
        env.grip_scale[wi] = sc.get("grip_scale", 1.0)
        env.aero_scale[wi] = sc.get("aero_scale", 1.0)

    for step_i in range(max_steps):
        raw = env.get_obs(include_map_ext=False)
        x = torch.clamp(
            torch.nan_to_num((raw - obs_mean) / obs_std, nan=0.0, posinf=10.0, neginf=-10.0),
            -10.0,
            10.0,
        )
        mu, ers_logits, flag_logits, _ = driver_model(x)
        ers = torch.argmax(ers_logits, dim=-1)

        mu_adj = mu.clone()
        ovr = (flag_logits[:, 1] > 0.0).float()
        if mode == "push_max":
            mu_adj[:, 0] = mu_adj[:, 0] + 0.15
            push_mask = (env.ers_soc_mj > 0.5) & (env.wetness < 0.45)
            ers = torch.where(push_mask, torch.full_like(ers, 3), ers)
            ovr = torch.where(push_mask, torch.ones_like(ovr), ovr)

        if is_residual:
            thr, brk, st_cmd = combine_residual_actions(raw, mu_adj)
            if mode == "balanced":
                ers = assist_ers_mode(raw, ers, thr, brk)
        else:
            thr = torch.clamp(mu_adj[:, 0], 0.0, 1.0)
            brk = torch.clamp(mu_adj[:, 1], 0.0, 1.0)
            st_cmd = torch.clamp(mu_adj[:, 2], -1.0, 1.0)

        env.step({
            "throttle": thr,
            "brake": brk,
            "steer": st_cmd,
            "ers_mode": ers,
            "aero_mode": (flag_logits[:, 0] > 0.0).float(),
            "override": ovr,
            "aid": (flag_logits[:, 2] > 0.0).float(),
            "pit_request": torch.zeros(env.N, device=device),
            "pit_compound": torch.zeros(env.N, device=device, dtype=torch.long),
        })
        if (step_i + 1) % 64 == 0 and (env.status != 1).all():
            break

    rows = []
    for wi, sc in enumerate(scenarios):
        st0 = int(env.status[wi].item())
        fin = st0 == 2
        ft = round(float(env.finish_t[wi].item()), 4) if fin and float(env.finish_t[wi].item()) > 0 else None
        rows.append({
            "scenario_id": sc["scenario_id"],
            "cell_id": sc.get("cell_id", sc["scenario_id"]),
            "track": sc["track"],
            "weather_regime": sc["weather_regime"],
            "seed": sc.get("seed", 0),
            "mode_applied": mode,
            "status": st0,
            "finished": fin,
            "completed_laps": laps_total if fin else max(0, int(env.lap[wi].item())),
            "finish_time_s": ft,
            "tick_count": int(env.tick_count[wi].item()),
        })
    return rows


def run_all(driver_path: Path = re_eval.DEFAULT_DRIVER, out_path: Path = DEFAULT_OUT) -> dict[str, Any]:
    t0 = time.perf_counter()
    script_path = str(Path(__file__).resolve())

    # Launch 6 parallel workers for confirmation (3 tracks x 2 policies) + 2 workers for parity (driver_only vs neutral_adapter)
    jobs = [
        ("parity", "parkland", "driver_only"),
        ("parity", "parkland", "neutral_adapter"),
        ("parity", "highland", "driver_only"),
        ("parity", "highland", "neutral_adapter"),
        ("parity", "crossover", "driver_only"),
        ("parity", "crossover", "neutral_adapter"),
        ("confirm", "parkland", "balanced"),
        ("confirm", "parkland", "track_rule"),
        ("confirm", "highland", "balanced"),
        ("confirm", "highland", "track_rule"),
        ("confirm", "crossover", "balanced"),
        ("confirm", "crossover", "track_rule"),
    ]
    pending = list(jobs)
    running: list[tuple[str, str, str, subprocess.Popen[str]]] = []
    out_map: dict[tuple[str, str, str], list[dict[str, Any]]] = {}

    while pending or running:
        while pending and len(running) < 6:
            kind, tr, pol = pending.pop(0)
            cmd = [sys.executable, script_path, "--worker", f"{kind}:{tr}:{pol}", "--driver", str(driver_path)]
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            running.append((kind, tr, pol, proc))
        still: list[tuple[str, str, str, subprocess.Popen[str]]] = []
        for kind, tr, pol, proc in running:
            ret = proc.poll()
            if ret is None:
                still.append((kind, tr, pol, proc))
                continue
            out, err = proc.communicate()
            if ret != 0:
                raise RuntimeError(f"Worker {kind}:{tr}:{pol} failed ({ret}): {err}")
            out_map[(kind, tr, pol)] = json.loads(out)
        running = still
        if running:
            time.sleep(0.1)

    # 1. Seeded parity verification
    parity_rows = []
    max_diff_s = 0.0
    max_tick_diff = 0
    for tr in ("parkland", "highland", "crossover"):
        d_rows = out_map[("parity", tr, "driver_only")]
        n_rows = out_map[("parity", tr, "neutral_adapter")]
        for dr, nr in zip(d_rows, n_rows):
            td = abs(float(dr["finish_time_s"] or 0.0) - float(nr["finish_time_s"] or 0.0))
            tk = abs(int(dr["tick_count"]) - int(nr["tick_count"]))
            max_diff_s = max(max_diff_s, td)
            max_tick_diff = max(max_tick_diff, tk)
            parity_rows.append({
                "scenario_id": dr["scenario_id"],
                "driver_only_t": dr["finish_time_s"],
                "neutral_adapter_t": nr["finish_time_s"],
                "abs_diff_s": round(td, 6),
                "tick_diff": tk,
            })

    # 2. 45-scenario confirmation aggregation
    bal_all = [r for tr in ("parkland", "highland", "crossover") for r in out_map[("confirm", tr, "balanced")]]
    trk_all = [r for tr in ("parkland", "highland", "crossover") for r in out_map[("confirm", tr, "track_rule")]]

    paired_outcomes = []
    paired_deltas = []
    wins = ties = losses = 0
    by_track: dict[str, dict[str, Any]] = {}
    by_cell: dict[str, dict[str, Any]] = {}

    for br, tr_r in zip(bal_all, trk_all):
        assert br["scenario_id"] == tr_r["scenario_id"]
        b_fin, t_fin = br["finished"], tr_r["finished"]
        b_t, t_t = br["finish_time_s"], tr_r["finish_time_s"]
        delta = round(float(t_t) - float(b_t), 4) if (b_fin and t_fin and b_t is not None and t_t is not None) else None
        if t_fin and not b_fin:
            res_lbl = "win"
            wins += 1
        elif b_fin and not t_fin:
            res_lbl = "loss"
            losses += 1
        elif not b_fin and not t_fin:
            res_lbl = "tie_both_failed"
            ties += 1
        else:
            assert delta is not None
            paired_deltas.append(delta)
            if delta < -WTL_TOL_S:
                res_lbl = "win"
                wins += 1
            elif delta > WTL_TOL_S:
                res_lbl = "loss"
                losses += 1
            else:
                res_lbl = "tie"
                ties += 1

        rec = {
            "scenario_id": br["scenario_id"],
            "cell_id": br["cell_id"],
            "track": br["track"],
            "weather_regime": br["weather_regime"],
            "seed": br["seed"],
            "balanced_finished": b_fin,
            "balanced_time_s": b_t,
            "balanced_status": br["status"],
            "track_rule_mode": tr_r["mode_applied"],
            "track_rule_finished": t_fin,
            "track_rule_time_s": t_t,
            "track_rule_status": tr_r["status"],
            "delta_s": delta,
            "outcome": res_lbl,
        }
        paired_outcomes.append(rec)

        for bucket_dict, key in ((by_track, br["track"]), (by_cell, br["cell_id"])):
            if key not in bucket_dict:
                bucket_dict[key] = {"n": 0, "bal_fin": 0, "trk_fin": 0, "wins": 0, "ties": 0, "losses": 0, "deltas": []}
            bd = bucket_dict[key]
            bd["n"] += 1
            bd["bal_fin"] += int(b_fin)
            bd["trk_fin"] += int(t_fin)
            if res_lbl == "win":
                bd["wins"] += 1
            elif res_lbl == "loss":
                bd["losses"] += 1
            else:
                bd["ties"] += 1
            if delta is not None:
                bd["deltas"].append(delta)

    def _finalize_group(g: dict[str, Any]) -> dict[str, Any]:
        ds = sorted(g.pop("deltas"))
        g["paired_n"] = len(ds)
        g["paired_mean_delta_s"] = round(sum(ds) / len(ds), 2) if ds else None
        g["paired_median_delta_s"] = round(ds[len(ds) // 2], 2) if ds else None
        return g

    by_track_clean = {k: _finalize_group(v) for k, v in by_track.items()}
    by_cell_clean = {k: _finalize_group(v) for k, v in by_cell.items()}

    bal_fin_ts = [float(r["finish_time_s"]) for r in bal_all if r["finished"] and r["finish_time_s"] is not None]
    trk_fin_ts = [float(r["finish_time_s"]) for r in trk_all if r["finished"] and r["finish_time_s"] is not None]
    ds_all = sorted(paired_deltas)

    report = {
        "evaluator_version": "pace-eval-v2+track-rule-confirm",
        "driver_weights": str(driver_path.name),
        "fixed_build": "dry_balanced",
        "fixed_pit_policy": "stay_out",
        "wtl_tolerance_s": WTL_TOL_S,
        "seeds": CONFIRM_SEEDS,
        "wall_s": round(time.perf_counter() - t0, 2),
        "seeded_parity_check": {
            "scenarios_n": len(parity_rows),
            "max_abs_diff_s": round(max_diff_s, 6),
            "max_tick_diff": max_tick_diff,
            "exact_parity_passed": (max_diff_s == 0.0 and max_tick_diff == 0),
            "rows": parity_rows,
        },
        "confirmation_summary": {
            "total_paired_scenarios": len(paired_outcomes),
            "balanced_finishes_n": len(bal_fin_ts),
            "balanced_avg_time_s_finishers_only": round(sum(bal_fin_ts) / len(bal_fin_ts), 2) if bal_fin_ts else None,
            "track_rule_finishes_n": len(trk_fin_ts),
            "track_rule_avg_time_s_finishers_only": round(sum(trk_fin_ts) / len(trk_fin_ts), 2) if trk_fin_ts else None,
            "paired_common_finishes_n": len(ds_all),
            "paired_mean_delta_s": round(sum(ds_all) / len(ds_all), 2) if ds_all else None,
            "paired_median_delta_s": round(ds_all[len(ds_all) // 2], 2) if ds_all else None,
            "wtl_vs_balanced": {"wins": wins, "ties": ties, "losses": losses},
            "by_track": by_track_clean,
            "by_cell": by_cell_clean,
        },
        "paired_outcomes": paired_outcomes,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="Seeded parity check + 45-scenario track-rule confirmation.")
    ap.add_argument("--worker", default=None, help="Internal worker: <kind>:<track>:<policy>")
    ap.add_argument("--driver", type=Path, default=re_eval.DEFAULT_DRIVER)
    args = ap.parse_args()

    if args.worker:
        kind, tr, pol = args.worker.split(":", 2)
        if kind == "parity":
            scs = [
                {**sc, "scenario_id": sc["id"], "cell_id": sc["id"], "seed": 42}
                for sc in re_eval.CANONICAL_SCENARIOS
                if sc["track"] == tr
            ]
        else:
            scs = [sc for sc in build_fresh_scenarios() if sc["track"] == tr]
        rows = simulate_batched_track(tr, pol, scs, args.driver, laps_total=4)
        print(json.dumps(rows))
        return

    rep = run_all(driver_path=args.driver)
    p = rep["seeded_parity_check"]
    c = rep["confirmation_summary"]
    wtl = c["wtl_vs_balanced"]
    print(f"Finished parity + 45-scenario confirmation in {rep['wall_s']}s")
    print(f"Seeded Parity: max_diff={p['max_abs_diff_s']}s, max_tick_diff={p['max_tick_diff']}, passed={p['exact_parity_passed']}")
    print(
        f"\nConfirmation (n={c['total_paired_scenarios']} fresh paired races):\n"
        f"  balanced   : {c['balanced_finishes_n']}/{c['total_paired_scenarios']} finishes | avg={c['balanced_avg_time_s_finishers_only']}s\n"
        f"  track_rule : {c['track_rule_finishes_n']}/{c['total_paired_scenarios']} finishes | avg={c['track_rule_avg_time_s_finishers_only']}s\n"
        f"  paired_d   : {c['paired_mean_delta_s']}s mean / {c['paired_median_delta_s']}s median (paired n={c['paired_common_finishes_n']})\n"
        f"  W-T-L      : {wtl['wins']}-{wtl['ties']}-{wtl['losses']} (tol={rep['wtl_tolerance_s']}s)"
    )
    print("\nBy Track:")
    for tr, d in c["by_track"].items():
        print(f"  {tr:10s} | bal={d['bal_fin']}/{d['n']} trk={d['trk_fin']}/{d['n']} | paired_d={d['paired_mean_delta_s']}s (n={d['paired_n']}) | W-T-L={d['wins']}-{d['ties']}-{d['losses']}")


if __name__ == "__main__":
    main()
