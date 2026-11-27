#!/usr/bin/env python3
"""Closed-loop 4-lap simulator test of AGX winner-derived builds vs hand-crafted builds.

Step 2 of the recommended workflow:
  Tests each regime's matched AGX winner build (from data/winner-builds-and-controls.json)
  against our 4 hand-crafted builds (dry_low_drag, dry_balanced, damp_mid_downforce,
  wet_high_downforce) using the existing frozen 30 Hz RL driver (models/jev_rl_v7.mapdata.npz)
  across all 16 covered (track, weather_profile) scenarios, using the exact observed
  (wet_start, wet_end) from each winner's trace.

Run with project virtualenv:
  .venv-decision/bin/python scripts/eval_winner_builds.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from self_jev.car_build_nn.builds import BUILD_NAMES, get_candidate_builds
from self_jev.car_build_nn.dataset import _load_frozen_driver
from self_jev.sim.design import CarParams, load_design, validate_design_limits
from self_jev.sim.env import RaceEnv
from self_jev.sim.physics import PhysicsConfig
from self_jev.sim.residual import assist_ers_mode, combine_residual_actions
from self_jev.sim.track import load_track

DEFAULT_WINNER_JSON = REPO_ROOT / "data" / "winner-builds-and-controls.json"
DEFAULT_DRIVER = REPO_ROOT / "models" / "jev_rl_v7.mapdata.npz"
DEFAULT_OUT = REPO_ROOT / "reports" / "winner_builds_sim_test.json"


def build_dict_to_car_params(b: dict[str, Any], base: CarParams) -> CarParams:
    return replace(
        base,
        front_wing=float(b["front_wing"]),
        rear_wing=float(b["rear_wing"]),
        ballast_kg=float(b["ballast_kg"]),
        weight_front_pct=float(b["weight_front_pct"]),
        final_drive=float(b["final_drive"]),
        gear_ratios=tuple(float(x) for x in b["gear_ratios"]),
        bias_front_pct=float(b["brake_bias_front_pct"]),
        radiator_opening=float(b["radiator_opening"]),
        fuel_load_kg=float(b["fuel_load_kg"]),
        ers_harvest_level=int(b["ers_harvest_level"]),
        start_compound=str(b["start_compound"]),
    )


@torch.inference_mode()
def run_single_4lap(
    track_id: str,
    car: CarParams,
    wet_start: float,
    wet_end: float,
    driver_model: torch.nn.Module,
    obs_mean: torch.Tensor,
    obs_std: torch.Tensor,
    is_residual: bool,
    cfg: PhysicsConfig,
    device: torch.device,
    laps_total: int = 4,
    max_steps: int = 30 * 800,
) -> dict[str, Any]:
    env = RaceEnv(
        track_id=track_id,
        worlds=1,
        cars_per_world=1,
        laps_total=laps_total,
        phase="race",
        weather_regime="dry",
        car=car,
        rain_car=car,  # Do NOT silently swap to rain_design when testing a specific build
        cfg=cfg,
        device=device,
    )
    env.w_state["wet_start"][0] = wet_start
    env.w_state["wet_end"][0] = wet_end
    env.w_state["ramp_start_s"][0] = 40.0
    env.w_state["ramp_end_s"][0] = 360.0
    env.w_state["wetness"][0] = wet_start
    env.w_state["rain_intensity"][0] = wet_start * 0.5
    env.w_state["track_temp_c"][0] = 30.0 - 8.0 * wet_start
    env.w_state["forecast_rain_prob"][0] = max(wet_start, wet_end)
    env.reset_all()

    for _ in range(max_steps):
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
            "pit_request": torch.zeros(1, device=device),
            "pit_compound": torch.ones(1, device=device, dtype=torch.long),
        })
        if int(env.status[0].item()) != 1:
            break

    st0 = int(env.status[0].item())
    finished = st0 == 2
    ft = round(float(env.finish_t[0].item()), 3) if finished and float(env.finish_t[0].item()) > 0 else None
    return {
        "status": st0,
        "finished": finished,
        "completed_laps": laps_total if finished else max(0, int(env.lap[0].item())),
        "finish_time_s": ft,
        "dist_m": round(float(env.dist[0].item()), 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Test AGX winner builds vs hand-crafted builds with existing driver.")
    ap.add_argument("--winner-json", type=Path, default=DEFAULT_WINNER_JSON)
    ap.add_argument("--driver", type=Path, default=DEFAULT_DRIVER)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    t0 = time.perf_counter()
    device = torch.device(args.device)
    cal_path = REPO_ROOT / "self_jev" / "sim" / "calibrated_params.json"
    cfg = PhysicsConfig.from_json(cal_path) if cal_path.is_file() else PhysicsConfig()
    base = load_design()
    hc_builds = get_candidate_builds(base)
    driver_model, obs_mean, obs_std, is_residual = _load_frozen_driver(args.driver, device)

    doc = json.loads(args.winner_json.read_text(encoding="utf-8"))
    cols = doc["control_trace_columns"]
    w_idx = cols.index("track_wetness")
    tracks_cache = {t: load_track(t) for t in ("parkland", "highland", "crossover")}

    rows = []
    for key, e in sorted(doc["winners"].items()):
        tr = e["track"].lower()
        wp = e["weather_profile"].lower()
        trace = e["control_trace"]
        w_start = float(trace[0][w_idx])
        w_end = float(trace[-1][w_idx])
        agx_t = float(e["official_finish_s"])

        win_car = build_dict_to_car_params(e["build"], base)
        legality_errs = validate_design_limits(win_car, getattr(tracks_cache[tr], "design_limits", None) or {})

        res_map: dict[str, Any] = {}
        res_map["winner_matched"] = run_single_4lap(
            tr, win_car, w_start, w_end, driver_model, obs_mean, obs_std, is_residual, cfg, device
        )
        for bname in BUILD_NAMES:
            res_map[bname] = run_single_4lap(
                tr, hc_builds[bname], w_start, w_end, driver_model, obs_mean, obs_std, is_residual, cfg, device
            )

        wt = res_map["winner_matched"]["finish_time_s"]
        db_t = res_map["dry_balanced"]["finish_time_s"]
        rows.append({
            "key": key,
            "track": tr,
            "weather_profile": wp,
            "wet_start": round(w_start, 3),
            "wet_end": round(w_end, 3),
            "agx_winner_s": round(agx_t, 3),
            "legality_errors": legality_errs,
            "results": res_map,
            "winner_build_vs_dry_balanced_s": round(wt - db_t, 3) if wt and db_t else None,
            "winner_build_vs_agx_s": round(wt - agx_t, 3) if wt else None,
        })
        print(
            f"{key:22s} | AGX={agx_t:6.1f}s | win_bld={str(wt):>7s}s | dry_bal={str(db_t):>7s}s "
            f"| delta_vs_dry_bal={rows[-1]['winner_build_vs_dry_balanced_s']}"
        )

    out_doc = {
        "driver_weights": args.driver.name,
        "wall_s": round(time.perf_counter() - t0, 2),
        "scenarios_n": len(rows),
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out_doc, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved {args.out} in {out_doc['wall_s']}s")


if __name__ == "__main__":
    main()
