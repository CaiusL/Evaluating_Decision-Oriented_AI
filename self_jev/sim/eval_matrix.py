"""Automated 3 x 5 x 2 (30-cell) evaluation matrix & fast in-training gate for self_jev.sim."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from self_jev.sim.design import load_design
from self_jev.sim.env import RaceEnv
from self_jev.sim.physics import PhysicsConfig
from self_jev.sim.residual import assist_ers_mode, combine_residual_actions

EVAL_TRACKS = ("parkland", "highland", "crossover")
EVAL_WEATHERS = ("dry", "damp", "wet", "ramp_wet", "ramp_dry")


def evaluate_policy_cell(
    policy_fn,
    track_id: str,
    weather_regime: str,
    phase: str,
    cfg: PhysicsConfig,
    device: torch.device,
    max_steps: int = 30 * 240,
) -> dict[str, Any]:
    """Run one deterministic cell (1 world: C=1 for qualifying, C=10 for race)."""
    C = 1 if phase == "qualifying" else 10
    laps = 1 if phase == "qualifying" else 4
    env = RaceEnv(
        track_id=track_id,
        worlds=1,
        cars_per_world=C,
        laps_total=laps,
        phase=phase,
        weather_regime=weather_regime,
        car=load_design(),
        cfg=cfg,
        device=device,
    )

    off_ticks = 0
    contact_ticks = 0
    graze_ticks = 0
    for step_i in range(max_steps):
        obs_181 = env.get_obs(include_map_ext=False)
        actions = policy_fn(obs_181, env)
        out = env.step(actions)
        off_ticks += int(out["off_track"][0].item())
        contact_ticks += int(out["car_contact"][0].item())
        graze_ticks += int(out["wall_graze"][0].item())
        if (step_i + 1) % 32 == 0 and (env.status != 1).all():
            break

    st0 = int(env.status[0].item())
    classified = bool(env.classified()[0].item()) if phase == "race" else (st0 == 2 and int(env.lap_dirty[0].item()) == 0)
    legal_2cmp = bool(env.dry_compound_legal()[0].item()) if phase == "race" else True
    return {
        "track": track_id,
        "weather": weather_regime,
        "phase": phase,
        "status": st0,
        "laps": int(env.lap[0].item()),
        "dist_m": round(float(env.dist[0].item()), 1),
        "finish_t": round(float(env.finish_t[0].item()), 3),
        "best_lap_t": round(float(env.best_lap_t[0].item()), 3),
        "pit_stops": int(env.pit_stops[0].item()),
        "classified": classified,
        "dry_compound_legal": legal_2cmp,
        "clean_lap": int(env.lap_dirty[0].item()) == 0 and off_ticks == 0,
        "off_track_ticks": off_ticks,
        "contact_ticks": contact_ticks,
        "wall_graze_ticks": graze_ticks,
    }


def evaluate_model_fast_30(
    model: torch.nn.Module,
    obs_mean: torch.Tensor,
    obs_std: torch.Tensor,
    cfg: PhysicsConfig,
    device: torch.device,
    max_steps: int = 30 * 210,
) -> dict[str, Any]:
    """Batched 30-cell zero-damage / zero-off-track gate (3 tracks x 5 weathers x 2 phases) on GPU."""
    was_training = model.training
    model.eval()
    car = load_design()

    qual_times: list[float] = []
    qual_clean = 0
    qual_total = 0
    race_classified = 0
    race_total = 0
    race_dists: list[float] = []
    walls = 0
    stalls = 0
    total_off_ticks = 0
    total_contact_ticks = 0
    total_graze_ticks = 0

    for tid in EVAL_TRACKS:
        # Batched 5 qualifying worlds (1 per weather regime)
        q_env = RaceEnv(tid, worlds=len(EVAL_WEATHERS), cars_per_world=1, laps_total=1, phase="qualifying",
                        weather_regime="mixed", car=car, cfg=cfg, device=device)
        # Override world 0..4 to exact deterministic 5 weather regimes
        for wi, wreg in enumerate(EVAL_WEATHERS):
            from self_jev.sim.weather import build_weather_tensors
            ws = build_weather_tensors(1, 1, wreg, device)
            for k in ("wet_start", "wet_end", "ramp_start_s", "ramp_end_s", "wetness", "rain_intensity", "track_temp_c", "forecast_rain_prob"):
                q_env.w_state[k][wi] = ws[k][0]
        q_env.reset_all()
        q_off = torch.zeros(q_env.N, device=device, dtype=torch.long)
        q_graze = torch.zeros(q_env.N, device=device, dtype=torch.long)

        for step_i in range(max_steps):
            raw = q_env.get_obs(include_map_ext=False)
            x = torch.clamp(torch.nan_to_num((raw - obs_mean) / obs_std, nan=0.0, posinf=10.0, neginf=-10.0), -10.0, 10.0)
            with torch.no_grad():
                mu, ers_logits, flag_logits, _ = model(x)
            thr, brk, st = combine_residual_actions(raw, mu)
            ers = assist_ers_mode(raw, torch.argmax(ers_logits, dim=-1), thr, brk)
            out = q_env.step({
                "throttle": thr, "brake": brk, "steer": st, "ers_mode": ers,
                "aero_mode": (flag_logits[:, 0] > 0.0).float(),
                "override": (flag_logits[:, 1] > 0.0).float(),
                "aid": (flag_logits[:, 2] > 0.0).float(),
                "pit_request": torch.zeros(q_env.N, device=device),
                "pit_compound": torch.ones(q_env.N, device=device, dtype=torch.long),
            })
            q_off += out["off_track"]
            q_graze += out["wall_graze"]
            if (step_i + 1) % 32 == 0 and (q_env.status != 1).all():
                break

        for wi in range(len(EVAL_WEATHERS)):
            st0 = int(q_env.status[wi].item())
            off0 = int(q_off[wi].item())
            grz0 = int(q_graze[wi].item())
            ft = float(q_env.finish_t[wi].item())
            qual_total += 1
            if st0 == 5:
                walls += 1
            if st0 == 4:
                stalls += 1
            total_off_ticks += off0
            total_graze_ticks += grz0
            if st0 == 2 and int(q_env.lap_dirty[wi].item()) == 0 and off0 == 0 and grz0 == 0:
                qual_clean += 1
                qual_times.append(ft)

        # Batched 5 race worlds (10 cars each) for 200s (6000 steps)
        r_env = RaceEnv(tid, worlds=len(EVAL_WEATHERS), cars_per_world=10, laps_total=4, phase="race",
                        weather_regime="mixed", car=car, cfg=cfg, device=device)
        for wi, wreg in enumerate(EVAL_WEATHERS):
            from self_jev.sim.weather import build_weather_tensors
            ws = build_weather_tensors(1, 10, wreg, device)
            sl = slice(wi * 10, (wi + 1) * 10)
            for k in ("wet_start", "wet_end", "ramp_start_s", "ramp_end_s", "wetness", "rain_intensity", "track_temp_c", "forecast_rain_prob"):
                r_env.w_state[k][sl] = ws[k]
        r_env.reset_all()
        r_off = torch.zeros(r_env.N, device=device, dtype=torch.long)
        r_cnt = torch.zeros(r_env.N, device=device, dtype=torch.long)
        r_grz = torch.zeros(r_env.N, device=device, dtype=torch.long)

        for step_i in range(min(max_steps, 30 * 190)):
            raw = r_env.get_obs(include_map_ext=False)
            x = torch.clamp(torch.nan_to_num((raw - obs_mean) / obs_std, nan=0.0, posinf=10.0, neginf=-10.0), -10.0, 10.0)
            with torch.no_grad():
                mu, ers_logits, flag_logits, _ = model(x)
            thr, brk, st = combine_residual_actions(raw, mu)
            ers = assist_ers_mode(raw, torch.argmax(ers_logits, dim=-1), thr, brk)
            pit_req = (
                (r_env.lap >= 1)
                & (~r_env.wet_waiver_latched)
                & (r_env.pit_stops == 0)
                & (r_env.s >= r_env.track.pit_entry_s - 220.0)
            )
            out = r_env.step({
                "throttle": thr, "brake": brk, "steer": st, "ers_mode": ers,
                "aero_mode": (flag_logits[:, 0] > 0.0).float(),
                "override": (flag_logits[:, 1] > 0.0).float(),
                "aid": (flag_logits[:, 2] > 0.0).float(),
                "pit_request": pit_req.float(),
                "pit_compound": torch.ones(r_env.N, device=device, dtype=torch.long),
            })
            r_off += out["off_track"]
            r_cnt += out["car_contact"]
            r_grz += out["wall_graze"]
            if (step_i + 1) % 32 == 0 and (r_env.status != 1).all():
                break

        cls_vec = r_env.classified()
        for wi in range(len(EVAL_WEATHERS)):
            ci = wi * 10  # ego car 0 in world wi
            st0 = int(r_env.status[ci].item())
            race_total += 1
            if st0 == 5:
                walls += 1
            if st0 == 4:
                stalls += 1
            if bool(cls_vec[ci].item()):
                race_classified += 1
            race_dists.append(float(r_env.dist[ci].item()))
            total_off_ticks += int(r_off[ci].item())
            total_contact_ticks += int(r_cnt[ci].item())
            total_graze_ticks += int(r_grz[ci].item())

    if was_training:
        model.train()

    return {
        "total_cells": qual_total + race_total,
        "qual_clean_rate": round(qual_clean / max(1, qual_total), 4),
        "race_classified_rate": round(race_classified / max(1, race_total), 4),
        "wall_retirements": walls,
        "stall_retirements": stalls,
        "off_track_ticks": total_off_ticks,
        "contact_ticks": total_contact_ticks,
        "wall_graze_ticks": total_graze_ticks,
        "mean_qual_lap_t": round(float(np.mean(qual_times)), 3) if qual_times else 999.0,
        "mean_race_dist_m": round(float(np.mean(race_dists)), 1) if race_dists else 0.0,
    }


def run_eval_matrix(
    weights_npz: Path,
    out_json: Path,
    max_steps: int = 30 * 240,
    device_str: str = "cpu",
) -> dict[str, Any]:
    """Evaluate a .mapdata.npz policy across 3 tracks x 5 weather regimes x 2 phases (30 cells)."""
    from self_jev.train_rl_sim import ActorCritic

    device = torch.device(device_str)
    cal_path = Path(__file__).resolve().parent / "calibrated_params.json"
    cfg = PhysicsConfig.from_json(cal_path) if cal_path.is_file() else PhysicsConfig()

    bc = np.load(weights_npz, allow_pickle=False)
    obs_mean = torch.from_numpy(bc["obs_mean"]).to(device)
    obs_std = torch.from_numpy(bc["obs_std"]).to(device)

    h1 = int(bc["net__0__weight"].shape[0])
    h2 = int(bc["net__3__weight"].shape[0])
    h3 = int(bc["net__6__weight"].shape[0])
    model = ActorCritic(len(obs_mean), h1=h1, h2=h2, h3=h3).to(device)
    sd = model.state_dict()
    for k in bc.files:
        if k in ("obs_mean", "obs_std", "policy_mode"):
            continue
        tk = k.replace("__", ".")
        if tk in sd and sd[tk].shape == tuple(bc[k].shape):
            sd[tk] = torch.from_numpy(bc[k]).to(device)
    model.load_state_dict(sd)
    model.eval()

    is_residual = "policy_mode" in bc.files and int(bc["policy_mode"]) == 1

    def _policy(obs_181: torch.Tensor, env: RaceEnv) -> dict[str, torch.Tensor]:
        x = torch.clamp(torch.nan_to_num((obs_181 - obs_mean) / obs_std, nan=0.0, posinf=10.0, neginf=-10.0), -10.0, 10.0)
        with torch.no_grad():
            mu, ers_logits, flag_logits, _ = model(x)
        ers = torch.argmax(ers_logits, dim=-1)
        if is_residual:
            thr, brk, str_cmd = combine_residual_actions(obs_181, mu)
            ers = assist_ers_mode(obs_181, ers, thr, brk)
        else:
            thr = torch.clamp(mu[:, 0], 0.0, 1.0)
            brk = torch.clamp(mu[:, 1], 0.0, 1.0)
            str_cmd = torch.clamp(mu[:, 2], -1.0, 1.0)
        pit_req = (
            (env.phase == "race")
            & (env.lap >= 1)
            & (~env.wet_waiver_latched)
            & (env.pit_stops == 0)
            & (env.s >= env.track.pit_entry_s - 220.0)
        )
        return {
            "throttle": thr,
            "brake": brk,
            "steer": str_cmd,
            "ers_mode": ers,
            "aero_mode": (flag_logits[:, 0] > 0.0).float(),
            "override": (flag_logits[:, 1] > 0.0).float(),
            "aid": (flag_logits[:, 2] > 0.0).float(),
            "pit_request": pit_req.float(),
            "pit_compound": torch.ones(env.N, device=device, dtype=torch.long),
        }

    cells = []
    for tid in EVAL_TRACKS:
        for w_reg in EVAL_WEATHERS:
            for ph in ("qualifying", "race"):
                cells.append(evaluate_policy_cell(_policy, tid, w_reg, ph, cfg, device, max_steps=max_steps))

    race_cells = [c for c in cells if c["phase"] == "race"]
    qual_cells = [c for c in cells if c["phase"] == "qualifying"]
    q_times = [c["finish_t"] for c in qual_cells if c["status"] == 2 and c["clean_lap"]]
    summary = {
        "weights": str(weights_npz),
        "total_cells": len(cells),
        "race_classified_rate": round(sum(1 for c in race_cells if c["classified"]) / max(1, len(race_cells)), 4),
        "race_2cmp_legal_rate": round(sum(1 for c in race_cells if c["dry_compound_legal"]) / max(1, len(race_cells)), 4),
        "qual_clean_rate": round(sum(1 for c in qual_cells if c["clean_lap"]) / max(1, len(qual_cells)), 4),
        "wall_retirements": sum(1 for c in cells if c["status"] == 5),
        "stall_retirements": sum(1 for c in cells if c["status"] == 4),
        "total_off_track_ticks": sum(c["off_track_ticks"] for c in cells),
        "total_contact_ticks": sum(c["contact_ticks"] for c in cells),
        "total_wall_graze_ticks": sum(c["wall_graze_ticks"] for c in cells),
        "mean_qual_lap_t": round(float(np.mean(q_times)), 3) if q_times else 999.0,
        "cells": cells,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 3x5x2 (30-cell) v3 evaluation matrix.")
    parser.add_argument("--weights", type=Path, default=Path("models/jev_rl.mapdata.npz"))
    parser.add_argument("--out", type=Path, default=Path("reports/eval_matrix_latest.json"))
    parser.add_argument("--max-steps", type=int, default=30 * 240)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    res = run_eval_matrix(args.weights, args.out, max_steps=args.max_steps, device_str=args.device)
    print(json.dumps({k: v for k, v in res.items() if k != "cells"}, indent=2))


if __name__ == "__main__":
    main()
