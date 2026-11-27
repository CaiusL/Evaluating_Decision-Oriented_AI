"""Exact RaceEnv snapshot/restore, vectorized closed-loop runner, and batched counterfactual rollout labeler."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from decision_eval.racing_contract import (
    BUILD_OPTIONS,
    FALLBACK_DEFAULTS,
    PACE_OPTIONS,
    PIT_OPTIONS,
    VERIFIED_BUILD_DESCS,
    VERIFIED_PACE_DESCS,
    VERIFIED_PIT_DESCS,
    DelegationTracker,
    build_live_lap_obs,
    build_pre_race_obs,
)
from self_jev.car_build_nn.builds import get_candidate_builds
from self_jev.car_build_nn.dataset import _load_frozen_driver
from self_jev.sim.design import COMPOUNDS
from self_jev.sim.env import RaceEnv
from self_jev.sim.physics import PhysicsConfig
from self_jev.sim.residual import assist_ers_mode, combine_residual_actions

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DRIVER = REPO_ROOT / "models" / "jev_rl_v7.mapdata.npz"
DEFAULT_CAL_PHYSICS = REPO_ROOT / "self_jev" / "sim" / "calibrated_params.json"

_ENV_TENSOR_ATTRS: tuple[str, ...] = (
    "vx", "vy", "yaw_rate", "x", "y", "yaw", "steer", "gear", "rpm",
    "prev_throttle", "prev_brake", "ers_soc_mj", "fuel_kg", "g_lat", "g_long",
    "tyre_temp_c", "tyre_wear", "front_sliding", "rear_sliding",
    "s", "n_lat", "heading_error", "trk_idx",
    "dist", "lap", "status", "compound", "compounds_used_mask", "wet_waiver_latched",
    "pit_state", "pit_t", "pit_stops", "pit_target_cmp", "pit_pending",
    "off_track", "lap_dirty", "wall_graze", "car_contact",
    "t_race", "lap_start_t", "best_lap_t", "finish_t", "tick_count",
    "ring_dist", "wing_factor", "grip_scale", "aero_scale",
)


def load_frozen_sim_assets(
    driver_path: Path = DEFAULT_DRIVER,
    device: torch.device | str = "cpu",
) -> tuple[torch.nn.Module, torch.Tensor, torch.Tensor, bool, PhysicsConfig, dict[str, Any]]:
    dev = torch.device(device)
    cfg = PhysicsConfig.from_json(DEFAULT_CAL_PHYSICS) if DEFAULT_CAL_PHYSICS.is_file() else PhysicsConfig()
    builds = get_candidate_builds()
    driver_model, obs_mean, obs_std, is_residual = _load_frozen_driver(driver_path, dev)
    return driver_model, obs_mean, obs_std, is_residual, cfg, builds


def snapshot_env(env: RaceEnv, world_idx: int = 0) -> dict[str, Any]:
    """Capture complete mutable state for a single world (C=1) plus torch RNG state."""
    i = int(world_idx)
    tensors: dict[str, torch.Tensor] = {}
    for attr in _ENV_TENSOR_ATTRS:
        val = getattr(env, attr)
        if attr == "ring_dist":
            tensors[attr] = val[:, i].detach().clone()
        else:
            tensors[attr] = val[i].detach().clone()
    w_state = {k: v[i].detach().clone() for k, v in env.w_state.items()}
    return {
        "tensors": tensors,
        "w_state": w_state,
        "rng_state": torch.get_rng_state().clone(),
    }


def restore_env_from_snapshots(env: RaceEnv, snaps: list[dict[str, Any]]) -> None:
    """Restore a list of N=env.W single-world snapshots into a batched RaceEnv (C=1)."""
    assert len(snaps) == env.W, f"Expected {env.W} snapshots, got {len(snaps)}"
    dev = env.device
    for attr in _ENV_TENSOR_ATTRS:
        if attr == "ring_dist":
            stacked = torch.stack([s["tensors"][attr] for s in snaps], dim=1).to(dev).clone()
        else:
            stacked = torch.stack([s["tensors"][attr] for s in snaps], dim=0).to(dev).clone()
        setattr(env, attr, stacked)
    w_keys = list(snaps[0]["w_state"].keys())
    for k in w_keys:
        env.w_state[k] = torch.stack([s["w_state"][k] for s in snaps], dim=0).to(dev).clone()
    env.wetness = env.w_state["wetness"]
    env.rain_intensity = env.w_state["rain_intensity"]
    env.track_temp_c = env.w_state["track_temp_c"]
    env.forecast_rain_prob = env.w_state["forecast_rain_prob"]
    torch.set_rng_state(snaps[0]["rng_state"].clone())


def restore_env_replicated(env: RaceEnv, snap: dict[str, Any]) -> None:
    restore_env_from_snapshots(env, [snap] * env.W)


def init_env_for_scenarios(
    track_id: str,
    scenarios: list[dict[str, Any]],
    build_car: Any,
    cfg: PhysicsConfig,
    device: torch.device,
    laps_total: int = 4,
) -> RaceEnv:
    torch.manual_seed(42)
    env = RaceEnv(
        track_id=track_id,
        worlds=len(scenarios),
        cars_per_world=1,
        laps_total=laps_total,
        phase="race",
        weather_regime="dry",
        car=build_car,
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
    for wi, sc in enumerate(scenarios):
        env.grip_scale[wi] = sc.get("grip_scale", 1.0)
        env.aero_scale[wi] = sc.get("aero_scale", 1.0)
    return env


@torch.inference_mode()
def step_driver_batch(
    env: RaceEnv,
    cur_modes: list[str],
    pit_latched: list[bool],
    pit_target_idx: list[int],
    driver_model: torch.nn.Module,
    obs_mean: torch.Tensor,
    obs_std: torch.Tensor,
    is_residual: bool,
) -> None:
    raw = env.get_obs(include_map_ext=False)
    x = torch.clamp(torch.nan_to_num((raw - obs_mean) / obs_std, nan=0.0, posinf=10.0, neginf=-10.0), -10.0, 10.0)
    mu, ers_logits, flag_logits, _ = driver_model(x)
    ers = torch.argmax(ers_logits, dim=-1)
    mu_adj = mu.clone()
    ovr = (flag_logits[:, 1] > 0.0).float()

    for wi, m in enumerate(cur_modes):
        if m == "push_max":
            mu_adj[wi, 0] = mu_adj[wi, 0] + 0.15
            if float(env.ers_soc_mj[wi].item()) > 0.5 and float(env.wetness[wi].item()) < 0.45:
                ers[wi] = 3
                ovr[wi] = 1.0
        elif m == "conserve_tyres":
            mu_adj[wi, 0] = mu_adj[wi, 0] - 0.08
            ers[wi] = 1
            ovr[wi] = 0.0
        elif m == "wet_caution":
            mu_adj[wi, 0] = mu_adj[wi, 0] - 0.15
            ers[wi] = 0
            ovr[wi] = 0.0

    if is_residual:
        thr, brk, st_cmd = combine_residual_actions(raw, mu_adj)
        bal_ers = assist_ers_mode(raw, ers, thr, brk)
        for wi, m in enumerate(cur_modes):
            if m == "balanced":
                ers[wi] = bal_ers[wi]
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
        "pit_request": torch.tensor([1.0 if p else 0.0 for p in pit_latched], device=env.device),
        "pit_compound": torch.tensor(pit_target_idx, device=env.device, dtype=torch.long),
    })


@torch.inference_mode()
def verify_snapshot_replay_invariance(
    sc: dict[str, Any],
    driver_model: torch.nn.Module,
    obs_mean: torch.Tensor,
    obs_std: torch.Tensor,
    is_residual: bool,
    cfg: PhysicsConfig,
    builds: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    env = init_env_for_scenarios(sc["track"], [sc], builds["dry_balanced"], cfg, device, laps_total=4)
    snap: dict[str, Any] | None = None
    for _ in range(30 * 800):
        if snap is None and int(env.lap[0].item()) == 1:
            snap = snapshot_env(env, 0)
        step_driver_batch(env, ["balanced"], [False], [0], driver_model, obs_mean, obs_std, is_residual)
        if int(env.status[0].item()) != 1:
            break
    orig_t = float(env.finish_t[0].item())
    orig_ticks = int(env.tick_count[0].item())
    orig_status = int(env.status[0].item())

    assert snap is not None, "Failed to reach lap 1 during replay invariance check"
    env2 = init_env_for_scenarios(sc["track"], [sc], builds["dry_balanced"], cfg, device, laps_total=4)
    restore_env_replicated(env2, snap)
    for _ in range(30 * 800):
        step_driver_batch(env2, ["balanced"], [False], [0], driver_model, obs_mean, obs_std, is_residual)
        if int(env2.status[0].item()) != 1:
            break
    rep_t = float(env2.finish_t[0].item())
    rep_ticks = int(env2.tick_count[0].item())
    rep_status = int(env2.status[0].item())
    passed = (orig_status == rep_status) and (orig_ticks == rep_ticks) and (abs(orig_t - rep_t) < 1e-6)
    return {
        "scenario_id": sc["scenario_id"],
        "orig_status": orig_status,
        "replay_status": rep_status,
        "orig_finish_t": round(orig_t, 6),
        "replay_finish_t": round(rep_t, 6),
        "abs_diff_s": round(abs(orig_t - rep_t), 8),
        "tick_diff": abs(orig_ticks - rep_ticks),
        "passed": passed,
    }


def rank_candidate_outcomes(
    outcomes: list[dict[str, Any]],
    time_tol_s: float = 0.50,
) -> dict[str, Any]:
    """Rank candidate actions under completion_then_time_v1."""
    fin_idxs = [i for i, o in enumerate(outcomes) if o["finished"] and o["finish_time_s"] is not None]
    if fin_idxs:
        best_t = min(float(outcomes[i]["finish_time_s"]) for i in fin_idxs)
        best_idx = min(fin_idxs, key=lambda i: float(outcomes[i]["finish_time_s"]))
        acceptable = [
            outcomes[i]["option"]
            for i in fin_idxs
            if float(outcomes[i]["finish_time_s"]) <= best_t + time_tol_s
        ]
        status = "resolved" if len(acceptable) == 1 else "tie_acceptable_set"
        return {
            "gold_index": best_idx,
            "gold_option_id": outcomes[best_idx]["option"],
            "acceptable_option_ids": acceptable,
            "best_finish_time_s": round(best_t, 4),
            "status": status,
        }
    best_dist = max(float(o["dist_m"]) for o in outcomes)
    best_idx = max(range(len(outcomes)), key=lambda i: float(outcomes[i]["dist_m"]))
    acceptable = [o["option"] for o in outcomes if float(o["dist_m"]) >= best_dist - 100.0]
    return {
        "gold_index": best_idx,
        "gold_option_id": outcomes[best_idx]["option"],
        "acceptable_option_ids": acceptable,
        "best_finish_time_s": None,
        "status": "all_failed",
    }


@torch.inference_mode()
def evaluate_pace_counterfactuals_batched(
    track_id: str,
    build_car: Any,
    items: list[dict[str, Any]],
    continuation_mode: str,
    driver_model: torch.nn.Module,
    obs_mean: torch.Tensor,
    obs_std: torch.Tensor,
    is_residual: bool,
    cfg: PhysicsConfig,
    device: torch.device,
    laps_total: int = 4,
    max_steps: int = 30 * 800,
) -> list[list[dict[str, Any]]]:
    """Evaluate all 4 pace options for M snapshots in ONE vectorized RaceEnv(worlds=M*4) pass."""
    if not items:
        return []
    K = len(PACE_OPTIONS)
    M = len(items)
    expanded_snaps: list[dict[str, Any]] = []
    init_modes: list[str] = []
    dec_laps: list[int] = []
    for it in items:
        for opt in PACE_OPTIONS:
            expanded_snaps.append(it["snap"])
            init_modes.append(opt)
            dec_laps.append(int(it["lap"]))

    env = RaceEnv(
        track_id=track_id, worlds=M * K, cars_per_world=1, laps_total=laps_total,
        phase="race", weather_regime="dry", car=build_car, cfg=cfg, device=device, domain_rand=False,
    )
    restore_env_from_snapshots(env, expanded_snaps)
    cur_modes = list(init_modes)
    pit_latched = [False] * (M * K)
    pit_target = [0] * (M * K)

    for step_i in range(max_steps):
        for wi in range(M * K):
            if int(env.lap[wi].item()) > dec_laps[wi]:
                cur_modes[wi] = continuation_mode
        step_driver_batch(env, cur_modes, pit_latched, pit_target, driver_model, obs_mean, obs_std, is_residual)
        if (step_i + 1) % 32 == 0 and (env.status != 1).all():
            break

    all_out: list[list[dict[str, Any]]] = []
    for m_i in range(M):
        sub = []
        for k_i, opt in enumerate(PACE_OPTIONS):
            wi = m_i * K + k_i
            st = int(env.status[wi].item())
            fin = st == 2
            ft = round(float(env.finish_t[wi].item()), 4) if fin and float(env.finish_t[wi].item()) > 0 else None
            sub.append({
                "option": opt,
                "finished": fin,
                "status": st,
                "finish_time_s": ft,
                "completed_laps": laps_total if fin else max(0, int(env.lap[wi].item())),
                "dist_m": round(float(env.dist[wi].item()), 1),
            })
        all_out.append(sub)
    return all_out


@torch.inference_mode()
def evaluate_pit_counterfactuals_batched(
    track_id: str,
    build_car: Any,
    items: list[dict[str, Any]],
    continuation_mode: str,
    driver_model: torch.nn.Module,
    obs_mean: torch.Tensor,
    obs_std: torch.Tensor,
    is_residual: bool,
    cfg: PhysicsConfig,
    device: torch.device,
    laps_total: int = 4,
    max_steps: int = 30 * 800,
) -> list[list[dict[str, Any]]]:
    """Evaluate all 3 pit options for M snapshots in ONE vectorized RaceEnv(worlds=M*3) pass."""
    if not items:
        return []
    K = len(PIT_OPTIONS)
    M = len(items)
    expanded_snaps: list[dict[str, Any]] = []
    pit_latched: list[bool] = []
    pit_target: list[int] = []
    for it in items:
        for k_i in range(K):
            expanded_snaps.append(it["snap"])
            pit_latched.append(k_i > 0)
            pit_target.append(0 if k_i <= 1 else 1)

    env = RaceEnv(
        track_id=track_id, worlds=M * K, cars_per_world=1, laps_total=laps_total,
        phase="race", weather_regime="dry", car=build_car, cfg=cfg, device=device, domain_rand=False,
    )
    restore_env_from_snapshots(env, expanded_snaps)
    cur_modes = [continuation_mode] * (M * K)
    init_pits = [int(env.pit_stops[wi].item()) for wi in range(M * K)]

    for step_i in range(max_steps):
        for wi in range(M * K):
            if int(env.pit_stops[wi].item()) > init_pits[wi]:
                pit_latched[wi] = False
        step_driver_batch(env, cur_modes, pit_latched, pit_target, driver_model, obs_mean, obs_std, is_residual)
        if (step_i + 1) % 32 == 0 and (env.status != 1).all():
            break

    all_out: list[list[dict[str, Any]]] = []
    for m_i in range(M):
        sub = []
        for k_i, opt in enumerate(PIT_OPTIONS):
            wi = m_i * K + k_i
            st = int(env.status[wi].item())
            fin = st == 2
            ft = round(float(env.finish_t[wi].item()), 4) if fin and float(env.finish_t[wi].item()) > 0 else None
            sub.append({
                "option": opt,
                "finished": fin,
                "status": st,
                "finish_time_s": ft,
                "pit_stops": int(env.pit_stops[wi].item()),
                "completed_laps": laps_total if fin else max(0, int(env.lap[wi].item())),
                "dist_m": round(float(env.dist[wi].item()), 1),
            })
        all_out.append(sub)
    return all_out


@torch.inference_mode()
def run_closed_loop_policy(
    scenarios: list[dict[str, Any]],
    decide_fn: Callable[[str, dict[str, Any], list[str], list[str]], tuple[str | None, bool, float | None]],
    driver_model: torch.nn.Module,
    obs_mean: torch.Tensor,
    obs_std: torch.Tensor,
    is_residual: bool,
    cfg: PhysicsConfig,
    builds: dict[str, Any],
    device: torch.device,
    pits_enabled: bool = False,
    confidence_threshold: float | None = None,
    laps_total: int = 4,
    max_steps: int = 30 * 800,
) -> list[dict[str, Any]]:
    by_track: dict[str, list[int]] = {}
    for idx, sc in enumerate(scenarios):
        by_track.setdefault(sc["track"], []).append(idx)

    results: list[dict[str, Any] | None] = [None] * len(scenarios)

    for track_id, track_idxs in by_track.items():
        trackers = {idx: DelegationTracker(confidence_threshold=confidence_threshold) for idx in track_idxs}
        chosen_builds: dict[int, str] = {}
        for idx in track_idxs:
            obs = build_pre_race_obs(scenarios[idx], laps_total=laps_total)
            prop, ok, conf = decide_fn("build", obs, BUILD_OPTIONS, VERIFIED_BUILD_DESCS)
            chosen_builds[idx] = trackers[idx].resolve("build", prop, BUILD_OPTIONS, parse_ok=ok, confidence=conf)

        by_build: dict[str, list[int]] = {}
        for idx in track_idxs:
            by_build.setdefault(chosen_builds[idx], []).append(idx)

        for b_name, idxs in by_build.items():
            sub_scs = [scenarios[i] for i in idxs]
            env = init_env_for_scenarios(track_id, sub_scs, builds[b_name], cfg, device, laps_total=laps_total)
            w_n = len(idxs)
            cur_modes = ["balanced"] * w_n
            pit_latched = [False] * w_n
            pit_target_idx = [0] * w_n
            prev_pit_stops = [0] * w_n
            last_decided_lap = [-1] * w_n
            prev_lap_start_t = [0.0] * w_n
            last_lap_time: list[float | None] = [None] * w_n
            mode_histories: list[list[str]] = [[] for _ in range(w_n)]
            pit_histories: list[list[str]] = [[] for _ in range(w_n)]
            off_ticks = [0] * w_n
            grazes = [0] * w_n

            for step_i in range(max_steps):
                for wi, orig_i in enumerate(idxs):
                    if int(env.status[wi].item()) != 1:
                        continue
                    cpits = int(env.pit_stops[wi].item())
                    if cpits > prev_pit_stops[wi]:
                        pit_latched[wi] = False
                        prev_pit_stops[wi] = cpits
                    clap = int(env.lap[wi].item())
                    if clap > last_decided_lap[wi] and clap < laps_total:
                        t_now = float(env.t_race[wi].item())
                        if clap > 0:
                            last_lap_time[wi] = t_now - prev_lap_start_t[wi]
                            prev_lap_start_t[wi] = t_now
                        last_decided_lap[wi] = clap
                        obs_pace = build_live_lap_obs(
                            sc=scenarios[orig_i],
                            chosen_build=b_name,
                            current_mode=cur_modes[wi],
                            lap_completed=clap,
                            laps_total=laps_total,
                            elapsed_s=t_now,
                            wetness=float(env.wetness[wi].item()),
                            rain_intensity=float(env.rain_intensity[wi].item()),
                            track_temp_c=float(env.track_temp_c[wi].item()),
                            tyre_compound=COMPOUNDS[int(env.compound[wi].item())],
                            tyre_wear=float(env.tyre_wear[wi].item()),
                            tyre_temp_c=float(env.tyre_temp_c[wi].item()),
                            ers_soc_mj=float(env.ers_soc_mj[wi].item()),
                            fuel_kg=float(env.fuel_kg[wi].item()),
                            pit_stops=cpits,
                            off_track_ticks_so_far=off_ticks[wi],
                            wall_grazes_so_far=grazes[wi],
                            last_lap_time_s=last_lap_time[wi],
                            decision_type="pace",
                        )
                        prop_m, ok_m, conf_m = decide_fn("pace", obs_pace, PACE_OPTIONS, VERIFIED_PACE_DESCS)
                        chosen_m = trackers[orig_i].resolve("pace", prop_m, PACE_OPTIONS, parse_ok=ok_m, confidence=conf_m)
                        cur_modes[wi] = chosen_m
                        mode_histories[wi].append(chosen_m)

                        if pits_enabled and clap in (1, 2) and not pit_latched[wi]:
                            obs_pit = dict(obs_pace, decision_type="pit", legal_options=list(PIT_OPTIONS))
                            prop_p, ok_p, conf_p = decide_fn("pit", obs_pit, PIT_OPTIONS, VERIFIED_PIT_DESCS)
                            chosen_p = trackers[orig_i].resolve("pit", prop_p, PIT_OPTIONS, parse_ok=ok_p, confidence=conf_p)
                            pit_histories[wi].append(chosen_p)
                            if chosen_p == "pit_soft":
                                pit_latched[wi] = True
                                pit_target_idx[wi] = 0
                            elif chosen_p == "pit_medium":
                                pit_latched[wi] = True
                                pit_target_idx[wi] = 1

                step_driver_batch(env, cur_modes, pit_latched, pit_target_idx, driver_model, obs_mean, obs_std, is_residual)
                for wi in range(w_n):
                    if int(env.status[wi].item()) == 1:
                        if int(env.off_track[wi].item()) > 0:
                            off_ticks[wi] += 1
                        if int(env.wall_graze[wi].item()) > 0:
                            grazes[wi] += 1
                if (step_i + 1) % 32 == 0 and (env.status != 1).all():
                    break

            for wi, orig_i in enumerate(idxs):
                sc = scenarios[orig_i]
                st0 = int(env.status[wi].item())
                fin = st0 == 2
                ft = round(float(env.finish_t[wi].item()), 4) if fin and float(env.finish_t[wi].item()) > 0 else None
                tr_recs = trackers[orig_i].records
                results[orig_i] = {
                    "scenario_id": sc["scenario_id"],
                    "cell_id": sc["cell_id"],
                    "track": sc["track"],
                    "weather_regime": sc["weather_regime"],
                    "seed": sc["seed"],
                    "chosen_build": b_name,
                    "mode_history": mode_histories[wi],
                    "pit_calls": pit_histories[wi],
                    "status": st0,
                    "finished": fin,
                    "completed_laps": laps_total if fin else max(0, int(env.lap[wi].item())),
                    "finish_time_s": ft,
                    "dist_m": round(float(env.dist[wi].item()), 1),
                    "pit_stops": int(env.pit_stops[wi].item()),
                    "delegated_calls": sum(1 for r in tr_recs if r.delegated),
                    "total_calls": len(tr_recs),
                    "illegal_proposed": sum(1 for r in tr_recs if r.delegation_reason == "illegal_action"),
                }
    return [r for r in results if r is not None]
