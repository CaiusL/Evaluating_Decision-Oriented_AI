"""Generate clean multi-track, multi-weather teacher rollouts in RaceEnv and train bc_clean_sim.mapdata.npz."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from self_jev.sim.design import load_design, make_rain_design
from self_jev.sim.env import RaceEnv
from self_jev.sim.physics import PhysicsConfig
from self_jev.train_rl_sim import ActorCritic


def teacher_policy(env: RaceEnv) -> dict[str, torch.Tensor]:
    """Clean, wall-safe braking-envelope + centerline + traffic + ERS teacher across all tracks & weather."""
    dev = env.device
    wet_scale = torch.clamp(1.0 - 0.35 * env.wetness, 0.55, 1.0)
    ay_budget = 9.0 * wet_scale
    a_brake = 12.0 * wet_scale

    v_lim = torch.full((env.N,), 82.0, device=dev)
    for k in range(0, 25):
        c_k, _, _, _, _ = env.track.sample_at_s(env.s + 15.0 * k)
        v_corner = torch.clamp(torch.sqrt(ay_budget / torch.clamp(c_k.abs(), min=1e-4)), 8.0, 82.0)
        d_ahead = max(0.0, 15.0 * k - 20.0)
        v_allow = torch.sqrt(v_corner * v_corner + 2.0 * a_brake * d_ahead)
        v_lim = torch.minimum(v_lim, v_allow)
    target_v = v_lim
    near_curv = 0.5 * env.track.sample_at_s(env.s + 5.0)[0] + 0.5 * env.track.sample_at_s(env.s + 18.0)[0]

    obs_181 = env.get_obs(include_map_ext=False)
    rv = obs_181[:, 142:178].view(env.N, 9, 4)
    gap_m = rv[:, :, 0] * 100.0
    dn_m = rv[:, :, 1] * 10.0
    rv_v = rv[:, :, 2] * 100.0
    present = rv[:, :, 3] > 0.5
    ahead_in_lane = present & (gap_m > 0.0) & (gap_m < 35.0) & (dn_m.abs() < 2.2)
    if ahead_in_lane.any():
        min_rv_v = torch.where(ahead_in_lane, rv_v, torch.full_like(rv_v, 100.0)).min(dim=1).values
        target_v = torch.minimum(target_v, min_rv_v * 0.92)

    spd = torch.hypot(env.vx, env.vy)
    err = target_v - spd
    throttle = torch.clamp(0.25 * err + 0.10, 0.0, 1.0)
    brake = torch.clamp(-0.22 * err, 0.0, 1.0)
    throttle = torch.where(brake > 0.05, torch.zeros_like(throttle), throttle)

    # Speed-scaled Stanley + curvature feedforward + yaw-rate damping (stable at 80+ m/s)
    delta_ff = torch.atan(3.6 * near_curv)
    delta_fb = -0.65 * env.heading_error - torch.atan2(0.9 * env.n_lat, spd + 12.0) - 0.12 * env.yaw_rate
    steer = torch.clamp((delta_ff + delta_fb) / 0.312, -1.0, 1.0)

    # ERS deploy on full-throttle straights when SoC > 0.25, harvest under braking
    ers_mode = torch.ones(env.N, device=dev, dtype=torch.long)
    ers_mode = torch.where((throttle > 0.85) & (env.ers_soc_mj > 0.25), torch.full_like(ers_mode, 3), ers_mode)
    ers_mode = torch.where((brake > 0.2) & (env.ers_soc_mj < 3.8), torch.zeros_like(ers_mode), ers_mode)

    _, _, _, _, aero_zone = env.track.sample_at_s(env.s)
    aero_mode = ((aero_zone > 0.5) & (throttle > 0.8)).float()
    override = ((throttle > 0.9) & (env.wetness < 0.5) & (env.ers_soc_mj > 0.5)).float()
    aid = ((env.phase == "race") & (env.wetness < 0.5) & (env.lap >= 1) & (throttle > 0.9)).float()

    pit_req = (
        (env.phase == "race")
        & (env.lap == 1)
        & (~env.wet_waiver_latched)
        & (env.pit_stops == 0)
        & (env.s >= env.track.pit_entry_s - 220.0)
    )
    return {
        "throttle": throttle,
        "brake": brake,
        "steer": steer,
        "ers_mode": ers_mode,
        "aero_mode": aero_mode,
        "override": override,
        "aid": aid,
        "pit_request": pit_req.float(),
        "pit_compound": torch.ones(env.N, device=dev, dtype=torch.long),
    }


def pretrain_clean_bc(
    out_npz: Path = Path("models/bc_clean_sim.mapdata.npz"),
    epochs: int = 25,
    steps_per_track: int = 600,
    worlds_per_track: int = 4,
) -> dict[str, float]:
    car = load_design()
    rain_car = make_rain_design(car)
    cal_path = Path(__file__).resolve().parent / "calibrated_params.json"
    cfg = PhysicsConfig.from_json(cal_path) if cal_path.is_file() else PhysicsConfig()

    xs, cs, es, fs = [], [], [], []
    for tid in ("parkland", "highland", "crossover"):
        for phase, C in (("qualifying", 1), ("race", 10)):
            env = RaceEnv(tid, worlds=worlds_per_track, cars_per_world=C, laps_total=2,
                          phase=phase, weather_regime="mixed", car=car, rain_car=rain_car, cfg=cfg)
            # Half rolling starts so we cover the whole circuit quickly
            env.reset_worlds(torch.ones(worlds_per_track, dtype=torch.bool), rolling_frac=0.5)
            for step in range(steps_per_track):
                obs = env.get_obs(include_map_ext=False)
                act = teacher_policy(env)
                active = env.status == 1
                if active.any() and (step % 2 == 0):
                    xs.append(obs[active].cpu().numpy())
                    cs.append(torch.stack([act["throttle"], act["brake"], act["steer"]], dim=1)[active].cpu().numpy())
                    es.append(act["ers_mode"][active].cpu().numpy())
                    fs.append(torch.stack([act["aero_mode"], act["override"], act["aid"]], dim=1)[active].cpu().numpy())
                env.step(act)
                if (env.status != 1).all():
                    env.reset_worlds(torch.ones(worlds_per_track, dtype=torch.bool), rolling_frac=0.5)

    x_all = np.concatenate(xs, axis=0).astype(np.float32)
    c_all = np.concatenate(cs, axis=0).astype(np.float32)
    e_all = np.concatenate(es, axis=0).astype(np.int64)
    f_all = np.concatenate(fs, axis=0).astype(np.float32)

    mean = x_all.mean(axis=0)
    std = np.where(x_all.std(axis=0) < 1e-4, 1.0, x_all.std(axis=0)).astype(np.float32)
    x_norm = np.clip((x_all - mean) / std, -10.0, 10.0).astype(np.float32)

    torch.manual_seed(42)
    model = ActorCritic(x_all.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_norm), torch.from_numpy(c_all), torch.from_numpy(e_all), torch.from_numpy(f_all)),
        batch_size=512, shuffle=True,
    )
    l1 = nn.SmoothL1Loss(beta=0.05)
    ce = nn.CrossEntropyLoss()
    bce = nn.BCEWithLogitsLoss()

    for _ in range(epochs):
        model.train()
        for bx, bc_t, be_t, bf_t in loader:
            pc, pe, pf, _ = model(bx)
            loss = l1(pc, bc_t) + 0.3 * ce(pe, be_t) + 0.2 * bce(pf, bf_t)
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    save_dict = {"obs_mean": mean, "obs_std": std}
    for k, v in model.state_dict().items():
        if k.startswith(("net.", "cont_head.", "ers_head.", "flag_head.")):
            save_dict[k.replace(".", "__")] = v.detach().cpu().numpy().astype(np.float32)
    np.savez(out_npz, **save_dict)

    return {"rows": int(len(x_all)), "weights": str(out_npz)}


if __name__ == "__main__":
    res = pretrain_clean_bc()
    print(json.dumps(res, indent=2))
