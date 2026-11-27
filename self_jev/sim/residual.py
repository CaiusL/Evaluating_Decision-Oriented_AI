"""Residual RL base controller (smooth braking-envelope + sharp hairpin Stanley) on 181-dim obs."""
from __future__ import annotations

import torch


def base_controls_from_obs181(obs_181: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute (v_base, steer_base) from [N, 181] observations with zero simulator internals."""
    dev = obs_181.device
    speed = obs_181[:, 0]
    n_lat = obs_181[:, 3]
    heading_error = obs_181[:, 4]
    yaw_rate = obs_181[:, 5]
    wetness = obs_181[:, 178]

    # Calibrated against live AGX Formula dry & wet telemetry (ce025466 / 61bb13c7)
    wet_scale = torch.clamp(1.0 - 0.36 * wetness, 0.56, 1.0)
    ay_budget = 13.0 * wet_scale
    a_brake = 15.5 * wet_scale

    # Smooth preview braking envelope (speed-proportional lookahead buffer eliminates sawtooth braking)
    pv = obs_181[:, 22:142].view(obs_181.shape[0], 24, 5)
    curvs = pv[:, :, 0] / 100.0  # [N, 24]
    raw_dists = torch.arange(1, 25, device=dev, dtype=torch.float32).unsqueeze(0) * 15.0  # [1, 24]
    buf_m = torch.clamp(0.45 * speed + 18.0, 18.0, 48.0).unsqueeze(1)  # [N, 1]
    dists = torch.clamp(raw_dists - buf_m, min=0.0)

    v_corner = torch.clamp(torch.sqrt(ay_budget.unsqueeze(1) / torch.clamp(curvs.abs(), min=1e-4)), 9.0, 94.0)
    v_allow = torch.sqrt(v_corner * v_corner + 2.0 * a_brake.unsqueeze(1) * dists)
    curv_now = yaw_rate.abs() / torch.clamp(speed, min=5.0)
    v_apex_now = torch.clamp(torch.sqrt(ay_budget / torch.clamp(curv_now, min=1e-4)), 9.0, 94.0)
    v_base = torch.clamp(torch.minimum(v_allow.min(dim=1).values, v_apex_now), 9.0, 94.0)

    # Traffic speed cap for in-lane cars ahead
    rv = obs_181[:, 142:178].view(obs_181.shape[0], 9, 4)
    gap_m = rv[:, :, 0] * 100.0
    dn_m = rv[:, :, 1] * 10.0
    rv_v = rv[:, :, 2] * 100.0
    present = rv[:, :, 3] > 0.5
    ahead = present & (gap_m > 0.0) & (gap_m < 32.0) & (dn_m.abs() < 2.2)
    min_rv_v = torch.where(ahead, rv_v, torch.full_like(rv_v, 100.0)).min(dim=1).values
    v_base = torch.minimum(v_base, min_rv_v * 0.92)

    # Hairpin-safe Stanley steering: blend local yaw curvature with near preview at low speed
    # so the car does not over-cut before tight hairpins (fixes Parkland Turn 1 s=1124m off-track)
    blend = torch.clamp((speed - 12.0) / 25.0, 0.25, 1.0)
    curv_signed_now = yaw_rate / torch.clamp(speed, min=6.0)
    eff_curv = blend * curvs[:, 0] + (1.0 - blend) * curv_signed_now
    delta_ff = torch.atan(3.6 * eff_curv)
    delta_fb = -1.05 * heading_error - torch.atan2(1.45 * n_lat, speed + 8.0) - 0.14 * yaw_rate
    steer_base = torch.clamp((delta_ff + delta_fb) / 0.312, -1.0, 1.0)
    return v_base, steer_base


def combine_residual_actions(
    obs_181: torch.Tensor,
    raw_res: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Combine [N, 3] network residuals with base controller -> (throttle, brake, steer)."""
    v_base, steer_base = base_controls_from_obs181(obs_181)
    speed = obs_181[:, 0]

    # Full physical headroom for 500-update 384->384->192 RL: +/-30% speed, +/-0.14 steer, +/-0.12 brake
    v_mult = 1.0 + 0.30 * torch.tanh(raw_res[:, 0])
    res_steer = torch.tanh(raw_res[:, 1]) * 0.14
    res_brk = torch.tanh(raw_res[:, 2]) * 0.12

    target_v = torch.clamp(v_base * v_mult, 7.0, 100.0)
    err = target_v - speed
    throttle = torch.clamp(0.28 * err + 0.12, 0.0, 1.0)
    brake = torch.clamp(-0.26 * err + res_brk, 0.0, 1.0)
    throttle = torch.where(brake > 0.04, torch.zeros_like(throttle), throttle)
    steer = torch.clamp(steer_base + res_steer, -1.0, 1.0)
    return throttle, brake, steer


def assist_ers_mode(
    obs_181: torch.Tensor,
    ers_mode: torch.Tensor,
    throttle: torch.Tensor,
    brake: torch.Tensor,
) -> torch.Tensor:
    """Match agent.py straight-line ERS deployment & braking harvest assist."""
    speed = obs_181[:, 0]
    soc = obs_181[:, 11]
    deploy = torch.where(soc > 1.8, torch.full_like(ers_mode, 3), torch.full_like(ers_mode, 2))
    out = torch.where((throttle > 0.85) & (speed > 35.0) & (soc > 0.45) & (ers_mode <= 1), deploy, ers_mode)
    out = torch.where((brake > 0.15) & (soc < 3.85), torch.zeros_like(out), out)
    return out
