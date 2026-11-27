"""Clean-room batched PyTorch dynamic bicycle + tyre + ERS vehicle physics (v2)."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from self_jev.sim.design import CarParams

GRAVITY = 9.80665


@dataclass
class PhysicsConfig:
    """Calibratable physical parameters for open-loop & RL simulation."""

    ice_power_w: float = 550000.0
    ers_power_w: float = 350000.0
    ers_harvest_w: float = 120000.0
    override_power_w: float = 120000.0
    brake_force_max_n: float = 42000.0
    aero_drag_coeff: float = 0.85
    aero_downforce_coeff: float = 2.60
    aero_drs_drag_cut: float = 0.22
    mu_dry_soft: float = 1.72
    mu_dry_medium: float = 1.64
    mu_dry_hard: float = 1.56
    wet_grip_drop: float = 0.36
    cornering_stiffness: float = 85000.0
    max_steer_rad: float = 0.312
    yaw_damping: float = 2.5
    rolling_resist_c: float = 0.012
    fuel_burn_kg_per_mj: float = 0.00072
    wear_rate_per_s: float = 0.00055
    temp_heat_coeff: float = 0.085
    temp_cool_coeff: float = 0.045

    def to_dict(self) -> dict[str, float]:
        return asdict(self)

    @classmethod
    def from_json(cls, path: Path | str) -> "PhysicsConfig":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        valid = {k: float(v) for k, v in data.items() if hasattr(cls, k)}
        return cls(**valid)


def step_physics(
    state: dict[str, torch.Tensor],
    actions: dict[str, torch.Tensor],
    grade: torch.Tensor,
    aero_zone: torch.Tensor,
    wetness: torch.Tensor,
    track_temp_c: torch.Tensor,
    car: CarParams,
    cfg: PhysicsConfig,
    dt: float = 1.0 / 30.0,
    substeps: int = 4,
    bank_rad: torch.Tensor | None = None,
    wing_factor_t: torch.Tensor | None = None,
    grip_scale_t: torch.Tensor | None = None,
    aero_scale_t: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Advance batched vehicle state [N] by one decision step dt using clean-room bicycle dynamics."""
    sub_dt = dt / float(substeps)
    device = state["vx"].device

    vx = state["vx"]
    vy = state["vy"]
    yaw_rate = state["yaw_rate"]
    x = state["x"]
    y = state["y"]
    yaw = state["yaw"]
    soc = state["ers_soc_mj"]
    fuel = state["fuel_kg"]
    tyre_temp = state["tyre_temp_c"]
    tyre_wear = state["tyre_wear"]
    compound = state["compound"]

    throttle = torch.clamp(actions["throttle"], 0.0, 1.0)
    brake = torch.clamp(actions["brake"], 0.0, 1.0)
    steer_cmd = torch.clamp(actions["steer"], -1.0, 1.0)
    ers_mode = actions["ers_mode"]
    aero_cmd = actions["aero_mode"]
    ovr_cmd = actions.get("override", torch.zeros_like(vx))

    ratios = torch.tensor(car.gear_ratios, device=device, dtype=torch.float32)
    a_lf = car.wheelbase_m * (1.0 - car.weight_front_pct / 100.0)
    a_lr = car.wheelbase_m * (car.weight_front_pct / 100.0)
    iz = 1450.0
    bank_g = GRAVITY * torch.sin(bank_rad) if bank_rad is not None else 0.0

    mu_base = torch.where(
        compound == 0,
        cfg.mu_dry_soft,
        torch.where(compound == 1, cfg.mu_dry_medium, cfg.mu_dry_hard),
    )
    wear_mult = torch.clamp(1.0 - 0.12 * tyre_wear - 0.35 * torch.clamp(tyre_wear - 0.75, min=0.0) ** 2, 0.45, 1.0)
    wet_mult = torch.clamp(1.0 - cfg.wet_grip_drop * torch.pow(torch.clamp(wetness, 0.0, 1.0), 0.85), 0.45, 1.0)
    temp_opt = 82.0
    temp_mult = torch.clamp(1.0 - 0.00018 * (tyre_temp - temp_opt) ** 2, 0.75, 1.0)
    mu_eff = mu_base * wear_mult * wet_mult * temp_mult
    if grip_scale_t is not None:
        mu_eff = mu_eff * grip_scale_t

    active_aero = (aero_cmd > 0.5) & (aero_zone > 0.5)
    # Override disabled when track_wetness >= 0.5 (low_grip_wetness) or low SoC
    override_active = (ovr_cmd > 0.5) & (wetness < 0.5) & (soc > 0.3)

    wf = wing_factor_t if wing_factor_t is not None else ((car.front_wing + car.rear_wing) / 42.0)
    aero_s = aero_scale_t if aero_scale_t is not None else 1.0
    cd_area = cfg.aero_drag_coeff * (0.85 + 0.15 * wf) * aero_s * torch.where(active_aero, 1.0 - cfg.aero_drs_drag_cut, 1.0)
    cl_area = cfg.aero_downforce_coeff * (0.85 + 0.15 * wf) * aero_s * torch.where(active_aero, 0.88, 1.0)

    g_long_out = torch.zeros_like(vx)
    g_lat_out = torch.zeros_like(vx)
    front_slide_out = torch.zeros_like(vx)
    rear_slide_out = torch.zeros_like(vx)
    gear_out = torch.ones_like(vx)
    rpm_out = torch.zeros_like(vx)

    for _ in range(substeps):
        mass = car.dry_mass_kg + car.ballast_kg + fuel
        q_dyn = 0.5 * 1.225 * vx * torch.abs(vx)
        downforce = torch.clamp(cl_area * q_dyn, min=0.0)
        drag = cd_area * q_dyn
        fz_total = mass * GRAVITY + downforce
        fz_front = fz_total * (car.weight_front_pct / 100.0)
        fz_rear = fz_total - fz_front

        wheel_omega = torch.clamp(vx, min=0.0) / car.wheel_radius_m
        rpms_all = wheel_omega.unsqueeze(1) * ratios.unsqueeze(0) * car.final_drive * (60.0 / (2.0 * torch.pi))
        valid_gears = (rpms_all >= 7500.0) & (rpms_all <= 14500.0)
        gear_idx = torch.where(
            valid_gears.any(dim=1),
            valid_gears.float().argmax(dim=1),
            torch.clamp((rpms_all - 11000.0).abs().argmin(dim=1), 0, len(ratios) - 1),
        )
        rpm = torch.clamp(rpms_all.gather(1, gear_idx.unsqueeze(1)).squeeze(1), 2500.0, 15000.0)
        gear_out = (gear_idx + 1).float()
        rpm_out = rpm

        v_drive = torch.clamp(vx, min=4.0)
        ice_torque_factor = torch.clamp(1.0 - 0.12 * ((rpm - 12500.0) / 3500.0) ** 2, 0.55, 1.0)
        p_ice = cfg.ice_power_w * throttle * ice_torque_factor

        v_kph = vx * 3.6
        taper_norm = torch.clamp(1.0 - (v_kph - 290.0) / (345.0 - 290.0), 0.0, 1.0)
        taper_ovr = torch.clamp(1.0 - (v_kph - 337.5) / (355.0 - 337.5), 0.0, 1.0)
        taper_scale = torch.where(override_active, taper_ovr, taper_norm)

        can_deploy = soc > 0.02
        can_harvest = soc < 3.98
        p_ers = torch.zeros_like(vx)
        p_ers = torch.where((ers_mode == 3) & can_deploy, cfg.ers_power_w * throttle * taper_scale, p_ers)
        p_ers = torch.where((ers_mode == 2) & can_deploy, 0.5 * cfg.ers_power_w * throttle * taper_scale, p_ers)
        p_ers = torch.where((ers_mode == 0) & can_harvest & (throttle > 0.02), -cfg.ers_harvest_w * throttle, p_ers)
        p_ers = torch.where(override_active & can_deploy, p_ers + cfg.override_power_w * throttle * taper_scale, p_ers)

        f_drive = (p_ice + p_ers) / v_drive
        f_drive = torch.where(vx < 30.0, torch.minimum(f_drive, torch.full_like(f_drive, 1.2 * GRAVITY) * mass), f_drive)
        f_lift = torch.where((throttle <= 0.02) & (ers_mode == 0) & can_harvest & (vx > 0.5), 0.04 * float(car.ers_harvest_level) * GRAVITY * mass, torch.zeros_like(vx))
        f_brake = cfg.brake_force_max_n * brake * torch.where(vx > 0.2, 1.0, 0.0) + f_lift
        f_roll = cfg.rolling_resist_c * fz_total * torch.sign(vx)
        f_grade = mass * GRAVITY * grade

        delta = steer_cmd * cfg.max_steer_rad
        v_safe = torch.clamp(torch.abs(vx), min=2.5)
        alpha_f = torch.atan2(vy + a_lf * yaw_rate, v_safe) - delta
        alpha_r = torch.atan2(vy - a_lr * yaw_rate, v_safe)

        fy_f_lin = -cfg.cornering_stiffness * alpha_f
        fy_r_lin = -cfg.cornering_stiffness * alpha_r

        fy_f_max = mu_eff * fz_front
        fy_r_max = mu_eff * fz_rear
        fy_f = fy_f_max * torch.tanh(fy_f_lin / torch.clamp(fy_f_max, min=100.0))
        fy_r = fy_r_max * torch.tanh(fy_r_lin / torch.clamp(fy_r_max, min=100.0))

        front_slide_out = (torch.abs(fy_f_lin) > 0.92 * fy_f_max).float()
        rear_slide_out = (torch.abs(fy_r_lin) > 0.92 * fy_r_max).float()

        fx_raw = f_drive - f_brake - drag - f_roll - f_grade
        lat_util = torch.clamp(torch.abs(fy_f * torch.cos(delta) + fy_r) / torch.clamp(mu_eff * fz_total, min=500.0), 0.0, 0.99)
        fx_cap = torch.clamp(mu_eff * fz_total * torch.sqrt(1.0 - lat_util * lat_util), min=500.0)
        fx_net = torch.clamp(fx_raw, -fx_cap, fx_cap)

        ax = fx_net / mass + vy * yaw_rate
        ay = (fy_f * torch.cos(delta) + fy_r) / mass - vx * yaw_rate - bank_g
        r_dot = (a_lf * fy_f * torch.cos(delta) - a_lr * fy_r) / iz - cfg.yaw_damping * yaw_rate

        vx = torch.clamp(vx + ax * sub_dt, 0.0, 110.0)
        vy = torch.clamp(vy + ay * sub_dt, -25.0, 25.0)
        yaw_rate = torch.clamp(yaw_rate + r_dot * sub_dt, -4.5, 4.5)

        cos_y = torch.cos(yaw)
        sin_y = torch.sin(yaw)
        x = x + (vx * cos_y - vy * sin_y) * sub_dt
        y = y + (vx * sin_y + vy * cos_y) * sub_dt
        yaw = torch.atan2(torch.sin(yaw + yaw_rate * sub_dt), torch.cos(yaw + yaw_rate * sub_dt))

        soc = torch.clamp(soc - (p_ers * sub_dt) / 1e6, 0.0, 4.0)
        fuel = torch.clamp(fuel - (p_ice * sub_dt / 1e6) * cfg.fuel_burn_kg_per_mj, 0.0, 120.0)

        work_factor = (torch.abs(ax) + torch.abs(ay)) / GRAVITY
        tyre_wear = torch.clamp(tyre_wear + cfg.wear_rate_per_s * (0.4 + 0.3 * work_factor) * sub_dt, 0.0, 1.0)
        target_temp = track_temp_c + 45.0 + 14.0 * work_factor - 12.0 * wetness
        tyre_temp = tyre_temp + (cfg.temp_heat_coeff * work_factor - cfg.temp_cool_coeff) * (target_temp - tyre_temp) * sub_dt

        g_long_out = (fx_net / mass) / GRAVITY
        g_lat_out = ((fy_f * torch.cos(delta) + fy_r) / mass) / GRAVITY

    return {
        "vx": vx,
        "vy": vy,
        "yaw_rate": yaw_rate,
        "x": x,
        "y": y,
        "yaw": yaw,
        "steer": steer_cmd,
        "gear": gear_out,
        "rpm": rpm_out,
        "ers_soc_mj": soc,
        "fuel_kg": fuel,
        "tyre_temp_c": tyre_temp,
        "tyre_wear": tyre_wear,
        "g_long": g_long_out,
        "g_lat": g_lat_out,
        "front_sliding": front_slide_out,
        "rear_sliding": rear_slide_out,
        "prev_throttle": throttle,
        "prev_brake": brake,
        "override_active": override_active.float(),
    }
