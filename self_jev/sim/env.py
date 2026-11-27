"""Clean-room GPU-vectorized multi-phase race environment (v2)."""
from __future__ import annotations

import math
from typing import Any

import torch

from self_jev.sim.design import COMPOUNDS, CarParams, load_design, make_rain_design, validate_design_limits
from self_jev.sim.physics import PhysicsConfig, step_physics
from self_jev.sim.track import TrackTable, load_track
from self_jev.sim.weather import build_weather_tensors, update_weather


class RaceEnv:
    """Batched [worlds, C] racing environment with v2 sporting rules & 181/206-dim observations."""

    def __init__(
        self,
        track_id: str = "parkland",
        worlds: int = 64,
        cars_per_world: int = 10,
        laps_total: int = 4,
        phase: str = "race",
        weather_regime: str = "mixed",
        wetness_palette: tuple[float, ...] | None = None,
        car: CarParams | None = None,
        rain_car: CarParams | None = None,
        cfg: PhysicsConfig | None = None,
        device: torch.device | str = "cpu",
        dt: float = 1.0 / 30.0,
        end_time_s: float | None = None,
        domain_rand: bool = False,
    ) -> None:
        self.device = torch.device(device)
        self.track: TrackTable = load_track(track_id, device=self.device)
        self.W = int(worlds)
        self.C = int(cars_per_world)
        self.N = self.W * self.C
        self.laps_total = int(laps_total)
        self.phase = phase
        self.weather_regime = weather_regime
        self.wetness_palette = wetness_palette
        self.dt = float(dt)
        self.domain_rand = bool(domain_rand)
        self.car = car or load_design()
        self.rain_car = rain_car or make_rain_design(self.car)
        self.cfg = cfg or PhysicsConfig()

        # Validate both books against per-track design_limits
        self.design_errors = validate_design_limits(self.car, self.track.design_limits)
        if end_time_s is None:
            t_trig = 7200.0 * 4.0 / (math.floor(305000.0 / self.track.length_m) + 1.0)
            self.end_time_s = t_trig + 120.0
        else:
            self.end_time_s = float(end_time_s)

        self._allocate_buffers()
        self.reset_all()

    def _allocate_buffers(self) -> None:
        N, dev = self.N, self.device
        self.vx = torch.zeros(N, device=dev, dtype=torch.float32)
        self.vy = torch.zeros(N, device=dev, dtype=torch.float32)
        self.yaw_rate = torch.zeros(N, device=dev, dtype=torch.float32)
        self.x = torch.zeros(N, device=dev, dtype=torch.float32)
        self.y = torch.zeros(N, device=dev, dtype=torch.float32)
        self.yaw = torch.zeros(N, device=dev, dtype=torch.float32)
        self.steer = torch.zeros(N, device=dev, dtype=torch.float32)
        self.gear = torch.ones(N, device=dev, dtype=torch.float32)
        self.rpm = torch.zeros(N, device=dev, dtype=torch.float32)
        self.prev_throttle = torch.zeros(N, device=dev, dtype=torch.float32)
        self.prev_brake = torch.zeros(N, device=dev, dtype=torch.float32)
        self.ers_soc_mj = torch.full((N,), 3.0, device=dev, dtype=torch.float32)
        self.fuel_kg = torch.full((N,), self.car.fuel_load_kg, device=dev, dtype=torch.float32)
        self.g_lat = torch.zeros(N, device=dev, dtype=torch.float32)
        self.g_long = torch.zeros(N, device=dev, dtype=torch.float32)
        self.tyre_temp_c = torch.full((N,), 75.0, device=dev, dtype=torch.float32)
        self.tyre_wear = torch.zeros(N, device=dev, dtype=torch.float32)
        self.front_sliding = torch.zeros(N, device=dev, dtype=torch.float32)
        self.rear_sliding = torch.zeros(N, device=dev, dtype=torch.float32)

        self.s = torch.zeros(N, device=dev, dtype=torch.float32)
        self.n_lat = torch.zeros(N, device=dev, dtype=torch.float32)
        self.heading_error = torch.zeros(N, device=dev, dtype=torch.float32)
        self.trk_idx = torch.zeros(N, device=dev, dtype=torch.long)

        self.dist = torch.zeros(N, device=dev, dtype=torch.float32)
        self.lap = torch.zeros(N, device=dev, dtype=torch.long)
        # status: 1=racing, 2=finished, 3=unfinished(time_limit), 4=stalled, 5=wall_retired
        self.status = torch.ones(N, device=dev, dtype=torch.long)
        self.compound = torch.full((N,), self.car.start_compound_idx, device=dev, dtype=torch.long)
        self.compounds_used_mask = torch.zeros(N, device=dev, dtype=torch.long)
        self.wet_waiver_latched = torch.zeros(N, device=dev, dtype=torch.bool)
        self.pit_state = torch.zeros(N, device=dev, dtype=torch.long)
        self.pit_t = torch.zeros(N, device=dev, dtype=torch.float32)
        self.pit_stops = torch.zeros(N, device=dev, dtype=torch.long)
        self.pit_target_cmp = torch.full((N,), -1, device=dev, dtype=torch.long)
        self.pit_pending = torch.zeros(N, device=dev, dtype=torch.bool)

        self.off_track = torch.zeros(N, device=dev, dtype=torch.long)
        self.lap_dirty = torch.zeros(N, device=dev, dtype=torch.long)
        self.wall_graze = torch.zeros(N, device=dev, dtype=torch.long)
        self.car_contact = torch.zeros(N, device=dev, dtype=torch.long)
        self.t_race = torch.zeros(N, device=dev, dtype=torch.float32)
        self.lap_start_t = torch.zeros(N, device=dev, dtype=torch.float32)
        self.best_lap_t = torch.zeros(N, device=dev, dtype=torch.float32)
        self.finish_t = torch.zeros(N, device=dev, dtype=torch.float32)
        self.tick_count = torch.zeros(N, device=dev, dtype=torch.long)

        self.ring_dist = torch.full((30, N), -1e9, device=dev, dtype=torch.float32)

        # Weather & design-book tensors
        self.w_state = build_weather_tensors(self.W, self.C, self.weather_regime, dev)
        self.wetness = self.w_state["wetness"]
        self.rain_intensity = self.w_state["rain_intensity"]
        self.track_temp_c = self.w_state["track_temp_c"]
        self.forecast_rain_prob = self.w_state["forecast_rain_prob"]
        self.wing_factor = torch.full((N,), (self.car.front_wing + self.car.rear_wing) / 42.0, device=dev)
        self.grip_scale = torch.ones(N, device=dev, dtype=torch.float32)
        self.aero_scale = torch.ones(N, device=dev, dtype=torch.float32)

    def reset_all(self) -> None:
        self.reset_worlds(torch.ones(self.W, device=self.device, dtype=torch.bool))

    def reset_worlds(self, world_mask: torch.Tensor, rolling_frac: float = 0.0) -> None:
        if world_mask.dtype != torch.bool:
            world_mask = world_mask.bool()
        if not world_mask.any():
            return
        car_mask = world_mask.unsqueeze(1).expand(self.W, self.C).reshape(self.N)

        grid = self.track.start_grid
        s_init = torch.zeros(self.C, device=self.device, dtype=torch.float32)
        n_init = torch.zeros(self.C, device=self.device, dtype=torch.float32)
        x_init = torch.zeros(self.C, device=self.device, dtype=torch.float32)
        y_init = torch.zeros(self.C, device=self.device, dtype=torch.float32)
        h_init = torch.zeros(self.C, device=self.device, dtype=torch.float32)
        for j in range(self.C):
            g = grid[j % len(grid)]
            s_init[j] = float(g["s_m"])
            n_init[j] = float(g["n_m"])
            x_init[j] = float(g["x_m"])
            y_init[j] = float(g["y_m"])
            h_init[j] = float(g["heading_rad"])

        s_full = s_init.unsqueeze(0).expand(self.W, self.C).reshape(self.N)
        n_full = n_init.unsqueeze(0).expand(self.W, self.C).reshape(self.N)
        x_full = x_init.unsqueeze(0).expand(self.W, self.C).reshape(self.N)
        y_full = y_init.unsqueeze(0).expand(self.W, self.C).reshape(self.N)
        h_full = h_init.unsqueeze(0).expand(self.W, self.C).reshape(self.N)
        vx_full = torch.zeros(self.N, device=self.device, dtype=torch.float32)
        t_init = torch.zeros(self.N, device=self.device, dtype=torch.float32)
        dist_full = s_full - self.track.length_m

        if rolling_frac > 0.0:
            roll_w = world_mask & (torch.rand(self.W, device=self.device) < rolling_frac)
            if roll_w.any():
                roll_c = roll_w.unsqueeze(1).expand(self.W, self.C).reshape(self.N)
                base_s = torch.rand(self.W, device=self.device) * (self.track.length_m - 120.0) + 40.0
                car_off = torch.arange(self.C, device=self.device, dtype=torch.float32) * 16.0
                s_roll = torch.remainder(base_s.unsqueeze(1) - car_off.unsqueeze(0), self.track.length_m).reshape(self.N)
                n_roll = ((torch.arange(self.C, device=self.device, dtype=torch.float32) % 2) * 2.0 - 1.0) * 1.8
                n_roll = n_roll.unsqueeze(0).expand(self.W, self.C).reshape(self.N)
                rx, ry, rh = self.track.frenet_to_world(s_roll, n_roll)
                s_full = torch.where(roll_c, s_roll, s_full)
                n_full = torch.where(roll_c, n_roll, n_full)
                x_full = torch.where(roll_c, rx, x_full)
                y_full = torch.where(roll_c, ry, y_full)
                h_full = torch.where(roll_c, rh, h_full)
                vx_full = torch.where(roll_c, 45.0, vx_full)
                t_init = torch.where(roll_c, 15.0, t_init)
                dist_full = torch.where(roll_c, s_roll, dist_full)

        idx_full = self.track.s_to_index(s_full)

        # Override weather if explicit wetness_palette was passed (for deterministic tests)
        if self.wetness_palette is not None:
            w_vals = torch.tensor(
                [self.wetness_palette[w % len(self.wetness_palette)] for w in range(self.W)],
                device=self.device, dtype=torch.float32,
            )
            wet_full = w_vals.unsqueeze(1).expand(self.W, self.C).reshape(self.N)
            self.w_state["wet_start"] = torch.where(car_mask, wet_full, self.w_state["wet_start"])
            self.w_state["wet_end"] = torch.where(car_mask, wet_full, self.w_state["wet_end"])
            self.w_state["forecast_rain_prob"] = torch.where(car_mask, wet_full, self.w_state["forecast_rain_prob"])

        wet_0 = self.w_state["wet_start"]
        self.wetness = torch.where(car_mask, wet_0, self.wetness)
        self.rain_intensity = torch.where(car_mask, wet_0 * 0.5, self.rain_intensity)
        self.track_temp_c = torch.where(car_mask, 30.0 - 8.0 * wet_0, self.track_temp_c)
        self.forecast_rain_prob = self.w_state["forecast_rain_prob"]

        # Select rain_design when frozen pre-race rain_probability > 0.40
        use_rain_book = self.forecast_rain_prob > 0.40
        wf_dry = (self.car.front_wing + self.car.rear_wing) / 42.0
        wf_rain = (self.rain_car.front_wing + self.rain_car.rear_wing) / 42.0
        wf_sel = torch.where(use_rain_book, wf_rain, wf_dry)
        self.wing_factor = torch.where(car_mask, wf_sel, self.wing_factor)
        if self.domain_rand:
            gs = (0.92 + 0.16 * torch.rand(self.W, device=self.device)).unsqueeze(1).expand(self.W, self.C).reshape(self.N)
            as_ = (0.92 + 0.16 * torch.rand(self.W, device=self.device)).unsqueeze(1).expand(self.W, self.C).reshape(self.N)
            self.grip_scale = torch.where(car_mask, gs, self.grip_scale)
            self.aero_scale = torch.where(car_mask, as_, self.aero_scale)
        else:
            self.grip_scale = torch.where(car_mask, 1.0, self.grip_scale)
            self.aero_scale = torch.where(car_mask, 1.0, self.aero_scale)

        self.vx = torch.where(car_mask, vx_full, self.vx)
        self.vy = torch.where(car_mask, 0.0, self.vy)
        self.yaw_rate = torch.where(car_mask, 0.0, self.yaw_rate)
        self.x = torch.where(car_mask, x_full, self.x)
        self.y = torch.where(car_mask, y_full, self.y)
        self.yaw = torch.where(car_mask, h_full, self.yaw)
        self.steer = torch.where(car_mask, 0.0, self.steer)
        self.gear = torch.where(car_mask, 1.0, self.gear)
        self.rpm = torch.where(car_mask, 0.0, self.rpm)
        self.prev_throttle = torch.where(car_mask, 0.0, self.prev_throttle)
        self.prev_brake = torch.where(car_mask, 0.0, self.prev_brake)
        self.ers_soc_mj = torch.where(car_mask, 3.0, self.ers_soc_mj)
        self.fuel_kg = torch.where(car_mask, self.car.fuel_load_kg, self.fuel_kg)
        self.g_lat = torch.where(car_mask, 0.0, self.g_lat)
        self.g_long = torch.where(car_mask, 0.0, self.g_long)
        self.tyre_temp_c = torch.where(car_mask, 75.0, self.tyre_temp_c)
        self.tyre_wear = torch.where(car_mask, 0.0, self.tyre_wear)
        self.front_sliding = torch.where(car_mask, 0.0, self.front_sliding)
        self.rear_sliding = torch.where(car_mask, 0.0, self.rear_sliding)

        self.s = torch.where(car_mask, s_full, self.s)
        self.n_lat = torch.where(car_mask, n_full, self.n_lat)
        self.heading_error = torch.where(car_mask, 0.0, self.heading_error)
        self.trk_idx = torch.where(car_mask, idx_full, self.trk_idx)

        self.dist = torch.where(car_mask, dist_full, self.dist)
        self.lap = torch.where(car_mask, 0, self.lap)
        self.status = torch.where(car_mask, 1, self.status)
        self.compound = torch.where(car_mask, self.car.start_compound_idx, self.compound)
        self.compounds_used_mask = torch.where(car_mask, 1 << self.car.start_compound_idx, self.compounds_used_mask)
        self.wet_waiver_latched = torch.where(car_mask, False, self.wet_waiver_latched)
        self.pit_state = torch.where(car_mask, 0, self.pit_state)
        self.pit_t = torch.where(car_mask, 0.0, self.pit_t)
        self.pit_stops = torch.where(car_mask, 0, self.pit_stops)
        self.pit_target_cmp = torch.where(car_mask, -1, self.pit_target_cmp)
        self.pit_pending = torch.where(car_mask, False, self.pit_pending)

        self.off_track = torch.where(car_mask, 0, self.off_track)
        self.lap_dirty = torch.where(car_mask, 0, self.lap_dirty)
        self.wall_graze = torch.where(car_mask, 0, self.wall_graze)
        self.car_contact = torch.where(car_mask, 0, self.car_contact)
        self.t_race = torch.where(car_mask, t_init, self.t_race)
        self.lap_start_t = torch.where(car_mask, 0.0, self.lap_start_t)
        self.best_lap_t = torch.where(car_mask, 0.0, self.best_lap_t)
        self.finish_t = torch.where(car_mask, 0.0, self.finish_t)
        self.tick_count = torch.where(car_mask, 0, self.tick_count)
        self.ring_dist[:, car_mask] = -1e9

    def _compute_rivals_36(self) -> torch.Tensor:
        W, C, dev = self.W, self.C, self.device
        if C <= 1:
            empty = torch.tensor([5.0, 0.0, 0.0, 0.0], device=dev).view(1, 1, 4).expand(self.N, 9, 4)
            return empty.reshape(self.N, 36)

        d_w = self.dist.view(W, C)
        n_w = self.n_lat.view(W, C)
        v_w = torch.sqrt(self.vx ** 2 + self.vy ** 2).view(W, C)
        act_w = (self.status == 1).view(W, C)

        L = self.track.length_m
        raw_gap = d_w.unsqueeze(1) - d_w.unsqueeze(2)
        gap = torch.remainder(raw_gap + L * 0.5, L) - L * 0.5
        dn = n_w.unsqueeze(1) - n_w.unsqueeze(2)
        rv_spd = v_w.unsqueeze(1).expand(W, C, C)

        eye = torch.eye(C, device=dev, dtype=torch.bool).unsqueeze(0)
        valid = act_w.unsqueeze(1) & (~eye)
        tie = torch.arange(C, device=dev, dtype=torch.float32).view(1, 1, C) * 1e-4
        abs_gap = torch.where(valid, torch.abs(gap) + tie, torch.full_like(gap, 1e9))

        k_take = min(9, C - 1)
        _, top_idx = torch.topk(abs_gap, k=k_take, dim=2, largest=False)

        sel_gap = gap.gather(2, top_idx)
        sel_dn = dn.gather(2, top_idx)
        sel_spd = rv_spd.gather(2, top_idx)
        sel_valid = valid.gather(2, top_idx)

        feat = torch.stack([
            torch.where(sel_valid, torch.clamp(sel_gap / 100.0, -5.0, 5.0), torch.full_like(sel_gap, 5.0)),
            torch.where(sel_valid, torch.clamp(sel_dn / 10.0, -2.0, 2.0), torch.zeros_like(sel_dn)),
            torch.where(sel_valid, sel_spd / 100.0, torch.zeros_like(sel_spd)),
            sel_valid.float(),
        ], dim=3)

        if k_take < 9:
            pad = torch.tensor([5.0, 0.0, 0.0, 0.0], device=dev).view(1, 1, 1, 4).expand(W, C, 9 - k_take, 4)
            feat = torch.cat([feat, pad], dim=2)
        return feat.reshape(self.N, 36)

    def get_obs(self, include_map_ext: bool = False) -> torch.Tensor:
        """Return [N, 181] (or [N, 206] with mapdata extensions) float32 tensor."""
        speed = torch.sqrt(self.vx ** 2 + self.vy ** 2)
        is_race = torch.full((self.N,), 1.0 if self.phase == "race" else 0.0, device=self.device)
        aid_avail = (self.phase == "race") & (self.wetness < 0.5) & (self.lap >= 1)
        ovr_avail = (self.ers_soc_mj > 0.3) & (self.wetness < 0.5)

        self_22 = torch.stack([
            speed,
            self.vx,
            self.vy,
            self.n_lat,
            self.heading_error,
            self.yaw_rate,
            self.steer,
            self.gear,
            self.rpm / 10000.0,
            self.prev_throttle,
            self.prev_brake,
            self.ers_soc_mj,
            self.fuel_kg,
            self.g_lat,
            self.g_long,
            self.tyre_temp_c / 100.0,
            self.tyre_wear,
            self.front_sliding,
            self.rear_sliding,
            aid_avail.float(),
            ovr_avail.float(),
            is_race,
        ], dim=1)

        preview_120 = self.track.preview_24(self.s)
        rivals_36 = self._compute_rivals_36()
        weather_3 = torch.stack([
            self.wetness,
            self.rain_intensity,
            self.track_temp_c / 40.0,
        ], dim=1)

        parts = [self_22, preview_120, rivals_36, weather_3]
        if include_map_ext:
            parts.append(self.track.map_ext_25(self.s))
        vec = torch.cat(parts, dim=1)
        return torch.nan_to_num(vec, nan=0.0, posinf=5.0, neginf=-5.0)

    def get_obs_181(self) -> torch.Tensor:
        return self.get_obs(include_map_ext=False)

    def _resolve_car_contacts(self) -> None:
        """Detect pairwise car-to-car overlaps within each world and emit car_contact flag."""
        self.car_contact.zero_()
        if self.C <= 1:
            return
        W, C, dev = self.W, self.C, self.device
        active = ((self.status == 1) & (self.pit_state == 0)).view(W, C)
        xw = self.x.view(W, C)
        yw = self.y.view(W, C)
        dx = xw.unsqueeze(2) - xw.unsqueeze(1)
        dy = yw.unsqueeze(2) - yw.unsqueeze(1)
        d2 = dx * dx + dy * dy
        eye = torch.eye(C, device=dev, dtype=torch.bool).unsqueeze(0)
        pair_act = active.unsqueeze(2) & active.unsqueeze(1) & (~eye)
        hit = pair_act & (d2 < 4.5)  # ~2.12m center-to-center overlap
        self.car_contact = hit.any(dim=2).reshape(self.N).long()

    def dry_compound_legal(self) -> torch.Tensor:
        """True if wet waiver latched during race OR >= 2 distinct dry compounds used on track."""
        m = self.compounds_used_mask
        n_used = (m & 1) + ((m >> 1) & 1) + ((m >> 2) & 1)
        return self.wet_waiver_latched | (n_used >= 2)

    def classified(self) -> torch.Tensor:
        """True if completed_laps >= floor(0.9 * winner_completed_laps) and winner completed >= 1 lap."""
        laps_w = self.lap.view(self.W, self.C)
        max_laps = laps_w.max(dim=1, keepdim=True).values
        req = torch.floor(0.9 * max_laps.float()).long()
        return ((max_laps > 0) & (laps_w >= req)).reshape(self.N)

    def step(self, actions: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        _curv, grade, wl, wr, aero_zone = self.track.sample_at_s(self.s)
        bank_rad = self.track.sample_bank_at_s(self.s)

        # Update dynamic weather if not overridden by fixed palette
        if self.wetness_palette is None:
            update_weather(self.w_state, self.t_race)
            self.wetness = self.w_state["wetness"]
            self.rain_intensity = self.w_state["rain_intensity"]
            self.track_temp_c = self.w_state["track_temp_c"]

        # Latch wet-race waiver ONLY when phase == "race" and track_wetness >= 0.5
        if self.phase == "race":
            self.wet_waiver_latched = self.wet_waiver_latched | ((self.status == 1) & (self.wetness >= 0.5))

        active = (self.status == 1) & (self.pit_state == 0)
        s_old = self.s.clone()

        phys_state = {
            "vx": self.vx, "vy": self.vy, "yaw_rate": self.yaw_rate,
            "x": self.x, "y": self.y, "yaw": self.yaw,
            "ers_soc_mj": self.ers_soc_mj, "fuel_kg": self.fuel_kg,
            "tyre_temp_c": self.tyre_temp_c, "tyre_wear": self.tyre_wear,
            "compound": self.compound,
        }
        nxt = step_physics(
            phys_state, actions, grade, aero_zone, self.wetness, self.track_temp_c,
            self.car, self.cfg, dt=self.dt, substeps=4,
            bank_rad=bank_rad, wing_factor_t=self.wing_factor,
            grip_scale_t=self.grip_scale, aero_scale_t=self.aero_scale,
        )

        self.vx = torch.where(active, nxt["vx"], self.vx)
        self.vy = torch.where(active, nxt["vy"], self.vy)
        self.yaw_rate = torch.where(active, nxt["yaw_rate"], self.yaw_rate)
        self.x = torch.where(active, nxt["x"], self.x)
        self.y = torch.where(active, nxt["y"], self.y)
        self.yaw = torch.where(active, nxt["yaw"], self.yaw)
        self.steer = torch.where(active, nxt["steer"], self.steer)
        self.gear = torch.where(active, nxt["gear"], self.gear)
        self.rpm = torch.where(active, nxt["rpm"], self.rpm)
        self.prev_throttle = torch.where(active, nxt["prev_throttle"], self.prev_throttle)
        self.prev_brake = torch.where(active, nxt["prev_brake"], self.prev_brake)
        self.ers_soc_mj = torch.where(active, nxt["ers_soc_mj"], self.ers_soc_mj)
        self.fuel_kg = torch.where(active, nxt["fuel_kg"], self.fuel_kg)
        self.tyre_temp_c = torch.where(active, nxt["tyre_temp_c"], self.tyre_temp_c)
        self.tyre_wear = torch.where(active, nxt["tyre_wear"], self.tyre_wear)
        self.g_long = torch.where(active, nxt["g_long"], self.g_long)
        self.g_lat = torch.where(active, nxt["g_lat"], self.g_lat)
        self.front_sliding = torch.where(active, nxt["front_sliding"], self.front_sliding)
        self.rear_sliding = torch.where(active, nxt["rear_sliding"], self.rear_sliding)

        s_new, n_new, he_new, idx_new = self.track.project(self.x, self.y, self.yaw, prev_idx=self.trk_idx)
        self.s = torch.where(active, s_new, self.s)
        self.n_lat = torch.where(active, n_new, self.n_lat)
        self.heading_error = torch.where(active, he_new, self.heading_error)
        self.trk_idx = torch.where(active, idx_new, self.trk_idx)

        racing = self.status == 1
        self.t_race = torch.where(racing, self.t_race + self.dt, self.t_race)
        self.tick_count = torch.where(racing, self.tick_count + 1, self.tick_count)

        # Off-track & wall contact
        is_off = active & ((self.n_lat > wl) | (self.n_lat < -wr))
        self.off_track = torch.where(active, is_off.long(), self.off_track)
        self.lap_dirty = torch.where(is_off, 1, self.lap_dirty)

        self.wall_graze.zero_()
        wall_margin = 3.0
        hit_left = active & (self.n_lat > wl + wall_margin)
        hit_right = active & (self.n_lat < -(wr + wall_margin))
        hit_wall = hit_left | hit_right
        v_lat_wall = torch.abs(self.vx * torch.sin(self.heading_error) + self.vy * torch.cos(self.heading_error))
        fatal = hit_wall & (v_lat_wall >= 12.0)
        graze = hit_wall & (~fatal)
        self.wall_graze = graze.long()
        self.status = torch.where(fatal, 5, self.status)
        self.vx = torch.where(fatal, 0.0, torch.where(graze, self.vx * 0.5, self.vx))
        self.vy = torch.where(fatal, 0.0, torch.where(graze, self.vy * 0.5, self.vy))
        n_clamped = torch.where(hit_left, wl + wall_margin, -(wr + wall_margin))
        self.n_lat = torch.where(graze, n_clamped, self.n_lat)
        wx_c, wy_c, _ = self.track.frenet_to_world(self.s, self.n_lat)
        self.x = torch.where(graze, wx_c, self.x)
        self.y = torch.where(graze, wy_c, self.y)

        self._resolve_car_contacts()

        still_on_track = (self.status == 1) & (self.pit_state == 0)
        ds = self.s - s_old
        L = self.track.length_m
        ds = torch.where(ds < -0.5 * L, ds + L, ds)
        ds = torch.where(ds > 0.5 * L, ds - L, ds)
        self.dist = torch.where(still_on_track, self.dist + ds, self.dist)

        # Record compound used on track only when making forward on-track progress
        prog_on_track = still_on_track & (ds > 0.05)
        self.compounds_used_mask = torch.where(
            prog_on_track,
            self.compounds_used_mask | (1 << self.compound),
            self.compounds_used_mask,
        )

        crossed = still_on_track & (self.dist >= (self.lap.float() + 1.0) * L)
        lap_dt = self.t_race - self.lap_start_t
        clean_fast = crossed & (self.lap_dirty == 0) & ((self.best_lap_t <= 0.0) | (lap_dt < self.best_lap_t))
        self.best_lap_t = torch.where(clean_fast, lap_dt, self.best_lap_t)
        self.lap_start_t = torch.where(crossed, self.t_race, self.lap_start_t)
        self.lap_dirty = torch.where(crossed, 0, self.lap_dirty)
        self.lap = torch.where(crossed, self.lap + 1, self.lap)
        finished = crossed & (self.lap >= self.laps_total)
        self.status = torch.where(finished, 2, self.status)
        self.finish_t = torch.where(finished, self.t_race, self.finish_t)

        # Non-latching pit_request: False cancels pending request; True must stay True until entry_s
        pit_req = actions.get("pit_request", torch.zeros(self.N, device=self.device)) > 0.5
        pit_cmp = actions.get("pit_compound", torch.full((self.N,), -1, device=self.device, dtype=torch.long))
        self.pit_pending = pit_req & still_on_track
        self.pit_target_cmp = torch.where(pit_req & (pit_cmp >= 0), pit_cmp, self.pit_target_cmp)

        enter_pit = (
            still_on_track & (self.status == 1) & self.pit_pending
            & (s_old < self.track.pit_entry_s) & (self.s >= self.track.pit_entry_s) & (ds > 0)
        )
        self.pit_state = torch.where(enter_pit, 1, self.pit_state)
        self.pit_t = torch.where(enter_pit, 0.0, self.pit_t)
        self.off_track = torch.where(enter_pit, 0, self.off_track)

        in_pit = (self.status == 1) & (self.pit_state != 0) & (~enter_pit)
        self.pit_t = torch.where(in_pit, self.pit_t + self.dt, self.pit_t)
        pit_exit = in_pit & (self.pit_t >= 35.766)
        self.pit_state = torch.where(pit_exit, 0, self.pit_state)
        self.pit_stops = torch.where(pit_exit, self.pit_stops + 1, self.pit_stops)
        new_c = torch.where(self.pit_target_cmp >= 0, self.pit_target_cmp, self.compound)
        self.compound = torch.where(pit_exit, new_c, self.compound)
        self.tyre_wear = torch.where(pit_exit, 0.0, self.tyre_wear)
        self.tyre_temp_c = torch.where(pit_exit, 72.0, self.tyre_temp_c)
        ex_s = torch.full((self.N,), self.track.pit_exit_s, device=self.device)
        ex_n = torch.zeros(self.N, device=self.device)
        ex_x, ex_y, ex_h = self.track.frenet_to_world(ex_s, ex_n)
        self.s = torch.where(pit_exit, ex_s, self.s)
        self.n_lat = torch.where(pit_exit, 0.0, self.n_lat)
        self.x = torch.where(pit_exit, ex_x, self.x)
        self.y = torch.where(pit_exit, ex_y, self.y)
        self.yaw = torch.where(pit_exit, ex_h, self.yaw)
        self.vx = torch.where(pit_exit, self.track.pit_speed_mps, self.vx)
        self.vy = torch.where(pit_exit, 0.0, self.vy)
        self.trk_idx = torch.where(pit_exit, self.track.s_to_index(ex_s), self.trk_idx)

        # Stall rule (branchless GPU gather/scatter across 30s ring buffer)
        sample_ring = (self.status == 1) & ((self.tick_count % 30) == 0)
        slot_t = (self.tick_count // 30) % 30  # [N]
        old_d = self.ring_dist.gather(0, slot_t.unsqueeze(0)).squeeze(0)
        stalled = sample_ring & (self.t_race > 40.0) & (self.pit_state == 0) & (old_d > -1e8) & ((self.dist - old_d) < 10.0)
        self.status = torch.where(stalled, 4, self.status)
        new_ring = torch.where(sample_ring, torch.where(self.pit_state != 0, torch.full_like(self.dist, -1e9), self.dist), old_d)
        self.ring_dist.scatter_(0, slot_t.unsqueeze(0), new_ring.unsqueeze(0))

        # Absolute time limit -> status = 3 (unfinished)
        timed_out = (self.status == 1) & (self.t_race >= self.end_time_s)
        self.status = torch.where(timed_out, 3, self.status)

        return {
            "status": self.status,
            "dist": self.dist,
            "lap": self.lap,
            "off_track": self.off_track,
            "wall_graze": self.wall_graze,
            "car_contact": self.car_contact,
        }

    # ---------- Single-car dict API ----------

    def get_obs_dict(self, car_idx: int = 0) -> dict[str, Any]:
        i = int(car_idx)
        s_val = float(self.s[i].item())
        offsets = torch.arange(1, 25, device=self.device, dtype=torch.float32) * 15.0
        curv, grade, wl, wr, zone = self.track.sample_at_s(s_val + offsets)
        pv_rows = [
            [
                float((k + 1) * 15.0),
                float(curv[k].item()),
                float(grade[k].item()),
                float(wl[k].item()),
                float(wr[k].item()),
                int(round(zone[k].item())),
            ]
            for k in range(24)
        ]

        w_idx = i // self.C
        c_base = w_idx * self.C
        L = self.track.length_m
        d_me = float(self.dist[i].item())
        n_me = float(self.n_lat[i].item())
        rivals = []
        for j in range(self.C):
            rj = c_base + j
            if rj == i:
                continue
            gap = ((float(self.dist[rj].item()) - d_me + 0.5 * L) % L) - 0.5 * L
            rivals.append({
                "seat": j,
                "gap_m": gap,
                "dn": float(self.n_lat[rj].item()) - n_me,
                "speed_mps": float(torch.hypot(self.vx[rj], self.vy[rj]).item()),
                "lap": int(self.lap[rj].item()),
                "status": "racing" if int(self.status[rj].item()) == 1 else "retired",
            })

        speed = float(torch.hypot(self.vx[i], self.vy[i]).item())
        t_temp = float(self.tyre_temp_c[i].item())
        t_wear = float(self.tyre_wear[i].item())
        wet = float(self.wetness[i].item())
        fc_prob = float(self.forecast_rain_prob[i].item())
        return {
            "game_id": "racing",
            "seat": i % self.C,
            "phase": self.phase,
            "track_length_m": L,
            "laps_total": self.laps_total,
            "self": {
                "speed_mps": speed,
                "vx": float(self.vx[i].item()),
                "vy": float(self.vy[i].item()),
                "n": n_me,
                "s": s_val,
                "lap": int(self.lap[i].item()),
                "heading_error": float(self.heading_error[i].item()),
                "yaw_rate": float(self.yaw_rate[i].item()),
                "steer": float(self.steer[i].item()),
                "gear": int(round(self.gear[i].item())),
                "rpm": float(self.rpm[i].item()),
                "throttle": float(self.prev_throttle[i].item()),
                "brake": float(self.prev_brake[i].item()),
                "ers_soc_mj": float(self.ers_soc_mj[i].item()),
                "fuel_kg": float(self.fuel_kg[i].item()),
                "g_lat": float(self.g_lat[i].item()),
                "g_long": float(self.g_long[i].item()),
                "tyre_temp_c": [t_temp] * 4,
                "tyre_wear": [t_wear] * 4,
                "front_sliding": bool(self.front_sliding[i].item() > 0.5),
                "rear_sliding": bool(self.rear_sliding[i].item() > 0.5),
                "aid_available": bool(self.phase == "race" and wet < 0.5 and int(self.lap[i].item()) >= 1),
                "override_available": bool(float(self.ers_soc_mj[i].item()) > 0.3 and wet < 0.5),
                "pit_stops": int(self.pit_stops[i].item()),
                "compound": COMPOUNDS[int(self.compound[i].item())],
                "status": "racing" if int(self.status[i].item()) == 1 else "retired",
            },
            "preview": {
                "columns": ["distance_m", "curvature_1pm", "grade", "half_width_left_m", "half_width_right_m", "aero_zone"],
                "rows": pv_rows,
            },
            "rivals": rivals,
            "weather": {
                "track_wetness": wet,
                "rain_intensity": float(self.rain_intensity[i].item()),
                "track_temp_c": float(self.track_temp_c[i].item()),
                "condition": "wet" if wet >= 0.5 else ("damp" if wet > 0.05 else "dry"),
            },
            "forecast": {
                "rain_probability": fc_prob,
            },
            "limits": {
                "recharge_mj": 8.5,
                "soc_window_mj": 4.0,
            },
        }

    def step_dict(self, action_dict: dict[str, Any], car_idx: int = 0) -> dict[str, torch.Tensor]:
        act_t = {
            "throttle": torch.zeros(self.N, device=self.device),
            "brake": torch.zeros(self.N, device=self.device),
            "steer": torch.zeros(self.N, device=self.device),
            "ers_mode": torch.ones(self.N, device=self.device, dtype=torch.long),
            "aero_mode": torch.zeros(self.N, device=self.device),
            "override": torch.zeros(self.N, device=self.device),
            "aid": torch.zeros(self.N, device=self.device),
            "pit_request": torch.zeros(self.N, device=self.device),
            "pit_compound": torch.full((self.N,), -1, device=self.device, dtype=torch.long),
        }
        i = int(car_idx)
        act_t["throttle"][i] = float(action_dict.get("throttle", 0.0))
        act_t["brake"][i] = float(action_dict.get("brake", 0.0))
        act_t["steer"][i] = float(action_dict.get("steer", 0.0))
        act_t["ers_mode"][i] = int(action_dict.get("ers_mode", 1))
        act_t["aero_mode"][i] = float(action_dict.get("aero_mode", 0))
        act_t["override"][i] = 1.0 if action_dict.get("override") else 0.0
        act_t["aid"][i] = 1.0 if action_dict.get("aid") else 0.0
        act_t["pit_request"][i] = 1.0 if action_dict.get("pit_request") else 0.0
        cmp_name = action_dict.get("pit_compound")
        if cmp_name in COMPOUNDS:
            act_t["pit_compound"][i] = COMPOUNDS.index(cmp_name)
        return self.step(act_t)
