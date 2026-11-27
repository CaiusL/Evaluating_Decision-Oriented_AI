"""Automated test suite for the clean-room self_jev.sim v2 racing simulator."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

from self_jev.bc_policy import extract_features
from self_jev.sim.design import load_design, make_rain_design, validate_design_limits
from self_jev.sim.env import RaceEnv
from self_jev.sim.eval_matrix import run_eval_matrix
from self_jev.sim.physics import PhysicsConfig, step_physics
from self_jev.sim.track import load_track
from self_jev.train_rl_sim import train_rl_sim


def test_track_frenet_roundtrip_and_map_ext() -> None:
    for track_id in ("parkland", "highland", "crossover"):
        trk = load_track(track_id)
        s_in = torch.linspace(5.0, trk.length_m - 5.0, 200)
        n_zero = torch.zeros_like(s_in)
        wx, wy, hdg = trk.frenet_to_world(s_in, n_zero)
        s_out, n_out, he_out, _ = trk.project(wx, wy, hdg, prev_idx=trk.s_to_index(s_in))
        ds = torch.minimum(torch.abs(s_out - s_in), trk.length_m - torch.abs(s_out - s_in))
        dn = torch.abs(n_out - n_zero)
        assert ds.max().item() < 1e-3
        assert dn.max().item() < 1e-3
        assert he_out.abs().max().item() < 1e-3

        pv = trk.preview_24(torch.tensor([trk.length_m - 10.0]))
        ext = trk.map_ext_25(torch.tensor([trk.length_m - 10.0]))
        assert pv.shape == (1, 120) and ext.shape == (1, 25)
        assert torch.isfinite(pv).all() and torch.isfinite(ext).all()


def test_physics_stability_banking_and_override() -> None:
    car = load_design()
    cfg = PhysicsConfig()
    N = 4
    state = {
        "vx": torch.full((N,), 40.0),
        "vy": torch.zeros(N),
        "yaw_rate": torch.zeros(N),
        "x": torch.zeros(N),
        "y": torch.zeros(N),
        "yaw": torch.zeros(N),
        "ers_soc_mj": torch.full((N,), 3.0),
        "fuel_kg": torch.full((N,), 9.0),
        "tyre_temp_c": torch.full((N,), 80.0),
        "tyre_wear": torch.zeros(N),
        "compound": torch.zeros(N, dtype=torch.long),
    }
    act = {
        "throttle": torch.ones(N),
        "brake": torch.zeros(N),
        "steer": torch.zeros(N),
        "ers_mode": torch.full((N,), 3, dtype=torch.long),
        "aero_mode": torch.ones(N),
        "override": torch.ones(N),
    }
    # Override active only when wetness < 0.5
    nxt = step_physics(
        state, act,
        grade=torch.zeros(N),
        aero_zone=torch.ones(N),
        wetness=torch.tensor([0.0, 0.2, 0.5, 0.9]),
        track_temp_c=torch.full((N,), 25.0),
        car=car, cfg=cfg,
        bank_rad=torch.full((N,), 0.05),
    )
    assert nxt["override_active"][0].item() == 1.0
    assert nxt["override_active"][1].item() == 1.0
    assert nxt["override_active"][2].item() == 0.0
    assert nxt["override_active"][3].item() == 0.0
    assert torch.isfinite(nxt["vx"]).all()


def test_design_limits_and_rain_design_selection() -> None:
    car = load_design()
    rain_car = make_rain_design(car)
    trk = load_track("parkland")
    assert validate_design_limits(car, trk.design_limits) == []
    assert rain_car.rear_wing > car.rear_wing

    env = RaceEnv("parkland", worlds=2, cars_per_world=1, weather_regime="wet", car=car, rain_car=rain_car)
    # Wet regime has forecast rain_prob 0.85 > 0.40 -> rain_design selected
    assert env.wing_factor[0].item() > (car.front_wing + car.rear_wing) / 42.0


def test_feature_parity_181_and_206_ext() -> None:
    env = RaceEnv("parkland", worlds=2, cars_per_world=10, weather_regime="mixed")
    obs_181 = env.get_obs(include_map_ext=False)
    obs_206 = env.get_obs(include_map_ext=True)
    assert obs_181.shape == (20, 181)
    assert obs_206.shape == (20, 206)
    for idx in range(20):
        ref = torch.from_numpy(extract_features(env.get_obs_dict(idx)))
        assert (obs_181[idx] - ref).abs().max().item() < 1e-5


def test_v2_sporting_rules_weather_ramp_pit_legality_and_timeout() -> None:
    # 1. Dynamic weather ramp + wet waiver latch only in race
    env_r = RaceEnv("parkland", worlds=1, cars_per_world=1, weather_regime="ramp_wet", phase="race")
    assert env_r.wetness[0].item() == 0.0
    assert not env_r.wet_waiver_latched[0].item()
    env_r.t_race[0] = 200.0
    env_r.step({"throttle": torch.tensor([0.5]), "brake": torch.tensor([0.0]), "steer": torch.tensor([0.0]),
                "ers_mode": torch.tensor([1]), "aero_mode": torch.tensor([0.0])})
    assert env_r.wetness[0].item() > 0.5
    assert env_r.wet_waiver_latched[0].item()

    # 2. Non-latching pit_request: False cancels pending request; True through entry pits and switches compound
    env_p = RaceEnv("parkland", worlds=1, cars_per_world=1, weather_regime="dry")
    assert not env_p.dry_compound_legal()[0].item()  # Only 1 compound used so far
    env_p.s[0] = env_p.track.pit_entry_s - 5.0
    wx, wy, hd = env_p.track.frenet_to_world(env_p.s, env_p.n_lat)
    env_p.x[0], env_p.y[0], env_p.yaw[0] = wx[0], wy[0], hd[0]
    env_p.vx[0] = 50.0
    env_p.trk_idx[0] = env_p.track.s_to_index(env_p.s)[0]
    # Request True then False before entry -> cancels
    env_p.step({"throttle": torch.tensor([0.5]), "brake": torch.tensor([0.0]), "steer": torch.tensor([0.0]),
                "ers_mode": torch.tensor([1]), "aero_mode": torch.tensor([0.0]),
                "pit_request": torch.tensor([1.0]), "pit_compound": torch.tensor([1])})
    assert env_p.pit_pending[0].item()
    env_p.step({"throttle": torch.tensor([0.5]), "brake": torch.tensor([0.0]), "steer": torch.tensor([0.0]),
                "ers_mode": torch.tensor([1]), "aero_mode": torch.tensor([0.0]),
                "pit_request": torch.tensor([0.0]), "pit_compound": torch.tensor([-1])})
    assert not env_p.pit_pending[0].item()

    # Now hold pit_request=True across entry_s
    env_p.s[0] = env_p.track.pit_entry_s - 1.0
    wx, wy, hd = env_p.track.frenet_to_world(env_p.s, env_p.n_lat)
    env_p.x[0], env_p.y[0], env_p.yaw[0] = wx[0], wy[0], hd[0]
    env_p.trk_idx[0] = env_p.track.s_to_index(env_p.s)[0]
    pit_act = {"throttle": torch.tensor([0.5]), "brake": torch.tensor([0.0]), "steer": torch.tensor([0.0]),
               "ers_mode": torch.tensor([1]), "aero_mode": torch.tensor([0.0]),
               "pit_request": torch.tensor([1.0]), "pit_compound": torch.tensor([1])}
    env_p.step(pit_act)
    assert env_p.pit_state[0].item() == 1
    for _ in range(30 * 37):
        env_p.step(pit_act)
    assert env_p.pit_stops[0].item() == 1
    assert env_p.compound[0].item() == 1
    assert env_p.dry_compound_legal()[0].item()

    # 3. Absolute time limit -> status = 3
    env_t = RaceEnv("parkland", worlds=1, cars_per_world=1, weather_regime="dry", end_time_s=1.0)
    for _ in range(35):
        env_t.step({"throttle": torch.tensor([0.5]), "brake": torch.tensor([0.0]), "steer": torch.tensor([0.0]),
                    "ers_mode": torch.tensor([1]), "aero_mode": torch.tensor([0.0])})
    assert env_t.status[0].item() == 3


def test_v2_ppo_and_eval_matrix_smoke(tmp_path: Path) -> None:
    out_npz = tmp_path / "test_rl_v2.mapdata.npz"
    train_rl_sim(
        init_npz=Path("models/bc_61bb13c7.mapdata.npz"),
        out_npz=out_npz,
        worlds_per_track=4,
        updates=2,
        rollout_steps=16,
        mb_size=256,
        device_str="cpu",
    )
    assert out_npz.is_file()
    out_eval = tmp_path / "eval_matrix.json"
    summary = run_eval_matrix(out_npz, out_eval, max_steps=15, device_str="cpu")
    assert summary["total_cells"] == 30
    assert out_eval.is_file()
