"""Open-loop physics calibration against real AGX Formula telemetry logs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from self_jev.sim.design import COMPOUNDS, load_design
from self_jev.sim.physics import PhysicsConfig, step_physics
from self_jev.sim.track import load_track


def load_rollout_segments(
    jsonl_path: Path,
    max_rows: int = 2400,
    seq_len: int = 20,
) -> list[dict[str, Any]]:
    """Load contiguous race-phase segments from a training export .jsonl."""
    rows: list[dict[str, Any]] = []
    with jsonl_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            obj = json.loads(line)
            if "observation" not in obj:
                continue
            obs = obj["observation"]
            if obs.get("phase") != "race":
                continue
            s = obs["self"]
            if s.get("status") != "racing" or s.get("pit_state", 0) != 0:
                continue
            if float(s.get("speed_mps", 0.0)) < 5.0:
                continue
            rows.append(obj)
            if len(rows) >= max_rows:
                break

    segments: list[dict[str, Any]] = []
    for start in range(0, len(rows) - seq_len - 1, seq_len):
        chunk = rows[start : start + seq_len + 1]
        # Ensure contiguous ticks (dt ~ 0.0333s)
        t0 = float(chunk[0]["observation"]["time"])
        t1 = float(chunk[-1]["observation"]["time"])
        if abs((t1 - t0) - seq_len / 30.0) > 0.15:
            continue
        segments.append({"rows": chunk})
    return segments


def evaluate_config(
    segments: list[dict[str, Any]],
    track,
    car,
    cfg: PhysicsConfig,
    horizon: int = 20,
) -> dict[str, float]:
    """Evaluate 1-step and horizon-step open-loop MAE on speed, g_long, g_lat, yaw_rate."""
    if not segments:
        return {"speed_mae_1": 0.0, "speed_mae_20": 0.0, "g_long_mae_1": 0.0, "g_lat_mae_1": 0.0}

    dev = track.device
    B = len(segments)

    def _get(idx: int, key: str, default: float = 0.0) -> torch.Tensor:
        return torch.tensor(
            [float(seg["rows"][idx]["observation"]["self"].get(key, default)) for seg in segments],
            device=dev,
            dtype=torch.float32,
        )

    def _get_w(idx: int, key: str, default: float = 0.0) -> torch.Tensor:
        return torch.tensor(
            [float(seg["rows"][idx]["observation"]["weather"].get(key, default)) for seg in segments],
            device=dev,
            dtype=torch.float32,
        )

    def _get_act(idx: int, key: str, default: float = 0.0) -> torch.Tensor:
        return torch.tensor(
            [float(seg["rows"][idx]["action"].get(key, default)) for seg in segments],
            device=dev,
            dtype=torch.float32,
        )

    def _get_mean_list(idx: int, key: str, default: float = 0.0) -> torch.Tensor:
        vals = []
        for seg in segments:
            v = seg["rows"][idx]["observation"]["self"].get(key, default)
            vals.append(float(np.mean(v)) if isinstance(v, list) else float(v))
        return torch.tensor(vals, device=dev, dtype=torch.float32)

    cmp_idx = torch.tensor(
        [
            COMPOUNDS.index(seg["rows"][0]["observation"]["self"].get("compound", "soft"))
            if seg["rows"][0]["observation"]["self"].get("compound", "soft") in COMPOUNDS
            else 0
            for seg in segments
        ],
        device=dev,
        dtype=torch.long,
    )

    state = {
        "vx": _get(0, "vx"),
        "vy": _get(0, "vy"),
        "yaw_rate": _get(0, "yaw_rate"),
        "x": _get(0, "x"),
        "y": _get(0, "y"),
        "yaw": _get(0, "yaw"),
        "ers_soc_mj": _get(0, "ers_soc_mj", 3.0),
        "fuel_kg": _get(0, "fuel_kg", 9.0),
        "tyre_temp_c": _get_mean_list(0, "tyre_temp_c", 75.0),
        "tyre_wear": _get_mean_list(0, "tyre_wear", 0.0),
        "compound": cmp_idx,
    }
    s_cur = _get(0, "s")

    err_spd_1, err_gl_1, err_ga_1 = 0.0, 0.0, 0.0
    for step_i in range(horizon):
        _c, grade, _wl, _wr, zone = track.sample_at_s(s_cur)
        act = {
            "throttle": _get_act(step_i, "throttle"),
            "brake": _get_act(step_i, "brake"),
            "steer": _get_act(step_i, "steer"),
            "ers_mode": _get_act(step_i, "ers_mode", 1).long(),
            "aero_mode": _get_act(step_i, "aero_mode", 0),
        }
        nxt = step_physics(
            state, act, grade, zone,
            _get_w(step_i, "track_wetness"),
            _get_w(step_i, "track_temp_c", 25.0),
            car, cfg, dt=1.0 / 30.0, substeps=4,
        )
        for k in ("vx", "vy", "yaw_rate", "x", "y", "yaw", "ers_soc_mj", "fuel_kg", "tyre_temp_c", "tyre_wear"):
            state[k] = nxt[k]
        s_cur = _get(step_i + 1, "s")
        if step_i == 0:
            spd_pred = torch.hypot(state["vx"], state["vy"])
            err_spd_1 = float((spd_pred - _get(1, "speed_mps")).abs().mean().item())
            err_gl_1 = float((nxt["g_long"] - _get(1, "g_long")).abs().mean().item())
            err_ga_1 = float((nxt["g_lat"] - _get(1, "g_lat")).abs().mean().item())

    spd_20 = torch.hypot(state["vx"], state["vy"])
    err_spd_20 = float((spd_20 - _get(horizon, "speed_mps")).abs().mean().item())
    return {
        "speed_mae_1": round(err_spd_1, 4),
        "speed_mae_20": round(err_spd_20, 4),
        "g_long_mae_1": round(err_gl_1, 4),
        "g_lat_mae_1": round(err_ga_1, 4),
    }


def calibrate(
    wet_jsonl: Path,
    damp_jsonl: Path,
    out_params: Path,
    out_report: Path,
) -> dict[str, Any]:
    track = load_track("parkland")
    car = load_design()
    wet_segs = load_rollout_segments(wet_jsonl, max_rows=2000, seq_len=20)
    damp_segs = load_rollout_segments(damp_jsonl, max_rows=2000, seq_len=20)

    cfg = PhysicsConfig()
    baseline_wet = evaluate_config(wet_segs, track, car, cfg)
    baseline_damp = evaluate_config(damp_segs, track, car, cfg)

    # Coordinate search over key longitudinal & lateral parameters
    best_cfg = PhysicsConfig(**cfg.to_dict())

    def _loss(c: PhysicsConfig) -> float:
        ew = evaluate_config(wet_segs, track, car, c)
        ed = evaluate_config(damp_segs, track, car, c)
        return (
            ew["speed_mae_20"] + ed["speed_mae_20"]
            + 2.0 * (ew["g_long_mae_1"] + ed["g_long_mae_1"])
            + 2.0 * (ew["g_lat_mae_1"] + ed["g_lat_mae_1"])
        )

    best_loss = _loss(best_cfg)
    grid = {
        "ice_power_w": [500000.0, 550000.0, 620000.0, 680000.0],
        "aero_drag_coeff": [0.72, 0.85, 0.98],
        "brake_force_max_n": [36000.0, 42000.0, 48000.0],
        "mu_dry_soft": [1.55, 1.72, 1.88],
        "wet_grip_drop": [0.24, 0.30, 0.36],
        "cornering_stiffness": [70000.0, 85000.0, 105000.0],
    }
    for param, candidates in grid.items():
        for val in candidates:
            trial = PhysicsConfig(**best_cfg.to_dict())
            setattr(trial, param, val)
            l = _loss(trial)
            if l < best_loss:
                best_loss = l
                best_cfg = trial

    out_params.parent.mkdir(parents=True, exist_ok=True)
    out_params.write_text(json.dumps(best_cfg.to_dict(), indent=2) + "\n")

    report = {
        "wet_segments": len(wet_segs),
        "damp_segments": len(damp_segs),
        "baseline": {"wet": baseline_wet, "damp": baseline_damp},
        "calibrated": {
            "wet": evaluate_config(wet_segs, track, car, best_cfg),
            "damp": evaluate_config(damp_segs, track, car, best_cfg),
        },
        "params_path": str(out_params),
    }
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_report.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate self_jev.sim physics against telemetry.")
    parser.add_argument("--wet", type=Path, default=Path("1st/training-61bb13c7-8b8f-47e4-aca9-70920619993a.jsonl"))
    parser.add_argument("--damp", type=Path, default=Path("data/training-ebe0e4b9-c62f-415c-8000-4aa51f9f662a.jsonl"))
    parser.add_argument("--out-params", type=Path, default=Path("self_jev/sim/calibrated_params.json"))
    parser.add_argument("--out-report", type=Path, default=Path("reports/sim_calibration.json"))
    args = parser.parse_args()
    res = calibrate(args.wet, args.damp, args.out_params, args.out_report)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
