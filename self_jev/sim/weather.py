"""Static, ramped, squall, and forecast weather state manager for self_jev.sim."""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class WeatherScenario:
    mode: str = "static"  # "static", "dry_to_wet", "wet_to_dry", "random_squall"
    start_wetness: float = 0.0
    end_wetness: float = 0.0
    ramp_start_s: float = 60.0
    ramp_end_s: float = 240.0
    rain_probability: float = 0.0


def build_weather_tensors(
    worlds: int,
    cars_per_world: int,
    regime: str,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Create per-car initial weather, ramp endpoints, ramp timings, and frozen forecast tensors.

    Regimes:
      - "dry": wetness 0.0, forecast rain_prob 0.05
      - "damp": wetness 0.25, forecast rain_prob 0.30
      - "wet": wetness 0.85, forecast rain_prob 0.85
      - "ramp_wet": 0.0 -> 0.85 over 40s..200s, forecast rain_prob 0.65
      - "ramp_dry": 0.85 -> 0.05 over 40s..200s, forecast rain_prob 0.55
      - "random_squall": randomized start/end wetness and onset timing per world
      - "mixed": spread dry, damp, wet, ramp_wet, ramp_dry, and random_squall across worlds
    """
    N = worlds * cars_per_world
    w_start = torch.zeros(worlds, device=device, dtype=torch.float32)
    w_end = torch.zeros(worlds, device=device, dtype=torch.float32)
    r_prob = torch.zeros(worlds, device=device, dtype=torch.float32)
    r_start = torch.full((worlds,), 40.0, device=device, dtype=torch.float32)
    r_end = torch.full((worlds,), 200.0, device=device, dtype=torch.float32)

    palette = [
        (0.0, 0.0, 0.05),    # dry
        (0.25, 0.25, 0.30),  # damp
        (0.85, 0.85, 0.85),  # wet
        (0.0, 0.85, 0.65),   # dry -> wet ramp
        (0.85, 0.05, 0.55),  # wet -> dry ramp
        None,                # random_squall
    ]
    fixed = {
        "dry": (0.0, 0.0, 0.05),
        "damp": (0.25, 0.25, 0.30),
        "wet": (0.85, 0.85, 0.85),
        "ramp_wet": (0.0, 0.85, 0.65),
        "ramp_dry": (0.85, 0.05, 0.55),
    }

    for w in range(worlds):
        if regime == "random_squall" or (regime not in fixed and palette[w % len(palette)] is None):
            ws = float(torch.rand(1).item() * 0.6)
            we = float(torch.clamp(torch.rand(1), 0.1, 1.0).item())
            rp = float(max(ws, we) * 0.85)
            rs = float(15.0 + torch.rand(1).item() * 95.0)
            re = float(rs + 45.0 + torch.rand(1).item() * 135.0)
            w_start[w], w_end[w], r_prob[w], r_start[w], r_end[w] = ws, we, rp, rs, re
        else:
            ws, we, rp = fixed[regime] if regime in fixed else palette[w % len(palette)]
            w_start[w], w_end[w], r_prob[w] = ws, we, rp

    expand = lambda t: t.unsqueeze(1).expand(worlds, cars_per_world).reshape(N)
    wet_s = expand(w_start)
    wet_e = expand(w_end)
    fc_prob = expand(r_prob)
    rs_t = expand(r_start)
    re_t = expand(r_end)
    rain_int = wet_s * 0.5
    trk_temp = 30.0 - 8.0 * wet_s

    return {
        "wet_start": wet_s,
        "wet_end": wet_e,
        "ramp_start_s": rs_t,
        "ramp_end_s": re_t,
        "wetness": wet_s.clone(),
        "rain_intensity": rain_int,
        "track_temp_c": trk_temp,
        "forecast_rain_prob": fc_prob,
    }


def update_weather(
    w_state: dict[str, torch.Tensor],
    t_race: torch.Tensor,
    ramp_start_s: float = 40.0,
    ramp_end_s: float = 200.0,
) -> None:
    """Update live wetness, rain intensity, and track temperature along piecewise-linear ramps."""
    rs = w_state.get("ramp_start_s", ramp_start_s)
    re = w_state.get("ramp_end_s", ramp_end_s)
    denom = torch.clamp(re - rs, min=1e-3) if isinstance(re, torch.Tensor) else max(1e-3, re - rs)
    alpha = torch.clamp((t_race - rs) / denom, 0.0, 1.0)
    wet = w_state["wet_start"] + alpha * (w_state["wet_end"] - w_state["wet_start"])
    w_state["wetness"] = wet
    w_state["rain_intensity"] = wet * 0.5
    w_state["track_temp_c"] = 30.0 - 8.0 * wet
