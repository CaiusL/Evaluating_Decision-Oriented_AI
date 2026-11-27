"""Canonical racing observation, action contract, renderer, and delegation policy for racing_engineer_v1."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np

SCHEMA_VERSION = "racing_engineer_v1"
OBJECTIVE_VERSION = "completion_then_time_v1"
FALLBACK_POLICY_ID = "driver_only_v1"

TRACKS: tuple[str, ...] = ("parkland", "highland", "crossover")
TRACK_LENGTHS_M: dict[str, float] = {
    "parkland": 5795.8,
    "highland": 6999.4,
    "crossover": 5803.2,
}
WEATHER_REGIMES: tuple[str, ...] = ("Dry", "Wetting", "Saturated")

CANONICAL_CELLS: list[dict[str, Any]] = [
    {"id": "parkland_dry", "track": "parkland", "weather_regime": "Dry", "wet_start": 0.02, "wet_end": 0.02, "rain_prob": 0.05, "track_temp_c": 30.0},
    {"id": "parkland_wetting", "track": "parkland", "weather_regime": "Wetting", "wet_start": 0.05, "wet_end": 0.72, "rain_prob": 0.75, "track_temp_c": 24.0},
    {"id": "parkland_saturated", "track": "parkland", "weather_regime": "Saturated", "wet_start": 0.85, "wet_end": 0.88, "rain_prob": 0.95, "track_temp_c": 19.0},
    {"id": "highland_dry", "track": "highland", "weather_regime": "Dry", "wet_start": 0.02, "wet_end": 0.02, "rain_prob": 0.05, "track_temp_c": 29.0},
    {"id": "highland_wetting", "track": "highland", "weather_regime": "Wetting", "wet_start": 0.08, "wet_end": 0.75, "rain_prob": 0.78, "track_temp_c": 23.0},
    {"id": "highland_saturated", "track": "highland", "weather_regime": "Saturated", "wet_start": 0.85, "wet_end": 0.90, "rain_prob": 0.95, "track_temp_c": 18.0},
    {"id": "crossover_dry", "track": "crossover", "weather_regime": "Dry", "wet_start": 0.02, "wet_end": 0.02, "rain_prob": 0.05, "track_temp_c": 30.0},
    {"id": "crossover_wetting", "track": "crossover", "weather_regime": "Wetting", "wet_start": 0.06, "wet_end": 0.74, "rain_prob": 0.76, "track_temp_c": 24.0},
    {"id": "crossover_saturated", "track": "crossover", "weather_regime": "Saturated", "wet_start": 0.86, "wet_end": 0.90, "rain_prob": 0.96, "track_temp_c": 18.5},
]

BUILD_OPTIONS: list[str] = [
    "dry_low_drag",
    "dry_balanced",
    "damp_mid_downforce",
    "wet_high_downforce",
]

# Verified directly against self_jev/car_build_nn/builds.py
VERIFIED_BUILD_DESCS: list[str] = [
    "wings 15/16, brake_bias 60%, final_drive 5.15, ballast 16 kg, start_compound medium (lowest aero drag and downforce).",
    "wings 20/22, brake_bias 59%, final_drive 5.40, ballast 20 kg, start_compound soft (balanced drag and downforce; reference baseline).",
    "wings 24/26, brake_bias 57%, final_drive 5.55, ballast 22 kg, start_compound medium (medium-high downforce).",
    "wings 29/31, brake_bias 55%, final_drive 5.75, ballast 24 kg, start_compound soft (highest downforce and drag).",
]

PACE_OPTIONS: list[str] = [
    "push_max",
    "balanced",
    "conserve_tyres",
    "wet_caution",
]

# Verified directly against scripts/eval_race_engineer_v2.py lines 253-271
VERIFIED_PACE_DESCS: list[str] = [
    "Throttle residual +0.15; if ers_soc_mj > 0.5 and wetness < 0.45 forces ers_mode=3 and override=1 (higher speed, higher wall-crash/wear risk).",
    "Zero throttle offset (standard frozen 30 Hz RL driver output with automatic assist_ers_mode).",
    "Throttle residual -0.08; forces ers_mode=1 and override=0 (slower, lower slip/wear).",
    "Throttle residual -0.15; forces ers_mode=0 (harvest) and override=0 (slowest, lowest slip/crash risk).",
]

PIT_OPTIONS: list[str] = [
    "stay_out",
    "pit_soft",
    "pit_medium",
]

# Verified directly against self_jev/sim/env.py lines 331-354
VERIFIED_PIT_DESCS: list[str] = [
    "Do not pit; keep current tyres and wear, incur 0.0 s pit-lane loss.",
    "Latch pit_request for soft compound; adds 35.77 s pit-lane transit/service time and resets tyre_wear to 0.00 at pit exit.",
    "Latch pit_request for medium compound; adds 35.77 s pit-lane transit/service time and resets tyre_wear to 0.00 at pit exit.",
]

VERIFIED_RACING_CONTRACT_TEXT = (
    "Racing Contract (4-lap RaceEnv, frozen 30 Hz RL driver):\n"
    "- Objective: Prioritize completing all 4 laps without wall crash, stall, or timeout; among valid finishes, minimize total 4-lap race time.\n"
    "- Pre-race build is fixed for the entire 4-lap race (if forecast rain_prob > 0.40, rain_design wing scaling applies on top of the selected build).\n"
    "- Pace mode is applied at the start of each lap (Laps 1-4) until the next lap crossing.\n"
    "- Pit call (when enabled on Laps 2-3) costs 35.77 s in the pit lane and resets tyre_wear to 0.00; only pit if remaining-race completion or time benefit exceeds 35.77 s."
)

FALLBACK_DEFAULTS: dict[str, str] = {
    "build": "dry_balanced",
    "pace": "balanced",
    "pit": "stay_out",
}


def action_spec_hash() -> str:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "objective_version": OBJECTIVE_VERSION,
        "builds": list(zip(BUILD_OPTIONS, VERIFIED_BUILD_DESCS)),
        "pace": list(zip(PACE_OPTIONS, VERIFIED_PACE_DESCS)),
        "pit": list(zip(PIT_OPTIONS, VERIFIED_PIT_DESCS)),
        "contract": VERIFIED_RACING_CONTRACT_TEXT,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def make_scenario(base_cell: dict[str, Any], cell_idx: int, seed: int) -> dict[str, Any]:
    """Deterministic scenario generator matching confirm_track_rule_v2.py & eval_feedback_f1.py."""
    rng = np.random.default_rng(seed * 100 + cell_idx)
    reg = base_cell["weather_regime"]
    if reg == "Dry":
        ws = float(np.clip(rng.uniform(0.0, 0.06), 0.0, 0.06))
        we = float(np.clip(rng.uniform(0.0, 0.06), 0.0, 0.06))
        rp = float(rng.uniform(0.0, 0.15))
        temp = float(rng.uniform(26.0, 34.0))
    elif reg == "Wetting":
        ws = float(rng.uniform(0.02, 0.14))
        we = float(rng.uniform(0.58, 0.88))
        rp = float(rng.uniform(0.55, 0.90))
        temp = float(rng.uniform(20.0, 27.0))
    else:  # Saturated
        ws = float(rng.uniform(0.76, 0.95))
        we = float(rng.uniform(0.78, 0.96))
        rp = float(rng.uniform(0.85, 0.99))
        temp = float(rng.uniform(15.0, 22.0))
    return {
        "scenario_id": f"{base_cell['id']}_s{seed}",
        "cell_id": base_cell["id"],
        "track": base_cell["track"],
        "weather_regime": reg,
        "seed": int(seed),
        "wet_start": round(ws, 4),
        "wet_end": round(we, 4),
        "rain_prob": round(rp, 4),
        "track_temp_c": round(temp, 2),
        "ramp_start_s": round(float(rng.uniform(25.0, 80.0)), 1),
        "ramp_end_s": round(float(rng.uniform(280.0, 420.0)), 1),
        "grip_scale": round(float(rng.uniform(0.94, 1.06)), 4),
        "aero_scale": round(float(rng.uniform(0.94, 1.06)), 4),
    }


def generate_scenarios_for_seeds(seeds: list[int]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for cell_idx, base in enumerate(CANONICAL_CELLS):
        for seed in seeds:
            out.append(make_scenario(base, cell_idx, int(seed)))
    return out


# ---------------------------------------------------------------------------
# Canonical Whitelisted Observations (Shared by Qwen Renderer & NN-3 Extractor)
# ---------------------------------------------------------------------------

def build_pre_race_obs(sc: dict[str, Any], laps_total: int = 4) -> dict[str, Any]:
    """Whitelisted pre-race observation (no seed, no hidden grip/aero scales, no ramp timings)."""
    tr = str(sc["track"]).lower()
    return {
        "decision_type": "build",
        "track": tr,
        "track_length_m": TRACK_LENGTHS_M.get(tr, 5800.0),
        "laps_total": int(laps_total),
        "weather_regime": str(sc["weather_regime"]),
        "forecast_wet_start": float(sc["wet_start"]),
        "forecast_wet_end": float(sc["wet_end"]),
        "forecast_rain_prob": float(sc["rain_prob"]),
        "track_temp_c": float(sc["track_temp_c"]),
        "legal_options": list(BUILD_OPTIONS),
    }


def build_live_lap_obs(
    sc: dict[str, Any],
    chosen_build: str,
    current_mode: str,
    lap_completed: int,
    laps_total: int,
    elapsed_s: float,
    wetness: float,
    rain_intensity: float,
    track_temp_c: float,
    tyre_compound: str,
    tyre_wear: float,
    tyre_temp_c: float,
    ers_soc_mj: float,
    fuel_kg: float,
    pit_stops: int,
    off_track_ticks_so_far: int,
    wall_grazes_so_far: int,
    last_lap_time_s: float | None,
    decision_type: str = "pace",
) -> dict[str, Any]:
    """Whitelisted live-lap observation (no seed, no hidden scales, explicit missingness)."""
    tr = str(sc["track"]).lower()
    legal = list(PACE_OPTIONS) if decision_type == "pace" else list(PIT_OPTIONS)
    return {
        "decision_type": decision_type,
        "track": tr,
        "track_length_m": TRACK_LENGTHS_M.get(tr, 5800.0),
        "chosen_build": str(chosen_build),
        "current_mode": str(current_mode),
        "laps_completed": int(lap_completed),
        "laps_remaining": int(laps_total - lap_completed),
        "laps_total": int(laps_total),
        "elapsed_s": round(float(elapsed_s), 2),
        "weather_regime": str(sc["weather_regime"]),
        "forecast_rain_prob": float(sc["rain_prob"]),
        "forecast_wet_end": float(sc["wet_end"]),
        "wetness": round(float(wetness), 4),
        "rain_intensity": round(float(rain_intensity), 4),
        "track_temp_c": round(float(track_temp_c), 2),
        "tyre_compound": str(tyre_compound),
        "tyre_wear": round(float(tyre_wear), 4),
        "tyre_temp_c": round(float(tyre_temp_c), 2),
        "ers_soc_mj": round(float(ers_soc_mj), 4),
        "fuel_kg": round(float(fuel_kg), 3),
        "pit_stops": int(pit_stops),
        "off_track_ticks_so_far": int(off_track_ticks_so_far),
        "wall_grazes_so_far": int(wall_grazes_so_far),
        "last_lap_time_s": round(float(last_lap_time_s), 2) if last_lap_time_s is not None else None,
        "legal_options": legal,
    }


def render_obs_text(obs: dict[str, Any], include_contract: bool = True) -> str:
    """Deterministic text rendering of canonical observation for Qwen."""
    prefix = f"{VERIFIED_RACING_CONTRACT_TEXT}\n\n" if include_contract else ""
    if obs["decision_type"] == "build":
        return (
            f"{prefix}Pre-Race Forecast:\n"
            f"- Track: {obs['track']} ({obs['track_length_m']:.1f} m, {obs['laps_total']} laps)\n"
            f"- Weather regime: {obs['weather_regime']}, rain_prob={obs['forecast_rain_prob']:.2f}, "
            f"wetness_start={obs['forecast_wet_start']:.2f}, wetness_end={obs['forecast_wet_end']:.2f}, "
            f"track_temp={obs['track_temp_c']:.1f} C\n"
            f"- Decision boundary: Pre-race build (applies for all {obs['laps_total']} laps)."
        )
    last_lap_str = f"{obs['last_lap_time_s']:.2f}s" if obs["last_lap_time_s"] is not None else "missing (lap 1 start)"
    return (
        f"{prefix}Live Telemetry (Lap {obs['laps_completed'] + 1}/{obs['laps_total']}, {obs['laps_remaining']} laps remaining):\n"
        f"- Track: {obs['track']} ({obs['track_length_m']:.1f} m) | Build: {obs['chosen_build']} | Prev mode: {obs['current_mode']}\n"
        f"- Time: elapsed={obs['elapsed_s']:.1f}s, last_lap={last_lap_str}\n"
        f"- Weather: regime={obs['weather_regime']}, wetness={obs['wetness']:.2f} (forecast_end={obs['forecast_wet_end']:.2f}, rain_prob={obs['forecast_rain_prob']:.2f}), temp={obs['track_temp_c']:.1f} C\n"
        f"- Car state: compound={obs['tyre_compound']}, tyre_wear={obs['tyre_wear']:.3f}, tyre_temp={obs['tyre_temp_c']:.1f} C, ers_soc={obs['ers_soc_mj']:.2f} MJ, fuel={obs['fuel_kg']:.2f} kg\n"
        f"- Incidents & pits: pit_stops={obs['pit_stops']}, off_track_ticks={obs['off_track_ticks_so_far']}, wall_grazes={obs['wall_grazes_so_far']}"
    )


def extract_build_features(obs: dict[str, Any]) -> np.ndarray:
    """11-dim numeric feature vector for NN-3 build head (strictly from canonical pre-race obs)."""
    tr = obs["track"]
    reg = obs["weather_regime"]
    return np.array([
        1.0 if tr == "parkland" else 0.0,
        1.0 if tr == "highland" else 0.0,
        1.0 if tr == "crossover" else 0.0,
        obs["track_length_m"] / 7000.0,
        1.0 if reg == "Dry" else 0.0,
        1.0 if reg == "Wetting" else 0.0,
        1.0 if reg == "Saturated" else 0.0,
        float(obs["forecast_wet_start"]),
        float(obs["forecast_wet_end"]),
        float(obs["forecast_rain_prob"]),
        float(obs["track_temp_c"]) / 40.0,
    ], dtype=np.float32)


def extract_pace_features(obs: dict[str, Any]) -> np.ndarray:
    """27-dim numeric feature vector for NN-3 pace head (strictly from canonical live-lap obs)."""
    tr = obs["track"]
    reg = obs["weather_regime"]
    b = obs["chosen_build"]
    m = obs["current_mode"]
    c = obs["tyre_compound"]
    last_lap_missing = 1.0 if obs["last_lap_time_s"] is None else 0.0
    last_lap_val = float(obs["last_lap_time_s"]) / 200.0 if obs["last_lap_time_s"] is not None else 0.0
    return np.array([
        1.0 if tr == "parkland" else 0.0,
        1.0 if tr == "highland" else 0.0,
        1.0 if tr == "crossover" else 0.0,
        obs["track_length_m"] / 7000.0,
        1.0 if reg == "Dry" else 0.0,
        1.0 if reg == "Wetting" else 0.0,
        1.0 if reg == "Saturated" else 0.0,
        1.0 if b == "dry_low_drag" else 0.0,
        1.0 if b == "dry_balanced" else 0.0,
        1.0 if b == "damp_mid_downforce" else 0.0,
        1.0 if b == "wet_high_downforce" else 0.0,
        1.0 if m == "push_max" else 0.0,
        1.0 if m == "balanced" else 0.0,
        1.0 if m == "conserve_tyres" else 0.0,
        1.0 if m == "wet_caution" else 0.0,
        1.0 if c == "soft" else 0.0,
        1.0 if c == "medium" else 0.0,
        float(obs["laps_completed"]) / max(float(obs["laps_total"]), 1.0),
        float(obs["laps_remaining"]) / max(float(obs["laps_total"]), 1.0),
        float(obs["elapsed_s"]) / 600.0,
        float(obs["wetness"]),
        float(obs["forecast_wet_end"]),
        float(obs["tyre_wear"]),
        float(obs["tyre_temp_c"]) / 100.0,
        float(obs["ers_soc_mj"]) / 4.0,
        min(float(obs["off_track_ticks_so_far"]) / 100.0, 2.0),
        last_lap_val,
        last_lap_missing,
        min(float(obs["wall_grazes_so_far"]) / 5.0, 2.0),
    ], dtype=np.float32)


@dataclass
class DecisionAuditRecord:
    decision_type: str
    proposed_action: str | None
    applied_action: str
    delegated: bool
    delegation_reason: str | None
    confidence: float | None
    fallback_policy_id: str = FALLBACK_POLICY_ID


@dataclass
class DelegationTracker:
    confidence_threshold: float | None = None
    fallback_policy_id: str = FALLBACK_POLICY_ID
    records: list[DecisionAuditRecord] = field(default_factory=list)

    def resolve(
        self,
        decision_type: str,
        proposed: str | None,
        legal_options: list[str],
        parse_ok: bool = True,
        confidence: float | None = None,
    ) -> str:
        fallback = FALLBACK_DEFAULTS[decision_type]
        if fallback not in legal_options:
            fallback = legal_options[0]

        reason: str | None = None
        if not parse_ok or proposed is None:
            reason = "invalid_or_unmapped_output"
        elif proposed not in legal_options:
            reason = "illegal_action"
        elif (
            self.confidence_threshold is not None
            and confidence is not None
            and confidence < self.confidence_threshold
        ):
            reason = f"low_confidence<{self.confidence_threshold:.2f}"

        applied = fallback if reason is not None else str(proposed)
        self.records.append(
            DecisionAuditRecord(
                decision_type=decision_type,
                proposed_action=proposed,
                applied_action=applied,
                delegated=(reason is not None),
                delegation_reason=reason,
                confidence=confidence,
                fallback_policy_id=self.fallback_policy_id,
            )
        )
        return applied
