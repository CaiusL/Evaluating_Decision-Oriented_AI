"""Mac-native wrapper for the 4-wheel AGX Formula reference model (agx_racing_gpusim/sim/ref)."""
from __future__ import annotations

import copy
import json
import math
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
_REF_DIR = _REPO_ROOT / "agx_racing_gpusim" / "sim" / "ref"
if str(_REF_DIR) not in sys.path:
    sys.path.insert(0, str(_REF_DIR))

import model as MD  # type: ignore  # noqa: E402
from track import Track  # type: ignore  # noqa: E402


def build_dict_to_ref_design(build: dict[str, Any]) -> dict[str, Any]:
    """Convert a winner/kit build dict into the nested design dict expected by ref/model.py."""
    design = copy.deepcopy(MD.STARTER_DESIGN)
    design["aero"]["front_wing"] = float(build["front_wing"])
    design["aero"]["rear_wing"] = float(build["rear_wing"])
    if "ride_height_mm" in build:
        rh = build["ride_height_mm"]
        if isinstance(rh, (list, tuple)):
            design["ride_height_mm"]["front"] = float(rh[0])
            design["ride_height_mm"]["rear"] = float(rh[1])
        elif isinstance(rh, dict):
            design["ride_height_mm"]["front"] = float(rh["front"])
            design["ride_height_mm"]["rear"] = float(rh["rear"])
    design["mass"]["ballast_kg"] = float(build["ballast_kg"])
    design["mass"]["weight_front_pct"] = float(build["weight_front_pct"])
    design["gearbox"]["ratios"] = [float(x) for x in build["gear_ratios"]]
    design["gearbox"]["final_drive"] = float(build["final_drive"])
    design["brakes"]["bias_front_pct"] = float(build["brake_bias_front_pct"])
    design["cooling"]["radiator_opening"] = float(build["radiator_opening"])
    design["power_unit"]["fuel_load_kg"] = float(build["fuel_load_kg"])
    design["power_unit"]["ers_harvest_level"] = int(build["ers_harvest_level"])
    design["tyres"]["start_compound"] = str(build["start_compound"])
    return design


DEFAULT_CALIBRATED_TUNE: dict[str, Any] = {
    "use_meff": True,
    "wet_grip": 0.46,
    "mux": 0.85,
    "mux_wet_drive": 0.98,
    "wet_launch_capacity_gain": 0.20,
    "mu_scale": 1.22,
    "mu_f": 1.03,
    "mu_r": 1.02,
    "k_scale": 1.0,
    "k_f": 0.98,
    "sens_form": "power",
    "front_heat_scale": 0.85,
}


class RefRaceSim:
    """Single-car Mac/CPU wrapper around agx_racing_gpusim/sim/ref/model.py."""

    def __init__(self, track_id: str, build: dict[str, Any] | None = None, tune: dict[str, Any] | None = None) -> None:
        self.track_id = track_id.lower()
        self.track = Track(self.track_id)
        self.design = build_dict_to_ref_design(build) if build else copy.deepcopy(MD.STARTER_DESIGN)
        merged_tune = dict(DEFAULT_CALIBRATED_TUNE)
        if tune:
            merged_tune.update(tune)
        self.car = MD.Car(self.design, self.track_id, tune=merged_tune)
        self.state = self.reset()

    def reset(
        self,
        x: float | None = None,
        y: float | None = None,
        yaw: float | None = None,
        speed: float = 0.0,
        gear: int = 1,
        pre_spool_ice: float = 1.0,
    ) -> MD.State:
        if x is None or y is None or yaw is None:
            g0 = self.track.raw["start_grid"][0]
            x = float(g0["x"])
            y = float(g0["y"])
            yaw = float(g0["yaw"])
        self.state = MD.new_state(
            self.car,
            x,
            y,
            yaw,
            speed,
            fuel=float(self.design["power_unit"]["fuel_load_kg"]),
            gear=gear,
        )
        self.state.ice = float(pre_spool_ice)
        s, n, i = self.track.project(x, y)
        self.state.s, self.state.n, self.state.i_trk = s, n, i
        return self.state

    def step(
        self,
        throttle: float,
        brake: float,
        steer_norm: float,
        ers_mode: int = 1,
        aero_mode: int = 0,
        override: bool = False,
        wetness: float = 0.0,
        air_temp_c: float = 24.0,
        track_temp_c: float | None = None,
        dt: float = 1.0 / 30.0,
        delta_override: float | None = None,
    ) -> MD.State:
        if track_temp_c is None:
            track_temp_c = 30.0 - 8.0 * wetness
        act = {
            "throttle": float(throttle),
            "brake": float(brake),
            "steer": max(-1.0, min(1.0, float(steer_norm))),
            "ers_mode": int(ers_mode),
            "aero_mode": int(aero_mode),
            "override": bool(override),
        }
        env = {"wet": float(wetness), "air": float(air_temp_c), "trk_t": float(track_temp_c)}
        n_ticks = max(1, int(round(dt * 60.0)))
        dt_tick = dt / float(n_ticks)
        for _ in range(n_ticks):
            MD.tick(self.car, self.track, self.state, act, env, dt_tick=dt_tick, substeps=2, delta_override=delta_override)
            self.state.wear = self.state.wear.clip(0.0, 0.99)
        return self.state

    def pit_exit(
        self,
        x: float,
        y: float,
        yaw: float,
        speed: float,
        compound: str | None = None,
    ) -> MD.State:
        """Place car at pit exit with fresh tyres and reset track projection hint."""
        self.state.x = float(x)
        self.state.y = float(y)
        self.state.yaw = float(yaw)
        self.state.vx = max(float(speed), 22.0)
        self.state.vy = 0.0
        self.state.r = 0.0
        self.state.wear[:] = 0.0
        if compound in ("soft", "medium", "hard"):
            self.car.set_compound(compound)
        self.state.i_trk = None
        s, n, i = self.track.project(self.state.x, self.state.y, None)
        self.state.s, self.state.n, self.state.i_trk = s, n, i
        return self.state
