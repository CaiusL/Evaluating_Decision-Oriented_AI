"""Parser & design_limits validator for design.json and rain_design.json."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

COMPOUNDS = ("soft", "medium", "hard")


@dataclass
class CarParams:
    dry_mass_kg: float
    ballast_kg: float
    weight_front_pct: float
    front_wing: float
    rear_wing: float
    bias_front_pct: float
    brake_duct: float
    radiator_opening: float
    final_drive: float
    gear_ratios: tuple[float, ...]
    ers_harvest_level: int
    fuel_load_kg: float
    start_compound: str
    pressure_front_psi: float = 29.0
    pressure_rear_psi: float = 28.0
    camber_front_deg: float = -2.5
    camber_rear_deg: float = -1.5
    wheel_radius_m: float = 0.33
    wheelbase_m: float = 3.60

    @property
    def start_compound_idx(self) -> int:
        return COMPOUNDS.index(self.start_compound) if self.start_compound in COMPOUNDS else 0


def load_design(path: Path | str | None = None) -> CarParams:
    if path is None:
        path = Path(__file__).resolve().parents[2] / "submissions" / "racing" / "jev_rl" / "design.json"
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    d = raw["design"] if "design" in raw else raw
    susp = d.get("suspension", {})
    tyres = d.get("tyres", {})
    return CarParams(
        dry_mass_kg=798.0,
        ballast_kg=float(d["mass"]["ballast_kg"]),
        weight_front_pct=float(d["mass"]["weight_front_pct"]),
        front_wing=float(d["aero"]["front_wing"]),
        rear_wing=float(d["aero"]["rear_wing"]),
        bias_front_pct=float(d["brakes"]["bias_front_pct"]),
        brake_duct=float(d["brakes"]["duct"]),
        radiator_opening=float(d["cooling"]["radiator_opening"]),
        final_drive=float(d["gearbox"]["final_drive"]),
        gear_ratios=tuple(float(r) for r in d["gearbox"]["ratios"]),
        ers_harvest_level=int(d["power_unit"]["ers_harvest_level"]),
        fuel_load_kg=float(d["power_unit"]["fuel_load_kg"]),
        start_compound=str(tyres["start_compound"]),
        pressure_front_psi=float(tyres.get("pressure_front_psi", 29.0)),
        pressure_rear_psi=float(tyres.get("pressure_rear_psi", 28.0)),
        camber_front_deg=float(susp.get("camber_front_deg", -2.5)),
        camber_rear_deg=float(susp.get("camber_rear_deg", -1.5)),
    )


def make_rain_design(base: CarParams) -> CarParams:
    """Create a wet-weather rain_design book with higher downforce and rearward brake balance."""
    return CarParams(
        dry_mass_kg=base.dry_mass_kg,
        ballast_kg=base.ballast_kg,
        weight_front_pct=base.weight_front_pct,
        front_wing=min(35.0, base.front_wing + 4.0),
        rear_wing=min(35.0, base.rear_wing + 4.0),
        bias_front_pct=max(52.0, base.bias_front_pct - 2.0),
        brake_duct=base.brake_duct,
        radiator_opening=base.radiator_opening,
        final_drive=base.final_drive,
        gear_ratios=base.gear_ratios,
        ers_harvest_level=base.ers_harvest_level,
        fuel_load_kg=base.fuel_load_kg,
        start_compound="soft",
        pressure_front_psi=base.pressure_front_psi,
        pressure_rear_psi=base.pressure_rear_psi,
        camber_front_deg=base.camber_front_deg,
        camber_rear_deg=base.camber_rear_deg,
    )


def validate_design_limits(car: CarParams, limits: dict[str, Any]) -> list[str]:
    """Validate CarParams against per-track design_limits from mapdata.json."""
    errors: list[str] = []
    p_floor = limits.get("effective_pressure_floor_psi") or limits.get("pressure_floor_psi") or {}
    if isinstance(p_floor, dict):
        f_min = float(p_floor.get("front", 0.0))
        r_min = float(p_floor.get("rear", 0.0))
        if car.pressure_front_psi < f_min:
            errors.append(f"front pressure {car.pressure_front_psi} < {f_min}")
        if car.pressure_rear_psi < r_min:
            errors.append(f"rear pressure {car.pressure_rear_psi} < {r_min}")
    c_lim = limits.get("camber_limit_deg") or {}
    if isinstance(c_lim, dict):
        f_lim = c_lim.get("front")
        if isinstance(f_lim, (list, tuple)) and len(f_lim) == 2:
            if not (float(f_lim[0]) <= car.camber_front_deg <= float(f_lim[1])):
                errors.append(f"front camber {car.camber_front_deg} outside {f_lim}")
    return errors
