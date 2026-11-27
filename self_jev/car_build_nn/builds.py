"""Candidate rule-legal car-build library for between-game setup selection."""
from __future__ import annotations

from dataclasses import replace
from typing import Any

from self_jev.sim.design import CarParams, load_design, validate_design_limits
from self_jev.sim.track import load_track

BUILD_NAMES: tuple[str, ...] = (
    "dry_low_drag",
    "dry_balanced",
    "damp_mid_downforce",
    "wet_high_downforce",
)
ABSTAIN_LABEL = "Abstain / Unresolved"
OPTION_NAMES: tuple[str, ...] = (*BUILD_NAMES, ABSTAIN_LABEL)


def get_candidate_builds(base: CarParams | None = None) -> dict[str, CarParams]:
    """Return 4 distinct, realistic formula-v4-design-1 candidate builds."""
    b = base or load_design()
    return {
        "dry_low_drag": replace(
            b,
            front_wing=15.0,
            rear_wing=16.0,
            bias_front_pct=60.0,
            final_drive=5.15,
            ballast_kg=16.0,
            weight_front_pct=46.0,
            start_compound="medium",
        ),
        "dry_balanced": replace(
            b,
            front_wing=20.0,
            rear_wing=22.0,
            bias_front_pct=59.0,
            final_drive=5.40,
            ballast_kg=20.0,
            weight_front_pct=45.0,
            start_compound="soft",
        ),
        "damp_mid_downforce": replace(
            b,
            front_wing=24.0,
            rear_wing=26.0,
            bias_front_pct=57.0,
            final_drive=5.55,
            ballast_kg=22.0,
            weight_front_pct=44.5,
            start_compound="medium",
        ),
        "wet_high_downforce": replace(
            b,
            front_wing=29.0,
            rear_wing=31.0,
            bias_front_pct=55.0,
            final_drive=5.75,
            ballast_kg=24.0,
            weight_front_pct=44.0,
            start_compound="soft",
        ),
    }


def describe_build(name: str, car: CarParams | None = None) -> str:
    """Short human-readable summary of a candidate build for LLM/Kev option descriptions."""
    if name == ABSTAIN_LABEL or car is None:
        return "Hold decision / escalate when forecast or telemetry is ambiguous, missing, or unsafe."
    return (
        f"wings={car.front_wing:.0f}/{car.rear_wing:.0f}, "
        f"brake_bias={car.bias_front_pct:.0f}%, "
        f"final_drive={car.final_drive:.2f}, "
        f"start_compound={car.start_compound}"
    )


def validate_all_builds(tracks: tuple[str, ...] = ("parkland", "highland", "crossover")) -> dict[str, Any]:
    """Validate all candidate builds against per-track design_limits."""
    builds = get_candidate_builds()
    report: dict[str, Any] = {"ok": True, "tracks": {}}
    for tid in tracks:
        trk = load_track(tid)
        limits = getattr(trk, "design_limits", None) or {}
        track_errs: dict[str, list[str]] = {}
        for bname, car in builds.items():
            errs = validate_design_limits(car, limits)
            track_errs[bname] = errs
            if errs:
                report["ok"] = False
        report["tracks"][tid] = track_errs
    return report
