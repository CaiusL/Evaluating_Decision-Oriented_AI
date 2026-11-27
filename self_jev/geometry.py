"""Visible-part aim geometry. Missing parts are not invented."""
from __future__ import annotations

import math

HEIGHTS = {"head": 1.64, "chest": 1.18, "abdomen": 0.88}
RADII = {"head": 0.12, "chest": 0.25, "abdomen": 0.25}
LEGAL_PARTS = ("head", "chest", "abdomen")


def aim_samples(origin: dict, enemy: dict, yaw: float, pitch: float):
    """Yield error, horizontal distance, and allowance for each reported part."""
    if enemy.get("alive", True) is not True:
        return
    if any(not _finite(enemy.get(key)) for key in ("x", "y", "z")):
        return
    parts = enemy.get("parts", None)
    if not isinstance(parts, (list, tuple)):
        return
    dx = enemy["x"] - _num(origin.get("x"))
    dy = enemy["y"] - _num(origin.get("y"))
    distance = math.hypot(dx, dy)
    eye = 0.95 if origin.get("crouch") else 1.62
    scale = 0.6 if enemy.get("crouch") else 1.0
    for part in parts:
        if part not in HEIGHTS:
            continue
        dz = enemy["z"] + HEIGHTS[part] * scale - _num(origin.get("z")) - eye
        want_yaw = math.atan2(-dy, dx)
        want_pitch = math.atan2(dz, max(0.01, distance))
        dyaw = (want_yaw - yaw + math.pi) % (2 * math.pi) - math.pi
        error = math.hypot(dyaw, want_pitch - pitch)
        allowance = math.atan2(RADII[part] * scale, max(0.01, distance))
        yield error, distance, allowance, part


def _finite(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _num(value) -> float:
    return float(value) if _finite(value) else 0.0
