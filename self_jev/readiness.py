"""Combat-readiness labels. The first matching rule wins."""
from __future__ import annotations

import math

from self_jev.geometry import LEGAL_PARTS, aim_samples

QUESTION = "What should this seat do now for combat readiness?"
CHOICES = ("fire", "adjust_aim", "reload", "wait")
UNSUPPORTED = "unsupported"
LABELER_VERSION = "1"
RELEASE_REQUIRED = ("pistol", "usp")
ENGAGE_METRES = 60.0
STATIONARY_SPEED = 0.5


def label(situation: dict) -> tuple[str, str]:
    """Return the readiness choice and the rule that produced it."""
    if situation.get("missing"):
        return UNSUPPORTED, "missing_" + str(situation["missing"])
    phase = situation.get("phase")
    if phase != "live":
        return UNSUPPORTED, "phase"
    if situation.get("alive") is not True:
        return UNSUPPORTED, "dead"
    tick = situation.get("match_tick")
    if isinstance(tick, bool) or not isinstance(tick, int):
        return UNSUPPORTED, "tick"
    weapon = situation.get("weapon") or {}
    magazine = _number(situation.get("ammo"))
    weapon_ammo = _number(weapon.get("ammo")) if "ammo" in weapon else None
    if magazine is not None and weapon_ammo is not None and abs(magazine - weapon_ammo) > 1e-9:
        return UNSUPPORTED, "ammo_mismatch"
    reload_mark = _reload_mark(situation.get("reload"))
    if reload_mark == "bad":
        return UNSUPPORTED, "reload_type"
    if _busy_value(weapon.get("reload_state")):
        reload_mark = "busy"
    reload_until = _deadline(weapon.get("reload_until"), tick)
    switch_until = _deadline(situation.get("switch_until"), tick)
    if reload_until == "bad" or switch_until == "bad":
        return UNSUPPORTED, "deadline_type"
    if reload_until == "busy":
        reload_mark = "busy"
    enemies = [enemy for enemy in situation.get("enemies") or [] if isinstance(enemy, dict)]
    visible = [enemy for enemy in enemies if enemy.get("alive", True) is True]
    if _objective(situation, visible):
        return UNSUPPORTED, _objective(situation, visible)
    slot = situation.get("active_slot")
    weapon_id = weapon.get("id")
    gun = slot in (0, 1) and isinstance(weapon_id, str) and weapon_id not in ("", "knife")
    usable = _usable(situation, visible)
    if not gun:
        if usable:
            return UNSUPPORTED, "switch_required"
        return "wait", "no_gun"
    if reload_mark == "busy" or switch_until == "busy":
        return "wait", "busy"
    reserve = _number(weapon.get("reserve")) if "reserve" in weapon else None
    if magazine is None:
        return UNSUPPORTED, "ammo_missing"
    if magazine <= 0 and reserve is None:
        return UNSUPPORTED, "reserve_missing"
    if magazine <= 0 and reserve > 0:
        return "reload", "empty_magazine"
    if magazine <= 0 and reserve <= 0:
        other = situation.get("other_gun") or {}
        other_ammo = _number(other.get("ammo")) or 0
        other_reserve = _number(other.get("reserve")) or 0
        if other_ammo > 0 or other_reserve > 0:
            return UNSUPPORTED, "switch_required"
        return "wait", "no_ammunition"
    blind = _number(situation.get("blind_ticks"))
    if blind is None:
        return UNSUPPORTED, "blind_missing"
    if blind > 0:
        return "wait", "blind"
    if not visible:
        return "wait", "no_enemy"
    if not usable:
        return UNSUPPORTED, "parts"
    enemy, distance = usable
    if distance > ENGAGE_METRES:
        return "wait", "out_of_range"
    aim = _aim(situation, enemy)
    if aim is None:
        return UNSUPPORTED, "aim_unknown"
    aligned, _error, _allowance = aim
    if not aligned:
        return "adjust_aim", "misaligned"
    speed = _speed(situation)
    if speed is None:
        return UNSUPPORTED, "speed_unknown"
    if speed >= STATIONARY_SPEED:
        return "wait", "moving"
    ready = _deadline(weapon.get("shot_ready"), tick)
    if ready == "bad":
        return UNSUPPORTED, "cooldown_type"
    if ready == "busy":
        return "wait", "cooldown"
    if _release_blocked(situation, weapon_id):
        if weapon.get("trigger_pressed") is None and "trigger_pressed" not in weapon:
            return UNSUPPORTED, "trigger_missing"
        return "wait", "trigger"
    if reserve is None:
        return UNSUPPORTED, "reserve_missing"
    return "fire", "ready"


def objective(situation: dict) -> str:
    """Plant or defuse opportunity, or an empty string."""
    enemies = [enemy for enemy in situation.get("enemies") or [] if isinstance(enemy, dict)]
    visible = [enemy for enemy in enemies if enemy.get("alive", True) is True]
    return _objective(situation, visible)


def describe_aim(situation: dict):
    """Nearest usable enemy and the punched-aim error the labeler uses."""
    enemies = [enemy for enemy in situation.get("enemies") or [] if isinstance(enemy, dict)]
    visible = [enemy for enemy in enemies if enemy.get("alive", True) is True]
    usable = _usable(situation, visible)
    if not usable:
        return None
    enemy, distance = usable
    aim = _aim(situation, enemy)
    if aim is None:
        return None
    aligned, error, allowance = aim
    return {
        "enemy": enemy,
        "distance": distance,
        "aligned": aligned,
        "error": error,
        "allowance": allowance,
    }


def _objective(situation: dict, visible: list) -> str:
    blind = _number(situation.get("blind_ticks"))
    if visible or blind is None or blind > 0:
        return ""
    bomb = situation.get("bomb") or {}
    if situation.get("side") == "attack" and situation.get("carrying_bomb") and bomb.get("state") != "planted":
        for site in situation.get("sites") or []:
            center = site.get("center") or []
            radius = _number(site.get("radius"))
            if radius is None or len(center) < 2:
                continue
            cx, cy = _number(center[0]), _number(center[1])
            x, y = _number(situation.get("x")), _number(situation.get("y"))
            if None in (cx, cy, x, y):
                continue
            cz = _number(center[2]) if len(center) > 2 else 0.0
            cz = 0.0 if cz is None else cz
            z = _number(situation.get("z"))
            z = 0.0 if z is None else z
            if math.hypot(x - cx, y - cy) < radius * 0.8 and abs(z - cz) < 1.5:
                return "plant"
    position = bomb.get("position")
    x, y = _number(situation.get("x")), _number(situation.get("y"))
    px = _number(position[0]) if isinstance(position, (list, tuple)) and len(position) >= 2 else None
    py = _number(position[1]) if isinstance(position, (list, tuple)) and len(position) >= 2 else None
    if (situation.get("side") == "defend" and bomb.get("state") == "planted"
            and None not in (x, y, px, py) and math.hypot(px - x, py - y) < 2.5):
        return "defuse"
    return ""


def _usable(situation: dict, visible: list):
    found = []
    for enemy in visible:
        parts = enemy.get("parts", None)
        legal = [part for part in parts or [] if part in LEGAL_PARTS] if isinstance(parts, (list, tuple)) else []
        if not legal or any(not _finite(enemy.get(key)) for key in ("x", "y", "z")):
            continue
        distance = math.hypot(enemy["x"] - _num(situation.get("x")), enemy["y"] - _num(situation.get("y")))
        found.append((distance, enemy.get("seat", 0), enemy))
    if not found:
        return None
    found.sort(key=lambda item: (item[0], item[1]))
    distance, _seat, enemy = found[0]
    return enemy, distance


def _aim(situation: dict, enemy: dict):
    yaw = _number(situation.get("yaw"))
    pitch = _number(situation.get("pitch"))
    if yaw is None or pitch is None:
        return None
    weapon = situation.get("weapon") or {}
    punch_yaw = _number(weapon.get("aim_punch_yaw")) or 0.0
    punch_pitch = _number(weapon.get("aim_punch_pitch")) or 0.0
    samples = list(aim_samples(situation, enemy, yaw + 2 * punch_yaw, pitch + 2 * punch_pitch))
    if not samples:
        return None
    error, _distance, allowance, _part = min(samples, key=lambda sample: sample[0])
    # Angle wrapping can put a mathematically equal error a few ulps over the allowance.
    return error <= allowance + 1e-9, error, allowance


def _speed(situation: dict):
    vx = _number(situation.get("vx"))
    vy = _number(situation.get("vy"))
    if vx is None or vy is None:
        return None
    return math.hypot(vx, vy)


def _release_blocked(situation: dict, weapon_id: str) -> bool:
    extra = situation.get("release_required") or []
    required = set(RELEASE_REQUIRED) | set(extra)
    if weapon_id not in required:
        return False
    weapon = situation.get("weapon") or {}
    if "trigger_pressed" not in weapon:
        return True
    return weapon.get("trigger_pressed") is True


def _reload_mark(value) -> str:
    if value is None or value is False or value == 0 or value == "":
        return "clear"
    if value is True:
        return "busy"
    if _finite(value) and value > 0:
        return "busy"
    if _finite(value):
        return "clear"
    return "bad"


def _busy_value(value) -> bool:
    return value not in (None, False, "")


def _deadline(value, tick: int) -> str:
    if value is None or value is False or value == "":
        return "clear"
    number = _number(value)
    if number is None:
        return "bad"
    return "busy" if number > tick else "clear"


def _finite(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _number(value):
    if not _finite(value):
        return None
    return float(value)


def _num(value) -> float:
    number = _number(value)
    return 0.0 if number is None else number
