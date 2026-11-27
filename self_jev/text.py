"""Handwritten English renders. Kept numbers are printed unchanged."""
from __future__ import annotations

import re

from self_jev.readiness import CHOICES, QUESTION, UNSUPPORTED, describe_aim, objective

DESCRIPTIONS = {
    "fire": "Fire the equipped gun.",
    "adjust_aim": "Adjust aim toward the visible enemy.",
    "reload": "Reload the equipped gun.",
    "wait": "Wait without firing, aiming, or reloading.",
    UNSUPPORTED: "Do not choose a combat action; the situation is unsupported.",
}
_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


def render(situation: dict, frame: str = "canonical") -> tuple[str, list[str]]:
    missing = situation.get("missing")
    pairs: list[tuple[str, str, list[str]]] = []

    def add(canon: str, para: str, numbers: list[str]) -> None:
        kept = [number for number in numbers if _NUMBER.fullmatch(number)]
        for number in kept:
            if number not in canon or number not in para:
                raise RuntimeError(f"template dropped literal {number}")
        pairs.append((canon, para, kept))

    phase = situation.get("phase") or "not reported"
    add(f"Phase {phase}.", f"Round phase {phase}.", [])
    alive = "alive" if situation.get("alive") is True else "dead"
    add(f"The seat is {alive}.", f"This seat is {alive}.", [])
    tick = _fmt(situation.get("match_tick"))
    add(f"Tick {tick}.", f"Match tick {tick}.", [tick])
    px, py, pz = _fmt(situation.get("x")), _fmt(situation.get("y")), _fmt(situation.get("z"))
    add(f"Position {px} {py} {pz}.", f"Coordinates {px} {py} {pz}.", [px, py, pz])
    speed = _fmt(_speed(situation))
    add(f"Speed {speed} metres per second.", f"Movement speed {speed} metres per second.", [speed])
    weapon = situation.get("weapon") or {}
    slot = _fmt(situation.get("active_slot"))
    weapon_id = weapon.get("id") if isinstance(weapon.get("id"), str) and weapon.get("id") else "unnamed"
    add(f"Weapon {weapon_id} in slot {slot}.", f"Equipped weapon {weapon_id} slot {slot}.", [slot])
    if missing == "ammo":
        add("The magazine count is unknown.", "Magazine count is unknown.", [])
    else:
        ammo = _fmt(situation.get("ammo"))
        add(f"Magazine {ammo}.", f"Rounds in the magazine: {ammo}.", [ammo])
    if missing == "reserve":
        add("The reserve count is unknown.", "Reserve count is unknown.", [])
    else:
        reserve = _fmt(weapon.get("reserve"))
        add(f"Reserve {reserve}.", f"Rounds in reserve: {reserve}.", [reserve])
    held = _fmt(weapon.get("ammo"))
    if missing != "ammo":
        add(f"Weapon ammo {held}.", f"Equipped magazine {held}.", [held])
    if missing == "shot_ready":
        add("The shot cooldown tick is unknown.", "Cooldown tick is unknown.", [])
    else:
        ready = _fmt(weapon.get("shot_ready"))
        add(f"Cooldown tick {ready}.", f"Shot ready tick {ready}.", [ready])
    switch = _fmt(situation.get("switch_until"))
    add(f"Switch tick {switch}.", f"Switch completes at tick {switch}.", [switch])
    reload = _fmt(situation.get("reload"))
    add(f"Reload {reload}.", f"Reload value {reload}.", [reload])
    if missing == "trigger":
        add("The trigger state is unknown.", "Trigger state is unknown.", [])
    else:
        pressed = "pressed" if weapon.get("trigger_pressed") is True else "released"
        add(f"Trigger {pressed}.", f"The trigger is {pressed}.", [])
    blind = _fmt(situation.get("blind_ticks"))
    add(f"Blind ticks {blind}.", f"Remaining blind ticks {blind}.", [blind])
    side = situation.get("side") or "not reported"
    add(f"Side {side}.", f"Team side {side}.", [])
    carrying = "yes" if situation.get("carrying_bomb") else "no"
    add(f"Carrying bomb {carrying}.", f"Bomb carrier {carrying}.", [])
    bomb = situation.get("bomb") or {}
    state = bomb.get("state")
    if isinstance(state, str) and state:
        add(f"Bomb state {state}.", f"The bomb state is {state}.", [])
    else:
        add("The bomb state is not reported.", "Bomb state was not reported.", [])
    if isinstance(bomb.get("position"), (list, tuple)) and len(bomb["position"]) >= 2:
        bx, by = _fmt(bomb["position"][0]), _fmt(bomb["position"][1])
        add(f"Bomb position {bx} {by}.", f"Planted bomb coordinates {bx} {by}.", [bx, by])
    _enemy_sentences(situation, missing, add)
    site_bits = []
    site_para = []
    site_numbers = []
    for site in situation.get("sites") or []:
        center = site.get("center") or []
        if len(center) < 2:
            continue
        cx, cy = _fmt(center[0]), _fmt(center[1])
        cz = _fmt(center[2] if len(center) > 2 else 0)
        radius = _fmt(site.get("radius"))
        name = site.get("id") or "site"
        site_bits.append(f"{name} {cx} {cy} {cz} r {radius}")
        site_para.append(f"{name} at {cx} {cy} {cz} radius {radius}")
        site_numbers.extend([cx, cy, cz, radius])
    if site_bits:
        add("Sites " + ", ".join(site_bits) + ".", "Bombsites " + ", ".join(site_para) + ".", site_numbers)
    _comparisons(situation, missing, add)
    name = objective(situation) if not missing else ""
    if name == "plant":
        add("The bomb carrier is inside a site.", "A bomb carrier is inside a site.", [])
    elif name == "defuse":
        add("The defender is beside a planted bomb.", "A defender is beside a planted bomb.", [])
    if frame == "paraphrase_b":
        body = " ".join(para for _canon, para, _numbers in pairs)
    elif frame == "paraphrase_c":
        body = "Readiness facts. " + " ".join(canon for canon, _para, _numbers in pairs)
    else:
        body = " ".join(canon for canon, _para, _numbers in pairs)
    literals = [number for _canon, _para, numbers in pairs for number in numbers]
    if frame == "negation":
        negation = _negation(situation, missing)
        body = negation + " " + body
    if "not empty" in body and _magazine_empty(situation, missing):
        raise RuntimeError("negation contradicts the magazine count")
    tokens = set(_NUMBER.findall(body))
    for number in literals:
        if number not in tokens:
            raise RuntimeError(f"render dropped literal {number}")
    return body, literals


def candidates(ids: dict[str, str]) -> list[dict]:
    order = (*CHOICES, UNSUPPORTED)
    return [{"id": ids[key], "key": key, "description": DESCRIPTIONS[key]} for key in order]


def _enemy_sentences(situation: dict, missing: str | None, add) -> None:
    if missing == "visible":
        add("The visible-enemy list is unknown.", "Visible enemies are unknown.", [])
        return
    enemies = [enemy for enemy in situation.get("enemies") or [] if isinstance(enemy, dict)]
    if not enemies:
        if missing == "parts":
            add("The visible parts are unknown.", "Visible parts are unknown.", [])
        elif missing == "distance":
            add("The enemy distance is unknown.", "Enemy distance is unknown.", [])
        else:
            add("There is no visible enemy.", "No enemy is visible.", [])
        return
    detail = None if missing in ("parts", "distance") else describe_aim(situation)
    enemy = detail["enemy"] if detail else enemies[0]
    if missing == "parts":
        add("The visible parts are unknown.", "Visible parts are unknown.", [])
    else:
        parts = enemy.get("parts")
        shown = "none" if not isinstance(parts, (list, tuple)) or not parts else ", ".join(str(part) for part in parts)
        add(f"Visible parts {shown}.", f"Reported parts {shown}.", [])
    if missing == "distance":
        add("The enemy distance is unknown.", "Enemy distance is unknown.", [])
        return
    if detail is None:
        add("There is no usable aim target.", "No usable aim target is present.", [])
        return
    distance = _fmt(detail["distance"])
    error = _fmt(detail["error"])
    allowance = _fmt(detail["allowance"])
    add(
        f"Nearest enemy {distance} metres. Aim error {error} radians. Allowance {allowance} radians.",
        f"Enemy range {distance} metres. Angular error {error} radians against allowance {allowance} radians.",
        [distance, error, allowance],
    )


def _comparisons(situation: dict, missing: str | None, add) -> None:
    canon, para, numbers = [], [], []

    def put(left: str, right: str, values: list[str]) -> None:
        canon.append(left)
        para.append(right)
        numbers.extend(values)

    if missing != "ammo":
        ammo = situation.get("ammo")
        if isinstance(ammo, (int, float)) and not isinstance(ammo, bool):
            put("magazine empty" if ammo <= 0 else "magazine loaded", "empty magazine" if ammo <= 0 else "loaded magazine", [])
    speed = _speed(situation)
    if speed is not None:
        put("moving" if speed >= 0.5 else "stationary", "in motion" if speed >= 0.5 else "standing still", [])
    blind = situation.get("blind_ticks")
    if isinstance(blind, (int, float)) and not isinstance(blind, bool):
        put("blinded" if blind > 0 else "sighted", "eyes blocked" if blind > 0 else "eyes clear", [])
    tick = situation.get("match_tick")
    switch = situation.get("switch_until")
    if all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in (tick, switch)):
        put("switch busy" if switch > tick else "switch clear", "switch unfinished" if switch > tick else "switch finished", [])
    ready = (situation.get("weapon") or {}).get("shot_ready")
    if missing != "shot_ready" and all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in (tick, ready)):
        put("cooldown busy" if ready > tick else "cooldown clear", "cooldown unfinished" if ready > tick else "cooldown finished", [])
    if missing == "visible":
        pass
    elif not (situation.get("enemies") or []):
        put("no enemy", "nobody visible", [])
    elif missing != "distance":
        detail = describe_aim(situation)
        if detail is None:
            put("no aim target", "no shootable part", [])
        else:
            if detail["distance"] > 60:
                put("enemy out of range", "enemy too far", [])
            else:
                put("enemy in range", "enemy close enough", [])
                outside = detail["error"] > detail["allowance"] + 1e-9
                put("aim off" if outside else "aim aligned", "aim is wide" if outside else "aim is on target", [])
    if canon:
        add("Checks: " + ", ".join(canon) + ".", "Facts: " + ", ".join(para) + ".", numbers)


def _negation(situation: dict, missing: str | None) -> str:
    ammo = situation.get("ammo")
    if missing != "ammo" and isinstance(ammo, (int, float)) and not isinstance(ammo, bool):
        shown = _fmt(ammo)
        if ammo > 0:
            return f"The magazine is not empty ({shown} rounds)."
        return f"The magazine is not loaded ({shown} rounds)."
    if missing != "visible" and not (situation.get("enemies") or []):
        return "There is not a visible enemy."
    blind = _fmt(situation.get("blind_ticks"))
    return f"Blindness is not being guessed ({blind} blind ticks)."


def _magazine_empty(situation: dict, missing: str | None) -> bool:
    ammo = situation.get("ammo")
    if missing == "ammo" or isinstance(ammo, bool) or not isinstance(ammo, (int, float)):
        return False
    return ammo <= 0


def _fmt(value) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "not reported"
    if isinstance(value, int) or float(value).is_integer():
        return str(int(value))
    return format(float(value), ".6g")


def _speed(situation: dict):
    vx, vy = situation.get("vx"), situation.get("vy")
    if isinstance(vx, bool) or isinstance(vy, bool):
        return None
    if not isinstance(vx, (int, float)) or not isinstance(vy, (int, float)):
        return None
    return (float(vx) ** 2 + float(vy) ** 2) ** 0.5
