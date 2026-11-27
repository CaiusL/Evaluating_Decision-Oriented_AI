"""One legal strategy lever per RSI cycle."""
from __future__ import annotations

import copy

FPS_LEVERS = ("site", "ct_setup", "post_plant", "buy_style", "utility", "timing")
RACING_LEVERS = ("preview", "kv", "fuel", "front_wing", "stop")

FPS_KEYS = {
    "site": ("default_site", "expect_site"),
    "ct_setup": ("ct_setup",),
    "post_plant": ("post_plant",),
    "buy_style": ("buy_style",),
    "utility": ("utility",),
    "timing": ("execute_live_seconds",),
}
RACING_KEYS = {
    "preview": ("preview",),
    "kv": ("kv",),
    "fuel": ("fuel_load_kg",),
    "front_wing": ("front_wing",),
    "stop": ("stops",),
}

_CT = ("mid", "retake", "stack_expect")
_POST = ("hunt", "hold", "molotov")
_BUY = ("full_rifle", "force_rifle", "smg_force", "armor_pistol")
_UTILITY = ("flashbang", "smokegrenade", "molotov", "none")


def levers(game: str) -> tuple[str, ...]:
    if game == "fps":
        return FPS_LEVERS
    if game == "racing":
        return RACING_LEVERS
    raise ValueError(game)


def lever_keys(game: str, lever: str) -> tuple[str, ...]:
    table = FPS_KEYS if game == "fps" else RACING_KEYS
    return table[lever]


def changed_keys(before: dict, after: dict) -> set[str]:
    keys = set(before) | set(after)
    return {key for key in keys if before.get(key) != after.get(key) and key not in ("name", "thesis")}


def revise(game: str, strategy: dict, lever: str, worse: bool, generation: int, match_id: str) -> dict:
    book = copy.deepcopy(strategy)
    if game == "fps":
        _revise_fps(book, lever, worse)
    else:
        _revise_racing(book, lever, worse)
    book["name"] = f"sj{generation}-{lever}"[:48]
    direction = "after a weaker result" if worse else "after a stronger result"
    book["thesis"] = (
        f"Self-JEV changed only {lever} {direction} on match {match_id}. "
        "The agent file is the published base."
    )[:600]
    changed = changed_keys(strategy, book)
    expected = set(lever_keys(game, lever))
    if changed != expected:
        raise RuntimeError(f"{game} {lever} changed {sorted(changed)}, expected {sorted(expected)}")
    return book


def _cycle(options: tuple[str, ...], current: str, worse: bool) -> str:
    index = options.index(current) if current in options else 0
    step = 1 if worse else -1
    return options[(index + step) % len(options)]


def _nudge(value, step, low, high):
    updated = min(high, max(low, value + step))
    if updated == value:
        updated = min(high, max(low, value - step))
    return updated


def _revise_fps(book: dict, lever: str, worse: bool) -> None:
    if lever == "site":
        site = "B" if book.get("default_site") == "A" else "A"
        book["default_site"] = site
        book["expect_site"] = site
    elif lever == "ct_setup":
        book["ct_setup"] = _cycle(_CT, book.get("ct_setup", "mid"), worse)
    elif lever == "post_plant":
        book["post_plant"] = _cycle(_POST, book.get("post_plant", "hunt"), worse)
    elif lever == "buy_style":
        book["buy_style"] = _cycle(_BUY, book.get("buy_style", "full_rifle"), worse)
    elif lever == "utility":
        book["utility"] = _cycle(_UTILITY, book.get("utility", "flashbang"), worse)
    elif lever == "timing":
        book["execute_live_seconds"] = int(_nudge(int(book.get("execute_live_seconds", 12)), 8 if worse else -8, 12, 45))
    else:
        raise ValueError(lever)


def _revise_racing(book: dict, lever: str, worse: bool) -> None:
    if lever == "preview":
        book["preview"] = round(_nudge(float(book["preview"]), 0.02 if worse else -0.02, 0.12, 0.40), 3)
    elif lever == "kv":
        book["kv"] = round(_nudge(float(book["kv"]), -0.1 if worse else 0.1, 1.0, 2.2), 3)
    elif lever == "fuel":
        book["fuel_load_kg"] = int(_nudge(int(book["fuel_load_kg"]), 1 if worse else -1, 7, 14))
    elif lever == "front_wing":
        book["front_wing"] = int(_nudge(int(book["front_wing"]), 1 if worse else -1, 18, 25))
    elif lever == "stop":
        stop = dict(book["stops"][0])
        stop["lap"] = 2 if int(stop.get("lap", 1)) == 1 else 1
        book["stops"] = [stop]
    else:
        raise ValueError(lever)
