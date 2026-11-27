import math

from self_jev.dataset import example_situation
from self_jev.readiness import label
from self_jev.text import render


def choice(**over):
    return label(example_situation(**over))


def test_each_priority_step_and_both_sides_of_each_boundary():
    assert choice() == ("fire", "ready")
    assert choice(phase="buy")[0] == "unsupported"
    assert choice(alive=False)[0] == "unsupported"
    assert choice(match_tick=True)[0] == "unsupported"
    assert choice(match_tick="1000")[0] == "unsupported"
    assert choice(ammo=10) == ("unsupported", "ammo_mismatch")
    assert choice(reload="slow")[0] == "unsupported"
    assert choice(enemies=[], carrying_bomb=True, sites=[{"center": [0, 0, 0], "radius": 5}])[0] == "unsupported"
    assert choice(enemies=[], side="defend", bomb={"state": "planted", "position": [1.0, 0.0, 0.0]})[0] == "unsupported"
    assert choice(weapon={"id": "knife", "ammo": 0, "reserve": 0, "shot_ready": 1000})[0] == "unsupported"
    assert choice(weapon={"id": "knife", "ammo": 0, "reserve": 0, "shot_ready": 1000}, enemies=[]) == ("wait", "no_gun")
    assert choice(
        weapon={"ammo": 0, "reserve": 0, "shot_ready": 1000},
        other_gun={"id": "usp", "ammo": 12, "reserve": 0},
    )[0] == "unsupported"
    assert choice(reload=True) == ("wait", "busy")
    assert choice(weapon={"ammo": 0, "reserve": 30, "shot_ready": 1000}, reload=True) == ("wait", "busy")
    assert choice(switch_until=1001) == ("wait", "busy")
    assert choice(switch_until=1000) == ("fire", "ready")
    assert choice(weapon={"ammo": 0, "reserve": 1, "shot_ready": 1000}) == ("reload", "empty_magazine")
    assert choice(weapon={"ammo": 0, "reserve": 30, "shot_ready": 1000}, blind_ticks=4) == ("reload", "empty_magazine")
    assert choice(weapon={"ammo": 1, "reserve": 90, "shot_ready": 1000}) == ("fire", "ready")
    assert choice(
        weapon={"ammo": 0, "reserve": 0, "shot_ready": 1000},
        other_gun={"id": "usp", "ammo": 0, "reserve": 0},
    ) == ("wait", "no_ammunition")
    missing_reserve = example_situation(weapon={"ammo": 0, "shot_ready": 1000})
    del missing_reserve["weapon"]["reserve"]
    assert label(missing_reserve)[0] == "unsupported"
    assert choice(blind_ticks=0) == ("fire", "ready")
    assert choice(blind_ticks=1) == ("wait", "blind")
    assert choice(enemies=[]) == ("wait", "no_enemy")
    assert choice(enemy={"parts": []})[0] == "unsupported"
    assert choice(enemy={"parts": None})[0] == "unsupported"
    assert choice(distance=60.0) == ("fire", "ready")
    assert choice(distance=60.01) == ("wait", "out_of_range")
    allowance = math.atan2(0.25, 10.0)
    assert choice(yaw_extra=allowance) == ("fire", "ready")
    assert choice(yaw_extra=allowance + 1e-3) == ("adjust_aim", "misaligned")
    assert choice(yaw_extra=0.4, vx=2.0) == ("adjust_aim", "misaligned")
    assert choice(vx=0.49) == ("fire", "ready")
    assert choice(vx=0.5) == ("wait", "moving")
    assert choice(weapon={"ammo": 30, "reserve": 90, "shot_ready": 1000}) == ("fire", "ready")
    assert choice(weapon={"ammo": 30, "reserve": 90, "shot_ready": 1001}) == ("wait", "cooldown")
    assert choice(weapon={"id": "usp", "ammo": 12, "reserve": 24, "shot_ready": 1000, "trigger_pressed": True}) == ("wait", "trigger")
    assert choice(weapon={"id": "m4a1", "ammo": 30, "reserve": 90, "shot_ready": 1000, "trigger_pressed": True}) == ("fire", "ready")
    missing_trigger = example_situation(weapon={"id": "usp", "ammo": 12, "reserve": 24, "shot_ready": 1000})
    del missing_trigger["weapon"]["trigger_pressed"]
    assert label(missing_trigger) == ("unsupported", "trigger_missing")


def test_rendered_literals_match_the_row():
    import re

    situation = example_situation()
    text, literals = render(situation, "canonical")
    tokens = set(re.findall(r"-?\d+(?:\.\d+)?", text))
    assert literals
    assert set(literals) <= tokens
    assert "30" in tokens
    assert "1000" in tokens
    negation, _numbers = render(situation, "negation")
    assert "not empty" in negation
    assert negation != text
    empty = example_situation(weapon={"ammo": 0, "reserve": 1, "shot_ready": 1000})
    empty_text, _numbers = render(empty, "negation")
    assert "not loaded" in empty_text
    assert "not empty" not in empty_text
    missing = example_situation()
    missing["missing"] = "ammo"
    missing_text, _numbers = render(missing, "canonical")
    assert "unknown" in missing_text
    assert "30" not in set(re.findall(r"-?\d+(?:\.\d+)?", missing_text))
