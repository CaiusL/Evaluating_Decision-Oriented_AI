"""Deterministic readiness groups. Seed 17. Variants of one group share a split."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from self_jev.readiness import CHOICES, LABELER_VERSION, QUESTION, UNSUPPORTED, label
from self_jev.text import candidates, render

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "readiness"
SEED = 17
FRAMES = ("canonical", "paraphrase_b", "paraphrase_c", "negation", "shuffled", "arbitrary_ids")
TEXT_FRAME = {
    "canonical": "canonical",
    "paraphrase_b": "paraphrase_b",
    "paraphrase_c": "paraphrase_c",
    "negation": "negation",
    "shuffled": "canonical",
    "arbitrary_ids": "canonical",
}
MISSING_FIELDS = ("ammo", "reserve", "parts", "shot_ready", "visible", "trigger", "distance")
ID_POOL = ("k7", "north", "slot4", "quartz", "moth", "river", "cedar", "plover", "ember", "nimbus")


def generate(seed: int = SEED) -> list[dict]:
    groups = _groups(np.random.default_rng(seed))
    _assign_splits(groups, np.random.default_rng(seed))
    examples = []
    for group in groups:
        examples.extend(_renders(group, np.random.default_rng(seed + group["index"] + 1)))
    return examples


def write_dataset(examples: list[dict] | None = None) -> Path:
    examples = generate() if examples is None else examples
    DATA.mkdir(parents=True, exist_ok=True)
    path = DATA / "examples.jsonl"
    with path.open("w") as handle:
        for example in examples:
            handle.write(json.dumps(example, ensure_ascii=False) + "\n")
    (DATA / "manifest.json").write_text(json.dumps(manifest(examples), indent=2) + "\n")
    return path


def load_dataset() -> list[dict]:
    path = DATA / "examples.jsonl"
    if not path.exists():
        write_dataset()
    rows = []
    with path.open() as handle:
        for line in handle:
            rows.append(json.loads(line))
    return rows


def manifest(examples: list[dict]) -> dict:
    groups = {}
    canonical = {choice: 0 for choice in (*CHOICES, UNSUPPORTED)}
    for example in examples:
        groups[example["group_id"]] = example["split"]
        if example["render"] == "canonical":
            canonical[example["canon_label"]] += 1
    splits = {name: 0 for name in ("train", "select", "calibrate", "test")}
    for split in groups.values():
        splits[split] += 1
    return {
        "seed": SEED,
        "labeler_version": LABELER_VERSION,
        "examples": len(examples),
        "groups": len(groups),
        "canonical_by_label": canonical,
        "groups_by_split": splits,
        "boundary_groups": "held in the test split",
    }


def _groups(rng: np.random.Generator) -> list[dict]:
    made = {choice: [] for choice in (*CHOICES, UNSUPPORTED)}
    made["fire"].extend(_mark(_boundaries_fire()))
    made["adjust_aim"].extend(_mark(_boundaries_aim()))
    made["reload"].extend(_mark(_boundaries_reload()))
    made["wait"].extend(_mark(_boundaries_wait()))
    made[UNSUPPORTED].extend(_unsupported_seeds())
    attempts = 0
    while any(len(rows) < 400 for rows in made.values()):
        for choice in made:
            if len(made[choice]) >= 400:
                continue
            attempts += 1
            if attempts > 20000:
                raise RuntimeError(f"could not fill labels: { {key: len(rows) for key, rows in made.items()} }")
            situation = _random_situation(rng, choice)
            got, _reason = label(situation)
            if got != choice:
                continue
            if choice in CHOICES and float(rng.random()) < 0.5:
                situation["side"] = None
                situation["bomb"] = {}
            dx = float(rng.uniform(0, 130))
            dy = float(rng.uniform(0, 120))
            situation["x"] = float(situation.get("x") or 0) + dx
            situation["y"] = float(situation.get("y") or 0) + dy
            for enemy in situation.get("enemies") or []:
                if isinstance(enemy, dict):
                    if isinstance(enemy.get("x"), (int, float)) and not isinstance(enemy.get("x"), bool):
                        enemy["x"] = float(enemy["x"]) + dx
                    if isinstance(enemy.get("y"), (int, float)) and not isinstance(enemy.get("y"), bool):
                        enemy["y"] = float(enemy["y"]) + dy
            new_tick = int(rng.integers(1, 70000))
            delta = new_tick - int(situation["match_tick"])
            situation["match_tick"] = new_tick
            weapon = situation["weapon"]
            if isinstance(weapon.get("shot_ready"), int):
                weapon["shot_ready"] += delta
            if isinstance(situation.get("switch_until"), int):
                situation["switch_until"] += delta
            if label(situation)[0] != choice:
                continue
            made[choice].append(situation)
    groups = []
    number = 0
    for choice in (*CHOICES, UNSUPPORTED):
        for situation in made[choice][:400]:
            groups.append({
                "index": number,
                "group_id": f"g{number:04d}",
                "canon_label": choice,
                "situation": situation,
                "boundary": bool(situation.pop("_boundary", False)),
                "missing_variant": choice in CHOICES and number % 2 == 0,
                "missing_field": MISSING_FIELDS[number % len(MISSING_FIELDS)],
            })
            number += 1
    return groups


def _assign_splits(groups: list[dict], rng: np.random.Generator) -> None:
    """Keep the 240/60/60/40 counts. Hand-built numeric boundaries stay in test."""
    by_label: dict[str, list] = {}
    for group in groups:
        by_label.setdefault(group["canon_label"], []).append(group)
    for rows in by_label.values():
        boundary = [row for row in rows if row["boundary"]]
        rest = [row for row in rows if not row["boundary"]]
        if len(boundary) > 40:
            raise RuntimeError("boundary groups exceed the test split")
        order = rng.permutation(len(rest))
        rest = [rest[int(index)] for index in order]
        test_fill = 40 - len(boundary)
        test = boundary + rest[:test_fill]
        rest = rest[test_fill:]
        calibrate, select, train = rest[:60], rest[60:120], rest[120:]
        if (len(test), len(calibrate), len(select), len(train)) != (40, 60, 60, 240):
            raise RuntimeError("split counts drifted")
        for group in test:
            group["split"] = "test"
        for group in calibrate:
            group["split"] = "calibrate"
        for group in select:
            group["split"] = "select"
        for group in train:
            group["split"] = "train"


def _renders(group: dict, rng: np.random.Generator) -> list[dict]:
    rows = []
    ids = {key: f"c{offset}" for offset, key in enumerate((*CHOICES, UNSUPPORTED))}
    canonical_text = None
    for frame in FRAMES:
        text, literals = render(group["situation"], TEXT_FRAME[frame])
        if frame == "canonical":
            canonical_text = text
        if frame == "negation" and text == canonical_text:
            raise RuntimeError("negation render matches the canonical text")
        if frame in ("paraphrase_b", "paraphrase_c") and text == canonical_text:
            raise RuntimeError(f"{frame} matches the canonical text")
        chosen = candidates(ids)
        if frame == "shuffled":
            rng.shuffle(chosen)
        if frame == "arbitrary_ids":
            pool = list(ID_POOL)
            rng.shuffle(pool)
            for item, token in zip(chosen, pool):
                item["id"] = token
        rows.append(_example(group, frame, text, literals, chosen, group["canon_label"]))
    if group["missing_variant"]:
        situation = json.loads(json.dumps(group["situation"]))
        situation["missing"] = group["missing_field"]
        got, _reason = label(situation)
        if got != UNSUPPORTED:
            raise RuntimeError("missing field was not labelled unsupported")
        text, literals = render(situation, "canonical")
        if "unknown" not in text:
            raise RuntimeError("missing-field render does not say unknown")
        rows.append(_example(group, "missing", text, literals, candidates(ids), UNSUPPORTED))
    return rows


def _example(group, frame, text, literals, chosen, label_key) -> dict:
    if not any(item["key"] == label_key for item in chosen):
        raise RuntimeError("correct choice missing from candidates")
    return {
        "group_id": group["group_id"],
        "split": group["split"],
        "render": frame,
        "question": QUESTION,
        "situation": text,
        "literals": literals,
        "candidates": chosen,
        "label_key": label_key,
        "canon_label": group["canon_label"],
        "boundary": bool(group["boundary"] and frame == "canonical" and label_key == group["canon_label"]),
    }


def _mark(rows: list[dict]) -> list[dict]:
    for row in rows:
        row["_boundary"] = True
    return rows


def example_situation(**over) -> dict:
    distance = float(over.pop("distance", 10.0))
    yaw_extra = float(over.pop("yaw_extra", 0.0))
    pitch = _aligned_pitch(distance)
    enemy = {
        "seat": 2, "x": distance, "y": 0.0, "z": 0.0, "alive": True, "crouch": False, "parts": ["chest"],
    }
    enemy.update(over.pop("enemy", {}))
    weapon = {
        "id": "m4a1", "ammo": 30, "reserve": 90, "shot_ready": 1000, "reload_until": None,
        "reload_state": None, "trigger_pressed": False, "aim_punch_yaw": 0.0, "aim_punch_pitch": 0.0,
    }
    weapon.update(over.pop("weapon", {}))
    row = {
        "phase": "live", "alive": True, "match_tick": 1000, "x": 0.0, "y": 0.0, "z": 0.0,
        "yaw": yaw_extra, "pitch": pitch, "vx": 0.0, "vy": 0.0, "crouch": False,
        "side": "attack", "carrying_bomb": False, "blind_ticks": 0, "active_slot": 0,
        "ammo": weapon["ammo"], "reload": 0, "switch_until": 1000, "weapon": weapon,
        "other_gun": {"id": "usp", "ammo": 0, "reserve": 0}, "enemies": [enemy],
        "bomb": {"state": "carried"},
        "sites": [
            {"id": "A", "center": [100.0, 0.0, 0.0], "radius": 5},
            {"id": "B", "center": [40.0, 80.0, 0.0], "radius": 5},
        ],
        "release_required": ["pistol", "usp"], "missing": None,
    }
    row.update(over)
    _aim_at(row, yaw_extra)
    return row


def _aim_at(row: dict, yaw_extra: float) -> None:
    enemies = [enemy for enemy in row.get("enemies") or [] if isinstance(enemy, dict)]
    enemy = enemies[0] if enemies else None
    if enemy is None or any(not isinstance(enemy.get(key), (int, float)) or isinstance(enemy.get(key), bool) for key in ("x", "y", "z")):
        row["yaw"] = yaw_extra
        return
    dx = enemy["x"] - (row.get("x") or 0)
    dy = enemy["y"] - (row.get("y") or 0)
    distance = math.hypot(dx, dy)
    eye = 0.95 if row.get("crouch") else 1.62
    scale = 0.6 if enemy.get("crouch") else 1.0
    dz = enemy["z"] + 1.18 * scale - (row.get("z") or 0) - eye
    row["yaw"] = math.atan2(-dy, dx) + yaw_extra
    row["pitch"] = math.atan2(dz, max(0.01, distance))


def _aligned_pitch(distance: float) -> float:
    return math.atan2(1.18 - 1.62, max(0.01, distance))


def _boundaries_fire() -> list[dict]:
    allowance = math.atan2(0.25, 10.0)
    return [
        example_situation(weapon={"ammo": 1, "reserve": 90, "shot_ready": 1000}),
        example_situation(distance=60.0),
        example_situation(vx=0.49),
        example_situation(yaw_extra=allowance),
        example_situation(weapon={"id": "m4a1", "ammo": 30, "reserve": 90, "shot_ready": 1000, "trigger_pressed": True}),
        example_situation(switch_until=1000),
        example_situation(blind_ticks=0),
    ]


def _boundaries_aim() -> list[dict]:
    allowance = math.atan2(0.25, 10.0)
    return [example_situation(yaw_extra=allowance + 1e-3), example_situation(yaw_extra=0.4)]


def _boundaries_reload() -> list[dict]:
    return [
        example_situation(weapon={"ammo": 0, "reserve": 1, "shot_ready": 1000}),
        example_situation(weapon={"ammo": 0, "reserve": 30, "shot_ready": 1000}, blind_ticks=4),
    ]


def _boundaries_wait() -> list[dict]:
    return [
        example_situation(enemies=[]),
        example_situation(distance=60.01),
        example_situation(vx=0.5),
        example_situation(weapon={"ammo": 30, "reserve": 90, "shot_ready": 1001}),
        example_situation(switch_until=1001),
        example_situation(blind_ticks=1),
        example_situation(weapon={"id": "usp", "ammo": 12, "reserve": 24, "shot_ready": 1000, "trigger_pressed": True}),
        example_situation(weapon={"ammo": 0, "reserve": 0, "shot_ready": 1000}, other_gun={"id": "usp", "ammo": 0, "reserve": 0}),
    ]


def _unsupported_seeds() -> list[dict]:
    return [
        example_situation(weapon={"id": "knife", "ammo": 0, "reserve": 0, "shot_ready": 1000}),
        example_situation(weapon={"ammo": 0, "reserve": 0, "shot_ready": 1000}, other_gun={"id": "usp", "ammo": 12, "reserve": 0}),
        example_situation(enemies=[], carrying_bomb=True, sites=[{"center": [0, 0, 0], "radius": 5}], bomb={"state": "carried"}),
        example_situation(enemies=[], side="defend", bomb={"state": "planted", "position": [1.0, 0.0, 0.0]}),
        example_situation(alive=False),
        example_situation(phase="buy"),
        example_situation(enemy={"parts": []}),
        example_situation(enemy={"parts": None}),
        example_situation(ammo=10, weapon={"ammo": 30, "reserve": 90, "shot_ready": 1000}),
        example_situation(missing="ammo"),
    ]


def _random_situation(rng: np.random.Generator, choice: str) -> dict:
    tick = int(rng.integers(200, 5000))
    if choice == "fire":
        distance = float(rng.choice([rng.uniform(5.0, 50.0), 60.0]))
        allowance = math.atan2(0.25, distance)
        situation = example_situation(
            distance=distance,
            yaw_extra=float(rng.choice([0.0, allowance])),
            vx=float(rng.choice([rng.uniform(0.0, 0.3), 0.49])),
            weapon={
                "id": str(rng.choice(["m4a1", "ak47", "galilar"])),
                "ammo": int(rng.integers(1, 31)),
                "reserve": int(rng.integers(1, 91)),
                "shot_ready": tick,
                "trigger_pressed": False,
            },
        )
        return _at_tick(situation, tick)
    if choice == "adjust_aim":
        distance = float(rng.uniform(5.0, 40.0))
        allowance = math.atan2(0.25, distance)
        situation = example_situation(
            distance=distance,
            yaw_extra=allowance + float(rng.uniform(0.001, 0.8)),
            vx=float(rng.uniform(0.0, 2.0)),
        )
        return _at_tick(situation, tick)
    if choice == "reload":
        situation = example_situation(
            distance=float(rng.uniform(5.0, 40.0)),
            weapon={"ammo": 0, "reserve": int(rng.integers(1, 91)), "shot_ready": tick},
            blind_ticks=int(rng.integers(0, 4)),
            vx=float(rng.uniform(0.0, 1.5)),
        )
        return _at_tick(situation, tick)
    if choice == "wait":
        return _random_wait(rng, tick)
    return _random_unsupported(rng, tick)


def _random_wait(rng: np.random.Generator, tick: int) -> dict:
    mode = int(rng.integers(0, 8))
    if mode == 0:
        return _at_tick(example_situation(enemies=[]), tick)
    if mode == 1:
        return _at_tick(example_situation(distance=float(rng.uniform(60.01, 90.0))), tick)
    if mode == 2:
        return _at_tick(example_situation(vx=float(rng.uniform(0.5, 3.0))), tick)
    if mode == 3:
        situation = example_situation(
            weapon={"ammo": 30, "reserve": 90, "shot_ready": tick + 1},
            switch_until=tick,
        )
        return _at_tick(situation, tick, ready=False)
    if mode == 4:
        return _at_tick(example_situation(switch_until=tick + 1), tick, ready=False)
    if mode == 5:
        return _at_tick(example_situation(blind_ticks=int(rng.integers(1, 6))), tick)
    if mode == 6:
        situation = example_situation(weapon={
            "id": "usp", "ammo": 12, "reserve": 24, "shot_ready": tick, "trigger_pressed": True,
        })
        return _at_tick(situation, tick)
    return _at_tick(example_situation(
        weapon={"ammo": 0, "reserve": 0, "shot_ready": tick},
        other_gun={"id": "usp", "ammo": 0, "reserve": 0},
    ), tick, ready=False)


def _random_unsupported(rng: np.random.Generator, tick: int) -> dict:
    mode = int(rng.integers(0, 10))
    if mode == 0:
        return _at_tick(example_situation(weapon={"id": "knife", "ammo": 0, "reserve": 0, "shot_ready": tick}), tick, ready=False)
    if mode == 1:
        return _at_tick(example_situation(
            weapon={"ammo": 0, "reserve": 0, "shot_ready": tick},
            other_gun={"id": "usp", "ammo": int(rng.integers(1, 13)), "reserve": 0},
        ), tick, ready=False)
    if mode == 2:
        return _at_tick(example_situation(
            enemies=[], carrying_bomb=True, x=float(rng.uniform(-1.0, 1.0)), y=float(rng.uniform(-1.0, 1.0)),
            sites=[{"id": "A", "center": [0, 0, 0], "radius": 5}], bomb={"state": "carried"},
        ), tick)
    if mode == 3:
        return _at_tick(example_situation(
            enemies=[], side="defend", bomb={"state": "planted", "position": [float(rng.uniform(0.2, 2.0)), 0.0, 0.0]},
        ), tick)
    if mode == 4:
        return _at_tick(example_situation(alive=False), tick)
    if mode == 5:
        return _at_tick(example_situation(phase=str(rng.choice(["buy", "post", "freeze"]))), tick)
    if mode == 6:
        return _at_tick(example_situation(enemy={"parts": []}), tick)
    if mode == 7:
        return _at_tick(example_situation(enemy={"parts": None}), tick)
    if mode == 8:
        return _at_tick(example_situation(ammo=int(rng.integers(0, 10)), weapon={"ammo": 30, "reserve": 90, "shot_ready": tick}), tick)
    situation = _at_tick(example_situation(), tick)
    situation["missing"] = str(rng.choice(MISSING_FIELDS))
    return situation


def _at_tick(situation: dict, tick: int, ready: bool = True) -> dict:
    situation["match_tick"] = tick
    if ready:
        situation["weapon"]["shot_ready"] = tick
        situation["switch_until"] = min(situation.get("switch_until", tick), tick)
    return situation
