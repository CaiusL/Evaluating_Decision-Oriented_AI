"""Element decisions for Self-JEV.

Each example is one element in one state. The text the model sees does not
include the trace. Train, select, and test do not share a situation id, and
train phrases are not reused in the held-out text.
"""
from __future__ import annotations

import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path

from self_jev._live_notes import DRAFT_NOTE, NOTES
from self_jev.dataset import example_situation
from self_jev.readiness import describe_aim, label

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "elements"
SHARD = ROOT / "data" / "shards" / "fps" / "2df4b71e-08bb-41a0-b899-ff4de556fb74-seat6.jsonl"
POINTER = ROOT / "state" / "element_checkpoint.txt"
ABSTAIN = "abstain"
SUPPORTED = {"fire", "adjust_aim", "reload", "wait"}
CAP = 80
SEED = 17
ID_POOL = ("k7", "north", "slot4", "quartz", "moth", "river", "cedar", "plover", "ember", "nimbus")
FAMILY = {"train": "alpha", "select": "beta", "calibrate": "beta", "test": "gamma"}
QUESTION = {
    "alpha": "Which setting should this element take?",
    "beta": "Select the preferred setting.",
    "gamma": "Choose the setting for the next submission.",
}
CANDIDATES = {
    "site": ("A", "B", ABSTAIN),
    "buy": ("rifle", "armor_pistol", ABSTAIN),
    "steer_speed": ("1.0", "0.85", ABSTAIN),
    "burst": ("4", "9", ABSTAIN),
    "settle": ("0.5", "0.6", ABSTAIN),
    "peek_trigger": ("6", "9", ABSTAIN),
    "peek_cadence": ("4/16", "8/24", ABSTAIN),
    "escape": ("50/25", "25/12", ABSTAIN),
    "enemy_memory": ("160/90/40", "80/45/20", ABSTAIN),
    "margins": ("1.0", "0.9", ABSTAIN),
    "preview": ("0.18", "0.35", ABSTAIN),
    "latency": ("0.05", "0.08", ABSTAIN),
    "lookahead": ("7/0.35", "8/0.40", ABSTAIN),
    "yaw_gain": ("0.7", "0.8", ABSTAIN),
    "kv": ("1.9", "1.6", ABSTAIN),
    "brake_gain": ("1.04", "1.08", ABSTAIN),
    "stops": ("lap1-medium-180", "lap2-soft-160", ABSTAIN),
    "front_slide_steer": ("0.75", "0.5", ABSTAIN),
    "front_slide_brake": ("0.75", "0.6", ABSTAIN),
}
PREFIX = {"alpha": "value", "beta": "setting", "gamma": "option"}
TICK_ELEMENTS = ("burst", "settle", "peek_trigger")
FPS_ELEMENTS = (
    "site", "buy", "steer_speed", "burst", "settle", "peek_trigger",
    "peek_cadence", "escape", "enemy_memory",
)
RACING_ELEMENTS = (
    "margins", "preview", "latency", "lookahead", "yaw_gain", "kv",
    "brake_gain", "stops", "front_slide_steer", "front_slide_brake",
)
WORD = re.compile(r"[a-z0-9./_+=-]+")


def write_dataset() -> Path:
    examples, evidence = build()
    assert_holdout(examples, extra_texts=list(DRAFT_NOTE.values()))
    DATA.mkdir(parents=True, exist_ok=True)
    path = DATA / "examples.jsonl"
    with path.open("w") as handle:
        for example in examples:
            handle.write(json.dumps(example, ensure_ascii=False) + "\n")
    (DATA / "draft_evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
    (DATA / "manifest.json").write_text(json.dumps(manifest(examples), indent=2) + "\n")
    return path


def load_examples() -> list[dict]:
    path = DATA / "examples.jsonl"
    if not path.exists():
        write_dataset()
    rows = []
    with path.open() as handle:
        for line in handle:
            rows.append(json.loads(line))
    return rows


def load_evidence() -> dict:
    path = DATA / "draft_evidence.json"
    if not path.exists():
        write_dataset()
    return json.loads(path.read_text())


def manifest(examples: list[dict]) -> dict:
    splits = defaultdict(int)
    origins = defaultdict(int)
    for example in examples:
        splits[example["split"]] += 1
        origins[example["origin"]] += 1
    return {"examples": len(examples), "splits": dict(splits), "origins": dict(origins), "cap": CAP}


def build() -> tuple[list[dict], dict]:
    examples = []
    reservoirs: dict[tuple, dict] = {}
    supported = {name: defaultdict(int) for name in TICK_ELEMENTS}
    by_split = {split: {name: defaultdict(int) for name in TICK_ELEMENTS} for split in FAMILY}
    subsets = {split: {name: defaultdict(int) for name in ("burst_fire", "aim_band", "peek_band", "speed_gap")} for split in FAMILY}
    rng = random.Random(SEED)
    with SHARD.open() as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("record") == "header" or "label" not in row:
                continue
            facts = _shard_facts(row)
            split = _split_for_step(int(row["step"]))
            for element in TICK_ELEMENTS:
                key, trace = judge_tick(element, facts)
                if facts["readiness"] in SUPPORTED:
                    supported[element][key] += 1
                    by_split[split][element][key] += 1
                _subset(subsets[split], element, facts, key)
                item = _tick_example(element, facts, split, key, trace, rng)
                _keep(reservoirs, (split, element, key), item, rng)
    for bucket in reservoirs.values():
        examples.extend(bucket["rows"])
    examples.extend(_synthetics(rng))
    examples.extend(_summaries(by_split, subsets, rng))
    examples.extend(_notes(rng))
    examples.sort(key=lambda item: (item["split"], item["element"], item["situation_id"], item["origin"]))
    evidence = {
        "supported": {name: dict(supported[name]) for name in TICK_ELEMENTS},
        "source_shard": str(SHARD),
        "notes": DRAFT_NOTE,
    }
    return examples, evidence


def judge_tick(element: str, facts: dict) -> tuple[str, str]:
    if element == "burst":
        return _burst(facts)
    if element == "peek_trigger":
        return _peek(facts)
    if element == "settle":
        return _speed(facts)
    raise KeyError(element)


def draft_cases(game: str, evidence: dict | None = None) -> list[dict]:
    evidence = evidence or load_evidence()
    if game == "fps":
        elements = FPS_ELEMENTS
    elif game == "racing":
        elements = RACING_ELEMENTS
    else:
        raise ValueError(game)
    cases = []
    for element in elements:
        if element in TICK_ELEMENTS:
            counts = evidence["supported"][element]
            question, situation = _render_summary("gamma", element, counts)
            kind = "supported tally"
        else:
            question = QUESTION["gamma"]
            situation = evidence["notes"][element]
            counts = None
            kind = "record note"
        cases.append({
            "element": element,
            "kind": kind,
            "question": question,
            "situation": situation,
            "candidates": _candidates(element, "gamma", random.Random(element)),
            "counts": counts,
        })
    return cases


def assert_holdout(examples: list[dict], extra_texts: list[str] | None = None) -> None:
    by_id = defaultdict(set)
    alpha, heldout = [], []
    for example in examples:
        by_id[example["situation_id"]].add(example["split"])
        text = _model_text(example)
        if example["split"] == "train":
            alpha.append(text)
            if example["render"] != "alpha":
                raise RuntimeError("train row is not alpha text")
        else:
            heldout.append(text)
            if example["split"] == "test" and example["render"] != "gamma":
                raise RuntimeError("test row is not gamma text")
            if example["split"] in ("select", "calibrate") and example["render"] != "beta":
                raise RuntimeError("select row is not beta text")
    leaked = [key for key, splits in by_id.items() if len(splits) > 1]
    if leaked:
        raise RuntimeError(f"situation ids cross splits: {leaked[:5]}")
    shared = shared_ngrams(alpha, heldout + list(extra_texts or []), 4)
    if shared:
        raise RuntimeError(f"train text shares a 4-word window with held-out text: {shared[:5]}")


def shared_ngrams(left: list[str], right: list[str], size: int) -> list[str]:
    pool = set()
    for text in left:
        pool.update(_ngrams(text, size))
    found = []
    seen = set()
    for text in right:
        for gram in _ngrams(text, size):
            if gram in pool and gram not in seen:
                seen.add(gram)
                found.append(gram)
    return found


def _burst(facts: dict) -> tuple[str, str]:
    readiness = facts["readiness"]
    if readiness == "fire":
        return "4", "readiness fire: four shots then the pause"
    if readiness in ("wait", "adjust_aim", "reload"):
        return "9", f"readiness {readiness}: a nine-tick pause withholds a shot the label does not want"
    return ABSTAIN, f"readiness {readiness} does not choose a pause"


def _peek(facts: dict) -> tuple[str, str]:
    distance = facts["distance"]
    readiness = facts["readiness"]
    if distance is None or not 6.0 < distance <= 9.0:
        return ABSTAIN, "distance is outside the 6 m to 9 m band"
    if readiness == "fire":
        return "6", "readiness fire inside the long peek and outside the short peek"
    if readiness == "adjust_aim":
        return "9", "readiness adjust_aim while the enemy is outside the 6 m peek"
    return ABSTAIN, f"readiness {readiness} does not choose a peek"


def _speed(facts: dict) -> tuple[str, str]:
    speed = facts["speed"]
    if speed is None:
        return ABSTAIN, "speed missing"
    if facts["reason"] == "moving" and 0.5 <= speed < 0.6:
        return "0.5", "readiness waits while moving, and only the 0.5 gate blocks this speed"
    return ABSTAIN, "speed does not separate the gates on this tick"


def _shard_facts(row: dict) -> dict:
    vx, vy = row.get("vx"), row.get("vy")
    speed = None
    if _finite(vx) and _finite(vy):
        speed = math.hypot(float(vx), float(vy))
    return {
        "situation_id": f"shard-{row['match_id']}-{row['step']}",
        "origin": "shard",
        "readiness": row["label"],
        "reason": row.get("reason") or "none",
        "speed": speed,
        "aim": float(row["aim_error_rad"]) if _finite(row.get("aim_error_rad")) else None,
        "distance": float(row["nearest_distance_m"]) if _finite(row.get("nearest_distance_m")) else None,
    }


def _split_for_step(step: int) -> str:
    bucket = step % 10
    if bucket <= 5:
        return "train"
    if bucket <= 7:
        return "select"
    if bucket == 8:
        return "calibrate"
    return "test"


def _subset(subsets: dict, element: str, facts: dict, key: str) -> None:
    if element == "burst" and facts["readiness"] == "fire":
        subsets["burst_fire"][key] += 1
    if element == "peek_trigger" and key != ABSTAIN:
        subsets["peek_band"][key] += 1
    if element == "settle" and key != ABSTAIN:
        subsets["speed_gap"][key] += 1


def _tick_example(element, facts, split, key, trace, rng) -> dict:
    family = FAMILY[split]
    question, situation = _render_tick(family, element, facts)
    return _row(element, facts["situation_id"], facts["origin"], split, family, question, situation, key, trace, rng)


def _render_tick(family: str, element: str, facts: dict) -> tuple[str, str]:
    speed, aim, dist = _fmt(facts["speed"]), _fmt(facts["aim"]), _fmt(facts["distance"])
    label, reason = facts["readiness"], facts["reason"]
    if family == "alpha":
        return QUESTION[family], f"Element {element}. Mark {label} via {reason}. Speed {speed}. Aim miss {aim}. Range {dist}."
    if family == "beta":
        return QUESTION[family], f"Item {element}. Tag {label} ({reason}). Pace {speed}. Yaw delta {aim}. Separation {dist}."
    return QUESTION[family], f"Next {element}. Call {label} reason {reason}. Velocity {speed}. Residual {aim}. Gap {dist}."


def _render_summary(family: str, element: str, counts: dict) -> tuple[str, str]:
    ordered = " ".join(f"{key}={int(counts.get(key, 0))}" for key in CANDIDATES[element])
    if family == "alpha":
        return QUESTION[family], f"Element {element}. Supported tally {ordered}."
    if family == "beta":
        return QUESTION[family], f"Item {element}. Kept totals {ordered}."
    return QUESTION[family], f"Next {element}. Live totals {ordered}."


def _synthetics(rng: random.Random) -> list[dict]:
    rows = []
    specs = (
        ("moving-gap", "wait", "moving"),
        ("aim-band", "adjust_aim", "misaligned"),
        ("peek-fire", "fire", "ready"),
        ("peek-aim", "adjust_aim", "misaligned"),
        ("ready-fire", "fire", "ready"),
    )
    for split in ("train", "select", "calibrate", "test"):
        for name, expect_label, expect_reason in specs:
            for index in range(16):
                situation = _synthetic_situation(name, index)
                got, reason = label(situation)
                if got != expect_label or reason != expect_reason:
                    raise RuntimeError(f"{name} labelled {got}/{reason}")
                facts = _facts_from_situation(situation, f"synth-{split}-{name}-{index}")
                for element in TICK_ELEMENTS:
                    key, trace = judge_tick(element, facts)
                    rows.append(_tick_example(element, facts, split, key, trace, rng))
    return rows


def _synthetic_situation(name: str, index: int) -> dict:
    tick = 3000 + index
    if name == "moving-gap":
        return example_situation(vx=0.52 + (index % 5) * 0.01, match_tick=tick)
    if name == "aim-band":
        return example_situation(yaw_extra=0.13, distance=12, match_tick=tick)
    if name == "peek-fire":
        return example_situation(distance=6.2 + (index % 6) * 0.4, match_tick=tick)
    if name == "peek-aim":
        return example_situation(distance=6.2 + (index % 6) * 0.4, yaw_extra=0.4, match_tick=tick)
    if name == "ready-fire":
        return example_situation(match_tick=tick)
    raise KeyError(name)


def _facts_from_situation(situation: dict, situation_id: str) -> dict:
    got, reason = label(situation)
    aim = describe_aim(situation)
    return {
        "situation_id": situation_id,
        "origin": "synthetic-readiness",
        "readiness": got,
        "reason": reason,
        "speed": math.hypot(situation["vx"], situation["vy"]),
        "aim": None if aim is None else float(aim["error"]),
        "distance": None if aim is None else float(aim["distance"]),
    }


def _summaries(by_split, subsets, rng) -> list[dict]:
    rows = []
    extras = {
        "burst": "burst_fire",
        "peek_trigger": "peek_band",
        "settle": "speed_gap",
    }
    for split in FAMILY:
        for element in TICK_ELEMENTS:
            rows.append(_summary_row(element, f"{split}-supported", by_split[split][element], split, rng))
            subset = subsets[split][extras[element]]
            if sum(subset.values()):
                rows.append(_summary_row(element, f"{split}-{extras[element]}", subset, split, rng))
    return [row for row in rows if row is not None]


def _summary_row(element: str, name: str, counts: dict, split: str, rng) -> dict | None:
    if sum(int(value) for value in counts.values()) <= 0:
        return None
    key, trace = _majority(counts)
    family = FAMILY[split]
    question, situation = _render_summary(family, element, counts)
    return _row(element, f"summary-{name}", "shard-summary", split, family, question, situation, key, trace, rng)


def _majority(counts: dict) -> tuple[str, str]:
    total = sum(int(value) for value in counts.values())
    if total <= 0:
        return ABSTAIN, "empty tally"
    best = max(counts, key=lambda key: (int(counts[key]), key))
    if int(counts[best]) * 2 > total:
        return best, f"{best} is {int(counts[best])} of {total} supported rows"
    return ABSTAIN, f"no key is a majority of {total}"


def _notes(rng) -> list[dict]:
    rows = []
    for note in NOTES:
        family = FAMILY[note["split"]]
        question = QUESTION[family]
        rows.append(_row(
            note["element"], note["id"], "record", note["split"], family, question, note["text"],
            note["label"], note["trace"], rng,
        ))
    return rows


def _row(element, situation_id, origin, split, family, question, situation, key, trace, rng) -> dict:
    candidates = _candidates(element, family, rng)
    if key not in {item["key"] for item in candidates}:
        raise RuntimeError(f"{element} label {key} is not a candidate")
    return {
        "situation_id": situation_id,
        "element": element,
        "origin": origin,
        "split": split,
        "render": family,
        "question": question,
        "situation": situation,
        "candidates": candidates,
        "label_key": key,
        "trace": trace,
    }


def _candidates(element: str, family: str, rng: random.Random) -> list[dict]:
    keys = list(CANDIDATES[element])
    ids = rng.sample(ID_POOL, len(keys))
    prefix = PREFIX[family]
    return [
        {"id": token, "key": key, "description": f"{prefix} {key}"}
        for token, key in zip(ids, keys)
    ]


def _keep(reservoirs, key, item, rng) -> None:
    bucket = reservoirs.setdefault(key, {"seen": 0, "rows": []})
    bucket["seen"] += 1
    if len(bucket["rows"]) < CAP:
        bucket["rows"].append(item)
        return
    slot = rng.randrange(bucket["seen"])
    if slot < CAP:
        bucket["rows"][slot] = item


def _model_text(example: dict) -> str:
    pieces = [example["question"], example["situation"]]
    pieces.extend(item["description"] for item in example["candidates"])
    return " ".join(pieces)


def _ngrams(text: str, size: int) -> set[str]:
    words = WORD.findall(text.lower())
    return {" ".join(words[index:index + size]) for index in range(0, max(0, len(words) - size + 1))}


def _fmt(value) -> str:
    if value is None:
        return "missing"
    return f"{float(value):.4f}"


def _finite(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def checkpoint_path() -> Path:
    if not POINTER.exists():
        raise FileNotFoundError("no element checkpoint pointer at state/element_checkpoint.txt")
    path = Path(POINTER.read_text().strip())
    if not path.is_absolute():
        path = ROOT / path
    if not (path / "best.pt").exists():
        raise FileNotFoundError(f"element checkpoint missing best.pt: {path}")
    return path
