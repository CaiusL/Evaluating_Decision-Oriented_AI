"""Outcome-linked preferences between cars that actually raced.

A pair is eligible when both matches are ours, they share a track and a recorded
wet/dry band, and their played setups differ. The label is the setup with the
better recorded outcome. Identical setups, equal outcomes, and setup pairs whose
winner flips are unresolved. Opponent cars are not labels.

The split unit is the connected component of match ids. Train wording does not
share a 4-word window with the other splits or the draft prompt.
"""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

from self_jev.element_data import _ngrams

ROOT = Path(__file__).resolve().parents[1]
IMPORT = ROOT / "data" / "imports" / "comp-racing-history-20261117"
DATA = ROOT / "data" / "outcome"

READ = (
    "qual_margin",
    "race_margin",
    "preview",
    "latency",
    "ld_min",
    "ld_time",
    "yaw_gain",
    "kv",
    "brake_gain",
    "stops",
)
BOOK = (
    "front_wing",
    "rear_wing",
    "fuel_load_kg",
    "ballast_kg",
    "radiator_opening",
    "start_compound",
    "pressure_front_psi",
    "pressure_rear_psi",
    "ers_harvest_level",
)
FIELDS = READ + BOOK

OUTCOME_RULE = (
    "Lower observed_result.rank is the better finishing place. "
    "When both matches have finish_time_s and the same non-null completed_laps, "
    "the lower finish_time_s is the better outcome instead. "
    "Equal outcomes are unresolved."
)
SPLIT_RULE = (
    "Split unit is the connected component of match ids joined by a kept pair. "
    "Components are sorted by minimum match id. The third-from-last component "
    "is select and the second-from-last is calibrate. Test starts as the last "
    "component and absorbs earlier components until it holds at least four "
    "decisive comparisons, without leaving train with fewer than four. "
    "Every example from a match stays in that match's component."
)

# Glue is one word, and it differs by split, so a shared setup cannot form the
# same 4-word window on both sides of the holdout.
# marker and token differ by split. Both candidates in a pair share the token, so a
# random option id cannot be the feature the encoder memorizes. The marker is
# repeated beside each changed field so a 4-word window cannot cross splits.
TEXT = {
    "train": {
        "question": "Choose the recorded car tied to the stronger finishing place.",
        "situation": "Session on {track} in a {wet} record. Both cars raced there.",
        "glue": "with",
        "marker": "delta",
        "token": "car",
    },
    "select": {
        "question": "Mark the car whose finish place led.",
        "situation": "Fixture {track}, moisture band {wet}. Two recorded cars.",
        "glue": "alongside",
        "marker": "aside",
        "token": "cab",
    },
    "calibrate": {
        "question": "Indicate the car with the leading place.",
        "situation": "Venue {track} during {wet} running. Pair of recorded cars.",
        "glue": "beside",
        "marker": "apart",
        "token": "van",
    },
    "test": {
        "question": "Name the car that took the leading place.",
        "situation": "Layout {track} with {wet} conditions. Two supplied cars.",
        "glue": "under",
        "marker": "gap",
        "token": "bus",
    },
}
TRAIN_PARAPHRASE = {
    "question": "Prefer the car whose recorded place was stronger.",
    "situation": "Course {track} under {wet} notes. Compare the two cars.",
    "glue": "plus",
    "marker": "shift",
    "token": "auto",
}
DRAFT_TEXT = {
    "question": "Which supplied car should the next entry keep?",
    "situation": "Upcoming entry. Circuit {track}. Moisture noted as {wet}.",
    "glue": "via",
    "marker": "ahead",
    "token": "rig",
}


def build_examples() -> tuple[list[dict], dict]:
    matches = [
        json.loads(line)
        for line in (IMPORT / "matches.jsonl").read_text().splitlines()
        if line.strip()
    ]
    bundles = [
        json.loads(line)
        for line in (IMPORT / "bundles.jsonl").read_text().splitlines()
        if line.strip()
    ]
    kept, dropped = _kept_pairs(matches)
    components = _components(kept)
    assignment = _assign(kept, components)
    examples = []
    for pair in kept:
        split = assignment[pair["component"]]
        examples.append(_example(pair, split, TEXT[split], paraphrase=False))
        if split == "train":
            examples.append(_example(pair, split, TRAIN_PARAPHRASE, paraphrase=True))
    _assert_split(examples)
    stats = _inventory(matches, bundles, kept, dropped, components, examples)
    return examples, stats


def write_dataset() -> dict:
    examples, stats = build_examples()
    DATA.mkdir(parents=True, exist_ok=True)
    path = DATA / "examples.jsonl"
    with path.open("w") as handle:
        for example in examples:
            handle.write(json.dumps(example, sort_keys=True) + "\n")
    (DATA / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    return stats


def model_text(example: dict) -> str:
    pieces = [example["question"], example["situation"]]
    for candidate in example["candidates"]:
        pieces.append(f"{candidate['id']}: {candidate['description']}")
    return " ".join(pieces)


def describe_pair(setup: dict, other: dict, glue: str, marker: str, fields: tuple[str, ...] | None = None) -> str:
    """Put the fields that differ first. The full setup follows, joined by glue."""
    fields = FIELDS if fields is None else fields
    changed = [
        f"{field}={setup.get(field, 'missing')}"
        for field in fields
        if setup.get(field, "missing") != other.get(field, "missing")
    ]
    body = render_setup(setup, glue, fields)
    if not changed:
        return body
    prefix = " ".join(f"{marker} {item}" for item in changed)
    return f"{prefix} {marker} {body}"


def draft_prompt(track: str, wet: str, setups: list[dict]) -> dict:
    """Held-out wording for a live comparison. Setups are field/value maps."""
    if len(setups) != 2:
        raise RuntimeError("a draft comparison has two setups")
    rng = random.Random("draft-" + track + wet + setup_key(setups[0]) + setup_key(setups[1]))
    order = [0, 1]
    rng.shuffle(order)
    candidates = []
    for index in order:
        other = 1 - index
        candidates.append({
            "id": DRAFT_TEXT["token"],
            "key": setup_key(setups[index]),
            "description": describe_pair(setups[index], setups[other], DRAFT_TEXT["glue"], DRAFT_TEXT["marker"]),
        })
    return {
        "question": DRAFT_TEXT["question"],
        "situation": DRAFT_TEXT["situation"].format(track=track, wet=wet),
        "candidates": candidates,
    }


def render_setup(setup: dict, glue: str, fields: tuple[str, ...] | None = None) -> str:
    fields = FIELDS if fields is None else fields
    parts = [f"{field}={setup.get(field, 'missing')}" for field in fields]
    return f" {glue} ".join(parts)


def setup_key(setup: dict) -> str:
    payload = json.dumps({field: setup.get(field, "missing") for field in FIELDS}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def live_setup(strategy: dict) -> dict:
    return _bag(strategy)


def _kept_pairs(matches: list[dict]) -> tuple[list[dict], dict]:
    dropped = {"wet_or_track": 0, "identical_setup": 0, "equal_outcome": 0, "contradictory_direction": 0}
    grouped: dict[tuple, list[dict]] = {}
    for index, left in enumerate(matches):
        for right in matches[index + 1:]:
            pair, reason = _compare(left, right)
            if reason == "skip":
                continue
            if reason:
                dropped[reason] += 1
                continue
            grouped.setdefault(pair["group_key"], []).append(pair)
    kept = []
    for group in grouped.values():
        winners = {item["label_key"] for item in group}
        if len(winners) == 1:
            kept.extend(group)
        else:
            dropped["contradictory_direction"] += len(group)
    return kept, dropped


def _compare(left: dict, right: dict) -> tuple[dict | None, str | None]:
    if left["track_id"] != right["track_id"]:
        return None, "skip"
    wet_left, wet_right = _wet(left), _wet(right)
    if wet_left == "unknown" or wet_right == "unknown" or wet_left != wet_right:
        return None, "wet_or_track"
    left_setup, right_setup = _bag(left["played_strategy"]), _bag(right["played_strategy"])
    if left_setup == right_setup:
        return None, "identical_setup"
    field, winner = _outcome(left, right)
    if winner is None:
        return None, "equal_outcome"
    winning = left_setup if winner == "left" else right_setup
    group_key = (left["track_id"], wet_left, frozenset((json.dumps(left_setup, sort_keys=True), json.dumps(right_setup, sort_keys=True))))
    return {
        "group_key": group_key,
        "track": left["track_id"],
        "wet": wet_left,
        "match_ids": tuple(sorted((left["match_id"], right["match_id"]))),
        "setups": {setup_key(left_setup): left_setup, setup_key(right_setup): right_setup},
        "label_key": setup_key(winning),
        "outcome_field": field,
        "code_identity": _identity(left, right),
        "ranks": {
            left["match_id"]: left["observed_result"].get("rank"),
            right["match_id"]: right["observed_result"].get("rank"),
        },
    }, None


def _outcome(left: dict, right: dict) -> tuple[str | None, str | None]:
    laps_left = left["recording_facts"].get("completed_laps")
    laps_right = right["recording_facts"].get("completed_laps")
    time_left = left["recording_facts"].get("finish_time_s")
    time_right = right["recording_facts"].get("finish_time_s")
    if laps_left is not None and laps_left == laps_right and time_left is not None and time_right is not None:
        if time_left == time_right:
            return "recording_facts.finish_time_s", None
        return "recording_facts.finish_time_s", "left" if time_left < time_right else "right"
    rank_left = left["observed_result"].get("rank")
    rank_right = right["observed_result"].get("rank")
    if rank_left is None or rank_right is None or rank_left == rank_right:
        return "observed_result.rank", None
    return "observed_result.rank", "left" if rank_left < rank_right else "right"


def _components(pairs: list[dict]) -> dict[str, str]:
    parent: dict[str, str] = {}

    def find(match_id: str) -> str:
        parent.setdefault(match_id, match_id)
        while parent[match_id] != match_id:
            parent[match_id] = parent[parent[match_id]]
            match_id = parent[match_id]
        return match_id

    def union(left: str, right: str) -> None:
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    for pair in pairs:
        union(pair["match_ids"][0], pair["match_ids"][1])
    members: dict[str, list[str]] = {}
    for match_id in parent:
        members.setdefault(find(match_id), []).append(match_id)
    component_of = {}
    for group in members.values():
        name = min(group)
        for match_id in group:
            component_of[match_id] = name
    for pair in pairs:
        pair["component"] = component_of[pair["match_ids"][0]]
    return component_of


def _assign(pairs: list[dict], component_of: dict[str, str]) -> dict[str, str]:
    counts: dict[str, int] = {}
    for pair in pairs:
        counts[pair["component"]] = counts.get(pair["component"], 0) + 1
    names = sorted(set(component_of.values()))
    if len(names) < 4:
        raise RuntimeError(f"need at least 4 match components, found {len(names)}")
    select, calibrate = names[-3], names[-2]
    test = [names[-1]]
    train = names[:-3]

    def total(group: list[str]) -> int:
        return sum(counts.get(name, 0) for name in group)

    while total(test) < 4 and len(train) > 1 and total(train[:-1]) >= 4:
        test.insert(0, train.pop())
    assignment = {name: "train" for name in train}
    assignment[select] = "select"
    assignment[calibrate] = "calibrate"
    for name in test:
        assignment[name] = "test"
    return assignment


def _example(pair: dict, split: str, template: dict, paraphrase: bool) -> dict:
    situation_id = ("paraphrase-" if paraphrase else "") + (
        f"{split}-{pair['match_ids'][0][:8]}-{pair['match_ids'][1][:8]}"
    )
    rng = random.Random(situation_id)
    order = list(pair["setups"])
    rng.shuffle(order)
    candidates = []
    for key in order:
        other = next(item for item in pair["setups"] if item != key)
        candidates.append({
            "id": template["token"],
            "key": key,
            "description": describe_pair(pair["setups"][key], pair["setups"][other], template["glue"], template["marker"]),
        })
    return {
        "situation_id": situation_id,
        "split": split,
        "paraphrase": paraphrase,
        "group_id": pair["component"],
        "match_ids": list(pair["match_ids"]),
        "track": pair["track"],
        "wet": pair["wet"],
        "question": template["question"],
        "situation": template["situation"].format(track=pair["track"], wet=pair["wet"]),
        "candidates": candidates,
        "label_key": pair["label_key"],
        "label_origin": pair["outcome_field"],
        "outcome_rule": OUTCOME_RULE,
        "code_identity": pair["code_identity"],
        "ranks": pair["ranks"],
        "setups": pair["setups"],
    }


def _assert_split(examples: list[dict]) -> None:
    by_split: dict[str, set[str]] = {}
    for example in examples:
        by_split.setdefault(example["split"], set()).update(example["match_ids"])
        text = model_text(example)
        for match_id in example["match_ids"]:
            if match_id in text:
                raise RuntimeError("match id leaked into model text")
    for left_name, right_name in _split_pairs(by_split):
        shared = by_split[left_name] & by_split[right_name]
        if shared:
            raise RuntimeError(f"match ids cross {left_name} and {right_name}")
    train_grams: set[str] = set()
    for example in examples:
        if example["split"] == "train":
            train_grams.update(_ngrams(model_text(example), 4))
    held = [model_text(example) for example in examples if example["split"] != "train"]
    held.extend([
        DRAFT_TEXT["question"],
        DRAFT_TEXT["situation"],
        DRAFT_TEXT["glue"],
        DRAFT_TEXT["marker"],
        DRAFT_TEXT["token"],
    ])
    leaked = []
    for text in held:
        for gram in _ngrams(text, 4):
            if gram in train_grams and gram not in leaked:
                leaked.append(gram)
    if leaked:
        raise RuntimeError(f"train text shares a 4-word window with held-out text: {leaked[:8]}")


def _split_pairs(by_split: dict[str, set[str]]) -> list[tuple[str, str]]:
    names = [name for name in ("train", "select", "calibrate", "test") if name in by_split]
    pairs = []
    for index, left_name in enumerate(names):
        for right_name in names[index + 1:]:
            pairs.append((left_name, right_name))
    return pairs


def _inventory(matches, bundles, kept, dropped, components, examples) -> dict:
    ranks = sum(match["observed_result"].get("rank") is not None for match in matches)
    finish_times = sum(match["recording_facts"].get("finish_time_s") is not None for match in matches)
    laps = sum(match["recording_facts"].get("completed_laps") is not None for match in matches)
    car_rows = 0
    for match in matches:
        cars = match["recording_facts"].get("cars")
        if isinstance(cars, list):
            car_rows += len(cars)
    origins = {}
    for example in examples:
        if not example["paraphrase"]:
            origins[example["label_origin"]] = origins.get(example["label_origin"], 0) + 1
    return {
        "matches": len(matches),
        "bundles": len(bundles),
        "bundle_links": sum(len(match.get("bundle_links") or []) for match in matches),
        "linked_identity": sum(match["code_identity"] == "linked_by_recorded_identity" for match in matches),
        "unresolved_identity": sum(match["code_identity"] != "linked_by_recorded_identity" for match in matches),
        "played_strategies": sum(bool(match.get("played_strategy")) for match in matches),
        "rank_fields": ranks,
        "finish_time_fields": finish_times,
        "completed_lap_fields": laps,
        "recording_car_rows": car_rows,
        "dropped": dropped,
        "decisive_pairs": len(kept),
        "examples": len(examples),
        "components": len(set(components.values())),
        "label_origins": origins,
        "split_counts": {
            split: sum(1 for row in examples if row["split"] == split and not row["paraphrase"])
            for split in ("train", "select", "calibrate", "test")
        },
        "outcome_rule": OUTCOME_RULE,
        "split_rule": SPLIT_RULE,
        "split_unit": "match_component",
    }


def _bag(strategy: dict) -> dict:
    values = {}
    for field in FIELDS:
        value = strategy.get(field)
        if field == "stops":
            value = tuple(
                f"{stop.get('lap')}-{stop.get('compound')}-{stop.get('request_ahead_m')}"
                for stop in value or []
            )
        values[field] = _canon(value)
    return values


def _canon(value) -> str:
    if isinstance(value, tuple):
        return "-".join(_canon(part) for part in value)
    if isinstance(value, float):
        return format(value, ".4f").rstrip("0").rstrip(".")
    if value is None:
        return "missing"
    return str(value)


def _wet(match: dict) -> str:
    flag = match["recording_facts"].get("wet_race")
    if flag is True:
        return "wet"
    if flag is False:
        return "dry"
    return "unknown"


def _identity(left: dict, right: dict) -> str:
    if left["code_identity"] == "linked_by_recorded_identity" and right["code_identity"] == "linked_by_recorded_identity":
        return "both_linked"
    return "unresolved_code"
