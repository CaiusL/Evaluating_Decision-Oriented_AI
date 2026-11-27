"""Pace-cap outcome rows and the gate that refuses a flat training set.

One pace row is one version that was racing when its pace was measured. A pair
is kept only when both cars were racing, the pace source or the speed cap
differs, and the measured pace differs. Rank is not the label. A damage
retirement is a separate label and is not a slower speed cap.
"""
from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path

from self_jev.outcome_data import (
    SPLIT_RULE,
    TEXT,
    TRAIN_PARAPHRASE,
    _assert_split,
    _assign,
    _canon,
    _components,
    describe_pair,
)

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "data" / "pace_outcome" / "examples.jsonl"
RETIREMENTS = ROOT / "data" / "pace_outcome" / "retirements.jsonl"
POINTER = ROOT / "state" / "pace_outcome_checkpoint.txt"
REPORT = ROOT / "reports" / "pace_outcome_model.json"
PACE_RETIREMENT_RULE = (
    "A row labeled damage retirement or damage-retired, or whose match is listed in "
    "data/pace_outcome/retirements.jsonl, is not a pace row."
)

CONTROLS = (
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
COLD_FIELDS = ("compound", "fuel_load_kg", "wetness", "tyre_temp_c")
PACE_FIELDS = CONTROLS + ("pace_source", "speed_cap") + COLD_FIELDS
PACE_NUMBERS = ("race_mean_speed_mps", "peak_speed_mps", "peak_lateral_g")

MISSING_DATA = (
    "Missing data: data/pace_outcome/examples.jsonl is absent. "
    "Version 12e5e35b match a2585d41 finished parkland still racing: "
    "mean 52.10 m/s, peak 92.84 m/s, peak lateral 3.23 g, clean lap 103.0 s, "
    "speed percentiles 21.7/52.1/84.8. "
    "Version d2b6ac91 match d124dc58 finished highland still racing: "
    "mean 38.38 m/s, peak 79.14 m/s, peak lateral 2.56 g. "
    "Those two races are different tracks, so they are not a pace pair. "
    "Version 5 job 6214cc20 finished parkland still racing, seat 3, rank 4, 4 laps in 432.638 s. "
    "Qualifying was valid at 100.798 s. It pitted from soft to medium. "
    "The race condition reached damp at wetness 0.250 and did not reach wet. "
    "While racing: mean 53.64 m/s, peak 95.63 m/s, peak lateral 3.087 g. "
    "a2585d41 is grip on the same track at peak wetness 0.250, mean 52.10, peak 92.84, lateral 3.23. "
    "The cold car is ahead on mean and peak, so this pair does not publish the grip floor. "
    "It is not a speed-table answer. sj4-pace has preview 0.18, yaw_gain 0.7, fuel 9, and ballast 20. "
    "sj6-track has preview 0.33, yaw_gain 0.8, fuel 11, and ballast 40. "
    "The speed-table question remains open. "
    "The 90 m/s floor already lifts the parkland peak. "
    "The 103.0 s lap is medium and the compared 90.5 s lap is soft, so those laps are not an equal-tyre target comparison. "
    "The same-compound laps are lap 0 on soft and include the start: J 110.0 s, mean 52.82 m/s, peak 88.08 m/s, "
    "against the rank-0 car at 92.9 s, mean 62.24 m/s, peak 94.62 m/s. That opponent lap is not a setup label. "
    "The driver scales the pre-limit speed with race_margin in prepare_pace "
    "(squared = raw * race_margin ** 2). Every kept outcome pair has race_margin 1. "
    "The only other played value is 0.995 on parkland matches 4761d4f5, 7bf8515d, and 828f3f2d, "
    "and their wet_race is absent, so they are not a pair. A margin pair needs two still-racing "
    "finishes on the same track and the same wet band, with different race_margin and a measured "
    "lap time or finish time. Corner speed is then cut by the fixed 0.985 ** 2 * ay / curve term, "
    "which no raced control changes. "
    "GET /api/v1/python-submissions/jobs/{job}/training.jsonl for d124dc58 seat 9 and "
    "5a14af31 seat 5 is still missing. The a2585d41 seat 0 export is complete: 16291 decisions, "
    "sha256 539e87b155174c492e67c4673d989f050745bdae20179fb5c833be6cc0d18fcc. "
    "Replaying sj4-pace matches throttle, brake, and ers_mode on every row. "
    "While racing, the free envelope averages 76.65 m/s, the follow cap 71.42, "
    "the longitudinal stopping cap 63.17, and the car 52.09. "
    "Of the stopping-cap cuts, 2497 meter-seconds are an in-lane car and 1118 are a car "
    "already 2.7 m off line. That names the stopping guard. It is not a pace_source label "
    "and not a higher speed cap. A guard pair needs a second still-racing finish on the same "
    "track and wet band, with a different stopping rule and a measured lap time. "
    "Saved in state/racing_matches/a2585d41-trace-caps.json. "
    "sj7-gap version 6 e9c57c03 races that second stopping rule on the a2585d41 setup. "
    "A stable or growing gap eases the limit, and a closing gap keeps the emergency stop. "
    "Preview, yaw, fuel, ballast, and the grip table stay as they were. "
    "Version 6 jobs 21f76602, c767b15a, and ebe0e4b9 are not that label. "
    "21f76602 is highland, seat 2, rank 8, damage retirement on lap 2. "
    "c767b15a is wet parkland, peak wetness 0.999972, seat 8, rank 8, damage retirement on lap 0. "
    "ebe0e4b9 is damp parkland, peak wetness 0.249972, seat 6, rank 5, damage retirement on lap 3. "
    "J has no finish event in any of the three. "
    "049a8010 is highland, seat 4, 8th, damage retirement after 1 lap. "
    "f96edeef is highland, seat 9, 5th, damage retirement after 3 laps. "
    "37b81ebd is highland, seat 4, 10th, damage retirement on lap 0. "
    "08569f9b is 7th and a damage retirement after 2 laps. "
    "None of those version 6 races is a still-racing lap time. "
    "sj8-energy restores the emergency stop and turns the energy budget on for lap 0. "
    "Preview, yaw, fuel, ballast, and grip stay at the a2585d41 values. "
    "The measured lap time is still missing. A label needs a still-racing damp Parkland finish of that car. "
    "Lap 0 distance is measured from each car's own grid slot, because raw s already starts near 5770. "
    "Both cars are on soft. J is 3.03 s behind at 400 m and 17.12 s behind when the lap ends, "
    "110.033 s versus 92.917 s. Grid order is 9, 2, 3, 0. Qualifying is 91.026 s for seat 0 and "
    "89.641 s for seat 9. Contact with seat 4 is at 1025.25 s, s 70.58. "
    "Seat 0's race ERS mode 3 is recorded 114 times and state of charge falls to 0.03 MJ. "
    "That is one arm. The missing arm is another still-racing finish on the same track and wet band "
    "with a different ERS deployment and a measured lap time or finish time. "
    "Seat 9's deployment is not that arm. "
    "On the clean medium lap 3, tyres are 57-63 C. The 4500-5500 m straights reach within 6 m/s "
    "of this grip envelope at full throttle. Braking at s 829-913 while the local envelope is still 96 "
    "is the stop for the corner near s 1128. The 16.9 m/s average gap is not a higher straight-cap label. "
    "The qualifying lap of the same version is clear air, soft, 91.026 s, tyres 65-89 C. "
    "From 0-1000 m it is within 2 m/s of this grip envelope at full throttle. "
    "The apex at s 1100 is 20.4 m/s against an envelope of 22.0. "
    "That lap does not ask for a higher straight cap. "
    "The next round payload names 2026-11-17T18:00:00Z and does not name a track. "
    "Qualifying at s 750-950 is 94 m/s with brake 0. Lap 3 at s 742 is 82 m/s with brake 0.21, "
    "and the frame weather is damp with track wetness 0.163. Seat 6 is 91.7 m ahead, dn 2.14, at 81 m/s. "
    "The replay envelope is 95.9. path_value stays at 95.9 and the longitudinal guard stopping cap is 81.7, "
    "then the replay brake is 0 against the recorded 0.21. "
    "At lights-out the same version's envelope is 90 m/s and the guard target is 5.3 m/s. "
    "Seat 2 is 16 m ahead and 0.24 m off line. The recorded throttle is 0.43. "
    "The replay throttle is 0.31 and the brake is 0. Cars with nobody in their lane show throttle 1. "
    "The 18:00Z round assigned job 5f2ea146 on version 4. It finished parkland, seat 8, rank 3, "
    "4 laps in 458.811 s. The winner, seat 6, did 4 laps in 406.598 s and did not pit. "
    "From lights-out at 1161.283 s the winner's laps are 95.02, 94.70, 101.70 and 115.20 s. "
    "J's observation laps are 126.83, 101.37, 113.30 and 117.32 s. "
    "The winner's first two laps end at 1256.3 and 1351.0, before wetness reaches 0.50 at t 1380, "
    "so those 95.02 and 94.70 s laps are not wet laps. "
    "J lap 0 is 126.83 s with 13.7 s inside 8 m of another car, and the battery is under 0.15 MJ "
    "for 35.3 s of full throttle. Lap 0 includes contact with seat 7 at t 1169.7 s, s 206, "
    "after which seat 7 retires, and contact with seat 2 at t 1228.8 s, s 2794. "
    "At t 1169.4 seat 7 is 5.5 m behind and closing. "
    "J lap 2 is 113.30 s, wetness 0.540 to 0.645, with 0.6 s inside 8 m. "
    "The winner's only lap entirely after t 1440, when wetness is 0.75, is 115.20 s. "
    "J's clean wet lap is faster than that lap, so the wet race is not a higher-cap label. "
    "J did not pit. "
    "Seats 4 and 9 pitted to medium and back to soft and finished behind J. Those stops are not a setup label. "
    "Peak race wetness is 0.749972. Match a2585d41 peaked at 0.249972, so these two parkland finishes "
    "are not the same wet band and are not a pace pair. "
    "While racing, 5f2ea146 mean is 50.55 m/s, peak 93.66 m/s, peak lateral 2.965 g. "
    "a2585d41 mean is 52.10 m/s, peak 92.84 m/s, peak lateral 3.23 g. "
    "The higher cold peak is not a pace label across those wet bands. "
    "At t 1426.8 s, s 1833 m, wetness 0.695, the cold envelope is 89.0 m/s and the car is at 77.6 m/s, "
    "throttle 1, still accelerating to 82.2 m/s before braking at s 2019. That gap is not a higher-cap label. "
    "Finished matches from other accounts are not J's track. "
    "Version d2b6ac91 on highland has no qualifying lap time; the event reason is retired at race start. "
    "The contact is in the race, 15.8 s in, with seat 7 at s 237. "
    "That version's own highland envelope maxes at 85 m/s. The recorded peak is 79. "
    "That peak is not a higher-cap label. "
    "Version 5 dadb68c1 selects the same cold-cap driver after comparing track distance in meters. "
    "Version 5 then raced four times. 6214cc20 was fifth on parkland. "
    "2f37ed84 and a462f780 retired from damage on lap 0 at crossover, peak wetness 1.0. "
    "2f3ce9c4 retired from damage on lap 3 at highland, peak wetness 0.75, after a valid qualifying lap of 201.059 s. "
    "Those three retirements are not pace rows. No race has rank 0. "
    "The training log for 5f2ea146 seat 8 has 13765 race decisions and no faults. "
    "Of 6343 full-throttle race rows, 4682 use ERS mode 1, 819 mode 2, and 842 mode 3. "
    "At 1195.5 s, still lap 0, s 1279 m, speed 62.5 m/s, throttle is full and state of charge is 0.037 MJ. "
    "Lights-out state of charge is 3.0 MJ at s 5747. It is 0.49 MJ by s 673 and 0.05 MJ by s 1276, "
    "which is after the 0-1012 m window and before the 3104-4028 m and 4368-5196 m windows. "
    "The energy budget stays inactive until lap 1, so this early spend is one arm. "
    "The missing arm is a still-racing finish on the same track and wet band whose lap-0 deployment "
    "does not fall to 0.04 MJ that early, with a measured lap time. "
    "No checkpoint was written."
)
RETIREMENT_REFUSAL = (
    "refusing to train: a damage retirement is its own label and is not a pace row. "
    "It is not stored as a slower speed cap. No checkpoint was written."
)
CAP_TRUST_RULE = (
    "trusted if and only if the selected checkpoint fits the training pairs "
    "(train pairwise accuracy at least 0.75) and its held-out pairwise accuracy "
    "on pairs that differ on pace_source or the speed cap is at least 0.10 above "
    "the unadapted MiniLM on those same pairs. Overall accuracy on preview and yaw "
    "does not count. The pointer moves only when this holds."
)
LABEL_RULE = "higher race mean speed, then higher peak speed, then higher peak lateral g"


class PaceRefusal(RuntimeError):
    """The pace file cannot be trained. No checkpoint should be written."""


def fail(message: str) -> None:
    print(message, file=sys.stderr)
    raise SystemExit(1)


def refuse_flat_outcome() -> None:
    """Stop outcome training when the built pairs cannot move the cap."""
    from self_jev.outcome_data import build_examples

    examples, _stats = build_examples()
    message = outcome_refusal_message(examples)
    if message:
        fail(message)


def outcome_refusal_message(examples: list[dict]) -> str | None:
    pace, cap, differing = _outcome_cap_stats(examples)
    if len(pace) > 1 or len(cap) > 1:
        return None
    return (
        "refusing to train: the built outcome dataset has zero variance on pace_source "
        f"and zero variance on the speed cap. {differing} pairs differ on either field. "
        "Preview and yaw accuracy cannot move the cold-tyre cap. "
        "See reports/jev_data_plan.md. Examples and checkpoints were left in place."
    )


MEASURED_RACES = (
    ROOT / "state" / "racing_matches" / "a2585d41-pace-measured.json",
    ROOT / "state" / "racing_matches" / "d124dc58-finish.json",
)


def measured_tracks() -> list[tuple[str, str]]:
    """Match id and track from the saved racing measurements. No pace pair is inferred."""
    found = []
    for path in MEASURED_RACES:
        if not path.is_file():
            continue
        row = json.loads(path.read_text())
        match_id = row.get("match_id")
        track = row.get("track_id")
        if match_id and track:
            found.append((str(match_id), str(track)))
    return found


# strategy.py in the zip that raced this version. Both set pace_source grip, and the
# driver lifts that source with a 90 m/s floor. Neither value is a measured lap time.
DOWNLOADED_CARS = {
    "e9c57c03-8bb3-4b2d-90eb-568b1d067a3b": {"name": "sj7-gap", "pace_source": "grip", "speed_cap": 90},
    "641d0146-fb80-4fa4-9e5b-5b6cb23bacc2": {"name": "sj8-energy", "pace_source": "grip", "speed_cap": 90},
}
DOWNLOADS = ROOT / "data" / "DL"


def _downloaded_retirement(folder: Path) -> dict | None:
    """One damage retirement from a downloaded match. A finish is not rewritten into a pace row."""
    training = next(folder.glob("training-*.jsonl"), None)
    events_path = folder / "public-events.jsonl"
    static_path = folder / "public-static.jsonl"
    if training is None or not events_path.is_file() or not static_path.is_file():
        return None
    metadata = outcome = None
    with training.open() as stream:
        for line in stream:
            if line.startswith('{"type":"metadata"'):
                metadata = json.loads(line)
            elif line.startswith('{"type":"outcome"'):
                outcome = json.loads(line)
    if not metadata or not outcome:
        return None
    participant = (metadata.get("participants") or [None])[0]
    if not isinstance(participant, dict) or not isinstance(participant.get("seat"), int):
        return None
    seat = participant["seat"]
    classification = next(
        (row for row in outcome.get("classification") or [] if row.get("seat") == seat),
        None,
    )
    if classification is None:
        return None
    track = json.loads(static_path.open().readline())["metadata"]["track"]["id"]
    peak = 0.0
    retired = None
    incidents = []
    with events_path.open() as stream:
        for line in stream:
            if not any(kind in line for kind in ('"kind":"weather_change"', '"kind":"retired"', '"kind":"incident"', '"kind":"finished"')):
                continue
            event = json.loads(line)
            seats = event.get("seats") or []
            kind = event.get("kind")
            if kind == "weather_change":
                peak = max(peak, float((event.get("payload") or {}).get("track_wetness") or 0))
            elif kind == "finished" and seat in seats:
                return None
            elif kind == "retired" and seat in seats:
                retired = event
            elif kind == "incident" and seat in seats:
                incidents.append(event)
    if retired is None or (retired.get("payload") or {}).get("reason") != "damage":
        return None
    prior = [event for event in incidents if event.get("time", 0) <= retired.get("time", 0)]
    incident = (prior or [None])[-1]
    payload = retired.get("payload") or {}
    retirement = {"reason": "damage"}
    if "lap" in payload:
        retirement["lap"] = payload["lap"]
    if "time" in retired:
        retirement["time_s"] = retired["time"]
    if incident is not None:
        detail = incident.get("payload") or {}
        if detail.get("reason"):
            retirement["incident"] = detail["reason"]
        if "s" in detail:
            retirement["s"] = detail["s"]
    ranks = outcome.get("ranks") or []
    car = DOWNLOADED_CARS.get(participant.get("version_id"), {})
    row = {
        "version_id": participant.get("version_id"),
        "name": car.get("name"),
        "match_id": metadata.get("job_id") or folder.name,
        "account": "J",
        "seat": seat,
        "rank": ranks[seat] if isinstance(ranks, list) and seat < len(ranks) else None,
        "ranks": ranks,
        "label": "damage retirement",
        "status": classification.get("status"),
        "racing": False,
        "pace_source": car.get("pace_source"),
        "speed_cap": car.get("speed_cap"),
        "track_id": track,
        "peak_track_wetness": round(peak, 6),
        "retirement": retirement,
    }
    return {key: value for key, value in row.items() if value not in (None, "")}


def record_downloaded_retirements() -> int:
    """Store data/DL damage retirements. Returns how many pace rows were written, which is zero."""
    index_path = DOWNLOADS / "index.json"
    if not index_path.is_file():
        return 0
    folders = []
    for item in json.loads(index_path.read_text()):
        match_id = item.get("match_id")
        if match_id:
            folders.append(DOWNLOADS / match_id)
    found = {}
    for folder in folders:
        row = _downloaded_retirement(folder)
        if row and row.get("match_id"):
            found[row["match_id"]] = row
    if not found:
        return 0
    kept = []
    seen = set()
    if RETIREMENTS.is_file():
        for line in RETIREMENTS.read_text().splitlines():
            if not line.strip():
                continue
            current = json.loads(line)
            match_id = current.get("match_id")
            if match_id in found:
                kept.append(json.dumps(found[match_id], ensure_ascii=False))
                seen.add(match_id)
            else:
                kept.append(line)
    for match_id, row in found.items():
        if match_id not in seen:
            kept.append(json.dumps(row, ensure_ascii=False))
    text = "\n".join(kept) + "\n"
    if not RETIREMENTS.is_file() or RETIREMENTS.read_text() != text:
        RETIREMENTS.parent.mkdir(parents=True, exist_ok=True)
        RETIREMENTS.write_text(text)
    return 0


def main() -> None:
    record_downloaded_retirements()
    if not EXAMPLES.is_file():
        tracks = dict(measured_tracks())
        needed = {
            "a2585d41-ff5d-4175-8a56-d95413d3a662": "parkland",
            "d124dc58-0057-4880-bcc2-14159ebab663": "highland",
        }
        if any(tracks.get(match) != track for match, track in needed.items()):
            fail(
                "refusing to train: saved race measurements do not match the parkland and highland rows. "
                "No checkpoint was written."
            )
        fail(MISSING_DATA)
    try:
        rows = load_rows(EXAMPLES)
    except PaceRefusal as exc:
        fail(str(exc))
    message = row_variance_message(rows)
    if message:
        fail(message)
    try:
        examples, stats = rows_to_examples(rows)
    except PaceRefusal as exc:
        fail(str(exc))
    from self_jev.outcome_train import train

    train(prepared=pace_job(examples, stats, EXAMPLES))


def load_rows(path: Path) -> list[dict]:
    rows = []
    missing: list[str] = []
    for number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise PaceRefusal(
                f"refusing to train: {_shown(path)} line {number} is not JSON. No checkpoint was written."
            ) from exc
        if not isinstance(row, dict):
            raise PaceRefusal(
                f"refusing to train: {_shown(path)} line {number} is not a row. No checkpoint was written."
            )
        if _is_damage_retirement(row):
            raise PaceRefusal(RETIREMENT_REFUSAL)
        missing.extend(f"line {number} {name}" for name in _missing_fields(row))
        rows.append(row)
    if missing:
        shown = ", ".join(missing[:12])
        raise PaceRefusal(
            f"refusing to train: {_shown(path)} is missing {shown}. No checkpoint was written."
        )
    if not rows:
        raise PaceRefusal(f"refusing to train: {_shown(path)} has no rows. No checkpoint was written.")
    return rows


def _shown(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return path.name


def row_variance_message(rows: list[dict]) -> str | None:
    pace = {_canon_source(row.get("pace_source")) for row in rows}
    cap = {_canon_cap(row.get("speed_cap")) for row in rows}
    if len(pace) > 1 or len(cap) > 1:
        return None
    return (
        "refusing to train: data/pace_outcome/examples.jsonl has zero variance on pace_source "
        "and zero variance on the speed cap. No checkpoint was written."
    )


def rows_to_examples(rows: list[dict]) -> tuple[list[dict], dict]:
    kept = _pairs(rows)
    if not kept:
        raise PaceRefusal(
            "refusing to train: data/pace_outcome/examples.jsonl has no pair where pace_source "
            "or the speed cap differs and the measured pace differs. No checkpoint was written."
        )
    try:
        components = _components(kept)
        assignment = _assign(kept, components)
    except RuntimeError as exc:
        if "match components" not in str(exc):
            raise
        raise PaceRefusal(
            "refusing to train: the pace rows do not form enough match components to hold out a cap pair. "
            "No checkpoint was written."
        ) from exc
    examples = []
    for pair in kept:
        split = assignment[pair["component"]]
        examples.append(_example(pair, split, TEXT[split], paraphrase=False))
        if split == "train":
            examples.append(_example(pair, split, TRAIN_PARAPHRASE, paraphrase=True))
    _assert_split(examples)
    test_caps = [
        row for row in examples
        if row["split"] == "test" and not row["paraphrase"] and row["differs_on_cap"]
    ]
    if not test_caps:
        raise PaceRefusal(
            "refusing to train: the match-component split held out no pair that differs on the cap. "
            "No checkpoint was written."
        )
    stats = {
        "rows": len(rows),
        "decisive_pairs": len(kept),
        "examples": len(examples),
        "components": len(set(components.values())),
        "pace_source_values": sorted({_canon_source(row["pace_source"]) for row in rows}),
        "speed_cap_values": sorted({_canon_cap(row["speed_cap"]) for row in rows}),
        "label_rule": LABEL_RULE,
        "split_counts": {
            split: sum(1 for row in examples if row["split"] == split and not row["paraphrase"])
            for split in ("train", "select", "calibrate", "test")
        },
        "split_rule": SPLIT_RULE,
        "split_unit": "match_component",
    }
    return examples, stats


def pace_job(examples: list[dict], stats: dict, dataset_path: Path) -> dict:
    return {
        "examples": examples,
        "stats": stats,
        "dataset_path": dataset_path,
        "task": "racing-pace-outcome",
        "run_prefix": "pace-outcome",
        "report_path": REPORT,
        "pointer": POINTER,
        "trust_mode": "cap",
        "cause": (
            "Outcome pairs do not differ on pace_source or the speed cap, so this run uses rows "
            "whose cap behavior differs and whose measured pace differs. Changed fields are prefixed, "
            "both candidates share a neutral id, and the learning rate is 5e-5 with no weight decay."
        ),
        "publish_outcome": False,
        "trust_rule": CAP_TRUST_RULE,
        "min_test_comparisons": 1,
        "held_out_floor": None,
        "held_out_bar": (
            "at least 0.10 above the unadapted encoder on held-out pairs that differ on "
            "pace_source or the speed cap. Preview and yaw accuracy does not count."
        ),
        "not": (
            "Does not replace the outcome checkpoint, readiness, elements, or racing-pace, "
            "and does not update runs/latest or state/outcome_checkpoint.txt."
        ),
    }


def _outcome_cap_stats(examples: list[dict]) -> tuple[set[str], set[str], int]:
    pace: set[str] = set()
    cap: set[str] = set()
    differing = 0
    for row in examples:
        if row.get("paraphrase"):
            continue
        setups = list(row.get("setups", {}).values())
        for setup in setups:
            pace.add(str(setup.get("pace_source", "missing")))
            cap.add(str(setup.get("speed_cap", "missing")))
        if len(setups) >= 2:
            left, right = setups[0], setups[1]
            pace_differs = left.get("pace_source", "missing") != right.get("pace_source", "missing")
            cap_differs = left.get("speed_cap", "missing") != right.get("speed_cap", "missing")
            if pace_differs or cap_differs:
                differing += 1
    return pace, cap, differing


def _missing_fields(row: dict) -> list[str]:
    missing = []
    for key in ("version_id", "match_id", "track", "wet", "controls", "pace_source", "speed_cap", "cold_table", "pace"):
        if key not in row or row[key] in ("", None):
            missing.append(key)
    if row.get("racing") is not True:
        missing.append("racing")
    if row.get("wet") not in (None, "", "wet", "dry"):
        missing.append("wet")
    controls = row.get("controls")
    if isinstance(controls, dict):
        for name in CONTROLS:
            if name not in controls or controls[name] in ("", None):
                missing.append(f"controls.{name}")
    elif "controls" in row and row.get("controls") not in ("", None):
        missing.append("controls")
    cold = row.get("cold_table")
    if isinstance(cold, dict):
        for name in ("compound", "fuel_load_kg", "wetness"):
            if name not in cold or cold[name] in ("", None):
                missing.append(f"cold_table.{name}")
        if "tyre_temp_c" not in cold:
            missing.append("cold_table.tyre_temp_c")
    elif "cold_table" in row and row.get("cold_table") not in ("", None):
        missing.append("cold_table")
    pace = row.get("pace")
    if isinstance(pace, dict):
        for name in PACE_NUMBERS:
            if not _is_number(pace.get(name)):
                missing.append(f"pace.{name}")
    elif "pace" in row and row.get("pace") not in ("", None):
        missing.append("pace")
    if "pace_source" in row and row.get("pace_source") not in ("", None) and not isinstance(row.get("pace_source"), str):
        missing.append("pace_source")
    if "speed_cap" in row and row.get("speed_cap") not in ("", None) and not _is_cap(row.get("speed_cap")):
        missing.append("speed_cap")
    return missing


def _pairs(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        if not _was_racing(row):
            continue
        grouped.setdefault((row["track"], row["wet"]), []).append(row)
    kept = []
    for (track, wet), group in grouped.items():
        for index, left in enumerate(group):
            for right in group[index + 1:]:
                pair = _compare(left, right, track, wet)
                if pair:
                    kept.append(pair)
    return kept


def _is_damage_retirement(row: dict) -> bool:
    """A contact or damage retirement is not a pace measurement."""
    if _retirement_markers(row):
        return True
    return _matches_listed_retirement(row.get("match_id"))


def _retirement_markers(row: dict) -> bool:
    label = _norm_label(row.get("label"))
    if label in {"damage retirement", "damage retired", "retirement", "retired"}:
        return True
    status = _norm_label(row.get("status"))
    if status in {"retired", "damage", "damage retirement", "damage retired"}:
        return True
    event = row.get("retirement")
    if isinstance(event, dict):
        reason = _norm_label(event.get("reason"))
        if reason in {"damage", "contact", "damage retirement", "damage retired"}:
            return True
        if event.get("contact_time_s") not in (None, ""):
            return True
    return False


def _norm_label(value) -> str:
    return " ".join(str(value or "").strip().lower().replace("_", " ").replace("-", " ").split())


def _listed_retirement_ids() -> set[str]:
    if not RETIREMENTS.is_file():
        return set()
    found: set[str] = set()
    for number, line in enumerate(RETIREMENTS.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise PaceRefusal(
                "refusing to train: data/pace_outcome/retirements.jsonl "
                f"line {number} is not JSON. No checkpoint was written."
            ) from exc
        if isinstance(row, dict) and _retirement_markers(row) and row.get("match_id"):
            found.add(str(row["match_id"]))
    return found


def _matches_listed_retirement(match_id) -> bool:
    if not match_id:
        return False
    text = str(match_id)
    for listed in _listed_retirement_ids():
        if text == listed:
            return True
        short, full = (text, listed) if len(text) < len(listed) else (listed, text)
        if len(short) >= 8 and full.startswith(short):
            return True
    return False


def _was_racing(row: dict) -> bool:
    return row.get("racing") is True and not _is_damage_retirement(row)


def _ballast(row: dict) -> str:
    if "ballast_kg" in row:
        return _canon(row["ballast_kg"])
    cold = row.get("cold_table") or {}
    if isinstance(cold, dict) and "ballast_kg" in cold:
        return _canon(cold["ballast_kg"])
    return ""


def _isolated_cap(left: dict, right: dict, left_setup: dict, right_setup: dict) -> bool:
    """A cap pair may differ only in pace source and speed cap."""
    for name in CONTROLS + ("compound", "fuel_load_kg"):
        if left_setup[name] != right_setup[name]:
            return False
    return _ballast(left) == _ballast(right)


def _compare(left: dict, right: dict, track: str, wet: str) -> dict | None:
    if not _was_racing(left) or not _was_racing(right):
        return None
    left_setup, right_setup = _flat(left), _flat(right)
    pace_differs = left_setup["pace_source"] != right_setup["pace_source"]
    cap_differs = left_setup["speed_cap"] != right_setup["speed_cap"]
    if not pace_differs and not cap_differs:
        return None
    if not _isolated_cap(left, right, left_setup, right_setup):
        return None
    if _pace_key(left) == _pace_key(right):
        return None
    winner = left if _pace_key(left) > _pace_key(right) else right
    winning = left_setup if winner is left else right_setup
    return {
        "track": track,
        "wet": wet,
        "match_ids": tuple(sorted((left["match_id"], right["match_id"]))),
        "setups": {_setup_key(left_setup): left_setup, _setup_key(right_setup): right_setup},
        "label_key": _setup_key(winning),
        "label_origin": _label_origin(left, right),
        "version_ids": {left["match_id"]: left["version_id"], right["match_id"]: right["version_id"]},
        "measured_pace": {left["match_id"]: left["pace"], right["match_id"]: right["pace"]},
        "opponent": {left["match_id"]: left.get("opponent"), right["match_id"]: right.get("opponent")},
        "differs_on_cap": True,
    }


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
            "description": describe_pair(
                pair["setups"][key], pair["setups"][other], template["glue"], template["marker"], PACE_FIELDS
            ),
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
        "label_origin": pair["label_origin"],
        "label_rule": LABEL_RULE,
        "differs_on_cap": True,
        "version_ids": pair["version_ids"],
        "measured_pace": pair["measured_pace"],
        "opponent": pair["opponent"],
        "setups": pair["setups"],
    }


def _flat(row: dict) -> dict:
    cold = row["cold_table"]
    values = {name: _canon_control(name, row["controls"][name]) for name in CONTROLS}
    values["pace_source"] = _canon_source(row["pace_source"])
    values["speed_cap"] = _canon_cap(row["speed_cap"])
    values["compound"] = _canon(cold["compound"])
    values["fuel_load_kg"] = _canon(cold["fuel_load_kg"])
    values["wetness"] = _canon(cold["wetness"])
    values["tyre_temp_c"] = _canon(cold.get("tyre_temp_c"))
    return values


def _canon_control(name: str, value) -> str:
    if name == "stops" and isinstance(value, list):
        return "-".join(
            f"{stop.get('lap')}-{stop.get('compound')}-{stop.get('request_ahead_m')}"
            for stop in value
        )
    return _canon(value)


def _canon_source(value) -> str:
    return str(value).strip()


def _canon_cap(value) -> str:
    if isinstance(value, bool) or value is None:
        return "missing"
    if isinstance(value, (int, float)):
        return _canon(value)
    return str(value).strip()


def _pace_key(row: dict) -> tuple[float, float, float]:
    pace = row["pace"]
    return (float(pace["race_mean_speed_mps"]), float(pace["peak_speed_mps"]), float(pace["peak_lateral_g"]))


def _label_origin(left: dict, right: dict) -> str:
    for name in PACE_NUMBERS:
        if float(left["pace"][name]) != float(right["pace"][name]):
            return name
    return PACE_NUMBERS[0]


def _setup_key(setup: dict) -> str:
    payload = json.dumps({field: setup.get(field, "missing") for field in PACE_FIELDS}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_cap(value) -> bool:
    return isinstance(value, str) or _is_number(value)
