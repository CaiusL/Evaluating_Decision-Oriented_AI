"""The pace gate refuses a flat cap and does not invent a checkpoint."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from self_jev.outcome_data import build_examples, model_text
from self_jev.pace_outcome import (
    EXAMPLES,
    MISSING_DATA,
    PACE_RETIREMENT_RULE,
    RETIREMENTS,
    ROOT,
    PaceRefusal,
    _flat,
    _is_damage_retirement,
    _pairs,
    _setup_key,
    load_rows,
    main,
    measured_tracks,
    outcome_refusal_message,
    refuse_flat_outcome,
    row_variance_message,
    rows_to_examples,
)

PLAN = ROOT / "reports" / "jev_data_plan.md"
OUTCOME_EXAMPLES = ROOT / "data" / "outcome" / "examples.jsonl"
CHECKPOINT = ROOT / "runs" / "outcome-6db695d99db9"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _row(match, track, wet, source, cap, mean, peak, lat, preview=0.18):
    return {
        "version_id": f"version-{match}",
        "match_id": match,
        "track": track,
        "wet": wet,
        "controls": {
            "qual_margin": 1,
            "race_margin": 1,
            "preview": preview,
            "latency": 0.05,
            "ld_min": 7,
            "ld_time": 0.35,
            "yaw_gain": 0.7,
            "kv": 1.9,
            "brake_gain": 1.04,
            "stops": "1-medium-180",
        },
        "pace_source": source,
        "speed_cap": cap,
        "cold_table": {"compound": "soft", "fuel_load_kg": 11, "wetness": 0.05, "tyre_temp_c": None},
        "pace": {"race_mean_speed_mps": mean, "peak_speed_mps": peak, "peak_lateral_g": lat},
        "racing": True,
        "opponent": {"peak_speed_mps": 87.84, "peak_lateral_g": 3.53},
    }


def test_plan_and_missing_command_share_one_message():
    assert MISSING_DATA in PLAN.read_text()
    assert not EXAMPLES.exists()
    with pytest.raises(SystemExit) as caught:
        main()
    assert caught.value.code == 1


def test_outcome_rows_have_no_cap_variance_and_are_left_in_place(capsys):
    before = _sha(OUTCOME_EXAMPLES)
    examples, _stats = build_examples()
    message = outcome_refusal_message(examples)
    assert message is not None
    assert "zero variance on pace_source" in message
    assert "zero variance on the speed cap" in message
    assert "0 pairs differ" in message
    with pytest.raises(SystemExit) as caught:
        refuse_flat_outcome()
    assert caught.value.code == 1
    assert message in capsys.readouterr().err
    assert _sha(OUTCOME_EXAMPLES) == before
    assert CHECKPOINT.is_dir()
    assert not EXAMPLES.exists()


def test_train_commands_refuse_without_writing(capsys):
    before = _sha(OUTCOME_EXAMPLES)
    runs_before = sorted(path.name for path in (ROOT / "runs").iterdir())
    outcome = subprocess.run(
        [sys.executable, "-m", "self_jev", "train-outcome"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    pace = subprocess.run(
        [sys.executable, "-m", "self_jev", "train-pace-outcome"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert outcome.returncode == 1
    assert outcome.stderr.strip() == outcome_refusal_message(build_examples()[0])
    assert outcome.stdout == ""
    assert pace.returncode == 1
    assert pace.stderr.strip() == MISSING_DATA
    assert pace.stdout == ""
    assert _sha(OUTCOME_EXAMPLES) == before
    assert sorted(path.name for path in (ROOT / "runs").iterdir()) == runs_before
    assert not EXAMPLES.exists()
    assert capsys.readouterr().out == ""


def test_confounded_setup_is_not_a_speed_table_pair():
    grip = _row("a2585d41-ff5d-4175-8a56-d95413d3a662", "parkland", "damp", "grip", 90, 52.10, 92.84, 3.23)
    cold = _row("6214cc20-cdbc-4cef-85f8-6587c4a6e277", "parkland", "damp", "table", "cold", 53.64, 95.63, 3.087, preview=0.33)
    grip["cold_table"]["fuel_load_kg"] = 9
    grip["controls"]["yaw_gain"] = 0.7
    grip["ballast_kg"] = 20
    cold["controls"]["yaw_gain"] = 0.8
    cold["ballast_kg"] = 40
    assert _pairs([grip, cold]) == []
    assert "speed-table question remains open" in MISSING_DATA


def test_same_cap_is_not_a_pair_even_when_pace_and_preview_differ():
    rows = [
        _row("m-cold-a", "crossover", "wet", "table", "cold", 30, 70, 1.5, preview=0.18),
        _row("m-cold-b", "crossover", "wet", "table", "cold", 48, 80, 2.4, preview=0.35),
    ]
    assert row_variance_message(rows) is not None
    assert _pairs(rows) == []


def test_equal_pace_does_not_become_a_pair():
    rows = [
        _row("m-grip", "crossover", "wet", "grip", 90, 40, 74, 2.0),
        _row("m-cold", "crossover", "wet", "table", "cold", 40, 74, 2.0),
    ]
    assert row_variance_message(rows) is None
    with pytest.raises(PaceRefusal, match="measured pace differs"):
        rows_to_examples(rows)


def test_one_contrast_cannot_hold_out_a_component():
    rows = [
        _row("m-grip", "crossover", "wet", "grip", 90, 50, 88, 2.4),
        _row("m-cold", "crossover", "wet", "table", "cold", 38, 74, 2.0),
    ]
    with pytest.raises(PaceRefusal, match="hold out a cap pair"):
        rows_to_examples(rows)
    assert not EXAMPLES.exists()
    assert not (ROOT / "state" / "pace_outcome_checkpoint.txt").exists()


def test_held_out_cap_pair_uses_measured_pace_and_hides_opponents():
    bands = [
        ("crossover", "wet", "aa-grip", "aa-cold", 52, 80, 2.2, 40, 70, 1.5, "race_mean_speed_mps", "grip"),
        ("highland", "wet", "bb-grip", "bb-cold", 40, 88, 2.0, 40, 70, 2.0, "peak_speed_mps", "grip"),
        ("parkland", "wet", "cc-grip", "cc-cold", 40, 74, 2.6, 40, 74, 1.4, "peak_lateral_g", "grip"),
        ("crossover", "dry", "dd-grip", "dd-cold", 45, 80, 2.0, 60, 90, 2.5, "race_mean_speed_mps", "table"),
    ]
    rows = []
    expected = {}
    for track, wet, grip_id, cold_id, g_mean, g_peak, g_lat, c_mean, c_peak, c_lat, origin, winner in bands:
        grip = _row(grip_id, track, wet, "grip", 90, g_mean, g_peak, g_lat)
        cold = _row(cold_id, track, wet, "table", "cold", c_mean, c_peak, c_lat)
        rows.extend((grip, cold))
        expected[tuple(sorted((grip_id, cold_id)))] = (origin, grip if winner == "grip" else cold)
    examples, stats = rows_to_examples(rows)
    assert stats["split_counts"]["test"] >= 1
    seen = [row for row in examples if not row["paraphrase"]]
    assert {row["split"] for row in seen} == {"train", "select", "calibrate", "test"}
    for example in seen:
        assert example["differs_on_cap"] is True
        assert len({candidate["id"] for candidate in example["candidates"]}) == 1
        text = model_text(example)
        assert "87.84" not in text
        assert "3.53" not in text
        origin, winner = expected[tuple(example["match_ids"])]
        assert example["label_origin"] == origin
        assert example["label_key"] == _setup_key(_flat(winner))
        assert winner["match_id"] not in text


def test_damage_retirement_is_not_a_slower_cap(tmp_path):
    retired = _row("08640570-c76a-422d-b794-5a7f9e9edd73", "crossover", "dry", "grip", 90, 4.38, 72.17, 2.88)
    retired["racing"] = False
    retired["label"] = "damage_retirement"
    retired["status"] = "retired"
    retired["retirement"] = {"reason": "damage", "contact_seat": 5, "contact_time_s": 1197.45}
    cold = _row("e12c8a81-a972-4964-a0e7-a095048ccc80", "crossover", "dry", "table", "cold", 38.91, 74.59, 2.0)
    assert _pairs([retired, cold]) == []
    path = tmp_path / "examples.jsonl"
    path.write_text(json.dumps(retired) + "\n" + json.dumps(cold) + "\n")
    with pytest.raises(PaceRefusal, match="damage retirement"):
        load_rows(path)
    assert not EXAMPLES.exists()
    assert not (ROOT / "state" / "pace_outcome_checkpoint.txt").exists()
    assert (ROOT / "runs" / "outcome-6db695d99db9").is_dir()


def test_recorded_retirements_cannot_label_pace():
    assert PACE_RETIREMENT_RULE in PLAN.read_text()
    assert "d2b6ac91-acda-4426-8a87-8cfed0f5f126" in PLAN.read_text()
    lines = [line for line in RETIREMENTS.read_text().splitlines() if line.strip()]
    rows = [json.loads(line) for line in lines]
    assert [row["match_id"] for row in rows] == [
        "08640570-c76a-422d-b794-5a7f9e9edd73",
        "898ec364-8d50-42b9-8dae-7fe7ba2f2eba",
        "5a14af31-6979-4939-9c36-76133aab1236",
        "21f76602-d0ef-4deb-94c4-26248c994593",
        "c767b15a-718f-41ae-bceb-a2bceefff0ff",
        "ebe0e4b9-c62f-415c-8000-4aa51f9f662a",
        "049a8010-6985-416c-abf9-d3d076d0b560",
        "f96edeef-2bba-46b7-a0b4-90157102e8ba",
        "37b81ebd-33ee-4b35-9765-00c8e995f001",
        "f9ad2bd4-9d7e-4aab-8f09-dd16f16b9b6e",
        "d106e664-70ff-463c-9406-64a2a7e52e5c",
        "1f6b7466-3ce0-4d10-8242-f8d192ca4bd9",
        "08569f9b-52d9-42f2-89b4-b8fbf9de03e2",
        "a94b748b-eac4-48c1-bd5d-b019d61e0b61",
        "ede97a2e-04e9-4b30-af22-e5c130352b86",
    ]
    assert rows[0]["seat"] == 3 and rows[0]["rank"] == 8
    assert rows[1]["seat"] == 0 and rows[1]["rank"] == 9
    assert rows[2]["seat"] == 5 and rows[2]["rank"] == 9
    assert rows[2]["track_id"] == "crossover" and rows[2]["speed_cap"] == "cold"
    assert rows[3]["seat"] == 2 and rows[3]["rank"] == 8 and rows[3]["track_id"] == "highland"
    assert rows[4]["seat"] == 8 and rows[4]["rank"] == 8 and rows[4]["track_id"] == "parkland"
    assert rows[5]["seat"] == 6 and rows[5]["rank"] == 5 and rows[5]["peak_track_wetness"] == 0.249972
    assert rows[6]["seat"] == 4 and rows[6]["rank"] == 7 and rows[6]["track_id"] == "highland"
    assert rows[6]["status"] == "unclassified"
    assert rows[7]["seat"] == 9 and rows[7]["rank"] == 4 and rows[7]["track_id"] == "highland"
    assert rows[8]["seat"] == 4 and rows[8]["rank"] == 9 and rows[8]["retirement"]["lap"] == 0
    assert rows[9]["seat"] == 0 and rows[9]["rank"] == 7 and rows[9]["retirement"]["lap"] == 1
    assert rows[10]["seat"] == 9 and rows[10]["rank"] == 4 and rows[10]["retirement"]["lap"] == 3
    assert rows[11]["seat"] == 6 and rows[11]["rank"] == 6 and rows[11]["retirement"]["lap"] == 3
    assert rows[12]["seat"] == 6 and rows[12]["rank"] == 6 and rows[12]["retirement"]["lap"] == 2
    assert rows[13]["seat"] == 3 and rows[13]["rank"] == 8 and rows[13]["retirement"]["lap"] == 0
    assert rows[14]["name"] == "sj8-energy"
    assert rows[14]["seat"] == 8 and rows[14]["rank"] == 1 and rows[14]["track_id"] == "crossover"
    assert rows[14]["pace_source"] == "grip" and rows[14]["speed_cap"] == 90
    assert rows[14]["retirement"]["reason"] == "damage" and rows[14]["retirement"]["lap"] == 3
    for row in rows:
        assert row["label"] == "damage retirement"
        assert row["racing"] is False
        assert "pace" not in row
        assert _is_damage_retirement(row)
    cold = _row("e12c8a81-a972-4964-a0e7-a095048ccc80", "crossover", "wet", "table", "cold", 38.91, 74.59, 2.0)
    assert not _is_damage_retirement(cold)
    assert _pairs(rows + [cold]) == []
    spoken = _row("08640570-c76a-422d-b794-5a7f9e9edd73", "crossover", "wet", "grip", 90, 4.38, 72.17, 2.88)
    spoken["label"] = "damage retirement"
    spoken["racing"] = True
    assert _pairs([spoken, cold]) == []
    spoken["label"] = "damage-retired"
    assert _pairs([spoken, cold]) == []
    stripped = {
        "version_id": "12e5e35b-2e00-4eb6-8dee-d9835a7915d1",
        "match_id": "898ec364",
        "track": "crossover",
        "wet": "wet",
        "racing": True,
        "pace_source": "grip",
        "speed_cap": 90,
    }
    assert _is_damage_retirement(stripped)
    assert _pairs([stripped, cold]) == []
    with pytest.raises(PaceRefusal, match="damage retirement"):
        load_rows(RETIREMENTS)
    assert not EXAMPLES.exists()


def test_saved_measurements_are_different_tracks():
    tracks = dict(measured_tracks())
    assert tracks["a2585d41-ff5d-4175-8a56-d95413d3a662"] == "parkland"
    assert tracks["d124dc58-0057-4880-bcc2-14159ebab663"] == "highland"


def test_measured_parkland_and_highland_are_not_a_pair():
    assert "a2585d41" in MISSING_DATA and "d124dc58" in MISSING_DATA
    assert "different tracks" in MISSING_DATA
    parkland = _row("a2585d41-ff5d-4175-8a56-d95413d3a662", "parkland", "dry", "grip", 90, 52.10, 92.84, 3.23)
    highland = _row("d124dc58-0057-4880-bcc2-14159ebab663", "highland", "dry", "table", "cold", 38.38, 79.14, 2.56)
    assert _pairs([parkland, highland]) == []
    assert not EXAMPLES.exists()


def test_incomplete_row_names_the_missing_pace(tmp_path):
    row = _row("m-grip", "crossover", "wet", "grip", 90, 50, 88, 2.4)
    del row["pace"]
    path = tmp_path / "examples.jsonl"
    path.write_text(json.dumps(row) + "\n")
    with pytest.raises(PaceRefusal, match="pace"):
        load_rows(path)
