"""The outcome split keeps a match, and its phrase, on one side of the holdout."""
import json

from self_jev.outcome_data import IMPORT, _kept_pairs, _wet, build_examples
UNKNOWN_MARGIN = (
    "4761d4f5-3009-4ad8-a8e2-5458864560fb",
    "7bf8515d-24b1-46b3-9ff5-c1e2e51b99ad",
    "828f3f2d-c1c3-428f-8464-afed1c19a14b",
)


def test_outcome_split_does_not_share_matches_or_phrases():
    examples, stats = build_examples()
    assert stats["decisive_pairs"] >= 20
    assert stats["split_counts"]["test"] >= 20
    assert stats["split_counts"]["train"] >= 4
    assert stats["dropped"]["contradictory_direction"] >= 1
    train_ids = {match_id for row in examples if row["split"] == "train" for match_id in row["match_ids"]}
    test_ids = {match_id for row in examples if row["split"] == "test" for match_id in row["match_ids"]}
    assert train_ids.isdisjoint(test_ids)
    origins = {row["label_origin"] for row in examples}
    assert origins <= {"observed_result.rank", "recording_facts.finish_time_s"}
    assert all(row["label_key"] for row in examples)
    # build_examples raises if a match id or a 4-word window crosses splits.
    assert examples


def test_unknown_wet_margin_is_not_a_speed_pair():
    rows = [
        json.loads(line)
        for line in (IMPORT / "matches.jsonl").read_text().splitlines()
        if line.strip()
    ]
    odd = [row for row in rows if row["match_id"] in UNKNOWN_MARGIN]
    assert len(odd) == 3
    assert {row["played_strategy"]["race_margin"] for row in odd} == {0.995}
    assert {row["track_id"] for row in odd} == {"parkland"}
    assert {_wet(row) for row in odd} == {"unknown"}
    kept, _dropped = _kept_pairs(rows)
    assert all(match_id not in pair["match_ids"] for pair in kept for match_id in UNKNOWN_MARGIN)
    assert all(
        left["race_margin"] == right["race_margin"] == "1"
        for pair in kept
        for left, right in [tuple(pair["setups"].values())]
    )
