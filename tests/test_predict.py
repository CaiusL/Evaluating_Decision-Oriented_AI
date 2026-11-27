import pytest

from self_jev.predict import predict


def _score(situation, question, candidates):
    return [len(candidate["description"]) for candidate in candidates]


def test_predict_sums_to_one_and_is_shuffle_invariant():
    candidates = [
        {"id": "a", "description": "short"},
        {"id": "b", "description": "a much longer description"},
    ]
    first = predict("The seat is alive.", "What should this seat do now?", candidates, scorer=_score)
    second = predict("The seat is alive.", "What should this seat do now?", list(reversed(candidates)), scorer=_score)
    assert first["choice_id"] == second["choice_id"] == "b"
    assert abs(sum(first["probabilities"].values()) - 1) < 1e-9
    seen = {}

    def counting(situation, question, given):
        seen["count"] = len(given)
        return [1.0, 0.0]

    result = predict(
        "The seat is alive.",
        "What should this seat do now?",
        [{"id": "a", "description": "Fire the equipped gun."}, {"id": "b", "description": "Wait."}],
        scorer=counting,
    )
    assert seen["count"] == 2
    assert result["choice_id"] == "a"
    with pytest.raises(ValueError):
        predict("", "What should this seat do now?", candidates, scorer=_score)
    with pytest.raises(ValueError):
        predict("The seat is alive.", "What should this seat do now?", [], scorer=_score)
    with pytest.raises(ValueError):
        predict(
            "The seat is alive.",
            "What should this seat do now?",
            [{"id": "a", "description": "one"}, {"id": "a", "description": "two"}],
            scorer=_score,
        )
