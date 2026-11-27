import zipfile
from pathlib import Path

from self_jev.bundle import read_strategy
from self_jev.check import check_bundle
from self_jev.element_data import DRAFT_NOTE, assert_holdout, judge_tick, write_dataset
from self_jev.elements import LIVE, audit, draft_game

ROOT = Path(__file__).resolve().parents[1]
BASES = LIVE


def _evidence():
    return {
        "supported": {
            "burst": {"4": 10, "9": 90, "abstain": 0},
            "settle": {"0.5": 2, "0.6": 0, "abstain": 40},
            "peek_trigger": {"6": 4, "9": 3, "abstain": 40},
        },
        "notes": DRAFT_NOTE,
    }


def _predict(situation, question, candidates):
    if "Next burst" in situation:
        key = "4"
    else:
        key = "abstain"
    return _score(candidates, key, 0.9)


def _score(candidates, key, top):
    probabilities = {item["id"]: (1 - top) / (len(candidates) - 1) for item in candidates}
    chosen = next(item for item in candidates if item["key"] == key)
    probabilities[chosen["id"]] = top
    return {"choice_id": chosen["id"], "probabilities": probabilities, "scores": {item["id"]: 0.0 for item in candidates}}


def test_tick_labels_follow_readiness_not_the_old_rule_table():
    fire = {"readiness": "fire", "reason": "ready", "speed": 0.0, "aim": 0.0, "distance": 7.5}
    moving = {"readiness": "wait", "reason": "moving", "speed": 0.55, "aim": 0.0, "distance": 10.0}
    band = {"readiness": "adjust_aim", "reason": "misaligned", "speed": 0.0, "aim": 0.13, "distance": 7.5}
    assert judge_tick("burst", fire)[0] == "4"
    assert judge_tick("burst", moving)[0] == "9"
    assert judge_tick("peek_trigger", fire)[0] == "6"
    assert judge_tick("peek_trigger", band)[0] == "9"
    assert judge_tick("settle", moving)[0] == "0.5"
    assert judge_tick("settle", fire)[0] == "abstain"
    assert judge_tick("peek_trigger", moving)[0] == "abstain"


def test_holdout_keeps_train_phrases_and_situations_apart():
    write_dataset()
    from self_jev.element_data import load_examples
    examples = load_examples()
    assert_holdout(examples, extra_texts=list(DRAFT_NOTE.values()))
    ids = {}
    for example in examples:
        ids.setdefault(example["situation_id"], set()).add(example["split"])
    assert all(len(splits) == 1 for splits in ids.values())


def test_audit_finds_the_real_elements():
    for game, path in BASES.items():
        strategy = read_strategy(path)
        agent = zipfile.ZipFile(path).read("agent.py").decode()
        found = {item["id"] for item in audit(game, strategy, agent)}
        assert len(found) >= 8


def test_model_choice_is_applied_and_abstain_falls_back(tmp_path):
    dest = tmp_path / "fps.zip"
    critique = draft_game("fps", BASES["fps"], dest, _predict, _evidence())
    by_id = {item["id"]: item for item in critique["elements"]}
    assert by_id["burst"]["source"] == "model"
    assert by_id["burst"]["old"] == 9 and by_id["burst"]["new"] == 4
    assert by_id["site"]["source"] == "fallback" and by_id["site"]["changed"] is False
    assert by_id["buy"]["source"] == "fallback" and by_id["buy"]["new"] == "rifle"
    agent = zipfile.ZipFile(dest).read("agent.py").decode()
    assert 'MEM["burst_until"] = tick + 4' in agent
    assert "execute_live_seconds" not in agent
    assert check_bundle(dest, "fps") == "act ok"
    for item in critique["elements"]:
        assert item["source"] in {"model", "fallback"}
        assert "probability" in item


def test_a_thin_margin_is_a_fallback(tmp_path):
    def predict(situation, question, candidates):
        if "Next burst" not in situation:
            return _score(candidates, "abstain", 0.9)
        probabilities = {item["id"]: 0.09 for item in candidates}
        first = next(item for item in candidates if item["key"] == "4")
        second = next(item for item in candidates if item["key"] == "9")
        probabilities[first["id"]] = 0.42
        probabilities[second["id"]] = 0.40
        return {"choice_id": first["id"], "probabilities": probabilities, "scores": {item["id"]: 0.0 for item in candidates}}

    critique = draft_game("fps", BASES["fps"], tmp_path / "fps.zip", predict, _evidence())
    burst = next(item for item in critique["elements"] if item["id"] == "burst")
    assert burst["source"] == "fallback" and burst["changed"] is False
