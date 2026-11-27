from self_jev.levers import changed_keys, lever_keys
from self_jev.rewrite import candidates_for, choose_rewrite

FPS = {
    "name": "rifle-force-A",
    "thesis": "Coach base.",
    "default_site": "A",
    "expect_site": "A",
    "buy_style": "force_rifle",
    "utility": "flashbang",
    "post_plant": "hunt",
    "ct_setup": "mid",
    "execute_live_seconds": 12,
}
RACING = {
    "name": "wet-preview-widen-01",
    "thesis": "Coach base.",
    "qual_margin": 1.0,
    "race_margin": 1.0,
    "preview": 0.35,
    "latency": 0.05,
    "ld_min": 7,
    "ld_time": 0.35,
    "yaw_gain": 0.8,
    "kv": 1.9,
    "brake_gain": 1.04,
    "stops": [{"lap": 1, "compound": "medium", "request_ahead_m": 180}],
    "front_wing": 20,
    "rear_wing": 22.0,
    "fuel_load_kg": 11,
    "radiator_opening": 0.7,
    "ers_harvest_level": 5,
    "start_compound": "soft",
    "pressure_front_psi": 29,
    "pressure_rear_psi": 28,
}


def test_candidates_are_single_lever_changes():
    for game, book in (("fps", FPS), ("racing", RACING)):
        options = candidates_for(game, book)
        ids = [item["id"] for item in options]
        assert len(ids) == len(set(ids))
        assert len(options) >= 5
        for item in options:
            assert changed_keys(book, item["strategy"]) == set(lever_keys(game, item["lever"]))
            assert item["description"].startswith("Change only ")


def test_choose_rewrite_uses_the_judge_choice():
    options = candidates_for("fps", FPS)

    def predict_fn(situation, question, candidates):
        assert "rifle-force-A" in situation
        assert candidates[0]["id"] == options[0]["id"]
        return {
            "choice_id": options[-1]["id"],
            "probabilities": {item["id"]: 0.1 for item in candidates},
            "scores": {item["id"]: 0.0 for item in candidates},
        }

    chosen = choose_rewrite("fps", FPS, predict_fn)
    assert chosen["choice_id"] == options[-1]["id"]
    assert chosen["strategy"]["name"].startswith("sj1-")
    assert "Self-JEV chose" in chosen["strategy"]["thesis"]
