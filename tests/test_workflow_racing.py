from self_jev.workflow import _cycle_game


def test_racing_cycle_does_not_publish_a_lever_nudge():
    pending = {"archive": "do-not-upload", "submission_id": None}
    state = {
        "games": {
            "racing": {
                "incumbent": {
                    "name": "sj6-track",
                    "version_id": "dadb68c1-f21e-4935-9bd9-d8c66700be7a",
                },
                "pending": pending,
                "latest_own_matches": [{"match_id": "new-match", "worse": True}],
                "seen_match_ids": [],
                "lever_cursor": 0,
            }
        }
    }
    text = _cycle_game(state, "racing", True)
    assert "not changing a lever" in text
    assert "dadb68c1-f21e-4935-9bd9-d8c66700be7a" in text
    assert state["games"]["racing"]["pending"] is pending
    assert state["games"]["racing"]["lever_cursor"] == 0
