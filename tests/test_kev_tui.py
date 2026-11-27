import json
from pathlib import Path

from self_jev.kev_tui import load_latest, screen, serve


def _report():
    return {
        "id": "abc123",
        "bar": 0.70,
        "incumbent": "sj7-gap",
        "backend": "mlx",
        "dtype": "bfloat16",
        "input_tokens": 12,
        "latency_ms": 4.0,
        "upload_step": "ran",
        "uploaded": False,
        "selected": None,
        "answers": {
            "any_fair_pair": {"type": "noul", "noul": 0.3776},
            "stopping": {
                "type": "choice",
                "choice": "stay_eased",
                "probabilities": {"stay_eased": 0.4045, "restore_stop": 0.2893, "unresolved": 0.3062},
            },
            "energy": {
                "type": "choice",
                "choice": "turn_on",
                "probabilities": {"leave_off": 0.10, "turn_on": 0.80, "unresolved": 0.10},
            },
        },
    }


def test_screen_shows_the_judge_and_the_bar():
    text = screen(_report())
    assert "incumbent sj7-gap" in text
    assert "judge abc123" in text
    assert "p(yes) 0.3776" in text
    assert "stay_eased" in text
    assert "* energy" in text
    assert "sj8-energy" in text
    assert "j  create a judge and hold the upload" in text
    assert "upload step ran. no built car cleared the bar" in text


def test_empty_screen_says_there_is_no_judge():
    text = screen(None)
    assert "no judge yet" in text
    assert "no answers yet" in text


def test_keys_judge_then_upload_then_quit(tmp_path):
    calls = []
    stored = _report()
    pointer = tmp_path / "latest.json"
    judge_path = tmp_path / "abc123.json"
    judge_path.write_text(json.dumps(stored))
    pointer.write_text(json.dumps({"id": "abc123", "path": str(judge_path)}))

    def judge(judges_dir=None):
        calls.append(("j", judges_dir))
        created = dict(stored, new=True, upload_step="held", selected={"name": "sj8-energy", "archive": "sj8", "choices": {}})
        return created

    def upload(judges_dir=None):
        calls.append(("u", judges_dir))
        return dict(stored, uploaded=True, selected={"name": "sj8-energy", "choices": {}}, receipt={"version_id": "ver"})

    seen = []
    result = serve(["j", "u", "q"], judge=judge, upload=upload, write=seen.append, judges_dir=tmp_path)
    assert calls == [("j", tmp_path), ("u", tmp_path)]
    assert any("scoring with Kev" in item for item in seen)
    assert any("incumbent sj7-gap" in item for item in seen)
    assert result["uploaded"] is True
    assert load_latest(tmp_path)["id"] == "abc123"


def test_a_missing_judge_keeps_the_screen_up(tmp_path):
    seen = []

    def upload(judges_dir=None):
        raise RuntimeError("no judge has been created")

    result = serve(["u", "q"], judge=lambda **kwargs: None, upload=upload, write=seen.append, judges_dir=tmp_path)
    assert result is None
    assert any("no judge has been created" in item for item in seen)
