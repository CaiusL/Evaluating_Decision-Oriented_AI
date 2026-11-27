from pathlib import Path

from self_jev.hashing import sha256_file
from self_jev.kev_harness import HELD, STATE, run, select_car, upload_latest


def _choice(name, probabilities):
    return {"type": "choice", "choice": name, "probabilities": probabilities}


def _low():
    return {
        "stopping": _choice("unresolved", {"stay_eased": 0.30, "restore_stop": 0.30, "unresolved": 0.40}),
        "energy": _choice("unresolved", {"leave_off": 0.30, "turn_on": 0.20, "unresolved": 0.50}),
        "pit": _choice("unresolved", {"keep_stop": 0.30, "skip_stop": 0.30, "unresolved": 0.40}),
    }


def _cars(tmp: Path):
    archives = {}
    for name in ("sj7-gap", "sj4-pace", "sj8-energy"):
        path = tmp / f"{name}.zip"
        path.write_bytes(name.encode())
        archives[name] = path
    return {
        "sj7-gap": {"archive": archives["sj7-gap"], "choices": {"stopping": "stay_eased", "energy": "leave_off"}},
        "sj4-pace": {"archive": archives["sj4-pace"], "choices": {"stopping": "restore_stop", "energy": "leave_off"}},
        "sj8-energy": {"archive": archives["sj8-energy"], "choices": {"stopping": "restore_stop", "energy": "turn_on"}},
    }


def _incumbent(cars):
    return {"name": "sj7-gap", "sha256": sha256_file(cars["sj7-gap"]["archive"])}


def _run(tmp, answers, cars, incumbent=None, upload=True):
    calls = []

    def ask(state, questions):
        return {"answers": answers, "backend": "test"}

    def submit(game, archive):
        calls.append((game, Path(archive)))
        return {"sha256": sha256_file(Path(archive)), "submission_id": "sub", "version_id": "ver"}

    report = run(
        upload=upload,
        ask=ask,
        submit=submit,
        report_path=tmp / "report.json",
        cars=cars,
        incumbent=incumbent or _incumbent(cars),
        remember=False,
        judges_dir=tmp / "judges",
    )
    return report, calls


def test_state_includes_the_latest_crossover_retirement():
    assert "ede97a2e" in STATE
    assert "crossover" in STATE
    assert "retired from damage on lap 3" in STATE
    assert "not a still-racing lap time" in STATE


def test_nothing_at_the_bar_uploads_nothing(tmp_path):
    cars = _cars(tmp_path)
    assert select_car(_low(), cars, HELD, "sj7-gap") is None
    report, calls = _run(tmp_path, _low(), cars)
    assert report["new"] is True
    assert report["upload_step"] == "ran"
    assert report["selected"] is None
    assert report["uploaded"] is False
    assert calls == []


def test_restored_stop_and_budget_on_uploads_sj8(tmp_path):
    cars = _cars(tmp_path)
    answers = _low()
    answers["stopping"] = _choice("restore_stop", {"stay_eased": 0.10, "restore_stop": 0.80, "unresolved": 0.10})
    answers["energy"] = _choice("turn_on", {"leave_off": 0.10, "turn_on": 0.85, "unresolved": 0.05})
    report, calls = _run(tmp_path, answers, cars)
    assert report["selected"]["name"] == "sj8-energy"
    assert report["uploaded"] is True
    assert report["backend"] == "test"
    assert calls == [("racing", cars["sj8-energy"]["archive"])]


def test_incumbent_bytes_are_not_uploaded_again(tmp_path):
    cars = _cars(tmp_path)
    answers = _low()
    answers["stopping"] = _choice("restore_stop", {"stay_eased": 0.10, "restore_stop": 0.80, "unresolved": 0.10})
    answers["energy"] = _choice("turn_on", {"leave_off": 0.10, "turn_on": 0.85, "unresolved": 0.05})
    incumbent = {"name": "sj7-gap", "sha256": sha256_file(cars["sj8-energy"]["archive"])}
    report, calls = _run(tmp_path, answers, cars, incumbent)
    assert report["selected"] is None
    assert report["uploaded"] is False
    assert calls == []


def test_budget_alone_does_not_upload(tmp_path):
    cars = _cars(tmp_path)
    answers = _low()
    answers["energy"] = _choice("turn_on", {"leave_off": 0.10, "turn_on": 0.80, "unresolved": 0.10})
    report, calls = _run(tmp_path, answers, cars)
    assert report["selected"] is None
    assert calls == []


def test_published_choices_do_not_upload(tmp_path):
    cars = _cars(tmp_path)
    answers = _low()
    answers["stopping"] = _choice("stay_eased", {"stay_eased": 0.80, "restore_stop": 0.10, "unresolved": 0.10})
    answers["energy"] = _choice("leave_off", {"leave_off": 0.75, "turn_on": 0.10, "unresolved": 0.15})
    report, calls = _run(tmp_path, answers, cars)
    assert report["selected"] is None
    assert calls == []


def test_a_lever_with_no_car_blocks_upload(tmp_path):
    cars = _cars(tmp_path)
    answers = _low()
    answers["stopping"] = _choice("restore_stop", {"stay_eased": 0.10, "restore_stop": 0.80, "unresolved": 0.10})
    answers["energy"] = _choice("turn_on", {"leave_off": 0.10, "turn_on": 0.85, "unresolved": 0.05})
    answers["pit"] = _choice("skip_stop", {"keep_stop": 0.10, "skip_stop": 0.80, "unresolved": 0.10})
    answers["brake"] = _choice("change", {"keep": 0.10, "change": 0.80, "unresolved": 0.10})
    report, calls = _run(tmp_path, answers, cars)
    assert report["selected"] is None
    assert calls == []


def test_restored_stop_with_budget_off_uploads_sj4(tmp_path):
    cars = _cars(tmp_path)
    answers = _low()
    answers["stopping"] = _choice("restore_stop", {"stay_eased": 0.10, "restore_stop": 0.80, "unresolved": 0.10})
    answers["energy"] = _choice("leave_off", {"leave_off": 0.75, "turn_on": 0.10, "unresolved": 0.15})
    report, calls = _run(tmp_path, answers, cars)
    assert report["selected"]["name"] == "sj4-pace"
    assert calls == [("racing", cars["sj4-pace"]["archive"])]


def test_judge_only_records_the_car_without_uploading(tmp_path):
    cars = _cars(tmp_path)
    answers = _low()
    answers["stopping"] = _choice("restore_stop", {"stay_eased": 0.10, "restore_stop": 0.80, "unresolved": 0.10})
    answers["energy"] = _choice("turn_on", {"leave_off": 0.10, "turn_on": 0.85, "unresolved": 0.05})
    report, calls = _run(tmp_path, answers, cars, upload=False)
    assert report["selected"]["name"] == "sj8-energy"
    assert report["upload_step"] == "held"
    assert report["uploaded"] is False
    assert calls == []


def test_a_new_judge_uploads_once(tmp_path):
    cars = _cars(tmp_path)
    answers = _low()
    answers["stopping"] = _choice("restore_stop", {"stay_eased": 0.10, "restore_stop": 0.80, "unresolved": 0.10})
    answers["energy"] = _choice("turn_on", {"leave_off": 0.10, "turn_on": 0.85, "unresolved": 0.05})
    calls = []
    judges = tmp_path / "judges"

    def ask(state, questions):
        return {"answers": answers}

    def submit(game, archive):
        calls.append(Path(archive))
        return {"sha256": sha256_file(Path(archive)), "submission_id": "sub", "version_id": "ver"}

    kwargs = dict(
        ask=ask,
        submit=submit,
        report_path=tmp_path / "report.json",
        cars=cars,
        incumbent=_incumbent(cars),
        remember=False,
        judges_dir=judges,
    )
    first = run(upload=True, **kwargs)
    second = run(upload=True, **kwargs)
    assert first["new"] is True
    assert first["upload_step"] == "ran"
    assert first["uploaded"] is True
    assert second["new"] is False
    assert calls == [cars["sj8-energy"]["archive"]]


def test_a_held_judge_uploads_on_the_upload_step(tmp_path):
    cars = _cars(tmp_path)
    answers = _low()
    answers["stopping"] = _choice("restore_stop", {"stay_eased": 0.10, "restore_stop": 0.80, "unresolved": 0.10})
    answers["energy"] = _choice("turn_on", {"leave_off": 0.10, "turn_on": 0.85, "unresolved": 0.05})
    calls = []
    judges = tmp_path / "judges"

    def ask(state, questions):
        return {"answers": answers}

    def submit(game, archive):
        calls.append(Path(archive))
        return {"sha256": sha256_file(Path(archive)), "submission_id": "sub", "version_id": "ver"}

    held = run(
        upload=False,
        ask=ask,
        submit=submit,
        report_path=tmp_path / "report.json",
        cars=cars,
        incumbent=_incumbent(cars),
        remember=False,
        judges_dir=judges,
    )
    assert held["upload_step"] == "held"
    released = upload_latest(submit=submit, remember=False, judges_dir=judges, report_path=tmp_path / "report.json")
    again = upload_latest(submit=submit, remember=False, judges_dir=judges, report_path=tmp_path / "report.json")
    assert released["uploaded"] is True
    assert released["upload_step"] == "ran"
    assert again["new"] is False
    assert calls == [cars["sj8-energy"]["archive"]]
