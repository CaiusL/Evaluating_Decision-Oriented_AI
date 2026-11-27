"""sj7-gap changes only the stopping limit on the a2585d41 setup."""
import zipfile

from self_jev.pace_outcome import _pairs
from tests.test_pace_outcome import _row


ROOT_ZIP = {
    "sj4": "submissions/racing/sj4-pace.zip",
    "sj7": "submissions/racing/sj7-gap.zip",
}


def _strategy(name):
    source = zipfile.ZipFile(ROOT_ZIP[name]).read("strategy.py").decode()
    namespace = {}
    exec(source, namespace)
    return namespace["STRATEGY"]


def _driver_source(name):
    return zipfile.ZipFile(ROOT_ZIP[name]).read("formula_driver.py").decode()


def test_only_the_stopping_rule_and_the_name_differ_from_a2585d41():
    old, new = _strategy("sj4"), _strategy("sj7")
    assert new["name"] == "sj7-gap"
    assert "stable or growing" in new["thesis"]
    for key in set(old) | set(new):
        if key in ("name", "thesis"):
            continue
        assert new[key] == old[key]
    assert zipfile.ZipFile(ROOT_ZIP["sj4"]).read("design.json") == zipfile.ZipFile(ROOT_ZIP["sj7"]).read("design.json")
    old_lines = _driver_source("sj4").splitlines()
    new_lines = _driver_source("sj7").splitlines()
    added = [line for line in new_lines if line not in old_lines]
    assert any("def gap_eases_stop" in line for line in added)
    assert any("gap_eases_stop(r['rate'])" in line for line in added)
    assert not any(line.startswith("-") for line in added)


def test_a_stable_or_growing_gap_eases_and_a_closing_gap_does_not():
    namespace = {}
    exec(_driver_source("sj7"), namespace)
    ease = namespace["gap_eases_stop"]
    assert ease(0) is True
    assert ease(2.0) is True
    assert ease(-0.01) is False


def test_parkland_speed_pair_stays_open_while_preview_yaw_fuel_or_ballast_differ():
    grip = _row("a2585d41-ff5d-4175-8a56-d95413d3a662", "parkland", "damp", "grip", 90, 52.10, 92.84, 3.23)
    cold = _row("6214cc20-cdbc-4cef-85f8-6587c4a6e277", "parkland", "damp", "table", "cold", 53.64, 95.63, 3.087)
    grip["cold_table"]["fuel_load_kg"] = 9
    grip["ballast_kg"] = 20
    cold["controls"]["preview"] = 0.33
    cold["controls"]["yaw_gain"] = 0.8
    cold["ballast_kg"] = 40
    assert _pairs([grip, cold]) == []
