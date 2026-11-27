"""sj8-energy turns the lap-0 budget on and leaves the a2585d41 setup."""
import zipfile


ROOT_ZIP = {
    "sj4": "submissions/racing/sj4-pace.zip",
    "sj8": "submissions/racing/sj8-energy.zip",
}


def _strategy(name):
    source = zipfile.ZipFile(ROOT_ZIP[name]).read("strategy.py").decode()
    namespace = {}
    exec(source, namespace)
    return namespace["STRATEGY"]


def _driver(name):
    return zipfile.ZipFile(ROOT_ZIP[name]).read("formula_driver.py").decode()


def test_only_the_lap_zero_budget_and_the_name_differ_from_a2585d41():
    old, new = _strategy("sj4"), _strategy("sj8")
    assert new["name"] == "sj8-energy"
    assert "lap-0 energy budget" in new["thesis"]
    for key in set(old) | set(new):
        if key in ("name", "thesis"):
            continue
        assert new[key] == old[key]
    assert zipfile.ZipFile(ROOT_ZIP["sj4"]).read("design.json") == zipfile.ZipFile(ROOT_ZIP["sj8"]).read("design.json")
    old_driver, new_driver = _driver("sj4"), _driver("sj8")
    assert "gap_eases_stop" not in new_driver
    assert old_driver.count("c['lap'] >= 1") == new_driver.count("c['lap'] >= 1") + 1
    assert "energy_windows and self.o['phase'] == 'race' and c['lap'] >= 1" in old_driver
    assert "energy_windows and self.o['phase'] == 'race' and c['lap'] >= 1" not in new_driver
