import zipfile
from pathlib import Path

from self_jev.bundle import read_strategy, rebuild
from self_jev.check import check_bundle
from self_jev.levers import changed_keys, lever_keys, levers, revise

ROOT = Path(__file__).resolve().parents[1]
BASES = {
    "fps": ROOT / "bases" / "fps" / "fps-v17.zip",
    "racing": ROOT / "bases" / "racing" / "racing-v25.zip",
}


def test_each_lever_changes_only_itself(tmp_path):
    for game, archive in BASES.items():
        current = read_strategy(archive)
        for index, lever in enumerate(levers(game)):
            proposed = revise(game, current, lever, worse=index % 2 == 0, generation=2 + index, match_id="match-1")
            assert changed_keys(current, proposed) == set(lever_keys(game, lever))
            dest = tmp_path / game / f"{lever}.zip"
            rebuild(archive, game, proposed, dest)
            with zipfile.ZipFile(archive) as old, zipfile.ZipFile(dest) as new:
                assert old.read("agent.py") == new.read("agent.py")
            assert read_strategy(dest)["name"] == proposed["name"]


def test_base_bundles_act():
    for game, archive in BASES.items():
        assert check_bundle(archive, game) == "act ok"
