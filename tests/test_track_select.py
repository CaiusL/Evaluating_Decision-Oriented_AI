"""The Highland qualifying release must select Highland, not the idle brake."""
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "submissions" / "racing" / "sj6-track.zip"


def _agent():
    folder = Path(tempfile.mkdtemp(prefix="track-select-"))
    sys.path.insert(0, str(folder))
    with zipfile.ZipFile(ARCHIVE) as bundle:
        bundle.extractall(folder)
    sys.modules.pop("strategy", None)
    sys.modules.pop("formula_driver", None)
    import importlib.util
    spec = importlib.util.spec_from_file_location("sj6_agent", folder / "agent.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.init({"weights_dir": str(folder)})
    module._cleanup = folder
    return module


def test_highland_qualifying_release_selects_highland():
    agent = _agent()
    try:
        # d124dc58 seat 9 at 1498.30 s, the first braking frame of qualifying.
        assert agent.select_track({"self": {"x": -192.571, "y": 768.601}}) == "highland"
    finally:
        shutil.rmtree(agent._cleanup, ignore_errors=True)


def test_parkland_start_selects_parkland():
    agent = _agent()
    try:
        # 5f2ea146 seat 8 qualifying release, t 936.37 s.
        assert agent.select_track({"self": {"x": -317.367, "y": -1230.152}}) == "parkland"
        # Same job, lights-out at t 1161.28 s, speed 0.
        assert agent.select_track({"self": {"x": -496.485, "y": -971.417}}) == "parkland"
    finally:
        shutil.rmtree(agent._cleanup, ignore_errors=True)


def test_crossover_start_selects_crossover():
    agent = _agent()
    try:
        # 5a14af31 seat 5 qualifying release, t 841.00 s, waiting.
        assert agent.select_track({"self": {"x": 292.3, "y": 183.6}}) == "crossover"
        # Same job, lights-out at t 1778.72 s, speed 0.
        assert agent.select_track({"self": {"x": 583.0, "y": -45.1}}) == "crossover"
        # Mid-lap points where Crossover and Parkland are a few meters apart stay closed.
        assert agent.select_track({"self": {"x": -100.4, "y": -135.3}}) is None
        assert agent.select_track({"self": {"x": 70.9, "y": 99.4}}) is None
    finally:
        shutil.rmtree(agent._cleanup, ignore_errors=True)


def test_far_and_missing_coordinates_stay_closed():
    agent = _agent()
    try:
        assert agent.select_track({"self": {"x": 1.0e6, "y": 1.0e6}}) is None
        assert agent.select_track({"self": {}}) is None
    finally:
        shutil.rmtree(agent._cleanup, ignore_errors=True)
