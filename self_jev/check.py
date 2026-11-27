"""Smoke-check a rebuilt bundle without uploading it."""
from __future__ import annotations

import importlib.util
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path


def check_bundle(archive: Path, game: str) -> str:
    folder = Path(tempfile.mkdtemp(prefix=f"self-jev-{game}-"))
    try:
        with zipfile.ZipFile(archive) as bundle:
            bundle.extractall(folder)
        if game == "fps":
            _fps(folder)
        elif game == "racing":
            _racing(folder)
        else:
            raise ValueError(game)
    finally:
        shutil.rmtree(folder, ignore_errors=True)
    return "act ok"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _fps(folder: Path) -> None:
    sys.path.insert(0, str(folder))
    try:
        sys.modules.pop("strategy", None)
        agent = _load(folder / "agent.py", "self_jev_fps_agent")
        agent.init({"contract_id": "rsi-fps-5.5", "seat": 0, "role": 0, "seed": 1, "weights_dir": str(folder)})
        action = agent.act({
            "phase": "live",
            "side": "attack",
            "round_index": 0,
            "round_tick": 2000,
            "match_tick": 2000,
            "buy_deadline_tick": 1860,
            "me": {
                "alive": True, "x": 0, "y": 0, "z": 0, "yaw": 0, "pitch": 0, "vx": 0, "vy": 0,
                "cash": 0, "armor": 650, "active_slot": 1, "ammo": 12, "carrying_bomb": False,
                "weapons": [None, {"id": "pistol", "ammo": 12}, {"id": "knife"}],
                "throwable_kinds": [],
            },
            "visible": [{"x": 8, "y": 0, "z": 0, "seat": 9, "parts": ["head"], "alive": True}],
            "teammates": [],
            "bomb": {"state": "dropped"},
        })
    finally:
        sys.path.pop(0)
        sys.modules.pop("strategy", None)
    if not isinstance(action, dict) or "yaw_rate" not in action:
        raise RuntimeError(f"fps act returned {action!r}")


def _racing(folder: Path) -> None:
    sys.path.insert(0, str(folder))
    try:
        sys.modules.pop("strategy", None)
        sys.modules.pop("formula_driver", None)
        agent = _load(folder / "agent.py", "self_jev_racing_agent")
        agent.init({"contract_id": "rsi-racing-4.4", "seat": 0, "role": 0, "seed": 1, "weights_dir": str(folder)})
        action = agent.act({"phase": "qualifying"})
    finally:
        sys.path.pop(0)
        sys.modules.pop("strategy", None)
        sys.modules.pop("formula_driver", None)
    if not isinstance(action, dict) or "throttle" not in action:
        raise RuntimeError(f"racing act returned {action!r}")
