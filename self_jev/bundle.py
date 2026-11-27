"""Read and rebuild a submission zip. Agent code stays byte-for-byte."""
from __future__ import annotations

import ast
import hashlib
import io
import json
import pprint
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

REQUIRED = {
    "fps": ("agent.py", "strategy.py"),
    "racing": (
        "agent.py", "strategy.py", "formula_driver.py", "plan.npz",
        "mechanics.npz", "design.json",
    ),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_strategy(archive: Path) -> dict:
    with zipfile.ZipFile(archive) as bundle:
        text = bundle.read("strategy.py").decode()
    prefix = "STRATEGY = "
    if not text.startswith(prefix):
        raise ValueError(f"{archive} strategy.py does not start with STRATEGY")
    strategy = ast.literal_eval(text[len(prefix):])
    if not isinstance(strategy, dict):
        raise ValueError("STRATEGY is not an object")
    return strategy


def render_strategy(strategy: dict) -> str:
    body = pprint.pformat(strategy, width=100, sort_dicts=False)
    return f"STRATEGY = {body}\n"


def design_from_strategy(document: dict, strategy: dict) -> dict:
    updated = json.loads(json.dumps(document))
    design = updated["design"]
    design["aero"]["front_wing"] = strategy["front_wing"]
    design["aero"]["rear_wing"] = strategy["rear_wing"]
    design["power_unit"]["fuel_load_kg"] = strategy["fuel_load_kg"]
    design["power_unit"]["ers_harvest_level"] = strategy["ers_harvest_level"]
    if "ballast_kg" in strategy:
        design["mass"]["ballast_kg"] = strategy["ballast_kg"]
    design["cooling"]["radiator_opening"] = strategy["radiator_opening"]
    design["tyres"]["start_compound"] = strategy["start_compound"]
    design["tyres"]["pressure_front_psi"] = strategy["pressure_front_psi"]
    design["tyres"]["pressure_rear_psi"] = strategy["pressure_rear_psi"]
    return updated


def rebuild(source: Path, game: str, strategy: dict, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    rendered = render_strategy(strategy).encode()
    with zipfile.ZipFile(source) as incoming, io.BytesIO() as buffer:
        with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as outgoing:
            names = incoming.namelist()
            missing = [name for name in REQUIRED[game] if name not in names]
            if missing:
                raise ValueError(f"{game} bundle missing {missing}")
            design = None
            if game == "racing":
                design = design_from_strategy(json.loads(incoming.read("design.json")), strategy)
            for name in names:
                if name == "strategy.py":
                    payload = rendered
                elif name == "design.json":
                    payload = json.dumps(design, indent=2, sort_keys=True).encode()
                else:
                    payload = incoming.read(name)
                info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o100644 << 16
                outgoing.writestr(info, payload)
        dest.write_bytes(buffer.getvalue())
    return dest
