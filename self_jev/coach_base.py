"""Copy the coach's current slot-A bundles into this folder. Read-only on the coach."""
from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path

from self_jev.bundle import design_from_strategy, render_strategy

ROOT = Path(__file__).resolve().parents[1]
COACH = Path("/Users/caius/Projects/Comp")
TEMPLATE = COACH / "assets" / "racing" / "template"
TRACKS = COACH / "assets" / "racing" / "tracks"
DEST = ROOT / "bases" / "coach"


def _sha(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def _zip(files: list[tuple[str, bytes]]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for name, payload in files:
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            bundle.writestr(info, payload)
    return buffer.getvalue()


def _slot_a(game: str) -> dict:
    path = COACH / "state" / f"{game}.json"
    record = json.loads(path.read_text())["A"]
    strategy = record.get("strategy")
    code = record.get("code")
    if not isinstance(strategy, dict) or not isinstance(code, str) or not code.strip():
        raise ValueError(f"coach {game} slot A has no strategy and code")
    return {
        "game": game,
        "slot": "A",
        "source": str(path),
        "name": strategy.get("name"),
        "submission_id": record.get("submission_id"),
        "agent_id": record.get("agent_id"),
        "active_from": record.get("active_from"),
        "strategy": strategy,
        "code": code,
    }


def _racing_files(strategy: dict, code: str) -> list[tuple[str, bytes]]:
    baseline = (TEMPLATE / "agent.py").read_text()
    agent, driver = code, baseline
    if code.lstrip().startswith("{"):
        row = json.loads(code)
        if row.get("format") != "agx-racing-driver-v1":
            raise ValueError("coach racing slot A is an unknown source snapshot")
        agent, driver = row["agent"], row["driver"]
    if "def act" not in agent or "formula_driver" not in agent:
        raise ValueError("coach racing slot A dispatcher does not call formula_driver")
    design = design_from_strategy(json.loads((TEMPLATE / "design.json").read_text()), strategy)
    files = [
        ("agent.py", agent.encode()),
        ("strategy.py", render_strategy(strategy).encode()),
        ("formula_driver.py", driver.encode()),
        ("plan.npz", (TEMPLATE / "plan.npz").read_bytes()),
        ("mechanics.npz", (TEMPLATE / "mechanics.npz").read_bytes()),
        ("design.json", json.dumps(design, indent=2, sort_keys=True).encode()),
        ("weights/bindings.npz", (TRACKS / "bindings.npz").read_bytes()),
    ]
    for track in ("parkland", "highland", "crossover"):
        folder = TRACKS / track
        files.append((f"weights/{track}/plan.npz", (folder / "plan.npz").read_bytes()))
        files.append((f"weights/{track}/mechanics.npz", (folder / "mechanics.npz").read_bytes()))
    return files


def export_game(game: str, dest: Path | None = None) -> dict:
    """Write one coach slot-A zip under bases/coach. Does not write inside the coach."""
    if game not in ("fps", "racing"):
        raise ValueError(game)
    slot = _slot_a(game)
    if game == "fps":
        if "def act" not in slot["code"]:
            raise ValueError("coach fps slot A has no act()")
        files = [
            ("agent.py", slot["code"].encode()),
            ("strategy.py", render_strategy(slot["strategy"]).encode()),
        ]
    else:
        files = _racing_files(slot["strategy"], slot["code"])
    blob = _zip(files)
    target = dest or (DEST / f"{game}-a.zip")
    resolved = target.resolve()
    coach_root = COACH.resolve()
    if resolved == coach_root or coach_root in resolved.parents:
        raise ValueError("refusing to write a coach base export inside the coach project")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(blob)
    provenance = {
        "game": game,
        "slot": "A",
        "coach_state": slot["source"],
        "coach_submission_id": slot["submission_id"],
        "coach_agent_id": slot["agent_id"],
        "active_from": slot["active_from"],
        "name": slot["name"],
        "archive": str(target),
        "sha256": _sha(blob),
        "code_sha256": _sha(slot["code"].encode()),
    }
    target.with_suffix(".json").write_text(json.dumps(provenance, indent=2) + "\n")
    return provenance


def export_all(dest_dir: Path | None = None) -> dict[str, dict]:
    folder = dest_dir or DEST
    return {game: export_game(game, folder / f"{game}-a.zip") for game in ("fps", "racing")}
