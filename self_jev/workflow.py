"""Self-JEV RSI cycle.

Observe one new own match, change one strategy lever, smoke-check the bundle,
then publish it on this account. With no new own match, the cycle waits.
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path

from self_jev.agx import AgxError, request
from self_jev.bundle import read_strategy, rebuild, sha256
from self_jev.check import check_bundle
from self_jev.levers import levers, revise
from self_jev.observe import COMPETITIONS, own_matches

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "state" / "workflow.json"
ACCOUNT = ROOT / "account.json"
BASES = {
    "fps": ROOT / "bases" / "fps" / "fps-v17.zip",
    "racing": ROOT / "bases" / "racing" / "racing-v25.zip",
}


def load() -> dict:
    if STATE.exists():
        return json.loads(STATE.read_text())
    account = json.loads(ACCOUNT.read_text())
    games = {}
    for game, archive in BASES.items():
        published = account["submissions"][game]
        games[game] = {
            "generation": 1,
            "lever_cursor": 0,
            "incumbent": {
                "archive": str(archive),
                "sha256": published["sha256"],
                "submission_id": published["submission_id"],
                "version_id": published["version_id"],
                "name": published["name"],
            },
            "seen_match_ids": [],
            "pending": None,
            "cycles": [],
            "latest_own_matches": [],
        }
    state = {"schema": 1, "games": games}
    save(state)
    return state


def save(state: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    temporary = STATE.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(STATE)


def status_lines(state: dict | None = None) -> list[str]:
    state = state or load()
    lines = ["Self-JEV RSI"]
    for game, row in state["games"].items():
        incumbent = row["incumbent"]
        pending = "pending upload" if row.get("pending") else "idle"
        lines.append(
            f"{game} g{row['generation']} {incumbent['name']} {pending} "
            f"version {incumbent.get('version_id')} seen {len(row.get('seen_match_ids') or [])}"
        )
    return lines


def observe(state: dict | None = None) -> dict:
    state = state or load()
    for game, row in state["games"].items():
        row["latest_own_matches"] = own_matches(game)
    save(state)
    return state


def cycle(game: str | None = None, submit: bool = True) -> list[str]:
    state = observe()
    names = [game] if game else ["fps", "racing"]
    lines = []
    for name in names:
        lines.append(_cycle_game(state, name, submit))
    save(state)
    return lines


def _cycle_game(state: dict, game: str, submit: bool) -> str:
    row = state["games"][game]
    if game == "racing":
        # Rank is not a licence to nudge preview, kv, fuel, wing, or the stop.
        # A racing submission comes from the pace gate, then submit.py.
        incumbent = row.get("incumbent") or {}
        return (
            f"{game}: not changing a lever. The pace gate decides the next racing submission. "
            f"Selected car stays {incumbent.get('name')} version {incumbent.get('version_id')}"
        )
    if row.get("pending"):
        if not submit:
            return f"{game}: pending bundle held, upload skipped"
        _finish_pending(state, game)
        return f"{game}: resumed pending publish {row['incumbent']['version_id']}"
    fresh = [item for item in row.get("latest_own_matches") or [] if item["match_id"] not in set(row["seen_match_ids"])]
    if not fresh:
        return f"{game}: waiting for a finished own match"
    newest = fresh[0]
    source = Path(row["incumbent"]["archive"])
    current = read_strategy(source)
    lever = levers(game)[row["lever_cursor"] % len(levers(game))]
    generation = row["generation"] + 1
    proposed = revise(game, current, lever, bool(newest["worse"]), generation, newest["match_id"])
    archive = ROOT / "state" / "generations" / game / f"g{generation}-{lever}.zip"
    rebuild(source, game, proposed, archive)
    check_bundle(archive, game)
    digest = sha256(archive)
    row["pending"] = {
        "archive": str(archive),
        "sha256": digest,
        "name": proposed["name"],
        "lever": lever,
        "generation": generation,
        "match_id": newest["match_id"],
        "also_seen": [item["match_id"] for item in fresh],
        "upload_idempotency": str(uuid.uuid4()),
        "publish_idempotency": str(uuid.uuid4()),
        "submission_id": None,
    }
    save(state)
    if not submit:
        return f"{game}: built {proposed['name']} at {archive}, upload skipped"
    _finish_pending(state, game)
    return f"{game}: published {row['incumbent']['name']} version {row['incumbent']['version_id']}"


def _finish_pending(state: dict, game: str) -> None:
    row = state["games"][game]
    pending = row["pending"]
    agent_id = json.loads(ACCOUNT.read_text())["enrolled"][game]["agent_id"]
    submission_id = pending.get("submission_id")
    if not submission_id:
        blob = Path(pending["archive"]).read_bytes()
        _status, _headers, created = request(
            "POST", f"/api/v1/python-submissions/agents/{agent_id}/bundles",
            data=blob, content_type="application/zip", idempotency=pending["upload_idempotency"],
        )
        submission_id = _find_id(created)
        if not submission_id:
            raise AgxError(f"{game} upload returned no submission id")
        pending["submission_id"] = submission_id
        save(state)
    state_name, fault = _poll(submission_id)
    if state_name != "passed":
        raise AgxError(f"{game} admission {state_name or 'pending'}: {fault}")
    _status, _headers, published = request(
        "POST", f"/api/v1/python-submissions/competitions/{COMPETITIONS[game]}/versions",
        payload={"submission_id": submission_id, "auto_match": True},
        idempotency=pending["publish_idempotency"],
    )
    version_id = published.get("version_id") or (published.get("version") or {}).get("id")
    row["generation"] = pending["generation"]
    row["lever_cursor"] = (row["lever_cursor"] + 1) % len(levers(game))
    row["incumbent"] = {
        "archive": pending["archive"],
        "sha256": pending["sha256"],
        "submission_id": submission_id,
        "version_id": version_id,
        "name": pending["name"],
    }
    row["seen_match_ids"] = list(dict.fromkeys([*row["seen_match_ids"], *pending["also_seen"]]))
    row["cycles"].append({
        "generation": pending["generation"],
        "lever": pending["lever"],
        "match_id": pending["match_id"],
        "submission_id": submission_id,
        "version_id": version_id,
        "sha256": pending["sha256"],
    })
    row["pending"] = None


def _find_id(data: dict) -> str | None:
    for field in ("submission_id", "id", "bundle_id"):
        if data.get(field):
            return str(data[field])
    for nested in ("submission", "bundle"):
        child = data.get(nested)
        if isinstance(child, dict):
            found = _find_id(child)
            if found:
                return found
    return None


def _admission(data: dict) -> tuple[str, str]:
    item = data.get("admission")
    fault = str(data.get("fault_code") or data.get("error") or "")
    if isinstance(item, str):
        return item, fault
    if isinstance(item, dict):
        return str(item.get("status") or item.get("state") or ""), str(item.get("fault_code") or fault)
    if data.get("status") in ("passed", "failed", "rejected", "queued", "running"):
        return str(data["status"]), fault
    return "", fault


def _poll(submission_id: str) -> tuple[str, str]:
    import time
    state_name, fault = "", ""
    for _ in range(40):
        _status, _headers, current = request("GET", f"/api/v1/python-submissions/bundles/{submission_id}")
        state_name, fault = _admission(current)
        if state_name in ("passed", "failed", "rejected"):
            return state_name, fault
        time.sleep(3)
    return state_name, fault
