"""Build and submit the first Self-JEV revision of each coach base."""
from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

from self_jev.agx import AgxError, credentials, request
from self_jev.bundle import read_strategy
from self_jev.check import check_bundle
from self_jev.coach_base import export_all
from self_jev.observe import own_matches
from self_jev.rewrite import rewrite_archive
from self_jev.workflow import load as load_workflow
from self_jev.workflow import save as save_workflow

ROOT = Path(__file__).resolve().parents[1]
ACCOUNT = ROOT / "account.json"
MANIFEST = ROOT / "submissions" / "manifest.json"
EXCHANGES = ROOT / "state" / "exchanges"


def build(predict_fn=None) -> dict:
    bases = export_all()
    built = {}
    for game, provenance in bases.items():
        source = Path(provenance["archive"])
        critique = rewrite_archive(
            game,
            source,
            ROOT / "submissions" / game / "pending.zip",
            predict_fn=predict_fn,
        )
        name = read_strategy(Path(critique["archive"]))["name"]
        dest = ROOT / "submissions" / game / f"{name}.zip"
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.resolve() != Path(critique["archive"]).resolve():
            dest.write_bytes(Path(critique["archive"]).read_bytes())
            Path(critique["archive"]).unlink()
        critique["archive"] = str(dest)
        critique["coach_base"] = provenance
        critique["smoke"] = check_bundle(dest, game)
        record = dest.with_suffix(".json")
        record.write_text(json.dumps(critique, ensure_ascii=False, indent=2) + "\n")
        built[game] = critique
    return built


def submit(built: dict) -> dict:
    _base, secret = credentials()
    account = json.loads(ACCOUNT.read_text())
    manifest = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {"schema": 1, "games": {}}
    results = {}
    for game, critique in built.items():
        results[game] = _submit_game(account, manifest, game, critique, secret)
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    account["self_jev_submissions"] = {
        game: {key: value for key, value in row.items() if key != "response"}
        for game, row in results.items()
    }
    ACCOUNT.write_text(json.dumps(account, ensure_ascii=False, indent=2) + "\n")
    _remember(results)
    return results


def status_text(built: dict, results: dict | None = None) -> str:
    lines = ["Self-JEV first submissions"]
    for game, critique in built.items():
        lines.append(
            f"{game}: {critique['base_name']} -> {Path(critique['archive']).name} "
            f"lever {critique['lever']} p {critique['probabilities'][critique['choice_id']]:.3f} "
            f"sha {critique['sha256'][:12]}"
        )
        if results and game in results:
            row = results[game]
            lines.append(
                f"  submit {row.get('admission') or row.get('error')} "
                f"version {row.get('version_id')} submission {row.get('submission_id')}"
            )
    return "\n".join(lines)


def _submit_game(account: dict, manifest: dict, game: str, critique: dict, secret: str) -> dict:
    games = manifest.setdefault("games", {})
    row = games.setdefault(game, {})
    if row.get("version_id") and row.get("sha256") == critique["sha256"]:
        return row
    if row.get("submission_id") and row.get("uploaded_sha256") not in (None, critique["sha256"]):
        row["submission_id"] = None
        row["upload_idempotency"] = str(uuid.uuid4())
        row["publish_idempotency"] = str(uuid.uuid4())
    row.update({
        "name": Path(critique["archive"]).stem,
        "archive": critique["archive"],
        "sha256": critique["sha256"],
        "lever": critique["lever"],
        "choice_id": critique["choice_id"],
        "coach_submission_id": critique["coach_base"]["coach_submission_id"],
        "coach_name": critique["base_name"],
    })
    row.setdefault("upload_idempotency", str(uuid.uuid4()))
    row.setdefault("publish_idempotency", str(uuid.uuid4()))
    _save_manifest(manifest)
    agent_id = account["enrolled"][game]["agent_id"]
    competition = account["enrolled"][game]["competition_id"]
    try:
        submission_id = row.get("submission_id")
        if not submission_id:
            _status, _headers, created = request(
                "POST",
                f"/api/v1/python-submissions/agents/{agent_id}/bundles",
                data=Path(critique["archive"]).read_bytes(),
                content_type="application/zip",
                idempotency=row["upload_idempotency"],
            )
            _write_exchange(game, "upload", created, secret)
            submission_id = _find_id(created)
            if not submission_id:
                raise AgxError(f"{game} upload returned no submission id")
            row["submission_id"] = submission_id
            row["uploaded_sha256"] = critique["sha256"]
            _save_manifest(manifest)
        state_name, fault = _poll(submission_id)
        row["admission"] = state_name
        row["fault"] = fault
        if state_name != "passed":
            raise AgxError(f"{game} admission {state_name or 'pending'}: {fault}")
        payload = {"submission_id": submission_id, "auto_match": True}
        _status, _headers, published = request(
            "POST",
            f"/api/v1/python-submissions/competitions/{competition}/versions",
            payload=payload,
            idempotency=row["publish_idempotency"],
        )
        _write_exchange(game, "publish", {"request": payload, "response": published}, secret)
        version_id = published.get("version_id") or (published.get("version") or {}).get("id")
        row["version_id"] = version_id
        row["auto_match"] = True
        row["error"] = None
        _save_manifest(manifest)
        row["round"] = _round_snapshot(competition, secret)
        return row
    except AgxError as exc:
        row["error"] = str(exc)
        _write_exchange(game, "error", {"error": str(exc)}, secret)
        _save_manifest(manifest)
        return row


def _remember(results: dict) -> None:
    state = load_workflow()
    for game, row in results.items():
        if not row.get("version_id"):
            continue
        current = state["games"][game]
        current["generation"] = 1
        current["incumbent"] = {
            "archive": row["archive"],
            "sha256": row["sha256"],
            "submission_id": row.get("submission_id"),
            "version_id": row.get("version_id"),
            "name": row.get("name"),
        }
        cycle = {
            "generation": 1,
            "lever": row.get("lever"),
            "match_id": "coach-base",
            "submission_id": row.get("submission_id"),
            "version_id": row.get("version_id"),
            "sha256": row["sha256"],
        }
        if not any(item.get("sha256") == row["sha256"] for item in current["cycles"]):
            current["cycles"].append(cycle)
        try:
            played = [item["match_id"] for item in own_matches(game)]
        except AgxError:
            played = []
        already = current.get("seen_match_ids") or []
        current["seen_match_ids"] = list(dict.fromkeys([*already, *played]))
        current["pending"] = None
    save_workflow(state)


def _round_snapshot(competition: str, secret: str) -> dict:
    try:
        _status, _headers, body = request("GET", f"/api/v1/competitions/{competition}/round")
    except AgxError as exc:
        return {"error": str(exc)}
    match = body.get("round_match") or {}
    version = match.get("python_version") or {}
    snapshot = {
        "round_state": body.get("round_state"),
        "next_round": body.get("next_round"),
        "job_id": match.get("job_id"),
        "match_state": match.get("state"),
        "version_id": version.get("id"),
        "version_number": version.get("number"),
        "bundle_sha256": version.get("bundle_sha256"),
    }
    _write_exchange("round", competition, snapshot, secret)
    return snapshot


def _save_manifest(manifest: dict) -> None:
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")


def _write_exchange(game: str, kind: str, payload, secret: str) -> None:
    EXCHANGES.mkdir(parents=True, exist_ok=True)
    path = EXCHANGES / f"{game}-{kind}.json"
    path.write_text(json.dumps(_redact(payload, secret), ensure_ascii=False, indent=2) + "\n")


def _redact(value, secret: str):
    if isinstance(value, dict):
        return {key: _redact(item, secret) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact(item, secret) for item in value]
    if isinstance(value, str) and secret and secret in value:
        return value.replace(secret, secret[:4] + "…" + secret[-4:])
    return value


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
    state_name, fault = "", ""
    for _ in range(60):
        _status, _headers, current = request("GET", f"/api/v1/python-submissions/bundles/{submission_id}")
        state_name, fault = _admission(current)
        if state_name in ("passed", "failed", "rejected"):
            return state_name, fault
        time.sleep(15)
    return state_name, fault


def main() -> None:
    built = build()
    results = submit(built)
    print(status_text(built, results))
