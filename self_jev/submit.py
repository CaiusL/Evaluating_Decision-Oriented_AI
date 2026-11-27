"""Submit a Python bundle the way the API docs describe.

https://agx.cool/#/docs/python-submissions

Pack files only, upload the ZIP, poll admission until it passes or fails, then
select that version. The JavaScript `source` upload is a different endpoint and
is not used here. A queued admission is not a published version.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from self_jev.agx import AgxError, credentials, request
from self_jev.hashing import sha256_file

ROOT = Path(__file__).resolve().parents[1]
ACCOUNT = ROOT / "account.json"
RECEIPTS = ROOT / "state" / "python_submissions"
LOCAL = timezone(timedelta(hours=8))
TRANSIENT = {429, 502, 503, 504}


def pack_directory(source: Path, dest: Path) -> None:
    """ZIP layout from the Python submission guide. No directory entries."""
    source = Path(source)
    if not (source / "agent.py").is_file():
        raise AgxError(f"{source} has no agent.py")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(dest, "w", ZIP_DEFLATED) as bundle:
        for file in sorted(source.rglob("*")):
            if file.is_file() and "__pycache__" not in file.parts:
                bundle.write(file, file.relative_to(source).as_posix())


def local_stamp(value: str | None) -> str:
    if not value:
        return ""
    text = value.replace("Z", "+00:00")
    try:
        moment = datetime.fromisoformat(text)
    except ValueError:
        return value
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(LOCAL).strftime("%Y-%m-%d %H:%M")


def account_game(game: str) -> dict:
    enrolled = json.loads(ACCOUNT.read_text())["enrolled"][game]
    return {
        "agent_id": enrolled["agent_id"],
        "competition_id": enrolled["competition_id"],
    }


def bundles(agent_id: str) -> list[dict]:
    _status, _headers, body = request("GET", f"/api/v1/python-submissions/agents/{agent_id}/bundles")
    if isinstance(body, list):
        return body
    return list(body.get("items") or body.get("bundles") or [])


def find_sha(rows: list[dict], digest: str) -> dict | None:
    for row in rows:
        if row.get("bundle_sha256") == digest:
            return row
    return None


def admission_of(row: dict) -> tuple[str, str]:
    item = row.get("admission")
    fault = str(row.get("fault_code") or "")
    if isinstance(item, str):
        return item, fault
    if isinstance(item, dict):
        return str(item.get("status") or item.get("state") or ""), str(item.get("fault_code") or fault)
    return "", fault


def _retry_delay(headers, default: float) -> float:
    value = ""
    if headers is not None:
        value = headers.get("Retry-After") or ""
    if str(value).isdigit():
        return max(1.0, float(value))
    if value:
        try:
            moment = parsedate_to_datetime(value)
            if moment.tzinfo is None:
                moment = moment.replace(tzinfo=timezone.utc)
            return max(1.0, (moment - datetime.now(timezone.utc)).total_seconds())
        except (TypeError, ValueError, OSError):
            pass
    return default


def call(method: str, path: str, *, payload: dict | None = None, data: bytes | None = None,
         content_type: str | None = None, idempotency: str | None = None, attempts: int = 5):
    """Retry only transport failures, 429, and 502/503/504. Leave a 409 unchanged."""
    base, key = credentials()
    body = data
    headers = {"Accept": "application/json", "Authorization": "Bearer " + key}
    if payload is not None:
        body = json.dumps(payload).encode()
        headers["Content-Type"] = "application/json"
    if content_type:
        headers["Content-Type"] = content_type
    if method == "POST":
        headers["Idempotency-Key"] = idempotency or str(uuid.uuid4())
    for attempt in range(attempts):
        req = urllib.request.Request(base + path, data=body, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=120) as response:
                raw = response.read()
                parsed = json.loads(raw.decode() or "{}") if raw else {}
                return response.status, response.headers, parsed
        except urllib.error.HTTPError as exc:
            raw = exc.read().decode("utf-8", "replace")
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                parsed = {"raw": raw[:400]}
            if exc.code == 401:
                raise AgxError("401: the account API key is missing, invalid, or revoked", 401) from exc
            if exc.code not in TRANSIENT or attempt == attempts - 1:
                raise AgxError(f"{method} {path} -> HTTP {exc.code}: {json.dumps(parsed)[:400]}", exc.code) from exc
            time.sleep(_retry_delay(exc.headers, min(60.0, 2 ** attempt)))
        except OSError:
            if attempt == attempts - 1:
                raise
            time.sleep(min(60.0, 2 ** attempt))
    raise AgxError(f"{method} {path} failed")


def _receipt_path(digest: str) -> Path:
    return RECEIPTS / f"{digest}.json"


def _load_receipt(digest: str) -> dict:
    path = _receipt_path(digest)
    if not path.is_file():
        return {}
    return json.loads(path.read_text())


def _save_receipt(digest: str, row: dict) -> None:
    path = _receipt_path(digest)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row, indent=2) + "\n")


def _submission_id(created: dict) -> str | None:
    for field in ("submission_id", "id", "bundle_id"):
        if created.get(field):
            return str(created[field])
    for nested in ("submission", "bundle"):
        child = created.get(nested)
        if isinstance(child, dict):
            found = _submission_id(child)
            if found:
                return found
    return None


def poll_admission(submission_id: str, deadline_s: float = 180) -> dict:
    """Queued or running is still being checked. Stop on passed or failed."""
    deadline = time.monotonic() + deadline_s
    last = {}
    while True:
        _status, headers, last = call("GET", f"/api/v1/python-submissions/bundles/{submission_id}")
        state, _fault = admission_of(last)
        if state in ("passed", "failed", "rejected"):
            return last
        if time.monotonic() >= deadline:
            return last
        time.sleep(_retry_delay(headers, 5))


def select_version(competition_id: str, submission_id: str, idempotency: str) -> dict:
    _status, _headers, body = call(
        "POST",
        f"/api/v1/python-submissions/competitions/{competition_id}/versions",
        payload={"submission_id": submission_id, "auto_match": True},
        idempotency=idempotency,
    )
    return body


def submit_zip(game: str, archive: Path, *, select: bool = True) -> dict:
    """Upload one ZIP. The same bytes and idempotency key are reused on retry."""
    archive = Path(archive)
    digest = sha256_file(archive)
    ids = account_game(game)
    receipt = _load_receipt(digest)
    receipt.setdefault("sha256", digest)
    receipt.setdefault("archive", str(archive))
    receipt.setdefault("upload_idempotency", str(uuid.uuid4()))
    receipt.setdefault("publish_idempotency", str(uuid.uuid4()))
    existing = find_sha(bundles(ids["agent_id"]), digest)
    if existing and not receipt.get("submission_id"):
        receipt["submission_id"] = existing.get("id")
        receipt["version_id"] = existing.get("version_id")
        receipt["number"] = existing.get("number")
        receipt["admission"] = admission_of(existing)[0]
    if not receipt.get("submission_id"):
        _status, _headers, created = call(
            "POST",
            f"/api/v1/python-submissions/agents/{ids['agent_id']}/bundles",
            data=archive.read_bytes(),
            content_type="application/zip",
            idempotency=receipt["upload_idempotency"],
        )
        submission_id = _submission_id(created)
        if not submission_id:
            raise AgxError("upload returned no submission id")
        receipt["submission_id"] = submission_id
        _save_receipt(digest, receipt)
    current = poll_admission(receipt["submission_id"])
    state, fault = admission_of(current)
    receipt["admission"] = state
    receipt["fault_code"] = fault or None
    receipt["version_id"] = current.get("version_id") or receipt.get("version_id")
    receipt["number"] = current.get("number") or receipt.get("number")
    _save_receipt(digest, receipt)
    if state != "passed":
        raise AgxError(f"admission {state or 'pending'}: {fault}".strip())
    if select and not receipt.get("published"):
        try:
            published = select_version(ids["competition_id"], receipt["submission_id"], receipt["publish_idempotency"])
        except AgxError as exc:
            if exc.status != 409 or not receipt.get("version_id"):
                raise
            published = {"version_id": receipt.get("version_id")}
        receipt["published"] = True
        receipt["version_id"] = published.get("version_id") or receipt.get("version_id")
        _save_receipt(digest, receipt)
    return receipt


def format_bundles(rows: list[dict]) -> list[str]:
    lines = ["number  local_time       utc                   admission  version_id                            sha"]
    ordered = sorted(rows, key=lambda row: row.get("number") or 0, reverse=True)
    for row in ordered:
        state, _fault = admission_of(row)
        lines.append(
            f"{row.get('number')!s:>6}  {local_stamp(row.get('created_at')):<16} "
            f"{(row.get('created_at') or '')[:19]:<21} {state:<10} "
            f"{row.get('version_id') or '-':<37} {(row.get('bundle_sha256') or '')[:12]}"
        )
    return lines


def format_round(competition_id: str) -> str:
    _status, _headers, body = request("GET", f"/api/v1/competitions/{competition_id}/round")
    match = body.get("round_match") or {}
    version = match.get("python_version") or {}
    return (
        f"round {body.get('round_state')} job {match.get('job_id') or '-'} "
        f"state {match.get('state') or '-'} version {version.get('number')} {version.get('id') or '-'}"
    )


def run(game: str, archive: Path | None, directory: Path | None, *, select: bool) -> None:
    ids = account_game(game)
    if directory:
        archive = ROOT / "submissions" / game / f"{directory.name}.zip"
        pack_directory(directory, archive)
        print(f"packed {archive}")
    if archive:
        receipt = submit_zip(game, archive, select=select)
        print(
            f"submission {receipt.get('submission_id')} admission {receipt.get('admission')} "
            f"number {receipt.get('number')} version {receipt.get('version_id')} "
            f"published {bool(receipt.get('published'))}"
        )
    print("\n".join(format_bundles(bundles(ids["agent_id"]))))
    print(format_round(ids["competition_id"]))
    # GET /me/competitions/{id}/versions is the JavaScript source list. Python bundles are above.
    _status, _headers, listed = request("GET", f"/api/v1/me/competitions/{ids['competition_id']}/versions")
    source_versions = listed.get("versions") if isinstance(listed, dict) else None
    print(f"javascript source versions: {len(source_versions) if isinstance(source_versions, list) else 'unavailable'}")


def main(argv: list[str] | None = None) -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Submit a Python bundle and list the versions the site stores")
    parser.add_argument("--game", choices=("racing", "fps"), default="racing")
    parser.add_argument("--archive", type=Path, help="ZIP to upload. Omit to only list versions.")
    parser.add_argument("--directory", type=Path, help="Directory to pack, then upload.")
    parser.add_argument("--no-select", action="store_true", help="Upload and wait for admission, but do not select the version.")
    args = parser.parse_args(argv)
    run(args.game, args.archive, args.directory, select=not args.no_select)


if __name__ == "__main__":
    main()
