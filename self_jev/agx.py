"""AGX client for the self_jev account key only."""
from __future__ import annotations

import json
import uuid
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV = ROOT / ".env"


class AgxError(RuntimeError):
    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


def load_env() -> dict[str, str]:
    values = {}
    for line in ENV.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        values[key] = value
    return values


def credentials() -> tuple[str, str]:
    values = load_env()
    base = values.get("AGX_BASE_URL", "").rstrip("/")
    key = values.get("SELF_JEV_AGX_API_KEY", "")
    if not base or not key:
        raise AgxError("self_jev/.env needs AGX_BASE_URL and SELF_JEV_AGX_API_KEY")
    return base, key


def request(method: str, path: str, params: dict | None = None, payload: dict | None = None,
            data: bytes | None = None, content_type: str | None = None, idempotency: str | None = None):
    base, key = credentials()
    query = ("?" + urllib.parse.urlencode(params)) if params else ""
    headers = {"Accept": "application/json", "Authorization": "Bearer " + key}
    body = data
    if payload is not None:
        body = json.dumps(payload).encode()
        headers["Content-Type"] = "application/json"
    if content_type:
        headers["Content-Type"] = content_type
    if method == "POST":
        headers["Idempotency-Key"] = idempotency or str(uuid.uuid4())
    req = urllib.request.Request(base + path + query, data=body, headers=headers, method=method)
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
        raise AgxError(f"{method} {path} -> HTTP {exc.code}: {json.dumps(parsed)[:400]}", exc.code) from exc
