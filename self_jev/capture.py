"""Read one verified FPS export and write a readiness shard under data/."""
from __future__ import annotations

import hashlib
import json
import math
import shutil
from pathlib import Path

from self_jev.hashing import sha256_file
from self_jev.readiness import LABELER_VERSION, QUESTION, describe_aim, label
from self_jev.text import candidates, render

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DEFAULT_RAW = Path(
    "/Users/caius/Projects/Comp/state/training/fps/2df4b71e-08bb-41a0-b899-ff4de556fb74-seat6.jsonl"
)
REAL_IDS = {
    "fire": "north",
    "adjust_aim": "k7",
    "reload": "quartz",
    "wait": "river",
    "unsupported": "moth",
}


class CaptureError(RuntimeError):
    pass


def main() -> None:
    receipt = capture(DEFAULT_RAW, DEFAULT_RAW.with_suffix(".summary.json"), DATA)
    print(receipt)


def capture(raw_path: Path, summary_path: Path, data_root: Path = DATA) -> Path:
    raw_path = Path(raw_path)
    summary_path = Path(summary_path)
    data_root = Path(data_root)
    summary = json.loads(summary_path.read_text())
    if summary.get("status") != "verified":
        raise CaptureError("summary is not verified")
    digest, decision_count = _decision_digest(raw_path)
    stat = raw_path.stat()
    if digest != summary.get("sha256") or stat.st_size != summary.get("file_size") or stat.st_mtime_ns != summary.get("file_mtime_ns"):
        raise CaptureError("raw file does not match the summary")
    if decision_count != summary.get("calls"):
        raise CaptureError(f"decision count {decision_count} does not match the summary")
    if summary.get("schema") != 5:
        raise CaptureError("summary schema is not 5")
    match_id = summary.get("match_id")
    seat = summary.get("seat")
    if not match_id or seat is None:
        raise CaptureError("summary is missing the match or the seat")
    partial = data_root / ".capture-partial"
    if partial.exists():
        shutil.rmtree(partial)
    partial.mkdir(parents=True)
    try:
        header, decisions, real_rows, counts = _read(raw_path, summary, digest)
        if len(decisions) != summary.get("calls"):
            raise CaptureError(f"decision count {len(decisions)} does not match the summary")
        stem = f"{match_id}-seat{seat}"
        shard_path = partial / f"{stem}.jsonl"
        with shard_path.open("w") as handle:
            handle.write(json.dumps(header) + "\n")
            for row in decisions:
                handle.write(json.dumps(row) + "\n")
        real_path = partial / "real.jsonl"
        with real_path.open("w") as handle:
            for row in real_rows:
                handle.write(json.dumps(row) + "\n")
        meta = {
            "schema": "self_jev_readiness_v1",
            "labeler_version": LABELER_VERSION,
            "source_path": str(raw_path),
            "byte_size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "raw_sha256": digest,
            "verified_summary_sha256": sha256_file(summary_path),
            "summary_schema": summary.get("schema"),
            "match_id": match_id,
            "seat": seat,
            "rules_version": summary.get("rules_version"),
            "source_hash": summary.get("source_hash"),
            "version_id": summary.get("version_id"),
            "decisions": len(decisions),
            "counts_by_label": counts,
            "extractor_sha256": sha256_file(Path(__file__)),
            "shard_sha256": sha256_file(shard_path),
            "real_sha256": sha256_file(real_path),
        }
        meta_path = partial / f"{stem}.meta.json"
        meta_path.write_text(json.dumps(meta, indent=2) + "\n")
        shard_dir = data_root / "shards" / "fps"
        split_dir = data_root / "splits"
        receipt_dir = data_root / "receipts"
        for directory in (shard_dir, split_dir, receipt_dir):
            directory.mkdir(parents=True, exist_ok=True)
        final_shard = shard_dir / shard_path.name
        final_meta = shard_dir / meta_path.name
        final_real = split_dir / "real.jsonl"
        shutil.move(shard_path, final_shard)
        shutil.move(meta_path, final_meta)
        shutil.move(real_path, final_real)
        receipt = {
            "raw_deletion_allowed": True,
            "raw_sha256": digest,
            "shard_sha256": meta["shard_sha256"],
            "meta_sha256": sha256_file(final_meta),
            "match_id": match_id,
            "seat": seat,
        }
        receipt_path = receipt_dir / f"{stem}.json"
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
        return receipt_path
    finally:
        if partial.exists():
            shutil.rmtree(partial)


def _decision_digest(path: Path) -> tuple[str, int]:
    """Hash the original decision lines. That is the digest stored on a verified summary."""
    digest = hashlib.sha256()
    calls = 0
    with Path(path).open("rb") as stream:
        for line in stream:
            row = json.loads(line)
            if isinstance(row.get("observation"), dict) and row.get("type") in (None, "decision"):
                digest.update(line)
                calls += 1
    return digest.hexdigest(), calls


def _read(raw_path: Path, summary: dict, digest: str):
    seat = summary["seat"]
    release = (summary.get("calibration") or {}).get("release_required_weapons") or ["pistol", "usp"]
    map_id = None
    sites = []
    decisions = []
    real_rows = []
    counts = {}
    with raw_path.open() as handle:
        for line in handle:
            row = json.loads(line)
            if "observation" not in row:
                continue
            if row.get("seat") != seat:
                raise CaptureError("seat does not match the summary")
            observation = row["observation"]
            if map_id is None and isinstance(observation.get("map"), dict):
                map_id = observation.get("map_id") or observation["map"].get("id")
                sites = [
                    {"id": site.get("id"), "center": site.get("center"), "radius": site.get("radius")}
                    for site in observation["map"].get("sites") or []
                    if isinstance(site, dict)
                ]
            situation = situation_from_observation(observation, sites, release)
            try:
                choice, reason = label(situation)
            except Exception as exc:
                raise CaptureError(f"label failed at step {row.get('step')}") from exc
            detail = describe_aim(situation)
            speed = _speed(situation)
            compact = {
                "match_id": summary["match_id"],
                "seat": seat,
                "step": row.get("step"),
                "round_index": observation.get("round_index"),
                "match_tick": situation.get("match_tick"),
                "phase": situation.get("phase"),
                "alive": situation.get("alive"),
                "x": situation.get("x"),
                "y": situation.get("y"),
                "z": situation.get("z"),
                "yaw": situation.get("yaw"),
                "pitch": situation.get("pitch"),
                "vx": situation.get("vx"),
                "vy": situation.get("vy"),
                "crouch": situation.get("crouch"),
                "active_slot": situation.get("active_slot"),
                "ammo": situation.get("ammo"),
                "blind_ticks": situation.get("blind_ticks"),
                "reload": situation.get("reload"),
                "switch_until": situation.get("switch_until"),
                "weapon": situation.get("weapon"),
                "other_gun": situation.get("other_gun"),
                "enemies": situation.get("enemies"),
                "nearest_distance_m": None if detail is None else detail["distance"],
                "aim_error_rad": None if detail is None else detail["error"],
                "aim_tolerance_rad": None if detail is None else detail["allowance"],
                "aligned": None if detail is None else detail["aligned"],
                "stationary": None if speed is None else speed < 0.5,
                "label": choice,
                "reason": reason,
                "source_sha256": digest,
            }
            decisions.append(compact)
            text, _literals = render(situation)
            real_rows.append({
                "match_id": summary["match_id"],
                "seat": seat,
                "step": row.get("step"),
                "label": choice,
                "reason": reason,
                "recorded": recorded_action(row.get("action") or {}),
                "question": QUESTION,
                "situation": text,
                "candidates": candidates(REAL_IDS),
            })
            counts[choice] = counts.get(choice, 0) + 1
    header = {
        "record": "header",
        "schema": "self_jev_readiness_v1",
        "match_id": summary["match_id"],
        "seat": seat,
        "map_id": map_id,
        "sites": sites,
    }
    return header, decisions, real_rows, counts


def situation_from_observation(observation: dict, sites: list, release_required: list) -> dict:
    me = observation.get("me") or {}
    weapons = me.get("weapons") or []
    slot = me.get("active_slot")
    weapon = weapons[slot] if isinstance(slot, int) and 0 <= slot < len(weapons) and isinstance(weapons[slot], dict) else {}
    punch = weapon.get("aim_punch") if isinstance(weapon.get("aim_punch"), dict) else {}
    weapon_out = {
        "id": weapon.get("id"),
        "ammo": weapon.get("ammo"),
        "reserve": weapon.get("reserve"),
        "shot_ready": weapon.get("shot_ready"),
        "reload_until": weapon.get("reload_until"),
        "reload_state": weapon.get("reload_state"),
        "aim_punch_yaw": punch.get("yaw") or 0.0,
        "aim_punch_pitch": punch.get("pitch") or 0.0,
    }
    trigger = weapon.get("trigger_state") if isinstance(weapon.get("trigger_state"), dict) else None
    if isinstance(trigger, dict) and "pressed" in trigger:
        weapon_out["trigger_pressed"] = trigger.get("pressed")
    other = {}
    for index in (0, 1):
        if index == slot or index >= len(weapons) or not isinstance(weapons[index], dict):
            continue
        item = weapons[index]
        other = {"id": item.get("id"), "ammo": item.get("ammo"), "reserve": item.get("reserve")}
        break
    enemies = []
    for enemy in observation.get("visible") or []:
        if not isinstance(enemy, dict):
            continue
        enemies.append({
            "seat": enemy.get("seat"),
            "x": enemy.get("x"),
            "y": enemy.get("y"),
            "z": enemy.get("z"),
            "alive": enemy.get("alive", True),
            "crouch": enemy.get("crouch", False),
            "parts": enemy.get("parts"),
        })
    return {
        "phase": observation.get("phase"),
        "alive": me.get("alive"),
        "match_tick": observation.get("match_tick"),
        "x": me.get("x"),
        "y": me.get("y"),
        "z": me.get("z"),
        "yaw": me.get("yaw"),
        "pitch": me.get("pitch"),
        "vx": me.get("vx"),
        "vy": me.get("vy"),
        "crouch": me.get("crouch"),
        "side": observation.get("side"),
        "carrying_bomb": me.get("carrying_bomb"),
        "blind_ticks": me.get("blind_ticks"),
        "active_slot": slot,
        "ammo": me.get("ammo"),
        "reload": me.get("reload"),
        "switch_until": me.get("switch_until"),
        "weapon": weapon_out,
        "other_gun": other,
        "enemies": enemies,
        "bomb": observation.get("bomb") or {},
        "sites": sites,
        "release_required": list(release_required),
        "missing": None,
    }


def recorded_action(action: dict) -> str:
    if action.get("fire") is True:
        return "fire"
    if action.get("reload") is True:
        return "reload"
    try:
        yaw = float(action.get("yaw_rate") or 0)
        pitch = float(action.get("pitch_rate") or 0)
    except (TypeError, ValueError):
        return "wait"
    if abs(yaw) > 0.05 or abs(pitch) > 0.05:
        return "adjust_aim"
    return "wait"


def _speed(situation: dict):
    vx, vy = situation.get("vx"), situation.get("vy")
    if isinstance(vx, bool) or isinstance(vy, bool) or not isinstance(vx, (int, float)) or not isinstance(vy, (int, float)):
        return None
    return math.hypot(vx, vy)


if __name__ == "__main__":
    main()
