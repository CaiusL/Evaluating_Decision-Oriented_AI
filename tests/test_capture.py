import json

import pytest

from self_jev.capture import CaptureError, capture
from self_jev.dataset import example_situation
from self_jev.hashing import sha256_file


def _export(situation, seat=6):
    weapon = situation["weapon"]
    return {
        "seat": seat,
        "step": 1,
        "observation": {
            "match_tick": situation["match_tick"],
            "round_index": 0,
            "phase": situation["phase"],
            "side": situation["side"],
            "map_id": "test-map",
            "map": {"id": "test-map", "sites": situation.get("sites") or []},
            "me": {
                "alive": situation["alive"],
                "x": situation["x"], "y": situation["y"], "z": situation["z"],
                "yaw": situation["yaw"], "pitch": situation["pitch"],
                "vx": situation["vx"], "vy": situation["vy"],
                "crouch": situation["crouch"],
                "ammo": situation["ammo"],
                "reload": situation["reload"],
                "active_slot": situation["active_slot"],
                "switch_until": situation["switch_until"],
                "blind_ticks": situation["blind_ticks"],
                "carrying_bomb": situation["carrying_bomb"],
                "weapons": [{
                    "id": weapon["id"],
                    "ammo": weapon.get("ammo"),
                    "reserve": weapon.get("reserve"),
                    "shot_ready": weapon.get("shot_ready"),
                    "reload_until": weapon.get("reload_until"),
                    "reload_state": weapon.get("reload_state"),
                    "trigger_state": {"pressed": weapon.get("trigger_pressed", False)},
                    "aim_punch": {"yaw": 0, "pitch": 0},
                }],
            },
            "visible": situation["enemies"],
            "bomb": situation["bomb"],
        },
        "action": {"fire": False, "reload": False, "yaw_rate": 0.0, "pitch_rate": 0.0},
    }


def _write(tmp_path, status="verified", sha=None, calls=1, seat=6, row_seat=6):
    raw = tmp_path / "match.jsonl"
    raw.write_text(json.dumps(_export(example_situation(), row_seat)) + "\n")
    stat = raw.stat()
    summary = {
        "status": status,
        "schema": 5,
        "calls": calls,
        "seat": seat,
        "match_id": "match-1",
        "sha256": sha if sha is not None else sha256_file(raw),
        "file_size": stat.st_size,
        "file_mtime_ns": stat.st_mtime_ns,
        "rules_version": "rsi-fps-5.5",
        "source_hash": "abc",
        "version_id": "ver",
    }
    path = tmp_path / "match.summary.json"
    path.write_text(json.dumps(summary))
    return raw, path


def test_capture_refuses_bad_summaries_and_writes_a_receipt(tmp_path):
    raw, summary = _write(tmp_path, status="pending")
    with pytest.raises(CaptureError, match="not verified"):
        capture(raw, summary, tmp_path / "data")
    assert not (tmp_path / "data" / "receipts").exists()
    raw, summary = _write(tmp_path, sha="0" * 64)
    with pytest.raises(CaptureError, match="does not match"):
        capture(raw, summary, tmp_path / "data")
    assert list((tmp_path / "data").glob("receipts/*")) == []
    raw, summary = _write(tmp_path, calls=2)
    with pytest.raises(CaptureError, match="decision count"):
        capture(raw, summary, tmp_path / "data")
    raw, summary = _write(tmp_path, row_seat=1)
    with pytest.raises(CaptureError, match="seat"):
        capture(raw, summary, tmp_path / "data")
    raw, summary = _write(tmp_path)
    receipt = capture(raw, summary, tmp_path / "data")
    saved = json.loads(receipt.read_text())
    assert saved["raw_deletion_allowed"] is True
    assert saved["raw_sha256"] == sha256_file(raw)
    assert saved["shard_sha256"]
    shard = tmp_path / "data" / "shards" / "fps" / "match-1-seat6.jsonl"
    assert shard.exists()
    with (tmp_path / "data" / "splits" / "real.jsonl").open() as handle:
        assert json.loads(handle.readline())["label"]
