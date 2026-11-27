"""The Python submission ZIP matches the documented layout."""
import zipfile
from pathlib import Path

from self_jev.submit import find_sha, local_stamp, pack_directory


def test_pack_keeps_files_and_drops_bytecode(tmp_path: Path):
    source = tmp_path / "my-submission"
    (source / "weights").mkdir(parents=True)
    (source / "__pycache__").mkdir()
    (source / "agent.py").write_text("def act(observation):\n    return {}\n", encoding="utf-8")
    (source / "weights" / "model.npz").write_bytes(b"not-a-real-npz")
    (source / "__pycache__" / "agent.cpython-314.pyc").write_bytes(b"pyc")
    dest = tmp_path / "output" / "agent.zip"
    pack_directory(source, dest)
    with zipfile.ZipFile(dest) as bundle:
        names = bundle.namelist()
    assert names == ["agent.py", "weights/model.npz"]
    assert all(not name.endswith("/") for name in names)


def test_same_hash_is_not_a_new_upload():
    rows = [{"bundle_sha256": "abc", "id": "already", "number": 4}]
    assert find_sha(rows, "abc")["id"] == "already"
    assert find_sha(rows, "missing") is None


def test_version_four_timestamp_is_2144_local():
    assert local_stamp("2026-11-17T13:44:09.440607+00:00") == "2026-11-17 21:44"
