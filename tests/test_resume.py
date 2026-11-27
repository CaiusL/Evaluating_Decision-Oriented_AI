import pytest

from self_jev.train import resume_check


def test_resume_accepts_a_matching_hash_and_refuses_a_different_one():
    resume_check("abc", "abc")
    with pytest.raises(RuntimeError, match="dataset hash mismatch"):
        resume_check("abc", "different")
