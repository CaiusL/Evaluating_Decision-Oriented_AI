from collections import Counter

from self_jev.dataset import generate
from self_jev.readiness import CHOICES, UNSUPPORTED


def test_groups_do_not_leak_across_splits():
    examples = generate()
    splits = {}
    for example in examples:
        splits.setdefault(example["group_id"], set()).add(example["split"])
    assert splits
    assert all(len(found) == 1 for found in splits.values())
    canonical = [example for example in examples if example["render"] == "canonical"]
    assert len(canonical) == 2000
    labels = Counter(example["canon_label"] for example in canonical)
    for label in (*CHOICES, UNSUPPORTED):
        assert labels[label] == 400
    counts = Counter(example["split"] for example in canonical)
    assert counts == {"train": 1200, "select": 300, "calibrate": 300, "test": 200}
    assert all(example["split"] == "test" for example in examples if example["boundary"])
    missing = [example for example in examples if example["render"] == "missing"]
    assert missing
    assert all(example["label_key"] == UNSUPPORTED for example in missing)
    assert all(example["group_id"] in splits for example in missing)
