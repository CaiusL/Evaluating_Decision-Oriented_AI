import json

from self_jev.benchmark import GATES, build_report, write_reports


def _measurements(good: bool) -> dict:
    value = 0.95 if good else 0.1
    gap = 0.0 if good else 0.5
    rows = {
        "supported_canonical_accuracy": value,
        "recall_fire": value,
        "recall_adjust_aim": value,
        "recall_reload": value,
        "recall_wait": value,
        "paraphrase_b_gap": gap,
        "paraphrase_c_gap": gap,
        "shuffled_gap": gap,
        "arbitrary_ids_gap": gap,
        "negation_accuracy": value,
        "numeric_boundary_accuracy": value,
        "unsupported_recall": value,
        "abstain_unsupported": value,
        "abstain_supported": 0.0 if good else 0.5,
        "expected_calibration_error": 0.01 if good else 0.5,
        "brier": 0.01 if good else 0.5,
        "real_agreement": value,
        "latency_ms_median": 10.0 if good else 80.0,
        "supported_real_rows": 500 if good else 10,
        "unsupported_real_rows": 20,
        "recorded_action_agreement": 0.2,
        "latency_ms_p95": 20.0,
    }
    return rows


def _context(measurements):
    return {
        "dataset": {"groups_by_split": {"train": 1200, "select": 300, "calibrate": 300, "test": 200}},
        "selected_epoch": 1,
        "temperature": 1.0,
        "abstain_threshold": 0.5,
        "device": "mps",
        "model_sha256": "abc",
        "raw_export_sha256": "def",
        "supported_real_rows": measurements["supported_real_rows"],
        "unsupported_real_rows": measurements["unsupported_real_rows"],
        "recorded_action_agreement": measurements["recorded_action_agreement"],
        "latency_ms_p95": measurements["latency_ms_p95"],
    }


def test_report_contains_every_gate(tmp_path):
    good = _measurements(True)
    report = build_report(good, _context(good))
    write_reports(tmp_path, report)
    saved = json.loads((tmp_path / "readiness_benchmark.json").read_text())
    assert saved["milestone"] == "passed"
    assert {gate["name"] for gate in saved["gates"]} == {name for name, _direction, _limit in GATES}
    assert all(isinstance(gate["passed"], bool) for gate in saved["gates"])
    assert "not a hit" in saved["limits"]
    assert (tmp_path / "readiness_benchmark.md").exists()
    failed = build_report(_measurements(False), _context(_measurements(False)))
    assert failed["milestone"] == "failed"
