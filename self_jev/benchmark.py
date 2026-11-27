"""Score the frozen checkpoint and write the readiness report."""
from __future__ import annotations

import json
import math
import statistics
import time
from pathlib import Path

from self_jev.dataset import load_dataset, manifest
from self_jev.hashing import sha256_file
from self_jev.metrics import brier, expected_calibration_error, recall
from self_jev.predict import load_bundle, predict, predict_many
from self_jev.readiness import CHOICES, UNSUPPORTED
from self_jev.text import DESCRIPTIONS

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
LIMITS = (
    "Fire is a readiness decision, not a hit. This run does not estimate kills or match outcome. "
    "Damage-risk rows and synthetic aim transitions are not evidence for these gates."
)
GATES = (
    ("supported_canonical_accuracy", "min", 0.90),
    ("recall_fire", "min", 0.85),
    ("recall_adjust_aim", "min", 0.85),
    ("recall_reload", "min", 0.85),
    ("recall_wait", "min", 0.85),
    ("paraphrase_b_gap", "max", 0.03),
    ("paraphrase_c_gap", "max", 0.03),
    ("shuffled_gap", "max", 0.02),
    ("arbitrary_ids_gap", "max", 0.02),
    ("negation_accuracy", "min", 0.85),
    ("numeric_boundary_accuracy", "min", 0.85),
    ("unsupported_recall", "min", 0.90),
    ("abstain_unsupported", "min", 0.80),
    ("abstain_supported", "max", 0.05),
    ("expected_calibration_error", "max", 0.10),
    ("brier", "max", 0.12),
    ("real_agreement", "min", 0.80),
    ("latency_ms_median", "max", 50.0),
)


def main() -> None:
    report = run()
    print(report["milestone"])
    for gate in report["gates"]:
        print(f"{gate['name']}: {gate['value']} passed={gate['passed']}")


def run() -> dict:
    examples = load_dataset()
    bundle = load_bundle()
    test = [example for example in examples if example["split"] == "test"]
    calibrate = [example for example in examples if example["split"] == "calibrate" and example["label_key"] in CHOICES]
    scored = {id(example): result for example, result in zip(test, predict_many(test))}
    if test:
        check = predict(test[0]["situation"], test[0]["question"], test[0]["candidates"])
        if check["choice_id"] != scored[id(test[0])]["choice_id"]:
            raise RuntimeError("predict disagrees with the batched scorer")
    measurements = _generated_measurements(test, calibrate, scored, bundle["abstain_threshold"])
    real_rows = _load_real()
    measurements.update(_real_measurements(real_rows))
    measurements.update(_latency(test[0] if test else None))
    context = {
        "dataset": manifest(examples),
        "selected_epoch": bundle.get("epoch"),
        "eligible": bundle.get("eligible"),
        "temperature": bundle.get("temperature"),
        "abstain_threshold": bundle.get("abstain_threshold"),
        "device": str(bundle["device"]),
        "model_sha256": sha256_file(bundle["directory"] / "best.pt"),
        "raw_export_sha256": _raw_sha(),
        "supported_real_rows": measurements.get("supported_real_rows", 0),
        "unsupported_real_rows": measurements.get("unsupported_real_rows", 0),
        "recorded_action_agreement": measurements.get("recorded_action_agreement"),
        "latency_ms_p95": measurements.get("latency_ms_p95"),
    }
    report = build_report(measurements, context)
    write_reports(REPORTS, report)
    return report


def build_report(measurements: dict, context: dict) -> dict:
    gates = []
    for name, direction, limit in GATES:
        value = measurements.get(name)
        if name == "real_agreement":
            enough = measurements.get("supported_real_rows", 0) >= 200
            passed = enough and value is not None and value >= limit
        elif value is None:
            passed = False
        elif direction == "min":
            passed = value >= limit
        else:
            passed = value <= limit
        gates.append({"name": name, "value": value, "limit": limit, "direction": direction, "passed": bool(passed)})
    return {
        "milestone": "passed" if all(gate["passed"] for gate in gates) else "failed",
        "gates": gates,
        "dataset": context.get("dataset"),
        "split_sizes": (context.get("dataset") or {}).get("groups_by_split"),
        "selected_epoch": context.get("selected_epoch"),
        "eligible": context.get("eligible"),
        "temperature": context.get("temperature"),
        "abstain_threshold": context.get("abstain_threshold"),
        "supported_real_rows": context.get("supported_real_rows"),
        "unsupported_real_rows": context.get("unsupported_real_rows"),
        "recorded_action_agreement": context.get("recorded_action_agreement"),
        "device": context.get("device"),
        "latency_ms_median": measurements.get("latency_ms_median"),
        "latency_ms_p95": context.get("latency_ms_p95"),
        "model_sha256": context.get("model_sha256"),
        "raw_export_sha256": context.get("raw_export_sha256"),
        "limits": LIMITS,
    }


def write_reports(directory: Path, report: dict) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "readiness_benchmark.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = [
        "# Self-JEV readiness benchmark",
        "",
        f"Milestone: {report['milestone']}.",
        "",
        f"Selected epoch: {report['selected_epoch']}. Temperature: {report['temperature']}. Abstain threshold: {report['abstain_threshold']}.",
        "",
        f"Device: {report['device']}. Median latency: {report['latency_ms_median']} ms. 95th percentile: {report['latency_ms_p95']} ms.",
        "",
        "| Gate | Value | Limit | Passed |",
        "| --- | --- | --- | --- |",
    ]
    for gate in report["gates"]:
        lines.append(f"| {gate['name']} | {gate['value']} | {gate['direction']} {gate['limit']} | {gate['passed']} |")
    lines.extend([
        "",
        f"Supported real rows: {report['supported_real_rows']}. Unsupported real rows: {report['unsupported_real_rows']}.",
        f"Recorded-action agreement: {report['recorded_action_agreement']}.",
        f"Model sha256: {report['model_sha256']}.",
        f"Raw export sha256: {report['raw_export_sha256']}.",
        "",
        report["limits"],
        "",
    ])
    (directory / "readiness_benchmark.md").write_text("\n".join(lines))


def _generated_measurements(test, calibrate, scored, threshold: float) -> dict:
    def rows(predicate):
        return [example for example in test if predicate(example)]

    def accuracy(chosen):
        if not chosen:
            return None
        hits = sum(_key(example, scored[id(example)]) == example["label_key"] for example in chosen)
        return hits / len(chosen)

    canonical = rows(lambda example: example["render"] == "canonical" and example["label_key"] in CHOICES)
    canonical_accuracy = accuracy(canonical)
    labels = [_key(example, scored[id(example)]) for example in canonical]
    truth = [example["label_key"] for example in canonical]
    measurements = {
        "supported_canonical_accuracy": canonical_accuracy,
        "negation_accuracy": accuracy(rows(lambda example: example["render"] == "negation")),
        "numeric_boundary_accuracy": accuracy(rows(lambda example: example["boundary"])),
        "unsupported_recall": _unsupported_recall(rows(lambda example: example["label_key"] == UNSUPPORTED), scored),
    }
    for label in CHOICES:
        measurements[f"recall_{label}"] = recall(truth, labels, label) if truth else None
    for frame, limit_name in (
        ("paraphrase_b", "paraphrase_b_gap"),
        ("paraphrase_c", "paraphrase_c_gap"),
        ("shuffled", "shuffled_gap"),
        ("arbitrary_ids", "arbitrary_ids_gap"),
    ):
        frame_accuracy = accuracy(rows(lambda example, frame=frame: example["render"] == frame and example["label_key"] in CHOICES))
        measurements[limit_name] = None if None in (canonical_accuracy, frame_accuracy) else abs(frame_accuracy - canonical_accuracy)
    supported = rows(lambda example: example["label_key"] in CHOICES)
    probability_rows = []
    confidence = []
    correct = []
    for example in supported:
        probabilities = _keyed_probabilities(example, scored[id(example)])
        probability_rows.append(probabilities)
        choice = _key(example, scored[id(example)])
        confidence.append(max(scored[id(example)]["probabilities"].values()))
        correct.append(choice == example["label_key"])
    measurements["brier"] = brier(probability_rows, [example["label_key"] for example in supported]) if supported else None
    measurements["expected_calibration_error"] = expected_calibration_error(confidence, correct) if supported else None
    measurements["abstain_supported"] = _abstain_rate(calibrate, threshold, drop_sentinel=False)
    measurements["abstain_unsupported"] = _abstain_rate(
        rows(lambda example: example["label_key"] == UNSUPPORTED), threshold, drop_sentinel=True,
    )
    return measurements


def _abstain_rate(examples, threshold: float, drop_sentinel: bool) -> float | None:
    if not examples:
        return None
    prepared = []
    for example in examples:
        candidates = example["candidates"]
        if drop_sentinel:
            candidates = [
                candidate for candidate in candidates
                if candidate.get("key") != UNSUPPORTED and candidate.get("description") != DESCRIPTIONS[UNSUPPORTED]
            ]
        prepared.append({**example, "candidates": candidates})
    scored = predict_many(prepared)
    abstained = 0
    for result in scored:
        if max(result["probabilities"].values()) < threshold:
            abstained += 1
    return abstained / len(scored)


def _real_measurements(rows: list[dict]) -> dict:
    supported = [row for row in rows if row.get("label") in CHOICES]
    unsupported = [row for row in rows if row.get("label") == UNSUPPORTED]
    agreement = None
    recorded = None
    if supported:
        scored = predict_many(supported)
        hits = 0
        recorded_hits = 0
        for row, result in zip(supported, scored):
            choice = _key(row, result)
            hits += choice == row["label"]
            recorded_hits += row.get("recorded") == row["label"]
        agreement = hits / len(supported)
        recorded = recorded_hits / len(supported)
    return {
        "real_agreement": agreement,
        "recorded_action_agreement": recorded,
        "supported_real_rows": len(supported),
        "unsupported_real_rows": len(unsupported),
    }


def _latency(example: dict | None) -> dict:
    if example is None:
        return {"latency_ms_median": None, "latency_ms_p95": None}
    candidates = [candidate for candidate in example["candidates"] if candidate.get("key") in CHOICES][:4]
    for _ in range(10):
        predict(example["situation"], example["question"], candidates)
    samples = []
    for _ in range(50):
        started = time.perf_counter()
        predict(example["situation"], example["question"], candidates)
        samples.append((time.perf_counter() - started) * 1000)
    ordered = sorted(samples)
    index = min(len(ordered) - 1, max(0, math.ceil(0.95 * len(ordered)) - 1))
    return {"latency_ms_median": statistics.median(samples), "latency_ms_p95": ordered[index]}


def _load_real() -> list[dict]:
    path = ROOT / "data" / "splits" / "real.jsonl"
    if not path.exists():
        return []
    rows = []
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            row["label_key"] = row.get("label")
            rows.append(row)
    return rows


def _raw_sha():
    path = ROOT / "data" / "receipts"
    if not path.exists():
        return None
    receipts = sorted(path.glob("*.json"))
    if not receipts:
        return None
    return json.loads(receipts[-1].read_text()).get("raw_sha256")


def _key(example: dict, result: dict) -> str:
    by_id = {candidate["id"]: candidate.get("key") for candidate in example["candidates"]}
    return by_id[result["choice_id"]]


def _keyed_probabilities(example: dict, result: dict) -> dict[str, float]:
    by_id = {candidate["id"]: candidate.get("key") for candidate in example["candidates"]}
    return {by_id[key]: value for key, value in result["probabilities"].items()}


def _unsupported_recall(examples, scored) -> float | None:
    if not examples:
        return None
    hits = sum(_key(example, scored[id(example)]) == UNSUPPORTED for example in examples)
    return hits / len(examples)


if __name__ == "__main__":
    main()
