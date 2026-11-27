"""Fine-tune a small element judge. The readiness checkpoint is left in place."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from self_jev.element_data import POINTER, load_examples, write_dataset
from self_jev.hashing import sha256_file
from self_jev.metrics import macro_f1, tempered_softmax
from self_jev.model import ENCODER_NAME
from self_jev.train import _epoch, _fit_temperature, _score_split

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
READINESS = RUNS / "readiness-6f722792ffed"
EPOCHS = 4
OLD_NAME = "readiness-6f722792ffed"


def main() -> None:
    run_dir = train()
    print(run_dir)


def train() -> Path:
    import torch

    from self_jev.model import load_encoder, pair_text, training_logits

    if not torch.backends.mps.is_available():
        raise RuntimeError("Self-JEV training requires MPS")
    path = write_dataset()
    examples = load_examples()
    dataset_sha = sha256_file(path)
    run_dir = RUNS / f"elements-{dataset_sha[:12]}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "manifest.json").write_text(json.dumps({
        "task": "elements",
        "dataset_sha256": dataset_sha,
        "model_name": ENCODER_NAME,
        "examples": len(examples),
        "init": "ms-marco-MiniLM-L6-v2",
        "not": "This run does not replace readiness-6f722792ffed and does not pass the readiness gates.",
    }, indent=2) + "\n")
    device = torch.device("mps")
    torch.manual_seed(17)
    model, tokenizer = load_encoder(device)
    groups = defaultdict(list)
    for example in examples:
        if example["split"] == "train":
            groups[len(example["candidates"])].append(example)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    steps = sum((len(rows) + 15) // 16 for rows in groups.values()) * EPOCHS
    from transformers import get_linear_schedule_with_warmup
    scheduler = get_linear_schedule_with_warmup(optimizer, max(1, steps // 10), max(steps, 1))
    select_rows = [example for example in examples if example["split"] == "select"]
    best = None
    for epoch in range(EPOCHS):
        losses = []
        seen = 0
        for rows in groups.values():
            loss = _epoch(model, tokenizer, optimizer, scheduler, rows, device, epoch, training_logits, pair_text, torch)
            losses.append(loss * len(rows))
            seen += len(rows)
        scores = _score_split(model, tokenizer, select_rows, device, pair_text)
        metrics = _metrics(select_rows, scores, temperature=1.0)
        print(
            f"epoch {epoch} loss {sum(losses) / seen:.4f} "
            f"select_accuracy {metrics['accuracy']:.3f} select_macro_f1 {metrics['macro_f1']:.3f}"
        )
        if best is None or metrics["macro_f1"] > best["macro_f1"] + 1e-12:
            best = {"macro_f1": metrics["macro_f1"], "accuracy": metrics["accuracy"], "epoch": epoch}
            torch.save({
                "model": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                "epoch": epoch,
                "dataset_sha256": dataset_sha,
                "task": "elements",
                "selection": best,
            }, run_dir / "best.pt")
    chosen = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(chosen["model"])
    model.to(device)
    calibrate_rows = [example for example in examples if example["split"] == "calibrate"]
    calibrate_scores = _score_split(model, tokenizer, calibrate_rows, device, pair_text)
    temperature = _fit_temperature(calibrate_rows, calibrate_scores) if calibrate_rows else 1.0
    chosen["temperature"] = temperature
    chosen["abstain_threshold"] = 0.0
    torch.save(chosen, run_dir / "best.pt")
    test_rows = [example for example in examples if example["split"] == "test"]
    test_scores = _score_split(model, tokenizer, test_rows, device, pair_text)
    report = _report(examples, test_rows, test_scores, temperature, run_dir, dataset_sha)
    report["old_checkpoint"] = _old_report(test_rows)
    _write_report(report)
    POINTER.parent.mkdir(parents=True, exist_ok=True)
    POINTER.write_text(str(run_dir.relative_to(ROOT)) + "\n")
    print(
        f"held-out accuracy {report['accuracy']:.3f} macro_f1 {report['macro_f1']:.3f} "
        f"majority {report['majority_accuracy']:.3f} always_abstain {report['always_abstain_accuracy']:.3f} "
        f"old {report['old_checkpoint']['accuracy']:.3f}"
    )
    print(f"trusted {report['trusted']} temperature {temperature:.4f}")
    return run_dir


def _report(examples, rows, scores, temperature, run_dir, dataset_sha) -> dict:
    metrics = _metrics(rows, scores, temperature)
    majority = _majority_label(examples)
    labels = _label_set(rows)
    majority_pred = [majority] * len(rows)
    abstain_pred = ["abstain"] * len(rows)
    truth = [row["label_key"] for row in rows]
    real = [index for index, row in enumerate(rows) if row["origin"] == "shard"]
    return {
        "task": "elements",
        "checkpoint": str(run_dir),
        "dataset_sha256": dataset_sha,
        "test_examples": len(rows),
        "real_shard_test_examples": len(real),
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "real_shard_accuracy": _accuracy([truth[index] for index in real], [metrics["predicted"][index] for index in real]),
        "majority_label": majority,
        "majority_accuracy": _accuracy(truth, majority_pred),
        "majority_macro_f1": macro_f1(truth, majority_pred, labels),
        "always_abstain_accuracy": _accuracy(truth, abstain_pred),
        "always_abstain_macro_f1": macro_f1(truth, abstain_pred, labels),
        "by_element": metrics["by_element"],
        "trusted": metrics["macro_f1"] >= macro_f1(truth, majority_pred, labels) + 0.10,
        "note": (
            "Held-out rows use gamma text and situation ids that are absent from train. "
            "The old readiness checkpoint was trained only on fire, adjust_aim, reload, wait, and unsupported. "
            "Always-abstain is its element-level behavior: it has no strategy label. "
            "This run does not pass the readiness gates."
        ),
    }


def _old_report(rows) -> dict:
    import torch

    from self_jev.model import load_encoder, pair_text
    from self_jev.train import _score_split

    blob = torch.load(READINESS / "best.pt", map_location="cpu", weights_only=False)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model, tokenizer = load_encoder(device)
    model.load_state_dict(blob["model"])
    model.eval()
    scores = _score_split(model, tokenizer, rows, device, pair_text)
    temperature = float(blob.get("temperature", 1.0))
    metrics = _metrics(rows, scores, temperature)
    return {
        "checkpoint": OLD_NAME,
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "predicted_counts": _counts(metrics["predicted"]),
    }


def _metrics(rows, scores, temperature: float) -> dict:
    truth, predicted = [], []
    for row, row_scores in zip(rows, scores):
        keys = [item["key"] for item in row["candidates"]]
        probabilities = tempered_softmax(row_scores, temperature)
        best = max(range(len(keys)), key=lambda index: (probabilities[index], -index))
        truth.append(row["label_key"])
        predicted.append(keys[best])
    labels = _label_set_from(truth, predicted)
    by_element = {}
    for element in sorted({row["element"] for row in rows}):
        pairs = [(label, pred) for row, label, pred in zip(rows, truth, predicted) if row["element"] == element]
        element_truth = [item[0] for item in pairs]
        element_pred = [item[1] for item in pairs]
        element_labels = _label_set_from(element_truth, element_pred)
        by_element[element] = {
            "n": len(pairs),
            "accuracy": _accuracy(element_truth, element_pred),
            "macro_f1": macro_f1(element_truth, element_pred, element_labels),
            "majority_accuracy": _accuracy(element_truth, [_mode(element_truth)] * len(pairs)),
        }
    return {
        "accuracy": _accuracy(truth, predicted),
        "macro_f1": macro_f1(truth, predicted, labels),
        "predicted": predicted,
        "by_element": by_element,
    }


def _majority_label(examples: list[dict]) -> str:
    train = [example["label_key"] for example in examples if example["split"] == "train"]
    return _mode(train)


def _mode(labels: list[str]) -> str:
    counts = _counts(labels)
    return max(counts, key=lambda key: (counts[key], key))


def _counts(labels: list[str]) -> dict:
    counts = defaultdict(int)
    for label in labels:
        counts[label] += 1
    return dict(counts)


def _label_set(rows) -> tuple[str, ...]:
    return _label_set_from([row["label_key"] for row in rows], [])


def _label_set_from(truth, predicted) -> tuple[str, ...]:
    return tuple(sorted(set(truth) | set(predicted)))


def _accuracy(truth, predicted) -> float:
    if not truth:
        return 0.0
    return sum(left == right for left, right in zip(truth, predicted)) / len(truth)


def _write_report(report: dict) -> None:
    path = ROOT / "reports" / "element_benchmark.json"
    serial = dict(report)
    path.write_text(json.dumps(serial, indent=2) + "\n")
    lines = [
        "# Self-JEV element benchmark",
        "",
        serial["note"],
        "",
        f"Checkpoint: `{serial['checkpoint']}`.",
        f"Test rows: {serial['test_examples']}. Real shard rows: {serial['real_shard_test_examples']}.",
        f"Accuracy: {serial['accuracy']:.3f}. Macro F1: {serial['macro_f1']:.3f}.",
        f"Majority baseline ({serial['majority_label']}): accuracy {serial['majority_accuracy']:.3f}, macro F1 {serial['majority_macro_f1']:.3f}.",
        f"Always abstain: accuracy {serial['always_abstain_accuracy']:.3f}, macro F1 {serial['always_abstain_macro_f1']:.3f}.",
        f"Old readiness checkpoint accuracy: {serial['old_checkpoint']['accuracy']:.3f}, macro F1 {serial['old_checkpoint']['macro_f1']:.3f}.",
        f"Beats the majority macro F1 by 0.10: {serial['trusted']}.",
        "",
    ]
    (ROOT / "reports" / "element_benchmark.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
