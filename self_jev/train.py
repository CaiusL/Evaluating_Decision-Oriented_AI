"""Fine-tune Self-JEV on the readiness groups."""
from __future__ import annotations

import json
import math
import random
from pathlib import Path

import numpy as np

from self_jev.dataset import DATA, load_dataset
from self_jev.hashing import sha256_file
from self_jev.metrics import brier, macro_f1, probabilities_for, predicted_key, tempered_softmax, unsupported_recall
from self_jev.readiness import CHOICES, LABELER_VERSION

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
SEED = 17
EPOCHS = 6
BATCH = 16
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
UNSUPPORTED_RECALL = 0.90
PATIENCE = 2


def resume_check(saved_sha: str, current_sha: str) -> None:
    if saved_sha != current_sha:
        raise RuntimeError("dataset hash mismatch")


def main() -> None:
    run_dir = train()
    print(run_dir)


def train() -> Path:
    import torch
    from transformers import get_linear_schedule_with_warmup

    from self_jev.model import ENCODER_NAME, load_encoder, pair_text, training_logits

    if not torch.backends.mps.is_available():
        raise RuntimeError("Self-JEV training requires MPS")
    device = torch.device("mps")
    _seed(torch)
    examples = load_dataset()
    dataset_path = DATA / "examples.jsonl"
    dataset_sha = sha256_file(dataset_path)
    groups = {}
    for example in examples:
        groups[example["group_id"]] = example["split"]
    run_id = f"readiness-{dataset_sha[:12]}"
    run_dir = RUNS / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "run_id": run_id,
        "dataset_sha256": dataset_sha,
        "labeler_version": LABELER_VERSION,
        "model_name": ENCODER_NAME,
        "groups": [{"id": key, "split": groups[key]} for key in sorted(groups)],
    }
    existing = run_dir / "manifest.json"
    if existing.exists():
        saved = json.loads(existing.read_text())
        resume_check(saved["dataset_sha256"], dataset_sha)
    else:
        existing.write_text(json.dumps(manifest, indent=2) + "\n")
    (RUNS / "latest").write_text(str(run_dir.relative_to(ROOT)) + "\n")
    model, tokenizer = load_encoder(device)
    _assert_lengths(tokenizer, examples, pair_text)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    train_rows = [example for example in examples if example["split"] == "train"]
    select_rows = [example for example in examples if example["split"] == "select"]
    calibrate_rows = [example for example in examples if example["split"] == "calibrate"]
    steps_per_epoch = math.ceil(len(train_rows) / BATCH)
    total_steps = steps_per_epoch * EPOCHS
    warmup = max(1, int(total_steps * 0.10))
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup, total_steps)
    start = 0
    stale = 0
    best = None
    fallback = None
    early_stop = False
    checkpoint_path = run_dir / "checkpoint.pt"
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        resume_check(checkpoint["dataset_sha256"], dataset_sha)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start = int(checkpoint["next_epoch"])
        stale = int(checkpoint.get("stale", 0))
        best = checkpoint.get("best")
        early_stop = bool(checkpoint.get("early_stop", False))
        _restore_rng(checkpoint.get("rng"), torch)
        if checkpoint.get("complete"):
            print(f"checkpoint already complete at {run_dir}")
            return run_dir
    log_path = run_dir / "train_log.jsonl"
    if early_stop:
        start = EPOCHS
    for epoch in range(start, EPOCHS):
        loss = _epoch(model, tokenizer, optimizer, scheduler, train_rows, device, epoch, training_logits, pair_text, torch)
        scores = _score_split(model, tokenizer, select_rows, device, pair_text)
        metrics = _selection_metrics(select_rows, scores)
        record = {"epoch": epoch, "loss": loss, **metrics}
        eligible = metrics["unsupported_recall"] >= UNSUPPORTED_RECALL
        if eligible and _better(metrics, best):
            best = {"f1": metrics["supported_macro_f1"], "brier": metrics["brier"], "epoch": epoch}
            _save_best(run_dir, model, best, dataset_sha, eligible=True)
            stale = 0
        elif best is not None:
            stale += 1
        if best is None and _better(metrics, fallback):
            fallback = {"f1": metrics["supported_macro_f1"], "brier": metrics["brier"], "epoch": epoch}
            _save_best(run_dir, model, fallback, dataset_sha, eligible=False)
        with log_path.open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        print(
            f"epoch {epoch} loss {loss:.4f} unsupported_recall {metrics['unsupported_recall']:.3f} "
            f"macro_f1 {metrics['supported_macro_f1']:.3f} brier {metrics['brier']:.3f} eligible {eligible}"
        )
        done = best is not None and stale >= PATIENCE
        early_stop = done
        _save_checkpoint(checkpoint_path, model, optimizer, scheduler, epoch, best, stale, dataset_sha, torch, complete=False, early_stop=done)
        if done:
            print(f"early stop at epoch {epoch}")
            break
    chosen = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(chosen["model"])
    model.to(device)
    calibrate_scores = _score_split(model, tokenizer, calibrate_rows, device, pair_text)
    temperature = _fit_temperature(calibrate_rows, calibrate_scores)
    threshold = _fit_threshold(calibrate_rows, calibrate_scores, temperature)
    chosen["temperature"] = temperature
    chosen["abstain_threshold"] = threshold
    torch.save(chosen, run_dir / "best.pt")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint["complete"] = True
    checkpoint["temperature"] = temperature
    checkpoint["abstain_threshold"] = threshold
    torch.save(checkpoint, checkpoint_path)
    print(f"temperature {temperature:.4f} abstain {threshold:.4f} eligible {chosen.get('eligible')}")
    return run_dir


def _better(metrics: dict, best: dict | None) -> bool:
    if best is None:
        return True
    if metrics["supported_macro_f1"] > best["f1"] + 1e-12:
        return True
    return abs(metrics["supported_macro_f1"] - best["f1"]) <= 1e-12 and metrics["brier"] < best["brier"] - 1e-12


def _epoch(model, tokenizer, optimizer, scheduler, rows, device, epoch, training_logits, pair_text, torch) -> float:
    model.train()
    order = np.random.default_rng(SEED + epoch).permutation(len(rows))
    total = 0.0
    seen = 0
    step = 0
    for start in range(0, len(order), BATCH):
        batch = [rows[int(index)] for index in order[start:start + BATCH]]
        pairs, targets = [], []
        width = len(batch[0]["candidates"])
        for example in batch:
            if len(example["candidates"]) != width:
                raise RuntimeError("training batch has mixed candidate counts")
            keys = [candidate["key"] for candidate in example["candidates"]]
            targets.append(keys.index(example["label_key"]))
            for candidate in example["candidates"]:
                pairs.append(pair_text(example["question"], example["situation"], candidate))
        optimizer.zero_grad(set_to_none=True)
        logits = training_logits(model, tokenizer, pairs, device).view(len(batch), width)
        target = torch.tensor(targets, device=device)
        loss = torch.nn.functional.cross_entropy(logits, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total += float(loss.detach()) * len(batch)
        seen += len(batch)
        step += 1
        if step % 50 == 0:
            print(f"  epoch {epoch} step {step} loss {total / seen:.4f}")
    return total / seen if seen else 0.0


def _score_split(model, tokenizer, rows, device, pair_text) -> list[list[float]]:
    from self_jev.model import score_pairs

    model.eval()
    scored = []
    for start in range(0, len(rows), BATCH):
        batch = rows[start:start + BATCH]
        pairs = []
        widths = []
        for example in batch:
            widths.append(len(example["candidates"]))
            for candidate in example["candidates"]:
                pairs.append(pair_text(example["question"], example["situation"], candidate))
        values = score_pairs(model, tokenizer, pairs, device)
        cursor = 0
        for width in widths:
            scored.append(values[cursor:cursor + width])
            cursor += width
    return scored


def _selection_metrics(rows: list[dict], scores: list[list[float]]) -> dict:
    truth, predicted, probability_rows = [], [], []
    for example, row_scores in zip(rows, scores):
        truth.append(example["label_key"])
        predicted.append(predicted_key(row_scores, example["candidates"]))
        probability_rows.append(probabilities_for(row_scores, example["candidates"]))
    supported_truth = [label for label in truth if label in CHOICES]
    supported_pred = [pred for label, pred in zip(truth, predicted) if label in CHOICES]
    supported_probs = [probs for label, probs in zip(truth, probability_rows) if label in CHOICES]
    return {
        "unsupported_recall": unsupported_recall(truth, predicted),
        "supported_macro_f1": macro_f1(supported_truth, supported_pred, CHOICES),
        "brier": brier(supported_probs, supported_truth),
    }


def _fit_temperature(rows: list[dict], scores: list[list[float]]) -> float:
    def loss(log_temperature: float) -> float:
        temperature = math.exp(log_temperature)
        total = 0.0
        for example, row_scores in zip(rows, scores):
            probabilities = tempered_softmax(row_scores, temperature)
            index = [candidate["key"] for candidate in example["candidates"]].index(example["label_key"])
            total -= math.log(max(probabilities[index], 1e-12))
        return total / len(rows)

    low, high = -3.0, 3.0
    for _ in range(40):
        left = low + (high - low) / 3
        right = high - (high - low) / 3
        if loss(left) < loss(right):
            high = right
        else:
            low = left
    return math.exp((low + high) / 2)


def _fit_threshold(rows: list[dict], scores: list[list[float]], temperature: float) -> float:
    peaks = []
    for example, row_scores in zip(rows, scores):
        if example["label_key"] not in CHOICES:
            continue
        peaks.append(max(tempered_softmax(row_scores, temperature)))
    if not peaks:
        return 1.0
    peaks.sort()
    return peaks[int(0.05 * len(peaks))]


def _assert_lengths(tokenizer, examples, pair_text) -> None:
    from self_jev.model import MAX_LENGTH

    for start in range(0, len(examples), 32):
        batch = examples[start:start + 32]
        left, right = [], []
        for example in batch:
            for candidate in example["candidates"]:
                first, second = pair_text(example["question"], example["situation"], candidate)
                left.append(first)
                right.append(second)
        encoded = tokenizer(left, right, padding=False, truncation=False)
        for token_ids in encoded["input_ids"]:
            if len(token_ids) > MAX_LENGTH:
                raise RuntimeError(f"a render is {len(token_ids)} tokens, above {MAX_LENGTH}")


def _save_best(run_dir: Path, model, summary: dict, dataset_sha: str, eligible: bool) -> None:
    import torch

    torch.save({
        "model": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "epoch": summary["epoch"],
        "eligible": eligible,
        "dataset_sha256": dataset_sha,
        "selection": summary,
    }, run_dir / "best.pt")


def _save_checkpoint(path, model, optimizer, scheduler, epoch, best, stale, dataset_sha, torch, complete, early_stop) -> None:
    torch.save({
        "model": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "next_epoch": epoch + 1,
        "step": (epoch + 1),
        "best": best,
        "stale": stale,
        "dataset_sha256": dataset_sha,
        "complete": complete,
        "early_stop": early_stop,
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
        },
    }, path)


def _restore_rng(state, torch) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])


def _seed(torch) -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if hasattr(torch, "mps") and hasattr(torch.mps, "manual_seed"):
        torch.mps.manual_seed(SEED)


if __name__ == "__main__":
    main()
