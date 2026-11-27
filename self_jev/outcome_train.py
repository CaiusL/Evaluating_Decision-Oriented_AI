"""Fine-tune the existing MiniLM encoder on outcome-linked racing comparisons.

The accept rule is fixed here, and written into the run manifest, before the
test split is scored.
"""
from __future__ import annotations

import json
from pathlib import Path

from self_jev.hashing import sha256_file
from self_jev.metrics import predicted_key, tempered_softmax
from self_jev.model import ENCODER_NAME
from self_jev.outcome_data import SPLIT_RULE, build_examples, write_dataset

ROOT = Path(__file__).resolve().parents[1]
POINTER = ROOT / "state" / "outcome_checkpoint.txt"
REPORT = ROOT / "reports" / "outcome_model.json"
BRIEF = ROOT / "reports" / "outcome_brief.md"

# Predeclared before any test score. The comparison is the unadapted encoder.
# The previous unadapted accuracy on these 69 pairs was 0.348, so 0.10 above it is 0.448.
METRIC = "pairwise_accuracy"
MARGIN = 0.10
MIN_TEST_COMPARISONS = 69
MIN_TRAIN_ACCURACY = 0.75
HELD_OUT_FLOOR = 0.448
TRUST_RULE = (
    "trusted if and only if the selected checkpoint fits the training pairs "
    "(train pairwise accuracy at least 0.75) and its held-out pairwise accuracy "
    "on these 69 comparisons is at least 0.10 above the unadapted MiniLM encoder "
    "on the same pairs. The previously measured unadapted accuracy is 0.348, so "
    "the held-out bar is 0.448. A train accuracy still near 0.5 is not trusted. "
    "The state pointer moves only when this holds."
)
EPOCHS = 8
BATCH = 8
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.0


def _outcome_job() -> dict:
    write_dataset()
    examples, stats = build_examples()
    return {
        "examples": examples,
        "stats": stats,
        "dataset_path": ROOT / "data" / "outcome" / "examples.jsonl",
        "task": "racing-outcome",
        "run_prefix": "outcome",
        "report_path": REPORT,
        "pointer": POINTER,
        "trust_mode": "overall",
        "cause": (
            "The two full setup strings differ by a few fields inside about 170 tokens, "
            "so the winner-minus-loser logit gap stayed near 0 at learning rate 2e-5. "
            "Changed fields are now prefixed, both candidates share a neutral id, "
            "and the learning rate is 5e-5 with no weight decay."
        ),
        "publish_outcome": True,
        "trust_rule": TRUST_RULE,
        "min_test_comparisons": MIN_TEST_COMPARISONS,
        "held_out_floor": HELD_OUT_FLOOR,
        "held_out_bar": "at least 0.448 and at least 0.10 above the unadapted encoder on these 69 comparisons",
        "not": "Does not replace readiness, elements, racing-pace, outcome-9713cfb87f79, or outcome-f36348cf67b3, and does not update runs/latest.",
    }


def train(prepared: dict | None = None) -> Path:
    if prepared is None:
        from self_jev.pace_outcome import refuse_flat_outcome

        refuse_flat_outcome()
        prepared = _outcome_job()
    import torch

    from self_jev.model import load_encoder, pair_text, training_logits
    from self_jev.train import _fit_temperature, _score_split

    if not torch.backends.mps.is_available():
        raise RuntimeError("Self-JEV training requires MPS")
    examples = prepared["examples"]
    stats = prepared["stats"]
    dataset_path = prepared["dataset_path"]
    dataset_sha = sha256_file(dataset_path)
    train_rows = [row for row in examples if row["split"] == "train"]
    select_rows = [row for row in examples if row["split"] == "select"]
    calibrate_rows = [row for row in examples if row["split"] == "calibrate"]
    test_rows = [row for row in examples if row["split"] == "test" and not row["paraphrase"]]
    if prepared["trust_mode"] == "cap":
        test_rows = [row for row in test_rows if row.get("differs_on_cap")]
    run_dir = ROOT / "runs" / f"{prepared['run_prefix']}-{dataset_sha[:12]}"
    if run_dir.name in {"outcome-9713cfb87f79", "outcome-f36348cf67b3"}:
        raise RuntimeError(f"refusing to overwrite {run_dir.name}")
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "task": prepared["task"],
        "dataset_sha256": dataset_sha,
        "metric": METRIC,
        "margin": MARGIN,
        "min_test_comparisons": prepared["min_test_comparisons"],
        "min_train_accuracy": MIN_TRAIN_ACCURACY,
        "held_out_floor": prepared["held_out_floor"],
        "held_out_bar": prepared["held_out_bar"],
        "trust_rule": prepared["trust_rule"],
        "selection": "train pairwise accuracy, then lower training loss",
        "cause": prepared["cause"],
        "split_unit": "match_component",
        "split_rule": SPLIT_RULE,
        "init": ENCODER_NAME,
        "local_encoder": "models/ms-marco-MiniLM-L6-v2",
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "seed": 17,
        "stats": stats,
        "not": prepared["not"],
    }
    # The trust rule is on disk before any test forward pass.
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    device = torch.device("mps")
    torch.manual_seed(17)
    model, tokenizer = load_encoder(device)
    model.config._attn_implementation = "eager"
    _assert_lengths(tokenizer, examples, pair_text)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    history = []
    best = None
    train_base = [row for row in train_rows if not row["paraphrase"]]
    for epoch in range(EPOCHS):
        loss = _epoch(model, tokenizer, optimizer, train_rows, device, epoch, training_logits, pair_text, torch)
        train_scores = _score_split(model, tokenizer, train_base, device, pair_text)
        train_metrics = _classify(train_base, train_scores, temperature=1.0)
        select_scores = _score_split(model, tokenizer, select_rows, device, pair_text) if select_rows else []
        select_metrics = _classify(select_rows, select_scores, temperature=1.0) if select_rows else {"accuracy": -1.0}
        record = {
            "epoch": epoch,
            "loss": loss,
            "train_pairwise_accuracy": train_metrics["accuracy"],
            "select_pairwise_accuracy": select_metrics["accuracy"],
        }
        history.append(record)
        print(
            f"epoch {epoch} loss {loss:.4f} "
            f"train_pairwise_accuracy {train_metrics['accuracy']:.3f} "
            f"select_pairwise_accuracy {select_metrics['accuracy']:.3f}"
        )
        if _better(train_metrics["accuracy"], loss, best):
            best = {**train_metrics, "epoch": epoch, "loss": loss}
            _save(run_dir, model, epoch, dataset_sha, prepared["task"])
    bundle = _load(run_dir, device)
    if calibrate_rows:
        calibrate_scores = _score_split(bundle["model"], bundle["tokenizer"], calibrate_rows, device, pair_text)
        temperature = _fit_temperature(calibrate_rows, calibrate_scores)
    else:
        temperature = 1.0
    (run_dir / "temperature.json").write_text(json.dumps({"temperature": temperature}) + "\n")
    (run_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    saved_train_scores = _score_split(bundle["model"], bundle["tokenizer"], train_base, device, pair_text)
    saved_train = _classify(train_base, saved_train_scores, temperature=1.0)
    manifest["selected_epoch"] = None if best is None else best["epoch"]
    manifest["train_pairwise_accuracy"] = saved_train["accuracy"]
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    # Test scoring starts only after the manifest records the train fit and the accept bar.
    test_scores = _score_split(bundle["model"], bundle["tokenizer"], test_rows, device, pair_text)
    report = _classify(test_rows, test_scores, temperature)
    fresh, fresh_tokenizer = load_encoder(device)
    fresh.eval()
    unadapted_scores = _score_split(fresh, fresh_tokenizer, test_rows, device, pair_text)
    unadapted = _classify(test_rows, unadapted_scores, temperature=1.0)
    report["unadapted_pairwise_accuracy"] = unadapted["accuracy"]
    report["train_pairwise_accuracy"] = saved_train["accuracy"]
    report["test_comparisons"] = len(test_rows)
    above_unadapted = report["accuracy"] >= unadapted["accuracy"] + MARGIN
    if prepared["trust_mode"] == "cap":
        report["margin_cleared"] = above_unadapted and len(test_rows) >= 1
        report["gate"] = "held-out pairs that differ on pace_source or the speed cap"
    else:
        report["margin_cleared"] = above_unadapted and report["accuracy"] >= HELD_OUT_FLOOR
    report["held_out_floor"] = prepared["held_out_floor"]
    report["count_cleared"] = len(test_rows) >= prepared["min_test_comparisons"]
    report["fit_cleared"] = saved_train["accuracy"] >= MIN_TRAIN_ACCURACY
    report["trusted"] = bool(report["margin_cleared"] and report["count_cleared"] and report["fit_cleared"])
    report["temperature"] = temperature
    report["checkpoint"] = str(run_dir)
    report["dataset_sha256"] = dataset_sha
    report["metric"] = METRIC
    report["trust_rule"] = prepared["trust_rule"]
    report["split_unit"] = "match_component"
    report["select_epoch"] = None if best is None else best["epoch"]
    pointer = prepared["pointer"]
    if report["trusted"]:
        pointer.parent.mkdir(parents=True, exist_ok=True)
        pointer.write_text(str(run_dir.relative_to(ROOT)) + "\n")
        report["pointer_updated"] = True
    else:
        report["pointer_updated"] = False
    (run_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    report_path = prepared["report_path"]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    if prepared["publish_outcome"]:
        if report["trusted"]:
            from self_jev.outcome_draft import draft_outcome
            draft_outcome()
        else:
            _refuse_draft(report)
        _write_brief(report, stats)
    print(
        f"held-out pairwise_accuracy {report['accuracy']:.3f} "
        f"unadapted {report['unadapted_pairwise_accuracy']:.3f} "
        f"trusted {report['trusted']}"
    )
    print(run_dir)
    return run_dir


def _better(accuracy: float, loss: float, best: dict | None) -> bool:
    if best is None:
        return True
    if accuracy > best["accuracy"] + 1e-12:
        return True
    return abs(accuracy - best["accuracy"]) <= 1e-12 and loss < best["loss"] - 1e-12


def _refuse_draft(report: dict) -> None:
    payload = {
        "game": "racing",
        "name": "outcome-draft",
        "submitted": False,
        "archive": None,
        "trusted": False,
        "checkpoint": report["checkpoint"],
        "metric": METRIC,
        "held_out_pairwise_accuracy": report["accuracy"],
        "unadapted_pairwise_accuracy": report["unadapted_pairwise_accuracy"],
        "elements": [],
        "changed": [],
        "why": "The checkpoint missed the predeclared bar against the unadapted encoder, so no control is changed.",
    }
    (ROOT / "reports" / "outcome_draft.json").write_text(json.dumps(payload, indent=2) + "\n")


def _classify(rows: list[dict], scores: list[list[float]], temperature: float) -> dict:
    truth, predicted = [], []
    details = []
    for example, row_scores in zip(rows, scores):
        probabilities = tempered_softmax(row_scores, temperature)
        choice = predicted_key(probabilities, example["candidates"])
        truth.append(example["label_key"])
        predicted.append(choice)
        order = sorted(range(len(probabilities)), key=lambda index: probabilities[index], reverse=True)
        margin = probabilities[order[0]] - probabilities[order[1]] if len(order) > 1 else probabilities[order[0]]
        details.append({
            "situation_id": example["situation_id"],
            "truth": example["label_key"],
            "predicted": choice,
            "margin": margin,
            "match_ids": example["match_ids"],
        })
    correct = sum(left == right for left, right in zip(truth, predicted))
    abstained = sum(1 for row in details if row["margin"] < 0.05)
    return {
        "accuracy": correct / len(truth) if truth else 0.0,
        "predicted": predicted,
        "truth": truth,
        "abstained": abstained,
        "coverage": (len(details) - abstained) / len(details) if details else 0.0,
        "rows": details,
    }


def _epoch(model, tokenizer, optimizer, rows, device, epoch, training_logits, pair_text, torch) -> float:
    import numpy as np

    model.train()
    order = np.random.default_rng(17 + epoch).permutation(len(rows))
    total = 0.0
    seen = 0
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
        total += float(loss.detach()) * len(batch)
        seen += len(batch)
    return total / seen if seen else 0.0


def _save(run_dir: Path, model, epoch: int, dataset_sha: str, task: str = "racing-outcome") -> None:
    import torch

    torch.save({
        "model": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "epoch": epoch,
        "dataset_sha256": dataset_sha,
        "task": task,
        "init": ENCODER_NAME,
    }, run_dir / "best.pt")


def _load(run_dir: Path, device):
    import torch
    from self_jev.model import load_encoder

    model, tokenizer = load_encoder(device)
    saved = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(saved["model"])
    model.to(device)
    model.eval()
    return {"model": model, "tokenizer": tokenizer, "dir": run_dir}


def _assert_lengths(tokenizer, examples, pair_text) -> None:
    from self_jev.model import MAX_LENGTH

    for example in examples:
        for candidate in example["candidates"]:
            first, second = pair_text(example["question"], example["situation"], candidate)
            encoded = tokenizer(first, second, truncation=False)
            if len(encoded["input_ids"]) > MAX_LENGTH:
                raise RuntimeError(f"a render is {len(encoded['input_ids'])} tokens, above {MAX_LENGTH}")


def _write_brief(report: dict, stats: dict) -> None:
    trusted = "yes" if report["trusted"] else "no"
    dropped = stats["dropped"]
    origins = stats["label_origins"]
    text = f"""# Outcome fine-tune

## What the brief asked

The imported brief says to keep the existing encoder and to supervise it with recorded outcomes:

> Continue by fine-tuning an open-source encoder. A focused racing model is achievable on the existing Mac.
>
> The project already fine-tunes `cross-encoder/ms-marco-MiniLM-L6-v2` through PyTorch. Continue with this backbone for the first experiment. It scores a question and situation against each candidate description. Train the encoder and scorer using measured preferences between racing options.
>
> Derive training preferences from recorded outcomes. Keep ambiguous comparisons explicitly unresolved.
>
> Apply cross-entropy over candidate scores for clear winning choices.

The import README names the outcome fields: "Use observed_result and recording_facts as outcome supervision." Opponent finishes are not labels. The 16 handwritten table/grip sentences are not this dataset.

## Import inventory

- Completed matches: {stats["matches"]}
- Strategy bundles: {stats["bundles"]}
- Matches with a recorded bundle link: {stats["bundle_links"]}
- Code identity linked: {stats["linked_identity"]}; unresolved: {stats["unresolved_identity"]}
- Played strategies: {stats["played_strategies"]}
- `observed_result.rank` present: {stats["rank_fields"]}
- `recording_facts.finish_time_s` present: {stats["finish_time_fields"]}
- `recording_facts.completed_laps` present: {stats["completed_lap_fields"]}
- Field-car rows in recordings: {stats["recording_car_rows"]} (not used as labels)

## Examples

A pair is two of our matches on the same track and the same recorded wet/dry band, with different played setups. The label is the setup with the better outcome. {stats["outcome_rule"]}

Label origins on kept pairs: {origins}.

Dropped among same-track comparisons: wet band differed or was unknown {dropped["wet_or_track"]}; identical setup {dropped["identical_setup"]}; equal outcome {dropped["equal_outcome"]}; the same two setups won in both directions {dropped["contradictory_direction"]}.

Kept comparisons: {stats["decisive_pairs"]}. Rendered examples, including one train paraphrase each: {stats["examples"]}.

## Split

Split unit: `match_component`.

{stats["split_rule"]}

Components: {stats["components"]}. Counts excluding paraphrases: train {stats["split_counts"]["train"]}, select {stats["split_counts"]["select"]}, calibrate {stats["split_counts"]["calibrate"]}, test {stats["split_counts"]["test"]}. No test match appears in train.

## Accept rule, written before the test score

{TRUST_RULE}

## Why the previous loss stayed at ln(2)

`runs/outcome-9713cfb87f79` trained on the full setup string. Pair `train-04722bed-646d6ba0` differs only in `fuel_load_kg=9` versus `fuel_load_kg=11`, but the first token difference was the random option id (`##65`/`##18` versus `##37`/`##17`) and only then the tokens `9` versus `11`. The two `[CLS]` vectors had cosine 0.999984. The first-batch gradient norm was 6.37 and was clipped to 1.0. All 105 parameter tensors required gradients, a scalar score per setup fitted the training labels at accuracy 1.000, and the target index was the faster setup. Six epochs at learning rate 2e-5 left the winner-minus-loser logit gap near 0.000, so the loss stayed near 0.693. The encoder was not frozen, the knob was in the text, and the labels were not contradictory. The difference was too small, and the step size never opened a gap.

The change prefixes each candidate with the fields that differ, gives both candidates the same neutral id, and trains at learning rate 5e-5 with weight decay 0.

## Result

Checkpoint: `{report["checkpoint"]}`

Train pairwise accuracy: {report["train_pairwise_accuracy"]:.3f}
Held-out pairwise accuracy: {report["accuracy"]:.3f}
Unadapted MiniLM pairwise accuracy: {report["unadapted_pairwise_accuracy"]:.3f}
Test comparisons: {report["test_comparisons"]}
Train fit cleared: {report["fit_cleared"]}
Margin cleared: {report["margin_cleared"]}
Count cleared: {report["count_cleared"]}
Trusted: {trusted}
Selected epoch: {report["select_epoch"]}
Pointer updated: {report["pointer_updated"]}

This score is a preference between recorded cars. It is not a first-place race result.

## Reproduce

```bash
python -m self_jev train-outcome
```
"""
    BRIEF.write_text(text)


def main() -> None:
    train()
