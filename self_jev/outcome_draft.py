"""Critique path for a trusted outcome checkpoint.

This scores the live car against one-control alternatives drawn from training
setups. It does not upload, and it does not write a racing zip. A checkpoint
that missed the predeclared bar keeps every live control.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from self_jev.bundle import read_strategy
from self_jev.metrics import tempered_softmax
from self_jev.outcome_data import DATA, READ, draft_prompt, live_setup, setup_key

ROOT = Path(__file__).resolve().parents[1]
POINTER = ROOT / "state" / "outcome_checkpoint.txt"
REPORT = ROOT / "reports" / "outcome_model.json"
LIVE = ROOT / "bases" / "live" / "racing-draft.zip"
DEST = ROOT / "reports" / "outcome_draft.json"
CHOICE_MARGIN = 0.05


def draft_outcome() -> dict:
    if not REPORT.exists():
        raise RuntimeError("train the outcome checkpoint before drafting")
    report = json.loads(REPORT.read_text())
    if not report.get("trusted"):
        critique = _refused(report, report.get("checkpoint"))
        DEST.write_text(json.dumps(critique, indent=2) + "\n")
        print("outcome draft trusted False; no control changed")
        return critique
    if not POINTER.exists():
        raise RuntimeError("a trusted outcome report has no state pointer")
    checkpoint = ROOT / POINTER.read_text().strip()
    if report.get("checkpoint") != str(checkpoint):
        raise RuntimeError("outcome report does not match state/outcome_checkpoint.txt")
    critique = _critique(report, checkpoint)
    DEST.write_text(json.dumps(critique, indent=2) + "\n")
    print(
        f"outcome draft trusted True changed {len(critique['changed'])} "
        f"fallbacks {sum(row['source'] == 'fallback' for row in critique['elements'])}"
    )
    return critique


def _refused(report: dict, checkpoint) -> dict:
    return {
        "game": "racing",
        "name": "outcome-draft",
        "submitted": False,
        "archive": None,
        "trusted": False,
        "checkpoint": checkpoint,
        "metric": report.get("metric"),
        "held_out_pairwise_accuracy": report.get("accuracy"),
        "unadapted_pairwise_accuracy": report.get("unadapted_pairwise_accuracy"),
        "elements": [],
        "changed": [],
        "why": "The checkpoint missed the predeclared bar against the unadapted encoder, so no control is changed.",
    }


def _critique(report: dict, checkpoint: Path) -> dict:
    examples = [
        json.loads(line)
        for line in (DATA / "examples.jsonl").read_text().splitlines()
        if line.strip()
    ]
    train_rows = [row for row in examples if row["split"] == "train" and not row["paraphrase"]]
    conditions = sorted({(row["track"], row["wet"]) for row in train_rows})
    setups = {}
    for row in train_rows:
        setups.update(row["setups"])
    live = live_setup(read_strategy(LIVE))
    temperature = float(report.get("temperature") or 1.0)
    scored = _score(checkpoint, live, setups, conditions, temperature)
    decisions = []
    for field in READ:
        row = scored[field]
        if row["source"] == "model":
            why = (
                f"On {', '.join(row['conditions'])} the outcome model prefers {field}={row['new']} "
                f"at probability {row['probability']:.3f}, margin {row['margin']:.3f}."
            )
        else:
            why = (
                f"No training-seen value of {field} beat the live value by 0.05 on every "
                f"training condition, so the live value stays."
            )
        decisions.append({**row, "why": why})
    return {
        "game": "racing",
        "name": "outcome-draft",
        "submitted": False,
        "archive": None,
        "trusted": True,
        "checkpoint": str(checkpoint),
        "metric": report.get("metric"),
        "held_out_pairwise_accuracy": report.get("accuracy"),
        "unadapted_pairwise_accuracy": report.get("unadapted_pairwise_accuracy"),
        "conditions": [f"{track} {wet}" for track, wet in conditions],
        "elements": decisions,
        "changed": [row["id"] for row in decisions if row["changed"]],
        "why": "Each change is one read control swapped onto the live car, preferred by the trusted checkpoint on every training condition.",
    }


def _score(checkpoint: Path, live: dict, setups: dict, conditions: list[tuple[str, str]], temperature: float) -> dict:
    import torch
    from self_jev.model import load_encoder, pair_text, score_pairs

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model, tokenizer = load_encoder(device)
    saved = torch.load(checkpoint / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(saved["model"])
    model.to(device)
    model.eval()
    values = {field: sorted({setup[field] for setup in setups.values()}) for field in READ}
    decisions = {}
    for field in READ:
        alternatives = [value for value in values[field] if value != live.get(field)]
        best = None
        for value in alternatives:
            modified = dict(live)
            modified[field] = value
            probabilities = []
            for track, wet in conditions:
                order = [live, modified]
                digest = hashlib.sha256(f"{field}|{value}|{track}|{wet}".encode()).digest()
                if digest[0] % 2:
                    order = [modified, live]
                example = draft_prompt(track, wet, order)
                pairs = [
                    pair_text(example["question"], example["situation"], candidate)
                    for candidate in example["candidates"]
                ]
                scores = tempered_softmax(score_pairs(model, tokenizer, pairs, device), temperature)
                by_key = {candidate["key"]: score for candidate, score in zip(example["candidates"], scores)}
                probabilities.append(by_key[setup_key(modified)])
            mean = sum(probabilities) / len(probabilities)
            margin = min(probability - (1.0 - probability) for probability in probabilities)
            if all(probability >= 0.5 + CHOICE_MARGIN / 2 for probability in probabilities) and margin >= CHOICE_MARGIN:
                if best is None or mean > best["probability"]:
                    best = {"new": value, "probability": mean, "margin": margin}
        if best is None:
            decisions[field] = {
                "id": field,
                "old": live.get(field),
                "new": live.get(field),
                "probability": None,
                "margin": None,
                "source": "fallback",
                "changed": False,
                "conditions": [f"{track} {wet}" for track, wet in conditions],
            }
        else:
            decisions[field] = {
                "id": field,
                "old": live.get(field),
                "new": best["new"],
                "probability": best["probability"],
                "margin": best["margin"],
                "source": "model",
                "changed": best["new"] != live.get(field),
                "conditions": [f"{track} {wet}" for track, wet in conditions],
            }
    return decisions
