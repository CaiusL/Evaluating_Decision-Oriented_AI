"""Score the candidates that were supplied. Nothing is inserted."""
from __future__ import annotations

from pathlib import Path

from self_jev.metrics import tempered_softmax

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
_BUNDLE = {}


def predict(situation: str, question: str, candidates: list[dict], scorer=None, temperature: float | None = None) -> dict:
    """Return choice_id, probabilities, and scores for the given candidates."""
    if not isinstance(situation, str) or not situation.strip():
        raise ValueError("empty input")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("empty input")
    if not candidates:
        raise ValueError("empty input")
    seen = []
    for candidate in candidates:
        if not isinstance(candidate, dict) or "id" not in candidate or "description" not in candidate:
            raise ValueError("each candidate needs an id and a description")
        if candidate["id"] in seen:
            raise ValueError("duplicate ids")
        seen.append(candidate["id"])
    if scorer is None:
        bundle = load_bundle()
        scores = score_candidates(bundle, situation, question, candidates)
        if temperature is None:
            temperature = float(bundle["temperature"])
    else:
        scores = [float(score) for score in scorer(situation, question, candidates)]
        if temperature is None:
            temperature = 1.0
    if len(scores) != len(candidates):
        raise ValueError("scorer returned the wrong number of scores")
    probabilities = tempered_softmax(scores, temperature)
    best = max(range(len(probabilities)), key=lambda index: (probabilities[index], -index))
    return {
        "choice_id": candidates[best]["id"],
        "probabilities": {candidate["id"]: probability for candidate, probability in zip(candidates, probabilities)},
        "scores": {candidate["id"]: score for candidate, score in zip(candidates, scores)},
    }


def load_bundle(run_dir: Path | None = None) -> dict:
    if _BUNDLE and run_dir is None:
        return _BUNDLE
    import torch

    from self_jev.model import load_encoder, score_pairs

    directory = run_dir or latest_run()
    blob = torch.load(directory / "best.pt", map_location="cpu", weights_only=False)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model, tokenizer = load_encoder(device)
    model.load_state_dict(blob["model"])
    model.eval()
    bundle = {
        "model": model,
        "tokenizer": tokenizer,
        "device": device,
        "temperature": float(blob.get("temperature", 1.0)),
        "abstain_threshold": float(blob.get("abstain_threshold", 1.0)),
        "eligible": bool(blob.get("eligible", False)),
        "epoch": blob.get("epoch"),
        "score_pairs": score_pairs,
        "directory": directory,
    }
    if run_dir is None:
        _BUNDLE.clear()
        _BUNDLE.update(bundle)
    return bundle


def predict_many(examples: list[dict], temperature: float | None = None) -> list[dict]:
    """Batched form of predict. It scores only the candidates on each example."""
    bundle = load_bundle()
    if temperature is None:
        temperature = float(bundle["temperature"])
    from self_jev.model import pair_text

    results = []
    for start in range(0, len(examples), 16):
        batch = examples[start:start + 16]
        pairs = []
        widths = []
        for example in batch:
            _validate(example["situation"], example["question"], example["candidates"])
            widths.append(len(example["candidates"]))
            for candidate in example["candidates"]:
                pairs.append(pair_text(example["question"], example["situation"], candidate))
        values = bundle["score_pairs"](bundle["model"], bundle["tokenizer"], pairs, bundle["device"])
        cursor = 0
        for example, width in zip(batch, widths):
            scores = values[cursor:cursor + width]
            cursor += width
            probabilities = tempered_softmax(scores, temperature)
            best = max(range(width), key=lambda index: (probabilities[index], -index))
            candidates = example["candidates"]
            results.append({
                "choice_id": candidates[best]["id"],
                "probabilities": {candidate["id"]: probability for candidate, probability in zip(candidates, probabilities)},
                "scores": {candidate["id"]: score for candidate, score in zip(candidates, scores)},
            })
    return results


def _validate(situation: str, question: str, candidates: list[dict]) -> None:
    if not isinstance(situation, str) or not situation.strip():
        raise ValueError("empty input")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("empty input")
    if not candidates:
        raise ValueError("empty input")
    seen = []
    for candidate in candidates:
        if not isinstance(candidate, dict) or "id" not in candidate or "description" not in candidate:
            raise ValueError("each candidate needs an id and a description")
        if candidate["id"] in seen:
            raise ValueError("duplicate ids")
        seen.append(candidate["id"])


def score_candidates(bundle: dict, situation: str, question: str, candidates: list[dict]) -> list[float]:
    from self_jev.model import pair_text

    pairs = [pair_text(question, situation, candidate) for candidate in candidates]
    return bundle["score_pairs"](bundle["model"], bundle["tokenizer"], pairs, bundle["device"])


def latest_run() -> Path:
    pointer = RUNS / "latest"
    if pointer.exists():
        directory = Path(pointer.read_text().strip())
        if not directory.is_absolute():
            directory = ROOT / directory
        if (directory / "best.pt").exists():
            return directory
    runs = sorted(path for path in RUNS.glob("readiness-*") if (path / "best.pt").exists())
    if not runs:
        raise FileNotFoundError("Self-JEV has no trained checkpoint yet")
    return runs[-1]


def clear_bundle() -> None:
    _BUNDLE.clear()
