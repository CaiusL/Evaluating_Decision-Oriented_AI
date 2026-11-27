"""Choice metrics used for checkpoint selection and the benchmark."""
from __future__ import annotations

import math

from self_jev.readiness import CHOICES, UNSUPPORTED


def tempered_softmax(scores: list[float], temperature: float) -> list[float]:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    scaled = [score / temperature for score in scores]
    peak = max(scaled)
    weights = [math.exp(score - peak) for score in scaled]
    total = sum(weights)
    return [weight / total for weight in weights]


def macro_f1(truth: list[str], predicted: list[str], labels: tuple[str, ...] = CHOICES) -> float:
    scores = []
    for label in labels:
        tp = sum(left == label and right == label for left, right in zip(truth, predicted))
        fp = sum(left != label and right == label for left, right in zip(truth, predicted))
        fn = sum(left == label and right != label for left, right in zip(truth, predicted))
        if tp == 0:
            scores.append(0.0)
            continue
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        scores.append(2 * precision * recall / (precision + recall))
    return sum(scores) / len(scores) if scores else 0.0


def recall(truth: list[str], predicted: list[str], label: str) -> float:
    rows = [right for left, right in zip(truth, predicted) if left == label]
    if not rows:
        return 0.0
    return sum(item == label for item in rows) / len(rows)


def brier(probability_rows: list[dict[str, float]], truth: list[str]) -> float:
    total = 0.0
    count = 0
    for probabilities, label in zip(probability_rows, truth):
        if label not in CHOICES:
            continue
        for key, probability in probabilities.items():
            target = 1.0 if key == label else 0.0
            total += (probability - target) ** 2
        count += 1
    return total / count if count else 1.0


def expected_calibration_error(confidence: list[float], correct: list[bool], bins: int = 10) -> float:
    count = len(confidence)
    if count == 0:
        return 1.0
    error = 0.0
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        chosen = []
        for confidence_value, hit in zip(confidence, correct):
            if lower <= confidence_value < upper or (index == bins - 1 and confidence_value == 1):
                chosen.append((confidence_value, hit))
        if not chosen:
            continue
        average = sum(item[0] for item in chosen) / len(chosen)
        accuracy = sum(1.0 if item[1] else 0.0 for item in chosen) / len(chosen)
        error += len(chosen) / count * abs(accuracy - average)
    return error


def probabilities_for(scores: list[float], candidates: list[dict], temperature: float = 1.0) -> dict[str, float]:
    values = tempered_softmax(scores, temperature)
    return {candidate["key"]: value for candidate, value in zip(candidates, values)}


def predicted_key(scores: list[float], candidates: list[dict]) -> str:
    best = max(range(len(scores)), key=lambda index: scores[index])
    return candidates[best]["key"]


def unsupported_recall(truth: list[str], predicted: list[str]) -> float:
    return recall(truth, predicted, UNSUPPORTED)
