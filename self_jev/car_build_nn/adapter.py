"""Unified benchmark adapter for Car-Build NN matching scripts/serve_decision_api.py."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from self_jev.car_build_nn.model import CarBuildNumPy


class TaskSpecificModelScorer:
    """Drop-in Tier 4 scorer compatible with the M4 benchmark unified response contract."""

    def __init__(self, weights_path: Path | str = Path("models/car_build_nn.npz")) -> None:
        self.weights_path = Path(weights_path)
        self.engine = CarBuildNumPy(self.weights_path)

    def score(
        self,
        features: list[float],
        options: list[str] | None = None,
    ) -> dict[str, Any]:
        t0 = time.perf_counter()
        probs_all = self.engine.predict_probs(features)
        latency_ms = round((time.perf_counter() - t0) * 1000.0, 4)

        opt_list = options or self.engine.option_names
        if opt_list == self.engine.option_names:
            probs = probs_all.tolist()
        else:
            name_to_p = dict(zip(self.engine.option_names, probs_all.tolist()))
            raw = [float(name_to_p.get(opt, 0.0)) for opt in opt_list]
            s = sum(raw) or 1.0
            probs = [p / s for p in raw]

        best_idx = max(range(len(probs)), key=lambda i: probs[i])
        return {
            "model": "car-build-nn",
            "model_id": str(self.weights_path),
            "backend": "numpy-cpu",
            "choice_index": best_idx,
            "choice": opt_list[best_idx],
            "probabilities": [round(float(p), 6) for p in probs],
            "confidence": round(float(probs[best_idx]), 6),
            "parse_ok": True,
            "latency_ms": latency_ms,
            "input_tokens": len(features),
            "state_truncated": False,
            "raw_output": None,
        }

    def score_record(self, record: dict[str, Any]) -> dict[str, Any]:
        return self.score(record["features"], record.get("options"))

    @staticmethod
    def should_escalate(
        result: dict[str, Any],
        conf_threshold: float = 0.70,
        margin_threshold: float = 0.05,
    ) -> bool:
        """Confidence-gated cascade rule: escalate to Kev-0.8B or Frontier LLM when uncertain."""
        probs = sorted(result["probabilities"], reverse=True)
        top1 = probs[0] if probs else 0.0
        top2 = probs[1] if len(probs) > 1 else 0.0
        return (top1 < conf_threshold) or ((top1 - top2) < margin_threshold)
