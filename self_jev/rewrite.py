"""One Self-JEV step: score coach-base lever changes and keep the chosen one."""
from __future__ import annotations

import json
from pathlib import Path

from self_jev.bundle import read_strategy, rebuild, sha256
from self_jev.levers import changed_keys, lever_keys, levers, revise

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "runs" / "readiness-6f722792ffed"
QUESTION = "Which single strategy change should the next submission make?"


def _compact(value) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return str(value)


def situation_text(game: str, strategy: dict) -> str:
    fields = []
    for lever in levers(game):
        for key in lever_keys(game, lever):
            if key in strategy:
                fields.append(f"{key}={_compact(strategy[key])}")
    thesis = " ".join(str(strategy.get("thesis") or "").split())[:120]
    return (
        f"Coach base {strategy.get('name')} for {game}. "
        + ", ".join(fields)
        + f". Thesis: {thesis}"
    )


def candidates_for(game: str, strategy: dict) -> list[dict]:
    """Both legal directions of each lever. Identical results are kept once."""
    found = []
    seen = set()
    for lever in levers(game):
        for worse in (False, True):
            proposed = revise(game, strategy, lever, worse, 1, "coach-base")
            signature = tuple(_compact(proposed.get(key)) for key in lever_keys(game, lever))
            if signature in seen:
                continue
            seen.add(signature)
            bits = [
                f"{key} {_compact(strategy.get(key))} -> {_compact(proposed.get(key))}"
                for key in lever_keys(game, lever)
            ]
            direction = "after-weaker" if worse else "after-stronger"
            found.append({
                "id": f"{lever}-{direction}",
                "description": "Change only " + "; ".join(bits) + ".",
                "lever": lever,
                "worse": worse,
                "strategy": proposed,
            })
    if not found:
        raise RuntimeError(f"{game} produced no Self-JEV candidates")
    return found


def choose_rewrite(game: str, strategy: dict, predict_fn) -> dict:
    options = candidates_for(game, strategy)
    situation = situation_text(game, strategy)
    scored = predict_fn(
        situation,
        QUESTION,
        [{"id": item["id"], "description": item["description"]} for item in options],
    )
    chosen_id = scored["choice_id"]
    chosen = next(item for item in options if item["id"] == chosen_id)
    book = dict(chosen["strategy"])
    book["name"] = f"sj1-{chosen['lever']}"[:48]
    probability = float(scored["probabilities"][chosen_id])
    book["thesis"] = (
        f"Self-JEV chose {chosen['lever']} ({probability:.3f}) from coach base "
        f"{strategy.get('name')}. {chosen['description']}"
    )[:600]
    if changed_keys(strategy, book) != set(lever_keys(game, chosen["lever"])):
        raise RuntimeError(f"{game} rewrite changed unexpected keys")
    return {
        "game": game,
        "question": QUESTION,
        "situation": situation,
        "checkpoint": str(CHECKPOINT),
        "choice_id": chosen_id,
        "lever": chosen["lever"],
        "worse": chosen["worse"],
        "probabilities": scored["probabilities"],
        "scores": scored["scores"],
        "candidates": [{"id": item["id"], "description": item["description"]} for item in options],
        "strategy": book,
        "base_name": strategy.get("name"),
    }


_LOADED = {}


def model_predict(situation: str, question: str, candidates: list[dict]) -> dict:
    from self_jev.predict import load_bundle, predict, score_candidates

    if "bundle" not in _LOADED:
        _LOADED["bundle"] = load_bundle(CHECKPOINT)
    bundle = _LOADED["bundle"]

    def scorer(text, asked, options):
        return score_candidates(bundle, text, asked, options)

    return predict(situation, question, candidates, scorer=scorer, temperature=float(bundle["temperature"]))


def rewrite_archive(game: str, source: Path, dest: Path, predict_fn=None) -> dict:
    strategy = read_strategy(source)
    critique = choose_rewrite(game, strategy, predict_fn or model_predict)
    rebuild(source, game, critique["strategy"], dest)
    critique["source_archive"] = str(source)
    critique["archive"] = str(dest)
    critique["sha256"] = sha256(dest)
    critique.pop("strategy")
    return critique
