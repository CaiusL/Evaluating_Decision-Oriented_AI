"""Multi-element draft.

`draft` asks the element checkpoint once per element. A non-abstain choice with
a clear margin is source `model`. Abstain, a margin under 0.05, an untrained
element, or a choice the agent cannot execute is source `fallback` and keeps
the current value.
"""
from __future__ import annotations

import ast
import copy
import io
import json
import re
import zipfile
from pathlib import Path

from self_jev.bundle import design_from_strategy, read_strategy, render_strategy, sha256
from self_jev.element_data import draft_cases

ROOT = Path(__file__).resolve().parents[1]
MARGIN = 0.05
LIVE = {
    "fps": ROOT / "bases" / "live" / "fps-draft.zip",
    "racing": ROOT / "bases" / "live" / "racing-draft.zip",
}


def audit(game: str, strategy: dict, agent: str) -> list[dict]:
    if game == "fps":
        return _audit_fps(strategy, agent)
    if game == "racing":
        return _audit_racing(strategy, agent)
    raise ValueError(game)


def _audit_fps(strategy: dict, agent: str) -> list[dict]:
    read = set(re.findall(r'STRATEGY\.get\("([^"]+)"', agent))
    return [
        {"id": "site", "old": _current_value("site", strategy), "read": "default_site" in read and "expect_site" in read},
        {"id": "buy", "old": _current_value("buy", strategy), "read": "buy_style" in read},
        {"id": "steer_speed", "old": 1.0, "read": "def steer_to(me, point, look=None, speed=1.0):" in agent},
        {"id": "burst", "old": 9, "read": 'MEM["burst_until"] = tick + 9' in agent},
        {"id": "settle", "old": 0.5, "read": 'math.hypot(me.get("vx", 0), me.get("vy", 0)) < .5' in agent and "speed < 0.5" in agent},
        {"id": "peek_trigger", "old": 6.0, "read": 'PEEK["near"]' in agent and 'PEEK["turn"]' in agent},
        {"id": "peek_cadence", "old": "4/16", "read": "cycle = 16" in agent and "phase < 4" in agent},
        {"id": "escape", "old": "50/25", "read": 'MEM["escape_until"] = tick + 50' in agent},
        {"id": "enemy_memory", "old": "160/90/40", "read": '< 160' in agent and '< 90' in agent and '< 40' in agent},
    ]


def _audit_racing(strategy: dict, agent: str, driver: str = "") -> list[dict]:
    control = _tuple_body(agent, "_CONTROL")
    text = agent + "\n" + driver
    return [
        {"id": "margins", "old": _current_value("margins", strategy), "read": "qual_margin" in control and "race_margin" in control and "_margin" in text},
        {"id": "preview", "old": strategy.get("preview"), "read": "preview" in control and "preview" in text},
        {"id": "latency", "old": strategy.get("latency"), "read": "latency" in control},
        {"id": "lookahead", "old": _current_value("lookahead", strategy), "read": "ld_min" in control and "ld_time" in control},
        {"id": "yaw_gain", "old": strategy.get("yaw_gain"), "read": "yaw_gain" in control},
        {"id": "kv", "old": strategy.get("kv"), "read": "kv" in control},
        {"id": "brake_gain", "old": strategy.get("brake_gain"), "read": "brake_gain" in control},
        {"id": "stops", "old": strategy.get("stops"), "read": 'book["stops"]' in agent},
        {"id": "front_slide_steer", "old": 0.75, "read": "rate *= 0.75" in driver},
        {"id": "front_slide_brake", "old": 0.75, "read": "brake *= .75" in driver},
    ]


def decide(game: str, strategy: dict, agent: str, predict_fn, evidence: dict | None = None) -> dict:
    _ = agent
    cases = draft_cases(game, evidence)
    elements = [_interpret(case, predict_fn(case["situation"], case["question"], case["candidates"]), strategy) for case in cases]
    return {
        "game": game,
        "name": "sj3-model",
        "model": {
            "task": "elements",
            "margin": MARGIN,
            "note": (
                "Each element is one question to the element checkpoint. "
                "source model means that checkpoint picked a value. "
                "source fallback means it abstained, the top two probabilities "
                f"were within {MARGIN:.2f}, or the agent cannot execute the choice."
            ),
        },
        "elements": elements,
    }


def _interpret(case: dict, scored: dict, strategy: dict) -> dict:
    ranked = sorted(scored["probabilities"].items(), key=lambda item: item[1], reverse=True)
    margin = ranked[0][1] - ranked[1][1] if len(ranked) > 1 else ranked[0][1]
    chosen = next(item for item in case["candidates"] if item["id"] == scored["choice_id"])
    key = chosen["key"]
    by_key = {item["key"]: scored["probabilities"][item["id"]] for item in case["candidates"]}
    old = _current_value(case["element"], strategy)
    probability = scored["probabilities"][scored["choice_id"]]
    base = {
        "id": case["element"],
        "old": old,
        "probability": probability,
        "margin": margin,
        "choice": key,
        "probabilities": by_key,
    }
    if key == "abstain" or margin < MARGIN:
        why = f"Element model top key {key} at {probability:.3f}, margin {margin:.3f}."
        if key == "abstain":
            why += " Abstain keeps the current value."
        else:
            why += " The margin is below 0.05, so this is a fallback."
        return {**base, "new": old, "changed": False, "source": "fallback", "why": why}
    try:
        new = _decode(case["element"], key)
    except KeyError:
        return {**base, "new": old, "changed": False, "source": "fallback",
                "why": f"Element model chose {key} at {probability:.3f}, and this agent cannot execute that choice."}
    changed = new != old
    return {**base, "new": new, "changed": changed, "source": "model",
            "why": f"Element model chose {key} at {probability:.3f}, margin {margin:.3f}, on the {case['kind']}."}


def _current_value(element: str, strategy: dict):
    if element == "site":
        attack, defend = strategy.get("default_site"), strategy.get("expect_site")
        return attack if attack == defend else {"default_site": attack, "expect_site": defend}
    if element == "buy":
        return "armor_pistol" if strategy.get("buy_style") == "armor_pistol" else "rifle"
    if element == "steer_speed":
        return 1.0
    if element == "burst":
        return 9
    if element == "settle":
        return 0.5
    if element == "peek_trigger":
        return 6.0
    if element == "peek_cadence":
        return "4/16"
    if element == "escape":
        return "50/25"
    if element == "enemy_memory":
        return "160/90/40"
    if element == "margins":
        return {"qual_margin": strategy.get("qual_margin"), "race_margin": strategy.get("race_margin")}
    if element == "lookahead":
        return {"ld_min": strategy.get("ld_min"), "ld_time": strategy.get("ld_time")}
    if element == "stops":
        return strategy.get("stops")
    if element == "front_slide_steer":
        return 0.75
    if element == "front_slide_brake":
        return 0.75
    return strategy.get(element)


def _decode(element: str, key: str):
    table = {
        "steer_speed": {"1.0": 1.0, "0.85": 0.85},
        "burst": {"4": 4, "9": 9},
        "settle": {"0.5": 0.5, "0.6": 0.6},
        "peek_trigger": {"6": 6.0, "9": 9.0},
        "margins": {
            "1.0": {"qual_margin": 1.0, "race_margin": 1.0},
            "0.9": {"qual_margin": 0.9, "race_margin": 0.9},
        },
        "lookahead": {
            "7/0.35": {"ld_min": 7, "ld_time": 0.35},
            "8/0.40": {"ld_min": 8, "ld_time": 0.40},
        },
        "stops": {
            "lap1-medium-180": [{"lap": 1, "compound": "medium", "request_ahead_m": 180}],
            "lap2-soft-160": [{"lap": 2, "compound": "soft", "request_ahead_m": 160}],
        },
        "preview": {"0.18": 0.18, "0.35": 0.35},
        "latency": {"0.05": 0.05, "0.08": 0.08},
        "yaw_gain": {"0.7": 0.7, "0.8": 0.8},
        "kv": {"1.9": 1.9, "1.6": 1.6},
        "brake_gain": {"1.04": 1.04, "1.08": 1.08},
        "front_slide_steer": {"0.75": 0.75, "0.5": 0.5},
        "front_slide_brake": {"0.75": 0.75, "0.6": 0.6},
    }
    if element in table:
        return table[element][key]
    return key


def apply_book(strategy: dict, decision: dict) -> dict:
    book = copy.deepcopy(strategy)
    for item in decision["elements"]:
        if not item["changed"]:
            continue
        key = item["id"]
        new = item["new"]
        if key == "site":
            book["default_site"] = new
            book["expect_site"] = new
        elif key == "buy":
            book["buy_style"] = "armor_pistol" if new == "armor_pistol" else "full_rifle"
        elif key == "margins":
            book.update(new)
        elif key == "lookahead":
            book["ld_min"] = new["ld_min"]
            book["ld_time"] = new["ld_time"]
        elif key == "stops":
            book["stops"] = new
        elif key in book:
            book[key] = new
    book["name"] = decision["name"]
    moved = [item for item in decision["elements"] if item["changed"]]
    bits = [f"{item['id']} {_shown(item['old'])} -> {_shown(item['new'])}" for item in moved]
    book["thesis"] = (
        f"Self-JEV changed {len(moved)} elements from {strategy.get('name')}: "
        + "; ".join(bits)
        + "."
    )[:600]
    return book


def patch_code(game: str, agent: str, driver: str, elements: list[dict]) -> tuple[str, str]:
    changed = {item["id"]: item for item in elements if item["changed"]}
    if game == "fps":
        return _patch_fps(agent, changed), driver
    if game == "racing":
        return agent, _patch_driver(driver, changed)
    raise ValueError(game)


def _patch_fps(agent: str, changed: dict) -> str:
    if "burst" in changed:
        agent = _once(agent, 'MEM["burst_until"] = tick + 9', f'MEM["burst_until"] = tick + {int(changed["burst"]["new"])}')
    if "settle" in changed:
        gate = changed["settle"]["new"]
        agent = _once(agent, 'math.hypot(me.get("vx", 0), me.get("vy", 0)) < .5',
                      f'math.hypot(me.get("vx", 0), me.get("vy", 0)) < {gate}')
        agent = _once(agent, "if speed < 0.5:", f"if speed < {gate}:")
    if "peek_trigger" in changed:
        agent = _once(agent, '"near": 6.0', f'"near": {changed["peek_trigger"]["new"]}')
    if "peek_cadence" in changed:
        agent = _once(agent, "cycle = 16", "cycle = 24")
        agent = _once(agent, 'phase = peek["tick"] % 8', 'phase = peek["tick"] % 16')
        agent = _once(agent, "if phase < 4:", "if phase < 8:")
    if "escape" in changed:
        agent = _once(agent, 'MEM["escape_until"] = tick + 50', 'MEM["escape_until"] = tick + 25')
        agent = _once(agent, 'MEM["escape_flip_tick"] = tick + 25', 'MEM["escape_flip_tick"] = tick + 12')
        agent = _once(agent, 'MEM["escape_until"] = tick + 25', 'MEM["escape_until"] = tick + 12')
    if "enemy_memory" in changed:
        agent = _once(agent, 'last_enemy_tick", 0) < 160', 'last_enemy_tick", 0) < 80')
        agent = _once(agent, 'last_enemy_tick", 0) < 90', 'last_enemy_tick", 0) < 45')
        agent = _once(agent, 'last_enemy_tick", 0) < 40', 'last_enemy_tick", 0) < 20')
    if "steer_speed" in changed:
        agent = _once(agent, "def steer_to(me, point, look=None, speed=1.0):",
                      f'def steer_to(me, point, look=None, speed={changed["steer_speed"]["new"]}):')
    return agent


def _patch_driver(driver: str, changed: dict) -> str:
    if "front_slide_steer" in changed:
        driver = _once(driver, "rate *= 0.75", f'rate *= {changed["front_slide_steer"]["new"]}')
    if "front_slide_brake" in changed:
        driver = _once(driver, "brake *= .75", f'brake *= {changed["front_slide_brake"]["new"]}')
    return driver


def write_archive(source: Path, game: str, strategy: dict, agent: str, dest: Path, driver: str | None = None) -> str:
    dest.parent.mkdir(parents=True, exist_ok=True)
    rendered = render_strategy(strategy).encode()
    with zipfile.ZipFile(source) as incoming, io.BytesIO() as buffer:
        with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as outgoing:
            design = None
            if game == "racing":
                design = design_from_strategy(json.loads(incoming.read("design.json")), strategy)
            for name in incoming.namelist():
                if name == "strategy.py":
                    payload = rendered
                elif name == "agent.py":
                    payload = agent.encode()
                elif name == "formula_driver.py" and driver is not None:
                    payload = driver.encode()
                elif name == "design.json":
                    payload = json.dumps(design, indent=2, sort_keys=True).encode()
                else:
                    payload = incoming.read(name)
                info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o100644 << 16
                outgoing.writestr(info, payload)
        dest.write_bytes(buffer.getvalue())
    return sha256(dest)


def draft_game(game: str, source: Path, dest: Path, predict_fn, evidence: dict | None = None) -> dict:
    from self_jev.element_data import checkpoint_path

    strategy = read_strategy(source)
    with zipfile.ZipFile(source) as bundle:
        agent = bundle.read("agent.py").decode()
        driver = bundle.read("formula_driver.py").decode() if game == "racing" else ""
    decision = decide(game, strategy, agent, predict_fn, evidence)
    book = apply_book(strategy, decision)
    patched, patched_driver = patch_code(game, agent, driver, decision["elements"])
    digest = write_archive(source, game, book, patched, dest, patched_driver or None)
    try:
        checkpoint = str(checkpoint_path())
    except FileNotFoundError:
        checkpoint = "untrained"
    return {
        "game": game,
        "name": decision["name"],
        "base_name": strategy.get("name"),
        "source_archive": str(source),
        "archive": str(dest),
        "sha256": digest,
        "submitted": False,
        "checkpoint": checkpoint,
        "model": decision["model"],
        "elements": decision["elements"],
        "changed": [item["id"] for item in decision["elements"] if item["changed"]],
        "strategy": book,
    }


def draft_main() -> None:
    """Write the next multi-element artifacts. This command does not submit."""
    from self_jev.check import check_bundle

    built = {}
    for game in ("fps", "racing"):
        source = LIVE[game]
        dest = ROOT / "submissions" / game / "pending-draft.zip"
        critique = draft_game(game, source, dest, model_predict)
        final = ROOT / "submissions" / game / f"{critique['name']}.zip"
        if final.resolve() != dest.resolve():
            final.write_bytes(dest.read_bytes())
            dest.unlink()
        critique["archive"] = str(final)
        strategy = critique.pop("strategy")
        final.with_suffix(".json").write_text(json.dumps(critique, ensure_ascii=False, indent=2) + "\n")
        smoke = check_bundle(final, game)
        print(f"{game}: {critique['name']} {smoke} changed {', '.join(critique['changed'])} sha {critique['sha256'][:12]}")
        built[game] = {
            "archive": str(final),
            "sha256": critique["sha256"],
            "changed": critique["changed"],
            "name": strategy["name"],
            "submitted": False,
        }
    (ROOT / "submissions" / "draft.json").write_text(json.dumps({"submitted": False, "games": built}, indent=2) + "\n")


def model_predict(situation: str, question: str, candidates: list[dict]) -> dict:
    from self_jev.element_data import checkpoint_path
    from self_jev.predict import load_bundle, predict, score_candidates

    if not hasattr(model_predict, "bundle"):
        model_predict.bundle = load_bundle(checkpoint_path())
    bundle = model_predict.bundle

    def scorer(text, asked, options):
        return score_candidates(bundle, text, asked, options)

    return predict(situation, question, candidates, scorer=scorer, temperature=float(bundle["temperature"]))


def _shown(value) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return str(value)


def _once(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise RuntimeError(f"agent patch missed a unique site: {old[:60]}")
    return text.replace(old, new, 1)


def _literal_dict(agent: str, name: str) -> dict:
    match = re.search(rf"{name} = (\{{.*\}})", agent)
    if not match:
        return {}
    return ast.literal_eval(match.group(1))


def _aim_gates(agent: str) -> dict:
    match = re.search(r"abs\(yaw_error\) < ([0-9.]+) and abs\(pitch_error\) < ([0-9.]+)", agent)
    if not match:
        return {}
    return {"yaw": float(match.group(1)), "pitch": float(match.group(2))}


def _chase_ticks(agent: str) -> int | None:
    match = re.search(r"last_enemy_tick\", 0\) < (\d+)", agent)
    return int(match.group(1)) if match else None


def _tuple_body(agent: str, name: str) -> tuple[str, ...]:
    match = re.search(rf"{name} = \((.*?)\)", agent, re.S)
    if not match:
        return ()
    return tuple(re.findall(r'"([^"]+)"', match.group(1)))
