#!/usr/bin/env python3
"""Bounded F1 Previous-Race Feedback Study (N1 + N3, Support/Query Paired Control).

Implements the locked methodological protocol:
  - 9 track x weather cells x 5 fresh ordered seed pairs (seed_A, seed_B) = 45 units
    (strictly disjoint from the prior confirmation seeds [101, 202, 303, 404, 505]).
  - Three races per unit for each feedback-capable model:
      1. Round 1 (Support, seed_A): zero-shot observation race; records actions + factual outcome.
      2. Round 2 Control (Query, seed_B): receives previous actions ONLY (no outcome fields).
      3. Round 2 Feedback (Query, seed_B): receives previous actions + deterministic factual
         Round 1 outcome telemetry (NO AGX winner gap, NO strategic advice/conclusions).
  - Primary causal comparison: Round 2 Feedback vs. Round 2 Control on matched seed_B.
  - Frozen references rerun on the exact same 45 seed_B query conditions:
      * nn-combo (N1: frozen build-only specialist; no feedback input; no weight updates)
      * driver_only (always dry_balanced + balanced + stay_out)
      * track_rule (Parkland -> balanced; Highland/Crossover -> push_max)
  - N3: Low-level 30 Hz RL driver (models/jev_rl_v7.mapdata.npz) and simulator physics
    remain strictly frozen across all runs; memory resets between every unit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import threading
import time
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
DECISION_EVAL_SRC = REPO_ROOT / "decision_eval" / "src"
for p in (str(REPO_ROOT), str(SCRIPTS_DIR), str(DECISION_EVAL_SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)

import eval_race_engineer_v2 as re_eval  # noqa: E402
from self_jev.car_build_nn.builds import BUILD_NAMES, get_candidate_builds  # noqa: E402
from self_jev.car_build_nn.dataset import _load_frozen_driver  # noqa: E402
from self_jev.car_build_nn.model import CarBuildNumPy  # noqa: E402
from self_jev.sim.design import COMPOUNDS  # noqa: E402
from self_jev.sim.env import RaceEnv  # noqa: E402
from self_jev.sim.physics import PhysicsConfig  # noqa: E402
from self_jev.sim.residual import assist_ers_mode, combine_residual_actions  # noqa: E402

DEFAULT_OUT = REPO_ROOT / "reports" / "feedback_f1_summary.json"
CACHE_DIR = REPO_ROOT / "reports" / "feedback_f1_cache"
WTL_TOL_S = 0.50

# Fresh ordered (support_seed_A, query_seed_B) pairs — strictly disjoint from [101..505]
FRESH_SEED_PAIRS: list[tuple[int, int]] = [
    (1001, 2001),
    (1002, 2002),
    (1003, 2003),
    (1004, 2004),
    (1005, 2005),
]

STATIC_TASK_RULES = (
    "Task Rules: 4-lap race with fixed 30Hz RL driver. Each pit stop adds ~35.8s in pit lane and resets tyre wear to 0.0. "
    "Builds: dry_low_drag, dry_balanced, damp_mid_downforce, wet_high_downforce. "
    "Modes: push_max (+0.15 throttle, aggressive ERS), balanced (standard RL), conserve_tyres (-0.08 throttle, gentle ERS), wet_caution (-0.15 throttle, harvest ERS)."
)

STATUS_LABELS = {
    1: "Racing",
    2: "Full finish (4/4 laps)",
    3: "Timeout DNF",
    4: "Stall DNF",
    5: "Wall crash DNF",
}


def load_dotenv(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    if not path.is_file():
        return env
    for ln in path.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if ln and not ln.startswith("#") and "=" in ln:
            k, v = ln.split("=", 1)
            env[k.strip()] = v.strip()
    return env


def make_scenario(base: dict[str, Any], cell_idx: int, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed * 100 + cell_idx)
    reg = base["weather_regime"]
    if reg == "Dry":
        ws = float(np.clip(rng.uniform(0.0, 0.06), 0.0, 0.06))
        we = float(np.clip(rng.uniform(0.0, 0.06), 0.0, 0.06))
        rp = float(rng.uniform(0.0, 0.15))
        temp = float(rng.uniform(26.0, 34.0))
    elif reg == "Wetting":
        ws = float(rng.uniform(0.02, 0.14))
        we = float(rng.uniform(0.58, 0.88))
        rp = float(rng.uniform(0.55, 0.90))
        temp = float(rng.uniform(20.0, 27.0))
    else:  # Saturated
        ws = float(rng.uniform(0.76, 0.95))
        we = float(rng.uniform(0.78, 0.96))
        rp = float(rng.uniform(0.85, 0.99))
        temp = float(rng.uniform(15.0, 22.0))
    return {
        "scenario_id": f"{base['id']}_s{seed}",
        "cell_id": base["id"],
        "track": base["track"],
        "weather_regime": reg,
        "seed": seed,
        "wet_start": round(ws, 4),
        "wet_end": round(we, 4),
        "rain_prob": round(rp, 4),
        "track_temp_c": round(temp, 2),
        "ramp_start_s": round(float(rng.uniform(25.0, 80.0)), 1),
        "ramp_end_s": round(float(rng.uniform(280.0, 420.0)), 1),
        "grip_scale": round(float(rng.uniform(0.94, 1.06)), 4),
        "aero_scale": round(float(rng.uniform(0.94, 1.06)), 4),
    }


def build_units(seed_pairs: list[tuple[int, int]]) -> list[dict[str, Any]]:
    units = []
    for cell_idx, base in enumerate(re_eval.CANONICAL_SCENARIOS):
        for pair_idx, (seed_a, seed_b) in enumerate(seed_pairs):
            units.append({
                "unit_id": f"{base['id']}_p{pair_idx + 1}",
                "cell_id": base["id"],
                "track": base["track"],
                "weather_regime": base["weather_regime"],
                "seed_a": seed_a,
                "seed_b": seed_b,
                "support_sc": make_scenario(base, cell_idx, seed_a),
                "query_sc": make_scenario(base, cell_idx, seed_b),
            })
    return units


def format_history_prefix(r1: dict[str, Any] | None, include_outcomes: bool) -> str:
    """Deterministic factual formatter (no AGX winner gap, no strategic advice)."""
    if r1 is None:
        return ""
    modes_str = ",".join(r1["mode_history"]) if r1["mode_history"] else "none"
    pits_str = ",".join(r1["pit_calls"]) if r1["pit_calls"] else "none"
    base = (
        f"Previous Race Actions (same track/weather family, prior seed): "
        f"build={r1['chosen_build']}; lap_modes=[{modes_str}]; pit_calls=[{pits_str}]."
    )
    if not include_outcomes:
        return base
    st_lbl = STATUS_LABELS.get(int(r1["status"]), f"status_{r1['status']}")
    time_lbl = (
        f"finish_time={r1['finish_time_s']:.1f}s"
        if r1["finished"] and r1["finish_time_s"] is not None
        else f"elapsed_time={r1['elapsed_time_s']:.1f}s"
    )
    return (
        f"Previous Race Record (same track/weather family, prior seed): "
        f"build={r1['chosen_build']}; lap_modes=[{modes_str}]; pit_calls=[{pits_str}]; "
        f"outcome={st_lbl}; completed_laps={r1['completed_laps']}/4; {time_lbl}; "
        f"pits_completed={r1['pit_stops']} (recorded_pit_time={r1['pit_time_s']:.1f}s); "
        f"off_track_ticks={r1['off_track_ticks']}; wall_grazes={r1['wall_grazes']}; "
        f"final_tyre_wear={r1['final_tyre_wear']:.2f}; final_ers_soc={r1['final_ers_soc_mj']:.2f}MJ; "
        f"invalid_outputs={r1['invalid_outputs']}."
    )


class FeedbackDecisionAdapter:
    """Frozen decision adapter supporting self-jev-minilm, frontier-llm, nn-combo, driver_only, track_rule."""

    def __init__(self, name: str, env_vars: dict[str, str] | None = None, cache_dir: Path = CACHE_DIR) -> None:
        self.name = name
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.minilm_tok = None
        self.minilm_mod = None
        self.nn_engine = None
        self.env_vars = env_vars or {}
        self._lock = threading.Lock()
        self.api_calls = 0
        self.api_in_tokens = 0
        self.api_out_tokens = 0
        self.api_latencies_ms: list[float] = []
        self.max_prompt_tokens_seen = 0

        if name == "self-jev-minilm":
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            self.minilm_tok = AutoTokenizer.from_pretrained(str(re_eval.DEFAULT_MINILM))
            self.minilm_mod = AutoModelForSequenceClassification.from_pretrained(str(re_eval.DEFAULT_MINILM))
            self.minilm_mod.eval()
        elif name == "nn-combo":
            self.nn_engine = CarBuildNumPy(REPO_ROOT / "models" / "car_build_nn.npz")
        elif name == "frontier-llm":
            if not (self.env_vars.get("QWEN_API_KEY") and self.env_vars.get("QWEN_BASE_URL") and self.env_vars.get("QWEN_MODEL")):
                raise RuntimeError("Missing QWEN_API_KEY / QWEN_BASE_URL / QWEN_MODEL in .env for frontier-llm")
        elif name not in ("driver_only", "track_rule"):
            raise ValueError(f"Unsupported model: {name}")

    def _call_frontier(self, prompt: str, n_opts: int) -> tuple[int, bool]:
        base_url = self.env_vars["QWEN_BASE_URL"]
        api_key = self.env_vars["QWEN_API_KEY"]
        model = self.env_vars["QWEN_MODEL"]
        body_obj: dict[str, Any] = {
            "model": model,
            "temperature": 0.0,
            "max_tokens": 64,
            "response_format": {"type": "json_object"},
            "messages": [{"role": "user", "content": prompt}],
        }
        if "openrouter.ai" in base_url:
            body_obj["reasoning"] = {"effort": "none", "exclude": True}
        else:
            body_obj["enable_thinking"] = False

        raw_req = json.dumps(body_obj, sort_keys=True, ensure_ascii=False)
        req_hash = hashlib.sha256(f"{base_url}|{raw_req}".encode("utf-8")).hexdigest()[:24]
        cfile = self.cache_dir / f"f1_{req_hash}.json"
        if cfile.is_file():
            rec = json.loads(cfile.read_text(encoding="utf-8"))
        else:
            url = base_url.rstrip("/") + "/chat/completions"
            data = json.dumps(body_obj).encode("utf-8")
            headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
            rec = {"ok": False, "content": "", "latency_ms": 0.0, "in_tok": 0, "out_tok": 0}
            for attempt in range(4):
                t0 = time.perf_counter()
                req = urllib.request.Request(url, data=data, headers=headers, method="POST")
                try:
                    with urllib.request.urlopen(req, timeout=45) as resp:
                        raw_json = json.loads(resp.read().decode("utf-8"))
                    lat = round((time.perf_counter() - t0) * 1000.0, 2)
                    content = str(raw_json.get("choices", [{}])[0].get("message", {}).get("content", "")).strip()
                    usage = raw_json.get("usage") or {}
                    rec = {
                        "ok": True,
                        "content": content,
                        "latency_ms": lat,
                        "in_tok": int(usage.get("prompt_tokens") or 0),
                        "out_tok": int(usage.get("completion_tokens") or 0),
                    }
                    tmp = cfile.with_suffix(f".{threading.get_ident()}.tmp")
                    tmp.write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
                    tmp.replace(cfile)
                    break
                except Exception:
                    time.sleep(1.2 * (attempt + 1))

        with self._lock:
            self.api_calls += 1
            self.api_in_tokens += int(rec.get("in_tok", 0))
            self.api_out_tokens += int(rec.get("out_tok", 0))
            if rec.get("latency_ms", 0) > 0:
                self.api_latencies_ms.append(float(rec["latency_ms"]))
        try:
            m = re.search(r"\{.*?\}", str(rec.get("content", "")), flags=re.DOTALL)
            parsed = json.loads(m.group(0) if m else str(rec.get("content", "")))
            idx = int(parsed["choice_index"])
            if 0 <= idx < n_opts:
                return idx, True
        except Exception:
            pass
        return 1 if n_opts > 1 else 0, False

    def decide(
        self,
        kind: str,
        state: str,
        question: str,
        options: list[str],
        option_descriptions: list[str],
        sc: dict[str, Any],
    ) -> tuple[str, bool]:
        """Returns (chosen_option, parse_ok)."""
        if self.name == "driver_only":
            if kind == "build":
                return "dry_balanced", True
            if kind == "pace":
                return "balanced", True
            return "stay_out", True

        if self.name == "track_rule":
            if kind == "build":
                return "dry_balanced", True
            if kind == "pace":
                return ("balanced" if sc["track"] == "parkland" else "push_max"), True
            return "stay_out", True

        if self.name == "nn-combo":
            if kind == "build":
                # Exact 12-feature vector schema from self_jev/car_build_nn/dataset.py
                tr = sc["track"].lower()
                tlen = {"parkland": 5795.8, "highland": 6999.4, "crossover": 5803.2}.get(tr, 5800.0)
                feats = [
                    tlen,
                    1.0 if tr == "parkland" else 0.0,
                    1.0 if tr == "highland" else 0.0,
                    1.0 if tr == "crossover" else 0.0,
                    float(sc["rain_prob"]),
                    float(sc["wet_start"]),
                    float(sc["wet_end"]),
                    float(sc["track_temp_c"]),
                    0.95,  # forecast_confidence
                    0.0,   # sensor_missing
                    0.05,  # recent_off_track_rate
                    0.05,  # recent_wear_rate
                ]
                probs = self.nn_engine.predict_probs(feats)
                best_idx = int(np.argmax(probs[:4]))  # actionable build choice (0..3)
                return options[best_idx], True
            if kind == "pace":
                return "balanced", True
            return "stay_out", True

        if self.name == "self-jev-minilm":
            query = f"{state} {question}"
            pairs = [(query, f"{opt}: {desc}") for opt, desc in zip(options, option_descriptions)]
            with torch.inference_mode():
                enc = self.minilm_tok(
                    [p[0] for p in pairs],
                    [p[1] for p in pairs],
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                )
                seq_len = int(enc["input_ids"].shape[1])
                self.max_prompt_tokens_seen = max(self.max_prompt_tokens_seen, seq_len)
                logits = self.minilm_mod(**enc).logits.squeeze(-1)
                best_idx = int(torch.argmax(logits).item())
            return options[best_idx], True

        if self.name == "frontier-llm":
            numbered = [f"{i}: {opt} ({desc})" for i, (opt, desc) in enumerate(zip(options, option_descriptions))]
            prompt = (
                f"{STATIC_TASK_RULES}\n\n"
                f"{state}\n\n"
                f"Question: {question}\n"
                f"Options:\n" + "\n".join(numbered) + "\n\n"
                'Return ONLY valid JSON: {"choice_index": <int>}'
            )
            idx, ok = self._call_frontier(prompt, len(options))
            return options[idx], ok

        return options[0], False


@torch.inference_mode()
def run_batched_track_branch(
    track_id: str,
    scenarios: list[dict[str, Any]],
    history_prefixes: list[str],
    adapter: FeedbackDecisionAdapter,
    driver_model: torch.nn.Module,
    obs_mean: torch.Tensor,
    obs_std: torch.Tensor,
    is_residual: bool,
    cfg: PhysicsConfig,
    builds: dict[str, Any],
    laps_total: int = 4,
    max_steps: int = 30 * 800,
) -> list[dict[str, Any]]:
    """Run all scenarios for one track in a single vectorized batch (grouped by chosen build)."""
    torch.set_num_threads(1)
    device = torch.device("cpu")
    n_sc = len(scenarios)

    # Step 1: Pre-race build decision per scenario (parallelized for frontier-llm)
    chosen_builds: list[str] = ["dry_balanced"] * n_sc
    invalid_counts = [0] * n_sc

    def _decide_build(i: int) -> tuple[int, str, bool]:
        sc = scenarios[i]
        hpref = history_prefixes[i]
        pre_state = (
            (f"{hpref}\n" if hpref else "")
            + f"Current Race Forecast — Track: {sc['track']}, regime={sc['weather_regime']}, "
            f"wetness_start={sc['wet_start']:.2f}, wetness_end={sc['wet_end']:.2f}, "
            f"rain_prob={sc['rain_prob']:.2f}, track_temp_c={sc['track_temp_c']:.1f}."
        )
        b_choice, ok = adapter.decide(
            "build", pre_state, "Which car build should we start this 4-lap race with?",
            list(BUILD_NAMES), re_eval.BUILD_DESCS, sc,
        )
        return i, b_choice, ok

    if adapter.name == "frontier-llm" and n_sc > 1:
        with ThreadPoolExecutor(max_workers=8) as pool:
            for i, b_choice, ok in pool.map(_decide_build, range(n_sc)):
                if not ok:
                    invalid_counts[i] += 1
                chosen_builds[i] = b_choice if b_choice in builds else "dry_balanced"
    else:
        for i in range(n_sc):
            _, b_choice, ok = _decide_build(i)
            if not ok:
                invalid_counts[i] += 1
            chosen_builds[i] = b_choice if b_choice in builds else "dry_balanced"

    # Group scenario indices by chosen build (RaceEnv takes one CarParams per batch)
    results: list[dict[str, Any] | None] = [None] * n_sc
    by_build: dict[str, list[int]] = {}
    for i, b in enumerate(chosen_builds):
        by_build.setdefault(b, []).append(i)

    for b_name, idxs in by_build.items():
        w_n = len(idxs)
        torch.manual_seed(42)
        env = RaceEnv(
            track_id=track_id,
            worlds=w_n,
            cars_per_world=1,
            laps_total=laps_total,
            phase="race",
            weather_regime="dry",
            car=builds[b_name],
            cfg=cfg,
            device=device,
            domain_rand=False,
        )
        for wi, orig_i in enumerate(idxs):
            sc = scenarios[orig_i]
            env.w_state["wet_start"][wi] = sc["wet_start"]
            env.w_state["wet_end"][wi] = sc["wet_end"]
            env.w_state["ramp_start_s"][wi] = sc["ramp_start_s"]
            env.w_state["ramp_end_s"][wi] = sc["ramp_end_s"]
            env.w_state["wetness"][wi] = sc["wet_start"]
            env.w_state["rain_intensity"][wi] = sc["wet_start"] * 0.5
            env.w_state["track_temp_c"][wi] = sc["track_temp_c"]
            env.w_state["forecast_rain_prob"][wi] = sc["rain_prob"]
        env.reset_all()
        for wi, orig_i in enumerate(idxs):
            sc = scenarios[orig_i]
            env.grip_scale[wi] = sc["grip_scale"]
            env.aero_scale[wi] = sc["aero_scale"]

        cur_modes = ["balanced"] * w_n
        pit_latched = [False] * w_n
        pit_target_idx = [0] * w_n
        prev_pit_stops = [0] * w_n
        last_decided_lap = [-1] * w_n
        mode_histories: list[list[str]] = [[] for _ in range(w_n)]
        pit_calls_hist: list[list[str]] = [[] for _ in range(w_n)]
        off_track_ticks = [0] * w_n
        wall_grazes = [0] * w_n
        pit_time_ticks = [0] * w_n

        lap_pool = ThreadPoolExecutor(max_workers=8) if adapter.name == "frontier-llm" else None
        try:
          for step_i in range(max_steps):
            pending_lap_calls: list[tuple[int, int, int, str, dict[str, Any], bool]] = []
            for wi, orig_i in enumerate(idxs):
                if int(env.status[wi].item()) != 1:
                    continue
                cur_pit_stops = int(env.pit_stops[wi].item())
                if cur_pit_stops > prev_pit_stops[wi]:
                    pit_latched[wi] = False
                    prev_pit_stops[wi] = cur_pit_stops

                cur_lap = int(env.lap[wi].item())
                if cur_lap > last_decided_lap[wi] and cur_lap < laps_total:
                    last_decided_lap[wi] = cur_lap
                    sc = scenarios[orig_i]
                    hpref = history_prefixes[orig_i]
                    wet_now = float(env.wetness[wi].item())
                    wear_now = float(env.tyre_wear[wi].item())
                    soc_now = float(env.ers_soc_mj[wi].item())
                    cmp_now = COMPOUNDS[int(env.compound[wi].item())]
                    t_now = float(env.t_race[wi].item())
                    live_state = (
                        (f"{hpref}\n" if hpref else "")
                        + f"Current Race Telemetry — Track: {sc['track']}, Lap {cur_lap + 1}/{laps_total}, "
                        f"elapsed={t_now:.1f}s, wetness={wet_now:.2f} (forecast_end={sc['wet_end']:.2f}), "
                        f"tyre_compound={cmp_now}, tyre_wear={wear_now:.2f}, ers_soc={soc_now:.2f}MJ, "
                        f"pit_stops={cur_pit_stops}."
                    )
                    need_pit = cur_lap in (1, 2) and not pit_latched[wi]
                    pending_lap_calls.append((wi, orig_i, cur_lap, live_state, sc, need_pit))

            if pending_lap_calls:
                def _run_lap_decisions(item: tuple[int, int, int, str, dict[str, Any], bool]) -> tuple[int, int, str, bool, str | None, bool]:
                    wi_l, orig_l, _cl, lstate, sc_l, do_pit = item
                    m_ch, ok_m = adapter.decide(
                        "pace", lstate, "Which driving pace & ERS mode should the RL driver use for this lap?",
                        re_eval.PACE_OPTIONS, re_eval.PACE_DESCS, sc_l,
                    )
                    p_ch, ok_p = (None, True)
                    if do_pit:
                        p_ch, ok_p = adapter.decide(
                            "pit", lstate, "Should the car stay out or pit for fresh tyres at the end of this lap?",
                            re_eval.PIT_OPTIONS, re_eval.PIT_DESCS, sc_l,
                        )
                    return wi_l, orig_l, m_ch, ok_m, p_ch, ok_p

                outputs = (
                    list(lap_pool.map(_run_lap_decisions, pending_lap_calls))
                    if (lap_pool is not None and len(pending_lap_calls) > 1)
                    else [_run_lap_decisions(it) for it in pending_lap_calls]
                )
                for wi, orig_i, m_choice, ok, p_choice, ok_p in outputs:
                    if not ok:
                        invalid_counts[orig_i] += 1
                    cur_modes[wi] = m_choice
                    mode_histories[wi].append(m_choice)
                    if p_choice is not None:
                        if not ok_p:
                            invalid_counts[orig_i] += 1
                        pit_calls_hist[wi].append(p_choice)
                        if p_choice == "pit_soft":
                            pit_latched[wi] = True
                            pit_target_idx[wi] = 0
                        elif p_choice == "pit_medium":
                            pit_latched[wi] = True
                            pit_target_idx[wi] = 1

            raw = env.get_obs(include_map_ext=False)
            x = torch.clamp(torch.nan_to_num((raw - obs_mean) / obs_std, nan=0.0, posinf=10.0, neginf=-10.0), -10.0, 10.0)
            mu, ers_logits, flag_logits, _ = driver_model(x)
            ers = torch.argmax(ers_logits, dim=-1)
            mu_adj = mu.clone()
            ovr = (flag_logits[:, 1] > 0.0).float()

            for wi in range(w_n):
                m = cur_modes[wi]
                if m == "push_max":
                    mu_adj[wi, 0] = mu_adj[wi, 0] + 0.15
                    if float(env.ers_soc_mj[wi].item()) > 0.5 and float(env.wetness[wi].item()) < 0.45:
                        ers[wi] = 3
                        ovr[wi] = 1.0
                elif m == "conserve_tyres":
                    mu_adj[wi, 0] = mu_adj[wi, 0] - 0.08
                    ers[wi] = 1
                    ovr[wi] = 0.0
                elif m == "wet_caution":
                    mu_adj[wi, 0] = mu_adj[wi, 0] - 0.15
                    ers[wi] = 0
                    ovr[wi] = 0.0

            if is_residual:
                thr, brk, st_cmd = combine_residual_actions(raw, mu_adj)
                bal_ers = assist_ers_mode(raw, ers, thr, brk)
                for wi in range(w_n):
                    if cur_modes[wi] == "balanced":
                        ers[wi] = bal_ers[wi]
            else:
                thr = torch.clamp(mu_adj[:, 0], 0.0, 1.0)
                brk = torch.clamp(mu_adj[:, 1], 0.0, 1.0)
                st_cmd = torch.clamp(mu_adj[:, 2], -1.0, 1.0)

            env.step({
                "throttle": thr,
                "brake": brk,
                "steer": st_cmd,
                "ers_mode": ers,
                "aero_mode": (flag_logits[:, 0] > 0.0).float(),
                "override": ovr,
                "aid": (flag_logits[:, 2] > 0.0).float(),
                "pit_request": torch.tensor([1.0 if p else 0.0 for p in pit_latched], device=device),
                "pit_compound": torch.tensor(pit_target_idx, device=device, dtype=torch.long),
            })

            for wi in range(w_n):
                if int(env.status[wi].item()) == 1:
                    if int(env.off_track[wi].item()) > 0:
                        off_track_ticks[wi] += 1
                    if int(env.wall_graze[wi].item()) > 0:
                        wall_grazes[wi] += 1
                    if int(env.pit_state[wi].item()) != 0:
                        pit_time_ticks[wi] += 1

            if (step_i + 1) % 32 == 0 and (env.status != 1).all():
                break
        finally:
          if lap_pool is not None:
            lap_pool.shutdown(wait=True)

        for wi, orig_i in enumerate(idxs):
            sc = scenarios[orig_i]
            st0 = int(env.status[wi].item())
            fin = st0 == 2
            ft = round(float(env.finish_t[wi].item()), 3) if fin and float(env.finish_t[wi].item()) > 0 else None
            results[orig_i] = {
                "scenario_id": sc["scenario_id"],
                "cell_id": sc["cell_id"],
                "track": sc["track"],
                "weather_regime": sc["weather_regime"],
                "seed": sc["seed"],
                "chosen_build": b_name,
                "mode_history": mode_histories[wi],
                "pit_calls": pit_calls_hist[wi],
                "status": st0,
                "finished": fin,
                "completed_laps": laps_total if fin else max(0, int(env.lap[wi].item())),
                "finish_time_s": ft,
                "elapsed_time_s": round(float(env.t_race[wi].item()), 2),
                "pit_stops": int(env.pit_stops[wi].item()),
                "pit_time_s": round(pit_time_ticks[wi] / 30.0, 1),
                "off_track_ticks": off_track_ticks[wi],
                "wall_grazes": wall_grazes[wi],
                "final_tyre_wear": round(float(env.tyre_wear[wi].item()), 3),
                "final_ers_soc_mj": round(float(env.ers_soc_mj[wi].item()), 3),
                "invalid_outputs": invalid_counts[orig_i],
            }

    return [r for r in results if r is not None]


def summarize_paired_feedback(
    units: list[dict[str, Any]],
    r1_list: list[dict[str, Any]],
    ctrl_list: list[dict[str, Any]],
    fb_list: list[dict[str, Any]],
) -> dict[str, Any]:
    n = len(units)
    ctrl_fin = sum(1 for r in ctrl_list if r["finished"])
    fb_fin = sum(1 for r in fb_list if r["finished"])
    r1_fin = sum(1 for r in r1_list if r["finished"])

    rescued = 0
    regressed = 0
    both_failed = 0
    both_finished = 0
    wins = ties_finished = losses = 0
    deltas: list[float] = []
    build_changed = 0
    mode_changed = 0
    pit_changed = 0
    r1_failed_units = 0
    r1_failed_repaired = 0
    r1_passed_units = 0
    r1_passed_preserved = 0

    for r1, c, f in zip(r1_list, ctrl_list, fb_list):
        cf, ff = c["finished"], f["finished"]
        if not r1["finished"]:
            r1_failed_units += 1
            if ff:
                r1_failed_repaired += 1
        else:
            r1_passed_units += 1
            if ff:
                r1_passed_preserved += 1

        if f["chosen_build"] != c["chosen_build"]:
            build_changed += 1
        if f["mode_history"] != c["mode_history"]:
            mode_changed += 1
        if f["pit_calls"] != c["pit_calls"]:
            pit_changed += 1

        if ff and not cf:
            rescued += 1
            wins += 1
        elif cf and not ff:
            regressed += 1
            losses += 1
        elif not cf and not ff:
            both_failed += 1
        else:
            both_finished += 1
            d = round(float(f["finish_time_s"]) - float(c["finish_time_s"]), 4)
            deltas.append(d)
            if d < -WTL_TOL_S:
                wins += 1
            elif d > WTL_TOL_S:
                losses += 1
            else:
                ties_finished += 1

    ds = sorted(deltas)
    ctrl_ts = [float(r["finish_time_s"]) for r in ctrl_list if r["finished"] and r["finish_time_s"] is not None]
    fb_ts = [float(r["finish_time_s"]) for r in fb_list if r["finished"] and r["finish_time_s"] is not None]

    return {
        "n_units": n,
        "r1_support_finishes": r1_fin,
        "control_finishes": ctrl_fin,
        "feedback_finishes": fb_fin,
        "rescued_failures": rescued,
        "newly_introduced_failures": regressed,
        "net_completion_delta": round((rescued - regressed) / max(n, 1), 4),
        "wtl_feedback_vs_control": {
            "wins": wins,
            "ties_both_finished": ties_finished,
            "ties_both_failed": both_failed,
            "losses": losses,
        },
        "common_finishers_n": both_finished,
        "paired_mean_delta_s": round(sum(ds) / len(ds), 2) if ds else None,
        "paired_median_delta_s": round(ds[len(ds) // 2], 2) if ds else None,
        "control_avg_time_finishers_s": round(sum(ctrl_ts) / len(ctrl_ts), 2) if ctrl_ts else None,
        "feedback_avg_time_finishers_s": round(sum(fb_ts) / len(fb_ts), 2) if fb_ts else None,
        "control_pits_total": sum(r["pit_stops"] for r in ctrl_list),
        "feedback_pits_total": sum(r["pit_stops"] for r in fb_list),
        "control_invalid_outputs": sum(r["invalid_outputs"] for r in ctrl_list),
        "feedback_invalid_outputs": sum(r["invalid_outputs"] for r in fb_list),
        "behavior_changes": {
            "build_changed_units": build_changed,
            "mode_changed_units": mode_changed,
            "pit_changed_units": pit_changed,
        },
        "r1_conditional": {
            "r1_failed_units": r1_failed_units,
            "r1_failed_repaired_in_r2_fb": r1_failed_repaired,
            "r1_passed_units": r1_passed_units,
            "r1_passed_preserved_in_r2_fb": r1_passed_preserved,
        },
        "control_mode_counts": dict(Counter(m for r in ctrl_list for m in r["mode_history"])),
        "feedback_mode_counts": dict(Counter(m for r in fb_list for m in r["mode_history"])),
    }


def summarize_static_reference(query_list: list[dict[str, Any]], label: str, role: str) -> dict[str, Any]:
    fin = sum(1 for r in query_list if r["finished"])
    ts = [float(r["finish_time_s"]) for r in query_list if r["finished"] and r["finish_time_s"] is not None]
    return {
        "label": label,
        "role": role,
        "is_non_adaptive_reference": True,
        "n_units": len(query_list),
        "query_finishes": fin,
        "query_avg_time_finishers_s": round(sum(ts) / len(ts), 2) if ts else None,
        "pits_total": sum(r["pit_stops"] for r in query_list),
        "mode_counts": dict(Counter(m for r in query_list for m in r["mode_history"])),
        "build_counts": dict(Counter(r["chosen_build"] for r in query_list)),
    }


def run_study(models: list[str], limit_pairs: int = 5, out_path: Path = DEFAULT_OUT) -> dict[str, Any]:
    t0 = time.perf_counter()
    env_vars = load_dotenv(REPO_ROOT / ".env")
    seed_pairs = FRESH_SEED_PAIRS[:limit_pairs]
    units = build_units(seed_pairs)
    tracks = ("parkland", "highland", "crossover")

    device = torch.device("cpu")
    cal_path = REPO_ROOT / "self_jev" / "sim" / "calibrated_params.json"
    cfg = PhysicsConfig.from_json(cal_path) if cal_path.is_file() else PhysicsConfig()
    builds = get_candidate_builds()
    driver_model, obs_mean, obs_std, is_residual = _load_frozen_driver(re_eval.DEFAULT_DRIVER, device)

    results_models: dict[str, Any] = {}
    feedback_models = {"self-jev-minilm", "frontier-llm"}

    for m_name in models:
        tm0 = time.perf_counter()
        adapter = FeedbackDecisionAdapter(m_name, env_vars=env_vars)
        print(f"\n[F1 Study] Running {m_name} across {len(units)} units ({len(seed_pairs)} pairs x 9 cells)...", flush=True)

        if m_name in feedback_models:
            r1_all: list[dict[str, Any]] = []
            ctrl_all: list[dict[str, Any]] = []
            fb_all: list[dict[str, Any]] = []
            for tr in tracks:
                tr_units = [u for u in units if u["track"] == tr]
                sup_scs = [u["support_sc"] for u in tr_units]
                qry_scs = [u["query_sc"] for u in tr_units]
                # 1. Round 1 (Support, seed_A)
                r1_rows = run_batched_track_branch(
                    tr, sup_scs, [""] * len(sup_scs), adapter, driver_model, obs_mean, obs_std, is_residual, cfg, builds
                )
                # 2. Round 2 Control (Query, seed_B — previous actions ONLY, no outcomes)
                ctrl_prefs = [format_history_prefix(r, include_outcomes=False) for r in r1_rows]
                ctrl_rows = run_batched_track_branch(
                    tr, qry_scs, ctrl_prefs, adapter, driver_model, obs_mean, obs_std, is_residual, cfg, builds
                )
                # 3. Round 2 Feedback (Query, seed_B — previous actions + factual Round 1 outcomes)
                fb_prefs = [format_history_prefix(r, include_outcomes=True) for r in r1_rows]
                fb_rows = run_batched_track_branch(
                    tr, qry_scs, fb_prefs, adapter, driver_model, obs_mean, obs_std, is_residual, cfg, builds
                )
                r1_all.extend(r1_rows)
                ctrl_all.extend(ctrl_rows)
                fb_all.extend(fb_rows)
                print(
                    f"  {m_name:16s} | {tr:9s} | R1={sum(r['finished'] for r in r1_rows)}/{len(r1_rows)} "
                    f"| R2_ctrl={sum(r['finished'] for r in ctrl_rows)}/{len(ctrl_rows)} "
                    f"| R2_fb={sum(r['finished'] for r in fb_rows)}/{len(fb_rows)}",
                    flush=True,
                )

            summary = summarize_paired_feedback(units, r1_all, ctrl_all, fb_all)
            summary["label"] = f"{m_name} ({env_vars.get('QWEN_MODEL', '')})" if m_name == "frontier-llm" else m_name
            summary["role"] = "Frozen decision model; in-context F1 factual feedback vs matched no-outcome control"
            summary["is_non_adaptive_reference"] = False
            summary["wall_s"] = round(time.perf_counter() - tm0, 2)
            summary["max_prompt_tokens_seen"] = adapter.max_prompt_tokens_seen or None
            if m_name == "frontier-llm":
                summary["api_usage"] = {
                    "calls": adapter.api_calls,
                    "prompt_tokens": adapter.api_in_tokens,
                    "completion_tokens": adapter.api_out_tokens,
                    "latency_p50_ms": round(float(np.percentile(adapter.api_latencies_ms, 50)), 1) if adapter.api_latencies_ms else None,
                }
            results_models[m_name] = summary
        else:
            # Non-adaptive frozen references rerun on the exact same 45 seed_B query conditions
            qry_all: list[dict[str, Any]] = []
            for tr in tracks:
                tr_units = [u for u in units if u["track"] == tr]
                qry_scs = [u["query_sc"] for u in tr_units]
                qry_rows = run_batched_track_branch(
                    tr, qry_scs, [""] * len(qry_scs), adapter, driver_model, obs_mean, obs_std, is_residual, cfg, builds
                )
                qry_all.extend(qry_rows)
            role_map = {
                "nn-combo": "N1: Frozen build-only specialist; no previous-race feedback input; no weight updates",
                "driver_only": "Frozen baseline (dry_balanced + balanced + stay_out)",
                "track_rule": "Frozen strong reference (Parkland->balanced; Highland/Crossover->push_max)",
            }
            ref_sum = summarize_static_reference(qry_all, m_name, role_map.get(m_name, "Frozen reference"))
            ref_sum["wall_s"] = round(time.perf_counter() - tm0, 2)
            results_models[m_name] = ref_sum
            print(f"  {m_name:16s} | query finishes={ref_sum['query_finishes']}/{ref_sum['n_units']} | avg={ref_sum['query_avg_time_finishers_s']}s", flush=True)

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "F1_own_run_factual_telemetry_vs_matched_no_outcome_control",
        "driver_weights": str(re_eval.DEFAULT_DRIVER.name),
        "weights_updated": False,
        "external_agx_gap_included_in_feedback": False,
        "n_units": len(units),
        "n_cells": len(re_eval.CANONICAL_SCENARIOS),
        "seed_pairs": seed_pairs,
        "wtl_tolerance_s": WTL_TOL_S,
        "wall_s": round(time.perf_counter() - t0, 2),
        "models": results_models,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved F1 feedback study report -> {out_path} ({report['wall_s']}s)")
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="Run F1 Previous-Race Feedback Study (N1 + N3).")
    ap.add_argument(
        "--models",
        nargs="+",
        default=["frontier-llm", "self-jev-minilm", "nn-combo", "driver_only", "track_rule"],
    )
    ap.add_argument("--limit-pairs", type=int, default=5, help="Number of seed pairs per cell (5 = 45 units)")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    run_study(args.models, limit_pairs=args.limit_pairs, out_path=args.out)


if __name__ == "__main__":
    main()
