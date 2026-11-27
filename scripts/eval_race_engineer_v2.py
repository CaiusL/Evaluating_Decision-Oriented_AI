#!/usr/bin/env python3
"""Closed-loop 4-lap Decision Model + Driver Evaluation (Pace Eval v2).

Instead of only picking a static pre-race car build, each decision model acts as
a live in-race Decision Model controlling BOTH:
  1. Pre-race car build (dry_low_drag, dry_balanced, damp_mid_downforce, wet_high_downforce)
  2. Live in-race driver actions at race start and every lap crossing:
       - Pace & ERS mode: push_max, balanced, conserve_tyres, wet_caution
       - Pit & tyre call: stay_out, pit_soft, pit_medium (Laps 1-2)

Paired with frozen 30 Hz RL driver (models/jev_rl_v7.mapdata.npz) in RaceEnv,
and scored with the auditable v2 Pace-Gap Evaluation contract from
scripts/eval_pace_accuracy.py against:
  - External AGX Winner Reference (data/agx_winner_times.json)
  - Driver-Only Baseline (jev_rl_v7 alone with dry_balanced + balanced mode)

Supported --models:
  driver_only, if_else_baseline, self-jev-minilm, kev-0.8b, qwen-0.8b-ll, qwen-0.8b-json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
DECISION_EVAL_SRC = REPO_ROOT / "decision_eval" / "src"
for p in (str(REPO_ROOT), str(SCRIPTS_DIR), str(DECISION_EVAL_SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)

import eval_pace_accuracy as ep
from self_jev.car_build_nn.builds import BUILD_NAMES, get_candidate_builds
from self_jev.car_build_nn.dataset import _load_frozen_driver
from self_jev.sim.design import COMPOUNDS
from self_jev.sim.env import RaceEnv
from self_jev.sim.physics import PhysicsConfig
from self_jev.sim.residual import assist_ers_mode, combine_residual_actions

DEFAULT_DRIVER = REPO_ROOT / "models" / "jev_rl_v7.mapdata.npz"
DEFAULT_WINNERS = REPO_ROOT / "data" / "agx_winner_times.json"
DEFAULT_MINILM = REPO_ROOT / "models" / "ms-marco-MiniLM-L6-v2"
DEFAULT_OUT = REPO_ROOT / "reports" / "race_engineer_v2_summary.json"

# 9 canonical track x weather scenarios (every cell has a non-null AGX winner time)
CANONICAL_SCENARIOS: list[dict[str, Any]] = [
    {"id": "parkland_dry", "track": "parkland", "weather_regime": "Dry", "wet_start": 0.02, "wet_end": 0.02, "rain_prob": 0.05, "track_temp_c": 30.0},
    {"id": "parkland_wetting", "track": "parkland", "weather_regime": "Wetting", "wet_start": 0.05, "wet_end": 0.72, "rain_prob": 0.75, "track_temp_c": 24.0},
    {"id": "parkland_saturated", "track": "parkland", "weather_regime": "Saturated", "wet_start": 0.85, "wet_end": 0.88, "rain_prob": 0.95, "track_temp_c": 19.0},
    {"id": "highland_dry", "track": "highland", "weather_regime": "Dry", "wet_start": 0.02, "wet_end": 0.02, "rain_prob": 0.05, "track_temp_c": 29.0},
    {"id": "highland_wetting", "track": "highland", "weather_regime": "Wetting", "wet_start": 0.08, "wet_end": 0.75, "rain_prob": 0.78, "track_temp_c": 23.0},
    {"id": "highland_saturated", "track": "highland", "weather_regime": "Saturated", "wet_start": 0.85, "wet_end": 0.90, "rain_prob": 0.95, "track_temp_c": 18.0},
    {"id": "crossover_dry", "track": "crossover", "weather_regime": "Dry", "wet_start": 0.02, "wet_end": 0.02, "rain_prob": 0.05, "track_temp_c": 30.0},
    {"id": "crossover_wetting", "track": "crossover", "weather_regime": "Wetting", "wet_start": 0.06, "wet_end": 0.74, "rain_prob": 0.76, "track_temp_c": 24.0},
    {"id": "crossover_saturated", "track": "crossover", "weather_regime": "Saturated", "wet_start": 0.86, "wet_end": 0.90, "rain_prob": 0.96, "track_temp_c": 18.5},
]

PACE_OPTIONS = ["push_max", "balanced", "conserve_tyres", "wet_caution"]
PACE_DESCS = [
    "Push maximum attack (+12% target speed, full ERS deploy & override) when track is dry and tyre wear < 0.65.",
    "Balanced race pace (standard RL speed and automatic ERS) for moderate grip or transitional conditions.",
    "Conserve tyres (-8% target speed, gentle ERS) when tyre wear >= 0.65 or car is sliding to avoid crashing.",
    "Wet caution (-15% target speed, harvest ERS, no override) when track wetness >= 0.55 to prevent wall crashes.",
]

PIT_OPTIONS = ["stay_out", "pit_soft", "pit_medium"]
PIT_DESCS = [
    "Stay out on current tyres (saves 35.8s pit-lane time; best unless tyre wear >= 0.80).",
    "Pit now for fresh soft tyres (high grip, costs 35.8s pit stop; use if tyre wear >= 0.80).",
    "Pit now for fresh medium tyres (durable, costs 35.8s pit stop; use if tyre wear >= 0.80 and heavy wear expected).",
]

BUILD_DESCS = [
    "Low-drag dry setup (medium start tyres, low wings 15/16) — fastest on dry straights, fragile in rain.",
    "Balanced dry setup (soft start tyres, wings 20/22) — stable and fast on dry or light-damp tracks.",
    "Mid-downforce damp setup (medium start tyres, wings 24/26) — transitional damp/wetting conditions.",
    "High-downforce wet setup (soft start tyres, wings 29/31) — maximum grip for wet/saturated tracks.",
]


class DecisionAdapter:
    """Unified adapter for any local decision model acting as live Decision Model."""

    def __init__(self, name: str, device: str = "cpu", backend: str = "auto") -> None:
        self.name = name
        self.runner: Any = None
        self.minilm_tok: Any = None
        self.minilm_mod: Any = None

        if name == "self-jev-minilm":
            trained_dir = REPO_ROOT / "models" / "self_jev_minilm"
            if not trained_dir.is_dir():
                raise FileNotFoundError(
                    f"Requested trained checkpoint '{name}' not found at {trained_dir}. "
                    "Use 'ms-marco-minilm-l6-v2' explicitly to run the off-the-shelf MS MARCO cross-encoder."
                )
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            self.minilm_tok = AutoTokenizer.from_pretrained(str(trained_dir))
            self.minilm_mod = AutoModelForSequenceClassification.from_pretrained(str(trained_dir))
            self.minilm_mod.eval()
        elif name == "ms-marco-minilm-l6-v2":
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            self.minilm_tok = AutoTokenizer.from_pretrained(str(DEFAULT_MINILM))
            self.minilm_mod = AutoModelForSequenceClassification.from_pretrained(str(DEFAULT_MINILM))
            self.minilm_mod.eval()
        elif name == "kev-0.8b":
            from decision_eval.runners import KevRunner
            self.runner = KevRunner("jaredpalmer/kev-0.8b", device=device, backend=backend)
        elif name == "qwen-0.8b-ll":
            from decision_eval.runners import create_qwen_runner
            self.runner = create_qwen_runner("Qwen/Qwen3.5-0.8B-Base", device=device, backend=backend)
        elif name == "qwen-0.8b-json":
            from decision_eval.runners import create_qwen_runner
            self.runner = create_qwen_runner("Qwen/Qwen3.5-0.8B", device=device, backend=backend)
        elif name not in ("driver_only", "if_else_baseline"):
            raise ValueError(f"Unsupported model: {name}")

    def decide(
        self,
        kind: str,
        state: str,
        question: str,
        options: list[str],
        option_descriptions: list[str],
        telemetry: dict[str, Any],
    ) -> str:
        if self.name == "driver_only":
            if kind == "build":
                return "dry_balanced"
            if kind == "pace":
                return "balanced"
            return "stay_out"

        if self.name == "if_else_baseline":
            if kind == "build":
                wet_avg = 0.5 * (telemetry["wet_start"] + telemetry["wet_end"])
                if wet_avg >= 0.45:
                    return "wet_high_downforce"
                return "dry_balanced"
            if kind == "pace":
                wet = telemetry["wetness"]
                wear = telemetry["tyre_wear"]
                if wet >= 0.55:
                    return "wet_caution"
                if wear >= 0.65:
                    return "conserve_tyres"
                if wet < 0.20 and wear < 0.50:
                    return "push_max"
                return "balanced"
            return "pit_soft" if telemetry["tyre_wear"] >= 0.85 and telemetry["lap"] <= 2 else "stay_out"

        if self.name == "self-jev-minilm":
            query = f"{state} {question}"
            pairs = [(query, f"{opt}: {desc}") for opt, desc in zip(options, option_descriptions)]
            with torch.inference_mode():
                enc = self.minilm_tok(
                    [p[0] for p in pairs],
                    [p[1] for p in pairs],
                    padding=True,
                    truncation=True,
                    max_length=256,
                    return_tensors="pt",
                )
                logits = self.minilm_mod(**enc).logits.squeeze(-1)
                best_idx = int(torch.argmax(logits).item())
            return options[best_idx]

        if self.name == "kev-0.8b":
            res = self.runner.score_unified(state, question, options, option_descriptions)
            return res["choice"] or options[0]
        if self.name == "qwen-0.8b-ll":
            res = self.runner.score_loglik(state, question, options, option_descriptions)
            return res["choice"] or options[0]
        if self.name == "qwen-0.8b-json":
            res = self.runner.score_json(state, question, options, option_descriptions)
            return res["choice"] if res.get("parse_ok") and res.get("choice") in options else options[1]
        return options[0]


@torch.inference_mode()
def run_closed_loop_race(
    sc: dict[str, Any],
    adapter: DecisionAdapter,
    driver_model: torch.nn.Module,
    obs_mean: torch.Tensor,
    obs_std: torch.Tensor,
    is_residual: bool,
    cfg: PhysicsConfig,
    builds: dict[str, Any],
    device: torch.device,
    laps_total: int = 4,
    max_steps: int = 30 * 800,
) -> dict[str, Any]:
    pre_state = (
        f"Track: {sc['track']} | Weather forecast: regime={sc['weather_regime']}, "
        f"wetness_start={sc['wet_start']:.2f}, wetness_end={sc['wet_end']:.2f}, "
        f"rain_prob={sc['rain_prob']:.2f}, track_temp_c={sc['track_temp_c']:.1f}."
    )
    chosen_build = adapter.decide(
        "build",
        pre_state,
        "Which car build should we start this 4-lap race with?",
        list(BUILD_NAMES),
        BUILD_DESCS,
        sc,
    )
    if chosen_build not in builds:
        chosen_build = "dry_balanced"

    env = RaceEnv(
        track_id=sc["track"],
        worlds=1,
        cars_per_world=1,
        laps_total=laps_total,
        phase="race",
        weather_regime="dry",
        car=builds[chosen_build],
        cfg=cfg,
        device=device,
    )
    env.w_state["wet_start"][0] = sc["wet_start"]
    env.w_state["wet_end"][0] = sc["wet_end"]
    env.w_state["ramp_start_s"][0] = 40.0
    env.w_state["ramp_end_s"][0] = 360.0
    env.w_state["wetness"][0] = sc["wet_start"]
    env.w_state["rain_intensity"][0] = sc["wet_start"] * 0.5
    env.w_state["track_temp_c"][0] = sc["track_temp_c"]
    env.w_state["forecast_rain_prob"][0] = sc["rain_prob"]
    env.reset_all()

    current_mode = "balanced"
    pit_latched = False
    pit_target_idx = 0
    prev_pit_stops = 0
    last_decided_lap = -1
    mode_history: list[str] = []
    pit_calls: list[str] = []
    decision_calls = 1

    for _step_i in range(max_steps):
        cur_lap = int(env.lap[0].item())
        cur_pit_stops = int(env.pit_stops[0].item())
        if cur_pit_stops > prev_pit_stops:
            pit_latched = False
            prev_pit_stops = cur_pit_stops

        if cur_lap > last_decided_lap and cur_lap < laps_total and int(env.status[0].item()) == 1:
            last_decided_lap = cur_lap
            wet_now = float(env.wetness[0].item())
            wear_now = float(env.tyre_wear[0].item())
            soc_now = float(env.ers_soc_mj[0].item())
            cmp_now = COMPOUNDS[int(env.compound[0].item())]
            t_now = float(env.t_race[0].item())
            tel = {
                "lap": cur_lap,
                "wetness": wet_now,
                "tyre_wear": wear_now,
                "ers_soc_mj": soc_now,
                "compound": cmp_now,
                "t_race": t_now,
            }
            live_state = (
                f"Live Race Telemetry — Track: {sc['track']}, Lap {cur_lap + 1}/{laps_total}, "
                f"elapsed={t_now:.1f}s, wetness={wet_now:.2f} (forecast_end={sc['wet_end']:.2f}), "
                f"tyre_compound={cmp_now}, tyre_wear={wear_now:.2f}, ers_soc={soc_now:.2f} MJ, "
                f"pit_stops={cur_pit_stops}."
            )
            current_mode = adapter.decide(
                "pace",
                live_state,
                "Which driving pace & ERS mode should the RL driver use for this lap?",
                PACE_OPTIONS,
                PACE_DESCS,
                tel,
            )
            mode_history.append(current_mode)
            decision_calls += 1

            if cur_lap in (1, 2) and not pit_latched:
                pcall = adapter.decide(
                    "pit",
                    live_state,
                    "Should the car stay out or pit for fresh tyres at the end of this lap?",
                    PIT_OPTIONS,
                    PIT_DESCS,
                    tel,
                )
                pit_calls.append(pcall)
                decision_calls += 1
                if pcall == "pit_soft":
                    pit_latched = True
                    pit_target_idx = 0
                elif pcall == "pit_medium":
                    pit_latched = True
                    pit_target_idx = 1

        raw = env.get_obs(include_map_ext=False)
        x = torch.clamp(
            torch.nan_to_num((raw - obs_mean) / obs_std, nan=0.0, posinf=10.0, neginf=-10.0),
            -10.0,
            10.0,
        )
        mu, ers_logits, flag_logits, _ = driver_model(x)
        ers = torch.argmax(ers_logits, dim=-1)

        mu_adj = mu.clone()
        ovr = (flag_logits[:, 1] > 0.0).float()
        if current_mode == "push_max":
            mu_adj[:, 0] = mu_adj[:, 0] + 0.15
            if float(env.ers_soc_mj[0].item()) > 0.5 and float(env.wetness[0].item()) < 0.45:
                ers = torch.full_like(ers, 3)
                ovr = torch.ones_like(ovr)
        elif current_mode == "conserve_tyres":
            mu_adj[:, 0] = mu_adj[:, 0] - 0.08
            ers = torch.full_like(ers, 1)
            ovr = torch.zeros_like(ovr)
        elif current_mode == "wet_caution":
            mu_adj[:, 0] = mu_adj[:, 0] - 0.15
            ers = torch.zeros_like(ers)
            ovr = torch.zeros_like(ovr)

        if is_residual:
            thr, brk, st_cmd = combine_residual_actions(raw, mu_adj)
            if current_mode == "balanced":
                ers = assist_ers_mode(raw, ers, thr, brk)
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
            "pit_request": torch.tensor([1.0 if pit_latched else 0.0], device=device),
            "pit_compound": torch.tensor([pit_target_idx], device=device, dtype=torch.long),
        })
        if int(env.status[0].item()) != 1:
            break

    st0 = int(env.status[0].item())
    finished = st0 == 2
    ft = round(float(env.finish_t[0].item()), 3) if finished and float(env.finish_t[0].item()) > 0 else None
    return {
        "scenario_id": sc["id"],
        "track": sc["track"],
        "weather_regime": sc["weather_regime"],
        "chosen_build": chosen_build,
        "status": st0,
        "finished": finished,
        "completed_laps": laps_total if finished else max(0, int(env.lap[0].item())),
        "finish_time_s": ft,
        "pit_stops": int(env.pit_stops[0].item()),
        "mode_history": mode_history,
        "pit_calls": pit_calls,
        "decision_calls": decision_calls,
    }


def evaluate_models(
    model_names: list[str],
    driver_path: Path = DEFAULT_DRIVER,
    winners_path: Path = DEFAULT_WINNERS,
    out_path: Path = DEFAULT_OUT,
    device_str: str = "cpu",
    backend: str = "auto",
    max_workers: int = 6,
) -> dict[str, Any]:
    w_doc = json.loads(winners_path.read_text(encoding="utf-8"))
    meta = w_doc.get("metadata") or {}
    required_laps = int(meta.get("required_laps", 4))
    ref_version = str(meta.get("reference_version", "unversioned"))
    winners = {tr.lower(): dict(d) for tr, d in w_doc["agx_winner_times_seconds"].items()}

    t0 = time.perf_counter()
    raw_by_model: dict[str, list[dict[str, Any]]] = {m: [] for m in model_names}

    heavy_models = {"kev-0.8b", "qwen-0.8b-ll", "qwen-0.8b-json"}
    light_jobs = [(m, sc["id"]) for m in model_names if m not in heavy_models for sc in CANONICAL_SCENARIOS]
    heavy_list = [m for m in model_names if m in heavy_models]

    if light_jobs and max_workers > 1:
        script_path = str(Path(__file__).resolve())
        pending = list(light_jobs)
        running: list[tuple[str, str, subprocess.Popen[str]]] = []
        results_map: dict[tuple[str, str], dict[str, Any]] = {}
        while pending or running:
            while pending and len(running) < max_workers:
                m, sid = pending.pop(0)
                cmd = [
                    sys.executable,
                    script_path,
                    "--worker",
                    f"{m}:{sid}",
                    "--driver",
                    str(driver_path),
                    "--device",
                    device_str,
                    "--backend",
                    backend,
                ]
                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                running.append((m, sid, proc))
            still: list[tuple[str, str, subprocess.Popen[str]]] = []
            for m, sid, proc in running:
                ret = proc.poll()
                if ret is None:
                    still.append((m, sid, proc))
                    continue
                out, err = proc.communicate()
                if ret != 0:
                    raise RuntimeError(f"Worker {m}:{sid} failed ({ret}): {err}")
                results_map[(m, sid)] = json.loads(out)
            running = still
            if running:
                time.sleep(0.15)
        for m in model_names:
            if m not in heavy_models:
                raw_by_model[m] = [results_map[(m, sc["id"])] for sc in CANONICAL_SCENARIOS]
    else:
        heavy_list = list(model_names)

    if heavy_list:
        device = torch.device(device_str)
        cal_path = REPO_ROOT / "self_jev" / "sim" / "calibrated_params.json"
        cfg = PhysicsConfig.from_json(cal_path) if cal_path.is_file() else PhysicsConfig()
        builds = get_candidate_builds()
        driver_model, obs_mean, obs_std, is_residual = _load_frozen_driver(driver_path, device)
        for m in heavy_list:
            adapter = DecisionAdapter(m, device=device_str, backend=backend)
            runs = []
            for sc in CANONICAL_SCENARIOS:
                runs.append(
                    run_closed_loop_race(
                        sc, adapter, driver_model, obs_mean, obs_std, is_residual, cfg, builds, device, laps_total=required_laps
                    )
                )
            raw_by_model[m] = runs

    best_by_sc: dict[str, float | None] = {}
    baseline_by_sc: dict[str, float | None] = {}
    for sc in CANONICAL_SCENARIOS:
        sid = sc["id"]
        fin_ts = [r["finish_time_s"] for runs in raw_by_model.values() for r in runs if r["scenario_id"] == sid and r["finished"] and r["finish_time_s"] is not None]
        best_by_sc[sid] = min(fin_ts) if fin_ts else None
        if "driver_only" in raw_by_model:
            base_r = next((r for r in raw_by_model["driver_only"] if r["scenario_id"] == sid), None)
            baseline_by_sc[sid] = base_r["finish_time_s"] if base_r and base_r["finished"] else None

    models_summary: dict[str, Any] = {}
    for m, runs in raw_by_model.items():
        ext_entries = []
        best_entries = []
        mode_counts: Counter[str] = Counter()
        total_pits = 0
        total_calls = 0
        deltas_vs_base: list[float] = []

        for r in runs:
            mode_counts.update(r["mode_history"])
            total_pits += r["pit_stops"]
            total_calls += r["decision_calls"]
            tr = r["track"].lower()
            reg = r["weather_regime"]
            norm = {
                "match_id": r["scenario_id"],
                "track": tr,
                "track_display": tr.capitalize(),
                "classified": r["finished"],
                "classification_source": "official",
                "legal": True,
                "terminal_failure": not r["finished"],
                "full_distance_completed": r["finished"] and r["completed_laps"] == required_laps,
                "finish_time_s": r["finish_time_s"],
                "timing_compatible": r["finished"] and r["finish_time_s"] is not None,
                "weather": {"regime": reg, "source": "official", "mapping_version": "canonical-sim-v1", "evidence": {}, "ambiguities": []},
                "observed_laps": r["completed_laps"],
                "rank": None,
            }
            ref_t = winners.get(tr, {}).get(reg)
            ext_sc = ep.score_normalized_run(norm, reference_applicable=ref_t is not None, reference_time_s=ref_t)
            ext_entries.append((norm, ext_sc))

            bt = best_by_sc[r["scenario_id"]]
            best_sc = ep.score_normalized_run(norm, reference_applicable=bt is not None, reference_time_s=bt)
            best_entries.append((norm, best_sc))

            base_t = baseline_by_sc.get(r["scenario_id"])
            if base_t is not None and r["finished"] and r["finish_time_s"] is not None:
                deltas_vs_base.append(round(r["finish_time_s"] - base_t, 3))

        deltas_sorted = sorted(deltas_vs_base)
        med_delta = deltas_sorted[len(deltas_sorted) // 2] if deltas_sorted else None
        mean_delta = round(sum(deltas_vs_base) / len(deltas_vs_base), 2) if deltas_vs_base else None

        models_summary[m] = {
            "scenarios_n": len(runs),
            "decision_calls_total": total_calls,
            "pit_stops_total": total_pits,
            "mode_counts": dict(mode_counts),
            "mean_delta_vs_driver_only_s": mean_delta,
            "median_delta_vs_driver_only_s": round(med_delta, 2) if med_delta is not None else None,
            "external_agx_reference": ep.summarize_scored_slice(ext_entries),
            "same_scenario_best_model": ep.summarize_scored_slice(best_entries),
            "runs": runs,
        }

    report = {
        "evaluator_version": f"{ep.EVALUATOR_VERSION}+race-engineer",
        "reference_version": ref_version,
        "required_laps": required_laps,
        "driver_weights": str(driver_path.name),
        "scenarios_n": len(CANONICAL_SCENARIOS),
        "wall_s": round(time.perf_counter() - t0, 2),
        "models": models_summary,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="Closed-loop 4-lap Decision Model + Driver Pace Eval v2.")
    ap.add_argument("--worker", default=None, help="Internal worker: <model>:<scenario_id>")
    ap.add_argument(
        "--models",
        nargs="+",
        default=["driver_only", "if_else_baseline", "self-jev-minilm"],
        help="Models to evaluate (driver_only, if_else_baseline, self-jev-minilm, kev-0.8b, qwen-0.8b-ll, qwen-0.8b-json)",
    )
    ap.add_argument("--driver", type=Path, default=DEFAULT_DRIVER)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--backend", default="auto")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    if args.worker:
        torch.set_num_threads(1)
        m, sid = args.worker.split(":", 1)
        sc = next(s for s in CANONICAL_SCENARIOS if s["id"] == sid)
        device = torch.device(args.device)
        cal_path = REPO_ROOT / "self_jev" / "sim" / "calibrated_params.json"
        cfg = PhysicsConfig.from_json(cal_path) if cal_path.is_file() else PhysicsConfig()
        builds = get_candidate_builds()
        driver_model, obs_mean, obs_std, is_residual = _load_frozen_driver(args.driver, device)
        adapter = DecisionAdapter(m, device=args.device, backend=args.backend)
        row = run_closed_loop_race(sc, adapter, driver_model, obs_mean, obs_std, is_residual, cfg, builds, device, laps_total=4)
        print(json.dumps(row))
        return

    rep = evaluate_models(
        args.models, driver_path=args.driver, out_path=args.out, device_str=args.device, backend=args.backend, max_workers=args.workers
    )
    print(f"Finished Decision Model + Driver 4-lap eval in {rep['wall_s']}s (driver={rep['driver_weights']}, n={rep['scenarios_n']})\n")
    for m, d in rep["models"].items():
        ext = d["external_agx_reference"]
        bst = d["same_scenario_best_model"]
        er = ext["competitive_completion_rate_by_threshold"]
        br = bst["competitive_completion_rate_by_threshold"]
        print(
            f"  {m:16s} | fin4={ext['valid_full_distance_gap_n']}/{d['scenarios_n']} "
            f"| vs_driver_only={d['mean_delta_vs_driver_only_s']}s (med {d['median_delta_vs_driver_only_s']}s) "
            f"| best@0%={br['le_0pct']} best@5%={br['le_5pct']} "
            f"| AGX_med_gap={ext['median_gap_s_among_comparable_finishes']}s ({ext['median_gap_pct_among_comparable_finishes']}%) "
            f"AGX@10%={er['le_10pct']} AGX@20%={er['le_20pct']} "
            f"| pits={d['pit_stops_total']} modes={d['mode_counts']}"
        )


if __name__ == "__main__":
    main()
