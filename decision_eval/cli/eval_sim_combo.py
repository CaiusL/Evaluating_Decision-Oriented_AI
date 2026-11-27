"""Closed-loop simulator evaluation: evaluate each decision model paired with the jev_rl Driver NN.

Compares end-to-end racing systems on Suite 3 (suite3_car_build_test.jsonl, n=34):
  - nn-combo (car-build-nn + jev-rl)
  - kev-0.8b + jev-rl
  - qwen-0.8b-ll + jev-rl
  - qwen-0.8b-json + jev-rl
  - self-jev-minilm + jev-rl (when present in benchmark_raw.jsonl)
  - oracle + jev-rl (upper bound)

Measures real downstream simulator outcomes (Finish Rate, Rule-Legal Finish Rate, Crash/DNF Rate,
Mean Finish Time, Mean Sim Score, Mean Oracle Regret, Safe Abstain Rate) plus combined model
footprint and measured per-tick jev_rl driving latency.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = Path(__file__).resolve().parents[1] / "src"
for p in (str(REPO_ROOT), str(SRC_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from decision_eval import paths
from self_jev.car_build_nn.builds import ABSTAIN_LABEL, BUILD_NAMES, get_candidate_builds
from self_jev.car_build_nn.dataset import _load_frozen_driver, _score_builds_in_sim
from self_jev.sim.physics import PhysicsConfig

APPROX_MODEL_SIZE_MB = {
    "nn-combo": 0.0,          # Computed dynamically from car_build_nn.npz + driver_npz
    "car-build-nn": 0.0,      # Alias for nn-combo
    "kev-0.8b": 1650.0,       # Qwen3.5-0.8B-Base BF16 + Kev pointer/LoRA
    "qwen-0.8b-ll": 1600.0,   # Qwen3.5-0.8B-Base BF16
    "qwen-0.8b-json": 1600.0, # Qwen3.5-0.8B BF16
    "self-jev-minilm": 90.0,  # MiniLM cross-encoder
    "frontier-llm": 0.0,      # Cloud API
}


def _benchmark_driver_tick_ms(driver_model: torch.nn.Module, obs_dim: int, device: torch.device, n_ticks: int = 300) -> float:
    x = torch.zeros((1, obs_dim), device=device, dtype=torch.float32)
    with torch.inference_mode():
        for _ in range(20):
            driver_model(x)
        t0 = time.perf_counter()
        for _ in range(n_ticks):
            driver_model(x)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return round(elapsed_ms / n_ticks, 4)


def _extract_scenario(item: dict[str, Any]) -> dict[str, float]:
    feats = item["features"]
    return {
        "rain_prob": float(feats[4]),
        "wet_start": float(feats[5]),
        "wet_end": float(feats[6]),
        "track_temp_c": float(feats[7]),
        "recent_off_track_rate": float(feats[8]),
        "recent_wear_rate": float(feats[9]),
        "forecast_confidence": float(feats[10]),
        "sensor_missing_flag": float(feats[11]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Closed-loop Suite 3 simulator evaluation of Car-Build + Driver NN combos.")
    parser.add_argument("--suite3-file", type=Path, default=paths.BENCHMARKS_DIR / "suite3_car_build_test.jsonl")
    parser.add_argument("--raw-file", type=Path, default=paths.REPORTS_DIR / "benchmark_raw.jsonl")
    parser.add_argument("--driver", type=Path, default=paths.MODELS_DIR / "jev_rl_v7.mapdata.npz")
    parser.add_argument("--out-dir", type=Path, default=paths.REPORTS_DIR)
    parser.add_argument("--max-steps", type=int, default=30 * 110)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    items = [json.loads(ln) for ln in args.suite3_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
    raw_records = [json.loads(ln) for ln in args.raw_file.read_text(encoding="utf-8").splitlines() if ln.strip()]

    # Map model -> {item_id: record} for Suite 3 test orig
    preds_by_model: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for r in raw_records:
        if r.get("suite") == "suite3_car_build" and r.get("split") == "test" and r.get("order") == "orig":
            m = "nn-combo" if r["model"] == "car-build-nn" else r["model"]
            preds_by_model[m][r["id"]] = r

    # Group scenarios by track and simulate all 4 candidate builds once with jev_rl
    device = torch.device(args.device)
    cal_path = REPO_ROOT / "self_jev" / "sim" / "calibrated_params.json"
    cfg = PhysicsConfig.from_json(cal_path) if cal_path.is_file() else PhysicsConfig()
    builds = get_candidate_builds()
    driver_model, obs_mean, obs_std, is_residual = _load_frozen_driver(args.driver, device)
    driver_tick_ms = _benchmark_driver_tick_ms(driver_model, len(obs_mean), device)
    driver_mb = round(args.driver.stat().st_size / (1024 * 1024), 3)
    car_nn_mb = round(paths.CAR_BUILD_NN_NPZ.stat().st_size / (1024 * 1024), 4) if paths.CAR_BUILD_NN_NPZ.is_file() else 0.0

    by_track: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for it in items:
        by_track[it["track"]].append(it)

    sim_lookup: dict[tuple[str, str], dict[str, float]] = {}
    t_sim0 = time.perf_counter()
    for track_id, t_items in by_track.items():
        scenarios = [_extract_scenario(it) for it in t_items]
        sim_out = _score_builds_in_sim(
            track_id, scenarios, builds, driver_model, obs_mean, obs_std, is_residual, cfg, device, max_steps=args.max_steps
        )
        for idx, it in enumerate(t_items):
            for bname in BUILD_NAMES:
                sim_lookup[(it["id"], bname)] = sim_out[bname][idx]
    sim_wall_s = round(time.perf_counter() - t_sim0, 2)

    # Evaluate each decision model + oracle
    eval_models = list(preds_by_model.keys()) + ["oracle"]
    rows_out: list[dict[str, Any]] = []

    for m in eval_models:
        actionable_n = 0
        abstain_gold_n = 0
        correct_n = 0
        clean_finish_n = 0
        legal_finish_n = 0
        rule_violation_n = 0
        crash_dnf_n = 0
        false_abstain_n = 0
        safe_hold_n = 0
        unsafe_launch_n = 0
        finish_times: list[float] = []
        sim_scores: list[float] = []
        regrets: list[float] = []
        pre_race_lats: list[float] = []

        for it in items:
            iid = it["id"]
            is_unseen = it.get("bucket") == "unseen_lever"
            is_abstain_gold = bool(it.get("is_abstain_gold", False))
            oscores = it.get("oracle_scores") or {}

            if m == "oracle":
                pred_label = it["gold_label"]
                pre_race_lats.append(0.0)
            else:
                rec = preds_by_model[m].get(iid)
                if not rec:
                    continue
                pred_label = rec.get("pred_label")
                pre_race_lats.append(float(rec.get("latency_ms") or 0.0))

            if pred_label == it["gold_label"]:
                correct_n += 1

            if is_abstain_gold:
                abstain_gold_n += 1
                if pred_label == ABSTAIN_LABEL:
                    safe_hold_n += 1
                else:
                    unsafe_launch_n += 1
                continue

            actionable_n += 1
            best_oracle = max(float(v) for v in oscores.values()) if oscores else 0.0

            if pred_label == ABSTAIN_LABEL or pred_label not in builds:
                false_abstain_n += 1
                if oscores:
                    regrets.append(max(0.0, best_oracle - min(float(v) for v in oscores.values())))
                continue

            sim_res = sim_lookup[(iid, pred_label)]
            st = int(sim_res["status"])
            finished = (st == 2)
            rule_legal = not (is_unseen and builds[pred_label].start_compound == "soft")
            if not rule_legal:
                rule_violation_n += 1
            if finished:
                clean_finish_n += 1
                finish_times.append(float(sim_res["finish_t"]))
                if rule_legal:
                    legal_finish_n += 1
            elif st in (4, 5):
                crash_dnf_n += 1

            sim_scores.append(float(sim_res["oracle_score"]))
            if oscores and pred_label in oscores:
                regrets.append(max(0.0, best_oracle - float(oscores[pred_label])))

        total_n = actionable_n + abstain_gold_n
        base_mb = car_nn_mb if m == "nn-combo" else APPROX_MODEL_SIZE_MB.get(m, 0.0)
        combo_mb = round(base_mb + driver_mb, 2) if m != "oracle" else driver_mb
        pre_sorted = sorted(pre_race_lats)
        p50_pre = round(pre_sorted[len(pre_sorted) // 2], 3) if pre_sorted else 0.0

        rows_out.append({
            "system": f"{m} + jev-rl" if m != "nn-combo" else "nn-combo (car-build-nn + jev-rl)",
            "decision_model": m,
            "n_total": total_n,
            "suite3_accuracy": round(correct_n / max(1, total_n), 4),
            "actionable_n": actionable_n,
            "clean_finish_rate": round(clean_finish_n / max(1, actionable_n), 4),
            "rule_legal_finish_rate": round(legal_finish_n / max(1, actionable_n), 4),
            "rule_violation_rate": round(rule_violation_n / max(1, actionable_n), 4),
            "crash_dnf_rate": round(crash_dnf_n / max(1, actionable_n), 4),
            "false_abstain_rate": round(false_abstain_n / max(1, actionable_n), 4),
            "mean_finish_time_s": round(sum(finish_times) / len(finish_times), 3) if finish_times else None,
            "mean_sim_score": round(sum(sim_scores) / len(sim_scores), 3) if sim_scores else None,
            "mean_oracle_regret": round(sum(regrets) / len(regrets), 3) if regrets else None,
            "abstain_gold_n": abstain_gold_n,
            "safe_hold_rate": round(safe_hold_n / max(1, abstain_gold_n), 4),
            "unsafe_launch_rate": round(unsafe_launch_n / max(1, abstain_gold_n), 4),
            "pre_race_p50_ms": p50_pre,
            "driver_tick_ms": driver_tick_ms,
            "combined_size_mb": combo_mb,
        })

    report = {
        "driver_weights": str(args.driver),
        "driver_size_mb": driver_mb,
        "car_build_nn_size_mb": car_nn_mb,
        "driver_tick_ms": driver_tick_ms,
        "realtime_30hz_budget_ms": 33.33,
        "sim_wall_s": sim_wall_s,
        "suite3_test_items": len(items),
        "systems": rows_out,
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "sim_combo_report.json"
    csv_path = args.out_dir / "sim_combo_report.csv"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    if rows_out:
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows_out[0].keys()))
            writer.writeheader()
            writer.writerows(rows_out)

    print(f"Closed-loop simulator eval complete in {sim_wall_s}s | driver={args.driver.name} ({driver_tick_ms} ms/tick)")
    print(f"  JSON: {json_path}\n  CSV : {csv_path}\n")
    for r in rows_out:
        print(
            f"  {r['system']:34s} | acc={r['suite3_accuracy']*100:5.1f}% | "
            f"legal_fin={r['rule_legal_finish_rate']*100:5.1f}% | crash={r['crash_dnf_rate']*100:4.1f}% | "
            f"safe_hold={r['safe_hold_rate']*100:5.1f}% | regret={r['mean_oracle_regret'] or 0:6.2f} | "
            f"pre={r['pre_race_p50_ms']:7.2f}ms | size={r['combined_size_mb']:7.2f}MB"
        )


if __name__ == "__main__":
    main()
