#!/usr/bin/env python3
"""Run m2-qwen3.5-0.8b-lora-s42 on the 9 canonical 4-lap closed-loop racing scenarios
(zero-shot domain-transfer diagnostic) and merge cleanly into reports/race_engineer_v2_summary.json.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT / "decision_eval" / "src"
for p in (str(ROOT), str(SCRIPTS_DIR), str(SRC_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import eval_pace_accuracy as ep  # noqa: E402
import eval_race_engineer_v2 as re_eval  # noqa: E402
from decision_eval import paths  # noqa: E402
from decision_eval.runners import M2QwenLoRARunner  # noqa: E402
from self_jev.car_build_nn.builds import get_candidate_builds  # noqa: E402
from self_jev.car_build_nn.dataset import _load_frozen_driver  # noqa: E402
from self_jev.sim.physics import PhysicsConfig  # noqa: E402

MODEL_KEY = "m2-qwen3.5-0.8b-lora-s42"


class M2RacingAdapter:
    """Wraps M2QwenLoRARunner in the DecisionAdapter interface for RaceEnv."""

    def __init__(self, device: str = "cpu") -> None:
        self.name = MODEL_KEY
        self.runner = M2QwenLoRARunner(paths.M2_QWEN_LORA_S42_DIR, device=device)

    def decide(
        self,
        kind: str,
        state: str,
        question: str,
        options: list[str],
        option_descriptions: list[str],
        telemetry: dict[str, Any],
    ) -> str:
        res = self.runner.score_decision(state, question, options, option_descriptions)
        return res["choice"] or options[0]


def run_worker_subset(indices: list[int], device_str: str) -> list[dict[str, Any]]:
    torch.set_num_threads(2)
    device = torch.device(device_str)
    cal_path = ROOT / "self_jev" / "sim" / "calibrated_params.json"
    cfg = PhysicsConfig.from_json(cal_path) if cal_path.is_file() else PhysicsConfig()
    builds = get_candidate_builds()
    driver_model, obs_mean, obs_std, is_residual = _load_frozen_driver(re_eval.DEFAULT_DRIVER, device)
    adapter = M2RacingAdapter(device=device_str)
    out = []
    for idx in indices:
        sc = re_eval.CANONICAL_SCENARIOS[idx]
        row = re_eval.run_closed_loop_race(
            sc, adapter, driver_model, obs_mean, obs_std, is_residual, cfg, builds, device, laps_total=4
        )
        out.append(row)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker-indices", default=None, help="Comma-separated scenario indices for parallel worker")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()

    if args.worker_indices is not None:
        idxs = [int(x) for x in args.worker_indices.split(",") if x.strip()]
        rows = run_worker_subset(idxs, args.device)
        print(json.dumps(rows))
        return

    t0 = time.perf_counter()
    # Partition 9 scenarios across N workers (3 scenarios per worker with 3 workers)
    n_sc = len(re_eval.CANONICAL_SCENARIOS)
    chunks: list[list[int]] = [[] for _ in range(max(1, min(args.workers, n_sc)))]
    for i in range(n_sc):
        chunks[i % len(chunks)].append(i)

    script_path = str(Path(__file__).resolve())
    procs = []
    print(f"[M2 Racing Pilot] Launching {len(chunks)} workers across {n_sc} canonical 4-lap scenarios...", flush=True)
    for ch in chunks:
        cmd = [
            sys.executable,
            script_path,
            "--worker-indices",
            ",".join(str(i) for i in ch),
            "--device",
            args.device,
        ]
        procs.append((ch, subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)))

    runs_by_idx: dict[int, dict[str, Any]] = {}
    for ch, proc in procs:
        out, err = proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(f"Worker {ch} failed ({proc.returncode}): {err}")
        rows = json.loads(out)
        for idx, r in zip(ch, rows):
            runs_by_idx[idx] = r
            print(
                f"  {r['scenario_id']:20s} | fin={r['finished']} | t={r['finish_time_s']}s | "
                f"build={r['chosen_build']} | pits={r['pit_stops']} | modes={r['mode_history']}",
                flush=True,
            )

    runs = [runs_by_idx[i] for i in range(n_sc)]

    # Load existing race_engineer_v2_summary.json & AGX winner reference
    summary_path = ROOT / "reports" / "race_engineer_v2_summary.json"
    rep = json.loads(summary_path.read_text(encoding="utf-8"))
    w_doc = json.loads(re_eval.DEFAULT_WINNERS.read_text(encoding="utf-8"))
    winners = {tr.lower(): dict(d) for tr, d in w_doc["agx_winner_times_seconds"].items()}
    required_laps = int(rep.get("required_laps", 4))

    driver_runs = {r["scenario_id"]: r for r in rep["models"]["driver_only"]["runs"]}
    best_by_sc: dict[str, float | None] = {}
    for sc in re_eval.CANONICAL_SCENARIOS:
        sid = sc["id"]
        ts = [
            r["finish_time_s"]
            for m_k, m_v in rep["models"].items()
            if m_k != MODEL_KEY
            for r in (m_v.get("runs") or [])
            if r.get("scenario_id") == sid and r.get("finished") and r.get("finish_time_s") is not None
        ]
        m2_r = next(r for r in runs if r["scenario_id"] == sid)
        if m2_r["finished"] and m2_r["finish_time_s"] is not None:
            ts.append(m2_r["finish_time_s"])
        best_by_sc[sid] = min(ts) if ts else None

    ext_entries, best_entries = [], []
    mode_counts: Counter[str] = Counter()
    total_pits = 0
    total_calls = 0
    deltas_vs_base: list[float] = []
    fin_times: list[float] = []
    wins = ties = losses = 0
    tol = 0.5

    for r in runs:
        mode_counts.update(r["mode_history"])
        total_pits += r["pit_stops"]
        total_calls += r["decision_calls"]
        if r["finished"] and r["finish_time_s"] is not None:
            fin_times.append(float(r["finish_time_s"]))

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
        ext_entries.append((norm, ep.score_normalized_run(norm, reference_applicable=ref_t is not None, reference_time_s=ref_t)))
        bt = best_by_sc[r["scenario_id"]]
        best_entries.append((norm, ep.score_normalized_run(norm, reference_applicable=bt is not None, reference_time_s=bt)))

        br = driver_runs.get(r["scenario_id"])
        if br:
            bf, cf = bool(br["finished"]), bool(r["finished"])
            if cf and not bf:
                wins += 1
            elif bf and not cf:
                losses += 1
            elif not bf and not cf:
                ties += 1
            else:
                d = round(float(r["finish_time_s"]) - float(br["finish_time_s"]), 3)
                deltas_vs_base.append(d)
                if d < -tol:
                    wins += 1
                elif d > tol:
                    losses += 1
                else:
                    ties += 1

    ds = sorted(deltas_vs_base)
    model_entry = {
        "label": MODEL_KEY,
        "scenarios_n": len(runs),
        "decision_calls_total": total_calls,
        "pit_stops_total": total_pits,
        "mode_counts": dict(mode_counts),
        "avg_finish_time_s": round(sum(fin_times) / len(fin_times), 2) if fin_times else None,
        "mean_delta_vs_driver_only_s": round(sum(ds) / len(ds), 2) if ds else None,
        "median_delta_vs_driver_only_s": round(ds[len(ds) // 2], 2) if ds else None,
        "wtl_vs_driver_only": {"wins": wins, "ties": ties, "losses": losses, "tolerance_s": tol},
        "external_agx_reference": ep.summarize_scored_slice(ext_entries),
        "same_scenario_best_model": ep.summarize_scored_slice(best_entries),
        "runs": runs,
    }

    # Save standalone record + merge into race_engineer_v2_summary.json
    standalone_path = ROOT / "reports" / "m2_v5" / "m2_qwen_racing_pilot_9.json"
    standalone_path.write_text(json.dumps(model_entry, indent=2) + "\n", encoding="utf-8")

    rep["models"][MODEL_KEY] = model_entry
    summary_path.write_text(json.dumps(rep, indent=2) + "\n", encoding="utf-8")

    wall_s = round(time.perf_counter() - t0, 1)
    print(
        f"\n[M2 Racing Pilot DONE in {wall_s}s] {MODEL_KEY}: fin={len(fin_times)}/9, "
        f"avg={model_entry['avg_finish_time_s']}s, delta_vs_driver={model_entry['mean_delta_vs_driver_only_s']}s, "
        f"W-T-L={wins}-{ties}-{losses}, pits={total_pits}, modes={dict(mode_counts)}"
    )
    print(f"Updated -> {summary_path}")


if __name__ == "__main__":
    main()
