#!/usr/bin/env python3
"""Milestone 6 & 7: Closed-Loop Racing Evaluation & Statistical Promotion Gates (racing_engineer_v1).

Evaluates frozen references and trained racing students across:
  1. Historical 45-case regression panel (seed_B = 2001..2005)
  2. Fresh confirmation seed bank (3501..3520 -> 180 cases; configurable via --confirm-seeds-n)

Models evaluated (all pit-free in Stage B when pits_enabled=False):
  - driver_only          (frozen baseline)
  - track_rule           (frozen strong reference)
  - lookup_9cell         (train-derived 9-cell build lookup + track_rule pace)
  - nn3_racing_v1        (compact 2-layer 64-unit MLP trained on racing_v1 rollouts)
  - qwen_lora_racing_v1  (fresh Racing Qwen3.5-0.8B LoRA trained from Base on racing_v1 rollouts)

Preserves all closed baselines (reports/feedback_f1_summary.json, reports/feedback_f1_v2_summary.json,
reports/m2_v5/, reports/race_engineer_v2_summary.json) untouched.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "decision_eval" / "src"
for p in (str(REPO_ROOT), str(SRC_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from decision_eval.racing_contract import (  # noqa: E402
    BUILD_OPTIONS,
    PACE_OPTIONS,
    SCHEMA_VERSION,
    action_spec_hash,
    extract_build_features,
    extract_pace_features,
    generate_scenarios_for_seeds,
    render_obs_text,
)
from decision_eval.racing_rollouts import (  # noqa: E402
    DEFAULT_DRIVER,
    load_frozen_sim_assets,
    run_closed_loop_policy,
)
from decision_eval.runners import M2QwenLoRARunner  # noqa: E402
from scripts.train_racing_v1_students import RacingNN3Head, softmax_np  # noqa: E402

DATA_DIR = REPO_ROOT / "data" / "racing_v1"
MODELS_DIR = REPO_ROOT / "models" / "racing_engineer_v1"
OUT_DIR = REPO_ROOT / "reports" / "racing_engineer_v1"
WTL_TOL_S = 0.50


class NN3PolicyAdapter:
    def __init__(self, ckpt_path: Path) -> None:
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Missing NN-3 checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.b_mean = ckpt["build_mean"]
        self.b_std = ckpt["build_std"]
        self.b_temp = float(ckpt["build_temperature"])
        self.b_net = RacingNN3Head(len(self.b_mean), len(BUILD_OPTIONS))
        self.b_net.load_state_dict(ckpt["build_state_dict"])
        self.b_net.eval()

        self.p_mean = ckpt["pace_mean"]
        self.p_std = ckpt["pace_std"]
        self.p_temp = float(ckpt["pace_temperature"])
        self.p_net = RacingNN3Head(len(self.p_mean), len(PACE_OPTIONS))
        self.p_net.load_state_dict(ckpt["pace_state_dict"])
        self.p_net.eval()

    @torch.inference_mode()
    def decide(self, kind: str, obs: dict[str, Any], opts: list[str], descs: list[str]) -> tuple[str, bool, float]:
        if kind == "build":
            x = (extract_build_features(obs) - self.b_mean) / self.b_std
            lg = self.b_net(torch.tensor(x, dtype=torch.float32).unsqueeze(0)).squeeze(0).numpy()
            probs = softmax_np(lg, self.b_temp)
            idx = int(np.argmax(probs))
            return BUILD_OPTIONS[idx], True, float(probs[idx])
        if kind == "pace":
            x = (extract_pace_features(obs) - self.p_mean) / self.p_std
            lg = self.p_net(torch.tensor(x, dtype=torch.float32).unsqueeze(0)).squeeze(0).numpy()
            probs = softmax_np(lg, self.p_temp)
            idx = int(np.argmax(probs))
            return PACE_OPTIONS[idx], True, float(probs[idx])
        return "stay_out", True, 1.0


class RacingQwenAdapter:
    def __init__(self, ckpt_dir: Path, device: str = "cpu") -> None:
        self.runner = M2QwenLoRARunner(ckpt_dir, device=device)
        self.q_map = {
            "build": "Select the pre-race car build to maximize 4-lap completion and minimize race time.",
            "pace": "Select the driving pace & ERS mode for this lap.",
            "pit": "Select whether to stay out or pit at the end of this lap.",
        }

    def decide(self, kind: str, obs: dict[str, Any], opts: list[str], descs: list[str]) -> tuple[str | None, bool, float]:
        st = render_obs_text(obs, include_contract=True)
        res = self.runner.score_decision(st, self.q_map[kind], opts, descs)
        return res["choice"], bool(res["parse_ok"]), float(res["confidence"])


def run_worker(panel: str, model_name: str, track: str, seeds: list[int], qwen_ckpt: Path | None) -> list[dict[str, Any]]:
    torch.set_num_threads(4 if model_name == "qwen_lora_racing_v1" else 1)
    device = torch.device("cpu")
    driver_model, obs_mean, obs_std, is_residual, cfg, builds = load_frozen_sim_assets(device=device)
    scs = [sc for sc in generate_scenarios_for_seeds(seeds) if sc["track"] == track]

    if model_name == "driver_only":
        fn = lambda kind, obs, opts, descs: ({"build": "dry_balanced", "pace": "balanced", "pit": "stay_out"}[kind], True, 1.0)
    elif model_name == "track_rule":
        fn = lambda kind, obs, opts, descs: (
            ("dry_balanced" if kind == "build" else (("balanced" if obs["track"] == "parkland" else "push_max") if kind == "pace" else "stay_out")),
            True,
            1.0,
        )
    elif model_name == "lookup_9cell":
        lookup = json.loads((DATA_DIR / "lookup_9cell_train.json").read_text(encoding="utf-8"))["lookup_9cell_build"]

        def fn(kind: str, obs: dict[str, Any], opts: list[str], descs: list[str]):
            if kind == "build":
                cid = f"{obs['track']}_{obs['weather_regime'].lower()}"
                return lookup.get(cid, "dry_balanced"), True, 1.0
            if kind == "pace":
                return ("balanced" if obs["track"] == "parkland" else "push_max"), True, 1.0
            return "stay_out", True, 1.0
    elif model_name == "nn3_racing_v1":
        adapter = NN3PolicyAdapter(MODELS_DIR / "nn3" / "nn3_selected.pt")
        fn = adapter.decide
    elif model_name == "qwen_lora_racing_v1":
        assert qwen_ckpt is not None
        adapter = RacingQwenAdapter(qwen_ckpt, device="cpu")
        fn = adapter.decide
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return run_closed_loop_policy(
        scs, fn, driver_model, obs_mean, obs_std, is_residual, cfg, builds, device, pits_enabled=False
    )


def summarize_model_panel(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ts = [float(r["finish_time_s"]) for r in rows if r["finished"] and r["finish_time_s"] is not None]
    fin = sum(1 for r in rows if r["finished"])
    return {
        "n": len(rows),
        "finishes": fin,
        "completion_rate_pct": round(100.0 * fin / max(len(rows), 1), 2),
        "avg_finish_time_s": round(float(np.mean(ts)), 2) if ts else None,
        "pits_total": sum(r["pit_stops"] for r in rows),
        "delegated_calls": sum(r["delegated_calls"] for r in rows),
        "illegal_proposed": sum(r["illegal_proposed"] for r in rows),
        "build_counts": dict(Counter(r["chosen_build"] for r in rows)),
        "mode_counts": dict(Counter(m for r in rows for m in r["mode_history"])),
    }


def paired_stats_with_ci(base_rows: list[dict[str, Any]], cand_rows: list[dict[str, Any]], n_boot: int = 2000) -> dict[str, Any]:
    n = len(base_rows)
    comp_diffs = np.array([int(c["finished"]) - int(b["finished"]) for b, c in zip(base_rows, cand_rows)], dtype=np.float64)
    rescued = int(np.sum(comp_diffs > 0))
    regressed = int(np.sum(comp_diffs < 0))
    both_fail = sum(1 for b, c in zip(base_rows, cand_rows) if not b["finished"] and not c["finished"])

    deltas = []
    wins = ties_fin = losses = 0
    for b, c in zip(base_rows, cand_rows):
        bf, cf = b["finished"], c["finished"]
        if cf and not bf:
            wins += 1
        elif bf and not cf:
            losses += 1
        elif bf and cf:
            d = float(c["finish_time_s"]) - float(b["finish_time_s"])
            deltas.append(d)
            if d < -WTL_TOL_S:
                wins += 1
            elif d > WTL_TOL_S:
                losses += 1
            else:
                ties_fin += 1

    rng = np.random.default_rng(42)
    b_idx = rng.integers(0, n, size=(n_boot, n))
    comp_boot = comp_diffs[b_idx].mean(axis=1) * 100.0
    comp_ci95_lo = round(float(np.percentile(comp_boot, 5)), 2)

    time_mean = round(float(np.mean(deltas)), 3) if deltas else None
    time_med = round(float(np.median(deltas)), 3) if deltas else None
    if len(deltas) >= 2:
        d_arr = np.array(deltas, dtype=np.float64)
        t_idx = rng.integers(0, len(d_arr), size=(n_boot, len(d_arr)))
        t_boot = d_arr[t_idx].mean(axis=1)
        time_ci95_hi = round(float(np.percentile(t_boot, 95)), 3)
    else:
        time_ci95_hi = time_mean

    return {
        "n": n,
        "base_finishes": sum(1 for r in base_rows if r["finished"]),
        "cand_finishes": sum(1 for r in cand_rows if r["finished"]),
        "rescued": rescued,
        "regressed": regressed,
        "completion_diff_pp": round(float(comp_diffs.mean() * 100.0), 2),
        "completion_ci95_lower_pp": comp_ci95_lo,
        "common_finishers_n": len(deltas),
        "mean_paired_delta_s": time_mean,
        "median_paired_delta_s": time_med,
        "time_ci95_upper_s": time_ci95_hi,
        "wtl": {"wins": wins, "ties_finished": ties_fin, "ties_both_failed": both_fail, "losses": losses},
    }


def evaluate_historical_gate(arm_sum: dict[str, Any], vs_drv: dict[str, Any], vs_trk: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "finishes_ge_38": arm_sum["finishes"] >= 38,
        "regressions_vs_driver_le_1": vs_drv["regressed"] <= 1,
        "mean_delta_vs_driver_le_neg5s": (vs_drv["mean_paired_delta_s"] is not None and vs_drv["mean_paired_delta_s"] <= -5.0),
        "mean_delta_vs_track_rule_le_0s": (vs_trk["mean_paired_delta_s"] is not None and vs_trk["mean_paired_delta_s"] <= 0.0),
        "illegal_actions_eq_0": arm_sum["illegal_proposed"] == 0,
        "pits_eq_0": arm_sum["pits_total"] == 0,
    }
    return {"passed": all(checks.values()), "checks": checks}


def evaluate_fresh_gate(vs_drv: dict[str, Any], vs_trk: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "comp_diff_vs_driver_ge_0": vs_drv["completion_diff_pp"] >= 0.0,
        "comp_ci95_lo_vs_driver_gt_neg5": vs_drv["completion_ci95_lower_pp"] > -5.0,
        "comp_diff_vs_track_rule_ge_0": vs_trk["completion_diff_pp"] >= 0.0,
        "comp_ci95_lo_vs_track_rule_gt_neg5": vs_trk["completion_ci95_lower_pp"] > -5.0,
        "time_delta_vs_driver_le_neg5s": (vs_drv["mean_paired_delta_s"] is not None and vs_drv["mean_paired_delta_s"] <= -5.0),
        "time_ci95_hi_vs_driver_lt_0s": (vs_drv["time_ci95_upper_s"] is not None and vs_drv["time_ci95_upper_s"] < 0.0),
        "time_delta_vs_track_rule_le_0s": (vs_trk["mean_paired_delta_s"] is not None and vs_trk["mean_paired_delta_s"] <= 0.0),
        "time_ci95_hi_vs_track_rule_le_2s": (vs_trk["time_ci95_upper_s"] is not None and vs_trk["time_ci95_upper_s"] <= 2.0),
    }
    return {"passed": all(checks.values()), "checks": checks}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", default=None, help="<panel>:<model>:<track>")
    ap.add_argument("--seeds-json", default=None)
    ap.add_argument("--qwen-ckpt", type=Path, default=MODELS_DIR / "qwen_lora" / "racing_qwen3.5_0.8b_lora_s42")
    ap.add_argument("--out-file", type=Path, default=None)
    ap.add_argument("--confirm-seeds-n", type=int, default=20, help="Seeds per cell for fresh confirmation (20 => 180 races)")
    ap.add_argument("--qwen-confirm-seeds-n", type=int, default=5, help="Seeds per cell for Qwen on fresh confirmation (5 => 45 races)")
    args = ap.parse_args()

    if args.worker:
        panel, model_name, track = args.worker.split(":", 2)
        seeds = json.loads(args.seeds_json)
        rows = run_worker(panel, model_name, track, seeds, args.qwen_ckpt)
        payload = json.dumps(rows)
        if args.out_file:
            args.out_file.write_text(payload, encoding="utf-8")
        else:
            print(payload)
        return

    t0 = time.perf_counter()
    manifest = json.loads((DATA_DIR / "split_manifest.json").read_text(encoding="utf-8"))
    hist_seeds = manifest["splits"]["historical_regression"]["seeds"]
    confirm_seeds = manifest["splits"]["confirm"]["seeds"][: args.confirm_seeds_n]
    qwen_confirm_seeds = manifest["splits"]["confirm"]["seeds"][: args.qwen_confirm_seeds_n]

    has_qwen = (args.qwen_ckpt / "lora_adapters.pt").is_file()
    light_models = ["driver_only", "track_rule", "lookup_9cell", "nn3_racing_v1"]

    tmp_dir = OUT_DIR / "_tmp_eval_workers"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    script_path = str(Path(__file__).resolve())

    light_jobs: list[tuple[str, list[str], Path]] = []
    for panel, seeds in (("historical_45", hist_seeds), ("fresh_confirm", confirm_seeds)):
        s_json = json.dumps(seeds)
        for m in light_models:
            for tr in ("parkland", "highland", "crossover"):
                fp = tmp_dir / f"{panel}_{m}_{tr}.json"
                light_jobs.append((
                    f"{panel}:{m}:{tr}",
                    [sys.executable, script_path, "--worker", f"{panel}:{m}:{tr}", "--seeds-json", s_json, "--out-file", str(fp)],
                    fp,
                ))

    heavy_jobs: list[tuple[str, list[str], Path]] = []
    if has_qwen:
        for panel, seeds in (("historical_45", hist_seeds), ("fresh_confirm_qwen", qwen_confirm_seeds)):
            s_json = json.dumps(seeds)
            for tr in ("parkland", "highland", "crossover"):
                fp = tmp_dir / f"{panel}_qwen_lora_racing_v1_{tr}.json"
                heavy_jobs.append((
                    f"{panel}:qwen_lora_racing_v1:{tr}",
                    [
                        sys.executable, script_path,
                        "--worker", f"{panel}:qwen_lora_racing_v1:{tr}",
                        "--seeds-json", s_json,
                        "--qwen-ckpt", str(args.qwen_ckpt),
                        "--out-file", str(fp),
                    ],
                    fp,
                ))

    out_map: dict[str, list[dict[str, Any]]] = {}

    def _run_pool(job_list: list[tuple[str, list[str], Path]], max_conc: int) -> None:
        pending = list(job_list)
        running: list[tuple[str, subprocess.Popen[str], Path]] = []
        while pending or running:
            while pending and len(running) < max_conc:
                tag, cmd, fp = pending.pop(0)
                running.append((tag, subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True), fp))
            still = []
            for tag, proc, fp in running:
                ret = proc.poll()
                if ret is None:
                    still.append((tag, proc, fp))
                    continue
                _, err = proc.communicate()
                if ret != 0:
                    raise RuntimeError(f"Eval worker {tag} failed ({ret}): {err}")
                out_map[tag] = json.loads(fp.read_text(encoding="utf-8"))
                print(f"  completed {tag} ({time.perf_counter()-t0:.1f}s elapsed)", flush=True)
            running = still
            if running:
                time.sleep(0.2)

    print(f"[Closed-Loop Eval] Running {len(light_jobs)} light jobs (8 concurrent) + {len(heavy_jobs)} Qwen jobs (2 concurrent)...", flush=True)
    _run_pool(light_jobs, max_conc=8)
    if heavy_jobs:
        _run_pool(heavy_jobs, max_conc=2)

    def _collect(panel: str, m: str) -> list[dict[str, Any]]:
        return [r for tr in ("parkland", "highland", "crossover") for r in out_map[f"{panel}:{m}:{tr}"]]

    hist_models = light_models + (["qwen_lora_racing_v1"] if has_qwen else [])
    hist_rows = {m: _collect("historical_45", m) for m in hist_models}
    hist_summary: dict[str, Any] = {}
    for m in hist_models:
        ms = summarize_model_panel(hist_rows[m])
        vs_d = paired_stats_with_ci(hist_rows["driver_only"], hist_rows[m]) if m != "driver_only" else None
        vs_t = paired_stats_with_ci(hist_rows["track_rule"], hist_rows[m]) if m != "track_rule" else None
        gate = evaluate_historical_gate(ms, vs_d, vs_t) if (vs_d and vs_t) else None
        hist_summary[m] = {"summary": ms, "vs_driver_only": vs_d, "vs_track_rule": vs_t, "historical_45_gate": gate}
        print(
            f"  [HIST-45] {m:20s} | fin={ms['finishes']}/{ms['n']} | avg={ms['avg_finish_time_s']}s | "
            f"d_drv={vs_d['mean_paired_delta_s'] if vs_d else 0.0}s | d_trk={vs_t['mean_paired_delta_s'] if vs_t else 0.0}s | "
            f"gate={gate['passed'] if gate else 'ref'}",
            flush=True,
        )

    fresh_rows = {m: _collect("fresh_confirm", m) for m in light_models}
    fresh_summary: dict[str, Any] = {}
    for m in light_models:
        ms = summarize_model_panel(fresh_rows[m])
        vs_d = paired_stats_with_ci(fresh_rows["driver_only"], fresh_rows[m]) if m != "driver_only" else None
        vs_t = paired_stats_with_ci(fresh_rows["track_rule"], fresh_rows[m]) if m != "track_rule" else None
        gate = evaluate_fresh_gate(vs_d, vs_t) if (vs_d and vs_t) else None
        fresh_summary[m] = {"summary": ms, "vs_driver_only": vs_d, "vs_track_rule": vs_t, "fresh_confirm_gate": gate}
        print(
            f"  [FRESH-{ms['n']}] {m:20s} | fin={ms['finishes']}/{ms['n']} ({ms['completion_rate_pct']}%) | avg={ms['avg_finish_time_s']}s | "
            f"d_drv={vs_d['mean_paired_delta_s'] if vs_d else 0.0}s | d_trk={vs_t['mean_paired_delta_s'] if vs_t else 0.0}s | "
            f"gate={gate['passed'] if gate else 'ref'}",
            flush=True,
        )

    if has_qwen:
        q_rows = _collect("fresh_confirm_qwen", "qwen_lora_racing_v1")
        q_sids = {r["scenario_id"] for r in q_rows}
        d_sub = [r for r in fresh_rows["driver_only"] if r["scenario_id"] in q_sids]
        t_sub = [r for r in fresh_rows["track_rule"] if r["scenario_id"] in q_sids]
        ms = summarize_model_panel(q_rows)
        vs_d = paired_stats_with_ci(d_sub, q_rows)
        vs_t = paired_stats_with_ci(t_sub, q_rows)
        gate = evaluate_fresh_gate(vs_d, vs_t)
        fresh_summary["qwen_lora_racing_v1"] = {
            "summary": ms,
            "vs_driver_only": vs_d,
            "vs_track_rule": vs_t,
            "fresh_confirm_gate": gate,
            "matched_subset_seeds": qwen_confirm_seeds,
        }
        print(
            f"  [FRESH-{ms['n']}] {'qwen_lora_racing_v1':20s} | fin={ms['finishes']}/{ms['n']} ({ms['completion_rate_pct']}%) | avg={ms['avg_finish_time_s']}s | "
            f"d_drv={vs_d['mean_paired_delta_s']}s | d_trk={vs_t['mean_paired_delta_s']}s | gate={gate['passed']}",
            flush=True,
        )

    report = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "action_spec_hash": action_spec_hash(),
        "driver_weights": str(DEFAULT_DRIVER.name),
        "pits_enabled": False,
        "ui_display_label": "self-jev-minilm",
        "wall_s": round(time.perf_counter() - t0, 1),
        "historical_45_regression": {
            "seeds": hist_seeds,
            "models": hist_summary,
        },
        "fresh_confirmation": {
            "seeds": confirm_seeds,
            "models": fresh_summary,
        },
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "closed_loop_eval_summary.json"
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved closed-loop evaluation summary -> {out_path} ({report['wall_s']}s)")


if __name__ == "__main__":
    main()
