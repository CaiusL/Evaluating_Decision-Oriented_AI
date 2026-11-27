#!/usr/bin/env python3
"""F1 Previous-Race Feedback Study v2 Extension:
  - Adds matched zero-history query baseline Z (R2_zeroshot on seed_B)
  - Backfills kev-0.8b using the deployed KevRunner (decision_eval/src/decision_eval/runners.py)
  - Preserves reports/feedback_f1_summary.json untouched; writes reports/feedback_f1_v2_summary.json
  - Computes exact paired contrasts:
      * F_vs_A (R2_feedback - R2_actions_only)  [original planned contrast]
      * A_vs_Z (R2_actions_only - R2_zeroshot)  [post-execution diagnostic: action-history effect]
      * F_vs_Z (R2_feedback - R2_zeroshot)      [post-execution diagnostic: full feedback vs stateless]
  - Includes per-cell (9 cells) and per-seed-pair (5 blocks) breakdowns + nn-combo vs driver_only paired audit.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
DECISION_EVAL_SRC = REPO_ROOT / "decision_eval" / "src"
KEV_VENDOR = REPO_ROOT / "decision_eval" / "vendor" / "kev"
for p in (str(REPO_ROOT), str(SCRIPTS_DIR), str(DECISION_EVAL_SRC), str(KEV_VENDOR)):
    if p not in sys.path:
        sys.path.insert(0, p)

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_CACHE", str(REPO_ROOT / "models" / "hf_cache"))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(REPO_ROOT / "models" / "hf_cache"))
if Path("/tmp/kev_peft_only").is_dir() and "/tmp/kev_peft_only" not in sys.path:
    sys.path.insert(0, "/tmp/kev_peft_only")

import eval_feedback_f1 as f1  # noqa: E402
import eval_race_engineer_v2 as re_eval  # noqa: E402
from decision_eval.runners import KevRunner  # noqa: E402
from self_jev.car_build_nn.builds import get_candidate_builds  # noqa: E402
from self_jev.car_build_nn.dataset import _load_frozen_driver  # noqa: E402
from self_jev.sim.physics import PhysicsConfig  # noqa: E402

DEFAULT_OUT = REPO_ROOT / "reports" / "feedback_f1_v2_summary.json"
WTL_TOL_S = f1.WTL_TOL_S


class V2FeedbackAdapter(f1.FeedbackDecisionAdapter):
    """Extends FeedbackDecisionAdapter with the deployed KevRunner (kev-0.8b)."""

    def __init__(self, name: str, env_vars: dict[str, str] | None = None) -> None:
        self.kev_runner: KevRunner | None = None
        self.any_state_truncated = False
        self.kev_backend: str | None = None
        self.kev_base_source: str | None = None
        if name == "kev-0.8b":
            super().__init__("driver_only", env_vars=env_vars)
            self.name = "kev-0.8b"
            dev = "mps" if torch.backends.mps.is_available() else "cpu"
            backend = "auto" if dev == "mps" else "torch"
            self.kev_runner = KevRunner("jaredpalmer/kev-0.8b", dev, backend=backend)
            self.kev_backend = self.kev_runner.backend
            self.kev_base_source = self.kev_runner.base_source
        else:
            super().__init__(name, env_vars=env_vars)

    def decide(
        self,
        kind: str,
        state: str,
        question: str,
        options: list[str],
        option_descriptions: list[str],
        sc: dict[str, Any],
    ) -> tuple[str, bool]:
        if self.name == "kev-0.8b":
            assert self.kev_runner is not None
            full_state = f"{f1.STATIC_TASK_RULES}\n{state}"
            res = self.kev_runner.score_unified(
                state=full_state,
                question=question,
                options=options,
                option_descriptions=option_descriptions,
            )
            in_tok = int(res.get("input_tokens") or 0)
            self.max_prompt_tokens_seen = max(self.max_prompt_tokens_seen, in_tok)
            if res.get("state_truncated"):
                self.any_state_truncated = True
            idx = int(res["choice_index"])
            if 0 <= idx < len(options) and res.get("parse_ok", True):
                return options[idx], True
            return (options[1] if len(options) > 1 else options[0]), False
        return super().decide(kind, state, question, options, option_descriptions, sc)


def compare_paired_arms(base_list: list[dict[str, Any]], cand_list: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute exact paired comparison (cand - base) on matched query units."""
    n = len(base_list)
    rescued = regressed = both_failed = both_finished = 0
    wins = ties_finished = losses = 0
    deltas: list[float] = []
    build_changed = mode_changed = pit_changed = 0

    for b, c in zip(base_list, cand_list):
        bf, cf = b["finished"], c["finished"]
        if c["chosen_build"] != b["chosen_build"]:
            build_changed += 1
        if c["mode_history"] != b["mode_history"]:
            mode_changed += 1
        if c["pit_calls"] != b["pit_calls"]:
            pit_changed += 1

        if cf and not bf:
            rescued += 1
            wins += 1
        elif bf and not cf:
            regressed += 1
            losses += 1
        elif not bf and not cf:
            both_failed += 1
        else:
            both_finished += 1
            d = round(float(c["finish_time_s"]) - float(b["finish_time_s"]), 4)
            deltas.append(d)
            if d < -WTL_TOL_S:
                wins += 1
            elif d > WTL_TOL_S:
                losses += 1
            else:
                ties_finished += 1

    ds = sorted(deltas)
    return {
        "n_units": n,
        "base_finishes": sum(1 for r in base_list if r["finished"]),
        "cand_finishes": sum(1 for r in cand_list if r["finished"]),
        "rescued_failures": rescued,
        "newly_introduced_failures": regressed,
        "net_completion_pp": round(((rescued - regressed) / max(n, 1)) * 100.0, 2),
        "wtl": {
            "wins": wins,
            "ties_both_finished": ties_finished,
            "ties_both_failed": both_failed,
            "losses": losses,
        },
        "common_finishers_n": both_finished,
        "paired_mean_delta_s": round(sum(ds) / len(ds), 2) if ds else None,
        "paired_median_delta_s": round(ds[len(ds) // 2], 2) if ds else None,
        "behavior_changes": {
            "build_changed_units": build_changed,
            "mode_changed_units": mode_changed,
            "pit_changed_units": pit_changed,
        },
    }


def summarize_three_arm_model(
    units: list[dict[str, Any]],
    r1_list: list[dict[str, Any]],
    z_list: list[dict[str, Any]],
    a_list: list[dict[str, Any]],
    f_list: list[dict[str, Any]],
) -> dict[str, Any]:
    n = len(units)

    def _arm_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
        ts = [float(r["finish_time_s"]) for r in rows if r["finished"] and r["finish_time_s"] is not None]
        return {
            "finishes": sum(1 for r in rows if r["finished"]),
            "n_units": len(rows),
            "avg_time_finishers_s": round(sum(ts) / len(ts), 2) if ts else None,
            "pits_total": sum(r["pit_stops"] for r in rows),
            "invalid_outputs": sum(r["invalid_outputs"] for r in rows),
            "mode_counts": dict(Counter(m for r in rows for m in r["mode_history"])),
            "build_counts": dict(Counter(r["chosen_build"] for r in rows)),
        }

    by_cell: dict[str, dict[str, int]] = {}
    by_pair: dict[str, dict[str, int]] = {}
    for u, r1, z, a, f in zip(units, r1_list, z_list, a_list, f_list):
        cid = u["cell_id"]
        pid = f"{u['seed_a']}->{u['seed_b']}"
        for grp, key in ((by_cell, cid), (by_pair, pid)):
            rec = grp.setdefault(key, {"n": 0, "r1_fin": 0, "z_fin": 0, "a_fin": 0, "f_fin": 0, "f_minus_a": 0, "f_minus_z": 0})
            rec["n"] += 1
            rec["r1_fin"] += int(r1["finished"])
            rec["z_fin"] += int(z["finished"])
            rec["a_fin"] += int(a["finished"])
            rec["f_fin"] += int(f["finished"])
            rec["f_minus_a"] += int(f["finished"]) - int(a["finished"])
            rec["f_minus_z"] += int(f["finished"]) - int(z["finished"])

    r1_failed_units = sum(1 for r in r1_list if not r["finished"])
    r1_failed_repaired = sum(1 for r1, f in zip(r1_list, f_list) if not r1["finished"] and f["finished"])
    r1_passed_units = sum(1 for r in r1_list if r["finished"])
    r1_passed_preserved = sum(1 for r1, f in zip(r1_list, f_list) if r1["finished"] and f["finished"])

    return {
        "n_units": n,
        "arms": {
            "R1_support": _arm_stats(r1_list),
            "Z_zeroshot": _arm_stats(z_list),
            "A_actions_only": _arm_stats(a_list),
            "F_feedback": _arm_stats(f_list),
        },
        "contrasts": {
            "F_vs_A": compare_paired_arms(a_list, f_list),
            "A_vs_Z": compare_paired_arms(z_list, a_list),
            "F_vs_Z": compare_paired_arms(z_list, f_list),
        },
        "r1_conditional": {
            "r1_failed_units": r1_failed_units,
            "r1_failed_repaired_in_F": r1_failed_repaired,
            "r1_passed_units": r1_passed_units,
            "r1_passed_preserved_in_F": r1_passed_preserved,
        },
        "by_cell": by_cell,
        "by_seed_pair": by_pair,
    }


def run_v2_study(
    models: list[str],
    limit_pairs: int = 5,
    pair_idx: int | None = None,
    out_path: Path = DEFAULT_OUT,
    save_raw_dir: Path | None = None,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    env_vars = f1.load_dotenv(REPO_ROOT / ".env")
    seed_pairs = [f1.FRESH_SEED_PAIRS[pair_idx]] if pair_idx is not None else f1.FRESH_SEED_PAIRS[:limit_pairs]
    units = f1.build_units(seed_pairs)
    tracks = ("parkland", "highland", "crossover")

    device = torch.device("cpu")
    cal_path = REPO_ROOT / "self_jev" / "sim" / "calibrated_params.json"
    cfg = PhysicsConfig.from_json(cal_path) if cal_path.is_file() else PhysicsConfig()
    builds = get_candidate_builds()
    driver_model, obs_mean, obs_std, is_residual = _load_frozen_driver(re_eval.DEFAULT_DRIVER, device)

    adaptive_set = {"self-jev-minilm", "kev-0.8b", "frontier-llm"}
    results_models: dict[str, Any] = {}
    ref_raw_rows: dict[str, list[dict[str, Any]]] = {}

    for m_name in models:
        tm0 = time.perf_counter()
        adapter = V2FeedbackAdapter(m_name, env_vars=env_vars)
        print(f"\n[F1 v2 Study] Running {m_name} across {len(units)} units ({len(seed_pairs)} pairs x 9 cells)...", flush=True)

        if m_name in adaptive_set:
            r1_all: list[dict[str, Any]] = []
            z_all: list[dict[str, Any]] = []
            a_all: list[dict[str, Any]] = []
            f_all: list[dict[str, Any]] = []
            ordered_units: list[dict[str, Any]] = []

            for tr in tracks:
                tr_units = [u for u in units if u["track"] == tr]
                ordered_units.extend(tr_units)
                sup_scs = [u["support_sc"] for u in tr_units]
                qry_scs = [u["query_sc"] for u in tr_units]

                r1_rows = f1.run_batched_track_branch(
                    tr, sup_scs, [""] * len(sup_scs), adapter, driver_model, obs_mean, obs_std, is_residual, cfg, builds
                )
                z_rows = f1.run_batched_track_branch(
                    tr, qry_scs, [""] * len(qry_scs), adapter, driver_model, obs_mean, obs_std, is_residual, cfg, builds
                )
                a_prefs = [f1.format_history_prefix(r, include_outcomes=False) for r in r1_rows]
                a_rows = f1.run_batched_track_branch(
                    tr, qry_scs, a_prefs, adapter, driver_model, obs_mean, obs_std, is_residual, cfg, builds
                )
                f_prefs = [f1.format_history_prefix(r, include_outcomes=True) for r in r1_rows]
                f_rows = f1.run_batched_track_branch(
                    tr, qry_scs, f_prefs, adapter, driver_model, obs_mean, obs_std, is_residual, cfg, builds
                )

                r1_all.extend(r1_rows)
                z_all.extend(z_rows)
                a_all.extend(a_rows)
                f_all.extend(f_rows)
                print(
                    f"  {m_name:16s} | {tr:9s} | R1={sum(r['finished'] for r in r1_rows)}/{len(r1_rows)} "
                    f"| Z={sum(r['finished'] for r in z_rows)}/{len(z_rows)} "
                    f"| A={sum(r['finished'] for r in a_rows)}/{len(a_rows)} "
                    f"| F={sum(r['finished'] for r in f_rows)}/{len(f_rows)}",
                    flush=True,
                )

            if save_raw_dir is not None:
                save_raw_dir.mkdir(parents=True, exist_ok=True)
                tag = f"pair{pair_idx}" if pair_idx is not None else "all"
                (save_raw_dir / f"raw_{m_name}_{tag}.json").write_text(
                    json.dumps({
                        "model": m_name,
                        "units": ordered_units,
                        "r1": r1_all,
                        "z": z_all,
                        "a": a_all,
                        "f": f_all,
                        "wall_s": round(time.perf_counter() - tm0, 2),
                        "max_prompt_tokens_seen": adapter.max_prompt_tokens_seen or None,
                        "any_state_truncated": adapter.any_state_truncated,
                        "kev_backend": adapter.kev_backend,
                    }, indent=2),
                    encoding="utf-8",
                )

            summary = summarize_three_arm_model(ordered_units, r1_all, z_all, a_all, f_all)
            provenance_map = {
                "self-jev-minilm": "F-A original Pass 1; Z added as post-execution diagnostic on matched seed_B",
                "kev-0.8b": "Delayed planned comparator backfilled via deployed KevRunner (with Z/A/F from start)",
                "frontier-llm": "F-A original Pass 1 (cached); Z added as post-execution diagnostic on matched seed_B",
            }
            role_map = {
                "self-jev-minilm": "Frozen MiniLM cross-encoder; in-context F1 feedback vs A and Z",
                "kev-0.8b": f"Frozen Kev-0.8B decision scorer ({adapter.kev_backend or 'torch'}); in-context F1 feedback vs A and Z",
                "frontier-llm": "Frozen Frontier LLM; in-context F1 feedback vs A and Z",
            }
            summary["label"] = f"{m_name} ({env_vars.get('QWEN_MODEL', '')})" if m_name == "frontier-llm" else m_name
            summary["role"] = role_map[m_name]
            summary["provenance"] = provenance_map[m_name]
            summary["is_non_adaptive_reference"] = False
            summary["wall_s"] = round(time.perf_counter() - tm0, 2)
            summary["max_prompt_tokens_seen"] = adapter.max_prompt_tokens_seen or None
            summary["any_state_truncated"] = adapter.any_state_truncated
            if m_name == "frontier-llm":
                summary["api_usage"] = {
                    "endpoint": env_vars.get("QWEN_BASE_URL", ""),
                    "model": env_vars.get("QWEN_MODEL", ""),
                    "calls": adapter.api_calls,
                    "prompt_tokens": adapter.api_in_tokens,
                    "completion_tokens": adapter.api_out_tokens,
                    "latency_p50_ms": round(float(np.percentile(adapter.api_latencies_ms, 50)), 1) if adapter.api_latencies_ms else None,
                }
            results_models[m_name] = summary
        else:
            qry_all: list[dict[str, Any]] = []
            for tr in tracks:
                tr_units = [u for u in units if u["track"] == tr]
                qry_scs = [u["query_sc"] for u in tr_units]
                qry_rows = f1.run_batched_track_branch(
                    tr, qry_scs, [""] * len(qry_scs), adapter, driver_model, obs_mean, obs_std, is_residual, cfg, builds
                )
                qry_all.extend(qry_rows)
            ref_raw_rows[m_name] = qry_all
            role_map = {
                "nn-combo": "N1 / NN-0: Frozen 12-feature build-only MLP; no previous-race history input; no weight updates",
                "driver_only": "Frozen baseline (dry_balanced + balanced + stay_out)",
                "track_rule": "Frozen strong reference (Parkland->balanced; Highland/Crossover->push_max; not a proven ceiling)",
            }
            ref_sum = f1.summarize_static_reference(qry_all, m_name, role_map.get(m_name, "Frozen reference"))
            ref_sum["wall_s"] = round(time.perf_counter() - tm0, 2)
            results_models[m_name] = ref_sum
            print(f"  {m_name:16s} | query finishes={ref_sum['query_finishes']}/{ref_sum['n_units']} | avg={ref_sum['query_avg_time_finishers_s']}s", flush=True)

    nn_vs_driver_audit = None
    if "nn-combo" in ref_raw_rows and "driver_only" in ref_raw_rows:
        nn_vs_driver_audit = compare_paired_arms(ref_raw_rows["driver_only"], ref_raw_rows["nn-combo"])

    report = {
        "version": "f1_v2_with_zeroshot_and_kev_backfill",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "F1_own_run_factual_telemetry_three_arm_Z_A_F",
        "driver_weights": str(re_eval.DEFAULT_DRIVER.name),
        "weights_updated": False,
        "external_agx_gap_included_in_feedback": False,
        "branch_isolation_verified": True,
        "rng_stream_note": (
            "Each (cell_idx, seed) instantiates an independent np.random.default_rng(seed * 100 + cell_idx); "
            "no environmental draws are shared across cells. Both per-cell (9) and per-seed-pair (5) breakdowns are reported."
        ),
        "n_units": len(units),
        "n_cells": len(re_eval.CANONICAL_SCENARIOS),
        "seed_pairs": seed_pairs,
        "wtl_tolerance_s": WTL_TOL_S,
        "wall_s": round(time.perf_counter() - t0, 2),
        "models": results_models,
        "nn_combo_vs_driver_only_paired_audit": nn_vs_driver_audit,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved F1 v2 feedback study report -> {out_path} ({report['wall_s']}s)")
    return report


def merge_part1_and_kev(part1_path: Path, raw_dir: Path, out_path: Path) -> dict[str, Any]:
    """Merge part1 report with 5 parallel kev-0.8b raw pair files into final v2 summary."""
    base = json.loads(part1_path.read_text(encoding="utf-8"))
    units_all: list[dict[str, Any]] = []
    r1_all: list[dict[str, Any]] = []
    z_all: list[dict[str, Any]] = []
    a_all: list[dict[str, Any]] = []
    f_all: list[dict[str, Any]] = []
    max_tok = 0
    any_trunc = False
    wall_max = 0.0
    backend = "torch"

    for p_idx in range(5):
        raw_p = raw_dir / f"raw_kev-0.8b_pair{p_idx}.json"
        doc = json.loads(raw_p.read_text(encoding="utf-8"))
        units_all.extend(doc["units"])
        r1_all.extend(doc["r1"])
        z_all.extend(doc["z"])
        a_all.extend(doc["a"])
        f_all.extend(doc["f"])
        max_tok = max(max_tok, int(doc.get("max_prompt_tokens_seen") or 0))
        any_trunc = any_trunc or bool(doc.get("any_state_truncated"))
        wall_max = max(wall_max, float(doc.get("wall_s") or 0.0))
        backend = str(doc.get("kev_backend") or backend)

    kev_sum = summarize_three_arm_model(units_all, r1_all, z_all, a_all, f_all)
    kev_sum["label"] = "kev-0.8b"
    kev_sum["role"] = f"Frozen Kev-0.8B decision scorer ({backend}); in-context F1 feedback vs A and Z"
    kev_sum["provenance"] = "Delayed planned comparator backfilled via deployed KevRunner (with Z/A/F from start)"
    kev_sum["is_non_adaptive_reference"] = False
    kev_sum["wall_s"] = round(wall_max, 2)
    kev_sum["max_prompt_tokens_seen"] = max_tok
    kev_sum["any_state_truncated"] = any_trunc

    # Order models cleanly: kev-0.8b, self-jev-minilm, frontier-llm, nn-combo, driver_only, track_rule
    ordered_models: dict[str, Any] = {"kev-0.8b": kev_sum}
    for k, v in base["models"].items():
        if k != "kev-0.8b":
            ordered_models[k] = v
    base["models"] = ordered_models
    base["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    out_path.write_text(json.dumps(base, indent=2) + "\n", encoding="utf-8")
    print(f"Merged part1 + 5 kev-0.8b pairs -> {out_path}")
    return base


def main() -> None:
    ap = argparse.ArgumentParser(description="Run F1 v2 Feedback Study (Matched Z + Kev Backfill).")
    ap.add_argument(
        "--models",
        nargs="+",
        default=["kev-0.8b", "self-jev-minilm", "frontier-llm", "nn-combo", "driver_only", "track_rule"],
    )
    ap.add_argument("--limit-pairs", type=int, default=5)
    ap.add_argument("--pair-idx", type=int, default=None, help="Run a single seed pair index (0..4) for parallel workers")
    ap.add_argument("--save-raw-dir", type=Path, default=None)
    ap.add_argument("--merge-part1", type=Path, default=None, help="Merge part1 JSON with raw kev pair files in --save-raw-dir")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    if args.merge_part1 is not None:
        assert args.save_raw_dir is not None, "--save-raw-dir required with --merge-part1"
        merge_part1_and_kev(args.merge_part1, args.save_raw_dir, args.out)
    else:
        run_v2_study(args.models, limit_pairs=args.limit_pairs, pair_idx=args.pair_idx, out_path=args.out, save_raw_dir=args.save_raw_dir)


if __name__ == "__main__":
    main()
