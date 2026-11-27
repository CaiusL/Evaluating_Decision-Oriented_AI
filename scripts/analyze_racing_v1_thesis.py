#!/usr/bin/env python3
"""Bounded Thesis Analysis Pass for racing_engineer_v1.

Reads existing frozen logs in reports/racing_engineer_v1/_tmp_eval_workers/,
reports/racing_engineer_v1/milestone1_diagnostics.json, and data/racing_v1/*.jsonl
to produce:
  1. Matched panel tables (Fresh-180, Matched Fresh-45, Historical-45, Transfer-27)
  2. 4-cell paired reliability transitions, multi-tolerance W-T-L, and conditional
     paired timing with both cell-stratified (primary) and seed-block (sensitivity)
     two-sided 95% CIs and one-sided 95% bounds
  3. All-model common-finisher timing & successful-completion-by-time curves G_m(t)
  4. Constant-policy audit (live traces + offline dev/cal acceptable-set accuracy)
  5. Complete dataset & target-accounting breakdown (eligible vs all_failed,
     singleton vs tie-set, acceptable-set membership, per-track distribution)

Writes:
  - reports/racing_engineer_v1/thesis_analysis_v1.json
  - reports/racing_engineer_v1/completion_by_time_curves.png (if matplotlib available)
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "racing_v1"
OUT_DIR = REPO_ROOT / "reports" / "racing_engineer_v1"
WORKERS_DIR = OUT_DIR / "_tmp_eval_workers"

TRACKS = ("parkland", "highland", "crossover")
LIGHT_MODELS = ("driver_only", "track_rule", "lookup_9cell", "nn3_racing_v1")
ALL_MODELS = ("driver_only", "track_rule", "lookup_9cell", "nn3_racing_v1", "qwen_lora_racing_v1")
WTL_TOLS = (0.0, 0.5, 1.0, 2.0)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def load_panel_rows(panel_prefix: str, model: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for tr in TRACKS:
        fp = WORKERS_DIR / f"{panel_prefix}_{model}_{tr}.json"
        if not fp.is_file():
            raise FileNotFoundError(f"Missing worker log: {fp}")
        rows.extend(json.loads(fp.read_text(encoding="utf-8")))
    rows.sort(key=lambda r: (r["cell_id"], int(r["seed"])))
    return rows


def summarize_model(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ts = [float(r["finish_time_s"]) for r in rows if r["finished"] and r["finish_time_s"] is not None]
    fin = sum(1 for r in rows if r["finished"])
    statuses = Counter(int(r["status"]) for r in rows)
    fails_by_track = Counter(r["track"] for r in rows if not r["finished"])
    fails_by_cell = Counter(r["cell_id"] for r in rows if not r["finished"])
    return {
        "n": len(rows),
        "finishes": fin,
        "completion_rate_pct": round(100.0 * fin / max(len(rows), 1), 2),
        "mean_time_finishers_s": round(float(np.mean(ts)), 2) if ts else None,
        "median_time_finishers_s": round(float(np.median(ts)), 2) if ts else None,
        "pits_total": sum(int(r["pit_stops"]) for r in rows),
        "illegal_proposed": sum(int(r["illegal_proposed"]) for r in rows),
        "delegated_calls": sum(int(r["delegated_calls"]) for r in rows),
        "status_counts": {str(k): v for k, v in sorted(statuses.items())},
        "failures_by_track": dict(sorted(fails_by_track.items())),
        "failures_by_cell": dict(sorted(fails_by_cell.items())),
        "build_counts": dict(Counter(r["chosen_build"] for r in rows)),
        "mode_counts": dict(Counter(m for r in rows for m in r["mode_history"])),
    }


def _bootstrap_cell_stratified(
    rows_a: list[dict[str, Any]],
    rows_b: list[dict[str, Any]],
    n_boot: int = 4000,
    seed: int = 42,
) -> dict[str, Any]:
    """Cell-stratified paired bootstrap (exact match for independent seed*100+cell_idx streams)."""
    rng = np.random.default_rng(seed)
    by_cell: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(rows_a):
        by_cell[r["cell_id"]].append(i)
    cell_idxs = [np.array(idxs, dtype=np.int64) for _, idxs in sorted(by_cell.items())]

    comp_diffs = np.array([int(b["finished"]) - int(a["finished"]) for a, b in zip(rows_a, rows_b)], dtype=np.float64)
    both_fin = np.array([bool(a["finished"] and b["finished"]) for a, b in zip(rows_a, rows_b)], dtype=bool)
    time_diffs = np.array(
        [
            (float(b["finish_time_s"]) - float(a["finish_time_s"])) if (a["finished"] and b["finished"]) else np.nan
            for a, b in zip(rows_a, rows_b)
        ],
        dtype=np.float64,
    )

    comp_boots = np.empty(n_boot, dtype=np.float64)
    time_boots = []
    for bi in range(n_boot):
        sampled = np.concatenate([rng.choice(ci, size=len(ci), replace=True) for ci in cell_idxs])
        comp_boots[bi] = comp_diffs[sampled].mean() * 100.0
        bf = both_fin[sampled]
        if np.any(bf):
            time_boots.append(float(np.nanmean(time_diffs[sampled][bf])))

    t_arr = np.array(time_boots, dtype=np.float64) if time_boots else np.array([], dtype=np.float64)
    return {
        "method": "cell_stratified_paired_bootstrap",
        "n_boot": n_boot,
        "comp_ci95_2sided_pp": [round(float(np.percentile(comp_boots, 2.5)), 2), round(float(np.percentile(comp_boots, 97.5)), 2)],
        "comp_ci95_1sided_lower_pp": round(float(np.percentile(comp_boots, 5.0)), 2),
        "time_ci95_2sided_s": [round(float(np.percentile(t_arr, 2.5)), 3), round(float(np.percentile(t_arr, 97.5)), 3)] if len(t_arr) else None,
        "time_ci95_1sided_upper_s": round(float(np.percentile(t_arr, 95.0)), 3) if len(t_arr) else None,
    }


def _bootstrap_seed_block(
    rows_a: list[dict[str, Any]],
    rows_b: list[dict[str, Any]],
    n_boot: int = 4000,
    seed: int = 42,
) -> dict[str, Any]:
    """Seed-block paired bootstrap sensitivity check."""
    rng = np.random.default_rng(seed)
    by_seed: dict[int, list[int]] = defaultdict(list)
    for i, r in enumerate(rows_a):
        by_seed[int(r["seed"])].append(i)
    seed_blocks = [np.array(idxs, dtype=np.int64) for _, idxs in sorted(by_seed.items())]
    n_blocks = len(seed_blocks)

    comp_diffs = np.array([int(b["finished"]) - int(a["finished"]) for a, b in zip(rows_a, rows_b)], dtype=np.float64)
    both_fin = np.array([bool(a["finished"] and b["finished"]) for a, b in zip(rows_a, rows_b)], dtype=bool)
    time_diffs = np.array(
        [
            (float(b["finish_time_s"]) - float(a["finish_time_s"])) if (a["finished"] and b["finished"]) else np.nan
            for a, b in zip(rows_a, rows_b)
        ],
        dtype=np.float64,
    )

    comp_boots = np.empty(n_boot, dtype=np.float64)
    time_boots = []
    for bi in range(n_boot):
        chosen = rng.integers(0, n_blocks, size=n_blocks)
        sampled = np.concatenate([seed_blocks[c] for c in chosen])
        comp_boots[bi] = comp_diffs[sampled].mean() * 100.0
        bf = both_fin[sampled]
        if np.any(bf):
            time_boots.append(float(np.nanmean(time_diffs[sampled][bf])))

    t_arr = np.array(time_boots, dtype=np.float64) if time_boots else np.array([], dtype=np.float64)
    return {
        "method": "seed_block_paired_bootstrap",
        "n_blocks": n_blocks,
        "n_boot": n_boot,
        "comp_ci95_2sided_pp": [round(float(np.percentile(comp_boots, 2.5)), 2), round(float(np.percentile(comp_boots, 97.5)), 2)],
        "comp_ci95_1sided_lower_pp": round(float(np.percentile(comp_boots, 5.0)), 2),
        "time_ci95_2sided_s": [round(float(np.percentile(t_arr, 2.5)), 3), round(float(np.percentile(t_arr, 97.5)), 3)] if len(t_arr) else None,
        "time_ci95_1sided_upper_s": round(float(np.percentile(t_arr, 95.0)), 3) if len(t_arr) else None,
    }


def paired_comparison(base_rows: list[dict[str, Any]], cand_rows: list[dict[str, Any]]) -> dict[str, Any]:
    assert [r["scenario_id"] for r in base_rows] == [r["scenario_id"] for r in cand_rows], "Scenario mismatch!"
    n = len(base_rows)
    both_finish = rescue = regression = both_fail = 0
    deltas: list[float] = []
    for b, c in zip(base_rows, cand_rows):
        bf, cf = bool(b["finished"]), bool(c["finished"])
        if bf and cf:
            both_finish += 1
            deltas.append(float(c["finish_time_s"]) - float(b["finish_time_s"]))
        elif cf and not bf:
            rescue += 1
        elif bf and not cf:
            regression += 1
        else:
            both_fail += 1

    wtl_by_tol: dict[str, dict[str, int]] = {}
    for tol in WTL_TOLS:
        w = tf = l = 0
        w += rescue
        l += regression
        for d in deltas:
            if d < -tol:
                w += 1
            elif d > tol:
                l += 1
            else:
                tf += 1
        wtl_by_tol[f"{tol:.1f}s"] = {
            "wins": w,
            "ties_finished": tf,
            "ties_both_failed": both_fail,
            "losses": l,
        }

    return {
        "n": n,
        "base_finishes": both_finish + regression,
        "cand_finishes": both_finish + rescue,
        "completion_diff_races": rescue - regression,
        "completion_diff_pp": round(100.0 * (rescue - regression) / max(n, 1), 2),
        "paired_outcomes": {
            "both_finish": both_finish,
            "rescue": rescue,
            "regression": regression,
            "both_fail": both_fail,
        },
        "common_finishers_n": len(deltas),
        "mean_paired_delta_s": round(float(np.mean(deltas)), 3) if deltas else None,
        "median_paired_delta_s": round(float(np.median(deltas)), 3) if deltas else None,
        "wtl_by_tolerance": wtl_by_tol,
        "bootstrap_cell_stratified": _bootstrap_cell_stratified(base_rows, cand_rows),
        "bootstrap_seed_block": _bootstrap_seed_block(base_rows, cand_rows),
    }


def analyze_panel(model_rows: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    models = list(model_rows.keys())
    # All-model common finishers
    common_ids = set(r["scenario_id"] for r in model_rows[models[0]] if r["finished"])
    for m in models[1:]:
        common_ids &= set(r["scenario_id"] for r in model_rows[m] if r["finished"])

    common_finisher_means: dict[str, float | None] = {}
    for m in models:
        ts = [float(r["finish_time_s"]) for r in model_rows[m] if r["scenario_id"] in common_ids]
        common_finisher_means[m] = round(float(np.mean(ts)), 2) if ts else None

    per_model: dict[str, Any] = {}
    for m in models:
        per_model[m] = {
            "summary": summarize_model(model_rows[m]),
            "mean_time_all_model_common_s": common_finisher_means[m],
            "vs_driver_only": paired_comparison(model_rows["driver_only"], model_rows[m]) if m != "driver_only" else None,
            "vs_track_rule": paired_comparison(model_rows["track_rule"], model_rows[m]) if m != "track_rule" else None,
        }

    # Successful-completion-by-time curve checkpoints G_m(t)
    t_grid = [560, 580, 600, 620, 640, 660, 680, 700, 750, 800]
    g_curves: dict[str, dict[str, float]] = {}
    for m in models:
        n = len(model_rows[m])
        g_curves[m] = {
            f"{t}s": round(sum(1 for r in model_rows[m] if r["finished"] and float(r["finish_time_s"]) <= t) / n, 4)
            for t in t_grid
        }

    return {
        "n_races": len(model_rows[models[0]]),
        "all_model_common_finishers_n": len(common_ids),
        "models": per_model,
        "completion_by_time_checkpoints": g_curves,
    }


def analyze_transfer_27() -> dict[str, Any]:
    m1 = json.loads((OUT_DIR / "milestone1_diagnostics.json").read_text(encoding="utf-8"))
    pb = m1["prompt_bridge_27"]
    out_arms: dict[str, Any] = {}
    for arm, data in pb["arms"].items():
        paired = pb["paired_vs_driver_only"].get(arm)
        paired_outcomes = None
        if paired is not None:
            paired_outcomes = {
                "both_finish": paired["common_finishers_n"],
                "rescue": paired["rescued"],
                "regression": paired["regressed"],
                "both_fail": paired["wtl"]["ties_both_failed"],
            }
        out_arms[arm] = {
            "n": data["n"],
            "finishes": data["finishes"],
            "completion_rate_pct": round(100.0 * data["finishes"] / max(data["n"], 1), 2),
            "mean_time_finishers_s": data["avg_finish_time_s"],
            "pits_total": data["pits_total"],
            "illegal_proposed": data["illegal_proposed"],
            "build_counts": data["build_counts"],
            "mode_counts": data["mode_counts"],
            "paired_outcomes_vs_driver_only": paired_outcomes,
            "vs_driver_only": paired,
        }
    return {
        "seeds": pb["seeds"],
        "n_races": 27,
        "arms": out_arms,
        "paired_verified_vs_current_prompt": pb.get("paired_verified_vs_current_prompt"),
    }


def analyze_dataset_and_constant_baselines() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for sp in ("train", "dev", "cal"):
        rows = load_jsonl(DATA_DIR / f"{sp}.jsonl")
        sp_info: dict[str, Any] = {"total_rows": len(rows)}
        for dt in ("build", "pace"):
            sub = [r for r in rows if r["decision_type"] == dt]
            elig = [r for r in sub if r["label_info"]["status"] != "all_failed"]
            all_fail = [r for r in sub if r["label_info"]["status"] == "all_failed"]
            resolved = [r for r in elig if r["label_info"]["status"] == "resolved"]
            tie_set = [r for r in elig if r["label_info"]["status"] == "tie_acceptable_set"]

            opts = sub[0]["options"]
            acc_membership = {
                opt: sum(1 for r in elig if opt in r["acceptable_option_ids"]) for opt in opts
            }
            acc_membership_pct = {
                opt: round(100.0 * cnt / max(len(elig), 1), 2) for opt, cnt in acc_membership.items()
            }
            exact_gold_elig = {opt: sum(1 for r in elig if r["gold_option_id"] == opt) for opt in opts}
            exact_gold_elig_pct = {
                opt: round(100.0 * cnt / max(len(elig), 1), 2) for opt, cnt in exact_gold_elig.items()
            }
            resolved_singleton = {opt: sum(1 for r in resolved if r["gold_option_id"] == opt) for opt in opts}

            # Per-track acceptable-set membership (especially critical for parkland vs highland/crossover pace)
            by_track: dict[str, Any] = {}
            for tr in TRACKS:
                tr_elig = [r for r in elig if r["telemetry"]["track"] == tr]
                by_track[tr] = {
                    "eligible_n": len(tr_elig),
                    "gold_counts": dict(Counter(r["gold_option_id"] for r in tr_elig)),
                    "acceptable_membership_counts": {
                        opt: sum(1 for r in tr_elig if opt in r["acceptable_option_ids"]) for opt in opts
                    },
                    "acceptable_membership_pct": {
                        opt: round(100.0 * sum(1 for r in tr_elig if opt in r["acceptable_option_ids"]) / max(len(tr_elig), 1), 2)
                        for opt in opts
                    },
                }

            # Heuristic baseline accuracy on eligible rows
            heur_accs: dict[str, float] = {}
            if dt == "build":
                # constant dry_balanced
                heur_accs["const_dry_balanced_acc_pct"] = acc_membership_pct["dry_balanced"]
                # lookup_9cell (highland_dry -> wet_high_downforce, else dry_balanced)
                lk_ok = sum(
                    1 for r in elig
                    if ("wet_high_downforce" if r["provenance"]["cell_id"] == "highland_dry" else "dry_balanced") in r["acceptable_option_ids"]
                )
                heur_accs["lookup_9cell_acc_pct"] = round(100.0 * lk_ok / max(len(elig), 1), 2)
            else:
                heur_accs["const_push_max_acc_pct"] = acc_membership_pct["push_max"]
                heur_accs["const_balanced_acc_pct"] = acc_membership_pct["balanced"]
                tr_ok = sum(
                    1 for r in elig
                    if ("balanced" if r["telemetry"]["track"] == "parkland" else "push_max") in r["acceptable_option_ids"]
                )
                heur_accs["track_rule_pace_acc_pct"] = round(100.0 * tr_ok / max(len(elig), 1), 2)

            sp_info[dt] = {
                "total": len(sub),
                "eligible": len(elig),
                "all_failed_excluded": len(all_fail),
                "resolved_singleton": len(resolved),
                "tie_acceptable_set": len(tie_set),
                "all_rows_gold_histogram_incl_all_failed": dict(Counter(r["gold_option_id"] for r in sub)),
                "eligible_gold_counts": exact_gold_elig,
                "eligible_gold_pct": exact_gold_elig_pct,
                "resolved_singleton_counts": resolved_singleton,
                "acceptable_set_membership_counts": acc_membership,
                "acceptable_set_membership_pct": acc_membership_pct,
                "constant_and_rule_baselines": heur_accs,
                "by_track": by_track,
            }

        # Combined micro-average (build + pace eligible) for constant (dry_balanced + push_max) vs track_rule
        b_elig = [r for r in rows if r["decision_type"] == "build" and r["label_info"]["status"] != "all_failed"]
        p_elig = [r for r in rows if r["decision_type"] == "pace" and r["label_info"]["status"] != "all_failed"]
        tot_elig = len(b_elig) + len(p_elig)
        const_ok = sum(1 for r in b_elig if "dry_balanced" in r["acceptable_option_ids"]) + sum(
            1 for r in p_elig if "push_max" in r["acceptable_option_ids"]
        )
        tr_ok = sum(1 for r in b_elig if "dry_balanced" in r["acceptable_option_ids"]) + sum(
            1 for r in p_elig if ("balanced" if r["telemetry"]["track"] == "parkland" else "push_max") in r["acceptable_option_ids"]
        )
        sp_info["combined_eligible_n"] = tot_elig
        sp_info["combined_excluded_all_failed_n"] = len(rows) - tot_elig
        sp_info["qwen_build_loss_weight"] = round(len(p_elig) / max(len(b_elig), 1), 4)
        sp_info["const_dry_balanced_plus_push_max_combined_acc_pct"] = round(100.0 * const_ok / max(tot_elig, 1), 2)
        sp_info["track_rule_combined_acc_pct"] = round(100.0 * tr_ok / max(tot_elig, 1), 2)
        out[sp] = sp_info
    return out


def plot_completion_by_time(
    fresh_180: dict[str, list[dict[str, Any]]],
    fresh_45: dict[str, list[dict[str, Any]]],
    hist_45: dict[str, list[dict[str, Any]]],
    out_png: Path,
) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[Plot skipped] matplotlib unavailable: {e}")
        return False

    labels = {
        "driver_only": "Static engineer (driver_only)",
        "track_rule": "Domain heuristic (track_rule)",
        "lookup_9cell": "Contextual lookup (lookup_9cell)",
        "nn3_racing_v1": "Compact MLP (NN-3)",
        "qwen_lora_racing_v1": "Racing Qwen3.5-0.8B LoRA",
    }
    colors = {
        "driver_only": "#2b5c8f",
        "track_rule": "#2a9d8f",
        "lookup_9cell": "#e9c46a",
        "nn3_racing_v1": "#f4a261",
        "qwen_lora_racing_v1": "#e76f51",
    }
    linestyles = {
        "driver_only": "-",
        "track_rule": "--",
        "lookup_9cell": "-.",
        "nn3_racing_v1": ":",
        "qwen_lora_racing_v1": "-",
    }

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8), sharey=True)
    panels = [
        ("Fresh Confirmation (N=180, seeds 3501–3520)", fresh_180),
        ("Matched Fresh Subset (N=45, seeds 3501–3505)", fresh_45),
        ("Historical Regression (N=45, seeds 2001–2005)", hist_45),
    ]

    t_vals = np.linspace(540.0, 740.0, 401)
    for ax, (title, m_rows) in zip(axes, panels):
        for m, rows in m_rows.items():
            n = len(rows)
            fin_times = np.array([float(r["finish_time_s"]) for r in rows if r["finished"] and r["finish_time_s"] is not None])
            y = np.array([np.sum(fin_times <= t) / n for t in t_vals])
            ax.plot(t_vals, y * 100.0, label=f"{labels[m]} ({len(fin_times)}/{n})", color=colors[m], linestyle=linestyles[m], linewidth=2.0)
        ax.set_title(title, fontsize=10.5, fontweight="bold")
        ax.set_xlabel("Race Finish Time t (seconds)", fontsize=9.5)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 100)
        ax.set_xlim(540, 740)
        ax.legend(loc="lower right", fontsize=8.0, frameon=True)

    axes[0].set_ylabel("Completed Races by Time t: G_m(t) (%)", fontsize=9.5)
    fig.suptitle("Successful-Completion-by-Time Curves G_m(t) (Failed Races Remain in Denominator)", fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return True


def main() -> None:
    fresh_180 = {m: load_panel_rows("fresh_confirm", m) for m in LIGHT_MODELS}
    qwen_fresh_45 = load_panel_rows("fresh_confirm_qwen", "qwen_lora_racing_v1")
    q_sids = {r["scenario_id"] for r in qwen_fresh_45}
    fresh_45: dict[str, list[dict[str, Any]]] = {
        m: [r for r in fresh_180[m] if r["scenario_id"] in q_sids] for m in LIGHT_MODELS
    }
    fresh_45["qwen_lora_racing_v1"] = qwen_fresh_45
    hist_45 = {m: load_panel_rows("historical_45", m) for m in ALL_MODELS}

    result = {
        "rng_construction": {
            "source": "decision_eval/src/decision_eval/racing_contract.py:make_scenario",
            "formula": "np.random.default_rng(seed * 100 + cell_idx)",
            "independent_cell_seed_streams": True,
            "primary_bootstrap": "cell_stratified_paired_bootstrap (4000 resamples)",
            "sensitivity_bootstrap": "seed_block_paired_bootstrap (4000 resamples)",
        },
        "transfer_diagnostic_27": analyze_transfer_27(),
        "historical_45_all_models": analyze_panel(hist_45),
        "fresh_180_light_models": analyze_panel(fresh_180),
        "fresh_45_matched_all_models": analyze_panel(fresh_45),
        "dataset_and_constant_baselines": analyze_dataset_and_constant_baselines(),
    }

    out_json = OUT_DIR / "thesis_analysis_v1.json"
    out_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    out_png = OUT_DIR / "completion_by_time_curves.png"
    plotted = plot_completion_by_time(fresh_180, fresh_45, hist_45, out_png)
    print(f"Saved {out_json} (plotted={plotted})")

    # Print concise highlights to stdout for verification
    ds = result["dataset_and_constant_baselines"]
    print("\n=== CONSTANT POLICY & DATASET ACCOUNTING ===")
    for sp in ("train", "dev", "cal"):
        b = ds[sp]["build"]
        p = ds[sp]["pace"]
        print(
            f"[{sp}] total={ds[sp]['total_rows']} elig={ds[sp]['combined_eligible_n']} excl_all_fail={ds[sp]['combined_excluded_all_failed_n']} | "
            f"build_elig={b['eligible']} (dry_bal acc={b['acceptable_set_membership_pct']['dry_balanced']}%, lk9={b['constant_and_rule_baselines']['lookup_9cell_acc_pct']}%) | "
            f"pace_elig={p['eligible']} (push_max acc={p['acceptable_set_membership_pct']['push_max']}%, tr_pace={p['constant_and_rule_baselines']['track_rule_pace_acc_pct']}%) | "
            f"comb const(dry_bal+push_max)={ds[sp]['const_dry_balanced_plus_push_max_combined_acc_pct']}%, track_rule={ds[sp]['track_rule_combined_acc_pct']}%"
        )

    print("\n=== MATCHED FRESH-45 (seeds 3501..3505, ALL 5 MODELS) ===")
    f45 = result["fresh_45_matched_all_models"]
    print(f"All-model common finishers: {f45['all_model_common_finishers_n']}/45")
    for m, md in f45["models"].items():
        s = md["summary"]
        vt = md["vs_track_rule"]
        vd = md["vs_driver_only"]
        print(
            f"  {m:20s} | fin={s['finishes']}/{s['n']} ({s['completion_rate_pct']}%) | mean_fin={s['mean_time_finishers_s']}s | common_all={md['mean_time_all_model_common_s']}s | "
            f"vs_drv={vd['mean_paired_delta_s'] if vd else 'ref'}s ({vd['paired_outcomes'] if vd else ''}) | "
            f"vs_trk={vt['mean_paired_delta_s'] if vt else 'ref'}s ({vt['paired_outcomes'] if vt else ''})"
        )

    print("\n=== FRESH-180 (seeds 3501..3520, LIGHT 4 MODELS) ===")
    f180 = result["fresh_180_light_models"]
    print(f"All-model common finishers: {f180['all_model_common_finishers_n']}/180")
    for m, md in f180["models"].items():
        s = md["summary"]
        vt = md["vs_track_rule"]
        vd = md["vs_driver_only"]
        print(
            f"  {m:20s} | fin={s['finishes']}/{s['n']} ({s['completion_rate_pct']}%) | mean_fin={s['mean_time_finishers_s']}s | common_all={md['mean_time_all_model_common_s']}s | "
            f"vs_drv={vd['mean_paired_delta_s'] if vd else 'ref'}s {vd['bootstrap_cell_stratified']['time_ci95_2sided_s'] if vd else ''} ({vd['paired_outcomes'] if vd else ''}) | "
            f"vs_trk={vt['mean_paired_delta_s'] if vt else 'ref'}s {vt['bootstrap_cell_stratified']['time_ci95_2sided_s'] if vt else ''} ({vt['paired_outcomes'] if vt else ''})"
        )


if __name__ == "__main__":
    main()
