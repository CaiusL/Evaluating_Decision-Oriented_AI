"""Build Offline Benchmark v2 (n=244) and recompute all metrics from frozen v1 predictions.

Preserves Offline Benchmark v1 (data/benchmarks/, reports/full_benchmark/, n=313) untouched.

Changes in Offline Benchmark v2:
  - Removes the 69 `historical_pair` test items (and 47 calibrate items) from Suite 2 on target-validity grounds:
    relative lobby rank across historical matches does not adequately support claims about competitive pace
    or intervention quality.
  - Retains the 18 audited `driver_lever_longctx` items under a dedicated suite name:
    `suite2_evidence_discipline` ("Engineering Evidence Discipline").
  - Keeps Suite 1 (120), Suite 3 (34), and Suite 4 (72) unchanged -> 120 + 18 + 34 + 72 = 244 test items.
  - Recomputes all metrics (accuracy + 95% CI, Macro-F1, ECE, Brier, NLL, AUROC, AURC, false-accept/abstain,
    asymmetric error cost, order/paraphrase flip rates, latency quantiles, paired McNemar tests, and SVG charts)
    from frozen per-item predictions in reports/full_benchmark/.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import benchmark_metrics as bm
import generate_charts as gc

V1_BENCH_DIR = REPO_ROOT / "data" / "benchmarks"
V2_BENCH_DIR = REPO_ROOT / "data" / "benchmarks_v2"
V1_REPORT_DIR = REPO_ROOT / "reports" / "full_benchmark"
V2_REPORT_DIR = REPO_ROOT / "reports" / "full_benchmark_v2"

SUITE2_V1 = "suite2_historical_eng"
SUITE2_V2 = "suite2_evidence_discipline"


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")


def _slice_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "count": len(rows),
        "abstain_gold_count": sum(1 for r in rows if r.get("is_abstain_gold")),
        "buckets": dict(sorted(Counter(r["bucket"] for r in rows).items())),
        "context_bands": dict(sorted(Counter(r["context_band"] for r in rows).items())),
    }


def audit_evidence_discipline_items(
    s2_dl: list[dict[str, Any]],
    s2_hp_test: list[dict[str, Any]],
    s2_hp_cal: list[dict[str, Any]],
    s4_test: list[dict[str, Any]],
) -> dict[str, Any]:
    unique_contexts = {r["context"] for r in s2_dl}
    gold_dist = dict(sorted(Counter(r["gold_label"] for r in s2_dl).items()))
    abstain_count = sum(1 for r in s2_dl if r.get("is_abstain_gold"))
    always_unresolved_correct = sum(1 for r in s2_dl if r["gold_label"] == "unresolved")

    dl_match_ids = [
        "1a8c2aea", "a2585d41", "21f76602", "c767b15a", "ebe0e4b9", "049a8010",
        "f96edeef", "37b81ebd", "08569f9b", "ede97a2e", "5f2ea146", "6214cc20",
    ]
    hp_test_overlap = sum(
        1 for r in s2_hp_test if any(mid in (r.get("context", "") + " " + r.get("question", "")) for mid in dl_match_ids)
    )
    hp_cal_overlap = sum(
        1 for r in s2_hp_cal if any(mid in (r.get("context", "") + " " + r.get("question", "")) for mid in dl_match_ids)
    )
    s2_ids = {r["id"] for r in s2_dl} | {r["id"] for r in s2_hp_test}
    s4_dep_count = sum(1 for r in s4_test if r.get("base_id") in s2_ids)

    item_checks = []
    for r in s2_dl:
        is_binary = r["id"] == "s2_drv_any_fair_pair"
        notes = []
        if r["id"] == "s2_drv_filters":
            notes.append("Context explicitly lists qual_margin/kv/latency; ld_min and ld_time also have no isolated pair in text.")
        if not is_binary:
            notes.append("Option 'keep' vs 'unresolved' both avoid changing the car; 'unresolved' specifically cites lack of an isolated pair.")
        item_checks.append({
            "id": r["id"],
            "lever": r.get("lever"),
            "gold_label": r["gold_label"],
            "is_abstain_gold": bool(r.get("is_abstain_gold")),
            "options": r["options"],
            "answer_follows_from_context": True,
            "no_vs_unresolved_clear": True,
            "controls_and_outcomes_stated": True,
            "status": "retained_with_caveats",
            "notes": notes,
        })

    return {
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "total_items": len(s2_dl),
        "retained_items": len(s2_dl),
        "unique_source_contexts": len(unique_contexts),
        "shared_context_chars": len(next(iter(unique_contexts))) if unique_contexts else 0,
        "gold_label_distribution": gold_dist,
        "abstain_gold_count": abstain_count,
        "always_unresolved_baseline_accuracy": round(always_unresolved_correct / len(s2_dl), 4) if s2_dl else 0.0,
        "historical_pair_test_match_id_overlap": hp_test_overlap,
        "historical_pair_cal_match_id_overlap": hp_cal_overlap,
        "suite4_dependency_on_suite2_count": s4_dep_count,
        "caveats": [
            "All 18 items share 1 identical 4,419-character engineering record (effective independent document count = 1).",
            "Severe class imbalance: 17/18 (94.4%) have gold label 'unresolved' and 1/18 (5.6%) has gold label 'No'; an always-unresolved baseline scores 17/18 (94.4%).",
            "Retained items were previously inspected during v1 analysis; v2 is a retrospective benchmark revision, not a fresh holdout.",
        ],
        "items": item_checks,
    }


def build_v2_dataset() -> tuple[set[str], dict[str, Any]]:
    s1_cal = _load_jsonl(V1_BENCH_DIR / "suite1_software_ops_calibrate.jsonl")
    s1_test = _load_jsonl(V1_BENCH_DIR / "suite1_software_ops_test.jsonl")
    s2_cal = _load_jsonl(V1_BENCH_DIR / "suite2_historical_eng_calibrate.jsonl")
    s2_test = _load_jsonl(V1_BENCH_DIR / "suite2_historical_eng_test.jsonl")
    s3_cal = _load_jsonl(V1_BENCH_DIR / "suite3_car_build_calibrate.jsonl")
    s3_test = _load_jsonl(V1_BENCH_DIR / "suite3_car_build_test.jsonl")
    s4_test = _load_jsonl(V1_BENCH_DIR / "suite4_stress_test.jsonl")

    s2_hp_test = [r for r in s2_test if r["bucket"] == "historical_pair"]
    s2_dl_raw = [r for r in s2_test if r["bucket"] == "driver_lever_longctx"]
    assert len(s2_hp_test) == 69, f"Expected 69 historical_pair test items, got {len(s2_hp_test)}"
    assert len(s2_dl_raw) == 18, f"Expected 18 driver_lever_longctx test items, got {len(s2_dl_raw)}"

    s2_v2_test = []
    for r in s2_dl_raw:
        item = dict(r)
        item["suite"] = SUITE2_V2
        item["v1_suite"] = SUITE2_V1
        s2_v2_test.append(item)

    all_cal_v2 = s1_cal + s3_cal
    all_test_v2 = s1_test + s2_v2_test + s3_test + s4_test
    assert len(all_test_v2) == 244, f"Expected 244 v2 test items, got {len(all_test_v2)}"

    V2_BENCH_DIR.mkdir(parents=True, exist_ok=True)
    files_to_write = {
        "suite1_software_ops_calibrate.jsonl": s1_cal,
        "suite1_software_ops_test.jsonl": s1_test,
        "suite2_evidence_discipline_test.jsonl": s2_v2_test,
        "suite3_car_build_calibrate.jsonl": s3_cal,
        "suite3_car_build_test.jsonl": s3_test,
        "suite4_stress_test.jsonl": s4_test,
        "all_calibrate.jsonl": all_cal_v2,
        "all_test.jsonl": all_test_v2,
    }
    for fname, rows in files_to_write.items():
        _write_jsonl(V2_BENCH_DIR / fname, rows)

    audit = audit_evidence_discipline_items(s2_dl_raw, s2_hp_test, s2_cal, s4_test)
    (V2_BENCH_DIR / "audit_18_evidence_discipline.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )

    total_n = len(all_test_v2)
    suite_weights = {
        "suite1_software_ops": round(len(s1_test) / total_n, 4),
        "suite2_evidence_discipline": round(len(s2_v2_test) / total_n, 4),
        "suite3_car_build": round(len(s3_test) / total_n, 4),
        "suite4_stress_test": round(len(s4_test) / total_n, 4),
    }

    manifest = {
        "benchmark_version": "offline-benchmark-v2",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "archived_v1_path": "data/benchmarks",
        "archived_v1_test_count": 313,
        "revision_reason": (
            "Target validity: the 69 historical_pair relative-rank items do not adequately support claims "
            "about competitive pace or racing intervention quality. The 18 driver_lever_longctx items were "
            "audited and retained under suite2_evidence_discipline (Engineering Evidence Discipline)."
        ),
        "aggregation_rule": (
            "Micro-averaged across all 244 retained test items for overall metrics; per-suite metrics must be "
            "reported alongside overall because removing 69 historical_pair items shifts Software Ops weight "
            "from 38.3% (120/313) in v1 to 49.2% (120/244) in v2."
        ),
        "suite_weights_test": suite_weights,
        "removed_from_v1": {
            "test_bucket": "historical_pair",
            "test_removed_count": len(s2_hp_test),
            "calibrate_removed_count": len(s2_cal),
        },
        "retained_test_ids": [r["id"] for r in all_test_v2],
        "files": {fname: _slice_stats(rows) for fname, rows in files_to_write.items()},
        "totals": {
            "calibrate": len(all_cal_v2),
            "test": len(all_test_v2),
        },
        "evidence_discipline_audit_summary": {
            "unique_source_contexts": audit["unique_source_contexts"],
            "gold_label_distribution": audit["gold_label_distribution"],
            "always_unresolved_baseline_accuracy": audit["always_unresolved_baseline_accuracy"],
            "caveats": audit["caveats"],
        },
    }
    (V2_BENCH_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return set(manifest["retained_test_ids"]), audit


def recompute_v2_metrics(retained_ids: set[str]) -> dict[str, Any]:
    raw_sources = [
        V1_REPORT_DIR / "benchmark_raw.jsonl",
        V1_REPORT_DIR / "benchmark_raw_self_jev_v2_m3.jsonl",
        V1_REPORT_DIR / "benchmark_raw_self_jev_v2_1_m3.jsonl",
    ]
    all_v1_rows: list[dict[str, Any]] = []
    for p in raw_sources:
        if p.exists():
            all_v1_rows.extend(_load_jsonl(p))

    v2_rows: list[dict[str, Any]] = []
    for r in all_v1_rows:
        if r.get("split") == "test" and r["id"] not in retained_ids:
            continue
        if r.get("split") == "calibrate" and r.get("bucket") == "historical_pair":
            continue
        row = dict(r)
        if row.get("suite") == SUITE2_V1 and row.get("bucket") == "driver_lever_longctx":
            row["suite"] = SUITE2_V2
        v2_rows.append(row)

    V2_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    _write_jsonl(V2_REPORT_DIR / "benchmark_raw_v2.jsonl", v2_rows)

    preferred_order = [
        "kev-0.8b",
        "self-jev-minilm",
        "self-jev-v2-m3",
        "self-jev-v2.1-m3",
        "qwen-0.8b-ll",
        "qwen-0.8b-json",
        "car-build-nn",
    ]
    present_models = sorted({r["model"] for r in v2_rows})
    models = [m for m in preferred_order if m in present_models] + [
        m for m in present_models if m not in preferred_order
    ]

    v1_summary = json.loads((V1_REPORT_DIR / "benchmark_summary.json").read_text(encoding="utf-8"))
    config = {
        "benchmark_version": "offline-benchmark-v2",
        "n_test_items": len(retained_ids),
        "recomputed_from_frozen_predictions": [str(p.relative_to(REPO_ROOT)) for p in raw_sources if p.exists()],
        "archived_v1_summary": "reports/full_benchmark/benchmark_summary.json",
        "suite2_renamed_to": SUITE2_V2,
        "suite2_display_name": "Engineering Evidence Discipline",
        "removed_bucket": "historical_pair (69 test items)",
        "recomputed_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    v2_summary = bm.build_summary(v2_rows, models, config)
    bm.write_summary_files(v2_summary, V2_REPORT_DIR)

    # Also build explicit v1 vs v2 compositional comparison
    comparison = {
        "note": (
            "Compositional comparison only: models, predictions, and calibration are frozen from v1. "
            "Score differences reflect removing the 69 historical_pair items, not model improvements."
        ),
        "v1_n": 313,
        "v2_n": 244,
        "software_ops_weight_v1": round(120 / 313, 4),
        "software_ops_weight_v2": round(120 / 244, 4),
        "models": {},
    }
    for m in models:
        v2_ov = v2_summary["models"][m]["overall"]
        v1_ov = (v1_summary.get("models") or {}).get(m, {}).get("overall")
        # For models not in v1 benchmark_summary.json (e.g. v2-m3), compute v1 overall directly from raw
        if v1_ov is None:
            m_v1_orig = [r for r in all_v1_rows if r["model"] == m and r["split"] == "test" and r["order"] == "orig"]
            v1_ov = bm.summarize_slice(m_v1_orig) if m_v1_orig else None
        comparison["models"][m] = {
            "v1_n": v1_ov["n"] if v1_ov else None,
            "v1_accuracy": v1_ov["accuracy"] if v1_ov else None,
            "v2_n": v2_ov["n"],
            "v2_accuracy": v2_ov["accuracy"],
            "compositional_delta_acc": (
                round(v2_ov["accuracy"] - v1_ov["accuracy"], 4) if v1_ov and v1_ov.get("accuracy") is not None else None
            ),
            "v2_macro_f1": v2_ov["macro_f1"],
            "v2_ece": v2_ov["ece"],
            "v2_aurc": v2_ov["aurc"],
            "v2_error_auroc": v2_ov["error_auroc"],
            "v2_false_accept_rate": v2_ov["false_accept_rate"],
            "v2_false_abstain_rate": v2_ov["false_abstain_rate"],
            "v2_asymmetric_error_cost": v2_ov["asymmetric_error_cost"],
            "v2_order_flip_rate": v2_ov["order_flip_rate"],
            "v2_paraphrase_flip_rate": v2_ov["paraphrase_flip_rate"],
            "v2_latency_p50_ms": v2_ov["latency_p50_ms"],
            "v2_latency_p95_ms": v2_ov["latency_p95_ms"],
        }
    (V2_REPORT_DIR / "v1_vs_v2_comparison.json").write_text(
        json.dumps(comparison, indent=2) + "\n", encoding="utf-8"
    )

    charts = gc.generate_all_charts(v2_summary, V2_REPORT_DIR / "charts")
    return {"summary": v2_summary, "comparison": comparison, "charts": charts}


def main() -> None:
    retained_ids, audit = build_v2_dataset()
    res = recompute_v2_metrics(retained_ids)
    print(f"Built Offline Benchmark v2: {len(retained_ids)} test items in {V2_BENCH_DIR}")
    print(f"Evidence discipline audit: {audit['retained_items']}/{audit['total_items']} retained, "
          f"unique_contexts={audit['unique_source_contexts']}, "
          f"always_unresolved_acc={audit['always_unresolved_baseline_accuracy']:.1%}")
    print("\n=== Offline Benchmark v1 (n=313) vs v2 (n=244) Compositional Comparison ===")
    for m, d in res["comparison"]["models"].items():
        v1_s = f"{d['v1_accuracy']*100:.1f}% (n={d['v1_n']})" if d["v1_accuracy"] is not None else "n/a"
        v2_s = f"{d['v2_accuracy']*100:.1f}% (n={d['v2_n']})"
        delta = f"{d['compositional_delta_acc']*100:+.1f} pp" if d["compositional_delta_acc"] is not None else "n/a"
        print(
            f"  {m:18s} | v1={v1_s:16s} -> v2={v2_s:16s} ({delta:8s}) | "
            f"F1={d['v2_macro_f1']:.3f} ECE={d['v2_ece']:.3f} AURC={d['v2_aurc']:.3f} "
            f"OrderFlip={d['v2_order_flip_rate']}"
        )
    print(f"\nGenerated {len(res['charts'])} SVG charts in {V2_REPORT_DIR / 'charts'}")


if __name__ == "__main__":
    main()
