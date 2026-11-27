"""Pure-Python metrics engine for decision-oriented AI benchmarking.

Computes:
  - Accuracy + 95% bootstrap CI, Macro-F1, Valid-Output Rate (parse_ok)
  - Calibration & Uncertainty: 10-bin ECE, Reliability bins, Brier, NLL, Error-Detection AUROC, Risk-Coverage AURC
  - Safety & Economics: False-Accept Rate, False-Abstain Rate, Weighted Asymmetric Error Cost (5x),
    Mean Oracle Regret, Cost per 1,000 Decisions, Cost per 1,000 Correct Decisions
  - Robustness & Speed: Option-Order Flip Rate, Paraphrase Flip Rate, Context-Length Accuracy, p50/p95/p99 Latency
  - Statistical Significance: Paired McNemar's test across every model pair
  - Confidence-Gated Hybrid Cascades: Local 0.8B -> Frontier LLM Pareto curve (0%-100% escalation)
  - Contrast Cases: Top items where Kev-0.8B succeeds and Qwen-0.8B-JSON fails (for slide visuals)
"""
from __future__ import annotations

import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


def percentile(vals: list[float], p: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    k = (len(s) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return round(s[int(k)], 3)
    return round(s[f] + (s[c] - s[f]) * (k - f), 3)


def bootstrap_ci_mean(indicators: list[int], n_boot: int = 1000, seed: int = 42) -> tuple[float, float]:
    if not indicators:
        return 0.0, 0.0
    rng = random.Random(seed)
    n = len(indicators)
    means = sorted(sum(indicators[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    return round(means[int(0.025 * n_boot)], 4), round(means[min(n_boot - 1, int(0.975 * n_boot))], 4)


def compute_ece_and_bins(records: list[dict[str, Any]], n_bins: int = 10) -> tuple[float, list[dict[str, Any]]]:
    if not records:
        return 0.0, []
    bins: list[list[dict[str, Any]]] = [[] for _ in range(n_bins)]
    for r in records:
        c = max(0.0, min(1.0, float(r["confidence"])))
        bins[min(n_bins - 1, int(c * n_bins))].append(r)
    ece = 0.0
    bin_rows = []
    n_total = len(records)
    for idx, bucket in enumerate(bins):
        lo, hi = idx / n_bins, (idx + 1) / n_bins
        if not bucket:
            bin_rows.append({"bin_lo": round(lo, 2), "bin_hi": round(hi, 2), "count": 0, "avg_conf": None, "avg_acc": None})
            continue
        avg_conf = sum(float(r["confidence"]) for r in bucket) / len(bucket)
        avg_acc = sum(1.0 if r["correct"] else 0.0 for r in bucket) / len(bucket)
        ece += (len(bucket) / n_total) * abs(avg_acc - avg_conf)
        bin_rows.append({
            "bin_lo": round(lo, 2),
            "bin_hi": round(hi, 2),
            "count": len(bucket),
            "avg_conf": round(avg_conf, 4),
            "avg_acc": round(avg_acc, 4),
        })
    return round(ece, 4), bin_rows


def compute_auroc(records: list[dict[str, Any]]) -> float | None:
    pos = [float(r["confidence"]) for r in records if r["correct"]]
    neg = [float(r["confidence"]) for r in records if not r["correct"]]
    if not pos or not neg:
        return None
    wins = sum(1.0 if p > q else (0.5 if p == q else 0.0) for p in pos for q in neg)
    return round(wins / (len(pos) * len(neg)), 4)


def compute_aurc_and_curve(records: list[dict[str, Any]]) -> tuple[float, list[dict[str, float]]]:
    if not records:
        return 0.0, []
    ordered = sorted(records, key=lambda r: float(r["confidence"]), reverse=True)
    n = len(ordered)
    errors = 0
    risks = []
    curve = []
    for k, r in enumerate(ordered, start=1):
        if not r["correct"]:
            errors += 1
        cov = k / n
        risk = errors / k
        risks.append(risk)
        if k == 1 or k == n or k % max(1, n // 20) == 0:
            curve.append({"coverage": round(cov, 4), "risk": round(risk, 4), "accuracy": round(1.0 - risk, 4)})
    return round(sum(risks) / n, 4), curve


def macro_f1(records: list[dict[str, Any]]) -> float:
    labels = sorted({r["gold_label"] for r in records})
    if not labels:
        return 0.0
    f1s = []
    for lab in labels:
        tp = sum(1 for r in records if r["pred_label"] == lab and r["gold_label"] == lab)
        fp = sum(1 for r in records if r["pred_label"] == lab and r["gold_label"] != lab)
        fn = sum(1 for r in records if r["pred_label"] != lab and r["gold_label"] == lab)
        denom = 2 * tp + fp + fn
        f1s.append((2 * tp / denom) if denom > 0 else 0.0)
    return round(sum(f1s) / len(f1s), 4)


def mcnemar_test(recs_a: list[dict[str, Any]], recs_b: list[dict[str, Any]]) -> dict[str, Any]:
    map_a = {r["id"]: bool(r["correct"]) for r in recs_a}
    map_b = {r["id"]: bool(r["correct"]) for r in recs_b}
    shared = sorted(set(map_a.keys()) & set(map_b.keys()))
    b_cnt = sum(1 for i in shared if map_a[i] and not map_b[i])
    c_cnt = sum(1 for i in shared if not map_a[i] and map_b[i])
    n_disc = b_cnt + c_cnt
    if n_disc == 0:
        p_val = 1.0
    elif n_disc <= 40:
        k = min(b_cnt, c_cnt)
        tail = sum(math.comb(n_disc, j) for j in range(0, k + 1)) * (0.5 ** n_disc)
        p_val = min(1.0, 2.0 * tail)
    else:
        chi2 = ((abs(b_cnt - c_cnt) - 1.0) ** 2) / n_disc
        p_val = math.erfc(math.sqrt(chi2 / 2.0))
    return {
        "shared_n": len(shared),
        "a_only_correct": b_cnt,
        "b_only_correct": c_cnt,
        "p_value": round(p_val, 6),
        "significant_0_05": p_val < 0.05,
    }


def summarize_slice(
    records: list[dict[str, Any]],
    in_price_per_m: float = 0.0,
    out_price_per_m: float = 0.0,
) -> dict[str, Any]:
    if not records:
        return {"n": 0}
    n = len(records)
    correct_flags = [1 if r["correct"] else 0 for r in records]
    acc = sum(correct_flags) / n
    ci_lo, ci_hi = bootstrap_ci_mean(correct_flags)
    parse_rate = sum(1 for r in records if r["parse_ok"]) / n
    ece, rel_bins = compute_ece_and_bins(records)
    auroc = compute_auroc(records)
    aurc, rc_curve = compute_aurc_and_curve(records)

    brier_sum = 0.0
    nll_sum = 0.0
    for r in records:
        probs = r["probabilities"]
        g = int(r["gold_index"])
        if probs and len(probs) > g:
            brier_sum += sum((float(p) - (1.0 if i == g else 0.0)) ** 2 for i, p in enumerate(probs))
            nll_sum += -math.log(max(1e-9, min(1.0, float(probs[g]))))
        else:
            brier_sum += 2.0
            nll_sum += -math.log(1e-9)

    abstain_gold = [r for r in records if r.get("is_abstain_gold")]
    action_gold = [r for r in records if r.get("abstain_index") is not None and not r.get("is_abstain_gold")]
    false_accept = (
        sum(1 for r in abstain_gold if r["pred_index"] != r["abstain_index"]) / len(abstain_gold)
        if abstain_gold else None
    )
    false_abstain = (
        sum(1 for r in action_gold if r["pred_index"] == r["abstain_index"]) / len(action_gold)
        if action_gold else None
    )
    asym_cost = sum((5.0 if r.get("is_abstain_gold") else 1.0) for r in records if not r["correct"]) / n

    regrets = []
    for r in records:
        oscores = r.get("oracle_scores")
        if oscores and not r.get("is_abstain_gold"):
            best = max(float(v) for v in oscores.values())
            pred_lab = r.get("pred_label")
            if pred_lab in oscores:
                regrets.append(max(0.0, best - float(oscores[pred_lab])))
            else:
                regrets.append(max(0.0, best - min(float(v) for v in oscores.values())))

    latencies = [float(r["latency_ms"]) for r in records]
    mean_lat = (sum(latencies) / len(latencies)) if latencies else 0.0
    compute_sec_per_1k = round(mean_lat, 3)
    compute_sec_per_1k_correct = round(mean_lat / acc, 3) if acc > 0 else None
    total_cost = sum(
        (float(r.get("input_tokens", 0)) * in_price_per_m + float(r.get("output_tokens", 0)) * out_price_per_m) / 1_000_000.0
        for r in records
    )
    cost_per_1k = (total_cost / n) * 1000.0
    n_correct = sum(correct_flags)
    cost_per_1k_correct = (total_cost / n_correct) * 1000.0 if n_correct > 0 else None

    return {
        "n": n,
        "accuracy": round(acc, 4),
        "acc_ci_95": [ci_lo, ci_hi],
        "macro_f1": macro_f1(records),
        "parse_ok_rate": round(parse_rate, 4),
        "ece": ece,
        "brier": round(brier_sum / n, 4),
        "nll": round(nll_sum / n, 4),
        "error_auroc": auroc,
        "aurc": aurc,
        "false_accept_rate": round(false_accept, 4) if false_accept is not None else None,
        "false_abstain_rate": round(false_abstain, 4) if false_abstain is not None else None,
        "asymmetric_error_cost": round(asym_cost, 4),
        "mean_oracle_regret": round(sum(regrets) / len(regrets), 4) if regrets else None,
        "latency_p50_ms": percentile(latencies, 50),
        "latency_p95_ms": percentile(latencies, 95),
        "latency_p99_ms": percentile(latencies, 99),
        "compute_sec_per_1k": compute_sec_per_1k,
        "compute_sec_per_1k_correct": compute_sec_per_1k_correct,
        "cost_per_1k_usd": round(cost_per_1k, 4),
        "cost_per_1k_correct_usd": round(cost_per_1k_correct, 4) if cost_per_1k_correct is not None else None,
        "reliability_bins": rel_bins,
        "risk_coverage_curve": rc_curve,
    }


def compute_cascade_pareto(
    local_test: list[dict[str, Any]],
    frontier_test: list[dict[str, Any]],
    local_cal: list[dict[str, Any]] | None,
    frontier_in_price: float,
    frontier_out_price: float,
) -> list[dict[str, Any]]:
    f_map = {r["id"]: r for r in frontier_test}
    paired = [(r, f_map[r["id"]]) for r in local_test if r["id"] in f_map]
    if not paired:
        return []
    cal_confs = sorted(float(r["confidence"]) for r in (local_cal or local_test))
    points = []
    for target_esc in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        if target_esc <= 0.0:
            tau = -1.0
        elif target_esc >= 1.0:
            tau = 2.0
        else:
            tau = cal_confs[min(len(cal_confs) - 1, max(0, int(target_esc * len(cal_confs))))]

        correct_cnt = 0
        esc_cnt = 0
        latencies = []
        total_cost = 0.0
        for loc, fr in paired:
            if (not loc["parse_ok"]) or (float(loc["confidence"]) < tau):
                esc_cnt += 1
                correct_cnt += 1 if fr["correct"] else 0
                latencies.append(float(loc["latency_ms"]) + float(fr["latency_ms"]))
                total_cost += (
                    float(fr.get("input_tokens", 0)) * frontier_in_price
                    + float(fr.get("output_tokens", 0)) * frontier_out_price
                ) / 1_000_000.0
            else:
                correct_cnt += 1 if loc["correct"] else 0
                latencies.append(float(loc["latency_ms"]))

        n = len(paired)
        points.append({
            "target_escalation": target_esc,
            "actual_escalation_rate": round(esc_cnt / n, 4),
            "threshold_tau": round(tau, 4),
            "accuracy": round(correct_cnt / n, 4),
            "cost_per_1k_usd": round((total_cost / n) * 1000.0, 4),
            "latency_p50_ms": percentile(latencies, 50),
            "latency_p95_ms": percentile(latencies, 95),
        })
    return points


def build_summary(
    all_records: list[dict[str, Any]],
    models: list[str],
    config: dict[str, Any],
    frontier_in_price: float = 0.15,
    frontier_out_price: float = 0.60,
) -> dict[str, Any]:
    test_orig = [r for r in all_records if r["split"] == "test" and r["order"] == "orig"]
    test_swap = [r for r in all_records if r["split"] == "test" and r["order"] == "swapped"]
    cal_orig = [r for r in all_records if r["split"] == "calibrate" and r["order"] == "orig"]

    summary: dict[str, Any] = {
        "config": config,
        "models": {},
        "mcnemar_pairwise": {},
        "cascades": {},
        "contrast_cases": [],
    }

    by_model_orig: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_model_swap: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_model_cal: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in test_orig:
        by_model_orig[r["model"]].append(r)
    for r in test_swap:
        by_model_swap[r["model"]].append(r)
    for r in cal_orig:
        by_model_cal[r["model"]].append(r)

    for m in models:
        recs = by_model_orig.get(m, [])
        in_p = frontier_in_price if m == "frontier-llm" else 0.0
        out_p = frontier_out_price if m == "frontier-llm" else 0.0
        overall = summarize_slice(recs, in_p, out_p)

        swap_map = {r["id"]: r["pred_index"] for r in by_model_swap.get(m, [])}
        order_pairs = [(r["pred_index"], swap_map[r["id"]]) for r in recs if r["id"] in swap_map]
        overall["order_flip_rate"] = (
            round(sum(1 for a, b in order_pairs if a != b) / len(order_pairs), 4) if order_pairs else None
        )

        pred_by_id = {r["id"]: r["pred_index"] for r in recs}
        para_pairs = [
            (r["pred_index"], pred_by_id[r["base_id"]])
            for r in recs
            if r["bucket"] == "paraphrase" and r.get("base_id") in pred_by_id
        ]
        overall["paraphrase_flip_rate"] = (
            round(sum(1 for a, b in para_pairs if a != b) / len(para_pairs), 4) if para_pairs else None
        )

        by_suite = {s: summarize_slice([r for r in recs if r["suite"] == s], in_p, out_p) for s in sorted({r["suite"] for r in recs})}
        by_bucket = {b: summarize_slice([r for r in recs if r["bucket"] == b], in_p, out_p) for b in sorted({r["bucket"] for r in recs})}
        by_ctx = {}
        for c in ["short", "medium", "long"]:
            sub = [r for r in recs if r.get("context_band") == c]
            if sub:
                by_ctx[c] = summarize_slice(sub, in_p, out_p)

        s3_clean = [r for r in recs if r["suite"] == "suite3_car_build" and not r.get("hidden_oracle_abstain")]
        if s3_clean:
            by_suite["suite3_car_build_clean"] = summarize_slice(s3_clean, in_p, out_p)

        summary["models"][m] = {
            "overall": overall,
            "by_suite": by_suite,
            "by_bucket": by_bucket,
            "by_context_band": by_ctx,
        }

    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            ma, mb = models[i], models[j]
            summary["mcnemar_pairwise"][f"{ma}__vs__{mb}"] = mcnemar_test(
                by_model_orig.get(ma, []), by_model_orig.get(mb, [])
            )

    if "frontier-llm" in by_model_orig:
        for m in ["kev-0.8b", "qwen-0.8b-ll", "qwen-0.8b-json"]:
            if m in by_model_orig:
                summary["cascades"][f"{m}->frontier-llm"] = compute_cascade_pareto(
                    by_model_orig[m],
                    by_model_orig["frontier-llm"],
                    by_model_cal.get(m),
                    frontier_in_price,
                    frontier_out_price,
                )

    if "kev-0.8b" in by_model_orig and "qwen-0.8b-json" in by_model_orig:
        k_map = {r["id"]: r for r in by_model_orig["kev-0.8b"]}
        q_map = {r["id"]: r for r in by_model_orig["qwen-0.8b-json"]}
        contrasts = []
        for iid, kr in k_map.items():
            qr = q_map.get(iid)
            if qr and kr["correct"] and not qr["correct"]:
                contrasts.append({
                    "id": iid,
                    "suite": kr["suite"],
                    "bucket": kr["bucket"],
                    "gold_label": kr["gold_label"],
                    "kev_pred": kr["pred_label"],
                    "kev_conf": kr["confidence"],
                    "qwen_json_pred": qr["pred_label"],
                    "qwen_json_conf": qr["confidence"],
                    "qwen_json_parse_ok": qr["parse_ok"],
                })
        parsed_wrong = [c for c in contrasts if c["qwen_json_parse_ok"]]
        parse_failed = [c for c in contrasts if not c["qwen_json_parse_ok"]]
        parsed_wrong.sort(key=lambda x: (x["qwen_json_conf"], x["kev_conf"]), reverse=True)
        parse_failed.sort(key=lambda x: x["kev_conf"], reverse=True)
        # Show up to 10 high-confidence overconfident errors first, then up to 5 parse failures
        summary["contrast_cases"] = (parsed_wrong[:10] + parse_failed[:5])[:15]

    return summary


def write_summary_files(summary: dict[str, Any], out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "benchmark_summary.json"
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    csv_path = out_dir / "benchmark_summary.csv"
    fieldnames = [
        "model", "scope", "slice", "n", "accuracy", "acc_ci_lo", "acc_ci_hi", "macro_f1",
        "parse_ok_rate", "ece", "brier", "nll", "error_auroc", "aurc",
        "false_accept_rate", "false_abstain_rate", "asymmetric_error_cost",
        "mean_oracle_regret", "order_flip_rate", "paraphrase_flip_rate",
        "latency_p50_ms", "latency_p95_ms", "cost_per_1k_usd", "cost_per_1k_correct_usd",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as fcsv:
        writer = csv.DictWriter(fcsv, fieldnames=fieldnames)
        writer.writeheader()
        for m, mdata in summary["models"].items():
            rows_to_emit = [("overall", "all", mdata["overall"])]
            for s, sdata in mdata["by_suite"].items():
                rows_to_emit.append(("suite", s, sdata))
            for b, bdata in mdata["by_bucket"].items():
                rows_to_emit.append(("bucket", b, bdata))
            for c, cdata in mdata["by_context_band"].items():
                rows_to_emit.append(("context_band", c, cdata))
            for scope, sl, d in rows_to_emit:
                ci = d.get("acc_ci_95") or [None, None]
                writer.writerow({
                    "model": m,
                    "scope": scope,
                    "slice": sl,
                    "n": d.get("n"),
                    "accuracy": d.get("accuracy"),
                    "acc_ci_lo": ci[0],
                    "acc_ci_hi": ci[1],
                    "macro_f1": d.get("macro_f1"),
                    "parse_ok_rate": d.get("parse_ok_rate"),
                    "ece": d.get("ece"),
                    "brier": d.get("brier"),
                    "nll": d.get("nll"),
                    "error_auroc": d.get("error_auroc"),
                    "aurc": d.get("aurc"),
                    "false_accept_rate": d.get("false_accept_rate"),
                    "false_abstain_rate": d.get("false_abstain_rate"),
                    "asymmetric_error_cost": d.get("asymmetric_error_cost"),
                    "mean_oracle_regret": d.get("mean_oracle_regret"),
                    "order_flip_rate": d.get("order_flip_rate"),
                    "paraphrase_flip_rate": d.get("paraphrase_flip_rate"),
                    "latency_p50_ms": d.get("latency_p50_ms"),
                    "latency_p95_ms": d.get("latency_p95_ms"),
                    "cost_per_1k_usd": d.get("cost_per_1k_usd"),
                    "cost_per_1k_correct_usd": d.get("cost_per_1k_correct_usd"),
                })
    return json_path, csv_path
