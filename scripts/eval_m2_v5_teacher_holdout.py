#!/usr/bin/env python3
"""Q2 Bounded Frontier Teacher Reference Pass on Untouched holdout_600.jsonl.

Uses the exact original fixed teacher decision prompt and request settings from
scripts/run_frontier_and_distill.py:
  - Gold answers and verification metadata excluded from request
  - All 600 original-order cases retained in denominator (no exclusion of disagreements)
  - No prompt edits, no accuracy-triggered retries, no student checkpoint changes
  - Reports accuracy, invalid outputs, per-bucket accuracy, clean/wrapped accuracy,
    abstention recall/precision/false-abstention rate, latency, token counts, and
    exact pairwise McNemar vs. frozen student predictions.
  - Does NOT manufacture NLL/ECE from textual self-confidence.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT / "decision_eval" / "src"
for p in (str(SCRIPTS_DIR), str(SRC_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import run_frontier_and_distill as rfd  # noqa: E402
from decision_eval.runners import parse_json_decision  # noqa: E402


def exact_mcnemar_p(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) * (0.5 ** n)
    return float(min(1.0, 2.0 * tail))


def run_teacher_holdout(env: dict[str, str], workers: int = 8) -> dict[str, Any]:
    holdout_path = ROOT / "data" / "holdout" / "holdout_600.jsonl"
    out_dir = ROOT / "reports" / "m2_v5"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / "teacher_holdout_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    raw_out_path = out_dir / "m2_v5_teacher_holdout_raw.jsonl"
    summary_path = out_dir / "m2_v5_teacher_holdout_summary.json"

    rows = rfd.load_jsonl(holdout_path)
    base_url = env["QWEN_BASE_URL"]
    api_key = env["QWEN_API_KEY"]
    model = env["QWEN_MODEL"]

    print(
        f"[Q2 Teacher Reference] Running single original-order pass on {len(rows)} holdout_600 items "
        f"with {model} ({workers} workers)...",
        flush=True,
    )
    t_start = time.perf_counter()

    def _eval_one(row: dict[str, Any]) -> dict[str, Any]:
        opts = list(row["options"])
        descs = list(row.get("option_descriptions") or opts)
        n_opts = len(opts)
        numbered = [f"{i}: {opts[i]} ({descs[i]})" for i in range(n_opts)]
        # Exact original fixed teacher decision prompt from run_frontier_and_distill.py
        prompt = (
            "You are a precise decision teacher. Pick the single best option index based strictly on the policy and facts in Context. "
            'Return ONLY valid JSON with keys "choice_index" (integer), "confidence" (float 0.0-1.0), and "rationale" (one concise sentence citing the decisive metric/rule).\n\n'
            f"Context:\n{row['context']}\n\nQuestion: {row['question']}\nOptions:\n" + "\n".join(numbered) + "\n\nJSON:"
        )
        res = rfd.call_chat_api(base_url, api_key, model, prompt, cache_dir, "hold600", max_tokens=200, max_retries=3)
        parse_ok, choice_idx, _conf, _probs = parse_json_decision(res["content"], n_opts)
        pred_idx = choice_idx if parse_ok else -1
        gold_idx = int(row["gold_index"])
        return {
            "id": row["id"],
            "source_id": row["source_id"],
            "bucket": row["bucket"],
            "context_band": row.get("context_band", "clean"),
            "gold_index": gold_idx,
            "abstain_index": int(row["abstain_index"]),
            "is_abstain_gold": bool(row.get("is_abstain_gold", False)),
            "teacher_model_requested": model,
            "teacher_model_returned": res.get("model_returned"),
            "pred_index": pred_idx,
            "pred_label": opts[pred_idx] if 0 <= pred_idx < n_opts else None,
            "parse_ok": bool(parse_ok),
            "correct": bool(parse_ok and pred_idx == gold_idx),
            "latency_ms": float(res.get("latency_ms") or 0.0),
            "input_tokens": int(res.get("input_tokens") or 0),
            "output_tokens": int(res.get("output_tokens") or 0),
            "timestamp_utc": res.get("timestamp_utc") or datetime.now(timezone.utc).isoformat(),
            "raw_response": res.get("content", ""),
            "error": res.get("error"),
        }

    records: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        for i, rec in enumerate(ex.map(_eval_one, rows), 1):
            records.append(rec)
            if i % 100 == 0 or i == len(rows):
                print(f"  [Q2 Teacher Reference] {i}/{len(rows)} done", flush=True)

    raw_out_path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n", encoding="utf-8")

    n = len(records)
    n_correct = sum(1 for r in records if r["correct"])
    n_parse_ok = sum(1 for r in records if r["parse_ok"])
    n_invalid = n - n_parse_ok

    # Abstention recall, precision, and false-abstention rate
    tp = fp = fn = tn = 0
    for r in records:
        is_abs = r["is_abstain_gold"]
        pred_abs = r["parse_ok"] and (r["pred_index"] == r["abstain_index"])
        if is_abs and pred_abs:
            tp += 1
        elif is_abs and not pred_abs:
            fn += 1
        elif not is_abs and pred_abs:
            fp += 1
        else:
            tn += 1

    by_bucket: dict[str, dict[str, Any]] = {}
    for b in sorted({r["bucket"] for r in records}):
        br = [r for r in records if r["bucket"] == b]
        bc = sum(1 for r in br if r["correct"])
        by_bucket[b] = {"n": len(br), "correct": bc, "accuracy": round(bc / len(br), 4)}

    by_band: dict[str, dict[str, Any]] = {}
    for b in sorted({r["context_band"] for r in records}):
        br = [r for r in records if r["context_band"] == b]
        bc = sum(1 for r in br if r["correct"])
        by_band[b] = {"n": len(br), "correct": bc, "accuracy": round(bc / len(br), 4)}

    lats = sorted(r["latency_ms"] for r in records if r["latency_ms"] > 0)
    tot_in = sum(r["input_tokens"] for r in records)
    tot_out = sum(r["output_tokens"] for r in records)

    # Pairwise McNemar vs frozen student predictions
    mcnemar: dict[str, Any] = {}
    student_preds_path = out_dir / "m2_v5_holdout_predictions.jsonl"
    if student_preds_path.is_file():
        sp_map = {r["id"]: r["preds"] for r in rfd.load_jsonl(student_preds_path)}
        for m_name in ("qwen_trained_deploy", "kev_0.8b_frozen", "qwen_zeroshot", "bge_trained_deploy"):
            t_only = s_only = 0
            for r in records:
                s_ok = (int(sp_map[r["id"]][m_name]) == r["gold_index"])
                t_ok = r["correct"]
                if t_ok and not s_ok:
                    t_only += 1
                elif s_ok and not t_ok:
                    s_only += 1
            mcnemar[f"teacher_ref__vs__{m_name}"] = {
                "n": n,
                "teacher_only_correct": t_only,
                "other_only_correct": s_only,
                "exact_p_value": round(exact_mcnemar_p(t_only, s_only), 6),
            }

    summary = {
        "protocol": "Q2_bounded_frontier_teacher_reference_holdout_600_orig_only",
        "role": "Frontier teacher reference on independent gold (not an upper bound)",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "teacher_model_requested": model,
        "teacher_models_returned": sorted({r["teacher_model_returned"] for r in records if r["teacher_model_returned"]}),
        "request_settings": {
            "temperature": 0.0,
            "max_tokens": 200,
            "response_format": "json_object",
            "max_retries": 3,
            "order": "orig_only",
        },
        "holdout_file": "holdout_600.jsonl",
        "holdout_sha256": hashlib.sha256(holdout_path.read_bytes()).hexdigest(),
        "n_total": n,
        "n_correct": n_correct,
        "acc_orig": round(n_correct / n, 6),
        "acc_swapped": None,
        "order_flip_rate": None,
        "n_parse_ok": n_parse_ok,
        "n_invalid_outputs": n_invalid,
        "abstention": {
            "gold_abstain_n": tp + fn,
            "gold_non_abstain_n": fp + tn,
            "recall": round(tp / max(1, tp + fn), 6),
            "recall_counts": f"{tp}/{tp + fn}",
            "precision": round(tp / max(1, tp + fp), 6),
            "precision_counts": f"{tp}/{tp + fp}",
            "false_abstention_rate": round(fp / max(1, fp + tn), 6),
            "false_abstention_counts": f"{fp}/{fp + tn}",
        },
        "by_bucket": by_bucket,
        "by_band": by_band,
        "latency_p50_ms": round(lats[len(lats) // 2], 1) if lats else None,
        "latency_p95_ms": round(lats[int(len(lats) * 0.95)], 1) if lats else None,
        "total_input_tokens": tot_in,
        "total_output_tokens": tot_out,
        "wall_s": round(time.perf_counter() - t_start, 2),
        "mcnemar_vs_students": mcnemar,
        "note_on_calibration_metrics": "NLL/ECE/AURC intentionally omitted because API returns textual self-confidence rather than option probability distributions.",
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(
        f"\n[Q2 Teacher Reference DONE] {n_correct}/{n} ({summary['acc_orig']*100:.2f}%), "
        f"invalid={n_invalid}, abstain_recall={tp}/{tp+fn}, abstain_prec={tp}/{tp+fp}, "
        f"false_abstain={fp}/{fp+tn}, p50={summary['latency_p50_ms']}ms, wall={summary['wall_s']}s"
    )
    print(f"  Summary: {summary_path}")
    print(f"  Raw    : {raw_out_path}")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-file", type=Path, default=ROOT / ".env")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    env = rfd.load_dotenv(args.env_file)
    run_teacher_holdout(env, workers=args.workers)


if __name__ == "__main__":
    main()
