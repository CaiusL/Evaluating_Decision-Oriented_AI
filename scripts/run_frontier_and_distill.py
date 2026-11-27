#!/usr/bin/env python3
"""Run Frontier LLM on Benchmark v2 (unlocking Chart 01) and Verified Distillation on train_840."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import re
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "decision_eval" / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from decision_eval import charts, metrics
from decision_eval.runners import parse_json_decision


def load_dotenv(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    for ln in path.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if ln and not ln.startswith("#") and "=" in ln:
            k, v = ln.split("=", 1)
            env[k.strip()] = v.strip()
    return env


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def call_chat_api(
    base_url: str,
    api_key: str,
    model: str,
    prompt: str,
    cache_dir: Path,
    cache_prefix: str,
    max_tokens: int = 128,
    max_retries: int = 3,
) -> dict[str, Any]:
    body_obj: dict[str, Any] = {
        "model": model,
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
        "messages": [{"role": "user", "content": prompt}],
    }
    if "openrouter.ai" in base_url:
        body_obj["reasoning"] = {"effort": "none", "exclude": True}
    else:
        body_obj["enable_thinking"] = False

    raw_req = json.dumps(body_obj, sort_keys=True, ensure_ascii=False)
    req_hash = hashlib.sha256(f"{base_url}|{raw_req}".encode("utf-8")).hexdigest()[:24]
    cache_file = cache_dir / f"{cache_prefix}_{req_hash}.json"
    if cache_file.is_file():
        return json.loads(cache_file.read_text(encoding="utf-8"))

    url = base_url.rstrip("/") + "/chat/completions"
    data = json.dumps(body_obj).encode("utf-8")
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}

    last_err = ""
    for attempt in range(max_retries + 1):
        t0 = time.perf_counter()
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                raw_json = json.loads(resp.read().decode("utf-8"))
            latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            gen_text = str(raw_json.get("choices", [{}])[0].get("message", {}).get("content", "")).strip()
            usage = raw_json.get("usage") or {}
            out = {
                "ok": True,
                "content": gen_text,
                "latency_ms": latency_ms,
                "input_tokens": int(usage.get("prompt_tokens") or 0),
                "output_tokens": int(usage.get("completion_tokens") or 0),
                "model_returned": raw_json.get("model"),
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            }
            cache_file.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
            return out
        except Exception as e:
            last_err = str(e)
            if attempt < max_retries:
                time.sleep(1.5 * (attempt + 1))

    return {
        "ok": False,
        "error": last_err,
        "content": "",
        "latency_ms": 0.0,
        "input_tokens": 0,
        "output_tokens": 0,
    }


def run_benchmark_v2_frontier(env: dict[str, str], workers: int) -> None:
    test_path = ROOT / "data" / "benchmarks_v2" / "all_test.jsonl"
    out_dir = ROOT / "reports" / "full_benchmark_v2"
    raw_path = out_dir / "benchmark_raw.jsonl"
    cache_dir = out_dir / "frontier_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    items = load_jsonl(test_path)
    base_url, api_key, model = env["QWEN_BASE_URL"], env["QWEN_API_KEY"], env["QWEN_MODEL"]

    jobs: list[tuple[dict[str, Any], str]] = []
    for item in items:
        jobs.append((item, "orig"))
        jobs.append((item, "swapped"))

    print(f"[Benchmark v2 Frontier] Scoring {len(jobs)} calls ({len(items)} items x 2 orders) with {model} ({workers} workers)...", flush=True)

    def _run_one(job: tuple[dict[str, Any], str]) -> dict[str, Any]:
        item, order = job
        opts_orig = list(item["options"])
        descs_orig = list(item.get("option_descriptions") or opts_orig)
        n_opts = len(opts_orig)
        opts = list(reversed(opts_orig)) if order == "swapped" else opts_orig
        descs = list(reversed(descs_orig)) if order == "swapped" else descs_orig

        numbered = []
        for idx, opt in enumerate(opts):
            extra = f" ({descs[idx]})" if idx < len(descs) and descs[idx] != opt else ""
            numbered.append(f"{idx}: {opt}{extra}")
        prompt = (
            "You are a decision assistant. Pick exactly one option index and output ONLY valid JSON "
            'with keys "choice_index" (integer) and "confidence" (float between 0.0 and 1.0).\n\n'
            f"State:\n{item['context']}\n\nQuestion: {item['question']}\nOptions:\n" + "\n".join(numbered) + "\n\nJSON:"
        )
        res = call_chat_api(base_url, api_key, model, prompt, cache_dir, "v2", max_tokens=128)
        parse_ok, choice_idx, confidence, probs = parse_json_decision(res["content"], n_opts)
        pred_idx = choice_idx if parse_ok else -1
        if order == "swapped":
            if 0 <= pred_idx < n_opts:
                pred_idx = n_opts - 1 - pred_idx
            probs = list(reversed(probs))

        gold_idx = int(item["gold_index"])
        return {
            "id": item["id"],
            "split": item["split"],
            "suite": item["suite"],
            "bucket": item["bucket"],
            "context_band": item.get("context_band", "short"),
            "est_tokens": item.get("est_tokens", 0),
            "base_id": item.get("base_id"),
            "order": order,
            "model": "frontier-llm",
            "model_id": model,
            "gold_index": gold_idx,
            "gold_label": item["gold_label"],
            "abstain_index": item.get("abstain_index"),
            "is_abstain_gold": bool(item.get("is_abstain_gold", False)),
            "hidden_oracle_abstain": bool(item.get("hidden_oracle_abstain", False)),
            "oracle_scores": item.get("oracle_scores"),
            "pred_index": pred_idx,
            "pred_label": opts_orig[pred_idx] if 0 <= pred_idx < n_opts else None,
            "correct": (pred_idx == gold_idx),
            "parse_ok": bool(parse_ok),
            "confidence": round(float(confidence), 6),
            "probabilities": [round(float(p), 6) for p in probs],
            "latency_ms": float(res["latency_ms"]),
            "input_tokens": int(res["input_tokens"]),
            "output_tokens": int(res["output_tokens"]),
            "raw_output": res["content"],
        }

    frontier_recs: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        for i, rec in enumerate(ex.map(_run_one, jobs), 1):
            frontier_recs.append(rec)
            if i % 100 == 0 or i == len(jobs):
                print(f"  [Benchmark v2 Frontier] {i}/{len(jobs)} done", flush=True)

    # Merge idempotently into benchmark_raw.jsonl (replacing any previous frontier-llm rows)
    existing = [r for r in load_jsonl(raw_path) if r.get("model") != "frontier-llm"]
    merged = existing + frontier_recs
    raw_path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in merged) + "\n", encoding="utf-8")

    # Regenerate summary JSON/CSV + all 6 SVG charts (including 01_cascade_pareto.svg)
    models = sorted({r["model"] for r in merged if r.get("model")})
    config = {
        "source": str(raw_path),
        "models": models,
        "frontier_model": model,
        "record_count": len(merged),
    }
    summary = metrics.build_summary(merged, models, config, frontier_in_price=0.26, frontier_out_price=0.38)
    json_path, csv_path = metrics.write_summary_files(summary, out_dir)
    created = charts.generate_all_charts(summary, out_dir / "charts")

    ov = summary["models"]["frontier-llm"]["overall"]
    print(f"\n[Benchmark v2 Frontier] DONE:")
    print(f"  frontier-llm ({model}): n={ov['n']} | acc={ov['accuracy']*100:.2f}% | ECE={ov['ece']:.4f} | parse={ov['parse_ok_rate']*100:.1f}% | flip={ov.get('order_flip_rate')} | p50={ov['latency_p50_ms']:.1f}ms")
    print(f"  Updated: {json_path}, {csv_path}")
    print(f"  Charts ({len(created)}): {[c.get('filename', c) if isinstance(c, dict) else str(c) for c in created]}")


def run_distill_train_840(env: dict[str, str], workers: int) -> None:
    train_path = ROOT / "data" / "teacher" / "train_840.jsonl"
    out_all = ROOT / "data" / "teacher" / "deepseek_v41_flash_distill_train_840.jsonl"
    out_verified = ROOT / "data" / "teacher" / "deepseek_v41_flash_verified_train.jsonl"
    summary_path = ROOT / "reports" / "distill_teacher_train_840_summary.json"
    cache_dir = ROOT / "data" / "teacher" / "distill_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    rows = load_jsonl(train_path)
    base_url, api_key, model = env["QWEN_BASE_URL"], env["QWEN_API_KEY"], env["QWEN_MODEL"]
    print(f"\n[Distill Teacher] Labeling {len(rows)} items in train_840.jsonl with {model} ({workers} workers)...", flush=True)

    def _label_one(row: dict[str, Any]) -> dict[str, Any]:
        opts = list(row["options"])
        descs = list(row.get("option_descriptions") or opts)
        n_opts = len(opts)
        numbered = [f"{i}: {opts[i]} ({descs[i]})" for i in range(n_opts)]
        prompt = (
            "You are a precise decision teacher. Pick the single best option index based strictly on the policy and facts in Context. "
            'Return ONLY valid JSON with keys "choice_index" (integer), "confidence" (float 0.0-1.0), and "rationale" (one concise sentence citing the decisive metric/rule).\n\n'
            f"Context:\n{row['context']}\n\nQuestion: {row['question']}\nOptions:\n" + "\n".join(numbered) + "\n\nJSON:"
        )
        res = call_chat_api(base_url, api_key, model, prompt, cache_dir, "distill", max_tokens=200)
        parse_ok, choice_idx, confidence, probs = parse_json_decision(res["content"], n_opts)
        rationale = ""
        try:
            m = re.search(r"\{.*\}", res["content"], flags=re.DOTALL)
            parsed = json.loads(m.group(0) if m else res["content"])
            rationale = str(parsed.get("rationale", "")).strip()
        except Exception:
            pass
        gold_idx = int(row["gold_index"])
        pred_idx = choice_idx if parse_ok else -1
        out_row = dict(row)
        out_row["teacher_model"] = model
        out_row["teacher_pred_index"] = pred_idx
        out_row["teacher_pred_label"] = opts[pred_idx] if 0 <= pred_idx < n_opts else None
        out_row["teacher_confidence"] = round(float(confidence), 6)
        out_row["teacher_probabilities"] = [round(float(p), 6) for p in probs]
        out_row["teacher_rationale"] = rationale
        out_row["teacher_parse_ok"] = bool(parse_ok)
        out_row["teacher_matches_gold"] = bool(parse_ok and pred_idx == gold_idx)
        out_row["teacher_latency_ms"] = float(res["latency_ms"])
        out_row["teacher_input_tokens"] = int(res["input_tokens"])
        out_row["teacher_output_tokens"] = int(res["output_tokens"])
        return out_row

    labeled: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        for i, r in enumerate(ex.map(_label_one, rows), 1):
            labeled.append(r)
            if i % 100 == 0 or i == len(rows):
                print(f"  [Distill Teacher] {i}/{len(rows)} done", flush=True)

    verified = [r for r in labeled if r["teacher_matches_gold"]]
    out_all.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in labeled) + "\n", encoding="utf-8")
    out_verified.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in verified) + "\n", encoding="utf-8")

    # Summary stats
    by_bucket: dict[str, dict[str, Any]] = {}
    for b in sorted({r["bucket"] for r in labeled}):
        br = [r for r in labeled if r["bucket"] == b]
        bm = sum(1 for r in br if r["teacher_matches_gold"])
        bp = sum(1 for r in br if r["teacher_parse_ok"])
        by_bucket[b] = {
            "n": len(br),
            "verified_matches": bm,
            "agreement_rate": round(bm / max(len(br), 1), 4),
            "parse_ok_rate": round(bp / max(len(br), 1), 4),
        }

    lats = sorted(r["teacher_latency_ms"] for r in labeled if r["teacher_latency_ms"] > 0)
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "teacher_model": model,
        "source_file": str(train_path),
        "out_all_file": str(out_all),
        "out_verified_file": str(out_verified),
        "n_total": len(labeled),
        "n_parse_ok": sum(1 for r in labeled if r["teacher_parse_ok"]),
        "n_verified_matches": len(verified),
        "agreement_rate": round(len(verified) / max(len(labeled), 1), 4),
        "abstain_agreement_rate": round(
            sum(1 for r in labeled if r.get("is_abstain_gold") and r["teacher_matches_gold"])
            / max(sum(1 for r in labeled if r.get("is_abstain_gold")), 1),
            4,
        ),
        "total_input_tokens": sum(r["teacher_input_tokens"] for r in labeled),
        "total_output_tokens": sum(r["teacher_output_tokens"] for r in labeled),
        "latency_p50_ms": round(lats[len(lats) // 2], 1) if lats else None,
        "latency_p95_ms": round(lats[int(len(lats) * 0.95)], 1) if lats else None,
        "by_bucket": by_bucket,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n[Distill Teacher] DONE: {len(verified)}/{len(labeled)} verified ({summary['agreement_rate']*100:.2f}%), parse={summary['n_parse_ok']}/{len(labeled)}")
    print(f"  All     : {out_all}")
    print(f"  Verified: {out_verified}")
    print(f"  Summary : {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--mode", choices=["v2", "distill", "both"], default="both")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    env = load_dotenv(args.env_file)
    if args.mode in ("v2", "both"):
        run_benchmark_v2_frontier(env, args.workers)
    if args.mode in ("distill", "both"):
        run_distill_train_840(env, args.workers)


if __name__ == "__main__":
    main()
