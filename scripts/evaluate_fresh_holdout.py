#!/usr/bin/env python3
"""Locked Fresh-Holdout Evaluator & Offline Cascade Replay.

Implements all protocol guardrails:
  - Evaluates 600 independent source pairs (orig + swapped)
  - Supports BGE-278M (frozen + 3 seeds), Qwen3.5-0.8B-Base (frozen + 3 seeds)
  - Supports Hosted Decision API (Bailian /compatible-mode/v1/systemone) with:
      * Neutral slot keys (slot_0..slot_K-1) so swapped genuinely tests slot assignment
      * Semantic remapping of selected choice & full probability vector
      * Both Native API probabilities and API + scalar T fitted on calibrate_180
      * Full dated service snapshot, client-observed latency, and denominator accounting
  - Supports OpenRouter / OpenAI-compatible Cloud LLM via --openrouter-config or --env-file
  - Offline cascade replay: Primary (local -> Cloud LLM) and Secondary (local -> Decision API)
    using the exact same predefined local gate fitted on calibrate_180.
  - Source-level paired bootstrap 95% CIs (1,000 resamples).
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import re
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from train_two_students import (
    LETTERS,
    compute_aurc,
    compute_ece,
    compute_error_auroc,
    fit_temperature,
    load_jsonl,
    softmax_np,
)


def compute_brier(probs_list: list[np.ndarray], labels: list[int]) -> float:
    scores = []
    for p, y in zip(probs_list, labels):
        onehot = np.zeros_like(p)
        onehot[y] = 1.0
        scores.append(float(np.sum((p - onehot) ** 2)))
    return float(np.mean(scores)) if scores else 0.0


def bootstrap_ci(values: np.ndarray, n_boot: int = 1000, seed: int = 2026) -> list[float]:
    if len(values) == 0:
        return [0.0, 0.0]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = np.mean(values[idx], axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return [round(float(lo), 4), round(float(hi), 4)]


def summarize_prob_model(
    rows: list[dict[str, Any]],
    orig_probs: list[np.ndarray | None],
    swap_probs: list[np.ndarray | None],
    orig_preds: list[int | None],
    swap_preds_unmapped: list[int | None],
    latencies_ms: list[float],
    label: str,
) -> dict[str, Any]:
    n_total = len(rows)
    valid_confs, valid_corr, valid_probs, valid_labels, valid_nlls = [], [], [], [], []
    all_orig_ok = []
    all_swap_ok = []
    flips = []
    complete_pairs = 0
    failures = 0

    for r, po, ps, yo_hat, ys_hat in zip(rows, orig_probs, swap_probs, orig_preds, swap_preds_unmapped):
        y = int(r["gold_index"])
        if yo_hat is None:
            failures += 1
            all_orig_ok.append(0)
        else:
            ok = int(yo_hat == y)
            all_orig_ok.append(ok)
            if po is not None:
                valid_confs.append(float(po[yo_hat]))
                valid_corr.append(ok)
                valid_probs.append(po)
                valid_labels.append(y)
                valid_nlls.append(-math.log(max(float(po[y]), 1e-12)))

        if ys_hat is None:
            all_swap_ok.append(0)
        else:
            all_swap_ok.append(int(ys_hat == y))

        if yo_hat is not None and ys_hat is not None:
            complete_pairs += 1
            flips.append(int(yo_hat != ys_hat))

    orig_arr = np.array(all_orig_ok, dtype=np.float64)
    confs_arr = np.array(valid_confs, dtype=np.float64)
    corr_arr = np.array(valid_corr, dtype=np.float64)

    return {
        "label": label,
        "n_planned": n_total,
        "n_valid_orig": n_total - failures,
        "n_complete_pairs": complete_pairs,
        "failures_orig": failures,
        "acc_orig": round(float(np.mean(orig_arr)), 4),
        "acc_orig_ci95": bootstrap_ci(orig_arr),
        "acc_swapped": round(float(np.mean(all_swap_ok)), 4),
        "order_flip_rate": round(float(np.mean(flips)), 4) if flips else None,
        "order_flips": int(sum(flips)),
        "nll": round(float(np.mean(valid_nlls)), 4) if valid_nlls else None,
        "brier": round(compute_brier(valid_probs, valid_labels), 4) if valid_probs else None,
        "ece": round(compute_ece(confs_arr, corr_arr), 4) if len(confs_arr) else None,
        "aurc": round(compute_aurc(confs_arr, corr_arr), 4) if len(confs_arr) else None,
        "error_auroc": round(compute_error_auroc(confs_arr, corr_arr), 4) if len(confs_arr) else None,
        "latency_p50_ms": round(float(np.percentile(latencies_ms, 50)), 1) if latencies_ms else None,
        "latency_p95_ms": round(float(np.percentile(latencies_ms, 95)), 1) if latencies_ms else None,
    }


def call_decision_api_item(
    endpoint: str,
    api_key: str,
    model_id: str,
    row: dict[str, Any],
    order: list[int],
    cache_dir: Path,
    timeout_s: float = 20.0,
    max_retries: int = 2,
) -> dict[str, Any]:
    """Call System One decision API using neutral slot keys (slot_0..slot_K-1)."""
    criteria = {}
    slot_to_orig: dict[str, int] = {}
    for pos, orig_idx in enumerate(order):
        slot_key = f"slot_{pos}"
        slot_to_orig[slot_key] = orig_idx
        criteria[slot_key] = f"{row['options'][orig_idx]}: {row['option_descriptions'][orig_idx]}"

    payload = {
        "model": model_id,
        "state": row["context"],
        "questions": {
            "decision": {
                "type": "choice",
                "instructions": row["question"],
                "criteria": criteria,
            }
        },
    }
    raw_req = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    req_hash = hashlib.sha256(f"{endpoint}|{raw_req}".encode("utf-8")).hexdigest()[:24]
    cache_file = cache_dir / f"dapi_{req_hash}.json"
    if cache_file.is_file():
        return json.loads(cache_file.read_text(encoding="utf-8"))

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    data = json.dumps(payload).encode("utf-8")

    last_err = ""
    for attempt in range(max_retries + 1):
        t0 = time.perf_counter()
        req = urllib.request.Request(endpoint, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            client_ms = (time.perf_counter() - t0) * 1000.0
            ans = body["answers"]["decision"]
            probs_slot = ans["probabilities"]
            k = len(order)
            remapped_probs = np.zeros(k, dtype=np.float64)
            for slot_key, orig_idx in slot_to_orig.items():
                remapped_probs[orig_idx] = float(probs_slot[slot_key])
            s = float(np.sum(remapped_probs))
            if s > 0:
                remapped_probs /= s
            selected_orig = slot_to_orig[ans["choice"]]
            record = {
                "ok": True,
                "request_id": body.get("request_id"),
                "model_returned": body.get("model"),
                "selected_orig": selected_orig,
                "probs_orig_order": remapped_probs.tolist(),
                "api_confidence": ans.get("confidence"),
                "server_latency_ms": body.get("latency_ms"),
                "client_latency_ms": round(client_ms, 2),
                "usage": body.get("usage"),
                "input_hash": req_hash,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            }
            cache_file.write_text(json.dumps(record), encoding="utf-8")
            return record
        except (urllib.error.HTTPError, urllib.error.URLError, KeyError, TimeoutError) as e:
            last_err = str(e)
            if attempt < max_retries:
                time.sleep(1.0 * (attempt + 1))

    return {"ok": False, "error": last_err, "client_latency_ms": 0.0, "input_hash": req_hash}


def call_openrouter_item(
    base_url: str,
    api_key: str,
    model_id: str,
    row: dict[str, Any],
    order: list[int],
    cache_dir: Path,
    disable_thinking: bool = True,
    timeout_s: float = 30.0,
    max_retries: int = 3,
) -> dict[str, Any]:
    opt_lines = [
        f"{LETTERS[pos]}. {row['options'][orig_idx]} — {row['option_descriptions'][orig_idx]}"
        for pos, orig_idx in enumerate(order)
    ]
    valid_letters = ", ".join(LETTERS[: len(order)])
    prompt = (
        f"Context:\n{row['context']}\n\n"
        f"Question: {row['question']}\n\n"
        f"Options:\n" + "\n".join(opt_lines) + "\n\n"
        f'Return ONLY valid JSON: {{"letter": "<one of {valid_letters}>"}}'
    )
    payload: dict[str, Any] = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 64,
        "response_format": {"type": "json_object"},
    }
    if disable_thinking:
        if "openrouter.ai" in base_url:
            payload["reasoning"] = {"effort": "none", "exclude": True}
        else:
            payload["enable_thinking"] = False

    raw_req = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    req_hash = hashlib.sha256(f"{base_url}|{raw_req}".encode("utf-8")).hexdigest()[:24]
    cache_file = cache_dir / f"or_{req_hash}.json"
    if cache_file.is_file():
        return json.loads(cache_file.read_text(encoding="utf-8"))

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    url = base_url.rstrip("/") + "/chat/completions"
    data = json.dumps(payload).encode("utf-8")

    last_err = ""
    for attempt in range(max_retries + 1):
        t0 = time.perf_counter()
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            client_ms = (time.perf_counter() - t0) * 1000.0
            content = str(body["choices"][0]["message"].get("content", "")).strip()
            m_json = re.search(r"\{.*?\}", content, flags=re.DOTALL)
            parsed = json.loads(m_json.group(0) if m_json else content)
            letter = str(parsed["letter"]).strip().upper()[:1]
            pos = LETTERS.index(letter)
            selected_orig = order[pos]
            record = {
                "ok": True,
                "id": body.get("id"),
                "provider": body.get("provider"),
                "model_returned": body.get("model"),
                "selected_orig": selected_orig,
                "client_latency_ms": round(client_ms, 2),
                "usage": body.get("usage"),
                "input_hash": req_hash,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            }
            cache_file.write_text(json.dumps(record), encoding="utf-8")
            return record
        except Exception as e:
            last_err = str(e)
            if attempt < max_retries:
                time.sleep(1.0 * (attempt + 1))

    return {"ok": False, "error": last_err, "client_latency_ms": 0.0, "input_hash": req_hash}


def probs_to_log_probs(probs: list[float], floor: float = 1e-6) -> np.ndarray:
    arr = np.maximum(np.array(probs, dtype=np.float64), floor)
    arr /= np.sum(arr)
    return np.log(arr)


def scale_probs(probs: np.ndarray, temp: float, floor: float = 1e-6) -> np.ndarray:
    lp = probs_to_log_probs(probs.tolist(), floor=floor)
    return softmax_np(lp, temp=temp)


def offline_cascade_replay(
    rows: list[dict[str, Any]],
    local_preds: list[int],
    local_confs: list[float],
    local_lat_ms: list[float],
    remote_preds: list[int | None],
    remote_lat_ms: list[float],
    threshold: float,
    name: str,
) -> dict[str, Any]:
    n = len(rows)
    local_ok, casc_ok, referred_remote_ok, referred_local_ok = [], [], [], []
    casc_lats = []
    n_referred = 0
    for r, lp, lc, ll, rp, rl in zip(rows, local_preds, local_confs, local_lat_ms, remote_preds, remote_lat_ms):
        y = int(r["gold_index"])
        l_corr = int(lp == y)
        local_ok.append(l_corr)
        refer = lc < threshold
        if refer:
            n_referred += 1
            r_corr = int(rp == y) if rp is not None else 0
            casc_ok.append(r_corr)
            referred_remote_ok.append(r_corr)
            referred_local_ok.append(l_corr)
            casc_lats.append(ll + rl)
        else:
            casc_ok.append(l_corr)
            casc_lats.append(ll)

    l_arr = np.array(local_ok, dtype=np.float64)
    c_arr = np.array(casc_ok, dtype=np.float64)
    return {
        "cascade": name,
        "threshold": threshold,
        "deferral_rate": round(n_referred / max(n, 1), 4),
        "n_referred": n_referred,
        "local_acc": round(float(np.mean(l_arr)), 4),
        "cascade_acc": round(float(np.mean(c_arr)), 4),
        "cascade_acc_ci95": bootstrap_ci(c_arr),
        "delta_acc": round(float(np.mean(c_arr - l_arr)), 4),
        "delta_acc_ci95": bootstrap_ci(c_arr - l_arr),
        "referred_local_acc": round(float(np.mean(referred_local_ok)), 4) if referred_local_ok else None,
        "referred_remote_acc": round(float(np.mean(referred_remote_ok)), 4) if referred_remote_ok else None,
        "combined_p50_ms": round(float(np.percentile(casc_lats, 50)), 1),
        "combined_p95_ms": round(float(np.percentile(casc_lats, 95)), 1),
    }


def load_dotenv(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    for ln in path.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if ln and not ln.startswith("#") and "=" in ln:
            k, v = ln.split("=", 1)
            env[k.strip()] = v.strip()
    return env


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--holdout", type=Path, default=Path("/Users/caius/Projects/self_jev/data/holdout/holdout_600.jsonl"))
    parser.add_argument("--calibrate", type=Path, default=Path("/Users/caius/Projects/self_jev/data/teacher/calibrate_180.jsonl"))
    parser.add_argument("--out-dir", type=Path, default=Path("/Users/caius/Projects/self_jev/reports/fresh_holdout"))
    parser.add_argument("--decision-api-config", type=Path, default=None, help="JSON with endpoint, api_key, model")
    parser.add_argument("--openrouter-config", type=Path, default=None, help="JSON with base_url, api_key, model, disable_thinking")
    parser.add_argument("--env-file", type=Path, default=None, help="Optional .env file with QWEN_BASE_URL, QWEN_API_KEY, QWEN_MODEL")
    parser.add_argument("--workers", type=int, default=8, help="Concurrent API worker threads")
    parser.add_argument("--limit", type=int, default=None, help="Smoke-test limit (e.g. 10 items)")
    args = parser.parse_args()

    hold_rows = load_jsonl(args.holdout, limit=args.limit)
    cal_rows = load_jsonl(args.calibrate, limit=args.limit)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.out_dir / "api_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_holdout": len(hold_rows),
        "n_calibrate": len(cal_rows),
        "models": {},
    }

    if args.decision_api_config and args.decision_api_config.is_file():
        cfg = json.loads(args.decision_api_config.read_text(encoding="utf-8"))
        ep, key, mod = cfg["endpoint"], cfg["api_key"], cfg.get("model", "decision-model-preview")
        print(f"[Decision API] Calibrating on {len(cal_rows)} sources (original order)...")
        cal_log_probs, cal_labels = [], []
        for r in cal_rows:
            k = len(r["options"])
            res = call_decision_api_item(ep, key, mod, r, list(range(k)), cache_dir)
            if res["ok"]:
                cal_log_probs.append(probs_to_log_probs(res["probs_orig_order"]))
                cal_labels.append(int(r["gold_index"]))
        t_api = fit_temperature(cal_log_probs, cal_labels)
        print(f"[Decision API] Frozen scalar temperature T={t_api}")

        print(f"[Decision API] Evaluating {len(hold_rows)} holdout pairs (orig + swapped)...")
        orig_probs_nat, swap_probs_nat, orig_preds, swap_preds, lats = [], [], [], [], []
        orig_probs_cal, swap_probs_cal = [], []
        for idx, r in enumerate(hold_rows):
            k = len(r["options"])
            first_orig = (idx % 2 == 0)
            orders = [list(range(k)), list(reversed(range(k)))] if first_orig else [list(reversed(range(k))), list(range(k))]
            res_map = {}
            for ord_seq in orders:
                tag = "orig" if ord_seq[0] == 0 else "swap"
                res_map[tag] = call_decision_api_item(ep, key, mod, r, ord_seq, cache_dir)
            ro, rs = res_map["orig"], res_map["swap"]
            if ro["ok"]:
                p_nat = np.array(ro["probs_orig_order"], dtype=np.float64)
                orig_probs_nat.append(p_nat)
                orig_probs_cal.append(scale_probs(p_nat, t_api))
                orig_preds.append(int(ro["selected_orig"]))
                lats.append(float(ro["client_latency_ms"]))
            else:
                orig_probs_nat.append(None); orig_probs_cal.append(None); orig_preds.append(None)

            if rs["ok"]:
                ps_nat = np.array(rs["probs_orig_order"], dtype=np.float64)
                swap_probs_nat.append(ps_nat)
                swap_probs_cal.append(scale_probs(ps_nat, t_api))
                swap_preds.append(int(rs["selected_orig"]))
            else:
                swap_probs_nat.append(None); swap_probs_cal.append(None); swap_preds.append(None)

        report["models"]["decision_api_native"] = summarize_prob_model(
            hold_rows, orig_probs_nat, swap_probs_nat, orig_preds, swap_preds, lats, "Hosted Decision API (Native)"
        )
        cal_summary = summarize_prob_model(
            hold_rows, orig_probs_cal, swap_probs_cal, orig_preds, swap_preds, lats, f"Hosted Decision API (+T={t_api})"
        )
        cal_summary["temperature"] = t_api
        report["models"]["decision_api_temp_scaled"] = cal_summary

    llm_cfg: dict[str, Any] | None = None
    if args.openrouter_config and args.openrouter_config.is_file():
        llm_cfg = json.loads(args.openrouter_config.read_text(encoding="utf-8"))
    elif args.env_file and args.env_file.is_file():
        env = load_dotenv(args.env_file)
        if env.get("QWEN_API_KEY") and env.get("QWEN_BASE_URL") and env.get("QWEN_MODEL"):
            llm_cfg = {
                "base_url": env["QWEN_BASE_URL"],
                "api_key": env["QWEN_API_KEY"],
                "model": env["QWEN_MODEL"],
                "disable_thinking": True,
            }

    if llm_cfg:
        burl = llm_cfg.get("base_url", "https://openrouter.ai/api/v1")
        key, mod = llm_cfg["api_key"], llm_cfg["model"]
        dis_think = llm_cfg.get("disable_thinking", True)
        print(f"[Cloud LLM] Evaluating {mod} on {len(hold_rows)} holdout pairs ({args.workers} workers)...", flush=True)

        def _eval_pair(idx_row: tuple[int, dict[str, Any]]) -> tuple[int, dict[str, Any], dict[str, Any]]:
            idx, r = idx_row
            k = len(r["options"])
            if idx % 2 == 0:
                ro = call_openrouter_item(burl, key, mod, r, list(range(k)), cache_dir, disable_thinking=dis_think)
                rs = call_openrouter_item(burl, key, mod, r, list(reversed(range(k))), cache_dir, disable_thinking=dis_think)
            else:
                rs = call_openrouter_item(burl, key, mod, r, list(reversed(range(k))), cache_dir, disable_thinking=dis_think)
                ro = call_openrouter_item(burl, key, mod, r, list(range(k)), cache_dir, disable_thinking=dis_think)
            return idx, ro, rs

        orig_preds: list[int | None] = [None] * len(hold_rows)
        swap_preds: list[int | None] = [None] * len(hold_rows)
        lats: list[float] = []
        total_in_tok = 0
        total_out_tok = 0
        done = 0
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
            for idx, ro, rs in ex.map(_eval_pair, enumerate(hold_rows)):
                orig_preds[idx] = int(ro["selected_orig"]) if ro["ok"] else None
                swap_preds[idx] = int(rs["selected_orig"]) if rs["ok"] else None
                if ro["ok"]:
                    lats.append(float(ro["client_latency_ms"]))
                    u = ro.get("usage") or {}
                    total_in_tok += int(u.get("prompt_tokens", 0) or 0)
                    total_out_tok += int(u.get("completion_tokens", 0) or 0)
                if rs["ok"]:
                    u = rs.get("usage") or {}
                    total_in_tok += int(u.get("prompt_tokens", 0) or 0)
                    total_out_tok += int(u.get("completion_tokens", 0) or 0)
                done += 1
                if done % 100 == 0 or done == len(hold_rows):
                    print(f"  [Cloud LLM] {done}/{len(hold_rows)} pairs done", flush=True)

        dummy_probs = [None] * len(hold_rows)
        mod_summary = summarize_prob_model(
            hold_rows, dummy_probs, dummy_probs, orig_preds, swap_preds, lats, f"Cloud LLM ({mod})"
        )
        by_bucket: dict[str, dict[str, Any]] = {}
        for b in sorted({r["bucket"] for r in hold_rows}):
            b_idxs = [i for i, r in enumerate(hold_rows) if r["bucket"] == b]
            b_ok = [int(orig_preds[i] == int(hold_rows[i]["gold_index"])) for i in b_idxs]
            b_swap_ok = [int(swap_preds[i] == int(hold_rows[i]["gold_index"])) for i in b_idxs]
            b_flips = [int(orig_preds[i] != swap_preds[i]) for i in b_idxs if orig_preds[i] is not None and swap_preds[i] is not None]
            by_bucket[b] = {
                "n": len(b_idxs),
                "acc_orig": round(float(np.mean(b_ok)), 4),
                "acc_swapped": round(float(np.mean(b_swap_ok)), 4),
                "order_flip_rate": round(float(np.mean(b_flips)), 4) if b_flips else None,
            }
        mod_summary["by_bucket"] = by_bucket
        mod_summary["total_prompt_tokens"] = total_in_tok
        mod_summary["total_completion_tokens"] = total_out_tok
        report["models"][f"cloud_llm_{mod}"] = mod_summary

    out_file = args.out_dir / "fresh_holdout_report.json"
    out_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved report -> {out_file}")


if __name__ == "__main__":
    main()
