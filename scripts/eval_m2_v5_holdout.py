#!/usr/bin/env python3
"""M2 (V5) Frozen Single-Pass Evaluation on Untouched holdout_600.jsonl.

Run ONLY ONCE after all training, dev checkpoint selection, and calibrate_180
temperature fitting are completely frozen.

Evaluates the complete predeclared roster on data/holdout/holdout_600.jsonl (orig + swapped):
  1. Zero-shot BAAI/bge-reranker-base (calibrated on calibrate_180)
  2. Trained BAAI/bge-reranker-base (all 3 seeds summarized + dev-selected deployment seed)
  3. Zero-shot Qwen/Qwen3.5-0.8B-Base (calibrated on calibrate_180, verified no Kev adapter)
  4. Trained Qwen/Qwen3.5-0.8B-Base (all 3 seeds summarized + dev-selected deployment seed)
  5. Frozen off-the-shelf jaredpalmer/kev-0.8b (optional --include-kev, enabled by default)
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import train_m2_v5_students as m2  # noqa: E402


def exact_mcnemar_p(b: int, c: int) -> float:
    """Two-sided exact binomial McNemar p-value for discordant counts (b, c)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    # Sum_{i=0..k} C(n, i) * 0.5^n, doubled, capped at 1.0
    tail = sum(math.comb(n, i) for i in range(k + 1)) * (0.5 ** n)
    return float(min(1.0, 2.0 * tail))


def pairwise_mcnemar(preds_a: list[int], preds_b: list[int], golds: list[int]) -> dict[str, Any]:
    a_only = sum(1 for pa, pb, g in zip(preds_a, preds_b, golds) if pa == g and pb != g)
    b_only = sum(1 for pa, pb, g in zip(preds_a, preds_b, golds) if pa != g and pb == g)
    return {
        "n": len(golds),
        "a_only_correct": a_only,
        "b_only_correct": b_only,
        "exact_p_value": round(exact_mcnemar_p(a_only, b_only), 6),
    }


def extract_preds(rows: list[dict[str, Any]], orig_logits: list[np.ndarray], temp: float) -> list[int]:
    return [int(np.argmax(m2.softmax_np(lg, temp))) for lg in orig_logits]


def eval_kev_holdout(
    rows: list[dict[str, Any]],
    kev_vendor_dir: Path,
    device: torch.device,
) -> tuple[dict[str, Any], list[int]]:
    if str(kev_vendor_dir) not in sys.path:
        sys.path.insert(0, str(kev_vendor_dir))
    from kev.api import Choice, SystemOneRequest, to_answers, to_record
    from kev.checkpoint import Checkpoint, LoadOptions
    from kev.model import SERVE_MAX_BRANCH, SERVE_MAX_STATE

    dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
    options = LoadOptions(backend="torch", dtype=dtype, merge=True)
    tok, model = Checkpoint("jaredpalmer/kev-0.8b").load(str(device), options)

    def _score_order(r: dict[str, Any], order: list[int]) -> tuple[np.ndarray, float]:
        opts = [r["options"][i] for i in order]
        descs = [r["option_descriptions"][i] for i in order]
        criteria = {o: d for o, d in zip(opts, descs)}
        req = SystemOneRequest(
            state=r["context"],
            model="jaredpalmer/kev-0.8b",
            questions={"decision": Choice(type="choice", instructions=r["question"], criteria=criteria)},
        )
        record, meta = to_record(req)
        encoded = model.encode(tok, record, max_state=SERVE_MAX_STATE, max_branch=SERVE_MAX_BRANCH, strict=True)
        t0 = time.perf_counter()
        probs = model.probs(encoded)
        if device.type == "cuda":
            torch.cuda.synchronize()
        lat_ms = (time.perf_counter() - t0) * 1000.0
        ans = to_answers([[float(x) for x in row.tolist()] for row in probs], meta)
        pmap = ans["decision"].get("probabilities") or {}
        p_vec = np.array([max(float(pmap.get(o, 1e-12)), 1e-12) for o in opts], dtype=np.float64)
        return np.log(p_vec), lat_ms

    orig_logits, swap_logits, lats = [], [], []
    for r in rows:
        k = len(r["options"])
        lg_o, lat_ms = _score_order(r, list(range(k)))
        lg_s, _ = _score_order(r, list(reversed(range(k))))
        orig_logits.append(lg_o)
        swap_logits.append(lg_s)
        lats.append(lat_ms)

    summary = m2.summarize_eval(rows, orig_logits, swap_logits, 1.0, lats)
    preds = extract_preds(rows, orig_logits, 1.0)
    del model, tok
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary, preds


def main() -> None:
    ap = argparse.ArgumentParser(description="M2 (V5) Frozen Single-Pass Holdout-600 Evaluation")
    ap.add_argument("--train-out-dir", type=Path, required=True, help="Directory with m2_v5_train_summary.json & checkpoints")
    ap.add_argument("--holdout-file", type=Path, required=True, help="Path to data/holdout/holdout_600.jsonl")
    ap.add_argument("--data-dir", type=Path, required=True, help="Path to data/teacher/ (for zero-overlap check)")
    ap.add_argument("--kev-vendor-dir", type=Path, default=None, help="Path to decision_eval/vendor/kev for frozen Kev eval")
    ap.add_argument("--skip-kev", action="store_true")
    args = ap.parse_args()

    train_sum_path = args.train_out_dir / "m2_v5_train_summary.json"
    train_sum = json.loads(train_sum_path.read_text(encoding="utf-8"))
    holdout_rows = m2.load_jsonl(args.holdout_file)

    # Zero-leakage assertion against all non-test splits
    hold_srcs = {r.get("source_id") for r in holdout_rows if r.get("source_id")}
    assert len(hold_srcs) == len(holdout_rows), f"Expected 600 unique source_ids, got {len(hold_srcs)}/{len(holdout_rows)}"
    for split_fn in ("deepseek_v41_flash_verified_train.jsonl", "dev_180.jsonl", "calibrate_180.jsonl"):
        p = args.data_dir / split_fn
        if p.is_file():
            s_ids = {r.get("source_id") for r in m2.load_jsonl(p) if r.get("source_id")}
            assert not (hold_srcs & s_ids), f"Holdout leakage detected against {split_fn}!"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    golds = [int(r["gold_index"]) for r in holdout_rows]
    preds_map: dict[str, list[int]] = {}
    results: dict[str, Any] = {
        "protocol": "M2_V5_frozen_single_pass_holdout_600",
        "holdout_file": str(args.holdout_file.name),
        "holdout_n": len(holdout_rows),
        "holdout_unique_source_ids": len(hold_srcs),
        "holdout_sha256_prefix": m2.sha256_file(args.holdout_file),
        "models": {},
    }

    # 1. BGE zero-shot + 3 trained seeds
    if "bge" in train_sum:
        bge_info = train_sum["bge"]
        bge_id = bge_info["model_id"]
        tok = AutoTokenizer.from_pretrained(bge_id)

        print(f"\n[Holdout] Evaluating BGE zero-shot: {bge_id}")
        t_fz = float(bge_info["frozen_baseline"]["temperature"])
        mod = AutoModelForSequenceClassification.from_pretrained(bge_id).to(device)
        o_lg, s_lg, lats, _ = m2.eval_bge(mod, tok, holdout_rows, device)
        results["models"]["bge_zeroshot"] = m2.summarize_eval(holdout_rows, o_lg, s_lg, t_fz, lats)
        preds_map["bge_zeroshot"] = extract_preds(holdout_rows, o_lg, t_fz)
        del mod
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        bge_seed_metrics = []
        deploy_seed = bge_info["deployment_seed_selected_on_dev"]
        for s_rec in bge_info["seeds"]:
            seed = s_rec["seed"]
            t_cal = float(s_rec["temperature"])
            ckpt = Path(s_rec["checkpoint"])
            print(f"[Holdout] Evaluating BGE trained seed={seed} (T={t_cal})")
            mod = AutoModelForSequenceClassification.from_pretrained(ckpt).to(device)
            o_lg, s_lg, lats, _ = m2.eval_bge(mod, tok, holdout_rows, device)
            met = m2.summarize_eval(holdout_rows, o_lg, s_lg, t_cal, lats)
            met["seed"] = seed
            bge_seed_metrics.append(met)
            if seed == deploy_seed:
                results["models"]["bge_trained_deploy"] = met
                preds_map["bge_trained_deploy"] = extract_preds(holdout_rows, o_lg, t_cal)
            del mod
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        results["models"]["bge_trained_3seed_aggregate"] = m2.aggregate_seeds(bge_seed_metrics)
        results["models"]["bge_trained_per_seed"] = bge_seed_metrics

    # 2. Qwen zero-shot + 3 trained seeds
    if "qwen" in train_sum:
        qw_info = train_sum["qwen"]
        qw_id = qw_info["model_id"]
        letter_ids = qw_info["letter_ids"]
        tok = AutoTokenizer.from_pretrained(qw_id, trust_remote_code=True)
        load_dtype = torch.float16 if device.type == "cuda" else torch.float32

        print(f"\n[Holdout] Evaluating Qwen zero-shot: {qw_id}")
        t_fz = float(qw_info["frozen_baseline"]["temperature"])
        mod = AutoModelForCausalLM.from_pretrained(qw_id, torch_dtype=load_dtype, trust_remote_code=True).to(device)
        o_lg, s_lg, lats, _ = m2.eval_qwen(mod, tok, letter_ids, holdout_rows, device)
        results["models"]["qwen_zeroshot"] = m2.summarize_eval(holdout_rows, o_lg, s_lg, t_fz, lats)
        preds_map["qwen_zeroshot"] = extract_preds(holdout_rows, o_lg, t_fz)
        del mod
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        qw_seed_metrics = []
        deploy_seed = qw_info["deployment_seed_selected_on_dev"]
        for s_rec in qw_info["seeds"]:
            seed = s_rec["seed"]
            t_cal = float(s_rec["temperature"])
            ckpt = Path(s_rec["checkpoint"])
            acfg = json.loads((ckpt / "adapter_config.json").read_text())
            print(f"[Holdout] Evaluating Qwen trained seed={seed} (T={t_cal})")
            mod = AutoModelForCausalLM.from_pretrained(qw_id, torch_dtype=load_dtype, trust_remote_code=True).to(device)
            m2.inject_lora(mod, tuple(acfg["target_modules"]), r=int(acfg["lora_r"]), alpha=float(acfg["lora_alpha"]))
            m2.load_lora_state_dict(mod, torch.load(ckpt / "lora_adapters.pt", map_location="cpu"))
            mod.to(device)
            o_lg, s_lg, lats, _ = m2.eval_qwen(mod, tok, letter_ids, holdout_rows, device)
            met = m2.summarize_eval(holdout_rows, o_lg, s_lg, t_cal, lats)
            met["seed"] = seed
            qw_seed_metrics.append(met)
            if seed == deploy_seed:
                results["models"]["qwen_trained_deploy"] = met
                preds_map["qwen_trained_deploy"] = extract_preds(holdout_rows, o_lg, t_cal)
            del mod
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        results["models"]["qwen_trained_3seed_aggregate"] = m2.aggregate_seeds(qw_seed_metrics)
        results["models"]["qwen_trained_per_seed"] = qw_seed_metrics

    # 3. Frozen off-the-shelf jaredpalmer/kev-0.8b reference
    if not args.skip_kev and args.kev_vendor_dir and args.kev_vendor_dir.is_dir():
        print("\n[Holdout] Evaluating frozen reference: jaredpalmer/kev-0.8b")
        kev_met, kev_preds = eval_kev_holdout(holdout_rows, args.kev_vendor_dir, device)
        results["models"]["kev_0.8b_frozen"] = kev_met
        preds_map["kev_0.8b_frozen"] = kev_preds

    # 4. Pairwise exact McNemar comparisons on holdout_600
    pairs = [
        ("bge_trained_deploy", "bge_zeroshot"),
        ("qwen_trained_deploy", "qwen_zeroshot"),
        ("bge_trained_deploy", "qwen_trained_deploy"),
        ("bge_trained_deploy", "kev_0.8b_frozen"),
        ("qwen_trained_deploy", "kev_0.8b_frozen"),
    ]
    results["mcnemar_pairwise"] = {
        f"{a}__vs__{b}": pairwise_mcnemar(preds_map[a], preds_map[b], golds)
        for a, b in pairs
        if a in preds_map and b in preds_map
    }

    out_report = args.train_out_dir / "m2_v5_holdout_report.json"
    out_report.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    # Save per-item predictions for independent auditing
    out_preds = args.train_out_dir / "m2_v5_holdout_predictions.jsonl"
    with out_preds.open("w", encoding="utf-8") as fh:
        for i, r in enumerate(holdout_rows):
            rec = {
                "id": r["id"],
                "source_id": r["source_id"],
                "bucket": r["bucket"],
                "context_band": r.get("context_band"),
                "gold_index": int(r["gold_index"]),
                "is_abstain_gold": bool(r.get("is_abstain_gold")),
                "preds": {m: preds_map[m][i] for m in preds_map},
            }
            fh.write(json.dumps(rec) + "\n")

    print(f"\nSaved frozen holdout-600 report -> {out_report}")
    print(f"Saved per-item holdout predictions -> {out_preds}")


if __name__ == "__main__":
    main()
