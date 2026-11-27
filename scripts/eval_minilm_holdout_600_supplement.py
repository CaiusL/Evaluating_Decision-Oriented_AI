#!/usr/bin/env python3
"""Bounded Option D Supplement — Step 1: Off-the-shelf MS MARCO MiniLM on holdout_600.

Evaluates the precisely identified local checkpoint `models/ms-marco-MiniLM-L6-v2`
(cross-encoder/ms-marco-MiniLM-L6-v2, 22.7M, off-the-shelf, no Self-JEV fine-tuning)
as an additional frozen reference on `data/holdout/holdout_600.jsonl` (orig + swapped),
with scalar temperature T fitted strictly on `data/teacher/calibrate_180.jsonl`.

Preserves `reports/m2_v5/` untouched; writes strictly to `reports/alignment_supplement/`.
"""
from __future__ import annotations

import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import train_m2_v5_students as m2  # noqa: E402


def exact_mcnemar_p(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) * (0.5 ** n)
    return float(min(1.0, 2.0 * tail))


def abstention_audit(rows: list[dict[str, Any]], preds: list[int]) -> dict[str, Any]:
    tp = fp = fn = tn = 0
    for r, p in zip(rows, preds):
        gold_abs = bool(r.get("is_abstain_gold"))
        abs_idx = r.get("abstain_index")
        if abs_idx is None:
            # Infer abstain option index from option text if not explicitly tagged
            abs_idx = next(
                (i for i, o in enumerate(r["options"]) if "abstain" in o.lower() or "escalat" in o.lower()),
                len(r["options"]) - 1,
            )
        pred_abs = (p == int(abs_idx))
        if gold_abs and pred_abs:
            tp += 1
        elif (not gold_abs) and pred_abs:
            fp += 1
        elif gold_abs and (not pred_abs):
            fn += 1
        else:
            tn += 1
    recall = tp / max(tp + fn, 1)
    precision = tp / max(tp + fp, 1)
    far = fp / max(fp + tn, 1)
    return {
        "gold_abstain_n": tp + fn,
        "gold_non_abstain_n": fp + tn,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "recall": round(recall, 4),
        "precision": round(precision, 4),
        "false_abstention_rate": round(far, 4),
    }


def main() -> None:
    t0 = time.perf_counter()
    ckpt_dir = REPO_ROOT / "models" / "ms-marco-MiniLM-L6-v2"
    cal_path = REPO_ROOT / "data" / "teacher" / "calibrate_180.jsonl"
    holdout_path = REPO_ROOT / "data" / "holdout" / "holdout_600.jsonl"
    out_dir = REPO_ROOT / "reports" / "alignment_supplement"
    out_dir.mkdir(parents=True, exist_ok=True)

    cal_rows = m2.load_jsonl(cal_path)
    holdout_rows = m2.load_jsonl(holdout_path)

    # Zero-overlap verification
    cal_srcs = {r.get("source_id") for r in cal_rows if r.get("source_id")}
    hold_srcs = {r.get("source_id") for r in holdout_rows if r.get("source_id")}
    assert not (cal_srcs & hold_srcs), "Leakage detected between calibrate_180 and holdout_600!"

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[MiniLM Supplement] Loading {ckpt_dir} on {device}...", flush=True)
    tok = AutoTokenizer.from_pretrained(str(ckpt_dir), local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(str(ckpt_dir), local_files_only=True).to(device)
    model.eval()
    total_params = sum(p.numel() for p in model.parameters())

    print(f"[MiniLM Supplement] Fitting temperature T on calibrate_180 (n={len(cal_rows)})...", flush=True)
    cal_o, cal_s, cal_lat, cal_max_tok = m2.eval_bge(model, tok, cal_rows, device, max_len=512)
    t_cal = m2.fit_temperature(cal_o, [int(r["gold_index"]) for r in cal_rows])
    cal_summary = m2.summarize_eval(cal_rows, cal_o, cal_s, t_cal, cal_lat)

    print(f"[MiniLM Supplement] Evaluating holdout_600 (n={len(holdout_rows)}, T={t_cal})...", flush=True)
    ho_o, ho_s, ho_lat, ho_max_tok = m2.eval_bge(model, tok, holdout_rows, device, max_len=512)
    ho_uncal = m2.summarize_eval(holdout_rows, ho_o, ho_s, 1.0, ho_lat)
    ho_cal = m2.summarize_eval(holdout_rows, ho_o, ho_s, t_cal, ho_lat)
    preds = [int(np.argmax(m2.softmax_np(lg, t_cal))) for lg in ho_o]
    abs_audit = abstention_audit(holdout_rows, preds)

    # Optional pairwise McNemar against frozen M2 V5 holdout predictions
    mcnemar: dict[str, Any] = {}
    m2_preds_path = REPO_ROOT / "reports" / "m2_v5" / "m2_v5_holdout_predictions.jsonl"
    if m2_preds_path.is_file():
        m2_rows = m2.load_jsonl(m2_preds_path)
        if len(m2_rows) == len(holdout_rows):
            golds = [int(r["gold_index"]) for r in holdout_rows]
            other_models = list(m2_rows[0].get("preds", {}).keys())
            for om in other_models:
                opreds = [int(r["preds"][om]) for r in m2_rows]
                a_only = sum(1 for pa, pb, g in zip(preds, opreds, golds) if pa == g and pb != g)
                b_only = sum(1 for pa, pb, g in zip(preds, opreds, golds) if pa != g and pb == g)
                mcnemar[f"minilm_offshelf__vs__{om}"] = {
                    "n": len(golds),
                    "a_only_correct": a_only,
                    "b_only_correct": b_only,
                    "exact_p_value": round(exact_mcnemar_p(a_only, b_only), 6),
                }

    report = {
        "protocol": "Bounded_Option_D_MiniLM_Holdout_600_Supplement",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "role": "Additional frozen off-the-shelf reference (not a teacher-trained student)",
        "checkpoint": {
            "hf_model_id": "cross-encoder/ms-marco-MiniLM-L6-v2",
            "local_dir": str(ckpt_dir.relative_to(REPO_ROOT)),
            "config_sha256_prefix": m2.sha256_file(ckpt_dir / "config.json"),
            "total_params": int(total_params),
            "device": str(device),
            "max_length": 512,
            "cal_max_tokens_seen": cal_max_tok,
            "holdout_max_tokens_seen": ho_max_tok,
        },
        "splits": {
            "calibrate_180_sha256_prefix": m2.sha256_file(cal_path),
            "holdout_600_sha256_prefix": m2.sha256_file(holdout_path),
        },
        "temperature_fitted_on_cal_180": t_cal,
        "cal_180_calibrated": cal_summary,
        "holdout_600_uncalibrated": ho_uncal,
        "holdout_600_calibrated": ho_cal,
        "holdout_600_abstention_audit": abs_audit,
        "mcnemar_vs_m2_v5": mcnemar,
        "wall_s": round(time.perf_counter() - t0, 1),
    }

    out_path = out_dir / "minilm_holdout_600_supplement.json"
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"[MiniLM Supplement DONE] acc_orig={ho_cal['acc_orig']*100:.2f}% "
        f"acc_swap={ho_cal['acc_swapped']*100:.2f}% flip={ho_cal['order_flip_rate']*100:.2f}% "
        f"nll={ho_cal['nll']:.3f} ece={ho_cal['ece']:.3f} aurc={ho_cal['aurc']:.3f} "
        f"T={t_cal} p50={ho_cal['latency_p50_ms']:.1f}ms ({report['wall_s']}s)",
        flush=True,
    )
    print(f"Abstention: {abs_audit}", flush=True)
    print(f"By bucket: {ho_cal['by_bucket']}", flush=True)
    print(f"By band: {ho_cal['by_band']}", flush=True)


if __name__ == "__main__":
    main()
