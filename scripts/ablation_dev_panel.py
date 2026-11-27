"""Controlled 4-condition development panel comparing Arm A (Stage A only) vs. Arm B (Stage A + B).

Strictly uses permitted non-test development sources (NEVER touches data/benchmarks/all_test.jsonl):
  - 4-condition paired sufficiency/noise panel (16 disjoint dev sources x 4 conditions = 64 paired checks):
      1. Clean, sufficient evidence     -> should select supported action
      2. Wrapped, sufficient evidence   -> should select same supported action (tests false-abstain on noise)
      3. Clean, insufficient evidence   -> should select unresolved/abstain
      4. Wrapped, insufficient evidence -> should select unresolved/abstain
  - Real out-of-fold validation panel (each fold evaluated ONLY by its own out-of-fold checkpoint,
    both clean and wrapped with the held-out `dev` wrapper family).
  - Logs exact unresolved-vs-best-substantive-action margins (z_U - z_{a*}) and recoverable false-abstention rate.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from train_self_jev_minilm import (
    generate_sufficiency_curriculum,
    load_nontest_pool,
    make_grouped_stratified_folds,
    score_item_logits,
    wrap_with_distractors,
)


def _load_fold_models(model_dir: Path, device: torch.device) -> tuple[Any, list[torch.nn.Module], float]:
    cfg = json.loads((model_dir / "ensemble_config.json").read_text(encoding="utf-8"))
    fold_dirs = [model_dir / f for f in cfg["folds"]]
    tok = AutoTokenizer.from_pretrained(fold_dirs[0])
    models = [AutoModelForSequenceClassification.from_pretrained(fd).to(device).eval() for fd in fold_dirs]
    temp = float(cfg.get("temperature", 1.0))
    return tok, models, temp


def _score_ensemble_logits(
    models: list[torch.nn.Module],
    tokenizer: Any,
    item: dict[str, Any],
    device: torch.device,
    use_amp: bool,
) -> np.ndarray:
    with torch.inference_mode():
        acc = None
        for m in models:
            with torch.amp.autocast("cuda", enabled=use_amp):
                lg = score_item_logits(m, tokenizer, item, device, use_retrieval=True).float()
            acc = lg if acc is None else (acc + lg)
        avg = (acc / float(len(models))).detach().cpu().numpy()
    return avg


def _summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(records)
    if n == 0:
        return {"n": 0}
    acc = sum(r["correct"] for r in records) / n
    false_abs = [r for r in records if r["is_false_abstain"]]
    false_acc = [r for r in records if r["is_false_accept"]]
    suff_items = [r for r in records if not r["is_abstain_gold"]]
    insuff_items = [r for r in records if r["is_abstain_gold"]]
    recoverable = [r for r in false_abs if r["best_substantive_correct"]]
    margins_fa = [r["unresolved_margin"] for r in false_abs if r["unresolved_margin"] is not None]
    margins_insuff = [r["unresolved_margin"] for r in insuff_items if r["unresolved_margin"] is not None]
    return {
        "n": n,
        "accuracy": round(acc, 4),
        "correct_count": sum(r["correct"] for r in records),
        "false_abstain_rate": round(len(false_abs) / max(1, len(suff_items)), 4) if suff_items else None,
        "false_accept_rate": round(len(false_acc) / max(1, len(insuff_items)), 4) if insuff_items else None,
        "false_abstain_count": len(false_abs),
        "recoverable_false_abstain_count": len(recoverable),
        "best_substantive_correct_among_fa": round(len(recoverable) / max(1, len(false_abs)), 4) if false_abs else None,
        "mean_margin_false_abstain": round(float(np.mean(margins_fa)), 4) if margins_fa else None,
        "mean_margin_true_insufficient": round(float(np.mean(margins_insuff)), 4) if margins_insuff else None,
    }


def _record_from_logits(item: dict[str, Any], logits: np.ndarray, condition: str) -> dict[str, Any]:
    gold = int(item["gold_index"])
    pred = int(np.argmax(logits))
    n_opts = len(item["options"])
    abstain_idx = item.get("abstain_index")
    if abstain_idx is None:
        # Detect explicit unresolved/abstain option by name if present
        for i, o in enumerate(item["options"]):
            ol = o.lower()
            if "abstain" in ol or "unresolved" in ol:
                abstain_idx = i
                break
    is_abs_gold = (abstain_idx is not None and gold == abstain_idx)
    is_false_abstain = (abstain_idx is not None and not is_abs_gold and pred == abstain_idx)
    is_false_accept = (abstain_idx is not None and is_abs_gold and pred != abstain_idx)

    best_sub_idx = None
    best_sub_correct = False
    margin = None
    if abstain_idx is not None and n_opts >= 2:
        sub_indices = [i for i in range(n_opts) if i != abstain_idx]
        best_sub_idx = max(sub_indices, key=lambda i: float(logits[i]))
        best_sub_correct = (best_sub_idx == gold)
        margin = float(logits[abstain_idx] - logits[best_sub_idx])

    return {
        "id": item["id"],
        "condition": condition,
        "gold": gold,
        "pred": pred,
        "correct": (pred == gold),
        "abstain_idx": abstain_idx,
        "is_abstain_gold": is_abs_gold,
        "is_false_abstain": is_false_abstain,
        "is_false_accept": is_false_accept,
        "best_substantive_idx": best_sub_idx,
        "best_substantive_correct": best_sub_correct,
        "unresolved_margin": margin,
    }


def evaluate_arm(name: str, model_dir: Path, device: torch.device, seed: int = 20260401) -> dict[str, Any]:
    use_amp = device.type == "cuda"
    tok, models, temp = _load_fold_models(model_dir, device)

    # 1. 4-condition paired sufficiency/noise dev panel (16 disjoint dev sources)
    suff_dev = generate_sufficiency_curriculum(n_sources=16, seed=seed + 909)
    rng = random.Random(seed + 777)
    panel_records: dict[str, list[dict[str, Any]]] = {
        "clean_sufficient": [],
        "wrapped_sufficient": [],
        "clean_insufficient": [],
        "wrapped_insufficient": [],
    }
    for it in suff_dev:
        if it["bucket"] == "suff_irrel_removed":
            continue
        is_insuff = (it["gold_index"] == len(it["options"]) - 1)
        c_key = "clean_insufficient" if is_insuff else "clean_sufficient"
        w_key = "wrapped_insufficient" if is_insuff else "wrapped_sufficient"

        lg_clean = _score_ensemble_logits(models, tok, it, device, use_amp)
        panel_records[c_key].append(_record_from_logits(it, lg_clean, c_key))

        wrapped_it = wrap_with_distractors(it, rng, family="dev", total_words=700)
        lg_wrap = _score_ensemble_logits(models, tok, wrapped_it, device, use_amp)
        panel_records[w_key].append(_record_from_logits(wrapped_it, lg_wrap, w_key))

    # 2. Strict out-of-fold real dev evaluation (each fold scored ONLY by its own OOF model)
    pool = load_nontest_pool()
    folds = make_grouped_stratified_folds(pool, n_folds=len(models), seed=seed)
    oof_clean_recs: list[dict[str, Any]] = []
    oof_wrap_recs: list[dict[str, Any]] = []
    rng_oof = random.Random(seed + 555)
    for fi, val_items in enumerate(folds):
        mdl = models[fi]
        for it in val_items:
            with torch.inference_mode():
                with torch.amp.autocast("cuda", enabled=use_amp):
                    lg_c = score_item_logits(mdl, tok, it, device, use_retrieval=True).float().detach().cpu().numpy()
                w_it = wrap_with_distractors(it, rng_oof, family="dev", total_words=700)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    lg_w = score_item_logits(mdl, tok, w_it, device, use_retrieval=True).float().detach().cpu().numpy()
            oof_clean_recs.append(_record_from_logits(it, lg_c, "oof_clean"))
            oof_wrap_recs.append(_record_from_logits(w_it, lg_w, "oof_wrapped"))

    return {
        "arm": name,
        "model_dir": str(model_dir),
        "temperature": temp,
        "four_condition_panel": {k: _summarize_records(v) for k, v in panel_records.items()},
        "real_oof_clean": _summarize_records(oof_clean_recs),
        "real_oof_wrapped": _summarize_records(oof_wrap_recs),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 4-condition dev ablation panel comparing Stage A vs Stage A+B.")
    parser.add_argument("--arm-a-dir", type=Path, required=True, help="Stage A only model directory.")
    parser.add_argument("--arm-b-dir", type=Path, default=None, help="Optional Stage A+B model directory for direct comparison.")
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260401)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    report: dict[str, Any] = {"device": str(device), "arms": {}}
    report["arms"]["arm_a_stage_a_only"] = evaluate_arm("arm_a_stage_a_only", args.arm_a_dir, device, seed=args.seed)
    if args.arm_b_dir and (args.arm_b_dir / "ensemble_config.json").is_file():
        report["arms"]["arm_b_stage_a_plus_b"] = evaluate_arm("arm_b_stage_a_plus_b", args.arm_b_dir, device, seed=args.seed)

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Saved ablation dev report to {args.out_json}\n")
    for arm_name, data in report["arms"].items():
        print(f"=== {arm_name} ===")
        print(f"  Real OOF clean  : acc={data['real_oof_clean']['accuracy']*100:.1f}% | FA={data['real_oof_clean']['false_abstain_count']} (recov={data['real_oof_clean']['recoverable_false_abstain_count']})")
        print(f"  Real OOF wrapped: acc={data['real_oof_wrapped']['accuracy']*100:.1f}% | FA={data['real_oof_wrapped']['false_abstain_count']} (recov={data['real_oof_wrapped']['recoverable_false_abstain_count']})")
        for ck, cv in data["four_condition_panel"].items():
            print(f"  {ck:20s}: acc={cv['accuracy']*100:5.1f}% (n={cv['n']}) | FA={cv['false_abstain_rate']} | FAcc={cv['false_accept_rate']} | margin_fa={cv['mean_margin_false_abstain']}")


if __name__ == "__main__":
    main()
