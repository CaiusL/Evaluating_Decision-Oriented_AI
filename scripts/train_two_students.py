#!/usr/bin/env python3
"""Controlled Two-Student Training & Calibration Pipeline (Colab T4 Safe).

Trains and evaluates two single-checkpoint students on identical verified splits:
  1. Student 1: BAAI/bge-reranker-base (278M XLM-RoBERTa cross-encoder, full FT)
  2. Student 2: Qwen/Qwen3.5-0.8B-Base (800M causal LM, option-letter logits, native FP32 LoRA)

Strict data hygiene:
  - Train on train_840.jsonl
  - Select best epoch on dev_180.jsonl
  - Fit scalar temperature T strictly on calibrate_180.jsonl
  - Evaluate both frozen zero-shot baselines and 3 trained seeds (orig + swapped order)
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

LETTERS = ["A", "B", "C", "D", "E", "F", "G", "H"]


# ---------------------------------------------------------------------------
# Reproducibility & Metrics
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    return rows[:limit] if limit else rows


def softmax_np(logits: np.ndarray, temp: float = 1.0) -> np.ndarray:
    z = logits / max(temp, 1e-4)
    z = z - np.max(z)
    e = np.exp(z)
    return e / np.sum(e)


def compute_ece(confs: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(confs)
    if n == 0:
        return 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (confs >= lo) & (confs <= hi if i == n_bins - 1 else confs < hi)
        if np.any(mask):
            ece += (np.sum(mask) / n) * abs(float(np.mean(correct[mask])) - float(np.mean(confs[mask])))
    return float(ece)


def compute_aurc(confs: np.ndarray, correct: np.ndarray) -> float:
    n = len(confs)
    if n == 0:
        return 0.0
    order = np.argsort(-confs)
    sorted_err = 1.0 - correct[order]
    cum_err = np.cumsum(sorted_err)
    risks = cum_err / np.arange(1, n + 1)
    return float(np.mean(risks))


def compute_error_auroc(confs: np.ndarray, correct: np.ndarray) -> float:
    # Score = 1 - conf (uncertainty); positive class = error (correct == 0)
    errors = (1.0 - correct).astype(np.int32)
    n_pos = int(np.sum(errors == 1))
    n_neg = int(np.sum(errors == 0))
    if n_pos == 0 or n_neg == 0:
        return 0.5
    scores = 1.0 - confs
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    # Average ranks for ties
    sorted_s = scores[order]
    i = 0
    while i < len(sorted_s):
        j = i
        while j + 1 < len(sorted_s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        if j > i:
            avg_rank = 0.5 * (i + j) + 1.0
            ranks[order[i : j + 1]] = avg_rank
        i = j + 1
    sum_pos_ranks = float(np.sum(ranks[errors == 1]))
    auroc = (sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auroc)


def fit_temperature(logits_list: list[np.ndarray], labels: list[int]) -> float:
    """Grid + local refinement for scalar temperature T > 0 minimizing NLL."""
    def nll_at(t: float) -> float:
        total = 0.0
        for lg, y in zip(logits_list, labels):
            p = softmax_np(lg, t)
            total -= math.log(max(float(p[y]), 1e-12))
        return total / max(len(labels), 1)

    grid = np.concatenate([np.linspace(0.25, 2.0, 36), np.linspace(2.2, 6.0, 20)])
    best_t = float(min(grid, key=nll_at))
    # Local fine search around best_t
    lo = max(0.1, best_t - 0.15)
    hi = best_t + 0.15
    fine = np.linspace(lo, hi, 31)
    best_t = float(min(fine, key=nll_at))
    return round(best_t, 4)


def summarize_eval(
    rows: list[dict[str, Any]],
    orig_logits: list[np.ndarray],
    swap_logits: list[np.ndarray],
    temp: float,
    latencies_ms: list[float],
) -> dict[str, Any]:
    confs, correct, nlls = [], [], []
    orig_preds, swap_preds_unmapped = [], []
    by_bucket: dict[str, list[int]] = {}
    by_band: dict[str, list[int]] = {}
    abstain_corr, non_abstain_corr = [], []

    for r, lg_o, lg_s in zip(rows, orig_logits, swap_logits):
        y = int(r["gold_index"])
        k = len(r["options"])
        p_o = softmax_np(lg_o, temp)
        pred_o = int(np.argmax(p_o))
        conf_o = float(p_o[pred_o])
        is_ok = int(pred_o == y)

        # Swapped reverses options: index j in swapped corresponds to (k - 1 - j) in orig
        p_s = softmax_np(lg_s, temp)
        pred_s_raw = int(np.argmax(p_s))
        pred_s_orig = (k - 1) - pred_s_raw

        confs.append(conf_o)
        correct.append(is_ok)
        nlls.append(-math.log(max(float(p_o[y]), 1e-12)))
        orig_preds.append(pred_o)
        swap_preds_unmapped.append(pred_s_orig)

        by_bucket.setdefault(r["bucket"], []).append(is_ok)
        by_band.setdefault(r.get("context_band", "clean"), []).append(is_ok)
        if r.get("is_abstain_gold"):
            abstain_corr.append(is_ok)
        else:
            non_abstain_corr.append(is_ok)

    confs_arr = np.array(confs, dtype=np.float64)
    corr_arr = np.array(correct, dtype=np.float64)
    swap_ok = [int(p == int(r["gold_index"])) for p, r in zip(swap_preds_unmapped, rows)]
    flips = [int(a != b) for a, b in zip(orig_preds, swap_preds_unmapped)]

    return {
        "n": len(rows),
        "temperature": float(temp),
        "acc_orig": float(np.mean(corr_arr)),
        "acc_swapped": float(np.mean(swap_ok)),
        "order_flip_rate": float(np.mean(flips)),
        "order_flips": int(sum(flips)),
        "nll": float(np.mean(nlls)),
        "ece": compute_ece(confs_arr, corr_arr),
        "aurc": compute_aurc(confs_arr, corr_arr),
        "error_auroc": compute_error_auroc(confs_arr, corr_arr),
        "abstain_acc": float(np.mean(abstain_corr)) if abstain_corr else None,
        "non_abstain_acc": float(np.mean(non_abstain_corr)) if non_abstain_corr else None,
        "by_bucket": {b: float(np.mean(v)) for b, v in sorted(by_bucket.items())},
        "by_band": {b: float(np.mean(v)) for b, v in sorted(by_band.items())},
        "latency_p50_ms": float(np.percentile(latencies_ms, 50)) if latencies_ms else 0.0,
        "latency_p95_ms": float(np.percentile(latencies_ms, 95)) if latencies_ms else 0.0,
    }


# ---------------------------------------------------------------------------
# Native Zero-Dependency LoRA (FP32 Adapters over FP16 Base — T4 Safe)
# ---------------------------------------------------------------------------

class LoRALinear(nn.Module):
    def __init__(self, base_layer: nn.Linear, r: int = 16, alpha: float = 32.0, dropout: float = 0.05):
        super().__init__()
        self.base = base_layer
        for p in self.base.parameters():
            p.requires_grad = False
        in_f = base_layer.in_features
        out_f = base_layer.out_features
        self.r = r
        self.scaling = alpha / float(r)
        self.drop = nn.Dropout(dropout)
        # Keep LoRA weights in FP32 for rock-solid stability on T4 (no GradScaler NaNs)
        self.lora_A = nn.Parameter(torch.empty((r, in_f), dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros((out_f, r), dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        x_f32 = self.drop(x.float())
        lora_out = F.linear(F.linear(x_f32, self.lora_A), self.lora_B) * self.scaling
        return base_out + lora_out.to(base_out.dtype)


def inject_lora(model: nn.Module, target_names: tuple[str, ...], r: int = 16, alpha: float = 32.0) -> int:
    for p in model.parameters():
        p.requires_grad = False
    replaced = 0
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if isinstance(child, nn.Linear) and child_name in target_names:
                setattr(module, child_name, LoRALinear(child, r=r, alpha=alpha))
                replaced += 1
    return replaced


def get_lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if "lora_" in k}


def load_lora_state_dict(model: nn.Module, state: dict[str, torch.Tensor]) -> None:
    model.load_state_dict(state, strict=False)


# ---------------------------------------------------------------------------
# Student 1: BGE-reranker-base (278M Cross-Encoder)
# ---------------------------------------------------------------------------

def format_bge_pairs(row: dict[str, Any], order: list[int]) -> tuple[list[str], list[str]]:
    q_text = f"{row['question']}\n{row['context']}"
    queries, docs = [], []
    for idx in order:
        opt = row["options"][idx]
        desc = row["option_descriptions"][idx]
        queries.append(q_text)
        docs.append(f"{opt}: {desc}")
    return queries, docs


@torch.no_grad()
def eval_bge(
    model: nn.Module,
    tok: Any,
    rows: list[dict[str, Any]],
    device: torch.device,
    max_len: int = 512,
) -> tuple[list[np.ndarray], list[np.ndarray], list[float], int]:
    model.eval()
    orig_logits, swap_logits, latencies = [], [], []
    max_seen = 0
    use_amp = device.type == "cuda"

    for r in rows:
        k = len(r["options"])
        t0 = time.perf_counter()
        q_o, d_o = format_bge_pairs(r, list(range(k)))
        enc_o = tok(q_o, d_o, padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(device)
        max_seen = max(max_seen, int(enc_o["input_ids"].shape[1]))
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            lg_o = model(**enc_o).logits.squeeze(-1).float().cpu().numpy()
        latencies.append((time.perf_counter() - t0) * 1000.0)
        orig_logits.append(lg_o)

        # Swapped order (cross-encoder is invariant, but we still measure explicitly)
        q_s, d_s = format_bge_pairs(r, list(reversed(range(k))))
        enc_s = tok(q_s, d_s, padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(device)
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            lg_s = model(**enc_s).logits.squeeze(-1).float().cpu().numpy()
        swap_logits.append(lg_s)

    return orig_logits, swap_logits, latencies, max_seen


def run_bge_study(
    model_id: str,
    train_rows: list[dict[str, Any]],
    dev_rows: list[dict[str, Any]],
    cal_rows: list[dict[str, Any]],
    seeds: list[int],
    epochs: int,
    lr: float,
    grad_accum: int,
    out_dir: Path,
    device: torch.device,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(model_id)
    use_amp = device.type == "cuda"

    # 1. Frozen zero-shot baseline
    print(f"\n[BGE] Evaluating frozen baseline: {model_id}")
    base_model = AutoModelForSequenceClassification.from_pretrained(model_id).to(device)
    total_params = sum(p.numel() for p in base_model.parameters())
    dev_o, dev_s, dev_lat, max_tok_dev = eval_bge(base_model, tok, dev_rows, device)
    cal_o, cal_s, cal_lat, _ = eval_bge(base_model, tok, cal_rows, device)
    t_frozen = fit_temperature(cal_o, [int(r["gold_index"]) for r in cal_rows])
    frozen_report = {
        "dev_uncalibrated": summarize_eval(dev_rows, dev_o, dev_s, 1.0, dev_lat),
        "dev_calibrated": summarize_eval(dev_rows, dev_o, dev_s, t_frozen, dev_lat),
        "cal_calibrated": summarize_eval(cal_rows, cal_o, cal_s, t_frozen, cal_lat),
        "max_tokens_seen": max_tok_dev,
    }
    print(
        f"[BGE Frozen] dev_acc={frozen_report['dev_calibrated']['acc_orig']:.4f} "
        f"ece={frozen_report['dev_calibrated']['ece']:.4f} "
        f"aurc={frozen_report['dev_calibrated']['aurc']:.4f} T={t_frozen}"
    )
    del base_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 2. Train across seeds
    seed_reports = []
    for seed in seeds:
        print(f"\n[BGE] Training seed={seed} (epochs={epochs}, lr={lr}, grad_accum={grad_accum})")
        set_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        model = AutoModelForSequenceClassification.from_pretrained(model_id).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        total_steps = math.ceil(len(train_rows) / grad_accum) * epochs
        warmup_steps = max(1, int(0.1 * total_steps))

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return float(step + 1) / float(warmup_steps)
            prog = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return 0.5 * (1.0 + math.cos(math.pi * prog))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        best_dev_nll = float("inf")
        best_state = None
        best_epoch = 0
        t_train0 = time.perf_counter()

        for ep in range(1, epochs + 1):
            model.train()
            order_idx = list(range(len(train_rows)))
            random.shuffle(order_idx)
            optimizer.zero_grad(set_to_none=True)
            ep_loss = 0.0

            for step_i, idx in enumerate(order_idx, 1):
                r = train_rows[idx]
                k = len(r["options"])
                q_t, d_t = format_bge_pairs(r, list(range(k)))
                enc = tok(q_t, d_t, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
                target = torch.tensor([int(r["gold_index"])], dtype=torch.long, device=device)

                with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                    logits = model(**enc).logits.squeeze(-1).unsqueeze(0)
                    loss = F.cross_entropy(logits.float(), target) / grad_accum

                scaler.scale(loss).backward()
                ep_loss += float(loss.item()) * grad_accum

                if step_i % grad_accum == 0 or step_i == len(order_idx):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()

            dev_o, dev_s, dev_lat, _ = eval_bge(model, tok, dev_rows, device)
            dev_m = summarize_eval(dev_rows, dev_o, dev_s, 1.0, dev_lat)
            print(
                f"  [BGE seed={seed} ep={ep}] train_loss={ep_loss/len(train_rows):.4f} "
                f"dev_acc={dev_m['acc_orig']:.4f} dev_nll={dev_m['nll']:.4f}"
            )
            if dev_m["nll"] < best_dev_nll:
                best_dev_nll = dev_m["nll"]
                best_epoch = ep
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        train_sec = time.perf_counter() - t_train0
        peak_vram_gb = torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0

        assert best_state is not None
        model.load_state_dict(best_state)
        ckpt_dir = out_dir / f"bge_seed{seed}"
        model.save_pretrained(ckpt_dir)
        tok.save_pretrained(ckpt_dir)

        # Calibrate strictly on calibrate_180
        cal_o, cal_s, cal_lat, _ = eval_bge(model, tok, cal_rows, device)
        t_cal = fit_temperature(cal_o, [int(r["gold_index"]) for r in cal_rows])
        dev_o, dev_s, dev_lat, _ = eval_bge(model, tok, dev_rows, device)

        s_rep = {
            "seed": seed,
            "best_epoch": best_epoch,
            "train_seconds": round(train_sec, 1),
            "peak_vram_gb": round(peak_vram_gb, 2),
            "temperature": t_cal,
            "dev_calibrated": summarize_eval(dev_rows, dev_o, dev_s, t_cal, dev_lat),
            "cal_calibrated": summarize_eval(cal_rows, cal_o, cal_s, t_cal, cal_lat),
            "checkpoint": str(ckpt_dir),
        }
        (ckpt_dir / "calibration.json").write_text(json.dumps({"temperature": t_cal, "seed": seed}, indent=2))
        seed_reports.append(s_rep)
        print(
            f"  [BGE seed={seed} DONE] best_ep={best_epoch} T={t_cal} "
            f"dev_acc={s_rep['dev_calibrated']['acc_orig']:.4f} "
            f"ece={s_rep['dev_calibrated']['ece']:.4f} "
            f"aurc={s_rep['dev_calibrated']['aurc']:.4f} "
            f"vram={peak_vram_gb:.2f}GB"
        )

        del model, best_state, optimizer, scheduler, scaler
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return {
        "model_id": model_id,
        "total_params": int(total_params),
        "trainable_params": int(total_params),
        "fine_tune_method": "full_ft_fp16_amp",
        "frozen_baseline": frozen_report,
        "seeds": seed_reports,
        "aggregate_dev": aggregate_seeds([s["dev_calibrated"] for s in seed_reports]),
        "aggregate_cal": aggregate_seeds([s["cal_calibrated"] for s in seed_reports]),
    }


# ---------------------------------------------------------------------------
# Student 2: Qwen3.5-0.8B-Base (800M Causal LM, Option-Letter Logits + LoRA)
# ---------------------------------------------------------------------------

def resolve_letter_token_ids(tok: Any, max_k: int = 5) -> list[int]:
    """Verify each option letter A..E maps cleanly to a single token ID."""
    ids = []
    for ch in LETTERS[:max_k]:
        # Prompt ends with 'Answer:' so next token is ' A' (with leading space) or 'A'
        toks_space = tok.encode(f" {ch}", add_special_tokens=False)
        toks_plain = tok.encode(ch, add_special_tokens=False)
        if len(toks_space) == 1:
            ids.append(toks_space[0])
        elif len(toks_plain) == 1:
            ids.append(toks_plain[0])
        else:
            raise ValueError(f"Letter {ch} did not encode to a single token: space={toks_space}, plain={toks_plain}")
    assert len(set(ids)) == max_k, f"Duplicate letter token IDs: {ids}"
    return ids


def format_qwen_prompt(row: dict[str, Any], order: list[int]) -> str:
    opt_lines = []
    for pos, orig_idx in enumerate(order):
        letter = LETTERS[pos]
        opt = row["options"][orig_idx]
        desc = row["option_descriptions"][orig_idx]
        opt_lines.append(f"{letter}. {opt} — {desc}")
    opts_block = "\n".join(opt_lines)
    return (
        f"Context:\n{row['context']}\n\n"
        f"Question: {row['question']}\n\n"
        f"Options:\n{opts_block}\n\n"
        f"Select the single best option letter (A-{LETTERS[len(order)-1]}).\nAnswer:"
    )


@torch.no_grad()
def eval_qwen(
    model: nn.Module,
    tok: Any,
    letter_ids: list[int],
    rows: list[dict[str, Any]],
    device: torch.device,
    max_len: int = 512,
) -> tuple[list[np.ndarray], list[np.ndarray], list[float], int]:
    model.eval()
    orig_logits, swap_logits, latencies = [], [], []
    max_seen = 0
    use_amp = device.type == "cuda"
    lid_tensor = torch.tensor(letter_ids, dtype=torch.long, device=device)

    for r in rows:
        k = len(r["options"])
        t0 = time.perf_counter()
        p_o = format_qwen_prompt(r, list(range(k)))
        enc_o = tok(p_o, truncation=True, max_length=max_len, return_tensors="pt").to(device)
        max_seen = max(max_seen, int(enc_o["input_ids"].shape[1]))
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            out_o = model(**enc_o).logits[0, -1, lid_tensor[:k]].float().cpu().numpy()
        latencies.append((time.perf_counter() - t0) * 1000.0)
        orig_logits.append(out_o)

        p_s = format_qwen_prompt(r, list(reversed(range(k))))
        enc_s = tok(p_s, truncation=True, max_length=max_len, return_tensors="pt").to(device)
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            out_s = model(**enc_s).logits[0, -1, lid_tensor[:k]].float().cpu().numpy()
        swap_logits.append(out_s)

    return orig_logits, swap_logits, latencies, max_seen


def run_qwen_study(
    model_id: str,
    train_rows: list[dict[str, Any]],
    dev_rows: list[dict[str, Any]],
    cal_rows: list[dict[str, Any]],
    seeds: list[int],
    epochs: int,
    lr: float,
    grad_accum: int,
    lora_r: int,
    out_dir: Path,
    device: torch.device,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    letter_ids = resolve_letter_token_ids(tok, max_k=5)
    print(f"[Qwen] Verified single-token letter IDs A..E: {letter_ids}")
    use_amp = device.type == "cuda"
    load_dtype = torch.float16 if use_amp else torch.float32

    # 1. Frozen zero-shot baseline
    print(f"\n[Qwen] Evaluating frozen baseline: {model_id}")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=load_dtype, trust_remote_code=True
    ).to(device)
    total_params = sum(p.numel() for p in base_model.parameters())
    dev_o, dev_s, dev_lat, max_tok_dev = eval_qwen(base_model, tok, letter_ids, dev_rows, device)
    cal_o, cal_s, cal_lat, _ = eval_qwen(base_model, tok, letter_ids, cal_rows, device)
    t_frozen = fit_temperature(cal_o, [int(r["gold_index"]) for r in cal_rows])
    frozen_report = {
        "dev_uncalibrated": summarize_eval(dev_rows, dev_o, dev_s, 1.0, dev_lat),
        "dev_calibrated": summarize_eval(dev_rows, dev_o, dev_s, t_frozen, dev_lat),
        "cal_calibrated": summarize_eval(cal_rows, cal_o, cal_s, t_frozen, cal_lat),
        "max_tokens_seen": max_tok_dev,
    }
    print(
        f"[Qwen Frozen] dev_acc={frozen_report['dev_calibrated']['acc_orig']:.4f} "
        f"flip={frozen_report['dev_calibrated']['order_flip_rate']:.4f} "
        f"ece={frozen_report['dev_calibrated']['ece']:.4f} T={t_frozen}"
    )
    del base_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 2. Train across seeds with native FP32 LoRA + gradient checkpointing
    target_modules = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
    seed_reports = []
    trainable_params = 0
    lid_tensor = torch.tensor(letter_ids, dtype=torch.long, device=device)

    for seed in seeds:
        print(f"\n[Qwen] Training seed={seed} (epochs={epochs}, lr={lr}, lora_r={lora_r}, grad_accum={grad_accum})")
        set_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=load_dtype, trust_remote_code=True
        ).to(device)
        n_rep = inject_lora(model, target_modules, r=lora_r, alpha=float(lora_r * 2))
        model.to(device)
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
        model.config.use_cache = False

        lora_params = [p for p in model.parameters() if p.requires_grad]
        trainable_params = sum(p.numel() for p in lora_params)
        print(f"  Injected LoRA into {n_rep} linear layers ({trainable_params:,} trainable FP32 params)")

        optimizer = torch.optim.AdamW(lora_params, lr=lr, weight_decay=0.01)
        total_steps = math.ceil(len(train_rows) / grad_accum) * epochs
        warmup_steps = max(1, int(0.1 * total_steps))

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return float(step + 1) / float(warmup_steps)
            prog = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return 0.5 * (1.0 + math.cos(math.pi * prog))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        best_dev_nll = float("inf")
        best_lora_state = None
        best_epoch = 0
        t_train0 = time.perf_counter()

        for ep in range(1, epochs + 1):
            model.train()
            order_idx = list(range(len(train_rows)))
            random.shuffle(order_idx)
            optimizer.zero_grad(set_to_none=True)
            ep_loss = 0.0

            for step_i, idx in enumerate(order_idx, 1):
                r = train_rows[idx]
                k = len(r["options"])
                # Permute option order during training so Qwen cannot memorize letter priors
                perm = list(range(k))
                random.shuffle(perm)
                new_gold = perm.index(int(r["gold_index"]))
                prompt = format_qwen_prompt(r, perm)
                enc = tok(prompt, truncation=True, max_length=512, return_tensors="pt").to(device)
                target = torch.tensor([new_gold], dtype=torch.long, device=device)

                with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                    logits = model(**enc).logits[:, -1, lid_tensor[:k]]
                loss = F.cross_entropy(logits.float(), target) / grad_accum
                loss.backward()
                ep_loss += float(loss.item()) * grad_accum

                if step_i % grad_accum == 0 or step_i == len(order_idx):
                    torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()

            dev_o, dev_s, dev_lat, _ = eval_qwen(model, tok, letter_ids, dev_rows, device)
            dev_m = summarize_eval(dev_rows, dev_o, dev_s, 1.0, dev_lat)
            print(
                f"  [Qwen seed={seed} ep={ep}] train_loss={ep_loss/len(train_rows):.4f} "
                f"dev_acc={dev_m['acc_orig']:.4f} dev_nll={dev_m['nll']:.4f} flip={dev_m['order_flip_rate']:.4f}"
            )
            if dev_m["nll"] < best_dev_nll:
                best_dev_nll = dev_m["nll"]
                best_epoch = ep
                best_lora_state = get_lora_state_dict(model)

        train_sec = time.perf_counter() - t_train0
        peak_vram_gb = torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0

        assert best_lora_state is not None
        load_lora_state_dict(model, best_lora_state)
        ckpt_dir = out_dir / f"qwen_seed{seed}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save(best_lora_state, ckpt_dir / "lora_adapters.pt")
        (ckpt_dir / "adapter_config.json").write_text(
            json.dumps(
                {
                    "base_model": model_id,
                    "lora_r": lora_r,
                    "lora_alpha": lora_r * 2,
                    "target_modules": list(target_modules),
                    "letter_ids": letter_ids,
                },
                indent=2,
            )
        )

        # Calibrate strictly on calibrate_180
        cal_o, cal_s, cal_lat, _ = eval_qwen(model, tok, letter_ids, cal_rows, device)
        t_cal = fit_temperature(cal_o, [int(r["gold_index"]) for r in cal_rows])
        dev_o, dev_s, dev_lat, _ = eval_qwen(model, tok, letter_ids, dev_rows, device)

        s_rep = {
            "seed": seed,
            "best_epoch": best_epoch,
            "train_seconds": round(train_sec, 1),
            "peak_vram_gb": round(peak_vram_gb, 2),
            "temperature": t_cal,
            "dev_calibrated": summarize_eval(dev_rows, dev_o, dev_s, t_cal, dev_lat),
            "cal_calibrated": summarize_eval(cal_rows, cal_o, cal_s, t_cal, cal_lat),
            "checkpoint": str(ckpt_dir),
        }
        (ckpt_dir / "calibration.json").write_text(json.dumps({"temperature": t_cal, "seed": seed}, indent=2))
        seed_reports.append(s_rep)
        print(
            f"  [Qwen seed={seed} DONE] best_ep={best_epoch} T={t_cal} "
            f"dev_acc={s_rep['dev_calibrated']['acc_orig']:.4f} "
            f"flip={s_rep['dev_calibrated']['order_flip_rate']:.4f} "
            f"ece={s_rep['dev_calibrated']['ece']:.4f} "
            f"vram={peak_vram_gb:.2f}GB"
        )

        del model, best_lora_state, optimizer, scheduler
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return {
        "model_id": model_id,
        "total_params": int(total_params),
        "trainable_params": int(trainable_params),
        "fine_tune_method": f"native_fp32_lora_r{lora_r}_fp16_base",
        "letter_ids": letter_ids,
        "frozen_baseline": frozen_report,
        "seeds": seed_reports,
        "aggregate_dev": aggregate_seeds([s["dev_calibrated"] for s in seed_reports]),
        "aggregate_cal": aggregate_seeds([s["cal_calibrated"] for s in seed_reports]),
    }


def aggregate_seeds(metrics_list: list[dict[str, Any]]) -> dict[str, Any]:
    keys = ["acc_orig", "acc_swapped", "order_flip_rate", "nll", "ece", "aurc", "error_auroc", "latency_p50_ms", "latency_p95_ms"]
    agg = {}
    for k in keys:
        vals = [float(m[k]) for m in metrics_list]
        agg[k] = {"mean": round(float(np.mean(vals)), 4), "sd": round(float(np.std(vals, ddof=0)), 4)}
    return agg


# ---------------------------------------------------------------------------
# Main Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--student", choices=["bge", "qwen", "both"], default="both")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 1024, 2026])
    parser.add_argument("--bge-model", type=str, default="BAAI/bge-reranker-base")
    parser.add_argument("--qwen-model", type=str, default="Qwen/Qwen3.5-0.8B-Base")
    parser.add_argument("--bge-epochs", type=int, default=4)
    parser.add_argument("--qwen-epochs", type=int, default=4)
    parser.add_argument("--bge-lr", type=float, default=2e-5)
    parser.add_argument("--qwen-lr", type=float, default=2e-4)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--smoke-test", action="store_true", help="Run fast 12-item smoke test (1 epoch, 1 seed)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    limit = 12 if args.smoke_test else None
    seeds = [42] if args.smoke_test else args.seeds
    bge_epochs = 1 if args.smoke_test else args.bge_epochs
    qwen_epochs = 1 if args.smoke_test else args.qwen_epochs

    train_rows = load_jsonl(args.data_dir / "train_840.jsonl", limit=limit)
    dev_rows = load_jsonl(args.data_dir / "dev_180.jsonl", limit=limit)
    cal_rows = load_jsonl(args.data_dir / "calibrate_180.jsonl", limit=limit)
    print(f"Loaded splits: train={len(train_rows)}, dev={len(dev_rows)}, cal={len(cal_rows)}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "smoke_test": args.smoke_test,
        "seeds": seeds,
        "splits": {"train": len(train_rows), "dev": len(dev_rows), "cal": len(cal_rows)},
    }

    if args.student in ("bge", "both"):
        summary["bge"] = run_bge_study(
            model_id=args.bge_model,
            train_rows=train_rows,
            dev_rows=dev_rows,
            cal_rows=cal_rows,
            seeds=seeds,
            epochs=bge_epochs,
            lr=args.bge_lr,
            grad_accum=args.grad_accum,
            out_dir=args.out_dir / "bge",
            device=device,
        )
        (args.out_dir / "two_student_summary.json").write_text(json.dumps(summary, indent=2))

    if args.student in ("qwen", "both"):
        summary["qwen"] = run_qwen_study(
            model_id=args.qwen_model,
            train_rows=train_rows,
            dev_rows=dev_rows,
            cal_rows=cal_rows,
            seeds=seeds,
            epochs=qwen_epochs,
            lr=args.qwen_lr,
            grad_accum=args.grad_accum,
            lora_r=args.lora_r,
            out_dir=args.out_dir / "qwen",
            device=device,
        )
        (args.out_dir / "two_student_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\nSaved consolidated summary -> {args.out_dir / 'two_student_summary.json'}")


if __name__ == "__main__":
    main()
