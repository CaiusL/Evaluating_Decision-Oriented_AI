#!/usr/bin/env python3
"""Milestone 5: Matched Racing Student Training & Calibration (NN-3 MLP + Fresh Racing Qwen3.5-0.8B LoRA).

Colab T4 & Local CPU safe:
  - Set-valued acceptable-action loss: L = -log sum_{a in S_acceptable} p_theta(a | o)
  - Balanced build vs. pace loss weighting so pace records (~4x count) do not overwhelm build learning
  - Multi-seed training + checkpoint/seed selection strictly on dev.jsonl
  - Racing-specific scalar temperature T fit strictly on cal.jsonl AFTER checkpoint selection
  - NEVER touches historical_regression (2001..2005) or confirm (3501..3520) splits
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "decision_eval" / "src"
for p in (str(REPO_ROOT), str(SRC_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from decision_eval import paths  # noqa: E402
from decision_eval.racing_contract import (  # noqa: E402
    BUILD_OPTIONS,
    PACE_OPTIONS,
    SCHEMA_VERSION,
    action_spec_hash,
)

DEFAULT_DATA_DIR = REPO_ROOT / "data" / "racing_v1"
DEFAULT_MODELS_DIR = REPO_ROOT / "models" / "racing_engineer_v1"
DEFAULT_REPORTS_DIR = REPO_ROOT / "reports" / "racing_engineer_v1"
LETTERS = ["A", "B", "C", "D", "E"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def softmax_np(logits: np.ndarray, temp: float = 1.0) -> np.ndarray:
    z = logits / max(temp, 1e-4)
    z = z - np.max(z)
    e = np.exp(z)
    return e / np.sum(e)


def set_valued_loss_torch(logits: torch.Tensor, acc_mask: torch.Tensor) -> torch.Tensor:
    """L = -log sum_{a in S_acceptable} softmax(logits)_a."""
    log_probs = F.log_softmax(logits, dim=-1)
    masked = torch.where(acc_mask > 0.5, log_probs, torch.full_like(log_probs, -1e9))
    log_sum = torch.logsumexp(masked, dim=-1)
    return -log_sum.mean()


def fit_temperature_set_valued(logits_list: list[np.ndarray], acc_masks: list[np.ndarray]) -> float:
    def nll_at(t: float) -> float:
        total = 0.0
        for lg, m in zip(logits_list, acc_masks):
            p = softmax_np(lg, t)
            p_acc = float(np.sum(p * m))
            total -= math.log(max(p_acc, 1e-12))
        return total / max(len(logits_list), 1)

    grid = np.concatenate([np.linspace(0.25, 2.0, 36), np.linspace(2.2, 6.0, 20)])
    best_t = float(min(grid, key=nll_at))
    fine = np.linspace(max(0.1, best_t - 0.15), best_t + 0.15, 31)
    return round(float(min(fine, key=nll_at)), 4)


def eval_logits_metrics(
    logits_list: list[np.ndarray],
    gold_indices: list[int],
    acc_masks: list[np.ndarray],
    temp: float,
) -> dict[str, Any]:
    exact_ok, acc_ok, nlls, confs = [], [], [], []
    for lg, y, m in zip(logits_list, gold_indices, acc_masks):
        p = softmax_np(lg, temp)
        pred = int(np.argmax(p))
        exact_ok.append(int(pred == y))
        acc_ok.append(int(m[pred] > 0.5))
        confs.append(float(p[pred]))
        nlls.append(-math.log(max(float(np.sum(p * m)), 1e-12)))
    confs_arr = np.array(confs, dtype=np.float64)
    acc_arr = np.array(acc_ok, dtype=np.float64)
    bins = np.linspace(0.0, 1.0, 11)
    ece = 0.0
    if len(confs_arr) > 0:
        for i in range(10):
            lo, hi = bins[i], bins[i + 1]
            mask = (confs_arr >= lo) & (confs_arr <= hi if i == 9 else confs_arr < hi)
            if np.any(mask):
                ece += (np.sum(mask) / len(confs_arr)) * abs(float(np.mean(acc_arr[mask])) - float(np.mean(confs_arr[mask])))
    return {
        "n": len(logits_list),
        "temperature": float(temp),
        "exact_acc": round(float(np.mean(exact_ok)), 4) if exact_ok else 0.0,
        "acceptable_acc": round(float(np.mean(acc_ok)), 4) if acc_ok else 0.0,
        "set_nll": round(float(np.mean(nlls)), 4) if nlls else 0.0,
        "ece": round(float(ece), 4),
    }


# ---------------------------------------------------------------------------
# Student A: Compact NN-3 (Two-layer 64-unit MLP with Build & Pace Heads)
# ---------------------------------------------------------------------------

class RacingNN3Head(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 64, dropout: float = 0.10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_single_nn3_head(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    m_tr: np.ndarray,
    X_dev: np.ndarray,
    y_dev: np.ndarray,
    m_dev: np.ndarray,
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    m_cal: np.ndarray,
    out_dim: int,
    seed: int,
    epochs: int = 120,
    lr: float = 3e-3,
) -> dict[str, Any]:
    set_seed(seed)
    mean = X_tr.mean(axis=0, keepdims=True).astype(np.float32)
    std = (X_tr.std(axis=0, keepdims=True) + 1e-6).astype(np.float32)

    x_tr_t = torch.tensor((X_tr - mean) / std, dtype=torch.float32)
    m_tr_t = torch.tensor(m_tr, dtype=torch.float32)
    x_dev_t = torch.tensor((X_dev - mean) / std, dtype=torch.float32)
    m_dev_t = torch.tensor(m_dev, dtype=torch.float32)
    x_cal_t = torch.tensor((X_cal - mean) / std, dtype=torch.float32)

    model = RacingNN3Head(X_tr.shape[1], out_dim, hidden=64, dropout=0.10)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)

    best_nll = float("inf")
    best_state = None
    best_ep = 0

    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(x_tr_t))
        for b0 in range(0, len(x_tr_t), 32):
            idx = perm[b0 : b0 + 32]
            opt.zero_grad(set_to_none=True)
            loss = set_valued_loss_torch(model(x_tr_t[idx]), m_tr_t[idx])
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            dev_loss = float(set_valued_loss_torch(model(x_dev_t), m_dev_t).item())
        if dev_loss < best_nll:
            best_nll = dev_loss
            best_ep = ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        dev_logits = model(x_dev_t).numpy()
        cal_logits = model(x_cal_t).numpy()

    t_cal = fit_temperature_set_valued(list(cal_logits), list(m_cal))
    dev_m = eval_logits_metrics(list(dev_logits), y_dev.tolist(), list(m_dev), t_cal)
    cal_m = eval_logits_metrics(list(cal_logits), y_cal.tolist(), list(m_cal), t_cal)

    return {
        "seed": seed,
        "best_epoch": best_ep,
        "temperature": t_cal,
        "dev_metrics": dev_m,
        "cal_metrics": cal_m,
        "mean": mean.squeeze(0),
        "std": std.squeeze(0),
        "state_dict": best_state,
    }


def run_nn3_training(data_dir: Path, out_dir: Path, seeds: list[int]) -> dict[str, Any]:
    t0 = time.perf_counter()
    npz = np.load(data_dir / "features_nn3.npz")
    seed_reports = []
    best_seed_obj = None
    best_combined_nll = float("inf")

    for seed in seeds:
        b_res = train_single_nn3_head(
            npz["train_build_X"], npz["train_build_y"], npz["train_build_acc_mask"],
            npz["dev_build_X"], npz["dev_build_y"], npz["dev_build_acc_mask"],
            npz["cal_build_X"], npz["cal_build_y"], npz["cal_build_acc_mask"],
            out_dim=len(BUILD_OPTIONS), seed=seed,
        )
        p_res = train_single_nn3_head(
            npz["train_pace_X"], npz["train_pace_y"], npz["train_pace_acc_mask"],
            npz["dev_pace_X"], npz["dev_pace_y"], npz["dev_pace_acc_mask"],
            npz["cal_pace_X"], npz["cal_pace_y"], npz["cal_pace_acc_mask"],
            out_dim=len(PACE_OPTIONS), seed=seed,
        )
        comb_nll = 0.5 * (b_res["dev_metrics"]["set_nll"] + p_res["dev_metrics"]["set_nll"])
        ckpt_path = out_dir / f"nn3_seed{seed}.pt"
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "schema_version": SCHEMA_VERSION,
                "action_spec_hash": action_spec_hash(),
                "seed": seed,
                "build_mean": b_res["mean"],
                "build_std": b_res["std"],
                "build_temperature": b_res["temperature"],
                "build_state_dict": b_res["state_dict"],
                "pace_mean": p_res["mean"],
                "pace_std": p_res["std"],
                "pace_temperature": p_res["temperature"],
                "pace_state_dict": p_res["state_dict"],
            },
            ckpt_path,
        )
        s_rep = {
            "seed": seed,
            "checkpoint": str(ckpt_path),
            "combined_dev_set_nll": round(comb_nll, 4),
            "build_dev": b_res["dev_metrics"],
            "build_cal": b_res["cal_metrics"],
            "pace_dev": p_res["dev_metrics"],
            "pace_cal": p_res["cal_metrics"],
        }
        seed_reports.append(s_rep)
        print(
            f"  [NN-3 seed={seed}] build_acc={b_res['dev_metrics']['acceptable_acc']:.3f} (T={b_res['temperature']}) | "
            f"pace_acc={p_res['dev_metrics']['acceptable_acc']:.3f} (T={p_res['temperature']}) | comb_nll={comb_nll:.4f}",
            flush=True,
        )
        if comb_nll < best_combined_nll:
            best_combined_nll = comb_nll
            best_seed_obj = s_rep

    assert best_seed_obj is not None
    deploy_ckpt = out_dir / "nn3_selected.pt"
    deploy_ckpt.write_bytes(Path(best_seed_obj["checkpoint"]).read_bytes())
    return {
        "model_family": "NN-3_two_layer_64u_MLP",
        "wall_s": round(time.perf_counter() - t0, 2),
        "seeds": seed_reports,
        "selected_seed": best_seed_obj["seed"],
        "selected_checkpoint": str(deploy_ckpt),
    }


# ---------------------------------------------------------------------------
# Student B: Fresh Racing Qwen3.5-0.8B LoRA (from Qwen/Qwen3.5-0.8B-Base)
# ---------------------------------------------------------------------------

class LoRALinear(nn.Module):
    def __init__(self, base_layer: nn.Linear, r: int = 16, alpha: float = 32.0, dropout: float = 0.05):
        super().__init__()
        self.base = base_layer
        for p in self.base.parameters():
            p.requires_grad = False
        self.r = r
        self.scaling = alpha / float(r)
        self.drop = nn.Dropout(dropout)
        self.lora_A = nn.Parameter(torch.empty((r, base_layer.in_features), dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros((base_layer.out_features, r), dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_out = F.linear(F.linear(self.drop(x.float()), self.lora_A), self.lora_B) * self.scaling
        return base_out + lora_out.to(base_out.dtype)


def inject_lora(model: nn.Module, target_names: tuple[str, ...], r: int, alpha: float) -> int:
    for p in model.parameters():
        p.requires_grad = False
    replaced = 0
    for _name, mod in list(model.named_modules()):
        for child_name, child in list(mod.named_children()):
            if isinstance(child, nn.Linear) and child_name in target_names:
                setattr(mod, child_name, LoRALinear(child, r=r, alpha=alpha))
                replaced += 1
    return replaced


def format_qwen_racing_prompt(row: dict[str, Any], order: list[int]) -> str:
    opt_lines = [
        f"{LETTERS[pos]}. {row['options'][orig_i]} — {row['option_descriptions'][orig_i]}"
        for pos, orig_i in enumerate(order)
    ]
    return (
        f"Context:\n{row['context']}\n\n"
        f"Question: {row['question']}\n\n"
        f"Options:\n" + "\n".join(opt_lines) + "\n\n"
        f"Select the single best option letter (A-{LETTERS[len(order)-1]}).\nAnswer:"
    )


def resolve_letter_ids(tok: Any, max_k: int = 5) -> list[int]:
    ids = []
    for ch in LETTERS[:max_k]:
        t_sp = tok.encode(f" {ch}", add_special_tokens=False)
        t_pl = tok.encode(ch, add_special_tokens=False)
        ids.append(int(t_sp[0] if len(t_sp) == 1 else t_pl[0]))
    assert len(set(ids)) == max_k, f"Duplicate letter token IDs: {ids}"
    return ids


@torch.no_grad()
def eval_qwen_rows(
    model: nn.Module,
    tok: Any,
    lid_tensor: torch.Tensor,
    rows: list[dict[str, Any]],
    device: torch.device,
    max_len: int = 512,
) -> tuple[list[np.ndarray], list[int], list[np.ndarray], list[str]]:
    model.eval()
    use_amp = device.type == "cuda"
    logits_list, golds, masks, dtypes = [], [], [], []
    for r in rows:
        k = len(r["options"])
        prompt = format_qwen_racing_prompt(r, list(range(k)))
        enc = tok(prompt, truncation=True, max_length=max_len, return_tensors="pt").to(device)
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            lg = model(**enc).logits[0, -1, lid_tensor[:k]].float().cpu().numpy()
        logits_list.append(lg)
        golds.append(int(r["gold_index"]))
        dtypes.append(str(r["decision_type"]))
        m = np.zeros(k, dtype=np.float32)
        for opt in r["acceptable_option_ids"]:
            m[r["options"].index(opt)] = 1.0
        masks.append(m)
    return logits_list, golds, masks, dtypes


def split_metrics_by_type(
    logits_list: list[np.ndarray],
    golds: list[int],
    masks: list[np.ndarray],
    dtypes: list[str],
    temp: float,
) -> dict[str, Any]:
    overall = eval_logits_metrics(logits_list, golds, masks, temp)
    by_type = {}
    for dt in ("build", "pace"):
        idxs = [i for i, t in enumerate(dtypes) if t == dt]
        if idxs:
            by_type[dt] = eval_logits_metrics(
                [logits_list[i] for i in idxs],
                [golds[i] for i in idxs],
                [masks[i] for i in idxs],
                temp,
            )
    overall["by_type"] = by_type
    return overall


def run_qwen_racing_training(
    data_dir: Path,
    out_dir: Path,
    seeds: list[int],
    epochs: int = 4,
    lora_r: int = 16,
    lr: float = 2e-4,
    grad_accum: int = 8,
    max_train_per_type: int | None = None,
    max_eval_per_type: int | None = None,
    smoke_test: bool = False,
) -> dict[str, Any]:
    """Train fresh Racing Qwen3.5-0.8B LoRA from Qwen/Qwen3.5-0.8B-Base on T4 GPU or CPU."""
    t0 = time.perf_counter()
    paths.configure_bundled_hf_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    base_model_id = "Qwen/Qwen3.5-0.8B-Base"
    resolved_base = paths.resolve_local_model_path(base_model_id)
    cache_kw = {"cache_dir": str(paths.HF_CACHE_DIR)} if paths.HF_CACHE_DIR.is_dir() else {}

    try:
        tok = AutoTokenizer.from_pretrained(resolved_base, trust_remote_code=True, local_files_only=True, **cache_kw)
        local_only = True
    except Exception:
        tok = AutoTokenizer.from_pretrained(resolved_base, trust_remote_code=True, **cache_kw)
        local_only = False

    letter_ids = resolve_letter_ids(tok, max_k=5)
    lid_tensor = torch.tensor(letter_ids, dtype=torch.long, device=device)
    print(f"[Racing Qwen LoRA] Device={device} (AMP={use_amp}) | letter_ids={letter_ids} | local_only={local_only}", flush=True)

    train_all = [r for r in load_jsonl(data_dir / "train.jsonl") if r["label_info"]["status"] != "all_failed"]
    dev_all = [r for r in load_jsonl(data_dir / "dev.jsonl") if r["label_info"]["status"] != "all_failed"]
    cal_all = [r for r in load_jsonl(data_dir / "cal.jsonl") if r["label_info"]["status"] != "all_failed"]

    def _cap_by_type(rows: list[dict[str, Any]], cap: int | None, seed_i: int) -> list[dict[str, Any]]:
        if cap is None:
            return list(rows)
        rng = random.Random(seed_i)
        b_r = [r for r in rows if r["decision_type"] == "build"]
        p_r = [r for r in rows if r["decision_type"] == "pace"]
        rng.shuffle(b_r)
        rng.shuffle(p_r)
        sub = b_r[:cap] + p_r[:cap]
        rng.shuffle(sub)
        return sub

    if smoke_test:
        seeds = [seeds[0]]
        epochs = 1
        max_train_per_type = 8
        max_eval_per_type = 8

    target_modules = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
    load_dtype = torch.float16 if use_amp else torch.float32
    seed_reports: list[dict[str, Any]] = []
    best_deploy_rep: dict[str, Any] | None = None
    best_deploy_nll = float("inf")

    for seed in seeds:
        ts0 = time.perf_counter()
        set_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        train_rows = _cap_by_type(train_all, max_train_per_type, seed)
        dev_rows = _cap_by_type(dev_all, max_eval_per_type, seed)
        cal_rows = _cap_by_type(cal_all, max_eval_per_type, seed)

        n_build = max(1, sum(1 for r in train_rows if r["decision_type"] == "build"))
        n_pace = max(1, sum(1 for r in train_rows if r["decision_type"] == "pace"))
        build_weight = float(n_pace) / float(n_build)
        print(
            f"\n[Racing Qwen seed={seed}] train={len(train_rows)} (build={n_build}, pace={n_pace}, build_w={build_weight:.2f}), "
            f"dev={len(dev_rows)}, cal={len(cal_rows)}, epochs={epochs}, lora_r={lora_r}",
            flush=True,
        )

        model = AutoModelForCausalLM.from_pretrained(
            resolved_base, dtype=load_dtype, trust_remote_code=True, local_files_only=local_only, **cache_kw
        ).to(device)
        n_rep = inject_lora(model, target_modules, r=lora_r, alpha=float(lora_r * 2))
        model.to(device)
        if use_amp and hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
        model.config.use_cache = False

        lora_params = [p for p in model.parameters() if p.requires_grad]
        trainable_n = sum(p.numel() for p in lora_params)
        print(f"  Injected LoRA into {n_rep} layers ({trainable_n:,} trainable FP32 params)", flush=True)

        opt = torch.optim.AdamW(lora_params, lr=lr, weight_decay=0.01)
        total_steps = math.ceil(len(train_rows) / grad_accum) * epochs
        warmup_steps = max(1, int(0.1 * total_steps))

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return float(step + 1) / float(warmup_steps)
            prog = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return 0.5 * (1.0 + math.cos(math.pi * prog))

        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

        best_dev_nll = float("inf")
        best_state = None
        best_ep = 0

        for ep in range(1, epochs + 1):
            model.train()
            random.shuffle(train_rows)
            opt.zero_grad(set_to_none=True)
            ep_loss = 0.0
            for step_i, r in enumerate(train_rows, 1):
                k = len(r["options"])
                perm = list(range(k))
                random.shuffle(perm)
                prompt = format_qwen_racing_prompt(r, perm)
                enc = tok(prompt, truncation=True, max_length=512, return_tensors="pt").to(device)
                acc_mask = torch.zeros((1, k), dtype=torch.float32, device=device)
                for pos, orig_i in enumerate(perm):
                    if r["options"][orig_i] in r["acceptable_option_ids"]:
                        acc_mask[0, pos] = 1.0
                w = build_weight if r["decision_type"] == "build" else 1.0
                with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                    logits = model(**enc).logits[:, -1, lid_tensor[:k]]
                loss = (set_valued_loss_torch(logits.float(), acc_mask) * w) / grad_accum
                loss.backward()
                ep_loss += float(loss.item()) * grad_accum
                if step_i % grad_accum == 0 or step_i == len(train_rows):
                    torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
                    opt.step()
                    opt.zero_grad(set_to_none=True)
                    sched.step()

            dev_lg, dev_g, dev_m, dev_dt = eval_qwen_rows(model, tok, lid_tensor, dev_rows, device)
            dm = split_metrics_by_type(dev_lg, dev_g, dev_m, dev_dt, 1.0)
            b_nll = dm["by_type"].get("build", {}).get("set_nll", dm["set_nll"])
            p_nll = dm["by_type"].get("pace", {}).get("set_nll", dm["set_nll"])
            comb_nll = 0.5 * (b_nll + p_nll)
            print(
                f"  [Racing Qwen seed={seed} ep={ep}] train_loss={ep_loss/len(train_rows):.4f} "
                f"dev_acc={dm['acceptable_acc']:.4f} (build={dm['by_type'].get('build',{}).get('acceptable_acc',0):.3f}, "
                f"pace={dm['by_type'].get('pace',{}).get('acceptable_acc',0):.3f}) comb_nll={comb_nll:.4f}",
                flush=True,
            )
            if comb_nll < best_dev_nll:
                best_dev_nll = comb_nll
                best_ep = ep
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if "lora_" in k}

        assert best_state is not None
        model.load_state_dict(best_state, strict=False)
        cal_lg, cal_g, cal_m, cal_dt = eval_qwen_rows(model, tok, lid_tensor, cal_rows, device)
        t_cal = fit_temperature_set_valued(cal_lg, cal_m)
        dev_lg, dev_g, dev_m, dev_dt = eval_qwen_rows(model, tok, lid_tensor, dev_rows, device)
        dev_cal = split_metrics_by_type(dev_lg, dev_g, dev_m, dev_dt, t_cal)
        cal_cal = split_metrics_by_type(cal_lg, cal_g, cal_m, cal_dt, t_cal)
        comb_cal_nll = 0.5 * (
            dev_cal["by_type"].get("build", {}).get("set_nll", dev_cal["set_nll"])
            + dev_cal["by_type"].get("pace", {}).get("set_nll", dev_cal["set_nll"])
        )
        peak_vram_gb = round(torch.cuda.max_memory_allocated() / (1024**3), 2) if torch.cuda.is_available() else 0.0

        ckpt_dir = out_dir / f"racing_qwen3.5_0.8b_lora_s{seed}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save(best_state, ckpt_dir / "lora_adapters.pt")
        (ckpt_dir / "adapter_config.json").write_text(
            json.dumps(
                {
                    "base_model": base_model_id,
                    "lora_r": lora_r,
                    "lora_alpha": lora_r * 2,
                    "target_modules": list(target_modules),
                    "letter_ids": letter_ids,
                    "schema_version": SCHEMA_VERSION,
                    "action_spec_hash": action_spec_hash(),
                },
                indent=2,
            ) + "\n",
            encoding="utf-8",
        )
        (ckpt_dir / "calibration.json").write_text(
            json.dumps(
                {
                    "temperature": t_cal,
                    "seed": seed,
                    "best_epoch": best_ep,
                    "model_id": f"{base_model_id}+racing_v1_lora_s{seed}",
                    "calibration_split": "data/racing_v1/cal.jsonl",
                },
                indent=2,
            ) + "\n",
            encoding="utf-8",
        )

        s_rep = {
            "seed": seed,
            "best_epoch": best_ep,
            "train_seconds": round(time.perf_counter() - ts0, 1),
            "peak_vram_gb": peak_vram_gb,
            "temperature": t_cal,
            "combined_dev_set_nll": round(comb_cal_nll, 4),
            "dev_calibrated": dev_cal,
            "cal_calibrated": cal_cal,
            "checkpoint_dir": str(ckpt_dir),
        }
        seed_reports.append(s_rep)
        if comb_cal_nll < best_deploy_nll:
            best_deploy_nll = comb_cal_nll
            best_deploy_rep = s_rep

        del model, best_state, opt, sched
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    assert best_deploy_rep is not None
    # Copy selected seed artifacts to racing_qwen3.5_0.8b_lora_selected/
    sel_dir = out_dir / "racing_qwen3.5_0.8b_lora_selected"
    sel_dir.mkdir(parents=True, exist_ok=True)
    src_dir = Path(best_deploy_rep["checkpoint_dir"])
    for fname in ("lora_adapters.pt", "adapter_config.json", "calibration.json"):
        (sel_dir / fname).write_bytes((src_dir / fname).read_bytes())

    return {
        "model_family": "Racing_Qwen3.5_0.8B_LoRA_fresh_from_Base",
        "base_model": base_model_id,
        "lora_r": lora_r,
        "letter_ids": letter_ids,
        "seeds": seed_reports,
        "selected_seed": best_deploy_rep["seed"],
        "selected_checkpoint_dir": str(sel_dir),
        "wall_s": round(time.perf_counter() - t0, 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--out-models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    ap.add_argument("--out-reports-dir", type=Path, default=DEFAULT_REPORTS_DIR)
    ap.add_argument("--student", choices=["nn3", "qwen", "both"], default="both")
    ap.add_argument("--nn3-seeds", nargs="+", type=int, default=[42, 1024, 2026])
    ap.add_argument("--qwen-seeds", nargs="+", type=int, default=[42, 1024, 2026])
    ap.add_argument("--qwen-epochs", type=int, default=4)
    ap.add_argument("--qwen-lora-r", type=int, default=16)
    ap.add_argument("--qwen-lr", type=float, default=2e-4)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--qwen-max-train-per-type", type=int, default=None)
    ap.add_argument("--qwen-max-eval-per-type", type=int, default=None)
    ap.add_argument("--smoke-test", action="store_true")
    args = ap.parse_args()

    args.out_models_dir.mkdir(parents=True, exist_ok=True)
    args.out_reports_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.out_reports_dir / "training_summary.json"
    summary: dict[str, Any] = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}
    summary["schema_version"] = SCHEMA_VERSION
    summary["action_spec_hash"] = action_spec_hash()

    if args.student in ("nn3", "both"):
        print("[Training] Student A: Compact NN-3 MLP...", flush=True)
        summary["nn3"] = run_nn3_training(args.data_dir, args.out_models_dir / "nn3", args.nn3_seeds)
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    if args.student in ("qwen", "both"):
        print("[Training] Student B: Fresh Racing Qwen3.5-0.8B LoRA from Base...", flush=True)
        summary["qwen_lora"] = run_qwen_racing_training(
            args.data_dir,
            args.out_models_dir / "qwen_lora",
            seeds=args.qwen_seeds,
            epochs=args.qwen_epochs,
            lora_r=args.qwen_lora_r,
            lr=args.qwen_lr,
            grad_accum=args.grad_accum,
            max_train_per_type=args.qwen_max_train_per_type,
            max_eval_per_type=args.qwen_max_eval_per_type,
            smoke_test=args.smoke_test,
        )
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"\nSaved training & calibration summary -> {summary_path}")


if __name__ == "__main__":
    main()
