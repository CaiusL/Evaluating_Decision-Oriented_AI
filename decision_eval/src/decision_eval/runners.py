"""Model inference runners for Kev-0.8B, Qwen-0.8B, Frontier LLM API, and Tier 4 NN."""
from __future__ import annotations

import json
import math
import os
import re
import time
import urllib.request
from pathlib import Path
from typing import Any


def pick_device(requested: str) -> str:
    import torch

    if requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _sync_device(device: str) -> None:
    import torch

    if device == "mps" and torch.backends.mps.is_available():
        torch.mps.synchronize()
    elif device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()


def _clear_device_cache(device: str) -> None:
    import torch

    if device == "mps" and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()
    elif device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()


def _softmax(logits: list[float]) -> list[float]:
    m = max(logits)
    exps = [math.exp(x - m) for x in logits]
    s = sum(exps) or 1.0
    return [x / s for x in exps]


def _maybe_patch_local_qwen_base() -> str | None:
    """Reuse local ModelScope Qwen3.5-0.8B-Base directory if present."""
    from decision_eval.paths import resolve_local_model_path

    base = resolve_local_model_path("Qwen/Qwen3.5-0.8B-Base")
    if base == "Qwen/Qwen3.5-0.8B-Base" or not Path(base).is_dir():
        return None

    import kev.checkpoint as checkpoint
    import kev.model as model_mod
    from transformers import AutoTokenizer

    original = checkpoint.resolve_run

    def resolve_run(run):
        if str(run).startswith("Qwen/Qwen3.5-0.8B-Base"):
            return base
        return original(run)

    def load_tokenizer(name, revision=None):
        if str(name).startswith("Qwen/Qwen3.5-0.8B-Base"):
            return AutoTokenizer.from_pretrained(base)
        return AutoTokenizer.from_pretrained(name, revision=revision)

    checkpoint.resolve_run = resolve_run
    model_mod.load_tokenizer = load_tokenizer
    checkpoint.load_tokenizer = load_tokenizer
    return base


class KevRunner:
    """Tier 3: Decision-oriented option scorer (jaredpalmer/kev-0.8b on MLX or PyTorch)."""

    def __init__(self, run_id: str, device: str, backend: str = "auto") -> None:
        import torch
        from kev.checkpoint import Checkpoint, LoadOptions
        from decision_eval.paths import resolve_local_model_path

        self.run_id = run_id
        resolved_run = resolve_local_model_path(run_id)
        self.device = device
        self.base_source = _maybe_patch_local_qwen_base() or "huggingface"

        t0 = time.perf_counter()
        kev_backend = "torch" if backend == "torch" else ("mlx" if backend == "mlx" else "auto")
        options = LoadOptions(backend=kev_backend, dtype=torch.bfloat16, merge=True)
        ckpt = Checkpoint(resolved_run)
        self.tokenizer, self.model = ckpt.load(device, options)
        self.load_s = round(time.perf_counter() - t0, 2)
        self.backend = str(getattr(self.model, "backend", device))

    def score_systemone(self, state: str, questions: dict[str, dict[str, Any]]) -> dict[str, Any]:
        from kev.api import Choice, Noul, SystemOneRequest, to_answers, to_record
        from kev.model import SERVE_MAX_BRANCH, SERVE_MAX_STATE

        typed = {}
        for key, spec in questions.items():
            if spec.get("type") == "noul":
                typed[key] = Noul(type="noul", instructions=spec.get("instructions"))
            else:
                typed[key] = Choice(
                    type="choice",
                    instructions=spec.get("instructions"),
                    criteria=spec["criteria"],
                )

        req = SystemOneRequest(state=state, model=self.run_id, questions=typed)
        record, meta = to_record(req)
        encoded = self.model.encode(
            self.tokenizer,
            record,
            max_state=SERVE_MAX_STATE,
            max_branch=SERVE_MAX_BRANCH,
            strict=True,
        )
        t0 = time.perf_counter()
        probs = self.model.probs(encoded)
        _sync_device(self.device)
        latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        if "mlx" in self.backend:
            try:
                import mlx.core as mx
                (getattr(mx, "clear_cache", None) or getattr(mx.metal, "clear_cache", lambda: None))()
            except Exception:
                pass

        return {
            "model": self.run_id,
            "backend": self.backend,
            "dtype": str(getattr(self.model, "dtype", "bfloat16")),
            "temperature": float(self.model.head.temperature),
            "input_tokens": len(encoded["ids"]),
            "state_tokens": encoded["state_tokens"],
            "state_truncated": encoded["state_truncated"],
            "latency_ms": latency_ms,
            "answers": to_answers([[float(x) for x in row.tolist()] for row in probs], meta),
        }

    def score_unified(
        self,
        state: str,
        question: str,
        options: list[str],
        option_descriptions: list[str] | None = None,
    ) -> dict[str, Any]:
        criteria = {
            opt: (
                option_descriptions[idx]
                if option_descriptions and idx < len(option_descriptions)
                else opt
            )
            for idx, opt in enumerate(options)
        }
        raw = self.score_systemone(
            state=state,
            questions={"decision": {"type": "choice", "instructions": question, "criteria": criteria}},
        )
        prob_map = raw["answers"]["decision"].get("probabilities") or {}
        probs = [float(prob_map.get(opt, 0.0)) for opt in options]
        best_idx = max(range(len(probs)), key=lambda i: probs[i])
        return {
            "model": "kev-0.8b",
            "model_id": self.run_id,
            "backend": self.backend,
            "choice_index": best_idx,
            "choice": options[best_idx],
            "probabilities": [round(p, 6) for p in probs],
            "confidence": round(probs[best_idx], 6),
            "parse_ok": True,
            "latency_ms": raw["latency_ms"],
            "input_tokens": raw["input_tokens"],
            "state_truncated": raw["state_truncated"],
            "raw_output": None,
        }


class QwenRunner:
    """Tier 2a/2b: Qwen-0.8B length-normalized log-likelihood and structured-JSON baselines."""

    def __init__(self, model_id: str, device: str) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from decision_eval.paths import resolve_local_model_path

        self.model_id = model_id
        resolved = resolve_local_model_path(model_id)
        self.device = device
        dtype = torch.bfloat16 if device in ("mps", "cuda") else torch.float32

        t0 = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(resolved)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(resolved, dtype=dtype)
        self.model.to(device)
        self.model.eval()
        self.load_s = round(time.perf_counter() - t0, 2)
        self.dtype = str(dtype)

    def score_loglik(
        self,
        state: str,
        question: str,
        options: list[str],
        option_descriptions: list[str] | None = None,
    ) -> dict[str, Any]:
        import torch

        opt_lines = []
        for idx, opt in enumerate(options):
            extra = (
                f": {option_descriptions[idx]}"
                if option_descriptions and idx < len(option_descriptions) and option_descriptions[idx] != opt
                else ""
            )
            opt_lines.append(f"- {opt}{extra}")
        prefix = f"State:\n{state}\n\nQuestion: {question}\nOptions:\n" + "\n".join(opt_lines) + "\nAnswer: "
        prefix_ids = self.tokenizer.encode(prefix, add_special_tokens=True)

        with torch.inference_mode():
            _sync_device(self.device)
            t0 = time.perf_counter()
            prefix_tensor = torch.tensor([prefix_ids], dtype=torch.long, device=self.device)
            try:
                prefix_out = self.model(input_ids=prefix_tensor, logits_to_keep=1)
            except TypeError:
                prefix_out = self.model(input_ids=prefix_tensor)
            last_prefix_logprob = torch.log_softmax(prefix_out.logits[:, -1, :].float(), dim=-1)

            scores: list[float] = []
            start = len(prefix_ids)
            for opt in options:
                opt_ids = self.tokenizer.encode(opt, add_special_tokens=False)
                if not opt_ids:
                    scores.append(-1e9)
                    continue
                total_lp = float(last_prefix_logprob[0, opt_ids[0]].item())
                if len(opt_ids) > 1:
                    full_ids = torch.tensor([prefix_ids + opt_ids], dtype=torch.long, device=self.device)
                    need = len(opt_ids)
                    try:
                        tail_logits = self.model(input_ids=full_ids, logits_to_keep=need).logits.float()
                        for j in range(1, len(opt_ids)):
                            lp = torch.log_softmax(tail_logits[0, j - 1, :], dim=-1)
                            total_lp += float(lp[opt_ids[j]].item())
                    except TypeError:
                        out_logits = self.model(input_ids=full_ids).logits
                        tail_logits = out_logits[0, start : start + len(opt_ids) - 1, :].float()
                        for j in range(1, len(opt_ids)):
                            lp = torch.log_softmax(tail_logits[j - 1], dim=-1)
                            total_lp += float(lp[opt_ids[j]].item())
                scores.append(total_lp / len(opt_ids))
            _sync_device(self.device)
            latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        _clear_device_cache(self.device)

        probs = _softmax(scores)
        best_idx = max(range(len(probs)), key=lambda i: probs[i])
        return {
            "model": "qwen-0.8b-ll",
            "model_id": self.model_id,
            "backend": f"torch-{self.device}",
            "choice_index": best_idx,
            "choice": options[best_idx],
            "probabilities": [round(p, 6) for p in probs],
            "confidence": round(probs[best_idx], 6),
            "parse_ok": True,
            "latency_ms": latency_ms,
            "input_tokens": len(prefix_ids),
            "state_truncated": False,
            "raw_output": None,
        }

    def score_json(
        self,
        state: str,
        question: str,
        options: list[str],
        option_descriptions: list[str] | None = None,
        max_new_tokens: int = 64,
    ) -> dict[str, Any]:
        import torch

        numbered = []
        for idx, opt in enumerate(options):
            extra = (
                f" ({option_descriptions[idx]})"
                if option_descriptions and idx < len(option_descriptions) and option_descriptions[idx] != opt
                else ""
            )
            numbered.append(f"{idx}: {opt}{extra}")
        user_prompt = (
            "You are a decision assistant. Pick exactly one option index and output ONLY valid JSON "
            'with keys "choice_index" (integer) and "confidence" (float between 0.0 and 1.0).\n\n'
            f"State:\n{state}\n\nQuestion: {question}\nOptions:\n" + "\n".join(numbered) + "\n\nJSON:"
        )
        if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
            text = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": user_prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            text = user_prompt

        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        in_len = int(inputs["input_ids"].shape[1])

        with torch.inference_mode():
            _sync_device(self.device)
            t0 = time.perf_counter()
            out_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            _sync_device(self.device)
            latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            gen_text = self.tokenizer.decode(out_ids[0][in_len:], skip_special_tokens=True).strip()
            del inputs, out_ids
        _clear_device_cache(self.device)
        parse_ok, choice_idx, confidence, probs = parse_json_decision(gen_text, len(options))
        return {
            "model": "qwen-0.8b-json",
            "model_id": self.model_id,
            "backend": f"torch-{self.device}",
            "choice_index": choice_idx if parse_ok else -1,
            "choice": options[choice_idx] if parse_ok else None,
            "probabilities": [round(p, 6) for p in probs],
            "confidence": round(confidence, 6),
            "parse_ok": parse_ok,
            "latency_ms": latency_ms,
            "input_tokens": in_len,
            "state_truncated": False,
            "raw_output": gen_text,
        }


class MlxQwenRunner:
    """Tier 2a/2b on native Apple Silicon MLX (via mlx-lm) for fair MLX-vs-MLX comparison with Kev-0.8B."""

    def __init__(self, model_id: str) -> None:
        import mlx_lm
        from decision_eval.paths import resolve_local_model_path

        self.model_id = model_id
        resolved = resolve_local_model_path(model_id)
        self.device = "mlx"
        self.dtype = "bfloat16"
        self.backend = "mlx"
        t0 = time.perf_counter()
        self.model, self.tokenizer = mlx_lm.load(resolved)
        self.load_s = round(time.perf_counter() - t0, 2)

    @staticmethod
    def _clear_mlx_cache() -> None:
        try:
            import mlx.core as mx
            (getattr(mx, "clear_cache", None) or getattr(mx.metal, "clear_cache", lambda: None))()
        except Exception:
            pass

    def score_loglik(
        self,
        state: str,
        question: str,
        options: list[str],
        option_descriptions: list[str] | None = None,
    ) -> dict[str, Any]:
        import mlx.core as mx
        import mlx.nn as nn

        opt_lines = []
        for idx, opt in enumerate(options):
            extra = (
                f": {option_descriptions[idx]}"
                if option_descriptions and idx < len(option_descriptions) and option_descriptions[idx] != opt
                else ""
            )
            opt_lines.append(f"- {opt}{extra}")
        prefix = f"State:\n{state}\n\nQuestion: {question}\nOptions:\n" + "\n".join(opt_lines) + "\nAnswer: "
        prefix_ids = self.tokenizer.encode(prefix)
        start = len(prefix_ids)

        t0 = time.perf_counter()
        scores: list[float] = []
        for opt in options:
            opt_ids = self.tokenizer.encode(opt, add_special_tokens=False)
            if not opt_ids:
                scores.append(-1e9)
                continue
            full_ids = mx.array([prefix_ids + opt_ids], dtype=mx.int32)
            logits = self.model(full_ids)
            seg = logits[0, start - 1 : start + len(opt_ids) - 1, :].astype(mx.float32)
            log_probs = nn.log_softmax(seg, axis=-1)
            idx_arr = mx.array(opt_ids, dtype=mx.int32)
            tok_lps = log_probs[mx.arange(len(opt_ids)), idx_arr]
            mean_lp = mx.mean(tok_lps)
            mx.eval(mean_lp)
            scores.append(float(mean_lp.item()))
        latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        self._clear_mlx_cache()

        probs = _softmax(scores)
        best_idx = max(range(len(probs)), key=lambda i: probs[i])
        return {
            "model": "qwen-0.8b-ll",
            "model_id": self.model_id,
            "backend": "mlx",
            "choice_index": best_idx,
            "choice": options[best_idx],
            "probabilities": [round(p, 6) for p in probs],
            "confidence": round(probs[best_idx], 6),
            "parse_ok": True,
            "latency_ms": latency_ms,
            "input_tokens": len(prefix_ids),
            "state_truncated": False,
            "raw_output": None,
        }

    def score_json(
        self,
        state: str,
        question: str,
        options: list[str],
        option_descriptions: list[str] | None = None,
        max_new_tokens: int = 64,
    ) -> dict[str, Any]:
        import mlx_lm

        numbered = []
        for idx, opt in enumerate(options):
            extra = (
                f" ({option_descriptions[idx]})"
                if option_descriptions and idx < len(option_descriptions) and option_descriptions[idx] != opt
                else ""
            )
            numbered.append(f"{idx}: {opt}{extra}")
        user_prompt = (
            "You are a decision assistant. Pick exactly one option index and output ONLY valid JSON "
            'with keys "choice_index" (integer) and "confidence" (float between 0.0 and 1.0).\n\n'
            f"State:\n{state}\n\nQuestion: {question}\nOptions:\n" + "\n".join(numbered) + "\n\nJSON:"
        )
        if hasattr(self.tokenizer, "apply_chat_template") and getattr(self.tokenizer, "chat_template", None):
            text = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": user_prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            text = user_prompt
        in_len = len(self.tokenizer.encode(text))

        t0 = time.perf_counter()
        try:
            from mlx_lm.sample_utils import make_sampler
            gen_text = mlx_lm.generate(
                self.model, self.tokenizer, prompt=text, max_tokens=max_new_tokens, sampler=make_sampler(temp=0.0)
            )
        except Exception:
            gen_text = mlx_lm.generate(self.model, self.tokenizer, prompt=text, max_tokens=max_new_tokens, temp=0.0)
        latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        self._clear_mlx_cache()

        gen_text = str(gen_text).strip()
        parse_ok, choice_idx, confidence, probs = parse_json_decision(gen_text, len(options))
        return {
            "model": "qwen-0.8b-json",
            "model_id": self.model_id,
            "backend": "mlx",
            "choice_index": choice_idx if parse_ok else -1,
            "choice": options[choice_idx] if parse_ok else None,
            "probabilities": [round(p, 6) for p in probs],
            "confidence": round(confidence, 6),
            "parse_ok": parse_ok,
            "latency_ms": latency_ms,
            "input_tokens": in_len,
            "state_truncated": False,
            "raw_output": gen_text,
        }


def create_qwen_runner(model_id: str, device: str, backend: str = "auto") -> Any:
    """Instantiate Qwen runner on MLX (mlx-lm) when available/requested, else PyTorch."""
    if backend in ("auto", "mlx"):
        try:
            import mlx_lm  # noqa: F401
            return MlxQwenRunner(model_id)
        except ImportError:
            if backend == "mlx":
                raise RuntimeError(
                    "Requested --backend mlx for Qwen, but 'mlx-lm' is not installed. "
                    "Install it with: pip install mlx-lm"
                )
    return QwenRunner(model_id, device)


class OptionalNNAdapter:
    """Tier 4: Plug-and-play adapter for models/car_build_nn.npz."""

    def __init__(self, npz_path: Path) -> None:
        import numpy as np

        self.np = np
        self.engine = None
        data = np.load(npz_path)
        if "feat_mean" in data:
            from self_jev.car_build_nn.model import CarBuildNumPy
            self.engine = CarBuildNumPy(npz_path)
        else:
            self.mean, self.std = data["mean"], data["std"]
            self.w1, self.b1 = data["w1"], data["b1"]
            self.w2, self.b2 = data["w2"], data["b2"]
            self.w3, self.b3 = data["w3"], data["b3"]
            self.temperature = float(data["temperature"][0]) if "temperature" in data else 1.0

    def score_features(self, features: list[float]) -> dict[str, Any]:
        t0 = time.perf_counter()
        if self.engine is not None:
            probs = self.engine.predict_probs(features).tolist()
        else:
            np = self.np
            x = (np.array(features, dtype=np.float32) - self.mean) / (self.std + 1e-8)
            h1 = np.maximum(0.0, x @ self.w1 + self.b1)
            h2 = np.maximum(0.0, h1 @ self.w2 + self.b2)
            logits = (h2 @ self.w3 + self.b3) / max(1e-6, self.temperature)
            logits = logits - np.max(logits)
            exps = np.exp(logits)
            probs = (exps / np.sum(exps)).tolist()
        latency_ms = round((time.perf_counter() - t0) * 1000.0, 4)
        best_idx = max(range(len(probs)), key=lambda i: probs[i])
        return {
            "choice_index": best_idx,
            "probabilities": [round(float(p), 6) for p in probs],
            "confidence": round(float(probs[best_idx]), 6),
            "parse_ok": True,
            "latency_ms": latency_ms,
            "input_tokens": 0,
            "output_tokens": 0,
            "raw_output": None,
        }


class M2QwenLoRARunner:
    """Frozen M2 V5 deployment runner: m2-qwen3.5-0.8b-lora-s42 (T=2.02, option-letter logits).

    Reproduces the exact Colab T4 frozen scorer contract:
      - Clean public Qwen/Qwen3.5-0.8B-Base (never Kev-merged)
      - Native FP32 LoRA adapters (r=16, alpha=32) loaded strictly from checkpoint_dir
      - Exact single-token option-letter IDs validated against tokenizer
      - Exact prompt template, max_length=512 truncation, and T=2.02 softmax over K legal options
      - Fail-fast FileNotFoundError if any required artifact is missing (never silent fallback)
    """

    LETTERS = ("A", "B", "C", "D", "E", "F", "G", "H")

    def __init__(self, checkpoint_dir: Path, device: str = "auto") -> None:
        from decision_eval.paths import HF_CACHE_DIR, configure_bundled_hf_cache, resolve_local_model_path

        configure_bundled_hf_cache()
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.checkpoint_dir = Path(checkpoint_dir)
        weights_path = self.checkpoint_dir / "lora_adapters.pt"
        cfg_path = self.checkpoint_dir / "adapter_config.json"
        cal_path = self.checkpoint_dir / "calibration.json"
        for req_p in (weights_path, cfg_path, cal_path):
            if not req_p.is_file():
                raise FileNotFoundError(
                    f"Required M2 Qwen LoRA artifact missing: {req_p}. Refusing silent fallback."
                )

        self.adapter_cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        self.cal_cfg = json.loads(cal_path.read_text(encoding="utf-8"))
        self.base_model_id: str = str(self.adapter_cfg["base_model"])
        self.temperature: float = float(self.cal_cfg["temperature"])
        self.expected_letter_ids: list[int] = [int(x) for x in self.adapter_cfg["letter_ids"]]
        self.device = pick_device(device)
        # Preserve FP16 base on CUDA/MPS (matching Colab FP16 base + FP32 LoRA), FP32 on CPU
        self.base_dtype = torch.float16 if self.device in ("cuda", "mps") else torch.float32
        self.adapter_dtype = "float32"

        resolved_base = resolve_local_model_path(self.base_model_id)
        cache_kw: dict[str, Any] = {"cache_dir": str(HF_CACHE_DIR)} if HF_CACHE_DIR.is_dir() else {}
        t0 = time.perf_counter()
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                resolved_base, trust_remote_code=True, local_files_only=True, **cache_kw
            )
            local_only = True
        except Exception:
            self.tokenizer = AutoTokenizer.from_pretrained(resolved_base, trust_remote_code=True, **cache_kw)
            local_only = False

        # Validate exact letter token IDs against loaded tokenizer
        resolved_ids: list[int] = []
        for ch in self.LETTERS[: len(self.expected_letter_ids)]:
            t_sp = self.tokenizer.encode(f" {ch}", add_special_tokens=False)
            t_pl = self.tokenizer.encode(ch, add_special_tokens=False)
            if len(t_sp) == 1:
                resolved_ids.append(int(t_sp[0]))
            elif len(t_pl) == 1:
                resolved_ids.append(int(t_pl[0]))
            else:
                raise ValueError(f"Option letter {ch} failed single-token validation: {t_sp}, {t_pl}")
        if resolved_ids != self.expected_letter_ids:
            raise ValueError(
                f"Tokenizer letter_ids {resolved_ids} != saved adapter_config letter_ids {self.expected_letter_ids}"
            )
        self.letter_ids = resolved_ids

        class _EvalLoRALinear(nn.Module):
            def __init__(self, base_layer: nn.Linear, r: int, alpha: float):
                super().__init__()
                self.base = base_layer
                self.scaling = float(alpha) / float(r)
                self.lora_A = nn.Parameter(torch.empty((r, base_layer.in_features), dtype=torch.float32))
                self.lora_B = nn.Parameter(torch.zeros((base_layer.out_features, r), dtype=torch.float32))

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                base_out = self.base(x)
                lora_out = F.linear(F.linear(x.float(), self.lora_A), self.lora_B) * self.scaling
                return base_out + lora_out.to(base_out.dtype)

        model = AutoModelForCausalLM.from_pretrained(
            resolved_base, dtype=self.base_dtype, trust_remote_code=True, local_files_only=local_only, **cache_kw
        )
        targets = set(self.adapter_cfg["target_modules"])
        lora_r = int(self.adapter_cfg["lora_r"])
        lora_alpha = float(self.adapter_cfg["lora_alpha"])
        replaced = 0
        for _name, mod in list(model.named_modules()):
            for child_name, child in list(mod.named_children()):
                if isinstance(child, nn.Linear) and child_name in targets:
                    setattr(mod, child_name, _EvalLoRALinear(child, r=lora_r, alpha=lora_alpha))
                    replaced += 1
        if replaced == 0:
            raise RuntimeError("No target linear layers found for M2 LoRA injection.")

        lora_state = torch.load(weights_path, map_location="cpu", weights_only=True)
        missing, unexpected = model.load_state_dict(lora_state, strict=False)
        missing_lora = [k for k in missing if "lora_" in k]
        if missing_lora or unexpected:
            raise RuntimeError(f"M2 LoRA weight mismatch: missing_lora={missing_lora}, unexpected={unexpected}")

        model.to(self.device)
        model.eval()
        self.model = model
        self.lid_tensor = torch.tensor(self.letter_ids, dtype=torch.long, device=self.device)
        self.load_s = round(time.perf_counter() - t0, 2)
        self.backend = f"torch-{self.device}"
        self.dtype = f"base={str(self.base_dtype).replace('torch.', '')}+lora=float32"

    def format_prompt(
        self,
        state: str,
        question: str,
        options: list[str],
        option_descriptions: list[str] | None = None,
    ) -> str:
        descs = option_descriptions if (option_descriptions and len(option_descriptions) == len(options)) else options
        opt_lines = [
            f"{self.LETTERS[i]}. {options[i]} — {descs[i]}"
            for i in range(len(options))
        ]
        return (
            f"Context:\n{state}\n\n"
            f"Question: {question}\n\n"
            f"Options:\n" + "\n".join(opt_lines) + "\n\n"
            f"Select the single best option letter (A-{self.LETTERS[len(options) - 1]}).\nAnswer:"
        )

    def score_decision(
        self,
        state: str,
        question: str,
        options: list[str],
        option_descriptions: list[str] | None = None,
        max_len: int = 512,
    ) -> dict[str, Any]:
        import torch

        k = len(options)
        if k < 2 or k > len(self.letter_ids):
            raise ValueError(f"m2-qwen3.5-0.8b-lora-s42 supports 2..{len(self.letter_ids)} options, got {k}")
        prompt = self.format_prompt(state, question, options, option_descriptions)
        enc = self.tokenizer(prompt, truncation=True, max_length=max_len, return_tensors="pt")
        in_tokens = int(enc["input_ids"].shape[1])
        truncated = in_tokens >= max_len
        enc = {key: val.to(self.device) for key, val in enc.items()}

        with torch.inference_mode():
            _sync_device(self.device)
            t0 = time.perf_counter()
            out_logits = self.model(**enc).logits[0, -1, self.lid_tensor[:k]].float().cpu().tolist()
            _sync_device(self.device)
            latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        _clear_device_cache(self.device)

        scaled = [float(x) / max(self.temperature, 1e-4) for x in out_logits]
        probs = _softmax(scaled)
        best_idx = max(range(k), key=lambda i: probs[i])
        return {
            "model": "m2-qwen3.5-0.8b-lora-s42",
            "model_id": f"{self.base_model_id}+m2_lora_s42",
            "backend": self.backend,
            "dtype": self.dtype,
            "temperature": self.temperature,
            "choice_index": best_idx,
            "choice": options[best_idx],
            "raw_logits": [round(float(x), 6) for x in out_logits],
            "probabilities": [round(float(p), 6) for p in probs],
            "confidence": round(float(probs[best_idx]), 6),
            "parse_ok": True,
            "latency_ms": latency_ms,
            "input_tokens": in_tokens,
            "state_truncated": truncated,
            "raw_output": None,
        }


def parse_json_decision(gen_text: str, n_options: int) -> tuple[bool, int, float, list[float]]:
    parse_ok = False
    choice_idx = 0
    confidence = 0.0
    try:
        match = re.search(r"\{.*?\}", gen_text, flags=re.DOTALL)
        parsed = json.loads(match.group(0) if match else gen_text)
        idx_val = int(parsed["choice_index"])
        conf_val = float(parsed.get("confidence", 0.5))
        if 0 <= idx_val < n_options:
            choice_idx = idx_val
            confidence = max(0.0, min(1.0, conf_val))
            parse_ok = True
    except Exception:
        parse_ok = False

    probs = [0.0] * n_options
    if parse_ok:
        rem = (1.0 - confidence) / max(1, n_options - 1)
        probs = [confidence if i == choice_idx else rem for i in range(n_options)]
    return parse_ok, choice_idx, confidence, probs


def call_frontier_llm(
    base_url: str,
    model: str,
    api_key: str,
    state: str,
    question: str,
    options: list[str],
    option_descriptions: list[str] | None = None,
) -> dict[str, Any]:
    """Tier 1: Call any OpenAI-compatible /chat/completions endpoint."""
    numbered = []
    for idx, opt in enumerate(options):
        extra = (
            f" ({option_descriptions[idx]})"
            if option_descriptions and idx < len(option_descriptions) and option_descriptions[idx] != opt
            else ""
        )
        numbered.append(f"{idx}: {opt}{extra}")
    prompt = (
        "You are a decision assistant. Pick exactly one option index and output ONLY valid JSON "
        'with keys "choice_index" (integer) and "confidence" (float between 0.0 and 1.0).\n\n'
        f"State:\n{state}\n\nQuestion: {question}\nOptions:\n" + "\n".join(numbered) + "\n\nJSON:"
    )
    url = base_url.rstrip("/") + "/chat/completions"
    body_obj: dict[str, Any] = {
        "model": model,
        "temperature": 0.0,
        "max_tokens": 128,
        "response_format": {"type": "json_object"},
        "messages": [{"role": "user", "content": prompt}],
    }
    if "openrouter.ai" in base_url:
        body_obj["reasoning"] = {"effort": "none", "exclude": True}
    else:
        body_obj["enable_thinking"] = False
    payload = json.dumps(body_obj).encode("utf-8")
    http_req = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(http_req, timeout=60) as resp:
        raw_json = json.loads(resp.read().decode("utf-8"))
    latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)

    gen_text = str(raw_json.get("choices", [{}])[0].get("message", {}).get("content", "")).strip()
    usage = raw_json.get("usage") or {}
    parse_ok, choice_idx, confidence, probs = parse_json_decision(gen_text, len(options))
    return {
        "model": "frontier-llm",
        "model_id": model,
        "backend": "api",
        "choice_index": choice_idx if parse_ok else -1,
        "choice": options[choice_idx] if parse_ok else None,
        "probabilities": [round(p, 6) for p in probs],
        "confidence": round(confidence, 6),
        "parse_ok": parse_ok,
        "latency_ms": latency_ms,
        "input_tokens": usage.get("prompt_tokens"),
        "output_tokens": usage.get("completion_tokens"),
        "state_truncated": False,
        "raw_output": gen_text,
    }
