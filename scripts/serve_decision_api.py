"""Unified local API server + built-in Web UI hosting Kev-0.8B and Qwen-0.8B on Apple Silicon (M4 Mac).

Endpoints:
  GET  /              - Built-in no-code Web UI (side-by-side demo + batch benchmark + CSV/JSON export)
  GET  /health        - Loaded models, active backends (mlx/mps/cpu), and memory/load stats
  GET  /v1/datasets   - Auto-discovered local JSONL benchmark files
  POST /v1/score      - Unified option-scoring API across kev-0.8b, qwen-0.8b-ll, and qwen-0.8b-json
  POST /v1/compare    - Run all loaded models side-by-side on one (state, question, options) item
  POST /v1/frontier   - Proxy to any OpenAI-compatible Frontier LLM API (OpenAI, OpenRouter, Gemini, etc.)
  POST /v1/systemone  - Native Kev / TypeSafe System One compatible endpoint

Quick start (Python 3.12 or 3.13 venv on M4 Mac):
  pip install fastapi uvicorn torch transformers accelerate mlx
  # plus jaredpalmer/kev installed in the same venv
  python scripts/serve_decision_api.py --port 8008 --open-browser
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request
import webbrowser
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Request / Response Schemas
# ---------------------------------------------------------------------------


class ScoreRequest(BaseModel):
    model: Literal["kev-0.8b", "qwen-0.8b-ll", "qwen-0.8b-json"] = "kev-0.8b"
    state: str = Field(..., description="Document, telemetry summary, or workflow context.")
    question: str = Field(..., description="Decision question to answer.")
    options: list[str] = Field(..., min_length=2, description="Allowed candidate answers.")
    option_descriptions: list[str] | None = Field(
        default=None,
        description="Optional per-option descriptions (same length as options).",
    )


class CompareRequest(BaseModel):
    state: str
    question: str
    options: list[str] = Field(..., min_length=2)
    option_descriptions: list[str] | None = None
    models: list[Literal["kev-0.8b", "qwen-0.8b-ll", "qwen-0.8b-json"]] | None = None


class SystemOnePayload(BaseModel):
    state: str
    questions: dict[str, dict[str, Any]]
    model: str = "jaredpalmer/kev-0.8b"


class FrontierRequest(BaseModel):
    base_url: str = "https://openrouter.ai/api/v1"
    model: str = "openai/gpt-4o-mini"
    api_key: str
    state: str
    question: str
    options: list[str] = Field(..., min_length=2)
    option_descriptions: list[str] | None = None


class NNScoreRequest(BaseModel):
    features: list[float]
    options: list[str] = Field(..., min_length=2)


class AnalyzeRequest(BaseModel):
    records: list[dict[str, Any]]
    models: list[str]
    config: dict[str, Any] = Field(default_factory=dict)
    frontier_in_price: float = 0.15
    frontier_out_price: float = 0.60


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _pick_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _sync_device(device: str) -> None:
    if device == "mps" and torch.backends.mps.is_available():
        torch.mps.synchronize()
    elif device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()


def _softmax(logits: list[float]) -> list[float]:
    m = max(logits)
    exps = [math.exp(x - m) for x in logits]
    s = sum(exps) or 1.0
    return [x / s for x in exps]


def _maybe_patch_local_qwen_base() -> str | None:
    """Reuse local ModelScope Qwen3.5-0.8B-Base cache if present (matches self_jev/kev_ask.py)."""
    base = os.environ.get("MODELSCOPE_BASE") or os.path.expanduser(
        "~/.cache/modelscope/Qwen/Qwen3.5-0.8B-Base"
    )
    weights = Path(base) / "model.safetensors-00001-of-00001.safetensors"
    if not weights.is_file():
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


# ---------------------------------------------------------------------------
# Model Runners
# ---------------------------------------------------------------------------


class KevRunner:
    def __init__(self, run_id: str, device: str) -> None:
        self.run_id = run_id
        self.device = device
        self.base_source = _maybe_patch_local_qwen_base() or "huggingface"

        from kev.checkpoint import Checkpoint, LoadOptions

        t0 = time.perf_counter()
        options = LoadOptions(backend="auto", dtype=torch.bfloat16, merge=True)
        ckpt = Checkpoint(run_id)
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
        criteria = {}
        for idx, opt in enumerate(options):
            desc = (
                option_descriptions[idx]
                if option_descriptions and idx < len(option_descriptions)
                else opt
            )
            criteria[opt] = desc

        raw = self.score_systemone(
            state=state,
            questions={"decision": {"type": "choice", "instructions": question, "criteria": criteria}},
        )
        ans = raw["answers"]["decision"]
        prob_map = ans.get("probabilities") or {}
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
    """Runs Qwen-0.8B for both length-normalized log-likelihood scoring and structured JSON."""

    def __init__(self, model_id: str, device: str) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_id = model_id
        self.device = device
        dtype = torch.bfloat16 if device in ("mps", "cuda") else torch.float32

        t0 = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
        self.model.to(device)
        self.model.eval()
        self.load_s = round(time.perf_counter() - t0, 2)
        self.dtype = str(dtype)

    @torch.inference_mode()
    def score_loglik(
        self,
        state: str,
        question: str,
        options: list[str],
        option_descriptions: list[str] | None = None,
    ) -> dict[str, Any]:
        """Standard MMLU-style length-normalized log-likelihood over candidate options."""
        opt_lines = []
        for idx, opt in enumerate(options):
            extra = (
                f": {option_descriptions[idx]}"
                if option_descriptions and idx < len(option_descriptions) and option_descriptions[idx] != opt
                else ""
            )
            opt_lines.append(f"- {opt}{extra}")
        prefix = (
            f"State:\n{state}\n\n"
            f"Question: {question}\n"
            f"Options:\n" + "\n".join(opt_lines) + "\n"
            f"Answer: "
        )
        prefix_ids = self.tokenizer.encode(prefix, add_special_tokens=True)

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
        if self.device == "mps" and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()

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

    @torch.inference_mode()
    def score_json(
        self,
        state: str,
        question: str,
        options: list[str],
        option_descriptions: list[str] | None = None,
        max_new_tokens: int = 64,
    ) -> dict[str, Any]:
        """Conventional structured-JSON generation baseline (tracks parse_ok & verbalized confidence)."""
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
            f"State:\n{state}\n\n"
            f"Question: {question}\n"
            f"Options:\n" + "\n".join(numbered) + "\n\n"
            "JSON:"
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
        if self.device == "mps" and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
        parse_ok = False
        choice_idx = 0
        confidence = 0.0
        try:
            match = re.search(r"\{.*?\}", gen_text, flags=re.DOTALL)
            parsed = json.loads(match.group(0) if match else gen_text)
            idx_val = int(parsed["choice_index"])
            conf_val = float(parsed.get("confidence", 0.5))
            if 0 <= idx_val < len(options):
                choice_idx = idx_val
                confidence = max(0.0, min(1.0, conf_val))
                parse_ok = True
        except Exception:
            parse_ok = False

        probs = [0.0] * len(options)
        if parse_ok:
            rem = (1.0 - confidence) / max(1, len(options) - 1)
            probs = [confidence if i == choice_idx else rem for i in range(len(options))]

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


def call_frontier_llm(
    base_url: str,
    model: str,
    api_key: str,
    state: str,
    question: str,
    options: list[str],
    option_descriptions: list[str] | None = None,
) -> dict[str, Any]:
    """Call any OpenAI-compatible /chat/completions endpoint and return unified score JSON."""
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
        f"State:\n{state}\n\n"
        f"Question: {question}\n"
        "Options:\n" + "\n".join(numbered) + "\n\nJSON:"
    )
    url = base_url.rstrip("/") + "/chat/completions"
    payload = json.dumps({
        "model": model,
        "temperature": 0.0,
        "max_tokens": 80,
        "messages": [{"role": "user", "content": prompt}],
    }).encode("utf-8")
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
    parse_ok = False
    choice_idx = 0
    confidence = 0.0
    try:
        match = re.search(r"\{.*?\}", gen_text, flags=re.DOTALL)
        parsed = json.loads(match.group(0) if match else gen_text)
        idx_val = int(parsed["choice_index"])
        conf_val = float(parsed.get("confidence", 0.5))
        if 0 <= idx_val < len(options):
            choice_idx = idx_val
            confidence = max(0.0, min(1.0, conf_val))
            parse_ok = True
    except Exception:
        parse_ok = False

    probs = [0.0] * len(options)
    if parse_ok:
        rem = (1.0 - confidence) / max(1, len(options) - 1)
        probs = [confidence if i == choice_idx else rem for i in range(len(options))]

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


# ---------------------------------------------------------------------------
# FastAPI App Factory
# ---------------------------------------------------------------------------


def create_app(
    kev_run: str | None,
    qwen_base_model: str | None,
    qwen_instruct_model: str | None,
    device: str,
) -> FastAPI:
    state_store: dict[str, Any] = {}

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        if kev_run:
            state_store["kev"] = KevRunner(kev_run, device)
        if qwen_base_model:
            state_store["qwen_ll"] = QwenRunner(qwen_base_model, device)
        if qwen_instruct_model:
            if qwen_instruct_model == qwen_base_model and "qwen_ll" in state_store:
                state_store["qwen_json"] = state_store["qwen_ll"]
            else:
                state_store["qwen_json"] = QwenRunner(qwen_instruct_model, device)
        elif "qwen_ll" in state_store:
            # Allow JSON endpoint to reuse base model if no separate instruct checkpoint is supplied
            state_store["qwen_json"] = state_store["qwen_ll"]
        yield
        state_store.clear()

    app = FastAPI(title="Decision Models Local API (Kev-0.8B + Qwen-0.8B)", lifespan=lifespan)

    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    ui_file = script_dir / "decision_ui.html"

    def _discover_datasets() -> dict[str, tuple[str, Path]]:
        bench_dir = repo_root / "data" / "benchmarks_v2"
        candidates = [
            ("all_test", "Full Multi-Suite Benchmark v2 (all_test.jsonl)", bench_dir / "all_test.jsonl"),
            ("all_calibrate", "Full Multi-Suite Calibrate v2 (all_calibrate.jsonl)", bench_dir / "all_calibrate.jsonl"),
            ("suite1_software_ops_test", "Suite 1: Software Ops (test)", bench_dir / "suite1_software_ops_test.jsonl"),
            ("suite2_evidence_discipline_test", "Suite 2: Engineering Evidence Discipline (test)", bench_dir / "suite2_evidence_discipline_test.jsonl"),
            ("suite3_car_build_test", "Suite 3: Car-Build Enriched (test)", bench_dir / "suite3_car_build_test.jsonl"),
            ("suite4_stress_test", "Suite 4: Stress Test (Context Length + Paraphrase)", bench_dir / "suite4_stress_test.jsonl"),
            ("car_build_test", "Legacy Car-Build Benchmark (test.jsonl)", repo_root / "data" / "car_build_bench" / "test.jsonl"),
            ("car_build_calibrate", "Legacy Car-Build Benchmark (calibrate.jsonl)", repo_root / "data" / "car_build_bench" / "calibrate.jsonl"),
            ("car_build_train", "Legacy Car-Build Benchmark (train.jsonl)", repo_root / "data" / "car_build_bench" / "train.jsonl"),
        ]
        return {ds_id: (label, p) for ds_id, label, p in candidates if p.is_file()}

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        if not ui_file.is_file():
            return HTMLResponse("<h3>decision_ui.html not found next to serve_decision_api.py</h3>", status_code=404)
        return HTMLResponse(ui_file.read_text(encoding="utf-8"))

    @app.get("/v1/datasets")
    def list_datasets() -> dict[str, Any]:
        found = _discover_datasets()
        out = []
        for ds_id, (label, path) in found.items():
            count = sum(1 for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip())
            out.append({"id": ds_id, "label": label, "count": count})
        return {"datasets": out}

    @app.get("/v1/datasets/{dataset_id}")
    def get_dataset(dataset_id: str) -> dict[str, Any]:
        found = _discover_datasets()
        if dataset_id not in found:
            raise HTTPException(status_code=404, detail=f"Dataset not found: {dataset_id}")
        label, path = found[dataset_id]
        rows = [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        return {"id": dataset_id, "label": label, "rows": rows}

    @app.post("/v1/frontier")
    def frontier_score(req: FrontierRequest) -> dict[str, Any]:
        try:
            return call_frontier_llm(
                req.base_url,
                req.model,
                req.api_key,
                req.state,
                req.question,
                req.options,
                req.option_descriptions,
            )
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            raise HTTPException(status_code=502, detail=f"Frontier API HTTP {e.code}: {body[:300]}")
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Frontier API error: {e}")

    @app.get("/health")
    def health() -> dict[str, Any]:
        info: dict[str, Any] = {"status": "ok", "device": device, "models": {}}
        if "kev" in state_store:
            k: KevRunner = state_store["kev"]
            info["models"]["kev-0.8b"] = {
                "model_id": k.run_id,
                "backend": k.backend,
                "base_source": k.base_source,
                "load_s": k.load_s,
            }
        if "qwen_ll" in state_store:
            q: QwenRunner = state_store["qwen_ll"]
            info["models"]["qwen-0.8b-ll"] = {
                "model_id": q.model_id,
                "backend": f"torch-{q.device}",
                "dtype": q.dtype,
                "load_s": q.load_s,
            }
        if "qwen_json" in state_store:
            qj: QwenRunner = state_store["qwen_json"]
            info["models"]["qwen-0.8b-json"] = {
                "model_id": qj.model_id,
                "backend": f"torch-{qj.device}",
                "dtype": qj.dtype,
                "load_s": qj.load_s,
            }
        return info

    @app.post("/v1/score")
    def score(req: ScoreRequest) -> dict[str, Any]:
        if req.model == "kev-0.8b":
            if "kev" not in state_store:
                raise HTTPException(status_code=503, detail="kev-0.8b is not loaded")
            return state_store["kev"].score_unified(
                req.state, req.question, req.options, req.option_descriptions
            )
        if req.model == "qwen-0.8b-ll":
            if "qwen_ll" not in state_store:
                raise HTTPException(status_code=503, detail="qwen-0.8b-ll is not loaded")
            return state_store["qwen_ll"].score_loglik(
                req.state, req.question, req.options, req.option_descriptions
            )
        if req.model == "qwen-0.8b-json":
            if "qwen_json" not in state_store:
                raise HTTPException(status_code=503, detail="qwen-0.8b-json is not loaded")
            return state_store["qwen_json"].score_json(
                req.state, req.question, req.options, req.option_descriptions
            )
        raise HTTPException(status_code=400, detail=f"Unknown model: {req.model}")

    @app.post("/v1/compare")
    def compare(req: CompareRequest) -> dict[str, Any]:
        targets = req.models or [
            m
            for m, key in [
                ("kev-0.8b", "kev"),
                ("qwen-0.8b-ll", "qwen_ll"),
                ("qwen-0.8b-json", "qwen_json"),
            ]
            if key in state_store
        ]
        results = {}
        for m in targets:
            sub = ScoreRequest(
                model=m,
                state=req.state,
                question=req.question,
                options=req.options,
                option_descriptions=req.option_descriptions,
            )
            results[m] = score(sub)
        return {"question": req.question, "options": req.options, "results": results}

    @app.post("/v1/systemone")
    def systemone(req: SystemOnePayload) -> dict[str, Any]:
        if "kev" not in state_store:
            raise HTTPException(status_code=503, detail="kev-0.8b is not loaded")
        return state_store["kev"].score_systemone(req.state, req.questions)

    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    reports_dir = repo_root / "reports" / "full_benchmark_v2"
    charts_dir = reports_dir / "charts"
    nn_path = repo_root / "models" / "car_build_nn.npz"

    @app.get("/v1/nn_status")
    def nn_status() -> dict[str, Any]:
        return {"available": nn_path.is_file(), "path": str(nn_path)}

    @app.post("/v1/score_nn")
    def score_nn(req: NNScoreRequest) -> dict[str, Any]:
        if not nn_path.is_file():
            raise HTTPException(status_code=404, detail="models/car_build_nn.npz not found")
        try:
            from evaluate_models import OptionalNNAdapter

            if "nn_adapter" not in state_store:
                state_store["nn_adapter"] = OptionalNNAdapter(nn_path)
            raw = state_store["nn_adapter"].score_features(req.features)
            idx = int(raw["choice_index"])
            return {
                "model": "car-build-nn",
                "model_id": "models/car_build_nn.npz",
                "backend": "numpy-cpu",
                "choice_index": idx,
                "choice": req.options[idx] if 0 <= idx < len(req.options) else None,
                "probabilities": raw["probabilities"],
                "confidence": raw["confidence"],
                "parse_ok": True,
                "latency_ms": raw["latency_ms"],
                "input_tokens": 0,
                "output_tokens": 0,
                "state_truncated": False,
                "raw_output": None,
            }
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Tier 4 NN scoring failed: {exc}")

    @app.post("/v1/analyze")
    def analyze_batch(req: AnalyzeRequest) -> dict[str, Any]:
        import benchmark_metrics
        import generate_charts

        summary = benchmark_metrics.build_summary(
            req.records,
            req.models,
            req.config,
            frontier_in_price=req.frontier_in_price,
            frontier_out_price=req.frontier_out_price,
        )
        benchmark_metrics.write_summary_files(summary, reports_dir)
        charts = generate_charts.generate_all_charts(summary, charts_dir)
        return {"summary": summary, "charts": charts}

    @app.get("/v1/reports/latest")
    def latest_report() -> dict[str, Any]:
        summary_path = reports_dir / "benchmark_summary.json"
        if not summary_path.is_file():
            return {"available": False, "summary": None, "charts": []}
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        charts = []
        if charts_dir.is_dir():
            for p in sorted(charts_dir.glob("*.svg")):
                charts.append({"filename": p.name, "title": p.stem.replace("_", " ").title()})
        return {"available": True, "summary": summary, "charts": charts}

    @app.get("/v1/reports/csv")
    def download_report_csv() -> Response:
        csv_path = reports_dir / "benchmark_summary.csv"
        if not csv_path.is_file():
            raise HTTPException(status_code=404, detail="No benchmark_summary.csv generated yet")
        return Response(content=csv_path.read_text(encoding="utf-8"), media_type="text/csv")

    @app.get("/v1/charts/{filename}")
    def get_chart(filename: str) -> Response:
        safe = Path(filename).name
        if not safe.endswith(".svg"):
            raise HTTPException(status_code=400, detail="Only .svg charts are served")
        target = charts_dir / safe
        if not target.is_file():
            raise HTTPException(status_code=404, detail=f"Chart not found: {safe}")
        return Response(content=target.read_text(encoding="utf-8"), media_type="image/svg+xml")

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Host Kev-0.8B and Qwen-0.8B behind one local HTTP API.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8008)
    parser.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"])
    parser.add_argument("--kev-run", default="jaredpalmer/kev-0.8b", help="Set empty '' to disable Kev.")
    parser.add_argument(
        "--qwen-base-model",
        default="Qwen/Qwen3.5-0.8B-Base",
        help="HuggingFace ID or local path for log-likelihood scoring. Set '' to disable.",
    )
    parser.add_argument(
        "--qwen-instruct-model",
        default="",
        help="Optional Instruct model ID for structured JSON baseline (e.g., Qwen/Qwen3.5-0.8B).",
    )
    parser.add_argument("--open-browser", action="store_true", help="Open the Web UI automatically in browser.")
    args = parser.parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    device = _pick_device(args.device)
    app = create_app(
        kev_run=args.kev_run or None,
        qwen_base_model=args.qwen_base_model or None,
        qwen_instruct_model=args.qwen_instruct_model or None,
        device=device,
    )
    if args.open_browser:
        url = f"http://{args.host}:{args.port}"
        threading.Timer(2.0, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
