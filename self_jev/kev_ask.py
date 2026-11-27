"""Score one Kev request. Imported only by the Kev interpreter, never by the project runtime.

stdin: {"state": str, "questions": {id: {type, instructions, criteria}}}
argv[1]: path that receives the answer JSON. The loader's own logs stay on stderr.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path


def _use_local_base() -> str | None:
    base = os.environ.get("MODELSCOPE_BASE") or os.path.expanduser("~/.cache/modelscope/Qwen/Qwen3.5-0.8B-Base")
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


def _typed(questions: dict):
    from kev.api import Choice, Noul

    typed = {}
    for key, spec in questions.items():
        if spec["type"] == "noul":
            typed[key] = Noul(type="noul", instructions=spec.get("instructions"))
        else:
            typed[key] = Choice(type="choice", instructions=spec.get("instructions"), criteria=spec["criteria"])
    return typed


def main() -> None:
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    payload = json.load(sys.stdin)
    base = _use_local_base()
    import torch
    from kev.api import SystemOneRequest, to_answers, to_record
    from kev.checkpoint import Checkpoint, LoadOptions
    from kev.model import SERVE_MAX_BRANCH, SERVE_MAX_STATE

    print("loading jaredpalmer/kev-0.8b", file=sys.stderr, flush=True)
    started = time.time()
    options = LoadOptions(backend="auto", dtype=torch.bfloat16, merge=True)
    checkpoint = Checkpoint("jaredpalmer/kev-0.8b")
    tokenizer, model = checkpoint.load("mps", options)
    request = SystemOneRequest(state=payload["state"], model="jaredpalmer/kev-0.8b", questions=_typed(payload["questions"]))
    record, meta = to_record(request)
    # The training default keeps the first 384 state tokens and drops the rest. The serving limit reads the whole record.
    encoded = model.encode(
        tokenizer,
        record,
        max_state=SERVE_MAX_STATE,
        max_branch=SERVE_MAX_BRANCH,
        strict=True,
    )
    scored = time.time()
    probabilities = model.probs(encoded)
    body = {
        "model": "jaredpalmer/kev-0.8b",
        "base_source": base or "huggingface",
        "backend": model.backend,
        "dtype": str(model.dtype),
        "temperature": float(model.head.temperature),
        "input_tokens": len(encoded["ids"]),
        "state_tokens": encoded["state_tokens"],
        "state_truncated": encoded["state_truncated"],
        "latency_ms": round((time.time() - scored) * 1000, 1),
        "load_s": round(time.time() - started, 1),
        "answers": to_answers([[float(item) for item in row.tolist()] for row in probabilities], meta),
    }
    text = json.dumps(body)
    if len(sys.argv) > 1:
        Path(sys.argv[1]).write_text(text)
    else:
        sys.stdout.write(text)
    print(f"scored {body['input_tokens']} tokens in {body['latency_ms']} ms", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
