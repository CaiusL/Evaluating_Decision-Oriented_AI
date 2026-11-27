"""Self-JEV cross-encoder. MiniLM initializes the language front end."""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
ENCODER_NAME = "cross-encoder/ms-marco-MiniLM-L6-v2"
ENCODER_DIR = ROOT / "models" / "ms-marco-MiniLM-L6-v2"
MAX_LENGTH = 256


def ensure_encoder() -> Path:
    if (ENCODER_DIR / "config.json").exists():
        return ENCODER_DIR
    from huggingface_hub import snapshot_download

    ENCODER_DIR.parent.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=ENCODER_NAME, local_dir=ENCODER_DIR)
    return ENCODER_DIR


def load_encoder(device: torch.device):
    path = ensure_encoder()
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path)
    model.to(device)
    return model, tokenizer


def pair_text(question: str, situation: str, candidate: dict) -> tuple[str, str]:
    return f"{question} {situation}", f"{candidate['id']}: {candidate['description']}"


def encode(tokenizer, pairs: list[tuple[str, str]]):
    left = [item[0] for item in pairs]
    right = [item[1] for item in pairs]
    encoded = tokenizer(left, right, padding=True, truncation=False, return_tensors="pt")
    length = int(encoded["input_ids"].shape[-1])
    if length > MAX_LENGTH:
        raise RuntimeError(f"a render is {length} tokens, above {MAX_LENGTH}")
    return encoded


def training_logits(model, tokenizer, pairs: list[tuple[str, str]], device: torch.device):
    encoded = {key: value.to(device) for key, value in encode(tokenizer, pairs).items()}
    logits = model(**encoded).logits.squeeze(-1)
    if logits.ndim == 0:
        logits = logits.unsqueeze(0)
    return logits


def score_pairs(model, tokenizer, pairs: list[tuple[str, str]], device: torch.device) -> list[float]:
    with torch.inference_mode():
        logits = training_logits(model, tokenizer, pairs, device)
    values = logits.detach().float().cpu().tolist()
    if isinstance(values, float):
        return [values]
    return values
