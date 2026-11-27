"""Comprehensive offline benchmark runner for decision-oriented AI.

Evaluates directly in-process (reusing KevRunner, QwenRunner, and Frontier proxy from serve_decision_api.py):
  - Local 0.8B models: kev-0.8b, qwen-0.8b-ll, qwen-0.8b-json
  - Configurable Frontier LLM API via scripts/frontier_config.json (fill in base_url, model, api_key later)
  - Optional Tier 4 task-specific NN (models/car_build_nn.npz) on tabular Suite 3 items
  - Confidence-gated hybrid cascades (Local 0.8B -> Frontier LLM) with thresholds fitted on calibrate split

Supports --swap-order, --resume, --calibrate-file, --suites, --limit, and --mock.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from benchmark_metrics import build_summary, write_summary_files

DEFAULT_TEST = REPO_ROOT / "data" / "benchmarks" / "all_test.jsonl"
DEFAULT_OUT = REPO_ROOT / "reports" / "full_benchmark"


def _normalize_response(res: dict[str, Any], n_opts: int) -> dict[str, Any]:
    return {
        "choice_index": int(res.get("choice_index", -1)),
        "probabilities": [float(p) for p in (res.get("probabilities") or [0.0] * n_opts)],
        "confidence": float(res.get("confidence", 0.0)),
        "parse_ok": bool(res.get("parse_ok", False)),
        "latency_ms": float(res.get("latency_ms", 0.0)),
        "input_tokens": int(res.get("input_tokens") or 0),
        "output_tokens": int(res.get("output_tokens") or 0),
        "raw_output": res.get("raw_output"),
    }


class OptionalNNAdapter:
    """Plug-and-play adapter for models/car_build_nn.npz (Tier 4 task-specific NN)."""

    def __init__(self, npz_path: Path) -> None:
        if str(REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(REPO_ROOT))
        from self_jev.car_build_nn.adapter import TaskSpecificModelScorer

        self.scorer = TaskSpecificModelScorer(npz_path)

    def score_features(self, features: list[float]) -> dict[str, Any]:
        return self.scorer.score(features)


class MiniLMDecisionAdapter:
    """Multi-suite self-trained cross-encoder adapter (single model, K-fold, or LoRA 8k ensemble)."""

    def __init__(self, model_dir: Path) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        from train_self_jev_minilm import score_item_logits

        self._torch = torch
        self._score_chunked = score_item_logits
        self.device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
        self.use_amp = self.device.type == "cuda"

        ens_cfg_path = model_dir / "ensemble_config.json"
        cfg = json.loads(ens_cfg_path.read_text(encoding="utf-8")) if ens_cfg_path.is_file() else {}
        fold_dirs = [model_dir / f for f in cfg["folds"]] if cfg.get("folds") else [model_dir]
        self.is_lora = bool(cfg.get("is_lora", False))
        self.option_first = bool(cfg.get("option_first", False))
        self.max_length = int(cfg.get("max_length", 512))

        self.tokenizer = AutoTokenizer.from_pretrained(fold_dirs[0])
        self.models = []
        if self.is_lora:
            from peft import PeftModel
            base_id = str(cfg["base_model"])
            for fd in fold_dirs:
                base = AutoModelForSequenceClassification.from_pretrained(base_id)
                mdl = PeftModel.from_pretrained(base, fd).merge_and_unload().to(self.device).eval()
                self.models.append(mdl)
        else:
            for fd in fold_dirs:
                self.models.append(AutoModelForSequenceClassification.from_pretrained(fd).to(self.device).eval())

        temp_path = model_dir / "temperature.json"
        self.temperature = 1.0
        if temp_path.is_file():
            self.temperature = float(json.loads(temp_path.read_text(encoding="utf-8")).get("temperature", 1.0))

    def score_text(self, context: str, question: str, options: list[str], descs: list[str]) -> dict[str, Any]:
        t0 = time.perf_counter()
        item = {"context": context, "question": question, "options": options, "option_descriptions": descs}
        with self._torch.inference_mode():
            logits_sum = None
            for mdl in self.models:
                with self._torch.amp.autocast("cuda", enabled=self.use_amp):
                    if self.max_length > 512:
                        from train_self_jev_v2 import score_item_logits_native
                        lg = score_item_logits_native(
                            mdl,
                            self.tokenizer,
                            item,
                            self.device,
                            max_length=self.max_length,
                            option_first=self.option_first,
                        ).float()
                    else:
                        lg = self._score_chunked(mdl, self.tokenizer, item, self.device).float()
                logits_sum = lg if logits_sum is None else (logits_sum + lg)
            avg_logits = logits_sum / float(len(self.models))
            probs = self._torch.softmax(avg_logits / max(self.temperature, 1e-4), dim=-1).detach().cpu().tolist()
        lat_ms = (time.perf_counter() - t0) * 1000.0
        pred_idx = int(max(range(len(probs)), key=lambda i: probs[i]))
        return {
            "choice_index": pred_idx,
            "probabilities": [float(p) for p in probs],
            "confidence": float(probs[pred_idx]),
            "parse_ok": True,
            "latency_ms": round(lat_ms, 3),
            "input_tokens": len(context.split()) + len(question.split()),
            "output_tokens": 0,
            "raw_output": None,
        }


def _score_mock(model_name: str, item: dict[str, Any], order: str, rng: random.Random) -> dict[str, Any]:
    n = len(item["options"])
    gold = int(item["gold_index"])
    gold_in_order = (n - 1 - gold) if order == "swapped" else gold
    is_long = item.get("context_band") == "long"
    profiles = {
        "kev-0.8b": (0.84 if not is_long else 0.78, 0.99, 0.04, 145.0),
        "qwen-0.8b-ll": (0.64 if not is_long else 0.56, 1.00, 0.16, 190.0),
        "qwen-0.8b-json": (0.58 if not is_long else 0.45, 0.86, 0.24, 620.0),
        "frontier-llm": (0.91 if not is_long else 0.88, 0.99, 0.08, 1180.0),
    }
    base_acc, parse_rate, overconf, lat_base = profiles.get(model_name, (0.7, 0.95, 0.12, 300.0))
    if rng.random() >= parse_rate:
        return {
            "choice_index": -1,
            "probabilities": [0.0] * n,
            "confidence": 0.0,
            "parse_ok": False,
            "latency_ms": round(lat_base * rng.uniform(0.8, 1.4), 2),
            "input_tokens": int(item.get("est_tokens", 200)),
            "output_tokens": 24,
            "raw_output": "INVALID_JSON_MOCK",
        }
    if rng.random() < base_acc:
        pred = gold_in_order
        conf = min(0.99, max(0.35, rng.gauss(0.84, 0.11)))
    else:
        pred = rng.choice([i for i in range(n) if i != gold_in_order])
        conf = min(0.98, max(0.25, rng.gauss(0.55 + overconf, 0.14)))
    rem = (1.0 - conf) / max(1, n - 1)
    probs = [conf if i == pred else rem for i in range(n)]
    return {
        "choice_index": pred,
        "probabilities": [round(p, 6) for p in probs],
        "confidence": round(conf, 6),
        "parse_ok": True,
        "latency_ms": round(lat_base * rng.uniform(0.75, 1.35), 2),
        "input_tokens": int(item.get("est_tokens", 200)),
        "output_tokens": 18 if ("json" in model_name or "frontier" in model_name) else 0,
        "raw_output": None,
    }


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def _init_live_runners(args: argparse.Namespace, models: list[str]) -> dict[str, Any]:
    import serve_decision_api as api_mod

    device = api_mod._pick_device(args.device)
    runners: dict[str, Any] = {}
    if "kev-0.8b" in models:
        print(f"Loading kev-0.8b ({args.kev_run}) on {device}...")
        runners["kev-0.8b"] = api_mod.KevRunner(args.kev_run, device)
    if "qwen-0.8b-ll" in models:
        print(f"Loading qwen-0.8b-ll ({args.qwen_base_model}) on {device}...")
        runners["qwen-0.8b-ll"] = api_mod.QwenRunner(args.qwen_base_model, device)
    if "qwen-0.8b-json" in models:
        q_json_id = args.qwen_instruct_model or args.qwen_base_model
        if q_json_id == args.qwen_base_model and "qwen-0.8b-ll" in runners:
            runners["qwen-0.8b-json"] = runners["qwen-0.8b-ll"]
        else:
            print(f"Loading qwen-0.8b-json ({q_json_id}) on {device}...")
            runners["qwen-0.8b-json"] = api_mod.QwenRunner(q_json_id, device)
    if "frontier-llm" in models:
        runners["frontier_fn"] = api_mod.call_frontier_llm
    return runners


def main() -> None:
    parser = argparse.ArgumentParser(description="Run comprehensive offline benchmark across models & suites.")
    parser.add_argument("--test-file", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--calibrate-file", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--models", default="kev-0.8b,qwen-0.8b-ll,qwen-0.8b-json")
    parser.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"])
    parser.add_argument("--kev-run", default="jaredpalmer/kev-0.8b")
    parser.add_argument("--qwen-base-model", default="Qwen/Qwen3.5-0.8B-Base")
    parser.add_argument("--qwen-instruct-model", default="", help="Optional Qwen/Qwen3.5-0.8B for JSON baseline.")
    parser.add_argument("--frontier-config", type=Path, default=None, help="Path to frontier_config.json (optional).")
    parser.add_argument("--nn-npz", type=Path, default=None, help="Optional Tier 4 car_build_nn.npz.")
    parser.add_argument("--minilm-dir", type=Path, default=None, help="Optional Tier 3 self-jev cross-encoder model directory.")
    parser.add_argument("--minilm-name", default="self-jev-minilm", help="Model name label for --minilm-dir.")
    parser.add_argument("--suites", default="", help="Optional comma-separated suite filter.")
    parser.add_argument("--limit", type=int, default=0, help="Optional cap on items per split (0 = all).")
    parser.add_argument("--swap-order", action="store_true", help="Also score reversed option order.")
    parser.add_argument("--resume", action="store_true", help="Resume from existing benchmark_raw.jsonl.")
    parser.add_argument("--mock", action="store_true", help="Run deterministic mock scorer.")
    parser.add_argument("--seed", type=int, default=20260401)
    args = parser.parse_args()

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "benchmark_raw.jsonl"
    if not args.resume and raw_path.exists():
        raw_path.unlink()

    existing_keys: set[tuple[str, str, str]] = set()
    all_records: list[dict[str, Any]] = []
    if args.resume and raw_path.is_file():
        for ln in raw_path.read_text(encoding="utf-8").splitlines():
            if ln.strip():
                r = json.loads(ln)
                existing_keys.add((r["id"], r["order"], r["model"]))
                all_records.append(r)

    frontier_cfg: dict[str, Any] | None = None
    in_price, out_price = 0.15, 0.60
    if args.frontier_config and args.frontier_config.is_file():
        loaded = json.loads(args.frontier_config.read_text(encoding="utf-8"))
        token_val = str(loaded.get("api_key", "")).strip()
        if token_val and not token_val.startswith("PASTE_"):
            frontier_cfg = loaded
            in_price = float(loaded.get("in_price_per_1m", 0.15))
            out_price = float(loaded.get("out_price_per_1m", 0.60))

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if (frontier_cfg or args.mock) and "frontier-llm" not in models:
        models.append("frontier-llm")

    nn_adapter = None
    if args.nn_npz and args.nn_npz.is_file():
        try:
            nn_adapter = OptionalNNAdapter(args.nn_npz)
            models.append("car-build-nn")
        except Exception as exc:
            print(f"Warning: could not load Tier 4 NN ({exc}); skipping car-build-nn.")

    minilm_adapter = None
    minilm_name = args.minilm_name.strip() or "self-jev-minilm"
    if args.minilm_dir and args.minilm_dir.is_dir():
        try:
            minilm_adapter = MiniLMDecisionAdapter(args.minilm_dir)
            models.append(minilm_name)
        except Exception as exc:
            print(f"Warning: could not load Tier 3 cross-encoder ({exc}); skipping {minilm_name}.")

    suite_filter = {s.strip() for s in args.suites.split(",") if s.strip()}
    test_items = [it for it in _load_jsonl(args.test_file) if (not suite_filter or it["suite"] in suite_filter)]
    if args.limit > 0:
        test_items = test_items[: args.limit]

    cal_items: list[dict[str, Any]] = []
    if args.calibrate_file and args.calibrate_file.is_file():
        cal_items = [it for it in _load_jsonl(args.calibrate_file) if (not suite_filter or it["suite"] in suite_filter)]
        if args.limit > 0:
            cal_items = cal_items[: args.limit]

    orders = ["orig", "swapped"] if args.swap_order else ["orig"]
    items = cal_items + test_items
    pending_models = {
        m
        for item in items
        for order in orders
        for m in models
        if (item["id"], order, m) not in existing_keys and m not in ("car-build-nn", minilm_name)
    }
    runners: dict[str, Any] = {} if (args.mock or not pending_models) else _init_live_runners(args, sorted(pending_models))
    print(f"Running benchmark | models={models} | pending_live={sorted(pending_models)} | test={len(test_items)} | cal={len(cal_items)} | orders={orders}")

    rng = random.Random(args.seed)
    total_jobs = len(items) * len(orders) * len(models)
    done = 0
    with raw_path.open("a", encoding="utf-8") as fout:
        for item in items:
            opts_orig = list(item["options"])
            descs_orig = list(item.get("option_descriptions") or opts_orig)
            n_opts = len(opts_orig)

            for order in orders:
                opts = list(reversed(opts_orig)) if order == "swapped" else opts_orig
                descs = list(reversed(descs_orig)) if order == "swapped" else descs_orig

                for model in models:
                    done += 1
                    key = (item["id"], order, model)
                    if key in existing_keys:
                        continue

                    if model == "car-build-nn":
                        if nn_adapter is None or not item.get("features") or order != "orig":
                            continue
                        raw = nn_adapter.score_features(item["features"])
                    elif model == minilm_name:
                        if minilm_adapter is None:
                            continue
                        raw = minilm_adapter.score_text(item["context"], item["question"], opts, descs)
                    elif args.mock:
                        raw = _score_mock(model, item, order, rng)
                    elif model == "kev-0.8b":
                        raw = _normalize_response(
                            runners["kev-0.8b"].score_unified(item["context"], item["question"], opts, descs),
                            n_opts,
                        )
                    elif model == "qwen-0.8b-ll":
                        raw = _normalize_response(
                            runners["qwen-0.8b-ll"].score_loglik(item["context"], item["question"], opts, descs),
                            n_opts,
                        )
                    elif model == "qwen-0.8b-json":
                        raw = _normalize_response(
                            runners["qwen-0.8b-json"].score_json(item["context"], item["question"], opts, descs),
                            n_opts,
                        )
                    elif model == "frontier-llm":
                        assert frontier_cfg is not None
                        raw = _normalize_response(
                            runners["frontier_fn"](
                                frontier_cfg["base_url"],
                                frontier_cfg["model"],
                                frontier_cfg["api_key"],
                                item["context"],
                                item["question"],
                                opts,
                                descs,
                            ),
                            n_opts,
                        )
                    else:
                        raise ValueError(f"Unsupported model: {model}")

                    pred_idx = int(raw["choice_index"])
                    probs = list(raw["probabilities"])
                    if order == "swapped":
                        if 0 <= pred_idx < n_opts:
                            pred_idx = n_opts - 1 - pred_idx
                        probs = list(reversed(probs))

                    pred_label = opts_orig[pred_idx] if 0 <= pred_idx < n_opts else None
                    gold_idx = int(item["gold_index"])
                    rec = {
                        "id": item["id"],
                        "split": item["split"],
                        "suite": item["suite"],
                        "bucket": item["bucket"],
                        "context_band": item.get("context_band", "short"),
                        "est_tokens": item.get("est_tokens", 0),
                        "base_id": item.get("base_id"),
                        "order": order,
                        "model": model,
                        "gold_index": gold_idx,
                        "gold_label": item["gold_label"],
                        "abstain_index": item.get("abstain_index"),
                        "is_abstain_gold": bool(item.get("is_abstain_gold", False)),
                        "hidden_oracle_abstain": bool(item.get("hidden_oracle_abstain", False)),
                        "oracle_scores": item.get("oracle_scores"),
                        "pred_index": pred_idx,
                        "pred_label": pred_label,
                        "correct": (pred_idx == gold_idx),
                        "parse_ok": bool(raw["parse_ok"]),
                        "confidence": float(raw["confidence"]),
                        "probabilities": probs,
                        "latency_ms": float(raw["latency_ms"]),
                        "input_tokens": int(raw.get("input_tokens", 0)),
                        "output_tokens": int(raw.get("output_tokens", 0)),
                        "raw_output": raw.get("raw_output"),
                    }
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    fout.flush()
                    existing_keys.add(key)
                    all_records.append(rec)
                    if done % 100 == 0:
                        print(f"  progress: {done}/{total_jobs} evaluations processed...")

    config = {
        "test_file": str(args.test_file),
        "calibrate_file": str(args.calibrate_file) if args.calibrate_file else None,
        "models": models,
        "frontier_model": frontier_cfg.get("model") if frontier_cfg else ("mock-frontier" if args.mock else None),
        "swap_order": args.swap_order,
        "mock": args.mock,
    }
    summary = build_summary(all_records, models, config, frontier_in_price=in_price, frontier_out_price=out_price)
    json_path, csv_path = write_summary_files(summary, out_dir)
    print(f"\nSaved benchmark summary:\n  JSON: {json_path}\n  CSV : {csv_path}")
    for m, mdata in summary["models"].items():
        ov = mdata["overall"]
        print(
            f"  {m:16s} | n={ov.get('n',0):3d} | acc={ov.get('accuracy',0)*100:5.1f}% "
            f"| ECE={ov.get('ece',0):.3f} | AURC={ov.get('aurc',0):.3f} "
            f"| parse={ov.get('parse_ok_rate',0)*100:5.1f}% | p50={ov.get('latency_p50_ms',0):6.1f}ms"
        )


if __name__ == "__main__":
    main()
