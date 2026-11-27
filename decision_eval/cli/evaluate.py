"""CLI entrypoint: Run comprehensive offline benchmark across models & suites."""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from decision_eval import paths
from decision_eval.metrics import build_summary, write_summary_files
from decision_eval.runners import KevRunner, OptionalNNAdapter, call_frontier_llm, create_qwen_runner, pick_device


def _normalize(res: dict[str, Any], n_opts: int) -> dict[str, Any]:
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
            "choice_index": -1, "probabilities": [0.0] * n, "confidence": 0.0, "parse_ok": False,
            "latency_ms": round(lat_base * rng.uniform(0.8, 1.4), 2),
            "input_tokens": int(item.get("est_tokens", 200)), "output_tokens": 24, "raw_output": "INVALID_JSON_MOCK",
        }
    if rng.random() < base_acc:
        pred = gold_in_order
        conf = min(0.99, max(0.35, rng.gauss(0.84, 0.11)))
    else:
        pred = rng.choice([i for i in range(n) if i != gold_in_order])
        conf = min(0.98, max(0.25, rng.gauss(0.55 + overconf, 0.14)))
    rem = (1.0 - conf) / max(1, n - 1)
    return {
        "choice_index": pred,
        "probabilities": [round(conf if i == pred else rem, 6) for i in range(n)],
        "confidence": round(conf, 6),
        "parse_ok": True,
        "latency_ms": round(lat_base * rng.uniform(0.75, 1.35), 2),
        "input_tokens": int(item.get("est_tokens", 200)),
        "output_tokens": 18 if ("json" in model_name or "frontier" in model_name) else 0,
        "raw_output": None,
    }


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def main() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.5")
    os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.0")
    bundled_cache = paths.configure_bundled_hf_cache()
    if bundled_cache:
        print(f"Using bundled model cache: {bundled_cache}")
    parser = argparse.ArgumentParser(description="Run comprehensive offline benchmark across models & suites.")
    parser.add_argument("--test-file", type=Path, default=paths.BENCHMARKS_DIR / "all_test.jsonl")
    parser.add_argument("--calibrate-file", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=paths.REPORTS_DIR)
    parser.add_argument("--models", default="kev-0.8b,qwen-0.8b-ll,qwen-0.8b-json")
    parser.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"])
    parser.add_argument("--backend", default="auto", choices=["auto", "mlx", "torch"], help="Force same backend (mlx or torch) across both Kev and Qwen.")
    parser.add_argument("--kev-run", default="jaredpalmer/kev-0.8b")
    parser.add_argument("--qwen-base-model", default="Qwen/Qwen3.5-0.8B-Base")
    parser.add_argument("--qwen-instruct-model", default="")
    parser.add_argument("--frontier-config", type=Path, default=None, help="Path to decision_eval/config/frontier.json")
    parser.add_argument("--nn-npz", type=Path, default=None, help="Optional Tier 4 models/car_build_nn.npz")
    parser.add_argument("--suites", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--swap-order", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--mock", action="store_true")
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
            models.append("nn-combo")
        except Exception as exc:
            print(f"Warning: could not load Tier 4 NN ({exc}); skipping.")

    suite_filter = {s.strip() for s in args.suites.split(",") if s.strip()}
    test_items = [it for it in _load_jsonl(args.test_file) if (not suite_filter or it["suite"] in suite_filter)]
    if args.limit > 0:
        test_items = test_items[: args.limit]

    cal_items: list[dict[str, Any]] = []
    if args.calibrate_file and args.calibrate_file.is_file():
        cal_items = [it for it in _load_jsonl(args.calibrate_file) if (not suite_filter or it["suite"] in suite_filter)]
        if args.limit > 0:
            cal_items = cal_items[: args.limit]

    runners: dict[str, Any] = {}
    if not args.mock:
        device = pick_device(args.device)
        if "kev-0.8b" in models:
            runners["kev-0.8b"] = KevRunner(args.kev_run, device, backend=args.backend)
        if "qwen-0.8b-ll" in models:
            runners["qwen-0.8b-ll"] = create_qwen_runner(args.qwen_base_model, device, backend=args.backend)
        if "qwen-0.8b-json" in models:
            q_json_id = args.qwen_instruct_model or args.qwen_base_model
            runners["qwen-0.8b-json"] = (
                runners["qwen-0.8b-ll"]
                if q_json_id == args.qwen_base_model and "qwen-0.8b-ll" in runners
                else create_qwen_runner(q_json_id, device, backend=args.backend)
            )

    orders = ["orig", "swapped"] if args.swap_order else ["orig"]
    items = cal_items + test_items
    print(f"Running benchmark | models={models} | test={len(test_items)} | cal={len(cal_items)} | orders={orders}")

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

                    if model in ("nn-combo", "car-build-nn"):
                        if nn_adapter is None or not item.get("features") or order != "orig":
                            continue
                        raw = nn_adapter.score_features(item["features"])
                    elif args.mock:
                        raw = _score_mock(model, item, order, rng)
                    elif model == "kev-0.8b":
                        raw = _normalize(runners["kev-0.8b"].score_unified(item["context"], item["question"], opts, descs), n_opts)
                    elif model == "qwen-0.8b-ll":
                        raw = _normalize(runners["qwen-0.8b-ll"].score_loglik(item["context"], item["question"], opts, descs), n_opts)
                    elif model == "qwen-0.8b-json":
                        raw = _normalize(runners["qwen-0.8b-json"].score_json(item["context"], item["question"], opts, descs), n_opts)
                    elif model == "frontier-llm":
                        assert frontier_cfg is not None
                        raw = _normalize(
                            call_frontier_llm(
                                frontier_cfg["base_url"], frontier_cfg["model"], frontier_cfg["api_key"],
                                item["context"], item["question"], opts, descs,
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

                    gold_idx = int(item["gold_index"])
                    rec = {
                        "id": item["id"], "split": item["split"], "suite": item["suite"], "bucket": item["bucket"],
                        "context_band": item.get("context_band", "short"), "est_tokens": item.get("est_tokens", 0),
                        "base_id": item.get("base_id"), "order": order, "model": model,
                        "gold_index": gold_idx, "gold_label": item["gold_label"],
                        "abstain_index": item.get("abstain_index"), "is_abstain_gold": bool(item.get("is_abstain_gold", False)),
                        "hidden_oracle_abstain": bool(item.get("hidden_oracle_abstain", False)),
                        "oracle_scores": item.get("oracle_scores"),
                        "pred_index": pred_idx, "pred_label": opts_orig[pred_idx] if 0 <= pred_idx < n_opts else None,
                        "correct": (pred_idx == gold_idx), "parse_ok": bool(raw["parse_ok"]),
                        "confidence": float(raw["confidence"]), "probabilities": probs,
                        "latency_ms": float(raw["latency_ms"]), "input_tokens": int(raw.get("input_tokens", 0)),
                        "output_tokens": int(raw.get("output_tokens", 0)), "raw_output": raw.get("raw_output"),
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
