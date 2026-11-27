
# Local Decision-Model API & Offline Benchmark Guide (M4 Mac 16 GB)

This guide covers setting up, pre-downloading, hosting, and benchmarking **`Kev-0.8B`** (`jaredpalmer/kev-0.8b`) and **`Qwen3.5-0.8B`** (`Base` + optional `Instruct`) on an **Apple Silicon M4 MacBook Air (16 GB unified memory)** for the project **Evaluating Decision-Oriented AI for Reliable Software Decisions**.

## 1. Scope & Division of Labor

- **This machine (M4 Mac 16 GB):** Hosts the local decision API (`scripts/serve_decision_api.py`), runs the 100% offline benchmark across `Kev-0.8B`, `Qwen3.5-0.8B` (Base log-likelihood + Instruct JSON), and the Frontier LLM API baseline, and provides a plug-and-play adapter slot for the project lead's trained task-specific model.
- **Hardware & environment notes:**
  - Use **Python 3.12 or 3.13** (`kev` has import issues on Python 3.14).
  - Install **`mlx`** so `Kev-0.8B` runs on Apple Silicon MLX (~45 ms warm on M4) instead of falling back to PyTorch MPS (~1.96 s). Verify `"backend": "mlx"` via `GET /health`.
  - Keep to `0.8B` models on 16 GB RAM (~4–5 GB combined in BF16). Do not load `Kev-4B` on 16 GB (requires 32 GB+).
  - Confirm `"state_truncated": false` in responses (`serve_decision_api.py` uses `max_state=SERVE_MAX_STATE`, `strict=True` to avoid the 384-token training default).
  - Run all local models on this same M4 Mac at batch size 1 after warm-up so latency comparisons stay apples-to-apples.

## 2. Setup, Model Download & Launch

### 2.1 Virtual Environment & Packages
```bash
python3.13 -m venv .venv-decision
source .venv-decision/bin/activate

pip install --upgrade pip
pip install -r scripts/requirements.txt
```

### 2.2 Pre-Download Both Models (`scripts/download_models.py`)
```bash
# Default: downloads jaredpalmer/kev-0.8b + Qwen/Qwen3.5-0.8B-Base and checks mlx/env
python scripts/download_models.py

# Optional: also download Qwen/Qwen3.5-0.8B Instruct for the structured-JSON baseline
python scripts/download_models.py --include-instruct

# Optional: pull Qwen weights from ModelScope or use a Hugging Face mirror
python scripts/download_models.py --qwen-source modelscope
# or:
python scripts/download_models.py --hf-endpoint https://hf-mirror.com
```

If `--qwen-source modelscope` is used, export `MODELSCOPE_BASE` using the path printed by the script before starting the server.

### 2.3 Start the API Server & Built-in Web UI (`scripts/serve_decision_api.py`)

```bash
# Default: loads Kev-0.8B (MLX/BF16) + Qwen3.5-0.8B-Base (MPS/BF16) and opens the browser UI
python scripts/serve_decision_api.py --port 8008 --open-browser

# With a separate Instruct checkpoint for the structured-JSON baseline
python scripts/serve_decision_api.py \
  --port 8008 \
  --open-browser \
  --kev-run jaredpalmer/kev-0.8b \
  --qwen-base-model Qwen/Qwen3.5-0.8B-Base \
  --qwen-instruct-model Qwen/Qwen3.5-0.8B
```

### 2.4 Using the Built-in Web UI (No Coding Needed)

Once the server finishes loading, open `http://127.0.0.1:8008` in Safari or Chrome (served from `scripts/decision_ui.html`):

- **Top status pills:** Confirm `kev-0.8b (mlx)` is green (if it shows yellow/MPS, install `mlx` in the same venv).
- **Tab 1 — Side-by-Side Demo (for Screenshots & Screen Recordings):** Click a preset (or pick a row from `data/car_build_bench/test.jsonl`), optionally check **Include Frontier LLM**, and click **Compare All Models** to view chosen answers, probability bars, confidence %, latency (ms), and raw JSON side-by-side. Click **Download This Result (JSON)** to save any demo case.
- **Tab 2 — Batch Benchmark & Export:** Select a local dataset or upload any `.jsonl` file, choose which models to run (`kev-0.8b`, `qwen-0.8b-ll`, `qwen-0.8b-json`, `frontier-llm`), keep **Run swapped option order** checked, and click **Run Full Benchmark**. When done, click **Download Full Report (.json)** or **Download Summary Table (.csv)** for Accuracy, Valid Output %, ECE, Order-Flip Rate, p50/p95 Latency, and Est. Cost / 1k Decisions.
- **Tab 3 — Frontier LLM Settings (Optional):** Enter any OpenAI-compatible Base URL (OpenAI, OpenRouter, Gemini), Model ID, API key, and $/1M token prices, then click **Test Frontier API**. Requests are proxied locally via `POST /v1/frontier`.

### 2.5 Optional: Packed Simulator (`scripts/simulator_bundle.zip`)

`scripts/simulator_bundle.zip` is a standalone archive of `self_jev/sim/`, `self_jev/train_rl_sim.py`, `design.json`, `models/jev_rl_v7.mapdata.npz`, and the three track `mapdata.json` files for evaluating self-trained models offline when needed.

### 2.6 Verify API Endpoints (CLI)

```bash
# 1. Check loaded models and confirm Kev backend is "mlx"
curl -s http://127.0.0.1:8008/health | python -m json.tool

# 2. Side-by-side comparison on one decision item
curl -s -X POST http://127.0.0.1:8008/v1/compare \
  -H "Content-Type: application/json" \
  -d '{
    "state": "Candidate A finished 4 laps cleanly with 0 damage. Candidate B retired on lap 2 from wall contact.",
    "question": "Which candidate should be promoted?",
    "options": ["Promote A", "Promote B", "Abstain / Unresolved"]
  }' | python -m json.tool

# 3. Unified scoring endpoint (model: kev-0.8b | qwen-0.8b-ll | qwen-0.8b-json)
curl -s -X POST http://127.0.0.1:8008/v1/score \
  -H "Content-Type: application/json" \
  -d '{
    "model": "kev-0.8b",
    "state": "No fair-pair finishes exist yet; all recent runs ended in damage retirements.",
    "question": "Should we promote a new version now?",
    "options": ["Promote", "Abstain / Unresolved"]
  }' | python -m json.tool
```

Native Kev multi-question requests are also exposed at `POST /v1/systemone`, and auto-discovered local datasets are listed at `GET /v1/datasets`.

3. Unified API Response Schema

Every POST /v1/score call returns the same structure across all models:

{
  "model": "kev-0.8b",
  "model_id": "jaredpalmer/kev-0.8b",
  "backend": "mlx",
  "choice_index": 0,
  "choice": "Promote A",
  "probabilities": [0.8421, 0.0612, 0.0967],
  "confidence": 0.8421,
  "parse_ok": true,
  "latency_ms": 46.8,
  "input_tokens": 118,
  "state_truncated": false,
  "raw_output": null
}


## 4. Comprehensive Multi-Suite Offline Benchmark Pipeline

The full evaluation pipeline lives in `scripts/` and covers **313 held-out test items** + **106 calibrate items** across four suites:

- **Suite 1 (`suite1_software_ops`)**: 120 test + 40 calibrate general software-operations decisions (incident triage, CI/CD release gating, API/support routing, and ambiguous/out-of-scope abstain traps). Zero racing bias.
- **Suite 2 (`suite2_historical_eng`)**: 69 test + 47 calibrate real historical `agx.cool` match comparisons (`data/outcome/examples.jsonl`) plus 18 long-context driver-lever questions from `self_jev/kev_harness.py` (87 test items total).
- **Suite 3 (`suite3_car_build`)**: 34 test + 19 calibrate simulator-backed car-build selection items (`data/car_build_bench/`) with `oracle_scores`, tabular `features` for Tier 4 NN, and clean-subset reporting.
- **Suite 4 (`suite4_stress_test`)**: 72 robustness items testing context-length scaling (`short` ~250 tok, `medium` ~750 tok, `long` ~2,000 tok) and prompt/label paraphrase stability.

### 4.1 Workflow Commands

```bash
# 1. Build (or rebuild) frozen benchmark splits under data/benchmarks/
python scripts/build_benchmarks.py

# 2. Configure Frontier LLM API when ready (copy template and fill in base_url, model, api_key)
cp scripts/frontier_config.example.json scripts/frontier_config.json

# 3. Run full offline evaluation (supports --swap-order, --resume, --nn-npz, --frontier-config)
python scripts/evaluate_models.py \
  --test-file data/benchmarks/all_test.jsonl \
  --calibrate-file data/benchmarks/all_calibrate.jsonl \
  --qwen-instruct-model Qwen/Qwen3.5-0.8B \
  --frontier-config scripts/frontier_config.json \
  --nn-npz models/car_build_nn.npz \
  --swap-order \
  --resume

# 4. Generate 6 publication-ready SVG charts + index.html gallery
python scripts/generate_charts.py
```

### 4.2 Metrics & Charts Produced

- `reports/full_benchmark/benchmark_summary.json` & `.csv`: Accuracy (with 95% bootstrap CIs), Macro-F1, `parse_ok` rate, 10-bin ECE, Brier score, NLL, Error-Detection AUROC, Risk–Coverage AURC, False-Accept & False-Abstain rates, Weighted Asymmetric Error Cost (5× false-accept), Mean Oracle Regret, Option-Order Flip Rate, Paraphrase Flip Rate, Context-Length accuracy breakdown, p50/p95/p99 latency, Cost per 1,000 decisions, pairwise McNemar significance tests, Confidence-Gated Cascade Pareto sweeps (`0% -> 100%` escalation to Frontier LLM), and top contrast cases for Web UI screenshots.
- `reports/full_benchmark/charts/index.html`: 6 vector SVG charts (Cascade Pareto Curve, Calibration Reliability Diagram, Risk–Coverage Curve, Suite Accuracy Comparison, Context-Length Degradation, and Latency Comparison).

## 5. Verification Checklist

1. `python scripts/download_models.py` finishes with `mlx: OK` and both models cached.
2. `GET /health` reports `kev-0.8b` (`backend: "mlx"`) and `qwen-0.8b-ll` loaded cleanly.
3. `python scripts/build_benchmarks.py` generates `data/benchmarks/all_test.jsonl` (313 items) and `all_calibrate.jsonl` (106 items).
4. `python scripts/evaluate_models.py --mock --swap-order` completes end-to-end and `python scripts/generate_charts.py` renders all 6 SVG charts cleanly.
5. `scripts/frontier_config.json` and `--nn-npz models/car_build_nn.npz` plug in without modifying any code.