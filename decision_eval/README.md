
# Local Decision-Model API & Offline Benchmark Guide (M4 Mac 16 GB)

This guide covers setting up, pre-downloading, hosting, and benchmarking **`Kev-0.8B`** (`jaredpalmer/kev-0.8b`) and **`Qwen3.5-0.8B`** (`Base` + optional `Instruct`) on an **Apple Silicon M4 MacBook Air (16 GB unified memory)** for the project **Evaluating Decision-Oriented AI for Reliable Software Decisions**.

## 1. Scope & Division of Labor

- **This machine (M4 Mac 16 GB):** Hosts the local decision API (`decision_eval/cli/serve.py`), runs the 100% offline benchmark across `Kev-0.8B`, `Qwen3.5-0.8B` (Base log-likelihood + Instruct JSON), and the Frontier LLM API baseline, and provides a plug-and-play adapter slot for the project lead's trained task-specific model.
- **Project layout (`decision_eval/`):**
  - `cli/` — runnable entrypoints (`quickstart.py`, `download_models.py`, `serve.py`, `build_benchmarks.py`, `evaluate.py`, `analyze_raw.py`, `generate_charts.py`)
  - `src/decision_eval/` — core modules (`paths.py`, `runners.py`, `server.py`, `benchmarks.py`, `metrics.py`, `charts.py`)
  - `vendor/kev/` — locally bundled `kev` package (no GitHub clone required)
  - `ui/index.html` — 4-tab Evaluation Studio Web UI
  - `config/frontier.example.json` — Frontier LLM API config template
  - `docs/` — `project_outline.md`, `full_report_guide.md`, and `interim_presentation_plan.md`
  - `bundles/simulator_bundle.zip` — optional standalone racing simulator archive
- **Hardware & environment notes:**
  - Use **Python 3.12 or 3.13** (`kev` has import issues on Python 3.14).
  - Both `mlx` and `mlx-lm` are included in `requirements.txt` so **both `Kev-0.8B` and `Qwen3.5-0.8B` run on Apple Silicon MLX** (`--backend mlx` or `--backend auto`) for fair latency comparisons. Verify `"backend": "mlx"` via `GET /health`.
  - Keep to `0.8B` models on 16 GB RAM (~4–5 GB combined in BF16). Do not load `Kev-4B` on 16 GB (requires 32 GB+).
  - Confirm `"state_truncated": false` in responses (`runners.py` uses `max_state=SERVE_MAX_STATE`, `strict=True` to avoid the 384-token training default).
  - Run all local models on this same M4 Mac at batch size 1 after warm-up so latency comparisons stay apples-to-apples.

## 2. Setup, Model Download & Launch

### 2.1 One-Time Mac Setup (`brew` + `uv`, Recommended)

Using **`uv`** automatically installs Python 3.13 for you (avoiding Python 3.14 issues with `kev`) and installs packages much faster than plain `pip`:

```bash
# 1. Install Xcode Command Line Tools (provides git; skip if already installed)
xcode-select --install

# 2. Install Homebrew (skip if `brew --version` already works)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# 3. Install uv via Homebrew
brew install uv

# 4. Create a Python 3.13 venv inside the unzipped project folder & install dependencies
uv venv --python 3.13 .venv-decision
source .venv-decision/bin/activate
uv pip install -r decision_eval/requirements.txt
```

*(Alternative without `uv`: if you already have `python3.13`, run `python3.13 -m venv .venv-decision && source .venv-decision/bin/activate && pip install -r decision_eval/requirements.txt`.)*

### 2.1b Fastest Option: One-Command Quickstart (`decision_eval/cli/quickstart.py`)

Once `brew` and `uv` are installed, you can run everything in a single command from the unzipped folder:

```bash
python3 decision_eval/cli/quickstart.py
```

This automatically creates `.venv-decision` (Python 3.13), installs `decision_eval/requirements.txt` (including `mlx`, `mlx-lm`, `modelscope`, and locally bundled `kev`), downloads the models from **ModelScope** into `models/modelscope/`, starts the API server on port `8008`, and opens the Web UI in your browser.

- Optional faster PyPI mirror: `UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple python3 decision_eval/cli/quickstart.py`
- Optional separate Instruct model: `python3 decision_eval/cli/quickstart.py --include-instruct`
- Optional Hugging Face source instead of ModelScope: `python3 decision_eval/cli/quickstart.py --source hf`

### 2.2 Manual Model Download (`decision_eval/cli/download_models.py`)

Models default to **ModelScope** (`https://www.modelscope.cn`) and are auto-detected from `models/modelscope/` (no manual env vars required), with automatic Hugging Face fallback:

- `jaredpalmer/kev-0.8b` (`https://www.modelscope.cn/models/jaredpalmer/kev-0.8b`)
- `Qwen/Qwen3.5-0.8B-Base` (`https://www.modelscope.cn/models/Qwen/Qwen3.5-0.8B-Base`)
- `Qwen/Qwen3.5-0.8B` (`https://www.modelscope.cn/models/Qwen/Qwen3.5-0.8B`, with `--include-instruct`)

```bash
# Default (ModelScope): downloads Kev-0.8B + Qwen3.5-0.8B-Base into models/modelscope/
python decision_eval/cli/download_models.py

# Optional: also download Qwen/Qwen3.5-0.8B Instruct for the structured-JSON baseline
python decision_eval/cli/download_models.py --include-instruct

# Optional: pull from Hugging Face Hub instead of ModelScope
python decision_eval/cli/download_models.py --source hf
```

### 2.3 Start the API Server & Built-in Web UI (`decision_eval/cli/serve.py`)

```bash
# Default: loads Kev-0.8B + Qwen3.5-0.8B-Base on MLX (auto) and opens the browser UI
python decision_eval/cli/serve.py --port 8008 --backend auto --open-browser

# Force the exact same backend across both models (mlx or torch)
python decision_eval/cli/serve.py --port 8008 --backend mlx --open-browser

# With a separate Instruct checkpoint for the structured-JSON baseline
python decision_eval/cli/serve.py \
  --port 8008 \
  --backend mlx \
  --open-browser \
  --kev-run jaredpalmer/kev-0.8b \
  --qwen-base-model Qwen/Qwen3.5-0.8B-Base \
  --qwen-instruct-model Qwen/Qwen3.5-0.8B
```

### 2.4 Using the Built-in Web UI (No Coding Needed)

Once the server finishes loading, open `http://127.0.0.1:8008` in Safari or Chrome (served from `decision_eval/ui/index.html`):

- **Top status pills:** Confirm `kev-0.8b (mlx)` is green (and `car-build-nn` if `models/car_build_nn.npz` is present).
- **Tab 1 — Side-by-Side Demo:** Pick a preset (Software Ops Incident Triage, Driver-Lever Abstain, Car-Build Dry/Trap/Unseen Rule) or any row from the 4-suite datasets, optionally include Frontier LLM, and click **Compare All Models** to inspect chosen answers, probability bars, confidence %, latency (ms), and raw JSON.
- **Tab 2 — Comprehensive Multi-Suite Benchmark:** Defaults to `all_test.jsonl` (313 items across 4 suites) with swapped option order and optional Tier 4 `car-build-nn`. Computes full publication metrics (95% CI, Macro-F1, ECE, Brier, AUROC, AURC, False-Accept, 5× Cost, Order/Paraphrase Flip, McNemar tests, Cascade Pareto, and Top Contrast Cases with one-click **Open in Demo Tab**), or click **Load Latest Saved Report** to view CLI runs.
- **Tab 3 — Charts & Visuals (6 SVGs):** View and download all 6 vector SVG charts generated automatically after any Web UI or CLI benchmark run.
- **Tab 4 — Frontier LLM API Settings (Optional):** Leave blank to run local `0.8B` models first, or paste any OpenAI-compatible Base URL, Model ID, API key, and $/1M token pricing whenever you're ready.

### 2.5 Optional: Packed Simulator (`decision_eval/bundles/simulator_bundle.zip`)

`decision_eval/bundles/simulator_bundle.zip` is a standalone archive of `self_jev/sim/`, `self_jev/train_rl_sim.py`, `design.json`, `models/jev_rl_v7.mapdata.npz`, and the three track `mapdata.json` files for evaluating self-trained models offline when needed.

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

## 3. Unified API Response Schema

Every `POST /v1/score` call returns the same structure across all models:

```json
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
```


## 4. Comprehensive Multi-Suite Offline Benchmark Pipeline

The full evaluation pipeline lives in `decision_eval/` and covers **313 held-out test items** + **106 calibrate items** across four suites:

- **Suite 1 (`suite1_software_ops`)**: 120 test + 40 calibrate general software-operations decisions (incident triage, CI/CD release gating, API/support routing, and ambiguous/out-of-scope abstain traps). Zero racing bias.
- **Suite 2 (`suite2_historical_eng`)**: 69 test + 47 calibrate real historical `agx.cool` match comparisons (`data/outcome/examples.jsonl`) plus 18 long-context driver-lever questions from `self_jev/kev_harness.py` (87 test items total).
- **Suite 3 (`suite3_car_build`)**: 34 test + 19 calibrate simulator-backed car-build selection items (`data/car_build_bench/`) with `oracle_scores`, tabular `features` for Tier 4 NN, and clean-subset reporting.
- **Suite 4 (`suite4_stress_test`)**: 72 robustness items testing context-length scaling (`short` ~250 tok, `medium` ~750 tok, `long` ~2,000 tok) and prompt/label paraphrase stability.

### 4.1 Workflow Commands

```bash
# 1. Build (or rebuild) frozen benchmark splits under data/benchmarks/
python decision_eval/cli/build_benchmarks.py

# 2. Configure Frontier LLM API when ready (copy template and fill in base_url, model, api_key)
cp decision_eval/config/frontier.example.json decision_eval/config/frontier.json

# 3. Run full offline evaluation (supports --swap-order, --resume, --nn-npz, --frontier-config)
python decision_eval/cli/evaluate.py \
  --test-file data/benchmarks/all_test.jsonl \
  --calibrate-file data/benchmarks/all_calibrate.jsonl \
  --qwen-instruct-model Qwen/Qwen3.5-0.8B \
  --frontier-config decision_eval/config/frontier.json \
  --nn-npz models/car_build_nn.npz \
  --swap-order \
  --resume

# 4. Generate 6 publication-ready SVG charts + index.html gallery
python decision_eval/cli/generate_charts.py
```

### 4.2 Metrics & Charts Produced

- `reports/full_benchmark/benchmark_summary.json` & `.csv`: Accuracy (with 95% bootstrap CIs), Macro-F1, `parse_ok` rate, 10-bin ECE, Brier score, NLL, Error-Detection AUROC, Risk–Coverage AURC, False-Accept & False-Abstain rates, Weighted Asymmetric Error Cost (5× false-accept), Mean Oracle Regret, Option-Order Flip Rate, Paraphrase Flip Rate, Context-Length accuracy breakdown, p50/p95/p99 latency, Cost per 1,000 decisions, pairwise McNemar significance tests, Confidence-Gated Cascade Pareto sweeps (`0% -> 100%` escalation to Frontier LLM), and top contrast cases for Web UI screenshots.
- `reports/full_benchmark/charts/index.html`: 6 vector SVG charts (Cascade Pareto Curve, Calibration Reliability Diagram, Risk–Coverage Curve, Suite Accuracy Comparison, Context-Length Degradation, and Latency Comparison).

## 5. Verification Checklist

1. `python decision_eval/cli/download_models.py` finishes with `mlx: OK` and both models cached.
2. `GET /health` reports `kev-0.8b` (`backend: "mlx"`) and `qwen-0.8b-ll` loaded cleanly.
3. `python decision_eval/cli/build_benchmarks.py` generates `data/benchmarks/all_test.jsonl` (313 items) and `all_calibrate.jsonl` (106 items).
4. `python decision_eval/cli/evaluate.py --mock --swap-order` completes end-to-end and `python decision_eval/cli/generate_charts.py` renders all 6 SVG charts cleanly.
5. `decision_eval/config/frontier.json` and `--nn-npz models/car_build_nn.npz` plug in without modifying any code.