# Full Group Project Report Guide & Template

**Project Title:** Evaluating Decision-Oriented AI for Reliable Software Decisions  
**Target Artifact:** Final Written Group Report  
**Evaluation Pipeline:** `scripts/build_benchmarks.py` → `scripts/evaluate_models.py` → `scripts/generate_charts.py` (or via Web UI `scripts/decision_ui.html`)

---

## Core Thesis

> **Can a lightweight, decision-oriented 0.8B model (`Kev-0.8B`)—and a confidence-gated cascade into a Frontier LLM—deliver more reliable, better-calibrated, and cheaper bounded software decisions than conventional generative LLMs?**

Anchor every section around three pillars:
1. **Format & Calibration Reliability:** Generative JSON prompting on small LLMs (`qwen-0.8b-json`) suffers from schema/parse failures, option-order sensitivity, and overconfident errors, whereas decision-oriented option scoring (`kev-0.8b`) produces structured probabilities by construction.
2. **Breadth Beyond a Single Toy Task:** Evaluating across **419 frozen items** (`313` held-out test + `106` calibrate) spanning general software operations, real historical engineering data, simulator-backed car builds, and robustness stress tests demonstrates generality.
3. **Practical Systems Economics:** A **confidence-gated cascade** (`Kev-0.8B -> Frontier LLM`, threshold fitted strictly on `calibrate`) trades a small escalation budget for near-Frontier accuracy at a fraction of the API cost and latency.

---

## Section-by-Section Report Blueprint

### 1. Introduction & Motivation
- Explain why bounded software decisions (incident triage, CI/CD release gating, ticket routing, safe abstention) fail under naive generative prompting: invalid JSON, uncalibrated verbalized confidence, high API cost, and tail latency.
- State three concrete research questions:
  - **RQ1 (Accuracy & Safety):** How does a decision-oriented `0.8B` model compare against matched-size generative baselines (`qwen-0.8b-ll`, `qwen-0.8b-json`), a Frontier LLM API, and a task-specific tabular NN across accuracy, valid output rate, and abstention safety?
  - **RQ2 (Calibration & Robustness):** Which paradigm produces the best-calibrated probabilities (`ECE`, `Brier`, `AUROC`, `AURC`) and remains most stable under option-order swaps, paraphrases, and long contexts (`~2,000` tokens)?
  - **RQ3 (Cost–Latency Pareto):** How much API cost and latency can a confidence-gated cascade (`0.8B -> Frontier LLM`) save while preserving near-Frontier accuracy?
- **Visual to include:** Architecture diagram of the 4-tier model ladder + confidence-gated cascade.

### 2. Model Ladder & Hybrid Cascades

| Tier | Model / Identifier | Interface & Output | Hardware / Backend | Applicable Suites |
| :--- | :--- | :--- | :--- | :--- |
| **Tier 1** | Frontier LLM (configurable OpenAI-compatible API) | Structured JSON (`choice_index`, `confidence`) | Cloud API (`temperature=0.0`) | Suites 1–4 |
| **Tier 2a** | `Qwen3.5-0.8B-Base` (`qwen-0.8b-ll`) | Length-normalized log-likelihood over options | Apple M4 (16 GB, PyTorch MPS, BF16) | Suites 1–4 |
| **Tier 2b** | `Qwen3.5-0.8B-Instruct` (`qwen-0.8b-json`) | Structured JSON generation (`parse_ok` tracked) | Apple M4 (16 GB, PyTorch MPS, BF16) | Suites 1–4 |
| **Tier 3** | `Kev-0.8B` (`jaredpalmer/kev-0.8b`) | Native decision-oriented option scorer | Apple M4 (16 GB, MLX, BF16) | Suites 1–4 |
| **Tier 4** | Self-trained Tabular NN (`models/car_build_nn.npz`) | Fixed 12-feature MLP with temperature scaling | Apple M4 (CPU / NumPy) | Suite 3 only |
| **Hybrid** | `Kev-0.8B -> Frontier` & `Qwen-0.8B -> Frontier` | Confidence-gated escalation (`conf < tau` or `!parse_ok`) | M4 Local + Cloud API | Suites 1–4 |

### 3. Multi-Suite Benchmark Design (`data/benchmarks/`)
Explain why the initial 34-row car-build benchmark was too narrow and present the frozen 4-suite benchmark (`data/benchmarks/manifest.json`):

| Suite | Calibrate Items | Test Items | Buckets Covered | Purpose |
| :--- | :---: | :---: | :--- | :--- |
| **Suite 1: `suite1_software_ops`** | 40 | 120 | `incident_triage` (30), `release_gate` (30), `api_routing` (30), `abstain_trap` (30) | General software workflows with zero racing bias |
| **Suite 2: `suite2_historical_eng`** | 47 | 87 | `historical_pair` (69 from `data/outcome/examples.jsonl`), `driver_lever_longctx` (18 from `self_jev/kev_harness.py`) | Real historical `agx.cool` match evidence + ~1,568-token long-context driver decisions |
| **Suite 3: `suite3_car_build`** | 19 | 34 | `in_dist` (15), `unseen_lever` (15), `abstain_trap` (4) | Simulator-backed builds with `oracle_scores` and tabular `features` for Tier 4 NN |
| **Suite 4: `suite4_stress_test`** | — | 72 | `ctx_short` (~250 tok, 18), `ctx_medium` (~750 tok, 18), `ctx_long` (~2,000 tok, 18), `paraphrase` (18) | Context-length scaling and prompt/label paraphrase stability |
| **Total** | **106** | **313** | — | **419 frozen items** (`all_calibrate.jsonl` + `all_test.jsonl`) |

### 4. Evaluation Protocol & Metrics
Document the exact scientific controls for reproducibility:
- **Zero-leakage split discipline:** Escalation thresholds (`tau`) and temperature scaling are fitted strictly on `all_calibrate.jsonl` (`n=106`); all reported numbers come from a single pass on `all_test.jsonl` (`n=313`).
- **Controlled hardware & decoding:** All local models run on the same Apple Silicon M4 (16 GB unified memory, batch size 1), with `state_truncated: false` (`SERVE_MAX_STATE`) and deterministic decoding (`temperature=0.0`).
- **Option-order randomization (`--swap-order`):** Every test item is scored in both original and reversed option order (including reversed `option_descriptions`) to compute **Order-Flip Rate**.
- **Metric suite (computed by `scripts/benchmark_metrics.py`):**
  - *Decision Quality:* Accuracy with **95% bootstrap CI** (1,000 resamples), Macro-F1, Valid Output Rate (`parse_ok`), and **paired McNemar's test** (`p`-values across every model pair).
  - *Calibration & Selective Prediction:* 10-bin **ECE**, **Brier score**, **NLL**, **Error-Detection AUROC**, and **Risk–Coverage AURC**.
  - *Safety & Regret:* **False-Accept Rate** (acting on an abstain trap), **False-Abstain Rate**, **Weighted Asymmetric Error Cost** (5× penalty on false-accept, 1× on other errors), and **Mean Oracle Regret** (on Suite 3 `oracle_scores`).
  - *Robustness, Speed & Cost:* **Order-Flip Rate**, **Paraphrase Flip Rate**, Context-Length accuracy (`short` / `medium` / `long`), **p50 / p95 / p99 latency (ms)**, **Cost per 1,000 decisions**, and **0%–100% Cascade Pareto curves**.

### 5. Quantitative Results & Charts
Fill this section directly from `reports/full_benchmark/benchmark_summary.csv` and `benchmark_summary.json`:

1. **Main Comparison Table:** Insert overall Accuracy (95% CI), Macro-F1, Valid %, ECE, Brier, AUROC, AURC, False-Accept Rate, 5× Cost, Order-Flip Rate, p50/p95 Latency, and Cost/1k.
2. **Statistical Significance:** Report paired McNemar's `p`-values comparing `kev-0.8b` vs. `qwen-0.8b-ll`, `qwen-0.8b-json`, and `frontier-llm`.
3. **Suite & Specialist Analysis:** Embed `04_suite_accuracy.svg` and compare generalist models (Suites 1–4) against Tier 4 NN (Suite 3 `in_dist` vs. `unseen_lever`). Report both raw `suite3_car_build` and `suite3_car_build_clean`.
4. **Calibration & Selective Risk:** Embed `02_reliability_diagram.svg` and `03_risk_coverage.svg`; explain how lower ECE and AURC enable safer automated thresholds.
5. **Robustness & Latency:** Embed `05_context_length.svg` and `06_latency_comparison.svg`; analyze accuracy retention from ~250 to ~2,000 tokens and local MLX latency vs. Frontier API latency.
6. **Cascade Cost–Accuracy Pareto:** Embed `01_cascade_pareto.svg`; highlight the operating point (e.g., 20%–30% escalation) that achieves near-Frontier accuracy at minimal API cost.

### 6. Qualitative Failure Analysis (Web UI Screenshots)
- Pick 2–3 high-signal cases from `contrast_cases` in `benchmark_summary.json` (accessible via the **Open in Demo Tab** button in Web UI Tab 2):
  1. A JSON parse failure or overconfident wrong action by `qwen-0.8b-json` on an `abstain_trap`.
  2. An `unseen_lever` or software runbook rule where `kev-0.8b` follows the text constraint while Tier 4 NN cannot read free text.
  3. A genuinely ambiguous case where `kev-0.8b` expresses low confidence and the cascade escalates cleanly to the Frontier LLM.
- Capture clean side-by-side cards from **Web UI Tab 1** (`decision_ui.html`).

### 7. Threats to Validity, Limitations & Future Work
- **Synthetic vs. live production logs:** Suites 1 and 4 are structured synthetic operational scenarios; live production telemetry may exhibit noisier formatting.
- **Legacy Suite 3 oracle labels:** Some `in_dist` Parkland rows in v1 `car_build_bench` were labeled `Abstain / Unresolved` because the frozen RL policy scored negatively on all 4 builds despite high forecast confidence; report both raw `suite3_car_build` and `suite3_car_build_clean` transparently.
- **Schema scope of Tier 4 NN:** The tabular MLP only consumes fixed numeric features on Suite 3 and cannot generalize to unseen text bulletins or Suites 1, 2, and 4.
- **Hardware scope:** Latency numbers reflect single-stream batch-size-1 inference on an Apple Silicon M4 (16 GB).
- **Future work:** Online shadow-mode deployment, larger calibration sets for tighter conformal guarantees, and evaluating larger `4B` decision-oriented checkpoints on 32 GB+ hardware.
