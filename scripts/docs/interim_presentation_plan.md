# Interim Presentation Plan (3–4 Minutes, 5 Slides)

**Project:** Evaluating Decision-Oriented AI for Reliable Software Decisions  
**Format:** Interim Checkpoint Presentation (**Target Length: 3 minutes 30 seconds**)  
**Goal:** Show a clear problem, a rigorous multi-suite evaluation design, a working local system (API + Web UI + chart pipeline), and a concrete path to the final report.

---

## Slide-by-Slide Plan & Timing

| Slide | Time | Title | Visuals on Slide | Core Takeaway |
| :---: | :---: | :--- | :--- | :--- |
| **1** | `0:00–0:35` (35s) | **Problem & Goal** | One-line thesis + 3 failure icons (Broken JSON, Overconfident Wrong Action, High API Bill) | Bounded software decisions need calibrated reliability, not free-form chat. |
| **2** | `0:35–1:15` (40s) | **4-Tier Model Ladder & Cascade** | 4-box diagram (Frontier API · `Qwen-0.8B` Base LL & Instruct JSON · `Kev-0.8B` MLX · Tier 4 Tabular NN) + dashed arrow for **Confidence-Gated Cascade** | Fair apples-to-apples comparison on M4 Apple Silicon + hybrid escalation. |
| **3** | `1:15–1:55` (40s) | **4-Suite Benchmark & Scientific Rigor** | 4-card breakdown of **419 frozen items** (`106` calibrate / `313` test) + bottom rigor strip (`Order-Swap · 95% CI · McNemar · ECE/AURC · 5× Cost`) | Expanded beyond 34 car-build rows to test general software ops, real engineering data, and stress tests. |
| **4** | `1:55–2:55` (60s) | **Working System & Early Results** | **1 crisp screenshot** (or 20s clip) from **Web UI Tab 1** comparing `Kev-0.8B` vs `Qwen-0.8B` on an `abstain_trap` or `unseen_lever` + **1 early chart** (`04_suite_accuracy.svg` or `06_latency_comparison.svg`) | Local API, Evaluation Studio Web UI, and automated SVG chart pipeline are already running on M4 MLX. |
| **5** | `2:55–3:30` (35s) | **Progress & Next Steps to Final Report** | 2-column checklist: **Done** vs. **Remaining for Final Report** | Plug in Frontier LLM API + final Tier 4 NN checkpoint to generate full Pareto curves & final report. |

---

## Detailed Slide Content & Speaker Cues

### Slide 1 — Problem & Goal (`0:00–0:35`)
- **On Slide:**
  - Title: *Evaluating Decision-Oriented AI for Reliable Software Decisions*
  - 3 short bullets:
    - Software pipelines need bounded choices (incident triage, CI/CD release gates, safe abstention).
    - Conventional generative LLMs fail via broken JSON, overconfident errors, or high API cost.
    - **Question:** Can a `0.8B` decision-oriented model (`Kev-0.8B`) + confidence-gated cascade beat standard generative prompting?
- **Speaker Cue:**
  > "Modern software workflows rely on bounded decisions—like paging an engineer, blocking a risky deploy, or abstaining when telemetry is missing. Conventional generative LLMs often output broken JSON, act overconfidently on missing data, or cost too much at scale. Our project evaluates whether a lightweight `0.8B` decision-oriented model, `Kev-0.8B`, and a confidence-gated cascade can deliver more reliable, better-calibrated decisions."

### Slide 2 — 4-Tier Model Ladder & Cascade (`0:35–1:15`)
- **On Slide:**
  - **Tier 1:** Frontier LLM (OpenAI-compatible API, structured JSON)
  - **Tier 2:** `Qwen3.5-0.8B` — Base log-likelihood (`qwen-0.8b-ll`) & Instruct JSON (`qwen-0.8b-json`)
  - **Tier 3:** `Kev-0.8B` (`jaredpalmer/kev-0.8b`, Apple Silicon MLX BF16)
  - **Tier 4:** Self-trained Task-Specific NN (`models/car_build_nn.npz`)
  - **Hybrid Cascade:** Route low-confidence local decisions (`conf < tau`, fitted on `calibrate`) to Frontier LLM.
- **Speaker Cue:**
  > "Rather than testing one model in isolation, we compare four tiers on the same M4 Mac hardware: a Frontier LLM API, matched-size `Qwen3.5-0.8B` baselines using both log-likelihood and structured JSON, `Kev-0.8B` on MLX, and a task-specific neural network. We also evaluate a confidence-gated cascade that handles easy decisions locally and only escalates uncertain ones to the Frontier API."

### Slide 3 — 4-Suite Frozen Benchmark (`1:15–1:55`)
- **On Slide:**
  - **Suite 1 — Software Ops (`120` test / `40` cal):** Incident triage, CI/CD release gates, API routing, abstain traps (zero racing bias).
  - **Suite 2 — Real Historical Engineering (`87` test / `47` cal):** `69` real `agx.cool` match pairs + `18` long-context driver-lever decisions (`~1,568` tokens).
  - **Suite 3 — Simulator Car-Build (`34` test / `19` cal):** `in_dist`, `unseen_lever` rules, `abstain_trap`, with `oracle_scores` + tabular features.
  - **Suite 4 — Stress Test (`72` test):** Context-length sweep (`~250` → `~2,000` tok) + prompt/label paraphrases.
  - **Controls:** Strict `calibrate`/`test` split · Option-order swap · 95% bootstrap CIs · McNemar's test · ECE, AURC & 5× False-Accept cost.
- **Speaker Cue:**
  > "Our initial 34-row car-build benchmark was too narrow to support general claims, so we built a frozen 4-suite offline benchmark with 419 total items—covering general software operations, real historical match data and long-context driver levers, simulator builds, and a context-length and paraphrase stress test. Every item is evaluated in both original and swapped option order with full calibration and statistical significance checks."

### Slide 4 — Working System & Early Visuals (`1:55–2:55`)
- **On Slide:**
  - Left: Clean screenshot from **Web UI Tab 1** (`decision_ui.html`) showing `Kev-0.8B` vs `Qwen-0.8B-LL` vs `Qwen-0.8B-JSON` on 1 concrete `abstain_trap` or `unseen_lever` item (showing chosen option, probability bars, and latency in ms).
  - Right: **1 early SVG chart** from `reports/full_benchmark/charts/` (recommended: `04_suite_accuracy.svg` or `06_latency_comparison.svg`).
- **Speaker Cue:**
  > "Our entire local API, 4-tab Evaluation Studio Web UI, and automated SVG chart pipeline are already working on M4 Apple Silicon. On the left is a real side-by-side case from our UI where `Kev-0.8B` outputs calibrated probabilities in milliseconds while generative JSON struggles with [abstention / unseen rules / formatting]. On the right is our early local `0.8B` comparison across suites."

### Slide 5 — Progress & Next Steps (`2:55–3:30`)
- **On Slide:**
  - **Completed:**
    - Frozen 4-suite benchmark (`419` items in `data/benchmarks/`)
    - Local FastAPI server + 4-tab Evaluation Studio Web UI
    - Automated metrics engine (ECE, AURC, McNemar, Cascades) & 6-chart SVG generator
    - Local `0.8B` baselines running on M4 MLX/MPS
  - **Next Steps for Final Report:**
    - Plug in Frontier LLM API key (`scripts/frontier_config.json`)
    - Finalize & plug in Tier 4 NN (`models/car_build_nn.npz`)
    - Run full 313-item swapped-order sweep and complete Accuracy-vs-Cost Pareto & failure analysis
- **Speaker Cue:**
  > "To summarize: our entire offline evaluation infrastructure, 4-suite benchmark, Web UI, and charting pipeline are built and verified. Our remaining steps for the final report are simply plugging in the Frontier LLM API and our final trained Tier 4 NN checkpoint, running the full 313-item sweep, and writing up the accuracy-vs-cost Pareto curves and failure taxonomy. Thank you—happy to take questions."

---

## Quick Delivery Tips
- **Skip for interim (save for final report):** The 15-column metrics table, math formulas (Brier/NLL/McNemar equations), and all 6 charts—keep Slide 4 to **1 screenshot + 1 chart max**.
- **Team split (if 2–3 presenters):**
  - *2 speakers:* Speaker A takes Slides 1–3 (`1:55`), Speaker B takes Slides 4–5 (`1:35`).
  - *3 speakers:* Speaker A takes Slides 1–2 (`1:15`), Speaker B takes Slide 3 (`0:40`), Speaker C takes Slides 4–5 (`1:35`).
- **Likely Q&A prep:**
  - *"Why include both software ops and racing?"* → Software ops (Suite 1) tests domain generality with zero racing bias; racing (Suites 2–3) adds real historical telemetry, long-context driver levers, and ground-truth simulator `oracle_scores` for measuring exact regret.
  - *"How do you avoid overfitting cascade thresholds?"* → Thresholds `tau` are fitted strictly on the 106-item `calibrate` split and evaluated once on the frozen 313-item `test` split.
