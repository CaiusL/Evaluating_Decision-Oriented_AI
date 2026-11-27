# Final Closed Report: M2 (V5) Controlled Two-Student Study & `m2-qwen3.5-0.8b-lora-s42` Integration

**Status:** Frozen & Closed (Final Project Endpoint per Judge Ruling, 2026-11-26)  
**Archived Artifacts (Untouched):**
- `reports/m2_v5/m2_v5_train_summary.json` (Colab T4 3-seed training & calibration summary)
- `reports/m2_v5/m2_v5_holdout_report.json` (Frozen single-pass `holdout_600` student & Kev results)
- `reports/m2_v5/m2_v5_holdout_predictions.jsonl` (Per-item `holdout_600` predictions for independent audit)
- `reports/m2_v5/m2_v5_teacher_holdout_summary.json` & `m2_v5_teacher_holdout_raw.jsonl` (Bounded Q2 Frontier Teacher Reference pass)
- `reports/m2_v5/m2_qwen_runner_dev_parity.json` (Local `decision_eval` dev-side parity record)
- `models/m2_qwen3.5_0.8b_lora_s42/` (`lora_adapters.pt`, `adapter_config.json`, `calibration.json`)

---

## 1. Executive Summary & Adopted Model

We adopt **`m2-qwen3.5-0.8b-lora-s42`** (`Qwen/Qwen3.5-0.8B-Base` + native FP32 LoRA `r=16, alpha=32`, seed `42`, scalar temperature `T = 2.02`) as the primary local model for the evaluated offline decision interface:

- On the reported `600`-source M2 holdout (`holdout_600.jsonl`), **`m2-qwen3.5-0.8b-lora-s42`** achieved **91.0% original-order accuracy** (`546/600`) and **88.5% swapped-order accuracy** (`531/600`), improving by **+62.7 pp** over its own zero-shot baseline (`28.3%`, exact McNemar `385` vs. `9`, `p < 1e-6`) and **+28.5 pp** over frozen off-the-shelf `jaredpalmer/kev-0.8b` (`62.5%`, `206` vs. `35`, `p < 1e-6`).
- Across three separate training seeds (`42, 1024, 2026`), Qwen averaged **89.5%** original-order accuracy (`91.0% / 88.0% / 89.5%`; population SD `1.2 pp` with `ddof=0`, sample SD `1.5 pp` with `ddof=1`; average of separate runs, not an ensemble).
- On the bounded Q2 **Frontier teacher reference** pass (`deepseek-v4.1-flash-1017`, original-order only, gold excluded, all 600 retained), the teacher scored **600/600 (100.0%)** with `0` invalid outputs (`p50 = 793.6 ms`). `m2-qwen3.5-0.8b-lora-s42` reached `91.0%` locally at `168.0 ms` p50 on Colab T4 (`$0` API cost).
- **Measured limitations preserved:** `44/600` (`7.3%`) semantic order flips under option reversal (net `-15` accuracy drop from `546/600` to `531/600`); weaker performance than frozen Kev on `engineering_tradeoff` (`117/150` vs. `127/150`); and demonstrated scope limited to the four evaluated software-ops decision families (not racing or arbitrary documents).

---

## 2. Corrected Supervision, Chronology & Reproducibility Provenance

### 2.1 Supervision Description (Gold-Filtered Hard-Label Training)
Both students were trained on the exact same **833 examples** for which the teacher's (`deepseek-v4.1-flash-1017`) parsed prediction agreed with the pre-existing deterministic rule-based gold label (`S = {i : y_hat_{T,i} = y_i}`, `|S| = 833`). Seven parsed teacher–gold disagreements were excluded before training (`4` `engineering_tradeoff`, `2` `incident_triage`, `1` `release_gate`) and remain archived in `data/teacher/deepseek_v41_flash_distill_train_840.jsonl`.
- Because every retained training target coincides with pre-existing gold, this is **gold-verified teacher hard-label distillation** (equivalently, supervised training on a teacher-filtered, gold-labeled subset). It does not isolate teacher labels vs. gold labels, the effect of excluding the 7 disagreements, or soft probability distillation.

### 2.2 Calibration & Selection Chronology (As Executed)
Recovered directly from `scripts/train_m2_v5_students.py`:
1. For each student and each seed `s ∈ {42, 1024, 2026}`, the best epoch was selected strictly on uncalibrated `dev_180` NLL.
2. Each seed's scalar temperature `T_s = argmin_T L_cal(theta_s, T)` was fitted on `calibrate_180.jsonl` via grid search (`[0.25..6.0]` + `±0.15` local refinement; hence the shared `T = 6.15` ceiling on both zero-shot models, where both hit the grid upper bound on their own calibration logits).
3. The deployment seed `s* = argmin_s L_dev(theta_s, T_s)` was then selected among the calibrated candidate pipelines on `dev_180` (`qwen` → seed `42`, `T = 2.02`; `bge` → seed `1024`, `T = 0.65`). Thus `calibrate_180` contributed indirectly to seed selection before `holdout_600` was ever loaded.
4. Frozen `jaredpalmer/kev-0.8b` was evaluated with its native as-shipped calibration (`T = 1.0` wrapper on model probabilities), making this a comparison of deployed pipelines rather than a calibration-matched ablation.

### 2.3 Full SHA-256 Reproducibility Manifest & Bounded Split Scope
Performance is reported on **600 reported source cases across four specified decision buckets, under the frozen generation and evaluation protocol, with no overlap under the listed exact, `source_id`, number-stripped skeleton, question-string, option-tuple, and Jaccard (`max = 0.571`, `0/600 ≥ 0.80`) checks**.

| Artifact | Full SHA-256 Digest |
|---|---|
| `data/teacher/deepseek_v41_flash_verified_train.jsonl` (`833`) | `2146a5b460e1da115fd28562f6d5411babd4355fea0fb0b8ad6917aa8a84bb08` |
| `data/teacher/dev_180.jsonl` (`180`) | `832a24c856ed194b10d134860628f8f95fa3962f72bd37224b80c0d43b18c79c` |
| `data/teacher/calibrate_180.jsonl` (`180`) | `8c8f19e4dcbcb59a24e4d19c8d1060cd311691bb4cea6cad3245b59468eefe0a` |
| `data/holdout/holdout_600.jsonl` (`600`) | `3782d9f6ba4a4164aa7d9c3b238ed8e31d1a5e8f20e0ede35ac0634f76d46bb8` |
| `models/m2_qwen3.5_0.8b_lora_s42/lora_adapters.pt` (`25.6 MB`) | `8542a3c8995051f10f148b0407c1477f7d14ec38463fa1e11b42184886f186c4` |
| `models/m2_qwen3.5_0.8b_lora_s42/adapter_config.json` | `2e3a806eefc99632f4a8f4401a08487c64b1218c383810c327867cc093580ad3` |
| `models/m2_qwen3.5_0.8b_lora_s42/calibration.json` | `f92722714c4660691ed7d6ae4a6a9c8ffd47b11f62b035c6927a71820afd4ce7` |

---

## 3. Complete `holdout_600` Results (Single Frozen Pass + Q2 Teacher Reference)

| System | Holdout Acc (Orig) | Swapped Acc | Order-Flip ↓ | NLL ↓ | ECE ↓ | AURC ↓ | Error AUROC ↑ | Abstain Recall (n=128) | Abstain Prec | False-Abstain (n=472) ↓ | p50 (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Frontier Teacher Ref** (`deepseek-v4.1-flash-1017`, orig-only) | **100.0%** (600/600) | N/A | N/A | N/A | N/A | N/A | N/A | **100.0%** (128/128) | **100.0%** (128/128) | **0.00%** (0/472) | 793.6 (API) |
| **`m2-qwen3.5-0.8b-lora-s42` (deploy seed 42, T=2.02)** | **91.0%** (546/600) | **88.5%** (531/600) | **7.3%** (44) | **0.271** | **0.030** | **0.026** | **0.881** | **89.8%** (115/128) | **87.8%** (115/131) | **3.39%** (16/472) | 168.0 (T4) |
| **Qwen3.5-0.8B LoRA (3-seed avg, `ddof=0` / `ddof=1`)** | **89.5%** (±1.2 / ±1.5 pp) | **86.6%** (±1.8 pp) | **8.8%** (±1.5 pp) | **0.352** | **0.047** | **0.032** | **0.833** | — | — | — | 167.8 (T4) |
| **Kev-0.8B (frozen off-the-shelf ref)** | 62.5% (375/600) | 58.5% (351/600) | 19.3% (116) | 1.133 | 0.177 | 0.287 | 0.639 | 41.4% (53/128) | 65.4% (53/81) | 5.93% (28/472) | 258.3 (T4) |
| **Qwen3.5-0.8B Zero-Shot (T=6.15)** | 28.3% (170/600) | 28.3% (170/600) | 88.2% (529) | 1.583 | 0.052 | 0.652 | 0.626 | 15.6% (20/128) | 33.3% (20/60) | 8.47% (40/472) | 145.3 (T4) |
| **BGE-reranker-base FP32 FT (failed config, seed 1024, T=0.65)** | 22.2% (133/600) | 23.2% (139/600) | 5.8% (35) | 1.625 | 0.040 | 0.770 | 0.532 | 31.2% (40/128) | 24.5% (40/163) | 26.06% (123/472) | 24.9 (T4) |
| **BGE-reranker-base Zero-Shot (T=6.15)** | 19.7% (118/600) | 19.7% (118/600) | 0.2% (1) | 1.634 | 0.077 | 0.802 | 0.507 | 30.5% (39/128) | 19.6% (39/199) | 33.90% (160/472) | 24.6 (T4) |

*Individual Qwen seeds on `holdout_600`: seed `42` = `546/600` (`91.0%`), seed `1024` = `528/600` (`88.0%`), seed `2026` = `537/600` (`89.5%`). Teacher reference NLL/ECE/AURC are omitted because the API returns textual self-confidence rather than option probability distributions.*

### 3.1 Bounded BGE Failure Audit
> The implemented BGE training configuration produced near-chance performance across the four decision buckets, with seed-dependent score collapse. FP32 removed the observed AMP skipped-update problem but did not resolve the failure. The experiment does not identify its cause or establish a limitation of cross-encoders generally.

- **Artifact checks verified in `train_m2_v5_students.py`:** Candidates were grouped listwise per decision (`logits = model(**enc).logits.squeeze(-1).unsqueeze(0)` over all `k=5` pairs, `F.cross_entropy(logits.float(), target)`); `max_tokens_seen = 413 <= 512` (zero truncation); checkpoints saved and reloaded from `bge_seed{s}`. On `holdout_600`, seed `42` collapsed to uniform probabilities (`NLL = 1.609438 ≈ ln 5`, `ECE ≈ 0.0`, `error_auroc = 0.500`, `600/600` order flips from `argmax` index-0 tie-breaking), confirming that low ECE on a collapsed uniform predictor reflects lack of discrimination rather than reliability.

---

## 4. `decision_eval` Integration & Dev-Side Parity Verification

Integrated **`m2-qwen3.5-0.8b-lora-s42`** into `decision_eval` (`paths.py`, `runners.py`, `server.py`, `ui/index.html`) as a distinct model identity reproducing the frozen Colab scorer contract:

- **Fail-fast missing-artifact check:** Verified `FileNotFoundError` when `lora_adapters.pt`, `adapter_config.json`, or `calibration.json` is absent (zero silent fallback).
- **Frozen scorer parity (`scripts/verify_m2_qwen_runner_parity.py` → `reports/m2_v5/m2_qwen_runner_dev_parity.json`):**
  - Validated exact single-token `letter_ids = [357, 417, 351, 414, 458]`, exact `T = 2.02`, exact prompt serialization match against `train_m2_v5_students.format_qwen_prompt`, and `K`-option legal softmax normalization.
  - Across **16 diverse `dev_180.jsonl` fixtures** (all 4 buckets × clean, wrapped, abstention, and swapped-order permutations): **16/16 original-order correct**, **16/16 swapped-order correct**, **0/16 order flips** (`backend = torch-cpu`, `dtype = base=float32+lora=float32`, `load_s = 1.41 s`, local CPU `p50 = 4029.9 ms`).
