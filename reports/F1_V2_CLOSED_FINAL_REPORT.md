# Closed Study Report: F1 v2 Previous-Race Feedback Study, Provenance Correction, and M2 Pre-Flight Audit

**Status:** Frozen & Closed (per Judge Ruling, 2026-11-25)  
**Original Numerical Logs Preserved Untouched:**
- `reports/feedback_f1_summary.json` (Pass 1, 45 units)
- `reports/feedback_f1_v2_summary.json` (Final v2 with matched `Z` + `kev-0.8b` backfill, 45 units)
- `reports/feedback_f1_v2_raw/` (per-unit raw traces for all adaptive models)

---

## 1. Substantive Provenance Correction & Training-History Reconciliation

### 1.1 Racing & F1 Cross-Encoder Identity (`MS MARCO MiniLM-L6-v2`)
All racing evaluations previously labeled `self-jev-minilm`—specifically the **9-case exploratory racing pilot** (`reports/race_engineer_v2_summary.json`), **F1 Pass 1** (`reports/feedback_f1_summary.json`), and **F1 v2** (`reports/feedback_f1_v2_summary.json`)—are formally relabeled as:

> **MS MARCO MiniLM-L6-v2 — off-the-shelf retrieval cross-encoder (`cross-encoder/ms-marco-MiniLM-L6-v2`); no Self-JEV fine-tuning**

| Provenance Field | Verified Record (Reconstructed from Hardcoded Loader & Local Checkpoint) |
|---|---|
| **Loaded Path** | `models/ms-marco-MiniLM-L6-v2` (hardcoded in `scripts/eval_race_engineer_v2.py:50` & `scripts/eval_feedback_f1.py:197-198`) |
| **Upstream Checkpoint** | `cross-encoder/ms-marco-MiniLM-L6-v2` (22.7M params, MS MARCO passage-ranking cross-encoder; **not** fine-tuned for Self-JEV) |
| **Architecture / Config** | `BertForSequenceClassification`, 6 layers, hidden 384, 12 attention heads, `max_position_embeddings=512`, `vocab_size=30522` |
| **Ensemble Member Count** | `1` (single model; contrast with intended 3-fold `BAAI/bge-reranker-base` ensemble at ~834M total params = `278M × 3`) |
| **Artifact SHA-256 (prefix)** | `model.safetensors`: `821d1aa69520101d...` (90,870,598 B); `config.json`: `380e02c93f431831...`; `tokenizer.json`: `d241a60d5e8f04cc...` |
| **Intended Trained Model Status** | **Not evaluated** in the 9-case pilot or F1 feedback study. Earlier offline BGE results remain subject to their own Colab manifests and are not invalidated by the local directory's absence. |

### 1.2 Training-History Conflict Resolved (Run #5 vs. Run #6)
Verified against `reports/full_benchmark/ablation_dev_report.json` and the V3/V4 briefs:
- **Run #5 (`arm_a_stage_a_only`):** `/content/drive/MyDrive/self_jev_minilm_stage_a/model` (`temperature = 1.7817`), packed-input Stage A model (`49.8%` on V3, 35% wrapping, folds `[156, 75, 75]`).
- **Run #6 (`arm_b_stage_a_plus_b`):** `/content/drive/MyDrive/self_jev_minilm_balanced/model` (`temperature = 1.6896`), Stage A + Stage B (`192`-item sufficiency curriculum, `46.3%` on V3).
- Prior reference in `F1_V2_FINAL_AND_NEXT_MILESTONE_BRIEF.md` attributing Stage A+B to Run #5 is hereby corrected.

---

## 2. Final Frozen F1 v2 Results (45 Fresh Support/Query Units)

**Experimental Boundaries:** 9 canonical track × weather cells × 5 fresh ordered seed pairs `(1001→2001)`..`(1005→2005)` (strictly disjoint from `[101..505]`). Independent RNG streams per `(cell_idx, seed)` via `np.random.default_rng(seed * 100 + cell_idx)`. Fresh `RaceEnv` and stateless model calls per arm (`Z`, `A`, `F`). Frozen driver (`jev_rl_v7.mapdata.npz`, N3), frozen NN (`nn-combo`, NN-0), zero weight updates, zero external AGX gap in feedback, zero invalid outputs, zero truncations.

### 2.1 Adaptive Decision Models (Matched `seed_B` Queries)

| System | Provenance | Z Completion (No History) | A Completion (Actions Only) | F Completion (Actions + Outcomes) | F − A Change | F − Z Change | Pits (Z / A / F) |
|---|---|---:|---:|---:|---:|---:|---:|
| **kev-0.8b** *(max 462 tok, trunc=no)* | Delayed planned backfill via `KevRunner` | 36 / 45 (80.0%) | 36 / 45 (80.0%) | 36 / 45 (80.0%) | **0.0 pp** (+0/−0; 0.00 s, n=36) | **0.0 pp** (+0/−0; +2.40 s, n=36) | 0 / 0 / 0 |
| **MS MARCO MiniLM-L6-v2** *(22.7M, off-the-shelf; max 279/512 tok)* | No Self-JEV fine-tuning; F−A Pass 1, Z diagnostic | **40 / 45 (88.9%)** | 25 / 45 (55.6%) | 37 / 45 (82.2%) | **+26.7 pp** (+14/−2; −43.81 s, n=23) | **−6.7 pp** (+0/−3; +15.68 s, n=37) | 0 / 84 / 23 |
| **frontier-llm** (`deepseek-v4.1-flash-1017`) | F−A Pass 1 (cached); Z diagnostic | 30 / 45 (66.7%) | 31 / 45 (68.9%) | 33 / 45 (73.3%) | **+4.4 pp** (+2/−0; −0.12 s, n=31) | **+6.7 pp** (+3/−0; +0.57 s, n=30) | 0 / 0 / 0 |

### 2.2 Frozen Non-Adaptive References (Same 45 `seed_B` Queries, No History)

| Reference | Role | Query Finishes | Mean Finish Time (Finishers Only) |
|---|---|---:|---:|
| **nn-combo** | NN-0 / N1: Frozen 12-feature build-only MLP | 35 / 45 (77.8%) | 639.96 s (n=35) |
| **driver_only** | Frozen baseline (`dry_balanced` + `balanced` + `stay_out`) | 38 / 45 (84.4%) | 624.31 s (n=38) |
| **track_rule** | Frozen strong reference (not a proven ceiling) | 37 / 45 (82.2%) | 612.44 s (n=37) |

- **Paired `nn-combo` vs. `driver_only` audit:** `0` rescued, `3` regressions (**−6.7 pp**), `0W–3T–35L` (`7` both failed; 32 slower common finishes + 3 completion losses), **+5.28 s** mean (**+4.37 s** median) across `n=35` common finishers.

### 2.3 Verified Diagnostic Checks & Bounded Conclusions

1. **MS MARCO MiniLM-L6-v2 (verified from `raw_self-jev-minilm_all.json`):**
   - Exact joint completion pattern across all 45 units: `(Z, A, F)` = `(Y,Y,Y): 23`, `(Y,Y,N): 2`, `(Y,N,Y): 14`, `(Y,N,N): 1`, `(N,N,N): 5`. Every `F`-over-`A` rescue (`14/14`) was already a `Z` completion.
   - Triple-common-finisher (`n=23`) time decomposition reconciles algebraically: `F−A = −43.81 s`, `A−Z = +69.03 s`, `F−Z = +25.23 s`.
   - **Conclusion:** For the tested MS MARCO MiniLM checkpoint and history renderer, actions-only history substantially worsened performance; outcome fields repaired much of that degradation, but zero-history inference (`Z`, 40/45, 0 pits) remained better on completion and common-finisher time.
2. **Kev-0.8B (serialized-input check passed):**
   - Verified in `eval_feedback_f1_v2.py` + `eval_feedback_f1.py`: `A` prefix (`162` chars) and `F` prefix (`377` chars, appending full deterministic outcome telemetry) were both passed intact to `KevRunner.score_unified(...)` with `max_prompt_tokens_seen = 462` and `any_state_truncated = False`.
   - **Conclusion:** Adding outcome fields did not change Kev's selected actions on these 45 units (`F−A = 0.0 pp`, `0.00 s`). Adding history (`A` or `F` vs. `Z`) shifted 20 build choices and increased mean time by `+2.40 s` (`n=36`) with the same completion set (`36/45`).
3. **Frontier LLM (`deepseek-v4.1-flash-1017`):**
   - **Conclusion:** Three observed additional finishes over `Z` (`+6.7 pp`, `+0.57 s` on `n=30` common finishers) and two over `A` (`+4.4 pp`, `−0.12 s` on `n=31`), with no observed completion regressions and substantial uncertainty (descriptive exact McNemar `p = 0.25` for `F−Z`).
4. **Closure Statement:**
   > We close this exploratory racing-feedback study because its planned comparisons are complete. The trained Self-JEV BGE ensemble was not evaluated in this study; conclusions about that model's feedback behavior remain open.

---

## 3. M2 Pre-Flight Manifest Audit (V5 Offline Two-Student Study)

All 7 pre-flight checks requested by the judge before touching the 600-item holdout were verified from local manifests:

1. **833/840 Teacher Exclusions (`data/teacher/deepseek_v41_flash_distill_train_840.jsonl` vs. `deepseek_v41_flash_verified_train.jsonl`):**
   - All **7 excluded items** (`t_bulk_0173`, `t_bulk_0555`, `t_bulk_0898`, `t_bulk_0728`, `t_bulk_1033`, `t_bulk_0306`, `t_bulk_0188`) had `teacher_parse_ok = True` (0 malformed outputs); all 7 were pure **teacher-vs-gold disagreements** (`4` `engineering_tradeoff`, `2` `incident_triage`, `1` `release_gate`; 4/7 were `gold_index = 4` abstain cases where the teacher chose a substantive option). Both students will train on the identical 833 verified items.
2. **Split Roles (`data/teacher/split_manifest.json`, seed `20261124`, `1,200` non-test items):**
   - `train_840.jsonl`: 840 items (833 verified for student training)
   - `dev_180.jsonl`: 180 items (checkpoint selection only; untouched by teacher filtering)
   - `calibrate_180.jsonl`: 180 items (post-selection temperature calibration only)
3. **Holdout 600 Structure (`data/holdout/holdout_manifest.json` + `holdout_600.jsonl`, seed `20261125`):**
   - **600 independent `source_id`s** (`src_hold_0001`..`src_hold_0600`), 150 per bucket (`incident_triage`, `release_gate`, `api_routing`, `engineering_tradeoff`), `448` clean / `152` wrapped, `128` abstain.
4. **Holdout Independence:** `overlap_hist_test = 0`, `overlap_nontest_1200 = 0`.
5. **Initializations to Verify Before Training:** Clean public `BAAI/bge-reranker-base` (never Run #5/Run #6) and clean public `Qwen/Qwen3.5-0.8B-Base` (verified no Kev LoRA adapter merged).
6. **Selection, Calibration & Evaluation Protocol:** 3 training seeds per student where feasible; select on `dev_180`, calibrate on `calibrate_180`, freeze all code/prompts/thresholds, then evaluate once on `holdout_600` alongside zero-shot student baselines, frozen `kev-0.8b`, and teacher-vs-gold.
