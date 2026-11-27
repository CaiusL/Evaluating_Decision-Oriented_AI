# Version-Identity Guide — Evaluating Decision-Oriented AI

## 1. Report Revision
- **Document:** `thesis/evaluating_decision_oriented_ai_report.pdf` (17 pages) and `thesis/evaluating_decision_oriented_ai_report.tex` with `thesis/figures/completion_by_time_curves.png`.
- **Snapshot:** Frozen November 26, 2026 evaluation record (`m2_v5`, `racing_engineer_v1`, `feedback_f1_v2`), finalized with explicit `$^\dagger$` markers on affected Qwen tables/figure, full `Qwen/Qwen3.5-0.8B-Base` BPE tokenizer pipeline provenance (revision `dc7cdfe2ee4154fa`), Level 2 Qwen interface qualification, and strict-threshold cascade wording.

## 2. Code & Model Snapshots That Generated the Tables
- **Simulator (`RaceEnv`, untouched since Nov 24):** `self_jev/sim/env.py`, `physics.py`, `track.py`, `weather.py`, `calibrated_params.json`, above frozen 30 Hz RL driver `models/jev_rl_v7.mapdata.npz`.
- **Scoring & Contract:** `decision_eval/src/decision_eval/runners.py` (`M2QwenLoRARunner`), `decision_eval/src/decision_eval/racing_contract.py` (`action_spec_hash = a65f275a8fff3740`).
- **Qwen Runner Interface Fidelity:** The included Qwen runner (`M2QwenLoRARunner`), `scripts/eval_stage_b_alignment_45.py`, and `scripts/train_racing_v1_students.py` **preserve the exact original `max_length=512` right-truncation interface** that produced the reported results (no silent post-hoc repair).

## 3. Frozen Raw Outputs & Analysis Scripts
- **Study 1A (Exploratory v2, N=244):** `reports/full_benchmark_v2/benchmark_summary.json`, `benchmark_raw.jsonl`, `reports/full_benchmark/benchmark_raw_self_jev_minilm_packed.jsonl`; metrics script `scripts/benchmark_metrics.py`.
- **Study 1B (M2 V5, N=600):** `reports/m2_v5/m2_v5_holdout_report.json`, `reports/M2_V5_FINAL_CLOSED_REPORT.md`, `reports/alignment_supplement/minilm_holdout_600_supplement.json`, `data/teacher/split_manifest.json`, `data/holdout/holdout_manifest.json`; scripts `scripts/train_m2_v5_students.py`, `scripts/eval_m2_v5_holdout.py`, `scripts/eval_minilm_holdout_600_supplement.py`.
- **Study 2 (Closed-Loop Racing & Post-Hoc Align-45):** `reports/race_engineer_v2_summary.json`, `reports/racing_engineer_v1/closed_loop_eval_summary.json`, `training_summary.json`, `qwen_colab_training_summary.json`, `thesis_analysis_v1.json`, `reports/alignment_supplement/stage_b_alignment_45_summary.json`, `frontier_stage_b_45_calls.jsonl`, `configs/racing_engineer_v1.yaml`; scripts `scripts/train_racing_v1_students.py`, `scripts/eval_racing_v1_closed_loop.py`, `scripts/eval_stage_b_alignment_45.py`, `scripts/analyze_racing_v1_thesis.py`.
- **Study 3 (Feedback F1 v2, N=45):** `reports/feedback_f1_v2_summary.json`, `reports/F1_V2_CLOSED_FINAL_REPORT.md`; script `scripts/eval_feedback_f1_v2.py`.

## 4. Separate November 27 Companion Simulator Update (Outside Studies 1–3)
Section 3.1 of the main report intentionally preserves the **November 26 frozen companion-simulator snapshot** (`mac_racing_sim` / `rsi-racing-4.4`: guided 4-lap replay within `6.35 s`, mean `2.16 s`; `4/4` regression tests). On November 27, 2026, the separate 4-wheel companion engine (`mac_racing_sim/`, `agx_racing_gpusim/sim/ref/model.py`, and `self_jev/sim/ref_env.py`) was updated independently (guided 4-lap replay `16/16` within `3.35 s`, mean `1.91 s`; `18/18` Comp harness PASS; `5/5` regression tests; plus calibration work on 83 public replays, distinct from any held-out benchmark cohort). Source verification confirmed **zero imports** of `ref_env`, `mac_racing_sim`, or `agx_racing_gpusim` in `RaceEnv` or any Study 1–3 script, and neither `calibrated_params.json` nor `jev_rl_v7.mapdata.npz` changed.

## 5. Viewing Cached Results vs. Rerunning Experiments
- **View all saved results & live replays in Eval Studio (read-only; runs NO training, NO model inference, and NO cloud API calls):**
  ```bash
  python3 decision_eval/cli/serve.py --port 8008
  ```
  Then open the local Eval Studio page on port 8008 in your browser to inspect all four tabs (**Offline Benchmark**, **Racing Evaluation / Post-Hoc Alignment-45**, **Feedback Study**, and **Demo** replays).
- **Recompile the report PDF from source:**
  ```bash
  cd thesis && pdflatex evaluating_decision_oriented_ai_report.tex && pdflatex evaluating_decision_oriented_ai_report.tex
  ```
- **Re-running model inference or training** (requires local weights, HuggingFace cache, or cloud API credentials, which are excluded from submission archives) uses `scripts/eval_racing_v1_closed_loop.py`, `scripts/eval_stage_b_alignment_45.py`, `scripts/eval_feedback_f1_v2.py`, and the Colab notebooks (`Train_M2_V5_Two_Students_Colab.ipynb`, `Train_Racing_V1_Qwen_Colab.ipynb`).
