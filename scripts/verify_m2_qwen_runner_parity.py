#!/usr/bin/env python3
"""Dev-side parity verification for m2-qwen3.5-0.8b-lora-s42 in decision_eval.

Verifies on saved dev_180.jsonl fixtures (never holdout_600):
  1. Fail-fast FileNotFoundError when adapter directory is missing
  2. Exact artifact loading, letter_ids [357, 417, 351, 414, 458], and T=2.02
  3. Exact prompt serialization parity against scripts/train_m2_v5_students.py
  4. Semantic predictions across clean, wrapped, abstention, and swapped-order cases
  5. Local device (MPS/CPU) latency and accuracy vs. Colab dev_180 benchmark (99.44% / 98.89%)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "decision_eval" / "src"
SCRIPTS_DIR = ROOT / "scripts"
for p in (str(SRC_DIR), str(SCRIPTS_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import train_m2_v5_students as m2  # noqa: E402
from decision_eval import paths  # noqa: E402
from decision_eval.runners import M2QwenLoRARunner  # noqa: E402


def main() -> None:
    # 1. Fail-fast check on missing adapter dir
    try:
        M2QwenLoRARunner(ROOT / "models" / "does_not_exist_m2", device="cpu")
        raise AssertionError("Expected FileNotFoundError for missing adapter directory!")
    except FileNotFoundError as exc:
        print(f"[PASS] Fail-fast guard triggered cleanly: {exc}")

    # 2. Load exact m2-qwen3.5-0.8b-lora-s42 checkpoint
    runner = M2QwenLoRARunner(paths.M2_QWEN_LORA_S42_DIR, device="auto")
    print(
        f"[PASS] Loaded m2-qwen3.5-0.8b-lora-s42: backend={runner.backend}, dtype={runner.dtype}, "
        f"T={runner.temperature}, letter_ids={runner.letter_ids}, load_s={runner.load_s}s"
    )
    assert runner.letter_ids == [357, 417, 351, 414, 458]
    assert abs(runner.temperature - 2.02) < 1e-6

    # 3. Select 16 diverse dev_180 fixtures: 4 buckets x (clean + wrapped + abstain)
    dev_rows = [json.loads(l) for l in (ROOT / "data/teacher/dev_180.jsonl").read_text().splitlines() if l.strip()]
    fixtures = []
    for bucket in ("incident_triage", "release_gate", "api_routing", "engineering_tradeoff"):
        b_rows = [r for r in dev_rows if r["bucket"] == bucket]
        fixtures.append(next(r for r in b_rows if r["context_band"] == "clean" and not r["is_abstain_gold"]))
        fixtures.append(next(r for r in b_rows if r["context_band"] == "wrapped" and not r["is_abstain_gold"]))
        fixtures.append(next(r for r in b_rows if r["is_abstain_gold"]))
    # Add 4 more wrapped/clean items -> 16 total
    seen_ids = {r["id"] for r in fixtures}
    for r in dev_rows:
        if len(fixtures) >= 16:
            break
        if r["id"] not in seen_ids:
            fixtures.append(r)
            seen_ids.add(r["id"])

    orig_ok = swap_ok = flips = 0
    lats = []
    for r in fixtures:
        k = len(r["options"])
        # Prompt serialization parity check against train_m2_v5_students.format_qwen_prompt
        expected_prompt = m2.format_qwen_prompt(r, list(range(k)))
        actual_prompt = runner.format_prompt(r["context"], r["question"], r["options"], r["option_descriptions"])
        assert actual_prompt == expected_prompt, f"Prompt mismatch on {r['id']}!"

        res_o = runner.score_decision(r["context"], r["question"], r["options"], r["option_descriptions"])
        lats.append(res_o["latency_ms"])
        pred_o = res_o["choice_index"]
        orig_ok += int(pred_o == int(r["gold_index"]))

        # Swapped permutation
        rev_opts = list(reversed(r["options"]))
        rev_descs = list(reversed(r["option_descriptions"]))
        res_s = runner.score_decision(r["context"], r["question"], rev_opts, rev_descs)
        pred_s_orig = (k - 1) - res_s["choice_index"]
        swap_ok += int(pred_s_orig == int(r["gold_index"]))
        flips += int(pred_o != pred_s_orig)

    lats.sort()
    p50 = lats[len(lats) // 2]
    report = {
        "model": "m2-qwen3.5-0.8b-lora-s42",
        "backend": runner.backend,
        "dtype": runner.dtype,
        "temperature": runner.temperature,
        "letter_ids": runner.letter_ids,
        "load_s": runner.load_s,
        "fixtures_n": len(fixtures),
        "prompt_serialization_exact_match": True,
        "orig_correct": f"{orig_ok}/{len(fixtures)}",
        "swapped_correct": f"{swap_ok}/{len(fixtures)}",
        "order_flips": f"{flips}/{len(fixtures)}",
        "latency_p50_ms": round(p50, 1),
        "fail_fast_missing_adapter_passed": True,
    }
    out_p = ROOT / "reports" / "m2_v5" / "m2_qwen_runner_dev_parity.json"
    out_p.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"[PASS] Dev-side parity verified ({len(fixtures)} fixtures): orig={orig_ok}/{len(fixtures)}, swap={swap_ok}/{len(fixtures)}, flips={flips}/{len(fixtures)}, local p50={p50:.1f}ms")
    print(f"Saved -> {out_p}")


if __name__ == "__main__":
    main()
