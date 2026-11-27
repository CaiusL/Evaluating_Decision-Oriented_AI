"""Train and calibrate Self-JEV (`self-jev-minilm`, BAAI/bge-reranker-base, XLM-RoBERTa-base 278M/fold).

Clean balanced restart implementing the full Stage A + Stage B supervised plan:
  1. Balanced source-grouped 3-fold splits using `situation_id` (fixing the batch `group_id` bug
     that previously caused [156, 75, 75] imbalance -> now ~[102, 102, 102] with all 3 suites per fold).
  2. Shared tokenizer-budgeted line/block `pack_evidence()` used identically in training, calibration,
     and inference (strictly <= 512 tokens; numeric masking used ONLY for boilerplate template detection).
  3. Stage A (Clean Preservation + Additive Distractor Retention): every real example ALWAYS trains on
     its clean target (weight 1.0); 25% of steps ADD a distractor-wrapped view (weight 0.5) instead of
     replacing clean data, plus 35% Paraphrase-Consistency KL (0.35) + Top-2 SWA.
  4. Stage B (Verified Evidence-Sufficiency Curriculum): 75% real replay + 25% independently labeled
     3-variant counterfactuals (64 sources x 3 = 192 items: complete, critical-fact-removed with
     rule-recomputed label, irrelevant-fact-removed) using varied abstain labels ('unresolved',
     'Abstain / Unresolved', 'Abstain / Escalate'), evaluated on a separate 16-source dev set.
  5. Out-of-fold temperature calibration on all 306 real non-test items (zero test leakage).
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import random
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_MODEL = "BAAI/bge-reranker-base"
OUT_DIR = REPO_ROOT / "models" / "self_jev_minilm"

_CAR_BUILD_POLICY = (
    "Build Selection Policy: Choose the single safest, highest-scoring car build for the forecast and track "
    "(dry_low_drag for dry/fast conditions; damp_mid_downforce for transitional/damp or higher-grip needs; "
    "wet_high_downforce for heavy wetness; follow any Event Bulletin tyre restrictions strictly). "
    "Choose 'Abstain / Unresolved' if forecast_confidence < 0.45, sensor_missing=1, or no candidate build is safe."
)

_STOPWORDS = {
    "the", "and", "for", "that", "this", "with", "from", "are", "was", "were", "have", "has",
    "had", "not", "but", "all", "can", "should", "which", "what", "when", "where", "how",
    "into", "than", "then", "them", "they", "their", "there", "these", "those", "will", "would",
    "could", "been", "being", "about", "after", "before", "under", "over", "only", "single",
    "most", "more", "less", "each", "other", "such", "also", "just", "any", "its", "our", "your",
}

_Q_PARAPHRASES = {
    "Which incident triage action should the automation take?": [
        "Select the appropriate operational response level for this alert.",
        "Determine the required severity and escalation action from the incident metrics.",
        "What triage decision matches the policy and observed signal?",
    ],
    "Which CI/CD release gate decision should be executed?": [
        "Determine the required deployment gate outcome for this release candidate.",
        "Should this build be promoted, held, rolled back, blocked, or escalated?",
        "Choose the correct release pipeline action based on the canary and security metrics.",
    ],
    "Which engineering queue should receive this ticket?": [
        "Route this incoming support issue to the responsible engineering team.",
        "Which team owns the problem described in this ticket?",
        "Assign this ticket to the single best owning queue or abstain.",
    ],
}

_TOKEN_RE = re.compile(r"[a-z0-9_]+")
_MASK_NUM_HEX_RE = re.compile(r"\b(?:0x[0-9a-f]+|[0-9a-f]{6,}|\d+(?:\.\d+)?%?)\b", re.IGNORECASE)
_MASK_TS_RE = re.compile(r"\d{4}-\d{2}-\d{2}[t ]\d{2}:\d{2}:\d{2}z?|\d{2}:\d{2}:\d{2}", re.IGNORECASE)

_TRAIN_SERVICES = ["edge-proxy", "cache-node", "telemetry-bus", "blob-worker", "queue-relay", "health-agent"]
_TRAIN_REGIONS = ["us-central-1", "eu-north-1", "ap-south-1", "sa-east-1"]

_DEV_SERVICES = ["metrics-collector", "log-shipper", "cron-daemon", "static-cdn", "dns-cache", "audit-sink"]
_DEV_REGIONS = ["ca-central-1", "me-south-1", "ap-southeast-2", "eu-south-1"]


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def load_nontest_pool() -> list[dict[str, Any]]:
    """Load all 306 strictly non-test items with per-situation source group keys & leakage checks."""
    test_items = _load_jsonl(REPO_ROOT / "data" / "benchmarks" / "all_test.jsonl")
    test_ids = {it["id"] for it in test_items}
    test_ctx_set = {it["context"] for it in test_items}

    pool: list[dict[str, Any]] = []
    for it in _load_jsonl(REPO_ROOT / "data" / "benchmarks" / "suite1_software_ops_calibrate.jsonl"):
        rec = dict(it)
        rec["group_key"] = rec["id"]
        pool.append(rec)

    outcome_path = REPO_ROOT / "data" / "outcome" / "examples.jsonl"
    if outcome_path.is_file():
        for idx, row in enumerate(_load_jsonl(outcome_path)):
            if row.get("split") not in ("train", "select"):
                continue
            cands = row.get("candidates") or []
            if len(cands) != 2:
                continue
            keys = [c["key"] for c in cands]
            if row.get("label_key") not in keys:
                continue
            gold_idx = keys.index(row["label_key"])
            options = [f"Candidate_A_{cands[0]['key'][:6]}", f"Candidate_B_{cands[1]['key'][:6]}"]
            descs = [cands[0]["description"], cands[1]["description"]]
            ctx = (
                f"Historical Match Context: {row.get('situation', '')}\n"
                f"Track: {row.get('track', 'unknown')} | Moisture band: {row.get('wet', 'unknown')}\n"
                f"Outcome Rule: {row.get('outcome_rule', '')}\n"
                f"Candidate A ({options[0]}): {descs[0]}\n"
                f"Candidate B ({options[1]}): {descs[1]}"
            )
            # Use per-situation ID (NOT batch group_id) so related situations stay grouped without a giant batch group
            sit = str(row.get("situation_id") or f"s2_tr_{idx}")
            pool.append(
                {
                    "id": f"s2_hist_train_{idx:04d}",
                    "suite": "suite2_historical_eng",
                    "bucket": "historical_pair",
                    "group_key": f"s2_{sit}",
                    "context": ctx,
                    "question": row.get("question", "Choose the recorded car setup tied to the stronger finishing place."),
                    "options": options,
                    "option_descriptions": descs,
                    "gold_index": gold_idx,
                }
            )
    for it in _load_jsonl(REPO_ROOT / "data" / "benchmarks" / "suite2_historical_eng_calibrate.jsonl"):
        rec = dict(it)
        sit = str(rec.get("situation_id") or rec["id"])
        rec["group_key"] = f"s2_{sit}"
        pool.append(rec)

    s3_train_path = REPO_ROOT / "data" / "car_build_bench" / "train.jsonl"
    if s3_train_path.is_file():
        for row in _load_jsonl(s3_train_path):
            options = list(row["options"])
            descs = list(row.get("option_descriptions") or options)
            pool.append(
                {
                    "id": f"s3_cb_train_{row['id']}",
                    "suite": "suite3_car_build",
                    "bucket": row.get("bucket", "in_dist"),
                    "group_key": f"s3_{row['id']}",
                    "context": f"{_CAR_BUILD_POLICY}\n{row['context']}",
                    "question": row["question"],
                    "options": options,
                    "option_descriptions": descs,
                    "gold_index": int(row["gold_index"]),
                }
            )
    for it in _load_jsonl(REPO_ROOT / "data" / "benchmarks" / "suite3_car_build_calibrate.jsonl"):
        rec = dict(it)
        rec["group_key"] = f"s3_{rec['id']}"
        pool.append(rec)

    for it in pool:
        assert it["id"] not in test_ids, f"ID leakage detected: {it['id']}"
        assert it["context"] not in test_ctx_set, f"Context leakage detected: {it['id']}"
    return pool


def make_grouped_stratified_folds(items: list[dict[str, Any]], n_folds: int, seed: int) -> list[list[dict[str, Any]]]:
    """Assign entire source groups to folds, balancing BOTH per-strata counts and total fold size."""
    rng = random.Random(seed)
    by_strata_group: dict[tuple[str, str], dict[str, list[dict[str, Any]]]] = {}
    for it in items:
        strata = (it["suite"], it.get("bucket", "default"))
        gkey = str(it.get("group_key") or it["id"])
        by_strata_group.setdefault(strata, {}).setdefault(gkey, []).append(it)

    folds: list[list[dict[str, Any]]] = [[] for _ in range(n_folds)]
    fold_totals = [0] * n_folds
    for strata in sorted(by_strata_group.keys()):
        gkeys = list(by_strata_group[strata].keys())
        rng.shuffle(gkeys)
        # Largest groups first (greedy LPT balancing) within each stratum
        gkeys.sort(key=lambda gk: len(by_strata_group[strata][gk]), reverse=True)
        strata_counts = [0] * n_folds
        for gk in gkeys:
            target_fold = min(range(n_folds), key=lambda f: (strata_counts[f], fold_totals[f]))
            grp_rows = by_strata_group[strata][gk]
            folds[target_fold].extend(grp_rows)
            strata_counts[target_fold] += len(grp_rows)
            fold_totals[target_fold] += len(grp_rows)
    return folds


def generate_sufficiency_curriculum(n_sources: int = 64, seed: int = 20260501) -> list[dict[str, Any]]:
    """Generate verified 3-variant evidence-sufficiency counterfactuals across 4 disjoint domains.

    Each source produces 3 variants with deterministic rule-recomputed labels:
      1. complete: all facts present -> supported active decision
      2. critical_removed: a decision-critical fact is missing/unknown -> rule engine recomputes
         label (becomes unresolved/abstain ONLY when no remaining independent OR-condition decides it)
      3. irrelevant_removed: non-critical metadata removed -> original active decision unchanged

    Varies abstain option wording ('unresolved', 'Abstain / Unresolved', 'Abstain / Escalate')
    to teach domain-agnostic evidence sufficiency rather than a single label string.
    """
    rng = random.Random(seed)
    abstain_labels = ["unresolved", "Abstain / Unresolved", "Abstain / Escalate"]
    items: list[dict[str, Any]] = []

    for s_idx in range(n_sources):
        dom = s_idx % 4
        abs_lbl = abstain_labels[s_idx % len(abstain_labels)]
        meta_zone = rng.choice(["zone-alpha", "zone-beta", "zone-gamma"])
        meta_owner = rng.choice(["team-core", "team-infra", "team-ops"])

        if dom == 0:
            # Domain 0: Autoscaling (OR rule for Urgent, AND rule for Scale_In)
            opts = ["Scale_Out_Urgent", "Scale_Out_Gradual", "Hold_Pool", "Scale_In_Idle", abs_lbl]
            descs = [
                "Urgent scale-out: cpu_pct >= 85% OR queue_depth > 500.",
                "Gradual scale-out: 65% <= cpu_pct < 85% AND queue_depth <= 500.",
                "Hold current pool: 25% <= cpu_pct < 65% AND queue_depth <= 500.",
                "Scale in idle nodes: cpu_pct < 25% AND queue_depth < 20.",
                "Insufficient or missing required metrics to determine scaling action; leave unresolved.",
            ]
            pol = (
                "Autoscaling Policy: Scale_Out_Urgent if cpu_pct >= 85% or queue_depth > 500; "
                "Scale_Out_Gradual if 65% <= cpu_pct < 85% and queue_depth <= 500; "
                "Hold_Pool if 25% <= cpu_pct < 65% and queue_depth <= 500; "
                "Scale_In_Idle if cpu_pct < 25% and queue_depth < 20; "
                f"choose '{abs_lbl}' if required metrics are missing or unknown."
            )
            q = "Which autoscaling action should the controller take?"

            def eval_dom0(cpu: float | None, qd: int | None) -> int:
                if (cpu is not None and cpu >= 85.0) or (qd is not None and qd > 500):
                    return 0
                if cpu is None or qd is None:
                    return 4
                if cpu >= 65.0:
                    return 1
                if cpu >= 25.0:
                    return 2
                if qd < 20:
                    return 3
                return 2

            cls = s_idx % 4
            if cls == 0:
                # Half have both OR triggers (removing one keeps Urgent!), half have only 1 trigger (removing it -> unresolved!)
                if (s_idx // 4) % 2 == 0:
                    cpu_v, qd_v = round(rng.uniform(86.0, 97.0), 1), rng.randint(550, 900)
                else:
                    cpu_v, qd_v = round(rng.uniform(86.0, 97.0), 1), rng.randint(40, 220)
            elif cls == 1:
                cpu_v, qd_v = round(rng.uniform(66.0, 83.0), 1), rng.randint(50, 380)
            elif cls == 2:
                cpu_v, qd_v = round(rng.uniform(28.0, 62.0), 1), rng.randint(30, 300)
            else:
                cpu_v, qd_v = round(rng.uniform(5.0, 22.0), 1), rng.randint(0, 16)

            variants = [
                ("complete", cpu_v, qd_v, True),
                ("crit_removed", None, qd_v, True),
                ("irrel_removed", cpu_v, qd_v, False),
            ]
            for vname, c_val, q_val, keep_meta in variants:
                gold = eval_dom0(c_val, q_val)
                parts = []
                if keep_meta:
                    parts.append(f"zone={meta_zone}, owner={meta_owner}")
                parts.append(f"cpu_pct={c_val}%" if c_val is not None else "cpu_pct=missing")
                parts.append(f"queue_depth={q_val}" if q_val is not None else "queue_depth=missing")
                ctx = f"{pol}\nTelemetry: " + ", ".join(parts) + "."
                items.append({
                    "id": f"suff_d0_s{s_idx:03d}_{vname}", "suite": "sufficiency_curriculum", "bucket": f"suff_{vname}",
                    "group_key": f"suff_src_{s_idx:03d}", "context": ctx, "question": q,
                    "options": opts, "option_descriptions": descs, "gold_index": gold,
                })

        elif dom == 1:
            # Domain 1: Fraud Risk Gate (strict vs inclusive + signed delta)
            opts = ["Block_Now", "Challenge_3DS", "Manual_Review", "Approve", abs_lbl]
            descs = [
                "Block immediately: risk_score >= 0.85 OR sanctioned=true.",
                "Challenge with 3DS: 0.55 <= risk_score < 0.85 AND sanctioned=false.",
                "Send to manual review: 0.30 <= risk_score < 0.55 AND sanctioned=false.",
                "Approve transaction: risk_score < 0.30 AND sanctioned=false.",
                "Required risk_score or sanctions check is missing/unavailable; leave unresolved.",
            ]
            pol = (
                "Risk Policy: Block_Now if risk_score >= 0.85 or sanctioned=true; "
                "Challenge_3DS if 0.55 <= risk_score < 0.85 and sanctioned=false; "
                "Manual_Review if 0.30 <= risk_score < 0.55 and sanctioned=false; "
                "Approve if risk_score < 0.30 and sanctioned=false; "
                f"choose '{abs_lbl}' if required risk fields are missing."
            )
            q = "Which risk gate decision should be executed?"

            def eval_dom1(rs: float | None, sanc: bool | None) -> int:
                if (rs is not None and rs >= 0.85) or (sanc is True):
                    return 0
                if rs is None or sanc is None:
                    return 4
                if rs >= 0.55:
                    return 1
                if rs >= 0.30:
                    return 2
                return 3

            cls = s_idx % 4
            if cls == 0:
                rs_v, s_v = (0.85 if s_idx % 8 == 0 else round(rng.uniform(0.87, 0.98), 2)), ((s_idx // 4) % 2 == 0)
            elif cls == 1:
                rs_v, s_v = round(rng.uniform(0.56, 0.82), 2), False
            elif cls == 2:
                rs_v, s_v = round(rng.uniform(0.31, 0.52), 2), False
            else:
                rs_v, s_v = round(rng.uniform(0.03, 0.26), 2), False

            variants = [
                ("complete", rs_v, s_v, True),
                ("crit_removed", None, s_v, True),
                ("irrel_removed", rs_v, s_v, False),
            ]
            for vname, r_val, sa_val, keep_meta in variants:
                gold = eval_dom1(r_val, sa_val)
                parts = []
                if keep_meta:
                    parts.append(f"region={meta_zone}, merchant_tier=standard")
                parts.append(f"risk_score={r_val}" if r_val is not None else "risk_score=unavailable")
                parts.append(f"sanctioned={'true' if sa_val else 'false'}" if sa_val is not None else "sanctioned=unknown")
                ctx = f"{pol}\nSignal: " + ", ".join(parts) + "."
                items.append({
                    "id": f"suff_d1_s{s_idx:03d}_{vname}", "suite": "sufficiency_curriculum", "bucket": f"suff_{vname}",
                    "group_key": f"suff_src_{s_idx:03d}", "context": ctx, "question": q,
                    "options": opts, "option_descriptions": descs, "gold_index": gold,
                })

        elif dom == 2:
            # Domain 2: Engineering Experiment Promotion (mirrors evidence-sufficiency / fair-pair logic)
            opts = ["Enable_Feature", "Revert_Feature", "Keep_Monitoring", abs_lbl]
            descs = [
                "Enable feature: fair_paired_runs >= 3 AND latency_delta_ms <= -5 AND error_delta_pct <= 0.0.",
                "Revert feature: fair_paired_runs >= 3 AND (error_delta_pct > 0.20 OR latency_delta_ms > +25).",
                "Keep monitoring: fair_paired_runs >= 3 AND latency_delta_ms between -4 and +25 AND error_delta_pct <= 0.20.",
                "No fair paired runs or missing required telemetry; decision is unresolved.",
            ]
            pol = (
                "Experiment Gate Policy: Enable_Feature if fair_paired_runs >= 3 and latency_delta_ms <= -5 and error_delta_pct <= 0.0; "
                "Revert_Feature if fair_paired_runs >= 3 and (error_delta_pct > 0.20 or latency_delta_ms > +25); "
                "Keep_Monitoring if fair_paired_runs >= 3 and -4 <= latency_delta_ms <= +25 and error_delta_pct <= 0.20; "
                f"choose '{abs_lbl}' if fair_paired_runs < 3 or metrics are missing."
            )
            q = "Should this engineering lever be enabled, reverted, monitored, or left unresolved?"

            def eval_dom2(runs: int | None, lat_d: int | None, err_d: float | None) -> int:
                if runs is None or runs < 3 or lat_d is None or err_d is None:
                    return 3
                if err_d > 0.20 or lat_d > 25:
                    return 1
                if lat_d <= -5 and err_d <= 0.0:
                    return 0
                return 2

            cls = s_idx % 3
            if cls == 0:
                r_v, l_v, e_v = rng.randint(3, 8), rng.randint(-35, -6), round(rng.uniform(-0.05, 0.0), 2)
            elif cls == 1:
                r_v, l_v, e_v = rng.randint(3, 8), rng.randint(30, 95), round(rng.uniform(0.25, 1.40), 2)
            else:
                r_v, l_v, e_v = rng.randint(3, 8), rng.randint(-2, 18), round(rng.uniform(0.0, 0.12), 2)

            variants = [
                ("complete", r_v, l_v, e_v, True),
                ("crit_removed", 0, l_v, e_v, True),  # 0 fair paired runs -> unresolved!
                ("irrel_removed", r_v, l_v, e_v, False),
            ]
            for vname, rn, ld, ed, keep_meta in variants:
                gold = eval_dom2(rn, ld, ed)
                parts = []
                if keep_meta:
                    parts.append(f"owner={meta_owner}, branch=exp-{s_idx:02d}")
                parts.append(f"fair_paired_runs={rn}")
                parts.append(f"latency_delta_ms={ld:+d}")
                parts.append(f"error_delta_pct={ed:+.2f}%")
                ctx = f"{pol}\nExperiment Summary: " + ", ".join(parts) + "."
                items.append({
                    "id": f"suff_d2_s{s_idx:03d}_{vname}", "suite": "sufficiency_curriculum", "bucket": f"suff_{vname}",
                    "group_key": f"suff_src_{s_idx:03d}", "context": ctx, "question": q,
                    "options": opts, "option_descriptions": descs, "gold_index": gold,
                })

        else:
            # Domain 3: Forecast & Equipment Safety Selection (mirrors confidence-gated physical setup)
            opts = ["Config_High_Grip", "Config_Balanced", "Config_Low_Drag", abs_lbl]
            descs = [
                "Use high-grip config when forecast_confidence >= 0.50 and moisture >= 0.60.",
                "Use balanced config when forecast_confidence >= 0.50 and 0.25 <= moisture < 0.60.",
                "Use low-drag config when forecast_confidence >= 0.50 and moisture < 0.25.",
                "Forecast confidence < 0.50 or moisture sensor missing; abstain / unresolved.",
            ]
            pol = (
                "Setup Policy: Config_High_Grip if forecast_confidence >= 0.50 and moisture >= 0.60; "
                "Config_Balanced if forecast_confidence >= 0.50 and 0.25 <= moisture < 0.60; "
                "Config_Low_Drag if forecast_confidence >= 0.50 and moisture < 0.25; "
                f"choose '{abs_lbl}' if forecast_confidence < 0.50 or moisture is missing."
            )
            q = "Which physical configuration should be selected?"

            def eval_dom3(conf: float | None, moist: float | None) -> int:
                if conf is None or moist is None or conf < 0.50:
                    return 3
                if moist >= 0.60:
                    return 0
                if moist >= 0.25:
                    return 1
                return 2

            cls = s_idx % 3
            c_v = round(rng.uniform(0.62, 0.95), 2)
            if cls == 0:
                m_v = round(rng.uniform(0.65, 0.95), 2)
            elif cls == 1:
                m_v = round(rng.uniform(0.28, 0.55), 2)
            else:
                m_v = round(rng.uniform(0.02, 0.20), 2)

            variants = [
                ("complete", c_v, m_v, True),
                ("crit_removed", round(rng.uniform(0.18, 0.42), 2), m_v, True),  # low forecast confidence -> abstain!
                ("irrel_removed", c_v, m_v, False),
            ]
            for vname, cf, ms, keep_meta in variants:
                gold = eval_dom3(cf, ms)
                parts = []
                if keep_meta:
                    parts.append(f"venue={meta_zone}, crew={meta_owner}")
                parts.append(f"forecast_confidence={cf}")
                parts.append(f"moisture={ms}")
                ctx = f"{pol}\nConditions: " + ", ".join(parts) + "."
                items.append({
                    "id": f"suff_d3_s{s_idx:03d}_{vname}", "suite": "sufficiency_curriculum", "bucket": f"suff_{vname}",
                    "group_key": f"suff_src_{s_idx:03d}", "context": ctx, "question": q,
                    "options": opts, "option_descriptions": descs, "gold_index": gold,
                })

    return items


def _make_wrapper_block(rng: random.Random, target_words: int, family: str = "train") -> str:
    services = _TRAIN_SERVICES if family == "train" else _DEV_SERVICES
    regions = _TRAIN_REGIONS if family == "train" else _DEV_REGIONS
    header = (
        "Routine Background Telemetry (healthy unrelated nodes):"
        if family == "train"
        else "Cluster Heartbeat Stream (nominal background services):"
    )
    lines = [header]
    words = len(header.split())
    idx = 0
    while words < target_words:
        idx += 1
        svc = rng.choice(services)
        reg = rng.choice(regions)
        if family == "train":
            ln = (
                f"[OK] ts=09:{idx%60:02d}:{(idx*11)%60:02d} node={svc}-{reg}-{idx%16:02d} "
                f"cpu={rng.randint(10,38)}% mem={rng.randint(25,58)}% err=0.00% lat={rng.randint(6,28)}ms state=nominal"
            )
        else:
            ln = (
                f"2026-05-0{1+(idx%8)}T14:{idx%60:02d}:{(idx*7)%60:02d}Z HEALTHY svc={svc} zone={reg} "
                f"load={round(rng.uniform(0.1,0.9),2)} rps={rng.randint(80,3200)} p50={rng.randint(5,30)}ms status=ok"
            )
        lines.append(ln)
        words += len(ln.split())
    return "\n".join(lines)


def wrap_with_distractors(item: dict[str, Any], rng: random.Random, family: str = "train", total_words: int | None = None) -> dict[str, Any]:
    it = dict(item)
    n_words = total_words if total_words is not None else rng.choice([250, 550, 900])
    top = _make_wrapper_block(rng, n_words // 2, family=family)
    bot = _make_wrapper_block(rng, n_words - n_words // 2, family=family)
    pos = rng.choice(["middle", "top", "bottom"])
    if pos == "middle":
        it["context"] = f"{top}\n\n{item['context']}\n\n{bot}"
    elif pos == "top":
        it["context"] = f"{item['context']}\n\n{top}\n{bot}"
    else:
        it["context"] = f"{top}\n{bot}\n\n{item['context']}"
    return it


def paraphrase_item(item: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    it = dict(item)
    ctx = it["context"]
    q = it["question"]
    opts = list(it["options"])
    descs = list(it.get("option_descriptions") or opts)

    if q in _Q_PARAPHRASES and rng.random() < 0.85:
        q = rng.choice(_Q_PARAPHRASES[q])
    if rng.random() < 0.80:
        opts = [o.replace("_", " ") for o in opts]

    replacements = [
        ("Runbook Policy:", rng.choice(["Standard Operating Rules:", "Operational Rules:", "Triage Policy:"])),
        ("Release Gate Policy:", rng.choice(["Deployment Policy:", "Rollout Rules:", "Gate Criteria:"])),
        ("Routing Policy:", rng.choice(["Team Ownership Rules:", "Assignment Policy:", "Queue Rules:"])),
        ("Incident Alert:", rng.choice(["Observed Production Signal:", "Active Alert:", "Telemetry Signal:"])),
        ("Release Candidate:", rng.choice(["Pending Build Summary:", "Canary Report:", "Deployment Candidate:"])),
        ("Build Selection Policy:", rng.choice(["Car Setup Policy:", "Setup Selection Rules:"])),
    ]
    for old, new in replacements:
        if old in ctx:
            ctx = ctx.replace(old, new)

    if rng.random() < 0.40:
        ctx = "Decision Context Brief:\n" + ctx

    it["context"] = ctx
    it["question"] = q
    it["options"] = opts
    it["option_descriptions"] = descs
    return it


def _tokenize_salience(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if len(t) >= 3 and t not in _STOPWORDS]


def _line_template_signature(line: str) -> str:
    """Used ONLY for boilerplate template detection; raw lines returned to the encoder stay unmasked."""
    s = _MASK_TS_RE.sub("<TS>", line.lower())
    s = _MASK_NUM_HEX_RE.sub("<N>", s)
    s = re.sub(r"([a-z_]+=)[a-z0-9_-]+", r"\1<V>", s)
    return " ".join(s.split())


def format_query(question: str, context: str) -> str:
    return f"Question: {question}\nContext: {context}"


def format_option(opt: str, desc: str) -> str:
    if desc and desc.strip() and desc.strip() != opt.strip():
        return f"{opt}: {desc}"
    return opt


def pack_evidence(
    context: str,
    question: str,
    options: list[str],
    descs: list[str],
    tokenizer: Any,
    max_length: int = 512,
) -> str:
    """Shared tokenizer-budgeted line/block evidence packer used identically in train, cal, and inference."""
    opt_texts = [format_option(o, d) for o, d in zip(options, descs)]
    max_opt_toks = max(len(tokenizer.encode(ot, add_special_tokens=False)) for ot in opt_texts)
    q_prefix_toks = len(tokenizer.encode(f"Question: {question}\nContext: ", add_special_tokens=False))
    ctx_budget = max(64, max_length - max_opt_toks - q_prefix_toks - 8)

    ctx_toks = tokenizer.encode(context, add_special_tokens=False)
    if len(ctx_toks) <= ctx_budget:
        return context

    raw_lines = [ln.strip() for ln in context.splitlines() if ln.strip()]
    if len(raw_lines) <= 1:
        raw_lines = [s.strip() for s in re.split(r"(?<=[.;])\s+", context) if s.strip()]
    if len(raw_lines) <= 1:
        return tokenizer.decode(ctx_toks[:ctx_budget], skip_special_tokens=True)

    sigs = [_line_template_signature(ln) for ln in raw_lines]
    sig_counts = Counter(sigs)
    line_toks = [set(_tokenize_salience(ln)) for ln in raw_lines]
    df: Counter[str] = Counter()
    for toks in line_toks:
        for t in toks:
            df[t] += 1
    n_lines = len(raw_lines)
    query_toks = set(_tokenize_salience(question + " " + " ".join(options) + " " + " ".join(descs)))

    line_scores: list[tuple[float, int, int]] = []
    for idx, (ln, toks, sig) in enumerate(zip(raw_lines, line_toks, sigs)):
        rep_penalty = 1.0 / (1.0 + 1.5 * math.log(float(sig_counts[sig])))
        q_score = sum(math.log((n_lines + 1.0) / (df[t] + 0.5)) for t in (toks & query_toks))
        info_score = (
            sum(math.log((n_lines + 1.0) / (df[t] + 0.5)) for t in toks) / math.sqrt(len(toks))
            if toks
            else 0.0
        )
        uniq_bonus = 1.5 if sig_counts[sig] == 1 else 0.0
        score = (q_score + 0.45 * info_score + uniq_bonus) * rep_penalty
        tok_len = len(tokenizer.encode(ln, add_special_tokens=False)) + 1
        line_scores.append((score, idx, tok_len))

    ranked = sorted(line_scores, key=lambda x: (-x[0], x[1]))
    chosen_indices: list[int] = []
    used = 0
    for score, idx, tlen in ranked:
        if used + tlen <= ctx_budget:
            chosen_indices.append(idx)
            used += tlen
        elif not chosen_indices:
            chosen_indices.append(idx)
            break

    chosen_indices.sort()
    return "\n".join(raw_lines[i] for i in chosen_indices)


def score_item_logits(
    model: torch.nn.Module,
    tokenizer: Any,
    item: dict[str, Any],
    device: torch.device,
    max_length: int = 512,
    use_retrieval: bool = True,
) -> torch.Tensor:
    options = list(item["options"])
    descs = list(item.get("option_descriptions") or options)
    opt_texts = [format_option(o, d) for o, d in zip(options, descs)]

    ctx = (
        pack_evidence(item["context"], item["question"], options, descs, tokenizer, max_length=max_length)
        if use_retrieval
        else item["context"]
    )
    q_text = format_query(item["question"], ctx)
    enc = tokenizer(
        [q_text] * len(opt_texts),
        opt_texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    logits = model(**enc).logits.squeeze(-1)
    if logits.ndim == 0:
        logits = logits.unsqueeze(0)
    return logits


def compute_decision_loss(logits: torch.Tensor, gold_idx: int, label_smoothing: float = 0.04, margin_weight: float = 0.20) -> torch.Tensor:
    target = torch.tensor([gold_idx], dtype=torch.long, device=logits.device)
    ce = F.cross_entropy(logits.unsqueeze(0), target, label_smoothing=label_smoothing)
    if logits.numel() < 2 or margin_weight <= 0.0:
        return ce
    gold_score = logits[gold_idx]
    mask = torch.ones_like(logits, dtype=torch.bool)
    mask[gold_idx] = False
    hardest_neg = logits[mask].max()
    return ce + margin_weight * F.relu(1.0 - (gold_score - hardest_neg))


def symmetric_kl_loss(logits_a: torch.Tensor, logits_b: torch.Tensor) -> torch.Tensor:
    p = F.log_softmax(logits_a, dim=-1)
    q = F.log_softmax(logits_b, dim=-1)
    kl_pq = F.kl_div(p, q.exp(), reduction="sum")
    kl_qp = F.kl_div(q, p.exp(), reduction="sum")
    return 0.5 * (kl_pq + kl_qp)


def evaluate_split(
    model: torch.nn.Module,
    tokenizer: Any,
    items: list[dict[str, Any]],
    device: torch.device,
    use_amp: bool = False,
    wrap_dev_family: bool = False,
    seed: int = 1234,
) -> dict[str, Any]:
    model.eval()
    by_suite_correct: Counter[str] = Counter()
    by_suite_total: Counter[str] = Counter()
    nlls: list[float] = []
    raw_logits_list: list[np.ndarray] = []
    golds: list[int] = []
    rng = random.Random(seed)

    with torch.inference_mode():
        for raw_it in items:
            it = wrap_with_distractors(raw_it, rng, family="dev", total_words=700) if wrap_dev_family else raw_it
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = score_item_logits(model, tokenizer, it, device, use_retrieval=True)
            logits_f32 = logits.float()
            log_probs = F.log_softmax(logits_f32, dim=-1)
            gold = int(it["gold_index"])
            pred = int(torch.argmax(logits_f32).item())
            suite = it["suite"]
            by_suite_total[suite] += 1
            if pred == gold:
                by_suite_correct[suite] += 1
            nlls.append(float(-log_probs[gold].item()))
            raw_logits_list.append(logits_f32.detach().cpu().numpy())
            golds.append(gold)

    suite_acc = {s: by_suite_correct[s] / max(1, by_suite_total[s]) for s in sorted(by_suite_total)}
    overall_acc = sum(by_suite_correct.values()) / max(1, sum(by_suite_total.values()))
    macro_suite_acc = float(np.mean(list(suite_acc.values()))) if suite_acc else 0.0
    return {
        "overall_acc": float(overall_acc),
        "macro_suite_acc": float(macro_suite_acc),
        "mean_nll": float(np.mean(nlls)) if nlls else 0.0,
        "suite_acc": suite_acc,
        "raw_logits": raw_logits_list,
        "golds": golds,
    }


def fit_temperature(raw_logits_list: list[np.ndarray], golds: list[int]) -> tuple[float, float]:
    best_t = 1.0
    best_nll = float("inf")
    for t in np.geomspace(0.25, 10.0, 140):
        total = 0.0
        for lg, g in zip(raw_logits_list, golds):
            scaled = lg.astype(np.float64) / float(t)
            scaled = scaled - np.max(scaled)
            log_p = float(scaled[g]) - math.log(float(np.sum(np.exp(scaled))))
            total -= log_p
        mean_nll = total / max(1, len(golds))
        if mean_nll < best_nll:
            best_nll = mean_nll
            best_t = float(t)
    return float(best_t), float(best_nll)


def average_state_dicts(states: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    if len(states) == 1:
        return states[0]
    avg: dict[str, torch.Tensor] = {}
    for k in states[0]:
        if states[0][k].is_floating_point():
            avg[k] = sum(s[k].float() for s in states) / float(len(states))
            avg[k] = avg[k].to(states[0][k].dtype)
        else:
            avg[k] = states[0][k].clone()
    return avg


def train_single_fold(
    *,
    fold_idx: int,
    base_model: str,
    train_items: list[dict[str, Any]],
    val_items: list[dict[str, Any]],
    suff_train: list[dict[str, Any]],
    suff_dev: list[dict[str, Any]],
    out_dir: Path,
    device: torch.device,
    use_amp: bool,
    epochs_a: int,
    epochs_b: int,
    lr_a: float,
    lr_b: float,
    weight_decay: float,
    grad_accum: int,
    kl_weight: float,
    wrap_add_prob: float,
    seed: int,
) -> dict[str, Any]:
    random.seed(seed + fold_idx * 97)
    np.random.seed(seed + fold_idx * 97)
    torch.manual_seed(seed + fold_idx * 97)

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSequenceClassification.from_pretrained(base_model)
    model.to(device)

    suite_counts = Counter(it["suite"] for it in train_items)
    max_cnt = max(suite_counts.values())
    balanced_train: list[dict[str, Any]] = []
    for it in train_items:
        reps = max(1, round(max_cnt / suite_counts[it["suite"]]))
        balanced_train.extend([it] * reps)

    zs_clean = evaluate_split(model, tokenizer, val_items, device, use_amp=use_amp, wrap_dev_family=False)
    zs_wrap = evaluate_split(model, tokenizer, val_items, device, use_amp=use_amp, wrap_dev_family=True)
    zs_suff = evaluate_split(model, tokenizer, suff_dev, device, use_amp=use_amp, wrap_dev_family=False)
    print(
        f"[Fold {fold_idx}] Ep 0 (zero-shot) | clean={zs_clean['overall_acc']*100:.1f}% "
        f"| wrapped={zs_wrap['overall_acc']*100:.1f}% | suff_dev={zs_suff['overall_acc']*100:.1f}% "
        f"| macro={zs_clean['macro_suite_acc']*100:.1f}% | nll={zs_clean['mean_nll']:.3f}"
    )

    # ---------------- Stage A: Clean Preservation + Additive Distractor Retention ----------------
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr_a, weight_decay=weight_decay)
    total_steps_a = max(1, (len(balanced_train) * epochs_a) // grad_accum)
    warmup_a = max(1, int(total_steps_a * 0.10))

    def lr_lambda_a(step: int) -> float:
        if step < warmup_a:
            return float(step + 1) / float(warmup_a)
        prog = float(step - warmup_a) / float(max(1, total_steps_a - warmup_a))
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda_a)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    top_ckpts_a: list[tuple[float, dict[str, torch.Tensor]]] = []
    history: list[dict[str, Any]] = []
    rng = random.Random(seed + fold_idx * 313)

    for ep in range(1, epochs_a + 1):
        ep_t0 = time.perf_counter()
        model.train()
        order = list(balanced_train)
        rng.shuffle(order)
        optimizer.zero_grad()
        losses: list[float] = []

        for step_idx, raw_it in enumerate(order, start=1):
            gold = int(raw_it["gold_index"])
            do_wrap_add = (rng.random() < wrap_add_prob)
            do_para = (kl_weight > 0.0 and rng.random() < 0.35)

            step_loss_val = 0.0
            # 1. Clean supervised target (always weight 1.0) — backward immediately to cut peak T4 VRAM 3x
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits_clean = score_item_logits(model, tokenizer, raw_it, device, use_retrieval=True)
                loss_clean = compute_decision_loss(logits_clean, gold) / grad_accum
            logits_clean_det = logits_clean.detach()
            scaler.scale(loss_clean).backward()
            step_loss_val += float(loss_clean.item() * grad_accum)

            # 2. Additive distractor-wrapped view (25%, weight 0.5) — separate backward pass
            if do_wrap_add:
                wrapped_it = wrap_with_distractors(raw_it, rng, family="train")
                with torch.amp.autocast("cuda", enabled=use_amp):
                    logits_wrap = score_item_logits(model, tokenizer, wrapped_it, device, use_retrieval=True)
                    loss_wrap = (0.50 * compute_decision_loss(logits_wrap, gold)) / grad_accum
                scaler.scale(loss_wrap).backward()
                step_loss_val += float(loss_wrap.item() * grad_accum)

            # 3. Paraphrase-consistency view (35%, weight 0.5 CE + KL) — separate backward pass
            if do_para:
                para_it = paraphrase_item(raw_it, rng)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    logits_para = score_item_logits(model, tokenizer, para_it, device, use_retrieval=True)
                    loss_para = (0.50 * compute_decision_loss(logits_para, gold) + kl_weight * symmetric_kl_loss(logits_clean_det, logits_para)) / grad_accum
                scaler.scale(loss_para).backward()
                step_loss_val += float(loss_para.item() * grad_accum)

            losses.append(step_loss_val)

            if step_idx % grad_accum == 0 or step_idx == len(order):
                prev_scale = scaler.get_scale() if use_amp else 1.0
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                if (not use_amp) or (scaler.get_scale() >= prev_scale):
                    scheduler.step()
                optimizer.zero_grad()

        val_clean = evaluate_split(model, tokenizer, val_items, device, use_amp=use_amp, wrap_dev_family=False)
        val_wrap = evaluate_split(model, tokenizer, val_items, device, use_amp=use_amp, wrap_dev_family=True)
        ep_sec = time.perf_counter() - ep_t0
        score = 0.65 * val_clean["macro_suite_acc"] + 0.35 * val_wrap["macro_suite_acc"] - 0.05 * val_clean["mean_nll"]
        row_log = {
            "stage": "A", "epoch": ep,
            "train_loss": round(float(np.mean(losses)), 4),
            "val_clean_acc": round(val_clean["overall_acc"], 4),
            "val_wrapped_acc": round(val_wrap["overall_acc"], 4),
            "val_macro": round(val_clean["macro_suite_acc"], 4),
            "val_nll": round(val_clean["mean_nll"], 4),
            "val_suites": {k: round(v, 4) for k, v in val_clean["suite_acc"].items()},
            "sec": round(ep_sec, 1),
        }
        history.append(row_log)
        print(
            f"[Fold {fold_idx} | Stage A] Ep {ep}/{epochs_a} ({ep_sec:.1f}s) | loss={row_log['train_loss']:.4f} "
            f"| clean={val_clean['overall_acc']*100:.1f}% | wrapped={val_wrap['overall_acc']*100:.1f}% "
            f"| macro={val_clean['macro_suite_acc']*100:.1f}% | nll={val_clean['mean_nll']:.3f} "
            f"| suites={ {k: round(v*100,1) for k,v in val_clean['suite_acc'].items()} }"
        )
        if len(top_ckpts_a) < 2 or score > top_ckpts_a[-1][0]:
            state_copy = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            top_ckpts_a.append((score, state_copy))
            top_ckpts_a.sort(key=lambda x: x[0], reverse=True)
            top_ckpts_a = top_ckpts_a[:2]

    # Pick Stage A best (Top-2 SWA vs Best Single)
    swa_state = average_state_dicts([st for _, st in top_ckpts_a])
    model.load_state_dict(swa_state)
    swa_clean = evaluate_split(model, tokenizer, val_items, device, use_amp=use_amp, wrap_dev_family=False)
    swa_wrap = evaluate_split(model, tokenizer, val_items, device, use_amp=use_amp, wrap_dev_family=True)
    swa_score = 0.65 * swa_clean["macro_suite_acc"] + 0.35 * swa_wrap["macro_suite_acc"] - 0.05 * swa_clean["mean_nll"]
    if swa_score >= top_ckpts_a[0][0]:
        stage_a_state = swa_state
        stage_a_clean, stage_a_wrap = swa_clean, swa_wrap
    else:
        stage_a_state = top_ckpts_a[0][1]
        model.load_state_dict(stage_a_state)
        stage_a_clean = evaluate_split(model, tokenizer, val_items, device, use_amp=use_amp, wrap_dev_family=False)
        stage_a_wrap = evaluate_split(model, tokenizer, val_items, device, use_amp=use_amp, wrap_dev_family=True)
    stage_a_suff = evaluate_split(model, tokenizer, suff_dev, device, use_amp=use_amp, wrap_dev_family=False)
    print(
        f"[Fold {fold_idx} | Stage A Best] clean={stage_a_clean['overall_acc']*100:.1f}% "
        f"| wrapped={stage_a_wrap['overall_acc']*100:.1f}% | suff_dev={stage_a_suff['overall_acc']*100:.1f}%"
    )

    # ---------------- Stage B: Verified Evidence-Sufficiency Curriculum (75% real replay + 25% counterfactuals) ----------------
    best_final_state = stage_a_state
    best_final_clean, best_final_wrap, best_final_suff = stage_a_clean, stage_a_wrap, stage_a_suff
    best_final_score = (
        0.55 * stage_a_clean["macro_suite_acc"]
        + 0.30 * stage_a_wrap["macro_suite_acc"]
        + 0.15 * stage_a_suff["overall_acc"]
        - 0.05 * stage_a_clean["mean_nll"]
    )
    used_stage_b = False

    if epochs_b > 0 and suff_train:
        model.load_state_dict(stage_a_state)
        # Build 75% real replay + 25% verified sufficiency counterfactuals
        n_suff_target = max(32, min(len(suff_train), len(balanced_train) // 3))
        opt_b = torch.optim.AdamW(model.parameters(), lr=lr_b, weight_decay=weight_decay)
        total_steps_b = max(1, ((len(balanced_train) + n_suff_target) * epochs_b) // grad_accum)
        sched_b = torch.optim.lr_scheduler.CosineAnnealingLR(opt_b, T_max=total_steps_b, eta_min=lr_b * 0.2)
        scaler_b = torch.amp.GradScaler("cuda", enabled=use_amp)

        for ep in range(1, epochs_b + 1):
            ep_t0 = time.perf_counter()
            model.train()
            suff_sample = rng.sample(suff_train, k=n_suff_target)
            order = list(balanced_train) + list(suff_sample)
            rng.shuffle(order)
            opt_b.zero_grad()
            losses_b: list[float] = []

            for step_idx, raw_it in enumerate(order, start=1):
                gold = int(raw_it["gold_index"])
                do_wrap_add = (rng.random() < wrap_add_prob)
                step_loss_b = 0.0
                with torch.amp.autocast("cuda", enabled=use_amp):
                    logits_clean = score_item_logits(model, tokenizer, raw_it, device, use_retrieval=True)
                    loss_c = compute_decision_loss(logits_clean, gold) / grad_accum
                scaler_b.scale(loss_c).backward()
                step_loss_b += float(loss_c.item() * grad_accum)
                if do_wrap_add:
                    wrapped_it = wrap_with_distractors(raw_it, rng, family="train")
                    with torch.amp.autocast("cuda", enabled=use_amp):
                        logits_wrap = score_item_logits(model, tokenizer, wrapped_it, device, use_retrieval=True)
                        loss_w = (0.50 * compute_decision_loss(logits_wrap, gold)) / grad_accum
                    scaler_b.scale(loss_w).backward()
                    step_loss_b += float(loss_w.item() * grad_accum)
                losses_b.append(step_loss_b)

                if step_idx % grad_accum == 0 or step_idx == len(order):
                    prev_scale = scaler_b.get_scale() if use_amp else 1.0
                    scaler_b.unscale_(opt_b)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler_b.step(opt_b)
                    scaler_b.update()
                    if (not use_amp) or (scaler_b.get_scale() >= prev_scale):
                        sched_b.step()
                    opt_b.zero_grad()

            val_clean = evaluate_split(model, tokenizer, val_items, device, use_amp=use_amp, wrap_dev_family=False)
            val_wrap = evaluate_split(model, tokenizer, val_items, device, use_amp=use_amp, wrap_dev_family=True)
            val_suff = evaluate_split(model, tokenizer, suff_dev, device, use_amp=use_amp, wrap_dev_family=False)
            ep_sec = time.perf_counter() - ep_t0
            sc_b = (
                0.55 * val_clean["macro_suite_acc"]
                + 0.30 * val_wrap["macro_suite_acc"]
                + 0.15 * val_suff["overall_acc"]
                - 0.05 * val_clean["mean_nll"]
            )
            history.append({
                "stage": "B", "epoch": ep,
                "train_loss": round(float(np.mean(losses_b)), 4),
                "val_clean_acc": round(val_clean["overall_acc"], 4),
                "val_wrapped_acc": round(val_wrap["overall_acc"], 4),
                "val_suff_acc": round(val_suff["overall_acc"], 4),
                "val_macro": round(val_clean["macro_suite_acc"], 4),
                "val_nll": round(val_clean["mean_nll"], 4),
                "val_suites": {k: round(v, 4) for k, v in val_clean["suite_acc"].items()},
                "sec": round(ep_sec, 1),
            })
            print(
                f"[Fold {fold_idx} | Stage B] Ep {ep}/{epochs_b} ({ep_sec:.1f}s) | loss={float(np.mean(losses_b)):.4f} "
                f"| clean={val_clean['overall_acc']*100:.1f}% | wrapped={val_wrap['overall_acc']*100:.1f}% "
                f"| suff_dev={val_suff['overall_acc']*100:.1f}% | nll={val_clean['mean_nll']:.3f}"
            )
            # Only adopt Stage B if it improves combined dev score AND does not regress clean macro > 2%
            if sc_b > best_final_score and val_clean["macro_suite_acc"] >= stage_a_clean["macro_suite_acc"] - 0.02:
                best_final_score = sc_b
                best_final_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                best_final_clean, best_final_wrap, best_final_suff = val_clean, val_wrap, val_suff
                used_stage_b = True

    model.load_state_dict(best_final_state)
    print(
        f"[Fold {fold_idx} | Final] stage={'B' if used_stage_b else 'A'} | clean={best_final_clean['overall_acc']*100:.1f}% "
        f"| wrapped={best_final_wrap['overall_acc']*100:.1f}% | suff_dev={best_final_suff['overall_acc']*100:.1f}% "
        f"| suites={ {k: round(v*100,1) for k,v in best_final_clean['suite_acc'].items()} }"
    )

    fold_dir = out_dir / f"fold_{fold_idx}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(fold_dir)
    tokenizer.save_pretrained(fold_dir)

    param_count = sum(p.numel() for p in model.parameters())
    del model, optimizer, scheduler, scaler, top_ckpts_a, swa_state, best_final_state
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "fold": fold_idx,
        "fold_dir": str(fold_dir.name),
        "param_count": int(param_count),
        "used_stage_b": used_stage_b,
        "val_clean_acc": round(best_final_clean["overall_acc"], 4),
        "val_wrapped_acc": round(best_final_wrap["overall_acc"], 4),
        "val_suff_dev_acc": round(best_final_suff["overall_acc"], 4),
        "val_macro": round(best_final_clean["macro_suite_acc"], 4),
        "val_nll": round(best_final_clean["mean_nll"], 4),
        "val_suites": {k: round(v, 4) for k, v in best_final_clean["suite_acc"].items()},
        "oof_logits": best_final_clean["raw_logits"],
        "oof_golds": best_final_clean["golds"],
        "history": history,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train balanced 3-fold self-jev-minilm (Stage A + Stage B sufficiency curriculum).")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--epochs-a", type=int, default=7)
    parser.add_argument("--epochs-b", type=int, default=2)
    parser.add_argument("--lr-a", type=float, default=2e-5)
    parser.add_argument("--lr-b", type=float, default=5e-6)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--kl-weight", type=float, default=0.35)
    parser.add_argument("--wrap-add-prob", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=20260401)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    use_amp = device.type == "cuda"

    t0 = time.perf_counter()
    pool = load_nontest_pool()
    folds = make_grouped_stratified_folds(pool, n_folds=max(2, args.folds), seed=args.seed)

    # Disjoint train (64 sources x 3 = 192) vs dev (16 sources x 3 = 48) verified sufficiency curriculum
    suff_train = generate_sufficiency_curriculum(n_sources=64, seed=args.seed + 101)
    suff_dev = generate_sufficiency_curriculum(n_sources=16, seed=args.seed + 909)

    print(f"Loaded {len(pool)} strictly non-test items (situation-grouped) | base={args.base_model} | device={device}")
    print(f"  fold sizes: {[len(f) for f in folds]} | suff_train={len(suff_train)} | suff_dev={len(suff_dev)}")
    for fi, f in enumerate(folds):
        print(f"  Fold {fi} suites: {dict(Counter(it['suite'] for it in f))}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    oof_logits: list[np.ndarray] = []
    oof_golds: list[int] = []
    fold_reports: list[dict[str, Any]] = []
    fold_names: list[str] = []
    param_count = 0

    for fi in range(args.folds):
        val_items = folds[fi]
        train_items = [it for j, f in enumerate(folds) if j != fi for it in f]
        res = train_single_fold(
            fold_idx=fi,
            base_model=args.base_model,
            train_items=train_items,
            val_items=val_items,
            suff_train=suff_train,
            suff_dev=suff_dev,
            out_dir=args.out_dir,
            device=device,
            use_amp=use_amp,
            epochs_a=args.epochs_a,
            epochs_b=args.epochs_b,
            lr_a=args.lr_a,
            lr_b=args.lr_b,
            weight_decay=args.weight_decay,
            grad_accum=args.grad_accum,
            kl_weight=args.kl_weight,
            wrap_add_prob=args.wrap_add_prob,
            seed=args.seed,
        )
        oof_logits.extend(res["oof_logits"])
        oof_golds.extend(res["oof_golds"])
        fold_names.append(res["fold_dir"])
        param_count = res["param_count"]
        fold_reports.append({k: v for k, v in res.items() if k not in ("oof_logits", "oof_golds")})

    oof_correct = sum(int(np.argmax(lg)) == g for lg, g in zip(oof_logits, oof_golds))
    oof_acc = float(oof_correct / max(1, len(oof_golds)))
    best_t, best_oof_nll = fit_temperature(oof_logits, oof_golds)
    print(f"\nOOF ({len(oof_golds)} real non-test items) | OOF acc={oof_acc*100:.1f}% | fitted T={best_t:.4f} | OOF NLL={best_oof_nll:.4f}")

    (args.out_dir / "temperature.json").write_text(
        json.dumps({"temperature": float(best_t), "oof_nll": float(best_oof_nll), "oof_acc": round(oof_acc, 4)}, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.out_dir / "ensemble_config.json").write_text(
        json.dumps(
            {
                "base_model": args.base_model,
                "architecture": "XLMRobertaForSequenceClassification",
                "is_lora": False,
                "max_length": 512,
                "use_pack_evidence": True,
                "folds": fold_names,
                "temperature": float(best_t),
                "param_count_per_fold": int(param_count),
                "total_ensemble_params": int(param_count * len(fold_names)),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    report = {
        "out_dir": str(args.out_dir),
        "base_model": args.base_model,
        "architecture": "XLMRobertaForSequenceClassification",
        "folds": int(args.folds),
        "fold_sizes": [len(f) for f in folds],
        "epochs_a": int(args.epochs_a),
        "epochs_b": int(args.epochs_b),
        "kl_weight": float(args.kl_weight),
        "wrap_add_prob": float(args.wrap_add_prob),
        "param_count_per_fold": int(param_count),
        "total_ensemble_params": int(param_count * len(fold_names)),
        "nontest_pool_size": len(pool),
        "oof_acc": round(oof_acc, 4),
        "oof_nll_after_temp": round(float(best_oof_nll), 4),
        "temperature": round(float(best_t), 4),
        "total_sec": round(float(time.perf_counter() - t0), 1),
        "fold_reports": fold_reports,
    }
    (args.out_dir / "training_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Saved balanced {args.folds}-fold self-jev-minilm to {args.out_dir} in {report['total_sec']}s")


if __name__ == "__main__":
    main()
