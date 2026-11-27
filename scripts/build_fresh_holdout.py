#!/usr/bin/env python3
"""Deterministic, Disjoint 600-Source Fresh Holdout Generator.

Generates 600 independent decision scenarios (150 per bucket) using completely
disjoint service names, queue names, candidate identifiers, and policy templates
from both the historical 313-item benchmark and the 1,200-item teacher non-test pool.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

OUT_PATH = Path("/Users/caius/Projects/self_jev/data/holdout/holdout_600.jsonl")
MANIFEST_PATH = Path("/Users/caius/Projects/self_jev/data/holdout/holdout_manifest.json")
HIST_TEST = Path("/Users/caius/Projects/self_jev/data/benchmarks/all_test.jsonl")
NONTEST_1200 = Path("/Users/caius/Projects/self_jev/data/teacher/merged_1200_nontest.jsonl")

SERVICES = [
    "aurora-gateway", "beacon-scheduler", "cobalt-ledger", "delta-ingestor",
    "ember-tokenizer", "falcon-router", "granite-indexer", "harbor-sync",
    "ion-aggregator", "jade-validator", "kestrel-proxy", "lantern-store",
    "mercury-dispatch", "nimbus-compiler", "onyx-verifier", "pulse-streamer",
    "quartz-planner", "ripple-broker", "sierra-cache", "topaz-worker",
]

WRAPPERS = [
    "Cluster maintenance window completed on zone-c with zero pod evictions.",
    "Periodic TLS rotation succeeded for internal-grpc-mesh (valid 89 days).",
    "Snapshot exporter uploaded 14.2 GB to cold-archive-bucket in 19s.",
    "Autoscaler held replica count steady at 16 across ap-southeast-1.",
    "CoreDNS cache hit ratio averaged 97.8% over the trailing 15 minutes.",
    "Dependency mirror sync finished cleanly with sha256 checksums intact.",
    "Background compaction reclaimed 4.6 GB on analytics-wal-volume.",
]


def wrap_text(rng: random.Random, core: str, is_wrapped: bool) -> str:
    if not is_wrapped:
        return core
    chatter = " ".join(rng.sample(WRAPPERS, 3))
    pos = rng.choice(["prefix", "suffix"])
    if pos == "prefix":
        return f"[Ops Telemetry Feed] {chatter}\n{core}"
    return f"{core}\n[Ops Telemetry Feed] {chatter}"


def gen_triage(rng: random.Random, idx: int, target_gold: int, is_wrapped: bool) -> dict[str, Any]:
    svc = SERVICES[(idx * 3 + 1) % len(SERVICES)] + f"-{100 + idx}"
    t_sev1 = rng.choice([4.0, 4.5, 5.5, 6.0])
    t_sev2_lo = rng.choice([1.2, 1.5, 1.8])
    t_p99 = rng.choice([1400, 1600, 1800])
    t_dur = rng.choice([8, 10, 12])
    t_sev3_lo = rng.choice([0.25, 0.30, 0.40])
    t_p99_s4 = rng.choice([300, 350, 400])
    t_age = rng.choice([4, 5, 6])

    env = "prod"
    age = rng.randint(1, t_age - 1)
    regions_down = 1
    dur = rng.randint(t_dur, t_dur + 8)
    conflict_note = ""

    if target_gold == 4:  # Abstain trap
        trap_kind = idx % 4
        if trap_kind == 0:
            env = "staging"
            err_5xx = round(t_sev1 + 2.5, 2)
            p99 = t_p99 + 400
        elif trap_kind == 1:
            age = t_age + rng.randint(3, 8)
            err_5xx = round(t_sev2_lo + 0.8, 2)
            p99 = t_p99 + 100
        elif trap_kind == 2:
            err_5xx = "null"
            p99 = 450
        else:
            err_5xx = round(t_sev1 + 3.0, 2)
            p99 = 210
            conflict_note = f" (Edge probes report 5xx={err_5xx}% while synthetic checks report 5xx=0.03%)"
    elif target_gold == 0:  # SEV1
        if idx % 3 == 0:
            regions_down = 2
            err_5xx = round(t_sev2_lo + 0.3, 2)
        else:
            err_5xx = round(t_sev1 + rng.choice([0.0, 0.7, 1.9]), 2)
        p99 = rng.randint(700, 2200)
    elif target_gold == 1:  # SEV2
        if idx % 3 == 0:
            err_5xx = round(t_sev3_lo - 0.1, 2)
            p99 = t_p99 + rng.choice([0, 250, 500])
            dur = t_dur + rng.choice([0, 4])
        else:
            err_5xx = round(t_sev2_lo + rng.choice([0.0, 0.5, 1.1]), 2)
            p99 = rng.randint(450, t_p99 - 100)
    elif target_gold == 2:  # SEV3
        err_5xx = round(t_sev3_lo + rng.choice([0.0, 0.25, 0.55]), 2)
        if idx % 3 == 0:
            # High latency but duration below SEV2 threshold
            p99 = t_p99 + 250
            dur = max(2, t_dur - 3)
        else:
            p99 = rng.randint(t_p99_s4 + 20, t_p99 - 100)
    else:  # SEV4
        err_5xx = round(max(0.01, t_sev3_lo - rng.choice([0.12, 0.18])), 2)
        p99 = rng.randint(80, t_p99_s4 - 30)

    policy = (
        f"Escalation Policy: P1_Critical if 5xx >= {t_sev1}% or regions_down >= 2; "
        f"P2_High if {t_sev2_lo}% <= 5xx < {t_sev1}% or (p99 >= {t_p99} ms for >= {t_dur} min); "
        f"P3_Normal if {t_sev3_lo}% <= 5xx < {t_sev2_lo}%; "
        f"P4_Info if 5xx < {t_sev3_lo}% and p99 < {t_p99_s4} ms; "
        f"Abstain if env != prod, telemetry_age > {t_age} min, metrics are null, or monitors conflict."
    )
    alert = (
        f"Incident Record: service={svc}, env={env}, regions_down={regions_down}, "
        f"5xx={err_5xx}{'%' if isinstance(err_5xx, float) else ''}, p99={p99} ms, "
        f"duration={dur} min, telemetry_age={age} min{conflict_note}."
    )
    opts = ["P1_Critical", "P2_High", "P3_Normal", "P4_Info", "Abstain / Escalate_Ambiguous"]
    descs = [
        f"Critical outage: 5xx >= {t_sev1}% or >= 2 regions down.",
        f"High severity: {t_sev2_lo}%-{t_sev1}% 5xx or p99 >= {t_p99} ms for >= {t_dur} min.",
        f"Normal ticket: {t_sev3_lo}%-{t_sev2_lo}% 5xx.",
        f"Healthy noise: 5xx < {t_sev3_lo}% and p99 < {t_p99_s4} ms.",
        f"Non-prod env, telemetry_age > {t_age} min, null metrics, or conflicting monitors.",
    ]
    return {
        "bucket": "incident_triage",
        "context_band": "wrapped" if is_wrapped else "clean",
        "context": wrap_text(rng, f"{policy}\n{alert}", is_wrapped),
        "question": "Which incident escalation tier applies?",
        "options": opts,
        "option_descriptions": descs,
        "gold_index": target_gold,
        "gold_label": opts[target_gold],
        "abstain_index": 4,
        "is_abstain_gold": target_gold == 4,
    }


def gen_gate(rng: random.Random, idx: int, target_gold: int, is_wrapped: bool) -> dict[str, Any]:
    svc = SERVICES[(idx * 5 + 2) % len(SERVICES)] + f"-rel-{idx}"
    min_req = rng.choice([6000, 8000, 10000])
    max_err = rng.choice([0.04, 0.05])
    rb_err = rng.choice([0.35, 0.40, 0.50])
    p99_ok = rng.choice([15, 20, 25])
    p99_rb = rng.choice([90, 110, 120])

    cves = 0
    secret = "false"
    leak = "false"
    tests = "pass"
    req_cnt: Any = min_req + rng.randint(1500, 6000)
    err_delta: Any = round(rng.choice([0.01, 0.02, max_err]), 2)
    p99_delta = rng.randint(3, p99_ok)
    fresh = "fresh"
    conflict = ""

    if target_gold == 4:
        trap = idx % 3
        if trap == 0:
            err_delta = "null"
            fresh = "partial"
        elif trap == 1:
            fresh = "stale_12m"
        else:
            conflict = f" (Monitor_X reports err_delta=+0.01%, Monitor_Y reports err_delta=+{rb_err + 0.25:.2f}%)"
    elif target_gold == 3:  # Block_Sec
        if idx % 2 == 0:
            cves = rng.choice([1, 2])
        else:
            secret = "true"
    elif target_gold == 2:  # Rollback
        mode = idx % 3
        if mode == 0:
            leak = "true"
        elif mode == 1:
            err_delta = round(rb_err + rng.choice([0.0, 0.18]), 2)
        else:
            p99_delta = p99_rb + rng.randint(10, 45)
    elif target_gold == 1:  # Hold
        if idx % 2 == 0:
            req_cnt = min_req - rng.randint(100, 1500)
        else:
            p99_delta = rng.randint(p99_ok + 6, p99_rb)
    # else target_gold == 0 -> Promote defaults

    policy = (
        f"Deployment Gate: Security_Halt if cves > 0 or secret_leak=true; "
        f"Immediate_Rollback if err_delta >= +{rb_err:.2f}% or p99_delta > {p99_rb} ms or oom_leak=true; "
        f"Extend_Canary if samples < {min_req} or ({p99_ok} < p99_delta <= {p99_rb}); "
        f"Ship_100pct if tests=pass, cves=0, secret_leak=false, oom_leak=false, samples >= {min_req}, "
        f"err_delta <= +{max_err:.2f}%, p99_delta <= {p99_ok} ms; Abstain if telemetry is null, stale, or conflicting."
    )
    cand = (
        f"Release Build: service={svc}, tests={tests}, cves={cves}, secret_leak={secret}, oom_leak={leak}, "
        f"samples={req_cnt}, err_delta={('+' + str(err_delta) + '%') if isinstance(err_delta, float) else err_delta}, "
        f"p99_delta=+{p99_delta} ms, telemetry={fresh}{conflict}."
    )
    opts = ["Ship_100pct", "Extend_Canary", "Immediate_Rollback", "Security_Halt", "Abstain / Escalate_Ambiguous"]
    descs = [
        f"All checks pass with samples >= {min_req}, err_delta <= +{max_err:.2f}%, p99_delta <= {p99_ok} ms.",
        f"Security/tests pass, but samples < {min_req} or p99_delta is +{p99_ok+1} to +{p99_rb} ms.",
        f"err_delta >= +{rb_err:.2f}%, p99_delta > {p99_rb} ms, or oom_leak=true.",
        "cves > 0 or secret_leak=true.",
        "Telemetry is null, stale, or monitors disagree.",
    ]
    return {
        "bucket": "release_gate",
        "context_band": "wrapped" if is_wrapped else "clean",
        "context": wrap_text(rng, f"{policy}\n{cand}", is_wrapped),
        "question": "Which deployment gate decision should execute?",
        "options": opts,
        "option_descriptions": descs,
        "gold_index": target_gold,
        "gold_label": opts[target_gold],
        "abstain_index": 4,
        "is_abstain_gold": target_gold == 4,
    }


ROUTING_QUEUES = [
    ("Identity_Access", "SAML2 assertion signature errors, OIDC token exchange 401s, and RBAC permission claims.", [
        "Enterprise tenant receives 401 invalid_assertion_signature during OIDC callback after IdP cert rotation.",
        "Service account API keys with admin_read scope fail RBAC policy checks on /v3/workspaces.",
        "SAML SSO login loop rejects users due to missing NameID format claim in IdP metadata.",
    ]),
    ("Ledger_Billing", "Invoice proration mismatches, VAT/GST tax rules, and failed Stripe payment retries.", [
        "Monthly plan upgrade generated duplicate prorated invoice lines and failed Stripe charge webhook.",
        "EU VAT reverse-charge exemption failed to apply on enterprise annual renewal invoice.",
        "Refund ledger entry stuck in pending state after credit card chargeback reversal.",
    ]),
    ("Edge_Networking", "Anycast DNS propagation, expired edge TLS certificates, and WAF rate-limit rules.", [
        "Tokyo edge PoP serves expired TLS 1.1 chain causing ERR_SSL_CIPHER_MISMATCH on static assets.",
        "GeoDNS CNAME flattening returns stale IPv6 AAAA record for ap-southeast-1 ingress.",
        "WAF managed rule falsely blocks legitimate GraphQL introspection requests with HTTP 403.",
    ]),
    ("Stream_Ingestion", "Kafka consumer lag, Iceberg schema evolution failures, and Airflow ETL backfills.", [
        "Hourly Airflow rollup job crashed with Iceberg schema cast error on partition dt=2026-11-20.",
        "Kafka consumer group event-sink-v4 fell 1.8M offsets behind after broker partition rebalance.",
        "Parquet compaction pipeline failed with incompatible decimal precision on orders_fact.",
    ]),
    ("Telemetry_Ops", "OpenTelemetry collector drops, Prometheus high cardinality, and Jaeger span loss.", [
        "OTel collector pods drop 40% of trace batches with memory_limiter_refused_data.",
        "Prometheus TSDB scrape fails on billing-api due to unbounded user_id label cardinality.",
        "Jaeger query UI missing downstream spans because sampling header x-b3-sampled is stripped.",
    ]),
    ("Vector_Search", "OpenSearch red cluster shards, HNSW index OOM, and analyzer tokenization bugs.", [
        "OpenSearch cluster catalog-prod entered red state with 5 unassigned primary shards.",
        "Vector HNSW index build hit OOM on 1536-dim embeddings during nightly reindex.",
        "Japanese kuromoji search analyzer splits compound product SKUs incorrectly.",
    ]),
]

OOS_TICKETS = [
    "User wrote: 'My screen looks weird since Tuesday, please fix.' No URL, screenshot, trace ID, or error code attached.",
    "Building facilities request: 4F badge reader beeps twice and fails Wiegand parity check for visitor cards.",
    "Corporate IT ticket: macOS laptop cannot connect to 6F Canon printer after MDM profile push.",
    "Conflicting report: ticket says both 'cannot log in via SAML' and 'Stripe VAT invoice PDF has wrong total' with zero logs attached.",
]


def gen_routing(rng: random.Random, idx: int, target_gold: int, is_wrapped: bool) -> dict[str, Any]:
    chosen_4 = rng.sample(ROUTING_QUEUES, 4)
    opts = [q[0] for q in chosen_4] + ["Abstain / Escalate_Ambiguous"]
    descs = [q[1] for q in chosen_4] + ["Ticket is vague, missing logs, conflicting, or outside the 4 listed queues."]
    policy = "Queue Rules: " + "; ".join(f"{q[0]} handles {q[1]}" for q in chosen_4) + "; Abstain if vague, conflicting, or outside all 4 queues."

    if target_gold == 4:
        ticket_txt = OOS_TICKETS[idx % len(OOS_TICKETS)]
    else:
        q_name, _, templates = chosen_4[target_gold]
        ticket_txt = templates[idx % len(templates)] + f" [ticket=#{5000 + idx}]"

    return {
        "bucket": "api_routing",
        "context_band": "wrapped" if is_wrapped else "clean",
        "context": wrap_text(rng, f"{policy}\nTicket: {ticket_txt}", is_wrapped),
        "question": "Which engineering queue owns this ticket?",
        "options": opts,
        "option_descriptions": descs,
        "gold_index": target_gold,
        "gold_label": opts[target_gold],
        "abstain_index": 4,
        "is_abstain_gold": target_gold == 4,
    }


def gen_tradeoff(rng: random.Random, idx: int, target_gold: int, is_wrapped: bool) -> dict[str, Any]:
    # 50% car-build / vehicle dynamics setups, 50% infra/systems setups
    is_car = (idx % 2 == 0)
    k_cands = 4
    opts = [f"Option_{i}" for i in range(k_cands)] + ["Abstain / Escalate_Ambiguous"]

    if is_car:
        max_lap = round(rng.uniform(79.5, 84.0), 1)
        max_wear = round(rng.uniform(2.6, 3.4), 1)
        max_temp = rng.randint(98, 104)
        req = f"Track Requirements: Pick the setup with lap_time_s <= {max_lap}, tire_wear_pct <= {max_wear}%, and coolant_temp_c <= {max_temp}. Abstain if none qualify or any required metric is null."
        cands = []
        for i in range(k_cands):
            if i == target_gold and target_gold != 4:
                lap = round(max_lap - rng.uniform(0.3, 1.2), 1)
                wear = round(max_wear - rng.uniform(0.2, 0.6), 1)
                temp = max_temp - rng.randint(2, 6)
            else:
                # Fail at least one constraint cleanly
                fail_mode = (i + idx) % 3
                lap = round(max_lap + rng.uniform(0.5, 1.8), 1) if fail_mode == 0 else round(max_lap - 0.4, 1)
                wear = round(max_wear + rng.uniform(0.4, 1.1), 1) if fail_mode == 1 else round(max_wear - 0.3, 1)
                temp = max_temp + rng.randint(3, 8) if fail_mode == 2 else max_temp - 2
                if target_gold == 4 and i == 1 and idx % 2 == 0:
                    temp = "null"
            wing_f = 18 + i * 2
            wing_r = 22 + i * 2
            ballast = 15 + i * 5
            cands.append((f"Setup {opts[i]} (fw={wing_f}, rw={wing_r}, ballast={ballast}kg)",
                          f"lap_time_s={lap}, tire_wear_pct={wear}%, coolant_temp_c={temp}"))
    else:
        max_p99 = rng.randint(15, 30)
        min_tps = rng.choice([8000, 10000, 12000])
        max_cost = rng.choice([550, 700, 900])
        req = f"Infra Requirements: Pick the topology with write_p99_ms <= {max_p99}, throughput_tps >= {min_tps}, and monthly_cost_usd <= ${max_cost}. Abstain if none qualify or any metric is null."
        cands = []
        for i in range(k_cands):
            if i == target_gold and target_gold != 4:
                p99 = max_p99 - rng.randint(2, 6)
                tps = min_tps + rng.randint(1000, 3500)
                cost = max_cost - rng.randint(40, 150)
            else:
                fail_mode = (i + idx) % 3
                p99 = max_p99 + rng.randint(6, 20) if fail_mode == 0 else max_p99 - 3
                tps = min_tps - rng.randint(1500, 3500) if fail_mode == 1 else min_tps + 1200
                cost = max_cost + rng.randint(120, 400) if fail_mode == 2 else max_cost - 50
                if target_gold == 4 and i == 2 and idx % 2 == 1:
                    tps = "null"
            cands.append((f"Topo {opts[i]} (cluster-{chr(65+i)})",
                          f"write_p99_ms={p99}, throughput_tps={tps}, monthly_cost_usd=${cost}"))

    cand_str = "; ".join(f"{name}: {metrics}" for name, metrics in cands)
    descs = [f"{name} — {metrics}." for name, metrics in cands] + ["No candidate meets all constraints or required metric is null."]

    return {
        "bucket": "engineering_tradeoff",
        "context_band": "wrapped" if is_wrapped else "clean",
        "context": wrap_text(rng, f"{req}\nCandidates: {cand_str}.", is_wrapped),
        "question": "Which candidate satisfies all requirements?",
        "options": opts,
        "option_descriptions": descs,
        "gold_index": target_gold,
        "gold_label": opts[target_gold],
        "abstain_index": 4,
        "is_abstain_gold": target_gold == 4,
    }


def main() -> None:
    rng = random.Random(20261125)
    rows: list[dict[str, Any]] = []
    # 150 per bucket = 600 total
    # Per 150: 32 abstain (21.3%) + 118 non-abstain (~29-30 per index 0..3)
    # 25% wrapped -> 38 wrapped, 112 clean per bucket (~152 wrapped, 448 clean total)
    gens = [gen_triage, gen_gate, gen_routing, gen_tradeoff]
    gold_plan = [4] * 32 + [0] * 30 + [1] * 30 + [2] * 29 + [3] * 29
    assert len(gold_plan) == 150

    seen_ctx: set[str] = set()
    hist_ctx = {json.loads(l)["context"] for l in HIST_TEST.read_text(encoding="utf-8").splitlines() if l.strip()}
    nt_ctx = {json.loads(l)["context"] for l in NONTEST_1200.read_text(encoding="utf-8").splitlines() if l.strip()}

    seq = 1
    for gen_fn in gens:
        plan = list(gold_plan)
        wrap_flags = [True] * 38 + [False] * 112
        rng.shuffle(plan)
        rng.shuffle(wrap_flags)
        for i, (g, w) in enumerate(zip(plan, wrap_flags)):
            item = gen_fn(rng, seq, g, w)
            # Ensure unique context
            while item["context"] in seen_ctx or item["context"] in hist_ctx or item["context"] in nt_ctx:
                item = gen_fn(rng, seq + 1000 + rng.randint(1, 9999), g, w)
            seen_ctx.add(item["context"])
            item["id"] = f"hold_{seq:04d}"
            item["source_id"] = f"src_hold_{seq:04d}"
            item["split"] = "holdout"
            rows.append(item)
            seq += 1

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")

    words = [len((" ".join([r["context"], r["question"]] + r["options"] + r["option_descriptions"])).split()) for r in rows]
    manifest = {
        "seed": 20261125,
        "count": len(rows),
        "buckets": dict(Counter(r["bucket"] for r in rows)),
        "context_bands": dict(Counter(r["context_band"] for r in rows)),
        "abstain_count": sum(1 for r in rows if r["is_abstain_gold"]),
        "gold_indices": dict(sorted(Counter(r["gold_index"] for r in rows).items())),
        "word_stats": {"min": min(words), "median": sorted(words)[len(words) // 2], "max": max(words)},
        "overlap_hist_test": sum(1 for r in rows if r["context"] in hist_ctx),
        "overlap_nontest_1200": sum(1 for r in rows if r["context"] in nt_ctx),
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
