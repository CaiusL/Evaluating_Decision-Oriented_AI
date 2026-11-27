"""Build frozen, 100% offline multi-suite evaluation benchmarks for decision-oriented AI.

Generates four complementary suites under data/benchmarks/:
  1. suite1_software_ops   - General software workflow decisions (incident triage, CI/CD release gating,
                             API/support routing, and out-of-scope/ambiguous abstain traps). Zero racing bias.
  2. suite2_historical_eng - Real historical agx.cool match setup comparisons (data/outcome/examples.jsonl)
                             plus the 18-question long-context driver-lever harness (self_jev/kev_harness.py).
  3. suite3_car_build      - Simulator-backed car-build selection benchmark (data/car_build_bench/) with
                             oracle_scores, tabular features (for Tier 4 NN), and transparent policy notes.
  4. suite4_stress_test    - Robustness & scaling stress test: context-length sweep (short ~250 tok,
                             medium ~750 tok, long ~2,000 tok) and prompt/label paraphrase variants.

Also writes unified splits (all_calibrate.jsonl, all_test.jsonl) and data/benchmarks/manifest.json.
Uses Python standard library only.
"""
from __future__ import annotations

import argparse
import ast
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from decision_eval.paths import BENCHMARKS_DIR as OUT_DIR
from decision_eval.paths import REPO_ROOT


def _est_tokens(text: str) -> int:
    return max(1, int(len(text.split()) * 1.33))


def _context_band(tokens: int) -> str:
    if tokens < 400:
        return "short"
    if tokens < 1000:
        return "medium"
    return "long"


def _make_item(
    *,
    item_id: str,
    suite: str,
    split: str,
    bucket: str,
    context: str,
    question: str,
    options: list[str],
    option_descriptions: list[str],
    gold_index: int,
    abstain_index: int | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    tokens = _est_tokens(context + " " + question + " " + " ".join(option_descriptions))
    rec: dict[str, Any] = {
        "id": item_id,
        "suite": suite,
        "split": split,
        "bucket": bucket,
        "est_tokens": tokens,
        "context_band": _context_band(tokens),
        "context": context,
        "question": question,
        "options": options,
        "option_descriptions": option_descriptions,
        "gold_index": gold_index,
        "gold_label": options[gold_index],
        "abstain_index": abstain_index,
        "is_abstain_gold": (abstain_index is not None and gold_index == abstain_index),
    }
    if extra:
        rec.update(extra)
    return rec


# ---------------------------------------------------------------------------
# Suite 1: General Software Operations (160 items: 40 calibrate, 120 test)
# ---------------------------------------------------------------------------

_SERVICES = [
    "checkout-api",
    "payments-ledger",
    "auth-gateway",
    "search-indexer",
    "notification-worker",
    "billing-reconciler",
    "inventory-sync",
    "session-store",
    "webhook-dispatcher",
    "analytics-ingest",
]
_REGIONS = ["us-east-1", "eu-west-1", "ap-northeast-1", "us-west-2", "eu-central-1"]


def _build_suite1(seed: int = 20260401) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed)
    cal: list[dict[str, Any]] = []
    test: list[dict[str, Any]] = []

    # 1A. Incident Triage (40 items: 10 cal, 30 test)
    triage_opts = [
        "SEV1_Page_Incident_Commander",
        "SEV2_Page_Service_Owner",
        "SEV3_Create_Next_Day_Ticket",
        "SEV4_Log_Only_No_Action",
        "Abstain / Escalate_Ambiguous",
    ]
    triage_descs = [
        "Critical customer-facing outage: 5xx >= 5.0% OR multi-region impact OR core checkout/auth down; page Incident Commander immediately.",
        "Major single-region degradation: 5xx between 1.0% and 4.9% OR p99 latency > 2000 ms for >= 10 min; page service owner.",
        "Minor non-urgent issue: 5xx between 0.2% and 0.9% OR background worker lag without user outage; create next-business-day ticket.",
        "Informational noise: 5xx < 0.2%, p99 < 400 ms, zero SLA impact; log only with no page or ticket.",
        "Telemetry is missing, contradictory, or sensor pipeline is stale; abstain from automated paging and escalate for manual verification.",
    ]
    triage_policy = (
        "Runbook Policy: SEV1 if 5xx >= 5.0% or multi-region outage; "
        "SEV2 if single-region and (1.0% <= 5xx < 5.0% or p99 >= 2000 ms for >= 10 min); "
        "SEV3 if 0.2% <= 5xx < 1.0% or non-urgent worker backlog; "
        "SEV4 if 5xx < 0.2% and p99 < 400 ms with zero SLA impact; "
        "Abstain if metrics are missing, stale, or contradictory."
    )

    for i in range(40):
        split = "calibrate" if i < 10 else "test"
        svc = rng.choice(_SERVICES)
        reg = rng.choice(_REGIONS)
        sha = f"{rng.randrange(16**7):07x}"
        cls = i % 4  # 0..3
        if cls == 0:
            err = round(rng.uniform(6.2, 24.5), 2)
            p99 = rng.randint(1800, 6200)
            regions_hit = f"{reg} and {rng.choice([r for r in _REGIONS if r != reg])}"
            dur = rng.randint(6, 25)
            sla = "breaching"
        elif cls == 1:
            err = round(rng.uniform(1.4, 4.4), 2)
            p99 = rng.randint(2150, 4100)
            regions_hit = f"{reg} (single region)"
            dur = rng.randint(12, 35)
            sla = "at_risk"
        elif cls == 2:
            err = round(rng.uniform(0.3, 0.85), 2)
            p99 = rng.randint(480, 1250)
            regions_hit = f"{reg} (single region, async queue)"
            dur = rng.randint(5, 20)
            sla = "within_budget"
        else:
            err = round(rng.uniform(0.01, 0.14), 2)
            p99 = rng.randint(95, 310)
            regions_hit = f"{reg} (single region)"
            dur = rng.randint(1, 5)
            sla = "healthy"

        ctx = (
            f"{triage_policy}\n"
            f"Incident Alert: service={svc}, deploy_sha={sha}, regions={regions_hit}, "
            f"http_5xx_rate={err}%, p99_latency_ms={p99}, duration_min={dur}, "
            f"sla_status={sla}, telemetry_freshness=fresh, dashboard_agreement=consistent."
        )
        item = _make_item(
            item_id=f"s1_triage_{i+1:03d}",
            suite="suite1_software_ops",
            split=split,
            bucket="incident_triage",
            context=ctx,
            question="Which incident triage action should the automation take?",
            options=triage_opts,
            option_descriptions=triage_descs,
            gold_index=cls,
            abstain_index=4,
        )
        (cal if split == "calibrate" else test).append(item)

    # 1B. CI/CD Release Gate (40 items: 10 cal, 30 test)
    gate_opts = [
        "Promote_To_100pct",
        "Hold_Extend_Canary",
        "Rollback_Immediately",
        "Block_Security_Review",
        "Abstain / Escalate_Ambiguous",
    ]
    gate_descs = [
        "All tests pass, zero critical/high CVEs, canary error delta <= +0.05%, latency delta <= +15 ms, no memory leak; promote to 100%.",
        "Tests pass and zero critical/high CVEs, but canary sample size is too small (< 5000 reqs) or latency delta is borderline (+16 to +60 ms); extend canary.",
        "Canary error delta >= +0.50% OR p99 latency delta > +150 ms OR confirmed memory leak; rollback immediately.",
        "Security scanner reports >= 1 critical or high unwaived CVE or leaked secret; block release for security review.",
        "Canary metrics or CI test artifacts are missing, corrupted, or conflicting across monitors; abstain and escalate.",
    ]
    gate_policy = (
        "Release Gate Policy: Block_Security_Review if critical_high_cves > 0 or secret_leak=true; "
        "Rollback_Immediately if canary_error_delta >= +0.50% or latency_delta_ms > 150 or memory_leak=true; "
        "Hold_Extend_Canary if canary_requests < 5000 or (15 < latency_delta_ms <= 60); "
        "Promote_To_100pct if tests=passed, critical_high_cves=0, canary_requests >= 5000, error_delta <= +0.05%, latency_delta_ms <= 15, memory_leak=false; "
        "Abstain if telemetry is missing or contradictory."
    )

    for i in range(40):
        split = "calibrate" if i < 10 else "test"
        svc = rng.choice(_SERVICES)
        ver = f"v{rng.randint(2,7)}.{rng.randint(0,19)}.{rng.randint(0,9)}"
        cls = i % 4
        if cls == 0:
            reqs = rng.randint(12000, 95000)
            err_d = round(rng.uniform(-0.02, 0.04), 3)
            lat_d = rng.randint(-5, 12)
            cves = 0
            leak = "false"
        elif cls == 1:
            reqs = rng.randint(800, 3200) if i % 2 == 0 else rng.randint(10000, 40000)
            err_d = round(rng.uniform(0.0, 0.04), 3)
            lat_d = rng.randint(22, 52) if reqs >= 5000 else rng.randint(2, 10)
            cves = 0
            leak = "false"
        elif cls == 2:
            reqs = rng.randint(8000, 50000)
            err_d = round(rng.uniform(0.65, 3.80), 3)
            lat_d = rng.randint(180, 520)
            cves = 0
            leak = "true" if i % 3 == 0 else "false"
        else:
            reqs = rng.randint(9000, 45000)
            err_d = round(rng.uniform(0.0, 0.03), 3)
            lat_d = rng.randint(0, 10)
            cves = rng.randint(1, 3)
            leak = "false"

        ctx = (
            f"{gate_policy}\n"
            f"Release Candidate: service={svc}, version={ver}, unit_and_integration_tests=passed, "
            f"critical_high_cves={cves}, secret_leak=false, canary_requests={reqs}, "
            f"canary_error_delta={err_d:+.3f}%, latency_delta_ms={lat_d:+d}, "
            f"memory_leak={leak}, artifact_checksum=verified."
        )
        item = _make_item(
            item_id=f"s1_gate_{i+1:03d}",
            suite="suite1_software_ops",
            split=split,
            bucket="release_gate",
            context=ctx,
            question="Which CI/CD release gate decision should be executed?",
            options=gate_opts,
            option_descriptions=gate_descs,
            gold_index=cls,
            abstain_index=4,
        )
        (cal if split == "calibrate" else test).append(item)

    # 1C. Engineering / API Ticket Routing (40 items: 10 cal, 30 test)
    route_opts = [
        "Queue_Auth_Identity",
        "Queue_Billing_Payments",
        "Queue_Database_Storage",
        "Queue_Frontend_SDK",
        "Abstain / Escalate_Ambiguous",
    ]
    route_descs = [
        "Authentication, OAuth2/OIDC tokens, SSO/SAML login failures, RBAC permissions, and API key rotation.",
        "Invoices, subscription upgrades, failed credit card charges, VAT/tax receipts, and refund disputes.",
        "SQL query timeouts, PostgreSQL/Redis connection pool exhaustion, replication lag, and schema migrations.",
        "React/TypeScript UI rendering bugs, browser console errors, mobile/JS SDK crashes, and CSS layout issues.",
        "Ticket lacks enough technical detail, belongs to an unsupported domain, or contains conflicting symptoms; escalate to human triage.",
    ]
    route_templates = [
        (0, [
            "Enterprise SSO users get HTTP 401 'invalid_grant' after SAML assertion refresh on /oauth/token.",
            "API keys created with read-only RBAC scope are unexpectedly rejected with 403 on /v2/projects.",
            "OIDC JWKS key rotation caused intermittent JWT signature verification failures across mobile logins.",
            "MFA enrollment email link expires immediately when users sign in through Okta federation.",
        ]),
        (1, [
            "Customer reports duplicate monthly invoice charge on Visa ending 4821 after upgrading to Team plan.",
            "Automated Stripe webhook failed to record seat proration credit; invoice PDF shows wrong VAT rate.",
            "Annual enterprise renewal quote needs PO number attached and refund issued for accidental overcharge.",
            "Self-serve checkout returns 'card_declined_insufficient_funds' even after updating default payment method.",
        ]),
        (2, [
            "Read replica lag on pg-primary-02 climbed to 48 seconds during the nightly batch index rebuild.",
            "HikariCP connection pool exhausted (active=100/100) with lock waits on orders_line_items table.",
            "Redis cluster failover left stale slot mappings and caused 12-second timeout spikes on cache writes.",
            "Alembic migration 2026_03_add_tenant_idx blocked concurrent writes with AccessExclusiveLock.",
        ]),
        (3, [
            "Dashboard charts fail to render in Safari 18 with TypeError: Cannot read properties of undefined in ChartCanvas.tsx.",
            "JavaScript SDK v4.2.1 throws unhandled promise rejection when offline retry queue flushes in React Native.",
            "Dropdown menu overlaps modal footer on 1280px viewports after upgrading Tailwind CSS component library.",
            "TypeScript client package exports broken ESM subpath for @acme/sdk/streaming in Vite builds.",
        ]),
    ]
    for i in range(40):
        split = "calibrate" if i < 10 else "test"
        cls = i % 4
        msg = route_templates[cls][1][(i // 4) % 4]
        acct_tier = rng.choice(["Enterprise", "Growth", "Starter"])
        ticket_no = 10400 + i
        ctx = (
            "Routing Policy: Assign each engineering support ticket to the single best owning queue "
            "(Auth/Identity, Billing/Payments, Database/Storage, or Frontend/SDK), or Abstain if "
            "the ticket is too vague, contradictory, or outside all four domains.\n"
            f"Ticket #{ticket_no} (tier={acct_tier}, channel=portal): \"{msg}\""
        )
        item = _make_item(
            item_id=f"s1_route_{i+1:03d}",
            suite="suite1_software_ops",
            split=split,
            bucket="api_routing",
            context=ctx,
            question="Which engineering queue should receive this ticket?",
            options=route_opts,
            option_descriptions=route_descs,
            gold_index=cls,
            abstain_index=4,
        )
        (cal if split == "calibrate" else test).append(item)

    # 1D. Abstain / Escalate Traps (40 items: 10 cal, 30 test)
    trap_cases = [
        # (domain_idx: 0=triage, 1=gate, 2=route, scenario_text)
        (0, "Prometheus shows 5xx=18.4% in us-east-1 while Datadog and synthetic probes show 5xx=0.01% and healthy latency; metrics collector reported a clock-skew alert 4 min ago."),
        (0, "Alert fired for checkout-api with p99_latency_ms=null and http_5xx_rate=missing because the telemetry ingestion pipeline has been stalled for 22 minutes."),
        (0, "Pager triggered with conflicting blast-radius tags: region=all_regions_down yet active_user_sessions=normal and load_balancer_health=unknown (telemetry_freshness=stale)."),
        (0, "Service payments-ledger shows 5xx=3.2% in staging-mirror only; production metrics are unavailable due to an expired monitoring scrape token."),
        (1, "Canary dashboard reports error_delta=-0.01% on 20,000 requests, but the CI artifact checksum failed verification and security scan status is 'UNKNOWN_TIMEOUT'."),
        (1, "Unit tests passed and CVEs=0, but canary_requests=missing and latency_delta_ms=NaN because the canary metrics exporter crashed mid-rollout."),
        (1, "Security scanner returned contradictory reports: Trivy found 0 CVEs while Snyk failed with HTTP 503; canary error rate is unavailable."),
        (1, "Deployment pipeline shows memory_leak=unknown and integration_tests=cancelled_by_runner_oom; partial canary logs disagree on 5xx rate."),
        (2, "Customer wrote: 'Something is broken on my screen since yesterday, please fix ASAP.' No product area, error message, screenshot, or account logs attached."),
        (2, "Employee submitted a request asking to replace their office ergonomic chair and renew their building parking badge."),
        (2, "Ticket says: 'Our legal counsel needs a redlined DPA and SOC2 Type II NDA countersigned before Friday.'"),
        (2, "User reports: 'Maybe it is a billing issue or maybe SSO or maybe our database? Nothing loads and we haven't checked which app it is yet.'"),
    ]
    for i in range(40):
        split = "calibrate" if i < 10 else "test"
        dom, base_desc = trap_cases[i % len(trap_cases)]
        svc = rng.choice(_SERVICES)
        reg = rng.choice(_REGIONS)
        variant_tag = f"event_id=TRAP-{i+1:03d}, service={svc}, region={reg}"
        if dom == 0:
            ctx = f"{triage_policy}\nIncident Alert ({variant_tag}): {base_desc}"
            opts, descs, q = triage_opts, triage_descs, "Which incident triage action should the automation take?"
        elif dom == 1:
            ctx = f"{gate_policy}\nRelease Candidate ({variant_tag}): {base_desc}"
            opts, descs, q = gate_opts, gate_descs, "Which CI/CD release gate decision should be executed?"
        else:
            ctx = (
                "Routing Policy: Assign each engineering support ticket to the single best owning queue "
                "(Auth/Identity, Billing/Payments, Database/Storage, or Frontend/SDK), or Abstain if "
                "the ticket is too vague, contradictory, or outside all four domains.\n"
                f"Ticket ({variant_tag}): {base_desc}"
            )
            opts, descs, q = route_opts, route_descs, "Which engineering queue should receive this ticket?"

        item = _make_item(
            item_id=f"s1_trap_{i+1:03d}",
            suite="suite1_software_ops",
            split=split,
            bucket="abstain_trap",
            context=ctx,
            question=q,
            options=opts,
            option_descriptions=descs,
            gold_index=4,
            abstain_index=4,
        )
        (cal if split == "calibrate" else test).append(item)

    return cal, test


# ---------------------------------------------------------------------------
# Suite 2: Real Historical Engineering & Long-Context Driver Levers
# ---------------------------------------------------------------------------


def _extract_kev_harness_constants() -> tuple[str, dict[str, Any]]:
    """Parse STATE and QUESTIONS from self_jev/kev_harness.py via AST (zero runtime imports)."""
    harness_path = REPO_ROOT / "self_jev" / "kev_harness.py"
    tree = ast.parse(harness_path.read_text(encoding="utf-8"))
    state_val: str | None = None
    questions_val: dict[str, Any] | None = None
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "STATE":
                    state_val = ast.literal_eval(node.value)
                elif isinstance(target, ast.Name) and target.id == "QUESTIONS":
                    questions_val = ast.literal_eval(node.value)
    if not state_val or not questions_val:
        raise RuntimeError("Failed to parse STATE / QUESTIONS from self_jev/kev_harness.py")
    return state_val, questions_val


def _build_suite2() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cal: list[dict[str, Any]] = []
    test: list[dict[str, Any]] = []

    # 2A. Real historical agx.cool match setup comparisons (data/outcome/examples.jsonl)
    outcome_path = REPO_ROOT / "data" / "outcome" / "examples.jsonl"
    if outcome_path.is_file():
        for idx, ln in enumerate(outcome_path.read_text(encoding="utf-8").splitlines()):
            if not ln.strip():
                continue
            row = json.loads(ln)
            sp = row.get("split")
            if sp not in ("calibrate", "test"):
                continue
            cands = row.get("candidates") or []
            if len(cands) != 2:
                continue
            label_key = row.get("label_key")
            keys = [c["key"] for c in cands]
            if label_key not in keys:
                continue
            gold_idx = keys.index(label_key)

            options = [f"Candidate_A_{cands[0]['key'][:6]}", f"Candidate_B_{cands[1]['key'][:6]}"]
            descs = [cands[0]["description"], cands[1]["description"]]
            ctx = (
                f"Historical Match Context: {row.get('situation', '')}\n"
                f"Track: {row.get('track', 'unknown')} | Moisture band: {row.get('wet', 'unknown')}\n"
                f"Outcome Rule: {row.get('outcome_rule', '')}\n"
                f"Candidate A ({options[0]}): {descs[0]}\n"
                f"Candidate B ({options[1]}): {descs[1]}"
            )
            item = _make_item(
                item_id=f"s2_hist_{sp}_{idx:04d}",
                suite="suite2_historical_eng",
                split=sp,
                bucket="historical_pair",
                context=ctx,
                question=row.get("question", "Choose the recorded car setup tied to the stronger finishing place."),
                options=options,
                option_descriptions=descs,
                gold_index=gold_idx,
                abstain_index=None,
                extra={
                    "track": row.get("track"),
                    "wet": row.get("wet"),
                    "situation_id": row.get("situation_id"),
                    "paraphrase": bool(row.get("paraphrase", False)),
                },
            )
            (cal if sp == "calibrate" else test).append(item)

    # 2B. Long-context driver-lever & release-gate harness (18 questions from self_jev/kev_harness.py)
    state_text, questions = _extract_kev_harness_constants()
    for q_key, spec in questions.items():
        q_type = spec.get("type")
        if q_type == "noul":
            options = ["Yes", "No"]
            descs = [
                "At least one valuable lap-time lever already has a fair isolated still-racing pair.",
                "No valuable lap-time lever has a fair isolated still-racing pair yet.",
            ]
            gold_idx = 1  # Ground truth from STATE: zero fair pairs exist
            abstain_idx = None
        else:
            criteria = spec.get("criteria") or {}
            options = list(criteria.keys())
            descs = [str(criteria[k]) for k in options]
            gold_idx = options.index("unresolved")
            abstain_idx = gold_idx

        item = _make_item(
            item_id=f"s2_drv_{q_key}",
            suite="suite2_historical_eng",
            split="test",
            bucket="driver_lever_longctx",
            context=state_text,
            question=str(spec.get("instructions", f"Decide lever: {q_key}")),
            options=options,
            option_descriptions=descs,
            gold_index=gold_idx,
            abstain_index=abstain_idx,
            extra={"lever": q_key, "question_type": q_type},
        )
        test.append(item)

    return cal, test


# ---------------------------------------------------------------------------
# Suite 3: Simulator-Backed Car-Build Selection (data/car_build_bench/)
# ---------------------------------------------------------------------------


_CAR_BUILD_POLICY = (
    "Build Selection Policy: Choose the single safest, highest-scoring car build for the forecast and track "
    "(dry_low_drag for dry/fast conditions; damp_mid_downforce for transitional/damp or higher-grip needs; "
    "wet_high_downforce for heavy wetness; follow any Event Bulletin tyre restrictions strictly). "
    "Choose 'Abstain / Unresolved' if forecast_confidence < 0.45, sensor_missing=1, or no candidate build is safe."
)


def _build_suite3() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cal: list[dict[str, Any]] = []
    test: list[dict[str, Any]] = []
    base = REPO_ROOT / "data" / "car_build_bench"

    for sp, target in [("calibrate", cal), ("test", test)]:
        p = base / f"{sp}.jsonl"
        if not p.is_file():
            continue
        for ln in p.read_text(encoding="utf-8").splitlines():
            if not ln.strip():
                continue
            row = json.loads(ln)
            options = list(row["options"])
            descs = list(row.get("option_descriptions") or options)
            abstain_idx = options.index("Abstain / Unresolved") if "Abstain / Unresolved" in options else None
            oracle_scores = row.get("oracle_scores") or {}
            best_oracle = max(oracle_scores.values()) if oracle_scores else None
            # Flag the known v1 Parkland cases where all 4 builds scored < 0 despite high forecast confidence
            hidden_oracle_abstain = (
                row.get("bucket") == "in_dist"
                and row.get("gold_label") == "Abstain / Unresolved"
                and best_oracle is not None
                and best_oracle < 0.0
            )
            ctx = f"{_CAR_BUILD_POLICY}\n{row['context']}"
            item = _make_item(
                item_id=f"s3_cb_{row['id']}",
                suite="suite3_car_build",
                split=sp,
                bucket=row.get("bucket", "in_dist"),
                context=ctx,
                question=row["question"],
                options=options,
                option_descriptions=descs,
                gold_index=int(row["gold_index"]),
                abstain_index=abstain_idx,
                extra={
                    "track": row.get("track"),
                    "features": row.get("features"),
                    "feature_names": row.get("feature_names"),
                    "oracle_scores": oracle_scores,
                    "hidden_oracle_abstain": hidden_oracle_abstain,
                },
            )
            target.append(item)
    return cal, test


# ---------------------------------------------------------------------------
# Suite 4: Stress Test (Context-Length Sweep + Paraphrase Stability)
# ---------------------------------------------------------------------------


def _make_distractor_block(rng: random.Random, target_extra_words: int) -> str:
    """Generate realistic, non-contradictory background telemetry/log lines."""
    lines = ["Background Cluster Heartbeats & Routine Audit Log (unrelated healthy subsystems):"]
    words = len(lines[0].split())
    idx = 0
    while words < target_extra_words:
        idx += 1
        svc = rng.choice(["cdn-edge", "static-assets", "metrics-relay", "cron-cleanup", "pdf-renderer", "dns-resolver"])
        reg = rng.choice(_REGIONS)
        cpu = round(rng.uniform(11.0, 42.0), 1)
        mem = round(rng.uniform(28.0, 61.0), 1)
        rps = rng.randint(120, 4800)
        line = (
            f"[2026-04-0{1 + (idx % 9)}T10:{idx % 60:02d}:{(idx * 7) % 60:02d}Z] "
            f"INFO pod={svc}-{reg}-{idx % 12:02d} status=HEALTHY cpu={cpu}% mem={mem}% "
            f"rps={rps} p50_ms={rng.randint(8, 35)} err_rate=0.00% note=routine_heartbeat_ok"
        )
        lines.append(line)
        words += len(line.split())
    return "\n".join(lines)


def _build_suite4(suite1_test: list[dict[str, Any]], seed: int = 20260402) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    # Pick 18 balanced base items from Suite 1 test (5 triage, 5 gate, 4 route, 4 trap)
    by_bucket: dict[str, list[dict[str, Any]]] = {}
    for it in suite1_test:
        by_bucket.setdefault(it["bucket"], []).append(it)

    base_items = (
        by_bucket.get("incident_triage", [])[:5]
        + by_bucket.get("release_gate", [])[:5]
        + by_bucket.get("api_routing", [])[:4]
        + by_bucket.get("abstain_trap", [])[:4]
    )

    stress_items: list[dict[str, Any]] = []

    # 4A. Context-Length Sweep: short (~250 tok), medium (~750 tok), long (~2,000 tok)
    for b_idx, base in enumerate(base_items):
        for band, extra_words in [("ctx_short", 110), ("ctx_medium", 480), ("ctx_long", 1400)]:
            pad_top = _make_distractor_block(rng, extra_words // 2)
            pad_bot = _make_distractor_block(rng, extra_words - extra_words // 2)
            ctx = f"{pad_top}\n\n--- ACTIVE DECISION RECORD ---\n{base['context']}\n--- END ACTIVE DECISION RECORD ---\n\n{pad_bot}"
            stress_items.append(
                _make_item(
                    item_id=f"s4_{band}_{b_idx+1:03d}",
                    suite="suite4_stress_test",
                    split="test",
                    bucket=band,
                    context=ctx,
                    question=base["question"],
                    options=list(base["options"]),
                    option_descriptions=list(base["option_descriptions"]),
                    gold_index=int(base["gold_index"]),
                    abstain_index=base.get("abstain_index"),
                    extra={"base_id": base["id"], "base_bucket": base["bucket"]},
                )
            )

    # 4B. Prompt & Label Paraphrase Slice (18 items)
    q_paraphrase = {
        "Which incident triage action should the automation take?":
            "Select the single most appropriate operational response level for this alert.",
        "Which CI/CD release gate decision should be executed?":
            "Determine the required deployment gate outcome for this release candidate.",
        "Which engineering queue should receive this ticket?":
            "Route this incoming support issue to the single responsible engineering team.",
    }
    for b_idx, base in enumerate(base_items):
        p_ctx = (
            "Operational Decision Brief (paraphrased format):\n"
            + base["context"]
            .replace("Runbook Policy:", "Standard Operating Rules:")
            .replace("Release Gate Policy:", "Deployment Policy:")
            .replace("Routing Policy:", "Team Ownership Rules:")
            .replace("Incident Alert:", "Observed Production Signal:")
            .replace("Release Candidate:", "Pending Build Summary:")
        )
        p_q = q_paraphrase.get(base["question"], base["question"])
        # Paraphrase option names cleanly while keeping semantic 1-to-1 mapping
        p_opts = [opt.replace("_", " ") for opt in base["options"]]
        stress_items.append(
            _make_item(
                item_id=f"s4_paraphrase_{b_idx+1:03d}",
                suite="suite4_stress_test",
                split="test",
                bucket="paraphrase",
                context=p_ctx,
                question=p_q,
                options=p_opts,
                option_descriptions=list(base["option_descriptions"]),
                gold_index=int(base["gold_index"]),
                abstain_index=base.get("abstain_index"),
                extra={"base_id": base["id"], "base_bucket": base["bucket"]},
            )
        )

    return stress_items


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build frozen multi-suite evaluation benchmarks.")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--seed", type=int, default=20260401)
    args = parser.parse_args()

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    s1_cal, s1_test = _build_suite1(seed=args.seed)
    s2_cal, s2_test = _build_suite2()
    s3_cal, s3_test = _build_suite3()
    s4_test = _build_suite4(s1_test, seed=args.seed + 1)

    files = {
        "suite1_software_ops_calibrate.jsonl": s1_cal,
        "suite1_software_ops_test.jsonl": s1_test,
        "suite2_historical_eng_calibrate.jsonl": s2_cal,
        "suite2_historical_eng_test.jsonl": s2_test,
        "suite3_car_build_calibrate.jsonl": s3_cal,
        "suite3_car_build_test.jsonl": s3_test,
        "suite4_stress_test.jsonl": s4_test,
        "all_calibrate.jsonl": s1_cal + s2_cal + s3_cal,
        "all_test.jsonl": s1_test + s2_test + s3_test + s4_test,
    }

    manifest: dict[str, Any] = {
        "seed": args.seed,
        "files": {},
        "totals": {
            "calibrate": len(files["all_calibrate.jsonl"]),
            "test": len(files["all_test.jsonl"]),
        },
    }

    for fname, rows in files.items():
        _write_jsonl(out_dir / fname, rows)
        buckets = Counter(r["bucket"] for r in rows)
        bands = Counter(r["context_band"] for r in rows)
        abstain_cnt = sum(1 for r in rows if r["is_abstain_gold"])
        manifest["files"][fname] = {
            "count": len(rows),
            "abstain_gold_count": abstain_cnt,
            "buckets": dict(sorted(buckets.items())),
            "context_bands": dict(sorted(bands.items())),
        }

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Built benchmarks in {out_dir}")
    print(f"  all_calibrate.jsonl : {manifest['totals']['calibrate']} items")
    print(f"  all_test.jsonl      : {manifest['totals']['test']} items")
    for fname, info in manifest["files"].items():
        if fname.startswith("suite"):
            print(f"  {fname:40s} : {info['count']:3d} items | buckets={info['buckets']}")


if __name__ == "__main__":
    main()
