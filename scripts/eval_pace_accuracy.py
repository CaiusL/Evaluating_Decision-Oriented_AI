#!/usr/bin/env python3
"""Auditable pace-gap evaluator (v2) for AGX racing match logs and simulator runs.

Implements a 3-stage pipeline:
  1. Normalize facts (strict booleans, classification provenance, full-distance check,
     timing compatibility, and weather-regime provenance).
  2. Resolve references separately for:
       - external_reference (data/agx_winner_times.json)
       - same_match_winner  (recording_facts.winner_time_s)
  3. Score and aggregate using Competitive Completion Rate at fixed gap-fraction
     thresholds (0%, 1%, 5%, 10%, 20%), keeping known failures in the eligible
     denominator, separating measurement_unknown with uncertainty bounds, and
     computing continuous pace-gap stats strictly over comparable full finishes.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

DEFAULT_WINNERS = Path("/Users/caius/Projects/self_jev/data/agx_winner_times.json")
DEFAULT_MATCHES = Path("/Users/caius/Projects/self_jev/data/imports/comp-racing-history-20261117/matches.jsonl")
DEFAULT_OUT = Path("/Users/caius/Projects/self_jev/reports/pace_accuracy_summary.json")

EVALUATOR_VERSION = "pace-eval-v2"
WEATHER_MAPPING_VERSION = "weather-map-v2"
THRESHOLD_FRACTIONS: tuple[float, ...] = (0.00, 0.01, 0.05, 0.10, 0.20)
_WET_ORDER = {"dry": 0, "damp": 1, "wet": 2, "saturated": 3}
_FAILURE_STATUSES = {"retired", "dnf", "disqualified", "dq", "failed", "aborted", "illegal"}


def positive_finite(x: Any) -> bool:
    return (
        isinstance(x, (int, float))
        and not isinstance(x, bool)
        and math.isfinite(x)
        and float(x) > 0.0
    )


def strict_bool(x: Any) -> bool | None:
    if isinstance(x, bool):
        return x
    return None


def infer_weather_regime(rec: dict[str, Any]) -> dict[str, Any]:
    """Infer weather regime with explicit provenance (official | heuristic | unknown)."""
    explicit = rec.get("weather_regime") or rec.get("official_weather_regime") or rec.get("weather")
    if isinstance(explicit, str) and explicit.strip():
        return {
            "regime": explicit.strip().capitalize(),
            "source": "official",
            "mapping_version": WEATHER_MAPPING_VERSION,
            "evidence": {"explicit_field": explicit.strip()},
            "ambiguities": [],
        }

    ws = rec.get("weather_start")
    we = rec.get("weather_end")
    wet_race = strict_bool(rec.get("wet_race"))
    max_wet = rec.get("max_track_wetness")
    changes = rec.get("weather_changes")

    evidence = {
        "weather_start": ws,
        "weather_end": we,
        "wet_race": wet_race,
        "max_track_wetness": max_wet,
        "weather_changes": changes,
    }

    if not ws and not we:
        return {
            "regime": None,
            "source": "unknown",
            "mapping_version": WEATHER_MAPPING_VERSION,
            "evidence": evidence,
            "ambiguities": ["missing_weather_start_and_end"],
        }

    ws_l = str(ws or "").lower()
    we_l = str(we or "").lower()
    ambiguities = ["official_regime_not_present"]

    if ws_l in _WET_ORDER and we_l in _WET_ORDER:
        s_ord, e_ord = _WET_ORDER[ws_l], _WET_ORDER[we_l]
        if s_ord < e_ord:
            return {
                "regime": "Wetting",
                "source": "heuristic",
                "mapping_version": WEATHER_MAPPING_VERSION,
                "evidence": evidence,
                "ambiguities": ambiguities,
            }
        if s_ord > e_ord:
            return {
                "regime": "Drying",
                "source": "heuristic",
                "mapping_version": WEATHER_MAPPING_VERSION,
                "evidence": evidence,
                "ambiguities": ambiguities,
            }
        if ws_l == "dry":
            max_wet_val = float(max_wet) if isinstance(max_wet, (int, float)) and not isinstance(max_wet, bool) else 0.0
            if wet_race is True or max_wet_val >= 0.15:
                return {
                    "regime": "Variable",
                    "source": "heuristic",
                    "mapping_version": WEATHER_MAPPING_VERSION,
                    "evidence": evidence,
                    "ambiguities": ambiguities + ["dry_start_end_with_midrace_wetness"],
                }
            return {
                "regime": "Dry",
                "source": "heuristic",
                "mapping_version": WEATHER_MAPPING_VERSION,
                "evidence": evidence,
                "ambiguities": ambiguities,
            }
        if isinstance(changes, int) and changes > 1:
            return {
                "regime": "Variable",
                "source": "heuristic",
                "mapping_version": WEATHER_MAPPING_VERSION,
                "evidence": evidence,
                "ambiguities": ambiguities + ["same_wet_start_end_with_multiple_changes"],
            }
        return {
            "regime": ws_l.capitalize(),
            "source": "heuristic",
            "mapping_version": WEATHER_MAPPING_VERSION,
            "evidence": evidence,
            "ambiguities": ambiguities,
        }

    return {
        "regime": None,
        "source": "unknown",
        "mapping_version": WEATHER_MAPPING_VERSION,
        "evidence": evidence,
        "ambiguities": ["unrecognized_weather_tokens"],
    }


def _seat_in_events(events: Any, seat: Any) -> bool:
    if not isinstance(events, list) or not events:
        return False
    if seat is None:
        return False
    for ev in events:
        if isinstance(ev, dict) and ev.get("seat") == seat:
            return True
    return False


def normalize_match_record(row: dict[str, Any], required_laps: int = 4) -> dict[str, Any]:
    """Stage 1: Normalize raw match/simulator record into auditable facts."""
    rec = row.get("recording_facts") if isinstance(row.get("recording_facts"), dict) else row
    obs = row.get("observed_result") if isinstance(row.get("observed_result"), dict) else {}

    seat = row.get("seat") if row.get("seat") is not None else rec.get("seat")
    track_raw = row.get("track_id") or rec.get("track_id") or ""
    track = str(track_raw).strip().lower()

    race_status = str(rec.get("race_status") or "").strip().lower()
    closure_reason = rec.get("race_closure_reason")

    retired_self = _seat_in_events(rec.get("retirements"), seat)
    rejected_self = _seat_in_events(rec.get("design_rejections"), seat)
    status_failed = race_status in _FAILURE_STATUSES

    explicit_cls = strict_bool(rec.get("classified"))
    explicit_fin = strict_bool(rec.get("finished"))
    ft = rec.get("finish_time_s")
    ft_valid = positive_finite(ft)

    # Conservative classification resolution with provenance
    if explicit_cls is not None:
        classified: bool | None = explicit_cls
        cls_source = "official"
    elif retired_self or rejected_self or status_failed:
        classified = False
        cls_source = "legacy_inferred"
    elif explicit_fin is True and ft_valid and not retired_self and not rejected_self:
        classified = True
        cls_source = "legacy_inferred"
    elif explicit_fin is False:
        classified = False
        cls_source = "legacy_inferred"
    else:
        classified = None
        cls_source = "unknown"

    legal: bool | None
    if rejected_self or race_status in {"disqualified", "dq", "illegal"}:
        legal = False
    elif classified is True or (explicit_fin is True and not rejected_self):
        legal = True
    elif classified is False:
        legal = True  # Non-classification due to DNF isn't automatically illegal unless flagged
    else:
        legal = None

    terminal_failure: bool | None
    if retired_self or rejected_self or status_failed or classified is False or explicit_fin is False:
        terminal_failure = True
    elif classified is True and explicit_fin is not False:
        terminal_failure = False
    else:
        terminal_failure = None

    # Distance verification: require full target distance (L_run == L_required)
    completed_laps = rec.get("completed_laps")
    winner_laps = rec.get("winner_completed_laps")
    finish_laps = rec.get("finish_laps")

    observed_laps = completed_laps if isinstance(completed_laps, int) and not isinstance(completed_laps, bool) else (
        finish_laps if isinstance(finish_laps, int) and not isinstance(finish_laps, bool) else None
    )

    if observed_laps is not None:
        if observed_laps < required_laps:
            full_distance_completed: bool | None = False
        elif isinstance(winner_laps, int) and not isinstance(winner_laps, bool) and observed_laps != winner_laps:
            full_distance_completed = False
        elif observed_laps == required_laps:
            full_distance_completed = True
        else:
            full_distance_completed = False
    else:
        full_distance_completed = False if terminal_failure is True else None

    # Timing compatibility
    if ft_valid and full_distance_completed is True:
        timing_compatible: bool | None = True
    elif full_distance_completed is False:
        timing_compatible = False
    else:
        timing_compatible = None

    weather_info = infer_weather_regime(rec)

    return {
        "match_id": row.get("match_id"),
        "track": track,
        "track_display": track.capitalize() if track else "Unknown",
        "seat": seat,
        "rank": obs.get("rank"),
        "required_laps": required_laps,
        "observed_laps": observed_laps,
        "winner_completed_laps": winner_laps,
        "race_closure_reason": closure_reason,
        "classified": classified,
        "classification_source": cls_source,
        "legal": legal,
        "terminal_failure": terminal_failure,
        "full_distance_completed": full_distance_completed,
        "finish_time_s": float(ft) if ft_valid else None,
        "timing_compatible": timing_compatible,
        "weather": weather_info,
        "same_match_winner_time_s": float(rec["winner_time_s"]) if positive_finite(rec.get("winner_time_s")) else None,
    }


def score_normalized_run(
    norm: dict[str, Any],
    *,
    reference_applicable: bool,
    reference_time_s: float | None,
    unavailable_reason: str | None = None,
) -> dict[str, Any]:
    """Stage 3 core: Score a single normalized run against a single validated reference."""
    thresholds = {f"le_{round(100 * t)}pct": t for t in THRESHOLD_FRACTIONS}
    out: dict[str, Any] = {
        "reference_applicable": False,
        "reference_time_s": reference_time_s if positive_finite(reference_time_s) else None,
        "unavailable_reason": unavailable_reason,
        "status": "reference_unavailable",
        "exclusion_reason": unavailable_reason,
        "gap_s": None,
        "gap_fraction": None,
        "gap_pct": None,
        "passes": {k: None for k in thresholds},
    }

    if reference_applicable is not True or not positive_finite(reference_time_s):
        if out["unavailable_reason"] is None:
            out["unavailable_reason"] = "invalid_or_missing_reference_time"
            out["exclusion_reason"] = out["unavailable_reason"]
        return out

    ref = float(reference_time_s)
    out["reference_applicable"] = True
    out["unavailable_reason"] = None

    required = [
        norm.get("legal"),
        norm.get("classified"),
        norm.get("full_distance_completed"),
    ]

    # Known unsuccessful outcomes remain in the eligible denominator as 0 (False)
    if norm.get("terminal_failure") is True or any(v is False for v in required):
        reasons = []
        if norm.get("terminal_failure") is True:
            reasons.append("terminal_failure")
        if norm.get("legal") is False:
            reasons.append("illegal")
        if norm.get("classified") is False:
            reasons.append("not_classified")
        if norm.get("full_distance_completed") is False:
            reasons.append("incomplete_distance")
        out.update(
            status="known_failure",
            exclusion_reason="+".join(reasons) or "known_failure",
            passes={k: False for k in thresholds},
        )
        return out

    # Missing evidence is kept distinct from known driving failures
    if norm.get("terminal_failure") is not False or not all(v is True for v in required):
        out.update(status="measurement_unknown", exclusion_reason="unknown_status_or_distance")
        return out

    t = norm.get("finish_time_s")
    if norm.get("timing_compatible") is not True or not positive_finite(t):
        out.update(status="measurement_unknown", exclusion_reason="incompatible_or_missing_timing")
        return out

    t_val = float(t)
    gap_s = t_val - ref
    gap_frac = gap_s / ref
    out.update(
        status="comparable_finish",
        exclusion_reason=None,
        gap_s=round(gap_s, 4),
        gap_fraction=round(gap_frac, 6),
        gap_pct=round(100.0 * gap_frac, 3),
        passes={
            k: t_val <= ref * (1.0 + thresh) + 1e-9
            for k, thresh in thresholds.items()
        },
    )
    return out


def summarize_scored_slice(entries: list[tuple[dict[str, Any], dict[str, Any]]]) -> dict[str, Any]:
    """Aggregate normalized runs + their scored results into an auditable summary."""
    total = len(entries)
    cls_known = [n for n, _ in entries if n["classified"] is not None]
    cls_true = [n for n in cls_known if n["classified"] is True]
    cls_sources = Counter(n["classification_source"] for n, _ in entries)
    weather_sources = Counter(n["weather"]["source"] for n, _ in entries)

    applicable = [(n, s) for n, s in entries if s["reference_applicable"]]
    unavail_reasons = Counter(s["unavailable_reason"] for _, s in entries if not s["reference_applicable"])
    known_fail = [(n, s) for n, s in applicable if s["status"] == "known_failure"]
    meas_unk = [(n, s) for n, s in applicable if s["status"] == "measurement_unknown"]
    comp_fin = [(n, s) for n, s in applicable if s["status"] == "comparable_finish"]
    cont_excl = Counter(s["exclusion_reason"] for _, s in applicable if s["status"] != "comparable_finish")

    n_elig = len(applicable)
    u_cnt = len(meas_unk)

    thresh_keys = [f"le_{round(100 * t)}pct" for t in THRESHOLD_FRACTIONS]
    succ_counts: dict[str, int] = {}
    comp_rates: dict[str, float | None] = {}
    bounds: dict[str, list[float] | None] = {}

    for k in thresh_keys:
        s_cnt = sum(1 for _, sc in comp_fin if sc["passes"].get(k) is True)
        succ_counts[k] = s_cnt
        if n_elig > 0:
            lo = round(s_cnt / n_elig, 4)
            hi = round((s_cnt + u_cnt) / n_elig, 4)
            comp_rates[k] = lo
            bounds[k] = [lo, hi]
        else:
            comp_rates[k] = None
            bounds[k] = None

    gaps_s = [sc["gap_s"] for _, sc in comp_fin if sc["gap_s"] is not None]
    gaps_pct = [sc["gap_pct"] for _, sc in comp_fin if sc["gap_pct"] is not None]

    return {
        "total_attempts": total,
        "classification_known_n": len(cls_known),
        "classified_n": len(cls_true),
        "classified_rate_among_known": round(len(cls_true) / len(cls_known), 4) if cls_known else None,
        "classification_by_source": dict(cls_sources),
        "weather_provenance_counts": dict(weather_sources),
        "reference_applicable_n": n_elig,
        "reference_coverage": round(n_elig / total, 4) if total else None,
        "known_outcome_failure_n": len(known_fail),
        "measurement_unknown_n": u_cnt,
        "valid_full_distance_gap_n": len(comp_fin),
        "competitive_success_n_by_threshold": succ_counts,
        "competitive_completion_rate_by_threshold": comp_rates,
        "rate_uncertainty_bounds_by_threshold": bounds,
        "median_gap_s_among_comparable_finishes": round(statistics.median(gaps_s), 2) if gaps_s else None,
        "min_gap_s_among_comparable_finishes": round(min(gaps_s), 2) if gaps_s else None,
        "median_gap_pct_among_comparable_finishes": round(statistics.median(gaps_pct), 2) if gaps_pct else None,
        "min_gap_pct_among_comparable_finishes": round(min(gaps_pct), 2) if gaps_pct else None,
        "reference_unavailable_by_reason": dict(unavail_reasons),
        "continuous_gap_exclusions_by_reason": dict(cont_excl),
    }


def evaluate(matches_path: Path, winners_path: Path, out_path: Path | None = None) -> dict[str, Any]:
    w_doc = json.loads(winners_path.read_text(encoding="utf-8"))
    meta = w_doc.get("metadata") or {}
    required_laps = int(meta.get("required_laps", 4))
    ref_version = str(meta.get("reference_version", "unversioned"))
    winners = {
        tr.lower(): {w: t for w, t in d.items()}
        for tr, d in w_doc["agx_winner_times_seconds"].items()
    }

    ext_entries: list[tuple[dict[str, Any], dict[str, Any]]] = []
    sm_entries: list[tuple[dict[str, Any], dict[str, Any]]] = []

    for line in matches_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        norm = normalize_match_record(row, required_laps=required_laps)

        # Resolve external reference
        tr = norm["track"]
        regime = norm["weather"]["regime"]
        if tr not in winners:
            ext_sc = score_normalized_run(norm, reference_applicable=False, reference_time_s=None, unavailable_reason="unknown_track")
        elif regime is None:
            ext_sc = score_normalized_run(norm, reference_applicable=False, reference_time_s=None, unavailable_reason="unknown_weather")
        elif winners[tr].get(regime) is None:
            ext_sc = score_normalized_run(norm, reference_applicable=False, reference_time_s=None, unavailable_reason=f"null_reference:{tr}/{regime}")
        else:
            ext_sc = score_normalized_run(norm, reference_applicable=True, reference_time_s=winners[tr][regime])
        ext_entries.append((norm, ext_sc))

        # Resolve same-match winner reference separately
        sm_t = norm["same_match_winner_time_s"]
        wl = norm["winner_completed_laps"]
        if sm_t is None:
            sm_sc = score_normalized_run(norm, reference_applicable=False, reference_time_s=None, unavailable_reason="missing_same_match_winner_time")
        elif isinstance(wl, int) and not isinstance(wl, bool) and wl != required_laps:
            sm_sc = score_normalized_run(norm, reference_applicable=False, reference_time_s=None, unavailable_reason="winner_incomplete_distance")
        else:
            sm_sc = score_normalized_run(norm, reference_applicable=True, reference_time_s=sm_t)
        sm_entries.append((norm, sm_sc))

    def group_summaries(entries: list[tuple[dict[str, Any], dict[str, Any]]]) -> dict[str, Any]:
        by_track: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
        by_tw: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
        for n, s in entries:
            by_track[n["track_display"]].append((n, s))
            w_lbl = n["weather"]["regime"] or "Unknown"
            by_tw[f"{n['track_display']} / {w_lbl}"].append((n, s))
        top_finishes = sorted(
            [
                {
                    "match_id": n["match_id"],
                    "track": n["track_display"],
                    "weather": n["weather"]["regime"],
                    "weather_source": n["weather"]["source"],
                    "observed_laps": n["observed_laps"],
                    "finish_time_s": n["finish_time_s"],
                    "reference_time_s": s["reference_time_s"],
                    "gap_s": s["gap_s"],
                    "gap_pct": s["gap_pct"],
                    "rank": n["rank"],
                }
                for n, s in entries
                if s["status"] == "comparable_finish"
            ],
            key=lambda x: x["gap_pct"],
        )[:5]
        return {
            "overall": summarize_scored_slice(entries),
            "by_track": {k: summarize_scored_slice(v) for k, v in sorted(by_track.items())},
            "by_track_weather": {k: summarize_scored_slice(v) for k, v in sorted(by_tw.items())},
            "top_5_comparable_finishes": top_finishes,
        }

    report = {
        "evaluator_version": EVALUATOR_VERSION,
        "reference_version": ref_version,
        "weather_mapping_version": WEATHER_MAPPING_VERSION,
        "required_laps": required_laps,
        "source": str(matches_path),
        "winners_benchmark": str(winners_path),
        "external_reference": group_summaries(ext_entries),
        "same_match_winner": group_summaries(sm_entries),
    }

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="Auditable pace-gap evaluator (v2) for AGX racing runs.")
    ap.add_argument("--matches", type=Path, default=DEFAULT_MATCHES)
    ap.add_argument("--winners", type=Path, default=DEFAULT_WINNERS)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    rep = evaluate(args.matches, args.winners, args.out)
    print(json.dumps(rep["external_reference"]["overall"], indent=2))
    print("\n--- BY TRACK (external_reference) ---")
    for tr, d in rep["external_reference"]["by_track"].items():
        print(
            f"{tr:10s}: n={d['total_attempts']:2d} elig={d['reference_applicable_n']:2d} "
            f"cls={d['classified_n']:2d} comp_fin={d['valid_full_distance_gap_n']:2d} "
            f"rate@5%={d['competitive_completion_rate_by_threshold']['le_5pct']} "
            f"rate@10%={d['competitive_completion_rate_by_threshold']['le_10pct']} "
            f"rate@20%={d['competitive_completion_rate_by_threshold']['le_20pct']} "
            f"med_gap={d['median_gap_s_among_comparable_finishes']}s ({d['median_gap_pct_among_comparable_finishes']}%)"
        )
    print("\n--- SAME-MATCH WINNER OVERALL ---")
    print(json.dumps(rep["same_match_winner"]["overall"], indent=2))


if __name__ == "__main__":
    main()
