#!/usr/bin/env python3
"""Regression tests for scripts/eval_pace_accuracy.py (v2 contract checks)."""
from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_pace_accuracy as ep


class TestPaceEvalV2(unittest.TestCase):
    def test_classified_three_of_four_laps_is_known_failure_not_fast(self):
        row = {
            "match_id": "short-3-of-4",
            "track_id": "highland",
            "seat": 6,
            "recording_facts": {
                "classified": True,
                "finished": True,
                "completed_laps": 3,
                "winner_completed_laps": 4,
                "finish_laps": 3,
                "finish_time_s": 300.0,
                "race_closure_reason": "absolute_time_limit",
                "weather_start": "dry",
                "weather_end": "damp",
            },
        }
        norm = ep.normalize_match_record(row, required_laps=4)
        sc = ep.score_normalized_run(norm, reference_applicable=True, reference_time_s=400.0)
        self.assertEqual(sc["status"], "known_failure")
        self.assertIsNone(sc["gap_fraction"])
        self.assertTrue(all(v is False for v in sc["passes"].values()))

    def test_exact_5pct_boundary_passes_5pct_and_10pct(self):
        row = {
            "match_id": "boundary-5pct",
            "track_id": "parkland",
            "seat": 0,
            "recording_facts": {
                "classified": True,
                "finished": True,
                "completed_laps": 4,
                "winner_completed_laps": 4,
                "finish_time_s": 420.0,
                "weather_regime": "Dry",
            },
        }
        norm = ep.normalize_match_record(row, required_laps=4)
        sc = ep.score_normalized_run(norm, reference_applicable=True, reference_time_s=400.0)
        self.assertEqual(sc["status"], "comparable_finish")
        self.assertAlmostEqual(sc["gap_pct"], 5.0, places=3)
        self.assertFalse(sc["passes"]["le_1pct"])
        self.assertTrue(sc["passes"]["le_5pct"])
        self.assertTrue(sc["passes"]["le_10pct"])

    def test_dnf_with_positive_timestamp_is_known_failure(self):
        row = {
            "match_id": "dnf-early",
            "track_id": "crossover",
            "seat": 2,
            "recording_facts": {
                "classified": False,
                "finished": False,
                "retirements": [{"seat": 2, "reason": "crash"}],
                "finish_time_s": 40.0,
                "weather_regime": "Dry",
            },
        }
        norm = ep.normalize_match_record(row, required_laps=4)
        sc = ep.score_normalized_run(norm, reference_applicable=True, reference_time_s=400.0)
        self.assertEqual(sc["status"], "known_failure")
        self.assertIsNone(sc["gap_s"])
        self.assertFalse(sc["passes"]["le_20pct"])

    def test_missing_classification_with_ambiguous_legacy_flags_is_unknown(self):
        row = {
            "match_id": "ambiguous-legacy",
            "track_id": "parkland",
            "seat": 1,
            "recording_facts": {
                "classified": None,
                "finished": None,
                "finish_time_s": 390.0,
                "completed_laps": 4,
                "weather_regime": "Dry",
            },
        }
        norm = ep.normalize_match_record(row, required_laps=4)
        self.assertEqual(norm["classification_source"], "unknown")
        sc = ep.score_normalized_run(norm, reference_applicable=True, reference_time_s=385.31)
        self.assertEqual(sc["status"], "measurement_unknown")
        self.assertIsNone(sc["passes"]["le_5pct"])

    def test_string_false_never_treated_as_truthy(self):
        self.assertIsNone(ep.strict_bool("false"))
        self.assertIs(ep.strict_bool(False), False)

    def test_missing_weather_reduces_coverage_but_keeps_classification(self):
        row = {
            "match_id": "no-weather",
            "track_id": "parkland",
            "seat": 0,
            "recording_facts": {
                "classified": True,
                "finished": True,
                "completed_laps": 4,
                "finish_time_s": 410.0,
            },
        }
        norm = ep.normalize_match_record(row, required_laps=4)
        self.assertEqual(norm["weather"]["source"], "unknown")
        sc = ep.score_normalized_run(norm, reference_applicable=False, reference_time_s=None, unavailable_reason="unknown_weather")
        summary = ep.summarize_scored_slice([(norm, sc)])
        self.assertEqual(summary["classified_n"], 1)
        self.assertEqual(summary["reference_applicable_n"], 0)
        self.assertEqual(summary["reference_unavailable_by_reason"]["unknown_weather"], 1)

    def test_zero_or_nan_reference_or_time_never_passes(self):
        norm = ep.normalize_match_record({
            "match_id": "nan-time",
            "track_id": "parkland",
            "seat": 0,
            "recording_facts": {
                "classified": True,
                "finished": True,
                "completed_laps": 4,
                "finish_time_s": float("nan"),
                "weather_regime": "Dry",
            },
        })
        sc_zero = ep.score_normalized_run(norm, reference_applicable=True, reference_time_s=0.0)
        self.assertEqual(sc_zero["status"], "reference_unavailable")
        sc_nan = ep.score_normalized_run(norm, reference_applicable=True, reference_time_s=400.0)
        self.assertEqual(sc_nan["status"], "measurement_unknown")

    def test_slow_same_match_winner_stays_separate_from_external_gap(self):
        row = {
            "match_id": "slow-win",
            "track_id": "parkland",
            "seat": 0,
            "recording_facts": {
                "classified": True,
                "finished": True,
                "completed_laps": 4,
                "winner_completed_laps": 4,
                "finish_time_s": 420.644,
                "winner_time_s": 420.644,
                "weather_start": "dry",
                "weather_end": "damp",
            },
        }
        norm = ep.normalize_match_record(row, required_laps=4)
        ext_sc = ep.score_normalized_run(norm, reference_applicable=True, reference_time_s=377.55)
        sm_sc = ep.score_normalized_run(norm, reference_applicable=True, reference_time_s=norm["same_match_winner_time_s"])
        self.assertAlmostEqual(sm_sc["gap_pct"], 0.0, places=3)
        self.assertAlmostEqual(ext_sc["gap_pct"], 11.414, places=2)

    def test_threshold_nesting_on_real_matches(self):
        rep = ep.evaluate(ep.DEFAULT_MATCHES, ep.DEFAULT_WINNERS, out_path=None)
        for section in ("external_reference", "same_match_winner"):
            ov = rep[section]["overall"]
            counts = ov["competitive_success_n_by_threshold"]
            vals = [counts["le_0pct"], counts["le_1pct"], counts["le_5pct"], counts["le_10pct"], counts["le_20pct"]]
            self.assertEqual(vals, sorted(vals))
            self.assertLessEqual(vals[-1], ov["valid_full_distance_gap_n"])


if __name__ == "__main__":
    unittest.main()
