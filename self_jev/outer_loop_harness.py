"""Automated between-match outer-loop feedback, grading, escalation, packaging & submission harness.

Supports both single-shot and fully automated daemon modes:
  1) Live agx.cool API polling (--poll-agx): detects finished matches via self_jev.observe.own_matches(),
     downloads training.jsonl, grades, runs Car-Build NN (+ optional LLM escalation API), repacks jev_rl.zip,
     and optionally auto-submits via self_jev.submit.submit_zip().
  2) Folder watcher (--watch-dir): watches a directory for newly exported *.jsonl logs and runs the full loop.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import stat
import tempfile
import time
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from self_jev.car_build_nn.adapter import TaskSpecificModelScorer
from self_jev.car_build_nn.builds import ABSTAIN_LABEL, get_candidate_builds
from self_jev.car_build_nn.dataset import FEATURE_NAMES
from self_jev.sim.design import CarParams, load_design, validate_design_limits
from self_jev.sim.track import load_track

ROOT = Path(__file__).resolve().parents[1]
SUB_DIR = ROOT / "submissions" / "racing" / "jev_rl"
ZIP_PATH = ROOT / "submissions" / "racing" / "jev_rl.zip"
STATE_DIR = ROOT / "state" / "outer_loop"
HISTORY_PATH = STATE_DIR / "history.json"
SEEN_PATH = STATE_DIR / "seen.json"
BACKUP_DIR = STATE_DIR / "last_promoted_backup"
DOWNLOAD_DIR = ROOT / "data" / "submission_data" / "auto_exports"
DEFAULT_CAR_BUILD_NPZ = ROOT / "models" / "car_build_nn.npz"


def _track_id_for_length(track_len: float) -> str:
    if track_len > 6500.0:
        return "highland"
    if track_len > 5800.0:
        return "crossover"
    return "parkland"


def summarize_match_jsonl(jsonl_path: Path) -> dict[str, Any]:
    """Parse an owned agx.cool training.jsonl into match diagnostics and the 12-float NN feature vector."""
    seat: int | None = None
    export_complete = False
    outcome: dict[str, Any] = {}
    decisions: list[dict[str, Any]] = []

    with jsonl_path.open("r", encoding="utf-8") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            row = json.loads(raw)
            rtype = row.get("type")
            if rtype == "metadata":
                if "seat" in row:
                    seat = int(row["seat"])
                continue
            if rtype == "outcome":
                outcome = row
                continue
            if rtype == "export_complete":
                export_complete = True
                continue
            if isinstance(row.get("observation"), dict):
                if seat is None or row.get("seat") is None or int(row["seat"]) == seat:
                    decisions.append(row)

    if not decisions:
        raise ValueError(f"No decision rows found in {jsonl_path}")

    first_obs = decisions[0]["observation"]
    last_obs = decisions[-1]["observation"]

    track_len = float(first_obs.get("track_length_m") or last_obs.get("track_length_m") or 5795.84)
    track_id = _track_id_for_length(track_len)

    forecast = first_obs.get("forecast") or {}
    rain_prob = float(forecast.get("rain_probability", (first_obs.get("weather") or {}).get("track_wetness", 0.0)))
    wet_start = float((first_obs.get("weather") or {}).get("track_wetness", 0.0))
    wet_end = float((last_obs.get("weather") or {}).get("track_wetness", wet_start))
    track_temp = float((first_obs.get("weather") or {}).get("track_temp_c", 26.0))

    speeds: list[float] = []
    g_lats: list[float] = []
    g_longs: list[float] = []
    off_ticks = 0
    ers_idle = 0
    max_lap = 0
    max_pits = 0
    max_wear = 0.0
    wet_waiver = False
    missing_fields = 0

    for row in decisions:
        obs = row["observation"]
        s = obs.get("self") or {}
        w = obs.get("weather") or {}
        act = row.get("action") or {}

        spd = s.get("speed_mps")
        if spd is None:
            missing_fields += 1
        else:
            speeds.append(float(spd))

        if s.get("g_lat") is not None:
            g_lats.append(abs(float(s["g_lat"])))
        if s.get("g_long") is not None:
            g_longs.append(float(s["g_long"]))

        n_val = float(s.get("n", 0.0))
        pv_rows = (obs.get("preview") or {}).get("rows") or []
        if pv_rows:
            wl = float(pv_rows[0][3])
            wr = float(pv_rows[0][4])
            if n_val > wl or n_val < -wr:
                off_ticks += 1

        if act.get("ers_mode") in (0, 1):
            ers_idle += 1

        max_lap = max(max_lap, int(s.get("lap", 0)))
        max_pits = max(max_pits, int(s.get("pit_stops", 0)))
        wear_raw = s.get("tyre_wear") or 0.0
        wear_val = float(np.mean(wear_raw)) if isinstance(wear_raw, list) else float(wear_raw)
        max_wear = max(max_wear, wear_val)
        if obs.get("phase") == "race" and float(w.get("track_wetness", 0.0)) >= 0.5:
            wet_waiver = True

    n_ticks = len(decisions)
    off_track_rate = off_ticks / max(1, n_ticks)
    wear_rate = max_wear / max(1, max_lap)
    ers_idle_rate = ers_idle / max(1, n_ticks)
    sensor_missing = 1.0 if (missing_fields > 0 or not export_complete) else 0.0
    forecast_conf = 0.90 if sensor_missing == 0.0 else 0.30

    features = [
        round(track_len, 1),
        1.0 if track_id == "parkland" else 0.0,
        1.0 if track_id == "highland" else 0.0,
        1.0 if track_id == "crossover" else 0.0,
        round(rain_prob, 3),
        round(wet_start, 3),
        round(wet_end, 3),
        round(track_temp, 1),
        round(off_track_rate, 4),
        round(wear_rate, 4),
        round(forecast_conf, 3),
        sensor_missing,
    ]

    last_self = last_obs.get("self") or {}
    return {
        "source_file": str(jsonl_path),
        "export_complete": export_complete,
        "seat": seat,
        "phase": str(last_obs.get("phase", "race")),
        "track_id": track_id,
        "track_length_m": round(track_len, 2),
        "ticks": n_ticks,
        "laps_completed": max_lap,
        "pit_stops": max_pits,
        "wet_waiver": wet_waiver,
        "final_status": str(last_self.get("status", "unknown")),
        "mean_speed_mps": round(float(np.mean(speeds)), 2) if speeds else 0.0,
        "p95_speed_mps": round(float(np.percentile(speeds, 95)), 2) if speeds else 0.0,
        "max_speed_mps": round(float(np.max(speeds)), 2) if speeds else 0.0,
        "p95_abs_g_lat": round(float(np.percentile(g_lats, 95)), 3) if g_lats else 0.0,
        "min_g_long": round(float(np.min(g_longs)), 3) if g_longs else 0.0,
        "off_track_rate": round(off_track_rate, 4),
        "wear_rate": round(wear_rate, 4),
        "ers_idle_rate": round(ers_idle_rate, 4),
        "outcome": outcome,
        "feature_names": list(FEATURE_NAMES),
        "features": features,
    }


def diagnose_match(summary: dict[str, Any], history: list[dict[str, Any]]) -> dict[str, Any]:
    """Grade whether the match was GOOD or BAD, list failure flags, and decide promote/rollback."""
    flags: list[str] = []
    if summary["final_status"] not in ("finished", "racing", "unknown"):
        flags.append(f"RETIRED:{summary['final_status']}")
    if summary["off_track_rate"] > 0.05:
        flags.append(f"HIGH_OFF_TRACK({summary['off_track_rate']*100:.1f}%)")
    if summary["wear_rate"] > 0.12:
        flags.append(f"HIGH_TYRE_WEAR({summary['wear_rate']:.3f}/lap)")
    if summary["ers_idle_rate"] > 0.50:
        flags.append(f"LOW_ERS_USAGE(idle={summary['ers_idle_rate']*100:.0f}%)")
    if summary["phase"] == "race" and not summary["wet_waiver"] and summary["pit_stops"] == 0 and summary["laps_completed"] >= 4:
        flags.append("MISSED_DRY_2CMP_PIT")
    if not summary["export_complete"]:
        flags.append("INCOMPLETE_EXPORT")

    prior_same = [h for h in history if h.get("summary", {}).get("track_id") == summary["track_id"]]
    prev_best_v = max((h["summary"]["mean_speed_mps"] for h in prior_same), default=0.0)
    improved_pace = summary["mean_speed_mps"] > prev_best_v + 0.3

    is_bad = any(f.startswith(("RETIRED", "HIGH_OFF_TRACK", "MISSED_DRY_2CMP_PIT")) for f in flags)
    if is_bad:
        verdict = "BAD_ROLLBACK_OR_FIX"
    elif improved_pace or not prior_same:
        verdict = "GOOD_PROMOTE"
    else:
        verdict = "NEUTRAL_HOLD"

    needs_recal = (summary["mean_speed_mps"] > 42.0) or (len(flags) >= 2)
    return {
        "verdict": verdict,
        "flags": flags,
        "prev_best_mean_speed_mps": round(prev_best_v, 2),
        "speed_delta_mps": round(summary["mean_speed_mps"] - prev_best_v, 2) if prior_same else None,
        "needs_sim_recalibration": needs_recal,
    }


def build_llm_escalation_prompt(summary: dict[str, Any], diag: dict[str, Any], nn_res: dict[str, Any]) -> str:
    builds = get_candidate_builds()
    opt_lines = [
        f"  - {name}: wings={c.front_wing:.0f}/{c.rear_wing:.0f}, bias={c.bias_front_pct:.0f}%, fd={c.final_drive:.2f}, ballast={c.ballast_kg:.0f}kg, start={c.start_compound}"
        for name, c in builds.items()
    ]
    return (
        f"Match Feedback Report ({summary['track_id']}, {summary['phase']}):\n"
        f"- Result: verdict={diag['verdict']}, flags={diag['flags'] or ['NONE']}\n"
        f"- Pace: mean_v={summary['mean_speed_mps']} m/s, p95_v={summary['p95_speed_mps']} m/s, max_v={summary['max_speed_mps']} m/s\n"
        f"- Grip/Wear: p95_|g_lat|={summary['p95_abs_g_lat']}g, min_g_long={summary['min_g_long']}g, off_track={summary['off_track_rate']*100:.2f}%, wear={summary['wear_rate']:.3f}/lap\n"
        f"- Weather: rain_prob={summary['features'][4]}, wet={summary['features'][5]}->{summary['features'][6]}, temp={summary['features'][7]}C\n"
        f"- Car-Build NN output: choice={nn_res['choice']} (conf={nn_res['confidence']:.3f}, escalated=True)\n"
        f"Available builds:\n" + "\n".join(opt_lines) + "\n"
        f"Return a JSON decision choosing the safest next build (or 'Abstain / Unresolved') and 1-sentence justification."
    )


def _call_decision_api(api_url: str, prompt: str, options: list[str]) -> dict[str, Any] | None:
    """Optionally call local scripts/serve_decision_api.py for automatic LLM escalation."""
    try:
        payload = json.dumps({
            "context": prompt,
            "question": "Which car build should be selected for the next match?",
            "options": options,
        }).encode("utf-8")
        req = urllib.request.Request(
            api_url.rstrip("/") + "/decide",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        return {"error": str(exc)}


def _clamp_to_agx_safe_design(car: CarParams) -> dict[str, Any]:
    """Keep design.json strictly inside proven agx.cool admission bounds."""
    base_raw = json.loads((SUB_DIR / "design.json").read_text(encoding="utf-8"))
    d = base_raw["design"]
    d["aero"]["front_wing"] = round(float(np.clip(car.front_wing, 15.0, 24.0)), 1)
    d["aero"]["rear_wing"] = round(float(np.clip(car.rear_wing, 16.0, 26.0)), 1)
    d["brakes"]["bias_front_pct"] = round(float(np.clip(car.bias_front_pct, 57.0, 61.0)), 1)
    d["gearbox"]["final_drive"] = round(float(np.clip(car.final_drive, 5.1, 5.5)), 2)
    d["mass"]["ballast_kg"] = round(float(np.clip(car.ballast_kg, 16.0, 40.0)), 1)
    d["mass"]["weight_front_pct"] = round(float(np.clip(car.weight_front_pct, 44.0, 46.0)), 1)
    d["tyres"]["pressure_front_psi"] = max(29.0, float(car.pressure_front_psi))
    d["tyres"]["pressure_rear_psi"] = max(28.0, float(car.pressure_rear_psi))
    d["tyres"]["start_compound"] = car.start_compound if car.start_compound in ("soft", "medium", "hard") else "soft"
    return {"design": d}


def pack_strict_agx_zip(sub_dir: Path = SUB_DIR, zip_path: Path = ZIP_PATH) -> Path:
    """Pack ONLY agent.py, design.json, and weights/*.npz with official pack.py ZipInfo headers."""
    files: dict[str, bytes] = {}
    for p in sorted(sub_dir.iterdir()):
        if p.is_file() and (p.suffix == ".py" or p.name == "design.json"):
            files[p.name] = p.read_bytes()
    for p in sorted((sub_dir / "weights").glob("*.npz")):
        if p.is_file():
            files[f"weights/{p.name}"] = p.read_bytes()

    assert "agent.py" in files and "design.json" in files, "Missing agent.py or design.json"
    assert len(files) <= 64, f"Too many zip entries: {len(files)}"

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for name, raw in sorted(files.items()):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.external_attr = (stat.S_IFREG | 0o644) << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(info, raw)

    with tempfile.TemporaryDirectory() as td:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(td)
        spec = importlib.util.spec_from_file_location("jev_harness_smoke", Path(td) / "agent.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod.init({"weights_dir": str(Path(td))})
        out = mod.act({"phase": "qualifying", "track_length_m": 5795.84, "self": {"speed_mps": 10.0}, "preview": {"rows": []}, "rivals": [], "weather": {"track_wetness": 0.0}})
        assert isinstance(out, dict)
    return zip_path


def _load_json(path: Path, default: Any) -> Any:
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return default


def _save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def backup_current_submission() -> None:
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SUB_DIR / "design.json", BACKUP_DIR / "design.json")
    shutil.copy2(SUB_DIR / "agent.py", BACKUP_DIR / "agent.py")


def rollback_submission() -> None:
    assert (BACKUP_DIR / "design.json").is_file(), f"No backup found in {BACKUP_DIR}"
    shutil.copy2(BACKUP_DIR / "design.json", SUB_DIR / "design.json")
    shutil.copy2(BACKUP_DIR / "agent.py", SUB_DIR / "agent.py")
    pack_strict_agx_zip()
    print(f"ROLLED BACK design.json & agent.py from {BACKUP_DIR} and repacked {ZIP_PATH}")


def _download_agx_training_jsonl(match_id: str) -> Path | None:
    """Download owned match training.jsonl from agx.cool API."""
    from self_jev.agx import credentials

    base, key = credentials()
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    dest = DOWNLOAD_DIR / f"training-{match_id}.jsonl"
    if dest.is_file() and dest.stat().st_size > 100:
        return dest

    for ep in (
        f"/api/v1/matches/{match_id}/training-data",
        f"/api/v1/matches/{match_id}/export",
        f"/api/v1/matches/{match_id}/training.jsonl",
    ):
        req = urllib.request.Request(base + ep, headers={"Authorization": "Bearer " + key})
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                raw = resp.read()
                if b"observation" in raw:
                    dest.write_bytes(raw)
                    return dest
        except Exception:
            continue
    return None


def run_harness(
    jsonl_path: Path,
    car_build_npz: Path = DEFAULT_CAR_BUILD_NPZ,
    apply_and_pack: bool = False,
    auto_submit: bool = False,
    decision_api_url: str | None = None,
    conf_threshold: float = 0.70,
    margin_threshold: float = 0.05,
) -> dict[str, Any]:
    history = _load_json(HISTORY_PATH, [])
    summary = summarize_match_jsonl(jsonl_path)
    diag = diagnose_match(summary, history)

    if not car_build_npz.is_file():
        raise FileNotFoundError(f"Car-Build NN weights not found: {car_build_npz}")
    scorer = TaskSpecificModelScorer(car_build_npz)
    nn_res = scorer.score(summary["features"])
    escalate = scorer.should_escalate(nn_res, conf_threshold=conf_threshold, margin_threshold=margin_threshold)
    if nn_res["choice"] == ABSTAIN_LABEL:
        escalate = True

    llm_prompt = build_llm_escalation_prompt(summary, diag, nn_res) if (escalate or diag["verdict"] == "BAD_ROLLBACK_OR_FIX") else None
    llm_response = None
    chosen_name = nn_res["choice"]

    if llm_prompt and decision_api_url:
        llm_response = _call_decision_api(decision_api_url, llm_prompt, list(get_candidate_builds().keys()) + [ABSTAIN_LABEL])
        if isinstance(llm_response, dict) and llm_response.get("choice") in get_candidate_builds():
            chosen_name = llm_response["choice"]
            escalate = False

    action_taken = "DRY_RUN_NO_FILES_MODIFIED"
    submit_receipt = None
    repacked = False

    if apply_and_pack:
        if diag["verdict"] == "GOOD_PROMOTE":
            backup_current_submission()
            action_taken = "PROMOTED_AND_BACKED_UP_CURRENT_SUBMISSION"
        elif not escalate and chosen_name in get_candidate_builds():
            if not (BACKUP_DIR / "design.json").is_file():
                backup_current_submission()
            chosen_car = get_candidate_builds()[chosen_name]
            safe_design = _clamp_to_agx_safe_design(chosen_car)
            tmp_path = STATE_DIR / "candidate_design.json"
            STATE_DIR.mkdir(parents=True, exist_ok=True)
            tmp_path.write_text(json.dumps(safe_design, indent=2) + "\n", encoding="utf-8")
            cand_params = load_design(tmp_path)
            for tid in ("parkland", "highland", "crossover"):
                errs = validate_design_limits(cand_params, load_track(tid).design_limits)
                assert not errs, f"Candidate design illegal on {tid}: {errs}"
            (SUB_DIR / "design.json").write_text(json.dumps(safe_design, indent=2) + "\n", encoding="utf-8")
            pack_strict_agx_zip()
            repacked = True
            action_taken = f"APPLIED_{chosen_name}_AND_REPACKED_ZIP"
        elif diag["verdict"] == "BAD_ROLLBACK_OR_FIX" and (BACKUP_DIR / "design.json").is_file():
            rollback_submission()
            repacked = True
            action_taken = "AUTO_ROLLED_BACK_TO_LAST_PROMOTED_AND_REPACKED"
        else:
            action_taken = "HELD_CURRENT_BUILD_ESCALATE_TO_LLM"

    if auto_submit and repacked:
        from self_jev.submit import submit_zip

        submit_receipt = submit_zip("racing", ZIP_PATH, select=True)
        action_taken += "_AND_SUBMITTED"

    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "summary": summary,
        "diagnosis": diag,
        "car_build_nn": nn_res,
        "escalate_to_llm": escalate,
        "llm_response": llm_response,
        "action_taken": action_taken,
        "submit_receipt": submit_receipt,
        "llm_escalation_prompt": llm_prompt,
    }
    history.append(record)
    _save_json(HISTORY_PATH, history)
    return record


def _sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def run_daemon(
    watch_dir: Path | None,
    poll_agx: bool,
    interval_s: int,
    car_build_npz: Path,
    apply_and_pack: bool,
    auto_submit: bool,
    decision_api_url: str | None,
) -> None:
    """Hands-free automation loop: polls agx.cool and/or watches a folder for new match logs."""
    seen = _load_json(SEEN_PATH, {"file_hashes": [], "match_ids": []})
    print(
        f"[AUTO-HARNESS] Started (poll_agx={poll_agx}, watch_dir={watch_dir}, "
        f"interval={interval_s}s, apply_and_pack={apply_and_pack}, auto_submit={auto_submit})",
        flush=True,
    )
    while True:
        try:
            # 1. Poll agx.cool for newly finished own-seat matches
            if poll_agx:
                from self_jev.observe import own_matches

                for m in own_matches("racing"):
                    mid = str(m["match_id"])
                    if mid in seen["match_ids"]:
                        continue
                    print(f"[AUTO-HARNESS] New finished match detected: {mid} (rank={m.get('rank')})", flush=True)
                    dl = _download_agx_training_jsonl(mid)
                    if dl is not None:
                        res = run_harness(
                            dl,
                            car_build_npz=car_build_npz,
                            apply_and_pack=apply_and_pack,
                            auto_submit=auto_submit,
                            decision_api_url=decision_api_url,
                        )
                        print(f"[AUTO-HARNESS] Match {mid} -> verdict={res['diagnosis']['verdict']} action={res['action_taken']}", flush=True)
                    else:
                        hist = _load_json(HISTORY_PATH, [])
                        hist.append({
                            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                            "match_id": mid,
                            "rank": m.get("rank"),
                            "finished_at": m.get("finished_at"),
                            "worse": m.get("worse"),
                            "action_taken": "LOGGED_MATCH_RANK_AWAITING_JSONL_EXPORT",
                        })
                        _save_json(HISTORY_PATH, hist)
                        print(
                            f"[AUTO-HARNESS] Logged match {mid} (rank={m.get('rank')}); "
                            f"drop its training.jsonl into {watch_dir or DOWNLOAD_DIR} for full telemetry grading.",
                            flush=True,
                        )
                    seen["match_ids"].append(mid)
                    _save_json(SEEN_PATH, seen)

            # 2. Watch local folder for newly dropped decision *.jsonl logs
            if watch_dir is not None and watch_dir.is_dir():
                for p in sorted(watch_dir.rglob("*.jsonl")):
                    if "events" in p.name:
                        continue
                    digest = _sha256_path(p)
                    if digest in seen["file_hashes"]:
                        continue
                    print(f"[AUTO-HARNESS] New log detected: {p}", flush=True)
                    try:
                        res = run_harness(
                            p,
                            car_build_npz=car_build_npz,
                            apply_and_pack=apply_and_pack,
                            auto_submit=auto_submit,
                            decision_api_url=decision_api_url,
                        )
                        print(f"[AUTO-HARNESS] File {p.name} -> verdict={res['diagnosis']['verdict']} action={res['action_taken']}", flush=True)
                    except ValueError as ve:
                        print(f"[AUTO-HARNESS] Skipping non-decision file {p.name}: {ve}", flush=True)
                    seen["file_hashes"].append(digest)
                    _save_json(SEEN_PATH, seen)
        except KeyboardInterrupt:
            print("\n[AUTO-HARNESS] Stopped by user.", flush=True)
            break
        except Exception as exc:
            print(f"[AUTO-HARNESS] Error (retrying in {interval_s}s): {exc}", flush=True)
        time.sleep(max(5, interval_s))


def main() -> None:
    parser = argparse.ArgumentParser(description="Automated between-match outer-loop harness.")
    parser.add_argument("--jsonl", type=Path, help="Single match training.jsonl to process once")
    parser.add_argument("--watch-dir", type=Path, default=None, help="Watch folder continuously for new *.jsonl exports")
    parser.add_argument("--poll-agx", action="store_true", help="Continuously poll agx.cool API for newly finished matches")
    parser.add_argument("--interval", type=int, default=60, help="Polling interval in seconds for daemon mode")
    parser.add_argument("--car-build-npz", type=Path, default=DEFAULT_CAR_BUILD_NPZ)
    parser.add_argument("--apply-and-pack", action="store_true", help="Auto-apply safe build/rollback and repack jev_rl.zip")
    parser.add_argument("--auto-submit", action="store_true", help="Auto-upload and publish repacked jev_rl.zip to agx.cool")
    parser.add_argument("--decision-api-url", type=str, default=None, help="Optional local serve_decision_api.py URL for auto LLM escalation")
    parser.add_argument("--rollback", action="store_true", help="Restore last promoted backup and repack jev_rl.zip")
    args = parser.parse_args()

    if args.rollback:
        rollback_submission()
        return
    if args.poll_agx or args.watch_dir is not None:
        run_daemon(
            watch_dir=args.watch_dir,
            poll_agx=args.poll_agx,
            interval_s=args.interval,
            car_build_npz=args.car_build_npz,
            apply_and_pack=args.apply_and_pack,
            auto_submit=args.auto_submit,
            decision_api_url=args.decision_api_url,
        )
        return
    if not args.jsonl:
        parser.error("Provide --poll-agx, --watch-dir <dir>, --jsonl <file>, or --rollback")

    res = run_harness(
        args.jsonl,
        car_build_npz=args.car_build_npz,
        apply_and_pack=args.apply_and_pack,
        auto_submit=args.auto_submit,
        decision_api_url=args.decision_api_url,
    )
    print(json.dumps({k: v for k, v in res.items() if k != "llm_escalation_prompt"}, indent=2))
    if res["llm_escalation_prompt"]:
        print("\n=== LLM ESCALATION / IMPROVEMENT PROMPT ===")
        print(res["llm_escalation_prompt"])


if __name__ == "__main__":
    main()
