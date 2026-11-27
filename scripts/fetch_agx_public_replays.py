#!/usr/bin/env python3
"""Acquire public agx.cool racing replays for mac_racing_sim calibration.

Endpoints (public, unauthenticated):
  GET /api/v1/matches?competition_id=<cid>&limit=&offset=      -> match list
  GET /api/v1/python-submissions/recordings/<match_id>/frames  -> gzip JSONL, all cars @24Hz
  GET /api/v1/python-submissions/recordings/<match_id>/events  -> JSONL event stream
  GET /api/v1/python-submissions/recordings/<match_id>/static  -> JSON scene metadata

Layout under data/agx_public_replays/:
  _index/matches.json                 all listed matches
  <match_id>/scene                    raw JSON scene (initial+terminal metadata)
  <match_id>/events                   raw JSONL events
  <match_id>/frames.gz                raw gzip JSONL frame stream (all cars)
  <match_id>/telemetry.npz            derived race-phase arrays (all seats)
  <match_id>/screen.json              derived classification summary
  index.json                          consolidated screen rows
  entries.json                        winner-style control traces (schema-compatible)
  manifest.json                       strict 70/30 grouped, stratified train/val split

Guardrails: read-only w.r.t. everything outside data/agx_public_replays; no secrets in
code; API key is only read from .env (and never logged) for the optional owned log.
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import random
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = REPO_ROOT / "data" / "agx_public_replays"
INDEX_DIR = OUT_ROOT / "_index"
BASE_URL = "https://agx.cool"
COMPETITION_ID = "1eb91364-950b-5725-aa4b-3501b16860ec"
TRACKS = ("parkland", "highland", "crossover")
REGIMES = ("dry", "damp", "wet", "saturated", "wetting", "drying", "variable")

# Race-phase per-seat telemetry vector (float32). Order is stable across versions.
TELEM_FIELDS = (
    "t", "s", "x", "y", "yaw", "speed", "gear", "throttle", "brake", "steer",
    "ers_soc", "ers_mode", "aero_mode", "override", "in_pit", "position",
    "tt_fl", "tt_fr", "tt_rl", "tt_rr", "wd_fl", "wd_fr", "wd_rl", "wd_rr",
    "fuel", "g_lat", "g_long", "downforce", "drag", "rpm",
)
WEATHER_FIELDS = ("track_wetness", "rain_intensity", "air_temp_c", "track_temp_c", "condition_idx")
CONDITIONS = ("dry", "damp", "wet")

_MIN_INTERVAL = 0.12  # seconds between network requests (politeness)
_throttle_lock = threading.Lock()
_last_req = [0.0]


def _throttle() -> None:
    with _throttle_lock:
        now = time.monotonic()
        wait = _MIN_INTERVAL - (now - _last_req[0])
        if wait > 0:
            time.sleep(wait)
        _last_req[0] = time.monotonic()


def _request(url: str, headers: dict[str, str] | None = None, timeout: int = 120):
    _throttle()
    req = urllib.request.Request(url, headers=headers or {"Accept": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)


def _remote_size(url: str) -> int | None:
    try:
        with _request(url, {"Range": "bytes=0-0"}) as r:
            cr = r.headers.get("Content-Range")
            if cr and "/" in cr:
                return int(cr.split("/")[-1])
            cl = r.headers.get("Content-Length")
            return int(cl) if cl else None
    except Exception:
        return None


def http_download(url: str, dest: Path, expect_gzip: bool = False, retries: int = 4) -> bool:
    """Resumable download. Skips when local file already complete & valid."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    total = _remote_size(url)
    if dest.exists():
        local = dest.stat().st_size
        if total is not None and local == total:
            if not expect_gzip or _valid_gzip(dest):
                return False  # already have it
        if total is not None and local > total:
            dest.unlink()  # corrupt oversize; restart
    for attempt in range(retries):
        try:
            start = dest.stat().st_size if dest.exists() else 0
            headers = {"Accept": "*/*"}
            mode = "wb"
            if start and (total is None or start < total):
                headers["Range"] = f"bytes={start}-"
                mode = "ab"
            with _request(url, headers) as r:
                with open(dest, mode) as f:
                    while True:
                        chunk = r.read(1 << 16)
                        if not chunk:
                            break
                        f.write(chunk)
            if expect_gzip and not _valid_gzip(dest):
                raise IOError("gzip integrity failed after download")
            if total is not None and dest.stat().st_size < total:
                raise IOError(f"short download {dest.stat().st_size}/{total}")
            return True
        except Exception as exc:  # noqa: BLE001
            if expect_gzip and dest.exists() and not _valid_gzip(dest):
                dest.unlink()
            if attempt == retries - 1:
                raise
            time.sleep(1.5 * (attempt + 1))
            print(f"    retry {attempt+1}/{retries} {dest.name}: {exc}", file=sys.stderr)
    return False


def _valid_gzip(path: Path) -> bool:
    try:
        with gzip.open(path, "rb") as f:
            while f.read(1 << 20):
                pass
        return True
    except Exception:
        return False


# ----------------------------------------------------------------------------- list
def list_matches(limit: int = 100, max_pages: int = 200) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    offset = 0
    total = None
    while True:
        url = f"{BASE_URL}/api/v1/matches?" + urllib.parse.urlencode(
            {"competition_id": COMPETITION_ID, "limit": limit, "offset": offset}
        )
        with _request(url) as r:
            doc = json.loads(r.read())
        page = doc.get("items", [])
        total = doc.get("total", total)
        items.extend(page)
        offset += limit
        if not page or len(page) < limit or (total is not None and offset >= total):
            break
        if offset >= limit * max_pages:
            break
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    (INDEX_DIR / "matches.json").write_text(
        json.dumps({"generated_at_utc": datetime.now(timezone.utc).isoformat(),
                    "competition_id": COMPETITION_ID, "total": total, "items": items}, indent=2) + "\n"
    )
    print(f"Listed {len(items)} matches (total reported {total})")
    return items


# ----------------------------------------------------------------------------- screen
def classify_regime(t: np.ndarray, wet: np.ndarray, rain: np.ndarray) -> dict[str, Any]:
    """Map a race-phase track_wetness trajectory to one of the 7 regimes."""
    if wet.size == 0:
        return {"regime": "dry", "reason": "no_data"}
    w0, w1 = float(wet[0]), float(wet[-1])
    wmin, wmax, wmean = float(wet.min()), float(wet.max()), float(wet.mean())
    amp = wmax - wmin
    d = w1 - w0
    # direction reversals on a lightly smoothed signal
    if wet.size >= 5:
        k = max(1, wet.size // 40)
        sm = np.convolve(wet, np.ones(k) / k, mode="valid")
        dif = np.diff(sm)
        dif = dif[np.abs(dif) > 0.01]
        reversals = int(np.sum(np.sign(dif[1:]) != np.sign(dif[:-1]))) if dif.size > 1 else 0
    else:
        reversals = 0
    reg: str
    if wmax < 0.03:
        reg = "dry"
    elif wmax >= 0.95:
        reg = "saturated"
    elif reversals >= 3 and amp >= 0.18:
        reg = "variable"
    elif d >= 0.15 and w0 < 0.30:
        reg = "wetting"
    elif d <= -0.15 and w0 > 0.25:
        reg = "drying"
    elif wmean >= 0.48:
        reg = "wet"
    else:
        reg = "damp"
    return {
        "regime": reg, "wet_start": round(w0, 4), "wet_end": round(w1, 4),
        "wet_min": round(wmin, 4), "wet_max": round(wmax, 4), "wet_mean": round(wmean, 4),
        "amplitude": round(amp, 4), "delta": round(d, 4), "reversals": reversals,
        "rain_max": round(float(rain.max()) if rain.size else 0.0, 4),
    }


def _parse_scene(path: Path) -> dict[str, Any]:
    meta = None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            if o.get("type") == "initial_metadata":
                meta = o["metadata"]
                break
    return meta or {}


def _seat_contacts(events: Iterable[dict], race_t0: float) -> list[dict[str, Any]]:
    out = []
    for e in events:
        if e.get("kind") != "incident":
            continue
        p = e.get("payload") or {}
        if p.get("reason") != "contact":
            continue
        out.append({"t_rel": round(float(e["time"]) - race_t0, 3),
                    "seats": e.get("seats") or [], "s": float(p.get("s", -1.0))})
    return out


def screen_match(match_id: str, force: bool = False) -> dict[str, Any] | None:
    mdir = OUT_ROOT / match_id
    scene_p, ev_p = mdir / "scene", mdir / "events"
    try:
        http_download(f"{BASE_URL}/api/v1/python-submissions/recordings/{match_id}/static", scene_p)
        http_download(f"{BASE_URL}/api/v1/python-submissions/recordings/{match_id}/events", ev_p)
    except Exception as exc:  # noqa: BLE001
        return {"match_id": match_id, "error": f"download:{exc}"}
    try:
        meta = _parse_scene(scene_p)
        events = [json.loads(l) for l in open(ev_p) if l.strip()]
    except Exception as exc:  # noqa: BLE001
        return {"match_id": match_id, "error": f"parse:{exc}"}

    track_id = (meta.get("track") or {}).get("id")
    rules = meta.get("rules_version")
    race_laps = meta.get("race_laps")
    designs = {}
    names = {}
    for c in meta.get("cars", []):
        designs[int(c["seat"])] = c.get("design_summary")
        names[int(c["seat"])] = c.get("display_name")

    # race phase window
    race_start = next((e for e in events if e.get("kind") == "phase_start" and e.get("phase") == "race"), None)
    lights = next((e for e in events if e.get("kind") == "lights_out"), None)
    race_t0 = float(race_start["time"]) if race_start else float(lights["time"]) if lights else 0.0

    # weather trajectory during race phase (from weather_change events; fall back to initial forecast)
    wc = [(float(e["time"]) - race_t0, e["payload"]) for e in events
          if e.get("kind") == "weather_change" and e.get("phase") == "race"]
    fc = meta.get("forecast") or {}
    if wc:
        wt = np.array([x[0] for x in wc], dtype=float)
        wet = np.array([float(x[1].get("track_wetness", 0.0)) for x in wc], dtype=float)
        rain = np.array([float(x[1].get("rain_intensity", 0.0)) for x in wc], dtype=float)
    else:
        wet = np.array([float(fc.get("track_wetness", 0.0))])
        rain = np.array([float(fc.get("rain_intensity", 0.0))])
        wt = np.array([0.0])

    regime = classify_regime(wt, wet, rain)
    contacts = _seat_contacts(events, race_t0)

    finished = {}
    for e in events:
        if e.get("kind") == "finished":
            for s in e.get("seats", []):
                finished[int(s)] = {"total_time_s": float(e["payload"]["total_time_s"]),
                                    "position": int(e["payload"].get("position", 0)),
                                    "lap": int(e["payload"].get("lap", 0))}
    pits = []
    for e in events:
        if e.get("kind") == "pit_exit":
            pits.append({"seats": e.get("seats") or [], "t_rel": round(float(e["time"]) - race_t0, 3),
                         **{k: e["payload"].get(k) for k in ("old_compound", "new_compound", "pit_elapsed_s")}})

    n_pit_cars = len({s for p in pits for s in p["seats"]})
    compounds = sorted({d.get("start_compound") for d in designs.values() if d} |
                       {p.get("new_compound") for p in pits if p.get("new_compound")})

    row = {
        "match_id": match_id,
        "track": track_id,
        "rules_version": rules,
        "race_laps": race_laps,
        "race_t0_s": round(race_t0, 3),
        "n_seats": len(designs),
        "regime": regime["regime"],
        "wet_stats": regime,
        "n_contact_events": len(contacts),
        "contacts": contacts[:40],
        "n_pit_cars": n_pit_cars,
        "pit_events": pits[:20],
        "compounds": compounds,
        "finished": {str(k): v for k, v in sorted(finished.items())},
        "winner_seat": (min(finished.items(), key=lambda kv: kv[1]["total_time_s"])[0]
                        if finished else None),
        "designs": {str(k): v for k, v in designs.items()},
        "names": {str(k): v for k, v in names.items()},
        "screened_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (mdir / "screen.json").write_text(json.dumps(row, indent=2) + "\n")
    return row


def screen_batch(ids: list[str], workers: int = 4) -> list[dict[str, Any]]:
    rows = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(screen_match, mid): mid for mid in ids}
        done = 0
        for fut in as_completed(futs):
            mid = futs[fut]
            try:
                r = fut.result()
            except Exception as exc:  # noqa: BLE001
                r = {"match_id": mid, "error": str(exc)}
            rows.append(r)
            done += 1
            if r and not r.get("error"):
                print(f"  [{done}/{len(ids)}] {mid[:8]} {r.get('track')}:{r.get('regime')} "
                      f"contacts={r.get('n_contact_events')} pits={r.get('n_pit_cars')}")
            else:
                print(f"  [{done}/{len(ids)}] {mid[:8]} SKIP {r.get('error') if r else ''}")
    _merge_index(rows)
    return rows


def _merge_index(rows: list[dict[str, Any]]) -> None:
    idx_p = OUT_ROOT / "index.json"
    cur = {}
    if idx_p.exists():
        for r in json.loads(idx_p.read_text()).get("matches", []):
            cur[r["match_id"]] = r
    for r in rows:
        if r:
            cur[r["match_id"]] = r
    idx_p.parent.mkdir(parents=True, exist_ok=True)
    idx_p.write_text(json.dumps({"generated_at_utc": datetime.now(timezone.utc).isoformat(),
                                 "matches": [cur[k] for k in sorted(cur)]}, indent=2) + "\n")


# ----------------------------------------------------------------------------- fetch frames
def fetch_frames(match_id: str) -> bool:
    dest = OUT_ROOT / match_id / "frames.gz"
    return http_download(f"{BASE_URL}/api/v1/python-submissions/recordings/{match_id}/frames",
                         dest, expect_gzip=True)


# ----------------------------------------------------------------------------- ingest
def ingest_match(match_id: str, primary_cars: int = 3) -> dict[str, Any] | None:
    mdir = OUT_ROOT / match_id
    scr_p = mdir / "screen.json"
    fgz = mdir / "frames.gz"
    if not (scr_p.exists() and fgz.exists()):
        return None
    scr = json.loads(scr_p.read_text())
    track_id = scr["track"]
    race_t0 = float(scr["race_t0_s"])
    designs = {int(k): v for k, v in scr["designs"].items()}
    contacts = scr.get("contacts", [])

    race_frames: list[dict] = []
    with gzip.open(fgz, "rt") as f:
        for line in f:
            if '"phase": "race"' not in line and '"phase":"race"' not in line:
                # cheap prefilter; still parse to be safe
                if '"race"' not in line:
                    continue
            o = json.loads(line)
            if o.get("phase") != "race":
                continue
            race_frames.append(o)
    if not race_frames:
        return None
    T = len(race_frames)
    seats = sorted({int(c["seat"]) for fr in race_frames for c in fr["cars"]})
    S = max(seats) + 1
    F = len(TELEM_FIELDS)
    fi = {name: i for i, name in enumerate(TELEM_FIELDS)}

    telem = np.full((T, S, F), np.nan, dtype=np.float32)
    contact_mask = np.zeros((T, S), dtype=bool)
    t_arr = np.zeros(T, dtype=np.float32)
    weather = np.zeros((T, len(WEATHER_FIELDS)), dtype=np.float32)
    lap_arr = np.zeros((T, S), dtype=np.float32)
    status_idx = np.full((T, S), -1, dtype=np.int8)
    statuses = {}

    def contact_at(t_rel: float, seat: int) -> bool:
        return any(abs(c["t_rel"] - t_rel) <= 0.25 and seat in c["seats"] for c in contacts)

    for ti, fr in enumerate(race_frames):
        t_rel = float(fr["time"]) - race_t0
        t_arr[ti] = t_rel
        w = fr.get("weather") or {}
        weather[ti] = (float(w.get("track_wetness", 0.0)), float(w.get("rain_intensity", 0.0)),
                       float(w.get("air_temp_c", 24.0)), float(w.get("track_temp_c", 30.0)),
                       CONDITIONS.index(w["condition"]) if w.get("condition") in CONDITIONS else 0.0)
        for c in fr["cars"]:
            s = int(c["seat"])
            st = str(c.get("status", "racing"))
            statuses.setdefault(st, len(statuses))
            status_idx[ti, s] = statuses[st]
            tt = c.get("tyre_temp_c") or [np.nan] * 4
            wd = c.get("tyre_wear") or [np.nan] * 4
            vals = {
                "t": t_rel, "s": c.get("s", np.nan), "x": c.get("x", np.nan), "y": c.get("y", np.nan),
                "yaw": c.get("yaw", np.nan), "speed": c.get("speed_mps", np.nan), "gear": c.get("gear", np.nan),
                "throttle": c.get("throttle", np.nan), "brake": c.get("brake", np.nan), "steer": c.get("steer", np.nan),
                "ers_soc": c.get("ers_soc_mj", np.nan), "ers_mode": c.get("ers_mode", np.nan),
                "aero_mode": c.get("aero_mode", np.nan), "override": float(bool(c.get("override_active"))),
                "in_pit": float(bool(c.get("in_pit"))), "position": c.get("position", np.nan),
                "tt_fl": tt[0], "tt_fr": tt[1], "tt_rl": tt[2], "tt_rr": tt[3],
                "wd_fl": wd[0], "wd_fr": wd[1], "wd_rl": wd[2], "wd_rr": wd[3],
                "fuel": c.get("fuel_kg", np.nan), "g_lat": c.get("g_lat", np.nan), "g_long": c.get("g_long", np.nan),
                "downforce": c.get("downforce_n", np.nan), "drag": c.get("drag_n", np.nan), "rpm": c.get("rpm", np.nan),
            }
            for k, v in vals.items():
                telem[ti, s, fi[k]] = v if v is not None else np.nan
            lap_arr[ti, s] = c.get("lap", np.nan)
            if contact_at(t_rel, s):
                contact_mask[ti, s] = True
        # store compound via a side array
        # (encoded below in a separate (T,S) array)

    # compound per (T,S)
    comp_code = np.full((T, S), 3, dtype=np.int8)
    for ti, fr in enumerate(race_frames):
        for c in fr["cars"]:
            comp_code[ti, int(c["seat"])] = {"soft": 0, "medium": 1, "hard": 2}.get(c.get("compound"), 3)

    np.savez_compressed(
        mdir / "telemetry.npz",
        telem=telem, weather=weather, t=t_arr, lap=lap_arr,
        status=status_idx, status_names=np.array(list(statuses.keys())),
        compound=comp_code, contact=contact_mask,
        fields=np.array(TELEM_FIELDS), weather_fields=np.array(WEATHER_FIELDS),
    )

    # Build winner-style entries for the fastest clean finishers
    finished = {int(k): v for k, v in (scr.get("finished") or {}).items()}
    candidates = []
    for s, fv in finished.items():
        if fv.get("lap", 0) < (scr.get("race_laps") or 4) - 1:
            continue
        n_contact = int(contact_mask[:, s].sum()) if s < contact_mask.shape[1] else 0
        candidates.append((n_contact > 0, fv["total_time_s"], s, fv, n_contact))
    candidates.sort(key=lambda x: (x[0], x[1]))
    entries = []
    for had_contact, ftime, s, fv, n_contact in candidates[:primary_cars]:
        if s >= telem.shape[1] or designs.get(s) is None:
            continue
        entries.append(_build_entry(match_id, scr, s, designs[s], fv, telem, weather, t_arr,
                                    lap_arr, comp_code, contact_mask, n_contact))
    sum_p = mdir / "entries.json"
    sum_p.write_text(json.dumps({"match_id": match_id, "track": track_id,
                                 "regime": scr["regime"], "entries": entries}, indent=2) + "\n")
    return {"match_id": match_id, "entries": len(entries), "frames_race": T, "seats": S}


def _build_entry(match_id, scr, seat, design, finish, telem, weather, t_arr, lap_arr, comp_code, contact_mask, n_contact):
    fi = {n: i for i, n in enumerate(TELEM_FIELDS)}
    N = telem.shape[0]
    cols = ["tick", "elapsed_s", "lap", "station_m", "x_m", "y_m", "heading_rad", "speed_mps",
            "track_wetness", "rain_intensity", "throttle", "brake", "steering_angle_rad", "gear",
            "aero_mode", "ers_mode", "override_active", "aid_active", "aid_mode", "in_pit", "compound"]
    comp_names = ("soft", "medium", "hard", "unknown")
    finish_t = float(finish.get("total_time_s", math.inf))
    rows = []
    for i in range(N):
        sp = telem[i, seat, fi["speed"]]
        if not np.isfinite(sp):
            continue
        t_rel = float(t_arr[i])
        if t_rel > finish_t + 1.0 / 24.0:
            break  # trace continues past this car's finish; trim here
        lap = int(np.nan_to_num(lap_arr[i, seat], nan=1))
        rows.append([
            int(round(t_rel * 60.0)), t_rel, lap,
            float(telem[i, seat, fi["s"]]), float(telem[i, seat, fi["x"]]), float(telem[i, seat, fi["y"]]),
            float(telem[i, seat, fi["yaw"]]), float(sp),
            float(weather[i, 0]), float(weather[i, 1]),
            float(np.nan_to_num(telem[i, seat, fi["throttle"]])), float(np.nan_to_num(telem[i, seat, fi["brake"]])),
            float(np.nan_to_num(telem[i, seat, fi["steer"]])), int(np.nan_to_num(telem[i, seat, fi["gear"]], nan=1)),
            int(np.nan_to_num(telem[i, seat, fi["aero_mode"]], nan=0)),
            int(np.nan_to_num(telem[i, seat, fi["ers_mode"]], nan=0)),
            bool(telem[i, seat, fi["override"]] > 0.5), False, "none",
            bool(telem[i, seat, fi["in_pit"]] > 0.5),
            comp_names[int(comp_code[i, seat])] if int(comp_code[i, seat]) < 4 else "unknown",
        ])
    face_mask = contact_mask[:, seat] if contact_mask.ndim == 2 else contact_mask
    contact_times_s = [round(float(t_arr[i]), 3) for i in np.flatnonzero(face_mask)]
    return {
        "track": scr["track"],
        "weather_profile": scr["regime"],
        "match_id": match_id,
        "match_url": f"https://agx.cool/#/matches/{match_id}",
        "winner": f"seat_{seat}",
        "winner_seat": int(seat),
        "completed_laps": int(finish.get("lap", scr.get("race_laps") or 4)),
        "official_finish_s": float(finish["total_time_s"]),
        "rules_version": scr.get("rules_version"),
        "kernel_version": "rsi-formula-4.0",
        "build": design,
        "pit_events": scr.get("pit_events", []),
        "contact_frames": int(n_contact),
        "contact_times_s": contact_times_s,
        "control_trace_columns": cols,
        "control_trace": rows,
    }


# ----------------------------------------------------------------------------- selection
def load_index() -> list[dict[str, Any]]:
    p = OUT_ROOT / "index.json"
    if not p.exists():
        return []
    return [r for r in json.loads(p.read_text())["matches"]]


def _cand_features(r: dict[str, Any]) -> dict[str, Any]:
    """Diversity buckets: pit strategy, compound mix, winner build, contact exposure."""
    comps = frozenset(c for c in (r.get("compounds") or []) if c in ("soft", "medium", "hard"))
    comps = comps | {p["new_compound"] for p in (r.get("pit_events") or []) if p.get("new_compound")}
    pit_cat = 0 if (r.get("n_pit_cars") or 0) == 0 else (1 if (r.get("n_pit_cars") or 0) <= 2 else 2)
    seat = r.get("winner_seat")
    design = (r.get("designs") or {}).get(str(seat), {})
    build_sig = json.dumps({k: design.get(k) for k in
                            ("front_wing", "rear_wing", "ballast_kg", "final_drive",
                             "brake_bias_front_pct", "radiator_opening", "fuel_load_kg",
                             "start_compound")}, sort_keys=True)
    return {
        "pit_cat": pit_cat,
        "compounds": comps,
        "build_sig": build_sig,
        "has_contacts": bool(r.get("n_contact_events")),
        "start_compounds": frozenset(c for c in comps),
    }


def select_stratified(per_cell: int = 4, max_total: int = 100, seed: int = 7,
                      want_regimes: tuple[str, ...] = REGIMES) -> list[str]:
    """Fill every track x regime cell, greedily maximizing pit/compound/build/contact diversity."""
    rows = [r for r in load_index()
            if not r.get("error") and r.get("rules_version") == "rsi-racing-4.4" and r.get("finished")]
    rng = random.Random(seed)
    by_cell: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        by_cell.setdefault((r["track"], r["regime"]), []).append(r)
    chosen: list[str] = []
    for t in TRACKS:
        for w in want_regimes:
            pool = list(by_cell.get((t, w), []))
            rng.shuffle(pool)
            feats = {r["match_id"]: _cand_features(r) for r in pool}
            picked: list[dict] = []
            seen_pit, seen_comps, seen_builds = set(), set(), set()
            while pool and len(picked) < per_cell:
                def score(r: dict) -> tuple:
                    f = feats[r["match_id"]]
                    return (
                        2 * (f["pit_cat"] not in seen_pit)
                        + 2 * len(f["compounds"] - seen_comps)
                        + (f["build_sig"] not in seen_builds)
                        + (f["has_contacts"] and not any(feats[p["match_id"]]["has_contacts"] for p in picked))
                        + (r.get("n_pit_cars", 0) in (0, 1, 2)) * 0.25,
                        -abs(r.get("n_pit_cars", 0) - 1),
                    )
                best = max(pool, key=score)
                pool.remove(best)
                picked.append(best)
                f = feats[best["match_id"]]
                seen_pit.add(f["pit_cat"]); seen_comps |= f["compounds"]; seen_builds.add(f["build_sig"])
            chosen.extend(r["match_id"] for r in picked)
            print(f"  cell {t:9s}:{w:10s} pool={len(by_cell.get((t, w), [])):3d} take={len(picked)} "
                  f"(pits={sorted(feats[r['match_id']]['pit_cat'] for r in picked)}, "
                  f"comps={sorted({c for r in picked for c in feats[r['match_id']]['compounds']})})")
    seen = set()
    out = []
    for mid in chosen:
        if mid not in seen:
            seen.add(mid)
            out.append(mid)
    return out[:max_total]


# ----------------------------------------------------------------------------- manifest
def build_manifest(ids: list[str], val_frac: float = 0.30, seed: int = 11) -> dict[str, Any]:
    idx = {r["match_id"]: r for r in load_index()}
    rows = [idx[m] for m in ids if m in idx and not idx[m].get("error") and idx[m].get("finished")]
    # group cells; shuffle with seed; then allocate a global 30/70 val quota across cells by largest
    # remainder so the split is exactly val_frac overall while every cell keeps >=1 val and >=1 train
    rng = random.Random(seed)
    by_cell: dict[tuple[str, str], list[str]] = {}
    for r in rows:
        by_cell.setdefault((r["track"], r["regime"]), []).append(r["match_id"])
    cells = sorted(by_cell)
    n_total = sum(len(by_cell[c]) for c in cells)
    target_val = int(round(val_frac * n_total))
    quotas = {}
    for c in cells:
        n = len(by_cell[c])
        raw = val_frac * n
        quotas[c] = [int(raw), raw - int(raw), n]
    assigned = sum(v[0] for v in quotas.values())
    for c in sorted(cells, key=lambda k: -quotas[k][1]):
        if assigned >= target_val:
            break
        if quotas[c][0] + 1 <= quotas[c][2] - 1:
            quotas[c][0] += 1
            assigned += 1
    train, val = [], []
    for cell in cells:
        mids = by_cell[cell]
        rng.shuffle(mids)
        n_val = min(max(1, quotas[cell][0]), max(0, len(mids) - 1)) if len(mids) >= 2 else 0
        val.extend(mids[:n_val])
        train.extend(mids[n_val:])
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "competition_id": COMPETITION_ID,
        "split_policy": "grouped_by_match_id, stratified_by_track_x_regime, 70/30; val never tuned",
        "val_fraction": val_frac,
        "n_train": len(train),
        "n_val": len(val),
        "n_total": len(train) + len(val),
        "train_match_ids": sorted(train),
        "val_match_ids": sorted(val),
        "cells": {},
    }
    for (t, w), mids in sorted(by_cell.items()):
        tr = sorted(m for m in mids if m in set(train))
        va = sorted(m for m in mids if m in set(val))
        manifest["cells"][f"{t}:{w}"] = {"n": len(mids), "train": len(tr), "val": len(va)}
    (OUT_ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


# ----------------------------------------------------------------------------- entries aggregation
def aggregate_entries() -> int:
    entries = []
    cols = None
    for mdir in sorted(OUT_ROOT.iterdir()):
        if not mdir.is_dir():
            continue
        ep = mdir / "entries.json"
        if not ep.exists():
            continue
        doc = json.loads(ep.read_text())
        for e in doc.get("entries", []):
            if cols is None:
                cols = e.pop("control_trace_columns", None)
            else:
                e.pop("control_trace_columns", None)
            entries.append(e)
    doc = {
        "schema_version": "agx-public-replays-1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "control_trace_columns": cols,
        "control_units": "SI (m, m/s, rad, s)",
        "entries": entries,
    }
    (OUT_ROOT / "entries.json").write_text(json.dumps(doc) + "\n")
    return len(entries)


# ----------------------------------------------------------------------------- CLI
def main() -> None:
    ap = argparse.ArgumentParser(description="Acquire + ingest public agx.cool racing replays.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list")

    sp = sub.add_parser("screen")
    sp.add_argument("--ids", nargs="*", default=None)
    sp.add_argument("--from-file", type=Path, default=None)
    sp.add_argument("--all-succeeded", action="store_true")
    sp.add_argument("--max", type=int, default=400)
    sp.add_argument("--workers", type=int, default=4)

    fp = sub.add_parser("fetch")
    fp.add_argument("--ids", nargs="*", default=None)
    fp.add_argument("--from-file", type=Path, default=None)

    ip = sub.add_parser("ingest")
    ip.add_argument("--ids", nargs="*", default=None)

    sel = sub.add_parser("select")
    sel.add_argument("--per-cell", type=int, default=4)
    sel.add_argument("--max-total", type=int, default=100)
    sel.add_argument("--out", type=Path, default=OUT_ROOT / "selection.json")

    mp = sub.add_parser("manifest")
    mp.add_argument("--from-file", type=Path, default=OUT_ROOT / "selection.json")

    sub.add_parser("aggregate")

    args = ap.parse_args()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    if args.cmd == "list":
        list_matches()
    elif args.cmd == "screen":
        if args.ids:
            ids = args.ids
        elif args.from_file:
            ids = json.loads(args.from_file.read_text())
            if isinstance(ids, dict):
                ids = ids.get("ids") or ids.get("match_ids")
        elif args.all_succeeded:
            items = list_matches()
            ids = [it["id"] for it in items
                   if it.get("status") == "succeeded" and it.get("replay_state") in (None, "available")]
        else:
            raise SystemExit("provide --ids, --from-file, or --all-succeeded")
        ids = ids[: args.max]
        print(f"Screening {len(ids)} matches ...")
        screen_batch(ids, workers=args.workers)
    elif args.cmd == "fetch":
        ids = args.ids or (json.loads(args.from_file.read_text()) if args.from_file else None)
        if not ids:
            raise SystemExit("provide --ids or --from-file")
        if isinstance(ids, dict):
            ids = ids.get("ids") or ids.get("match_ids")
        ok = 0
        for i, mid in enumerate(ids, 1):
            try:
                fetch_frames(mid)
                ok += 1
                print(f"  [{i}/{len(ids)}] frames {mid[:8]} ok")
            except Exception as exc:  # noqa: BLE001
                print(f"  [{i}/{len(ids)}] frames {mid[:8]} FAIL {exc}", file=sys.stderr)
        print(f"Fetched frames for {ok}/{len(ids)}")
    elif args.cmd == "ingest":
        ids = args.ids or [m.name for m in sorted(OUT_ROOT.iterdir())
                           if m.is_dir() and (m / "frames.gz").exists()]
        n = 0
        for i, mid in enumerate(ids, 1):
            try:
                r = ingest_match(mid)
                if r:
                    n += 1
                    print(f"  [{i}/{len(ids)}] ingest {mid[:8]} entries={r['entries']} T={r['frames_race']} S={r['seats']}")
            except Exception as exc:  # noqa: BLE001
                print(f"  [{i}/{len(ids)}] ingest {mid[:8]} FAIL {exc}", file=sys.stderr)
        print(f"Ingested {n} matches")
    elif args.cmd == "select":
        ids = select_stratified(per_cell=args.per_cell, max_total=args.max_total)
        args.out.write_text(json.dumps({"ids": ids, "n": len(ids)}, indent=2) + "\n")
        print(f"Selected {len(ids)} -> {args.out}")
    elif args.cmd == "manifest":
        ids = json.loads(args.from_file.read_text())
        ids = ids.get("ids") or ids.get("match_ids") if isinstance(ids, dict) else ids
        m = build_manifest(ids)
        print(f"Manifest: train={m['n_train']} val={m['n_val']} total={m['n_total']}")
    elif args.cmd == "aggregate":
        n = aggregate_entries()
        print(f"Aggregated {n} entries -> {OUT_ROOT / 'entries.json'}")


if __name__ == "__main__":
    main()