#!/usr/bin/env python3
"""Fit mac_racing_sim physics knobs on the *train* split of data/agx_public_replays.

Subsystem order (one at a time, per the calibration brief):
  1. wet   : nonlinear wetness -> grip curve  (wet_grip, wet_exp)
  2. wear  : tyre thermal/wear grip decay per compound (front_heat_scale; report-only by default)
  3. lat   : lateral stiffness / axle balance (mu_f, mu_r, k_scale) on open-loop cornering drift

Objective: 10 s zero-feedback open-loop windows seeded from the recorded state at race laps 1-4
(tyre wear/temp/fuel seeded from telemetry.npz), excluding car-to-car contact windows. Fitting only
touches the manifest train ids; held-out val ids are touched only by `--eval-split val` reporting.

Usage:
  python3 scripts/fit_public_physics.py wet --grid
  python3 scripts/fit_public_physics.py wet --eval-json mac_racing_sim/reports/fit_wet_after.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

SELF_JEV = Path(__file__).resolve().parents[1]
REPLAY_DIR = SELF_JEV / "data" / "agx_public_replays"
MAC = SELF_JEV / "mac_racing_sim"
sys.path.insert(0, str(SELF_JEV))
sys.path.insert(0, str(MAC))

from mac_racing_sim import DEFAULT_CALIBRATED_TUNE  # noqa: E402
from mac_racing_sim.ref_env import RefRaceSim  # noqa: E402
from mac_racing_sim.benchmark_winners import road_clearance_m  # noqa: E402

CONTACT_SKIP_S = 0.5
WINDOW_S = 10.0


def load_manifest() -> dict[str, Any]:
    return json.loads((REPLAY_DIR / "manifest.json").read_text())


def _telemetry(mdir: Path):
    tp = mdir / "telemetry.npz"
    return np.load(tp) if tp.exists() else None


def _extras(z, seat: int, elapsed_s: float) -> dict[str, Any]:
    if z is None:
        return {}
    fi = {str(n): i for i, n in enumerate(z["fields"])}
    i = int(np.argmin(np.abs(z["t"] - elapsed_s)))
    row = z["telem"][i, seat]
    wear = [float(row[fi[f]]) for f in ("wd_fl", "wd_fr", "wd_rl", "wd_rr")]
    tt = [float(row[fi[f]]) for f in ("tt_fl", "tt_fr", "tt_rl", "tt_rr")]
    if not all(math.isfinite(v) for v in wear + tt):
        return {}
    ex = {"wear": wear, "ts": tt, "tc": [v - 6.3 for v in tt]}
    fuel, soc = float(row[fi["fuel"]]), float(row[fi["ers_soc"]])
    if math.isfinite(fuel):
        ex["fuel"] = fuel
    if math.isfinite(soc):
        ex["soc"] = soc
    return ex


def _ref_end(z, seat: int, elapsed_s: float) -> dict[str, Any]:
    """Recorded per-wheel tyre temp / wear nearest to the window end (calibration target)."""
    if z is None:
        return {}
    fi = {str(n): i for i, n in enumerate(z["fields"])}
    i = int(np.argmin(np.abs(z["t"] - elapsed_s)))
    tt = [float(z["telem"][i, seat, fi[f]]) for f in ("tt_fl", "tt_fr", "tt_rl", "tt_rr")]
    wd = [float(z["telem"][i, seat, fi[f]]) for f in ("wd_fl", "wd_fr", "wd_rl", "wd_rr")]
    if not all(math.isfinite(v) for v in tt + wd):
        return {}
    return {"ts": tt, "wear": wd}


def _contact_spans(z, seat: int) -> list[tuple[float, float]]:
    if z is None:
        return []
    idx = np.flatnonzero(z["contact"][:, seat])
    return [(float(z["t"][i]) - CONTACT_SKIP_S, float(z["t"][i]) + CONTACT_SKIP_S) for i in idx]


def build_specs(ids: list[str], max_windows_per_entry: int = 3, seed: int = 3,
                window_s: float = WINDOW_S) -> list[dict[str, Any]]:
    """Control-input window specs (no physics) for every contact-free seed lap of every entry."""
    rng = random.Random(seed)
    specs: list[dict[str, Any]] = []
    for mid in ids:
        mdir = REPLAY_DIR / mid
        ep = mdir / "entries.json"
        if not ep.exists():
            continue
        doc = json.loads(ep.read_text())
        z = _telemetry(mdir)
        for ei, entry in enumerate(doc["entries"]):
            idx = {c: i for i, c in enumerate(entry["control_trace_columns"])}
            tr = entry["control_trace"]
            seat = int(entry.get("winner_seat", 0))
            spans = _contact_spans(z, seat)
            laps = [int(r[idx["lap"]]) for r in tr]
            cand: list[int] = []
            for lp in (1, 2, 3, 4):
                i0 = next((i for i, v in enumerate(laps) if v >= lp), None)
                if i0 is None:
                    continue
                t0 = float(tr[i0][idx["elapsed_s"]])
                if any(a <= t0 + window_s and b >= t0 for a, b in spans):
                    continue
                if any(int(tr[i][idx["in_pit"]]) for i in range(i0, min(i0 + 300, len(tr)))):
                    continue
                cand.append(i0)
            rng.shuffle(cand)
            for i0 in cand[:max_windows_per_entry]:
                r0 = tr[i0]
                steps = []
                for i in range(i0, len(tr) - 1):
                    r, rn = tr[i], tr[i + 1]
                    dt = float(rn[idx["elapsed_s"]]) - float(r[idx["elapsed_s"]])
                    if dt <= 0:
                        continue
                    steps.append((float(r[idx["throttle"]]), float(r[idx["brake"]]),
                                  float(r[idx["steering_angle_rad"]]), int(r[idx["ers_mode"]]),
                                  int(r[idx["aero_mode"]]), float(r[idx["override_active"]]),
                                  float(r[idx["track_wetness"]]), dt,
                                  float(rn[idx["speed_mps"]]),
                                  float(rn[idx["x_m"]]), float(rn[idx["y_m"]])))
                    if sum(s[7] for s in steps) >= window_s:
                        break
                if len(steps) < 10:
                    continue
                t0 = float(r0[idx["elapsed_s"]])
                ref_end = _ref_end(z, seat, t0 + window_s)
                specs.append({
                    "match_id": mid, "entry_i": ei, "track": entry["track"],
                    "regime": entry["weather_profile"], "build": entry["build"],
                    "x": float(r0[idx["x_m"]]), "y": float(r0[idx["y_m"]]),
                    "yaw": float(r0[idx["heading_rad"]]), "v": float(r0[idx["speed_mps"]]),
                    "gear": int(r0[idx["gear"]]), "wet": float(r0[idx["track_wetness"]]),
                    "lap": int(laps[i0]), "window_s": window_s, "extras": _extras(z, seat, t0),
                    "ref_end": ref_end, "steps": steps,
                })
    return specs


def eval_specs(specs: list[dict[str, Any]], tune: dict[str, Any], subset: int = 0,
               core_off: float = -6.3, collect_final: bool = False) -> dict[str, Any]:
    if subset and len(specs) > subset:
        specs = specs[:: max(1, len(specs) // subset)][:subset]
    sims: dict[tuple[str, int], RefRaceSim] = {}
    dv5, dv10, pos10, roads, wets = [], [], [], [], []
    finals: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    for sp in specs:
        key = (sp["match_id"], sp["entry_i"])
        sim = sims.get(key)
        if sim is None:
            sim = RefRaceSim(sp["track"], sp["build"], tune=tune)
            sims[key] = sim
        ex = sp["extras"]
        ts0 = ex.get("ts")
        sim.reset(sp["x"], sp["y"], sp["yaw"], sp["v"], sp["gear"], pre_spool_ice=1.0,
                  fuel=ex.get("fuel"), soc=ex.get("soc"), wear=ex.get("wear"),
                  ts=ts0, tc=[v + core_off for v in ts0] if ts0 is not None else None)
        t = 0.0; d5 = d10 = p10 = 0.0; road = 999.0
        for (thr, brk, steer_rad, ers, aero, ovr, wet, dt, v_ref, x_ref, y_ref) in sp["steps"]:
            sim.step(thr, brk, steer_rad / 0.3, ers, aero, bool(ovr), wet, dt=dt, delta_override=steer_rad)
            t += dt
            road = min(road, road_clearance_m(sim.track, sim.state.s, sim.state.n))
            if abs(t - 5.0) < dt:
                d5 = math.hypot(sim.state.vx, sim.state.vy) - v_ref
            if t >= sp.get("window_s", WINDOW_S) - dt * 0.5:
                d10 = math.hypot(sim.state.vx, sim.state.vy) - v_ref
                p10 = math.hypot(sim.state.x - x_ref, sim.state.y - y_ref)
                break
        dv5.append(d5); dv10.append(d10); pos10.append(p10); roads.append(road); wets.append(sp["wet"])
        if collect_final:
            finals.append({"wear": [float(v) for v in sim.state.wear],
                           "ts": [float(v) for v in sim.state.ts],
                           "ref": sp.get("ref_end") or {}})
    w = np.asarray(wets); d5 = np.asarray(dv5); d10 = np.asarray(dv10)
    bins = {}
    for lo, hi, name in ((0.0, 0.1, "dry"), (0.1, 0.45, "damp"), (0.45, 0.8, "wet"), (0.8, 1.01, "saturated")):
        m = (w >= lo) & (w < hi)
        if m.any():
            bins[name] = {"n": int(m.sum()), "mean_dv_10s": round(float(d10[m].mean()), 3),
                          "mean_abs_dv_10s": round(float(np.abs(d10[m]).mean()), 3)}
    out = {
        "n_windows": len(specs),
        "mean_abs_dv_5s": round(float(np.abs(d5).mean()), 4),
        "mean_abs_dv_10s": round(float(np.abs(d10).mean()), 4),
        "mean_dv_10s": round(float(d10.mean()), 4),
        "mean_pos_10s": round(float(np.mean(pos10)), 3),
        "p90_pos_10s": round(float(np.percentile(pos10, 90)), 3),
        "min_road_m": round(float(np.min(roads)), 3),
        "by_wetness": bins,
        "eval_s": round(time.perf_counter() - t0, 1),
    }
    if collect_final:
        out["finals"] = finals
    return out


def wear_score(finals: list[dict[str, Any]]) -> dict[str, Any]:
    we, te, n = [], [], 0
    for f in finals:
        ref = f["ref"]
        if not ref.get("wear") or not ref.get("ts"):
            continue
        we.extend(abs(a - b) for a, b in zip(f["wear"], ref["wear"]))
        te.extend(abs(a - b) for a, b in zip(f["ts"], ref["ts"]))
        n += 1
    if not n:
        return {"n": 0, "wear_mae": None, "ts_mae": None}
    return {"n": n, "wear_mae": round(float(np.mean(we)), 4), "ts_mae": round(float(np.mean(te)), 2)}


def cmd_wear(args) -> None:
    manifest = load_manifest()
    ids = manifest["train_match_ids"] if args.split == "train" else manifest["val_match_ids"]
    cache = Path(args.cache)
    if cache.exists() and not args.rebuild:
        specs = json.loads(cache.read_text())
    else:
        specs = build_specs(ids, max_windows_per_entry=args.windows_per_entry, window_s=args.window_s)
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(specs))
    print(f"{len(specs)} window specs")
    base = dict(DEFAULT_CALIBRATED_TUNE)
    if args.base_json:
        base.update(json.loads(Path(args.base_json).read_text()))
    results = []
    for core_off, hs, ws in itertools.product((0.0, -6.3, -12.0), (0.5, 0.65, 0.85), (0.7, 1.0, 1.4)):
        t = dict(base, front_heat_scale=hs, wear_scale=ws)
        r = eval_specs(specs, t, args.subset, core_off=core_off, collect_final=True)
        sc = wear_score(r.pop("finals"))
        row = {"core_off": core_off, "front_heat_scale": hs, "wear_scale": ws, **sc,
               "mean_abs_dv_10s": r["mean_abs_dv_10s"]}
        results.append(row)
        print(f"core_off={core_off:5.1f} heat={hs:.2f} wear={ws:.1f} | wear_mae={sc['wear_mae']} "
              f"ts_mae={sc['ts_mae']} |dv10|={r['mean_abs_dv_10s']}", flush=True)
    results.sort(key=lambda r: (r["wear_mae"] if r["wear_mae"] is not None else 9e9) + 0.002 * (r["ts_mae"] or 0))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "kind": "wear_thermal_grid", "split": args.split,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_tune": base, "n_windows": len(specs), "results": results,
    }, indent=1))
    print("best:", json.dumps(results[0], indent=1), "\nsaved", out)


def cmd_wet(args) -> None:
    manifest = load_manifest()
    split = "train" if args.split == "train" else args.split
    ids = manifest["train_match_ids"] if split == "train" else manifest["val_match_ids"]
    cache = Path(args.cache)
    if cache.exists() and not args.rebuild:
        specs = json.loads(cache.read_text())
        print(f"loaded {len(specs)} cached window specs from {cache}")
    else:
        specs = build_specs(ids, max_windows_per_entry=args.windows_per_entry, window_s=args.window_s)
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(specs))
        print(f"built {len(specs)} window specs -> {cache}")

    base = dict(DEFAULT_CALIBRATED_TUNE)
    if args.base_json:
        base.update(json.loads(Path(args.base_json).read_text()))

    if not args.grid:
        print(json.dumps(eval_specs(specs, base, args.subset), indent=1))
        return

    grid = [(round(wg, 2), round(we, 2))
            for wg, we in itertools.product(
                [0.20, 0.28, 0.36, 0.44, 0.52, 0.60],
                [0.6, 0.9, 1.2, 1.6, 2.2])]
    results = []
    for wg, we in grid:
        t = dict(base, wet_grip=wg, wet_exp=we)
        r = eval_specs(specs, t, args.subset)
        r.update({"wet_grip": wg, "wet_exp": we})
        results.append(r)
        print(f"wet_grip={wg:.2f} wet_exp={we:.1f} |dv10|={r['mean_abs_dv_10s']:.3f} "
              f"pos10={r['mean_pos_10s']:.1f} bins={json.dumps(r['by_wetness'])}", flush=True)
    results.sort(key=lambda r: r["mean_abs_dv_10s"] + 0.02 * r["mean_pos_10s"])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "kind": "wet_curve_grid", "split": split, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_tune": base, "n_windows": len(specs), "results": results,
    }, indent=1))
    print("best:", json.dumps(results[0], indent=1), "\nsaved", out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("wet")
    w.add_argument("--split", default="train")
    w.add_argument("--grid", action="store_true")
    w.add_argument("--subset", type=int, default=0, help="evaluate on a strided subset of windows")
    w.add_argument("--windows-per-entry", type=int, default=3)
    w.add_argument("--rebuild", action="store_true")
    w.add_argument("--cache", default=str(MAC / "reports" / "_fit_windows_train.json"))
    w.add_argument("--window-s", type=float, default=10.0)
    w.add_argument("--base-json", default=None)
    w.add_argument("--out", default=str(MAC / "reports" / "fit_wet_grid.json"))
    r = sub.add_parser("wear")
    r.add_argument("--split", default="train")
    r.add_argument("--subset", type=int, default=0)
    r.add_argument("--windows-per-entry", type=int, default=3)
    r.add_argument("--rebuild", action="store_true")
    r.add_argument("--cache", default=str(MAC / "reports" / "_fit_windows_train30.json"))
    r.add_argument("--window-s", type=float, default=30.0)
    r.add_argument("--base-json", default=None)
    r.add_argument("--out", default=str(MAC / "reports" / "fit_wear_grid.json"))
    args = ap.parse_args()
    if args.cmd == "wet":
        cmd_wet(args)
    else:
        cmd_wear(args)


if __name__ == "__main__":
    main()