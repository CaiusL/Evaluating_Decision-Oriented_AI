#!/usr/bin/env python3
"""Generate a self-contained HTML5 2D Replay for any Decision Model + jev_rl_v7 in self_jev/sim.

Replays the exact saved decisions (chosen_build, per-lap mode_history, pit_calls)
from reports/race_engineer_v2_summary.json (plus the confirmed track_rule policy)
and overlays the matching recorded AGX Winner Ghost when available.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from eval_race_engineer_v2 import CANONICAL_SCENARIOS, DEFAULT_DRIVER, DEFAULT_OUT  # noqa: E402
from mac_racing_sim.benchmark_winners import road_clearance_m  # noqa: E402
from mac_racing_sim.render_replay import build_track_geometry  # noqa: E402
from mac_racing_sim.sim.ref.track import Track as RefTrack  # noqa: E402
from self_jev.car_build_nn.builds import get_candidate_builds  # noqa: E402
from self_jev.car_build_nn.dataset import _load_frozen_driver  # noqa: E402
from self_jev.sim.env import RaceEnv  # noqa: E402
from self_jev.sim.physics import PhysicsConfig  # noqa: E402
from self_jev.sim.residual import assist_ers_mode, combine_residual_actions  # noqa: E402

MODEL_CATALOG: list[dict[str, str]] = [
    {"id": "track_rule", "label": "track_rule (Best Overall — 9/9, 596.1s, -8.5s confirmed)"},
    {"id": "driver_only", "label": "driver_only (jev_rl_v7 — 9/9, 603.8s)"},
    {"id": "kev-0.8b", "label": "kev-0.8b (Best ML Model — 9/9, 606.1s)"},
    {"id": "qwen-0.8b-ll", "label": "qwen-0.8b-ll (8/9, 607.0s)"},
    {"id": "nn-combo", "label": "nn-combo (8/9, 623.3s)"},
    {"id": "if_else_baseline", "label": "if_else_baseline (7/9, 620.6s)"},
    {"id": "qwen-0.8b-json", "label": "qwen-0.8b-json (6/9, 649.2s, 18 pits)"},
    {"id": "self-jev-minilm", "label": "self-jev-minilm (4/9, 634.1s, 18 pits)"},
]

STATUS_NAMES = {1: "Racing", 2: "Finished (4 Laps)", 3: "Timeout DNF", 4: "Stalled DNF", 5: "Wall Crash DNF"}

_CACHE: dict[str, Any] = {}


def _get_cached_assets() -> tuple[Any, Any, Any, Any, Any, Any]:
    if "driver" not in _CACHE:
        torch.set_num_threads(1)
        device = torch.device("cpu")
        cal_path = REPO_ROOT / "self_jev" / "sim" / "calibrated_params.json"
        _CACHE["cfg"] = PhysicsConfig.from_json(cal_path) if cal_path.is_file() else PhysicsConfig()
        _CACHE["builds"] = get_candidate_builds()
        _CACHE["driver"] = _load_frozen_driver(DEFAULT_DRIVER, device)
    driver_model, obs_mean, obs_std, is_residual = _CACHE["driver"]
    return _CACHE["cfg"], _CACHE["builds"], driver_model, obs_mean, obs_std, is_residual


def _get_ref_track(track_id: str) -> RefTrack:
    k = f"track_{track_id}"
    if k not in _CACHE:
        _CACHE[k] = RefTrack(track_id, root=str(REPO_ROOT / "mac_racing_sim"))
        _CACHE[f"geom_{track_id}"] = build_track_geometry(_CACHE[k], stride=2)
    return _CACHE[k]


def _load_saved_decisions(model_id: str, sc: dict[str, Any]) -> tuple[str, list[str], list[str]]:
    if model_id == "track_rule":
        mode = "balanced" if sc["track"] == "parkland" else "push_max"
        return "dry_balanced", [mode, mode, mode, mode], ["stay_out", "stay_out"]

    if "re_summary" not in _CACHE and DEFAULT_OUT.is_file():
        _CACHE["re_summary"] = json.loads(DEFAULT_OUT.read_text(encoding="utf-8"))
    doc = _CACHE.get("re_summary") or {}
    m_data = (doc.get("models") or {}).get(model_id)
    if m_data and isinstance(m_data.get("runs"), list):
        for r in m_data["runs"]:
            if r.get("scenario_id") == sc["id"]:
                return (
                    str(r.get("chosen_build") or "dry_balanced"),
                    list(r.get("mode_history") or ["balanced"] * 4),
                    list(r.get("pit_calls") or ["stay_out", "stay_out"]),
                )
    return "dry_balanced", ["balanced"] * 4, ["stay_out", "stay_out"]


def _load_agx_ghost(track: str, weather_regime: str) -> tuple[list[float], list[float], list[float], list[float], list[float], float | None]:
    key = f"{track.lower()}:{weather_regime.lower()}:4"
    gk = f"ghost_{key}"
    if gk in _CACHE:
        return _CACHE[gk]
    if "winner_doc" not in _CACHE:
        wpath = REPO_ROOT / "mac_racing_sim" / "data" / "winner-builds-and-controls.json"
        _CACHE["winner_doc"] = json.loads(wpath.read_text(encoding="utf-8")) if wpath.is_file() else {}
    doc = _CACHE["winner_doc"]
    entry = (doc.get("winners") or {}).get(key)
    if not entry:
        _CACHE[gk] = ([], [], [], [], [], None)
        return _CACHE[gk]
    idx = {c: i for i, c in enumerate(doc["control_trace_columns"])}
    tr = entry["control_trace"]
    gt, gx, gy, gh, gv = [], [], [], [], []
    for r in tr:
        gt.append(float(r[idx["elapsed_s"]]))
        gx.append(float(r[idx["x_m"]]))
        gy.append(float(r[idx["y_m"]]))
        gh.append(float(r[idx["heading_rad"]]))
        gv.append(float(r[idx["speed_mps"]]))
    _CACHE[gk] = (gt, gx, gy, gh, gv, float(entry["official_finish_s"]))
    return _CACHE[gk]


@torch.inference_mode()
def record_model_replay(
    model_id: str,
    scenario_id: str,
    target_hz: float = 10.0,
) -> tuple[dict[str, Any], list[list[Any]], dict[str, Any]]:
    sc = next((s for s in CANONICAL_SCENARIOS if s["id"] == scenario_id), None)
    if sc is None:
        raise ValueError(f"Unknown scenario_id: {scenario_id}")

    chosen_build, mode_history, pit_calls = _load_saved_decisions(model_id, sc)
    cfg, builds, driver_model, obs_mean, obs_std, is_residual = _get_cached_assets()
    if chosen_build not in builds:
        chosen_build = "dry_balanced"

    device = torch.device("cpu")
    torch.manual_seed(2026)
    env = RaceEnv(
        track_id=sc["track"],
        worlds=1,
        cars_per_world=1,
        laps_total=4,
        phase="race",
        weather_regime="dry",
        car=builds[chosen_build],
        cfg=cfg,
        device=device,
    )
    env.w_state["wet_start"][0] = sc["wet_start"]
    env.w_state["wet_end"][0] = sc["wet_end"]
    env.w_state["ramp_start_s"][0] = 40.0
    env.w_state["ramp_end_s"][0] = 360.0
    env.w_state["wetness"][0] = sc["wet_start"]
    env.w_state["rain_intensity"][0] = sc["wet_start"] * 0.5
    env.w_state["track_temp_c"][0] = sc["track_temp_c"]
    env.w_state["forecast_rain_prob"][0] = sc["rain_prob"]
    env.reset_all()

    ref_track = _get_ref_track(sc["track"])
    geom = _CACHE[f"geom_{sc['track']}"]
    gt, gx, gy, gh, gv, winner_s = _load_agx_ghost(sc["track"], sc["weather_regime"])
    has_ghost = len(gt) > 0

    current_mode = mode_history[0] if mode_history else "balanced"
    pit_latched = False
    pit_target_idx = 0
    prev_pit_stops = 0
    last_decided_lap = -1
    rec_stride = max(1, int(round(30.0 / target_hz)))
    frames: list[list[Any]] = []
    min_rd = 999.0
    g_i = 0

    for step_i in range(30 * 800):
        cur_lap = int(env.lap[0].item())
        cur_pit_stops = int(env.pit_stops[0].item())
        if cur_pit_stops > prev_pit_stops:
            pit_latched = False
            prev_pit_stops = cur_pit_stops

        if cur_lap > last_decided_lap and cur_lap < 4 and int(env.status[0].item()) == 1:
            last_decided_lap = cur_lap
            current_mode = mode_history[cur_lap] if cur_lap < len(mode_history) else mode_history[-1]
            if cur_lap in (1, 2) and not pit_latched:
                pidx = cur_lap - 1
                pcall = pit_calls[pidx] if pidx < len(pit_calls) else "stay_out"
                if pcall == "pit_soft":
                    pit_latched = True
                    pit_target_idx = 0
                elif pcall == "pit_medium":
                    pit_latched = True
                    pit_target_idx = 1

        raw = env.get_obs(include_map_ext=False)
        x = torch.clamp(torch.nan_to_num((raw - obs_mean) / obs_std, nan=0.0, posinf=10.0, neginf=-10.0), -10.0, 10.0)
        mu, ers_logits, flag_logits, _ = driver_model(x)
        ers = torch.argmax(ers_logits, dim=-1)
        mu_adj = mu.clone()
        ovr = (flag_logits[:, 1] > 0.0).float()

        if current_mode == "push_max":
            mu_adj[:, 0] = mu_adj[:, 0] + 0.15
            if float(env.ers_soc_mj[0].item()) > 0.5 and float(env.wetness[0].item()) < 0.45:
                ers = torch.full_like(ers, 3)
                ovr = torch.ones_like(ovr)
        elif current_mode == "conserve_tyres":
            mu_adj[:, 0] = mu_adj[:, 0] - 0.08
            ers = torch.full_like(ers, 1)
            ovr = torch.zeros_like(ovr)
        elif current_mode == "wet_caution":
            mu_adj[:, 0] = mu_adj[:, 0] - 0.15
            ers = torch.zeros_like(ers)
            ovr = torch.zeros_like(ovr)

        if is_residual:
            thr, brk, st_cmd = combine_residual_actions(raw, mu_adj)
            if current_mode == "balanced":
                ers = assist_ers_mode(raw, ers, thr, brk)
        else:
            thr = torch.clamp(mu_adj[:, 0], 0.0, 1.0)
            brk = torch.clamp(mu_adj[:, 1], 0.0, 1.0)
            st_cmd = torch.clamp(mu_adj[:, 2], -1.0, 1.0)

        env.step({
            "throttle": thr,
            "brake": brk,
            "steer": st_cmd,
            "ers_mode": ers,
            "aero_mode": (flag_logits[:, 0] > 0.0).float(),
            "override": ovr,
            "aid": (flag_logits[:, 2] > 0.0).float(),
            "pit_request": torch.tensor([1.0 if pit_latched else 0.0], device=device),
            "pit_compound": torch.tensor([pit_target_idx], device=device, dtype=torch.long),
        })

        t_now = float(env.t_race[0].item())
        s_now = float(env.s[0].item())
        n_now = float(env.n_lat[0].item())
        rd = road_clearance_m(ref_track, s_now, n_now)
        min_rd = min(min_rd, rd)

        if step_i % rec_stride == 0 or int(env.status[0].item()) != 1:
            if has_ghost:
                while g_i + 1 < len(gt) and gt[g_i + 1] <= t_now:
                    g_i += 1
                rx, ry, rh, rv = gx[g_i], gy[g_i], gh[g_i], gv[g_i]
            else:
                rx, ry, rh, rv = float(env.x[0].item()), float(env.y[0].item()), float(env.yaw[0].item()), 0.0

            v_sim = float(math.hypot(float(env.vx[0].item()), float(env.vy[0].item())))
            in_pit = int(env.pit_state[0].item()) != 0
            frames.append([
                round(t_now, 2),
                round(float(env.x[0].item()), 2), round(float(env.y[0].item()), 2), round(float(env.yaw[0].item()), 3), round(v_sim, 2),
                round(rx, 2), round(ry, 2), round(rh, 3), round(rv, 2),
                round(float(thr[0].item()), 2), round(float(brk[0].item()), 2), round(float(env.steer[0].item()), 3),
                int(round(float(env.gear[0].item()))), int(ers[0].item()),
                round(float(env.wetness[0].item()) * 100.0, 1), round(float(env.tyre_wear[0].item()) * 100.0, 1),
                round(rd, 2), min(4, int(env.lap[0].item()) + 1),
                f"{current_mode}{' (PIT)' if in_pit else ''}",
            ])

        if int(env.status[0].item()) != 1:
            break

    st0 = int(env.status[0].item())
    finished = st0 == 2
    sim_t = round(float(env.finish_t[0].item()), 2) if finished else round(float(env.t_race[0].item()), 2)
    meta = {
        "mode": f"Model: {model_id} + jev_rl_v7 · Build: {chosen_build} · {STATUS_NAMES.get(st0, str(st0))}",
        "model_id": model_id,
        "chosen_build": chosen_build,
        "status_label": STATUS_NAMES.get(st0, str(st0)),
        "pit_stops": int(env.pit_stops[0].item()),
        "has_ghost": has_ghost,
        "winner_s": round(winner_s, 2) if winner_s is not None else None,
        "sim_s": sim_t,
        "diff_s": round(sim_t - winner_s, 2) if (finished and winner_s is not None) else None,
        "min_road_clr_m": round(min_rd, 2),
    }
    return geom, frames, meta


def render_model_replay_html(model_id: str, scenario_id: str, out_path: Path | None = None) -> Path:
    from mac_racing_sim.render_replay import HTML_TEMPLATE

    geom, frames, meta = record_model_replay(model_id, scenario_id)
    out = out_path or (REPO_ROOT / "reports" / "sim_replays" / f"replay_{model_id}_{scenario_id}.html")
    out.parent.mkdir(parents=True, exist_ok=True)
    tpl = HTML_TEMPLATE.replace(
        "let showGhost = true;",
        "let showGhost = (M.has_ghost !== false);",
    ).replace(
        "document.getElementById(\"m-lap\").textContent = `L${lap}/4 · ${gapM.toFixed(1)}m`;",
        "const modeTag = fr[18] ? ` · ${fr[18]}` : ''; document.getElementById(\"m-lap\").textContent = `L${lap}/4${modeTag}`;",
    ).replace(
        "Winner time: <b>${M.winner_s} s</b><br>",
        "Model: <b>${M.model_id || 'sim'}</b> (${M.status_label || 'OK'})<br>Build: <b>${M.chosen_build || 'winner'}</b> · Pits: <b>${M.pit_stops ?? 0}</b><br>AGX Winner: <b>${M.winner_s ?? 'N/A'} s</b><br>",
    )
    payload = json.dumps({"geom": geom, "frames": frames, "meta": meta}, separators=(",", ":"))
    html = tpl.replace("__TITLE__", f"{model_id} on {scenario_id}").replace("__PAYLOAD__", payload)
    out.write_text(html, encoding="utf-8")
    return out


def _precache_job(job: tuple[str, str]) -> str:
    m, sc = job
    out = REPO_ROOT / "reports" / "sim_replays" / f"replay_{m}_{sc}.html"
    if out.is_file():
        return f"cached {m}/{sc}"
    t0 = time.perf_counter()
    render_model_replay_html(m, sc, out)
    return f"built {m}/{sc} in {time.perf_counter() - t0:.1f}s"


def main() -> None:
    ap = argparse.ArgumentParser(description="Render HTML5 2D replay for a Decision Model + jev_rl_v7.")
    ap.add_argument("--model", default="track_rule")
    ap.add_argument("--scenario", default="crossover_saturated")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--precache", action="store_true", help="Pre-render multiple models x scenarios in parallel")
    ap.add_argument("--scenarios", nargs="+", default=["crossover_saturated", "parkland_dry", "highland_saturated"])
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    if args.precache:
        jobs = [(m["id"], sc) for sc in args.scenarios for m in MODEL_CATALOG]
        t0 = time.perf_counter()
        pending = list(jobs)
        running: list[tuple[str, str, subprocess.Popen[str]]] = []
        while pending or running:
            while pending and len(running) < max(1, args.workers):
                m_id, sc_id = pending.pop(0)
                out_f = REPO_ROOT / "reports" / "sim_replays" / f"replay_{m_id}_{sc_id}.html"
                if out_f.is_file():
                    print(f"  cached {m_id}/{sc_id}", flush=True)
                    continue
                p = subprocess.Popen(
                    [sys.executable, str(Path(__file__).resolve()), "--model", m_id, "--scenario", sc_id],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                )
                running.append((m_id, sc_id, p))
            still = []
            for m_id, sc_id, p in running:
                ret = p.poll()
                if ret is None:
                    still.append((m_id, sc_id, p))
                else:
                    out_s, err_s = p.communicate()
                    if ret == 0:
                        print(f"  built {m_id}/{sc_id}: {out_s.strip()}", flush=True)
                    else:
                        print(f"  FAILED {m_id}/{sc_id}: {err_s.strip()[:200]}", flush=True)
            running = still
            if running:
                time.sleep(0.15)
        print(f"Pre-cached {len(jobs)} replays in {time.perf_counter() - t0:.1f}s")
        return

    out = render_model_replay_html(args.model, args.scenario, args.out)
    print(f"Saved model replay ({out.stat().st_size / 1024:.1f} KB) -> {out}")


if __name__ == "__main__":
    main()
