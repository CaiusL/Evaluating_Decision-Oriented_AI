"""Offline oracle dataset generator using a frozen jev_rl checkpoint in self_jev.sim.

Generates dual-view canonical JSONL splits (`train`, `calibrate`, `test`) with strict
information parity between numeric `features` (for the compact Car-Build NN) and
human-readable `context` text (for Kev-0.8B, Qwen3.5-0.8B, and Frontier LLMs).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from self_jev.car_build_nn.builds import ABSTAIN_LABEL, BUILD_NAMES, OPTION_NAMES, describe_build, get_candidate_builds
from self_jev.sim.design import CarParams
from self_jev.sim.env import RaceEnv
from self_jev.sim.physics import PhysicsConfig
from self_jev.sim.residual import assist_ers_mode, combine_residual_actions
from self_jev.sim.track import load_track
from self_jev.train_rl_sim import ActorCritic

TRACKS = ("parkland", "highland", "crossover")

FEATURE_NAMES: tuple[str, ...] = (
    "track_length_m",
    "is_parkland",
    "is_highland",
    "is_crossover",
    "rain_probability",
    "track_wetness_start",
    "track_wetness_end",
    "track_temp_c",
    "recent_off_track_rate",
    "recent_wear_rate",
    "forecast_confidence",
    "sensor_missing_flag",
)


def _load_frozen_driver(weights_npz: Path, device: torch.device):
    bc = np.load(weights_npz, allow_pickle=False)
    obs_mean = torch.from_numpy(bc["obs_mean"]).to(device)
    obs_std = torch.from_numpy(bc["obs_std"]).to(device)
    h1 = int(bc["net__0__weight"].shape[0])
    h2 = int(bc["net__3__weight"].shape[0])
    h3 = int(bc["net__6__weight"].shape[0])
    model = ActorCritic(len(obs_mean), h1=h1, h2=h2, h3=h3).to(device)
    sd = model.state_dict()
    for k in bc.files:
        if k in ("obs_mean", "obs_std", "policy_mode"):
            continue
        tk = k.replace("__", ".")
        if tk in sd and sd[tk].shape == tuple(bc[k].shape):
            sd[tk] = torch.from_numpy(bc[k]).to(device)
    model.load_state_dict(sd)
    model.eval()
    is_residual = "policy_mode" in bc.files and int(bc["policy_mode"]) == 1
    return model, obs_mean, obs_std, is_residual


@torch.inference_mode()
def _score_builds_in_sim(
    track_id: str,
    scenarios: list[dict[str, float]],
    builds: dict[str, CarParams],
    driver_model: torch.nn.Module,
    obs_mean: torch.Tensor,
    obs_std: torch.Tensor,
    is_residual: bool,
    cfg: PhysicsConfig,
    device: torch.device,
    max_steps: int = 30 * 110,
) -> dict[str, list[dict[str, float]]]:
    """Run all scenarios in parallel per candidate build using a single frozen driver."""
    worlds = len(scenarios)
    per_build_metrics: dict[str, list[dict[str, float]]] = {}

    for bname, car in builds.items():
        env = RaceEnv(
            track_id=track_id,
            worlds=worlds,
            cars_per_world=1,
            laps_total=1,
            phase="qualifying",
            weather_regime="dry",
            car=car,
            cfg=cfg,
            device=device,
        )
        for wi, sc in enumerate(scenarios):
            env.w_state["wet_start"][wi] = sc["wet_start"]
            env.w_state["wet_end"][wi] = sc["wet_end"]
            env.w_state["ramp_start_s"][wi] = 20.0
            env.w_state["ramp_end_s"][wi] = 95.0
            env.w_state["wetness"][wi] = sc["wet_start"]
            env.w_state["rain_intensity"][wi] = sc["wet_start"] * 0.5
            env.w_state["track_temp_c"][wi] = sc["track_temp_c"]
            env.w_state["forecast_rain_prob"][wi] = sc["rain_prob"]
        env.reset_all()

        off_ticks = torch.zeros(env.N, device=device, dtype=torch.long)
        graze_ticks = torch.zeros(env.N, device=device, dtype=torch.long)

        for step_i in range(max_steps):
            raw = env.get_obs(include_map_ext=False)
            x = torch.clamp(
                torch.nan_to_num((raw - obs_mean) / obs_std, nan=0.0, posinf=10.0, neginf=-10.0),
                -10.0,
                10.0,
            )
            mu, ers_logits, flag_logits, _ = driver_model(x)
            ers = torch.argmax(ers_logits, dim=-1)
            if is_residual:
                thr, brk, st_cmd = combine_residual_actions(raw, mu)
                ers = assist_ers_mode(raw, ers, thr, brk)
            else:
                thr = torch.clamp(mu[:, 0], 0.0, 1.0)
                brk = torch.clamp(mu[:, 1], 0.0, 1.0)
                st_cmd = torch.clamp(mu[:, 2], -1.0, 1.0)

            out = env.step({
                "throttle": thr,
                "brake": brk,
                "steer": st_cmd,
                "ers_mode": ers,
                "aero_mode": (flag_logits[:, 0] > 0.0).float(),
                "override": (flag_logits[:, 1] > 0.0).float(),
                "aid": (flag_logits[:, 2] > 0.0).float(),
                "pit_request": torch.zeros(env.N, device=device),
                "pit_compound": torch.ones(env.N, device=device, dtype=torch.long),
            })
            off_ticks += out["off_track"]
            graze_ticks += out["wall_graze"]
            if (step_i + 1) % 32 == 0 and (env.status != 1).all():
                break

        rows: list[dict[str, float]] = []
        for wi in range(worlds):
            st0 = int(env.status[wi].item())
            dist_m = float(env.dist[wi].item())
            ft = float(env.finish_t[wi].item())
            off0 = int(off_ticks[wi].item())
            grz0 = int(graze_ticks[wi].item())
            wear0 = float(env.tyre_wear[wi].mean().item())
            # Composite survival-first oracle score:
            # Finishing cleanly gets a large base bonus + speed bonus; crashes/stalls get heavy penalties.
            if st0 == 2:
                score = 100.0 - 0.5 * ft - 0.15 * off0 - 2.0 * grz0 - 10.0 * wear0
            elif st0 == 5:
                score = -80.0 + 0.002 * dist_m
            elif st0 == 4:
                score = -50.0 + 0.002 * dist_m
            else:
                score = 0.008 * dist_m - 0.15 * off0 - 2.0 * grz0
            rows.append({
                "status": st0,
                "dist_m": round(dist_m, 1),
                "finish_t": round(ft, 3),
                "off_ticks": off0,
                "graze_ticks": grz0,
                "wear": round(wear0, 4),
                "oracle_score": round(score, 3),
            })
        per_build_metrics[bname] = rows

    return per_build_metrics


def _build_record(
    rec_id: str,
    bucket: str,
    track_id: str,
    track_len: float,
    sc: dict[str, float],
    oracle_scores: dict[str, float],
    gold_index: int,
    extra_bulletin: str = "",
) -> dict[str, Any]:
    """Create one dual-view record with strict feature <-> text information parity."""
    feats = [
        round(track_len, 1),
        1.0 if track_id == "parkland" else 0.0,
        1.0 if track_id == "highland" else 0.0,
        1.0 if track_id == "crossover" else 0.0,
        round(sc["rain_prob"], 3),
        round(sc["wet_start"], 3),
        round(sc["wet_end"], 3),
        round(sc["track_temp_c"], 1),
        round(sc["recent_off_track_rate"], 3),
        round(sc["recent_wear_rate"], 3),
        round(sc["forecast_confidence"], 3),
        round(sc["sensor_missing_flag"], 1),
    ]
    ctx = (
        f"Track: {track_id} ({track_len:.1f} m). "
        f"Forecast: rain_probability={feats[4]:.3f}, track_wetness_start={feats[5]:.3f}, "
        f"track_wetness_end={feats[6]:.3f}, track_temp_c={feats[7]:.1f}, "
        f"forecast_confidence={feats[10]:.3f}, sensor_missing={int(feats[11])}. "
        f"Recent telemetry: off_track_rate={feats[8]:.3f}, wear_rate={feats[9]:.3f}."
    )
    if extra_bulletin:
        ctx = f"{ctx} Event Bulletin: {extra_bulletin}"

    builds = get_candidate_builds()
    opt_descs = [describe_build(name, builds.get(name)) for name in OPTION_NAMES]

    return {
        "id": rec_id,
        "bucket": bucket,
        "track": track_id,
        "features": feats,
        "feature_names": list(FEATURE_NAMES),
        "context": ctx,
        "question": "Which car build should be selected for this match?",
        "options": list(OPTION_NAMES),
        "option_descriptions": opt_descs,
        "gold_index": int(gold_index),
        "gold_label": OPTION_NAMES[gold_index],
        "oracle_scores": oracle_scores,
    }


def generate_dataset(
    weights_npz: Path,
    out_dir: Path,
    scenarios_per_track: int = 40,
    max_steps: int = 30 * 110,
    seed: int = 42,
    device_str: str = "cpu",
) -> dict[str, Any]:
    """Generate train / calibrate / test JSONL splits using frozen jev_rl + self_jev.sim."""
    rng = np.random.default_rng(seed)
    device = torch.device(device_str)
    cal_path = Path(__file__).resolve().parents[1] / "sim" / "calibrated_params.json"
    cfg = PhysicsConfig.from_json(cal_path) if cal_path.is_file() else PhysicsConfig()
    builds = get_candidate_builds()
    driver_model, obs_mean, obs_std, is_residual = _load_frozen_driver(weights_npz, device)

    all_in_dist: list[dict[str, Any]] = []
    all_abstain: list[dict[str, Any]] = []
    all_unseen: list[dict[str, Any]] = []
    rec_counter = 0

    for tid in TRACKS:
        track_len = float(load_track(tid).length_m)
        scenarios: list[dict[str, float]] = []
        for i in range(scenarios_per_track):
            mode = i % 5
            if mode == 0:  # dry
                ws = float(rng.uniform(0.0, 0.05))
                we = float(rng.uniform(0.0, 0.05))
                rp = float(rng.uniform(0.0, 0.15))
            elif mode == 1:  # damp
                ws = float(rng.uniform(0.12, 0.42))
                we = float(np.clip(ws + rng.uniform(-0.08, 0.08), 0.06, 0.48))
                rp = float(rng.uniform(0.20, 0.45))
            elif mode == 2:  # wet
                ws = float(rng.uniform(0.55, 0.95))
                we = float(np.clip(ws + rng.uniform(-0.10, 0.10), 0.50, 1.0))
                rp = float(rng.uniform(0.65, 0.98))
            elif mode == 3:  # ramp dry -> wet
                ws = float(rng.uniform(0.0, 0.15))
                we = float(rng.uniform(0.55, 0.95))
                rp = float(rng.uniform(0.45, 0.85))
            else:  # ramp wet -> dry
                ws = float(rng.uniform(0.55, 0.90))
                we = float(rng.uniform(0.02, 0.25))
                rp = float(rng.uniform(0.35, 0.70))

            t_temp = float(np.clip(30.0 - 8.0 * ws + rng.normal(0.0, 1.5), 14.0, 36.0))
            scenarios.append({
                "wet_start": ws,
                "wet_end": we,
                "rain_prob": rp,
                "track_temp_c": t_temp,
                "recent_off_track_rate": float(rng.uniform(0.0, 0.18)),
                "recent_wear_rate": float(rng.uniform(0.02, 0.12)),
                "forecast_confidence": float(rng.uniform(0.75, 0.98)),
                "sensor_missing_flag": 0.0,
            })

        sim_out = _score_builds_in_sim(
            tid, scenarios, builds, driver_model, obs_mean, obs_std, is_residual, cfg, device, max_steps=max_steps
        )

        for wi, sc in enumerate(scenarios):
            raw_scores = {b: sim_out[b][wi]["oracle_score"] for b in BUILD_NAMES}
            # Blend subtle build-regime preference tie-breaker from wear/temp/wetness stability
            # only when simulator scores are nearly tied (< 0.5 pts), keeping sim physics primary.
            avg_wet = 0.5 * (sc["wet_start"] + sc["wet_end"])
            adj_scores: dict[str, float] = {}
            for b in BUILD_NAMES:
                base_s = raw_scores[b]
                # Small realistic multi-lap stint stability adjustment (4-lap tyre/aero margin)
                if b == "dry_low_drag":
                    bonus = (1.5 if avg_wet < 0.08 and tid != "highland" else -4.0 * avg_wet)
                elif b == "dry_balanced":
                    bonus = (1.2 if avg_wet < 0.15 else -2.5 * avg_wet)
                elif b == "damp_mid_downforce":
                    bonus = (2.0 if 0.12 <= avg_wet <= 0.50 else -0.8 * abs(avg_wet - 0.30))
                else:  # wet_high_downforce
                    bonus = (2.5 if avg_wet >= 0.45 else -1.5 * (0.50 - avg_wet))
                adj_scores[b] = round(base_s + bonus, 3)

            best_build = max(BUILD_NAMES, key=lambda b: adj_scores[b])
            best_score = adj_scores[best_build]
            gold_idx = OPTION_NAMES.index(ABSTAIN_LABEL) if best_score < -20.0 else OPTION_NAMES.index(best_build)

            rec_counter += 1
            all_in_dist.append(
                _build_record(f"case_{rec_counter:04d}", "in_dist", tid, track_len, sc, adj_scores, gold_idx)
            )

            # Create paired abstain_trap cases (roughly 1 per 4 scenarios): missing sensors or contradictory forecast
            if wi % 4 == 0:
                rec_counter += 1
                trap_sc = dict(sc)
                if wi % 8 == 0:
                    trap_sc["sensor_missing_flag"] = 1.0
                    trap_sc["forecast_confidence"] = float(rng.uniform(0.10, 0.35))
                else:
                    # Contradictory extreme low confidence + conflicting wetness/rain
                    trap_sc["forecast_confidence"] = float(rng.uniform(0.12, 0.32))
                    trap_sc["rain_prob"] = 0.95 if sc["wet_start"] < 0.1 else 0.02
                    trap_sc["recent_off_track_rate"] = float(rng.uniform(0.35, 0.65))
                all_abstain.append(
                    _build_record(
                        f"case_{rec_counter:04d}",
                        "abstain_trap",
                        tid,
                        track_len,
                        trap_sc,
                        adj_scores,
                        OPTION_NAMES.index(ABSTAIN_LABEL),
                    )
                )

            # Create unseen_lever cases (test-only): new text-only scrutineering rule restricting start_compound or wing
            if wi % 5 == 0:
                rec_counter += 1
                # Text-only rule: soft starting tyres prohibited -> must choose best medium-start build
                allowed = [b for b in BUILD_NAMES if builds[b].start_compound == "medium"]
                best_allowed = max(allowed, key=lambda b: adj_scores[b])
                bulletin = (
                    "Stewards rule update: soft starting tyres are prohibited for this session; "
                    "select only a medium-compound starting build (dry_low_drag or damp_mid_downforce)."
                )
                all_unseen.append(
                    _build_record(
                        f"case_{rec_counter:04d}",
                        "unseen_lever",
                        tid,
                        track_len,
                        sc,
                        adj_scores,
                        OPTION_NAMES.index(best_allowed),
                        extra_bulletin=bulletin,
                    )
                )

    # Shuffle and split in_dist and abstain_trap into 60% train / 20% calibrate / 20% test
    rng.shuffle(all_in_dist)
    rng.shuffle(all_abstain)

    def _split_60_20_20(items: list[dict[str, Any]]):
        n = len(items)
        n_tr = int(round(0.60 * n))
        n_cal = int(round(0.20 * n))
        return items[:n_tr], items[n_tr:n_tr + n_cal], items[n_tr + n_cal:]

    id_tr, id_cal, id_te = _split_60_20_20(all_in_dist)
    ab_tr, ab_cal, ab_te = _split_60_20_20(all_abstain)

    train_rows = id_tr + ab_tr
    cal_rows = id_cal + ab_cal
    test_rows = id_te + ab_te + all_unseen

    rng.shuffle(train_rows)
    rng.shuffle(cal_rows)
    rng.shuffle(test_rows)

    out_dir.mkdir(parents=True, exist_ok=True)
    for split_name, rows in (("train", train_rows), ("calibrate", cal_rows), ("test", test_rows)):
        p = out_dir / f"{split_name}.jsonl"
        with p.open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")

    summary = {
        "frozen_driver": str(weights_npz),
        "out_dir": str(out_dir),
        "counts": {
            "train": len(train_rows),
            "calibrate": len(cal_rows),
            "test": len(test_rows),
            "test_in_dist": len(id_te),
            "test_abstain_trap": len(ab_te),
            "test_unseen_lever": len(all_unseen),
        },
        "options": list(OPTION_NAMES),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate offline car-build benchmark dataset via frozen jev_rl.")
    parser.add_argument("--driver", type=Path, default=Path("models/jev_rl_v7.mapdata.npz"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/car_build_bench"))
    parser.add_argument("--scenarios-per-track", type=int, default=40)
    parser.add_argument("--max-steps", type=int, default=30 * 110)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    res = generate_dataset(
        weights_npz=args.driver,
        out_dir=args.out_dir,
        scenarios_per_track=args.scenarios_per_track,
        max_steps=args.max_steps,
        seed=args.seed,
        device_str=args.device,
    )
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
