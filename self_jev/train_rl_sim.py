"""v3 Ultimate One-Shot 4-Stage Resumable Residual GPU PPO trainer (384->384->192, 500 updates, 762M steps)."""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from self_jev.sim.design import load_design, make_rain_design
from self_jev.sim.env import RaceEnv
from self_jev.sim.eval_matrix import evaluate_model_fast_30
from self_jev.sim.physics import PhysicsConfig
from self_jev.sim.residual import assist_ers_mode, combine_residual_actions


class ActorCritic(nn.Module):
    def __init__(self, in_dim: int = 181, h1: int = 384, h2: int = 384, h3: int = 192) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, h1),
            nn.LayerNorm(h1),
            nn.SiLU(),
            nn.Linear(h1, h2),
            nn.LayerNorm(h2),
            nn.SiLU(),
            nn.Linear(h2, h3),
            nn.SiLU(),
        )
        self.cont_head = nn.Linear(h3, 3)   # residuals: d_speed_mult, d_steer, d_brake
        self.ers_head = nn.Linear(h3, 4)    # ers_mode 0..3
        self.flag_head = nn.Linear(h3, 3)   # aero_mode, override, aid
        self.log_std = nn.Parameter(torch.tensor([-1.5, -1.9, -1.9]))
        self.value_head = nn.Linear(h3, 1)

    def forward(self, x: torch.Tensor):
        h = self.net(x)
        return self.cont_head(h), self.ers_head(h), self.flag_head(h), self.value_head(h).squeeze(-1)


def _stage_schedule(update: int, total_updates: int) -> dict[str, float | str]:
    """4-Stage automated curriculum: A (Pace) -> B (Clean/Wet) -> C (Racecraft) -> D (Speed Opt, 0 Damage/Off-Track)."""
    frac = (update - 1) / max(1, total_updates - 1)
    if frac < 0.20:
        return {
            "name": "A-Pace",
            "lr": 2.2e-4 - 0.7e-4 * (frac / 0.20),
            "ent_coef": 0.006,
            "res_reg": 0.0010,
            "off_qual": -0.25,
            "off_race": -0.15,
            "dirty_pen": -3.0,
            "graze_pen": -4.0,
            "contact_pen": -6.0,
            "wall_pen": -50.0,
            "time_cost": 0.0,
            "lap_speed_bonus": 0.0,
            "max_log_std": -0.3,
        }
    if frac < 0.40:
        p = (frac - 0.20) / 0.20
        return {
            "name": "B-CleanWet",
            "lr": 1.5e-4 - 0.7e-4 * p,
            "ent_coef": 0.004,
            "res_reg": 0.0004,
            "off_qual": -1.0,
            "off_race": -0.5,
            "dirty_pen": -6.0,
            "graze_pen": -8.0,
            "contact_pen": -10.0,
            "wall_pen": -75.0,
            "time_cost": -0.02,
            "lap_speed_bonus": 0.5,
            "max_log_std": -0.6,
        }
    if frac < 0.60:
        p = (frac - 0.40) / 0.20
        return {
            "name": "C-Racecraft",
            "lr": 8.0e-5 - 4.0e-5 * p,
            "ent_coef": 0.0025,
            "res_reg": 0.0002,
            "off_qual": -2.0,
            "off_race": -1.0,
            "dirty_pen": -10.0,
            "graze_pen": -12.0,
            "contact_pen": -18.0,
            "wall_pen": -95.0,
            "time_cost": -0.04,
            "lap_speed_bonus": 1.0,
            "max_log_std": -0.9,
        }
    # Stage D (60%..100%): Speed Optimization under strict Zero-Damage / Zero-Off-Track constraints
    p = (frac - 0.60) / 0.40
    lr = 1.0e-5 + 0.5 * (4.0e-5 - 1.0e-5) * (1.0 + math.cos(math.pi * p))
    return {
        "name": "D-SpeedOpt",
        "lr": lr,
        "ent_coef": 0.0012 * (1.0 - 0.5 * p),
        "res_reg": 0.0001,
        "off_qual": -6.0,
        "off_race": -4.0,
        "dirty_pen": -15.0,
        "graze_pen": -15.0,
        "contact_pen": -25.0,
        "wall_pen": -120.0,
        "time_cost": -0.06,
        "lap_speed_bonus": 2.0,
        "max_log_std": -1.2,
    }


def _compute_stage_reward(
    env: RaceEnv,
    prev_dist: torch.Tensor,
    prev_status: torch.Tensor,
    prev_dirty: torch.Tensor,
    prev_lap: torch.Tensor,
    step_out: dict[str, torch.Tensor],
    stg: dict[str, float | str],
) -> torch.Tensor:
    """Stage-aware reward combining pace/time minimization with zero-damage/zero-off-track barriers."""
    cur_dist = step_out["dist"]
    status = step_out["status"]
    off_track = step_out["off_track"]
    wall_graze = step_out["wall_graze"]
    car_contact = step_out["car_contact"]

    was_active = prev_status == 1
    progress = torch.where(was_active, (cur_dist - prev_dist) / 4.5, 0.0)
    time_pen_tick = torch.where(was_active & (env.pit_state == 0), float(stg["time_cost"]), 0.0)
    speed = torch.hypot(env.vx, env.vy)
    idle_pen = torch.where(was_active & (env.pit_state == 0) & (speed < 5.0), -0.6, 0.0)

    is_qual = env.phase == "qualifying"
    off_rate = float(stg["off_qual"]) if is_qual else float(stg["off_race"])
    off_pen = torch.where(was_active & (off_track == 1), off_rate, 0.0)
    dirty_trans = (prev_dirty == 0) & (env.lap_dirty == 1)
    off_pen = off_pen + torch.where(was_active & dirty_trans, float(stg["dirty_pen"]), 0.0)

    graze_pen = torch.where(was_active & (wall_graze == 1), float(stg["graze_pen"]), 0.0)
    contact_pen = torch.where(was_active & (car_contact == 1), float(stg["contact_pen"]), 0.0) if not is_qual else 0.0

    # Clean lap-completion speed bonus (rewards shaving seconds off lap time!)
    lap_crossed = was_active & (env.lap > prev_lap)
    ref_lap_s = 140.0 if env.track.track_id == "parkland" else (185.0 if env.track.track_id == "highland" else 165.0)
    saved_s = torch.clamp(ref_lap_s - env.best_lap_t, min=0.0, max=60.0)
    lap_bonus = torch.where(
        lap_crossed & (prev_dirty == 0) & (env.best_lap_t > 1.0),
        float(stg["lap_speed_bonus"]) * saved_s * 0.25,
        0.0,
    )

    newly_done = was_active & (status != 1)
    fin_bonus = torch.where(newly_done & (status == 2), 25.0 if is_qual else 60.0, 0.0)
    timeout_pen = torch.where(newly_done & (status == 3), -25.0, 0.0)
    stall_pen = torch.where(newly_done & (status == 4), -40.0, 0.0)
    wall_pen = torch.where(newly_done & (status == 5), float(stg["wall_pen"]), 0.0)

    return progress + time_pen_tick + idle_pen + off_pen + graze_pen + contact_pen + lap_bonus + fin_bonus + timeout_pen + stall_pen + wall_pen


def _atomic_save_torch(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _atomic_save_npz(save_dict: dict[str, np.ndarray], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez(tmp, **save_dict)
    os.replace(tmp, path)


def train_rl_sim(
    init_npz: Path,
    out_npz: Path,
    worlds_per_track: int = 128,
    updates: int = 500,
    rollout_steps: int = 512,
    mb_size: int = 32768,
    h1: int = 384,
    h2: int = 384,
    h3: int = 192,
    eval_every: int = 25,
    ckpt_every: int = 10,
    resume_dir: Path | None = None,
    device_str: str | None = None,
) -> Path:
    device = torch.device(device_str or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        print(f"Using GPU: {torch.cuda.get_device_name(0)} (worlds_per_track={worlds_per_track}, updates={updates}, net={h1}-{h2}-{h3})")

    car = load_design()
    rain_car = make_rain_design(car)
    cal_path = Path(__file__).resolve().parent / "sim" / "calibrated_params.json"
    cfg = PhysicsConfig.from_json(cal_path) if cal_path.is_file() else PhysicsConfig()

    track_names = ["parkland", "highland", "crossover"]
    qual_worlds = max(1, worlds_per_track // 4)
    race_worlds = max(1, worlds_per_track - qual_worlds)

    envs: list[RaceEnv] = []
    for t in track_names:
        envs.append(RaceEnv(t, worlds=qual_worlds, cars_per_world=1, laps_total=1, phase="qualifying",
                            weather_regime="mixed", car=car, rain_car=rain_car, cfg=cfg, device=device, domain_rand=True))
        envs.append(RaceEnv(t, worlds=race_worlds, cars_per_world=10, laps_total=4, phase="race",
                            weather_regime="mixed", car=car, rain_car=rain_car, cfg=cfg, device=device, domain_rand=True))

    bc = np.load(init_npz, allow_pickle=False)
    obs_mean_np = bc["obs_mean"].astype(np.float32)
    obs_std_np = bc["obs_std"].astype(np.float32)
    obs_mean = torch.from_numpy(obs_mean_np).to(device)
    obs_std = torch.from_numpy(obs_std_np).to(device)

    model = ActorCritic(len(obs_mean_np), h1=h1, h2=h2, h3=h3).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=2.2e-4)

    def build_npz_dict() -> dict[str, np.ndarray]:
        save_dict = {
            "obs_mean": obs_mean_np,
            "obs_std": obs_std_np,
            "policy_mode": np.array(1, dtype=np.int32),  # 1 = residual controller
        }
        for k, v in model.state_dict().items():
            if k.startswith(("net.", "cont_head.", "ers_head.", "flag_head.")):
                save_dict[k.replace(".", "__")] = v.detach().cpu().numpy().astype(np.float32)
        return save_dict

    def export_weights(path: Path) -> None:
        _atomic_save_npz(build_npz_dict(), path)

    start_update = 1
    # Gate score tuple (higher is better): (zero_wall_stall, qual_clean_rate, -damage_off_ticks, -mean_qual_lap_t, mean_race_dist_m)
    best_gate_score: tuple[float, ...] = (-1e9, -1e9, -1e9, -1e9, -1e9)
    best_eval_summary: dict | None = None
    saved_best = False

    ckpt_path = (resume_dir / "trainer_ckpt.pt") if resume_dir is not None else None
    drive_best_npz = (resume_dir / "jev_rl_best.mapdata.npz") if resume_dir is not None else None
    drive_eval_json = (resume_dir / "eval_gate_best.json") if resume_dir is not None else None

    if ckpt_path is not None and ckpt_path.is_file():
        print(f"Resuming training from checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        start_update = int(ckpt["update"]) + 1
        best_gate_score = tuple(ckpt.get("best_gate_score", best_gate_score))
        best_eval_summary = ckpt.get("best_eval_summary")
        saved_best = bool(ckpt.get("saved_best", False))
        print(f"Resumed at update {start_update}/{updates} (saved_best={saved_best}, best_gate={best_gate_score})")
    else:
        # Warm-start matching weights if shapes match, else clean residual init (cont_head near 0)
        sd = model.state_dict()
        loaded_any = False
        for k in bc.files:
            if k in ("obs_mean", "obs_std", "policy_mode"):
                continue
            tk = k.replace("__", ".")
            if tk in sd and tuple(sd[tk].shape) == tuple(bc[k].shape):
                sd[tk] = torch.from_numpy(bc[k]).to(device)
                loaded_any = True
        model.load_state_dict(sd)
        if not loaded_any or "policy_mode" not in bc.files:
            with torch.no_grad():
                model.cont_head.weight.mul_(0.01)
                model.cont_head.bias.zero_()
        # Immediately save update-0 baseline so out_npz is never empty even if interrupted early
        export_weights(out_npz)
        if drive_best_npz is not None:
            export_weights(drive_best_npz)

    prev_dist = [e.dist.clone() for e in envs]
    prev_status = [e.status.clone() for e in envs]
    prev_dirty = [e.lap_dirty.clone() for e in envs]
    prev_lap = [e.lap.clone() for e in envs]
    t0 = time.time()

    for update in range(start_update, updates + 1):
        stg = _stage_schedule(update, updates)
        cur_lr = float(stg["lr"])
        ent_coef = float(stg["ent_coef"])
        res_reg_coef = float(stg["res_reg"])
        max_log_std = float(stg["max_log_std"])
        for pg in opt.param_groups:
            pg["lr"] = cur_lr

        all_obs, all_c, all_e, all_f, all_logp, all_adv, all_ret = [], [], [], [], [], [], []
        step_rews = []

        for idx_e, env in enumerate(envs):
            n_cars = env.N
            obs_buf, c_buf, e_buf, f_buf, logp_buf, rew_buf, val_buf, done_buf = [], [], [], [], [], [], [], []

            for step_r in range(rollout_steps):
                raw_obs = env.get_obs(include_map_ext=False)
                x = torch.clamp(torch.nan_to_num((raw_obs - obs_mean) / obs_std, nan=0.0, posinf=10.0, neginf=-10.0), -10.0, 10.0)
                with torch.no_grad():
                    mu, ers_logits, flag_logits, val = model(x)
                    mu = torch.clamp(torch.nan_to_num(mu, nan=0.0, posinf=2.0, neginf=-2.0), -2.0, 2.0)
                    ers_logits = torch.nan_to_num(ers_logits, nan=0.0, posinf=10.0, neginf=-10.0)
                    flag_logits = torch.nan_to_num(flag_logits, nan=0.0, posinf=10.0, neginf=-10.0)
                    scale = torch.clamp(model.log_std, -4.0, max_log_std).exp()

                    c_dist = torch.distributions.Normal(mu, scale)
                    e_dist = torch.distributions.Categorical(logits=ers_logits)
                    f_dist = torch.distributions.Bernoulli(logits=flag_logits)

                    c_act = c_dist.sample()
                    e_act = e_dist.sample()
                    f_act = f_dist.sample()
                    logp = c_dist.log_prob(c_act).sum(-1) + e_dist.log_prob(e_act) + f_dist.log_prob(f_act).sum(-1)

                thr, brk, str_cmd = combine_residual_actions(raw_obs, c_act)
                ers_cmd = assist_ers_mode(raw_obs, e_act, thr, brk)
                pit_req = (
                    (env.phase == "race")
                    & (env.lap >= 1)
                    & (~env.wet_waiver_latched)
                    & (env.pit_stops == 0)
                    & (env.s >= env.track.pit_entry_s - 220.0)
                )
                actions = {
                    "throttle": thr,
                    "brake": brk,
                    "steer": str_cmd,
                    "ers_mode": ers_cmd,
                    "aero_mode": f_act[:, 0],
                    "override": f_act[:, 1],
                    "aid": f_act[:, 2],
                    "pit_request": pit_req.float(),
                    "pit_compound": torch.ones(n_cars, device=device, dtype=torch.long),
                }
                step_out = env.step(actions)
                rew = _compute_stage_reward(
                    env, prev_dist[idx_e], prev_status[idx_e], prev_dirty[idx_e], prev_lap[idx_e], step_out, stg
                )
                done = (step_out["status"] != 1).float()

                obs_buf.append(x)
                c_buf.append(c_act)
                e_buf.append(e_act)
                f_buf.append(f_act)
                logp_buf.append(logp)
                rew_buf.append(rew)
                val_buf.append(val)
                done_buf.append(done)

                if (step_r + 1) % 32 == 0:
                    world_done = (step_out["status"].view(env.W, env.C) != 1).all(dim=1)
                    if world_done.any():
                        env.reset_worlds(world_done, rolling_frac=0.5)
                prev_dist[idx_e] = env.dist.clone()
                prev_status[idx_e] = env.status.clone()
                prev_dirty[idx_e] = env.lap_dirty.clone()
                prev_lap[idx_e] = env.lap.clone()

            with torch.no_grad():
                x_next = torch.clamp(torch.nan_to_num((env.get_obs(include_map_ext=False) - obs_mean) / obs_std, nan=0.0, posinf=10.0, neginf=-10.0), -10.0, 10.0)
                _, _, _, next_val = model(x_next)

            T_steps = len(rew_buf)
            adv_buf = [None] * T_steps
            gae = torch.zeros(n_cars, device=device)
            for t in reversed(range(T_steps)):
                non_term = 1.0 - done_buf[t]
                nv = next_val if t == T_steps - 1 else val_buf[t + 1]
                delta = rew_buf[t] + 0.997 * nv * non_term - val_buf[t]
                gae = delta + 0.997 * 0.95 * non_term * gae
                adv_buf[t] = gae

            t_adv = torch.cat(adv_buf)
            t_val = torch.cat(val_buf)
            all_obs.append(torch.cat(obs_buf))
            all_c.append(torch.cat(c_buf))
            all_e.append(torch.cat(e_buf))
            all_f.append(torch.cat(f_buf))
            all_logp.append(torch.cat(logp_buf))
            all_adv.append(t_adv)
            all_ret.append(t_adv + t_val)
            step_rews.append(torch.stack(rew_buf).mean().item())

        b_obs = torch.cat(all_obs)
        b_c = torch.cat(all_c)
        b_e = torch.cat(all_e)
        b_f = torch.cat(all_f)
        b_old_logp = torch.cat(all_logp)
        b_adv = torch.cat(all_adv)
        b_ret = torch.cat(all_ret)
        b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

        total_n = b_obs.shape[0]
        eff_mb = min(mb_size, total_n)
        for _ in range(4):
            perm = torch.randperm(total_n, device=device)
            for start in range(0, total_n, eff_mb):
                idx = perm[start:start + eff_mb]
                mu_raw, ers_logits, flag_logits, val = model(b_obs[idx])
                mu_raw = torch.nan_to_num(mu_raw, nan=0.0, posinf=2.0, neginf=-2.0)
                mu = torch.clamp(mu_raw, -2.0, 2.0)
                ers_logits = torch.nan_to_num(ers_logits, nan=0.0, posinf=10.0, neginf=-10.0)
                flag_logits = torch.nan_to_num(flag_logits, nan=0.0, posinf=10.0, neginf=-10.0)
                scale = torch.clamp(model.log_std, -4.0, max_log_std).exp()

                c_dist = torch.distributions.Normal(mu, scale)
                e_dist = torch.distributions.Categorical(logits=ers_logits)
                f_dist = torch.distributions.Bernoulli(logits=flag_logits)

                logp = c_dist.log_prob(b_c[idx]).sum(-1) + e_dist.log_prob(b_e[idx]) + f_dist.log_prob(b_f[idx]).sum(-1)
                entropy = c_dist.entropy().sum(-1) + e_dist.entropy() + f_dist.entropy().sum(-1)

                ratio = torch.exp(logp - b_old_logp[idx])
                surr1 = ratio * b_adv[idx]
                surr2 = torch.clamp(ratio, 0.8, 1.2) * b_adv[idx]
                res_reg = res_reg_coef * mu_raw.pow(2).mean()
                loss = -torch.min(surr1, surr2).mean() + 0.5 * (b_ret[idx] - val).pow(2).mean() - ent_coef * entropy.mean() + res_reg

                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                opt.step()

        mean_rew = float(np.mean(step_rews))
        if update % 5 == 0 or update == start_update or update == updates:
            fin = sum(int((e.status == 2).sum().item()) for e in envs)
            wal = sum(int((e.status == 5).sum().item()) for e in envs)
            stl = sum(int((e.status == 4).sum().item()) for e in envs)
            mins = (time.time() - t0) / 60.0
            print(f"update {update}/{updates} [{stg['name']}] ({mins:.1f}m) mean_rew={mean_rew:.4f} fin={fin} stall={stl} wall={wal}", flush=True)

        # Periodic 30-cell Zero-Damage / Zero-Off-Track Promotion Gate
        if eval_every > 0 and (update % eval_every == 0 or update == updates):
            ev = evaluate_model_fast_30(model, obs_mean, obs_std, cfg, device)
            zero_crash = 1.0 if (ev["wall_retirements"] == 0 and ev["stall_retirements"] == 0) else 0.0
            dmg_off = ev["off_track_ticks"] + ev["contact_ticks"] + ev["wall_graze_ticks"]
            gate_score = (
                zero_crash,
                float(ev["qual_clean_rate"]),
                float(ev["race_classified_rate"]),
                -float(dmg_off),
                -float(ev["mean_qual_lap_t"]),
                float(ev["mean_race_dist_m"]),
            )
            promoted = gate_score > best_gate_score
            if promoted:
                best_gate_score = gate_score
                best_eval_summary = {**ev, "update": update, "stage": stg["name"]}
                export_weights(out_npz)
                if drive_best_npz is not None:
                    export_weights(drive_best_npz)
                if drive_eval_json is not None:
                    drive_eval_json.write_text(json.dumps(best_eval_summary, indent=2) + "\n")
                saved_best = True
            print(
                f"  [GATE u={update}] clean={ev['qual_clean_rate']*100:.0f}% cls={ev['race_classified_rate']*100:.0f}% "
                f"wall={ev['wall_retirements']} off={ev['off_track_ticks']} cnt={ev['contact_ticks']} "
                f"qual_t={ev['mean_qual_lap_t']:.2f}s race_d={ev['mean_race_dist_m']:.0f}m -> "
                f"{'PROMOTED' if promoted else 'kept best'}",
                flush=True,
            )

        # Frequent atomic resumable checkpoint save (protects against Colab disconnects)
        if ckpt_path is not None and (update % ckpt_every == 0 or update == updates):
            _atomic_save_torch(
                {
                    "update": update,
                    "model": model.state_dict(),
                    "opt": opt.state_dict(),
                    "best_gate_score": best_gate_score,
                    "best_eval_summary": best_eval_summary,
                    "saved_best": saved_best,
                },
                ckpt_path,
            )

    if not saved_best:
        export_weights(out_npz)
        if drive_best_npz is not None:
            export_weights(drive_best_npz)
    return out_npz


def main() -> None:
    parser = argparse.ArgumentParser(description="Train v3 4-stage resumable Residual PPO policy on self_jev.sim.")
    parser.add_argument("--init-npz", type=Path, default=Path("models/bc_clean_sim.mapdata.npz"))
    parser.add_argument("--out-npz", type=Path, default=Path("models/jev_rl.mapdata.npz"))
    parser.add_argument("--worlds-per-track", type=int, default=128)
    parser.add_argument("--updates", type=int, default=500)
    parser.add_argument("--rollout-steps", type=int, default=512)
    parser.add_argument("--mb-size", type=int, default=32768)
    parser.add_argument("--h1", type=int, default=384)
    parser.add_argument("--h2", type=int, default=384)
    parser.add_argument("--h3", type=int, default=192)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--ckpt-every", type=int, default=10)
    parser.add_argument("--resume-dir", type=Path, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()
    out = train_rl_sim(
        args.init_npz,
        args.out_npz,
        worlds_per_track=args.worlds_per_track,
        updates=args.updates,
        rollout_steps=args.rollout_steps,
        mb_size=args.mb_size,
        h1=args.h1,
        h2=args.h2,
        h3=args.h3,
        eval_every=args.eval_every,
        ckpt_every=args.ckpt_every,
        resume_dir=args.resume_dir,
        device_str=args.device,
    )
    print(f"Exported {out}")


if __name__ == "__main__":
    main()
