"""Comprehensive multi-track, multi-weather PPO RL trainer for agx_racing_gpusim.

Trains one shared policy across parkland, highland, and crossover with:
- Persistent multi-lap episodes (resets worlds when terminal, not every 8.5s)
- Mixed dry/damp/wet conditions per world fed directly into observations
- Joint PPO updates for continuous controls (throttle, brake, steer) + ERS + active aero
- Penalties for off-track, stalls (status 4), and wall crashes (status 5), plus finish bonus
- Mini-batched GAE(lambda) updates and best-checkpoint export to .mapdata.npz
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def build_obs_vector_gpu(obs_cuda, ix: dict[str, int], wet_tensor, rain_tensor, temp_tensor):
    """Map RaceSim [n, 215] CUDA observation tensor + weather tensors to 181-dim feature tensor."""
    import torch

    obs_cuda = obs_cuda.float()
    n = obs_cuda.shape[0]
    zeros = torch.zeros(n, device=obs_cuda.device)
    ones = torch.ones(n, device=obs_cuda.device)
    feats = [
        obs_cuda[:, ix["speed"]],
        obs_cuda[:, ix["vx"]],
        obs_cuda[:, ix["vy"]],
        obs_cuda[:, ix["n"]],
        obs_cuda[:, ix["heading_error"]],
        obs_cuda[:, ix["yaw_rate"]],
        obs_cuda[:, ix["steer"]],
        obs_cuda[:, ix["gear"]],
        obs_cuda[:, ix["rpm"]] / 10000.0,
        zeros,  # prev throttle
        zeros,  # prev brake
        obs_cuda[:, ix["soc"]],
        obs_cuda[:, ix["fuel"]],
        obs_cuda[:, ix["g_lat"]],
        obs_cuda[:, ix["g_long"]],
        (obs_cuda[:, ix["ts0"]] + obs_cuda[:, ix["ts1"]] + obs_cuda[:, ix["ts2"]] + obs_cuda[:, ix["ts3"]]) / 400.0,
        (obs_cuda[:, ix["wear0"]] + obs_cuda[:, ix["wear1"]] + obs_cuda[:, ix["wear2"]] + obs_cuda[:, ix["wear3"]]) / 4.0,
        obs_cuda[:, ix["front_sliding"]],
        obs_cuda[:, ix["rear_sliding"]],
        zeros,  # aid_available
        zeros,  # override_available
        ones,   # phase == race
    ]
    for k in range(1, 25):
        feats.extend([
            obs_cuda[:, ix[f"pv{k}_curv"]] * 100.0,
            obs_cuda[:, ix[f"pv{k}_grade"]] * 10.0,
            obs_cuda[:, ix[f"pv{k}_wl"]] / 10.0,
            obs_cuda[:, ix[f"pv{k}_wr"]] / 10.0,
            obs_cuda[:, ix[f"pv{k}_zone"]],
        ])
    for r in range(9):
        feats.extend([
            torch.clamp(obs_cuda[:, ix[f"rv{r}_gap"]] / 100.0, -5.0, 5.0),
            torch.clamp(obs_cuda[:, ix[f"rv{r}_dn"]] / 10.0, -2.0, 2.0),
            obs_cuda[:, ix[f"rv{r}_speed"]] / 100.0,
            obs_cuda[:, ix[f"rv{r}_present"]],
        ])
    feats.extend([wet_tensor, rain_tensor, temp_tensor])
    vec = torch.stack(feats, dim=1)
    return torch.nan_to_num(vec, nan=0.0, posinf=5.0, neginf=-5.0)


def make_track_sim(track_id: str, worlds: int, MD, Track, RaceSim, device):
    """Initialize a persistent RaceSim with varied dry/damp/wet conditions across worlds."""
    import torch

    track = Track(track_id)
    C = 10
    sim = RaceSim(worlds, C, MD.STARTER_DESIGN, track, 4, "f32")
    car = sim.base_cars[0]

    # Spread wetness across dry (0.0), damp (0.25), wet (0.60), heavy wet (0.90)
    wet_palette = [0.0, 0.0, 0.25, 0.60, 0.90]
    wet_per_car = np.zeros(worlds * C, dtype=np.float32)

    for w in range(worlds):
        wet = wet_palette[w % len(wet_palette)]
        trk_t = 30.0 - 8.0 * wet
        for j in range(C):
            g = track.raw["start_grid"][j]
            st = MD.new_state(car, g["x"], g["y"], g["yaw"], 0.0, t_surf=88.0)
            s, _n, i_trk = track.project(st.x, st.y)
            st.i_trk = i_trk
            st.s = s
            k = w * C + j
            wet_per_car[k] = wet
            sim.set_state(k, st, dict(wet=wet, air=24.0, trk_t=trk_t))
            sim.rq["status"][0][k] = 1
            sim.rq["pit_cmp"][0][k] = -1
            sim.rq["dist"][0][k] = s - track.L
    sim.upload()
    sim.push_race()
    sim.snapshot()
    sim.race2_step(1)

    wet_t = torch.from_numpy(wet_per_car).to(device)
    rain_t = wet_t * 0.5
    temp_t = (30.0 - 8.0 * wet_t) / 40.0
    return sim, track, wet_t, rain_t, temp_t


def main() -> None:
    parser = argparse.ArgumentParser(description="Comprehensive multi-track PPO RL on agx_racing_gpusim")
    parser.add_argument("--gpusim-dir", type=Path, default=Path("agx_racing_gpusim"))
    parser.add_argument("--init-npz", type=Path, default=Path("models/bc_61bb13c7.mapdata.npz"))
    parser.add_argument("--out-npz", type=Path, default=Path("models/jev_rl.mapdata.npz"))
    parser.add_argument("--worlds-per-track", type=int, default=200)
    parser.add_argument("--updates", type=int, default=360)
    parser.add_argument("--rollout-steps", type=int, default=300)
    parser.add_argument("--mb-size", type=int, default=32768)
    args = parser.parse_args()

    gpusim = args.gpusim_dir.resolve()
    sys.path.insert(0, str(gpusim / "sim" / "gpu"))
    sys.path.insert(0, str(gpusim / "sim" / "ref"))

    import torch
    import torch.nn as nn
    import model as MD
    from track import Track
    from race import RaceSim, OBS2_NAMES

    device = torch.device("cuda")
    ix = {name: i for i, name in enumerate(OBS2_NAMES)}
    track_names = ["parkland", "highland", "crossover"]
    sims = {t: make_track_sim(t, args.worlds_per_track, MD, Track, RaceSim, device) for t in track_names}

    bc = np.load(args.init_npz, allow_pickle=False)
    obs_mean = torch.from_numpy(bc["obs_mean"]).to(device)
    obs_std = torch.from_numpy(bc["obs_std"]).to(device)

    class ActorCritic(nn.Module):
        def __init__(self, in_dim: int) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, 256),
                nn.LayerNorm(256),
                nn.SiLU(),
                nn.Linear(256, 256),
                nn.LayerNorm(256),
                nn.SiLU(),
                nn.Linear(256, 128),
                nn.SiLU(),
            )
            self.cont_head = nn.Linear(128, 3)
            self.ers_head = nn.Linear(128, 4)
            self.flag_head = nn.Linear(128, 3)
            self.log_std = nn.Parameter(torch.tensor([-1.2, -1.5, -1.5]))
            self.value_head = nn.Linear(128, 1)

        def forward(self, x: torch.Tensor):
            h = self.net(x)
            return self.cont_head(h), self.ers_head(h), self.flag_head(h), self.value_head(h).squeeze(-1)

    model = ActorCritic(len(obs_mean)).to(device)
    state_dict = model.state_dict()
    for k in bc.files:
        if k in ("obs_mean", "obs_std"):
            continue
        torch_k = k.replace("__", ".")
        if torch_k in state_dict:
            state_dict[torch_k] = torch.from_numpy(bc[k]).to(device)
    model.load_state_dict(state_dict)

    opt = torch.optim.Adam(model.parameters(), lr=2.5e-4)
    C = 10
    n_cars = args.worlds_per_track * C

    def export_weights(path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        save_dict = {"obs_mean": bc["obs_mean"], "obs_std": bc["obs_std"]}
        for k, v in model.state_dict().items():
            if k.startswith(("net.", "cont_head.", "ers_head.", "flag_head.")):
                save_dict[k.replace(".", "__")] = v.detach().cpu().numpy().astype(np.float32)
        np.savez(path, **save_dict)

    best_score = -1e9
    prev_dist = {
        t: torch.as_tensor(sims[t][0].cuda_array_race("dist"), device=device).clone()
        for t in track_names
    }
    prev_status = {
        t: torch.as_tensor(sims[t][0].cuda_array_race("status"), device=device).clone()
        for t in track_names
    }

    for update in range(1, args.updates + 1):
        t_name = track_names[(update - 1) % len(track_names)]
        sim, _trk, wet_t, rain_t, temp_t = sims[t_name]

        obs_buf, c_buf, e_buf, a_buf, logp_buf, rew_buf, val_buf, done_buf = [], [], [], [], [], [], [], []
        for _ in range(args.rollout_steps):
            raw_obs = torch.as_tensor(sim.obs2_cuda(), device=device).T
            x = (build_obs_vector_gpu(raw_obs, ix, wet_t, rain_t, temp_t) - obs_mean) / obs_std
            x = torch.clamp(torch.nan_to_num(x, nan=0.0, posinf=10.0, neginf=-10.0), -10.0, 10.0)
            with torch.no_grad():
                mu, ers_logits, flag_logits, val = model(x)
                mu = torch.nan_to_num(mu, nan=0.0, posinf=1.0, neginf=-1.0)
                ers_logits = torch.nan_to_num(ers_logits, nan=0.0, posinf=10.0, neginf=-10.0)
                flag_logits = torch.nan_to_num(flag_logits, nan=0.0, posinf=10.0, neginf=-10.0)
                scale = torch.clamp(model.log_std, -4.0, 0.5).exp()
                c_dist = torch.distributions.Normal(mu, scale)
                e_dist = torch.distributions.Categorical(logits=ers_logits)
                a_dist = torch.distributions.Bernoulli(logits=flag_logits[:, 0])

                c_act = c_dist.sample()
                e_act = e_dist.sample()
                a_act = a_dist.sample()

                logp = c_dist.log_prob(c_act).sum(-1) + e_dist.log_prob(e_act) + a_dist.log_prob(a_act)
                act_gpu = torch.stack([
                    torch.clamp(c_act[:, 0], 0.0, 1.0),
                    torch.clamp(c_act[:, 1], 0.0, 1.0),
                    torch.clamp(c_act[:, 2], -1.0, 1.0),
                    e_act.float(),
                    a_act,
                    torch.zeros(n_cars, device=device),
                ], dim=1)

            sim.host["act"][:] = act_gpu.cpu().numpy()
            sim.race_step(2)

            cur_dist = torch.as_tensor(sim.cuda_array_race("dist"), device=device)
            status = torch.as_tensor(sim.cuda_array_race("status"), device=device)
            off_track = torch.as_tensor(sim.cuda_array_race("off_track"), device=device)

            was_active = prev_status[t_name] == 1
            progress = torch.where(was_active, (cur_dist - prev_dist[t_name]) / 10.0, 0.0)
            off_pen = torch.where(was_active & (off_track == 1), -0.05, 0.0)

            newly_done = was_active & (status != 1)
            fin_bonus = torch.where(newly_done & (status == 2), 15.0, 0.0)
            stall_pen = torch.where(newly_done & (status == 4), -5.0, 0.0)
            wall_pen = torch.where(newly_done & (status == 5), -10.0, 0.0)

            rew = progress + off_pen + fin_bonus + stall_pen + wall_pen
            done = (status != 1).float()

            obs_buf.append(x)
            c_buf.append(c_act)
            e_buf.append(e_act)
            a_buf.append(a_act)
            logp_buf.append(logp)
            rew_buf.append(rew)
            val_buf.append(val)
            done_buf.append(done)

            # Reset any world where all cars are terminal so episodes span full 4-lap races
            world_done = (status.view(args.worlds_per_track, C) != 1).all(dim=1)
            if world_done.any():
                sim.reset_worlds(world_done.cpu().numpy())
                sim.race2_step(1)
                cur_dist = torch.as_tensor(sim.cuda_array_race("dist"), device=device)
                status = torch.as_tensor(sim.cuda_array_race("status"), device=device)

            prev_dist[t_name] = cur_dist.clone()
            prev_status[t_name] = status.clone()

        # GAE(lambda=0.95, gamma=0.99)
        with torch.no_grad():
            raw_obs = torch.as_tensor(sim.obs2_cuda(), device=device).T
            x_next = (build_obs_vector_gpu(raw_obs, ix, wet_t, rain_t, temp_t) - obs_mean) / obs_std
            x_next = torch.clamp(torch.nan_to_num(x_next, nan=0.0, posinf=10.0, neginf=-10.0), -10.0, 10.0)
            _, _, _, next_val = model(x_next)

        T_steps = len(rew_buf)
        adv_buf = [None] * T_steps
        gae = torch.zeros(n_cars, device=device)
        for t in reversed(range(T_steps)):
            non_term = 1.0 - done_buf[t]
            nv = next_val if t == T_steps - 1 else val_buf[t + 1]
            delta = rew_buf[t] + 0.99 * nv * non_term - val_buf[t]
            gae = delta + 0.99 * 0.95 * non_term * gae
            adv_buf[t] = gae

        b_obs = torch.cat(obs_buf)
        b_c = torch.cat(c_buf)
        b_e = torch.cat(e_buf)
        b_a = torch.cat(a_buf)
        b_old_logp = torch.cat(logp_buf)
        b_adv = torch.cat(adv_buf)
        b_ret = b_adv + torch.cat(val_buf)
        b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

        total_n = b_obs.shape[0]
        for _ in range(4):
            perm = torch.randperm(total_n, device=device)
            for start in range(0, total_n, args.mb_size):
                idx = perm[start:start + args.mb_size]
                mu, ers_logits, flag_logits, val = model(b_obs[idx])
                mu = torch.nan_to_num(mu, nan=0.0, posinf=1.0, neginf=-1.0)
                ers_logits = torch.nan_to_num(ers_logits, nan=0.0, posinf=10.0, neginf=-10.0)
                flag_logits = torch.nan_to_num(flag_logits, nan=0.0, posinf=10.0, neginf=-10.0)
                scale = torch.clamp(model.log_std, -4.0, 0.5).exp()
                c_dist = torch.distributions.Normal(mu, scale)
                e_dist = torch.distributions.Categorical(logits=ers_logits)
                a_dist = torch.distributions.Bernoulli(logits=flag_logits[:, 0])

                logp = c_dist.log_prob(b_c[idx]).sum(-1) + e_dist.log_prob(b_e[idx]) + a_dist.log_prob(b_a[idx])
                entropy = c_dist.entropy().sum(-1) + e_dist.entropy() + a_dist.entropy()

                ratio = torch.exp(logp - b_old_logp[idx])
                surr1 = ratio * b_adv[idx]
                surr2 = torch.clamp(ratio, 0.8, 1.2) * b_adv[idx]
                pg_loss = -torch.min(surr1, surr2).mean()
                vf_loss = 0.5 * (b_ret[idx] - val).pow(2).mean()
                loss = pg_loss + vf_loss - 0.01 * entropy.mean()

                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                opt.step()

        mean_rew = torch.stack(rew_buf).mean().item()
        if mean_rew > best_score and update >= 30:
            best_score = mean_rew
            export_weights(args.out_npz)

        if update % 15 == 0 or update == 1:
            st = prev_status[t_name]
            fin = int((st == 2).sum().item())
            stl = int((st == 4).sum().item())
            wal = int((st == 5).sum().item())
            print(f"update {update}/{args.updates} [{t_name:9s}] mean_rew={mean_rew:.4f} fin={fin} stall={stl} wall={wal}")

    export_weights(args.out_npz)
    print(f"Saved comprehensive RL weights to {args.out_npz}")


if __name__ == "__main__":
    main()
