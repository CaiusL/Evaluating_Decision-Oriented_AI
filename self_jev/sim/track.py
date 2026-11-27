"""Vectorized PyTorch track geometry, Frenet projection, banking, and 24-slice lookahead."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

KIT_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "imports" / "comp-racing-history-20261117" / "sources" / "agx-racing-kit" / "data"

_DEFAULT_AERO_ZONES: dict[str, list[tuple[float, float]]] = {
    "parkland": [(0.0, 900.0), (1172.0, 2100.0), (4400.0, 5795.85)],
    "highland": [(0.0, 750.0), (3200.0, 4100.0)],
    "crossover": [(0.0, 800.0), (2600.0, 3500.0)],
}


class TrackTable:
    """GPU/CPU vectorized track lookup table built from agx-racing-kit mapdata.json."""

    def __init__(self, track_id: str, mapdata: dict[str, Any], device: torch.device | str = "cpu") -> None:
        self.track_id = track_id
        self.device = torch.device(device)
        self.length_m = float(mapdata["length_m"])
        self.design_limits = mapdata.get("design_limits", {})

        cl = mapdata["centerline"]
        n_pts = len(cl)
        s_np = np.array([float(p["s_m"]) for p in cl], dtype=np.float32)
        x_np = np.array([float(p["x_m"]) for p in cl], dtype=np.float32)
        y_np = np.array([float(p["y_m"]) for p in cl], dtype=np.float32)
        z_np = np.array([float(p.get("z_m", 0.0)) for p in cl], dtype=np.float32)
        hdg_np = np.array([float(p["heading_rad"]) for p in cl], dtype=np.float32)
        curv_np = np.array([float(p["curvature_1pm"]) for p in cl], dtype=np.float32)
        grade_np = np.array([float(p["grade_dz_ds"]) for p in cl], dtype=np.float32)
        bank_np = np.array([float(p.get("bank_rad", 0.0)) for p in cl], dtype=np.float32)
        wl_np = np.array([float(p["half_width_left_m"]) for p in cl], dtype=np.float32)
        wr_np = np.array([float(p["half_width_right_m"]) for p in cl], dtype=np.float32)

        zones = _DEFAULT_AERO_ZONES.get(track_id, [])
        aero_np = np.zeros(n_pts, dtype=np.float32)
        for s0, s1 in zones:
            aero_np[(s_np >= s0) & (s_np <= s1)] = 1.0

        self.n_pts = n_pts
        self.ds = self.length_m / float(n_pts)

        self.s = torch.tensor(s_np, device=self.device, dtype=torch.float32)
        self.x = torch.tensor(x_np, device=self.device, dtype=torch.float32)
        self.y = torch.tensor(y_np, device=self.device, dtype=torch.float32)
        self.z = torch.tensor(z_np, device=self.device, dtype=torch.float32)
        self.heading = torch.tensor(hdg_np, device=self.device, dtype=torch.float32)
        self.curv = torch.tensor(curv_np, device=self.device, dtype=torch.float32)
        self.grade = torch.tensor(grade_np, device=self.device, dtype=torch.float32)
        self.bank = torch.tensor(bank_np, device=self.device, dtype=torch.float32)
        self.wl = torch.tensor(wl_np, device=self.device, dtype=torch.float32)
        self.wr = torch.tensor(wr_np, device=self.device, dtype=torch.float32)
        self.aero_zone = torch.tensor(aero_np, device=self.device, dtype=torch.float32)

        i_next = torch.arange(n_pts, device=self.device)
        j_next = (i_next + 1) % n_pts
        self.seg_vx = self.x[j_next] - self.x[i_next]
        self.seg_vy = self.y[j_next] - self.y[i_next]
        self.seg_len = torch.sqrt(torch.clamp(self.seg_vx ** 2 + self.seg_vy ** 2, min=1e-10))
        self.seg_len2 = torch.clamp(self.seg_vx ** 2 + self.seg_vy ** 2, min=1e-10)
        self.seg_hdg = torch.atan2(self.seg_vy, self.seg_vx)
        self.seg_ds = torch.remainder(self.s[j_next] - self.s[i_next], self.length_m)

        pit = mapdata["pit"]
        self.pit_entry_s = float(pit["entry_s"])
        self.pit_exit_s = float(pit["exit_s"])
        self.pit_speed_mps = float(pit["speed_limit_kph"]) / 3.6
        self.start_grid = mapdata["start_grid"]

    def s_to_index(self, s: torch.Tensor) -> torch.Tensor:
        s_wrap = torch.remainder(s, self.length_m)
        return torch.clamp(torch.searchsorted(self.s, s_wrap, right=True) - 1, 0, self.n_pts - 1)

    def frenet_to_world(self, s: torch.Tensor, n: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        s_wrap = torch.remainder(s, self.length_m)
        i0 = self.s_to_index(s_wrap)
        ds_seg = torch.clamp(self.seg_ds[i0], min=1e-6)
        alpha = torch.clamp((s_wrap - self.s[i0]) / ds_seg, 0.0, 1.0)

        xc = self.x[i0] + alpha * self.seg_vx[i0]
        yc = self.y[i0] + alpha * self.seg_vy[i0]
        hdg = self.seg_hdg[i0]

        wx = xc - n * torch.sin(hdg)
        wy = yc + n * torch.cos(hdg)
        return wx, wy, hdg

    def project(
        self,
        wx: torch.Tensor,
        wy: torch.Tensor,
        yaw: torch.Tensor,
        prev_idx: torch.Tensor | None = None,
        window: int = 24,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if prev_idx is None:
            dx = wx.unsqueeze(1) - self.x.unsqueeze(0)
            dy = wy.unsqueeze(1) - self.y.unsqueeze(0)
            best_idx = torch.argmin(dx * dx + dy * dy, dim=1)
        else:
            offsets = torch.arange(-window, window + 1, device=self.device, dtype=torch.long)
            cand_idx = (prev_idx.unsqueeze(1) + offsets.unsqueeze(0)) % self.n_pts
            dx = wx.unsqueeze(1) - self.x[cand_idx]
            dy = wy.unsqueeze(1) - self.y[cand_idx]
            local_best = torch.argmin(dx * dx + dy * dy, dim=1)
            best_idx = cand_idx.gather(1, local_best.unsqueeze(1)).squeeze(1)

        i_prev = (best_idx - 1) % self.n_pts

        def _seg_proj(seg_i: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            ax, ay = self.x[seg_i], self.y[seg_i]
            vx, vy = self.seg_vx[seg_i], self.seg_vy[seg_i]
            t = torch.clamp(((wx - ax) * vx + (wy - ay) * vy) / self.seg_len2[seg_i], 0.0, 1.0)
            px = ax + t * vx
            py = ay + t * vy
            dist2 = (wx - px) ** 2 + (wy - py) ** 2
            s_val = self.s[seg_i] + t * self.seg_ds[seg_i]
            seg_h = self.seg_hdg[seg_i]
            n_val = -(wx - px) * torch.sin(seg_h) + (wy - py) * torch.cos(seg_h)
            return dist2, s_val, n_val, seg_h

        d_fwd, s_fwd, n_fwd, h_fwd = _seg_proj(best_idx)
        d_bwd, s_bwd, n_bwd, h_bwd = _seg_proj(i_prev)
        use_fwd = d_fwd <= d_bwd

        s_out = torch.remainder(torch.where(use_fwd, s_fwd, s_bwd), self.length_m)
        n_out = torch.where(use_fwd, n_fwd, n_bwd)
        hdg_out = torch.where(use_fwd, h_fwd, h_bwd)
        he_out = torch.atan2(torch.sin(yaw - hdg_out), torch.cos(yaw - hdg_out))
        return s_out, n_out, he_out, best_idx

    def sample_at_s(self, s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        idx = self.s_to_index(s)
        return self.curv[idx], self.grade[idx], self.wl[idx], self.wr[idx], self.aero_zone[idx]

    def sample_bank_at_s(self, s: torch.Tensor) -> torch.Tensor:
        return self.bank[self.s_to_index(s)]

    def preview_24(self, s: torch.Tensor) -> torch.Tensor:
        offsets = torch.arange(1, 25, device=self.device, dtype=torch.float32) * 15.0
        s_look = s.unsqueeze(1) + offsets.unsqueeze(0)
        curv, grade, wl, wr, zone = self.sample_at_s(s_look)
        stacked = torch.stack([
            curv * 100.0,
            grade * 10.0,
            wl / 10.0,
            wr / 10.0,
            zone,
        ], dim=2)
        return stacked.reshape(s.shape[0], 120)

    def map_ext_25(self, s: torch.Tensor) -> torch.Tensor:
        """Return [N, 25] optional mapdata extension: 24 lookahead bank_rad + normalized pit_entry approach."""
        offsets = torch.arange(1, 25, device=self.device, dtype=torch.float32) * 15.0
        s_look = s.unsqueeze(1) + offsets.unsqueeze(0)
        bank_24 = self.sample_bank_at_s(s_look)  # [N, 24]
        dist_to_pit = torch.remainder(self.pit_entry_s - torch.remainder(s, self.length_m), self.length_m) / self.length_m
        return torch.cat([bank_24, dist_to_pit.unsqueeze(1)], dim=1)


def load_track(track_id: str, device: torch.device | str = "cpu") -> TrackTable:
    path = KIT_DATA_DIR / f"racing-{track_id}.mapdata.json"
    if not path.is_file():
        raise FileNotFoundError(f"Track mapdata not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    return TrackTable(track_id, data, device=device)
