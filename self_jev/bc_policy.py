"""Behavioral cloning policy trainer (v1 single-trace & v2 stratified) and pure-NumPy inference."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

FEATURE_DIM = 22 + 24 * 5 + 9 * 4 + 3  # 181 features


def extract_features(obs: dict[str, Any]) -> np.ndarray:
    """Convert one platform observation dict into a flat float32 feature vector."""
    s = obs.get("self") or {}
    w = obs.get("weather") or {}

    tyre_temp = s.get("tyre_temp_c") or [70.0, 70.0, 70.0, 70.0]
    if isinstance(tyre_temp, (int, float)):
        tyre_temp = [float(tyre_temp)] * 4
    tyre_wear = s.get("tyre_wear") or [0.0, 0.0, 0.0, 0.0]
    if isinstance(tyre_wear, (int, float)):
        tyre_wear = [float(tyre_wear)] * 4

    self_vec = [
        float(s.get("speed_mps", 0.0)),
        float(s.get("vx", 0.0)),
        float(s.get("vy", 0.0)),
        float(s.get("n", 0.0)),
        float(s.get("heading_error", 0.0)),
        float(s.get("yaw_rate", 0.0)),
        float(s.get("steer", 0.0)),
        float(s.get("gear", 1.0)),
        float(s.get("rpm", 0.0)) / 10000.0,
        float(s.get("throttle", 0.0)),
        float(s.get("brake", 0.0)),
        float(s.get("ers_soc_mj", 0.0)),
        float(s.get("fuel_kg", 0.0)),
        float(s.get("g_lat", 0.0)),
        float(s.get("g_long", 0.0)),
        float(np.mean(tyre_temp)) / 100.0,
        float(np.mean(tyre_wear)),
        1.0 if s.get("front_sliding") else 0.0,
        1.0 if s.get("rear_sliding") else 0.0,
        1.0 if s.get("aid_available") else 0.0,
        1.0 if s.get("override_available") else 0.0,
        1.0 if obs.get("phase") == "race" else 0.0,
    ]

    pv_rows = (obs.get("preview") or {}).get("rows") or []
    pv_flat: list[float] = []
    for i in range(24):
        if i < len(pv_rows):
            row = pv_rows[i]
            pv_flat.extend([
                float(row[1]) * 100.0,
                float(row[2]) * 10.0,
                float(row[3]) / 10.0,
                float(row[4]) / 10.0,
                float(row[5]),
            ])
        else:
            pv_flat.extend([0.0, 0.0, 0.8, 0.8, 0.0])

    rivals = obs.get("rivals") or []
    active_rivals = [
        r for r in rivals
        if r.get("status") == "racing" and r.get("gap_m") is not None
    ]
    active_rivals.sort(key=lambda r: abs(float(r.get("gap_m", 9999.0))))
    rv_flat: list[float] = []
    for i in range(9):
        if i < len(active_rivals):
            r = active_rivals[i]
            gap = float(r.get("gap_m", 0.0))
            rv_flat.extend([
                float(np.clip(gap / 100.0, -5.0, 5.0)),
                float(np.clip(float(r.get("dn", 0.0)) / 10.0, -2.0, 2.0)),
                float(r.get("speed_mps", 0.0)) / 100.0,
                1.0,
            ])
        else:
            rv_flat.extend([5.0, 0.0, 0.0, 0.0])

    weather_vec = [
        float(w.get("track_wetness", 0.0)),
        float(w.get("rain_intensity", 0.0)),
        float(w.get("track_temp_c", 25.0)) / 40.0,
    ]

    vec = np.asarray(self_vec + pv_flat + rv_flat + weather_vec, dtype=np.float32)
    return np.nan_to_num(vec, nan=0.0, posinf=5.0, neginf=-5.0)


def _stratum_key(obs: dict[str, Any]) -> str:
    trk_len = float(obs.get("track_length_m", 5800.0))
    trk = "highland" if trk_len > 6500.0 else ("crossover" if trk_len > 5800.0 else "parkland")
    wet = float((obs.get("weather") or {}).get("track_wetness", 0.0))
    band = "wet" if wet >= 0.5 else ("damp" if wet > 0.05 else "dry")
    ph = str(obs.get("phase", "race"))
    return f"{trk}:{band}:{ph}"


def load_jsonl_dataset(path: Path, clean_only: bool = False) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray, np.ndarray, list[str]]:
    """Load decision rows from a training export jsonl."""
    xs: list[np.ndarray] = []
    throttle: list[float] = []
    brake: list[float] = []
    steer: list[float] = []
    ers_mode: list[int] = []
    aero_mode: list[float] = []
    override: list[float] = []
    aid: list[float] = []
    ovr_mask: list[float] = []
    laps: list[int] = []
    is_race: list[bool] = []
    strata: list[str] = []

    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            obj = json.loads(line)
            if "observation" not in obj:
                continue
            obs = obj["observation"]
            s = obs.get("self") or {}
            if clean_only and s.get("status") != "racing":
                continue
            act = obj["action"]
            wet = float((obs.get("weather") or {}).get("track_wetness", 0.0))
            xs.append(extract_features(obs))
            throttle.append(float(act.get("throttle", 0.0)))
            brake.append(float(act.get("brake", 0.0)))
            steer.append(float(act.get("steer", 0.0)))
            ers_mode.append(int(act.get("ers_mode", 1)))
            aero_mode.append(1.0 if act.get("aero_mode") else 0.0)
            override.append(1.0 if act.get("override") else 0.0)
            aid.append(1.0 if act.get("aid") else 0.0)
            ovr_mask.append(1.0 if (s.get("override_available") and wet < 0.5) else 0.0)
            laps.append(int(s.get("lap", 0)))
            is_race.append(obs.get("phase") == "race")
            strata.append(_stratum_key(obs))

    x_arr = np.stack(xs).astype(np.float32)
    targets = {
        "cont": np.stack([throttle, brake, steer], axis=1).astype(np.float32),
        "ers_mode": np.asarray(ers_mode, dtype=np.int64),
        "flags": np.stack([aero_mode, override, aid], axis=1).astype(np.float32),
        "ovr_mask": np.asarray(ovr_mask, dtype=np.float32),
    }
    return x_arr, targets, np.asarray(laps, dtype=np.int64), np.asarray(is_race, dtype=bool), strata


def _fit_and_export(
    x_train_raw: np.ndarray,
    y_train: dict[str, np.ndarray],
    sample_weights: np.ndarray,
    x_val_raw: np.ndarray,
    y_val: dict[str, np.ndarray],
    out_npz: Path,
    out_report: Path,
    meta: dict[str, Any],
    epochs: int = 35,
    batch_size: int = 256,
    lr: float = 1e-3,
    seed: int = 42,
) -> dict[str, Any]:
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler

    torch.manual_seed(seed)
    np.random.seed(seed)

    mean = x_train_raw.mean(axis=0)
    std = x_train_raw.std(axis=0)
    std = np.where(std < 1e-4, 1.0, std).astype(np.float32)

    x_train = ((x_train_raw - mean) / std).astype(np.float32)
    x_val = ((x_val_raw - mean) / std).astype(np.float32)

    class PolicyNet(nn.Module):
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

        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            h = self.net(x)
            return self.cont_head(h), self.ers_head(h), self.flag_head(h)

    model = PolicyNet(x_train.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    ds = TensorDataset(
        torch.from_numpy(x_train),
        torch.from_numpy(y_train["cont"]),
        torch.from_numpy(y_train["ers_mode"]),
        torch.from_numpy(y_train["flags"]),
        torch.from_numpy(y_train["ovr_mask"]),
    )
    sampler = WeightedRandomSampler(torch.from_numpy(sample_weights).double(), num_samples=len(ds), replacement=True)
    loader = DataLoader(ds, batch_size=batch_size, sampler=sampler)

    smooth_l1 = nn.SmoothL1Loss(beta=0.05)
    ce = nn.CrossEntropyLoss()
    bce_none = nn.BCEWithLogitsLoss(reduction="none")

    best_val_loss = float("inf")
    best_state: dict[str, np.ndarray] | None = None

    xv = torch.from_numpy(x_val)
    yv_cont = torch.from_numpy(y_val["cont"])
    yv_ers = torch.from_numpy(y_val["ers_mode"])
    yv_flags = torch.from_numpy(y_val["flags"])

    for _epoch in range(1, epochs + 1):
        model.train()
        for bx, bc, be, bf, bm in loader:
            pc, pe, pf = model(bx)
            f_loss = bce_none(pf, bf)
            f_mask = torch.stack([torch.ones_like(bm), bm, torch.ones_like(bm)], dim=1)
            masked_f = (f_loss * f_mask).sum() / torch.clamp(f_mask.sum(), min=1.0)
            loss = smooth_l1(pc, bc) + 0.35 * ce(pe, be) + 0.25 * masked_f
            opt.zero_grad()
            loss.backward()
            opt.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            pc, pe, pf = model(xv)
            val_loss = (smooth_l1(pc, yv_cont) + 0.35 * ce(pe, yv_ers) + 0.25 * bce_none(pf, yv_flags).mean()).item()
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}

    assert best_state is not None
    model.load_state_dict({k: torch.from_numpy(v) for k, v in best_state.items()})
    model.eval()
    with torch.no_grad():
        pc, pe, pf = model(xv)
        pred_cont = pc.numpy()
        pred_cont[:, 0] = np.clip(pred_cont[:, 0], 0.0, 1.0)
        pred_cont[:, 1] = np.clip(pred_cont[:, 1], 0.0, 1.0)
        pred_cont[:, 2] = np.clip(pred_cont[:, 2], -1.0, 1.0)
        true_cont = y_val["cont"]
        mae_throttle = float(np.abs(pred_cont[:, 0] - true_cont[:, 0]).mean())
        mae_brake = float(np.abs(pred_cont[:, 1] - true_cont[:, 1]).mean())
        mae_steer = float(np.abs(pred_cont[:, 2] - true_cont[:, 2]).mean())
        ers_acc = float((pe.argmax(dim=1).numpy() == y_val["ers_mode"]).mean())
        aero_acc = float(((torch.sigmoid(pf[:, 0]).numpy() >= 0.5) == (y_val["flags"][:, 0] >= 0.5)).mean())

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    save_dict = {"obs_mean": mean, "obs_std": std}
    for k, v in best_state.items():
        save_dict[k.replace(".", "__")] = v.astype(np.float32)
    np.savez(out_npz, **save_dict)

    report = {
        **meta,
        "weights_npz": str(out_npz),
        "train_rows": int(len(x_train)),
        "val_rows": int(len(x_val)),
        "feature_dim": int(x_train.shape[1]),
        "epochs": epochs,
        "best_val_loss": round(best_val_loss, 5),
        "val_mae_throttle": round(mae_throttle, 4),
        "val_mae_brake": round(mae_brake, 4),
        "val_mae_steer": round(mae_steer, 4),
        "val_ers_mode_acc": round(ers_acc, 4),
        "val_aero_mode_acc": round(aero_acc, 4),
    }
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_report.write_text(json.dumps(report, indent=2) + "\n")
    return report


def train_bc(
    jsonl_path: Path,
    out_npz: Path,
    out_report: Path,
    epochs: int = 35,
) -> dict[str, Any]:
    x_all, targets, laps, is_race, _ = load_jsonl_dataset(jsonl_path)
    val_mask = is_race & (laps == 3)
    if val_mask.sum() < 200:
        split = int(len(x_all) * 0.85)
        val_mask = np.zeros(len(x_all), dtype=bool)
        val_mask[split:] = True
    train_mask = ~val_mask
    y_tr = {k: v[train_mask] for k, v in targets.items()}
    y_va = {k: v[val_mask] for k, v in targets.items()}
    sw = np.ones(int(train_mask.sum()), dtype=np.float32)
    return _fit_and_export(
        x_all[train_mask], y_tr, sw, x_all[val_mask], y_va, out_npz, out_report,
        {"mode": "single_trace", "source_jsonl": str(jsonl_path)}, epochs=epochs,
    )


def train_bc_stratified(
    train_paths: list[Path],
    val_paths: list[Path],
    out_npz: Path,
    out_report: Path,
    epochs: int = 35,
) -> dict[str, Any]:
    """v2 Stratified BC across tracks & weather bands with whole-match holdouts."""
    xs_tr, ys_tr_list, strata_tr = [], [], []
    for p in train_paths:
        x, t, _, _, st = load_jsonl_dataset(p, clean_only=True)
        xs_tr.append(x)
        ys_tr_list.append(t)
        strata_tr.extend(st)
    x_tr = np.concatenate(xs_tr, axis=0)
    y_tr = {k: np.concatenate([y[k] for y in ys_tr_list], axis=0) for k in ys_tr_list[0]}

    # Inverse-stratum weighting
    counts: dict[str, int] = {}
    for s in strata_tr:
        counts[s] = counts.get(s, 0) + 1
    sw = np.array([1.0 / max(1, counts[s]) for s in strata_tr], dtype=np.float32)
    sw = sw / sw.mean()

    xs_va, ys_va_list = [], []
    for p in val_paths:
        x, t, _, _, _ = load_jsonl_dataset(p, clean_only=True)
        xs_va.append(x)
        ys_va_list.append(t)
    x_va = np.concatenate(xs_va, axis=0)
    y_va = {k: np.concatenate([y[k] for y in ys_va_list], axis=0) for k in ys_va_list[0]}

    return _fit_and_export(
        x_tr, y_tr, sw, x_va, y_va, out_npz, out_report,
        {
            "mode": "stratified_v2",
            "train_matches": [p.name for p in train_paths],
            "val_matches": [p.name for p in val_paths],
            "strata_counts": counts,
        },
        epochs=epochs,
    )


class BCPolicy:
    """Pure-NumPy inference wrapper compatible with agx.cool submissions."""

    def __init__(self, weights_path: Path) -> None:
        data = np.load(weights_path, allow_pickle=False)
        self.mean = data["obs_mean"]
        self.std = data["obs_std"]
        self.w0 = data["net__0__weight"]
        self.b0 = data["net__0__bias"]
        self.ln1_w = data["net__1__weight"]
        self.ln1_b = data["net__1__bias"]
        self.w3 = data["net__3__weight"]
        self.b3 = data["net__3__bias"]
        self.ln4_w = data["net__4__weight"]
        self.ln4_b = data["net__4__bias"]
        self.w6 = data["net__6__weight"]
        self.b6 = data["net__6__bias"]
        self.wc = data["cont_head__weight"]
        self.bc = data["cont_head__bias"]
        self.we = data["ers_head__weight"]
        self.be = data["ers_head__bias"]
        self.wf = data["flag_head__weight"]
        self.bf = data["flag_head__bias"]

    @staticmethod
    def _silu(x: np.ndarray) -> np.ndarray:
        return x / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))

    @staticmethod
    def _ln(x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
        m = x.mean()
        v = x.var()
        return ((x - m) / np.sqrt(v + 1e-5)) * w + b

    def act(self, obs: dict[str, Any]) -> dict[str, Any]:
        x = (extract_features(obs) - self.mean) / self.std
        h = self._silu(self._ln(x @ self.w0.T + self.b0, self.ln1_w, self.ln1_b))
        h = self._silu(self._ln(h @ self.w3.T + self.b3, self.ln4_w, self.ln4_b))
        h = self._silu(h @ self.w6.T + self.b6)
        cont = h @ self.wc.T + self.bc
        ers = int(np.argmax(h @ self.we.T + self.be))
        flags = h @ self.wf.T + self.bf
        return {
            "throttle": float(np.clip(cont[0], 0.0, 1.0)),
            "brake": float(np.clip(cont[1], 0.0, 1.0)),
            "steer": float(np.clip(cont[2], -1.0, 1.0)),
            "ers_mode": ers,
            "aero_mode": 1 if flags[0] > 0.0 else 0,
            "override": bool(flags[1] > 0.0),
            "aid": bool(flags[2] > 0.0),
            "pit_request": False,
            "pit_compound": None,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a behavioral-cloning racing policy.")
    parser.add_argument("--stratified", action="store_true", help="Run v2 stratified BC across owned matches.")
    parser.add_argument("--jsonl", type=Path, default=Path("1st/training-61bb13c7-8b8f-47e4-aca9-70920619993a.jsonl"))
    parser.add_argument("--out-npz", type=Path, default=Path("models/bc_61bb13c7.mapdata.npz"))
    parser.add_argument("--out-report", type=Path, default=Path("reports/bc_61bb13c7.json"))
    parser.add_argument("--epochs", type=int, default=35)
    args = parser.parse_args()
    if args.stratified:
        train_files = [
            Path("1st/training-61bb13c7-8b8f-47e4-aca9-70920619993a.jsonl"),
            Path("data/traces/a2585d41-seat0-training.jsonl"),
            Path("data/DL/049a8010-6985-416c-abf9-d3d076d0b560/training-049a8010-6985-416c-abf9-d3d076d0b560.jsonl"),
            Path("data/DL/37b81ebd-33ee-4b35-9765-00c8e995f001/training-37b81ebd-33ee-4b35-9765-00c8e995f001.jsonl"),
            Path("data/DL/a94b748b-eac4-48c1-bd5d-b019d61e0b61/training-a94b748b-eac4-48c1-bd5d-b019d61e0b61.jsonl"),
        ]
        val_files = [
            Path("data/training-ebe0e4b9-c62f-415c-8000-4aa51f9f662a.jsonl"),
            Path("data/DL/08569f9b-52d9-42f2-89b4-b8fbf9de03e2/training-08569f9b-52d9-42f2-89b4-b8fbf9de03e2.jsonl"),
        ]
        report = train_bc_stratified(train_files, val_files, args.out_npz, args.out_report, epochs=args.epochs)
    else:
        report = train_bc(args.jsonl, args.out_npz, args.out_report, epochs=args.epochs)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
