#!/usr/bin/env python3
"""Train Step 3 Behavior Cloning baseline (models/bc_winner_v1.mapdata.npz) on winner traces.

Trains an ActorCritic(181) policy on data/winner_bc/train.npz (12 train races), selects
best checkpoint by held-out validation loss on data/winner_bc/val.npz (4 val races),
saves models/bc_winner_v1.mapdata.npz (drop-in compatible with _load_frozen_driver),
and evaluates closed-loop 4-lap pace on the 4 held-out val scenarios (plus optional --eval-all).

Run with project virtualenv:
  .venv-decision/bin/python scripts/train_winner_bc.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
for p in (str(REPO_ROOT), str(SCRIPTS_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from eval_winner_builds import build_dict_to_car_params, run_single_4lap
from self_jev.sim.design import load_design
from self_jev.sim.physics import PhysicsConfig
from self_jev.train_rl_sim import ActorCritic

DEFAULT_DATA_DIR = REPO_ROOT / "data" / "winner_bc"
DEFAULT_WINNER_JSON = REPO_ROOT / "data" / "winner-builds-and-controls.json"
DEFAULT_OUT_NPZ = REPO_ROOT / "models" / "bc_winner_v1.mapdata.npz"
DEFAULT_REPORT = REPO_ROOT / "reports" / "bc_winner_v1_report.json"


def _eval_offline(model: nn.Module, x: torch.Tensor, c: torch.Tensor, e: torch.Tensor, f: torch.Tensor) -> dict[str, float]:
    model.eval()
    l1 = nn.SmoothL1Loss(beta=0.05)
    ce = nn.CrossEntropyLoss()
    bce = nn.BCEWithLogitsLoss()
    with torch.inference_mode():
        pc, pe, pf, _ = model(x)
        loss_c = float(l1(pc, c).item())
        loss_e = float(ce(pe, e).item())
        loss_f = float(bce(pf, f).item())
        total = loss_c + 0.3 * loss_e + 0.2 * loss_f
        mae = torch.abs(pc - c).mean(dim=0)
        ers_acc = float((torch.argmax(pe, dim=-1) == e).float().mean().item())
        aero_acc = float(((pf[:, 0] > 0.0).float() == f[:, 0]).float().mean().item())
    return {
        "total_loss": round(total, 5),
        "cont_loss": round(loss_c, 5),
        "thr_mae": round(float(mae[0].item()), 4),
        "brk_mae": round(float(mae[1].item()), 4),
        "steer_mae": round(float(mae[2].item()), 4),
        "ers_acc": round(ers_acc, 4),
        "aero_acc": round(aero_acc, 4),
    }


def _save_npz(model: nn.Module, obs_mean: np.ndarray, obs_std: np.ndarray, out_npz: Path) -> None:
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    save_dict: dict[str, Any] = {
        "obs_mean": obs_mean.astype(np.float32),
        "obs_std": obs_std.astype(np.float32),
    }
    for k, v in model.state_dict().items():
        if k.startswith(("net.", "cont_head.", "ers_head.", "flag_head.")):
            save_dict[k.replace(".", "__")] = v.detach().cpu().numpy().astype(np.float32)
    np.savez(out_npz, **save_dict)


def main() -> None:
    ap = argparse.ArgumentParser(description="Train winner BC baseline and evaluate in closed-loop 4-lap sim.")
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--winner-json", type=Path, default=DEFAULT_WINNER_JSON)
    ap.add_argument("--out-npz", type=Path, default=DEFAULT_OUT_NPZ)
    ap.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1.5e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--eval-all", action="store_true", help="Evaluate all 16 regimes in sim (default: 4 held-out val regimes)")
    args = ap.parse_args()

    t0 = time.perf_counter()
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    tr = np.load(args.data_dir / "train.npz")
    va = np.load(args.data_dir / "val.npz")
    manifest = json.loads((args.data_dir / "dataset_manifest.json").read_text(encoding="utf-8"))

    obs_mean = tr["obs_mean"].astype(np.float32)
    obs_std = tr["obs_std"].astype(np.float32)

    def _norm(arr: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.clip((arr - obs_mean) / obs_std, -10.0, 10.0).astype(np.float32))

    x_tr, c_tr, e_tr, f_tr = _norm(tr["obs"]), torch.from_numpy(tr["cont"]), torch.from_numpy(tr["ers"]), torch.from_numpy(tr["flags"])
    x_va, c_va, e_va, f_va = _norm(va["obs"]), torch.from_numpy(va["cont"]), torch.from_numpy(va["ers"]), torch.from_numpy(va["flags"])

    model = ActorCritic(x_tr.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loader = DataLoader(TensorDataset(x_tr, c_tr, e_tr, f_tr), batch_size=args.batch_size, shuffle=True)

    l1 = nn.SmoothL1Loss(beta=0.05)
    ce = nn.CrossEntropyLoss()
    bce = nn.BCEWithLogitsLoss()

    best_val = float("inf")
    best_state = None
    best_metrics: dict[str, float] = {}
    history = []

    x_va_d, c_va_d, e_va_d, f_va_d = x_va.to(device), c_va.to(device), e_va.to(device), f_va.to(device)

    for ep in range(1, args.epochs + 1):
        model.train()
        for bx, bc_t, be_t, bf_t in loader:
            bx, bc_t, be_t, bf_t = bx.to(device), bc_t.to(device), be_t.to(device), bf_t.to(device)
            pc, pe, pf, _ = model(bx)
            # Upweight steering slightly since lateral stability dominates closed-loop survival
            loss_c = l1(pc[:, :2], bc_t[:, :2]) + 2.0 * l1(pc[:, 2], bc_t[:, 2])
            loss = loss_c + 0.3 * ce(pe, be_t) + 0.2 * bce(pf, bf_t)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

        vm = _eval_offline(model, x_va_d, c_va_d, e_va_d, f_va_d)
        history.append({"epoch": ep, **vm})
        if vm["total_loss"] < best_val:
            best_val = vm["total_loss"]
            best_metrics = vm
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if ep == 1 or ep % 5 == 0 or ep == args.epochs:
            print(
                f"Epoch {ep:02d}/{args.epochs} | val_loss={vm['total_loss']:.4f} "
                f"| MAE thr={vm['thr_mae']:.3f} brk={vm['brk_mae']:.3f} steer={vm['steer_mae']:.4f} "
                f"| ers={vm['ers_acc']:.3f} aero={vm['aero_acc']:.3f}"
            )

    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    _save_npz(model, obs_mean, obs_std, args.out_npz)
    train_s = round(time.perf_counter() - t0, 2)
    print(f"\nSaved best BC checkpoint to {args.out_npz} in {train_s}s")

    # Closed-loop 4-lap validation in RaceEnv
    doc = json.loads(args.winner_json.read_text(encoding="utf-8"))
    w_idx = doc["control_trace_columns"].index("track_wetness")
    cal_path = REPO_ROOT / "self_jev" / "sim" / "calibrated_params.json"
    cfg = PhysicsConfig.from_json(cal_path) if cal_path.is_file() else PhysicsConfig()
    base = load_design()

    eval_keys = sorted(doc["winners"].keys()) if args.eval_all else sorted(manifest["val_entries"])
    obs_mean_t = torch.from_numpy(obs_mean).to(device)
    obs_std_t = torch.from_numpy(obs_std).to(device)

    sim_rows = []
    print(f"\n=== Closed-Loop 4-Lap Sim Eval ({len(eval_keys)} scenarios) ===")
    for key in eval_keys:
        e = doc["winners"][key]
        tr_name = e["track"].lower()
        trace = e["control_trace"]
        w0, w1 = float(trace[0][w_idx]), float(trace[-1][w_idx])
        agx_t = float(e["official_finish_s"])
        car = build_dict_to_car_params(e["build"], base)
        res = run_single_4lap(
            tr_name, car, w0, w1, model, obs_mean_t, obs_std_t, False, cfg, device
        )
        gap_s = round(res["finish_time_s"] - agx_t, 2) if res["finished"] and res["finish_time_s"] else None
        gap_pct = round(100.0 * gap_s / agx_t, 1) if gap_s is not None else None
        sim_rows.append({
            "key": key,
            "split": "val" if key in set(manifest["val_entries"]) else "train",
            "agx_winner_s": round(agx_t, 2),
            **res,
            "gap_vs_agx_s": gap_s,
            "gap_vs_agx_pct": gap_pct,
        })
        print(
            f"  {key:22s} | AGX={agx_t:6.1f}s | bc={str(res['finish_time_s']):>7s}s "
            f"| status={res['status']} laps={res['completed_laps']} | gap={gap_s}s ({gap_pct}%)"
        )

    report = {
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(args.out_npz.relative_to(REPO_ROOT)),
        "train_rows": int(len(x_tr)),
        "val_rows": int(len(x_va)),
        "epochs": args.epochs,
        "train_wall_s": train_s,
        "best_val_offline_metrics": best_metrics,
        "closed_loop_sim": sim_rows,
        "history": history,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved BC report to {args.report}")


if __name__ == "__main__":
    main()
