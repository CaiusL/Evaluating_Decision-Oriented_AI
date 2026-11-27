"""Train, temperature-calibrate, and evaluate the between-game Car-Build Decision NN."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from self_jev.car_build_nn.adapter import TaskSpecificModelScorer
from self_jev.car_build_nn.builds import ABSTAIN_LABEL, OPTION_NAMES
from self_jev.car_build_nn.model import CarBuildMLP, export_npz, fit_temperature


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    confs = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    accs = (preds == labels).astype(np.float64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(labels)
    if n == 0:
        return 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (confs > lo) & (confs <= hi) if i > 0 else (confs >= lo) & (confs <= hi)
        if mask.any():
            ece += (mask.sum() / n) * abs(accs[mask].mean() - confs[mask].mean())
    return float(ece)


def _compute_brier(probs: np.ndarray, labels: np.ndarray) -> float:
    n, k = probs.shape
    if n == 0:
        return 0.0
    one_hot = np.zeros((n, k), dtype=np.float64)
    one_hot[np.arange(n), labels] = 1.0
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


def _compute_aurc(probs: np.ndarray, labels: np.ndarray) -> float:
    """Area Under the Risk-Coverage curve (lower is better)."""
    n = len(labels)
    if n == 0:
        return 0.0
    confs = probs.max(axis=1)
    errs = (probs.argmax(axis=1) != labels).astype(np.float64)
    order = np.argsort(-confs)
    cum_err = np.cumsum(errs[order])
    coverages = np.arange(1, n + 1)
    risks = cum_err / coverages
    return float(np.mean(risks))


def _evaluate_split(scorer: TaskSpecificModelScorer, rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"count": 0}
    probs_list: list[list[float]] = []
    labels: list[int] = []
    latencies: list[float] = []
    regrets: list[float] = []

    abstain_idx = OPTION_NAMES.index(ABSTAIN_LABEL)
    false_accepts = 0
    abstain_total = 0
    false_abstains = 0
    non_abstain_total = 0

    # Warm-up
    scorer.score_record(rows[0])

    for r in rows:
        out = scorer.score_record(r)
        p = out["probabilities"]
        pred = out["choice_index"]
        gold = int(r["gold_index"])
        probs_list.append(p)
        labels.append(gold)
        latencies.append(float(out["latency_ms"]))

        if gold == abstain_idx:
            abstain_total += 1
            if pred != abstain_idx:
                false_accepts += 1
        else:
            non_abstain_total += 1
            if pred == abstain_idx:
                false_abstains += 1

        # Downstream oracle regret (for non-abstain predictions where oracle_scores exist)
        oscores = r.get("oracle_scores") or {}
        if gold != abstain_idx and oscores:
            gold_name = OPTION_NAMES[gold]
            best_s = float(oscores.get(gold_name, 0.0))
            if pred == abstain_idx:
                # Safe abstention has a small fixed escalation penalty (2.0 pts) vs wrong build
                regrets.append(2.0)
            else:
                pred_name = OPTION_NAMES[pred]
                regrets.append(max(0.0, best_s - float(oscores.get(pred_name, best_s))))

    probs_arr = np.asarray(probs_list, dtype=np.float64)
    labels_arr = np.asarray(labels, dtype=np.int64)
    acc = float((probs_arr.argmax(axis=1) == labels_arr).mean())
    return {
        "count": len(rows),
        "accuracy": round(acc, 4),
        "ece": round(_compute_ece(probs_arr, labels_arr), 4),
        "brier": round(_compute_brier(probs_arr, labels_arr), 4),
        "aurc": round(_compute_aurc(probs_arr, labels_arr), 4),
        "false_accept_rate": round(false_accepts / abstain_total, 4) if abstain_total else None,
        "false_abstain_rate": round(false_abstains / non_abstain_total, 4) if non_abstain_total else None,
        "mean_oracle_regret": round(float(np.mean(regrets)), 3) if regrets else None,
        "p50_latency_ms": round(float(np.percentile(latencies, 50)), 4),
        "p95_latency_ms": round(float(np.percentile(latencies, 95)), 4),
    }


def train_and_evaluate(
    data_dir: Path,
    out_npz: Path,
    out_report: Path,
    epochs: int = 120,
    batch_size: int = 32,
    lr: float = 3e-3,
    seed: int = 42,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_rows = _load_jsonl(data_dir / "train.jsonl")
    cal_rows = _load_jsonl(data_dir / "calibrate.jsonl")
    test_rows = _load_jsonl(data_dir / "test.jsonl")

    x_tr = np.asarray([r["features"] for r in train_rows], dtype=np.float32)
    y_tr = np.asarray([r["gold_index"] for r in train_rows], dtype=np.int64)
    x_cal = np.asarray([r["features"] for r in cal_rows], dtype=np.float32)
    y_cal = np.asarray([r["gold_index"] for r in cal_rows], dtype=np.int64)

    feat_mean = x_tr.mean(axis=0)
    feat_std = np.where(x_tr.std(axis=0) < 1e-4, 1.0, x_tr.std(axis=0)).astype(np.float32)

    x_tr_n = (x_tr - feat_mean) / feat_std
    x_cal_n = (x_cal - feat_mean) / feat_std

    model = CarBuildMLP(in_dim=x_tr.shape[1], num_classes=len(OPTION_NAMES), hidden=64, dropout=0.1)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    ce = nn.CrossEntropyLoss()

    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_tr_n), torch.from_numpy(y_tr)),
        batch_size=batch_size,
        shuffle=True,
    )
    xc_t = torch.from_numpy(x_cal_n)
    yc_t = torch.from_numpy(y_cal)

    best_cal_loss = float("inf")
    best_sd = None
    for _ in range(epochs):
        model.train()
        for bx, by in loader:
            loss = ce(model(bx), by)
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            cal_loss = float(ce(model(xc_t), yc_t).item())
            if cal_loss < best_cal_loss:
                best_cal_loss = cal_loss
                best_sd = {k: v.detach().clone() for k, v in model.state_dict().items()}

    assert best_sd is not None
    model.load_state_dict(best_sd)
    model.eval()

    with torch.no_grad():
        cal_logits = model(xc_t)
    temp = fit_temperature(cal_logits, yc_t)

    feature_names = train_rows[0]["feature_names"]
    export_npz(model, feat_mean, feat_std, temp, list(OPTION_NAMES), feature_names, out_npz)

    scorer = TaskSpecificModelScorer(out_npz)
    buckets = {
        "overall": _evaluate_split(scorer, test_rows),
        "in_dist": _evaluate_split(scorer, [r for r in test_rows if r["bucket"] == "in_dist"]),
        "abstain_trap": _evaluate_split(scorer, [r for r in test_rows if r["bucket"] == "abstain_trap"]),
        "unseen_lever": _evaluate_split(scorer, [r for r in test_rows if r["bucket"] == "unseen_lever"]),
    }

    param_count = sum(p.numel() for p in model.parameters())
    report = {
        "weights_npz": str(out_npz),
        "param_count": int(param_count),
        "temperature": round(temp, 4),
        "best_cal_nll": round(best_cal_loss, 4),
        "splits": {"train": len(train_rows), "calibrate": len(cal_rows), "test": len(test_rows)},
        "calibrate_metrics": _evaluate_split(scorer, cal_rows),
        "test_buckets": buckets,
    }
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_report.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate between-game Car-Build NN.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/car_build_bench"))
    parser.add_argument("--out-npz", type=Path, default=Path("models/car_build_nn.npz"))
    parser.add_argument("--out-report", type=Path, default=Path("reports/car_build_nn_report.json"))
    parser.add_argument("--epochs", type=int, default=120)
    args = parser.parse_args()
    res = train_and_evaluate(args.data_dir, args.out_npz, args.out_report, epochs=args.epochs)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
