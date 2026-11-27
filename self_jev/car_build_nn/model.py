"""Compact Car-Build MLP, post-hoc temperature scaling, and pure-NumPy .npz inference."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


class CarBuildMLP(nn.Module):
    """Lightweight 2-hidden-layer MLP for between-game car-build selection."""

    def __init__(self, in_dim: int, num_classes: int, hidden: int = 64, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Fit scalar temperature T > 0 strictly on the calibration split via NLL minimization."""
    log_t = nn.Parameter(torch.zeros(1, dtype=torch.float32))
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=60)
    ce = nn.CrossEntropyLoss()

    def closure():
        opt.zero_grad()
        temp = torch.exp(log_t).clamp(0.1, 20.0)
        loss = ce(logits / temp, labels)
        loss.backward()
        return loss

    opt.step(closure)
    return float(torch.exp(log_t).clamp(0.1, 20.0).item())


def export_npz(
    model: CarBuildMLP,
    feat_mean: np.ndarray,
    feat_std: np.ndarray,
    temperature: float,
    option_names: list[str],
    feature_names: list[str],
    out_path: Path,
) -> None:
    """Export trained Car-Build NN weights + calibration params to pure-NumPy .npz."""
    sd = model.state_dict()
    payload: dict[str, np.ndarray] = {
        "feat_mean": feat_mean.astype(np.float32),
        "feat_std": feat_std.astype(np.float32),
        "temperature": np.asarray([temperature], dtype=np.float32),
        "option_names_utf8": np.frombuffer("\n".join(option_names).encode("utf-8"), dtype=np.uint8),
        "feature_names_utf8": np.frombuffer("\n".join(feature_names).encode("utf-8"), dtype=np.uint8),
        "w0": sd["net.0.weight"].detach().cpu().numpy().astype(np.float32),
        "b0": sd["net.0.bias"].detach().cpu().numpy().astype(np.float32),
        "ln1_w": sd["net.1.weight"].detach().cpu().numpy().astype(np.float32),
        "ln1_b": sd["net.1.bias"].detach().cpu().numpy().astype(np.float32),
        "w4": sd["net.4.weight"].detach().cpu().numpy().astype(np.float32),
        "b4": sd["net.4.bias"].detach().cpu().numpy().astype(np.float32),
        "ln5_w": sd["net.5.weight"].detach().cpu().numpy().astype(np.float32),
        "ln5_b": sd["net.5.bias"].detach().cpu().numpy().astype(np.float32),
        "w8": sd["net.8.weight"].detach().cpu().numpy().astype(np.float32),
        "b8": sd["net.8.bias"].detach().cpu().numpy().astype(np.float32),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **payload)


class CarBuildNumPy:
    """Zero-dependency pure-NumPy inference wrapper (< 1 ms per decision)."""

    def __init__(self, npz_path: Path | str) -> None:
        d = np.load(npz_path, allow_pickle=False)
        self.feat_mean = d["feat_mean"]
        self.feat_std = d["feat_std"]
        self.temperature = float(d["temperature"][0])
        self.option_names = bytes(d["option_names_utf8"].tolist()).decode("utf-8").split("\n")
        self.feature_names = bytes(d["feature_names_utf8"].tolist()).decode("utf-8").split("\n")
        self.w0, self.b0 = d["w0"], d["b0"]
        self.ln1_w, self.ln1_b = d["ln1_w"], d["ln1_b"]
        self.w4, self.b4 = d["w4"], d["b4"]
        self.ln5_w, self.ln5_b = d["ln5_w"], d["ln5_b"]
        self.w8, self.b8 = d["w8"], d["b8"]

    @staticmethod
    def _silu(x: np.ndarray) -> np.ndarray:
        return x / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))

    @staticmethod
    def _ln(x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
        m = x.mean()
        v = x.var()
        return ((x - m) / np.sqrt(v + 1e-5)) * w + b

    def predict_probs(self, features: list[float] | np.ndarray) -> np.ndarray:
        x = (np.asarray(features, dtype=np.float32) - self.feat_mean) / self.feat_std
        x = np.clip(np.nan_to_num(x, nan=0.0, posinf=10.0, neginf=-10.0), -10.0, 10.0)
        h = self._silu(self._ln(x @ self.w0.T + self.b0, self.ln1_w, self.ln1_b))
        h = self._silu(self._ln(h @ self.w4.T + self.b4, self.ln5_w, self.ln5_b))
        logits = (h @ self.w8.T + self.b8) / max(1e-4, self.temperature)
        logits = logits - logits.max()
        exps = np.exp(logits)
        return (exps / exps.sum()).astype(np.float64)
