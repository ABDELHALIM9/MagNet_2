"""Loss and evaluation metrics (SRE / NERE-style, plus NRMSE)."""
from __future__ import annotations

import torch


def sequence_relative_error(H_pred: torch.Tensor, H_true: torch.Tensor, eps: float = 1e-12):
    """Per-sequence relative error  ||H_pred - H||_2 / ||H||_2  -> [batch].
    Scale-invariant, so it gives the same value on normalized or physical H."""
    num = torch.linalg.vector_norm(H_pred - H_true, dim=-1)
    den = torch.linalg.vector_norm(H_true, dim=-1).clamp_min(eps)
    return num / den


def normalized_energy_relative_error(H_pred, H_true, B, eps: float = 1e-12):
    """Relative error of the hysteresis loss energy, w = integral(H dB) per
    sequence, approximated with the trapezoidal rule. -> [batch].
    Meaningful when each sequence covers whole (periodic) cycles."""
    dB = B[:, 1:] - B[:, :-1]
    w_pred = (0.5 * (H_pred[:, 1:] + H_pred[:, :-1]) * dB).sum(-1)
    w_true = (0.5 * (H_true[:, 1:] + H_true[:, :-1]) * dB).sum(-1)
    return (w_pred - w_true).abs() / w_true.abs().clamp_min(eps)


def nrmse(H_pred: torch.Tensor, H_true: torch.Tensor, eps: float = 1e-12):
    rmse = torch.sqrt(((H_pred - H_true) ** 2).mean(-1))
    scale = H_true.abs().amax(dim=-1).clamp_min(eps)
    return rmse / scale


def summarize(values: torch.Tensor) -> dict:
    """mean / median / 95th percentile / max of a 1-D tensor."""
    v = values.detach().float().flatten()
    return {
        "mean": v.mean().item(),
        "median": v.median().item(),
        "p95": torch.quantile(v, 0.95).item(),
        "max": v.max().item(),
    }
