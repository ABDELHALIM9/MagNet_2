"""Helpers shared by the model files."""
from __future__ import annotations

import torch
import torch.nn as nn


def compute_dB(B: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """dB_t = B_t - B_{t-1} for B [batch, L]; the first sample has no
    predecessor and is set to 0. Multiplied by `scale` because dB is much
    smaller than B per step."""
    dB = torch.cat([torch.zeros_like(B[:, :1]), B[:, 1:] - B[:, :-1]], dim=1)
    return dB * scale


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)