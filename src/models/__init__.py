"""Model registry. To add a model: create src/models/<name>.py with a class that
has `config` (including "model_type"), `from_args(args)` and
`forward(B, T) -> H`, then register it in MODEL_TYPES below."""
from __future__ import annotations

import torch.nn as nn

from .common import compute_dB, count_parameters
from .gru import BTHGRU
from .mss_gru import MSSGRU
from .transformer import BTHTransformer

MODEL_TYPES = {
    "gru": BTHGRU,
    "mss_gru": MSSGRU,
    "transformer": BTHTransformer,
}


def build_model(model_type: str = "gru", **kwargs) -> nn.Module:
    if model_type not in MODEL_TYPES:
        raise ValueError(f"Unknown model '{model_type}'. Options: {list(MODEL_TYPES)}")
    return MODEL_TYPES[model_type](**kwargs)


def build_model_from_args(args) -> nn.Module:
    """Used by train.py: args.model selects the class, the class reads its own flags."""
    if args.model not in MODEL_TYPES:
        raise ValueError(f"Unknown model '{args.model}'. Options: {list(MODEL_TYPES)}")
    return MODEL_TYPES[args.model].from_args(args)


def build_model_from_config(config: dict) -> nn.Module:
    """Rebuild a model from checkpoint['model_config'].
    Old checkpoints without 'model_type' are treated as the baseline GRU."""
    cfg = dict(config)
    model_type = cfg.pop("model_type", "gru")
    return build_model(model_type, **cfg)


__all__ = [
    "MODEL_TYPES", "build_model", "build_model_from_args", "build_model_from_config",
    "count_parameters", "compute_dB", "BTHGRU", "MSSGRU", "BTHTransformer",
]